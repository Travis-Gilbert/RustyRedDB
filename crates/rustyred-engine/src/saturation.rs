use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::epistemic::{
    same_eclass_edge_id, EGRAPH_EPISTEMIC_ENGINE, EPISTEMIC_SHADOW_LABEL, HAS_EPISTEMIC_SHADOW,
    SAME_ECLASS,
};
use crate::graph::capability::{GraphSnapshotSource, GraphStore};
use crate::graph::error::GraphStoreResult;
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord, Provenance};
use crate::graph::ttl::now_ms;
use crate::hooks::{
    HookContext, HookError, HookOutcome, HookRegistration, MutationEvent, MutationKind,
    MutationMatcher,
};
use crate::plugin::{PluginCapability, PluginCapabilityKind, RustyRedPlugin};
use crate::state::stable_hash;
use crate::statement::{
    literal_ref, statement_id as hyper_statement_id, statement_incidence_edges,
    StatementFieldProvenance, StatementProvenance, StatementRecord, StatementSemiring, HAS_OBJECT,
    HAS_SUBJECT, STATEMENT_LABEL,
};
use crate::symbolic::{derive_datalog_receipt, stable_hash_value};
use crate::versioned_graph::{
    compile_graph_pack, compile_graph_pack_incremental, CommitCost, GraphCompileOptions,
};

pub const SATURATION_ENGINE: &str = "egglog-saturation";
pub const SATURATION_ENGINE_VERSION: &str = "egglog-saturation-v1";
pub const SATURATION_DERIVED_STATEMENT_LABEL: &str = "SaturationDerivedStatement";
pub const SATURATION_DERIVES_EDGE: &str = "SaturationDerives";
pub const SATURATION_SHARED_RULE_IDS: [&str; 3] = [
    "dependent_claim",
    "evidence_path_too_long",
    "object_in_unresolved_tension_neighborhood",
];

const SATURATION_FACT_LIMIT: usize = 100_000;
pub(crate) const SUPPORT_REACHABLE_RULE: &str = "support_reachability";
pub(crate) const CONTRADICTION_REACHES_RULE: &str = "contradiction_propagation";
pub(crate) const SAME_ENTITY_RULE: &str = "same_entity";
const MAX_SATURATION_PATH_LEN: usize = 32;
const MAX_ENTITY_EVENT_SCOPES: usize = 4_096;

#[derive(Clone)]
struct EntityEventScope {
    component_key: String,
    record_ids: Vec<String>,
}

fn entity_event_scopes() -> &'static Mutex<BTreeMap<(String, String), EntityEventScope>> {
    static SCOPES: OnceLock<Mutex<BTreeMap<(String, String), EntityEventScope>>> = OnceLock::new();
    SCOPES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn cache_entity_event_scope(tenant: &str, edge_id: String, scope: EntityEventScope) {
    let mut scopes = entity_event_scopes().lock().expect("component cache lock");
    scopes.insert((tenant.to_string(), edge_id), scope);
    while scopes.len() > MAX_ENTITY_EVENT_SCOPES {
        let Some(oldest_key) = scopes.keys().next().cloned() else {
            break;
        };
        scopes.remove(&oldest_key);
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SaturationBackend {
    /// Parse and run the public rule program through the real `egglog` crate.
    Egglog,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturationProgram {
    pub backend: SaturationBackend,
    pub engine: String,
    pub engine_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_id: Option<String>,
    #[serde(default = "default_saturation_writeback_policy")]
    pub writeback_policy: String,
    #[serde(default = "default_saturation_iteration_bound")]
    pub iteration_bound: usize,
    #[serde(default)]
    pub semiring: crate::provenance_semiring::ProvenanceSemiringKind,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub program: String,
    pub shared_rule_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SaturationTemplateId {
    EquivalenceDedup,
    EntityResolution,
    SupportReachability,
    ContradictionPropagation,
    SafeProof,
}

impl Default for SaturationProgram {
    fn default() -> Self {
        Self {
            backend: SaturationBackend::Egglog,
            engine: SATURATION_ENGINE.to_string(),
            engine_version: SATURATION_ENGINE_VERSION.to_string(),
            template_id: None,
            writeback_policy: default_saturation_writeback_policy(),
            iteration_bound: default_saturation_iteration_bound(),
            semiring: crate::provenance_semiring::ProvenanceSemiringKind::Viterbi,
            program: String::new(),
            shared_rule_ids: SATURATION_SHARED_RULE_IDS
                .iter()
                .map(|rule| (*rule).to_string())
                .collect(),
        }
    }
}

impl SaturationProgram {
    pub fn from_template(template_id: SaturationTemplateId) -> Self {
        let mut program = Self::default();
        program.template_id = Some(template_id.as_str().to_string());
        program.shared_rule_ids = template_id
            .rule_ids()
            .into_iter()
            .map(str::to_string)
            .collect();
        program
    }

    fn rule_enabled(&self, rule_id: &str) -> bool {
        if self.template_id.is_none() {
            return true;
        }
        if self.shared_rule_ids.is_empty() {
            return false;
        }
        self.shared_rule_ids
            .iter()
            .any(|candidate| candidate == rule_id)
    }

    /// Predicates that require DRed rather than counting under incremental
    /// maintenance. Computed once from the rule dependency graph (cached on
    /// the program via [`crate::incremental::stratify_recursive_predicates`]).
    pub fn recursive_predicates(&self) -> BTreeSet<String> {
        let rules = saturation_rules_for_stratification(self);
        crate::incremental::stratify_recursive_predicates(&rules)
    }
}

fn saturation_rules_for_stratification(
    program: &SaturationProgram,
) -> Vec<crate::incremental::Rule> {
    use crate::incremental::{Atom, Rule};
    let mut rules = Vec::new();
    if program.rule_enabled(SUPPORT_REACHABLE_RULE) {
        // support_reachable is transitive: recursive in the head predicate.
        rules.push(Rule::new(
            SUPPORT_REACHABLE_RULE,
            Atom::new("support_reachable", ["X", "Y"]),
            [Atom::new("support_edge", ["X", "Y"])],
        ));
        rules.push(Rule::new(
            format!("{SUPPORT_REACHABLE_RULE}_trans"),
            Atom::new("support_reachable", ["X", "Y"]),
            [
                Atom::new("support_reachable", ["X", "Z"]),
                Atom::new("support_edge", ["Z", "Y"]),
            ],
        ));
    }
    if program.rule_enabled(CONTRADICTION_REACHES_RULE) {
        rules.push(Rule::new(
            CONTRADICTION_REACHES_RULE,
            Atom::new("contradiction_reaches", ["X", "Y"]),
            [Atom::new("contradicts", ["X", "Y"])],
        ));
        rules.push(Rule::new(
            format!("{CONTRADICTION_REACHES_RULE}_via_support"),
            Atom::new("contradiction_reaches", ["X", "Y"]),
            [
                Atom::new("contradiction_reaches", ["X", "Z"]),
                Atom::new("support_reachable", ["Z", "Y"]),
            ],
        ));
    }
    if program.rule_enabled(SAME_ENTITY_RULE) {
        rules.push(Rule::new(
            SAME_ENTITY_RULE,
            Atom::new("same_entity", ["X", "Y"]),
            [Atom::new("match_edge", ["X", "Y"])],
        ));
        rules.push(Rule::new(
            format!("{SAME_ENTITY_RULE}_trans"),
            Atom::new("same_entity", ["X", "Y"]),
            [
                Atom::new("same_entity", ["X", "Z"]),
                Atom::new("same_entity", ["Z", "Y"]),
            ],
        ));
    }
    rules
}

impl SaturationTemplateId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EquivalenceDedup => "equivalence_dedup",
            Self::EntityResolution => "entity_resolution",
            Self::SupportReachability => "support_reachability",
            Self::ContradictionPropagation => "contradiction_propagation",
            Self::SafeProof => "safe_proof",
        }
    }

    pub fn rule_ids(self) -> Vec<&'static str> {
        match self {
            Self::EquivalenceDedup => Vec::new(),
            Self::EntityResolution => vec![SAME_ENTITY_RULE],
            Self::SupportReachability => vec![SUPPORT_REACHABLE_RULE],
            Self::ContradictionPropagation => vec![CONTRADICTION_REACHES_RULE],
            Self::SafeProof => SATURATION_SHARED_RULE_IDS.to_vec(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturationConfig {
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
    pub component_key: Option<String>,
    pub retracted_fact_ids: Vec<String>,
    pub prune_stale: bool,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        Self {
            engine: SATURATION_ENGINE.to_string(),
            engine_version: SATURATION_ENGINE_VERSION.to_string(),
            computed_at: now_ms(),
            component_key: None,
            retracted_fact_ids: Vec::new(),
            prune_stale: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturationFacts {
    pub component_key: String,
    pub content_node_ids: Vec<String>,
    pub nodes: Vec<NodeRecord>,
    pub edges: Vec<EdgeRecord>,
    pub facts: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturationContributor {
    pub dependency_fact_ids: Vec<String>,
    pub confidence: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturationDerivedStatement {
    pub fact_id: String,
    pub rule_id: String,
    pub relation: String,
    pub subject_id: String,
    pub reason: String,
    pub dependency_fact_ids: Vec<String>,
    pub contributors: Vec<SaturationContributor>,
    pub attributes: Value,
    pub confidence: f64,
    pub writeback_policy: String,
    pub engine: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SaturationEquivalenceClass {
    pub class_id: String,
    pub canonical_form: String,
    pub representative_content_id: String,
    pub representative_shadow_id: String,
    pub member_content_ids: Vec<String>,
    pub member_shadow_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SaturationClosure {
    pub component_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_id: Option<String>,
    #[serde(default)]
    pub writeback_policy: String,
    pub derived_statements: Vec<SaturationDerivedStatement>,
    pub equivalence_classes: Vec<SaturationEquivalenceClass>,
    pub iterations: usize,
    pub egglog_tuple_count: usize,
    pub egglog_output_count: usize,
    pub egglog_iteration_budget: usize,
    pub egglog_checked_pair_count: usize,
    pub egglog_error: Option<String>,
    #[serde(default = "default_closure_semiring")]
    pub semiring: StatementSemiring,
}

fn default_closure_semiring() -> StatementSemiring {
    StatementSemiring::Viterbi
}
fn default_saturation_writeback_policy() -> String {
    "read-only".to_string()
}

fn default_saturation_iteration_bound() -> usize {
    MAX_SATURATION_PATH_LEN
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SaturationRevisionReport {
    pub stale_nodes_tombstoned: usize,
    pub stale_edges_tombstoned: usize,
    pub retracted_fact_ids: Vec<String>,
    pub commit_cost: Option<CommitCost>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SaturationReport {
    pub component_key: String,
    pub derived_statement_count: usize,
    pub equivalence_class_count: usize,
    pub fixpoint_iterations: usize,
    pub nodes_written: usize,
    pub edges_written: usize,
    pub same_eclass_edges_written: usize,
    pub idempotent_skips: usize,
    pub revision: SaturationRevisionReport,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DifferentialReport {
    pub engine: String,
    pub reference_engine: String,
    pub shared_rule_ids: Vec<String>,
    pub expected_count: usize,
    pub actual_count: usize,
    pub missing_fact_ids: Vec<String>,
    pub extra_fact_ids: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct SupportEdge {
    pub(crate) from_id: String,
    pub(crate) to_id: String,
    pub(crate) confidence: f64,
    pub(crate) dependency_fact_id: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ContradictionEdge {
    pub(crate) from_id: String,
    pub(crate) to_id: String,
    pub(crate) confidence: f64,
    pub(crate) dependency_fact_id: String,
}

#[derive(Clone, Debug)]
pub(crate) struct MatchEdge {
    pub(crate) left_record_id: String,
    pub(crate) right_record_id: String,
    pub(crate) confidence: f64,
    pub(crate) dependency_fact_id: String,
}
pub(crate) struct StatementSeed<'a> {
    pub(crate) rule_id: &'a str,
    pub(crate) relation: &'a str,
    pub(crate) subject_id: &'a str,
    pub(crate) reason: &'a str,
    pub(crate) dependency_fact_ids: Vec<String>,
    pub(crate) attributes: Value,
    pub(crate) confidence: f64,
    pub(crate) writeback_policy: &'a str,
}
pub fn facts_from_subgraph<S: GraphStore>(
    store: &S,
    content_node_ids: &[String],
) -> GraphStoreResult<SaturationFacts> {
    let mut content_ids: BTreeSet<String> = content_node_ids
        .iter()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect();
    let full_store = content_ids.is_empty();

    if full_store {
        for node in store
            .query_nodes(NodeQuery::default().with_limit(SATURATION_FACT_LIMIT))
            .into_iter()
        {
            if is_saturation_artifact_node(&node) {
                continue;
            }
            if !node
                .labels
                .iter()
                .any(|label| label == EPISTEMIC_SHADOW_LABEL)
            {
                content_ids.insert(node.id);
            }
        }
    }

    let mut node_ids = content_ids.clone();
    let mut edge_ids: BTreeSet<String> = BTreeSet::new();
    for id in &content_ids {
        for hit in store
            .neighbors(NeighborQuery::out(id).with_include_expired(true))
            .into_iter()
            .chain(store.neighbors(NeighborQuery::in_(id).with_include_expired(true)))
        {
            if hit.edge_type == SATURATION_DERIVES_EDGE {
                continue;
            }
            edge_ids.insert(hit.edge_id.clone());
            node_ids.insert(hit.node_id);
        }
        for hit in store
            .neighbors(
                NeighborQuery::out(id)
                    .with_edge_type(HAS_EPISTEMIC_SHADOW)
                    .with_include_expired(true),
            )
            .into_iter()
        {
            edge_ids.insert(hit.edge_id.clone());
            node_ids.insert(hit.node_id);
        }
    }

    if full_store {
        for shadow in store
            .query_nodes(NodeQuery::label(EPISTEMIC_SHADOW_LABEL).with_limit(SATURATION_FACT_LIMIT))
            .into_iter()
        {
            node_ids.insert(shadow.id);
        }
    }

    let mut nodes = Vec::new();
    for id in node_ids {
        if let Some(node) = store.get_node(&id) {
            if !is_saturation_artifact_node(node) {
                nodes.push(node.clone());
            }
        }
    }
    nodes.sort_by(|left, right| left.id.cmp(&right.id));

    let mut edges = Vec::new();
    for id in edge_ids {
        if let Some(edge) = store.get_edge(&id) {
            if !is_saturation_artifact_edge(edge) {
                edges.push(edge.clone());
            }
        }
    }
    edges.sort_by(|left, right| left.id.cmp(&right.id));

    let facts = datalog_facts_from_graph_records(&nodes, &edges);
    let content_node_ids = content_ids.into_iter().collect::<Vec<_>>();
    let component_key = if full_store {
        "full-store".to_string()
    } else {
        format!("component:{}", stable_hash(&content_node_ids)?)
    };

    Ok(SaturationFacts {
        component_key,
        content_node_ids,
        nodes,
        edges,
        facts,
    })
}

pub fn facts_from_payload(payload: &Value) -> Result<SaturationFacts, String> {
    let raw_facts = payload_facts(payload)?;
    let mut facts = Vec::new();
    for raw in raw_facts {
        facts.push(normalize_datalog_fact(raw)?);
    }
    facts.sort_by(|left, right| object_field(left, "fact_id").cmp(object_field(right, "fact_id")));
    Ok(SaturationFacts {
        component_key: format!(
            "payload:{}",
            stable_hash(&facts).map_err(|error| format!("{}: {}", error.code, error.message))?
        ),
        content_node_ids: Vec::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
        facts,
    })
}

pub fn run_saturation(
    facts: SaturationFacts,
    program: &SaturationProgram,
) -> GraphStoreResult<SaturationClosure> {
    crate::saturation_engine::dispatch_run_saturation(facts, program)
}

pub fn validate_egglog_program(program: &SaturationProgram) -> Result<(usize, usize), String> {
    let facts = SaturationFacts {
        component_key: "egglog-validation".to_string(),
        content_node_ids: Vec::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
        facts: Vec::new(),
    };
    let rules = crate::saturation_engine::SaturationRuleSet::from_program(program);
    let source = match program.semiring {
        crate::provenance_semiring::ProvenanceSemiringKind::Viterbi => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::ViterbiSemiring,
            >(&rules)
        }
        crate::provenance_semiring::ProvenanceSemiringKind::Tropical => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::TropicalSemiring,
            >(&rules)
        }
        crate::provenance_semiring::ProvenanceSemiringKind::Boolean => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::BooleanSemiring,
            >(&rules)
        }
        crate::provenance_semiring::ProvenanceSemiringKind::Counting => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::CountingSemiring,
            >(&rules)
        }
        crate::provenance_semiring::ProvenanceSemiringKind::Corroboration => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::CorroborationSemiring,
            >(&rules)
        }
        crate::provenance_semiring::ProvenanceSemiringKind::Probability => {
            crate::saturation_engine::saturation_program_source::<
                rustyred_thg_derivation::ProbabilitySemiring,
            >(&rules)
        }
    };
    let report = crate::saturation_engine::run_egglog_program_engine::<
        rustyred_thg_derivation::ViterbiSemiring,
    >(&facts, &source, 1, &rules)?;
    Ok((report.tuple_count, report.output_count))
}

pub fn materialize_closure<S: GraphStore + GraphSnapshotSource>(
    store: &mut S,
    closure: SaturationClosure,
    mut config: SaturationConfig,
) -> GraphStoreResult<SaturationReport> {
    if config.component_key.is_none() {
        config.component_key = Some(closure.component_key.clone());
    }
    let component_key = config
        .component_key
        .clone()
        .unwrap_or_else(|| closure.component_key.clone());
    let before_pack = store
        .graph_snapshot()
        .ok()
        .map(|snapshot| compile_graph_pack(&snapshot, graph_compile_options("before saturation")))
        .transpose()?;
    let mut mutations = Vec::new();

    let mut report = SaturationReport {
        component_key: component_key.clone(),
        derived_statement_count: closure.derived_statements.len(),
        equivalence_class_count: closure.equivalence_classes.len(),
        fixpoint_iterations: closure.iterations,
        ..SaturationReport::default()
    };

    let live_statement_ids = closure
        .derived_statements
        .iter()
        .map(|statement| derived_statement_hyper_id(store, statement))
        .collect::<GraphStoreResult<BTreeSet<_>>>()?;
    if config.prune_stale {
        let revision = prune_stale_derived_artifacts(
            store,
            &component_key,
            &live_statement_ids,
            &config.retracted_fact_ids,
            &mut mutations,
        )?;
        report.revision.stale_nodes_tombstoned = revision.0;
        report.revision.stale_edges_tombstoned = revision.1;
        report.revision.retracted_fact_ids = sorted_strings(config.retracted_fact_ids.clone());
    }

    for class in &closure.equivalence_classes {
        let edge_count = materialize_same_eclass(store, class, &config, &mut mutations)?;
        report.same_eclass_edges_written += edge_count;
        report.edges_written += edge_count;
        report.idempotent_skips += class
            .member_shadow_ids
            .len()
            .saturating_sub(1)
            .saturating_sub(edge_count);
    }

    for statement in &closure.derived_statements {
        let node =
            derived_statement_node(store, statement, &component_key, &config, &closure.semiring)?;
        match upsert_node_if_changed(store, node.clone())? {
            WriteDisposition::Written => {
                report.nodes_written += 1;
                mutations.push(GraphMutation::NodeUpsert(node.clone()));
            }
            WriteDisposition::Skipped => report.idempotent_skips += 1,
        }

        for edge in statement_incidence_edges(&node)? {
            if store.get_node(&edge.to_id).is_none() {
                continue;
            }
            match upsert_edge_if_changed(store, edge.clone())? {
                WriteDisposition::Written => {
                    report.edges_written += 1;
                    mutations.push(GraphMutation::EdgeUpsert(edge));
                }
                WriteDisposition::Skipped => report.idempotent_skips += 1,
            }
        }

        if store.get_node(&statement.subject_id).is_some() {
            let edge = derived_statement_edge(statement, &node.id, &config)?;
            match upsert_edge_if_changed(store, edge.clone())? {
                WriteDisposition::Written => {
                    report.edges_written += 1;
                    mutations.push(GraphMutation::EdgeUpsert(edge));
                }
                WriteDisposition::Skipped => report.idempotent_skips += 1,
            }
        }
    }

    if let Some(before_pack) = before_pack.filter(|_| !mutations.is_empty()) {
        let incremental = compile_graph_pack_incremental(
            &before_pack,
            &GraphMutationBatch::new(mutations),
            graph_compile_options("saturation materialized view"),
        )?;
        report.revision.commit_cost = Some(incremental.commit_cost);
    }

    // Project derived statements into the incremental multiplicity index and
    // retract configured base facts through the same maintenance path.
    let _ = crate::incremental::bridge::sync_saturation_materialize(store, &closure, &config)?;

    Ok(report)
}

pub fn differential_check(payload: &Value) -> Result<DifferentialReport, String> {
    let reference_payload = shared_subset_payload(payload)?;
    let facts = facts_from_payload(&reference_payload)?;
    let closure = run_saturation(facts, &SaturationProgram::default())
        .map_err(|error| format!("{}: {}", error.code, error.message))?;
    differential_check_closure(&reference_payload, &closure)
}

fn differential_check_closure(
    reference_payload: &Value,
    closure: &SaturationClosure,
) -> Result<DifferentialReport, String> {
    let receipt = derive_datalog_receipt(reference_payload)?;
    let expected = fact_ids_from_receipt(&receipt);
    if let Some(error) = &closure.egglog_error {
        return Err(format!(
            "egglog saturation failed differential preflight: {error}"
        ));
    }
    let actual = closure
        .derived_statements
        .iter()
        .filter(|statement| SATURATION_SHARED_RULE_IDS.contains(&statement.rule_id.as_str()))
        .map(|statement| statement.fact_id.clone())
        .collect::<BTreeSet<_>>();

    let missing_fact_ids = expected
        .difference(&actual)
        .cloned()
        .collect::<Vec<String>>();
    let extra_fact_ids = actual
        .difference(&expected)
        .cloned()
        .collect::<Vec<String>>();
    let report = DifferentialReport {
        engine: SATURATION_ENGINE.to_string(),
        reference_engine: object_field(&receipt, "engine").to_string(),
        shared_rule_ids: SATURATION_SHARED_RULE_IDS
            .iter()
            .map(|rule| (*rule).to_string())
            .collect(),
        expected_count: expected.len(),
        actual_count: actual.len(),
        missing_fact_ids,
        extra_fact_ids,
    };
    if !report.missing_fact_ids.is_empty() {
        return Err(format!(
            "egglog saturation differential check missing {} fact(s): {:?}",
            report.missing_fact_ids.len(),
            report.missing_fact_ids
        ));
    }
    Ok(report)
}

pub fn coalesce_by_subgraph(event: &MutationEvent) -> Option<String> {
    let component_key = event.changed_props.iter().find_map(|property| {
        property
            .strip_prefix("component_key=")
            .or_else(|| property.strip_prefix("component_key:"))
            .filter(|value| !value.is_empty())
    });
    let component_key = if let Some(component_key) = component_key {
        let mut parts = component_key.split('|');
        let component_key = parts.next().unwrap_or(component_key).to_string();
        let record_ids = parts
            .map(str::to_string)
            .filter(|part| !part.is_empty())
            .collect();
        cache_entity_event_scope(
            &event.tenant,
            event.id.clone(),
            EntityEventScope {
                component_key: component_key.clone(),
                record_ids,
            },
        );
        Some(component_key)
    } else {
        entity_event_scopes()
            .lock()
            .expect("component cache lock")
            .get(&(event.tenant.clone(), event.id.clone()))
            .map(|scope| scope.component_key.clone())
    };
    component_key
        .map(|key| format!("saturation-subgraph:{key}"))
        .or_else(|| Some(format!("saturation-subgraph:unmapped:{}", event.id)))
}

pub fn saturation_hook_registration() -> HookRegistration {
    HookRegistration::new(
        "egglog-saturation",
        MutationMatcher::any()
            .with_kinds([
                MutationKind::NodeUpserted,
                MutationKind::EdgeUpserted,
                MutationKind::NodeDeleted,
                MutationKind::EdgeDeleted,
            ])
            .with_labels([
                "Claim",
                "Evidence",
                "EvidenceLink",
                "ClaimDependency",
                "CONTRADICTS",
                "Contradicts",
                "Undercuts",
                "UNDERCUTS",
                "SUPPORTS",
                "Supports",
                "match_edge",
                "MatchEdge",
            ]),
        coalesce_by_subgraph,
        Arc::new(saturation_handler),
    )
}

pub fn saturation_handler(
    ctx: &mut HookContext,
    events: &[MutationEvent],
) -> Result<HookOutcome, HookError> {
    let program = if events.iter().any(|event| {
        event
            .labels
            .iter()
            .any(|label| matches!(label.as_str(), "match_edge" | "MatchEdge"))
    }) {
        SaturationProgram::from_template(SaturationTemplateId::EntityResolution)
    } else {
        SaturationProgram::default()
    };
    let has_delete = events.iter().any(|event| {
        matches!(
            event.kind,
            MutationKind::NodeDeleted | MutationKind::EdgeDeleted
        )
    });
    let has_entity_retraction = events.iter().any(|event| {
        event
            .labels
            .iter()
            .any(|label| matches!(label.as_str(), "match_edge" | "MatchEdge"))
            && crate::GraphRead::get_edge(ctx.store, &event.id).is_none()
    });
    let mut ids = Vec::new();
    let mut seen = BTreeSet::new();
    for event in events {
        match event.kind {
            MutationKind::NodeUpserted | MutationKind::NodeDeleted => {
                if seen.insert(event.id.clone()) {
                    ids.push(event.id.clone());
                }
            }
            MutationKind::EdgeUpserted => {
                if let Some(edge) = crate::GraphRead::get_edge(ctx.store, &event.id).cloned() {
                    for id in [edge.from_id, edge.to_id] {
                        if seen.insert(id.clone()) {
                            ids.push(id);
                        }
                    }
                } else if let Some(scope) = entity_event_scopes()
                    .lock()
                    .expect("component cache lock")
                    .get(&(event.tenant.clone(), event.id.clone()))
                    .cloned()
                {
                    for id in scope.record_ids {
                        if seen.insert(id.clone()) {
                            ids.push(id);
                        }
                    }
                }
            }
            MutationKind::EdgeDeleted => {
                if let Some(edge) = &event.edge_context {
                    for endpoint in [&edge.from_id, &edge.to_id] {
                        let (record_ids, _) =
                            expand_match_component(ctx.store, std::slice::from_ref(endpoint));
                        for record_id in record_ids {
                            if seen.insert(record_id.clone()) {
                                ids.push(record_id);
                            }
                        }
                    }
                } else if let Some(scope) = entity_event_scopes()
                    .lock()
                    .expect("component cache lock")
                    .get(&(event.tenant.clone(), event.id.clone()))
                    .cloned()
                {
                    // Compatibility fallback for callers constructing legacy
                    // delete events without pre-mutation endpoint context.
                    for id in scope.record_ids {
                        if seen.insert(id.clone()) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }

    for event in events
        .iter()
        .filter(|event| event.kind == MutationKind::EdgeUpserted)
    {
        let key = (event.tenant.clone(), event.id.clone());
        let component_key = entity_event_scopes()
            .lock()
            .expect("component cache lock")
            .get(&key)
            .map(|scope| scope.component_key.clone());
        if let Some(edge) = crate::GraphRead::get_edge(ctx.store, &event.id) {
            let (record_ids, edge_ids) =
                expand_match_component(ctx.store, &[edge.from_id.clone(), edge.to_id.clone()]);
            let component_key =
                component_key.unwrap_or_else(|| format!("entity-component:{}", record_ids[0]));
            for record_id in &record_ids {
                if seen.insert(record_id.clone()) {
                    ids.push(record_id.clone());
                }
            }
            let scope = EntityEventScope {
                component_key,
                record_ids,
            };
            for edge_id in edge_ids {
                cache_entity_event_scope(&event.tenant, edge_id, scope.clone());
            }
        }
    }

    let facts = facts_from_subgraph(ctx.store, &ids)?;
    let closure = run_saturation(facts, &program)?;
    if let Some(error) = &closure.egglog_error {
        return Err(HookError::from(format!(
            "egglog saturation failed before materialization: {error}"
        )));
    }
    let report = materialize_closure(
        ctx.store,
        closure,
        SaturationConfig {
            computed_at: now_ms(),
            retracted_fact_ids: if has_delete || has_entity_retraction {
                events.iter().map(|event| event.id.clone()).collect()
            } else {
                Vec::new()
            },
            ..SaturationConfig::default()
        },
    )?;
    let mut mutations = report.nodes_written
        + report.edges_written
        + report.revision.stale_nodes_tombstoned
        + report.revision.stale_edges_tombstoned;
    for event in events
        .iter()
        .filter(|event| event.kind == MutationKind::EdgeDeleted)
    {
        let old_scope = {
            entity_event_scopes()
                .lock()
                .expect("component cache lock")
                .get(&(event.tenant.clone(), event.id.clone()))
                .cloned()
        };
        if let Some(old_scope) = &old_scope {
            let mut scopes = entity_event_scopes().lock().expect("component cache lock");
            scopes.retain(|(tenant, _), scope| {
                tenant != &event.tenant || scope.component_key != old_scope.component_key
            });
            drop(scopes);
        }

        let seeds = if let Some(edge) = &event.edge_context {
            vec![edge.from_id.clone(), edge.to_id.clone()]
        } else {
            // Legacy fallback only: endpoint-bearing events derive their
            // complete post-delete fragments from durable graph state.
            old_scope.map(|scope| scope.record_ids).unwrap_or_default()
        };
        let mut visited = BTreeSet::new();
        for seed in seeds {
            if visited.contains(&seed) {
                continue;
            }
            let (record_ids, edge_ids) =
                expand_match_component(ctx.store, std::slice::from_ref(&seed));
            visited.extend(record_ids.iter().cloned());
            if record_ids.is_empty() {
                continue;
            }
            let fragment_facts = facts_from_subgraph(ctx.store, &record_ids)?;
            let fragment_closure = run_saturation(fragment_facts, &program)?;
            if let Some(error) = &fragment_closure.egglog_error {
                return Err(HookError::from(format!(
                    "egglog saturation failed for retracted fragment: {error}"
                )));
            }
            let fragment_report = materialize_closure(
                ctx.store,
                fragment_closure,
                SaturationConfig {
                    computed_at: now_ms(),
                    retracted_fact_ids: vec![event.id.clone()],
                    ..SaturationConfig::default()
                },
            )?;
            mutations += fragment_report.nodes_written
                + fragment_report.edges_written
                + fragment_report.revision.stale_nodes_tombstoned
                + fragment_report.revision.stale_edges_tombstoned;

            if edge_ids.is_empty() {
                continue;
            }
            let component_key = format!("entity-component:{}", record_ids[0]);
            let scope = EntityEventScope {
                component_key,
                record_ids,
            };
            for edge_id in edge_ids {
                cache_entity_event_scope(&event.tenant, edge_id, scope.clone());
            }
        }
    }
    for event in events
        .iter()
        .filter(|event| event.kind == MutationKind::NodeDeleted)
    {
        entity_event_scopes()
            .lock()
            .expect("component cache lock")
            .remove(&(event.tenant.clone(), event.id.clone()));
    }
    if mutations == 0 {
        Ok(HookOutcome::Done)
    } else {
        Ok(HookOutcome::Wrote { mutations })
    }
}

fn expand_match_component<S: GraphStore>(
    store: &S,
    seeds: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut records = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut queue = VecDeque::from(seeds.to_vec());
    while let Some(record_id) = queue.pop_front() {
        if !records.insert(record_id.clone()) {
            continue;
        }
        for hit in store
            .neighbors(NeighborQuery::out(&record_id).with_edge_type("match_edge"))
            .into_iter()
            .chain(store.neighbors(NeighborQuery::in_(&record_id).with_edge_type("match_edge")))
        {
            edges.insert(hit.edge_id);
            if !records.contains(&hit.node_id) {
                queue.push_back(hit.node_id);
            }
        }
    }
    (records.into_iter().collect(), edges.into_iter().collect())
}

#[derive(Clone, Debug, Default)]
pub struct SaturationPlugin;

impl RustyRedPlugin for SaturationPlugin {
    fn name(&self) -> &'static str {
        "egglog-saturation"
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability {
            kind: PluginCapabilityKind::Hook,
            name: "egglog-saturation".to_string(),
        }]
    }

    fn hooks(&self) -> Vec<HookRegistration> {
        vec![saturation_hook_registration()]
    }
}
pub(crate) fn egglog_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

pub(crate) fn egglog_f64(value: f64) -> String {
    let value = if value.is_finite() { value } else { 0.0 };
    let mut rendered = format!("{value:.17}");
    while rendered.contains('.') && rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.push('0');
    }
    rendered
}

pub(crate) fn egglog_symbol(value: &str) -> String {
    let mut out = String::from("shadow_");
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}
fn materialize_same_eclass<S: GraphStore>(
    store: &mut S,
    class: &SaturationEquivalenceClass,
    config: &SaturationConfig,
    mutations: &mut Vec<GraphMutation>,
) -> GraphStoreResult<usize> {
    let mut written = 0;
    for member_shadow_id in &class.member_shadow_ids {
        if member_shadow_id == &class.representative_shadow_id {
            continue;
        }
        let edge = EdgeRecord::new(
            same_eclass_edge_id(
                member_shadow_id,
                &class.representative_shadow_id,
                crate::epistemic::DEFAULT_EPISTEMIC_ENGINE_VERSION,
            )?,
            member_shadow_id,
            SAME_ECLASS,
            &class.representative_shadow_id,
            json!({
                "class_id": class.class_id,
                "canonical_form": class.canonical_form,
                "confidence": 1.0,
                "evidence": "egraph_congruence",
                "source_kind": "structural",
                "engine": EGRAPH_EPISTEMIC_ENGINE,
                "engine_version": crate::epistemic::DEFAULT_EPISTEMIC_ENGINE_VERSION,
                "computed_at": config.computed_at,
                "quarantine": true,
            }),
        )
        .with_confidence(1.0)
        .with_provenance(Provenance {
            source_id: Some(EGRAPH_EPISTEMIC_ENGINE.to_string()),
            timestamp: Some(config.computed_at.to_string()),
            method: Some("egglog_saturation_equivalence".to_string()),
        });
        match upsert_edge_if_changed(store, edge.clone())? {
            WriteDisposition::Written => {
                written += 1;
                let committed = crate::GraphRead::get_edge(store, &edge.id)
                    .cloned()
                    .unwrap_or(edge);
                mutations.push(GraphMutation::EdgeUpsert(committed));
            }
            WriteDisposition::Skipped => {}
        }
    }
    Ok(written)
}

fn prune_stale_derived_artifacts<S: GraphStore>(
    store: &mut S,
    component_key: &str,
    live_statement_ids: &BTreeSet<String>,
    retracted_fact_ids: &[String],
    mutations: &mut Vec<GraphMutation>,
) -> GraphStoreResult<(usize, usize)> {
    let retracted = retracted_fact_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut nodes_tombstoned = 0;
    let mut edges_tombstoned = 0;
    let existing = store.query_nodes(
        NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(SATURATION_FACT_LIMIT),
    );
    for node in existing {
        if prop_str(&node.properties, "engine_version").as_deref()
            != Some(SATURATION_ENGINE_VERSION)
        {
            continue;
        }
        if prop_str(&node.properties, "component_key").as_deref() != Some(component_key) {
            continue;
        }
        if live_statement_ids.contains(&node.id) {
            continue;
        }
        let dependency_fact_ids = prop_string_vec(&node.properties, "dependency_fact_ids");
        let touched_retraction = dependency_fact_ids
            .iter()
            .any(|dep| retracted.contains(dep) || retracted.contains(&node.id));
        if !retracted.is_empty() && !touched_retraction {
            continue;
        }

        let outgoing = store.neighbors(NeighborQuery::out(&node.id).with_include_expired(true));
        for hit in outgoing {
            if let Some(edge) = store.get_edge(&hit.edge_id).cloned() {
                let mut tombstone = edge.clone();
                tombstone.tombstone = true;
                match upsert_edge_if_changed(store, tombstone.clone())? {
                    WriteDisposition::Written => {
                        edges_tombstoned += 1;
                        mutations.push(GraphMutation::EdgeUpsert(tombstone));
                    }
                    WriteDisposition::Skipped => {}
                }
            }
        }

        let mut tombstone = node.clone();
        tombstone.tombstone = true;
        match upsert_node_if_changed(store, tombstone.clone())? {
            WriteDisposition::Written => {
                nodes_tombstoned += 1;
                mutations.push(GraphMutation::NodeUpsert(tombstone));
            }
            WriteDisposition::Skipped => {}
        }
    }
    Ok((nodes_tombstoned, edges_tombstoned))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteDisposition {
    Written,
    Skipped,
}

fn upsert_node_if_changed<S: GraphStore>(
    store: &mut S,
    node: NodeRecord,
) -> GraphStoreResult<WriteDisposition> {
    if let Some(existing) = store.get_node(&node.id) {
        if same_node_payload(existing, &node) {
            return Ok(WriteDisposition::Skipped);
        }
    }
    store.upsert_node(node)?;
    Ok(WriteDisposition::Written)
}

fn upsert_edge_if_changed<S: GraphStore>(
    store: &mut S,
    edge: EdgeRecord,
) -> GraphStoreResult<WriteDisposition> {
    if let Some(existing) = store.get_edge(&edge.id) {
        if same_edge_payload(existing, &edge) {
            return Ok(WriteDisposition::Skipped);
        }
    }
    store.upsert_edge(edge)?;
    Ok(WriteDisposition::Written)
}

fn same_node_payload(left: &NodeRecord, right: &NodeRecord) -> bool {
    left.id == right.id
        && left.labels == right.labels
        && left.properties == right.properties
        && left.tombstone == right.tombstone
}

fn same_edge_payload(left: &EdgeRecord, right: &EdgeRecord) -> bool {
    left.id == right.id
        && left.from_id == right.from_id
        && left.to_id == right.to_id
        && left.edge_type == right.edge_type
        && left.properties == right.properties
        && left.tombstone == right.tombstone
        && left.confidence == right.confidence
        && left.epistemic_type == right.epistemic_type
        && left.provenance == right.provenance
}

fn derived_statement_node<S: GraphStore>(
    store: &S,
    statement: &SaturationDerivedStatement,
    component_key: &str,
    config: &SaturationConfig,
    semiring: &StatementSemiring,
) -> GraphStoreResult<NodeRecord> {
    let (object_ref, object_value) = derived_statement_object_ref(store, statement)?;
    let field_provenance = StatementFieldProvenance::new(
        crate::epistemic::EpistemicSourceKind::Structural,
        config.engine.clone(),
        config.engine_version.clone(),
        config.computed_at,
    );
    let provenance = StatementProvenance::new(
        field_provenance,
        statement.dependency_fact_ids.clone(),
        statement.rule_id.clone(),
        semiring.clone(),
    );
    let mut properties = json!({
            "fact_id": statement.fact_id,
            "rule_id": statement.rule_id,
            "relation": statement.relation,
            "subject_id": statement.subject_id,
            "subject_ref": statement.subject_id,
            "object_ref": object_ref.clone(),
            "reason": statement.reason,
            "dependency_fact_ids": statement.dependency_fact_ids,
            "contributors": statement.contributors,
            "attributes": statement.attributes,
            "confidence": statement.confidence,
            "writeback_policy": statement.writeback_policy,
            "engine": config.engine,
            "engine_version": config.engine_version,
            "computed_at": config.computed_at,
            "component_key": component_key,
            "quarantine": true,
            "field_provenance": {
                "confidence": {
                    "source_kind": "structural",
                    "engine": config.engine,
                    "engine_version": config.engine_version,
                    "computed_at": config.computed_at
                },
                "dependency_fact_ids": {
                    "source_kind": "structural",
                    "engine": config.engine,
                    "engine_version": config.engine_version,
                    "computed_at": config.computed_at
                }
            }
    });
    if let Some(object_value) = object_value {
        if let Some(object) = properties.as_object_mut() {
            object.insert("object_value".to_string(), object_value);
        }
    }
    let mut node = StatementRecord::derive(
        &statement.subject_id,
        &statement.relation,
        &object_ref,
        provenance,
        properties,
    )?;
    if !node
        .labels
        .iter()
        .any(|label| label == SATURATION_DERIVED_STATEMENT_LABEL)
    {
        node.labels
            .push(SATURATION_DERIVED_STATEMENT_LABEL.to_string());
        node.labels.sort();
        node.labels.dedup();
    }
    Ok(node)
}

fn derived_statement_edge(
    statement: &SaturationDerivedStatement,
    statement_node_id: &str,
    config: &SaturationConfig,
) -> GraphStoreResult<EdgeRecord> {
    Ok(EdgeRecord::new(
        statement_edge_id(&statement.fact_id, &statement.subject_id)?,
        statement_node_id,
        SATURATION_DERIVES_EDGE,
        &statement.subject_id,
        json!({
            "fact_id": statement.fact_id,
            "statement_id": statement_node_id,
            "rule_id": statement.rule_id,
            "confidence": statement.confidence,
            "dependency_fact_ids": statement.dependency_fact_ids,
            "engine": config.engine,
            "engine_version": config.engine_version,
            "computed_at": config.computed_at,
            "quarantine": true,
        }),
    )
    .with_confidence(statement.confidence)
    .with_epistemic_type(EpistemicType::Derives)
    .with_provenance(Provenance {
        source_id: Some(config.engine.clone()),
        timestamp: Some(config.computed_at.to_string()),
        method: Some("egglog_saturation".to_string()),
    }))
}

fn derived_statement_hyper_id<S: GraphStore>(
    store: &S,
    statement: &SaturationDerivedStatement,
) -> GraphStoreResult<String> {
    let (object_ref, _) = derived_statement_object_ref(store, statement)?;
    hyper_statement_id(&statement.subject_id, &statement.relation, &object_ref)
}

fn derived_statement_object_ref<S: GraphStore>(
    store: &S,
    statement: &SaturationDerivedStatement,
) -> GraphStoreResult<(String, Option<Value>)> {
    for key in ["target_id", "depends_on_object_id", "artifact_id"] {
        let candidate = prop_str(&statement.attributes, key);
        if let Some(candidate) = candidate.filter(|candidate| store.get_node(candidate).is_some()) {
            return Ok((candidate, None));
        }
    }
    if let Some(edge_id) = prop_str(&statement.attributes, "edge_id") {
        if store.get_node(&edge_id).is_some() {
            return Ok((edge_id, None));
        }
        let value = json!({
            "edge_id": edge_id,
            "attributes": statement.attributes,
        });
        return Ok((literal_ref(&value)?, Some(value)));
    }
    for key in ["target_id", "depends_on_object_id", "artifact_id"] {
        if let Some(candidate) = prop_str(&statement.attributes, key) {
            let value = Value::String(candidate);
            return Ok((literal_ref(&value)?, Some(value)));
        }
    }
    let value = if statement.attributes.is_null() {
        Value::String(statement.reason.clone())
    } else {
        statement.attributes.clone()
    };
    Ok((literal_ref(&value)?, Some(value)))
}

fn statement_edge_id(fact_id: &str, subject_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "saturation:edge:{}",
        stable_hash(json!({ "fact_id": fact_id, "subject_id": subject_id }))?
    ))
}

pub(crate) fn exact_derived_statement(
    seed: StatementSeed<'_>,
    engine: &str,
) -> Result<SaturationDerivedStatement, String> {
    let dependency_fact_ids = sorted_strings(seed.dependency_fact_ids);
    let hash_payload = json!({
        "rule_id": seed.rule_id,
        "relation": seed.relation,
        "subject_id": seed.subject_id,
        "attributes": seed.attributes,
        "dependency_fact_ids": dependency_fact_ids,
    });
    let fact_id = stable_hash_value(&hash_payload)?;
    Ok(SaturationDerivedStatement {
        fact_id,
        rule_id: seed.rule_id.to_string(),
        relation: seed.relation.to_string(),
        subject_id: seed.subject_id.to_string(),
        reason: seed.reason.to_string(),
        dependency_fact_ids: value_string_vec(&hash_payload["dependency_fact_ids"]),
        contributors: vec![SaturationContributor {
            dependency_fact_ids: value_string_vec(&hash_payload["dependency_fact_ids"]),
            confidence: seed.confidence,
        }],
        attributes: hash_payload["attributes"].clone(),
        confidence: seed.confidence,
        writeback_policy: seed.writeback_policy.to_string(),
        engine: engine.to_string(),
    })
}

pub(crate) fn saturation_statement(
    seed: StatementSeed<'_>,
    engine: &str,
) -> GraphStoreResult<SaturationDerivedStatement> {
    let dependency_fact_ids = sorted_strings(seed.dependency_fact_ids);
    let hash_payload = json!({
        "rule_id": seed.rule_id,
        "relation": seed.relation,
        "subject_id": seed.subject_id,
        "attributes": seed.attributes,
    });
    Ok(SaturationDerivedStatement {
        fact_id: stable_hash(hash_payload)?,
        rule_id: seed.rule_id.to_string(),
        relation: seed.relation.to_string(),
        subject_id: seed.subject_id.to_string(),
        reason: seed.reason.to_string(),
        dependency_fact_ids: dependency_fact_ids.clone(),
        contributors: vec![SaturationContributor {
            dependency_fact_ids,
            confidence: seed.confidence,
        }],
        attributes: seed.attributes,
        confidence: seed.confidence,
        writeback_policy: seed.writeback_policy.to_string(),
        engine: engine.to_string(),
    })
}

pub(crate) fn merge_statement(
    statements: &mut BTreeMap<String, SaturationDerivedStatement>,
    mut statement: SaturationDerivedStatement,
) {
    statement.dependency_fact_ids = sorted_strings(statement.dependency_fact_ids);
    if let Some(existing) = statements.get_mut(&statement.fact_id) {
        existing.confidence = existing.confidence.max(statement.confidence);
        existing.dependency_fact_ids = sorted_strings(
            existing
                .dependency_fact_ids
                .iter()
                .cloned()
                .chain(statement.dependency_fact_ids.iter().cloned())
                .collect(),
        );
        existing.contributors.extend(statement.contributors);
        existing.contributors.sort_by(|left, right| {
            left.dependency_fact_ids
                .cmp(&right.dependency_fact_ids)
                .then_with(|| left.confidence.total_cmp(&right.confidence))
        });
        existing.contributors.dedup_by(|left, right| {
            left.dependency_fact_ids == right.dependency_fact_ids
                && (left.confidence - right.confidence).abs() < f64::EPSILON
        });
    } else {
        statements.insert(statement.fact_id.clone(), statement);
    }
}

/// Convert already-admitted owned records from one revision into the canonical
/// relational facts. This function performs no storage traversal or admission;
/// callers own both the bounded record selection and its visibility policy.
pub fn datalog_facts_from_graph_records(nodes: &[NodeRecord], edges: &[EdgeRecord]) -> Vec<Value> {
    let mut facts = Vec::new();
    for node in nodes {
        if node.labels.iter().any(|label| label == "Claim")
            || prop_str(&node.properties, "claim_text").is_some()
        {
            facts.push(graph_fact(
                "claim",
                &node.id,
                &node.id,
                json!({
                    "status": prop_value(&node.properties, "status").unwrap_or_else(|| json!("")),
                    "claim_text": claim_text(node),
                }),
                "graph-node",
            ));
        }
    }
    for edge in edges {
        let edge_type = edge.edge_type.as_str();
        if matches!(edge_type, "match_edge" | "MatchEdge") {
            facts.push(graph_fact(
                "match_edge",
                &edge.id,
                &edge.id,
                json!({
                    "left_record_id": prop_str(&edge.properties, "left_record_id").unwrap_or_else(|| edge.from_id.clone()),
                    "right_record_id": prop_str(&edge.properties, "right_record_id").unwrap_or_else(|| edge.to_id.clone()),
                    "weight": prop_value(&edge.properties, "weight").unwrap_or_else(|| json!(edge.effective_confidence())),
                    "comparison_vector": prop_value(&edge.properties, "comparison_vector").unwrap_or_else(|| json!([])),
                }),
                "resolve-pair-decision",
            ));
        } else if is_claim_dependency_edge(edge_type) {
            facts.push(graph_fact(
                "claim_dependency",
                &edge.id,
                &edge.id,
                json!({
                    "claim_id": edge.from_id,
                    "depends_on_object_id": edge.to_id,
                    "justification_type": prop_value(&edge.properties, "justification_type").unwrap_or_else(|| json!("dependency")),
                    "strength": prop_value(&edge.properties, "strength").unwrap_or_else(|| json!(edge.effective_confidence())),
                }),
                "graph-edge",
            ));
        } else if is_evidence_link_edge(edge_type) {
            facts.push(graph_fact(
                "evidence_link",
                &edge.id,
                &edge.id,
                json!({
                    "claim_id": edge.from_id,
                    "artifact_id": edge.to_id,
                    "strength": prop_value(&edge.properties, "strength").unwrap_or_else(|| json!(edge.effective_confidence())),
                }),
                "graph-edge",
            ));
        } else if is_contradiction_edge(edge_type) {
            facts.push(graph_fact(
                "edge",
                &edge.id,
                &edge.id,
                json!({
                    "edge_type": "contradicts",
                    "from_object_id": edge.from_id,
                    "to_object_id": edge.to_id,
                    "acceptance_status": prop_value(&edge.properties, "acceptance_status").unwrap_or_else(|| json!("contested")),
                    "confidence": edge.effective_confidence(),
                }),
                "graph-edge",
            ));
        }
    }
    facts.extend(statement_facts_from_graph_records(nodes, edges));
    facts.sort_by(|left, right| object_field(left, "fact_id").cmp(object_field(right, "fact_id")));
    facts
}

fn statement_facts_from_graph_records(nodes: &[NodeRecord], edges: &[EdgeRecord]) -> Vec<Value> {
    let mut subject_by_statement = BTreeMap::<String, String>::new();
    let mut object_by_statement = BTreeMap::<String, String>::new();
    for edge in edges {
        match edge.edge_type.as_str() {
            HAS_SUBJECT => {
                subject_by_statement.insert(edge.from_id.clone(), edge.to_id.clone());
            }
            HAS_OBJECT => {
                object_by_statement.insert(edge.from_id.clone(), edge.to_id.clone());
            }
            _ => {}
        }
    }

    let mut facts = Vec::new();
    for node in nodes {
        if !node.labels.iter().any(|label| label == STATEMENT_LABEL) {
            continue;
        }
        if prop_bool(&node.properties, "derived") {
            continue;
        }
        let relation = prop_str(&node.properties, "relation")
            .or_else(|| prop_str(&node.properties, "predicate_key"))
            .unwrap_or_default();
        if relation.is_empty() {
            continue;
        }
        let subject_id = subject_by_statement
            .get(&node.id)
            .cloned()
            .or_else(|| prop_str(&node.properties, "subject_ref"))
            .or_else(|| prop_str(&node.properties, "subject_id"))
            .unwrap_or_default();
        if subject_id.is_empty() {
            continue;
        }
        let object_id = object_by_statement
            .get(&node.id)
            .cloned()
            .or_else(|| {
                prop_str(&node.properties, "object_ref").filter(|value| !value.starts_with("lit:"))
            })
            .unwrap_or_default();
        let confidence = node
            .properties
            .get("confidence")
            .and_then(value_to_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        if is_claim_dependency_edge(&relation) {
            if object_id.is_empty() {
                continue;
            }
            facts.push(graph_fact(
                "claim_dependency",
                &node.id,
                &node.id,
                json!({
                    "claim_id": subject_id,
                    "depends_on_object_id": object_id,
                    "justification_type": prop_value(&node.properties, "justification_type").unwrap_or_else(|| json!("dependency")),
                    "strength": confidence,
                }),
                &prop_str(&node.properties, "source_ref").unwrap_or_else(|| "statement".to_string()),
            ));
        } else if is_evidence_link_edge(&relation) {
            if object_id.is_empty() {
                continue;
            }
            facts.push(graph_fact(
                "evidence_link",
                &node.id,
                &node.id,
                json!({
                    "claim_id": subject_id,
                    "artifact_id": object_id,
                    "strength": confidence,
                }),
                &prop_str(&node.properties, "source_ref")
                    .unwrap_or_else(|| "statement".to_string()),
            ));
        } else if is_contradiction_edge(&relation) {
            if object_id.is_empty() {
                continue;
            }
            facts.push(graph_fact(
                "edge",
                &node.id,
                &node.id,
                json!({
                    "edge_type": "contradicts",
                    "from_object_id": subject_id,
                    "to_object_id": object_id,
                    "acceptance_status": prop_value(&node.properties, "acceptance_status").unwrap_or_else(|| json!("contested")),
                    "confidence": confidence,
                }),
                &prop_str(&node.properties, "source_ref").unwrap_or_else(|| "statement".to_string()),
            ));
        }
    }
    facts
}

fn graph_fact(
    relation: &str,
    fact_id: &str,
    entity_id: &str,
    attributes: Value,
    source_ref: &str,
) -> Value {
    let source_ref = source_ref.to_string();
    json!({
        "fact_id": fact_id,
        "relation": relation,
        "entity_id": entity_id,
        "attributes": attributes,
        "source_ref": source_ref,
    })
}

pub(crate) fn support_edges_from_facts(facts: &[Value]) -> Vec<SupportEdge> {
    let mut out = Vec::new();
    for fact in facts {
        match object_field(fact, "relation") {
            "claim_dependency" => {
                let from_id = attr_string(fact, "claim_id");
                let to_id = attr_string(fact, "depends_on_object_id");
                if from_id.is_empty() || to_id.is_empty() {
                    continue;
                }
                out.push(SupportEdge {
                    from_id,
                    to_id,
                    confidence: numeric_attr(fact, "strength", 1.0).clamp(0.0, 1.0),
                    dependency_fact_id: object_field(fact, "fact_id").to_string(),
                });
            }
            "evidence_link" => {
                let from_id = attr_string(fact, "claim_id");
                let to_id = attr_string(fact, "artifact_id");
                if from_id.is_empty() || to_id.is_empty() {
                    continue;
                }
                out.push(SupportEdge {
                    from_id,
                    to_id,
                    confidence: numeric_attr(fact, "strength", 1.0).clamp(0.0, 1.0),
                    dependency_fact_id: object_field(fact, "fact_id").to_string(),
                });
            }
            _ => {}
        }
    }
    out.sort_by(|left, right| {
        (&left.from_id, &left.to_id, &left.dependency_fact_id).cmp(&(
            &right.from_id,
            &right.to_id,
            &right.dependency_fact_id,
        ))
    });
    out
}

pub(crate) fn contradiction_edges_from_facts(facts: &[Value]) -> Vec<ContradictionEdge> {
    let mut out = Vec::new();
    for fact in facts {
        if object_field(fact, "relation") != "edge" {
            continue;
        }
        let edge_type = attr_string(fact, "edge_type").to_lowercase();
        let status = attr_string(fact, "acceptance_status").to_lowercase();
        if edge_type != "contradicts" && edge_type != "undercuts" && status != "contested" {
            continue;
        }
        let from_id = attr_string(fact, "from_object_id");
        let to_id = attr_string(fact, "to_object_id");
        if from_id.is_empty() || to_id.is_empty() {
            continue;
        }
        out.push(ContradictionEdge {
            from_id,
            to_id,
            confidence: numeric_attr(fact, "confidence", 1.0).clamp(0.0, 1.0),
            dependency_fact_id: object_field(fact, "fact_id").to_string(),
        });
    }
    out.sort_by(|left, right| {
        (&left.from_id, &left.to_id, &left.dependency_fact_id).cmp(&(
            &right.from_id,
            &right.to_id,
            &right.dependency_fact_id,
        ))
    });
    out
}

pub(crate) fn match_edges_from_facts(facts: &[Value]) -> Vec<MatchEdge> {
    let mut edges = BTreeMap::<(String, String), MatchEdge>::new();
    for fact in facts {
        if object_field(fact, "relation") != "match_edge" {
            continue;
        }
        let left = attr_string(fact, "left_record_id");
        let right = attr_string(fact, "right_record_id");
        if left.is_empty() || right.is_empty() || left == right {
            continue;
        }
        let (left_record_id, right_record_id) = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        let candidate = MatchEdge {
            left_record_id: left_record_id.clone(),
            right_record_id: right_record_id.clone(),
            confidence: log2_weight_probability(numeric_attr(fact, "weight", 0.0)),
            dependency_fact_id: object_field(fact, "fact_id").to_string(),
        };
        edges
            .entry((left_record_id, right_record_id))
            .and_modify(|existing| {
                if candidate.confidence > existing.confidence
                    || (candidate.confidence == existing.confidence
                        && candidate.dependency_fact_id < existing.dependency_fact_id)
                {
                    *existing = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
    edges.into_values().collect()
}
/// Convert a Fellegi-Sunter log2 likelihood ratio to a probability without
/// discarding negative or greater-than-one raw weights from the source fact.
pub(crate) fn log2_weight_probability(weight: f64) -> f64 {
    if weight >= 0.0 {
        1.0 / (1.0 + (-weight).exp2())
    } else {
        let odds = weight.exp2();
        odds / (1.0 + odds)
    }
}

fn is_claim_dependency_edge(edge_type: &str) -> bool {
    matches!(
        edge_type,
        "claim_dependency" | "ClaimDependency" | "CLAIM_DEPENDENCY" | "DEPENDS_ON" | "DependsOn"
    )
}

fn is_evidence_link_edge(edge_type: &str) -> bool {
    matches!(
        edge_type,
        "evidence_link" | "EvidenceLink" | "EVIDENCE_LINK" | "SUPPORTS" | "Supports" | "supports"
    )
}

fn is_contradiction_edge(edge_type: &str) -> bool {
    matches!(
        edge_type,
        "CONTRADICTS" | "Contradicts" | "contradicts" | "UNDERCUTS" | "Undercuts" | "undercuts"
    )
}
fn payload_facts(payload: &Value) -> Result<&Vec<Value>, String> {
    if let Some(facts) = payload.as_array() {
        return Ok(facts);
    }
    payload
        .get("facts")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "datalog payload expected a JSON array or an object with facts array".to_string()
        })
}

fn normalize_datalog_fact(raw: &Value) -> Result<Value, String> {
    let relation = field_string(raw, "relation");
    let entity_id = field_string(raw, "entity_id");
    if relation.trim().is_empty() {
        return Err("DatalogFact requires relation".to_string());
    }
    if entity_id.trim().is_empty() {
        return Err("DatalogFact requires entity_id".to_string());
    }
    let attributes = raw
        .get("attributes")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let source_ref = field_string(raw, "source_ref");
    let fallback_id = stable_hash_value(&json!({
        "relation": relation,
        "entity_id": entity_id,
        "attributes": attributes,
        "source_ref": source_ref,
    }))?;
    let fact_id = field_string(raw, "fact_id");
    Ok(json!({
        "fact_id": if fact_id.is_empty() { fallback_id } else { fact_id },
        "relation": field_string(raw, "relation"),
        "entity_id": field_string(raw, "entity_id"),
        "attributes": raw
            .get("attributes")
            .filter(|value| value.is_object())
            .cloned()
            .unwrap_or_else(|| json!({})),
        "source_ref": field_string(raw, "source_ref"),
    }))
}

fn shared_subset_payload(payload: &Value) -> Result<Value, String> {
    Ok(json!({
        "facts": payload_facts(payload)?.clone(),
        "rule_ids": SATURATION_SHARED_RULE_IDS,
    }))
}

fn fact_ids_from_receipt(receipt: &Value) -> BTreeSet<String> {
    receipt
        .get("derived_facts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|fact| fact.get("fact_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn is_saturation_artifact_node(node: &NodeRecord) -> bool {
    node.labels
        .iter()
        .any(|label| label == SATURATION_DERIVED_STATEMENT_LABEL)
        || prop_str(&node.properties, "engine").as_deref() == Some(SATURATION_ENGINE)
}

fn is_saturation_artifact_edge(edge: &EdgeRecord) -> bool {
    edge.edge_type == SATURATION_DERIVES_EDGE
        || prop_str(&edge.properties, "engine").as_deref() == Some(SATURATION_ENGINE)
}

pub(crate) fn claim_text(node: &NodeRecord) -> String {
    ["claim_text", "text", "title", "body", "content"]
        .iter()
        .find_map(|key| prop_str(&node.properties, key))
        .unwrap_or_else(|| node.id.clone())
}

fn graph_compile_options(message: &str) -> GraphCompileOptions {
    GraphCompileOptions {
        name: Some("saturation-revision".to_string()),
        branch: Some("saturation".to_string()),
        author: Some(SATURATION_ENGINE.to_string()),
        message: Some(message.to_string()),
        ..GraphCompileOptions::default()
    }
}

pub(crate) fn sorted_strings(values: Vec<String>) -> Vec<String> {
    let mut values = values
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn value_string_vec(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn prop_string_vec(value: &Value, key: &str) -> Vec<String> {
    value.get(key).map(value_string_vec).unwrap_or_default()
}

fn field_string(value: &Value, key: &str) -> String {
    value.get(key).map(value_to_string).unwrap_or_default()
}

pub(crate) fn object_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn attr<'a>(fact: &'a Value, key: &str) -> Option<&'a Value> {
    fact.get("attributes")
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get(key))
}

pub(crate) fn attr_string(fact: &Value, key: &str) -> String {
    attr(fact, key).map(value_to_string).unwrap_or_default()
}
pub(crate) fn numeric_attr(fact: &Value, key: &str, fallback: f64) -> f64 {
    attr(fact, key).and_then(value_to_f64).unwrap_or(fallback)
}

pub(crate) fn integer_attr(fact: &Value, key: &str, fallback: i64) -> i64 {
    attr(fact, key)
        .and_then(|value| match value {
            Value::Number(number) => number
                .as_i64()
                .or_else(|| number.as_u64().map(|v| v as i64)),
            Value::String(text) => text.parse::<i64>().ok(),
            _ => None,
        })
        .unwrap_or(fallback)
}

pub(crate) fn prop_str(value: &Value, key: &str) -> Option<String> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .map(value_to_string)
        .filter(|value| !value.trim().is_empty())
}

fn prop_value(value: &Value, key: &str) -> Option<Value> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .cloned()
}

fn prop_bool(value: &Value, key: &str) -> bool {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}
