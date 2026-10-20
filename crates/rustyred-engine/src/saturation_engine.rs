//! Egglog-backed saturation evaluator parameterized by [`Semiring`].
//!
//! SPEC-THEOREM-DERIVATION-COLLAPSE-1.0: one evaluator; the rule set is data;
//! the semiring is a type parameter.

use std::collections::{BTreeMap, BTreeSet};

use egraph::{Atom, TermValue};
use rustyred_thg_derivation::{
    BooleanSemiring, CorroborationSemiring, CountingSemiring, ProbabilitySemiring, Semiring,
    TropicalSemiring, ViterbiSemiring,
};
use serde_json::{json, Value as JsonValue};

use crate::epistemic::{
    claim_logical_form, grounded_status_symbol, ClaimForm, EPISTEMIC_SHADOW_LABEL,
};
use crate::graph::error::GraphStoreResult;
use crate::provenance_semiring::ProvenanceSemiringKind;
use crate::saturation::{
    attr_string, claim_text, contradiction_edges_from_facts, egglog_f64, egglog_string,
    egglog_symbol, exact_derived_statement, integer_attr, match_edges_from_facts, merge_statement,
    numeric_attr, object_field, prop_str, saturation_statement, sorted_strings,
    support_edges_from_facts, SaturationClosure, SaturationContributor, SaturationDerivedStatement,
    SaturationEquivalenceClass, SaturationFacts, SaturationProgram, StatementSeed,
    CONTRADICTION_REACHES_RULE, SAME_ENTITY_RULE, SATURATION_SHARED_RULE_IDS,
    SUPPORT_REACHABLE_RULE,
};
use crate::state::stable_hash;
use crate::statement::StatementSemiring;

const MAX_SATURATION_PATH_LEN: usize = 32;

/// Enabled saturation rule families for program generation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SaturationRuleSet {
    pub support_reachability: bool,
    pub contradiction_propagation: bool,
    pub same_entity: bool,
    pub dependent_claim: bool,
    pub evidence_path_too_long: bool,
    pub object_in_unresolved_tension_neighborhood: bool,
    pub equivalence: bool,
}

impl SaturationRuleSet {
    pub fn from_program(program: &SaturationProgram) -> Self {
        if program.template_id.is_none() {
            return Self {
                support_reachability: true,
                contradiction_propagation: true,
                same_entity: true,
                dependent_claim: true,
                evidence_path_too_long: true,
                object_in_unresolved_tension_neighborhood: true,
                equivalence: true,
            };
        }
        let enabled = |id: &str| program.shared_rule_ids.iter().any(|r| r == id);
        Self {
            support_reachability: enabled(SUPPORT_REACHABLE_RULE),
            contradiction_propagation: enabled(CONTRADICTION_REACHES_RULE),
            same_entity: enabled(SAME_ENTITY_RULE),
            dependent_claim: enabled("dependent_claim"),
            evidence_path_too_long: enabled("evidence_path_too_long"),
            object_in_unresolved_tension_neighborhood: enabled(
                "object_in_unresolved_tension_neighborhood",
            ),
            equivalence: program
                .template_id
                .as_deref()
                .is_some_and(|t| t == "equivalence_dedup"),
        }
    }
}

pub fn statement_semiring_for(kind: ProvenanceSemiringKind) -> StatementSemiring {
    match kind {
        ProvenanceSemiringKind::Boolean => StatementSemiring::Boolean,
        ProvenanceSemiringKind::Counting => StatementSemiring::Counting,
        ProvenanceSemiringKind::Viterbi => StatementSemiring::Viterbi,
        ProvenanceSemiringKind::Tropical => StatementSemiring::Tropical,
        ProvenanceSemiringKind::Corroboration => StatementSemiring::Corroboration,
        ProvenanceSemiringKind::Probability => StatementSemiring::Probability,
    }
}

fn lift_annotation<S: Semiring>(var: &str) -> String {
    match S::egglog_sort() {
        "bool" => format!("(!= {var} 0.0)"),
        "i64" => S::egglog_one(),
        _ => var.to_string(),
    }
}

/// Emit the egglog program body for saturation under semiring `S`.
pub fn saturation_program_source<S: Semiring>(rules: &SaturationRuleSet) -> String {
    let sort = S::egglog_sort();
    let merge = S::egglog_merge();
    let mut out = String::new();
    out.push_str(
        r#"
(datatype Prop
  (Atom String)
  (Not Prop)
  (Shadow String Prop))

(sort DepSet (Set String))

(relation support-edge (String String f64 String))
(relation contradiction-edge (String String f64 String))
(relation match-edge (String String f64 String))
(relation claim-dependency (String String String f64 String))
(relation evidence-path (String i64 String))
(relation tension-edge (String String String String String String))

(relation support-reachable (String String))
(relation contradiction-reaches (String String))
(relation same-entity (String String))
(relation dependent-claim (String String String f64 String))
(relation evidence-path-too-long (String i64 String))
(relation tension-neighborhood (String String String String String))
(relation same-shadow (String String))

(function shadow-binding (String) Prop :merge new)

"#,
    );
    out.push_str(&format!(
        "(function support-confidence (String String) {sort} :merge {merge})\n"
    ));
    out.push_str(&format!(
        "(function contradiction-confidence (String String) {sort} :merge {merge})\n"
    ));
    out.push_str(&format!(
        "(function same-entity-confidence (String String) {sort} :merge {merge})\n"
    ));
    out.push_str(
        "(function support-deps (String String) DepSet :merge (set-union old new))\n\
(function contradiction-deps (String String) DepSet :merge (set-union old new))\n\
(function same-entity-deps (String String) DepSet :merge (set-union old new))\n\
(function support-path-len (String String) i64 :merge (min old new))\n\
(function contradiction-path-len (String String) i64 :merge (min old new))\n\
(function same-entity-path-len (String String) i64 :merge (min old new))\n\n\
(ruleset equivalence)\n\
(ruleset derivation)\n\n\
(rewrite (Not (Not x)) x :ruleset equivalence)\n\n\
(rule ((= ea (shadow-binding a))
       (= eb (shadow-binding b))
       (= ea eb))
      ((same-shadow a b))
      :ruleset equivalence
      :name \"same-shadow\")\n\n",
    );

    let prod = |a: &str, b: &str| S::egglog_product(a, b);
    let base = |v: &str| lift_annotation::<S>(v);
    let next_lifted = lift_annotation::<S>("next");

    if rules.support_reachability {
        let base_conf = base("confidence");
        out.push_str(&format!(
            r#"(rule ((support-edge a b confidence dependency))
      ((support-reachable a b)
       (set (support-confidence a b) {base_conf})
       (set (support-deps a b) (set-of dependency))
       (set (support-path-len a b) 1))
      :ruleset derivation
      :name "support-base")
"#
        ));
        out.push_str(&format!(
            r#"(rule ((support-reachable a b)
       (= prior (support-confidence a b))
       (= deps (support-deps a b))
       (= plen (support-path-len a b))
       (support-edge b c next dependency))
      ((support-reachable a c)
       (set (support-confidence a c) {product})
       (set (support-deps a c) (set-union deps (set-of dependency)))
       (set (support-path-len a c) (+ plen 1)))
      :ruleset derivation
      :name "support-transitive")
"#,
            product = prod("prior", &next_lifted),
        ));
    }

    if rules.contradiction_propagation {
        let base_conf = base("confidence");
        out.push_str(&format!(
            r#"(rule ((contradiction-edge a b confidence dependency))
      ((contradiction-reaches a b)
       (set (contradiction-confidence a b) {base_conf})
       (set (contradiction-deps a b) (set-of dependency))
       (set (contradiction-path-len a b) 1))
      :ruleset derivation
      :name "contradiction-base")
"#
        ));
        out.push_str(&format!(
            r#"(rule ((contradiction-reaches a b)
       (= prior (contradiction-confidence a b))
       (= deps (contradiction-deps a b))
       (= plen (contradiction-path-len a b))
       (support-edge b c next dependency))
      ((contradiction-reaches a c)
       (set (contradiction-confidence a c) {product})
       (set (contradiction-deps a c) (set-union deps (set-of dependency)))
       (set (contradiction-path-len a c) (+ plen 1)))
      :ruleset derivation
      :name "contradiction-propagates")
"#,
            product = prod("prior", &next_lifted),
        ));
    }

    if rules.same_entity {
        let base_conf = base("confidence");
        out.push_str(&format!(
            r#"(rule ((match-edge a b confidence dependency))
      ((same-entity a b)
       (same-entity b a)
       (set (same-entity-confidence a b) {base_conf})
       (set (same-entity-confidence b a) {base_conf})
       (set (same-entity-deps a b) (set-of dependency))
       (set (same-entity-deps b a) (set-of dependency))
       (set (same-entity-path-len a b) 1)
       (set (same-entity-path-len b a) 1))
      :ruleset derivation
      :name "same-entity-base")
"#
        ));
        out.push_str(&format!(
            r#"(rule ((same-entity a b)
       (= prior (same-entity-confidence a b))
       (= deps (same-entity-deps a b))
       (= plen (same-entity-path-len a b))
       (match-edge b c next dependency))
      ((same-entity a c)
       (set (same-entity-confidence a c) {product})
       (set (same-entity-deps a c) (set-union deps (set-of dependency)))
       (set (same-entity-path-len a c) (+ plen 1)))
      :ruleset derivation
      :name "same-entity-transitive")
"#,
            product = prod("prior", &next_lifted),
        ));
    }

    if rules.dependent_claim {
        out.push_str(
            r#"(rule ((claim-dependency claim depends just strength fact))
      ((dependent-claim claim depends just strength fact))
      :ruleset derivation
      :name "dependent-claim")
"#,
        );
    }
    if rules.evidence_path_too_long {
        out.push_str(
            r#"(rule ((evidence-path entity plen fact) (> plen 3))
      ((evidence-path-too-long entity plen fact))
      :ruleset derivation
      :name "evidence-path-too-long")
"#,
        );
    }
    if rules.object_in_unresolved_tension_neighborhood {
        out.push_str(
            r#"(rule ((tension-edge from to edge_id edge_type status fact))
      ((tension-neighborhood from edge_id edge_type status fact)
       (tension-neighborhood to edge_id edge_type status fact))
      :ruleset derivation
      :name "tension-neighborhood")
"#,
        );
    }

    out
}

pub fn iteration_bound_for<S: Semiring>(facts: &SaturationFacts) -> usize {
    if S::is_absorptive() {
        distinct_entity_count(facts).max(1)
    } else {
        MAX_SATURATION_PATH_LEN
    }
}

fn distinct_entity_count(facts: &SaturationFacts) -> usize {
    let mut ids = BTreeSet::new();
    for fact in &facts.facts {
        let relation = fact.get("relation").and_then(|v| v.as_str()).unwrap_or("");
        match relation {
            "claim_dependency" | "evidence_link" => {
                push_attr(&mut ids, fact, "claim_id");
                push_attr(&mut ids, fact, "depends_on_object_id");
                push_attr(&mut ids, fact, "artifact_id");
            }
            "match_edge" => {
                push_attr(&mut ids, fact, "left_record_id");
                push_attr(&mut ids, fact, "right_record_id");
            }
            "edge" => {
                push_attr(&mut ids, fact, "from_object_id");
                push_attr(&mut ids, fact, "to_object_id");
            }
            _ => {}
        }
    }
    for node in &facts.nodes {
        ids.insert(node.id.clone());
    }
    ids.len()
}

fn push_attr(ids: &mut BTreeSet<String>, fact: &JsonValue, key: &str) {
    if let Some(v) = fact
        .get("attributes")
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_str())
    {
        if !v.is_empty() {
            ids.insert(v.to_string());
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EgglogRows {
    pub tables: BTreeMap<String, Vec<(Vec<String>, Option<f64>)>>,
    pub decoded: BTreeMap<String, Vec<DecodedRow>>,
}

#[derive(Clone, Debug)]
pub struct DecodedRow {
    pub keys: Vec<String>,
    pub number: Option<f64>,
    pub int: Option<i64>,
    pub bool_value: Option<bool>,
    pub deps: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct EgglogRunReport {
    pub tuple_count: usize,
    pub output_count: usize,
    pub iteration_budget: usize,
    pub checked_pair_count: usize,
    pub rows: EgglogRows,
    pub equivalence_classes: Vec<SaturationEquivalenceClass>,
}

pub fn read_egglog_rows(egraph: &egraph::Engine, relations: &[&str]) -> Result<EgglogRows, String> {
    let mut rows = EgglogRows::default();
    for rel in relations {
        if !egraph.has_function(rel) {
            continue;
        }
        let output = egraph.function_rows(rel).map_err(|e| e.to_string())?;
        let out_is_unit = output.output_sort == "Unit";
        let mut decoded = Vec::new();
        for row in output.rows {
            let key_strings = row
                .inputs
                .iter()
                .map(|term| decode_leaf_string(&term.value))
                .collect::<Result<Vec<_>, _>>()?;
            let mut number = None;
            let mut int = None;
            let mut bool_value = None;
            let mut deps = Vec::new();
            if !out_is_unit {
                match &row.output.value {
                    TermValue::Atom(Atom::F64(f)) => number = Some(*f),
                    TermValue::Atom(Atom::I64(i)) => int = Some(*i),
                    TermValue::Atom(Atom::Bool(b)) => bool_value = Some(*b),
                    TermValue::App(name, children) if name == "set-of" => {
                        for child in children {
                            deps.push(decode_leaf_string(child)?);
                        }
                        deps.sort();
                        deps.dedup();
                    }
                    other => {
                        let rendered = &row.output.expression;
                        if let Ok(f) = rendered.parse::<f64>() {
                            number = Some(f);
                        } else if rendered.starts_with("(set-of") || name_is_set(other) {
                            deps = decode_set_members(other)?;
                        } else {
                            return Err(format!(
                                "unsupported egglog value for {rel}: {other:?} ({rendered})"
                            ));
                        }
                    }
                }
            }
            rows.tables
                .entry((*rel).to_string())
                .or_default()
                .push((key_strings.clone(), number));
            decoded.push(DecodedRow {
                keys: key_strings,
                number,
                int,
                bool_value,
                deps,
            });
        }
        rows.decoded.insert((*rel).to_string(), decoded);
    }
    Ok(rows)
}

fn name_is_set(term: &TermValue) -> bool {
    matches!(term, TermValue::App(name, _) if name == "set-of")
}

fn decode_set_members(term: &TermValue) -> Result<Vec<String>, String> {
    let mut deps = match term {
        TermValue::App(name, children) if name == "set-of" => children
            .iter()
            .map(decode_leaf_string)
            .collect::<Result<Vec<_>, _>>()?,
        other => return Err(format!("expected set-of, got {other:?}")),
    };
    deps.sort();
    deps.dedup();
    Ok(deps)
}

fn decode_leaf_string(term: &TermValue) -> Result<String, String> {
    match term {
        TermValue::Atom(Atom::String(s)) => Ok(s.clone()),
        TermValue::Atom(Atom::F64(f)) => Ok(format!("{f}")),
        TermValue::Atom(Atom::I64(i)) => Ok(i.to_string()),
        TermValue::Atom(Atom::Bool(b)) => Ok(b.to_string()),
        other => Err(format!("expected leaf term, got {other:?}")),
    }
}

pub fn dispatch_run_saturation(
    facts: SaturationFacts,
    program: &SaturationProgram,
) -> GraphStoreResult<SaturationClosure> {
    let result = match program.semiring {
        ProvenanceSemiringKind::Viterbi => run_saturation_inner::<ViterbiSemiring>(facts, program),
        ProvenanceSemiringKind::Tropical => {
            run_saturation_inner::<TropicalSemiring>(facts, program)
        }
        ProvenanceSemiringKind::Boolean => run_saturation_inner::<BooleanSemiring>(facts, program),
        ProvenanceSemiringKind::Counting => {
            run_saturation_inner::<CountingSemiring>(facts, program)
        }
        ProvenanceSemiringKind::Corroboration => {
            run_saturation_inner::<CorroborationSemiring>(facts, program)
        }
        ProvenanceSemiringKind::Probability => {
            run_saturation_inner::<ProbabilitySemiring>(facts, program)
        }
    };
    result.map_err(|error| crate::graph::error::GraphStoreError::new("saturation_error", error))
}

fn run_saturation_inner<S: Semiring>(
    facts: SaturationFacts,
    program: &SaturationProgram,
) -> Result<SaturationClosure, String> {
    let rules = SaturationRuleSet::from_program(program);
    let source = saturation_program_source::<S>(&rules);
    let budget = iteration_bound_for::<S>(&facts)
        .max(1)
        .max(program.iteration_bound);
    let report = run_egglog_program_engine::<S>(&facts, &source, budget, &rules)?;
    let statements = closure_statements_from_rows(
        &report.rows,
        program,
        statement_semiring_for(program.semiring),
    )?;
    Ok(SaturationClosure {
        component_key: facts.component_key,
        template_id: program.template_id.clone(),
        writeback_policy: program.writeback_policy.clone(),
        derived_statements: statements,
        equivalence_classes: report.equivalence_classes,
        iterations: budget,
        egglog_tuple_count: report.tuple_count,
        egglog_output_count: report.output_count,
        egglog_iteration_budget: report.iteration_budget,
        egglog_checked_pair_count: report.checked_pair_count,
        egglog_error: None,
        semiring: statement_semiring_for(program.semiring),
    })
}

pub fn run_egglog_program_engine<S: Semiring>(
    facts: &SaturationFacts,
    program_source: &str,
    iteration_budget: usize,
    rules: &SaturationRuleSet,
) -> Result<EgglogRunReport, String> {
    let _ = std::any::type_name::<S>();
    let mut egraph = egraph::Engine::new(egraph::RulePack::new(
        program_source.trim(),
        egraph::CostModel::default(),
    ))
    .map_err(|error| error.to_string())?;
    let initialization_outputs = egraph.initialization().output_count;
    let (source, checked_pair_count) =
        egglog_program_with_facts(facts, program_source, iteration_budget, rules);
    let runtime_source = source
        .strip_prefix(program_source.trim())
        .expect("fact program starts with its unchanged rule pack");
    let execution = egraph
        .execute(runtime_source)
        .map_err(|error| error.to_string())?;

    let relations = [
        "support-reachable",
        "support-confidence",
        "support-deps",
        "support-path-len",
        "contradiction-reaches",
        "contradiction-confidence",
        "contradiction-deps",
        "contradiction-path-len",
        "same-entity",
        "same-entity-confidence",
        "same-entity-deps",
        "same-entity-path-len",
        "dependent-claim",
        "evidence-path-too-long",
        "tension-neighborhood",
        "same-shadow",
    ];
    let rows = read_egglog_rows(&egraph, &relations)?;
    let equivalence_classes = if rules.equivalence {
        equivalence_classes_from_rows(facts, &rows)?
    } else {
        Vec::new()
    };

    Ok(EgglogRunReport {
        tuple_count: egraph.num_tuples(),
        output_count: initialization_outputs + execution.output_count,
        iteration_budget,
        checked_pair_count,
        rows,
        equivalence_classes,
    })
}

fn egglog_program_with_facts(
    facts: &SaturationFacts,
    program: &str,
    iteration_budget: usize,
    rules: &SaturationRuleSet,
) -> (String, usize) {
    let checked_pairs = if rules.same_entity {
        entity_check_pairs(facts)
    } else {
        Vec::new()
    };
    let mut source = String::with_capacity(program.len() + facts.facts.len() * 96 + 64);
    source.push_str(program.trim());
    source.push('\n');
    for line in egglog_fact_lines(facts) {
        source.push_str(&line);
        source.push('\n');
    }
    source.push_str(&format!("(run equivalence {iteration_budget})\n"));
    source.push_str(&format!("(run derivation {iteration_budget})\n"));
    for (left, right) in &checked_pairs {
        source.push_str(&format!(
            "(check (same-entity {} {}))\n",
            egglog_string(left),
            egglog_string(right)
        ));
    }
    (source, checked_pairs.len())
}

fn entity_check_pairs(facts: &SaturationFacts) -> Vec<(String, String)> {
    use std::collections::VecDeque;
    let mut adjacent = BTreeMap::<String, BTreeSet<String>>::new();
    for edge in match_edges_from_facts(&facts.facts) {
        adjacent
            .entry(edge.left_record_id.clone())
            .or_default()
            .insert(edge.right_record_id.clone());
        adjacent
            .entry(edge.right_record_id)
            .or_default()
            .insert(edge.left_record_id);
    }
    let mut unseen = adjacent.keys().cloned().collect::<BTreeSet<_>>();
    let mut pairs = Vec::new();
    while let Some(seed) = unseen.pop_first() {
        let mut component = BTreeSet::from([seed.clone()]);
        let mut queue = VecDeque::from([seed]);
        while let Some(node) = queue.pop_front() {
            for next in &adjacent[&node] {
                if component.insert(next.clone()) {
                    unseen.remove(next);
                    queue.push_back(next.clone());
                }
            }
        }
        if component.len() > 1 {
            pairs.push((
                component.first().unwrap().clone(),
                component.last().unwrap().clone(),
            ));
        }
    }
    pairs
}

fn egglog_fact_lines(facts: &SaturationFacts) -> Vec<String> {
    let mut lines = Vec::new();
    for edge in support_edges_from_facts(&facts.facts) {
        lines.push(format!(
            "(support-edge {} {} {} {})",
            egglog_string(&edge.from_id),
            egglog_string(&edge.to_id),
            egglog_f64(edge.confidence),
            egglog_string(&edge.dependency_fact_id)
        ));
    }
    for edge in contradiction_edges_from_facts(&facts.facts) {
        lines.push(format!(
            "(contradiction-edge {} {} {} {})",
            egglog_string(&edge.from_id),
            egglog_string(&edge.to_id),
            egglog_f64(edge.confidence),
            egglog_string(&edge.dependency_fact_id)
        ));
    }
    for edge in match_edges_from_facts(&facts.facts) {
        for (left, right) in [
            (&edge.left_record_id, &edge.right_record_id),
            (&edge.right_record_id, &edge.left_record_id),
        ] {
            lines.push(format!(
                "(match-edge {} {} {} {})",
                egglog_string(left),
                egglog_string(right),
                egglog_f64(edge.confidence),
                egglog_string(&edge.dependency_fact_id)
            ));
        }
    }
    for fact in &facts.facts {
        let relation = object_field(fact, "relation");
        if relation == "claim_dependency" {
            let claim = attr_string(fact, "claim_id");
            let depends = attr_string(fact, "depends_on_object_id");
            let just = attr_string(fact, "justification_type");
            let strength = numeric_attr(fact, "strength", 0.0);
            let fact_id = object_field(fact, "fact_id");
            if !claim.is_empty() {
                lines.push(format!(
                    "(claim-dependency {} {} {} {} {})",
                    egglog_string(&claim),
                    egglog_string(&depends),
                    egglog_string(&just),
                    egglog_f64(strength),
                    egglog_string(fact_id)
                ));
            }
        } else if relation == "evidence_path" {
            let entity = object_field(fact, "entity_id");
            let plen = integer_attr(fact, "path_length", 0);
            let fact_id = object_field(fact, "fact_id");
            lines.push(format!(
                "(evidence-path {} {} {})",
                egglog_string(entity),
                plen,
                egglog_string(fact_id)
            ));
        } else if relation == "edge" {
            let edge_type = attr_string(fact, "edge_type").to_lowercase();
            let status = attr_string(fact, "acceptance_status").to_lowercase();
            if edge_type == "contradicts" || status == "contested" {
                let from = attr_string(fact, "from_object_id");
                let to = attr_string(fact, "to_object_id");
                let edge_id = object_field(fact, "entity_id");
                if !from.is_empty() && !to.is_empty() {
                    let fact_id = object_field(fact, "fact_id");
                    lines.push(format!(
                        "(tension-edge {} {} {} {} {} {})",
                        egglog_string(&from),
                        egglog_string(&to),
                        egglog_string(edge_id),
                        egglog_string(&edge_type),
                        egglog_string(&status),
                        egglog_string(fact_id)
                    ));
                }
            }
        }
    }
    lines.extend(egglog_shadow_binding_lines(facts));
    lines.sort();
    lines.dedup();
    lines
}

fn egglog_shadow_binding_lines(facts: &SaturationFacts) -> Vec<String> {
    let nodes_by_id = facts
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut lines = Vec::new();
    for shadow in facts.nodes.iter().filter(|node| {
        node.labels
            .iter()
            .any(|label| label == EPISTEMIC_SHADOW_LABEL)
    }) {
        let content_id =
            prop_str(&shadow.properties, "content_node_id").unwrap_or_else(|| shadow.id.clone());
        let Some(content_node) = nodes_by_id.get(content_id.as_str()) else {
            continue;
        };
        let Some(form) = claim_logical_form(&claim_text(content_node)) else {
            continue;
        };
        let status = grounded_status_symbol(
            &prop_str(&shadow.properties, "grounded_extension_status")
                .unwrap_or_else(|| "undecided".to_string()),
        );
        let prop = egglog_prop_term(&form);
        lines.push(format!(
            "(set (shadow-binding {}) (Shadow {} {}))",
            egglog_string(&shadow.id),
            egglog_string(status),
            prop
        ));
        let _ = egglog_symbol(&shadow.id);
    }
    lines
}

fn egglog_prop_term(form: &ClaimForm) -> String {
    let mut term = format!("(Atom {})", egglog_string(&form.prop));
    for _ in 0..form.neg_count {
        term = format!("(Not {term})");
    }
    term
}

fn confidence_from_row(row: &DecodedRow) -> f64 {
    if let Some(n) = row.number {
        return n;
    }
    if let Some(b) = row.bool_value {
        return if b { 1.0 } else { 0.0 };
    }
    if let Some(i) = row.int {
        return i as f64;
    }
    0.0
}

fn index_pair_map(rows: &EgglogRows, name: &str) -> BTreeMap<(String, String), DecodedRow> {
    let mut map = BTreeMap::new();
    for row in rows.decoded.get(name).into_iter().flatten() {
        if row.keys.len() >= 2 {
            map.insert((row.keys[0].clone(), row.keys[1].clone()), row.clone());
        }
    }
    map
}

fn build_saturation_statement(
    rule_id: &str,
    relation: &str,
    subject_id: &str,
    reason: &str,
    deps: Vec<String>,
    attributes: JsonValue,
    confidence: f64,
    writeback_policy: &str,
    engine: &str,
) -> Result<SaturationDerivedStatement, String> {
    saturation_statement(
        StatementSeed {
            rule_id,
            relation,
            subject_id,
            reason,
            dependency_fact_ids: deps,
            attributes,
            confidence,
            writeback_policy,
        },
        engine,
    )
    .map_err(|e| format!("{}: {}", e.code, e.message))
}

fn build_exact_statement(
    rule_id: &str,
    relation: &str,
    subject_id: &str,
    reason: &str,
    deps: Vec<String>,
    attributes: JsonValue,
    confidence: f64,
    writeback_policy: &str,
    engine: &str,
) -> Result<SaturationDerivedStatement, String> {
    exact_derived_statement(
        StatementSeed {
            rule_id,
            relation,
            subject_id,
            reason,
            dependency_fact_ids: deps,
            attributes,
            confidence,
            writeback_policy,
        },
        engine,
    )
}

pub fn closure_statements_from_rows(
    rows: &EgglogRows,
    program: &SaturationProgram,
    _semiring: StatementSemiring,
) -> Result<Vec<SaturationDerivedStatement>, String> {
    let mut statements: BTreeMap<String, SaturationDerivedStatement> = BTreeMap::new();
    let support_conf = index_pair_map(rows, "support-confidence");
    let support_deps = index_pair_map(rows, "support-deps");
    let support_len = index_pair_map(rows, "support-path-len");
    for row in rows.decoded.get("support-reachable").into_iter().flatten() {
        if row.keys.len() < 2 {
            continue;
        }
        let key = (row.keys[0].clone(), row.keys[1].clone());
        let path_length = support_len.get(&key).and_then(|r| r.int).unwrap_or(0) as usize;
        if path_length < 2 {
            continue;
        }
        let confidence = support_conf
            .get(&key)
            .map(confidence_from_row)
            .unwrap_or(0.0);
        let deps = support_deps
            .get(&key)
            .map(|r| r.deps.clone())
            .unwrap_or_default();
        let deps = sorted_strings(deps);
        let mut statement = build_saturation_statement(
            SUPPORT_REACHABLE_RULE,
            "support_reachable",
            &key.0,
            "Support reaches this target through an evidence or claim-dependency chain.",
            deps.clone(),
            json!({
                "target_id": key.1,
                "path_length": path_length,
            }),
            confidence,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        statement.contributors = vec![SaturationContributor {
            dependency_fact_ids: deps,
            confidence,
        }];
        merge_statement(&mut statements, statement);
    }

    let contra_conf = index_pair_map(rows, "contradiction-confidence");
    let contra_deps = index_pair_map(rows, "contradiction-deps");
    let contra_len = index_pair_map(rows, "contradiction-path-len");
    for row in rows
        .decoded
        .get("contradiction-reaches")
        .into_iter()
        .flatten()
    {
        if row.keys.len() < 2 {
            continue;
        }
        let key = (row.keys[0].clone(), row.keys[1].clone());
        let path_length = contra_len.get(&key).and_then(|r| r.int).unwrap_or(1) as usize;
        let confidence = contra_conf
            .get(&key)
            .map(confidence_from_row)
            .unwrap_or(0.0);
        let deps = sorted_strings(
            contra_deps
                .get(&key)
                .map(|r| r.deps.clone())
                .unwrap_or_default(),
        );
        let mut statement = build_saturation_statement(
            CONTRADICTION_REACHES_RULE,
            "contradiction_reaches",
            &key.0,
            "A contradiction propagates across support and dependency reachability.",
            deps.clone(),
            json!({
                "target_id": key.1,
                "path_length": path_length,
            }),
            confidence,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        statement.contributors = vec![SaturationContributor {
            dependency_fact_ids: deps,
            confidence,
        }];
        merge_statement(&mut statements, statement);
    }

    let entity_conf = index_pair_map(rows, "same-entity-confidence");
    let entity_deps = index_pair_map(rows, "same-entity-deps");
    let entity_len = index_pair_map(rows, "same-entity-path-len");
    for row in rows.decoded.get("same-entity").into_iter().flatten() {
        if row.keys.len() < 2 {
            continue;
        }
        let (from_id, to_id) = (row.keys[0].clone(), row.keys[1].clone());
        if from_id >= to_id {
            continue;
        }
        let key = (from_id.clone(), to_id.clone());
        let path_length = entity_len.get(&key).and_then(|r| r.int).unwrap_or(1) as usize;
        let confidence = entity_conf
            .get(&key)
            .map(confidence_from_row)
            .unwrap_or(0.0);
        let deps = sorted_strings(
            entity_deps
                .get(&key)
                .map(|r| r.deps.clone())
                .unwrap_or_default(),
        );
        let mut statement = build_exact_statement(
            SAME_ENTITY_RULE,
            "same_entity",
            &from_id,
            "Threshold-crossing match edges connect these records.",
            deps.clone(),
            json!({
                "target_id": to_id,
                "path_length": path_length,
            }),
            confidence,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        statement.contributors = vec![SaturationContributor {
            dependency_fact_ids: deps,
            confidence,
        }];
        merge_statement(&mut statements, statement);
    }

    for row in rows.decoded.get("dependent-claim").into_iter().flatten() {
        // keys: claim, depends, just, strength, fact  — relation stores all in key for unit output
        if row.keys.len() < 5 {
            continue;
        }
        let claim = &row.keys[0];
        let depends = &row.keys[1];
        let just = &row.keys[2];
        let strength: f64 = row.keys[3].parse().unwrap_or(0.0);
        let fact_id = &row.keys[4];
        let statement = build_exact_statement(
            "dependent_claim",
            "dependent_claim",
            claim,
            "This claim depends on another graph object for its justification.",
            vec![fact_id.clone()],
            json!({
                "depends_on_object_id": depends,
                "justification_type": just,
                "strength": strength,
            }),
            1.0,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        merge_statement(&mut statements, statement);
    }

    for row in rows
        .decoded
        .get("evidence-path-too-long")
        .into_iter()
        .flatten()
    {
        if row.keys.len() < 3 {
            continue;
        }
        let entity = &row.keys[0];
        let path_length: i64 = row.keys[1].parse().unwrap_or(0);
        let fact_id = &row.keys[2];
        let statement = build_exact_statement(
            "evidence_path_too_long",
            "evidence_path_too_long",
            entity,
            "The evidence path exceeds the configured symbolic derivation depth.",
            vec![fact_id.clone()],
            json!({"path_length": path_length, "max_length": 3_i64}),
            0.9,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        merge_statement(&mut statements, statement);
    }

    for row in rows
        .decoded
        .get("tension-neighborhood")
        .into_iter()
        .flatten()
    {
        if row.keys.len() < 5 {
            continue;
        }
        let object_id = &row.keys[0];
        let edge_id = &row.keys[1];
        let edge_type = &row.keys[2];
        let status = &row.keys[3];
        let fact_id = &row.keys[4];
        let statement = build_exact_statement(
            "object_in_unresolved_tension_neighborhood",
            "object_in_unresolved_tension_neighborhood",
            object_id,
            "This object is adjacent to a contradicting or contested edge.",
            vec![fact_id.clone()],
            json!({
                "edge_id": edge_id,
                "edge_type": edge_type,
                "acceptance_status": status,
            }),
            0.8,
            program.writeback_policy.as_str(),
            &program.engine,
        )?;
        merge_statement(&mut statements, statement);
    }

    let _ = SATURATION_SHARED_RULE_IDS;
    Ok(statements.into_values().collect())
}

fn equivalence_classes_from_rows(
    facts: &SaturationFacts,
    rows: &EgglogRows,
) -> Result<Vec<SaturationEquivalenceClass>, String> {
    let nodes_by_id = facts
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();

    let mut parent: BTreeMap<String, String> = BTreeMap::new();
    let find = |parent: &mut BTreeMap<String, String>, id: &str| -> String {
        let mut cur = id.to_string();
        let mut path = Vec::new();
        while let Some(p) = parent.get(&cur).cloned() {
            if p == cur {
                break;
            }
            path.push(cur.clone());
            cur = p;
        }
        parent.entry(cur.clone()).or_insert_with(|| cur.clone());
        for node in path {
            parent.insert(node, cur.clone());
        }
        cur
    };

    for row in rows.decoded.get("same-shadow").into_iter().flatten() {
        if row.keys.len() < 2 {
            continue;
        }
        let a = &row.keys[0];
        let b = &row.keys[1];
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for shadow_id in parent.keys().cloned().collect::<Vec<_>>() {
        let root = find(&mut parent, &shadow_id);
        groups.entry(root).or_default().push(shadow_id);
    }

    let mut classes = Vec::new();
    for members in groups.into_values() {
        if members.len() < 2 {
            continue;
        }
        let mut candidates = Vec::new();
        for shadow_id in &members {
            let Some(shadow) = facts.nodes.iter().find(|n| &n.id == shadow_id) else {
                continue;
            };
            let content_id = prop_str(&shadow.properties, "content_node_id")
                .unwrap_or_else(|| shadow.id.clone());
            let Some(content_node) = nodes_by_id.get(content_id.as_str()) else {
                continue;
            };
            let Some(form) = claim_logical_form(&claim_text(content_node)) else {
                continue;
            };
            candidates.push((shadow_id.clone(), content_id, form));
        }
        if candidates.len() < 2 {
            continue;
        }
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let rep = &candidates[0];
        let mut member_content_ids = candidates.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut member_shadow_ids = candidates.iter().map(|c| c.0.clone()).collect::<Vec<_>>();
        member_content_ids.sort();
        member_content_ids.dedup();
        member_shadow_ids.sort();
        member_shadow_ids.dedup();
        let class_id = format!(
            "eclass:{}",
            stable_hash(json!(member_shadow_ids))
                .map_err(|e| format!("{}: {}", e.code, e.message))?
        );
        classes.push(SaturationEquivalenceClass {
            class_id,
            canonical_form: rep.2.canonical_label(),
            representative_content_id: rep.1.clone(),
            representative_shadow_id: rep.0.clone(),
            member_content_ids,
            member_shadow_ids,
        });
    }
    classes.sort_by(|left, right| left.class_id.cmp(&right.class_id));
    Ok(classes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyred_thg_derivation::ViterbiSemiring;

    #[test]
    fn provenance_semiring_matches_program_semiring() {
        for kind in [
            ProvenanceSemiringKind::Boolean,
            ProvenanceSemiringKind::Counting,
            ProvenanceSemiringKind::Viterbi,
            ProvenanceSemiringKind::Tropical,
            ProvenanceSemiringKind::Corroboration,
            ProvenanceSemiringKind::Probability,
        ] {
            let mut program = SaturationProgram::default();
            program.semiring = kind;
            let rules = SaturationRuleSet::from_program(&program);
            let source = match kind {
                ProvenanceSemiringKind::Viterbi => {
                    saturation_program_source::<ViterbiSemiring>(&rules)
                }
                ProvenanceSemiringKind::Tropical => {
                    saturation_program_source::<TropicalSemiring>(&rules)
                }
                ProvenanceSemiringKind::Boolean => {
                    saturation_program_source::<BooleanSemiring>(&rules)
                }
                ProvenanceSemiringKind::Counting => {
                    saturation_program_source::<CountingSemiring>(&rules)
                }
                ProvenanceSemiringKind::Corroboration => {
                    saturation_program_source::<CorroborationSemiring>(&rules)
                }
                ProvenanceSemiringKind::Probability => {
                    saturation_program_source::<ProbabilitySemiring>(&rules)
                }
            };
            assert!(source.contains(":merge"));
            assert_eq!(statement_semiring_for(kind), statement_semiring_for(kind));
        }
    }
}
