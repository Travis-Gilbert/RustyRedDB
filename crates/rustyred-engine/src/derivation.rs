//! First-class derivations, assumptions, and semiring folds.
//!
//! Derivations are graph objects rather than metadata embedded in one feature.
//! This lets beliefs, skills, calibration, and diagnosis share one provenance
//! and retraction mechanism.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::graph::capability::{GraphAtomic, GraphResidency, GraphStore};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::query::{NeighborHit, NeighborQuery};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::saturation::SaturationDerivedStatement;
use crate::state::stable_hash;
use crate::stream::StreamEvent;

pub const DERIVATION_LABEL: &str = "Derivation";
pub const ASSUMPTION_LABEL: &str = "Assumption";
pub const OPERATOR_LABEL: &str = "Operator";
pub const VALUE_LABEL: &str = "Value";
pub const NOGOOD_LABEL: &str = "Nogood";
pub const CONSUMES_EDGE: &str = "CONSUMES";
pub const PRODUCES_EDGE: &str = "PRODUCES";
pub const VIA_EDGE: &str = "VIA";
pub const UNDER_EDGE: &str = "UNDER";
pub const CONFLICTS_WITH_EDGE: &str = "CONFLICTS_WITH";
// Compatibility vocabulary for the original derivation crate. New graph
// records use the typed consumes/produces/via/under shape above; these names
// remain public so existing folds and traces continue to read legacy graphs.
pub const DERIVATION_ANCESTOR_EDGE: &str = "derivation_ancestor";
pub const DERIVATION_ASSUMES_EDGE: &str = "derivation_assumes";
pub const ANNOTATION_PROPERTY: &str = "annotation";
pub const SEMIRING_KIND_PROPERTY: &str = "semiring_kind";
pub const FOLD_VALUE_PROPERTY: &str = "fold_value";
pub const MAX_DERIVATION_DEPTH: usize = 256;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivationKind {
    Functional,
    Observational,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DerivationAnnotation {
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub cost: f64,
}

impl Default for DerivationAnnotation {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            cost: 0.0,
        }
    }
}

fn default_confidence() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DerivationSpec {
    pub kind: DerivationKind,
    pub operator_id: String,
    #[serde(default)]
    pub input_refs: Vec<String>,
    #[serde(default)]
    pub output_refs: Vec<String>,
    #[serde(default)]
    pub assumption_refs: Vec<String>,
    #[serde(default)]
    pub params: Value,
    pub environment_version: String,
    #[serde(default)]
    pub annotation: DerivationAnnotation,
}

impl DerivationSpec {
    pub fn content_id(&self) -> GraphStoreResult<String> {
        #[derive(Serialize)]
        struct Identity<'a> {
            operator_id: &'a str,
            input_refs: Vec<&'a str>,
            params: Value,
            environment_version: &'a str,
        }

        let mut input_refs = self
            .input_refs
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        input_refs.sort_unstable();
        input_refs.dedup();
        Ok(format!(
            "derivation:{}",
            stable_hash(Identity {
                operator_id: &self.operator_id,
                input_refs,
                params: canonical_json(&self.params),
                environment_version: &self.environment_version,
            })?
        ))
    }

    fn normalized(mut self) -> Self {
        normalize_refs(&mut self.input_refs);
        normalize_refs(&mut self.output_refs);
        normalize_refs(&mut self.assumption_refs);
        self.annotation.confidence = self.annotation.confidence.clamp(0.0, 1.0);
        self.annotation.cost = self.annotation.cost.max(0.0);
        self.params = canonical_json(&self.params);
        self
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DerivationRecord {
    pub id: String,
    #[serde(flatten)]
    pub spec: DerivationSpec,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssumptionRecord {
    pub id: String,
    pub name: String,
    pub active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NogoodRecord {
    pub id: String,
    pub assumption_refs: Vec<String>,
    pub check_ref: String,
    pub reason: String,
    pub created_at: String,
}

impl NogoodRecord {
    pub fn new(
        assumption_refs: impl IntoIterator<Item = impl Into<String>>,
        check_ref: impl Into<String>,
        reason: impl Into<String>,
        created_at: impl Into<String>,
    ) -> GraphStoreResult<Self> {
        let mut assumption_refs = assumption_refs
            .into_iter()
            .map(Into::into)
            .collect::<Vec<String>>();
        normalize_refs(&mut assumption_refs);
        let check_ref = check_ref.into();
        let reason = reason.into();
        let created_at = created_at.into();
        let id = format!(
            "nogood:{}",
            stable_hash((&assumption_refs, &check_ref, &reason))?
        );
        Ok(Self {
            id,
            assumption_refs,
            check_ref,
            reason,
            created_at,
        })
    }
}

impl AssumptionRecord {
    pub fn active(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            active: true,
            scope: None,
        }
    }

    pub fn into_node(self) -> NodeRecord {
        NodeRecord::new(
            self.id,
            [ASSUMPTION_LABEL],
            json!({
                "name": self.name,
                "active": self.active,
                "scope": self.scope,
            }),
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "node_type", rename_all = "snake_case")]
pub enum WhyTree {
    ValueLeaf {
        value_id: String,
    },
    Derived {
        value_id: String,
        derivations: Vec<WhyDerivation>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WhyDerivation {
    pub derivation_id: String,
    pub kind: DerivationKind,
    pub operator_id: String,
    pub assumptions: Vec<WhyAssumption>,
    pub inputs: Vec<WhyTree>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WhyAssumption {
    pub id: String,
    pub active: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BooleanProvenance {
    /// Minimal DNF: alternatives are terms; each term is a conjunction of
    /// assumption ids. The empty set of terms is false.
    pub terms: BTreeSet<BTreeSet<String>>,
}

impl BooleanProvenance {
    pub fn holds(&self) -> bool {
        !self.terms.is_empty()
    }

    fn minimized(mut self) -> Self {
        let terms = self.terms.iter().cloned().collect::<Vec<_>>();
        self.terms.retain(|candidate| {
            !terms
                .iter()
                .any(|other| other != candidate && other.is_subset(candidate))
        });
        self
    }
}

pub trait DerivationSemiring {
    type Element: Clone;

    fn zero(&self) -> Self::Element;
    fn one(&self) -> Self::Element;
    fn annotation(&self, annotation: &DerivationAnnotation) -> Self::Element;
    fn assumption(&self, assumption_id: &str, active: bool) -> Self::Element;
    fn combine(&self, left: Self::Element, right: Self::Element) -> Self::Element;
    fn aggregate(&self, left: Self::Element, right: Self::Element) -> Self::Element;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BooleanSemiring;

impl DerivationSemiring for BooleanSemiring {
    type Element = BooleanProvenance;

    fn zero(&self) -> Self::Element {
        BooleanProvenance::default()
    }

    fn one(&self) -> Self::Element {
        BooleanProvenance {
            terms: [BTreeSet::new()].into_iter().collect(),
        }
    }

    fn annotation(&self, _annotation: &DerivationAnnotation) -> Self::Element {
        self.one()
    }

    fn assumption(&self, assumption_id: &str, active: bool) -> Self::Element {
        if !active {
            return self.zero();
        }
        BooleanProvenance {
            terms: [[assumption_id.to_string()].into_iter().collect()]
                .into_iter()
                .collect(),
        }
    }

    fn combine(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        let mut terms = BTreeSet::new();
        for left_term in &left.terms {
            for right_term in &right.terms {
                terms.insert(left_term.union(right_term).cloned().collect());
            }
        }
        BooleanProvenance { terms }.minimized()
    }

    fn aggregate(&self, mut left: Self::Element, right: Self::Element) -> Self::Element {
        left.terms.extend(right.terms);
        left.minimized()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CountSemiring;

impl DerivationSemiring for CountSemiring {
    type Element = u64;

    fn zero(&self) -> Self::Element {
        0
    }

    fn one(&self) -> Self::Element {
        1
    }

    fn annotation(&self, _annotation: &DerivationAnnotation) -> Self::Element {
        1
    }

    fn assumption(&self, _assumption_id: &str, active: bool) -> Self::Element {
        u64::from(active)
    }

    fn combine(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        left.saturating_mul(right)
    }

    fn aggregate(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        left.saturating_add(right)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TropicalSemiring;

impl DerivationSemiring for TropicalSemiring {
    type Element = f64;

    fn zero(&self) -> Self::Element {
        f64::INFINITY
    }

    fn one(&self) -> Self::Element {
        0.0
    }

    fn annotation(&self, annotation: &DerivationAnnotation) -> Self::Element {
        annotation.cost
    }

    fn assumption(&self, _assumption_id: &str, active: bool) -> Self::Element {
        if active {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn combine(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        left + right
    }

    fn aggregate(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        left.min(right)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProbabilitySemiring;

impl DerivationSemiring for ProbabilitySemiring {
    type Element = f64;

    fn zero(&self) -> Self::Element {
        0.0
    }

    fn one(&self) -> Self::Element {
        1.0
    }

    fn annotation(&self, annotation: &DerivationAnnotation) -> Self::Element {
        annotation.confidence
    }

    fn assumption(&self, _assumption_id: &str, active: bool) -> Self::Element {
        if active {
            1.0
        } else {
            0.0
        }
    }

    fn combine(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        (left * right).clamp(0.0, 1.0)
    }

    fn aggregate(&self, left: Self::Element, right: Self::Element) -> Self::Element {
        (1.0 - (1.0 - left) * (1.0 - right)).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivationSemiringKind {
    Boolean,
    Count,
    Tropical,
    Probability,
}

/// Minimal read contract for derivation queries. The owned-record shape lets
/// MCP/database adapters participate without pretending to be mutable stores.
pub trait DerivationGraphReader {
    fn derivation_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>>;
    fn derivation_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>>;
    fn derivation_neighbors(&self, query: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>>;
}

/// Minimal mutable contract for graph-native derivation publication.
///
/// Database and MCP adapters often return owned records and therefore cannot
/// implement the borrowed-record `GraphStore` API. This writer keeps the
/// derivation invariant reusable on those surfaces instead of encouraging each
/// adapter to rebuild derivation nodes and edges independently.
pub trait DerivationGraphWriter: DerivationGraphReader {
    fn derivation_upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<()>;
    fn derivation_upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()>;
    /// Project derivation premises into the incremental SupportedFact index.
    /// GraphStore writers sync; adapter writers no-op.
    fn project_incremental_support(&mut self, record: &DerivationRecord) -> GraphStoreResult<()> {
        let _ = record;
        Ok(())
    }
    fn derivation_insert_node_if_absent(&mut self, node: NodeRecord) -> GraphStoreResult<bool> {
        let _ = node;
        Err(GraphStoreError::new(
            "atomic_create_not_supported",
            "this derivation store does not support atomic node creation",
        ))
    }
    fn derivation_compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<bool> {
        let _ = (id, property, expected, new_value);
        Err(GraphStoreError::new(
            "compare_and_set_not_supported",
            "this derivation store does not support atomic node property updates",
        ))
    }
}

impl<T: GraphStore + ?Sized> DerivationGraphReader for T {
    fn derivation_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        Ok(self.get_node_record(id))
    }

    fn derivation_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        Ok(self.get_edge_record(id))
    }

    fn derivation_neighbors(&self, query: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>> {
        Ok(self.neighbors(query))
    }
}

impl<T: GraphStore + GraphAtomic + ?Sized> DerivationGraphWriter for T {
    fn derivation_upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.upsert_node(node).map(|_| ())
    }

    fn derivation_upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        self.upsert_edge(edge).map(|_| ())
    }

    fn project_incremental_support(&mut self, record: &DerivationRecord) -> GraphStoreResult<()> {
        crate::incremental::bridge::record_derivation_outputs(
            self,
            &record.id,
            &record.spec.input_refs,
            &record.spec.assumption_refs,
            &record.spec.output_refs,
        )
    }

    fn derivation_insert_node_if_absent(&mut self, node: NodeRecord) -> GraphStoreResult<bool> {
        self.insert_node_if_absent(node)
            .map(|write| write.is_some())
    }

    fn derivation_compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<bool> {
        self.compare_and_set_node_property(id, property, expected, new_value)
            .map(|write| write.is_some())
    }
}

pub fn put_assumption<S: GraphStore + ?Sized>(
    store: &mut S,
    assumption: AssumptionRecord,
) -> GraphStoreResult<()> {
    store.upsert_node(assumption.into_node()).map(|_| ())
}

pub fn record_nogood<S: GraphStore + GraphAtomic + ?Sized>(
    store: &mut S,
    nogood: NogoodRecord,
) -> GraphStoreResult<NogoodRecord> {
    if nogood.assumption_refs.is_empty() {
        return Err(GraphStoreError::new(
            "invalid_nogood",
            "nogood requires at least one conflicting assumption",
        ));
    }
    for assumption_ref in &nogood.assumption_refs {
        let assumption = store.get_node_record(assumption_ref).ok_or_else(|| {
            GraphStoreError::new(
                "missing_nogood_assumption",
                format!("nogood assumption {assumption_ref} does not exist"),
            )
        })?;
        if !assumption
            .labels
            .iter()
            .any(|label| label == ASSUMPTION_LABEL)
        {
            return Err(GraphStoreError::new(
                "invalid_nogood_assumption",
                format!("nogood input {assumption_ref} is not an assumption"),
            ));
        }
    }
    store.upsert_node(NodeRecord::new(
        &nogood.id,
        [NOGOOD_LABEL],
        serde_json::to_value(&nogood).map_err(json_error)?,
    ))?;
    for assumption_ref in &nogood.assumption_refs {
        put_relation(
            store,
            &nogood.id,
            CONFLICTS_WITH_EDGE,
            assumption_ref,
            json!({ "check_ref": nogood.check_ref }),
        )?;
    }
    Ok(nogood)
}

pub fn put_derivation(
    store: &mut (impl DerivationGraphWriter + ?Sized),
    spec: DerivationSpec,
) -> GraphStoreResult<DerivationRecord> {
    let spec = spec.normalized();
    validate_spec(&spec)?;
    let id = spec.content_id()?;

    for input_ref in &spec.input_refs {
        require_labeled_node(
            store,
            input_ref,
            VALUE_LABEL,
            "missing_derivation_input",
            "invalid_derivation_input",
            "derivation input",
        )?;
    }
    for output_ref in &spec.output_refs {
        require_labeled_node(
            store,
            output_ref,
            VALUE_LABEL,
            "missing_derivation_output",
            "invalid_derivation_output",
            "derivation output",
        )?;
    }
    for assumption_ref in &spec.assumption_refs {
        require_labeled_node(
            store,
            assumption_ref,
            ASSUMPTION_LABEL,
            "missing_derivation_assumption",
            "invalid_derivation_assumption",
            "derivation assumption",
        )?;
    }

    ensure_node(store, &spec.operator_id, OPERATOR_LABEL, json!({}))?;

    store.derivation_upsert_node(NodeRecord::new(
        &id,
        [DERIVATION_LABEL],
        serde_json::to_value(&spec)
            .map_err(|error| GraphStoreError::new("invalid_derivation", error.to_string()))?,
    ))?;

    for input_ref in &spec.input_refs {
        put_relation(store, &id, CONSUMES_EDGE, input_ref, json!({}))?;
    }
    for output_ref in &spec.output_refs {
        let expected_content_hash = store
            .derivation_node(output_ref)?
            .map(|node| node.content_address())
            .transpose()?;
        put_relation(
            store,
            &id,
            PRODUCES_EDGE,
            output_ref,
            json!({ "expected_content_hash": expected_content_hash }),
        )?;
    }
    put_relation(store, &id, VIA_EDGE, &spec.operator_id, json!({}))?;
    for assumption_ref in &spec.assumption_refs {
        put_relation(store, &id, UNDER_EDGE, assumption_ref, json!({}))?;
    }

    let record = DerivationRecord { id, spec };
    store.project_incremental_support(&record)?;
    Ok(record)
}

pub fn get_derivation(
    store: &(impl DerivationGraphReader + ?Sized),
    derivation_id: &str,
) -> GraphStoreResult<DerivationRecord> {
    let node = store.derivation_node(derivation_id)?.ok_or_else(|| {
        GraphStoreError::new(
            "derivation_not_found",
            format!("derivation {derivation_id} does not exist"),
        )
    })?;
    if !node.labels.iter().any(|label| label == DERIVATION_LABEL) {
        return Err(GraphStoreError::new(
            "not_a_derivation",
            format!("node {derivation_id} is not a derivation"),
        ));
    }
    let spec = serde_json::from_value(node.properties)
        .map_err(|error| GraphStoreError::new("invalid_derivation", error.to_string()))?;
    Ok(DerivationRecord {
        id: derivation_id.to_string(),
        spec,
    })
}

pub fn why(
    store: &(impl DerivationGraphReader + ?Sized),
    value_id: &str,
) -> GraphStoreResult<WhyTree> {
    why_inner(store, value_id, &mut BTreeSet::new())
}

pub fn fold<S: DerivationSemiring>(
    store: &(impl DerivationGraphReader + ?Sized),
    value_id: &str,
    semiring: &S,
) -> GraphStoreResult<S::Element> {
    fold_inner(
        store,
        value_id,
        semiring,
        &mut BTreeMap::new(),
        &mut BTreeSet::new(),
    )
}

pub fn fold_json(
    store: &(impl DerivationGraphReader + ?Sized),
    value_id: &str,
    semiring: DerivationSemiringKind,
) -> GraphStoreResult<Value> {
    match semiring {
        DerivationSemiringKind::Boolean => {
            serde_json::to_value(fold(store, value_id, &BooleanSemiring)?).map_err(json_error)
        }
        DerivationSemiringKind::Count => Ok(json!(fold(store, value_id, &CountSemiring)?)),
        DerivationSemiringKind::Tropical => {
            let cost = fold(store, value_id, &TropicalSemiring)?;
            Ok(if cost.is_finite() {
                json!(cost)
            } else {
                Value::Null
            })
        }
        DerivationSemiringKind::Probability => {
            Ok(json!(fold(store, value_id, &ProbabilitySemiring)?))
        }
    }
}

pub fn retract_assumption(
    store: &mut dyn GraphStore,
    assumption_id: &str,
) -> GraphStoreResult<Vec<String>> {
    let mut node = store.get_node_record(assumption_id).ok_or_else(|| {
        GraphStoreError::new(
            "assumption_not_found",
            format!("assumption {assumption_id} does not exist"),
        )
    })?;
    if !node.labels.iter().any(|label| label == ASSUMPTION_LABEL) {
        return Err(GraphStoreError::new(
            "not_an_assumption",
            format!("node {assumption_id} is not an assumption"),
        ));
    }
    let properties = node.properties.as_object_mut().ok_or_else(|| {
        GraphStoreError::new(
            "invalid_assumption",
            "assumption properties must be an object",
        )
    })?;
    properties.insert("active".to_string(), Value::Bool(false));
    store.upsert_node(node)?;

    // Graph-walk dependents (legacy return shape) plus incremental maintenance.
    let mut dependents = dependent_values(store, assumption_id)?;
    let delta = crate::incremental::bridge::retract_assumption_facts(store, assumption_id)?;
    for removed in delta.removed {
        if !dependents.iter().any(|id| id == &removed) {
            dependents.push(removed);
        }
    }
    dependents.sort();
    dependents.dedup();
    Ok(dependents)
}

pub fn dependent_values(
    store: &(impl DerivationGraphReader + ?Sized),
    assumption_id: &str,
) -> GraphStoreResult<Vec<String>> {
    let mut queue = VecDeque::new();
    let mut seen_derivations = BTreeSet::new();
    let mut values = BTreeSet::new();
    for hit in
        store.derivation_neighbors(NeighborQuery::in_(assumption_id).with_edge_type(UNDER_EDGE))?
    {
        queue.push_back(hit.node_id);
    }
    while let Some(derivation_id) = queue.pop_front() {
        if !seen_derivations.insert(derivation_id.clone()) {
            continue;
        }
        for output in store.derivation_neighbors(
            NeighborQuery::out(&derivation_id).with_edge_type(PRODUCES_EDGE),
        )? {
            if values.insert(output.node_id.clone()) {
                for consumer in store.derivation_neighbors(
                    NeighborQuery::in_(&output.node_id).with_edge_type(CONSUMES_EDGE),
                )? {
                    queue.push_back(consumer.node_id);
                }
            }
        }
    }
    Ok(values.into_iter().collect())
}

pub trait FunctionalDerivationExecutor {
    fn rederive(
        &self,
        derivation: &DerivationRecord,
        inputs: &[NodeRecord],
        output_id: &str,
    ) -> GraphStoreResult<NodeRecord>;
}

pub fn rederive_if_missing<S: GraphStore + GraphResidency + ?Sized>(
    store: &mut S,
    output_id: &str,
    executor: &dyn FunctionalDerivationExecutor,
) -> GraphStoreResult<Option<NodeRecord>> {
    if let Some(node) = store.get_node_record(output_id) {
        return Ok(Some(node));
    }
    let producer = store
        .neighbors(NeighborQuery::in_(output_id).with_edge_type(PRODUCES_EDGE))
        .into_iter()
        .find_map(|hit| get_derivation(store, &hit.node_id).ok())
        .filter(|derivation| derivation.spec.kind == DerivationKind::Functional);
    let Some(derivation) = producer else {
        return Ok(None);
    };
    let inputs = derivation
        .spec
        .input_refs
        .iter()
        .map(|input_id| {
            store.get_node_record(input_id).ok_or_else(|| {
                GraphStoreError::new(
                    "missing_derivation_input",
                    format!("functional derivation input {input_id} is not resident"),
                )
            })
        })
        .collect::<GraphStoreResult<Vec<_>>>()?;
    let node = executor.rederive(&derivation, &inputs, output_id)?;
    if node.id != output_id {
        return Err(GraphStoreError::new(
            "rederive_output_mismatch",
            format!("executor returned {}, expected {output_id}", node.id),
        ));
    }
    let expected_hash = expected_output_hash(store, &derivation.id, output_id);
    if let Some(expected_hash) = expected_hash {
        let actual_hash = node.content_address()?;
        if actual_hash != expected_hash {
            return Err(GraphStoreError::new(
                "rederive_content_mismatch",
                format!("re-derived {output_id} hash {actual_hash} != {expected_hash}"),
            ));
        }
    }
    store.readmit_node(node.clone())?;
    Ok(Some(node))
}

pub fn lift_stream_event<S: GraphStore + GraphAtomic + ?Sized>(
    store: &mut S,
    event: &StreamEvent,
) -> GraphStoreResult<DerivationRecord> {
    let output_id = format!("stream-event:{}", event.id);
    store.upsert_node(NodeRecord::new(
        &output_id,
        [VALUE_LABEL, "StreamEvent"],
        serde_json::to_value(event).map_err(json_error)?,
    ))?;
    put_derivation(
        store,
        DerivationSpec {
            kind: DerivationKind::Observational,
            operator_id: "operator:stream-log-append".to_string(),
            input_refs: Vec::new(),
            output_refs: vec![output_id],
            assumption_refs: Vec::new(),
            params: json!({ "stream_key": event.stream_key, "ordering_token": event.ordering_token }),
            environment_version: "stream-log:v1".to_string(),
            annotation: DerivationAnnotation::default(),
        },
    )
}

pub fn lift_rule_firing<S: GraphStore + GraphAtomic + ?Sized>(
    store: &mut S,
    statement: &SaturationDerivedStatement,
    environment_version: impl Into<String>,
) -> GraphStoreResult<DerivationRecord> {
    let environment_version = environment_version.into();
    let assumption_id = format!(
        "assumption:environment:{}",
        stable_hash((&statement.engine, &environment_version))?
    );
    if store.get_node_record(&assumption_id).is_none() {
        put_assumption(
            store,
            AssumptionRecord {
                id: assumption_id.clone(),
                name: format!("{} environment {}", statement.engine, environment_version),
                active: true,
                scope: Some("environment_version".to_string()),
            },
        )?;
    }
    if store.get_node_record(&statement.fact_id).is_none() {
        store.upsert_node(NodeRecord::new(
            &statement.fact_id,
            [VALUE_LABEL, "DerivedStatement"],
            serde_json::to_value(statement).map_err(json_error)?,
        ))?;
    }
    put_derivation(
        store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: format!("operator:rule:{}", statement.rule_id),
            input_refs: statement.dependency_fact_ids.clone(),
            output_refs: vec![statement.fact_id.clone()],
            assumption_refs: vec![assumption_id],
            params: json!({
                "relation": statement.relation,
                "subject_id": statement.subject_id,
                "attributes": statement.attributes,
                "writeback_policy": statement.writeback_policy,
            }),
            environment_version,
            annotation: DerivationAnnotation {
                confidence: statement.confidence,
                cost: 1.0,
            },
        },
    )
}

fn fold_inner<S: DerivationSemiring>(
    store: &(impl DerivationGraphReader + ?Sized),
    value_id: &str,
    semiring: &S,
    memo: &mut BTreeMap<String, S::Element>,
    visiting: &mut BTreeSet<String>,
) -> GraphStoreResult<S::Element> {
    if let Some(value) = memo.get(value_id) {
        return Ok(value.clone());
    }
    if !visiting.insert(value_id.to_string()) {
        return Err(GraphStoreError::new(
            "cyclic_derivation",
            format!("derivation cycle reaches {value_id}"),
        ));
    }
    let producers = store
        .derivation_neighbors(NeighborQuery::in_(value_id).with_edge_type(PRODUCES_EDGE))?
        .into_iter()
        .filter_map(|hit| get_derivation(store, &hit.node_id).ok())
        .collect::<Vec<_>>();
    let result = if producers.is_empty() {
        semiring.one()
    } else {
        let mut alternatives = semiring.zero();
        for producer in producers {
            let mut path = semiring.annotation(&producer.spec.annotation);
            for assumption_id in &producer.spec.assumption_refs {
                path = semiring.combine(
                    path,
                    semiring.assumption(assumption_id, assumption_active(store, assumption_id)?),
                );
            }
            for input_id in &producer.spec.input_refs {
                let input = fold_inner(store, input_id, semiring, memo, visiting)?;
                path = semiring.combine(path, input);
            }
            alternatives = semiring.aggregate(alternatives, path);
        }
        alternatives
    };
    visiting.remove(value_id);
    memo.insert(value_id.to_string(), result.clone());
    Ok(result)
}

fn why_inner(
    store: &(impl DerivationGraphReader + ?Sized),
    value_id: &str,
    visiting: &mut BTreeSet<String>,
) -> GraphStoreResult<WhyTree> {
    if !visiting.insert(value_id.to_string()) {
        return Err(GraphStoreError::new(
            "cyclic_derivation",
            format!("derivation cycle reaches {value_id}"),
        ));
    }
    let producers = store
        .derivation_neighbors(NeighborQuery::in_(value_id).with_edge_type(PRODUCES_EDGE))?
        .into_iter()
        .filter_map(|hit| get_derivation(store, &hit.node_id).ok())
        .collect::<Vec<_>>();
    if producers.is_empty() {
        visiting.remove(value_id);
        return Ok(WhyTree::ValueLeaf {
            value_id: value_id.to_string(),
        });
    }
    let mut derivations = Vec::with_capacity(producers.len());
    for producer in producers {
        let assumptions = producer
            .spec
            .assumption_refs
            .iter()
            .map(|id| {
                Ok(WhyAssumption {
                    id: id.clone(),
                    active: assumption_active(store, id)?,
                })
            })
            .collect::<GraphStoreResult<Vec<_>>>()?;
        let inputs = producer
            .spec
            .input_refs
            .iter()
            .map(|id| why_inner(store, id, visiting))
            .collect::<GraphStoreResult<Vec<_>>>()?;
        derivations.push(WhyDerivation {
            derivation_id: producer.id,
            kind: producer.spec.kind,
            operator_id: producer.spec.operator_id,
            assumptions,
            inputs,
        });
    }
    visiting.remove(value_id);
    Ok(WhyTree::Derived {
        value_id: value_id.to_string(),
        derivations,
    })
}

fn assumption_active(
    store: &(impl DerivationGraphReader + ?Sized),
    id: &str,
) -> GraphStoreResult<bool> {
    let node = store.derivation_node(id)?.ok_or_else(|| {
        GraphStoreError::new(
            "assumption_not_found",
            format!("assumption {id} does not exist"),
        )
    })?;
    Ok(node
        .properties
        .get("active")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

fn validate_spec(spec: &DerivationSpec) -> GraphStoreResult<()> {
    if spec.operator_id.trim().is_empty() {
        return Err(GraphStoreError::new(
            "invalid_derivation",
            "derivation operator id must not be empty",
        ));
    }
    if spec.output_refs.is_empty() {
        return Err(GraphStoreError::new(
            "invalid_derivation",
            "derivation must produce at least one value",
        ));
    }
    if spec.environment_version.trim().is_empty() {
        return Err(GraphStoreError::new(
            "invalid_derivation",
            "derivation environment version must not be empty",
        ));
    }
    Ok(())
}

fn ensure_node(
    store: &mut (impl DerivationGraphWriter + ?Sized),
    id: &str,
    label: &str,
    properties: Value,
) -> GraphStoreResult<()> {
    if let Some(node) = store.derivation_node(id)? {
        if !node.labels.iter().any(|existing| existing == label) {
            return Err(GraphStoreError::new(
                "invalid_derivation_operator",
                format!("derivation operator {id} is not labeled {label}"),
            ));
        }
    } else {
        store.derivation_upsert_node(NodeRecord::new(id, [label], properties))?;
    }
    Ok(())
}

fn require_labeled_node(
    store: &(impl DerivationGraphReader + ?Sized),
    id: &str,
    required_label: &str,
    missing_code: &str,
    invalid_code: &str,
    role: &str,
) -> GraphStoreResult<()> {
    let node = store
        .derivation_node(id)?
        .ok_or_else(|| GraphStoreError::new(missing_code, format!("{role} {id} does not exist")))?;
    if !node.labels.iter().any(|label| label == required_label) {
        return Err(GraphStoreError::new(
            invalid_code,
            format!("{role} {id} is not labeled {required_label}"),
        ));
    }
    Ok(())
}

fn put_relation(
    store: &mut (impl DerivationGraphWriter + ?Sized),
    from_id: &str,
    edge_type: &str,
    to_id: &str,
    properties: Value,
) -> GraphStoreResult<()> {
    let edge_id = format!(
        "derivation-edge:{}",
        stable_hash((from_id, edge_type, to_id))?
    );
    store.derivation_upsert_edge(EdgeRecord::new(
        edge_id, from_id, edge_type, to_id, properties,
    ))
}

fn expected_output_hash(
    store: &(impl DerivationGraphReader + ?Sized),
    derivation_id: &str,
    output_id: &str,
) -> Option<String> {
    store
        .derivation_neighbors(NeighborQuery::out(derivation_id).with_edge_type(PRODUCES_EDGE))
        .ok()?
        .into_iter()
        .find(|hit| hit.node_id == output_id)
        .and_then(|hit| store.derivation_edge(&hit.edge_id).ok().flatten())
        .and_then(|edge| {
            edge.properties
                .get("expected_content_hash")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn normalize_refs(refs: &mut Vec<String>) {
    refs.retain(|value| !value.trim().is_empty());
    refs.sort();
    refs.dedup();
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut canonical = Map::new();
            for key in keys {
                canonical.insert(key.clone(), canonical_json(&object[key]));
            }
            Value::Object(canonical)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

fn json_error(error: serde_json::Error) -> GraphStoreError {
    GraphStoreError::new("derivation_serialization_failed", error.to_string())
}
