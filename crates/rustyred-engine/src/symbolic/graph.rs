//! Bounded adapters over one actual retained graph revision. Admission belongs
//! to the caller; these adapters never expand a scope or scan the graph.
use crate::derivation::{
    put_derivation, AssumptionRecord, DerivationAnnotation, DerivationGraphReader,
    DerivationGraphWriter, DerivationKind, DerivationRecord, DerivationSpec, VALUE_LABEL,
};
use crate::graph_history::PinnedGraphRevision;
use crate::{
    node_is_expired, EdgeRecord, GraphMutation, GraphMutationBatch, GraphStoreError,
    GraphStoreResult, NeighborHit, NeighborQuery, NodeRecord,
};
use datalog::{ContentHash, DerivationReceipt, FactPack};
use egraph::{ExtractionReceipt, RulePack, Schedule, StopReason};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const MAX_RECORDS: usize = 16_384;
const MAX_BYTES: usize = 64 * 1024 * 1024;

/// Explicit IDs already admitted by the host's tenant/project policy. Empty
/// means empty, never the whole graph. Edge endpoints must both be admitted.
#[derive(Clone, Debug, Default)]
pub struct GraphScope {
    pub node_ids: BTreeSet<String>,
    pub edge_ids: BTreeSet<String>,
    pub observed_at_ms: i64,
}

struct CapturedScope {
    nodes: BTreeMap<String, NodeRecord>,
    edges: BTreeMap<String, EdgeRecord>,
    rows: Vec<Value>,
    facts: FactPack,
}

fn refusal(message: impl ToString) -> GraphStoreError {
    GraphStoreError::new("symbolic_adapter_refused", message.to_string())
}

fn capture(pin: &PinnedGraphRevision, scope: &GraphScope) -> GraphStoreResult<CapturedScope> {
    if scope.node_ids.len().saturating_add(scope.edge_ids.len()) > MAX_RECORDS {
        return Err(refusal("symbolic scope exceeds record bound"));
    }
    let mut bytes = 0usize;
    let mut count = |value: &Value| -> GraphStoreResult<()> {
        bytes = bytes.saturating_add(serde_json::to_vec(value).map_err(refusal)?.len());
        if bytes > MAX_BYTES {
            return Err(refusal("symbolic scope exceeds byte bound"));
        }
        Ok(())
    };
    let mut nodes = BTreeMap::new();
    for id in &scope.node_ids {
        let node = pin
            .get_node(id)?
            .ok_or_else(|| refusal("admitted node is missing"))?;
        if node.tombstone || node_is_expired(&node, scope.observed_at_ms) {
            return Err(refusal("admitted node is no longer live"));
        }
        count(&serde_json::to_value(&node).map_err(refusal)?)?;
        nodes.insert(id.clone(), node);
    }
    let mut edges = BTreeMap::new();
    for id in &scope.edge_ids {
        let edge = pin
            .get_edge(id)?
            .ok_or_else(|| refusal("admitted edge is missing"))?;
        if edge.tombstone || !nodes.contains_key(&edge.from_id) || !nodes.contains_key(&edge.to_id)
        {
            return Err(refusal("edge is not live with both endpoints admitted"));
        }
        count(&serde_json::to_value(&edge).map_err(refusal)?)?;
        edges.insert(id.clone(), edge);
    }
    let rows = crate::saturation::datalog_facts_from_graph_records(
        &nodes.values().cloned().collect::<Vec<_>>(),
        &edges.values().cloned().collect::<Vec<_>>(),
    );
    let facts = datalog::fact_pack_from_legacy_rows(&rows).map_err(refusal)?;
    Ok(CapturedScope {
        nodes,
        edges,
        rows,
        facts,
    })
}

pub fn fact_pack_from_graph(
    pin: &PinnedGraphRevision,
    scope: &GraphScope,
) -> GraphStoreResult<FactPack> {
    Ok(capture(pin, scope)?.facts)
}

enum ExpectedRun {
    Derivation(DerivationReceipt),
    Extraction {
        receipt: ExtractionReceipt,
        input: String,
        property: String,
    },
}

/// One verified engine invocation and its captured publication. `finish` emits
/// the existing expected-root batch for the host's attributed atomic commit.
/// No graph mutation occurs if evaluation, verification or preparation fails.
pub struct SymbolicGraph<'a> {
    pin: &'a PinnedGraphRevision,
    captured: CapturedScope,
    expected: ExpectedRun,
    batch: Option<GraphMutationBatch>,
}

impl<'a> SymbolicGraph<'a> {
    pub fn derivation(
        pin: &'a PinnedGraphRevision,
        scope: &GraphScope,
        engine: &dyn datalog::Engine,
    ) -> GraphStoreResult<Self> {
        let captured = capture(pin, scope)?;
        let receipt = engine.derive(&captured.facts);
        receipt.ensure_complete().map_err(refusal)?;
        Ok(Self {
            pin,
            captured,
            expected: ExpectedRun::Derivation(receipt),
            batch: None,
        })
    }

    /// The expression is read from a real admitted Value property. Neither a
    /// term handle nor an arbitrary string is mistaken for a graph node ID.
    pub fn extraction(
        pin: &'a PinnedGraphRevision,
        scope: &GraphScope,
        input: &str,
        property: &str,
        pack: RulePack,
        schedule: Schedule,
    ) -> GraphStoreResult<Self> {
        let captured = capture(pin, scope)?;
        let node = captured
            .nodes
            .get(input)
            .ok_or_else(|| refusal("extraction input is outside scope"))?;
        if !node.labels.iter().any(|label| label == VALUE_LABEL) {
            return Err(refusal("extraction input must already be a Value"));
        }
        let expression = node
            .properties
            .get(property)
            .and_then(Value::as_str)
            .ok_or_else(|| refusal("extraction input property must contain an expression"))?;
        let cost = pack.cost_model.clone();
        let mut engine = egraph::Engine::new(pack).map_err(refusal)?;
        let root = engine.root(expression).map_err(refusal)?;
        let saturation = engine.saturate(schedule).map_err(refusal)?;
        if saturation.stop_reason != StopReason::FixedPoint {
            return Err(refusal("graph extraction did not reach fixed point"));
        }
        let (_, receipt) = engine.extract(root, &cost).map_err(refusal)?;
        Ok(Self {
            pin,
            captured,
            expected: ExpectedRun::Extraction {
                receipt,
                input: input.into(),
                property: property.into(),
            },
            batch: None,
        })
    }

    pub fn derivation_receipt(&self) -> Option<&DerivationReceipt> {
        match &self.expected {
            ExpectedRun::Derivation(value) => Some(value),
            _ => None,
        }
    }
    pub fn extraction_receipt(&self) -> Option<&ExtractionReceipt> {
        match &self.expected {
            ExpectedRun::Extraction { receipt, .. } => Some(receipt),
            _ => None,
        }
    }
    pub fn finish(self) -> GraphStoreResult<GraphMutationBatch> {
        self.batch
            .ok_or_else(|| refusal("no verified receipt was applied"))
    }
}

fn derivation_identity(receipt: &DerivationReceipt) -> Value {
    // Observation timing is retained in the run record but cannot turn a repeat
    // of the same consequence into a new graph fact.
    json!({"engine":receipt.engine,"backend":receipt.backend,"rule_pack":receipt.rule_pack,
        "fact_pack":receipt.fact_pack,"derived":receipt.derived,"provenance":receipt.provenance})
}
fn extraction_identity(receipt: &ExtractionReceipt) -> Value {
    json!({"engine":receipt.engine,"rule_pack":receipt.rule_pack,"cost_model":receipt.cost_model,
        "root":receipt.root,"term":receipt.term,"term_hash":receipt.term_hash,"cost":receipt.cost})
}
fn content_id(prefix: &str, value: &Value) -> GraphStoreResult<String> {
    Ok(format!(
        "symbolic:{prefix}:{}",
        ContentHash::digest(serde_json::to_vec(value).map_err(refusal)?)
    ))
}

pub fn apply_derivation(
    store: &mut SymbolicGraph<'_>,
    receipt: &DerivationReceipt,
) -> GraphStoreResult<()> {
    receipt.ensure_complete().map_err(refusal)?;
    let ExpectedRun::Derivation(expected) = &store.expected else {
        return Err(refusal("graph binding expects an extraction"));
    };
    let identity = derivation_identity(receipt);
    let mut actual_stats = receipt.stats.clone();
    let mut expected_stats = expected.stats.clone();
    actual_stats.wall_time_ns = 0;
    expected_stats.wall_time_ns = 0;
    if identity != derivation_identity(expected) || actual_stats != expected_stats {
        return Err(refusal(
            "receipt differs from the engine run on the pinned facts",
        ));
    }
    let mut writes = CapturedWrites::new(store.pin);
    let run_id = content_id("run", &identity)?;
    if writes.existing_run(&run_id, &identity)? {
        store.batch = Some(writes.finish());
        return Ok(());
    }
    let environment = format!("symbolic:{}:{}", receipt.engine, receipt.rule_pack);
    let assumption = AssumptionRecord::active(
        content_id(
            "environment",
            &json!([receipt.engine, receipt.rule_pack, receipt.fact_pack]),
        )?,
        &environment,
    );
    let assumption_id = assumption.id.clone();
    writes.derivation_upsert_node(assumption.into_node())?;
    let mut values = BTreeMap::new();
    for row in &store.captured.rows {
        let fact = row["fact_id"]
            .as_str()
            .ok_or_else(|| refusal("graph fact has no identity"))?;
        let source = match (
            store.captured.nodes.get(fact),
            store.captured.edges.get(fact),
        ) {
            (Some(node), None) => json!({"node":node}),
            (None, Some(edge)) => json!({"edge":edge}),
            _ => {
                return Err(refusal(
                    "graph fact is not bound to exactly one admitted record",
                ))
            }
        };
        let properties =
            json!({"fact":row,"source":source,"graph_root":store.pin.roots.logical_tree.root});
        let id = content_id("input", &properties)?;
        writes.derivation_upsert_node(NodeRecord::new(
            &id,
            [VALUE_LABEL, "SymbolicGraphFact"],
            properties.clone(),
        ))?;
        put_derivation(
            &mut writes,
            DerivationSpec {
                kind: DerivationKind::Observational,
                operator_id: "operator:graph-fact-projection".into(),
                input_refs: vec![],
                output_refs: vec![id.clone()],
                assumption_refs: vec![],
                params: properties,
                environment_version: "symbolic-graph-facts:1".into(),
                annotation: DerivationAnnotation::default(),
            },
        )?;
        if values.insert(fact.to_owned(), id).is_some() {
            return Err(refusal("graph fact identity is ambiguous"));
        }
    }
    for row in &receipt.derived {
        let id = content_id(
            "derived",
            &json!([receipt.engine, receipt.rule_pack, receipt.fact_pack, row]),
        )?;
        if values.insert(row.fact_id.clone(), id.clone()).is_some() {
            return Err(refusal(
                "derived fact collides with another premise or output",
            ));
        }
        writes.derivation_upsert_node(NodeRecord::new(
            id,
            [VALUE_LABEL, "DerivedStatement"],
            serde_json::to_value(row).map_err(refusal)?,
        ))?;
    }
    for proof in &receipt.provenance {
        let output = values
            .get(&proof.conclusion)
            .ok_or_else(|| refusal("proof conclusion has no output"))?;
        let inputs = proof
            .premises
            .iter()
            .map(|id| {
                values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| refusal("proof premise is outside the pinned fact pack"))
            })
            .collect::<GraphStoreResult<Vec<_>>>()?;
        let mut assumptions = vec![assumption_id.clone()];
        for absence in &proof.negative_premises {
            if absence.fact_pack != receipt.fact_pack || absence.rule_pack != receipt.rule_pack {
                return Err(refusal(
                    "negative premise is bound to another input or rule pack",
                ));
            }
            let evidence = serde_json::to_value(absence).map_err(refusal)?;
            let id = content_id("absence", &evidence)?;
            let mut assumption = AssumptionRecord::active(
                &id,
                format!("{} absent in closed fact pack", absence.relation),
            );
            assumption.scope = Some(format!("closed_fact_pack:{}", absence.fact_pack));
            let mut node = assumption.into_node();
            node.properties["evidence"] = evidence;
            writes.derivation_upsert_node(node)?;
            assumptions.push(id);
        }
        put_derivation(
            &mut writes,
            DerivationSpec {
                kind: DerivationKind::Functional,
                operator_id: format!("operator:rule:{}", proof.rule),
                input_refs: inputs,
                output_refs: vec![output.clone()],
                assumption_refs: assumptions,
                params: json!({"engine":receipt.engine,"rule_pack":receipt.rule_pack,"fact_pack":receipt.fact_pack,"proof":proof}),
                environment_version: environment.clone(),
                annotation: DerivationAnnotation::default(),
            },
        )?;
    }
    writes.run_record(
        run_id,
        identity,
        serde_json::to_value(expected).map_err(refusal)?,
    )?;
    store.batch = Some(writes.finish());
    Ok(())
}

pub fn apply_extraction(
    store: &mut SymbolicGraph<'_>,
    receipt: &ExtractionReceipt,
) -> GraphStoreResult<()> {
    let ExpectedRun::Extraction {
        receipt: expected,
        input,
        property,
    } = &store.expected
    else {
        return Err(refusal("graph binding expects a derivation"));
    };
    if extraction_identity(receipt) != extraction_identity(expected) {
        return Err(refusal(
            "extraction differs from the actual bound engine run",
        ));
    }
    let identity = json!({"extraction":extraction_identity(receipt),"input":input,"property":property,
        "input_content":store.captured.nodes[input].content_address()?});
    let mut writes = CapturedWrites::new(store.pin);
    let run_id = content_id("run", &identity)?;
    if !writes.existing_run(&run_id, &identity)? {
        let output = content_id("extracted", &identity)?;
        writes.derivation_upsert_node(NodeRecord::new(
            &output,
            [VALUE_LABEL, "SymbolicExtraction"],
            json!({"expression":receipt.term,"receipt":extraction_identity(receipt)}),
        ))?;
        put_derivation(
            &mut writes,
            DerivationSpec {
                kind: DerivationKind::Functional,
                operator_id: format!("operator:egraph:{}", receipt.rule_pack),
                input_refs: vec![input.clone()],
                output_refs: vec![output],
                assumption_refs: vec![],
                params: identity.clone(),
                environment_version: format!("egraph:{}", receipt.rule_pack),
                annotation: DerivationAnnotation::default(),
            },
        )?;
        writes.run_record(
            run_id,
            identity,
            serde_json::to_value(expected).map_err(refusal)?,
        )?;
    }
    store.batch = Some(writes.finish());
    Ok(())
}

/// The fallible canonical writer needs only point reads for publication. A
/// neighbor traversal is explicitly refused instead of hiding a graph scan.
struct CapturedWrites<'a> {
    pin: &'a PinnedGraphRevision,
    nodes: BTreeMap<String, NodeRecord>,
    edges: BTreeMap<String, EdgeRecord>,
    bytes: usize,
}
impl<'a> CapturedWrites<'a> {
    fn new(pin: &'a PinnedGraphRevision) -> Self {
        Self {
            pin,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            bytes: 0,
        }
    }
    fn count(&mut self, bytes: usize) -> GraphStoreResult<()> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.nodes.len().saturating_add(self.edges.len()) >= MAX_RECORDS
            || self.bytes > MAX_BYTES
        {
            return Err(refusal("symbolic publication exceeds bounds"));
        }
        Ok(())
    }
    fn existing_run(&self, id: &str, identity: &Value) -> GraphStoreResult<bool> {
        let Some(node) = self.pin.get_node(id)? else {
            return Ok(false);
        };
        if node.tombstone
            || !node.labels.iter().any(|label| label == "SymbolicRun")
            || node.properties["identity"] != *identity
        {
            return Err(refusal(
                "symbolic run identity is occupied by another record",
            ));
        }
        Ok(true)
    }
    fn run_record(&mut self, id: String, identity: Value, receipt: Value) -> GraphStoreResult<()> {
        self.derivation_upsert_node(NodeRecord::new(id, [VALUE_LABEL, "SymbolicRun"],
            json!({"identity":identity,"receipt":receipt,"graph_root":self.pin.roots.logical_tree.root})))
    }
    fn finish(self) -> GraphMutationBatch {
        GraphMutationBatch::new(
            self.nodes
                .into_values()
                .map(GraphMutation::NodeUpsert)
                .chain(self.edges.into_values().map(GraphMutation::EdgeUpsert)),
        )
        .with_expected_root(self.pin.roots.logical_tree.root)
    }
}
impl DerivationGraphReader for CapturedWrites<'_> {
    fn derivation_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        match self.nodes.get(id) {
            Some(node) => Ok(Some(node.clone())),
            None => self.pin.get_node(id),
        }
    }
    fn derivation_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        match self.edges.get(id) {
            Some(edge) => Ok(Some(edge.clone())),
            None => self.pin.get_edge(id),
        }
    }
    fn derivation_neighbors(&self, _: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>> {
        Err(refusal("symbolic publication does not traverse neighbors"))
    }
}
impl DerivationGraphWriter for CapturedWrites<'_> {
    fn derivation_upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.count(serde_json::to_vec(&node).map_err(refusal)?.len())?;
        if let Some(existing) = self.derivation_node(&node.id)? {
            if existing.content_address()? != node.content_address()? {
                return Err(refusal(
                    "symbolic publication cannot overwrite an existing node",
                ));
            }
            return Ok(());
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }
    fn derivation_upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        self.count(serde_json::to_vec(&edge).map_err(refusal)?.len())?;
        if let Some(existing) = self.derivation_edge(&edge.id)? {
            if existing.content_address()? != edge.content_address()? {
                return Err(refusal(
                    "symbolic publication cannot overwrite an existing edge",
                ));
            }
            return Ok(());
        }
        self.edges.insert(edge.id.clone(), edge);
        Ok(())
    }
    fn project_incremental_support(&mut self, _: &DerivationRecord) -> GraphStoreResult<()> {
        // The native commit preparer performs the existing incremental fold on
        // this same complete batch; it is not a second authority in the adapter.
        Ok(())
    }
}
