use std::collections::BTreeSet;

use rustyred_thg_core::{
    fold, put_assumption, put_derivation, record_nogood, rederive_if_missing, retract_assumption,
    why, AssumptionRecord, BooleanSemiring, ConditionalNodePropertyUpdate, CountSemiring,
    DerivationAnnotation, DerivationKind, DerivationRecord, DerivationSpec, EdgeRecord,
    FunctionalDerivationExecutor, GraphAtomic, GraphMutationBatch, GraphRead, GraphRebuildReport,
    GraphResidency, GraphStats, GraphStoreResult, GraphVerify, GraphWrite, GraphWriteResult,
    InMemoryGraphStore, NeighborHit, NeighborQuery, NodeQuery, NodeRecord, NogoodRecord,
    ProbabilitySemiring, TropicalSemiring, VerifyReport, WhyTree,
};
use serde_json::json;

fn observational(store: &mut InMemoryGraphStore, output: &str, confidence: f64) -> String {
    store
        .upsert_node(NodeRecord::new(
            output,
            ["Value"],
            json!({ "value": output }),
        ))
        .unwrap();
    put_derivation(
        store,
        DerivationSpec {
            kind: DerivationKind::Observational,
            operator_id: format!("operator:observe:{output}"),
            input_refs: Vec::new(),
            output_refs: vec![output.to_string()],
            assumption_refs: Vec::new(),
            params: json!({}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation {
                confidence,
                cost: 1.0,
            },
        },
    )
    .unwrap()
    .id
}

#[test]
fn derivation_identity_sorts_inputs_and_canonicalizes_params() {
    let left = DerivationSpec {
        kind: DerivationKind::Functional,
        operator_id: "operator:test".to_string(),
        input_refs: vec!["value:b".to_string(), "value:a".to_string()],
        output_refs: vec!["value:out".to_string()],
        assumption_refs: Vec::new(),
        params: json!({ "z": 1, "a": { "y": 2, "b": 3 } }),
        environment_version: "env:v1".to_string(),
        annotation: DerivationAnnotation::default(),
    };
    let mut right = left.clone();
    right.input_refs.reverse();
    right.params = json!({ "a": { "b": 3, "y": 2 }, "z": 1 });
    right.kind = DerivationKind::Observational;
    right.annotation.confidence = 0.2;

    assert_eq!(left.content_id(), right.content_id());
}

#[test]
fn probability_fold_matches_a_three_layer_hand_computation() {
    let mut store = InMemoryGraphStore::new();
    observational(&mut store, "value:source", 0.8);
    store
        .upsert_node(NodeRecord::new("value:middle", ["Value"], json!({})))
        .unwrap();
    put_derivation(
        &mut store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: "operator:middle".to_string(),
            input_refs: vec!["value:source".to_string()],
            output_refs: vec!["value:middle".to_string()],
            assumption_refs: Vec::new(),
            params: json!({}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation {
                confidence: 0.75,
                cost: 2.0,
            },
        },
    )
    .unwrap();
    store
        .upsert_node(NodeRecord::new("value:result", ["Value"], json!({})))
        .unwrap();
    put_derivation(
        &mut store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: "operator:result".to_string(),
            input_refs: vec!["value:middle".to_string()],
            output_refs: vec!["value:result".to_string()],
            assumption_refs: Vec::new(),
            params: json!({}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation {
                confidence: 0.5,
                cost: 3.0,
            },
        },
    )
    .unwrap();

    let probability = fold(&store, "value:result", &ProbabilitySemiring).unwrap();
    assert!((probability - 0.8 * 0.75 * 0.5).abs() < 1e-12);
    assert_eq!(fold(&store, "value:result", &CountSemiring).unwrap(), 1);
    assert_eq!(
        fold(&store, "value:result", &TropicalSemiring).unwrap(),
        6.0
    );

    let why_tree = why(&store, "value:result").unwrap();
    let WhyTree::Derived { derivations, .. } = why_tree else {
        panic!("result must have a derivation");
    };
    assert_eq!(derivations[0].operator_id, "operator:result");
}

#[test]
fn one_assumption_retraction_mechanically_invalidates_all_dependents() {
    let mut store = InMemoryGraphStore::new();
    put_assumption(
        &mut store,
        AssumptionRecord::active("assumption:api-v1", "API v1 contract"),
    )
    .unwrap();
    observational(&mut store, "value:source", 1.0);

    for (input, output) in [
        ("value:source", "value:belief"),
        ("value:belief", "value:skill"),
        ("value:skill", "value:decision"),
    ] {
        store
            .upsert_node(NodeRecord::new(output, ["Value"], json!({})))
            .unwrap();
        put_derivation(
            &mut store,
            DerivationSpec {
                kind: DerivationKind::Functional,
                operator_id: format!("operator:{output}"),
                input_refs: vec![input.to_string()],
                output_refs: vec![output.to_string()],
                assumption_refs: vec!["assumption:api-v1".to_string()],
                params: json!({}),
                environment_version: "test:v1".to_string(),
                annotation: DerivationAnnotation::default(),
            },
        )
        .unwrap();
    }

    assert!(fold(&store, "value:decision", &BooleanSemiring)
        .unwrap()
        .holds());
    let impacted = retract_assumption(&mut store, "assumption:api-v1").unwrap();
    assert_eq!(
        impacted.into_iter().collect::<BTreeSet<_>>(),
        ["value:belief", "value:decision", "value:skill"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
    assert!(!fold(&store, "value:belief", &BooleanSemiring)
        .unwrap()
        .holds());
    assert!(!fold(&store, "value:skill", &BooleanSemiring)
        .unwrap()
        .holds());
    assert!(!fold(&store, "value:decision", &BooleanSemiring)
        .unwrap()
        .holds());
}

struct EvictingStore {
    inner: InMemoryGraphStore,
    evicted: BTreeSet<String>,
}

impl EvictingStore {
    fn new() -> Self {
        Self {
            inner: InMemoryGraphStore::new(),
            evicted: BTreeSet::new(),
        }
    }
}

impl GraphWrite for EvictingStore {
    fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        GraphWrite::upsert_node(&mut self.inner, node)
    }

    fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        GraphWrite::upsert_edge(&mut self.inner, edge)
    }
}

impl GraphAtomic for EvictingStore {
    fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        GraphAtomic::insert_node_if_absent(&mut self.inner, node)
    }

    fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        GraphAtomic::insert_node_if_never_existed(&mut self.inner, node)
    }

    fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &serde_json::Value,
        new_value: serde_json::Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        GraphAtomic::compare_and_set_node_property(
            &mut self.inner,
            id,
            property,
            expected,
            new_value,
        )
    }

    fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        GraphAtomic::commit_conditional_batch_atomic(&mut self.inner, conditions, batch)
    }
}

impl GraphRead for EvictingStore {
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        (!self.evicted.contains(id))
            .then(|| GraphRead::get_node(&self.inner, id))
            .flatten()
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        GraphRead::get_edge(&self.inner, id)
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        GraphRead::live_edge_records(&self.inner)
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        GraphRead::query_nodes(&self.inner, query)
            .into_iter()
            .filter(|node| !self.evicted.contains(&node.id))
            .collect()
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        GraphRead::neighbors(&self.inner, query)
    }

    fn stats(&self) -> GraphStats {
        GraphRead::stats(&self.inner)
    }
}

impl GraphVerify for EvictingStore {
    fn verify(&self) -> VerifyReport {
        GraphVerify::verify(&self.inner)
    }

    fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        GraphVerify::rebuild_indexes(&mut self.inner)
    }
}

impl GraphResidency for EvictingStore {
    fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        let node = GraphRead::get_node_record(&self.inner, id);
        if node.is_some() {
            self.evicted.insert(id.to_string());
        }
        Ok(node)
    }

    fn readmit_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.evicted.remove(&node.id);
        GraphWrite::upsert_node(&mut self.inner, node).map(|_| ())
    }

    fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        GraphResidency::evict_edge(&mut self.inner, id)
    }

    fn readmit_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        GraphResidency::readmit_edge(&mut self.inner, edge)
    }
}

struct DoubleExecutor;

impl FunctionalDerivationExecutor for DoubleExecutor {
    fn rederive(
        &self,
        _derivation: &DerivationRecord,
        inputs: &[NodeRecord],
        output_id: &str,
    ) -> GraphStoreResult<NodeRecord> {
        let value = inputs[0].properties["value"].as_i64().unwrap();
        Ok(NodeRecord::new(
            output_id,
            ["Value"],
            json!({ "value": value * 2 }),
        ))
    }
}

#[test]
fn evicted_functional_output_is_rederived_byte_identically() {
    let mut store = EvictingStore::new();
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("value:input", ["Value"], json!({ "value": 21 })),
    )
    .unwrap();
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("value:output", ["Value"], json!({ "value": 42 })),
    )
    .unwrap();
    put_derivation(
        &mut store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: "operator:double".to_string(),
            input_refs: vec!["value:input".to_string()],
            output_refs: vec!["value:output".to_string()],
            assumption_refs: Vec::new(),
            params: json!({ "factor": 2 }),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation::default(),
        },
    )
    .unwrap();
    let expected = GraphRead::get_node_record(&store, "value:output")
        .unwrap()
        .content_address();

    GraphResidency::evict_node(&mut store, "value:output").unwrap();
    assert!(GraphRead::get_node(&store, "value:output").is_none());
    let rederived = rederive_if_missing(&mut store, "value:output", &DoubleExecutor)
        .unwrap()
        .unwrap();

    assert_eq!(rederived.content_address(), expected);
    assert_eq!(
        GraphRead::get_node(&store, "value:output")
            .unwrap()
            .properties,
        json!({ "value": 42 })
    );
}

#[test]
fn nogood_records_the_minimal_conflicting_assumption_set() {
    let mut store = InMemoryGraphStore::new();
    for (id, name) in [
        ("assumption:source-a", "source A is trustworthy"),
        ("assumption:api-v1", "API v1 contract"),
    ] {
        put_assumption(&mut store, AssumptionRecord::active(id, name)).unwrap();
    }
    let nogood = record_nogood(
        &mut store,
        NogoodRecord::new(
            ["assumption:source-a", "assumption:api-v1"],
            "check:test-42",
            "seeded contradiction",
            "2026-07-12T00:00:00Z",
        )
        .expect("nogood record should be content-addressable"),
    )
    .unwrap();

    assert_eq!(
        nogood.assumption_refs,
        vec!["assumption:api-v1", "assumption:source-a"]
    );
    let conflicts = GraphRead::neighbors(
        &store,
        NeighborQuery::out(&nogood.id).with_edge_type("CONFLICTS_WITH"),
    );
    assert_eq!(conflicts.len(), 2);
}
