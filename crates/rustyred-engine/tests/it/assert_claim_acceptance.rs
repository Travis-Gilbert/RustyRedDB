use rustyred_thg_core::{
    assert_claim, fold, is_asserted_claim, put_assumption, put_derivation, retract_assumption,
    AssertClaimInput, AssertClaimStore, AssumptionRecord, BooleanSemiring, DerivationAnnotation,
    DerivationGraphReader, DerivationKind, DerivationSpec, EdgeRecord, GraphMutationBatch,
    GraphStoreError, GraphStoreResult, GraphWrite, InMemoryGraphStore, NeighborHit, NeighborQuery,
    NodeRecord,
};
use serde_json::json;

fn seed_support(store: &mut InMemoryGraphStore, suffix: &str, assumption_id: &str) -> String {
    put_assumption(
        store,
        AssumptionRecord::active(
            assumption_id,
            format!("source {suffix} remains trustworthy"),
        ),
    )
    .unwrap();
    let observation_id = format!("observation:{suffix}");
    GraphWrite::upsert_node(
        store,
        NodeRecord::new(&observation_id, ["Value"], json!({"source": suffix})),
    )
    .unwrap();
    put_derivation(
        store,
        DerivationSpec {
            kind: DerivationKind::Observational,
            operator_id: format!("operator:observe:{suffix}"),
            input_refs: Vec::new(),
            output_refs: vec![observation_id],
            assumption_refs: vec![assumption_id.to_string()],
            params: json!({"source": suffix}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation::default(),
        },
    )
    .unwrap()
    .id
}

fn input(source: &str, supporting: String, assumption: &str) -> AssertClaimInput {
    AssertClaimInput {
        tenant_slug: "tenant-a".to_string(),
        claim: "The API contract is stable".to_string(),
        claim_type: "api_contract".to_string(),
        source_ref: source.to_string(),
        supporting_derivation_ref: supporting,
        assumption_refs: vec![assumption.to_string()],
        stated_confidence: 0.9,
        effective_confidence: 0.8,
        tier: "load_bearing".to_string(),
    }
}

#[test]
fn two_sources_create_independent_support_and_one_retraction_does_not_kill_claim() {
    let mut store = InMemoryGraphStore::new();
    let support_a = seed_support(&mut store, "a", "assumption:source-a");
    let support_b = seed_support(&mut store, "b", "assumption:source-b");
    let asserted_a = assert_claim(
        &mut store,
        input("source:a", support_a, "assumption:source-a"),
    )
    .unwrap();
    let asserted_b = assert_claim(
        &mut store,
        input("source:b", support_b, "assumption:source-b"),
    )
    .unwrap();

    assert_eq!(asserted_a.claim_id, asserted_b.claim_id);
    assert_ne!(
        asserted_a.admission_derivation_id,
        asserted_b.admission_derivation_id
    );
    assert!(is_asserted_claim(&store, &asserted_a.claim_id).unwrap());
    let before = fold(&store, &asserted_a.claim_id, &BooleanSemiring).unwrap();
    assert_eq!(before.terms.len(), 2);

    retract_assumption(&mut store, "assumption:source-a").unwrap();
    let after = fold(&store, &asserted_a.claim_id, &BooleanSemiring).unwrap();
    assert!(after.holds());
    assert_eq!(after.terms.len(), 1);
    assert!(after
        .terms
        .iter()
        .all(|term| term.contains("assumption:source-b")));
}

#[test]
fn assertion_is_idempotent_and_raw_claim_shaped_nodes_are_not_asserted() {
    let mut store = InMemoryGraphStore::new();
    let support = seed_support(&mut store, "a", "assumption:source-a");
    let first = assert_claim(
        &mut store,
        input("source:a", support.clone(), "assumption:source-a"),
    )
    .unwrap();
    let second = assert_claim(
        &mut store,
        input("source:a", support, "assumption:source-a"),
    )
    .unwrap();
    assert_eq!(first.receipt_id, second.receipt_id);
    assert!(second.idempotent_replay);

    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "raw:claim-shaped",
            ["Claim", "Value"],
            json!({"claim": "raw writes remain available"}),
        ),
    )
    .unwrap();
    assert!(!is_asserted_claim(&store, "raw:claim-shaped").unwrap());
}

#[test]
fn inactive_assumptions_are_rejected_before_any_claim_write() {
    let mut store = InMemoryGraphStore::new();
    let support = seed_support(&mut store, "a", "assumption:source-a");
    retract_assumption(&mut store, "assumption:source-a").unwrap();
    let result = assert_claim(
        &mut store,
        input("source:a", support, "assumption:source-a"),
    );
    assert_eq!(result.unwrap_err().code, "inactive_assertion_assumption");
    assert!(store
        .query_nodes(rustyred_thg_core::NodeQuery::label("Claim"))
        .is_empty());
}

struct RejectingAtomicStore {
    inner: InMemoryGraphStore,
}

impl DerivationGraphReader for RejectingAtomicStore {
    fn derivation_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        Ok(self.inner.get_node(id).cloned())
    }

    fn derivation_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        Ok(self.inner.get_edge(id).cloned())
    }

    fn derivation_neighbors(&self, query: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>> {
        Ok(self.inner.neighbors(query))
    }
}

impl AssertClaimStore for RejectingAtomicStore {
    fn commit_assert_claim_batch(&mut self, _batch: GraphMutationBatch) -> GraphStoreResult<()> {
        Err(GraphStoreError::new(
            "fault_injected",
            "reject atomic batch",
        ))
    }
}

#[test]
fn failed_atomic_commit_leaves_no_partial_claim_or_receipt() {
    let mut seeded = InMemoryGraphStore::new();
    let support = seed_support(&mut seeded, "a", "assumption:source-a");
    let mut store = RejectingAtomicStore { inner: seeded };
    let result = assert_claim(
        &mut store,
        input("source:a", support, "assumption:source-a"),
    );
    assert_eq!(result.unwrap_err().code, "fault_injected");
    assert!(store
        .inner
        .query_nodes(rustyred_thg_core::NodeQuery::label("Claim"))
        .is_empty());
    assert!(store
        .inner
        .query_nodes(rustyred_thg_core::NodeQuery::label("AssertionReceipt"))
        .is_empty());
}
