use rustyred_thg_core::{
    collapse_if_corroborated, epistemic_shadow_node_id, flatten_statements,
    migrate_epistemic_shadows_to_statements, predicate_id, promote_statement_predicate,
    propose_same_as, write_statement, Confidence, EdgeRecord, EpistemicSourceKind, FlatObject,
    InMemoryGraphStore, NeighborQuery, NodeRecord, StatementFieldProvenance, StatementProvenance,
    StatementQuery, StatementRecord, StatementSemiring, CANONICAL_ENTITY_LABEL,
    HAS_EPISTEMIC_SHADOW, HAS_PREDICATE, SAME_ECLASS, STATEMENT_LABEL,
};
use serde_json::json;

fn entity(id: &str) -> NodeRecord {
    NodeRecord::new(id, ["Entity"], json!({}))
}

fn provenance(deps: &[&str], rule_id: &str, semiring: StatementSemiring) -> StatementProvenance {
    StatementProvenance::new(
        StatementFieldProvenance::new(EpistemicSourceKind::Structural, "test-engine", "test-v1", 1),
        deps.iter().copied(),
        rule_id,
        semiring,
    )
}

#[test]
fn statement_identity_converges_and_why_returns_monomial() {
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(entity("claim:a")).unwrap();
    store.upsert_node(entity("claim:b")).unwrap();

    let first = StatementRecord::derive(
        "claim:a",
        "supports",
        "claim:b",
        provenance(&["fact:1"], "rule:path:1", StatementSemiring::Viterbi),
        json!({ "confidence": 0.4 }),
    )
    .expect("first derived statement should be content-addressable");
    let second = StatementRecord::derive(
        "claim:a",
        "supports",
        "claim:b",
        provenance(&["fact:2"], "rule:path:2", StatementSemiring::Viterbi),
        json!({ "confidence": 0.8 }),
    )
    .expect("second derived statement should be content-addressable");

    assert_eq!(first.id, second.id);
    write_statement(&mut store, first, false).unwrap();
    write_statement(&mut store, second, false).unwrap();
    assert_eq!(
        store
            .query_nodes(rustyred_thg_core::NodeQuery::label(STATEMENT_LABEL).with_limit(10))
            .len(),
        1
    );

    let why = StatementRecord::derive(
        "claim:a",
        "derived_from",
        "claim:b",
        provenance(&["fact:b", "fact:a"], "why-rule", StatementSemiring::Why),
        json!({ "confidence": 1.0 }),
    )
    .expect("why statement should be content-addressable");
    assert_eq!(StatementRecord::why(&why), vec!["fact:a", "fact:b"]);
}

#[test]
fn confidence_lattice_joins_by_max_and_conjoins_by_decayed_min() {
    assert_eq!(
        Confidence::join(Confidence::new(0.25), Confidence::new(0.8)).value(),
        0.8
    );
    assert_eq!(
        Confidence::conjoin(&[Confidence::new(0.9), Confidence::new(0.6)], 0.5).value(),
        0.3
    );
    assert_eq!(Confidence::new(9.0).value(), 1.0);
    assert_eq!(Confidence::new(-9.0).value(), 0.0);
}

#[test]
fn flatten_view_is_invariant_under_predicate_promotion() {
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(entity("person:1")).unwrap();
    store.upsert_node(entity("parcel:1")).unwrap();
    let statement = StatementRecord::assert(
        "person:1",
        "owns",
        "parcel:1",
        json!({ "confidence": 0.85 }),
    )
    .expect("asserted statement should be content-addressable");
    write_statement(&mut store, statement.clone(), false).unwrap();

    let before = flatten_statements(
        &store,
        StatementQuery::subject("person:1").with_relation("owns"),
    );
    promote_statement_predicate(&mut store, &statement.id)
        .unwrap()
        .expect("predicate edge");
    let after = flatten_statements(
        &store,
        StatementQuery::subject("person:1").with_relation("owns"),
    );

    assert_eq!(before, after);
    assert!(store.get_node(&predicate_id("owns")).is_some());
    assert!(store
        .neighbors(NeighborQuery::out(&statement.id).with_edge_type(HAS_PREDICATE))
        .into_iter()
        .any(|hit| hit.node_id == predicate_id("owns")));
}

#[test]
fn same_as_stays_reversible_below_threshold_and_collapses_with_independent_support() {
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(entity("entity:a")).unwrap();
    store.upsert_node(entity("entity:b")).unwrap();
    let members = vec!["entity:a".to_string(), "entity:b".to_string()];

    let weak = propose_same_as(
        &mut store,
        "entity:a",
        "entity:b",
        Confidence::new(0.7),
        ["shared-key"],
    )
    .unwrap();
    assert!(
        collapse_if_corroborated(&mut store, &members, Confidence::new(0.9))
            .unwrap()
            .is_none()
    );
    assert!(store.get_node("entity:a").is_some());
    assert!(store.get_node("entity:b").is_some());

    let mut removed = store.get_edge(&weak.id).unwrap().clone();
    removed.tombstone = true;
    store.upsert_edge(removed).unwrap();
    assert!(store.get_edge(&weak.id).is_none());

    propose_same_as(
        &mut store,
        "entity:a",
        "entity:b",
        Confidence::new(0.95),
        ["source:a", "source:b"],
    )
    .unwrap();
    let before_collapse = store.snapshot();
    let canonical = collapse_if_corroborated(&mut store, &members, Confidence::new(0.9))
        .unwrap()
        .expect("canonical entity");

    assert!(canonical
        .labels
        .iter()
        .any(|label| label == CANONICAL_ENTITY_LABEL));
    assert!(store.get_node("entity:a").is_some());
    assert!(store.get_node("entity:b").is_some());
    assert!(store
        .neighbors(NeighborQuery::out("entity:a").with_edge_type(SAME_ECLASS))
        .into_iter()
        .any(|hit| hit.node_id == canonical.id));

    let reverted = InMemoryGraphStore::from_snapshot(before_collapse).unwrap();
    assert!(reverted.get_node(&canonical.id).is_none());
    assert!(reverted.get_node("entity:a").is_some());
    assert!(reverted.get_node("entity:b").is_some());
}

#[test]
fn epistemic_shadow_migration_keeps_shadow_ids_stable_and_writes_statement() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "claim:1",
            ["Claim"],
            json!({ "claim_text": "feature enabled" }),
        ))
        .unwrap();
    let shadow_id = epistemic_shadow_node_id("claim:1", "epistemic-v1")
        .expect("shadow node ID should be content-addressable");
    store
        .upsert_node(NodeRecord::new(
            &shadow_id,
            ["EpistemicShadow"],
            json!({
                "content_node_id": "claim:1",
                "grounded_extension_status": "in",
                "quarantine": true
            }),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "has:claim:1",
            "claim:1",
            HAS_EPISTEMIC_SHADOW,
            &shadow_id,
            json!({}),
        ))
        .unwrap();

    let report =
        migrate_epistemic_shadows_to_statements(&mut store, "migration-engine", "migration-v1", 9)
            .unwrap();

    assert_eq!(report.statements_written, 1);
    assert_eq!(report.shadow_ids, vec![shadow_id.clone()]);
    assert!(store.get_node(&shadow_id).is_some());
    let triples = flatten_statements(
        &store,
        StatementQuery::subject("claim:1").with_relation("epistemic_shadow_claim"),
    );
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].object, FlatObject::Entity(shadow_id.clone()));
    assert_eq!(
        triples[0].provenance.as_ref().expect("provenance").why(),
        vec![shadow_id]
    );
}
