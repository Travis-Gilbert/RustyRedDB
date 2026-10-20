use rustyred_thg_core::{
    ConditionalNodePropertyUpdate, EdgeRecord, EpistemicType, FactsQuery, FoldGraphStore,
    GraphAtomic, GraphMutation, GraphMutationBatch, GraphRead, GraphSnapshotSource, GraphVerify,
    GraphWrite, InMemoryGraphStore, NodeRecord, Provenance, GRAPH_INTENT_NAME,
    GRAPH_INTENT_NAMESPACE, GRAPH_INTENT_VERSION,
};
use serde_json::{json, Value};
use std::error::Error;
use theorem_oplog::{
    AccessPath, IntentKind, LogIntent, LogPosition, OpDigest, PhysicalLog, WriteSet,
};

fn node(id: &str, status: &str) -> NodeRecord {
    NodeRecord::new(
        id,
        ["Claim"],
        json!({
            "status": status,
            "_valid_from_ms": 100,
            "_valid_to_ms": 900,
            "_epistemic_type": "supports",
            "_confidence": 0.75,
            "_provenance": {"source_id": "source:fixture", "method": "test"}
        }),
    )
}

fn edge(id: &str, from: &str, to: &str) -> EdgeRecord {
    EdgeRecord::new(id, from, "SUPPORTS", to, json!({"weight": 3}))
        .with_confidence(0.8)
        .with_epistemic_type(EpistemicType::Supports)
        .with_provenance(Provenance {
            source_id: Some("source:fixture".to_string()),
            timestamp: Some("2026-08-12T00:00:00Z".to_string()),
            method: Some("test".to_string()),
        })
}

fn assert_same_graph(
    legacy: &InMemoryGraphStore,
    folded: &FoldGraphStore,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(legacy.graph_snapshot()?, folded.graph_snapshot()?);
    let legacy_verification = legacy.verify();
    let folded_verification = folded.verify();
    assert_eq!(legacy_verification.ok, folded_verification.ok);
    assert_eq!(legacy_verification.problems, folded_verification.problems);
    Ok(())
}

#[derive(Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
}

#[test]
fn graph_fold_matches_legacy_store_at_every_commit_boundary() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut folded = FoldGraphStore::open(directory.path(), deployment_id())?;
    let mut legacy = InMemoryGraphStore::new();

    let seed = GraphMutationBatch::new([
        GraphMutation::NodeUpsert(node("claim:a", "draft")),
        GraphMutation::NodeUpsert(node("claim:b", "draft")),
    ]);
    legacy.upsert_node(node("claim:a", "draft"))?;
    legacy.upsert_node(node("claim:b", "draft"))?;
    let seed_receipt = folded.commit_batch(seed)?;
    assert_eq!(seed_receipt.position, LogPosition(1));
    assert_same_graph(&legacy, &folded)?;

    let relationship = GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge(
        "edge:ab", "claim:a", "claim:b",
    ))]);
    legacy.upsert_edge(edge("edge:ab", "claim:a", "claim:b"))?;
    folded.commit_batch(relationship)?;
    assert_same_graph(&legacy, &folded)?;

    let mut revised = node("claim:a", "reviewed");
    revised.properties["op"] = json!("node_upsert");
    revised.properties["payload"] = json!({"op": "edge_delete"});
    let revision = GraphMutationBatch::new([GraphMutation::NodeUpsert(revised.clone())]);
    legacy.upsert_node(revised)?;
    folded.commit_batch(revision)?;
    assert_same_graph(&legacy, &folded)?;

    let legacy_cas = legacy.compare_and_set_node_property(
        "claim:a",
        "status",
        &json!("reviewed"),
        json!("accepted"),
    )?;
    let folded_cas = folded.compare_and_set_node_property(
        "claim:a",
        "status",
        &json!("reviewed"),
        json!("accepted"),
    )?;
    assert_eq!(legacy_cas, folded_cas);
    assert_same_graph(&legacy, &folded)?;

    let conditions = vec![ConditionalNodePropertyUpdate {
        node_id: "claim:b".to_string(),
        property: "status".to_string(),
        expected: json!("draft"),
        new_value: json!("accepted"),
    }];
    let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(node("claim:c", "accepted"))]);
    assert!(legacy.commit_conditional_batch_atomic(conditions.clone(), batch.clone())?);
    assert!(folded.commit_conditional_batch_atomic(conditions, batch)?);
    assert_same_graph(&legacy, &folded)?;

    let deletion = GraphMutationBatch::new([GraphMutation::NodeDelete("claim:b".to_string())]);
    folded.commit_batch(deletion)?;
    assert!(folded.get_edge("edge:ab").is_none());
    assert!(folded
        .get_edge_record("edge:ab")
        .is_some_and(|edge| edge.tombstone));
    assert!(folded
        .get_node_record("claim:b")
        .is_some_and(|node| node.tombstone));

    assert_eq!(
        folded.log().len(),
        6,
        "one durable append per public commit"
    );
    assert!(folded.graph_quarantine().is_empty());
    assert!(folded.facts_quarantine().is_empty());

    let original_status = folded.facts().facts_at(
        &FactsQuery {
            entity: "node:claim:a".to_string(),
            attribute: Some("property:status".to_string()),
        },
        seed_receipt.position,
    );
    assert_eq!(original_status.len(), 1);
    assert_eq!(original_status[0].value, json!("draft"));
    assert_eq!(original_status[0].valid_from_ms, Some(100));
    assert_eq!(original_status[0].valid_to_ms, Some(900));
    assert_eq!(
        original_status[0].epistemic_type.as_deref(),
        Some("supports")
    );
    assert_eq!(original_status[0].confidence, Some(0.75));
    assert!(
        folded.facts().eavt_len()
            > folded
                .facts()
                .facts_at(
                    &FactsQuery {
                        entity: "node:claim:a".to_string(),
                        attribute: None,
                    },
                    folded.latest_position(),
                )
                .len()
    );
    assert_eq!(folded.facts().provenance_source_count(), 1);

    let expected_snapshot = folded.graph_snapshot()?;
    let expected_position = folded.latest_position();
    drop(folded);
    let reopened = FoldGraphStore::open(directory.path(), deployment_id())?;
    assert_eq!(reopened.latest_position(), expected_position);
    assert_eq!(reopened.graph_snapshot()?, expected_snapshot);
    assert!(reopened.graph_quarantine().is_empty());
    assert!(reopened.facts_quarantine().is_empty());
    Ok(())
}

#[test]
fn seeded_random_histories_match_legacy_at_every_commit_and_replay_boundary(
) -> Result<(), Box<dyn Error>> {
    const NODE_COUNT: usize = 12;
    const EDGE_COUNT: usize = 24;
    const STEPS_PER_SEED: usize = 64;
    const SEEDS: [u64; 4] = [
        0x03a0_0000_0000_0001,
        0x03a0_0000_0000_0101,
        0x03a0_0000_0001_0001,
        0x03a0_0000_0100_0001,
    ];

    for seed in SEEDS {
        let directory = tempfile::tempdir()?;
        let mut folded = FoldGraphStore::open(directory.path(), deployment_id())?;
        let mut legacy = InMemoryGraphStore::new();
        let mut rng = DeterministicRng(seed);

        for index in 0..NODE_COUNT {
            let record = node(&format!("claim:{index}"), "seeded");
            legacy.upsert_node(record.clone())?;
            folded.upsert_node(record)?;
        }
        assert_same_graph(&legacy, &folded)?;

        for step in 0..STEPS_PER_SEED {
            match rng.index(4) {
                0 | 1 => {
                    let index = rng.index(NODE_COUNT);
                    let status = format!("seed-{seed:016x}-step-{step:03}");
                    let record = node(&format!("claim:{index}"), &status);
                    legacy.upsert_node(record.clone())?;
                    folded.upsert_node(record)?;
                }
                2 => {
                    let edge_index = rng.index(EDGE_COUNT);
                    let from = rng.index(NODE_COUNT);
                    let mut to = rng.index(NODE_COUNT);
                    if from == to {
                        to = (to + 1) % NODE_COUNT;
                    }
                    let record = edge(
                        &format!("edge:{edge_index}"),
                        &format!("claim:{from}"),
                        &format!("claim:{to}"),
                    );
                    legacy.upsert_edge(record.clone())?;
                    folded.upsert_edge(record)?;
                }
                _ => {
                    let index = rng.index(NODE_COUNT);
                    let id = format!("claim:{index}");
                    let current = legacy
                        .get_node_record(&id)
                        .expect("seeded node remains present")
                        .properties["status"]
                        .clone();
                    let expected = if rng.index(5) == 0 {
                        json!("deliberately-stale")
                    } else {
                        current
                    };
                    let replacement = json!(format!("cas-{seed:016x}-{step:03}"));
                    let legacy_receipt = legacy.compare_and_set_node_property(
                        &id,
                        "status",
                        &expected,
                        replacement.clone(),
                    )?;
                    let folded_receipt = folded.compare_and_set_node_property(
                        &id,
                        "status",
                        &expected,
                        replacement,
                    )?;
                    assert_eq!(legacy_receipt, folded_receipt, "seed {seed:#x} step {step}");
                }
            }
            assert_same_graph(&legacy, &folded)?;

            if step == STEPS_PER_SEED / 2 {
                let expected_position = folded.latest_position();
                drop(folded);
                folded = FoldGraphStore::open(directory.path(), deployment_id())?;
                assert_eq!(folded.latest_position(), expected_position);
                assert_same_graph(&legacy, &folded)?;
            }
        }

        let expected_position = folded.latest_position();
        drop(folded);
        let reopened = FoldGraphStore::open(directory.path(), deployment_id())?;
        assert_eq!(reopened.latest_position(), expected_position);
        assert_same_graph(&legacy, &reopened)?;
        assert!(reopened.graph_quarantine().is_empty());
        assert!(reopened.facts_quarantine().is_empty());
    }
    Ok(())
}

#[test]
fn malformed_schemaless_entry_is_quarantined_and_replay_continues() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut log = PhysicalLog::open(directory.path(), deployment_id())?;
    let kind = IntentKind::new(
        GRAPH_INTENT_NAMESPACE,
        GRAPH_INTENT_NAME,
        GRAPH_INTENT_VERSION,
    )?;
    let writes = WriteSet::new([AccessPath::Entity {
        entity: "node:broken".to_string(),
    }])?;
    log.append(
        LogIntent::new(
            kind.clone(),
            json!({"changes": [{"intent": "node_observed"}]}),
        )?,
        writes,
        OpDigest::from_bytes("malformed-graph-entry"),
    )?;
    log.append(
        LogIntent::new(
            kind,
            json!({
                "changes": [{
                    "intent": "node_observed",
                    "record": node("claim:valid", "accepted")
                }]
            }),
        )?,
        WriteSet::new([AccessPath::Entity {
            entity: "node:claim:valid".to_string(),
        }])?,
        OpDigest::from_bytes("valid-graph-entry"),
    )?;
    drop(log);

    let store = FoldGraphStore::open(directory.path(), deployment_id())?;
    assert_eq!(store.latest_position(), LogPosition(2));
    assert_eq!(store.graph_quarantine().len(), 1);
    assert_eq!(store.facts_quarantine().len(), 1);
    assert_eq!(store.graph_quarantine()[0].position, LogPosition(1));
    assert_eq!(store.facts_quarantine()[0].position, LogPosition(1));
    assert_eq!(
        store
            .get_node_record("claim:valid")
            .map(|node| node.properties["status"].clone()),
        Some(Value::String("accepted".to_string()))
    );
    Ok(())
}

#[test]
fn public_facade_rejects_fold_invalid_payload_before_append() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    let invalid = NodeRecord::new("claim:invalid", ["Claim"], json!("not-an-object"));

    let error = store.upsert_node(invalid).unwrap_err();
    assert_eq!(error.code, "facts_fold_validation_failed");
    assert_eq!(store.log().len(), 0);
    assert_eq!(store.latest_position(), LogPosition::ZERO);
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:graphstore-fold-differential").unwrap()
}
