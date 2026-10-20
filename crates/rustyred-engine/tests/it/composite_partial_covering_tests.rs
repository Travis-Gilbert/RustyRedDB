use rustyred_thg_core::{
    CompositeIndex, CompositeIndexDefinition, CompositeIndexKey, CoveringIndex,
    CoveringIndexDefinition, NodeRecord, PartialIndex, PartialIndexDefinition,
    PartialPredicateClause, ReceiptScope,
};
use serde_json::json;

fn tenant_scope() -> ReceiptScope {
    ReceiptScope::from([
        ("tenant".to_string(), "Travis-Gilbert".to_string()),
        ("project".to_string(), "Theorem".to_string()),
    ])
}

fn node(id: &str, label: &str, version: u64, properties: serde_json::Value) -> NodeRecord {
    let mut node = NodeRecord::new(id, [label], properties);
    node.version = version;
    node
}

#[test]
fn composite_index_serves_scope_first_exact_and_left_prefix_queries() {
    let definition = CompositeIndexDefinition::new(
        "composite:artifact-list",
        "ContextArtifact",
        ["tenant", "project"],
        ["tenant", "project", "artifact_type", "updated_at"],
    );
    let mut index = CompositeIndex::new();
    index.register_definition(definition).unwrap();

    let scope = tenant_scope();
    let artifact = node(
        "artifact:1",
        "ContextArtifact",
        7,
        json!({
            "artifact_type": "postmortem",
            "updated_at": "2026-06-26T12:00:00Z",
            "title": "Recall incident"
        }),
    );
    index
        .upsert_node("composite:artifact-list", &artifact, &scope)
        .unwrap();

    let exact_key = CompositeIndexKey::from_properties(
        "composite:artifact-list",
        &[
            "tenant".to_string(),
            "project".to_string(),
            "artifact_type".to_string(),
            "updated_at".to_string(),
        ],
        &json!({
            "tenant": "Travis-Gilbert",
            "project": "Theorem",
            "artifact_type": "postmortem",
            "updated_at": "2026-06-26T12:00:00Z"
        }),
    )
    .unwrap();

    let exact = index.query_exact(&exact_key);
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].node_id, "artifact:1");
    assert_eq!(exact[0].source_version, 7);

    let prefix = exact_key.left_prefix(2);
    let prefixed = index.query_left_prefix(&prefix);
    assert_eq!(prefixed.len(), 1);
    assert_eq!(prefixed[0].node_id, "artifact:1");
}

#[test]
fn composite_index_lookup_matches_manual_scan_baseline() {
    let definition = CompositeIndexDefinition::new(
        "composite:artifact-list",
        "ContextArtifact",
        ["tenant", "project"],
        ["tenant", "project", "artifact_type", "updated_at"],
    );
    let mut index = CompositeIndex::new();
    index.register_definition(definition).unwrap();

    let scope = tenant_scope();
    let nodes = vec![
        node(
            "artifact:1",
            "ContextArtifact",
            7,
            json!({
                "artifact_type": "postmortem",
                "updated_at": "2026-06-26T12:00:00Z"
            }),
        ),
        node(
            "artifact:2",
            "ContextArtifact",
            8,
            json!({
                "artifact_type": "runbook",
                "updated_at": "2026-06-26T13:00:00Z"
            }),
        ),
        node(
            "artifact:3",
            "ContextArtifact",
            9,
            json!({
                "artifact_type": "postmortem",
                "updated_at": "2026-06-26T14:00:00Z"
            }),
        ),
    ];
    for node in &nodes {
        index
            .upsert_node("composite:artifact-list", node, &scope)
            .unwrap();
    }

    let prefix = CompositeIndexKey::from_properties(
        "composite:artifact-list",
        &[
            "tenant".to_string(),
            "project".to_string(),
            "artifact_type".to_string(),
        ],
        &json!({
            "tenant": "Travis-Gilbert",
            "project": "Theorem",
            "artifact_type": "postmortem"
        }),
    )
    .unwrap();
    let mut indexed = index
        .query_left_prefix(&prefix)
        .into_iter()
        .map(|entry| entry.node_id)
        .collect::<Vec<_>>();
    indexed.sort();

    let mut scanned = nodes
        .iter()
        .filter(|node| node.properties["artifact_type"] == "postmortem")
        .map(|node| node.id.clone())
        .collect::<Vec<_>>();
    scanned.sort();

    assert_eq!(indexed, scanned);
}

#[test]
fn composite_index_moves_updated_nodes_and_removes_tombstones() {
    let definition = CompositeIndexDefinition::new(
        "composite:artifact-list",
        "ContextArtifact",
        ["tenant", "project"],
        ["tenant", "project", "artifact_type", "updated_at"],
    );
    let mut index = CompositeIndex::new();
    index.register_definition(definition).unwrap();
    let scope = tenant_scope();

    let original = node(
        "artifact:1",
        "ContextArtifact",
        7,
        json!({
            "artifact_type": "postmortem",
            "updated_at": "2026-06-26T12:00:00Z"
        }),
    );
    let updated = node(
        "artifact:1",
        "ContextArtifact",
        8,
        json!({
            "artifact_type": "runbook",
            "updated_at": "2026-06-26T13:00:00Z"
        }),
    );

    index
        .upsert_node("composite:artifact-list", &original, &scope)
        .unwrap();
    index
        .upsert_node("composite:artifact-list", &updated, &scope)
        .unwrap();

    let old_key = CompositeIndexKey::from_properties(
        "composite:artifact-list",
        &[
            "tenant".to_string(),
            "project".to_string(),
            "artifact_type".to_string(),
            "updated_at".to_string(),
        ],
        &json!({
            "tenant": "Travis-Gilbert",
            "project": "Theorem",
            "artifact_type": "postmortem",
            "updated_at": "2026-06-26T12:00:00Z"
        }),
    )
    .unwrap();
    let new_key = CompositeIndexKey::from_properties(
        "composite:artifact-list",
        &[
            "tenant".to_string(),
            "project".to_string(),
            "artifact_type".to_string(),
            "updated_at".to_string(),
        ],
        &json!({
            "tenant": "Travis-Gilbert",
            "project": "Theorem",
            "artifact_type": "runbook",
            "updated_at": "2026-06-26T13:00:00Z"
        }),
    )
    .unwrap();

    assert!(index.query_exact(&old_key).is_empty());
    assert_eq!(index.query_exact(&new_key)[0].source_version, 8);

    let mut tombstone = updated.clone();
    tombstone.tombstone = true;
    index
        .upsert_node("composite:artifact-list", &tombstone, &scope)
        .unwrap();

    assert!(index.query_exact(&new_key).is_empty());
}

#[test]
fn composite_index_removes_stale_entry_when_new_key_is_invalid() {
    let definition = CompositeIndexDefinition::new(
        "composite:artifact-list",
        "ContextArtifact",
        ["tenant", "project"],
        ["tenant", "project", "artifact_type"],
    );
    let mut index = CompositeIndex::new();
    index.register_definition(definition).unwrap();
    let scope = tenant_scope();
    let original = node(
        "artifact:1",
        "ContextArtifact",
        1,
        json!({"artifact_type": "postmortem"}),
    );
    index
        .upsert_node("composite:artifact-list", &original, &scope)
        .unwrap();
    let key = CompositeIndexKey::from_properties(
        "composite:artifact-list",
        &[
            "tenant".to_string(),
            "project".to_string(),
            "artifact_type".to_string(),
        ],
        &json!({
            "tenant": "Travis-Gilbert",
            "project": "Theorem",
            "artifact_type": "postmortem"
        }),
    )
    .unwrap();
    assert_eq!(index.query_exact(&key).len(), 1);

    let invalid = node("artifact:1", "ContextArtifact", 2, json!({}));
    assert!(index
        .upsert_node("composite:artifact-list", &invalid, &scope)
        .is_err());
    assert!(index.query_exact(&key).is_empty());
}

#[test]
fn composite_index_rejects_non_scope_first_keys() {
    let mut index = CompositeIndex::new();
    let err = index
        .register_definition(CompositeIndexDefinition::new(
            "composite:bad",
            "ContextArtifact",
            ["tenant", "project"],
            ["artifact_type", "tenant", "project"],
        ))
        .expect_err("scope fields must lead the composite key");

    assert_eq!(err.code, "composite_index_not_scope_first");
}

#[test]
fn partial_index_tracks_active_subset_and_detects_predicate_drift() {
    let definition = PartialIndexDefinition::new(
        "partial:active-runs",
        "HarnessRun",
        vec![PartialPredicateClause::equals("status", json!("open"))],
    )
    .expect("partial-index predicate should be content-addressable");
    let mut index = PartialIndex::new(definition);

    let open = node("run:open", "HarnessRun", 3, json!({ "status": "open" }));
    let closed = node("run:closed", "HarnessRun", 4, json!({ "status": "closed" }));

    assert!(index.upsert_node(&open));
    assert!(!index.upsert_node(&closed));
    assert_eq!(index.ids(), ["run:open".to_string()].into_iter().collect());

    index
        .definition
        .clauses
        .push(PartialPredicateClause::equals("redaction_ok", json!(true)));
    index
        .definition
        .refresh_predicate_hash()
        .expect("drifted predicate should remain content-addressable");

    assert!(index.predicate_drifted());
}

#[test]
fn covering_index_serves_card_rows_and_marks_stale_source_versions() {
    let definition = CoveringIndexDefinition::new(
        "covering:artifact-card",
        "ContextArtifact",
        ["title", "status", "outcome"],
    );
    let mut index = CoveringIndex::new(definition);
    let artifact = node(
        "artifact:card",
        "ContextArtifact",
        11,
        json!({
            "title": "Advisor replay",
            "status": "ready",
            "outcome": "accepted",
            "large_body": "not copied into the card row"
        }),
    );

    let row = index.upsert_node(&artifact).unwrap();
    assert_eq!(row.object_id, "artifact:card");
    assert_eq!(row.source_version, 11);
    assert!(row.fields.contains_key("title"));
    assert!(!row.fields.contains_key("large_body"));

    assert!(!index.mark_stale_if_older("artifact:card", 10));
    assert!(index.mark_stale_if_older("artifact:card", 12));
    assert!(index.get("artifact:card").unwrap().stale);
}
