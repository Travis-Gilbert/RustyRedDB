use theorem_surface_contracts::sync::{ChangesSince, MutationKind as WireKind, MutationLogEntry};

fn live_node(id: &str) -> NodeRecord {
    NodeRecord::new(id, ["Live"], json!({"k": id}))
}

fn live_edge(id: &str, from: &str, to: &str) -> EdgeRecord {
    EdgeRecord::new(id, from, "LINKS", to, json!({}))
}

fn mixed_batch() -> GraphMutationBatch {
    GraphMutationBatch::new([
        GraphMutation::NodeUpsert(live_node("node:a")),
        GraphMutation::NodeUpsert(live_node("node:b")),
        GraphMutation::EdgeUpsert(live_edge("edge:ab", "node:a", "node:b")),
    ])
}

fn change_entries(result: ChangesSince) -> Vec<MutationLogEntry> {
    match result {
        ChangesSince::Changes { entries, .. } => entries,
        ChangesSince::Truncated { cursor } => {
            panic!("expected Changes, got Truncated at txn {}", cursor.txn_id)
        }
    }
}

fn entry_shape(entry: &MutationLogEntry) -> (WireKind, &str, u64) {
    (entry.kind, entry.id.as_str(), entry.sequence)
}

#[test]
fn mutation_log_appends_without_a_hook_emitter() {
    let mut store = RedCoreGraphStore::memory();
    let before = store.version();
    store.commit_batch(mixed_batch()).unwrap();
    let entries = change_entries(store.changes_since(&before).unwrap());
    assert!(!entries.is_empty());
    assert!(entries.iter().any(|entry| entry.id == "node:a"));
    assert!(entries.iter().any(|entry| entry.id == "edge:ab"));
}

#[test]
fn journal_restart_preserves_sync_cursor() {
    let data_dir = unique_test_dir("live-substrate-restart-cursor");
    let options = redcore_journal_test_options();
    let cursor;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store.commit_batch(mixed_batch()).unwrap();
        cursor = store.version();
        assert_eq!(cursor.txn_id, 1);
    }
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(reopened.version(), cursor);
    match reopened.changes_since(&cursor).unwrap() {
        ChangesSince::Changes { entries, .. } => assert!(entries.is_empty()),
        ChangesSince::Truncated { .. } => panic!("same cursor must not truncate"),
    }
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn incremental_and_full_commit_emit_shape_equal_log_entries() {
    let batch = mixed_batch();
    let mut incremental = RedCoreGraphStore::memory();
    let mut full = RedCoreGraphStore::memory();
    let before_inc = incremental.version();
    let before_full = full.version();
    incremental.commit_batch(batch.clone()).unwrap();
    full.commit_batch_without_incremental(batch).unwrap();
    let inc_entries = change_entries(incremental.changes_since(&before_inc).unwrap());
    let full_entries = change_entries(full.changes_since(&before_full).unwrap());
    let inc_shape: Vec<_> = inc_entries.iter().map(entry_shape).collect();
    let full_shape: Vec<_> = full_entries.iter().map(entry_shape).collect();
    assert_eq!(inc_shape, full_shape);
}

#[test]
fn one_batch_sequences_events_from_zero() {
    let mut store = RedCoreGraphStore::memory();
    let before = store.version();
    store.commit_batch(mixed_batch()).unwrap();
    let entries = change_entries(store.changes_since(&before).unwrap());
    let sequences: Vec<u64> = entries.iter().map(|entry| entry.sequence).collect();
    assert_eq!(
        sequences,
        (0..entries.len() as u64).collect::<Vec<_>>(),
        "one commit must number its log events 0..N"
    );
}

#[test]
fn changes_since_current_cursor_is_empty() {
    let mut store = RedCoreGraphStore::memory();
    match store.changes_since(&store.version()).unwrap() {
        ChangesSince::Changes { entries, .. } => assert!(entries.is_empty()),
        ChangesSince::Truncated { .. } => panic!("head cursor must yield empty Changes"),
    }
    store.commit_batch(mixed_batch()).unwrap();
    match store.changes_since(&store.version()).unwrap() {
        ChangesSince::Changes { entries, .. } => assert!(entries.is_empty()),
        ChangesSince::Truncated { .. } => panic!("head cursor must yield empty Changes"),
    }
}

#[test]
fn three_batches_pull_from_pre_commit_cursor() {
    let mut store = RedCoreGraphStore::memory();
    let before = store.version();
    for idx in 0..3 {
        store
            .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                live_node(&format!("node:{idx}")),
            )]))
            .unwrap();
    }
    let entries = change_entries(store.changes_since(&before).unwrap());
    let ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
    assert!(ids.contains(&"node:0"));
    assert!(ids.contains(&"node:1"));
    assert!(ids.contains(&"node:2"));
    assert_eq!(store.version().txn_id, 3);
}

#[test]
fn changes_since_older_than_log_base_is_truncated() {
    let mut store = RedCoreGraphStore::memory();
    let before = store.version();
    for idx in 0..3 {
        store
            .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                live_node(&format!("node:{idx}")),
            )]))
            .unwrap();
    }
    store.drop_log_before_txn_for_test(3);
    match store.changes_since(&before).unwrap() {
        ChangesSince::Truncated { cursor } => assert_eq!(cursor, store.version()),
        ChangesSince::Changes { .. } => panic!("a cursor older than log_base must truncate"),
    }
}

#[test]
fn changes_since_swung_root_is_truncated() {
    let mut swung = RedCoreGraphStore::memory();
    swung.commit_batch(mixed_batch()).unwrap();
    let since = swung.version();
    swung.swing_sync_root_for_test(crate::hash::Hash::of(b"swung-sync-root"));
    match swung.changes_since(&since).unwrap() {
        ChangesSince::Truncated { cursor } => assert_eq!(cursor, swung.version()),
        ChangesSince::Changes { .. } => panic!("a swung root must truncate"),
    }
}

#[test]
fn mixed_log_ids_match_recorded_committed_mutations() {
    let mut store = RedCoreGraphStore::memory();
    store.start_recording_committed_mutations();
    let before = store.version();
    store.commit_batch(mixed_batch()).unwrap();
    let recorded = store
        .take_recorded_committed_mutations()
        .expect("recording session returns a batch");
    let log_ids: Vec<String> = change_entries(store.changes_since(&before).unwrap())
        .into_iter()
        .map(|entry| entry.id)
        .collect();
    let recorded_ids: Vec<String> = recorded
        .mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::NodeUpsert(node) => Some(node.id.clone()),
            GraphMutation::EdgeUpsert(edge) => Some(edge.id.clone()),
            GraphMutation::NodeDelete(id) | GraphMutation::EdgeDelete(id) => Some(id.clone()),
        })
        .collect();
    for id in ["node:a", "node:b", "edge:ab"] {
        assert!(log_ids.iter().any(|logged| logged == id), "log missing {id}");
        assert!(
            recorded_ids.iter().any(|recorded| recorded == id),
            "recorded mutations missing {id}"
        );
    }
    for id in &log_ids {
        assert!(
            recorded_ids.contains(id),
            "log id {id} missing from take_recorded_committed_mutations"
        );
    }
}

#[test]
fn stale_expected_root_refuses_and_mutates_nothing() {
    let mut store = RedCoreGraphStore::memory();
    store
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            live_node("node:keep"),
        )]))
        .unwrap();
    let before = store.version();
    let err = store
        .commit_batch(GraphMutationBatch {
            mutations: vec![GraphMutation::NodeUpsert(live_node("node:stale"))],
            expected_root: Some(crate::hash::Hash::of(b"not-the-current-root")),
        })
        .expect_err("stale expected_root must refuse");
    assert_eq!(err.code, "stale_write_base");
    assert_eq!(err.cursor, Some(before));
    assert!(store.get_node("node:stale").unwrap().is_none());
    assert!(store.get_node("node:keep").unwrap().is_some());
    assert_eq!(store.version(), before);
}

#[test]
fn expected_root_matching_current_commits() {
    let mut store = RedCoreGraphStore::memory();
    store.commit_batch(mixed_batch()).unwrap();
    let current = store.version();
    store
        .commit_batch(GraphMutationBatch {
            mutations: vec![GraphMutation::NodeUpsert(live_node("node:fresh"))],
            expected_root: Some(crate::hash::Hash::from_bytes(current.root)),
        })
        .unwrap();
    assert!(store.get_node("node:fresh").unwrap().is_some());
}

#[test]
fn expected_root_none_still_commits() {
    let mut store = RedCoreGraphStore::memory();
    store
        .commit_batch(GraphMutationBatch {
            mutations: vec![GraphMutation::NodeUpsert(live_node("node:open"))],
            expected_root: None,
        })
        .unwrap();
    assert!(store.get_node("node:open").unwrap().is_some());
}
