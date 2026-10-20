use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use rustyred_thg_core::{
    read_manifest, ChunkStore, GraphSnapshot, Hash, InMemoryThgExecutor, JournalChunkStore,
    NeighborQuery, NodeQuery, NodeRecord, RedCoreDurability, RedCoreGraphStore, RedCoreOptions,
    ThgCommand, ThgExecutor,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct SnapshotEnvelope {
    version: u32,
    txn_id: u64,
    graph: GraphSnapshot,
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", process::id()))
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let source_path = entry.path();
        let target_path = dst.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target_path);
        } else {
            fs::copy(&source_path, &target_path).unwrap();
        }
    }
}

fn copied_redcore_fixture() -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/redcore-v1");
    let dst = unique_temp_dir("redcore-snapshot-replay");
    copy_dir_recursive(&src, &dst);
    dst
}

fn thg_state_hash(snapshot: &GraphSnapshot) -> String {
    let mut executor = InMemoryThgExecutor::new();
    let response = executor
        .execute(
            ThgCommand::StateHash,
            json!({"state": serde_json::to_value(snapshot).unwrap()}),
        )
        .unwrap();
    response
        .payload
        .get("hash")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string()
}

fn durable_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    }
}

fn assert_no_legacy_artifacts(data_dir: &Path) {
    for artifact in [
        "graph.aof.current",
        "graph.snapshot.current",
        "graph.snapshot.previous",
        "manifest.json",
    ] {
        assert!(
            !data_dir.join(artifact).exists(),
            "normal journal path recreated legacy artifact {artifact}"
        );
    }
}

#[test]
fn redcore_fixture_migration_matches_snapshot_hash_stats_and_query_smoke() {
    let data_dir = copied_redcore_fixture();
    let manifest = read_manifest(&data_dir).unwrap().unwrap();
    assert_eq!(manifest.format_kind, "redcore");
    assert_eq!(manifest.snapshot_txn_id, 3);

    let raw_snapshot = fs::read_to_string(data_dir.join("graph.snapshot.current")).unwrap();
    let expected: SnapshotEnvelope = serde_json::from_str(&raw_snapshot).unwrap();
    assert_eq!(expected.version, 1);

    // The v1 fixture predates the historical-node registry. Recovery must still
    // classify every node present in that snapshot as historically observed,
    // while keeping the legacy history-completeness flag false.
    let mut expected_graph = expected.graph.clone();
    expected_graph
        .historical_node_ids
        .extend(expected_graph.nodes.iter().map(|node| node.id.clone()));
    let expected_hash = thg_state_hash(&expected_graph);

    let options = durable_options();
    let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();

    let replayed = store.graph_snapshot();
    assert_eq!(replayed, expected_graph);
    assert_eq!(thg_state_hash(&replayed), expected_hash);

    let status = store.status();
    assert_eq!(status.snapshot_txn_id, expected.txn_id);
    assert!(status.last_recovery_ok);

    let stats = store.stats().unwrap();
    assert_eq!(stats.version, 3);
    assert_eq!(stats.nodes_total, 2);
    assert_eq!(stats.edges_total, 1);
    assert_eq!(stats.labels_total, 1);
    assert_eq!(stats.edge_types_total, 1);

    let docs = store.query_nodes(NodeQuery::label("Doc")).unwrap();
    assert_eq!(docs.len(), 2);

    let cites = store
        .neighbors(NeighborQuery::out("doc:a").with_edge_type("CITES"))
        .unwrap();
    assert_eq!(cites.len(), 1);
    assert_eq!(cites[0].node_id, "doc:b");
    assert_eq!(cites[0].confidence, Some(0.9));

    let verify = store.verify().unwrap();
    assert!(verify.ok);
    assert_eq!(verify.stats, stats);

    drop(store);

    assert_no_legacy_artifacts(&data_dir);
    let backup_dir = data_dir.join("legacy-redcore-v1.backup");
    assert!(backup_dir.join("graph.snapshot.current").is_file());
    assert!(backup_dir.join("manifest.json").is_file());
    assert!(data_dir.join("redcore.journal").is_file());

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(reopened.graph_snapshot(), expected_graph);
    assert_eq!(thg_state_hash(&reopened.graph_snapshot()), expected_hash);
    assert_eq!(reopened.status().recovered_frames, 1);
    assert!(reopened.verify().unwrap().ok);
    drop(reopened);
    fs::remove_dir_all(data_dir).ok();
}

#[test]
fn uncommitted_journal_chunk_does_not_advance_recovery_root() {
    let data_dir = unique_temp_dir("redcore-uncommitted-chunk");
    let options = durable_options();
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new("node:one", ["Doc"], json!({"step": 1})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("node:two", ["Doc"], json!({"step": 2})))
            .unwrap();
    }

    let journal_path = data_dir.join("redcore.journal");
    let committed_root;
    let orphan_bytes = b"written without a root publication barrier";
    let orphan_address = Hash::of(orphan_bytes);
    {
        let journal = JournalChunkStore::open(&journal_path).unwrap();
        committed_root = journal.read_root().unwrap().unwrap();
        journal
            .write(&[(orphan_address, orphan_bytes.as_slice())])
            .unwrap();
        assert_eq!(journal.read_root().unwrap(), Some(committed_root));
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:one").unwrap().is_some());
    assert!(store.get_node("node:two").unwrap().is_some());
    assert_eq!(store.status().recovered_frames, 1);
    drop(store);

    let journal = JournalChunkStore::open(&journal_path).unwrap();
    assert_eq!(journal.read_root().unwrap(), Some(committed_root));
    assert!(journal.has(orphan_address).unwrap());
    drop(journal);
    assert_no_legacy_artifacts(&data_dir);
    fs::remove_dir_all(data_dir).ok();
}

#[test]
fn committed_journal_root_advances_and_recovers_large_payload() {
    let data_dir = unique_temp_dir("redcore-journal-root-advance");
    let options = durable_options();
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:base",
                ["Doc"],
                json!({"small": true}),
            ))
            .unwrap();
    }
    let first_root = {
        let journal = JournalChunkStore::open(data_dir.join("redcore.journal")).unwrap();
        journal.read_root().unwrap().unwrap()
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:large-delta",
                ["Doc"],
                json!({"payload": "x".repeat(128 * 1024)}),
            ))
            .unwrap();
    }

    let second_root = {
        let journal = JournalChunkStore::open(data_dir.join("redcore.journal")).unwrap();
        journal.read_root().unwrap().unwrap()
    };
    assert_ne!(second_root, first_root);

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:large-delta").unwrap().is_some());
    assert_eq!(store.status().last_txn_id, 2);
    assert_eq!(store.status().recovered_frames, 1);

    drop(store);
    assert_no_legacy_artifacts(&data_dir);
    fs::remove_dir_all(data_dir).ok();
}

#[test]
fn recovery_truncates_torn_journal_tail_after_committed_root() {
    let data_dir = unique_temp_dir("redcore-torn-journal-tail");
    let options = durable_options();
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:base",
                ["Doc"],
                json!({"small": true}),
            ))
            .unwrap();
    }

    let journal_path = data_dir.join("redcore.journal");
    let committed_len = fs::metadata(&journal_path).unwrap().len();
    let committed_root = {
        let journal = JournalChunkStore::open(&journal_path).unwrap();
        journal.read_root().unwrap().unwrap()
    };
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&journal_path)
        .unwrap();
    file.write_all(b"torn-journal-frame").unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert!(fs::metadata(&journal_path).unwrap().len() > committed_len);

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:base").unwrap().is_some());
    assert_eq!(store.status().recovered_frames, 1);
    drop(store);

    assert_eq!(fs::metadata(&journal_path).unwrap().len(), committed_len);
    let journal = JournalChunkStore::open(&journal_path).unwrap();
    assert_eq!(journal.read_root().unwrap(), Some(committed_root));
    drop(journal);
    assert_no_legacy_artifacts(&data_dir);
    fs::remove_dir_all(data_dir).ok();
}
