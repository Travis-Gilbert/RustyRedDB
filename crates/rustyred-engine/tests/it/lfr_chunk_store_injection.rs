//! W04-B prerequisite: `RedCoreGraphStore::open_with_chunk_store` composes a
//! durable graph over an injected `UnifiedChunkStore`. These tests use only
//! the public API and the real projection log (`NativeLogFile` via
//! `ProjectionChunkStore::open_at`) so reopen means a second process-visible
//! open of the same bytes, not a shared in-memory handle.

use std::collections::BTreeMap;
use std::sync::Arc;

use rustyred_thg_core::browser_projection::{browser_red_core_options, ProjectionChunkStore};
use rustyred_thg_core::chunk_store::{ChunkStore, ChunkStoreCatalog, CommitMeta, MemChunkStore};
use rustyred_thg_core::{
    EdgeRecord, Hash, NeighborQuery, NodeQuery, NodeRecord, RedCoreDurability, RedCoreGraphStore,
    RedCoreOptions, ROOT_NAME_REDCORE,
};
use serde_json::json;

fn write_small_graph(store: &mut RedCoreGraphStore) {
    store
        .upsert_node(NodeRecord::new(
            "lfr:a",
            ["File"],
            json!({"path": "notes/a.md"}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "lfr:b",
            ["File"],
            json!({"path": "notes/b.md"}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "lfr:ab",
            "lfr:a",
            "LINKS",
            "lfr:b",
            json!({"rank": 1}),
        ))
        .unwrap();
}

fn assert_small_graph_intact(store: &RedCoreGraphStore) {
    let hits = store.query_nodes(NodeQuery::label("File")).unwrap();
    assert_eq!(hits.len(), 2, "both nodes survive reopen");
    assert_eq!(
        store.neighbors(NeighborQuery::out("lfr:a")).unwrap()[0].node_id,
        "lfr:b",
        "the edge survives reopen"
    );
    assert!(store.verify().unwrap().ok, "reopened graph verifies");
}

#[test]
fn lfr_injected_projection_store_reopens_committed_graph_over_the_same_log() {
    let dir = tempfile::tempdir().unwrap();
    let (committed_version, committed_root) = {
        let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
        let mut store =
            RedCoreGraphStore::open_with_chunk_store(backend, browser_red_core_options())
                .expect("compose over an injected projection store");
        assert_eq!(
            store.unified_chunk_store().read_root().unwrap(),
            None,
            "empty construction must not publish a catalog before its first graph commit"
        );
        write_small_graph(&mut store);
        let version = store.version();
        let root = store
            .unified_chunk_store()
            .read_root()
            .unwrap()
            .expect("a commit publishes a root");
        (version, root)
    };

    let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
    let reopened = RedCoreGraphStore::open_with_chunk_store(backend, browser_red_core_options())
        .expect("reopen the same log");
    assert_eq!(reopened.version(), committed_version);
    assert_eq!(
        reopened.unified_chunk_store().read_root().unwrap(),
        Some(committed_root),
        "the committed catalog root is the one the first open published"
    );
    assert_small_graph_intact(&reopened);
}

#[test]
fn lfr_injected_memory_backend_recovers_committed_state_from_its_catalog() {
    let backend: Arc<MemChunkStore> = Arc::new(MemChunkStore::new());
    let committed_version = {
        let mut store = RedCoreGraphStore::open_with_chunk_store(
            backend.clone(),
            RedCoreOptions {
                durability: RedCoreDurability::AofAlways,
                ..RedCoreOptions::default()
            },
        )
        .unwrap();
        write_small_graph(&mut store);
        store.version()
    };

    // The same backend handle, a fresh facade: recovery reads the catalog only.
    let reopened = RedCoreGraphStore::open_with_chunk_store(
        backend,
        RedCoreOptions {
            durability: RedCoreDurability::AofAlways,
            ..RedCoreOptions::default()
        },
    )
    .unwrap();
    assert_eq!(reopened.version(), committed_version);
    assert_small_graph_intact(&reopened);
}

#[test]
fn lfr_injected_store_refuses_a_catalog_whose_state_chunk_is_missing() {
    // A foreign or damaged catalog: RedCore's named root points at a chunk the
    // backend does not hold. Recovery must refuse by name, never open empty.
    let backend: Arc<MemChunkStore> = Arc::new(MemChunkStore::new());
    let missing = Hash::of(b"lfr: state chunk that was never written");
    backend
        .commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(missing))]),
            CommitMeta {
                parents: Vec::new(),
                author: "lfr-test".to_string(),
                message: "publish a dangling RedCore root".to_string(),
                timestamp_unix_ms: 0,
                graph_version: 0,
            },
        )
        .unwrap();
    let error = match RedCoreGraphStore::open_with_chunk_store(backend, browser_red_core_options())
    {
        Ok(_) => panic!("a dangling RedCore root must not open as an empty graph"),
        Err(error) => error,
    };
    assert_eq!(error.code, "redcore_journal_state_missing");
}

#[test]
fn lfr_injected_store_rejects_strict_mode_without_a_durable_policy() {
    let backend: Arc<MemChunkStore> = Arc::new(MemChunkStore::new());
    let error = match RedCoreGraphStore::open_with_chunk_store(
        backend,
        RedCoreOptions {
            durability: RedCoreDurability::None,
            strict_acid: true,
            ..RedCoreOptions::default()
        },
    ) {
        Ok(_) => panic!("strict ACID over no durable policy must be refused"),
        Err(error) => error,
    };
    assert_eq!(error.code, "redcore_strict_mode_invalid");
}

#[test]
fn lfr_injected_store_reports_backend_owned_durability() {
    let backend = Arc::new(MemChunkStore::new());
    let store =
        RedCoreGraphStore::open_with_chunk_store(backend, browser_red_core_options()).unwrap();
    let status = store.status();
    assert_eq!(status.mode, "injected");
    assert_eq!(status.commit_policy, "backend_owned");
    assert_eq!(status.data_dir, None);
    let memory_status = RedCoreGraphStore::memory().status();
    assert_eq!(memory_status.mode, "memory");
    assert_eq!(memory_status.commit_policy, "memory_only");
}

#[test]
fn lfr_injected_store_refuses_malformed_committed_state() {
    let backend = Arc::new(MemChunkStore::new());
    let malformed = b"not a canonical RedCore journal state";
    let root = Hash::of(malformed);
    backend.write(&[(root, malformed)]).unwrap();
    backend
        .commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(root))]),
            CommitMeta {
                parents: Vec::new(),
                author: "lfr-test".into(),
                message: "malformed state".into(),
                timestamp_unix_ms: 0,
                graph_version: 0,
            },
        )
        .unwrap();
    let error = match RedCoreGraphStore::open_with_chunk_store(backend, browser_red_core_options())
    {
        Ok(_) => panic!("malformed committed state must not reopen as empty"),
        Err(error) => error,
    };
    assert_eq!(error.code, "redcore_journal_state_format_mismatch");
}

#[cfg(feature = "test-fault-injection")]
#[test]
fn lfr_injected_first_flush_failure_reopens_empty_and_allows_retry() {
    let dir = tempfile::tempdir().unwrap();
    {
        let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
        let mut store =
            RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
                .unwrap();
        let version = store.version();
        backend.fail_next_publication_flush_for_test().unwrap();
        let error = store
            .upsert_node(NodeRecord::new("lfr:first", ["File"], json!({})))
            .expect_err("first commit flush refusal must leave the graph empty");
        assert_eq!(error.code, "chunk_store_browser_storage");
        assert_eq!(store.version(), version);
        assert_eq!(backend.read_root().unwrap(), None);
        assert!(store.get_node("lfr:first").unwrap().is_none());
    }
    let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
    let mut reopened =
        RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
            .unwrap();
    assert_eq!(backend.read_root().unwrap(), None);
    assert!(reopened.get_node("lfr:first").unwrap().is_none());
    reopened
        .upsert_node(NodeRecord::new("lfr:first", ["File"], json!({})))
        .expect("retry can publish the first graph commit");
    assert!(reopened.get_node("lfr:first").unwrap().is_some());
    assert!(backend.read_root().unwrap().is_some());
}

#[cfg(feature = "test-fault-injection")]
#[test]
fn lfr_injected_graph_flush_failure_preserves_live_and_reopened_commit() {
    let dir = tempfile::tempdir().unwrap();
    let (accepted_version, accepted_root) = {
        let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
        let mut store =
            RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
                .unwrap();
        write_small_graph(&mut store);
        let version = store.version();
        let root = backend.read_root().unwrap();
        backend.fail_next_publication_flush_for_test().unwrap();
        let error = store
            .upsert_node(NodeRecord::new("lfr:refused", ["File"], json!({})))
            .expect_err("flush refusal must refuse graph publication");
        assert_eq!(error.code, "chunk_store_browser_storage");
        assert_eq!(store.version(), version);
        assert_eq!(backend.read_root().unwrap(), root);
        assert!(store.get_node("lfr:refused").unwrap().is_none());
        assert_small_graph_intact(&store);
        (version, root)
    };
    let backend = Arc::new(ProjectionChunkStore::open_at(dir.path()).unwrap());
    let mut reopened =
        RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
            .unwrap();
    assert_eq!(reopened.version(), accepted_version);
    assert_eq!(backend.read_root().unwrap(), accepted_root);
    assert!(reopened.get_node("lfr:refused").unwrap().is_none());
    assert_small_graph_intact(&reopened);
    reopened
        .upsert_node(NodeRecord::new("lfr:refused", ["File"], json!({})))
        .expect("successful retry after reopening prior commit");
    assert!(reopened.get_node("lfr:refused").unwrap().is_some());
    assert!(reopened.version().txn_id > accepted_version.txn_id);
}
