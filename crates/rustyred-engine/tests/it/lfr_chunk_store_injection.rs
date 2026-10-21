//! W04-B prerequisite: `RedCoreGraphStore::open_with_chunk_store` composes a
//! durable graph over an injected `UnifiedChunkStore`. Projection reopen tests
//! use the public API and real projection log (`NativeLogFile` via
//! `ProjectionChunkStore::open_at`), reopening the same process-visible bytes.
//! Memory-backed fixtures separately exercise catalog admission and decoding.

use std::collections::BTreeMap;
use std::sync::Arc;

use rustyred_thg_core::browser_projection::{browser_red_core_options, ProjectionChunkStore};
use rustyred_thg_core::chunk_store::{
    CatalogCommitResult, ChunkStore, ChunkStoreCatalog, CommitMeta, MemChunkStore, RootCatalog,
};
use rustyred_thg_core::{
    EdgeRecord, GraphStoreResult, Hash, NeighborQuery, NodeQuery, NodeRecord, RedCoreDurability,
    RedCoreGraphStore, RedCoreOptions, ROOT_NAME_REDCORE,
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

fn committed_memory_backend() -> Arc<MemChunkStore> {
    let backend = Arc::new(MemChunkStore::new());
    let mut store =
        RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
            .unwrap();
    write_small_graph(&mut store);
    backend
}

// A named damaged-read fixture, not a native persistence oracle: all actual
// storage and catalog publication stay on MemChunkStore. Only one committed
// state's reads are hidden to exercise recovery's fail-closed catalog boundary.
struct MissingStateReadStore {
    inner: Arc<MemChunkStore>,
    missing_state: Hash,
}

impl ChunkStore for MissingStateReadStore {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        if address == self.missing_state {
            return Ok(None);
        }
        self.inner.get(address)
    }

    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.get(address).map(|bytes| bytes.is_some())
    }

    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let mut missing = Vec::new();
        for address in addresses {
            if !self.has(*address)? {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.inner.write(chunks)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.inner.commit(root, meta)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.inner.read_root()
    }
}

impl ChunkStoreCatalog for MissingStateReadStore {
    fn commit_catalog_updates(
        &self,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        self.inner.commit_catalog_updates(updates, meta)
    }
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
fn lfr_injected_store_refuses_publication_with_missing_state_chunk() {
    let backend = committed_memory_backend();
    let committed_root = backend.read_root().unwrap();
    let missing = Hash::of(b"lfr: state chunk that was never written");
    // A valid graph catalog is required first. The backend refuses a missing
    // named state chunk at publication, before recovery can encounter it.
    let error = backend
        .commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(missing))]),
            CommitMeta {
                parents: Vec::new(),
                author: "lfr-test".to_string(),
                message: "attempt a dangling RedCore root".to_string(),
                timestamp_unix_ms: 0,
                graph_version: 0,
            },
        )
        .expect_err("a dangling RedCore state root must not be committed");
    assert_eq!(error.code, "chunk_store_named_root_missing");
    assert_eq!(backend.read_root().unwrap(), committed_root);
    let reopened =
        RedCoreGraphStore::open_with_chunk_store(backend.clone(), browser_red_core_options())
            .expect("refused publication preserves the prior recoverable graph");
    assert_small_graph_intact(&reopened);
}

#[test]
fn lfr_injected_store_refuses_a_catalog_whose_state_chunk_is_missing() {
    let backend = committed_memory_backend();
    let catalog_root = backend.read_root().unwrap().expect("seeded graph catalog");
    let catalog = RootCatalog::from_bytes(&backend.get(catalog_root).unwrap().unwrap()).unwrap();
    let missing_state = catalog.get(ROOT_NAME_REDCORE).unwrap();
    let damaged_reads = Arc::new(MissingStateReadStore {
        inner: backend,
        missing_state,
    });
    assert!(!damaged_reads.has(missing_state).unwrap());
    let error =
        match RedCoreGraphStore::open_with_chunk_store(damaged_reads, browser_red_core_options()) {
            Ok(_) => panic!("missing committed state must not reopen as an empty graph"),
            Err(error) => error,
        };
    assert_eq!(error.code, "chunk_store_named_root_missing");
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
    let backend = committed_memory_backend();
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
