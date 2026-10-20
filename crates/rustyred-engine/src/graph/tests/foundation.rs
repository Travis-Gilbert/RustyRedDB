use super::RecordedRedCoreMutation;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::chunk_store::{ChunkStore, ChunkStoreCatalog};
use crate::CONTROL_EVENT_LABEL;
// These four live in `crate::graph::snapshot`. Import them by that path rather
// than through `super`, which does not mean what this file's location suggests:
// the path says `graph/tests/foundation.rs`, but the file is `include!`d by
// `graph/tests.rs`, which `graph_store.rs` mounts as its own `mod tests`. So
// the code lands in `graph_store::tests` and `super` is `graph_store`, which
// carries none of these names. `graph/in_memory.rs` imports the same set the
// same way.
use crate::graph::snapshot::{
    serialized_json_bytes, BorrowedGraphSnapshot, EdgeValues, MapValues,
};
use crate::reachability::{
    compact_committed_reachable, walk_committed_reachable, CoreReachabilityCodec,
};
use crate::tensor_derivation::{InMemoryTensorBlockPayloadStore, TensorBlockPayloadStore};
use crate::vector_payload::{read_vector_property, VectorPropertyRef};
use crate::versioned_graph::{graph_payload_address, GraphPayloadRecord};
use serde_json::json;

use super::{
    legacy_redcore_artifact_names, sanitize_tenant_segment,
    BorrowedRedCoreSnapshotEnvelope, ConditionalNodePropertyUpdate, Direction, EdgeRecord,
    GraphMutation, GraphMutationBatch, GraphRead, GraphSnapshot, GraphStore, GraphStoreResult,
    GraphVectorPayloadAccess, InMemoryGraphStore, JournalCommitPolicy,
    MemoryDocumentQuery, NeighborQuery, NodeQuery, NodeRecord, RedCoreDurability,
    RedCoreGraphStore, RedCoreMutation, RedCoreOptions, RedCoreSnapshotEnvelope, VectorDesignation,
    VectorIndex, VectorIndexShape, REDCORE_GROUP_COMMIT_MAX_WAIT, REDCORE_MANIFEST_VERSION,
};
use super::{serialize_vector_fallback_reasons, VectorSearchBackend};

#[derive(Debug)]
struct FaultingVectorPinStore {
    state: Mutex<FaultingVectorPinState>,
}

struct FailOnceClaimedReleasePayloadStore {
    inner: Arc<dyn TensorBlockPayloadStore>,
    fail_next_claimed_release: AtomicBool,
    claimed_release_calls: AtomicUsize,
}

impl TensorBlockPayloadStore for FailOnceClaimedReleasePayloadStore {
    fn put_tensor_block_bytes(
        &self,
        expected_content_id: &str,
        bytes: &[u8],
    ) -> GraphStoreResult<()> {
        self.inner
            .put_tensor_block_bytes(expected_content_id, bytes)
    }

    fn get_tensor_block_bytes(&self, content_id: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        self.inner.get_tensor_block_bytes(content_id)
    }

    fn bind_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<()> {
        self.inner
            .bind_tensor_block_pending(content_id, publication_key)
    }

    fn release_bound_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        self.inner
            .release_bound_tensor_block_pending(content_id, publication_key)
    }

    fn claim_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        self.inner.claim_tensor_block_pin(content_id)
    }

    fn release_claimed_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        self.claimed_release_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_next_claimed_release.swap(false, Ordering::SeqCst) {
            return Err(super::GraphStoreError::new(
                "injected_postcommit_claimed_pin_release_failure",
                "injected post-commit claimed-pin release failure",
            ));
        }
        self.inner.release_claimed_tensor_block_pin(content_id)
    }

    fn release_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        self.inner.release_tensor_block_pin(content_id)
    }
}

#[derive(Debug)]
struct FaultingVectorPinState {
    bound: usize,
    claimed: usize,
    fail_next_claimed_release: bool,
}

impl TensorBlockPayloadStore for FaultingVectorPinStore {
    fn put_tensor_block_bytes(
        &self,
        _expected_content_id: &str,
        _bytes: &[u8],
    ) -> GraphStoreResult<()> {
        Ok(())
    }

    fn get_tensor_block_bytes(&self, _content_id: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn release_bound_tensor_block_pending(
        &self,
        _content_id: &str,
        _publication_key: &str,
    ) -> GraphStoreResult<bool> {
        let mut state = self.state.lock().expect("faulting pin state");
        if state.bound == 0 {
            return Ok(false);
        }
        state.bound -= 1;
        Ok(true)
    }

    fn release_claimed_tensor_block_pin(&self, _content_id: &str) -> GraphStoreResult<()> {
        let mut state = self.state.lock().expect("faulting pin state");
        if state.fail_next_claimed_release {
            state.fail_next_claimed_release = false;
            return Err(super::GraphStoreError::new(
                "injected_claimed_pin_release_failure",
                "injected claimed-pin release failure",
            ));
        }
        if state.claimed == 0 {
            return Err(super::GraphStoreError::new(
                "claimed_pin_underflow",
                "claimed pin underflow",
            ));
        }
        state.claimed -= 1;
        Ok(())
    }
}

#[test]
fn vector_pin_cleanup_does_not_repeat_a_successful_bound_release() {
    let store = Arc::new(FaultingVectorPinStore {
        state: Mutex::new(FaultingVectorPinState {
            bound: 2,
            claimed: 1,
            fail_next_claimed_release: true,
        }),
    });
    let handle: Arc<dyn TensorBlockPayloadStore> = store.clone();
    let mut guard = super::VectorPayloadPinGuard::new(handle);
    guard.extend([super::VectorPayloadPinReceipt {
        content_id: crate::hash::Hash::of(b"shared vector").to_prefixed(),
        publication_key: "vector-property:test-owner".to_string(),
        pending_released: false,
        claimed: true,
    }]);

    let error = guard.cancel().expect_err("first claimed release fails");
    assert_eq!(error.code, "injected_claimed_pin_release_failure");
    let state = store.state.lock().expect("faulting pin state");
    assert_eq!(state.bound, 1, "another writer's receipt must remain");
    assert_eq!(state.claimed, 0, "Drop retries only the claimed component");
}

#[test]
fn redcore_postcommit_vector_pin_cleanup_failure_does_not_fail_durable_transaction() {
    let data_dir = unique_test_dir("redcore-postcommit-vector-cleanup");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
    let inner_payload_store = store.store.vector_payload_store_handle();
    let faulting_store = Arc::new(FailOnceClaimedReleasePayloadStore {
        inner: inner_payload_store,
        fail_next_claimed_release: AtomicBool::new(true),
        claimed_release_calls: AtomicUsize::new(0),
    });
    let payload_handle: Arc<dyn TensorBlockPayloadStore> = faulting_store.clone();
    store.store.set_vector_payload_store(payload_handle);
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();
    let root_before = store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .unwrap();

    let transaction = store
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new(
                "doc:postcommit-cleanup",
                ["Doc"],
                json!({"embedding": [1.0, 0.0, -0.5]}),
            ),
        )]))
        .expect("durable graph publication remains successful");
    assert_eq!(transaction.txn_id, 2);
    let root_after = store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .unwrap();
    assert_ne!(root_after, root_before);
    assert_eq!(
        faulting_store.claimed_release_calls.load(Ordering::SeqCst),
        2
    );

    let node = store.get_node("doc:postcommit-cleanup").unwrap().unwrap();
    let reference = VectorPropertyRef::from_value(&node.properties["embedding"]).unwrap();
    let address = crate::Hash::parse_prefixed(&reference.content_id).unwrap();
    assert_eq!(
        store
            .journal
            .as_ref()
            .unwrap()
            .in_flight_pin_count(address)
            .unwrap(),
        0,
        "Drop retry releases the retained claimed component"
    );

    drop(store);
    drop(faulting_store);
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened
        .get_node("doc:postcommit-cleanup")
        .unwrap()
        .is_some());
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_reachability_decoder_returns_only_physical_tree_roots() {
    let logical_root = crate::hash::Hash::of(b"logical reachability root");
    let payload_root = crate::hash::Hash::of(b"payload reachability root");
    let historical_root = crate::hash::Hash::of(b"historical reachability root");
    let logical_commit = crate::hash::Hash::of(b"semantic commit metadata");
    let state = super::RedCoreJournalState {
        txn_id: 7,
        graph_version: 11,
        pack_graph_version: 11,
        logical_tree: crate::versioned_graph::GraphProllyTree {
            root: logical_root,
            entries_total: 0,
        },
        payload_tree: crate::versioned_graph::GraphProllyTree {
            root: payload_root,
            entries_total: 0,
        },
        logical_commit,
        nodes_total: 0,
        edges_total: 0,
        historical_node_tree: Some(crate::versioned_graph::GraphProllyTree {
            root: historical_root,
            entries_total: 0,
        }),
        legacy_historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: true,
        vector_designations: Vec::new(),
        ordered_designations: Vec::new(),
        mutation_log: crate::mutation_log::MutationLog::genesis(),
        history_root: None,
    };
    let bytes = state.to_bytes().unwrap();
    let address = crate::hash::Hash::of(&bytes);

    assert_eq!(
        super::decode_redcore_reachability_roots(address, &bytes).unwrap(),
        (logical_root, payload_root, Some(historical_root))
    );
    assert_ne!(logical_commit, logical_root);
    assert_ne!(logical_commit, payload_root);
    assert_ne!(logical_commit, historical_root);

    let error = super::decode_redcore_reachability_roots(
        crate::hash::Hash::of(b"wrong state address"),
        &bytes,
    )
    .unwrap_err();
    assert_eq!(error.code, "redcore_journal_state_address_mismatch");

    let mut unsupported = bytes;
    unsupported[1] = unsupported[1].wrapping_add(1);
    let unsupported_address = crate::hash::Hash::of(&unsupported);
    let error =
        super::decode_redcore_reachability_roots(unsupported_address, &unsupported).unwrap_err();
    assert_eq!(error.code, "redcore_journal_state_format_mismatch");
}

fn inline_history_v2_state_bytes_for_test(state: &super::RedCoreJournalState) -> Vec<u8> {
    assert!(state.historical_node_tree.is_none());
    assert!(state.vector_designations.is_empty());
    assert!(state.ordered_designations.is_empty());

    let mut output = vec![
        super::REDCORE_JOURNAL_STATE_DOMAIN,
        super::INLINE_HISTORY_REDCORE_JOURNAL_STATE_VERSION,
    ];
    output.extend_from_slice(&state.txn_id.to_le_bytes());
    output.extend_from_slice(&state.graph_version.to_le_bytes());
    output.extend_from_slice(&state.pack_graph_version.to_le_bytes());
    output.extend_from_slice(state.logical_tree.root.as_bytes());
    output.extend_from_slice(
        &super::redcore_state_u64(state.logical_tree.entries_total, "logical entries")
            .unwrap()
            .to_le_bytes(),
    );
    output.extend_from_slice(state.payload_tree.root.as_bytes());
    output.extend_from_slice(
        &super::redcore_state_u64(state.payload_tree.entries_total, "payload entries")
            .unwrap()
            .to_le_bytes(),
    );
    output.extend_from_slice(state.logical_commit.as_bytes());
    output.extend_from_slice(
        &super::redcore_state_u64(state.nodes_total, "node count")
            .unwrap()
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &super::redcore_state_u64(state.edges_total, "edge count")
            .unwrap()
            .to_le_bytes(),
    );
    output.push(u8::from(state.historical_node_history_complete));
    output.extend_from_slice(
        &super::redcore_state_u32(
            state.legacy_historical_node_ids.len(),
            "historical node id count",
        )
        .unwrap()
        .to_le_bytes(),
    );
    for id in &state.legacy_historical_node_ids {
        super::push_redcore_state_string(&mut output, id, "historical node id").unwrap();
    }
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output
}

#[test]
fn redcore_journal_state_v2_decodes_and_v3_size_is_history_cardinality_independent() {
    let logical_root = crate::Hash::of(b"v2 logical root");
    let payload_root = crate::Hash::of(b"v2 payload root");
    let mut state = super::RedCoreJournalState {
        txn_id: 4,
        graph_version: 9,
        pack_graph_version: 9,
        logical_tree: crate::versioned_graph::GraphProllyTree {
            root: logical_root,
            entries_total: 0,
        },
        payload_tree: crate::versioned_graph::GraphProllyTree {
            root: payload_root,
            entries_total: 0,
        },
        logical_commit: crate::Hash::of(b"v2 logical commit"),
        nodes_total: 0,
        edges_total: 0,
        historical_node_tree: None,
        legacy_historical_node_ids: BTreeSet::from([
            "history:a".to_string(),
            "history:b".to_string(),
        ]),
        historical_node_history_complete: true,
        vector_designations: Vec::new(),
        ordered_designations: Vec::new(),
        mutation_log: crate::mutation_log::MutationLog::genesis(),
        history_root: None,
    };

    let decoded =
        super::RedCoreJournalState::from_bytes(&inline_history_v2_state_bytes_for_test(&state))
            .unwrap();
    assert_eq!(
        decoded.legacy_historical_node_ids,
        state.legacy_historical_node_ids
    );
    assert!(decoded.historical_node_tree.is_none());
    assert!(decoded.historical_node_history_complete);

    state.legacy_historical_node_ids.clear();
    state.historical_node_tree = Some(crate::versioned_graph::GraphProllyTree {
        root: crate::Hash::of(b"small historical tree"),
        entries_total: 1,
    });
    let one_id_state_len = state.to_bytes().unwrap().len();
    state.historical_node_tree = Some(crate::versioned_graph::GraphProllyTree {
        root: crate::Hash::of(b"large historical tree"),
        entries_total: 500_000,
    });
    assert_eq!(state.to_bytes().unwrap().len(), one_id_state_len);
}

#[test]
fn redcore_historical_registry_is_exact_across_v2_migration_compaction_and_reopen() {
    let data_dir = unique_test_dir("redcore-v2-history-migration-compaction");
    let options = redcore_journal_test_options();
    let expected_historical_ids;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store.upsert_node(plain_node("history:retired")).unwrap();
        assert!(store.delete_node("history:retired").unwrap());

        let journal = Arc::clone(store.journal.as_ref().unwrap());
        let catalog_root = journal.read_root().unwrap().unwrap();
        let catalog =
            crate::chunk_store::validate_catalog_root(journal.as_ref(), catalog_root).unwrap();
        let state_address = catalog.get(crate::chunk_store::ROOT_NAME_REDCORE).unwrap();
        let state_bytes = journal.get(state_address).unwrap().unwrap();
        let mut inline_state = super::RedCoreJournalState::from_bytes(&state_bytes).unwrap();
        inline_state.historical_node_tree = None;
        inline_state.legacy_historical_node_ids = store.store.historical_node_ids.clone();
        let inline_bytes = inline_history_v2_state_bytes_for_test(&inline_state);
        let inline_address = crate::Hash::of(&inline_bytes);
        journal
            .write(&[(inline_address, inline_bytes.as_slice())])
            .unwrap();
        journal
            .commit_catalog_updates(
                &std::collections::BTreeMap::from([(
                    crate::chunk_store::ROOT_NAME_REDCORE.to_string(),
                    Some(inline_address),
                )]),
                crate::chunk_store::CommitMeta {
                    parents: Vec::new(),
                    author: "redcore-v2-migration-test".to_string(),
                    message: "republish inline-history v2 state".to_string(),
                    timestamp_unix_ms: super::unix_ms(),
                    graph_version: inline_state.graph_version,
                },
            )
            .unwrap();
    }

    {
        let mut migrated = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        assert!(migrated.historical_node_tree.is_none());
        assert!(migrated.get_node("history:retired").unwrap().is_none());
        assert!(migrated
            .insert_node_if_never_existed(plain_node("history:retired"))
            .unwrap()
            .is_none());
        assert!(migrated
            .insert_node_if_never_existed(plain_node("history:new"))
            .unwrap()
            .is_some());
        assert!(migrated.delete_node("history:new").unwrap());

        let journal = Arc::clone(migrated.journal.as_ref().unwrap());
        let catalog_root = journal.read_root().unwrap().unwrap();
        let catalog =
            crate::chunk_store::validate_catalog_root(journal.as_ref(), catalog_root).unwrap();
        let state_address = catalog.get(crate::chunk_store::ROOT_NAME_REDCORE).unwrap();
        let state_bytes = journal.get(state_address).unwrap().unwrap();
        let state = super::RedCoreJournalState::from_bytes(&state_bytes).unwrap();
        let historical_tree = state.historical_node_tree.as_ref().unwrap();
        expected_historical_ids = migrated.store.historical_node_ids.clone();
        assert!(expected_historical_ids.contains("history:new"));
        assert!(expected_historical_ids.contains("history:retired"));
        assert_eq!(
            super::hydrate_historical_node_registry(journal.as_ref(), historical_tree).unwrap(),
            expected_historical_ids
        );
        assert!(state.legacy_historical_node_ids.is_empty());

        compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec).unwrap();
    }

    let mut reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened
        .insert_node_if_never_existed(plain_node("history:retired"))
        .unwrap()
        .is_none());
    assert!(reopened
        .insert_node_if_never_existed(plain_node("history:new"))
        .unwrap()
        .is_none());
    assert_eq!(reopened.store.historical_node_ids, expected_historical_ids);
    assert!(reopened.verify().unwrap().ok);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_historical_registry_incremental_tree_hydrates_exactly_across_leaves() {
    let store = crate::chunk_store::MemChunkStore::new();
    let initial = (0..137)
        .map(|index| format!("history:{index:04}"))
        .collect::<BTreeSet<_>>();
    let tree =
        super::build_historical_node_registry(&store, initial.iter().map(String::as_str)).unwrap();
    assert_eq!(tree.entries_total, initial.len());
    assert_eq!(
        super::hydrate_historical_node_registry(&store, &tree).unwrap(),
        initial
    );

    let additions = BTreeSet::from([
        "history:0007".to_string(),
        "history:9000".to_string(),
        "history:9001".to_string(),
    ]);
    let updated =
        super::update_historical_node_registry(&store, &tree, additions.iter().map(String::as_str))
            .unwrap();
    let expected = initial.union(&additions).cloned().collect::<BTreeSet<_>>();
    assert_eq!(updated.entries_total, expected.len());
    assert_eq!(
        super::hydrate_historical_node_registry(&store, &updated).unwrap(),
        expected
    );
}

#[test]
fn historical_registry_write_cost_is_bounded_across_prior_cardinality() {
    fn one_id_cost(prior_count: usize) -> crate::versioned_graph::TreeBuildCost {
        let store = crate::chunk_store::MemChunkStore::new();
        let initial = (0..prior_count)
            .map(|index| format!("history:{index:08}"))
            .collect::<BTreeSet<_>>();
        let tree =
            super::build_historical_node_registry(&store, initial.iter().map(String::as_str))
                .unwrap();
        let next_id = format!("history:{prior_count:08}");
        let upserts = super::historical_node_registry_entries(&store, [next_id.as_str()]).unwrap();
        crate::versioned_graph::build_prolly_tree_incremental(&store, &tree, &upserts)
            .unwrap()
            .cost
    }

    let small = one_id_cost(61);
    let large = one_id_cost(4_096);
    eprintln!(
        "historical registry one-ID cost: prior=61 chunks={} bytes={}; prior=4096 chunks={} bytes={}",
        small.chunks_written,
        small.bytes_written,
        large.chunks_written,
        large.bytes_written,
    );
    assert!(
        large.chunks_written <= small.chunks_written + 4,
        "one-ID historical update changed {} chunks over 61 prior IDs but {} over 4096",
        small.chunks_written,
        large.chunks_written,
    );
    assert!(
        large.bytes_written < small.bytes_written.saturating_mul(4),
        "one-ID historical update wrote {} bytes over 61 prior IDs but {} over 4096",
        small.bytes_written,
        large.bytes_written,
    );
}

#[test]
fn redcore_journal_growth_is_subquadratic_in_commit_count() {
    const WINDOW_COMMITS: usize = 256;

    let data_dir = unique_test_dir("redcore-journal-growth-curve");
    let options = redcore_journal_test_options();
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let journal_path = store.journal.as_ref().unwrap().path().to_path_buf();
    let bytes_at_zero = std::fs::metadata(&journal_path).unwrap().len();

    for index in 0..WINDOW_COMMITS {
        store
            .upsert_node(plain_node(&format!("growth:{index:08}")))
            .unwrap();
    }
    let bytes_at_n = std::fs::metadata(&journal_path).unwrap().len();

    for index in WINDOW_COMMITS..(WINDOW_COMMITS * 2) {
        store
            .upsert_node(plain_node(&format!("growth:{index:08}")))
            .unwrap();
    }
    let bytes_at_2n = std::fs::metadata(&journal_path).unwrap().len();
    let first_window = bytes_at_n - bytes_at_zero;
    let second_window = bytes_at_2n - bytes_at_n;
    eprintln!(
        "journal growth: commits=0 bytes={bytes_at_zero}; commits={WINDOW_COMMITS} bytes={bytes_at_n}; commits={} bytes={bytes_at_2n}; windows={first_window},{second_window}",
        WINDOW_COMMITS * 2,
    );

    assert!(
        bytes_at_2n < bytes_at_n.saturating_mul(3),
        "journal grew {bytes_at_n} -> {bytes_at_2n} bytes over {WINDOW_COMMITS} + {WINDOW_COMMITS} commits"
    );
    assert!(
        second_window.saturating_mul(2) < first_window.saturating_mul(5),
        "equal commit windows wrote {first_window} then {second_window} bytes; the second window must remain below 2.5x instead of approaching v2's 3x quadratic increment"
    );

    drop(store);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn borrowed_snapshot_serialization_matches_materialized_snapshot_size() {
    let mut store = InMemoryGraphStore::new();
    for task_number in 1..=37 {
        store
            .upsert_node(NodeRecord::new(
                format!("plan:task:{task_number:03}"),
                ["PlanTask"],
                json!({
                    "title": format!("Harness capability task {task_number}"),
                    "description": "acceptance evidence ".repeat(32),
                    "status": "pending",
                }),
            ))
            .unwrap();
    }

    let borrowed_bytes = serialized_json_bytes(&BorrowedGraphSnapshot {
        version: store.version,
        nodes: MapValues(&store.nodes),
        edges: EdgeValues(&store.edges),
        historical_node_ids: &store.historical_node_ids,
        historical_node_history_complete: store.historical_node_history_complete,
    });
    let materialized_bytes = serde_json::to_vec(&store.snapshot()).unwrap().len();

    assert_eq!(borrowed_bytes, materialized_bytes);
}

#[test]
fn borrowed_redcore_snapshot_envelope_matches_owned_schema() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Snapshot"],
            json!({ "name": "Ada" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Snapshot"],
            json!({ "name": "Grace" }),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "weight": 1 }),
        ))
        .unwrap();

    let owned = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph: store.snapshot(),
        historical_node_ids: store.historical_node_ids.clone(),
        historical_node_history_complete: Some(store.historical_node_history_complete),
    };
    let borrowed = BorrowedRedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph: BorrowedGraphSnapshot {
            version: store.version,
            nodes: MapValues(&store.nodes),
            edges: EdgeValues(&store.edges),
            historical_node_ids: &store.historical_node_ids,
            historical_node_history_complete: store.historical_node_history_complete,
        },
        historical_node_ids: &store.historical_node_ids,
        historical_node_history_complete: Some(store.historical_node_history_complete),
    };

    assert_eq!(
        serde_json::to_vec_pretty(&borrowed).unwrap(),
        serde_json::to_vec_pretty(&owned).unwrap()
    );
}

fn redcore_journal_test_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1,
        strict_acid: true,
    }
}

fn assert_no_legacy_redcore_files(data_dir: &Path) {
    for artifact in legacy_redcore_artifact_names() {
        assert!(
            !data_dir.join(artifact).exists(),
            "journal path recreated retired artifact {artifact}"
        );
    }
}

#[test]
fn redcore_durability_status_reports_requested_and_effective_policy() {
    let cases = [
        (RedCoreDurability::None, None, "memory_only", 0_u64),
        (
            RedCoreDurability::AofEverysec,
            Some(JournalCommitPolicy::Grouped {
                max_wait: REDCORE_GROUP_COMMIT_MAX_WAIT,
            }),
            "grouped_fsync",
            1_u64,
        ),
        (
            RedCoreDurability::AofAlways,
            Some(JournalCommitPolicy::Immediate),
            "immediate_fsync",
            0_u64,
        ),
        (
            RedCoreDurability::SnapshotOnly,
            Some(JournalCommitPolicy::Immediate),
            "immediate_fsync",
            0_u64,
        ),
    ];

    for (durability, expected_policy, expected_label, expected_window_ms) in cases {
        let data_dir = unique_test_dir(&format!("redcore-policy-{}", durability.as_str()));
        let store = RedCoreGraphStore::open(
            &data_dir,
            RedCoreOptions {
                durability,
                snapshot_interval_writes: u64::MAX,
                strict_acid: false,
            },
        )
        .unwrap();
        let status = store.status();
        assert_eq!(status.durability, durability.as_str());
        assert_eq!(status.commit_policy, expected_label);
        assert_eq!(status.commit_batch_window_ms, expected_window_ms);
        assert_eq!(
            store
                .journal
                .as_ref()
                .map(|journal| journal.commit_policy()),
            expected_policy
        );
        drop(store);
        std::fs::remove_dir_all(data_dir).ok();
    }
}

#[test]
fn redcore_legacy_durable_modes_share_the_journal_root_path() {
    for durability in [
        RedCoreDurability::AofEverysec,
        RedCoreDurability::AofAlways,
        RedCoreDurability::SnapshotOnly,
    ] {
        let data_dir = unique_test_dir(&format!("redcore-journal-{}", durability.as_str()));
        let options = RedCoreOptions {
            durability,
            snapshot_interval_writes: 1,
            strict_acid: true,
        };
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:durable",
                ["Claim"],
                json!({"requested": durability.as_str()}),
            ))
            .unwrap();
        assert!(store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .is_some());
        assert_no_legacy_redcore_files(&data_dir);
        drop(store);

        let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
        assert!(reopened.get_node("node:durable").unwrap().is_some());
        assert_eq!(reopened.status().last_txn_id, 1);
        assert_no_legacy_redcore_files(&data_dir);
        drop(reopened);
        std::fs::remove_dir_all(data_dir).ok();
    }
}

#[test]
fn redcore_memory_exposes_one_shared_unified_chunk_store() {
    let store = RedCoreGraphStore::memory();
    assert!(store.chunk_store_handle().is_none());
    let chunks = store.unified_chunk_store();
    let objects =
        crate::object_store::DiskObjectStore::from_chunk_store(Arc::clone(&chunks)).unwrap();
    let content_id = objects
        .put_document_bytes(b"shared memory payload")
        .unwrap();
    let address = crate::Hash::parse_prefixed(&content_id).unwrap();
    assert_eq!(
        chunks.get(address).unwrap().as_deref(),
        Some(b"shared memory payload".as_slice())
    );
    assert_eq!(
        objects.get_chunk(&content_id).unwrap().as_deref(),
        Some(b"shared memory payload".as_slice())
    );
}

#[test]
fn redcore_unified_chunk_handle_is_the_durable_restart_store() {
    let data_dir = unique_test_dir("redcore-unified-handle-restart");
    let options = redcore_journal_test_options();
    let content_id;
    {
        let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let chunks = store.unified_chunk_store();
        let objects = crate::object_store::DiskObjectStore::from_chunk_store(chunks).unwrap();
        content_id = objects.put_document_bytes(b"restart payload").unwrap();
        assert!(!data_dir.join("cold_objects/chunk-store.journal").exists());
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let chunks = reopened.unified_chunk_store();
    let address = crate::Hash::parse_prefixed(&content_id).unwrap();
    assert_eq!(
        chunks.get(address).unwrap().as_deref(),
        Some(b"restart payload".as_slice())
    );
    assert!(reopened.chunk_store_handle().is_some());
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_reopens_committed_batch_without_legacy_files() {
    let data_dir = unique_test_dir("redcore-journal-reopen");
    let options = redcore_journal_test_options();
    let committed_root;
    let expected_snapshot;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let transaction = store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:a",
                    ["File"],
                    json!({"path": "src/lib.rs"}),
                )),
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:b",
                    ["File"],
                    json!({"path": "src/main.rs"}),
                )),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ab",
                    "node:a",
                    "IMPORTS",
                    "node:b",
                    json!({"rank": 1}),
                )),
            ]))
            .unwrap();
        assert_eq!(transaction.graph_version, 5);
        assert_eq!(transaction.writes.len(), 3);
        assert_eq!(
            store
                .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
                .unwrap()
                .len(),
            2
        );
        expected_snapshot = store.graph_snapshot();
        committed_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
    }

    for legacy in [
        super::REDCORE_AOF_FILE,
        super::REDCORE_SNAPSHOT_FILE,
        super::REDCORE_PREVIOUS_SNAPSHOT_FILE,
        super::REDCORE_MANIFEST_FILE,
    ] {
        assert!(!data_dir.join(legacy).exists(), "legacy artifact {legacy}");
    }
    assert!(data_dir.join(super::REDCORE_JOURNAL_FILE).is_file());

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let reopened_snapshot = reopened.graph_snapshot();
    assert_eq!(reopened_snapshot.version, expected_snapshot.version);
    assert_eq!(reopened_snapshot.nodes, expected_snapshot.nodes);
    assert_eq!(reopened_snapshot.edges, expected_snapshot.edges);
    assert_eq!(reopened.status().last_txn_id, 1);
    assert_eq!(reopened.status().snapshot_txn_id, 1);
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(committed_root)
    );
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_doc_tree_payload_shares_the_open_store_and_reopens() {
    let data_dir = unique_test_dir("redcore-journal-doc-tree-shared-store");
    let options = redcore_journal_test_options();
    let body = b"one journal owns graph and document payload bytes";
    let content_hash;
    {
        let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let journal = store.chunk_store_handle().unwrap();
        let cold_store = store.cold_object_store().unwrap().unwrap();
        let chunks_before = journal.chunk_count().unwrap();
        content_hash = cold_store.put_document_bytes(body).unwrap();
        assert_eq!(journal.chunk_count().unwrap(), chunks_before + 1);
        assert_eq!(cold_store.put_chunk(body).unwrap(), content_hash);
        assert_eq!(journal.chunk_count().unwrap(), chunks_before + 1);
        assert_eq!(
            store
                .get_cold_document_bytes(&content_hash)
                .unwrap()
                .as_deref(),
            Some(body.as_slice())
        );
        assert_eq!(
            journal.path().canonicalize().unwrap(),
            data_dir
                .join(super::REDCORE_JOURNAL_FILE)
                .canonicalize()
                .unwrap()
        );
        assert!(!data_dir.join("cold_objects/chunk-store.journal").exists());
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        reopened
            .get_cold_document_bytes(&content_hash)
            .unwrap()
            .as_deref(),
        Some(body.as_slice())
    );
    assert!(!data_dir.join("cold_objects/chunk-store.journal").exists());
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_orphan_chunks_do_not_advance_committed_root() {
    use crate::hash::Hash;
    use crate::journal::JournalChunkStore;

    let data_dir = unique_test_dir("redcore-journal-orphan");
    let options = redcore_journal_test_options();
    let prior_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new("node:committed", ["Claim"], json!({})))
            .unwrap();
        prior_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
    }

    let orphan_bytes = b"written before a killed root barrier";
    let orphan = Hash::of(orphan_bytes);
    {
        let journal = JournalChunkStore::open(data_dir.join(super::REDCORE_JOURNAL_FILE)).unwrap();
        journal.write(&[(orphan, orphan_bytes.as_slice())]).unwrap();
        assert_eq!(journal.read_root().unwrap(), Some(prior_root));
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:committed").unwrap().is_some());
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(prior_root)
    );
    assert!(reopened.journal.as_ref().unwrap().has(orphan).unwrap());
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

fn abrupt_exit_orphan_node() -> NodeRecord {
    NodeRecord::new(
        "node:orphaned-after-abrupt-exit",
        ["Claim"],
        json!({"phase": "after-chunks-before-root"}),
    )
}

#[test]
fn redcore_journal_abrupt_process_exit_after_chunks_child() {
    let Some(journal_path) = std::env::var_os(super::REDCORE_TEST_EXIT_AFTER_CHUNKS_JOURNAL_ENV)
    else {
        return;
    };
    let data_dir = std::path::PathBuf::from(journal_path)
        .parent()
        .expect("test journal path has a parent")
        .to_path_buf();
    let mut store = RedCoreGraphStore::open(&data_dir, redcore_journal_test_options()).unwrap();
    assert_eq!(store.status().last_txn_id, 1);
    store.upsert_node(abrupt_exit_orphan_node()).unwrap();
    panic!("test-only abrupt-exit failpoint did not terminate the child process");
}

#[test]
fn redcore_journal_abrupt_process_exit_leaves_chunks_unreferenced() {
    let data_dir = unique_test_dir("redcore-journal-abrupt-exit");
    let options = redcore_journal_test_options();
    let prior_root;
    let prior_snapshot;
    let journal_path;
    let journal_len_before;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:committed-before-abrupt-exit",
                ["Claim"],
                json!({"phase": "committed"}),
            ))
            .unwrap();
        prior_snapshot = store.graph_snapshot();
        let journal = store.journal.as_ref().unwrap();
        prior_root = journal.read_root().unwrap().unwrap();
        journal_path = journal.path().to_path_buf();
        journal_len_before = std::fs::metadata(&journal_path).unwrap().len();
    }

    let mut orphan_record = abrupt_exit_orphan_node();
    orphan_record.version = prior_snapshot.version + 1;
    orphan_record.content_hash = Some(orphan_record.checksum().unwrap());
    let orphan_payload_address =
        graph_payload_address(&GraphPayloadRecord::Node(orphan_record)).unwrap();

    let output = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("graph_store::tests::redcore_journal_abrupt_process_exit_after_chunks_child")
        .arg("--nocapture")
        .env(super::REDCORE_TEST_EXIT_AFTER_CHUNKS_TXN_ENV, "2")
        .env(
            super::REDCORE_TEST_EXIT_AFTER_CHUNKS_JOURNAL_ENV,
            &journal_path,
        )
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(super::REDCORE_TEST_ABRUPT_EXIT_CODE),
        "child stdout:\n{}\nchild stderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        std::fs::metadata(&journal_path).unwrap().len() > journal_len_before,
        "abrupt child must append novel chunk frames before exiting"
    );

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(reopened.status().last_txn_id, 1);
    let reopened_snapshot = reopened.graph_snapshot();
    assert_eq!(reopened_snapshot.version, prior_snapshot.version);
    assert_eq!(reopened_snapshot.nodes, prior_snapshot.nodes);
    assert_eq!(reopened_snapshot.edges, prior_snapshot.edges);
    assert_eq!(
        reopened_snapshot.historical_node_ids,
        prior_snapshot.historical_node_ids
    );
    assert!(
        reopened
            .get_node("node:orphaned-after-abrupt-exit")
            .unwrap()
            .is_none(),
        "the uncommitted payload must not enter the recovered mirror"
    );
    let journal = reopened.journal.as_ref().unwrap();
    assert_eq!(journal.read_root().unwrap(), Some(prior_root));
    assert!(
        journal.has(orphan_payload_address).unwrap(),
        "the txn-2 payload bytes must remain physically present as an orphan"
    );
    assert!(reopened.verify().unwrap().ok);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_node_tombstone_blocks_stale_readmission_before_and_after_reopen() {
    let data_dir = unique_test_dir("redcore-journal-node-tombstone-readmit");
    let options = redcore_journal_test_options();
    let stale;
    let deleted_version;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:deleted",
                ["Claim"],
                json!({"body": "stale"}),
            ))
            .unwrap();
        stale = store.store.nodes["node:deleted"].clone();
        assert!(store.delete_node("node:deleted").unwrap());
        deleted_version = store.store.version;
        store.store.readmit_node(stale.clone()).unwrap();
        assert!(store.get_node("node:deleted").unwrap().is_none());
    }

    let mut reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:deleted").unwrap().is_none());
    assert!(reopened.store.version >= deleted_version);
    reopened.store.readmit_node(stale).unwrap();
    assert!(
        reopened.get_node("node:deleted").unwrap().is_none(),
        "the recovered tombstone must reject a stale cold-tier record"
    );
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_node_delete_cascades_incident_edges_and_recovers_consistently() {
    let data_dir = unique_test_dir("redcore-journal-node-delete-cascade");
    let options = redcore_journal_test_options();
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["Doc"], json!({}))),
                GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["Doc"], json!({}))),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ab",
                    "node:a",
                    "LINKS",
                    "node:b",
                    json!({}),
                )),
            ]))
            .unwrap();
        assert!(store.delete_node("node:b").unwrap());
        assert!(store.get_node("node:b").unwrap().is_none());
        assert!(store.get_edge("edge:ab").unwrap().is_none());
        assert!(store.verify().unwrap().ok);
        let stats = store.stats_without_memory_estimate().unwrap();
        assert_eq!(stats.nodes_total, 4);
        assert_eq!(store.query_nodes(NodeQuery::label("Doc")).unwrap().len(), 1);
        assert_eq!(
            store
                .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(stats.edges_total, 0);
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:a").unwrap().is_some());
    assert!(reopened.get_node("node:b").unwrap().is_none());
    assert!(reopened.get_edge("edge:ab").unwrap().is_none());
    assert!(reopened.verify().unwrap().ok);
    let stats = reopened.stats_without_memory_estimate().unwrap();
    assert_eq!(stats.nodes_total, 4);
    assert_eq!(
        reopened.query_nodes(NodeQuery::label("Doc")).unwrap().len(),
        1
    );
    assert_eq!(
        reopened
            .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
            .unwrap()
            .len(),
        3
    );
    assert_eq!(stats.edges_total, 0);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_empty_graph_designation_bootstraps_and_recovers() {
    let data_dir = unique_test_dir("redcore-journal-empty-designation");
    let options = redcore_journal_test_options();
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .designate_ordered_property("File", "path_rank")
            .unwrap();
        assert!(store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .is_some());
        assert!(store.graph_snapshot().nodes.is_empty());

        let journal = store.chunk_store_handle().unwrap();
        let orphan_bytes = b"empty-redcore global-gc orphan";
        let orphan = crate::hash::Hash::of(orphan_bytes);
        journal.write(&[(orphan, orphan_bytes)]).unwrap();

        let report = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec).unwrap();
        assert!(!journal.has(orphan).unwrap());
        assert!(report.compaction.chunks_removed >= 1);
        for address in report.reachability.reachable {
            assert!(journal.has(address).unwrap());
        }
    }
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        reopened.ordered_designations(),
        vec![crate::ordered::OrderedDesignation {
            label: "File".to_string(),
            property: "path_rank".to_string(),
        }]
    );
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_migrates_legacy_snapshot_once() {
    let data_dir = unique_test_dir("redcore-journal-legacy-migration");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut legacy_store = InMemoryGraphStore::new();
    legacy_store
        .upsert_node(NodeRecord::new(
            "node:legacy",
            ["Claim"],
            json!({"source": "snapshot"}),
        ))
        .unwrap();
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph: legacy_store.snapshot(),
        historical_node_ids: legacy_store.historical_node_ids.clone(),
        historical_node_history_complete: Some(true),
    };
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        serde_json::to_vec_pretty(&envelope).unwrap(),
    )
    .unwrap();

    let options = redcore_journal_test_options();
    {
        let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        assert!(store.get_node("node:legacy").unwrap().is_some());
        assert_eq!(store.status().last_txn_id, 7);
    }
    assert!(!data_dir.join(super::REDCORE_SNAPSHOT_FILE).exists());
    assert!(data_dir
        .join(super::REDCORE_LEGACY_BACKUP_DIR)
        .join(super::REDCORE_SNAPSHOT_FILE)
        .is_file());
    assert!(data_dir.join(super::REDCORE_JOURNAL_FILE).is_file());

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:legacy").unwrap().is_some());
    assert_eq!(reopened.status().last_txn_id, 7);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_migrates_legacy_cold_record_without_losing_payload() {
    let data_dir = unique_test_dir("redcore-journal-legacy-cold-migration");
    std::fs::create_dir_all(&data_dir).unwrap();
    let original = NodeRecord::new(
        "node:legacy-cold",
        ["MemoryNode"],
        json!({"body": "retained across migration"}),
    );
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 0,
        graph: GraphSnapshot::default(),
        historical_node_ids: BTreeSet::from([original.id.clone()]),
        historical_node_history_complete: Some(true),
    };
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        serde_json::to_vec_pretty(&envelope).unwrap(),
    )
    .unwrap();

    let mut marked = original.clone();
    marked.properties.as_object_mut().unwrap().insert(
        super::COLD_ROLLBACK_RECORD_PROPERTY.to_string(),
        json!(format!("v1:{}", original.checksum().unwrap())),
    );
    marked.content_hash = None;
    marked.content_hash = Some(marked.checksum().unwrap());
    let mutation =
        serde_json::to_string(&RedCoreMutation::Batch(vec![RedCoreMutation::NodeUpsert(
            marked,
        )]))
        .unwrap();
    let frame = format!(
        "{{\"magic\":\"RRGDB_AOF\",\"version\":{},\"txn_id\":1,\"graph_version\":1,\"timestamp_unix_ms\":0,\"payload_checksum\":\"{}\",\"mutation\":{mutation}}}\n",
        super::REDCORE_MANIFEST_VERSION,
        super::aof_payload_checksum(&mutation),
    );
    std::fs::write(data_dir.join(super::REDCORE_AOF_FILE), frame).unwrap();

    let rollback_dir = data_dir.join(super::REDCORE_COLD_ROLLBACK_DIR);
    std::fs::create_dir_all(&rollback_dir).unwrap();
    let rollback_name = format!(
        "{}.json",
        super::safe_filename(&super::stable_hash(&original.id).unwrap())
    );
    std::fs::write(
        rollback_dir.join(rollback_name),
        serde_json::to_vec(&super::RedCoreColdRollbackRecord {
            id: original.id.clone(),
            node: original.clone(),
        })
        .unwrap(),
    )
    .unwrap();

    let options = redcore_journal_test_options();
    {
        let migrated = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let node = migrated.get_node(&original.id).unwrap().unwrap();
        assert_eq!(node.labels, original.labels);
        assert_eq!(node.properties, original.properties);
        assert!(migrated.store.cold_nonresident_node_ids.is_empty());
    }
    assert!(data_dir
        .join(super::REDCORE_LEGACY_BACKUP_DIR)
        .join(super::REDCORE_COLD_ROLLBACK_DIR)
        .is_dir());

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let node = reopened.get_node(&original.id).unwrap().unwrap();
    assert_eq!(node.labels, original.labels);
    assert_eq!(node.properties, original.properties);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_vector_property_legacy_snapshot_migration_persists_payload_across_reopen() {
    let data_dir = unique_test_dir("redcore-journal-legacy-vector-migration");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut legacy_store = InMemoryGraphStore::new();
    legacy_store
        .upsert_node(NodeRecord::new(
            "node:legacy-vector",
            ["Doc"],
            json!({"embedding": [1.0, -0.0, 0.5]}),
        ))
        .unwrap();
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph: legacy_store.snapshot(),
        historical_node_ids: legacy_store.historical_node_ids.clone(),
        historical_node_history_complete: Some(true),
    };
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        serde_json::to_vec_pretty(&envelope).unwrap(),
    )
    .unwrap();
    let designation = RedCoreMutation::VectorDesignation(VectorDesignation {
        label: "Doc".to_string(),
        property: "embedding".to_string(),
        dimension: 3,
    });
    let mutation = serde_json::to_string(&designation).unwrap();
    let frame = format!(
            "{{\"magic\":\"RRGDB_AOF\",\"version\":{},\"txn_id\":8,\"graph_version\":2,\"timestamp_unix_ms\":0,\"payload_checksum\":\"{}\",\"mutation\":{mutation}}}\n",
            super::REDCORE_MANIFEST_VERSION,
            super::aof_payload_checksum(&mutation),
        );
    std::fs::write(data_dir.join(super::REDCORE_AOF_FILE), frame).unwrap();

    let options = redcore_journal_test_options();
    let expected_bits = [1.0_f32, -0.0, 0.5].map(f32::to_bits);
    {
        let migrated = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let node = migrated.get_node("node:legacy-vector").unwrap().unwrap();
        assert!(node.properties["embedding"].is_object());
        let payload_store = migrated.vector_payload_store().unwrap();
        let vector = read_vector_property(&node, "embedding", payload_store.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(
            vector.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            expected_bits
        );
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let node = reopened.get_node("node:legacy-vector").unwrap().unwrap();
    let payload_store = reopened.vector_payload_store().unwrap();
    let vector = read_vector_property(&node, "embedding", payload_store.as_ref())
        .unwrap()
        .unwrap();
    assert_eq!(
        vector.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected_bits
    );
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_preserving_open_stages_without_rewriting_legacy_node_bytes() {
    let data_dir = unique_test_dir("redcore-preserved-legacy-migration-stage");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut legacy = InMemoryGraphStore::new();
    legacy
        .upsert_node(NodeRecord::new(
            "node:legacy-vector",
            ["Doc"],
            json!({"embedding":[1.0,-0.0,0.5]}),
        ))
        .unwrap();
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph: legacy.snapshot(),
        historical_node_ids: legacy.historical_node_ids.clone(),
        historical_node_history_complete: Some(true),
    };
    let snapshot_bytes = serde_json::to_vec_pretty(&envelope).unwrap();
    std::fs::write(data_dir.join(super::REDCORE_SNAPSHOT_FILE), &snapshot_bytes).unwrap();
    let designation = RedCoreMutation::VectorDesignation(VectorDesignation {
        label: "Doc".into(),
        property: "embedding".into(),
        dimension: 3,
    });
    let mutation = serde_json::to_string(&designation).unwrap();
    let frame = format!(
        "{{\"magic\":\"RRGDB_AOF\",\"version\":{},\"txn_id\":8,\"graph_version\":2,\"timestamp_unix_ms\":0,\"payload_checksum\":\"{}\",\"mutation\":{mutation}}}\n",
        super::REDCORE_MANIFEST_VERSION, super::aof_payload_checksum(&mutation),
    );
    std::fs::write(data_dir.join(super::REDCORE_AOF_FILE), &frame).unwrap();
    let options = redcore_journal_test_options();
    let mut store =
        RedCoreGraphStore::open_preserving_legacy_vectors(&data_dir, options.clone()).unwrap();
    let node = store.get_node("node:legacy-vector").unwrap().unwrap();
    assert!(node.properties["embedding"].is_array());
    let record_bytes = serde_json::to_vec(&node).unwrap();
    let address = graph_payload_address(&GraphPayloadRecord::Node(node.clone())).unwrap();
    let original_payload = store.unified_chunk_store().get(address).unwrap().unwrap();
    let version = store.graph_version();
    let progress = || {
        let mut vector = vec![0.0; 256];
        vector[0] = 1.0;
        NodeRecord::new(
            "stage:fixture",
            ["MigrationProgressFixture"],
            json!({"semantic_vec_eg2_256":vector}),
        )
    };
    // A mismatched manifest/source guard is checked before any designation,
    // node write or legacy-vector conversion can publish.
    assert!(!store
        .commit_conditional_batch_with_vector_designation_if_version(
            version,
            vec![ConditionalNodePropertyUpdate {
                node_id: node.id.clone(),
                property: "embedding".into(),
                expected: json!([9.0, 0.0, 0.0]),
                new_value: json!([9.0, 0.0, 0.0])
            }],
            "MigrationProgressFixture",
            "semantic_vec_eg2_256",
            256,
            GraphMutationBatch::new([GraphMutation::NodeUpsert(progress())]),
        )
        .unwrap());
    assert_eq!(store.graph_version(), version);
    assert!(store
        .vector_designations()
        .iter()
        .all(|d| d.property != "semantic_vec_eg2_256"));
    assert!(store
        .commit_conditional_batch_with_vector_designation_if_version(
            version,
            vec![ConditionalNodePropertyUpdate {
                node_id: node.id.clone(),
                property: "embedding".into(),
                expected: node.properties["embedding"].clone(),
                new_value: node.properties["embedding"].clone()
            }],
            "MigrationProgressFixture",
            "semantic_vec_eg2_256",
            256,
            GraphMutationBatch::new([GraphMutation::NodeUpsert(progress())]),
        )
        .unwrap());
    assert_eq!(
        serde_json::to_vec(&store.get_node(&node.id).unwrap().unwrap()).unwrap(),
        record_bytes
    );
    assert_eq!(
        store
            .unified_chunk_store()
            .get(address)
            .unwrap()
            .unwrap()
            .as_ref(),
        original_payload.as_ref()
    );
    drop(store);
    let preserved =
        RedCoreGraphStore::open_preserving_legacy_vectors(&data_dir, options.clone()).unwrap();
    assert_eq!(
        serde_json::to_vec(&preserved.get_node(&node.id).unwrap().unwrap()).unwrap(),
        record_bytes
    );
    assert_eq!(
        preserved
            .unified_chunk_store()
            .get(address)
            .unwrap()
            .unwrap()
            .as_ref(),
        original_payload.as_ref()
    );
    match std::fs::read(data_dir.join(super::REDCORE_SNAPSHOT_FILE)) {
        Ok(remaining_snapshot) => assert_eq!(remaining_snapshot, snapshot_bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("legacy snapshot read failed: {error}"),
    }
    // Journal checkpointing may retire the legacy AOF. Its canonical legacy
    // snapshot and exact physical node payload remain the custody oracle above.
    match std::fs::read_to_string(data_dir.join(super::REDCORE_AOF_FILE)) {
        Ok(remaining_aof) => assert_eq!(remaining_aof, frame),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("legacy AOF read failed: {error}"),
    }
    drop(preserved);
    let ordinary = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(
        ordinary.get_node(&node.id).unwrap().unwrap().properties["embedding"].is_object(),
        "ordinary open keeps its legacy migration behavior"
    );
    assert_eq!(
        ordinary
            .unified_chunk_store()
            .get(address)
            .unwrap()
            .unwrap()
            .as_ref(),
        original_payload.as_ref()
    );
    drop(ordinary);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_builtin_vector_legacy_snapshot_migration_is_durable_and_idempotent() {
    let data_dir = unique_test_dir("redcore-journal-legacy-builtin-vector-migration");
    std::fs::create_dir_all(&data_dir).unwrap();
    let fixtures = [
        ("item:legacy", "Item", "embedding"),
        ("collection:legacy", "Collection", "label_embedding"),
        ("entity:legacy", "Entity", "entity_embedding"),
        ("shadow:legacy", "EpistemicShadow", "structural_role_vector"),
    ];
    let nodes = fixtures
        .iter()
        .map(|(id, label, property)| {
            NodeRecord::new(*id, [*label], json!({(*property): [1.0, -0.0, 0.5]}))
        })
        .collect();
    let graph = GraphSnapshot {
        version: 1,
        nodes,
        edges: Vec::new(),
        historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: true,
    };
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 7,
        graph,
        historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: Some(true),
    };
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        serde_json::to_vec_pretty(&envelope).unwrap(),
    )
    .unwrap();

    let options = redcore_journal_test_options();
    let expected_bits = [1.0_f32, -0.0, 0.5].map(f32::to_bits);
    let migrated_txn_id;
    {
        let migrated = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        assert!(migrated.vector_designations().is_empty());
        for (id, _, property) in fixtures {
            let node = migrated.get_node(id).unwrap().unwrap();
            assert!(node.properties[property].is_object());
            let payload_store = migrated.vector_payload_store().unwrap();
            let vector = read_vector_property(&node, property, payload_store.as_ref())
                .unwrap()
                .unwrap();
            assert_eq!(
                vector.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                expected_bits
            );
        }
        migrated_txn_id = migrated.status().last_txn_id;
        assert!(migrated_txn_id > 7);
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(reopened.status().last_txn_id, migrated_txn_id);
    assert!(reopened.vector_designations().is_empty());
    for (id, _, property) in fixtures {
        let node = reopened.get_node(id).unwrap().unwrap();
        let payload_store = reopened.vector_payload_store().unwrap();
        let vector = read_vector_property(&node, property, payload_store.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(
            vector.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            expected_bits
        );
    }
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_journal_migrates_legacy_edge_delete_with_stale_readmission_barrier() {
    let data_dir = unique_test_dir("redcore-journal-legacy-edge-delete");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut legacy_store = InMemoryGraphStore::new();
    legacy_store
        .upsert_node(NodeRecord::new("node:a", ["Doc"], json!({})))
        .unwrap();
    legacy_store
        .upsert_node(NodeRecord::new("node:b", ["Doc"], json!({})))
        .unwrap();
    legacy_store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "LINKS",
            "node:b",
            json!({}),
        ))
        .unwrap();
    let stale = legacy_store.edges["edge:ab"].clone();
    let envelope = RedCoreSnapshotEnvelope {
        version: super::REDCORE_MANIFEST_VERSION,
        txn_id: 1,
        graph: legacy_store.snapshot(),
        historical_node_ids: legacy_store.historical_node_ids.clone(),
        historical_node_history_complete: Some(true),
    };
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        serde_json::to_vec_pretty(&envelope).unwrap(),
    )
    .unwrap();

    let mutation = serde_json::to_string(&RedCoreMutation::EdgeDelete("edge:ab".into())).unwrap();
    let frame = format!(
            "{{\"magic\":\"RRGDB_AOF\",\"version\":{},\"txn_id\":2,\"graph_version\":4,\"timestamp_unix_ms\":0,\"payload_checksum\":\"{}\",\"mutation\":{mutation}}}\n",
            super::REDCORE_MANIFEST_VERSION,
            super::aof_payload_checksum(&mutation),
        );
    std::fs::write(data_dir.join(super::REDCORE_AOF_FILE), frame).unwrap();

    let options = redcore_journal_test_options();
    {
        let mut migrated = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        assert!(migrated.get_edge("edge:ab").unwrap().is_none());
        migrated.store.readmit_edge(stale.clone()).unwrap();
        assert!(migrated.get_edge("edge:ab").unwrap().is_none());
        assert!(migrated.verify().unwrap().ok);
    }
    let mut reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_edge("edge:ab").unwrap().is_none());
    reopened.store.readmit_edge(stale).unwrap();
    assert!(
        reopened.get_edge("edge:ab").unwrap().is_none(),
        "the migrated tombstone must survive a second journal reopen"
    );
    assert!(reopened.verify().unwrap().ok);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn memory_attribution_counts_owned_edge_copies_json_and_vectors() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "doc:a",
            ["Doc"],
            json!({
                "embedding": [0.25, 0.75],
                "title": "Attribution fixture",
            }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new("doc:b", ["Doc"], json!({})))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:doc:a:links:doc:b",
            "doc:a",
            "LINKS",
            "doc:b",
            json!({ "weights": [1, 2, 3], "source": "fixture" }),
        ))
        .unwrap();
    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();

    let report = store.memory_attribution();

    assert_eq!(report.node_records, 2);
    assert_eq!(report.live_nodes, 2);
    assert_eq!(report.tombstoned_nodes, 0);
    assert_eq!(report.edge_records, 1);
    assert_eq!(report.live_edges, 1);
    assert_eq!(report.tombstoned_edges, 0);
    assert_eq!(report.primary_edge_keys.values, 1);
    assert_eq!(report.edge_type_index.memberships, 1);
    assert_eq!(report.out_adjacency.memberships, 1);
    assert_eq!(report.in_adjacency.memberships, 1);
    assert_eq!(
        report.edge_type_index.members.values, 0,
        "edge indexes must store compact ordinals instead of owning edge-id strings"
    );
    assert_eq!(report.out_adjacency.members.values, 0);
    assert_eq!(report.in_adjacency.members.values, 0);
    assert!(report.edge_record_strings.values >= 4);
    assert!(report.edge_properties.array_slots_capacity_bytes > 0);
    assert_eq!(report.vector_indexes.len(), 1);
    assert_eq!(report.vector_indexes[0].point_count, 1);
    assert_eq!(report.vector_indexes[0].dimension, 2);
    assert!(report.vector_indexes[0].point_values_capacity_bytes >= 2 * 4);
    assert!(report.serialized_graph_bytes > 0);
    assert!(report.accounted_capacity_lower_bound_bytes > 0);
}

#[test]
fn projected_memory_upper_bound_covers_plan_projection_replacement() {
    let mut store = InMemoryGraphStore::new();
    for task_number in 1..=37 {
        store
            .upsert_node(NodeRecord::new(
                format!("plan:task:{task_number:03}"),
                ["PlanTask"],
                json!({
                    "description": "durable plan task ".repeat(64),
                    "status": "pending",
                }),
            ))
            .unwrap();
    }
    store
        .upsert_node(NodeRecord::new(
            "plan:projection",
            ["PlanProjection"],
            json!({ "markdown": "initial projection" }),
        ))
        .unwrap();

    let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
        "plan:projection",
        ["PlanProjection"],
        json!({ "markdown": "updated projection\n".repeat(256) }),
    ))]);
    let projected = store.projected_memory_bytes_upper_bound(&batch);

    let mut actual_store = InMemoryGraphStore::from_snapshot(store.snapshot()).unwrap();
    for mutation in batch.mutations {
        match mutation {
            GraphMutation::NodeUpsert(node) => {
                actual_store.upsert_node(node).unwrap();
            }
            GraphMutation::EdgeUpsert(edge) => {
                actual_store.upsert_edge(edge).unwrap();
            }
            GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                unreachable!("replacement fixture contains only upserts")
            }
        }
    }

    assert!(projected >= actual_store.stats().memory_bytes);
}

#[test]
fn records_have_stable_hashes_and_metadata() -> GraphStoreResult<()> {
    let mut node = NodeRecord::new(
        "node:1",
        ["Person", "Person", " User "],
        json!({ "name": "Ada" }),
    );
    node.version = 7;

    assert_eq!(node.labels, vec!["Person".to_string(), "User".to_string()]);
    assert!(node.checksum()?.starts_with("sha256:"));

    let mut edge = EdgeRecord::new(
        "edge:1",
        "node:1",
        "KNOWS",
        "node:2",
        json!({ "confidence": 0.9 }),
    );
    edge.version = 8;

    assert_eq!(edge.from_id, "node:1");
    assert_eq!(edge.to_id, "node:2");
    assert_eq!(edge.edge_type, "KNOWS");
    assert!(edge.checksum()?.starts_with("sha256:"));
    Ok(())
}

#[test]
fn memory_store_records_content_hash_parent_chain() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:1",
            ["Person"],
            json!({ "name": "Ada" }),
        ))
        .unwrap();
    let first_hash = store
        .get_node("node:1")
        .unwrap()
        .content_hash
        .clone()
        .unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:1",
            ["Person"],
            json!({ "name": "Ada Lovelace" }),
        ))
        .unwrap();
    let updated = store.get_node("node:1").unwrap();

    assert_ne!(updated.content_hash.as_ref().unwrap(), &first_hash);
    assert_eq!(updated.parent_hashes, vec![first_hash]);
}

#[test]
fn tenant_segment_encoding_distinguishes_separator_collisions() {
    assert_ne!(
        sanitize_tenant_segment("acme/prod"),
        sanitize_tenant_segment("acme.prod")
    );
    assert_eq!(sanitize_tenant_segment("acme/prod"), "pct_acme%2Fprod");
    assert_eq!(sanitize_tenant_segment("acme.prod"), "pct_acme.prod");
}

#[test]
fn tenant_segment_encoding_is_injective_over_pseudo_random_strings() {
    let mut seen = std::collections::BTreeMap::new();
    let mut state = 0xC0DEC0DE_u64;
    for _ in 0..10_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let len = (state % 24) as usize;
        let mut bytes = Vec::with_capacity(len);
        for offset in 0..len {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let byte = match (state >> ((offset % 8) * 8)) as u8 {
                0 => b'_',
                1 => b'-',
                2 => b'.',
                3 => b'/',
                4 => b'%',
                5 => b':',
                6 => b'{',
                7 => b'}',
                value => 0x20 + (value % 0x5f),
            };
            bytes.push(byte);
        }
        let original = String::from_utf8_lossy(&bytes).to_string();
        let encoded = sanitize_tenant_segment(&original);
        if let Some(prior) = seen.insert(encoded.clone(), original.clone()) {
            assert_eq!(
                prior, original,
                "tenant segment collision for {prior:?} and {original:?}: {encoded}"
            );
        }
    }
}

#[test]
fn memory_store_upserts_nodes_edges_and_adjacency() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({ "name": "Ada" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Person", "Engineer"],
            json!({ "name": "Grace" }),
        ))
        .unwrap();

    let write = store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "since": 1952 }),
        ))
        .unwrap();

    assert_eq!(write.id, "edge:ab");
    assert_eq!(store.get_node("node:a").unwrap().version, 1);
    assert_eq!(store.get_edge("edge:ab").unwrap().version, 3);
    assert_eq!(
        store.neighbors(NeighborQuery::out("node:a")),
        vec![super::NeighborHit {
            edge_id: "edge:ab".to_string(),
            node_id: "node:b".to_string(),
            edge_type: "KNOWS".to_string(),
            confidence: None,
            epistemic_type: None,
        }]
    );
    assert_eq!(
        store.neighbors(NeighborQuery::in_("node:b").with_edge_type("KNOWS"))[0].node_id,
        "node:a"
    );
    assert_eq!(store.verify().ok, true);
}

#[test]
fn label_and_edge_type_indexes_track_updates() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({"name": "Ada", "kind": "scientist"}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Person"],
            json!({"name": "Grace", "kind": "engineer"}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({}),
        ))
        .unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["System"],
            json!({"name": "Ada", "kind": "engine"}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "CALLS",
            "node:b",
            json!({}),
        ))
        .unwrap();

    assert_eq!(
        store.node_ids_for_label("Person"),
        vec!["node:b".to_string()]
    );
    assert_eq!(
        store.node_ids_for_label("System"),
        vec!["node:a".to_string()]
    );
    assert!(store.edge_ids_for_type("KNOWS").is_empty());
    assert_eq!(
        store.edge_ids_for_type("CALLS"),
        vec!["edge:ab".to_string()]
    );
    assert_eq!(
        store
            .neighbors(NeighborQuery {
                node_id: "node:a".to_string(),
                direction: Direction::Out,
                edge_type: Some("CALLS".to_string()),
                include_expired: false,
            })
            .len(),
        1
    );
    assert_eq!(
        store.node_ids_for_property("kind", &json!("engine")),
        vec!["node:a".to_string()]
    );
    assert!(store
        .node_ids_for_property("kind", &json!("scientist"))
        .is_empty());
    assert_eq!(store.verify().ok, true);
}

#[test]
fn compact_edge_indexes_preserve_lexical_query_order() {
    let mut store = InMemoryGraphStore::new();
    for node_id in ["node:a", "node:b", "node:c"] {
        store
            .upsert_node(NodeRecord::new(node_id, ["Node"], json!({})))
            .unwrap();
    }
    for (edge_id, to_id) in [("edge:z", "node:b"), ("edge:a", "node:c")] {
        store
            .upsert_edge(EdgeRecord::new(
                edge_id,
                "node:a",
                "KNOWS",
                to_id,
                json!({}),
            ))
            .unwrap();
    }

    assert_eq!(
        store.edge_ids_for_type("KNOWS"),
        vec!["edge:a".to_string(), "edge:z".to_string()]
    );
    assert_eq!(
        store
            .neighbors(NeighborQuery::out("node:a"))
            .into_iter()
            .map(|hit| hit.edge_id)
            .collect::<Vec<_>>(),
        vec!["edge:a".to_string(), "edge:z".to_string()]
    );
}

#[test]
fn edge_arena_memory_accounting_retains_freed_slot_capacity() {
    let mut store = InMemoryGraphStore::new();
    for node_id in ["node:a", "node:b"] {
        store
            .upsert_node(NodeRecord::new(node_id, ["Node"], json!({})))
            .unwrap();
    }
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({}),
        ))
        .unwrap();
    store.evict_edge("edge:ab").unwrap();

    let report = store.memory_attribution();

    assert_eq!(report.edge_records, 0);
    assert!(
        report.record_inline_bytes
            >= 2 * std::mem::size_of::<NodeRecord>() + std::mem::size_of::<Option<EdgeRecord>>()
    );
    assert!(report.record_vec_slots_capacity_bytes >= std::mem::size_of::<usize>());
    assert!(
        store.estimated_memory_bytes()
            >= std::mem::size_of::<Option<EdgeRecord>>() + std::mem::size_of::<usize>()
    );
}

#[test]
fn property_indexes_support_exact_node_seek() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["File"],
            json!({"path": "src/lib.rs", "repo": "rusty-red", "rank": 1}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["File"],
            json!({"path": "src/main.rs", "repo": "rusty-red", "rank": 2}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:c",
            ["Symbol"],
            json!({"path": "src/lib.rs", "repo": "rusty-red"}),
        ))
        .unwrap();

    let hits = store.query_nodes(
        NodeQuery::label("File")
            .with_property("repo", json!("rusty-red"))
            .with_property("path", json!("src/lib.rs")),
    );

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "node:a");
    assert_eq!(
        store.property_keys(),
        vec!["path".to_string(), "rank".to_string(), "repo".to_string()]
    );
    assert_eq!(store.stats().property_indexes_total, 5);
    assert_eq!(store.verify().ok, true);
}

#[test]
fn raw_string_property_index_supports_nul_exact_and_ordered_prefix_scans() {
    let mut store = InMemoryGraphStore::new();
    for (id, path) in [
        ("node:b2", ["a", "b2", "child"]),
        ("node:b-z", ["a", "b", "zeta"]),
        ("node:b-a", ["a", "b", "alpha"]),
    ] {
        let key = crate::doc_tree::PathKey::from_segments(path).unwrap();
        store
            .upsert_node(NodeRecord::new(
                id,
                ["File"],
                json!({"path": key.as_str(), "rank": 1, "string_rank": "1"}),
            ))
            .unwrap();
    }

    let exact = crate::doc_tree::PathKey::from_slash_path("a/b/alpha").unwrap();
    assert_eq!(
        store.node_ids_for_string_property_exact("path", exact.as_str()),
        vec!["node:b-a".to_string()]
    );
    let prefix = crate::doc_tree::PathPrefix::from_segments(["a", "b"]).unwrap();
    assert_eq!(
        store.node_ids_for_string_property_prefix("path", prefix.as_str()),
        vec!["node:b-a".to_string(), "node:b-z".to_string()],
        "ordered prefix scan must stop before the a/b2 collision sibling"
    );
    assert_eq!(
        store.node_ids_for_string_property_prefix(
            "path",
            crate::doc_tree::PathPrefix::root().as_str()
        ),
        vec![
            "node:b-a".to_string(),
            "node:b-z".to_string(),
            "node:b2".to_string()
        ]
    );
    for index in 0..250 {
        let path =
            crate::doc_tree::PathKey::from_slash_path(&format!("a/b/foreign-{index:03}")).unwrap();
        store
            .upsert_node(NodeRecord::new(
                format!("memory:{index:03}"),
                ["Memory"],
                json!({"path": path.as_str()}),
            ))
            .unwrap();
    }
    assert_eq!(
        store.node_ids_for_label_and_string_property_prefix("File", "path", prefix.as_str()),
        vec!["node:b-a".to_string(), "node:b-z".to_string()],
        "label intersection must not materialize foreign path-bearing nodes"
    );
    assert!(store
        .property_index
        .contains_key(&("path".to_string(), format!("s:{}", exact.as_str()))));
    assert!(store
        .property_index
        .contains_key(&("rank".to_string(), "1".to_string())));
    assert!(store
        .property_index
        .contains_key(&("string_rank".to_string(), "s:1".to_string())));
    assert!(store.verify().ok);
}

#[test]
fn property_index_replacement_removes_only_prior_exact_tokens() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:target",
            ["Symbol"],
            json!({"kind": "old", "repo": "shared"}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:old-peer",
            ["Symbol"],
            json!({"kind": "old", "repo": "shared"}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:other",
            ["Symbol"],
            json!({"kind": "other", "repo": "separate"}),
        ))
        .unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:target",
            ["Symbol"],
            json!({"kind": "new", "repo": "shared"}),
        ))
        .unwrap();

    assert_eq!(
        store.node_ids_for_property("kind", &json!("old")),
        vec!["node:old-peer".to_string()]
    );
    assert_eq!(
        store.node_ids_for_property("kind", &json!("other")),
        vec!["node:other".to_string()]
    );
    assert_eq!(
        store.node_ids_for_property("kind", &json!("new")),
        vec!["node:target".to_string()]
    );
    assert_eq!(
        store.node_ids_for_property("repo", &json!("shared")),
        vec!["node:old-peer".to_string(), "node:target".to_string()]
    );
    assert_eq!(store.verify().ok, true);
}

#[test]
fn upserting_edge_requires_live_endpoints() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("node:a", ["Person"], json!({})))
        .unwrap();

    let error = store
        .upsert_edge(EdgeRecord::new(
            "edge:missing",
            "node:a",
            "KNOWS",
            "node:missing",
            json!({}),
        ))
        .unwrap_err();

    assert_eq!(error.code, "missing_graph_endpoint");
    assert!(store.edge_ids_for_type("KNOWS").is_empty());
}

#[test]
fn conditional_batch_failure_leaves_no_partial_nodes_or_properties() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "goal:atomic",
            ["Goal"],
            json!({"status":"open","closure_receipt":null}),
        ))
        .unwrap();
    let version_before = store.stats().version;

    let error = store
        .commit_conditional_batch_atomic(
            vec![
                ConditionalNodePropertyUpdate {
                    node_id: "goal:atomic".to_string(),
                    property: "status".to_string(),
                    expected: json!("open"),
                    new_value: json!("closed"),
                },
                ConditionalNodePropertyUpdate {
                    node_id: "goal:atomic".to_string(),
                    property: "closure_receipt".to_string(),
                    expected: serde_json::Value::Null,
                    new_value: json!("receipt:atomic"),
                },
            ],
            GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "receipt:atomic",
                    ["GoalClosureReceipt"],
                    json!({}),
                )),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:injected-failure",
                    "goal:atomic",
                    "CLOSES_WITH",
                    "missing:receipt",
                    json!({}),
                )),
            ]),
        )
        .unwrap_err();

    assert_eq!(error.code, "missing_graph_endpoint");
    assert_eq!(store.stats().version, version_before);
    assert_eq!(
        store.get_node("goal:atomic").unwrap().properties["status"],
        "open"
    );
    assert_eq!(
        store.get_node("goal:atomic").unwrap().properties["closure_receipt"],
        serde_json::Value::Null
    );
    assert!(store.get_node("receipt:atomic").is_none());
    assert!(store.get_edge("edge:injected-failure").is_none());
}

#[test]
fn conditional_batch_rejects_duplicate_node_property_conditions() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "goal:duplicate-condition",
            ["Goal"],
            json!({"status":"open"}),
        ))
        .unwrap();
    let version_before = store.stats().version;

    let error = store
        .commit_conditional_batch_atomic(
            vec![
                ConditionalNodePropertyUpdate {
                    node_id: "goal:duplicate-condition".to_string(),
                    property: "status".to_string(),
                    expected: json!("open"),
                    new_value: json!("closed"),
                },
                ConditionalNodePropertyUpdate {
                    node_id: "goal:duplicate-condition".to_string(),
                    property: "status".to_string(),
                    expected: json!("closed"),
                    new_value: json!("archived"),
                },
            ],
            GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "receipt:duplicate-condition",
                ["GoalClosureReceipt"],
                json!({}),
            ))]),
        )
        .unwrap_err();

    assert_eq!(error.code, "duplicate_conditional_node_property");
    assert_eq!(store.stats().version, version_before);
    assert_eq!(
        store
            .get_node("goal:duplicate-condition")
            .unwrap()
            .properties["status"],
        "open"
    );
    assert!(store.get_node("receipt:duplicate-condition").is_none());
}

#[test]
fn conditional_batch_merges_same_node_upsert_without_losing_guarded_or_latest_values() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "goal:owned",
            ["Goal", "LatestLabel"],
            json!({
                "status":"open",
                "latest_unknown":"preserve",
                "caller_overrides_unguarded":"latest"
            }),
        ))
        .unwrap();
    let version_before = store.stats().version;

    let committed = store
        .commit_conditional_batch_atomic(
            vec![ConditionalNodePropertyUpdate {
                node_id: "goal:owned".to_string(),
                property: "status".to_string(),
                expected: json!("open"),
                new_value: json!("closed"),
            }],
            GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "goal:owned",
                    ["Goal", "CallerLabel"],
                    json!({
                        "status":"caller-overwrite",
                        "caller_field":"included",
                        "caller_overrides_unguarded":"caller"
                    }),
                )),
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "receipt:merged",
                    ["GoalClosureReceipt"],
                    json!({}),
                )),
            ]),
        )
        .unwrap();

    assert!(committed);
    assert_eq!(store.stats().version, version_before + 2);
    let goal = store.get_node("goal:owned").unwrap();
    assert_eq!(goal.properties["status"], "closed");
    assert_eq!(goal.properties["latest_unknown"], "preserve");
    assert_eq!(goal.properties["caller_field"], "included");
    assert_eq!(goal.properties["caller_overrides_unguarded"], "caller");
    assert!(goal.labels.iter().any(|label| label == "LatestLabel"));
    assert!(goal.labels.iter().any(|label| label == "CallerLabel"));
    assert!(store.get_node("receipt:merged").is_some());
}

#[test]
fn conditional_batch_rejects_delete_of_condition_owned_node() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "goal:owned",
            ["Goal"],
            json!({"status":"open"}),
        ))
        .unwrap();
    let version_before = store.stats().version;

    let error = store
        .commit_conditional_batch_atomic(
            vec![ConditionalNodePropertyUpdate {
                node_id: "goal:owned".to_string(),
                property: "status".to_string(),
                expected: json!("open"),
                new_value: json!("closed"),
            }],
            GraphMutationBatch::new([GraphMutation::NodeDelete("goal:owned".to_string())]),
        )
        .unwrap_err();

    assert_eq!(error.code, "conditional_batch_node_conflict");
    assert_eq!(store.stats().version, version_before);
    assert_eq!(
        store.get_node("goal:owned").unwrap().properties["status"],
        "open"
    );
}

#[test]
fn verify_detects_index_drift() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("node:a", ["Person"], json!({})))
        .unwrap();
    store
        .upsert_node(NodeRecord::new("node:b", ["Person"], json!({})))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({}),
        ))
        .unwrap();

    let edge_ordinal = store.edges.ordinal("edge:ab").unwrap();
    store
        .out_adjacency
        .get_mut(&("node:a".to_string(), "KNOWS".to_string()))
        .unwrap()
        .remove(&edge_ordinal);
    store
        .property_index
        .entry(("name".to_string(), "s:Ada".to_string()))
        .or_default()
        .insert("node:a".to_string());

    let report = store.verify();

    assert_eq!(report.ok, false);
    assert!(report
        .problems
        .iter()
        .any(|problem| problem.kind == "out_adjacency_drift"));
    assert!(report
        .problems
        .iter()
        .any(|problem| problem.kind == "property_index_drift"));
}

#[test]
fn rebuild_indexes_repairs_derived_index_drift() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({ "name": "Ada" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Person"],
            json!({ "name": "Grace" }),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({}),
        ))
        .unwrap();

    store.label_index.clear();
    let edge_ordinal = store.edges.ordinal("edge:ab").unwrap();
    store
        .out_adjacency
        .get_mut(&("node:a".to_string(), "KNOWS".to_string()))
        .unwrap()
        .remove(&edge_ordinal);
    store
        .property_index
        .entry(("name".to_string(), "s:Wrong".to_string()))
        .or_default()
        .insert("node:a".to_string());

    assert!(store.query_nodes(NodeQuery::label("Person")).is_empty());
    let report = store.rebuild_indexes().unwrap();

    assert_eq!(report.repaired, true);
    assert_eq!(report.before.ok, false);
    assert_eq!(report.after.ok, true);
    assert_eq!(store.query_nodes(NodeQuery::label("Person")).len(), 2);
    assert_eq!(
        store.neighbors(NeighborQuery::out("node:a"))[0].node_id,
        "node:b"
    );
    assert_eq!(
        store.node_ids_for_property("name", &json!("Ada")),
        vec!["node:a".to_string()]
    );
}

#[test]
fn rebuild_indexes_does_not_hide_canonical_edge_corruption() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("node:a", ["Person"], json!({})))
        .unwrap();
    store.edges.insert(
        "edge:missing".to_string(),
        EdgeRecord {
            id: "edge:missing".to_string(),
            from_id: "node:a".to_string(),
            to_id: "node:missing".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: json!({}),
            version: 9,
            tombstone: false,
            confidence: None,
            epistemic_type: None,
            provenance: None,
            content_hash: None,
            parent_hashes: Vec::new(),
        },
    );

    let report = store.rebuild_indexes().unwrap();

    assert_eq!(report.repaired, false);
    assert_eq!(report.after.ok, false);
    assert!(report
        .after
        .problems
        .iter()
        .any(|problem| problem.kind == "missing_to_endpoint"));
}

// ---------------------------------------------------------------------------
// Incremental committed-footprint accounting
//
// `estimated_memory_bytes` is served from maintained counters so the quota path
// stops serde-serializing the whole graph per write; `recompute_memory_bytes` is
// the whole-graph oracle these tests compare against after every step.
// ---------------------------------------------------------------------------

use crate::graph::footprint::graph_memory_recompute_invocations;

/// The serialized-snapshot term measured directly, exactly as the recompute
/// oracle measures it.
fn measured_snapshot_bytes(store: &InMemoryGraphStore) -> usize {
    serialized_json_bytes(&BorrowedGraphSnapshot {
        version: store.version,
        nodes: MapValues(&store.nodes),
        edges: EdgeValues(&store.edges),
        historical_node_ids: &store.historical_node_ids,
        historical_node_history_complete: store.historical_node_history_complete,
    })
}

fn assert_footprint_matches_oracle(store: &InMemoryGraphStore, step: &str) {
    assert_eq!(
        store.estimated_memory_bytes(),
        store.recompute_memory_bytes(),
        "maintained committed footprint drifted from the oracle after: {step}"
    );
}

#[test]
fn committed_snapshot_term_matches_the_serialized_snapshot() {
    let mut store = InMemoryGraphStore::new();
    assert_eq!(store.committed_snapshot_bytes(), measured_snapshot_bytes(&store));

    store
        .upsert_node(NodeRecord::new("node:a", ["Person"], json!({})))
        .unwrap();
    assert_eq!(store.committed_snapshot_bytes(), measured_snapshot_bytes(&store));

    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Person"],
            json!({ "name": "Grace" }),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({}),
        ))
        .unwrap();
    assert_eq!(store.committed_snapshot_bytes(), measured_snapshot_bytes(&store));

    // Enough writes to push the graph version past one digit, plus a tombstone:
    // both change the framing the snapshot writer emits.
    for revision in 0..12 {
        store
            .upsert_node(NodeRecord::new(
                "node:a",
                ["Person"],
                json!({ "revision": revision }),
            ))
            .unwrap();
    }
    store.apply_logical_node_delete("node:b").unwrap();
    assert!(store.version >= 10, "framing must be checked with a multi-digit version");
    assert_eq!(store.committed_snapshot_bytes(), measured_snapshot_bytes(&store));
}

#[test]
fn committed_footprint_matches_the_recompute_oracle_across_mutations() {
    let mut store = InMemoryGraphStore::new();
    assert_footprint_matches_oracle(&store, "empty store");

    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({ "name": "Ada", "rank": 1 }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "node insert");

    // Same id, same field widths: a same-size replacement.
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({ "name": "Ada", "rank": 2 }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "same-size node replacement");

    // Different size, different label set, different property token: the record
    // bytes and both node index memberships change.
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person", "Author"],
            json!({ "name": "Ada Lovelace", "rank": 3, "city": "London" }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "different-size node replacement");

    store
        .upsert_node(NodeRecord::new(
            "node:b",
            ["Person"],
            json!({ "name": "Grace" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:c",
            ["Paper"],
            json!({ "title": "Notes" }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "further node inserts");

    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "since": 1843 }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "edge insert");

    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "since": 1844, "note": "revised" }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "edge replacement");

    store
        .upsert_edge(EdgeRecord::new(
            "edge:ac",
            "node:a",
            "CITES",
            "node:c",
            json!({}),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "edge insert with a new type and adjacency key");

    // Ordered designation: designation key bytes, then member inserts and
    // member updates under it.
    store.designate_ordered_property("Person", "rank").unwrap();
    assert_footprint_matches_oracle(&store, "ordered designation");
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person", "Author"],
            json!({ "name": "Ada Lovelace", "rank": 5, "city": "London" }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "ordered member update under a designation");
    store
        .upsert_node(NodeRecord::new(
            "node:d",
            ["Person"],
            json!({ "name": "Alan", "rank": 3 }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "ordered member insert");

    // Vector designation rewrites legacy arrays into packed references, which
    // changes the node record the snapshot term charges for.
    store
        .upsert_node(NodeRecord::new(
            "node:e",
            ["Person"],
            json!({ "name": "Edsger", "embedding": [0.25, 0.75] }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "pre-designation vector array");
    store.designate_vector_property("Person", "embedding", 2).unwrap();
    assert_footprint_matches_oracle(&store, "vector designation and legacy migration");
    store
        .upsert_node(NodeRecord::new(
            "node:f",
            ["Person"],
            json!({ "name": "Frances", "embedding": [0.5, 0.5] }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "vector-designated property insert");
    store.index_vector("node:a", "embedding", &[0.1, 0.9]).unwrap();
    assert_footprint_matches_oracle(&store, "direct vector index insert");

    // Tombstones stay in the snapshot but leave every index.
    store.apply_logical_node_delete("node:f").unwrap();
    assert_footprint_matches_oracle(&store, "live -> tombstone node transition");
    store
        .upsert_node(NodeRecord::new(
            "node:f",
            ["Person"],
            json!({ "name": "Frances", "embedding": [0.5, 0.5] }),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "tombstone -> live node transition");
    store.apply_logical_edge_delete("edge:ac").unwrap();
    assert_footprint_matches_oracle(&store, "live -> tombstone edge transition");

    // Residency: eviction must drop the resident bytes, rehydration must restore
    // them to the same total.
    let before_eviction = store.estimated_memory_bytes();
    let evicted = store
        .evict_node("node:c")
        .unwrap()
        .expect("node:c is resident before eviction");
    assert_footprint_matches_oracle(&store, "resident node eviction");
    assert!(
        store.estimated_memory_bytes() < before_eviction,
        "evicting a node must drop the committed footprint"
    );
    store.readmit_node(evicted).unwrap();
    assert_footprint_matches_oracle(&store, "cold node rehydration");
    assert_eq!(store.estimated_memory_bytes(), before_eviction);

    let edge = store.get_edge_record("edge:ab").expect("edge:ab exists");
    store.evict_edge("edge:ab").unwrap();
    assert_footprint_matches_oracle(&store, "edge eviction");
    store.readmit_edge(edge).unwrap();
    assert_footprint_matches_oracle(&store, "edge rehydration");

    // TTL purge removes the node, its incident edges, and their index entries.
    store
        .upsert_node(crate::graph::ttl::node_with_ttl_property(
            NodeRecord::new("node:ttl", ["Person"], json!({ "name": "Temporary" })),
            Some(1),
        ))
        .unwrap();
    assert_footprint_matches_oracle(&store, "ttl node insert");
    let purged = store.drain_expired_node_ids_before(crate::graph::ttl::now_ms());
    assert_eq!(purged, vec!["node:ttl".to_string()]);
    assert_footprint_matches_oracle(&store, "ttl purge");

    // Hard delete and ordered-designation clear.
    store.apply_logical_edge_delete("edge:ab").unwrap();
    store.apply_recovered_delete("node:b").unwrap();
    assert_footprint_matches_oracle(&store, "hard node delete");
    store.clear_ordered_property("Person", "rank").unwrap();
    assert_footprint_matches_oracle(&store, "ordered designation clear");
    store.clear_vector_property("Person", "embedding").unwrap();
    assert_footprint_matches_oracle(&store, "vector designation clear");

    assert!(
        store.historical_node_ids.len() >= 6,
        "every written node id must stay in the historical registry"
    );
}

#[test]
fn repeated_record_replacement_does_not_accumulate_committed_footprint() {
    let mut store = InMemoryGraphStore::new();
    for (id, name) in [("node:a", "Ada"), ("node:b", "Grace"), ("node:c", "Alan")] {
        store
            .upsert_node(NodeRecord::new(
                id,
                ["Person"],
                json!({ "name": name, "rank": 1 }),
            ))
            .unwrap();
    }
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "since": 1843 }),
        ))
        .unwrap();

    // Fixed-width revisions keep the replacement records byte-identical in
    // shape, so the only legitimate growth across iterations is the graph
    // version gaining a decimal digit.
    let replace = |store: &mut InMemoryGraphStore, revision: usize| {
        for (id, name) in [("node:a", "Ada"), ("node:b", "Grace"), ("node:c", "Alan")] {
            store
                .upsert_node(NodeRecord::new(
                    id,
                    ["Person"],
                    json!({ "name": name, "rank": 1, "revision": format!("rev-{revision:04}") }),
                ))
                .unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "KNOWS",
                "node:b",
                json!({ "since": 1843, "revision": format!("rev-{revision:04}") }),
            ))
            .unwrap();
    };

    // Warm up once so the first replacement's parent-hash insertion is not
    // counted as growth.
    replace(&mut store, 0);
    assert_footprint_matches_oracle(&store, "first replacement");
    let baseline = store.estimated_memory_bytes();

    let mut largest_total_growth = 0usize;
    let mut largest_single_step = 0usize;
    let mut previous = baseline;
    for revision in 1..2_000 {
        replace(&mut store, revision);
        assert_footprint_matches_oracle(&store, "repeated replacement");
        let current = store.estimated_memory_bytes();
        largest_total_growth = largest_total_growth.max(current.saturating_sub(baseline));
        largest_single_step = largest_single_step.max(current.saturating_sub(previous));
        previous = current;
    }

    assert_eq!(store.nodes.len(), 3, "replacements must not add records");
    assert_eq!(store.edges.len(), 1, "replacements must not add records");
    // The only legitimate growth is the graph version gaining a decimal digit:
    // once in the snapshot framing and once in each of the four replaced
    // records, so at most five bytes in one iteration. Re-charging a record
    // would add that record's entire serialized size, hundreds of bytes, every
    // iteration.
    assert!(
        largest_single_step <= 5,
        "one replacement grew the committed footprint by {largest_single_step} bytes"
    );
    assert!(
        largest_total_growth <= 16,
        "2000 replacements of byte-identical records grew the committed footprint by \
         {largest_total_growth} bytes; the footprint must track record content, not the \
         number of replacements"
    );
}

#[test]
fn committed_footprint_is_the_quota_projection_baseline() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["PlanTask"],
            json!({ "status": "pending" }),
        ))
        .unwrap();

    assert_eq!(
        store.stats().memory_bytes,
        store.estimated_memory_bytes(),
        "stats must report the same committed footprint the quota path sees"
    );

    let empty = GraphMutationBatch::new(Vec::new());
    assert_eq!(
        store.projected_memory_bytes_upper_bound(&empty),
        store.estimated_memory_bytes().saturating_add(20),
        "the projection baseline must be the maintained committed footprint"
    );

    // Deletes still contribute nothing, and repeated replacements of the same
    // record do not accumulate.
    let deletes = GraphMutationBatch::new([
        GraphMutation::NodeDelete("node:a".to_string()),
        GraphMutation::EdgeDelete("edge:missing".to_string()),
    ]);
    assert_eq!(
        store.projected_memory_bytes_upper_bound(&deletes),
        store.estimated_memory_bytes().saturating_add(20)
    );

    let replacement = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
        "node:a",
        ["PlanTask"],
        json!({ "status": "complete" }),
    ))]);
    let projected = store.projected_memory_bytes_upper_bound(&replacement);
    for _ in 0..64 {
        assert_eq!(
            store.projected_memory_bytes_upper_bound(&replacement),
            projected,
            "a repeated replacement must not accumulate in the projection"
        );
    }
    assert!(projected > store.estimated_memory_bytes());
}

#[test]
fn reopened_and_rebuilt_store_accounts_the_same_as_the_source() {
    let mut store = InMemoryGraphStore::new();
    for (id, label, name) in [
        ("node:a", "Person", "Ada"),
        ("node:b", "Person", "Grace"),
        ("node:c", "Paper", "Notes"),
    ] {
        store
            .upsert_node(NodeRecord::new(
                id,
                [label],
                json!({ "name": name, "rank": 7 }),
            ))
            .unwrap();
    }
    store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "KNOWS",
            "node:b",
            json!({ "since": 1843 }),
        ))
        .unwrap();
    store.designate_ordered_property("Person", "rank").unwrap();
    store
        .upsert_node(NodeRecord::new(
            "node:d",
            ["Person"],
            json!({ "name": "Alan", "rank": 3 }),
        ))
        .unwrap();
    let source_bytes = store.estimated_memory_bytes();
    assert_eq!(source_bytes, store.recompute_memory_bytes());

    let reopened = InMemoryGraphStore::from_snapshot(store.snapshot()).unwrap();
    assert_footprint_matches_oracle(&reopened, "snapshot reopen");
    // A snapshot carries records and the historical registry, not ordered
    // designations, so the reopened store is smaller by exactly the ordered
    // index bytes the source still holds. Everything else must match byte for
    // byte, which is what makes the maintained counters comparable across a
    // restart.
    let source_ordered_bytes = crate::graph::in_memory::ordered_index_bytes(&store.ordered_indexes);
    assert!(source_ordered_bytes > 0, "the fixture must hold ordered entries");
    assert_eq!(
        reopened.estimated_memory_bytes().saturating_add(source_ordered_bytes),
        source_bytes,
        "a reopened store must account identically to its source apart from ordered designations"
    );

    // A store that re-designates and re-indexes the same way lands back on the
    // source total.
    let mut redesignated = InMemoryGraphStore::from_snapshot(store.snapshot()).unwrap();
    redesignated
        .designate_ordered_property("Person", "rank")
        .unwrap();
    assert_footprint_matches_oracle(&redesignated, "ordered re-designation after reopen");
    assert_eq!(redesignated.estimated_memory_bytes(), source_bytes);

    // The reopen path reinstalls the persisted historical registry from the
    // snapshot; a bounded limit must keep the counters and the completeness flag
    // in step.
    let bounded =
        InMemoryGraphStore::from_snapshot_with_historical_limit(store.snapshot(), 2).unwrap();
    assert_eq!(bounded.historical_node_ids.len(), 2);
    assert!(!bounded.historical_node_history_complete);
    assert_footprint_matches_oracle(&bounded, "bounded snapshot reopen");

    // A rebuild re-derives the counters from the records, so even a store whose
    // containers were tampered with ends up consistent again.
    let mut tampered = InMemoryGraphStore::from_snapshot(store.snapshot()).unwrap();
    tampered.label_index.clear();
    assert_ne!(
        tampered.estimated_memory_bytes(),
        tampered.recompute_memory_bytes(),
        "the fixture must actually be inconsistent before the rebuild"
    );
    tampered.rebuild_indexes().unwrap();
    assert_footprint_matches_oracle(&tampered, "index rebuild");
    assert_eq!(
        tampered.estimated_memory_bytes(),
        reopened.estimated_memory_bytes(),
        "repairing a tampered store must land on the same maintained total as a clean reopen"
    );
}

#[test]
fn bounded_write_performs_no_whole_graph_recompute() {
    let mut store = InMemoryGraphStore::new();
    for index in 0..1_500 {
        store
            .upsert_node(NodeRecord::new(
                format!("node:{index:04}"),
                ["Person"],
                json!({ "name": format!("name-{index:04}"), "rank": index }),
            ))
            .unwrap();
    }
    for index in 0..1_500 {
        store
            .upsert_edge(EdgeRecord::new(
                format!("edge:{index:04}"),
                format!("node:{index:04}"),
                "KNOWS",
                format!("node:{:04}", (index + 1) % 1_500),
                json!({ "since": 1843 }),
            ))
            .unwrap();
    }

    // The counter is process-wide, so a window is the claim: a write that
    // recomputed would move it in every window, while a concurrent recompute in
    // a threaded `cargo test` run can only spoil an individual window.
    let mut observed_clean_window = false;
    for index in 0..4 {
        let before = graph_memory_recompute_invocations();
        store
            .upsert_node(NodeRecord::new(
                format!("node:write{index}"),
                ["Person"],
                json!({ "name": "bounded write", "rank": 1 }),
            ))
            .unwrap();
        let _ = store.projected_memory_bytes_upper_bound(&GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                format!("node:write{index}"),
                ["Person"],
                json!({ "name": "bounded write", "rank": 1 }),
            )),
        ]));
        if graph_memory_recompute_invocations() == before {
            observed_clean_window = true;
            break;
        }
    }
    assert!(
        observed_clean_window,
        "a bounded write over {} nodes must not recompute the whole graph",
        store.nodes.len()
    );
    assert_footprint_matches_oracle(&store, "bounded write");
}

/// A durable reopen hydrates records one at a time and then installs the
/// persisted historical registry wholesale, which no per-record delta can
/// express. The recovered store carries the tenant's live quota value for the
/// rest of the process, so it must land exactly on the oracle instead of
/// keeping whatever the incremental deltas happened to accumulate.
#[test]
fn durable_reopen_accounts_the_historical_registry_and_matches_the_oracle() {
    let data_dir = unique_test_dir("committed-footprint-durable-reopen");
    let options = redcore_journal_test_options();

    let source_bytes = {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for id in ["node:kept", "node:other", "node:dropped"] {
            store
                .upsert_node(NodeRecord::new(
                    id,
                    ["Claim"],
                    json!({ "name": id, "rank": 1 }),
                ))
                .unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new(
                "edge:kept",
                "node:kept",
                "SUPPORTS",
                "node:other",
                json!({ "weight": 1.0 }),
            ))
            .unwrap();
        assert!(store.delete_node("node:dropped").unwrap());
        assert_footprint_matches_oracle(&store.store, "durable source");
        store.store.estimated_memory_bytes()
    };

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_footprint_matches_oracle(&reopened.store, "durable reopen");
    assert!(
        !reopened.store.historical_node_ids.is_empty(),
        "the fixture must retain a historical id for the reopen to reinstall"
    );
    let expected_history_bytes = reopened
        .store
        .historical_node_ids
        .iter()
        .map(|id| serialized_json_bytes(id))
        .fold(0usize, usize::saturating_add);
    assert_eq!(
        reopened.store.committed_footprint.historical_ids_json_bytes, expected_history_bytes,
        "a recovered store must account the historical registry it installed"
    );
    assert_eq!(
        reopened.store.estimated_memory_bytes(),
        source_bytes,
        "a durable reopen must account identically to the store it recovered"
    );
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}
