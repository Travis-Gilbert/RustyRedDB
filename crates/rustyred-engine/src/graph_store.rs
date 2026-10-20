use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
use web_time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(test)]
use crate::chunk_store::ChunkStoreCompaction;
use crate::chunk_store::{
    validate_catalog_root, ChunkPublicationWindow, ChunkStore, ChunkStoreCatalog, CommitMeta,
    MemChunkStore, ROOT_NAME_GRAPH, ROOT_NAME_REDCORE,
};
use crate::events::{control_events_from_projections, NodeMutationProjection, CONTROL_EVENT_LABEL};
use crate::file_lock::FileExt;
use crate::graph_history::{
    append_history, load_history_root, GraphCommitMetadata, GraphCommitPublication,
    GraphCommitReceipt, GraphHistoryEntry, GraphRevisionRoots, PinnedGraphHistory,
};
use crate::hash::Hash;
use crate::hooks::{
    changed_property_keys, GraphCommitObserver, HookEmitter, MutationCommitContext,
    MutationEdgeContext, MutationEdgeRecordContext, MutationEdgeTransition, MutationEvent,
    MutationKind, MutationNodeContext, MutationNodeTransition, MutationScopeContext,
};
use crate::journal::{JournalChunkStore, JournalCommitPolicy};
use crate::mutation_log::{self, genesis_sync_root, MutationLog};
use crate::object_store::{safe_filename, DiskObjectStore, StagedRawPayload, UnifiedChunkStore};
use crate::ordered::{OrderedDesignation, OrderedIndex};
use crate::recovery_profile::RecoveryProfile;
use crate::state::stable_hash;
use crate::tensor_derivation::TensorBlockPayloadStore;
use crate::vector_eval::VectorSearchReceipt;
use crate::vector_payload::BUILTIN_VECTOR_PAYLOAD_PROPERTIES;
#[cfg(test)]
use crate::versioned_graph::hydrate_graph_snapshot_from_payload_index;
use crate::versioned_graph::{
    audit_published_prolly_parity, build_commit, build_graph_payload_index,
    build_prolly_tree_from_entries, build_prolly_tree_incremental,
    canonical_content_object_preimage, canonical_edge_preimage, canonical_graph_payload_preimage,
    canonical_leaf_preimage, canonical_node_preimage, collect_tree_entries, compile_graph_pack,
    compile_graph_pack_incremental_with_store, estimated_resident_edge_bytes,
    hydrate_in_memory_graph_from_payload_index_profiled, prolly_validation_selected,
    recovered_graph_pack_descriptor, retarget_graph_pack_version, update_graph_payload_index,
    CompiledGraphPack, GraphCompileOptions, GraphObjectKind, GraphPackManifest, GraphPayloadRecord,
    GraphProllyTree, GraphTreeEntry, GraphTreeParityAudit,
};
use serde::{Deserialize, Serialize};
use serde_json::{value::RawValue, Value};
use sha2::{Digest, Sha256};
use theorem_surface_contracts::sync::{ChangesSince, MutationLogEntry, SyncCursor};

pub use crate::graph::error::{GraphStoreError, GraphStoreResult};
pub use crate::graph::in_memory::InMemoryGraphStore;
pub use crate::graph::mutation::{
    ConditionalNodePropertyUpdate, GraphMutation, GraphMutationBatch, GraphTransaction,
    GraphWriteResult, RecordedRedCoreMutation, RecordedRedCoreMutationBatch,
};
pub use crate::graph::query::{NeighborHit, NeighborQuery, NodeQuery};
pub use crate::graph::record::{Direction, EdgeRecord, EpistemicType, NodeRecord, Provenance};
pub use crate::graph::temporal::{edge_time_interval, node_time_interval, TimeInterval};
pub use crate::memory_document::MemoryDocumentQuery;

use crate::graph::error::{is_recoverable_orphan_edge, vector_pin_cleanup_error};
pub(crate) use crate::graph::in_memory::MAX_RESIDENT_HISTORICAL_NODE_IDS;
use crate::graph::index::validate_edge_shape;
#[cfg(feature = "redis-store")]
use crate::graph::index::{indexed_properties, merge_candidates, property_index_token};
use crate::graph::mutation::prepare_conditional_batch;
use crate::graph::record::normalize_labels;
use crate::graph::residency::COLD_ROLLBACK_RECORD_PROPERTY;

#[cfg(test)]
use crate::graph::vector_index::serialize_vector_fallback_reasons;
pub use crate::graph::vector_index::{
    default_hybrid_edge_type_weights, HybridScoringConfig, VectorDesignation, VectorIndex,
    VectorIndexShape, VectorPoint,
};
use crate::graph::vector_pin::{
    PreparedVectorProperties, VectorPayloadPinGuard, VectorPayloadPinReceipt,
};
#[cfg(feature = "redis-store")]
use crate::graph::verify::ExpectedIndexes;
pub use crate::graph::verify::{GraphRebuildReport, VerifyProblem, VerifyReport};
#[cfg(test)]
use crate::vector_eval::VectorSearchBackend;

pub use crate::graph::capability::{
    GraphAtomic, GraphRead, GraphResidency, GraphSnapshotSource, GraphStore, GraphTtl,
    GraphVectorPayloadAccess, GraphVerify, GraphWrite,
};

use crate::graph::ttl::node_with_ttl_property;
pub use crate::graph::ttl::{node_is_expired, node_ttl_expires_at_ms, now_ms, TTL_PROPERTY};

pub use crate::graph::attribution::{
    GraphMemoryAttribution, GraphStats, JsonAllocationStats, StringAllocationStats,
    StringIndexAllocationStats, TupleIndexAllocationStats, VectorIndexAllocationStats,
};

pub use crate::graph::snapshot::GraphSnapshot;
#[cfg(test)]
use crate::graph::snapshot::{
    serialized_json_bytes, BorrowedGraphSnapshot, BorrowedRedCoreSnapshotEnvelope, EdgeValues,
    MapValues,
};

#[derive(Clone)]
struct UnifiedChunkStoreHandle {
    inner: Arc<dyn UnifiedChunkStore>,
}

impl std::fmt::Debug for UnifiedChunkStoreHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnifiedChunkStoreHandle")
            .finish_non_exhaustive()
    }
}

impl UnifiedChunkStoreHandle {
    fn new(inner: Arc<dyn UnifiedChunkStore>) -> Self {
        Self { inner }
    }

    fn get(&self) -> Arc<dyn UnifiedChunkStore> {
        Arc::clone(&self.inner)
    }
}

const REDCORE_AOF_MAGIC: &str = "RRGDB_AOF";
const REDCORE_MANIFEST_VERSION: u32 = 1;
pub const REDCORE_JOURNAL_FILE: &str = "redcore.journal";
// The legacy every-second mode now selects synchronous group commit. A short
// window preserves batching for concurrent journal users without imposing a
// one-second stall on RedCore's synchronous single-writer API.
const REDCORE_GROUP_COMMIT_MAX_WAIT: Duration = Duration::from_millis(1);
const REDCORE_LEGACY_BACKUP_DIR: &str = "legacy-redcore-v1.backup";
const REDCORE_SNAPSHOT_FILE: &str = "graph.snapshot.current";
const REDCORE_PREVIOUS_SNAPSHOT_FILE: &str = "graph.snapshot.previous";
const REDCORE_AOF_FILE: &str = "graph.aof.current";
const REDCORE_MANIFEST_FILE: &str = "manifest.json";
const REDCORE_LOCK_FILE: &str = ".redcore.lock";
const REDCORE_CURRENT_SNAPSHOT_TMP_FILE: &str = "graph.snapshot.current.tmp";
const REDCORE_PREVIOUS_SNAPSHOT_TMP_FILE: &str = "graph.snapshot.previous.tmp";
const REDCORE_AOF_TMP_FILE: &str = "graph.aof.current.tmp";
const REDCORE_MANIFEST_TMP_FILE: &str = "manifest.json.tmp";
const REDCORE_COLD_ROLLBACK_DIR: &str = "cold_rollback";
const REDCORE_BOOTSTRAP_CHUNK_BATCH_BYTES: usize = 8 * 1024 * 1024;
const REDCORE_BOOTSTRAP_CHUNK_BATCH_COUNT: usize = 1_024;
#[cfg(test)]
const REDCORE_TEST_EXIT_AFTER_CHUNKS_TXN_ENV: &str =
    "RUSTYRED_TEST_EXIT_AFTER_UNCOMMITTED_CHUNKS_TXN";
#[cfg(test)]
const REDCORE_TEST_EXIT_AFTER_CHUNKS_JOURNAL_ENV: &str =
    "RUSTYRED_TEST_EXIT_AFTER_UNCOMMITTED_CHUNKS_JOURNAL";
#[cfg(test)]
const REDCORE_TEST_ABRUPT_EXIT_CODE: i32 = 86;

fn legacy_redcore_artifact_names() -> &'static [&'static str] {
    &[
        REDCORE_SNAPSHOT_FILE,
        REDCORE_PREVIOUS_SNAPSHOT_FILE,
        REDCORE_AOF_FILE,
        REDCORE_MANIFEST_FILE,
        REDCORE_CURRENT_SNAPSHOT_TMP_FILE,
        REDCORE_PREVIOUS_SNAPSHOT_TMP_FILE,
        REDCORE_AOF_TMP_FILE,
        REDCORE_MANIFEST_TMP_FILE,
        REDCORE_COLD_ROLLBACK_DIR,
    ]
}

static REDCORE_PROCESS_LOCKS: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();

/// Process-wide count of publish-path branch selections, split by branch.
///
/// `commit_prepared_batch_internal` has two publish branches: the incremental
/// delta pack (`can_commit_incrementally`) and the staged branch, which clones
/// the whole in-memory graph (`let mut staged = self.store.clone()`) before
/// applying the batch. The staged branch is required for mutation sets the
/// delta compiler cannot express -- notably any batch containing a delete -- and
/// it is O(stored records) by construction, so which branch a bounded write
/// takes is the difference between a delta publish and a whole-store copy.
///
/// These counters exist so tests can assert that branch selection from the
/// outside, the same way `graph_memory_recompute_invocations` lets tests assert
/// the accounting path never traverses the graph.
static GRAPH_DELTA_COMMIT_PUBLICATIONS: AtomicU64 = AtomicU64::new(0);
static GRAPH_STAGED_COMMIT_PUBLICATIONS: AtomicU64 = AtomicU64::new(0);

/// How many commits in this process published through the incremental delta pack
/// branch. A bounded upsert-only write against a store with a published head
/// must move this by exactly one.
pub fn graph_delta_commit_publications() -> u64 {
    GRAPH_DELTA_COMMIT_PUBLICATIONS.load(Ordering::Relaxed)
}

/// How many commits in this process took the staged whole-store branch. That
/// branch is legitimate for deletes and for the first publish into an empty
/// store (the pack bootstrap), and is the cost a bounded write must not pay.
pub fn graph_staged_commit_publications() -> u64 {
    GRAPH_STAGED_COMMIT_PUBLICATIONS.load(Ordering::Relaxed)
}

/// Reset both publish-branch counters to zero.
pub fn reset_graph_commit_publication_counts() {
    GRAPH_DELTA_COMMIT_PUBLICATIONS.store(0, Ordering::Relaxed);
    GRAPH_STAGED_COMMIT_PUBLICATIONS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
fn exit_after_uncommitted_chunks_if_requested(journal: &JournalChunkStore, txn_id: u64) {
    let txn_matches = std::env::var(REDCORE_TEST_EXIT_AFTER_CHUNKS_TXN_ENV)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        == Some(txn_id);
    let journal_matches = std::env::var_os(REDCORE_TEST_EXIT_AFTER_CHUNKS_JOURNAL_ENV)
        .is_some_and(|raw| Path::new(&raw) == journal.path());
    if txn_matches && journal_matches {
        // Deliberately skip destructors and the catalog/root durability
        // barrier. The parent-process oracle reopens this journal and proves
        // that these chunk frames are present but were never published.
        std::process::exit(REDCORE_TEST_ABRUPT_EXIT_CODE);
    }
}

/// Legacy RedCore durability names retained for configuration compatibility.
///
/// Every durable variant writes the same journal format. The effective root
/// barrier policy is reported separately through [`RedCoreStatus`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RedCoreDurability {
    None,
    AofEverysec,
    AofAlways,
    SnapshotOnly,
}

impl RedCoreDurability {
    pub fn parse(value: &str) -> GraphStoreResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "memory" => Ok(Self::None),
            "aof_everysec" | "everysec" => Ok(Self::AofEverysec),
            "aof_always" | "always" => Ok(Self::AofAlways),
            "snapshot_only" | "snapshot" => Ok(Self::SnapshotOnly),
            unknown => Err(GraphStoreError::new(
                "redcore_invalid_durability",
                format!("unknown RedCore durability policy: {unknown}"),
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::AofEverysec => "aof_everysec",
            Self::AofAlways => "aof_always",
            Self::SnapshotOnly => "snapshot_only",
        }
    }

    fn journal_commit_policy(self) -> Option<JournalCommitPolicy> {
        match self {
            Self::None => None,
            Self::AofEverysec => Some(JournalCommitPolicy::Grouped {
                max_wait: REDCORE_GROUP_COMMIT_MAX_WAIT,
            }),
            Self::AofAlways | Self::SnapshotOnly => Some(JournalCommitPolicy::Immediate),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RedCoreOptions {
    pub durability: RedCoreDurability,
    /// Deprecated compatibility knob from the retired snapshot writer.
    /// Journal durability is governed by the effective commit policy instead.
    pub snapshot_interval_writes: u64,
    pub strict_acid: bool,
}

impl Default for RedCoreOptions {
    fn default() -> Self {
        Self {
            durability: RedCoreDurability::AofEverysec,
            snapshot_interval_writes: 1_000,
            strict_acid: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RedCoreStatus {
    pub mode: String,
    /// Requested legacy configuration name.
    pub durability: String,
    /// Effective journal behavior: `grouped_fsync`, `immediate_fsync`, or
    /// `memory_only`.
    pub commit_policy: String,
    pub commit_batch_window_ms: u64,
    pub data_dir: Option<String>,
    pub graph_version: u64,
    pub last_txn_id: u64,
    pub snapshot_txn_id: u64,
    pub txn_lag_since_snapshot: u64,
    pub recovered_frames: u64,
    pub skipped_aof_frames: u64,
    pub aof_size_bytes: u64,
    pub aof_replay_bytes: u64,
    pub recovery_duration_ms: u64,
    pub truncated_aof_tail: bool,
    pub last_fsync_unix_ms: Option<u128>,
    pub last_recovery_ok: bool,
    pub strict_acid: bool,
    pub durable_nodes_total: usize,
    pub durable_edges_total: usize,
    pub resident_edges_total: usize,
    pub resident_edge_bytes_estimate: usize,
    pub resident_edge_budget_bytes: Option<usize>,
    pub edges_fully_resident: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RedCoreManifest {
    pub version: u32,
    pub graph_version: u64,
    pub last_txn_id: u64,
    pub snapshot_txn_id: u64,
    pub durability: RedCoreDurability,
    pub snapshot_file: String,
    pub aof_file: String,
    pub updated_at_unix_ms: u128,
    #[serde(default = "default_format_kind")]
    pub format_kind: String,
    #[serde(default)]
    pub crate_version: String,
    #[serde(default)]
    pub created_at_unix_ms: u128,
}

fn default_format_kind() -> String {
    "redcore".to_string()
}

/// Maximum manifest format version this build is willing to load.
/// Bump when introducing a breaking on-disk format change.
pub const CURRENT_FORMAT_VERSION: u32 = REDCORE_MANIFEST_VERSION;

/// Returns true if a snapshot at `version` can be loaded directly without migration.
pub fn manifest_version_compatible(version: u32) -> bool {
    version <= CURRENT_FORMAT_VERSION
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RedCoreSnapshotEnvelope {
    version: u32,
    txn_id: u64,
    graph: GraphSnapshot,
    #[serde(default)]
    historical_node_ids: BTreeSet<String>,
    #[serde(default)]
    historical_node_history_complete: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RedCoreAofFrame {
    magic: String,
    version: u32,
    txn_id: u64,
    graph_version: u64,
    timestamp_unix_ms: u128,
    payload_checksum: String,
    mutation: RedCoreMutation,
}

#[derive(Deserialize)]
struct RedCoreAofWireFrame<'a> {
    magic: String,
    version: u32,
    txn_id: u64,
    graph_version: u64,
    timestamp_unix_ms: u128,
    payload_checksum: String,
    #[serde(borrow)]
    mutation: &'a RawValue,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", content = "record", rename_all = "snake_case")]
enum RedCoreMutation {
    NodeUpsert(NodeRecord),
    EdgeUpsert(EdgeRecord),
    Batch(Vec<RedCoreMutation>),
    VectorDesignation(VectorDesignation),
    OrderedDesignation(OrderedDesignation),
    VectorDesignationClear(OrderedDesignation),
    OrderedDesignationClear(OrderedDesignation),
    /// Durable node deletion (TTL-04). Recovery handler removes the node
    /// from storage and from every index. Idempotent on recovery: an
    /// AOF replay that deletes a node already absent in the snapshot is
    /// a no-op rather than an error. AOF files written by pre-TTL-04
    /// builds never contain this variant, so older snapshots replay
    /// fine; AOF files containing NodeDelete fail to load on pre-TTL-04
    /// builds with an `unknown variant` serde error, which is the
    /// loudest forward-compat signal available.
    NodeDelete(String),
    EdgeDelete(String),
}

const REDCORE_JOURNAL_STATE_DOMAIN: u8 = 7;
const LEGACY_REDCORE_JOURNAL_STATE_VERSION: u8 = 1;
const INLINE_HISTORY_REDCORE_JOURNAL_STATE_VERSION: u8 = 2;
const HISTORICAL_REGISTRY_REDCORE_JOURNAL_STATE_VERSION: u8 = 3;
const MUTATION_LOG_REDCORE_JOURNAL_STATE_VERSION: u8 = 4;
const REDCORE_JOURNAL_STATE_VERSION: u8 = 5;
const REDCORE_HISTORICAL_NODE_MARKER_DOMAIN: u8 = 8;
const REDCORE_HISTORICAL_NODE_MARKER_VERSION: u8 = 1;
const MAX_REDCORE_STATE_DESIGNATIONS: usize = 1_000_000;

/// Canonical durable state which is not represented by the frozen logical
/// graph tree itself. The logical and payload trees are separate by design:
/// Phase-C addresses remain metadata-independent while recovery retains every
/// field of each `NodeRecord` and `EdgeRecord`.
#[derive(Clone, Debug, PartialEq)]
struct RedCoreJournalState {
    txn_id: u64,
    graph_version: u64,
    pack_graph_version: u64,
    logical_tree: GraphProllyTree,
    payload_tree: GraphProllyTree,
    logical_commit: Hash,
    nodes_total: usize,
    edges_total: usize,
    historical_node_tree: Option<GraphProllyTree>,
    legacy_historical_node_ids: BTreeSet<String>,
    historical_node_history_complete: bool,
    vector_designations: Vec<VectorDesignation>,
    ordered_designations: Vec<OrderedDesignation>,
    mutation_log: MutationLog,
    history_root: Option<Hash>,
}

impl RedCoreJournalState {
    fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        if !self.legacy_historical_node_ids.is_empty() {
            return Err(GraphStoreError::new(
                "redcore_journal_state_inline_history_forbidden",
                "v3 RedCore journal state cannot serialize inline historical node ids",
            ));
        }
        let mut vectors = self.vector_designations.clone();
        vectors.sort_unstable_by(|left, right| {
            (&left.label, &left.property, left.dimension).cmp(&(
                &right.label,
                &right.property,
                right.dimension,
            ))
        });
        let mut ordered = self.ordered_designations.clone();
        ordered.sort_unstable_by(|left, right| {
            (&left.label, &left.property).cmp(&(&right.label, &right.property))
        });

        let mut output = Vec::new();
        output.push(REDCORE_JOURNAL_STATE_DOMAIN);
        output.push(REDCORE_JOURNAL_STATE_VERSION);
        output.extend_from_slice(&self.txn_id.to_le_bytes());
        output.extend_from_slice(&self.graph_version.to_le_bytes());
        output.extend_from_slice(&self.pack_graph_version.to_le_bytes());
        output.extend_from_slice(self.logical_tree.root.as_bytes());
        output.extend_from_slice(
            &redcore_state_u64(self.logical_tree.entries_total, "logical tree entry count")?
                .to_le_bytes(),
        );
        output.extend_from_slice(self.payload_tree.root.as_bytes());
        output.extend_from_slice(
            &redcore_state_u64(self.payload_tree.entries_total, "payload tree entry count")?
                .to_le_bytes(),
        );
        output.extend_from_slice(self.logical_commit.as_bytes());
        output.extend_from_slice(
            &redcore_state_u64(self.nodes_total, "live node count")?.to_le_bytes(),
        );
        output.extend_from_slice(
            &redcore_state_u64(self.edges_total, "live edge count")?.to_le_bytes(),
        );
        output.push(u8::from(self.historical_node_history_complete));
        let historical_node_tree = self.historical_node_tree.as_ref().ok_or_else(|| {
            GraphStoreError::new(
                "redcore_journal_state_history_root_missing",
                "v3 RedCore journal state requires a historical-node registry root",
            )
        })?;
        output.extend_from_slice(historical_node_tree.root.as_bytes());
        output.extend_from_slice(
            &redcore_state_u64(
                historical_node_tree.entries_total,
                "historical node tree entry count",
            )?
            .to_le_bytes(),
        );
        output.extend_from_slice(
            &redcore_state_u32(vectors.len(), "vector designation count")?.to_le_bytes(),
        );
        for designation in vectors {
            push_redcore_state_string(&mut output, &designation.label, "vector label")?;
            push_redcore_state_string(&mut output, &designation.property, "vector property")?;
            output.extend_from_slice(
                &redcore_state_u64(designation.dimension, "vector dimension")?.to_le_bytes(),
            );
        }
        output.extend_from_slice(
            &redcore_state_u32(ordered.len(), "ordered designation count")?.to_le_bytes(),
        );
        for designation in ordered {
            push_redcore_state_string(&mut output, &designation.label, "ordered label")?;
            push_redcore_state_string(&mut output, &designation.property, "ordered property")?;
        }
        output.push(u8::from(self.history_root.is_some()));
        if let Some(history_root) = self.history_root {
            output.extend_from_slice(history_root.as_bytes());
        }
        output.extend_from_slice(&self.mutation_log.encode_tail()?);
        Ok(output)
    }

    fn from_bytes(bytes: &[u8]) -> GraphStoreResult<Self> {
        let mut decoder = RedCoreStateDecoder::new(bytes);
        decoder.expect_byte(REDCORE_JOURNAL_STATE_DOMAIN, "state domain")?;
        let state_version = decoder.byte("state version")?;
        if !matches!(
            state_version,
            LEGACY_REDCORE_JOURNAL_STATE_VERSION
                | INLINE_HISTORY_REDCORE_JOURNAL_STATE_VERSION
                | HISTORICAL_REGISTRY_REDCORE_JOURNAL_STATE_VERSION
                | MUTATION_LOG_REDCORE_JOURNAL_STATE_VERSION
                | REDCORE_JOURNAL_STATE_VERSION
        ) {
            return Err(GraphStoreError::new(
                "redcore_journal_state_format_mismatch",
                format!(
                    "state version is {state_version}, expected {LEGACY_REDCORE_JOURNAL_STATE_VERSION}, {INLINE_HISTORY_REDCORE_JOURNAL_STATE_VERSION}, {HISTORICAL_REGISTRY_REDCORE_JOURNAL_STATE_VERSION}, or {REDCORE_JOURNAL_STATE_VERSION}"
                ),
            ));
        }
        let txn_id = decoder.u64("transaction id")?;
        let graph_version = decoder.u64("graph version")?;
        let pack_graph_version = decoder.u64("logical pack graph version")?;
        let logical_tree = GraphProllyTree {
            root: decoder.hash("logical tree root")?,
            entries_total: decoder.usize("logical tree entry count")?,
        };
        let payload_tree = GraphProllyTree {
            root: decoder.hash("payload tree root")?,
            entries_total: decoder.usize("payload tree entry count")?,
        };
        let logical_commit = decoder.hash("logical commit")?;
        let nodes_total = decoder.usize("live node count")?;
        let edges_total = decoder.usize("live edge count")?;
        let (historical_node_tree, legacy_historical_node_ids, historical_node_history_complete) =
            if state_version >= HISTORICAL_REGISTRY_REDCORE_JOURNAL_STATE_VERSION {
                let complete = match decoder.byte("historical node history completeness")? {
                    0 => false,
                    1 => true,
                    value => {
                        return Err(GraphStoreError::new(
                            "redcore_journal_state_history_invalid",
                            format!(
                            "historical node history completeness byte must be 0 or 1, got {value}"
                        ),
                        ))
                    }
                };
                let tree = GraphProllyTree {
                    root: decoder.hash("historical node tree root")?,
                    entries_total: decoder.usize("historical node tree entry count")?,
                };
                if tree.entries_total > MAX_RESIDENT_HISTORICAL_NODE_IDS {
                    return Err(GraphStoreError::new(
                        "redcore_journal_state_history_limit",
                        format!(
                            "historical node tree entry count {} exceeds {MAX_RESIDENT_HISTORICAL_NODE_IDS}",
                            tree.entries_total
                        ),
                    ));
                }
                (Some(tree), BTreeSet::new(), complete)
            } else if state_version == INLINE_HISTORY_REDCORE_JOURNAL_STATE_VERSION {
                let complete = match decoder.byte("historical node history completeness")? {
                    0 => false,
                    1 => true,
                    value => {
                        return Err(GraphStoreError::new(
                            "redcore_journal_state_history_invalid",
                            format!(
                            "historical node history completeness byte must be 0 or 1, got {value}"
                        ),
                        ))
                    }
                };
                let count = usize::try_from(decoder.u32("historical node id count")?)
                    .expect("u32 fits usize");
                if count > MAX_RESIDENT_HISTORICAL_NODE_IDS {
                    return Err(GraphStoreError::new(
                    "redcore_journal_state_history_limit",
                    format!(
                        "historical node id count {count} exceeds {MAX_RESIDENT_HISTORICAL_NODE_IDS}"
                    ),
                ));
                }
                let mut ids = BTreeSet::new();
                let mut previous: Option<String> = None;
                for _ in 0..count {
                    let id = decoder.string("historical node id")?;
                    if id.is_empty()
                        || previous
                            .as_deref()
                            .is_some_and(|previous| previous >= id.as_str())
                    {
                        return Err(GraphStoreError::new(
                            "redcore_journal_state_history_invalid",
                            "historical node ids must be non-empty, strictly sorted, and unique",
                        ));
                    }
                    previous = Some(id.clone());
                    ids.insert(id);
                }
                (None, ids, complete)
            } else {
                // V1 did not persist the registry, so never-before-used claims
                // must fail closed after recovery.
                (None, BTreeSet::new(), false)
            };
        let vector_count = decoder.count("vector designation count")?;
        let mut vector_designations = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            vector_designations.push(VectorDesignation {
                label: decoder.string("vector label")?,
                property: decoder.string("vector property")?,
                dimension: decoder.usize("vector dimension")?,
            });
        }
        let ordered_count = decoder.count("ordered designation count")?;
        let mut ordered_designations = Vec::with_capacity(ordered_count);
        for _ in 0..ordered_count {
            ordered_designations.push(OrderedDesignation {
                label: decoder.string("ordered label")?,
                property: decoder.string("ordered property")?,
            });
        }
        let history_root = if state_version >= REDCORE_JOURNAL_STATE_VERSION {
            match decoder.byte("logical history root presence")? {
                0 => None,
                1 => Some(decoder.hash("logical history root")?),
                _ => {
                    return Err(GraphStoreError::new(
                        "redcore_journal_state_history_invalid",
                        "logical history presence byte must be zero or one",
                    ))
                }
            }
        } else {
            None
        };
        let mutation_log = if state_version >= MUTATION_LOG_REDCORE_JOURNAL_STATE_VERSION {
            let log = MutationLog::decode_tail(decoder.rest())?;
            decoder.skip_remaining();
            log
        } else {
            MutationLog::empty_after(logical_tree.root, txn_id)
        };
        decoder.finish()?;
        ensure_redcore_state_designations_canonical(&vector_designations, &ordered_designations)?;
        if nodes_total
            .checked_add(edges_total)
            .is_none_or(|live| live > logical_tree.entries_total)
        {
            return Err(GraphStoreError::new(
                "redcore_journal_state_count_mismatch",
                "live graph counts exceed the logical tree entry count",
            ));
        }
        if payload_tree.entries_total != logical_tree.entries_total {
            return Err(GraphStoreError::new(
                "redcore_journal_state_tree_mismatch",
                format!(
                    "logical tree declares {} entries but payload tree declares {}",
                    logical_tree.entries_total, payload_tree.entries_total
                ),
            ));
        }
        Ok(Self {
            txn_id,
            graph_version,
            pack_graph_version,
            logical_tree,
            payload_tree,
            logical_commit,
            nodes_total,
            edges_total,
            historical_node_tree,
            legacy_historical_node_ids,
            historical_node_history_complete,
            vector_designations,
            ordered_designations,
            mutation_log,
            history_root,
        })
    }
}

fn canonical_historical_node_marker(id: &str) -> GraphStoreResult<Vec<u8>> {
    if id.is_empty() {
        return Err(GraphStoreError::new(
            "redcore_historical_node_id_empty",
            "historical node id must not be empty",
        ));
    }
    let mut bytes = vec![
        REDCORE_HISTORICAL_NODE_MARKER_DOMAIN,
        REDCORE_HISTORICAL_NODE_MARKER_VERSION,
    ];
    push_redcore_state_string(&mut bytes, id, "historical node marker id")?;
    Ok(bytes)
}

fn historical_node_registry_entries<'a>(
    store: &dyn ChunkStore,
    ids: impl IntoIterator<Item = &'a str>,
) -> GraphStoreResult<BTreeMap<String, GraphTreeEntry>> {
    let ids = ids.into_iter().collect::<BTreeSet<_>>();
    if ids.len() > MAX_RESIDENT_HISTORICAL_NODE_IDS {
        return Err(GraphStoreError::new(
            "redcore_historical_node_registry_limit",
            format!(
                "historical node registry has {} ids, exceeding {MAX_RESIDENT_HISTORICAL_NODE_IDS}",
                ids.len()
            ),
        ));
    }
    let mut markers = Vec::with_capacity(ids.len());
    let mut entries = BTreeMap::new();
    for id in ids {
        let bytes = canonical_historical_node_marker(id)?;
        let object_hash = Hash::of(&bytes);
        markers.push((object_hash, bytes));
        entries.insert(
            id.to_string(),
            GraphTreeEntry {
                key: id.to_string(),
                kind: GraphObjectKind::Node,
                object_hash,
                tombstone: false,
            },
        );
    }
    let borrowed = markers
        .iter()
        .map(|(address, bytes)| (*address, bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write(&borrowed)?;
    Ok(entries)
}

fn build_historical_node_registry<'a>(
    store: &dyn ChunkStore,
    ids: impl IntoIterator<Item = &'a str>,
) -> GraphStoreResult<GraphProllyTree> {
    let entries = historical_node_registry_entries(store, ids)?
        .into_values()
        .collect();
    build_prolly_tree_from_entries(store, entries)
}

fn update_historical_node_registry<'a>(
    store: &dyn ChunkStore,
    prior: &GraphProllyTree,
    ids: impl IntoIterator<Item = &'a str>,
) -> GraphStoreResult<GraphProllyTree> {
    let upserts = historical_node_registry_entries(store, ids)?;
    build_prolly_tree_incremental(store, prior, &upserts).map(|build| build.tree)
}

fn hydrate_historical_node_registry(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
) -> GraphStoreResult<BTreeSet<String>> {
    if tree.entries_total > MAX_RESIDENT_HISTORICAL_NODE_IDS {
        return Err(GraphStoreError::new(
            "redcore_historical_node_registry_limit",
            format!(
                "historical node tree declares {} ids, exceeding {MAX_RESIDENT_HISTORICAL_NODE_IDS}",
                tree.entries_total
            ),
        ));
    }
    let entries = collect_tree_entries(store, tree)?;
    if entries.len() != tree.entries_total {
        return Err(GraphStoreError::new(
            "redcore_historical_node_registry_count_mismatch",
            format!(
                "historical node tree declares {} ids but contains {} entries",
                tree.entries_total,
                entries.len()
            ),
        ));
    }
    let mut ids = BTreeSet::new();
    for entry in entries {
        if entry.kind != GraphObjectKind::Node || entry.tombstone || entry.key.is_empty() {
            return Err(GraphStoreError::new(
                "redcore_historical_node_registry_entry_invalid",
                format!(
                    "invalid historical node registry entry for key {:?}",
                    entry.key
                ),
            ));
        }
        let expected = canonical_historical_node_marker(&entry.key)?;
        if Hash::of(&expected) != entry.object_hash {
            return Err(GraphStoreError::new(
                "redcore_historical_node_marker_address_mismatch",
                format!(
                    "historical node marker for {:?} does not match tree address {}",
                    entry.key, entry.object_hash
                ),
            ));
        }
        let actual = store.get(entry.object_hash)?.ok_or_else(|| {
            GraphStoreError::new(
                "redcore_historical_node_marker_missing",
                format!(
                    "historical node marker {} for {:?} is missing",
                    entry.object_hash, entry.key
                ),
            )
        })?;
        if actual.as_ref() != expected.as_slice() {
            return Err(GraphStoreError::new(
                "redcore_historical_node_marker_invalid",
                format!(
                    "historical node marker for {:?} is not canonical",
                    entry.key
                ),
            ));
        }
        if !ids.insert(entry.key) {
            return Err(GraphStoreError::new(
                "redcore_historical_node_registry_duplicate",
                "historical node registry contains a duplicate key",
            ));
        }
    }
    Ok(ids)
}

/// Strictly decode the stored tree roots which participate in RedCore
/// reachability. `logical_commit` is semantic commit metadata rather than a
/// physical chunk address and is deliberately not returned.
pub(crate) fn decode_redcore_reachability_roots(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<(Hash, Hash, Option<Hash>)> {
    decode_redcore_reachability_state(expected_address, bytes)
        .map(|(logical, payload, historical, _)| (logical, payload, historical))
}

pub(crate) fn decode_redcore_reachability_state(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<(Hash, Hash, Option<Hash>, Vec<VectorDesignation>)> {
    let actual_address = Hash::of(bytes);
    if actual_address != expected_address {
        return Err(GraphStoreError::new(
            "redcore_journal_state_address_mismatch",
            format!("RedCore state bytes hash to {actual_address}, expected {expected_address}"),
        ));
    }

    let state = RedCoreJournalState::from_bytes(bytes)?;
    Ok((
        state.logical_tree.root,
        state.payload_tree.root,
        state.historical_node_tree.map(|tree| tree.root),
        state.vector_designations,
    ))
}

pub(crate) fn decode_redcore_history_root(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<Option<Hash>> {
    if Hash::of(bytes) != expected_address {
        return Err(GraphStoreError::new(
            "redcore_journal_state_address_mismatch",
            "state address mismatch",
        ));
    }
    Ok(RedCoreJournalState::from_bytes(bytes)?.history_root)
}

fn redcore_state_u32(value: usize, field: &str) -> GraphStoreResult<u32> {
    u32::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "redcore_journal_state_length_overflow",
            format!("{field} exceeds the canonical u32 limit"),
        )
    })
}

fn redcore_state_u64(value: usize, field: &str) -> GraphStoreResult<u64> {
    u64::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "redcore_journal_state_length_overflow",
            format!("{field} exceeds the canonical u64 limit"),
        )
    })
}

fn push_redcore_state_string(
    output: &mut Vec<u8>,
    value: &str,
    field: &str,
) -> GraphStoreResult<()> {
    output.extend_from_slice(&redcore_state_u32(value.len(), field)?.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn ensure_redcore_state_designations_canonical(
    vectors: &[VectorDesignation],
    ordered: &[OrderedDesignation],
) -> GraphStoreResult<()> {
    let vectors_canonical = vectors.windows(2).all(|pair| {
        (&pair[0].label, &pair[0].property, pair[0].dimension)
            < (&pair[1].label, &pair[1].property, pair[1].dimension)
    });
    let ordered_canonical = ordered
        .windows(2)
        .all(|pair| (&pair[0].label, &pair[0].property) < (&pair[1].label, &pair[1].property));
    if !vectors_canonical || !ordered_canonical {
        return Err(GraphStoreError::new(
            "redcore_journal_state_not_canonical",
            "durable designation lists must be strictly sorted and duplicate-free",
        ));
    }
    Ok(())
}

struct RedCoreStateDecoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> RedCoreStateDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self.offset.checked_add(length).ok_or_else(|| {
            GraphStoreError::new(
                "redcore_journal_state_truncated",
                format!("{field} offset overflowed"),
            )
        })?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            GraphStoreError::new(
                "redcore_journal_state_truncated",
                format!("durable RedCore state ended while reading {field}"),
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn expect_byte(&mut self, expected: u8, field: &str) -> GraphStoreResult<()> {
        let actual = self.byte(field)?;
        if actual != expected {
            return Err(GraphStoreError::new(
                "redcore_journal_state_format_mismatch",
                format!("{field} is {actual}, expected {expected}"),
            ));
        }
        Ok(())
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        let raw: [u8; 4] = self.take(4, field)?.try_into().expect("fixed-width slice");
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        let raw: [u8; 8] = self.take(8, field)?.try_into().expect("fixed-width slice");
        Ok(u64::from_le_bytes(raw))
    }

    fn usize(&mut self, field: &str) -> GraphStoreResult<usize> {
        usize::try_from(self.u64(field)?).map_err(|_| {
            GraphStoreError::new(
                "redcore_journal_state_length_overflow",
                format!("{field} exceeds this platform's usize limit"),
            )
        })
    }

    fn count(&mut self, field: &str) -> GraphStoreResult<usize> {
        let count = usize::try_from(self.u32(field)?).expect("u32 fits usize");
        if count > MAX_REDCORE_STATE_DESIGNATIONS {
            return Err(GraphStoreError::new(
                "redcore_journal_state_count_limit",
                format!("{field} {count} exceeds {MAX_REDCORE_STATE_DESIGNATIONS}"),
            ));
        }
        Ok(count)
    }

    fn hash(&mut self, field: &str) -> GraphStoreResult<Hash> {
        let raw: [u8; 32] = self.take(32, field)?.try_into().expect("fixed-width slice");
        Ok(Hash::from_bytes(raw))
    }

    fn string(&mut self, field: &str) -> GraphStoreResult<String> {
        let length = usize::try_from(self.u32(field)?).expect("u32 fits usize");
        String::from_utf8(self.take(length, field)?.to_vec()).map_err(|error| {
            GraphStoreError::new(
                "redcore_journal_state_utf8",
                format!("{field} is not UTF-8: {error}"),
            )
        })
    }

    fn rest(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
    }

    fn skip_remaining(&mut self) {
        self.offset = self.bytes.len();
    }

    fn finish(&self) -> GraphStoreResult<()> {
        if self.offset != self.bytes.len() {
            return Err(GraphStoreError::new(
                "redcore_journal_state_trailing_bytes",
                format!(
                    "durable RedCore state has {} trailing bytes",
                    self.bytes.len() - self.offset
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RedCoreColdRollbackRecord {
    id: String,
    node: NodeRecord,
}

fn cold_rollback_payload_matches(
    node: &NodeRecord,
    original: &NodeRecord,
) -> GraphStoreResult<bool> {
    if node.id != original.id
        || node.labels != original.labels
        || node.version != original.version
        || node.tombstone != original.tombstone
        || node.parent_hashes != original.parent_hashes
    {
        return Ok(false);
    }
    let (Some(properties), Some(original_properties)) =
        (node.properties.as_object(), original.properties.as_object())
    else {
        return Ok(false);
    };
    if original_properties.contains_key(COLD_ROLLBACK_RECORD_PROPERTY)
        || properties.len() != original_properties.len().saturating_add(1)
    {
        return Ok(false);
    }
    let expected_marker = format!("v1:{}", original.checksum()?);
    if properties
        .get(COLD_ROLLBACK_RECORD_PROPERTY)
        .and_then(Value::as_str)
        != Some(expected_marker.as_str())
        || !original_properties
            .iter()
            .all(|(key, value)| properties.get(key) == Some(value))
    {
        return Ok(false);
    }
    let marked_checksum = node.checksum()?;
    Ok(node.content_hash.as_deref() == Some(marked_checksum.as_str()))
}

fn take_cold_rollback_marker(
    node: &mut NodeRecord,
    original: &NodeRecord,
) -> GraphStoreResult<bool> {
    let marked = cold_rollback_payload_matches(node, original)?;
    if marked {
        if let Some(properties) = node.properties.as_object_mut() {
            properties.remove(COLD_ROLLBACK_RECORD_PROPERTY);
        }
        node.content_hash = original.content_hash.clone();
    }
    Ok(marked)
}

impl From<GraphMutation> for RedCoreMutation {
    fn from(mutation: GraphMutation) -> Self {
        match mutation {
            GraphMutation::NodeUpsert(node) => Self::NodeUpsert(node),
            GraphMutation::EdgeUpsert(edge) => Self::EdgeUpsert(edge),
            GraphMutation::NodeDelete(id) => Self::NodeDelete(id),
            GraphMutation::EdgeDelete(id) => Self::EdgeDelete(id),
        }
    }
}

impl RedCoreMutation {
    fn graph_batch(&self) -> GraphMutationBatch {
        let mut mutations = Vec::new();
        self.collect_graph_mutations(&mut mutations);
        GraphMutationBatch {
            mutations,
            ..GraphMutationBatch::default()
        }
    }

    fn collect_graph_mutations(&self, output: &mut Vec<GraphMutation>) {
        match self {
            Self::NodeUpsert(node) => output.push(GraphMutation::NodeUpsert(node.clone())),
            Self::EdgeUpsert(edge) => output.push(GraphMutation::EdgeUpsert(edge.clone())),
            Self::NodeDelete(id) => output.push(GraphMutation::NodeDelete(id.clone())),
            Self::EdgeDelete(id) => output.push(GraphMutation::EdgeDelete(id.clone())),
            Self::Batch(mutations) => {
                for mutation in mutations {
                    mutation.collect_graph_mutations(output);
                }
            }
            Self::VectorDesignation(_)
            | Self::OrderedDesignation(_)
            | Self::VectorDesignationClear(_)
            | Self::OrderedDesignationClear(_) => {}
        }
    }

    fn collect_historical_node_ids<'a>(&'a self, output: &mut Vec<&'a str>) {
        match self {
            Self::NodeUpsert(node) => output.push(node.id.as_str()),
            // Deletes keep the id in the never-before-used registry.
            Self::NodeDelete(id) => output.push(id.as_str()),
            Self::Batch(mutations) => {
                for mutation in mutations {
                    mutation.collect_historical_node_ids(output);
                }
            }
            Self::EdgeUpsert(_)
            | Self::EdgeDelete(_)
            | Self::VectorDesignation(_)
            | Self::OrderedDesignation(_)
            | Self::VectorDesignationClear(_)
            | Self::OrderedDesignationClear(_) => {}
        }
    }
}

/// A trusted, read-only command preparer. It may append derived mutations to
/// one incoming atomic batch; it cannot publish or mutate the writer.
#[derive(Clone)]
pub struct GraphBatchPreparer(
    Arc<
        dyn Fn(&RedCoreGraphStore, &GraphMutationBatch) -> GraphStoreResult<GraphMutationBatch>
            + Send
            + Sync,
    >,
);
impl GraphBatchPreparer {
    pub fn new(
        prepare: impl Fn(&RedCoreGraphStore, &GraphMutationBatch) -> GraphStoreResult<GraphMutationBatch>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self(Arc::new(prepare))
    }
}
impl std::fmt::Debug for GraphBatchPreparer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GraphBatchPreparer")
    }
}

/// Once-prepared final mutations, certified against this writer and base.
/// Deliberately neither Clone nor serializable: consumers cannot mint or replay
/// a certificate, or edit the batch after final quota/guard validation.
#[derive(Debug)]
pub struct PreparedGraphBatch {
    batch: GraphMutationBatch,
    base: SyncCursor,
    writer: Arc<()>,
}
impl PreparedGraphBatch {
    pub fn batch(&self) -> &GraphMutationBatch {
        &self.batch
    }
}

#[derive(Debug)]
pub struct RedCoreGraphStore {
    store: InMemoryGraphStore,
    data_dir: Option<PathBuf>,
    _directory_lock: Option<RedCoreDirectoryLock>,
    /// One physical address space for every RedCore mode. Durable stores point
    /// this at `journal`; memory-only stores point it at one shared MemChunkStore;
    /// injected stores retain the caller's already-open backend.
    chunk_store: UnifiedChunkStoreHandle,
    /// The caller, rather than this facade, owns physical persistence policy.
    injected_chunk_store: bool,
    /// Typed durable handle retained only for journal recovery, diagnostics,
    /// and fault-injection tests. Payload lenses use `chunk_store` above.
    journal: Option<Arc<JournalChunkStore>>,
    current_pack: Option<CompiledGraphPack>,
    payload_tree: Option<GraphProllyTree>,
    historical_node_tree: Option<GraphProllyTree>,
    history_root: Option<Hash>,
    durable_nodes_total: usize,
    durable_edges_total: usize,
    resident_edge_bytes_estimate: usize,
    resident_edge_budget_bytes: Option<usize>,
    edges_fully_resident: bool,
    options: RedCoreOptions,
    last_txn_id: u64,
    snapshot_txn_id: u64,
    recovered_frames: u64,
    skipped_aof_frames: u64,
    aof_size_bytes: u64,
    aof_replay_bytes: u64,
    recovery_duration_ms: u64,
    recovery_profile: RecoveryProfile,
    truncated_aof_tail: bool,
    last_recovery_ok: bool,
    last_fsync: Option<SystemTime>,
    transient_ordered_indexes: HashMap<String, OrderedIndex>,
    // ---- Graph-level hooks (additive; see crate::hooks) ----------------
    // Optional post-commit mutation-event sink. `None` (the default for every
    // existing caller) makes every emit a no-op, so the hook subsystem is
    // strictly opt-in and changes nothing for stores without a dispatcher.
    hook_emitter: Option<HookEmitter>,
    // Transport notification is independent of queued compute-hook enablement.
    commit_observer: Option<GraphCommitObserver>,
    batch_preparer: Option<GraphBatchPreparer>,
    preparation_identity: Arc<()>,
    pending_commit_publication: Option<GraphCommitPublication>,
    // Loop-guard generation stamped onto events this store emits. The hook
    // worker sets this to `g + 1` around a handler running at generation `g`,
    // so a handler's writes are tagged one deeper and converge under max_depth.
    // Foreground writes leave it at 0.
    hook_emit_depth: u32,
    // Tenant label stamped onto emitted events. Per-tenant embedders set this
    // when they open the store so events carry the right scope; empty by default.
    hook_tenant: String,
    // Optional actor/principal stamped onto control-plane events (CP1).
    hook_actor: String,
    hook_principal: String,
    // Optional recorder used by embedders that maintain an external read mirror.
    // When enabled around a hook batch, committed node/edge records are copied
    // here after durable publication so the embedder can refresh incrementally.
    recorded_committed_mutations: Option<Vec<GraphMutation>>,
    recorded_redcore_mutations: Option<Vec<RecordedRedCoreMutation>>,
    mutation_log: MutationLog,
    sync_root: Hash,
    sync_parent_by_root: HashMap<Hash, Hash>,
    #[cfg(test)]
    journal_precommit_compaction_reachable: Option<Vec<Hash>>,
    #[cfg(test)]
    journal_precommit_compaction_report: Option<crate::chunk_store::CompactionReport>,
}

#[derive(Debug)]
struct RedCoreDirectoryLock {
    file: File,
    process_key: PathBuf,
}

#[derive(Debug)]
struct PendingHookEvent {
    kind: MutationKind,
    id: String,
    labels: Vec<String>,
    changed_props: Vec<String>,
    scope_context: MutationScopeContext,
    node_transition: Option<MutationNodeTransition>,
    edge_transition: Option<MutationEdgeTransition>,
    edge_context: Option<MutationEdgeContext>,
}

#[derive(Debug)]
struct RedCoreCommitPlan {
    graph_version: u64,
    durable_mutation: RedCoreMutation,
    publish_mutations: Vec<RedCoreMutation>,
    prepared_vector_payloads: Vec<Option<BTreeMap<String, Vec<u8>>>>,
    acquired_vector_pin_receipts: Vec<VectorPayloadPinReceipt>,
    writes: Vec<GraphWriteResult>,
    pending_events: Vec<PendingHookEvent>,
}

fn property_hashes(properties: &Value) -> GraphStoreResult<BTreeMap<String, String>> {
    properties
        .as_object()
        .into_iter()
        .flat_map(|properties| properties.iter())
        .map(|(key, value)| Ok((key.clone(), stable_hash(value)?)))
        .collect()
}

fn mutation_node_context(
    hook_tenant: &str,
    node: &NodeRecord,
) -> GraphStoreResult<MutationNodeContext> {
    Ok(MutationNodeContext {
        id: node.id.clone(),
        labels: node.labels.clone(),
        property_hash: stable_hash(&node.properties)?,
        property_hashes: property_hashes(&node.properties)?,
        scope_context: mutation_scope_context(hook_tenant, [&node.properties]),
    })
}

fn mutation_edge_record_context(
    hook_tenant: &str,
    staged: &InMemoryGraphStore,
    edge: &EdgeRecord,
) -> GraphStoreResult<MutationEdgeRecordContext> {
    let from = staged
        .nodes
        .get(&edge.from_id)
        .filter(|node| !node.tombstone);
    let to = staged.nodes.get(&edge.to_id).filter(|node| !node.tombstone);
    Ok(MutationEdgeRecordContext {
        id: edge.id.clone(),
        from_id: edge.from_id.clone(),
        to_id: edge.to_id.clone(),
        edge_type: edge.edge_type.clone(),
        property_hash: stable_hash(&edge.properties)?,
        property_hashes: property_hashes(&edge.properties)?,
        from_scope_context: mutation_scope_context(
            hook_tenant,
            from.into_iter()
                .map(|node| &node.properties)
                .chain([&edge.properties]),
        ),
        to_scope_context: mutation_scope_context(
            hook_tenant,
            to.into_iter()
                .map(|node| &node.properties)
                .chain([&edge.properties]),
        ),
        from_labels: from.map(|node| node.labels.clone()).unwrap_or_default(),
        to_labels: to.map(|node| node.labels.clone()).unwrap_or_default(),
    })
}

fn mutation_edge_record_context_in_plan(
    hook_tenant: &str,
    base: &InMemoryGraphStore,
    staged_nodes: &BTreeMap<String, NodeRecord>,
    edge: &EdgeRecord,
) -> GraphStoreResult<MutationEdgeRecordContext> {
    let node = |id: &str| {
        staged_nodes
            .get(id)
            .or_else(|| base.nodes.get(id))
            .filter(|node| !node.tombstone)
    };
    let from = node(&edge.from_id);
    let to = node(&edge.to_id);
    Ok(MutationEdgeRecordContext {
        id: edge.id.clone(),
        from_id: edge.from_id.clone(),
        to_id: edge.to_id.clone(),
        edge_type: edge.edge_type.clone(),
        property_hash: stable_hash(&edge.properties)?,
        property_hashes: property_hashes(&edge.properties)?,
        from_scope_context: mutation_scope_context(
            hook_tenant,
            from.into_iter()
                .map(|node| &node.properties)
                .chain([&edge.properties]),
        ),
        to_scope_context: mutation_scope_context(
            hook_tenant,
            to.into_iter()
                .map(|node| &node.properties)
                .chain([&edge.properties]),
        ),
        from_labels: from.map(|node| node.labels.clone()).unwrap_or_default(),
        to_labels: to.map(|node| node.labels.clone()).unwrap_or_default(),
    })
}

fn mutation_scope_context<'a>(
    hook_tenant: &str,
    properties: impl IntoIterator<Item = &'a Value>,
) -> MutationScopeContext {
    let mut tenant_id = hook_tenant.trim().to_string();
    let mut repo_id = None;
    let mut generation = None;
    for properties in properties {
        if tenant_id.is_empty() {
            tenant_id = properties
                .get("tenant_id")
                .or_else(|| properties.get("tenant"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or_default()
                .to_string();
        }
        if repo_id.is_none() {
            repo_id = properties
                .get("repo_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);
        }
        if generation.is_none() {
            generation = properties.get("generation").and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|raw| raw.parse::<u64>().ok()))
            });
        }
    }
    MutationScopeContext {
        tenant_id,
        repo_id,
        generation,
    }
}

fn node_scope_context(
    hook_tenant: &str,
    prior: Option<&NodeRecord>,
    current: Option<&NodeRecord>,
) -> MutationScopeContext {
    mutation_scope_context(
        hook_tenant,
        current
            .into_iter()
            .map(|node| &node.properties)
            .chain(prior.into_iter().map(|node| &node.properties)),
    )
}

fn edge_scope_context(
    hook_tenant: &str,
    staged: &InMemoryGraphStore,
    prior: Option<&EdgeRecord>,
    current: Option<&EdgeRecord>,
) -> MutationScopeContext {
    let edge = current.or(prior);
    mutation_scope_context(
        hook_tenant,
        current
            .into_iter()
            .map(|edge| &edge.properties)
            .chain(prior.into_iter().map(|edge| &edge.properties))
            .chain(
                edge.into_iter()
                    .flat_map(|edge| [&edge.from_id, &edge.to_id])
                    .filter_map(|id| staged.nodes.get(id))
                    .map(|node| &node.properties),
            ),
    )
}

fn edge_scope_context_in_plan(
    hook_tenant: &str,
    base: &InMemoryGraphStore,
    staged_nodes: &BTreeMap<String, NodeRecord>,
    prior: Option<&EdgeRecord>,
    current: Option<&EdgeRecord>,
) -> MutationScopeContext {
    let edge = current.or(prior);
    mutation_scope_context(
        hook_tenant,
        current
            .into_iter()
            .map(|edge| &edge.properties)
            .chain(prior.into_iter().map(|edge| &edge.properties))
            .chain(
                edge.into_iter()
                    .flat_map(|edge| [&edge.from_id, &edge.to_id])
                    .filter_map(|id| {
                        staged_nodes
                            .get(id)
                            .or_else(|| base.nodes.get(id))
                            .filter(|node| !node.tombstone)
                    })
                    .map(|node| &node.properties),
            ),
    )
}

impl RedCoreGraphStore {
    pub fn memory() -> Self {
        let backend = Arc::new(MemChunkStore::new());
        let chunks: Arc<dyn UnifiedChunkStore> = backend;
        let payload_store: Arc<dyn TensorBlockPayloadStore> = Arc::new(
            DiskObjectStore::from_chunk_store(Arc::clone(&chunks))
                .expect("a fresh MemChunkStore must accept its canonical empty catalog"),
        );
        let mut store = InMemoryGraphStore::new();
        store.set_vector_payload_store(payload_store);
        Self {
            store,
            data_dir: None,
            _directory_lock: None,
            chunk_store: UnifiedChunkStoreHandle::new(chunks),
            injected_chunk_store: false,
            journal: None,
            current_pack: None,
            payload_tree: None,
            historical_node_tree: None,
            history_root: None,
            durable_nodes_total: 0,
            durable_edges_total: 0,
            resident_edge_bytes_estimate: 0,
            resident_edge_budget_bytes: None,
            edges_fully_resident: true,
            options: RedCoreOptions {
                durability: RedCoreDurability::None,
                snapshot_interval_writes: 0,
                strict_acid: false,
            },
            last_txn_id: 0,
            snapshot_txn_id: 0,
            recovered_frames: 0,
            skipped_aof_frames: 0,
            aof_size_bytes: 0,
            aof_replay_bytes: 0,
            recovery_duration_ms: 0,
            recovery_profile: RecoveryProfile::default(),
            truncated_aof_tail: false,
            last_recovery_ok: true,
            last_fsync: None,
            transient_ordered_indexes: HashMap::new(),
            hook_emitter: None,
            commit_observer: None,
            batch_preparer: None,
            preparation_identity: Arc::new(()),
            pending_commit_publication: None,
            hook_emit_depth: 0,
            hook_tenant: String::new(),
            hook_actor: String::new(),
            hook_principal: String::new(),
            recorded_committed_mutations: None,
            recorded_redcore_mutations: None,
            mutation_log: MutationLog::genesis(),
            sync_root: genesis_sync_root(),
            sync_parent_by_root: HashMap::new(),
            #[cfg(test)]
            journal_precommit_compaction_reachable: None,
            #[cfg(test)]
            journal_precommit_compaction_report: None,
        }
    }

    /// Build an isolated, non-durable RedCore store from a pinned graph
    /// snapshot. Derived-compute callers use this to stage speculative work
    /// without publishing into the live store.
    pub fn scratch_from_snapshot(snapshot: GraphSnapshot) -> GraphStoreResult<Self> {
        let mut scratch = Self::memory();
        let payload_store = scratch.store.vector_payload_store_handle();
        scratch.store = InMemoryGraphStore::from_snapshot(snapshot)?;
        scratch.store.set_vector_payload_store(payload_store);
        Ok(scratch)
    }

    /// Build a scratch graph over a snapshot whose vector references resolve
    /// through the same content-addressed payload store as the live graph.
    /// This keeps speculative graph mutations isolated while preserving the
    /// immutable payload identity captured by the snapshot.
    pub fn scratch_from_snapshot_with_vector_payload_store(
        snapshot: GraphSnapshot,
        payload_store: Arc<dyn TensorBlockPayloadStore>,
    ) -> GraphStoreResult<Self> {
        let mut scratch = Self::scratch_from_snapshot(snapshot)?;
        scratch.store.set_vector_payload_store(payload_store);
        Ok(scratch)
    }

    pub fn open(data_dir: impl Into<PathBuf>, options: RedCoreOptions) -> GraphStoreResult<Self> {
        Self::open_with_optional_resident_edge_budget(data_dir, options, None)
    }

    /// Open an explicitly selected migration store without the ordinary
    /// recovery-time conversion of legacy vector arrays. Migration owners must
    /// admit the immutable source/schema manifest before publishing changes.
    /// Existing recovery, journal ownership and durability rules still apply.
    pub fn open_preserving_legacy_vectors(
        data_dir: impl Into<PathBuf>,
        options: RedCoreOptions,
    ) -> GraphStoreResult<Self> {
        Self::open_with_vector_recovery_policy(data_dir, options, None, false)
    }

    /// Return the durable store root when this graph is disk-backed.
    ///
    /// Derived indexes use this accessor to stay anchored to the same physical
    /// authority as the graph instead of guessing from process environment.
    pub fn data_dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    fn migrate_legacy_vector_arrays(&mut self) -> GraphStoreResult<()> {
        let mut ids = BTreeSet::new();
        for designation in self.store.vector_designations() {
            ids.extend(
                self.store.legacy_vector_node_ids_for_designation(
                    &designation.label,
                    &designation.property,
                ),
            );
        }
        for (label, property) in BUILTIN_VECTOR_PAYLOAD_PROPERTIES {
            ids.extend(
                self.store
                    .legacy_vector_node_ids_for_designation(label, property),
            );
        }
        if ids.is_empty() {
            return Ok(());
        }
        let mutations = ids
            .into_iter()
            .map(|id| {
                self.store
                    .nodes
                    .get(&id)
                    .cloned()
                    .map(GraphMutation::NodeUpsert)
                    .ok_or_else(|| {
                        GraphStoreError::new(
                            "vector_migration_node_missing",
                            format!("legacy vector node {id} disappeared during recovery"),
                        )
                    })
            })
            .collect::<GraphStoreResult<Vec<_>>>()?;
        self.commit_batch(GraphMutationBatch::new(mutations))?;
        Ok(())
    }

    /// Clone the one already-open physical chunk store used by this RedCore.
    ///
    /// This handle is always available. Durable modes return the existing
    /// journal backend; [`RedCoreDurability::None`] and [`Self::memory`] return
    /// the one MemChunkStore shared by every payload lens in the engine.
    pub fn unified_chunk_store(&self) -> Arc<dyn UnifiedChunkStore> {
        self.chunk_store.get()
    }

    /// Clone the typed durable journal used by this RedCore, when present.
    ///
    /// This compatibility seam is for diagnostics and fault injection. Payload
    /// lenses must use [`Self::unified_chunk_store`] so memory mode works too.
    pub fn chunk_store_handle(&self) -> Option<Arc<JournalChunkStore>> {
        self.journal.clone()
    }

    pub fn recovery_profile(&self) -> &RecoveryProfile {
        &self.recovery_profile
    }

    /// Install one deterministic append failure for a downstream crash-safety
    /// oracle. This hook is absent from production builds.
    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn inject_journal_append_failure_for_test(
        &self,
        successful_appends_before_failure: usize,
        after_bytes: usize,
        fail_rollback: bool,
    ) -> GraphStoreResult<()> {
        let journal = self.journal.as_ref().ok_or_else(|| {
            GraphStoreError::new(
                "journal_fault_injection_unavailable",
                "RedCore has no durable journal",
            )
        })?;
        journal.inject_append_failure_after(
            successful_appends_before_failure,
            after_bytes,
            fail_rollback,
        );
        Ok(())
    }

    pub fn put_cold_document_bytes(&self, body: &[u8]) -> GraphStoreResult<Option<String>> {
        let Some(store) = self.cold_object_store()? else {
            return Ok(None);
        };
        store.put_document_bytes(body).map(Some)
    }

    /// Stage a raw payload in the unified store while retaining a scoped GC
    /// pin across the caller's subsequent graph transaction. On successful
    /// graph publication the caller must invoke
    /// [`StagedRawPayload::finish_after_publication`]; dropping the receipt on
    /// any error aborts the pin.
    pub fn stage_cold_document_bytes(
        &self,
        body: &[u8],
    ) -> GraphStoreResult<Option<StagedRawPayload>> {
        let Some(store) = self.cold_object_store()? else {
            return Ok(None);
        };
        store.stage_document_bytes(body).map(Some)
    }

    pub fn get_cold_document_bytes(&self, content_hash: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(store) = self.cold_object_store()? else {
            return Ok(None);
        };
        store.get_document_bytes(content_hash)
    }

    fn cold_rollback_directory(&self) -> Option<PathBuf> {
        self.data_dir
            .as_ref()
            .map(|data_dir| data_dir.join(REDCORE_COLD_ROLLBACK_DIR))
    }

    fn cold_rollback_record_path(&self, id: &str) -> GraphStoreResult<Option<PathBuf>> {
        self.cold_rollback_directory()
            .map(|directory| {
                Ok(directory.join(format!("{}.json", safe_filename(&stable_hash(id)?))))
            })
            .transpose()
    }

    fn remove_cold_rollback_record(&self, id: &str) -> GraphStoreResult<()> {
        let Some(path) = self.cold_rollback_record_path(id)? else {
            return Ok(());
        };
        match fs::remove_file(&path) {
            Ok(()) => sync_directory(
                path.parent().unwrap_or_else(|| Path::new(".")),
                "fsync cold rollback directory after removal",
            ),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(GraphStoreError::io("remove cold rollback record", error)),
        }
    }

    fn remove_superseded_cold_rollback_records(&self, mutations: &[RedCoreMutation]) {
        for mutation in mutations {
            match mutation {
                RedCoreMutation::NodeUpsert(node) => {
                    let _ = self.remove_cold_rollback_record(&node.id);
                }
                RedCoreMutation::NodeDelete(id) => {
                    let _ = self.remove_cold_rollback_record(id);
                }
                RedCoreMutation::Batch(nested) => {
                    self.remove_superseded_cold_rollback_records(nested);
                }
                RedCoreMutation::EdgeUpsert(_)
                | RedCoreMutation::EdgeDelete(_)
                | RedCoreMutation::VectorDesignation(_)
                | RedCoreMutation::OrderedDesignation(_)
                | RedCoreMutation::VectorDesignationClear(_)
                | RedCoreMutation::OrderedDesignationClear(_) => {}
            }
        }
    }

    fn read_cold_rollback_record(path: &Path) -> GraphStoreResult<RedCoreColdRollbackRecord> {
        let raw = fs::read(path)
            .map_err(|error| GraphStoreError::io("read cold rollback record", error))?;
        serde_json::from_slice(&raw)
            .map_err(|error| GraphStoreError::io("decode cold rollback record", error))
    }

    fn prune_resident_cold_rollback_records(&self) -> GraphStoreResult<()> {
        let Some(directory) = self.cold_rollback_directory() else {
            return Ok(());
        };
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(GraphStoreError::io("read cold rollback directory", error)),
        };
        for entry in entries {
            let path = entry
                .map_err(|error| GraphStoreError::io("read cold rollback entry", error))?
                .path();
            if !path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                continue;
            }
            let record = Self::read_cold_rollback_record(&path)?;
            if !self.store.cold_nonresident_node_ids.contains(&record.id) {
                fs::remove_file(path).map_err(|error| {
                    GraphStoreError::io("remove stale cold rollback record", error)
                })?;
            }
        }
        Ok(())
    }

    fn evict_node_residency(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        // Residency is a cache concern now. The committed logical and payload
        // roots retain the complete record, so eviction must not manufacture a
        // second durable mutation stream.
        self.store.evict_node(id)
    }

    fn readmit_node_residency(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        let id = node.id.clone();
        self.remove_cold_rollback_record(&id)?;
        self.store.readmit_node(node)
    }

    fn cold_object_store(&self) -> GraphStoreResult<Option<DiskObjectStore>> {
        let shared = self.unified_chunk_store();
        match self.data_dir.as_ref() {
            Some(data_dir) => DiskObjectStore::from_chunk_store_with_legacy_root(
                shared,
                data_dir.join("cold_objects"),
            )
            .map(Some),
            None => DiskObjectStore::from_chunk_store(shared).map(Some),
        }
    }

    pub fn readiness_check(
        data_dir: &Path,
        durability: RedCoreDurability,
        strict_acid: bool,
    ) -> GraphStoreResult<()> {
        validate_redcore_options(&RedCoreOptions {
            durability,
            snapshot_interval_writes: 1,
            strict_acid,
        })?;
        fs::create_dir_all(data_dir)
            .map_err(|err| GraphStoreError::io("create RedCore data directory", err))?;
        if let Some(policy) = durability.journal_commit_policy() {
            drop(JournalChunkStore::open_with_commit_policy(
                data_dir.join(REDCORE_JOURNAL_FILE),
                policy,
            )?);
        }
        Self::write_readiness_probe(data_dir)
    }

    /// Verify the durable root through this store's already-owned journal.
    ///
    /// A resident store must not reopen its journal just to answer a health
    /// check: the journal lock correctly rejects a second open in the same
    /// process. The live handle proves journal ownership while the probe below
    /// still proves that the mounted filesystem is writable and syncable.
    pub fn resident_readiness_check(&self) -> GraphStoreResult<()> {
        validate_redcore_options(&self.options)?;
        if let Some(journal) = self.journal.as_ref() {
            journal.readiness_check()?;
        }
        match self.data_dir.as_deref() {
            Some(data_dir) => Self::write_readiness_probe(data_dir),
            None => Ok(()),
        }
    }

    fn write_readiness_probe(data_dir: &Path) -> GraphStoreResult<()> {
        let probe_path = data_dir.join(".ready-probe");
        fs::write(&probe_path, b"ok")
            .map_err(|err| GraphStoreError::io("write RedCore readiness probe", err))?;
        sync_file_path(&probe_path, "fsync RedCore readiness probe")?;
        fs::remove_file(&probe_path)
            .map_err(|err| GraphStoreError::io("remove RedCore readiness probe", err))?;
        sync_directory(
            data_dir,
            "fsync RedCore data directory after readiness probe",
        )?;
        Ok(())
    }

    pub fn status(&self) -> RedCoreStatus {
        let (commit_policy, commit_batch_window_ms) =
            match self.journal.as_ref().map(|journal| journal.commit_policy()) {
                None if self.injected_chunk_store => ("backend_owned", 0),
                None => ("memory_only", 0),
                Some(JournalCommitPolicy::Immediate) => ("immediate_fsync", 0),
                Some(JournalCommitPolicy::Grouped { max_wait }) => (
                    "grouped_fsync",
                    u64::try_from(max_wait.as_millis()).unwrap_or(u64::MAX),
                ),
            };
        let resident_stats = self.store.stats_without_memory_estimate();
        RedCoreStatus {
            mode: if self.injected_chunk_store {
                "injected".to_string()
            } else if self.data_dir.is_some() {
                "embedded".to_string()
            } else {
                "memory".to_string()
            },
            durability: self.options.durability.as_str().to_string(),
            commit_policy: commit_policy.to_string(),
            commit_batch_window_ms,
            data_dir: self
                .data_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            graph_version: resident_stats.version,
            last_txn_id: self.last_txn_id,
            snapshot_txn_id: self.snapshot_txn_id,
            txn_lag_since_snapshot: self.last_txn_id.saturating_sub(self.snapshot_txn_id),
            recovered_frames: self.recovered_frames,
            skipped_aof_frames: self.skipped_aof_frames,
            aof_size_bytes: self.aof_size_bytes,
            aof_replay_bytes: self.aof_replay_bytes,
            recovery_duration_ms: self.recovery_duration_ms,
            truncated_aof_tail: self.truncated_aof_tail,
            last_fsync_unix_ms: self.last_fsync.and_then(system_time_unix_ms),
            last_recovery_ok: self.last_recovery_ok,
            strict_acid: self.options.strict_acid,
            durable_nodes_total: if self.edges_fully_resident {
                resident_stats.nodes_total
            } else {
                self.durable_nodes_total
            },
            durable_edges_total: if self.edges_fully_resident {
                resident_stats.edges_total
            } else {
                self.durable_edges_total
            },
            resident_edges_total: resident_stats.edges_total,
            resident_edge_bytes_estimate: self.resident_edge_bytes_estimate,
            resident_edge_budget_bytes: self.resident_edge_budget_bytes,
            edges_fully_resident: self.edges_fully_resident,
        }
    }

    pub fn graph_snapshot(&self) -> GraphSnapshot {
        self.store.snapshot()
    }

    pub fn start_recording_committed_mutations(&mut self) {
        self.recorded_committed_mutations = Some(Vec::new());
    }

    pub fn take_recorded_committed_mutations(&mut self) -> Option<GraphMutationBatch> {
        self.recorded_committed_mutations
            .take()
            .map(GraphMutationBatch::new)
    }

    pub fn start_recording_redcore_mutations(&mut self) {
        self.recorded_redcore_mutations = Some(Vec::new());
    }

    pub fn take_recorded_redcore_mutations(&mut self) -> Option<RecordedRedCoreMutationBatch> {
        self.recorded_redcore_mutations
            .take()
            .map(|mutations| RecordedRedCoreMutationBatch { mutations })
    }

    fn record_committed_redcore_mutations(&mut self, mutations: &[RedCoreMutation]) {
        if self.recorded_committed_mutations.is_none() && self.recorded_redcore_mutations.is_none()
        {
            return;
        }
        let mut recorded = Vec::new();
        for mutation in mutations {
            Self::collect_recordable_mutations(mutation, self.store.version, &mut recorded);
        }
        if let Some(recorder) = &mut self.recorded_redcore_mutations {
            recorder.extend(recorded.clone());
        }
        if let Some(recorder) = &mut self.recorded_committed_mutations {
            recorder.extend(recorded.into_iter().filter_map(|mutation| match mutation {
                RecordedRedCoreMutation::NodeUpsert { record } => {
                    Some(GraphMutation::NodeUpsert(record))
                }
                RecordedRedCoreMutation::EdgeUpsert { record } => {
                    Some(GraphMutation::EdgeUpsert(record))
                }
                RecordedRedCoreMutation::NodeDelete { id, .. } => {
                    Some(GraphMutation::NodeDelete(id))
                }
                RecordedRedCoreMutation::EdgeDelete { id, .. } => {
                    Some(GraphMutation::EdgeDelete(id))
                }
            }));
        }
    }

    fn collect_recordable_mutations(
        mutation: &RedCoreMutation,
        version: u64,
        out: &mut Vec<RecordedRedCoreMutation>,
    ) {
        match mutation {
            RedCoreMutation::NodeUpsert(node) => {
                out.push(RecordedRedCoreMutation::NodeUpsert {
                    record: node.clone(),
                });
            }
            RedCoreMutation::EdgeUpsert(edge) => {
                out.push(RecordedRedCoreMutation::EdgeUpsert {
                    record: edge.clone(),
                });
            }
            RedCoreMutation::Batch(mutations) => {
                for mutation in mutations {
                    Self::collect_recordable_mutations(mutation, version, out);
                }
            }
            RedCoreMutation::NodeDelete(id) => out.push(RecordedRedCoreMutation::NodeDelete {
                id: id.clone(),
                version,
            }),
            RedCoreMutation::EdgeDelete(id) => out.push(RecordedRedCoreMutation::EdgeDelete {
                id: id.clone(),
                version,
            }),
            RedCoreMutation::VectorDesignation(_)
            | RedCoreMutation::OrderedDesignation(_)
            | RedCoreMutation::VectorDesignationClear(_)
            | RedCoreMutation::OrderedDesignationClear(_) => {}
        }
    }

    // ---- Graph-level hooks (additive emit seam; see crate::hooks) -------
    //
    // The emit points live at the post-commit, post-publish boundary in
    // `commit_batch` and `purge_expired_nodes`. They are strictly additive:
    // when no emitter is attached (the default), they are skipped entirely.

    /// Attach transport publication independently of optional queued hooks.
    /// Changing the preparation policy invalidates outstanding certificates.
    pub fn attach_batch_preparer(&mut self, preparer: GraphBatchPreparer) {
        self.batch_preparer = Some(preparer);
        self.preparation_identity = Arc::new(());
    }

    pub fn prepare_batch(&self, batch: GraphMutationBatch) -> GraphStoreResult<PreparedGraphBatch> {
        let base = self.sync_cursor();
        if batch
            .expected_root
            .is_some_and(|root| root != Hash::from_bytes(base.root))
        {
            return Err(GraphStoreError::stale_write_base(base));
        }
        let final_batch = match &self.batch_preparer {
            Some(prepare) if !batch.mutations.is_empty() => (prepare.0)(self, &batch)?,
            _ => batch.clone(),
        };
        if final_batch.expected_root != batch.expected_root
            || !final_batch.mutations.starts_with(&batch.mutations)
        {
            return Err(GraphStoreError::new(
                "invalid_batch_preparation",
                "preparation must preserve the expected root and original mutation prefix",
            ));
        }
        Ok(PreparedGraphBatch {
            batch: final_batch,
            base,
            writer: self.preparation_identity.clone(),
        })
    }

    pub fn commit_prepared_batch(
        &mut self,
        prepared: PreparedGraphBatch,
    ) -> GraphStoreResult<GraphTransaction> {
        self.commit_prepared_batch_internal(prepared, None, true, None)
    }

    pub fn commit_prepared_batch_with_metadata(
        &mut self,
        prepared: PreparedGraphBatch,
        metadata: GraphCommitMetadata,
    ) -> GraphStoreResult<GraphCommitReceipt> {
        let transaction =
            self.commit_prepared_batch_internal(prepared, None, true, Some(&metadata))?;
        let commit = self
            .current_pack
            .as_ref()
            .ok_or_else(|| {
                GraphStoreError::new(
                    "graph_history_integrity",
                    "successful graph publication has no compiler header",
                )
            })?
            .commit
            .clone();
        Ok(GraphCommitReceipt {
            transaction,
            commit,
        })
    }

    pub fn attach_commit_observer(&mut self, observer: GraphCommitObserver) {
        self.commit_observer = Some(observer);
    }

    pub fn detach_commit_observer(&mut self) {
        self.commit_observer = None;
    }

    /// Install a post-commit hook emitter (handed out by a `HookDispatcher`).
    /// Until set, every mutation emit is a no-op, so hooks are strictly opt-in
    /// and existing callers are unaffected.
    pub fn attach_hook_emitter(&mut self, emitter: HookEmitter) {
        self.hook_emitter = Some(emitter);
    }

    /// Drop the hook emitter, returning the store to a no-emit state.
    pub fn detach_hook_emitter(&mut self) {
        self.hook_emitter = None;
    }

    pub fn has_hook_emitter(&self) -> bool {
        self.hook_emitter.is_some()
    }

    /// Set the tenant label stamped onto emitted mutation events. Per-tenant
    /// embedders set this when they open the store.
    pub fn set_hook_tenant(&mut self, tenant: impl Into<String>) {
        self.hook_tenant = tenant.into();
    }

    pub fn hook_tenant(&self) -> &str {
        &self.hook_tenant
    }

    pub fn set_hook_actor(&mut self, actor: impl Into<String>) {
        self.hook_actor = actor.into();
    }

    pub fn hook_actor(&self) -> &str {
        &self.hook_actor
    }

    pub fn set_hook_principal(&mut self, principal: impl Into<String>) {
        self.hook_principal = principal.into();
    }

    pub fn hook_principal(&self) -> &str {
        &self.hook_principal
    }

    fn append_control_event_mutations(
        &self,
        staged: &mut InMemoryGraphStore,
        durable_mutations: &mut Vec<RedCoreMutation>,
        projections: &[NodeMutationProjection],
        graph_version_before: u64,
        txn_id: u64,
    ) -> GraphStoreResult<()> {
        if projections.is_empty() {
            return Ok(());
        }
        let committed_at_ms = now_ms().max(0) as u64;
        // Version after user mutations but before control-event nodes land.
        let graph_version_after = staged.stats_without_memory_estimate().version;
        let tenant = if self.hook_tenant.trim().is_empty() {
            "default"
        } else {
            self.hook_tenant.as_str()
        };
        let actor = (!self.hook_actor.trim().is_empty()).then_some(self.hook_actor.as_str());
        let principal =
            (!self.hook_principal.trim().is_empty()).then_some(self.hook_principal.as_str());
        let events = control_events_from_projections(
            tenant,
            actor,
            principal,
            graph_version_before,
            graph_version_after,
            committed_at_ms,
            txn_id,
            projections,
        );
        for event in events {
            let node = event.to_node();
            // ControlEvent nodes must not recurse into further control events.
            debug_assert!(node.labels.iter().any(|label| label == CONTROL_EVENT_LABEL));
            staged.upsert_node(node.clone())?;
            durable_mutations.push(RedCoreMutation::NodeUpsert(node));
        }
        Ok(())
    }

    /// Set the loop-guard generation stamped onto subsequently emitted events.
    /// The hook worker sets this around handler execution so a handler's writes
    /// are tagged one generation deeper; foreground callers leave it at 0.
    pub fn set_hook_emit_depth(&mut self, depth: u32) {
        self.hook_emit_depth = depth;
    }

    pub fn hook_emit_depth(&self) -> u32 {
        self.hook_emit_depth
    }

    /// Non-blocking post-commit emit. No-op without an attached emitter. Called
    /// only after a commit is durably published, never inside the commit
    /// critical section.
    fn emit_hook_events(
        &mut self,
        pending_events: Vec<PendingHookEvent>,
        txn_id: u64,
        graph_version: u64,
    ) {
        if let Some(publication) = self.pending_commit_publication.take() {
            if let Some(observer) = &self.commit_observer {
                observer.publish(&publication);
            }
        }
        let Some(emitter) = &self.hook_emitter else {
            return;
        };
        let committed_at_ms = now_ms().max(0) as u64;
        let event_count = pending_events.len() as u64;
        for (sequence, pending) in pending_events.into_iter().enumerate() {
            let mut event = MutationEvent::new(
                pending.kind,
                self.hook_tenant.clone(),
                pending.id,
                pending.labels,
                pending.changed_props,
                committed_at_ms,
                self.hook_emit_depth,
            );
            event.commit_context = Some(MutationCommitContext {
                txn_id,
                graph_version,
                sequence: sequence as u64,
                event_count,
            });
            event.scope_context = Some(pending.scope_context);
            event.node_transition = pending.node_transition;
            event.edge_transition = pending.edge_transition;
            event.edge_context = pending.edge_context;
            emitter.try_emit(event);
        }
    }

    /// Content-addressed sync cursor. `root` is the Prolly tree (or memory
    /// chain) identity; `txn_id` is the linear index.
    pub fn version(&self) -> SyncCursor {
        self.sync_cursor()
    }

    pub fn sync_cursor(&self) -> SyncCursor {
        SyncCursor {
            root: *self.sync_root.as_bytes(),
            txn_id: self.last_txn_id,
        }
    }

    pub fn changes_since(&self, since: &SyncCursor) -> GraphStoreResult<ChangesSince> {
        Ok(mutation_log::changes_since(
            &self.mutation_log,
            &self.sync_parent_by_root,
            self.sync_cursor(),
            since,
        ))
    }

    #[cfg(test)]
    pub fn commit_batch_without_incremental(
        &mut self,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<GraphTransaction> {
        self.commit_batch_internal(batch, None, false, None)
    }

    #[cfg(test)]
    pub fn swing_sync_root_for_test(&mut self, root: Hash) {
        self.sync_root = root;
    }

    #[cfg(test)]
    pub fn drop_log_before_txn_for_test(&mut self, txn_id: u64) {
        self.mutation_log
            .entries
            .retain(|entry| entry.txn_id >= txn_id);
        if let Some(first) = self.mutation_log.entries.first() {
            self.mutation_log.log_base_txn_id = first.txn_id;
            self.mutation_log.log_base_root = first.root;
        } else {
            self.mutation_log.log_base_txn_id = txn_id;
        }
    }

    fn log_entries_from_pending(
        &self,
        pending: &[PendingHookEvent],
        txn_id: u64,
        root: Hash,
    ) -> Vec<MutationLogEntry> {
        let committed_at_ms = now_ms().max(0) as u64;
        pending
            .iter()
            .enumerate()
            .map(|(sequence, pending)| MutationLogEntry {
                txn_id,
                root: *root.as_bytes(),
                kind: mutation_log::wire_kind(pending.kind),
                id: pending.id.clone(),
                labels: pending.labels.clone(),
                changed_props: pending.changed_props.clone(),
                tenant: self.hook_tenant.clone(),
                committed_at_ms,
                sequence: sequence as u64,
            })
            .collect()
    }

    fn preview_mutation_log(
        &self,
        pending: &[PendingHookEvent],
        txn_id: u64,
        root: Hash,
    ) -> MutationLog {
        let mut preview = self.mutation_log.clone();
        if !pending.is_empty() && !preview.contains_txn(txn_id) {
            preview.append_entries(self.log_entries_from_pending(pending, txn_id, root));
        }
        preview
    }

    fn note_sync_root(&mut self, root: Hash) {
        if root != self.sync_root {
            self.sync_parent_by_root.insert(root, self.sync_root);
            self.sync_root = root;
        }
    }

    fn finish_mutation_log(&mut self, pending: &[PendingHookEvent], txn_id: u64) {
        let root = self
            .current_pack
            .as_ref()
            .map(|pack| pack.tree.root)
            .unwrap_or_else(|| mutation_log::memory_sync_root(self.sync_root, txn_id));
        if !self.mutation_log.contains_txn(txn_id) {
            let entries = self.log_entries_from_pending(pending, txn_id, root);
            self.mutation_log.append_entries(entries);
        }
        self.note_sync_root(root);
    }

    pub fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        self.commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(node)]))?
            .writes
            .into_iter()
            .next()
            .ok_or_else(|| GraphStoreError::new("redcore_missing_write", "node write vanished"))
    }

    /// Durably remove a live node from RedCore storage. This is a logical
    /// delete, not a working-set residency change: the incident-edge and node
    /// tombstones are rooted before the staged mirror is published. Returns
    /// `false` when the node is already absent.
    pub fn delete_node(&mut self, id: &str) -> GraphStoreResult<bool> {
        if id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if self
            .store
            .get_node_including_expired(id)
            .filter(|node| !node.tombstone)
            .is_none()
        {
            return Ok(false);
        }
        self.commit_batch(GraphMutationBatch::new([GraphMutation::NodeDelete(
            id.to_string(),
        )]))?;
        Ok(true)
    }

    /// Durably remove a live edge and emit an actual `EdgeDeleted` hook event.
    pub fn delete_edge(&mut self, id: &str) -> GraphStoreResult<bool> {
        self.require_fully_resident_edges("edge delete")?;
        if id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("edge.id"));
        }
        if self.store.get_edge(id).is_none() {
            return Ok(false);
        }
        self.commit_batch(GraphMutationBatch::new([GraphMutation::EdgeDelete(
            id.to_string(),
        )]))?;
        Ok(true)
    }

    /// Insert `node` only when no live record with the same id exists.
    /// Returns `Ok(None)` for ordinary dedup contention and roots the winning
    /// insert as an ordinary node upsert.
    pub fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_absent_internal(node, None, |_, _| Ok(()))
    }

    pub fn insert_node_if_absent_checked(
        &mut self,
        node: NodeRecord,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_absent_internal(node, None, check)
    }

    pub fn insert_node_if_absent_with_metadata_checked(
        &mut self,
        node: NodeRecord,
        metadata: GraphCommitMetadata,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_absent_internal(node, Some(&metadata), check)
    }

    fn insert_node_if_absent_internal(
        &mut self,
        node: NodeRecord,
        metadata: Option<&GraphCommitMetadata>,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if self.store.get_node(&node.id).is_some() {
            return Ok(None);
        }
        self.guarded_node_upsert(node, metadata, check).map(Some)
    }

    pub fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_never_existed_internal(node, None, |_, _| Ok(()))
    }

    pub fn insert_node_if_never_existed_checked(
        &mut self,
        node: NodeRecord,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_never_existed_internal(node, None, check)
    }

    pub fn insert_node_if_never_existed_with_metadata_checked(
        &mut self,
        node: NodeRecord,
        metadata: GraphCommitMetadata,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.insert_node_if_never_existed_internal(node, Some(&metadata), check)
    }

    fn insert_node_if_never_existed_internal(
        &mut self,
        node: NodeRecord,
        metadata: Option<&GraphCommitMetadata>,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if !self.store.historical_node_history_complete {
            return Err(GraphStoreError::new(
                "historical_registry_incomplete",
                "cannot reserve a never-before-used node id from incomplete historical state",
            ));
        }
        if self.store.has_historical_node_id(&node.id) {
            return Ok(None);
        }
        self.guarded_node_upsert(node, metadata, check).map(Some)
    }

    /// Atomically set a node property when its current value equals `expected`.
    /// Missing properties compare as JSON null. `Value::Null` as `new_value`
    /// removes the property. The winning update journals as a NodeUpsert AOF
    /// mutation, so replay reconstructs the claimed state.
    pub fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.compare_and_set_node_property_internal(
            id,
            property,
            expected,
            new_value,
            None,
            |_, _| Ok(()),
        )
    }

    pub fn compare_and_set_node_property_checked(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.compare_and_set_node_property_internal(id, property, expected, new_value, None, check)
    }

    pub fn compare_and_set_node_property_with_metadata_checked(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
        metadata: GraphCommitMetadata,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        self.compare_and_set_node_property_internal(
            id,
            property,
            expected,
            new_value,
            Some(&metadata),
            check,
        )
    }

    fn compare_and_set_node_property_internal(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
        metadata: Option<&GraphCommitMetadata>,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        let property = property.trim();
        if property.is_empty() {
            return Err(GraphStoreError::empty_field("node.property"));
        }
        let mut node = self
            .store
            .get_node_including_expired(id)
            .filter(|node| !node.tombstone)
            .cloned()
            .ok_or_else(|| {
                GraphStoreError::new("missing_graph_node", format!("node {id} does not exist"))
            })?;
        let current = node
            .properties
            .as_object()
            .and_then(|properties| properties.get(property))
            .cloned()
            .unwrap_or(Value::Null);
        if &current != expected {
            return Ok(None);
        }
        let mut properties = node
            .properties
            .as_object()
            .cloned()
            .unwrap_or_else(serde_json::Map::new);
        if new_value.is_null() {
            properties.remove(property);
        } else {
            properties.insert(property.to_string(), new_value);
        }
        node.properties = Value::Object(properties);
        self.guarded_node_upsert(node, metadata, check).map(Some)
    }

    fn guarded_node_upsert(
        &mut self,
        node: NodeRecord,
        metadata: Option<&GraphCommitMetadata>,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<GraphWriteResult> {
        let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(node)]);
        let prepared = self.prepare_batch(batch)?;
        check(self, prepared.batch())?;
        self.commit_prepared_batch_internal(prepared, None, true, metadata)?
            .writes
            .into_iter()
            .next()
            .ok_or_else(|| GraphStoreError::new("redcore_missing_write", "node write vanished"))
    }

    pub fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        self.commit_batch(GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge)]))?
            .writes
            .into_iter()
            .next()
            .ok_or_else(|| GraphStoreError::new("redcore_missing_write", "edge write vanished"))
    }

    pub fn commit_batch(
        &mut self,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<GraphTransaction> {
        self.commit_batch_internal(batch, None, true, None)
    }

    pub fn commit_batch_with_metadata(
        &mut self,
        batch: GraphMutationBatch,
        metadata: GraphCommitMetadata,
    ) -> GraphStoreResult<GraphCommitReceipt> {
        let transaction = self.commit_batch_internal(batch, None, true, Some(&metadata))?;
        let commit = self
            .current_pack
            .as_ref()
            .ok_or_else(|| {
                GraphStoreError::new(
                    "graph_history_integrity",
                    "successful graph publication has no compiler header",
                )
            })?
            .commit
            .clone();
        Ok(GraphCommitReceipt {
            transaction,
            commit,
        })
    }

    pub fn commit_conditional_batch_with_metadata(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
        metadata: GraphCommitMetadata,
    ) -> GraphStoreResult<Option<GraphCommitReceipt>> {
        let Some(batch) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(None);
        };
        self.commit_batch_with_metadata(batch, metadata).map(Some)
    }

    pub fn commit_conditional_batch_with_metadata_checked<F>(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
        metadata: GraphCommitMetadata,
        check: F,
    ) -> GraphStoreResult<Option<(GraphCommitReceipt, GraphMutationBatch)>>
    where
        F: FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    {
        let Some(prepared) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(None);
        };
        let prepared = self.prepare_batch(prepared)?;
        check(self, prepared.batch())?;
        let committed = prepared.batch().clone();
        let receipt = self.commit_prepared_batch_with_metadata(prepared, metadata)?;
        Ok(Some((receipt, committed)))
    }

    pub fn pin_graph_history(&self) -> GraphStoreResult<PinnedGraphHistory> {
        let roots = self.revision_roots()?.ok_or_else(|| {
            GraphStoreError::new(
                "graph_history_uninitialized",
                "no graph root has been published",
            )
        })?;
        PinnedGraphHistory::new(self.chunk_store.get(), self.history_root, roots)
    }

    fn revision_roots(&self) -> GraphStoreResult<Option<GraphRevisionRoots>> {
        match (&self.current_pack, &self.payload_tree) {
            (None, None) => Ok(None),
            (Some(pack), Some(payload_tree)) => Ok(Some(GraphRevisionRoots {
                transaction: self.last_txn_id,
                graph_version: pack.commit.graph_version,
                logical_commit: pack.commit.commit_hash,
                logical_tree: pack.tree.clone(),
                payload_tree: payload_tree.clone(),
                historical_node_tree: self.historical_node_tree.clone(),
                historical_node_history_complete: self.store.historical_node_history_complete,
                vector_designations: self.store.vector_designations(),
                ordered_designations: self.store.ordered_designations(),
            })),
            _ => Err(GraphStoreError::new(
                "graph_history_integrity",
                "graph root descriptors are incomplete",
            )),
        }
    }

    fn commit_batch_internal(
        &mut self,
        batch: GraphMutationBatch,
        vector_designation: Option<VectorDesignation>,
        allow_incremental: bool,
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<GraphTransaction> {
        let prepared = self.prepare_batch(batch)?;
        self.commit_prepared_batch_internal(
            prepared,
            vector_designation,
            allow_incremental,
            metadata,
        )
    }

    fn commit_prepared_batch_internal(
        &mut self,
        prepared: PreparedGraphBatch,
        vector_designation: Option<VectorDesignation>,
        allow_incremental: bool,
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<GraphTransaction> {
        let current = self.sync_cursor();
        if !Arc::ptr_eq(&prepared.writer, &self.preparation_identity) {
            return Err(GraphStoreError::new(
                "invalid_batch_preparation",
                "certificate belongs to another writer or preparation policy",
            ));
        }
        if prepared.base != current {
            return Err(GraphStoreError::stale_write_base(current));
        }
        let batch = prepared.batch;
        if batch.mutations.is_empty() && vector_designation.is_none() {
            return Err(GraphStoreError::empty_transaction());
        }
        if let Some(expected) = batch.expected_root {
            let current = self.sync_cursor();
            if expected != Hash::from_bytes(current.root) {
                return Err(GraphStoreError::stale_write_base(current));
            }
        }
        if !self.edges_fully_resident
            && batch.mutations.iter().any(|mutation| {
                matches!(
                    mutation,
                    GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_)
                )
            })
        {
            return Err(self.resident_edges_nonresident_error(
                "graph deletes require the complete prior edge set",
            ));
        }
        if !self.edges_fully_resident
            && !allow_incremental
            && batch
                .mutations
                .iter()
                .any(|mutation| matches!(mutation, GraphMutation::EdgeUpsert(_)))
        {
            return Err(self.resident_edges_nonresident_error(
                "non-incremental edge writes require the complete prior edge set",
            ));
        }

        let projected_resident_edge_bytes = if self.edges_fully_resident
            && self.resident_edge_budget_bytes.is_some()
        {
            let mut projected = self.resident_edge_bytes_estimate;
            let mut staged_edges = BTreeMap::<String, EdgeRecord>::new();
            for mutation in &batch.mutations {
                let GraphMutation::EdgeUpsert(edge) = mutation else {
                    continue;
                };
                let prior = staged_edges
                    .get(&edge.id)
                    .or_else(|| self.store.edges.get(&edge.id));
                if let Some(prior) = prior {
                    projected = projected.saturating_sub(estimated_resident_edge_bytes(prior)?);
                }
                projected = projected.saturating_add(estimated_resident_edge_bytes(edge)?);
                staged_edges.insert(edge.id.clone(), edge.clone());
            }
            if projected > self.resident_edge_budget_bytes.unwrap_or_default() {
                return Err(GraphStoreError::new(
                        "resident_edge_budget_exceeded",
                        format!(
                            "edge write would require approximately {projected} resident bytes, above the configured {} byte edge budget",
                            self.resident_edge_budget_bytes.unwrap_or_default()
                        ),
                    ));
            }
            Some(projected)
        } else {
            None
        };

        let txn_id = self.last_txn_id + 1;
        if allow_incremental
            && vector_designation.is_none()
            && self.can_commit_incrementally(txn_id, &batch)
        {
            GRAPH_DELTA_COMMIT_PUBLICATIONS.fetch_add(1, Ordering::Relaxed);
            return self.commit_batch_incremental(txn_id, batch, metadata);
        }
        GRAPH_STAGED_COMMIT_PUBLICATIONS.fetch_add(1, Ordering::Relaxed);
        let mut staged = self.store.clone();
        let mut vector_pins = VectorPayloadPinGuard::new(staged.vector_payload_store_handle());
        let mut writes = Vec::with_capacity(batch.mutations.len());
        let mut durable_mutations = Vec::with_capacity(batch.mutations.len());
        if let Some(designation) = vector_designation {
            let migration_ids = staged
                .legacy_vector_node_ids_for_designation(&designation.label, &designation.property);
            vector_pins.extend(staged.designate_vector_property_retaining_pins(
                &designation.label,
                &designation.property,
                designation.dimension,
            )?);
            durable_mutations.push(RedCoreMutation::VectorDesignation(designation));
            for id in migration_ids {
                let migrated = staged.nodes.get(&id).cloned().ok_or_else(|| {
                    GraphStoreError::new(
                        "vector_migration_node_missing",
                        format!("migrated vector node {id} disappeared"),
                    )
                })?;
                durable_mutations.push(RedCoreMutation::NodeUpsert(migrated));
            }
        }
        let emit_hooks = true;
        let mut pending_events = Vec::<PendingHookEvent>::new();
        let mut control_projections = Vec::<NodeMutationProjection>::new();
        let graph_version_before = self.store.version;
        let mut edge_deletes = Vec::new();
        let mut node_deletes = Vec::new();
        let mut node_upserts = Vec::new();
        let mut edge_upserts = Vec::new();
        let mut staged_edge_upsert_ids = BTreeSet::new();
        let batch_edge_upsert_ids = batch
            .mutations
            .iter()
            .filter_map(|mutation| match mutation {
                GraphMutation::EdgeUpsert(edge) => Some(edge.id.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::EdgeDelete(id) => edge_deletes.push(GraphMutation::EdgeDelete(id)),
                GraphMutation::NodeDelete(id) => {
                    // A rooted graph cannot retain live edges whose endpoint
                    // is tombstoned. Expand the node delete inside this same
                    // transaction so the resident mirror and recovery walk
                    // have identical edge sets before and after restart. An
                    // edge replaced later in this batch remains one logical
                    // transition: its upsert captures the immutable old side
                    // before installing endpoints that survive the deletion.
                    edge_deletes.extend(
                        staged
                            .edges
                            .values()
                            .filter(|edge| {
                                !edge.tombstone
                                    && (edge.from_id == id || edge.to_id == id)
                                    && !batch_edge_upsert_ids.contains(&edge.id)
                            })
                            .map(|edge| GraphMutation::EdgeDelete(edge.id.clone())),
                    );
                    node_deletes.push(GraphMutation::NodeDelete(id));
                }
                GraphMutation::NodeUpsert(node) => {
                    node_upserts.push(GraphMutation::NodeUpsert(node));
                }
                GraphMutation::EdgeUpsert(edge) => {
                    edge_upserts.push(GraphMutation::EdgeUpsert(edge));
                }
            }
        }

        for mutation in edge_deletes
            .into_iter()
            .chain(node_deletes)
            .chain(node_upserts)
            .chain(edge_upserts)
        {
            match mutation {
                GraphMutation::EdgeDelete(id) => {
                    if id.trim().is_empty() {
                        return Err(GraphStoreError::empty_field("edge.id"));
                    }
                    let Some(prior) = staged.get_edge(&id).cloned() else {
                        continue;
                    };
                    staged.apply_logical_edge_delete(&id)?;
                    if emit_hooks {
                        pending_events.push(PendingHookEvent {
                            kind: MutationKind::EdgeDeleted,
                            id: id.clone(),
                            labels: vec![prior.edge_type.clone()],
                            changed_props: Vec::new(),
                            scope_context: edge_scope_context(
                                &self.hook_tenant,
                                &staged,
                                Some(&prior),
                                None,
                            ),
                            node_transition: None,
                            edge_transition: Some(MutationEdgeTransition {
                                prior: Some(mutation_edge_record_context(
                                    &self.hook_tenant,
                                    &staged,
                                    &prior,
                                )?),
                                current: None,
                            }),
                            edge_context: Some(MutationEdgeContext {
                                from_id: prior.from_id.clone(),
                                to_id: prior.to_id.clone(),
                                edge_type: prior.edge_type.clone(),
                            }),
                        });
                    }
                    durable_mutations.push(RedCoreMutation::EdgeDelete(id));
                }
                GraphMutation::NodeDelete(id) => {
                    if id.trim().is_empty() {
                        return Err(GraphStoreError::empty_field("node.id"));
                    }
                    let Some(prior) = staged
                        .nodes
                        .get(&id)
                        .filter(|node| !node.tombstone)
                        .cloned()
                    else {
                        continue;
                    };
                    staged.apply_logical_node_delete(&id)?;
                    control_projections.push(NodeMutationProjection {
                        kind: MutationKind::NodeDeleted,
                        node_id: id.clone(),
                        labels: prior.labels.clone(),
                        prior_was_tombstone: false,
                        had_prior: true,
                    });
                    if emit_hooks {
                        pending_events.push(PendingHookEvent {
                            kind: MutationKind::NodeDeleted,
                            id: id.clone(),
                            labels: prior.labels.clone(),
                            changed_props: Vec::new(),
                            scope_context: node_scope_context(
                                &self.hook_tenant,
                                Some(&prior),
                                None,
                            ),
                            node_transition: Some(MutationNodeTransition {
                                prior: Some(mutation_node_context(&self.hook_tenant, &prior)?),
                                current: None,
                            }),
                            edge_transition: None,
                            edge_context: None,
                        });
                    }
                    durable_mutations.push(RedCoreMutation::NodeDelete(id));
                }
                GraphMutation::NodeUpsert(node) => {
                    let prior_any = staged.nodes.get(&node.id).cloned();
                    let prior_was_tombstone = prior_any.as_ref().is_some_and(|node| node.tombstone);
                    let prior = prior_any.as_ref().filter(|node| !node.tombstone).cloned();
                    let (write, acquired_pin_receipts) =
                        staged.upsert_node_retaining_vector_pins(node)?;
                    vector_pins.extend(acquired_pin_receipts);
                    let record = staged.nodes.get(&write.id).cloned().ok_or_else(|| {
                        GraphStoreError::new("redcore_missing_write", "node write vanished")
                    })?;
                    control_projections.push(NodeMutationProjection {
                        kind: MutationKind::NodeUpserted,
                        node_id: write.id.clone(),
                        labels: record.labels.clone(),
                        prior_was_tombstone,
                        had_prior: prior.is_some(),
                    });
                    if emit_hooks {
                        pending_events.push(PendingHookEvent {
                            kind: MutationKind::NodeUpserted,
                            id: write.id.clone(),
                            labels: record.labels.clone(),
                            changed_props: changed_property_keys(
                                prior.as_ref().map(|node| &node.properties),
                                &record.properties,
                            ),
                            scope_context: node_scope_context(
                                &self.hook_tenant,
                                prior.as_ref(),
                                Some(&record),
                            ),
                            node_transition: Some(MutationNodeTransition {
                                prior: prior
                                    .as_ref()
                                    .map(|node| mutation_node_context(&self.hook_tenant, node))
                                    .transpose()?,
                                current: Some(mutation_node_context(&self.hook_tenant, &record)?),
                            }),
                            edge_transition: None,
                            edge_context: None,
                        });
                    }
                    durable_mutations.push(RedCoreMutation::NodeUpsert(record));
                    writes.push(write);
                }
                GraphMutation::EdgeUpsert(edge) => {
                    let prior_was_staged = staged_edge_upsert_ids.contains(&edge.id);
                    let prior = staged
                        .edges
                        .get(&edge.id)
                        .filter(|edge| !edge.tombstone)
                        .cloned();
                    let write = staged.upsert_edge(edge)?;
                    let record = staged.edges.get(&write.id).cloned().ok_or_else(|| {
                        GraphStoreError::new("redcore_missing_write", "edge write vanished")
                    })?;
                    if emit_hooks {
                        pending_events.push(PendingHookEvent {
                            kind: MutationKind::EdgeUpserted,
                            id: write.id.clone(),
                            labels: vec![record.edge_type.clone()],
                            changed_props: changed_property_keys(
                                prior.as_ref().map(|edge| &edge.properties),
                                &record.properties,
                            ),
                            scope_context: edge_scope_context(
                                &self.hook_tenant,
                                &staged,
                                prior.as_ref(),
                                Some(&record),
                            ),
                            node_transition: None,
                            edge_transition: Some(MutationEdgeTransition {
                                prior: prior
                                    .as_ref()
                                    .map(|edge| {
                                        let endpoint_store = if prior_was_staged {
                                            &staged
                                        } else {
                                            &self.store
                                        };
                                        mutation_edge_record_context(
                                            &self.hook_tenant,
                                            endpoint_store,
                                            edge,
                                        )
                                    })
                                    .transpose()?,
                                current: Some(mutation_edge_record_context(
                                    &self.hook_tenant,
                                    &staged,
                                    &record,
                                )?),
                            }),
                            edge_context: None,
                        });
                    }
                    durable_mutations.push(RedCoreMutation::EdgeUpsert(record));
                    staged_edge_upsert_ids.insert(write.id.clone());
                    writes.push(write);
                }
            }
        }

        self.append_control_event_mutations(
            &mut staged,
            &mut durable_mutations,
            &control_projections,
            graph_version_before,
            txn_id,
        )?;

        if durable_mutations.is_empty() {
            return Err(GraphStoreError::empty_transaction());
        }

        let graph_version = staged.stats_without_memory_estimate().version;
        let committed_mutations = durable_mutations.clone();
        let durable_mutation = RedCoreMutation::Batch(durable_mutations);
        let (prepublished_snapshot_txn_id, durable_last_txn_id) = match self.persist_before_publish(
            txn_id,
            graph_version,
            &staged,
            durable_mutation,
            &pending_events,
            metadata,
        ) {
            Ok(published) => published,
            Err(publish_error) => {
                return match vector_pins.cancel() {
                    Ok(()) => Err(publish_error),
                    Err(release_error) => Err(vector_pin_cleanup_error(
                        "failed graph publication",
                        publish_error,
                        release_error,
                    )),
                };
            }
        };

        self.store = staged;
        if let Some(projected) = projected_resident_edge_bytes {
            self.resident_edge_bytes_estimate = projected;
        }
        self.last_txn_id = durable_last_txn_id;
        if let Some(snapshot_txn_id) = prepublished_snapshot_txn_id {
            self.snapshot_txn_id = snapshot_txn_id;
        }
        // The catalog/root barrier above is the publication point. Releasing
        // before it would reopen the payload-vs-GC deletion window.
        vector_pins.finish_after_publication();
        self.record_committed_redcore_mutations(&committed_mutations);
        self.remove_superseded_cold_rollback_records(&committed_mutations);

        self.finish_mutation_log(&pending_events, txn_id);
        self.emit_hook_events(pending_events, txn_id, graph_version);

        Ok(GraphTransaction {
            txn_id,
            graph_version,
            writes,
        })
    }

    /// Commit a mutation batch only while one node property still matches the
    /// caller's observed value. A missing node or property compares as JSON
    /// null. The check and durable batch publish share this exclusive writer
    /// boundary, so a stale caller advances no journal root and no partial state.
    pub fn commit_batch_if_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<Option<GraphTransaction>> {
        let property = property.trim();
        if property.is_empty() {
            return Err(GraphStoreError::empty_field("node.property"));
        }
        let current = self
            .store
            .get_node_including_expired(id)
            .filter(|node| !node.tombstone)
            .and_then(|node| node.properties.as_object())
            .and_then(|properties| properties.get(property))
            .cloned()
            .unwrap_or(Value::Null);
        if &current != expected {
            return Ok(None);
        }
        self.commit_batch(batch).map(Some)
    }

    /// Validate all property preconditions against the current durable mirror,
    /// apply their replacements to those latest records, and publish the
    /// replacements plus the supplied mutations in one AOF-backed batch.
    pub fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        self.commit_conditional_batch_atomic_checked(conditions, batch, |_, _| Ok(()))
            .map(|batch| batch.is_some())
    }

    /// Prepare a conditional batch against the current mirror, validate the
    /// complete synthesized mutation set, and commit it without releasing the
    /// caller's exclusive store access. The committed batch is returned so
    /// projections can observe condition-owned node replacements exactly.
    pub fn commit_conditional_batch_atomic_checked<F>(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
        validate: F,
    ) -> GraphStoreResult<Option<GraphMutationBatch>>
    where
        F: FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    {
        let Some(batch) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(None);
        };
        let prepared = self.prepare_batch(batch)?;
        validate(self, prepared.batch())?;
        let committed = prepared.batch().clone();
        self.commit_prepared_batch(prepared)?;
        Ok(Some(committed))
    }

    fn can_commit_incrementally(&self, txn_id: u64, batch: &GraphMutationBatch) -> bool {
        let _ = txn_id;
        (self.current_pack.is_some() && self.payload_tree.is_some())
            && batch.mutations.iter().all(|mutation| {
                matches!(
                    mutation,
                    GraphMutation::NodeUpsert(_) | GraphMutation::EdgeUpsert(_)
                )
            })
    }

    fn commit_batch_incremental(
        &mut self,
        txn_id: u64,
        batch: GraphMutationBatch,
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<GraphTransaction> {
        let emit_hooks = true;
        let RedCoreCommitPlan {
            graph_version,
            durable_mutation,
            publish_mutations,
            mut prepared_vector_payloads,
            acquired_vector_pin_receipts,
            writes,
            pending_events,
        } = self.prepare_commit_plan(batch, emit_hooks)?;
        let mut vector_pins = VectorPayloadPinGuard::new(self.store.vector_payload_store_handle());
        vector_pins.extend(acquired_vector_pin_receipts);
        if let Err(publish_error) = self.persist_delta_before_publish(
            txn_id,
            graph_version,
            durable_mutation,
            &pending_events,
            metadata,
        ) {
            return match vector_pins.cancel() {
                Ok(()) => Err(publish_error),
                Err(release_error) => Err(vector_pin_cleanup_error(
                    "failed incremental graph publication",
                    publish_error,
                    release_error,
                )),
            };
        }

        for (mutation, prepared) in publish_mutations
            .iter()
            .zip(prepared_vector_payloads.iter_mut())
        {
            match mutation {
                RedCoreMutation::NodeUpsert(node) => {
                    self.store.apply_recovered_node_with_prepared(
                        node.clone(),
                        prepared.get_or_insert_with(BTreeMap::new),
                    )?;
                }
                RedCoreMutation::EdgeUpsert(edge) => {
                    if self.edges_fully_resident {
                        let prior_bytes = self
                            .store
                            .edges
                            .get(&edge.id)
                            .map(estimated_resident_edge_bytes)
                            .transpose()?
                            .unwrap_or_default();
                        self.store.apply_recovered_edge(edge.clone())?;
                        if self.resident_edge_budget_bytes.is_some() {
                            self.resident_edge_bytes_estimate = self
                                .resident_edge_bytes_estimate
                                .saturating_sub(prior_bytes)
                                .saturating_add(estimated_resident_edge_bytes(edge)?);
                        }
                    }
                }
                _ => unreachable!("incremental commit plan contains only ordinary upserts"),
            }
        }
        self.store.version = graph_version;
        self.last_txn_id = txn_id;
        if self.journal.is_some() {
            self.snapshot_txn_id = txn_id;
        }
        vector_pins.finish_after_publication();
        self.record_committed_redcore_mutations(&publish_mutations);
        self.remove_superseded_cold_rollback_records(&publish_mutations);
        self.finish_mutation_log(&pending_events, txn_id);
        self.emit_hook_events(pending_events, txn_id, graph_version);

        Ok(GraphTransaction {
            txn_id,
            graph_version,
            writes,
        })
    }

    fn prepare_commit_plan(
        &self,
        batch: GraphMutationBatch,
        emit_hooks: bool,
    ) -> GraphStoreResult<RedCoreCommitPlan> {
        let mut graph_version = self.store.version;
        let mut staged_nodes = BTreeMap::<String, NodeRecord>::new();
        let mut staged_edges = BTreeMap::<String, EdgeRecord>::new();
        let mut durable_mutations = Vec::with_capacity(batch.mutations.len());
        let mut publish_mutations = Vec::with_capacity(batch.mutations.len());
        let mut prepared_vector_payloads = Vec::with_capacity(batch.mutations.len());
        let mut acquired_vector_pins =
            VectorPayloadPinGuard::new(self.store.vector_payload_store_handle());
        let mut writes = Vec::with_capacity(batch.mutations.len());
        let mut pending_events = Vec::<PendingHookEvent>::new();
        let mut control_projections = Vec::<NodeMutationProjection>::new();
        let graph_version_before = self.store.version;
        let mut node_upserts = Vec::new();
        let mut edge_upserts = Vec::new();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => node_upserts.push(node),
                GraphMutation::EdgeUpsert(edge) => edge_upserts.push(edge),
                GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                    return Err(GraphStoreError::new(
                        "redcore_incremental_delete_unsupported",
                        "delete mutations require a staged graph transaction",
                    ));
                }
            }
        }

        for node in node_upserts {
            let prior_any = staged_nodes
                .get(&node.id)
                .or_else(|| self.store.nodes.get(&node.id))
                .cloned();
            let prior_was_tombstone = prior_any.as_ref().is_some_and(|node| node.tombstone);
            let prior = prior_any.as_ref().filter(|node| !node.tombstone).cloned();
            let (record, mut prepared) =
                self.prepare_node_upsert(node, &staged_nodes, graph_version + 1)?;
            acquired_vector_pins.extend(std::mem::take(&mut prepared.acquired_pin_receipts));
            graph_version = record.version;
            let checksum = record.checksum()?;
            let id = record.id.clone();
            control_projections.push(NodeMutationProjection {
                kind: MutationKind::NodeUpserted,
                node_id: id.clone(),
                labels: record.labels.clone(),
                prior_was_tombstone,
                had_prior: prior.is_some(),
            });
            if emit_hooks {
                pending_events.push(PendingHookEvent {
                    kind: MutationKind::NodeUpserted,
                    id: id.clone(),
                    labels: record.labels.clone(),
                    changed_props: changed_property_keys(
                        prior.as_ref().map(|node| &node.properties),
                        &record.properties,
                    ),
                    scope_context: node_scope_context(
                        &self.hook_tenant,
                        prior.as_ref(),
                        Some(&record),
                    ),
                    node_transition: Some(MutationNodeTransition {
                        prior: prior
                            .as_ref()
                            .map(|node| mutation_node_context(&self.hook_tenant, node))
                            .transpose()?,
                        current: Some(mutation_node_context(&self.hook_tenant, &record)?),
                    }),
                    edge_transition: None,
                    edge_context: None,
                });
            }
            let mutation = RedCoreMutation::NodeUpsert(record.clone());
            durable_mutations.push(mutation.clone());
            publish_mutations.push(mutation);
            prepared_vector_payloads.push(Some(prepared.packed_by_property));
            staged_nodes.insert(id.clone(), record);
            writes.push(GraphWriteResult {
                id,
                version: graph_version,
                checksum,
            });
        }

        for edge in edge_upserts {
            let prior_was_staged = staged_edges.contains_key(&edge.id);
            let record =
                self.prepare_edge_upsert(edge, &staged_nodes, &staged_edges, graph_version + 1)?;
            graph_version = record.version;
            let checksum = record.checksum()?;
            let id = record.id.clone();
            if emit_hooks {
                let prior = staged_edges
                    .get(&id)
                    .or_else(|| self.store.edges.get(&id))
                    .filter(|edge| !edge.tombstone)
                    .cloned();
                pending_events.push(PendingHookEvent {
                    kind: MutationKind::EdgeUpserted,
                    id: id.clone(),
                    labels: vec![record.edge_type.clone()],
                    changed_props: changed_property_keys(
                        prior.as_ref().map(|edge| &edge.properties),
                        &record.properties,
                    ),
                    scope_context: edge_scope_context_in_plan(
                        &self.hook_tenant,
                        &self.store,
                        &staged_nodes,
                        prior.as_ref(),
                        Some(&record),
                    ),
                    node_transition: None,
                    edge_transition: Some(MutationEdgeTransition {
                        prior: prior
                            .as_ref()
                            .map(|edge| {
                                if prior_was_staged {
                                    mutation_edge_record_context_in_plan(
                                        &self.hook_tenant,
                                        &self.store,
                                        &staged_nodes,
                                        edge,
                                    )
                                } else {
                                    mutation_edge_record_context(
                                        &self.hook_tenant,
                                        &self.store,
                                        edge,
                                    )
                                }
                            })
                            .transpose()?,
                        current: Some(mutation_edge_record_context_in_plan(
                            &self.hook_tenant,
                            &self.store,
                            &staged_nodes,
                            &record,
                        )?),
                    }),
                    edge_context: None,
                });
            }
            let mutation = RedCoreMutation::EdgeUpsert(record.clone());
            durable_mutations.push(mutation.clone());
            publish_mutations.push(mutation);
            prepared_vector_payloads.push(None);
            staged_edges.insert(id.clone(), record);
            writes.push(GraphWriteResult {
                id,
                version: graph_version,
                checksum,
            });
        }

        let committed_at_ms = now_ms().max(0) as u64;
        let tenant = if self.hook_tenant.trim().is_empty() {
            "default"
        } else {
            self.hook_tenant.as_str()
        };
        let actor = (!self.hook_actor.trim().is_empty()).then_some(self.hook_actor.as_str());
        let principal =
            (!self.hook_principal.trim().is_empty()).then_some(self.hook_principal.as_str());
        let control_events = control_events_from_projections(
            tenant,
            actor,
            principal,
            graph_version_before,
            graph_version,
            committed_at_ms,
            self.last_txn_id + 1,
            &control_projections,
        );
        for event in control_events {
            let mut node = event.to_node();
            graph_version += 1;
            node.version = graph_version;
            let mutation = RedCoreMutation::NodeUpsert(node.clone());
            durable_mutations.push(mutation.clone());
            publish_mutations.push(mutation);
            prepared_vector_payloads.push(None);
            staged_nodes.insert(node.id.clone(), node);
        }

        let durable_mutation = if durable_mutations.len() == 1 {
            durable_mutations
                .pop()
                .expect("single durable mutation exists")
        } else {
            RedCoreMutation::Batch(durable_mutations)
        };

        Ok(RedCoreCommitPlan {
            graph_version,
            durable_mutation,
            publish_mutations,
            prepared_vector_payloads,
            acquired_vector_pin_receipts: acquired_vector_pins.into_receipts(),
            writes,
            pending_events,
        })
    }

    fn prepare_node_upsert(
        &self,
        mut node: NodeRecord,
        staged_nodes: &BTreeMap<String, NodeRecord>,
        version: u64,
    ) -> GraphStoreResult<(NodeRecord, PreparedVectorProperties)> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        node.labels = normalize_labels(node.labels);
        let mut prepared = self.store.prepare_designated_vector_properties(&mut node)?;
        let mut acquired_pins =
            VectorPayloadPinGuard::new(self.store.vector_payload_store_handle());
        acquired_pins.extend(std::mem::take(&mut prepared.acquired_pin_receipts));
        let parent_hash = match staged_nodes
            .get(&node.id)
            .or_else(|| self.store.nodes.get(&node.id))
        {
            Some(existing) => Some(match &existing.content_hash {
                Some(hash) => hash.clone(),
                None => existing.checksum()?,
            }),
            None => None,
        };
        node.version = version;
        let content_hash = node.checksum()?;
        if node.parent_hashes.is_empty() {
            if let Some(parent_hash) = parent_hash.filter(|parent| parent != &content_hash) {
                node.parent_hashes.push(parent_hash);
            }
        }
        node.content_hash = Some(content_hash);
        prepared.acquired_pin_receipts = acquired_pins.into_receipts();
        Ok((node, prepared))
    }

    fn prepare_edge_upsert(
        &self,
        mut edge: EdgeRecord,
        staged_nodes: &BTreeMap<String, NodeRecord>,
        staged_edges: &BTreeMap<String, EdgeRecord>,
        version: u64,
    ) -> GraphStoreResult<EdgeRecord> {
        validate_edge_shape(&edge)?;
        self.require_live_endpoint_in_plan(&edge, "from", &edge.from_id, staged_nodes)?;
        self.require_live_endpoint_in_plan(&edge, "to", &edge.to_id, staged_nodes)?;
        let prior_link = staged_edges
            .get(&edge.id)
            .or_else(|| self.store.edges.get(&edge.id));
        crate::graph::linked_to::stamp_first_version(&mut edge, prior_link, version)?;
        let parent_hash = match staged_edges
            .get(&edge.id)
            .or_else(|| self.store.edges.get(&edge.id))
        {
            Some(existing) => Some(match &existing.content_hash {
                Some(hash) => hash.clone(),
                None => existing.checksum()?,
            }),
            None => None,
        };
        edge.version = version;
        let content_hash = edge.checksum()?;
        if edge.parent_hashes.is_empty() {
            if let Some(parent_hash) = parent_hash.filter(|parent| parent != &content_hash) {
                edge.parent_hashes.push(parent_hash);
            }
        }
        edge.content_hash = Some(content_hash);
        Ok(edge)
    }

    fn require_live_endpoint_in_plan(
        &self,
        edge: &EdgeRecord,
        endpoint: &str,
        node_id: &str,
        staged_nodes: &BTreeMap<String, NodeRecord>,
    ) -> GraphStoreResult<()> {
        let node = staged_nodes
            .get(node_id)
            .or_else(|| self.store.nodes.get(node_id))
            .ok_or_else(|| GraphStoreError::missing_endpoint(&edge.id, endpoint, node_id))?;
        if node.tombstone {
            return Err(GraphStoreError::tombstoned_endpoint(
                &edge.id, endpoint, node_id,
            ));
        }
        Ok(())
    }

    /// Atomically install a vector designation and publish a mixed node/edge
    /// batch, provided the pinned input graph version is still current.
    pub fn commit_conditional_batch_with_vector_designation_if_version(
        &mut self,
        expected_graph_version: u64,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        label: &str,
        property_name: &str,
        dimension: usize,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        let Some(prepared) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(false);
        };
        self.commit_batch_with_vector_designation_if_version(
            expected_graph_version,
            label,
            property_name,
            dimension,
            prepared,
        )?;
        Ok(true)
    }

    /// Atomically install a vector designation and publish a mixed node/edge
    /// batch, provided the pinned input graph version is still current.
    pub fn commit_batch_with_vector_designation_if_version(
        &mut self,
        expected_graph_version: u64,
        label: &str,
        property_name: &str,
        dimension: usize,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<GraphTransaction> {
        let actual_graph_version = self.store.version;
        if actual_graph_version != expected_graph_version {
            return Err(GraphStoreError::new(
                "graph_version_conflict",
                format!(
                    "expected graph version {expected_graph_version}, current version is {actual_graph_version}"
                ),
            ));
        }
        self.commit_batch_internal(
            batch,
            Some(VectorDesignation {
                label: label.to_string(),
                property: property_name.to_string(),
                dimension,
            }),
            false,
            None,
        )
    }

    pub fn commit_batch_if_version(
        &mut self,
        expected_graph_version: u64,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<GraphTransaction> {
        self.commit_batch_if_version_internal(expected_graph_version, batch, None)
    }

    pub fn commit_batch_if_version_with_metadata(
        &mut self,
        expected_graph_version: u64,
        batch: GraphMutationBatch,
        metadata: GraphCommitMetadata,
    ) -> GraphStoreResult<GraphTransaction> {
        self.commit_batch_if_version_internal(expected_graph_version, batch, Some(&metadata))
    }

    pub fn commit_batch_if_version_checked(
        &mut self,
        expected_graph_version: u64,
        batch: GraphMutationBatch,
        metadata: Option<GraphCommitMetadata>,
        check: impl FnOnce(&Self, &GraphMutationBatch) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<(GraphTransaction, GraphMutationBatch)> {
        if self.store.version != expected_graph_version {
            return Err(GraphStoreError::new(
                "graph_version_conflict",
                format!(
                    "expected graph version {expected_graph_version}, current version is {}",
                    self.store.version
                ),
            ));
        }
        let prepared = self.prepare_batch(batch)?;
        check(self, prepared.batch())?;
        let committed = prepared.batch().clone();
        let transaction =
            self.commit_prepared_batch_internal(prepared, None, false, metadata.as_ref())?;
        Ok((transaction, committed))
    }

    fn commit_batch_if_version_internal(
        &mut self,
        expected_graph_version: u64,
        batch: GraphMutationBatch,
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<GraphTransaction> {
        let actual_graph_version = self.store.version;
        if actual_graph_version != expected_graph_version {
            return Err(GraphStoreError::new(
                "graph_version_conflict",
                format!(
                    "expected graph version {expected_graph_version}, current version is {actual_graph_version}"
                ),
            ));
        }
        self.commit_batch_internal(batch, None, false, metadata)
    }

    pub fn get_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        Ok(self.store.get_node(id).cloned())
    }

    pub fn node_ids_for_string_property_exact(
        &self,
        key: &str,
        raw: &str,
    ) -> GraphStoreResult<Vec<String>> {
        Ok(self.store.node_ids_for_string_property_exact(key, raw))
    }

    pub fn node_ids_for_string_property_prefix(
        &self,
        key: &str,
        raw_prefix: &str,
    ) -> GraphStoreResult<Vec<String>> {
        Ok(self
            .store
            .node_ids_for_string_property_prefix(key, raw_prefix))
    }

    pub fn node_ids_for_label_and_string_property_prefix(
        &self,
        label: &str,
        key: &str,
        raw_prefix: &str,
    ) -> GraphStoreResult<Vec<String>> {
        Ok(self
            .store
            .node_ids_for_label_and_string_property_prefix(label, key, raw_prefix))
    }

    // ---- TTL primitive (decorator pass-throughs) -----------------------
    //
    // Read paths delegate directly: the TTL filter inside
    // `InMemoryGraphStore::get_node` (etc.) fires through the wrapper
    // automatically, and rebuild_indexes on startup repopulates ttl_index
    // because RedCore replays all NodeUpsert mutations through
    // self.store.upsert_node during recovery, which calls add_node_indexes.
    //
    // The write path (set_node_ttl) reads the existing node, applies the
    // TTL property via the shared `node_with_ttl_property` helper, then
    // routes the result through self.upsert_node so the change journals
    // as a normal node upsert. No new RedCoreMutation variant is required;
    // root-based hydration reconstructs the TTL state automatically.
    //
    // Expiration purges use the same atomic mixed transaction path as explicit
    // deletes, including durable transition events.

    /// Set or clear the TTL on an existing node. `expires_at_ms = Some(t)`
    /// sets/extends; `None` removes TTL (node becomes permanent).
    /// Routes through `commit_batch` and the journal root barrier for durability.
    pub fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult> {
        let existing = self
            .store
            .get_node(id)
            .filter(|node| !node.tombstone)
            .cloned()
            .ok_or_else(|| {
                GraphStoreError::new("missing_graph_node", format!("node {id} does not exist"))
            })?;
        let updated = node_with_ttl_property(existing, expires_at_ms);
        self.upsert_node(updated)
    }

    /// Read a node regardless of TTL window. Audit/forensics path.
    /// Returns None only when the node was never inserted or has been
    /// tombstoned via a future delete op.
    pub fn get_node_including_expired(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        Ok(self.store.get_node_including_expired(id).cloned())
    }

    /// Return nodes whose `_ttl_expires_at_ms <= ts_ms`, ordered by
    /// expiration. Used by callers that need "what's about to time out"
    /// visibility (mentions queue, presence keys, coordinator agents).
    pub fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord> {
        self.store.nodes_expiring_before(ts_ms, limit)
    }

    /// Current count of TTL-bearing live nodes. Used for diagnostics.
    pub fn ttl_active_count(&self) -> usize {
        self.store.ttl_active_count()
    }

    /// Sweep expired nodes from storage durably. Walks the inner store's
    /// expiration index and publishes the resulting tombstones under one
    /// journal root so the purge survives process restart.
    ///
    /// Returns the count of nodes removed. Returns 0 with no root advance
    /// when nothing is expired (cheap polling for the background sweep).
    pub fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize> {
        let cutoff_ms = now_ms();
        if !self.store.has_expired_nodes_before(cutoff_ms) {
            return Ok(0);
        }

        // Discover the expired ids on a clone, then let commit_batch perform
        // the authoritative staged delete and persistence gate.
        let mut staged = self.store.clone();
        let purged_ids = staged.drain_expired_node_ids_before(cutoff_ms);
        if purged_ids.is_empty() {
            return Ok(0);
        }
        let count = purged_ids.len();
        self.commit_batch(GraphMutationBatch::new(
            purged_ids.into_iter().map(GraphMutation::NodeDelete),
        ))?;
        Ok(count)
    }

    pub fn get_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        self.require_fully_resident_edges("edge lookup")?;
        Ok(self.store.get_edge(id).cloned())
    }

    pub fn live_edges_snapshot(&self) -> Vec<EdgeRecord> {
        self.store.live_edges_snapshot()
    }

    pub fn query_nodes(&self, query: NodeQuery) -> GraphStoreResult<Vec<NodeRecord>> {
        Ok(self.store.query_nodes(query))
    }

    pub fn nodes_for_label_after_id(
        &self,
        label: &str,
        after_id: Option<&str>,
        limit: usize,
    ) -> GraphStoreResult<Vec<NodeRecord>> {
        Ok(self.store.nodes_for_label_after_id(label, after_id, limit))
    }

    pub fn memory_documents_by_updated_at(
        &self,
        query: MemoryDocumentQuery,
    ) -> GraphStoreResult<Vec<NodeRecord>> {
        self.store.try_memory_documents_by_updated_at(query)
    }

    pub fn neighbors(&self, query: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>> {
        self.require_fully_resident_edges("neighbor traversal")?;
        Ok(self.store.neighbors(query))
    }

    pub fn stats(&self) -> GraphStoreResult<GraphStats> {
        let mut stats = self.store.stats();
        if !self.edges_fully_resident {
            stats.nodes_total = self.durable_nodes_total;
            stats.edges_total = self.durable_edges_total;
        }
        Ok(stats)
    }

    /// Stats without the O(nodes+edges) resident-memory estimate. Callers that
    /// only need counts/version (diagnostics, hot-path metrics) use this to
    /// avoid walking the whole graph under the writer lock.
    pub fn stats_without_memory_estimate(&self) -> GraphStoreResult<GraphStats> {
        let mut stats = self.store.stats_without_memory_estimate();
        if !self.edges_fully_resident {
            stats.nodes_total = self.durable_nodes_total;
            stats.edges_total = self.durable_edges_total;
        }
        Ok(stats)
    }

    pub fn memory_attribution(&self) -> GraphMemoryAttribution {
        self.store.memory_attribution()
    }

    /// Conservative post-commit footprint used by quota enforcement. Unlike a
    /// projected snapshot, this walks records in place and allocates no buffer
    /// proportional to tenant size.
    pub fn projected_memory_bytes_upper_bound(&self, batch: &GraphMutationBatch) -> usize {
        self.store.projected_memory_bytes_upper_bound(batch)
    }

    pub fn verify(&self) -> GraphStoreResult<VerifyReport> {
        self.require_fully_resident_edges("graph verification")?;
        Ok(self.store.verify())
    }

    /// Whole-graph logical-tree parity audit, on demand.
    ///
    /// Collects every entry of the published logical tree, rebuilds the tree from
    /// scratch, and compares the rebuild with the tree this store published; then
    /// cross-checks the tree's counts against the manifest that names it. This is
    /// the oracle that used to be a per-commit publish precondition, reached here
    /// instead of on the write path, together with the recovery checkpoint.
    ///
    /// It is deliberately not folded into [`Self::verify`]: that surface is
    /// reachable per request (the admin and MCP verify tools, the index-status
    /// payload), and an O(entries) audit with a hash verification per chunk read
    /// belongs on a boundary a caller asks for, not on a status read.
    ///
    /// `Ok(None)` means nothing has been published yet, so there is no tree to
    /// audit. Divergence is an `Err` with code
    /// `graph_prolly_validation_divergence`; a tree whose counts disagree with
    /// its manifest is `graph_tree_entry_count_mismatch`.
    pub fn verify_tree_parity(&self) -> GraphStoreResult<Option<GraphTreeParityAudit>> {
        let Some(pack) = self.current_pack.as_ref() else {
            return Ok(None);
        };
        let audit =
            audit_published_prolly_parity(self.chunk_store.get().as_chunk_store(), &pack.tree)?;
        verify_published_tree_counts(&audit, &pack.manifest)?;
        Ok(Some(audit))
    }

    pub fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        self.require_fully_resident_edges("index rebuild")?;
        self.store.rebuild_indexes()
    }

    pub fn labels(&self) -> GraphStoreResult<Vec<String>> {
        Ok(self.store.labels())
    }

    pub fn edge_types(&self) -> GraphStoreResult<Vec<String>> {
        self.require_fully_resident_edges("edge-type enumeration")?;
        Ok(self.store.edge_types())
    }

    pub fn property_keys(&self) -> GraphStoreResult<Vec<String>> {
        Ok(self.store.property_keys())
    }

    pub fn designate_vector_property(
        &mut self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> GraphStoreResult<()> {
        if self
            .store
            .vector_designation_is_current_and_migrated(label, property_name, dimension)
        {
            return Ok(());
        }
        let designation = VectorDesignation {
            label: label.to_string(),
            property: property_name.to_string(),
            dimension,
        };
        let txn_id = self.last_txn_id + 1;
        let mut staged = self.store.clone();
        let migration_ids = staged.legacy_vector_node_ids_for_designation(label, property_name);
        let acquired_pin_receipts =
            staged.designate_vector_property_retaining_pins(label, property_name, dimension)?;
        let mut vector_pins = VectorPayloadPinGuard::new(staged.vector_payload_store_handle());
        vector_pins.extend(acquired_pin_receipts);
        let mut durable_mutations = vec![RedCoreMutation::VectorDesignation(designation)];
        for id in migration_ids {
            durable_mutations.push(RedCoreMutation::NodeUpsert(
                staged.nodes.get(&id).cloned().ok_or_else(|| {
                    GraphStoreError::new(
                        "vector_migration_node_missing",
                        format!("migrated vector node {id} disappeared"),
                    )
                })?,
            ));
        }
        let graph_version = staged.stats_without_memory_estimate().version;
        let durable_mutation = RedCoreMutation::Batch(durable_mutations);
        let (prepublished_snapshot_txn_id, durable_last_txn_id) = match self.persist_before_publish(
            txn_id,
            graph_version,
            &staged,
            durable_mutation,
            &[],
            None,
        ) {
            Ok(published) => published,
            Err(publish_error) => {
                return match vector_pins.cancel() {
                    Ok(()) => Err(publish_error),
                    Err(release_error) => Err(vector_pin_cleanup_error(
                        "failed vector-designation publication",
                        publish_error,
                        release_error,
                    )),
                };
            }
        };
        self.store = staged;
        self.last_txn_id = durable_last_txn_id;
        if let Some(snapshot_txn_id) = prepublished_snapshot_txn_id {
            self.snapshot_txn_id = snapshot_txn_id;
        }
        vector_pins.finish_after_publication();
        Ok(())
    }

    pub fn vector_search(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        Ok(self
            .vector_search_with_receipt(label, property_name, query, k)?
            .hits)
    }

    pub fn vector_search_with_receipt(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        self.store
            .vector_search_with_receipt(label, property_name, query, k)
    }

    /// Bounded point-directory search over a caller-admitted blocking posting.
    pub fn vector_search_candidates_with_receipt(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        candidates: &BTreeSet<String>,
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        self.store
            .vector_search_candidates_with_receipt(label, property, query, candidates, k)
    }

    pub fn hybrid_search(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
        graph_seeds: &[String],
        max_hops: usize,
        alpha: f32,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        self.require_fully_resident_edges("hybrid graph search")?;
        self.store
            .hybrid_search(label, property_name, query, k, graph_seeds, max_hops, alpha)
    }

    pub fn hybrid_search_with_config(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
        graph_seeds: &[String],
        max_hops: usize,
        config: &HybridScoringConfig,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        self.require_fully_resident_edges("hybrid graph search")?;
        self.store.hybrid_search_with_config(
            label,
            property_name,
            query,
            k,
            graph_seeds,
            max_hops,
            config,
        )
    }

    pub fn vector_designations(&self) -> Vec<VectorDesignation> {
        self.store.vector_designations()
    }

    pub fn clear_vector_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_vector_designation",
                "label and property_name must be non-empty",
            ));
        }
        if !self
            .store
            .vector_designations()
            .iter()
            .any(|designation| designation.label == label && designation.property == property_name)
        {
            return Ok(());
        }
        let designation = OrderedDesignation {
            label: label.to_string(),
            property: property_name.to_string(),
        };
        let txn_id = self.last_txn_id + 1;
        let mut staged = self.store.clone();
        staged.clear_vector_property(label, property_name)?;
        let graph_version = staged.stats_without_memory_estimate().version;
        let (prepublished_snapshot_txn_id, durable_last_txn_id) = self.persist_before_publish(
            txn_id,
            graph_version,
            &staged,
            RedCoreMutation::VectorDesignationClear(designation),
            &[],
            None,
        )?;
        self.store = staged;
        self.last_txn_id = durable_last_txn_id;
        if let Some(snapshot_txn_id) = prepublished_snapshot_txn_id {
            self.snapshot_txn_id = snapshot_txn_id;
        }
        Ok(())
    }

    /// Returns immutable exact-search cardinality for one designated index.
    pub fn vector_index_shape(&self, label: &str, property_name: &str) -> Option<VectorIndexShape> {
        self.store.vector_index_shape(label, property_name)
    }

    pub fn designate_ordered_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        let designation = OrderedDesignation {
            label: label.to_string(),
            property: property_name.to_string(),
        };
        let txn_id = self.last_txn_id + 1;
        let mut staged = self.store.clone();
        staged.designate_ordered_property(label, property_name)?;
        let graph_version = staged.stats_without_memory_estimate().version;
        let (prepublished_snapshot_txn_id, durable_last_txn_id) = self.persist_before_publish(
            txn_id,
            graph_version,
            &staged,
            RedCoreMutation::OrderedDesignation(designation),
            &[],
            None,
        )?;
        self.store = staged;
        self.last_txn_id = durable_last_txn_id;
        if let Some(snapshot_txn_id) = prepublished_snapshot_txn_id {
            self.snapshot_txn_id = snapshot_txn_id;
        }
        Ok(())
    }

    pub fn ordered_designations(&self) -> Vec<OrderedDesignation> {
        self.store.ordered_designations()
    }

    pub fn clear_ordered_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_ordered_designation",
                "label and property_name must be non-empty",
            ));
        }
        if !self
            .store
            .ordered_designations()
            .iter()
            .any(|designation| designation.label == label && designation.property == property_name)
        {
            return Ok(());
        }
        let designation = OrderedDesignation {
            label: label.to_string(),
            property: property_name.to_string(),
        };
        let txn_id = self.last_txn_id + 1;
        let mut staged = self.store.clone();
        staged.clear_ordered_property(label, property_name)?;
        let graph_version = staged.stats_without_memory_estimate().version;
        let (prepublished_snapshot_txn_id, durable_last_txn_id) = self.persist_before_publish(
            txn_id,
            graph_version,
            &staged,
            RedCoreMutation::OrderedDesignationClear(designation),
            &[],
            None,
        )?;
        self.store = staged;
        self.last_txn_id = durable_last_txn_id;
        if let Some(snapshot_txn_id) = prepublished_snapshot_txn_id {
            self.snapshot_txn_id = snapshot_txn_id;
        }
        Ok(())
    }

    pub fn ordered_range_by_score(
        &self,
        label: &str,
        property_name: &str,
        min: f64,
        max: f64,
        limit: Option<usize>,
    ) -> GraphStoreResult<Vec<(String, f64)>> {
        self.store
            .ordered_range_by_score(label, property_name, min, max, limit)
    }

    pub fn ordered_score(&self, label: &str, property_name: &str, node_id: &str) -> Option<f64> {
        self.store.ordered_score(label, property_name, node_id)
    }

    pub fn transient_ordered_zadd(
        &mut self,
        index_name: &str,
        member: impl Into<Vec<u8>>,
        score: f64,
    ) -> GraphStoreResult<bool> {
        self.transient_ordered_indexes
            .entry(index_name.to_string())
            .or_insert_with(OrderedIndex::transient)
            .zadd(member, score)
    }

    pub fn transient_ordered_zpop_max(&mut self, index_name: &str) -> Option<(Vec<u8>, f64)> {
        self.transient_ordered_indexes
            .get_mut(index_name)
            .and_then(OrderedIndex::zpop_max)
    }

    pub fn transient_ordered_zpop_min(&mut self, index_name: &str) -> Option<(Vec<u8>, f64)> {
        self.transient_ordered_indexes
            .get_mut(index_name)
            .and_then(OrderedIndex::zpop_min)
    }

    pub fn transient_ordered_zcard(&self, index_name: &str) -> usize {
        self.transient_ordered_indexes
            .get(index_name)
            .map(OrderedIndex::zcard)
            .unwrap_or_default()
    }

    pub fn transient_ordered_zrem(&mut self, index_name: &str, member: &[u8]) -> bool {
        self.transient_ordered_indexes
            .get_mut(index_name)
            .map(|index| index.zrem(member))
            .unwrap_or(false)
    }

    pub fn epistemic_neighbors(
        &self,
        node_id: &str,
        epistemic_types: Option<&[EpistemicType]>,
        min_confidence: Option<f64>,
        max_depth: Option<usize>,
    ) -> Vec<(EdgeRecord, NodeRecord)> {
        self.store
            .epistemic_neighbors(node_id, epistemic_types, min_confidence, max_depth)
    }

    pub fn snapshot_now(&mut self) -> GraphStoreResult<()> {
        let txn_id = self.last_txn_id;
        let graph_version = self.store.version;
        let Some(journal) = self.journal.clone() else {
            return Ok(());
        };
        match journal.read_root()? {
            Some(catalog_root) => {
                // Republish the exact current catalog under a new commit. An empty
                // persist_journal_commit path can still rewrite the redcore state
                // chunk and advance the catalog address; snapshot barriers must
                // keep the published catalog bytes stable.
                journal.commit(
                    catalog_root,
                    CommitMeta {
                        parents: Vec::new(),
                        author: "rustyred".to_string(),
                        message: format!("republish RedCore catalog at transaction {txn_id}"),
                        timestamp_unix_ms: unix_ms(),
                        graph_version,
                    },
                )?;
            }
            None => {
                let staged = self.store.clone();
                self.persist_journal_commit(
                    txn_id,
                    graph_version,
                    Some(&staged),
                    &RedCoreMutation::Batch(Vec::new()),
                    &[],
                    None,
                )?;
            }
        }
        self.snapshot_txn_id = txn_id;
        self.last_fsync = Some(SystemTime::now());
        self.aof_size_bytes = journal_file_len(journal.path())?;
        Ok(())
    }

    fn recover(&mut self) -> GraphStoreResult<()> {
        if !self.recovers_from_catalog() {
            return self.recover_legacy();
        }
        // Journal-backed and injected stores recover through the same catalog
        // reads. Only the legacy bootstrap below needs the concrete journal.
        let backend = self.unified_chunk_store();
        let Some(catalog_address) = backend.read_root()? else {
            if !self.has_legacy_redcore_artifacts() {
                self.aof_size_bytes = self.durable_log_len()?;
                return Ok(());
            }

            self.recover_legacy()?;
            let journal = self.journal.clone().ok_or_else(|| {
                GraphStoreError::new(
                    "redcore_legacy_bootstrap_requires_journal",
                    "legacy RedCore artifacts can only be published into a durable journal",
                )
            })?;
            self.publish_legacy_journal_bootstrap(&journal)?;
            self.readmit_legacy_cold_records()?;
            self.move_legacy_redcore_artifacts_to_backup()?;
            self.aof_size_bytes = self.durable_log_len()?;
            return Ok(());
        };

        let catalog = validate_catalog_root(backend.as_chunk_store(), catalog_address)?;
        let graph_root = catalog.graph_root()?;
        let state_address = match catalog.get(ROOT_NAME_REDCORE) {
            Some(state_address) => state_address,
            None => {
                // Another lens may publish its named root before RedCore's
                // first graph transaction. That is a valid empty RedCore only
                // when the catalog's graph root is the exact canonical empty
                // tree; a non-empty logical head without recovery state still
                // fails closed.
                let empty_graph = canonical_leaf_preimage(&[])?;
                let empty_graph_root = Hash::of(&empty_graph);
                let stored_empty_graph = backend.get(graph_root)?;
                if graph_root == empty_graph_root
                    && stored_empty_graph.as_deref() == Some(empty_graph.as_slice())
                {
                    self.aof_size_bytes = self.durable_log_len()?;
                    return Ok(());
                }
                return Err(GraphStoreError::new(
                    "redcore_journal_state_root_missing",
                    format!("root catalog has no {ROOT_NAME_REDCORE:?} entry"),
                ));
            }
        };
        let state_bytes = backend.get(state_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "redcore_journal_state_missing",
                format!("durable RedCore state chunk {state_address} is missing"),
            )
        })?;
        let state = RedCoreJournalState::from_bytes(&state_bytes)?;
        let journal_bytes = self.durable_log_len()?;
        self.recovery_profile.capture(
            "journal_replay",
            None::<String>,
            journal_bytes,
            state.nodes_total,
            state.edges_total,
        );
        if state.logical_tree.root != graph_root {
            return Err(GraphStoreError::new(
                "redcore_journal_graph_root_mismatch",
                format!(
                    "catalog graph root {graph_root} does not match RedCore state root {}",
                    state.logical_tree.root
                ),
            ));
        }

        let hydrated = hydrate_in_memory_graph_from_payload_index_profiled(
            backend.as_chunk_store(),
            &state.logical_tree,
            &state.payload_tree,
            state.graph_version,
            journal_bytes,
            self.resident_edge_budget_bytes,
            &mut self.recovery_profile,
        )?;
        if hydrated.live_nodes != state.nodes_total
            || (hydrated.edges_fully_resident && hydrated.live_edges != state.edges_total)
        {
            return Err(GraphStoreError::new(
                "redcore_journal_live_count_mismatch",
                format!(
                    "hydrated {} nodes and {} edges, state declares {} nodes and {} edges",
                    hydrated.live_nodes, hydrated.live_edges, state.nodes_total, state.edges_total
                ),
            ));
        }
        let historical_node_ids = match state.historical_node_tree.as_ref() {
            Some(tree) => hydrate_historical_node_registry(backend.as_chunk_store(), tree)?,
            None => state.legacy_historical_node_ids.clone(),
        };
        let mut recovered = hydrated.store;
        recovered.historical_node_ids = historical_node_ids;
        recovered.historical_node_history_complete = state.historical_node_history_complete;
        recovered.vector_payload_store = self.store.vector_payload_store.clone();
        for designation in &state.vector_designations {
            recovered.install_vector_designation(
                &designation.label,
                &designation.property,
                designation.dimension,
            )?;
            self.recovery_profile.capture(
                "index_build",
                Some(format!(
                    "vector:{}:{}:{}",
                    designation.label, designation.property, designation.dimension
                )),
                journal_bytes,
                hydrated.live_nodes,
                hydrated.live_edges,
            );
        }
        for designation in &state.ordered_designations {
            recovered.designate_ordered_property(&designation.label, &designation.property)?;
            self.recovery_profile.capture(
                "index_build",
                Some(format!(
                    "ordered:{}:{}",
                    designation.label, designation.property
                )),
                journal_bytes,
                hydrated.live_nodes,
                hydrated.live_edges,
            );
        }
        self.recovery_profile.capture(
            "index_build",
            Some("vector_acceleration_finalize"),
            journal_bytes,
            hydrated.live_nodes,
            hydrated.live_edges,
        );
        self.current_pack = Some(recovered_graph_pack_descriptor(
            state.logical_tree.clone(),
            state.pack_graph_version,
            state.nodes_total,
            state.edges_total,
            state.logical_commit,
        )?);
        self.history_root = state.history_root;
        if let Some(address) = self.history_root {
            let history = load_history_root(backend.as_chunk_store(), address)?;
            let pin = PinnedGraphHistory::new(
                self.chunk_store.get(),
                Some(address),
                GraphRevisionRoots {
                    transaction: state.txn_id,
                    graph_version: state.pack_graph_version,
                    logical_commit: state.logical_commit,
                    logical_tree: state.logical_tree.clone(),
                    payload_tree: state.payload_tree.clone(),
                    historical_node_tree: state.historical_node_tree.clone(),
                    historical_node_history_complete: state.historical_node_history_complete,
                    vector_designations: state.vector_designations.clone(),
                    ordered_designations: state.ordered_designations.clone(),
                },
            )?;
            let header = pin.entry(history.head_commit)?.commit;
            if header.commit_hash != state.logical_commit
                || header.tree_hash != state.logical_tree.root
            {
                return Err(GraphStoreError::new(
                    "graph_history_integrity",
                    "recovered graph and retained header differ",
                ));
            }
            self.current_pack
                .as_mut()
                .expect("recovery installed descriptor")
                .commit = header;
        }
        self.payload_tree = Some(state.payload_tree);
        self.historical_node_tree = state.historical_node_tree;
        // The historical registry above was installed wholesale, so re-derive
        // the maintained committed footprint from the graph before publishing
        // it as the live store. Recovery may spend O(graph) here; the per-write
        // quota path must not.
        recovered.reset_committed_footprint_from_recompute();
        self.store = recovered;
        self.durable_nodes_total = state.nodes_total;
        self.durable_edges_total = state.edges_total;
        self.resident_edge_bytes_estimate = hydrated.resident_edge_bytes_estimate;
        self.edges_fully_resident = hydrated.edges_fully_resident;
        self.last_txn_id = state.txn_id;
        self.snapshot_txn_id = state.txn_id;
        self.sync_root = state.logical_tree.root;
        self.mutation_log = state.mutation_log;
        self.sync_parent_by_root = self.mutation_log.parent_map(self.sync_root);
        self.recovered_frames = 1;
        self.aof_size_bytes = self.durable_log_len()?;
        self.move_legacy_redcore_artifacts_to_backup()?;

        // The parity oracle's boundary. A recovered head is audited once, here,
        // before this store serves a single read or write: recovery already walks
        // the whole graph to hydrate it, so one more whole-graph walk costs a
        // constant factor on a path that is O(graph) by construction. A diverged
        // or unreachable published tree fails the open closed, so drift is caught
        // before it can be handed to a caller; `RUSTYRED_PROLLY_VALIDATE=off` is
        // the only way to skip it, and an explicit `sampled` rate filters it
        // deterministically per published root.
        if let Some(pack) = self.current_pack.as_ref() {
            if prolly_validation_selected(pack.tree.root, pack.manifest.graph_version)? {
                let audit = audit_published_prolly_parity(
                    self.chunk_store.get().as_chunk_store(),
                    &pack.tree,
                )?;
                verify_published_tree_counts(&audit, &pack.manifest)?;
                self.recovery_profile.capture(
                    "prolly_parity_audit",
                    None::<String>,
                    journal_bytes,
                    audit.live_nodes,
                    audit.live_edges,
                );
            }
        }
        Ok(())
    }

    fn recover_legacy(&mut self) -> GraphStoreResult<()> {
        let Some(data_dir) = self.data_dir.clone() else {
            return Ok(());
        };
        if let Some(manifest) = read_manifest(&data_dir)? {
            if !manifest_version_compatible(manifest.version) {
                return Err(GraphStoreError::new(
                    "redcore_format_too_new",
                    format!(
                        "On-disk RedCore manifest version is {}, this build supports up to {}. \
                         Upgrade the binary or run rustyred-thg-upgrade-format.",
                        manifest.version, CURRENT_FORMAT_VERSION
                    ),
                ));
            }
        }
        let mut recovered_from_current_snapshot = false;
        if let Some((envelope, is_current)) = read_latest_valid_snapshot(&data_dir)? {
            recovered_from_current_snapshot = is_current;
            self.snapshot_txn_id = envelope.txn_id;
            self.last_txn_id = self.last_txn_id.max(envelope.txn_id);
            let historical_node_history_complete = envelope.historical_node_history_complete;
            let mut historical_node_ids = envelope.historical_node_ids;
            if historical_node_history_complete.is_none() {
                historical_node_ids.extend(envelope.graph.nodes.iter().map(|node| node.id.clone()));
            }
            let vector_payload_store = self.store.vector_payload_store.clone();
            let mut recovered = InMemoryGraphStore::from_snapshot(envelope.graph)?;
            recovered.vector_payload_store = vector_payload_store;
            self.store = recovered;
            self.store.historical_node_history_complete &=
                historical_node_history_complete.unwrap_or(false);
            for id in historical_node_ids {
                self.store.record_historical_node_id(&id);
            }
        }
        self.replay_aof(&data_dir)?;
        self.prune_resident_cold_rollback_records()?;
        let _ = recovered_from_current_snapshot;
        Ok(())
    }

    fn has_legacy_redcore_artifacts(&self) -> bool {
        self.data_dir.as_ref().is_some_and(|data_dir| {
            legacy_redcore_artifact_names()
                .iter()
                .any(|name| data_dir.join(name).exists())
        })
    }

    fn for_each_legacy_graph_record(
        &self,
        mut visit: impl FnMut(GraphPayloadRecord) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<()> {
        for edge in self.store.edges.values() {
            visit(GraphPayloadRecord::Edge(edge.clone()))?;
        }
        for node in self.store.nodes.values() {
            visit(GraphPayloadRecord::Node(node.clone()))?;
        }
        for id in &self.store.cold_nonresident_node_ids {
            if self.store.nodes.contains_key(id) {
                return Err(GraphStoreError::new(
                    "redcore_legacy_cold_payload_duplicate",
                    format!("cold node {id:?} is already resident during migration"),
                ));
            }
            let path = self
                .cold_rollback_record_path(id)?
                .filter(|path| path.is_file())
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "redcore_legacy_cold_payload_missing",
                        format!("cannot migrate cold node {id:?}: its rollback payload is missing"),
                    )
                })?;
            let record = Self::read_cold_rollback_record(&path)?;
            if record.id != *id || record.node.id != *id {
                return Err(GraphStoreError::new(
                    "redcore_legacy_cold_payload_mismatch",
                    format!("cold rollback payload does not match node {id:?}"),
                ));
            }
            visit(GraphPayloadRecord::Node(record.node))?;
        }
        Ok(())
    }

    fn legacy_logical_entries(
        &self,
        publication: &dyn ChunkStore,
    ) -> GraphStoreResult<(Vec<GraphTreeEntry>, usize, usize)> {
        let mut entries = Vec::with_capacity(self.legacy_record_capacity()?);
        let mut chunks = RedCoreBootstrapChunkBatch::default();
        let mut nodes_total = 0usize;
        let mut edges_total = 0usize;
        self.for_each_legacy_graph_record(|record| {
            let (bytes, key, kind, tombstone) = match record {
                GraphPayloadRecord::Node(node) => {
                    if !node.tombstone {
                        nodes_total = nodes_total.checked_add(1).ok_or_else(|| {
                            GraphStoreError::new(
                                "redcore_legacy_node_count_overflow",
                                "legacy live node count overflowed usize",
                            )
                        })?;
                    }
                    (
                        canonical_node_preimage(&node)?,
                        format!("node/{}", node.id),
                        GraphObjectKind::Node,
                        node.tombstone,
                    )
                }
                GraphPayloadRecord::Edge(edge) => {
                    if !edge.tombstone {
                        edges_total = edges_total.checked_add(1).ok_or_else(|| {
                            GraphStoreError::new(
                                "redcore_legacy_edge_count_overflow",
                                "legacy live edge count overflowed usize",
                            )
                        })?;
                    }
                    (
                        canonical_edge_preimage(&edge)?,
                        format!("edge/{}", edge.id),
                        GraphObjectKind::Edge,
                        edge.tombstone,
                    )
                }
            };
            let object_hash = Hash::of(&bytes);
            chunks.push(publication, object_hash, bytes)?;
            entries.push(GraphTreeEntry {
                key,
                kind,
                object_hash,
                tombstone,
            });
            Ok(())
        })?;
        chunks.flush(publication)?;
        sort_and_validate_legacy_entries(&mut entries)?;
        Ok((entries, nodes_total, edges_total))
    }

    fn legacy_payload_entries(
        &self,
        publication: &dyn ChunkStore,
    ) -> GraphStoreResult<Vec<GraphTreeEntry>> {
        let mut entries = Vec::with_capacity(self.legacy_record_capacity()?);
        let mut chunks = RedCoreBootstrapChunkBatch::default();
        self.for_each_legacy_graph_record(|record| {
            let bytes = canonical_graph_payload_preimage(&record)?;
            let object_hash = Hash::of(&bytes);
            let entry = GraphTreeEntry {
                key: record.key(),
                kind: record.kind(),
                object_hash,
                tombstone: record.tombstone(),
            };
            chunks.push(publication, object_hash, bytes)?;
            entries.push(entry);
            Ok(())
        })?;
        chunks.flush(publication)?;
        sort_and_validate_legacy_entries(&mut entries)?;
        Ok(entries)
    }

    fn legacy_record_capacity(&self) -> GraphStoreResult<usize> {
        self.store
            .nodes
            .len()
            .checked_add(self.store.edges.len())
            .and_then(|count| count.checked_add(self.store.cold_nonresident_node_ids.len()))
            .ok_or_else(|| {
                GraphStoreError::new(
                    "redcore_legacy_record_count_overflow",
                    "legacy graph record count overflowed usize",
                )
            })
    }

    fn publish_legacy_journal_bootstrap(
        &mut self,
        journal: &Arc<JournalChunkStore>,
    ) -> GraphStoreResult<()> {
        let txn_id = self.last_txn_id;
        let graph_version = self.store.version;
        let publication = ChunkPublicationWindow::new(
            Arc::clone(journal),
            &format!("redcore-legacy-bootstrap:{txn_id}:{graph_version}"),
        );
        let (logical_entries, nodes_total, edges_total) =
            self.legacy_logical_entries(&publication)?;
        let logical_tree = build_prolly_tree_from_entries(&publication, logical_entries)?;
        let payload_entries = self.legacy_payload_entries(&publication)?;
        let payload_tree = build_prolly_tree_from_entries(&publication, payload_entries)?;
        if logical_tree.entries_total != payload_tree.entries_total {
            return Err(GraphStoreError::new(
                "redcore_journal_tree_count_mismatch",
                format!(
                    "legacy logical tree has {} entries but payload tree has {}",
                    logical_tree.entries_total, payload_tree.entries_total
                ),
            ));
        }

        let commit = build_commit(
            &logical_tree,
            "main",
            Vec::new(),
            "rustyred",
            "bootstrap RedCore journal",
            unix_ms(),
            graph_version,
            None,
        )?;
        let pack = recovered_graph_pack_descriptor(
            logical_tree.clone(),
            graph_version,
            nodes_total,
            edges_total,
            commit.commit_hash,
        )?;
        let historical_node_tree = build_historical_node_registry(
            &publication,
            self.store.historical_node_ids.iter().map(String::as_str),
        )?;
        let state = RedCoreJournalState {
            txn_id,
            graph_version,
            pack_graph_version: graph_version,
            logical_tree: logical_tree.clone(),
            payload_tree: payload_tree.clone(),
            logical_commit: commit.commit_hash,
            nodes_total,
            edges_total,
            historical_node_tree: Some(historical_node_tree.clone()),
            legacy_historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: self.store.historical_node_history_complete,
            vector_designations: self.store.vector_designations(),
            ordered_designations: self.store.ordered_designations(),
            mutation_log: self.mutation_log.clone(),
            history_root: None,
        };
        let state_bytes = state.to_bytes()?;
        let state_address = Hash::of(&state_bytes);
        publication.write(&[(state_address, state_bytes.as_slice())])?;

        #[cfg(test)]
        exit_after_uncommitted_chunks_if_requested(journal.as_ref(), txn_id);

        journal.commit_catalog_updates(
            &BTreeMap::from([
                (ROOT_NAME_GRAPH.to_string(), Some(logical_tree.root)),
                (ROOT_NAME_REDCORE.to_string(), Some(state_address)),
            ]),
            CommitMeta {
                parents: Vec::new(),
                author: "rustyred".to_string(),
                message: format!("publish legacy RedCore transaction {txn_id}"),
                timestamp_unix_ms: unix_ms(),
                graph_version,
            },
        )?;
        publication.finish_after_publication();

        self.current_pack = Some(pack);
        self.payload_tree = Some(payload_tree);
        self.historical_node_tree = Some(historical_node_tree);
        self.snapshot_txn_id = txn_id;
        self.last_fsync = Some(SystemTime::now());
        self.aof_size_bytes = journal_file_len(journal.path())?;
        Ok(())
    }

    fn readmit_legacy_cold_records(&mut self) -> GraphStoreResult<()> {
        let ids = self
            .store
            .cold_nonresident_node_ids
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for id in ids {
            let path = self
                .cold_rollback_record_path(&id)?
                .filter(|path| path.is_file())
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "redcore_legacy_cold_payload_missing",
                        format!("cannot readmit cold node {id:?}: its rollback payload is missing"),
                    )
                })?;
            let record = Self::read_cold_rollback_record(&path)?;
            if record.id != id || record.node.id != id {
                return Err(GraphStoreError::new(
                    "redcore_legacy_cold_payload_mismatch",
                    format!("cold rollback payload does not match node {id:?}"),
                ));
            }
            self.store.readmit_node(record.node)?;
        }
        Ok(())
    }

    fn move_legacy_redcore_artifacts_to_backup(&self) -> GraphStoreResult<()> {
        let Some(data_dir) = self.data_dir.as_ref() else {
            return Ok(());
        };
        let artifacts = legacy_redcore_artifact_names()
            .iter()
            .map(|name| data_dir.join(name))
            .filter(|path| path.exists())
            .collect::<Vec<_>>();
        if artifacts.is_empty() {
            return Ok(());
        }
        let backup = data_dir.join(REDCORE_LEGACY_BACKUP_DIR);
        fs::create_dir_all(&backup)
            .map_err(|error| GraphStoreError::io("create RedCore legacy backup", error))?;
        for source in artifacts {
            let name = source.file_name().ok_or_else(|| {
                GraphStoreError::new(
                    "redcore_legacy_backup_name",
                    format!("legacy artifact {} has no file name", source.display()),
                )
            })?;
            let destination = backup.join(name);
            if destination.exists() {
                return Err(GraphStoreError::new(
                    "redcore_legacy_backup_exists",
                    format!(
                        "cannot move {} because backup target {} already exists",
                        source.display(),
                        destination.display()
                    ),
                ));
            }
            fs::rename(&source, &destination).map_err(|error| {
                GraphStoreError::io(
                    format!(
                        "move legacy RedCore artifact {} to {}",
                        source.display(),
                        destination.display()
                    ),
                    error,
                )
            })?;
        }
        sync_directory(&backup, "fsync RedCore legacy backup")?;
        sync_directory(data_dir, "fsync RedCore directory after legacy migration")
    }

    fn replay_aof(&mut self, data_dir: &Path) -> GraphStoreResult<()> {
        let path = data_dir.join(REDCORE_AOF_FILE);
        if !path.exists() {
            return Ok(());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|err| GraphStoreError::io("open RedCore AOF", err))?;
        self.aof_size_bytes = file
            .metadata()
            .map_err(|err| GraphStoreError::io("stat RedCore AOF", err))?
            .len();
        let read_file = file
            .try_clone()
            .map_err(|err| GraphStoreError::io("clone RedCore AOF reader", err))?;
        let mut reader = BufReader::new(read_file);
        let mut frame_start = 0_u64;
        loop {
            let mut line = String::new();
            let bytes_read = reader
                .read_line(&mut line)
                .map_err(|err| GraphStoreError::io("read RedCore AOF", err))?;
            if bytes_read == 0 {
                break;
            }
            let frame_end = frame_start + bytes_read as u64;
            if line.trim().is_empty() {
                frame_start = frame_end;
                self.aof_replay_bytes = frame_end;
                continue;
            }
            if !line.ends_with('\n') {
                truncate_aof_tail(&file, data_dir, frame_start)?;
                self.truncated_aof_tail = true;
                self.aof_replay_bytes = frame_start;
                break;
            }
            let raw = line.trim_end_matches(['\r', '\n']);
            if covered_aof_txn_id(raw, self.snapshot_txn_id).is_some() {
                frame_start = frame_end;
                self.skipped_aof_frames += 1;
                self.aof_replay_bytes = frame_end;
                continue;
            }
            let frame = match decode_aof_frame(raw) {
                Ok(frame) => frame,
                Err(error) => {
                    if aof_has_valid_frame_after(&mut reader)? {
                        return Err(GraphStoreError::new(
                            "redcore_aof_corrupt_frame",
                            format!(
                                "RedCore AOF frame at byte offset {frame_start} is corrupt; \
                                 the log was left intact because a valid later frame exists: {} ({})",
                                error.code, error.message
                            ),
                        ));
                    }
                    truncate_aof_tail(&file, data_dir, frame_start)?;
                    self.truncated_aof_tail = true;
                    self.aof_size_bytes = frame_start;
                    self.aof_replay_bytes = frame_start;
                    break;
                }
            };
            if frame.txn_id <= self.snapshot_txn_id {
                frame_start = frame_end;
                self.skipped_aof_frames += 1;
                self.aof_replay_bytes = frame_end;
                continue;
            }
            let graph_version = frame.graph_version;
            self.apply_recovered_mutation(frame.mutation)?;
            self.store.version = self.store.version.max(graph_version);
            self.last_txn_id = self.last_txn_id.max(frame.txn_id);
            self.recovered_frames += 1;
            frame_start = frame_end;
            self.aof_replay_bytes = frame_end;
        }
        Ok(())
    }

    fn apply_recovered_edge_lossy(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        // Ported from RustyRed-Graph-Database 365d073 "fix(core): tolerate orphan
        // edges during recovery". AOF replay must not let one orphan edge
        // (endpoint missing or tombstoned) make the whole tenant graph
        // unavailable; skip the orphan instead. The strict live-write path
        // (upsert_edge / apply_recovered_edge) is unchanged.
        match self.store.apply_recovered_edge(edge) {
            Ok(()) => Ok(()),
            Err(error) if is_recoverable_orphan_edge(&error) => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn apply_recovered_mutation(&mut self, mutation: RedCoreMutation) -> GraphStoreResult<()> {
        match mutation {
            RedCoreMutation::NodeUpsert(node) => self.store.apply_recovered_node(node),
            RedCoreMutation::EdgeUpsert(edge) => self.apply_recovered_edge_lossy(edge),
            RedCoreMutation::Batch(mutations) => {
                let mut cold_rollback_ids = Vec::new();
                for mutation in mutations {
                    match mutation {
                        RedCoreMutation::NodeUpsert(mut node) => {
                            let rollback_record = self
                                .cold_rollback_record_path(&node.id)?
                                .filter(|path| path.is_file())
                                .map(|path| Self::read_cold_rollback_record(&path))
                                .transpose()?;
                            if let Some(record) = rollback_record {
                                if record.id != node.id || record.node.id != node.id {
                                    return Err(GraphStoreError::new(
                                        "cold_rollback_id_mismatch",
                                        format!(
                                            "cold rollback record does not match replayed node {}",
                                            node.id
                                        ),
                                    ));
                                }
                                if take_cold_rollback_marker(&mut node, &record.node)? {
                                    cold_rollback_ids.push(node.id.clone());
                                }
                            }
                            self.store.apply_recovered_node(node)?;
                        }
                        mutation => self.apply_recovered_mutation(mutation)?,
                    }
                }
                for id in cold_rollback_ids {
                    self.store.evict_node(&id)?;
                }
                Ok(())
            }
            RedCoreMutation::VectorDesignation(d) => {
                self.store
                    .install_vector_designation(&d.label, &d.property, d.dimension)
            }
            RedCoreMutation::OrderedDesignation(d) => {
                self.store.designate_ordered_property(&d.label, &d.property)
            }
            RedCoreMutation::VectorDesignationClear(d) => {
                self.store.clear_vector_property(&d.label, &d.property)
            }
            RedCoreMutation::OrderedDesignationClear(d) => {
                self.store.clear_ordered_property(&d.label, &d.property)
            }
            RedCoreMutation::NodeDelete(id) => self.store.apply_recovered_delete(&id),
            RedCoreMutation::EdgeDelete(id) => {
                self.store.apply_logical_edge_delete(&id)?;
                Ok(())
            }
        }
    }

    fn persist_before_publish(
        &mut self,
        txn_id: u64,
        graph_version: u64,
        staged: &InMemoryGraphStore,
        mutation: RedCoreMutation,
        pending_events: &[PendingHookEvent],
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<(Option<u64>, u64)> {
        self.persist_journal_commit(
            txn_id,
            graph_version,
            Some(staged),
            &mutation,
            pending_events,
            metadata,
        )?;
        Ok((self.journal.as_ref().map(|_| txn_id), txn_id))
    }

    fn persist_delta_before_publish(
        &mut self,
        txn_id: u64,
        graph_version: u64,
        mutation: RedCoreMutation,
        pending_events: &[PendingHookEvent],
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<()> {
        self.persist_journal_commit(
            txn_id,
            graph_version,
            None,
            &mutation,
            pending_events,
            metadata,
        )
    }

    fn persist_journal_commit(
        &mut self,
        txn_id: u64,
        graph_version: u64,
        staged: Option<&InMemoryGraphStore>,
        mutation: &RedCoreMutation,
        pending_events: &[PendingHookEvent],
        metadata: Option<&GraphCommitMetadata>,
    ) -> GraphStoreResult<()> {
        let backend = self.chunk_store.get();
        let metadata = metadata.cloned().unwrap_or_else(|| GraphCommitMetadata {
            author: "rustyred".into(),
            message: "commit RedCore mutation batch".into(),
            timestamp_unix_ms: unix_ms(),
        });
        let publication = ChunkPublicationWindow::new(
            Arc::clone(&backend),
            &format!("redcore-journal:{txn_id}:{graph_version}"),
        );
        let batch = mutation.graph_batch();
        let (next_pack, next_payload_tree) = match (&self.current_pack, &self.payload_tree) {
            (None, None) => {
                let staged = staged.ok_or_else(|| {
                    GraphStoreError::new(
                        "redcore_journal_bootstrap_snapshot_missing",
                        "the first durable journal commit requires the staged graph mirror",
                    )
                })?;
                let snapshot = staged.snapshot();
                let pack = retarget_graph_pack_version(
                    compile_graph_pack(
                        &snapshot,
                        GraphCompileOptions {
                            name: Some("redcore".to_string()),
                            branch: Some("main".to_string()),
                            parent_commits: Vec::new(),
                            author: Some(metadata.author.clone()),
                            message: Some(metadata.message.clone()),
                            timestamp_unix_ms: Some(metadata.timestamp_unix_ms),
                            include_payloads: true,
                            ..GraphCompileOptions::default()
                        },
                    )?,
                    graph_version,
                )?;
                write_graph_pack_chunks(&publication, &pack)?;
                let payload_tree = build_graph_payload_index(&publication, &snapshot)?;
                (pack, payload_tree)
            }
            (Some(prior_pack), Some(prior_payload_tree)) if batch.mutations.is_empty() => {
                (prior_pack.clone(), prior_payload_tree.clone())
            }
            (Some(prior_pack), Some(prior_payload_tree)) => {
                let incremental = compile_graph_pack_incremental_with_store(
                    &publication,
                    prior_pack,
                    &batch,
                    GraphCompileOptions {
                        name: Some("redcore".to_string()),
                        branch: Some("main".to_string()),
                        parent_commits: Vec::new(),
                        author: Some(metadata.author.clone()),
                        message: Some(metadata.message.clone()),
                        timestamp_unix_ms: Some(metadata.timestamp_unix_ms),
                        include_payloads: true,
                        ..GraphCompileOptions::default()
                    },
                )?;
                let pack = retarget_graph_pack_version(incremental.pack, graph_version)?;
                write_graph_pack_chunks(&publication, &pack)?;
                let payload_tree =
                    update_graph_payload_index(&publication, prior_payload_tree, &batch)?;
                (pack, payload_tree)
            }
            _ => {
                return Err(GraphStoreError::new(
                    "redcore_journal_head_incomplete",
                    "logical and recovery-payload roots must be present or absent together",
                ))
            }
        };

        let designation_store = staged.unwrap_or(&self.store);
        let mut mutation_node_ids = Vec::new();
        mutation.collect_historical_node_ids(&mut mutation_node_ids);
        let (next_historical_node_tree, historical_node_history_complete) = match self
            .historical_node_tree
            .as_ref()
        {
            Some(prior) => {
                let mut accepted_ids = BTreeSet::new();
                let historical_node_history_complete = match staged {
                    Some(staged) => {
                        for id in mutation_node_ids {
                            if staged.historical_node_ids.contains(id)
                                && !self.store.historical_node_ids.contains(id)
                            {
                                accepted_ids.insert(id);
                            }
                        }
                        staged.historical_node_history_complete
                    }
                    None => {
                        let mut projected_len = self.store.historical_node_ids.len();
                        let mut complete = self.store.historical_node_history_complete;
                        for id in mutation_node_ids {
                            if self.store.historical_node_ids.contains(id)
                                || accepted_ids.contains(id)
                            {
                                continue;
                            }
                            if projected_len >= self.store.historical_node_id_limit {
                                complete = false;
                                continue;
                            }
                            accepted_ids.insert(id);
                            projected_len += 1;
                        }
                        complete
                    }
                };
                (
                    update_historical_node_registry(&publication, prior, accepted_ids.into_iter())?,
                    historical_node_history_complete,
                )
            }
            None => match staged {
                Some(staged) => (
                    build_historical_node_registry(
                        &publication,
                        staged.historical_node_ids.iter().map(String::as_str),
                    )?,
                    staged.historical_node_history_complete,
                ),
                // A v1/v2 recovery has no durable registry tree yet. Project the
                // post-mutation set once; subsequent v3 commits use only deltas.
                None => {
                    let (ids, complete) = self
                        .store
                        .projected_historical_registry_after(mutation_node_ids);
                    (
                        build_historical_node_registry(
                            &publication,
                            ids.iter().map(String::as_str),
                        )?,
                        complete,
                    )
                }
            },
        };
        if let Some(staged) = staged {
            let stats = staged.stats_without_memory_estimate();
            if stats.nodes_total > next_pack.manifest.nodes_total
                || stats.edges_total > next_pack.manifest.edges_total
            {
                return Err(GraphStoreError::new(
                    "redcore_journal_live_count_mismatch",
                    format!(
                        "staged resident mirror has {} nodes and {} edges, exceeding the logical head's {} nodes and {} edges",
                        stats.nodes_total,
                        stats.edges_total,
                        next_pack.manifest.nodes_total,
                        next_pack.manifest.edges_total
                    ),
                ));
            }
        }
        if next_payload_tree.entries_total != next_pack.tree.entries_total {
            return Err(GraphStoreError::new(
                "redcore_journal_tree_count_mismatch",
                format!(
                    "logical tree has {} entries but payload tree has {}",
                    next_pack.tree.entries_total, next_payload_tree.entries_total
                ),
            ));
        }

        let prior_roots = self.revision_roots()?;
        let next_roots = GraphRevisionRoots {
            transaction: txn_id,
            graph_version: next_pack.commit.graph_version,
            logical_commit: next_pack.commit.commit_hash,
            logical_tree: next_pack.tree.clone(),
            payload_tree: next_payload_tree.clone(),
            historical_node_tree: Some(next_historical_node_tree.clone()),
            historical_node_history_complete,
            vector_designations: designation_store.vector_designations(),
            ordered_designations: designation_store.ordered_designations(),
        };
        let mut next_commit_publication = None;
        let next_history_root = if batch.mutations.is_empty() {
            self.history_root
        } else {
            let prior_history = self
                .history_root
                .map(|address| load_history_root(&publication, address))
                .transpose()?;
            let mut candidates = BTreeSet::new();
            for mutation in &batch.mutations {
                candidates.insert(match mutation {
                    GraphMutation::NodeUpsert(node) => format!("node/{}", node.id),
                    GraphMutation::NodeDelete(id) => format!("node/{id}"),
                    GraphMutation::EdgeUpsert(edge) => format!("edge/{}", edge.id),
                    GraphMutation::EdgeDelete(id) => format!("edge/{id}"),
                });
            }
            let mut changed_object_keys = Vec::new();
            for key in candidates {
                let before = self
                    .payload_tree
                    .as_ref()
                    .map(|tree| crate::versioned_graph::graph_tree_entry(&publication, tree, &key))
                    .transpose()?
                    .flatten();
                let after = crate::versioned_graph::graph_tree_entry(
                    &publication,
                    &next_payload_tree,
                    &key,
                )?;
                if before != after {
                    changed_object_keys.push(key);
                }
            }
            let entry = GraphHistoryEntry {
                commit: next_pack.commit.clone(),
                roots: next_roots,
                changed_object_keys,
            };
            if self.commit_observer.is_some() {
                next_commit_publication = Some(GraphCommitPublication::prepare(
                    &publication,
                    &entry,
                    prior_roots.as_ref(),
                )?);
            }
            Some(
                append_history(
                    &publication,
                    prior_history.as_ref(),
                    prior_roots.as_ref(),
                    &entry,
                )?
                .0,
            )
        };

        let preview = self.preview_mutation_log(pending_events, txn_id, next_pack.tree.root);
        let state = RedCoreJournalState {
            txn_id,
            graph_version,
            pack_graph_version: next_pack.commit.graph_version,
            logical_tree: next_pack.tree.clone(),
            payload_tree: next_payload_tree.clone(),
            logical_commit: next_pack.commit.commit_hash,
            nodes_total: next_pack.manifest.nodes_total,
            edges_total: next_pack.manifest.edges_total,
            historical_node_tree: Some(next_historical_node_tree.clone()),
            legacy_historical_node_ids: BTreeSet::new(),
            historical_node_history_complete,
            vector_designations: designation_store.vector_designations(),
            ordered_designations: designation_store.ordered_designations(),
            mutation_log: preview.clone(),
            history_root: next_history_root,
        };
        let state_bytes = state.to_bytes()?;
        let state_address = Hash::of(&state_bytes);
        publication.write(&[(state_address, state_bytes.as_slice())])?;

        #[cfg(test)]
        if let Some(journal) = &self.journal {
            exit_after_uncommitted_chunks_if_requested(journal.as_ref(), txn_id);
        }

        #[cfg(test)]
        if let (Some(reachable), Some(journal)) = (
            self.journal_precommit_compaction_reachable.take(),
            self.journal.as_ref(),
        ) {
            let expected_catalog_root = journal.read_root()?;
            self.journal_precommit_compaction_report =
                Some(journal.compact_reachable(expected_catalog_root, &reachable)?);
        }

        backend.commit_catalog_updates(
            &BTreeMap::from([
                (ROOT_NAME_GRAPH.to_string(), Some(next_pack.tree.root)),
                (ROOT_NAME_REDCORE.to_string(), Some(state_address)),
            ]),
            CommitMeta {
                // The catalog implementation derives the physical parent
                // while holding the same lock used for the named-root merge.
                parents: Vec::new(),
                author: "rustyred".to_string(),
                message: format!("publish RedCore transaction {txn_id}"),
                timestamp_unix_ms: unix_ms(),
                graph_version,
            },
        )?;
        publication.finish_after_publication();

        self.durable_nodes_total = next_pack.manifest.nodes_total;
        self.durable_edges_total = next_pack.manifest.edges_total;
        self.mutation_log = preview;
        self.note_sync_root(next_pack.tree.root);
        self.current_pack = Some(next_pack);
        self.payload_tree = Some(next_payload_tree);
        self.historical_node_tree = Some(next_historical_node_tree);
        self.history_root = next_history_root;
        self.pending_commit_publication = next_commit_publication;
        if let Some(journal) = &self.journal {
            self.last_fsync = Some(SystemTime::now());
            self.aof_size_bytes = journal_file_len(journal.path())?;
        }
        Ok(())
    }

    /// Open RedCore while bounding the recovered edge mirror by an accounted
    /// resident-byte budget.
    ///
    /// Durable payload-tree state remains complete. When the recovered edge
    /// estimate exceeds `resident_edge_budget_bytes`, nodes stay resident and
    /// edge-dependent operations fail closed with `resident_edges_nonresident`
    /// instead of allowing process RSS to cross the watchdog ceiling.
    pub fn open_with_resident_edge_budget(
        data_dir: impl Into<PathBuf>,
        options: RedCoreOptions,
        resident_edge_budget_bytes: usize,
    ) -> GraphStoreResult<Self> {
        Self::open_with_optional_resident_edge_budget(
            data_dir,
            options,
            Some(resident_edge_budget_bytes),
        )
    }

    fn open_with_optional_resident_edge_budget(
        data_dir: impl Into<PathBuf>,
        options: RedCoreOptions,
        resident_edge_budget_bytes: Option<usize>,
    ) -> GraphStoreResult<Self> {
        Self::open_with_vector_recovery_policy(data_dir, options, resident_edge_budget_bytes, true)
    }

    fn open_with_vector_recovery_policy(
        data_dir: impl Into<PathBuf>,
        options: RedCoreOptions,
        resident_edge_budget_bytes: Option<usize>,
        migrate_legacy_vectors: bool,
    ) -> GraphStoreResult<Self> {
        validate_redcore_options(&options)?;
        let data_dir = data_dir.into();
        fs::create_dir_all(&data_dir)
            .map_err(|err| GraphStoreError::io("create RedCore data directory", err))?;
        let directory_lock = acquire_redcore_directory_lock(&data_dir)?;
        let (journal, chunks): (Option<Arc<JournalChunkStore>>, Arc<dyn UnifiedChunkStore>) =
            match options.durability.journal_commit_policy() {
                None => {
                    let memory = Arc::new(MemChunkStore::new());
                    (None, memory)
                }
                Some(policy) => {
                    let journal = Arc::new(JournalChunkStore::open_with_commit_policy(
                        data_dir.join(REDCORE_JOURNAL_FILE),
                        policy,
                    )?);
                    (Some(Arc::clone(&journal)), journal)
                }
            };
        let mut engine = Self::assemble(
            chunks,
            journal,
            Some(data_dir.clone()),
            Some(directory_lock),
            options,
            resident_edge_budget_bytes,
        );
        let journal_bytes = engine.durable_log_len()?;
        engine
            .recovery_profile
            .capture("journal_open", None::<String>, journal_bytes, 0, 0);
        let shared = engine.unified_chunk_store();
        let legacy_root = engine
            .data_dir
            .as_ref()
            .expect("opened RedCore retains its data directory")
            .join("cold_objects");
        let payload_store: Arc<dyn TensorBlockPayloadStore> = if engine.journal.is_some() {
            Arc::new(DiskObjectStore::from_redcore_chunk_store(
                shared,
                legacy_root,
            )?)
        } else {
            Arc::new(DiskObjectStore::from_chunk_store_with_legacy_root(
                shared,
                legacy_root,
            )?)
        };
        Self::recover_composed(engine, payload_store, migrate_legacy_vectors)
    }

    /// Compose a durable RedCore over an injected, already-open chunk store.
    ///
    /// The backend is whatever the caller opened: an OPFS-backed
    /// `ProjectionChunkStore` in the browser, or a log file or memory store in
    /// tests. The constructor opens no path, creates no directory and takes no
    /// directory lock. Durable state is the backend's committed named-root
    /// catalog, recovered by the same machinery as a journal-backed open. An
    /// empty backend opens as an empty graph; a catalog that names missing or
    /// corrupt state is refused by name.
    ///
    /// The caller owns the backend's writer exclusion and publication/flush
    /// policy. `options` validates RedCore configuration but does not change an
    /// already-open backend's policy or start a group-commit thread. In
    /// particular, injecting a memory backend does not make it durable.
    pub fn open_with_chunk_store(
        chunk_store: Arc<dyn UnifiedChunkStore>,
        options: RedCoreOptions,
    ) -> GraphStoreResult<Self> {
        validate_redcore_options(&options)?;
        let payload_store: Arc<dyn TensorBlockPayloadStore> = Arc::new(
            DiskObjectStore::from_recovered_chunk_store(Arc::clone(&chunk_store))?,
        );
        let engine = Self::assemble(chunk_store, None, None, None, options, None);
        Self::recover_composed(engine, payload_store, true)
    }

    /// The one place a RedCore facade is assembled around a chunk backend. The
    /// caller installs the payload lens and runs recovery.
    fn assemble(
        chunks: Arc<dyn UnifiedChunkStore>,
        journal: Option<Arc<JournalChunkStore>>,
        data_dir: Option<PathBuf>,
        directory_lock: Option<RedCoreDirectoryLock>,
        options: RedCoreOptions,
        resident_edge_budget_bytes: Option<usize>,
    ) -> Self {
        let injected_chunk_store = data_dir.is_none();
        Self {
            store: InMemoryGraphStore::new(),
            data_dir,
            _directory_lock: directory_lock,
            chunk_store: UnifiedChunkStoreHandle::new(chunks),
            injected_chunk_store,
            journal,
            current_pack: None,
            payload_tree: None,
            historical_node_tree: None,
            history_root: None,
            durable_nodes_total: 0,
            durable_edges_total: 0,
            resident_edge_bytes_estimate: 0,
            resident_edge_budget_bytes,
            edges_fully_resident: true,
            options,
            last_txn_id: 0,
            snapshot_txn_id: 0,
            recovered_frames: 0,
            skipped_aof_frames: 0,
            aof_size_bytes: 0,
            aof_replay_bytes: 0,
            recovery_duration_ms: 0,
            recovery_profile: RecoveryProfile::default(),
            truncated_aof_tail: false,
            last_recovery_ok: false,
            last_fsync: None,
            transient_ordered_indexes: HashMap::new(),
            hook_emitter: None,
            commit_observer: None,
            batch_preparer: None,
            preparation_identity: Arc::new(()),
            pending_commit_publication: None,
            hook_emit_depth: 0,
            hook_tenant: String::new(),
            hook_actor: String::new(),
            hook_principal: String::new(),
            recorded_committed_mutations: None,
            recorded_redcore_mutations: None,
            mutation_log: MutationLog::genesis(),
            sync_root: genesis_sync_root(),
            sync_parent_by_root: HashMap::new(),
            #[cfg(test)]
            journal_precommit_compaction_reachable: None,
            #[cfg(test)]
            journal_precommit_compaction_report: None,
        }
    }

    /// Whether this store recovers from its chunk catalog. Journal-backed
    /// stores do, and so does a store composed over an injected backend, which
    /// has no data directory. A store with an on-disk data directory but no
    /// journal (`RedCoreDurability::None`) keeps the legacy artifact recovery.
    fn recovers_from_catalog(&self) -> bool {
        self.journal.is_some() || self.injected_chunk_store
    }

    /// Byte length of the durable journal. An injected backend reports no
    /// journal length; its size belongs to the backend.
    fn durable_log_len(&self) -> GraphStoreResult<u64> {
        match self.journal.as_ref() {
            Some(journal) => journal_file_len(journal.path()),
            None => Ok(0),
        }
    }

    /// The shared tail of every durable open: install the payload lens, recover
    /// the committed state, then record the recovery time.
    fn recover_composed(
        mut engine: Self,
        payload_store: Arc<dyn TensorBlockPayloadStore>,
        migrate_legacy_vectors: bool,
    ) -> GraphStoreResult<Self> {
        engine.store.set_vector_payload_store(payload_store);
        let recovery_started = Instant::now();
        engine.recover()?;
        if migrate_legacy_vectors {
            engine.migrate_legacy_vector_arrays()?;
        }
        engine.recovery_duration_ms =
            recovery_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        engine.last_recovery_ok = true;
        Ok(engine)
    }

    pub fn graph_snapshot_checked(&self) -> GraphStoreResult<GraphSnapshot> {
        self.require_fully_resident_edges("graph snapshot")?;
        Ok(self.graph_snapshot())
    }

    pub fn edges_fully_resident(&self) -> bool {
        self.edges_fully_resident
    }

    pub fn live_edges_snapshot_checked(&self) -> GraphStoreResult<Vec<EdgeRecord>> {
        self.require_fully_resident_edges("edge enumeration")?;
        Ok(self.store.live_edges_snapshot())
    }

    fn require_fully_resident_edges(&self, operation: &str) -> GraphStoreResult<()> {
        if self.edges_fully_resident {
            Ok(())
        } else {
            Err(self.resident_edges_nonresident_error(operation))
        }
    }

    fn resident_edges_nonresident_error(&self, operation: &str) -> GraphStoreError {
        GraphStoreError::new(
            "resident_edges_nonresident",
            format!(
                "{operation} is backpressured because the durable graph has {} edges and its \
                 accounted resident edge footprint exceeded the configured {}-byte budget; \
                 node reads and incremental upserts remain available",
                self.durable_edges_total,
                self.resident_edge_budget_bytes.unwrap_or_default(),
            ),
        )
    }

    pub fn epistemic_neighbors_checked(
        &self,
        node_id: &str,
        epistemic_types: Option<&[EpistemicType]>,
        min_confidence: Option<f64>,
        max_depth: Option<usize>,
    ) -> GraphStoreResult<Vec<(EdgeRecord, NodeRecord)>> {
        self.require_fully_resident_edges("epistemic neighbor traversal")?;
        Ok(self.epistemic_neighbors(node_id, epistemic_types, min_confidence, max_depth))
    }
}

impl Drop for RedCoreDirectoryLock {
    fn drop(&mut self) {
        let _ = crate::file_lock::FileExt::unlock(&self.file);
        if let Some(locks) = REDCORE_PROCESS_LOCKS.get() {
            if let Ok(mut locks) = locks.lock() {
                locks.remove(&self.process_key);
            }
        }
    }
}

fn validate_redcore_options(options: &RedCoreOptions) -> GraphStoreResult<()> {
    if options.strict_acid && options.durability == RedCoreDurability::None {
        return Err(GraphStoreError::new(
            "redcore_strict_mode_invalid",
            "strict ACID mode requires a durable journal policy",
        ));
    }
    Ok(())
}

fn journal_file_len(path: &Path) -> GraphStoreResult<u64> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| GraphStoreError::io("stat RedCore journal", error))
}

#[derive(Default)]
struct RedCoreBootstrapChunkBatch {
    chunks: Vec<(Hash, Vec<u8>)>,
    bytes: usize,
}

impl RedCoreBootstrapChunkBatch {
    fn push(
        &mut self,
        store: &dyn ChunkStore,
        address: Hash,
        bytes: Vec<u8>,
    ) -> GraphStoreResult<()> {
        if !self.chunks.is_empty()
            && (self.bytes.saturating_add(bytes.len()) > REDCORE_BOOTSTRAP_CHUNK_BATCH_BYTES
                || self.chunks.len() >= REDCORE_BOOTSTRAP_CHUNK_BATCH_COUNT)
        {
            self.flush(store)?;
        }
        self.bytes = self.bytes.saturating_add(bytes.len());
        self.chunks.push((address, bytes));
        if self.bytes >= REDCORE_BOOTSTRAP_CHUNK_BATCH_BYTES {
            self.flush(store)?;
        }
        Ok(())
    }

    fn flush(&mut self, store: &dyn ChunkStore) -> GraphStoreResult<()> {
        if self.chunks.is_empty() {
            return Ok(());
        }
        let borrowed = self
            .chunks
            .iter()
            .map(|(address, bytes)| (*address, bytes.as_slice()))
            .collect::<Vec<_>>();
        store.write(&borrowed)?;
        self.chunks.clear();
        self.bytes = 0;
        Ok(())
    }
}

fn sort_and_validate_legacy_entries(entries: &mut [GraphTreeEntry]) -> GraphStoreResult<()> {
    entries.sort_unstable_by(|left, right| left.key.cmp(&right.key));
    if let Some(duplicate) = entries
        .windows(2)
        .find(|window| window[0].key == window[1].key)
    {
        return Err(GraphStoreError::new(
            "redcore_legacy_duplicate_key",
            format!(
                "legacy graph contains duplicate journal key {:?}",
                duplicate[0].key
            ),
        ));
    }
    Ok(())
}

/// Cross-check an audited published tree against the manifest that names it.
///
/// The parity audit proves the tree is the canonical tree for its own entries;
/// this proves the entries are the ones the pack claims. Both directions matter:
/// a tree that lost a delta *and* the header that counted it is caught by the
/// audit, and a header that counted a delta the tree never received is caught
/// here, because the manifest is the only durable record of what the tree was
/// supposed to hold.
fn verify_published_tree_counts(
    audit: &GraphTreeParityAudit,
    manifest: &GraphPackManifest,
) -> GraphStoreResult<()> {
    if audit.entries_total != manifest.objects_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "published tree {} holds {} physical entries, its manifest declares {}",
                audit.tree_root, audit.entries_total, manifest.objects_total
            ),
        ));
    }
    if audit.live_nodes != manifest.nodes_total || audit.live_edges != manifest.edges_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "published tree {} holds {} live nodes and {} live edges, its manifest declares {} \
                 and {}",
                audit.tree_root, audit.live_nodes, audit.live_edges, manifest.nodes_total,
                manifest.edges_total
            ),
        ));
    }
    Ok(())
}

fn write_graph_pack_chunks(
    store: &dyn ChunkStore,
    pack: &CompiledGraphPack,
) -> GraphStoreResult<()> {
    let mut chunks = Vec::with_capacity(pack.tree_chunks.len() + pack.objects.len());
    for tree_chunk in &pack.tree_chunks {
        chunks.push((tree_chunk.address, tree_chunk.bytes.clone()));
    }
    for object in &pack.objects {
        let bytes = canonical_content_object_preimage(object)?;
        let address = Hash::of(&bytes);
        if address != object.hash {
            return Err(GraphStoreError::new(
                "redcore_journal_object_address_mismatch",
                format!(
                    "graph object {:?} hashes to {address}, expected {}",
                    object.key, object.hash
                ),
            ));
        }
        chunks.push((address, bytes));
    }
    let borrowed = chunks
        .iter()
        .map(|(address, bytes)| (*address, bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write(&borrowed)
}

fn acquire_redcore_directory_lock(data_dir: &Path) -> GraphStoreResult<RedCoreDirectoryLock> {
    let process_key = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf());
    let locks = REDCORE_PROCESS_LOCKS.get_or_init(|| Mutex::new(BTreeSet::new()));
    {
        let mut locks = locks.lock().map_err(|_| {
            GraphStoreError::new(
                "redcore_lock_poisoned",
                "RedCore process lock registry is poisoned",
            )
        })?;
        if !locks.insert(process_key.clone()) {
            return Err(GraphStoreError::new(
                "redcore_lock_unavailable",
                format!(
                    "RedCore data directory {} is already open in this process",
                    data_dir.display()
                ),
            ));
        }
    }

    let lock_path = data_dir.join(REDCORE_LOCK_FILE);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|err| {
            release_redcore_process_lock(&process_key);
            GraphStoreError::io("open RedCore directory lock", err)
        })?;
    if let Err(err) = file.try_lock_exclusive() {
        release_redcore_process_lock(&process_key);
        let message = if err.kind() == ErrorKind::WouldBlock {
            format!(
                "RedCore data directory {} is locked by another process",
                data_dir.display()
            )
        } else {
            format!("lock RedCore data directory: {err}")
        };
        return Err(GraphStoreError::new("redcore_lock_unavailable", message));
    }

    Ok(RedCoreDirectoryLock { file, process_key })
}

fn release_redcore_process_lock(process_key: &Path) {
    if let Some(locks) = REDCORE_PROCESS_LOCKS.get() {
        if let Ok(mut locks) = locks.lock() {
            locks.remove(process_key);
        }
    }
}

fn decode_aof_frame(raw: &str) -> GraphStoreResult<RedCoreAofFrame> {
    let frame: RedCoreAofWireFrame<'_> = serde_json::from_str(raw)
        .map_err(|err| GraphStoreError::io("decode RedCore AOF frame", err))?;
    if frame.magic != REDCORE_AOF_MAGIC || frame.version != REDCORE_MANIFEST_VERSION {
        return Err(GraphStoreError::new(
            "redcore_aof_frame_invalid",
            "RedCore AOF frame magic or version is invalid",
        ));
    }
    // AOF v1 checksums cover the exact persisted mutation token. Re-serializing
    // the typed mutation here is not equivalent: serde_json map ordering can
    // change with feature unification or dependency upgrades.
    let checksum = aof_payload_checksum(frame.mutation.get());
    if checksum != frame.payload_checksum {
        return Err(GraphStoreError::new(
            "redcore_aof_checksum_mismatch",
            format!("AOF frame {} checksum mismatch", frame.txn_id),
        ));
    }
    let mutation = serde_json::from_str(frame.mutation.get())
        .map_err(|err| GraphStoreError::io("decode RedCore AOF mutation", err))?;
    Ok(RedCoreAofFrame {
        magic: frame.magic,
        version: frame.version,
        txn_id: frame.txn_id,
        graph_version: frame.graph_version,
        timestamp_unix_ms: frame.timestamp_unix_ms,
        payload_checksum: frame.payload_checksum,
        mutation,
    })
}

fn aof_payload_checksum(raw_mutation: &str) -> String {
    let digest = Sha256::digest(raw_mutation.as_bytes());
    format!("sha256:{digest:x}")
}

/// Returns the transaction id when `raw` is a canonical frame already covered
/// by the active legacy snapshot. Canonical v1 frames used a stable, compact
/// field order. Avoiding mutation deserialization here matters for large batch
/// frames; if the prefix differs for any reason, migration falls back to the
/// full decoder and its checksum/corruption handling.
fn covered_aof_txn_id(raw: &str, snapshot_txn_id: u64) -> Option<u64> {
    let rest = raw.strip_prefix(r#"{"magic":"RRGDB_AOF","version":"#)?;
    let (version, rest) = rest.split_once(',')?;
    if version.parse::<u32>().ok()? != REDCORE_MANIFEST_VERSION {
        return None;
    }
    let rest = rest.strip_prefix(r#""txn_id":"#)?;
    let digits = rest
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 || rest.as_bytes().get(digits) != Some(&b',') {
        return None;
    }
    let txn_id = rest[..digits].parse::<u64>().ok()?;
    (txn_id <= snapshot_txn_id).then_some(txn_id)
}

fn aof_has_valid_frame_after(reader: &mut impl BufRead) -> GraphStoreResult<bool> {
    loop {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|err| GraphStoreError::io("scan RedCore AOF after corrupt frame", err))?;
        if bytes_read == 0 {
            return Ok(false);
        }
        if line.trim().is_empty() || !line.ends_with('\n') {
            continue;
        }
        let raw = line.trim_end_matches(['\r', '\n']);
        if decode_aof_frame(raw).is_ok() {
            return Ok(true);
        }
    }
}

fn truncate_aof_tail(file: &File, data_dir: &Path, offset: u64) -> GraphStoreResult<()> {
    file.set_len(offset)
        .map_err(|err| GraphStoreError::io("truncate torn RedCore AOF tail", err))?;
    file.sync_all()
        .map_err(|err| GraphStoreError::io("fsync truncated RedCore AOF", err))?;
    sync_directory(
        data_dir,
        "fsync RedCore data directory after AOF truncation",
    )
}

fn sync_file_path(path: &Path, action: &str) -> GraphStoreResult<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|err| GraphStoreError::io(action, err))
}

fn sync_directory(data_dir: &Path, action: impl AsRef<str>) -> GraphStoreResult<()> {
    File::open(data_dir)
        .and_then(|dir| dir.sync_all())
        .map_err(|err| GraphStoreError::io(action.as_ref(), err))
}

pub fn read_manifest(data_dir: &Path) -> GraphStoreResult<Option<RedCoreManifest>> {
    let path = data_dir.join(REDCORE_MANIFEST_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path).map_err(|err| GraphStoreError::io("read manifest", err))?;
    serde_json::from_str::<RedCoreManifest>(&raw)
        .map(Some)
        .map_err(|err| GraphStoreError::io("decode manifest", err))
}

fn read_latest_valid_snapshot(
    data_dir: &Path,
) -> GraphStoreResult<Option<(RedCoreSnapshotEnvelope, bool)>> {
    match read_snapshot_file(&data_dir.join(REDCORE_SNAPSHOT_FILE), "current snapshot") {
        Ok(Some(snapshot)) => Ok(Some((snapshot, true))),
        Ok(None) => read_snapshot_file(
            &data_dir.join(REDCORE_PREVIOUS_SNAPSHOT_FILE),
            "previous snapshot",
        )
        .map(|snapshot| snapshot.map(|snapshot| (snapshot, false))),
        Err(current_error) => match read_snapshot_file(
            &data_dir.join(REDCORE_PREVIOUS_SNAPSHOT_FILE),
            "previous snapshot",
        ) {
            Ok(Some(snapshot)) => Ok(Some((snapshot, false))),
            Ok(None) => Err(current_error),
            Err(_) => Err(current_error),
        },
    }
}

fn read_snapshot_file(
    path: &Path,
    label: &str,
) -> GraphStoreResult<Option<RedCoreSnapshotEnvelope>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path)
        .map_err(|err| GraphStoreError::io(format!("open RedCore {label}"), err))?;
    serde_json::from_reader::<_, RedCoreSnapshotEnvelope>(BufReader::new(file))
        .map(Some)
        .map_err(|err| GraphStoreError::io(format!("decode RedCore {label}"), err))
}

pub fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn system_time_unix_ms(time: SystemTime) -> Option<u128> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

#[cfg(feature = "redis-store")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RedisGraphKeyspace {
    prefix: String,
}

#[cfg(feature = "redis-store")]
impl RedisGraphKeyspace {
    pub fn new(prefix: impl Into<String>) -> Self {
        let prefix = prefix.into().trim().trim_end_matches(':').to_string();
        Self {
            prefix: if prefix.is_empty() {
                "rrgdb:deployment:graph:v1".to_string()
            } else {
                prefix
            },
        }
    }

    pub fn deployment_prefix(
        base_prefix: &str,
        deployment_id: &theorem_oplog::DeploymentId,
    ) -> String {
        let base_prefix = base_prefix.trim().trim_end_matches(':');
        let digest = crate::stable_hash(deployment_id.as_str())
            .expect("validated deployment identity is always hashable");
        let segment = digest.trim_start_matches("sha256:");
        if base_prefix.is_empty() {
            format!("rrgdb:{{deployment:{segment}}}:graph:v1")
        } else {
            format!("{base_prefix}:{{deployment:{segment}}}:graph:v1")
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn version(&self) -> String {
        self.key("version")
    }

    pub fn nodes(&self) -> String {
        self.key("nodes")
    }

    pub fn edges(&self) -> String {
        self.key("edges")
    }

    pub fn labels(&self) -> String {
        self.key("labels")
    }

    pub fn edge_types(&self) -> String {
        self.key("edge_types")
    }

    pub fn property_index_entries(&self) -> String {
        self.key("property_index_entries")
    }

    pub fn out_adjacency_pairs(&self) -> String {
        self.key("out_adjacency_pairs")
    }

    pub fn in_adjacency_pairs(&self) -> String {
        self.key("in_adjacency_pairs")
    }

    pub fn events(&self) -> String {
        self.key("events")
    }

    pub fn node(&self, id: &str) -> String {
        self.key(&format!("node:{}", encode_key_segment(id)))
    }

    pub fn edge(&self, id: &str) -> String {
        self.key(&format!("edge:{}", encode_key_segment(id)))
    }

    pub fn label(&self, label: &str) -> String {
        self.key(&format!("label:{}", encode_key_segment(label)))
    }

    pub fn edge_type(&self, edge_type: &str) -> String {
        self.key(&format!("edge_type:{}", encode_key_segment(edge_type)))
    }

    pub fn property_value(&self, key: &str, token: &str) -> String {
        self.key(&format!(
            "property:{}:{}",
            encode_key_segment(key),
            encode_key_segment(token)
        ))
    }

    pub fn out_adjacency(&self, node_id: &str, edge_type: &str) -> String {
        self.key(&format!(
            "adj:out:{}:{}",
            encode_key_segment(node_id),
            encode_key_segment(edge_type)
        ))
    }

    pub fn in_adjacency(&self, node_id: &str, edge_type: &str) -> String {
        self.key(&format!(
            "adj:in:{}:{}",
            encode_key_segment(node_id),
            encode_key_segment(edge_type)
        ))
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}:{suffix}", self.prefix)
    }
}

#[cfg(feature = "redis-store")]
#[derive(Clone, Debug)]
pub struct RedisGraphStore {
    client: redis::Client,
    keyspace: RedisGraphKeyspace,
}

#[cfg(feature = "redis-store")]
impl RedisGraphStore {
    pub fn new(redis_url: &str, key_prefix: impl Into<String>) -> redis::RedisResult<Self> {
        Ok(Self {
            client: redis::Client::open(redis_url)?,
            keyspace: RedisGraphKeyspace::new(key_prefix),
        })
    }

    pub fn deployment(
        redis_url: &str,
        base_prefix: &str,
        deployment_id: &theorem_oplog::DeploymentId,
    ) -> redis::RedisResult<Self> {
        Self::new(
            redis_url,
            RedisGraphKeyspace::deployment_prefix(base_prefix, deployment_id),
        )
    }

    pub fn keyspace(&self) -> &RedisGraphKeyspace {
        &self.keyspace
    }

    pub fn ping(&self) -> GraphStoreResult<()> {
        let mut connection = self.connection()?;
        redis::cmd("PING").query::<String>(&mut connection)?;
        Ok(())
    }

    pub fn upsert_node(&mut self, mut node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }

        node.labels = normalize_labels(node.labels);
        let mut connection = self.connection()?;
        let keyspace = self.keyspace.clone();
        let watch_keys = vec![keyspace.version(), keyspace.node(&node.id)];
        let mutation: GraphStoreResult<(GraphWriteResult, Option<NodeRecord>)> =
            redis::transaction(&mut connection, &watch_keys, |connection, pipe| {
                let existing = match load_node_raw_from_connection(connection, &keyspace, &node.id)
                {
                    Ok(existing) => existing,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let current_version = redis::cmd("GET")
                    .arg(keyspace.version())
                    .query::<Option<u64>>(connection)?
                    .unwrap_or_default();
                let version = current_version + 1;
                let mut next_node = node.clone();
                next_node.version = version;
                let content_hash = match next_node.checksum() {
                    Ok(hash) => hash,
                    Err(error) => return Ok(Some(Err(error))),
                };
                if next_node.parent_hashes.is_empty() {
                    if let Some(record) = existing.as_ref() {
                        let parent_hash = match &record.content_hash {
                            Some(hash) => hash.clone(),
                            None => match record.checksum() {
                                Ok(hash) => hash,
                                Err(error) => return Ok(Some(Err(error))),
                            },
                        };
                        if parent_hash != content_hash {
                            next_node.parent_hashes.push(parent_hash);
                        }
                    }
                }
                next_node.content_hash = Some(content_hash.clone());
                let checksum = match next_node.checksum() {
                    Ok(hash) => hash,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let raw = match serde_json::to_string(&next_node) {
                    Ok(raw) => raw,
                    Err(error) => {
                        return Ok(Some(Err(GraphStoreError::invalid_record(
                            "node",
                            &next_node.id,
                            error,
                        ))))
                    }
                };
                let event = match graph_event("node.upsert", &next_node.id, version, &checksum) {
                    Ok(event) => event,
                    Err(error) => return Ok(Some(Err(error))),
                };

                pipe.cmd("SET")
                    .arg(keyspace.version())
                    .arg(version)
                    .ignore()
                    .cmd("SET")
                    .arg(keyspace.node(&next_node.id))
                    .arg(raw)
                    .ignore()
                    .cmd("SADD")
                    .arg(keyspace.nodes())
                    .arg(&next_node.id)
                    .ignore()
                    .cmd("RPUSH")
                    .arg(keyspace.events())
                    .arg(event)
                    .ignore();
                if let Some(existing) = existing.as_ref() {
                    for label in &existing.labels {
                        pipe.cmd("SREM")
                            .arg(keyspace.label(label))
                            .arg(&existing.id)
                            .ignore();
                    }
                    remove_node_from_redis_property_indexes(pipe, &keyspace, existing);
                }
                if !next_node.tombstone {
                    for label in &next_node.labels {
                        pipe.cmd("SADD")
                            .arg(keyspace.labels())
                            .arg(label)
                            .ignore()
                            .cmd("SADD")
                            .arg(keyspace.label(label))
                            .arg(&next_node.id)
                            .ignore();
                    }
                    add_node_to_redis_property_indexes(pipe, &keyspace, &next_node);
                }
                let write = GraphWriteResult {
                    id: next_node.id.clone(),
                    version,
                    checksum,
                };
                match pipe.query::<Option<()>>(connection)? {
                    Some(()) => Ok(Some(Ok((write, existing)))),
                    None => Ok(None),
                }
            })?;
        let (write, existing) = mutation?;

        if let Some(existing) = existing {
            self.cleanup_empty_labels(&mut connection, &existing.labels)?;
            self.cleanup_empty_properties(
                &mut connection,
                &indexed_properties(&existing.properties),
            )?;
        }

        Ok(write)
    }

    pub fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        validate_edge_shape(&edge)?;
        let mut connection = self.connection()?;
        let keyspace = self.keyspace.clone();
        let watch_keys = vec![
            keyspace.version(),
            keyspace.edge(&edge.id),
            keyspace.node(&edge.from_id),
            keyspace.node(&edge.to_id),
        ];
        let mutation: GraphStoreResult<(GraphWriteResult, Option<EdgeRecord>)> =
            redis::transaction(&mut connection, &watch_keys, |connection, pipe| {
                let from_node =
                    match load_node_raw_from_connection(connection, &keyspace, &edge.from_id) {
                        Ok(node) => node,
                        Err(error) => return Ok(Some(Err(error))),
                    };
                if let Err(error) =
                    require_live_endpoint_record(&edge, "from", &edge.from_id, from_node.as_ref())
                {
                    return Ok(Some(Err(error)));
                }
                let to_node =
                    match load_node_raw_from_connection(connection, &keyspace, &edge.to_id) {
                        Ok(node) => node,
                        Err(error) => return Ok(Some(Err(error))),
                    };
                if let Err(error) =
                    require_live_endpoint_record(&edge, "to", &edge.to_id, to_node.as_ref())
                {
                    return Ok(Some(Err(error)));
                }
                let existing = match load_edge_raw_from_connection(connection, &keyspace, &edge.id)
                {
                    Ok(existing) => existing,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let current_version = redis::cmd("GET")
                    .arg(keyspace.version())
                    .query::<Option<u64>>(connection)?
                    .unwrap_or_default();
                let version = current_version + 1;
                let mut next_edge = edge.clone();
                next_edge.version = version;
                let content_hash = match next_edge.checksum() {
                    Ok(hash) => hash,
                    Err(error) => return Ok(Some(Err(error))),
                };
                if next_edge.parent_hashes.is_empty() {
                    if let Some(record) = existing.as_ref() {
                        let parent_hash = match &record.content_hash {
                            Some(hash) => hash.clone(),
                            None => match record.checksum() {
                                Ok(hash) => hash,
                                Err(error) => return Ok(Some(Err(error))),
                            },
                        };
                        if parent_hash != content_hash {
                            next_edge.parent_hashes.push(parent_hash);
                        }
                    }
                }
                next_edge.content_hash = Some(content_hash.clone());
                let checksum = match next_edge.checksum() {
                    Ok(hash) => hash,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let raw = match serde_json::to_string(&next_edge) {
                    Ok(raw) => raw,
                    Err(error) => {
                        return Ok(Some(Err(GraphStoreError::invalid_record(
                            "edge",
                            &next_edge.id,
                            error,
                        ))))
                    }
                };
                let event = match graph_event("edge.upsert", &next_edge.id, version, &checksum) {
                    Ok(event) => event,
                    Err(error) => return Ok(Some(Err(error))),
                };

                pipe.cmd("SET")
                    .arg(keyspace.version())
                    .arg(version)
                    .ignore()
                    .cmd("SET")
                    .arg(keyspace.edge(&next_edge.id))
                    .arg(raw)
                    .ignore()
                    .cmd("SADD")
                    .arg(keyspace.edges())
                    .arg(&next_edge.id)
                    .ignore()
                    .cmd("RPUSH")
                    .arg(keyspace.events())
                    .arg(event)
                    .ignore();
                if let Some(existing) = existing.as_ref() {
                    remove_edge_from_redis_indexes(pipe, &keyspace, existing);
                }
                if !next_edge.tombstone {
                    add_edge_to_redis_indexes(pipe, &keyspace, &next_edge);
                }
                let write = GraphWriteResult {
                    id: next_edge.id.clone(),
                    version,
                    checksum,
                };
                match pipe.query::<Option<()>>(connection)? {
                    Some(()) => Ok(Some(Ok((write, existing)))),
                    None => Ok(None),
                }
            })?;
        let (write, existing) = mutation?;

        if let Some(existing) = existing {
            self.cleanup_empty_edge_type(&mut connection, &existing.edge_type)?;
            self.cleanup_empty_adjacency_pair(
                &mut connection,
                Direction::Out,
                &existing.from_id,
                &existing.edge_type,
            )?;
            self.cleanup_empty_adjacency_pair(
                &mut connection,
                Direction::In,
                &existing.to_id,
                &existing.edge_type,
            )?;
        }

        Ok(write)
    }

    pub fn get_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        Ok(self.load_node_raw(id)?.filter(|node| !node.tombstone))
    }

    pub fn get_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        Ok(self.load_edge_raw(id)?.filter(|edge| !edge.tombstone))
    }

    pub fn node_ids_for_label(&self, label: &str) -> GraphStoreResult<Vec<String>> {
        let mut connection = self.connection()?;
        Ok(
            redis_string_set(&mut connection, self.keyspace.label(label))?
                .into_iter()
                .collect(),
        )
    }

    pub fn edge_ids_for_type(&self, edge_type: &str) -> GraphStoreResult<Vec<String>> {
        let mut connection = self.connection()?;
        Ok(
            redis_string_set(&mut connection, self.keyspace.edge_type(edge_type))?
                .into_iter()
                .collect(),
        )
    }

    pub fn node_ids_for_property(&self, key: &str, value: &Value) -> GraphStoreResult<Vec<String>> {
        let Some(token) = property_index_token(value) else {
            return Ok(Vec::new());
        };
        let mut connection = self.connection()?;
        Ok(
            redis_string_set(&mut connection, self.keyspace.property_value(key, &token))?
                .into_iter()
                .collect(),
        )
    }

    pub fn labels(&self) -> GraphStoreResult<Vec<String>> {
        let mut connection = self.connection()?;
        Ok(redis_string_set(&mut connection, self.keyspace.labels())?
            .into_iter()
            .filter(|label| {
                redis_string_set(&mut connection, self.keyspace.label(label))
                    .map(|ids| !ids.is_empty())
                    .unwrap_or(false)
            })
            .collect())
    }

    pub fn edge_types(&self) -> GraphStoreResult<Vec<String>> {
        let mut connection = self.connection()?;
        Ok(
            redis_string_set(&mut connection, self.keyspace.edge_types())?
                .into_iter()
                .filter(|edge_type| {
                    redis_string_set(&mut connection, self.keyspace.edge_type(edge_type))
                        .map(|ids| !ids.is_empty())
                        .unwrap_or(false)
                })
                .collect(),
        )
    }

    pub fn property_keys(&self) -> GraphStoreResult<Vec<String>> {
        let mut connection = self.connection()?;
        let mut keys = BTreeSet::new();
        for entry in redis_string_set(&mut connection, self.keyspace.property_index_entries())? {
            let Some((key, token)) = decode_property_pair(&entry) else {
                continue;
            };
            if !redis_string_set(&mut connection, self.keyspace.property_value(&key, &token))?
                .is_empty()
            {
                keys.insert(key);
            }
        }
        Ok(keys.into_iter().collect())
    }

    pub fn query_nodes(&self, query: NodeQuery) -> GraphStoreResult<Vec<NodeRecord>> {
        let mut candidate_ids: Option<BTreeSet<String>> = None;
        if let Some(label) = query.normalized_label() {
            merge_candidates(
                &mut candidate_ids,
                Some(self.node_ids_for_label(&label)?.into_iter().collect()),
            );
        }
        for (key, value) in &query.properties {
            let key = key.trim();
            if key.is_empty() {
                return Ok(Vec::new());
            }
            let Some(token) = property_index_token(value) else {
                return Ok(Vec::new());
            };
            let mut connection = self.connection()?;
            merge_candidates(
                &mut candidate_ids,
                Some(
                    redis_string_set(&mut connection, self.keyspace.property_value(key, &token))?
                        .into_iter()
                        .collect(),
                ),
            );
        }

        let ids = match candidate_ids {
            Some(ids) => ids,
            None => self.live_nodes()?.into_keys().collect(),
        };
        let mut nodes = Vec::new();
        for id in ids.into_iter().take(query.bounded_limit()) {
            if let Some(node) = self.get_node(&id)? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    pub fn neighbors(&self, query: NeighborQuery) -> GraphStoreResult<Vec<NeighborHit>> {
        let mut connection = self.connection()?;
        let mut edge_ids = BTreeSet::new();
        match query.edge_type {
            Some(edge_type) => {
                let key = match query.direction {
                    Direction::Out => self.keyspace.out_adjacency(&query.node_id, &edge_type),
                    Direction::In => self.keyspace.in_adjacency(&query.node_id, &edge_type),
                };
                edge_ids.extend(redis_string_set(&mut connection, key)?);
            }
            None => {
                let edge_types = redis_string_set(&mut connection, self.keyspace.edge_types())?;
                for edge_type in edge_types {
                    let key = match query.direction {
                        Direction::Out => self.keyspace.out_adjacency(&query.node_id, &edge_type),
                        Direction::In => self.keyspace.in_adjacency(&query.node_id, &edge_type),
                    };
                    edge_ids.extend(redis_string_set(&mut connection, key)?);
                }
            }
        }

        let mut hits = Vec::new();
        for edge_id in edge_ids {
            let Some(edge) = self.get_edge(&edge_id)? else {
                continue;
            };
            let node_id = match query.direction {
                Direction::Out => edge.to_id.clone(),
                Direction::In => edge.from_id.clone(),
            };
            if self.get_node(&node_id)?.is_none() {
                continue;
            }
            hits.push(NeighborHit {
                edge_id: edge.id.clone(),
                node_id,
                edge_type: edge.edge_type.clone(),
                confidence: edge.confidence,
                epistemic_type: edge.epistemic_type.clone(),
            });
        }
        Ok(hits)
    }

    pub fn stats(&self) -> GraphStoreResult<GraphStats> {
        let live_nodes = self.live_nodes()?;
        let live_edges = self.live_edges()?;
        let mut connection = self.connection()?;
        let version = redis::cmd("GET")
            .arg(self.keyspace.version())
            .query::<Option<u64>>(&mut connection)?
            .unwrap_or_default();
        Ok(GraphStats {
            version,
            nodes_total: live_nodes.len(),
            edges_total: live_edges.len(),
            labels_total: self.labels()?.len(),
            edge_types_total: self.edge_types()?.len(),
            property_keys_total: self.property_keys()?.len(),
            property_indexes_total: self.redis_indexes()?.property_index.len(),
            memory_bytes: 0,
            memory_quota_bytes: 0,
        })
    }

    pub fn verify(&self) -> GraphStoreResult<VerifyReport> {
        let live_nodes = self.live_nodes()?;
        let live_edges = self.live_edges()?;
        let mut expected = ExpectedIndexes::default();
        let mut problems = Vec::new();

        for node in live_nodes.values() {
            for label in &node.labels {
                expected
                    .label_index
                    .entry(label.clone())
                    .or_default()
                    .insert(node.id.clone());
            }
            for (key, token) in indexed_properties(&node.properties) {
                expected
                    .property_index
                    .entry((key, token))
                    .or_default()
                    .insert(node.id.clone());
            }
        }

        for edge in live_edges.values() {
            if !live_nodes.contains_key(&edge.from_id) {
                problems.push(VerifyProblem {
                    kind: "missing_from_endpoint".to_string(),
                    id: edge.id.clone(),
                    detail: format!("from endpoint {} is not a live node", edge.from_id),
                });
            }
            if !live_nodes.contains_key(&edge.to_id) {
                problems.push(VerifyProblem {
                    kind: "missing_to_endpoint".to_string(),
                    id: edge.id.clone(),
                    detail: format!("to endpoint {} is not a live node", edge.to_id),
                });
            }
            expected
                .edge_type_index
                .entry(edge.edge_type.clone())
                .or_default()
                .insert(edge.id.clone());
            expected
                .out_adjacency
                .entry((edge.from_id.clone(), edge.edge_type.clone()))
                .or_default()
                .insert(edge.id.clone());
            expected
                .in_adjacency
                .entry((edge.to_id.clone(), edge.edge_type.clone()))
                .or_default()
                .insert(edge.id.clone());
        }

        let actual = self.redis_indexes()?;
        if actual.label_index != expected.label_index {
            problems.push(VerifyProblem {
                kind: "label_index_drift".to_string(),
                id: "label_index".to_string(),
                detail: "Redis label index does not match live node labels".to_string(),
            });
        }
        if actual.edge_type_index != expected.edge_type_index {
            problems.push(VerifyProblem {
                kind: "edge_type_index_drift".to_string(),
                id: "edge_type_index".to_string(),
                detail: "Redis edge type index does not match live edge types".to_string(),
            });
        }
        if actual.property_index != expected.property_index {
            problems.push(VerifyProblem {
                kind: "property_index_drift".to_string(),
                id: "property_index".to_string(),
                detail: "Redis property index does not match live scalar node properties"
                    .to_string(),
            });
        }
        if actual.out_adjacency != expected.out_adjacency {
            problems.push(VerifyProblem {
                kind: "out_adjacency_drift".to_string(),
                id: "out_adjacency".to_string(),
                detail: "Redis out adjacency index does not match live edges".to_string(),
            });
        }
        if actual.in_adjacency != expected.in_adjacency {
            problems.push(VerifyProblem {
                kind: "in_adjacency_drift".to_string(),
                id: "in_adjacency".to_string(),
                detail: "Redis in adjacency index does not match live edges".to_string(),
            });
        }

        Ok(VerifyReport {
            ok: problems.is_empty(),
            stats: self.stats()?,
            problems,
        })
    }

    pub fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        let before = self.verify()?;
        let mut connection = self.connection()?;
        let keyspace = self.keyspace.clone();
        let watch_keys = vec![keyspace.version()];
        let rebuild: GraphStoreResult<()> =
            redis::transaction(&mut connection, &watch_keys, |connection, pipe| {
                let live_nodes = match live_nodes_from_connection(connection, &keyspace) {
                    Ok(nodes) => nodes,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let live_edges = match live_edges_from_connection(connection, &keyspace) {
                    Ok(edges) => edges,
                    Err(error) => return Ok(Some(Err(error))),
                };
                let actual = match redis_indexes_from_connection(connection, &keyspace) {
                    Ok(indexes) => indexes,
                    Err(error) => return Ok(Some(Err(error))),
                };

                pipe.cmd("DEL")
                    .arg(keyspace.labels())
                    .arg(keyspace.edge_types())
                    .arg(keyspace.property_index_entries())
                    .arg(keyspace.out_adjacency_pairs())
                    .arg(keyspace.in_adjacency_pairs())
                    .ignore();
                for label in actual.label_index.keys() {
                    pipe.cmd("DEL").arg(keyspace.label(label)).ignore();
                }
                for edge_type in actual.edge_type_index.keys() {
                    pipe.cmd("DEL").arg(keyspace.edge_type(edge_type)).ignore();
                }
                for (key, token) in actual.property_index.keys() {
                    pipe.cmd("DEL")
                        .arg(keyspace.property_value(key, token))
                        .ignore();
                }
                for (node_id, edge_type) in actual.out_adjacency.keys() {
                    pipe.cmd("DEL")
                        .arg(keyspace.out_adjacency(node_id, edge_type))
                        .ignore();
                }
                for (node_id, edge_type) in actual.in_adjacency.keys() {
                    pipe.cmd("DEL")
                        .arg(keyspace.in_adjacency(node_id, edge_type))
                        .ignore();
                }

                for node in live_nodes.values() {
                    for label in &node.labels {
                        pipe.cmd("SADD")
                            .arg(keyspace.labels())
                            .arg(label)
                            .ignore()
                            .cmd("SADD")
                            .arg(keyspace.label(label))
                            .arg(&node.id)
                            .ignore();
                    }
                    add_node_to_redis_property_indexes(pipe, &keyspace, node);
                }
                for edge in live_edges.values() {
                    add_edge_to_redis_indexes(pipe, &keyspace, edge);
                }

                match pipe.query::<Option<()>>(connection)? {
                    Some(()) => Ok(Some(Ok(()))),
                    None => Ok(None),
                }
            })?;
        rebuild?;
        let after = self.verify()?;
        Ok(GraphRebuildReport {
            repaired: !before.ok && after.ok,
            before,
            after,
        })
    }

    fn connection(&self) -> GraphStoreResult<redis::Connection> {
        Ok(self.client.get_connection()?)
    }

    fn load_node_raw(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        let mut connection = self.connection()?;
        load_node_raw_from_connection(&mut connection, &self.keyspace, id)
    }

    fn load_edge_raw(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        let mut connection = self.connection()?;
        load_edge_raw_from_connection(&mut connection, &self.keyspace, id)
    }

    fn live_nodes(&self) -> GraphStoreResult<BTreeMap<String, NodeRecord>> {
        let mut connection = self.connection()?;
        live_nodes_from_connection(&mut connection, &self.keyspace)
    }

    fn live_edges(&self) -> GraphStoreResult<BTreeMap<String, EdgeRecord>> {
        let mut connection = self.connection()?;
        live_edges_from_connection(&mut connection, &self.keyspace)
    }

    fn redis_indexes(&self) -> GraphStoreResult<ExpectedIndexes> {
        let mut connection = self.connection()?;
        redis_indexes_from_connection(&mut connection, &self.keyspace)
    }

    fn cleanup_empty_labels(
        &self,
        connection: &mut redis::Connection,
        labels: &[String],
    ) -> GraphStoreResult<()> {
        for label in labels {
            cleanup_empty_redis_set(
                connection,
                self.keyspace.label(label),
                self.keyspace.labels(),
                label,
            )?;
        }
        Ok(())
    }

    fn cleanup_empty_properties(
        &self,
        connection: &mut redis::Connection,
        properties: &BTreeMap<String, String>,
    ) -> GraphStoreResult<()> {
        for (key, token) in properties {
            cleanup_empty_redis_set(
                connection,
                self.keyspace.property_value(key, token),
                self.keyspace.property_index_entries(),
                &property_pair(key, token),
            )?;
        }
        Ok(())
    }

    fn cleanup_empty_edge_type(
        &self,
        connection: &mut redis::Connection,
        edge_type: &str,
    ) -> GraphStoreResult<()> {
        cleanup_empty_redis_set(
            connection,
            self.keyspace.edge_type(edge_type),
            self.keyspace.edge_types(),
            edge_type,
        )
    }

    fn cleanup_empty_adjacency_pair(
        &self,
        connection: &mut redis::Connection,
        direction: Direction,
        node_id: &str,
        edge_type: &str,
    ) -> GraphStoreResult<()> {
        let pair = adjacency_pair(node_id, edge_type);
        let (index_key, catalog_key) = match direction {
            Direction::Out => (
                self.keyspace.out_adjacency(node_id, edge_type),
                self.keyspace.out_adjacency_pairs(),
            ),
            Direction::In => (
                self.keyspace.in_adjacency(node_id, edge_type),
                self.keyspace.in_adjacency_pairs(),
            ),
        };
        cleanup_empty_redis_set(connection, index_key, catalog_key, &pair)
    }
}

impl GraphWrite for RedCoreGraphStore {
    // Writes go through the inherent AOF-backed durable upserts (commit_batch),
    // so persistence holds when RedCore is used as a generic GraphStore (e.g.
    // CrawlGraph::apply_to_store / the browser page-ingest seam, instead of the
    // ephemeral InMemoryGraphStore). The fully-qualified `RedCoreGraphStore::`
    // path selects the inherent method, not this trait method: no recursion.
    fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        RedCoreGraphStore::upsert_node(self, node)
    }

    fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        RedCoreGraphStore::upsert_edge(self, edge)
    }

    fn commit_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        RedCoreGraphStore::commit_batch(self, batch).map(|_| ())
    }
}

impl GraphAtomic for RedCoreGraphStore {
    fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        RedCoreGraphStore::insert_node_if_absent(self, node)
    }

    fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        RedCoreGraphStore::insert_node_if_never_existed(self, node)
    }

    fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        RedCoreGraphStore::compare_and_set_node_property(self, id, property, expected, new_value)
    }

    fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        RedCoreGraphStore::commit_conditional_batch_atomic(self, conditions, batch)
    }
}

impl GraphRead for RedCoreGraphStore {
    // Reads serve from the in-memory mirror, which `recover()` rebuilds from the
    // durable AOF on open and which every committed batch keeps current.
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        self.store.get_node(id)
    }

    fn get_node_record(&self, id: &str) -> Option<NodeRecord> {
        self.store.nodes.get(id).cloned()
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        self.store.get_node_record_including_tombstone(id)
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        self.store.has_historical_node_id(id)
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        self.store.get_edge(id)
    }

    fn get_edge_record(&self, id: &str) -> Option<EdgeRecord> {
        self.store.edges.get(id).cloned()
    }

    fn live_node_records(&self) -> Vec<NodeRecord> {
        GraphRead::live_node_records(&self.store)
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        self.store.live_edges_snapshot()
    }

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        self.store.node_ids_for_label(label)
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        self.store.edge_ids_for_type(edge_type)
    }

    fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        self.store.node_ids_for_label_folded(label)
    }

    fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        self.store.edge_ids_for_type_folded(edge_type)
    }

    fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        self.store.node_ids_for_property(key, value)
    }

    fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        self.store.node_ids_for_string_property_exact(key, raw)
    }

    fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        self.store
            .node_ids_for_string_property_prefix(key, raw_prefix)
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        self.store.query_nodes(query)
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        self.store.neighbors(query)
    }

    fn stats(&self) -> GraphStats {
        self.store.stats()
    }
}

impl GraphSnapshotSource for RedCoreGraphStore {
    fn graph_snapshot(&self) -> GraphStoreResult<GraphSnapshot> {
        Ok(RedCoreGraphStore::graph_snapshot(self))
    }
}

impl GraphVerify for RedCoreGraphStore {
    fn verify(&self) -> VerifyReport {
        self.store.verify()
    }

    fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        self.store.rebuild_indexes()
    }
}

impl GraphVectorPayloadAccess for RedCoreGraphStore {
    fn vector_payload_store(&self) -> GraphStoreResult<Arc<dyn TensorBlockPayloadStore>> {
        Ok(self.store.vector_payload_store_handle())
    }
}

impl GraphTtl for RedCoreGraphStore {
    // TTL methods route through the durable RedCore implementations above.
    // Those implementations journal TTL-bearing NodeUpsert/NodeDelete frames;
    // do not delegate writes directly to the in-memory mirror.
    fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult> {
        RedCoreGraphStore::set_node_ttl(self, id, expires_at_ms)
    }

    fn get_node_including_expired(&self, id: &str) -> Option<&NodeRecord> {
        self.store.get_node_including_expired(id)
    }

    fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize> {
        RedCoreGraphStore::purge_expired_nodes(self)
    }

    fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord> {
        RedCoreGraphStore::nodes_expiring_before(self, ts_ms, limit)
    }

    fn ttl_active_count(&self) -> usize {
        RedCoreGraphStore::ttl_active_count(self)
    }
}

impl GraphResidency for RedCoreGraphStore {
    fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        self.evict_node_residency(id)
    }

    fn readmit_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.readmit_node_residency(node)
    }

    fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        self.require_fully_resident_edges("edge residency eviction")?;
        self.store.evict_edge(id)
    }

    fn readmit_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        self.require_fully_resident_edges("edge residency readmission")?;
        self.store.readmit_edge(edge)
    }
}

#[cfg(feature = "redis-store")]
fn redis_string_set(
    connection: &mut redis::Connection,
    key: String,
) -> GraphStoreResult<BTreeSet<String>> {
    let values = redis::cmd("SMEMBERS")
        .arg(key)
        .query::<Vec<String>>(connection)?;
    Ok(values.into_iter().collect())
}

#[cfg(feature = "redis-store")]
fn live_nodes_from_connection(
    connection: &mut redis::Connection,
    keyspace: &RedisGraphKeyspace,
) -> GraphStoreResult<BTreeMap<String, NodeRecord>> {
    let node_ids = redis_string_set(connection, keyspace.nodes())?;
    let mut nodes = BTreeMap::new();
    for node_id in node_ids {
        if let Some(node) = load_node_raw_from_connection(connection, keyspace, &node_id)?
            .filter(|node| !node.tombstone)
        {
            nodes.insert(node_id, node);
        }
    }
    Ok(nodes)
}

#[cfg(feature = "redis-store")]
fn live_edges_from_connection(
    connection: &mut redis::Connection,
    keyspace: &RedisGraphKeyspace,
) -> GraphStoreResult<BTreeMap<String, EdgeRecord>> {
    let edge_ids = redis_string_set(connection, keyspace.edges())?;
    let mut edges = BTreeMap::new();
    for edge_id in edge_ids {
        if let Some(edge) = load_edge_raw_from_connection(connection, keyspace, &edge_id)?
            .filter(|edge| !edge.tombstone)
        {
            edges.insert(edge_id, edge);
        }
    }
    Ok(edges)
}

#[cfg(feature = "redis-store")]
fn redis_indexes_from_connection(
    connection: &mut redis::Connection,
    keyspace: &RedisGraphKeyspace,
) -> GraphStoreResult<ExpectedIndexes> {
    let mut indexes = ExpectedIndexes::default();
    for label in redis_string_set(connection, keyspace.labels())? {
        let node_ids = redis_string_set(connection, keyspace.label(&label))?;
        if !node_ids.is_empty() {
            indexes.label_index.insert(label, node_ids);
        }
    }
    for edge_type in redis_string_set(connection, keyspace.edge_types())? {
        let edge_ids = redis_string_set(connection, keyspace.edge_type(&edge_type))?;
        if !edge_ids.is_empty() {
            indexes.edge_type_index.insert(edge_type, edge_ids);
        }
    }
    for entry in redis_string_set(connection, keyspace.property_index_entries())? {
        let Some((key, token)) = decode_property_pair(&entry) else {
            continue;
        };
        let node_ids = redis_string_set(connection, keyspace.property_value(&key, &token))?;
        if !node_ids.is_empty() {
            indexes.property_index.insert((key, token), node_ids);
        }
    }
    for pair in redis_string_set(connection, keyspace.out_adjacency_pairs())? {
        let Some((node_id, edge_type)) = decode_adjacency_pair(&pair) else {
            continue;
        };
        let edge_ids = redis_string_set(connection, keyspace.out_adjacency(&node_id, &edge_type))?;
        if !edge_ids.is_empty() {
            indexes.out_adjacency.insert((node_id, edge_type), edge_ids);
        }
    }
    for pair in redis_string_set(connection, keyspace.in_adjacency_pairs())? {
        let Some((node_id, edge_type)) = decode_adjacency_pair(&pair) else {
            continue;
        };
        let edge_ids = redis_string_set(connection, keyspace.in_adjacency(&node_id, &edge_type))?;
        if !edge_ids.is_empty() {
            indexes.in_adjacency.insert((node_id, edge_type), edge_ids);
        }
    }
    Ok(indexes)
}

#[cfg(feature = "redis-store")]
fn load_node_raw_from_connection(
    connection: &mut redis::Connection,
    keyspace: &RedisGraphKeyspace,
    id: &str,
) -> GraphStoreResult<Option<NodeRecord>> {
    let raw = redis::cmd("GET")
        .arg(keyspace.node(id))
        .query::<Option<String>>(connection)?;
    raw.map(|value| {
        serde_json::from_str::<NodeRecord>(&value)
            .map_err(|err| GraphStoreError::invalid_record("node", id, err))
    })
    .transpose()
}

#[cfg(feature = "redis-store")]
fn load_edge_raw_from_connection(
    connection: &mut redis::Connection,
    keyspace: &RedisGraphKeyspace,
    id: &str,
) -> GraphStoreResult<Option<EdgeRecord>> {
    let raw = redis::cmd("GET")
        .arg(keyspace.edge(id))
        .query::<Option<String>>(connection)?;
    raw.map(|value| {
        serde_json::from_str::<EdgeRecord>(&value)
            .map_err(|err| GraphStoreError::invalid_record("edge", id, err))
    })
    .transpose()
}

#[cfg(feature = "redis-store")]
fn require_live_endpoint_record(
    edge: &EdgeRecord,
    endpoint: &str,
    node_id: &str,
    node: Option<&NodeRecord>,
) -> GraphStoreResult<()> {
    let Some(node) = node else {
        return Err(GraphStoreError::missing_endpoint(
            &edge.id, endpoint, node_id,
        ));
    };
    if node.tombstone {
        return Err(GraphStoreError::tombstoned_endpoint(
            &edge.id, endpoint, node_id,
        ));
    }
    Ok(())
}

#[cfg(feature = "redis-store")]
fn add_edge_to_redis_indexes(
    pipe: &mut redis::Pipeline,
    keyspace: &RedisGraphKeyspace,
    edge: &EdgeRecord,
) {
    let out_pair = adjacency_pair(&edge.from_id, &edge.edge_type);
    let in_pair = adjacency_pair(&edge.to_id, &edge.edge_type);
    pipe.cmd("SADD")
        .arg(keyspace.edge_types())
        .arg(&edge.edge_type)
        .ignore()
        .cmd("SADD")
        .arg(keyspace.edge_type(&edge.edge_type))
        .arg(&edge.id)
        .ignore()
        .cmd("SADD")
        .arg(keyspace.out_adjacency_pairs())
        .arg(out_pair)
        .ignore()
        .cmd("SADD")
        .arg(keyspace.out_adjacency(&edge.from_id, &edge.edge_type))
        .arg(&edge.id)
        .ignore()
        .cmd("SADD")
        .arg(keyspace.in_adjacency_pairs())
        .arg(in_pair)
        .ignore()
        .cmd("SADD")
        .arg(keyspace.in_adjacency(&edge.to_id, &edge.edge_type))
        .arg(&edge.id)
        .ignore();
}

#[cfg(feature = "redis-store")]
fn remove_edge_from_redis_indexes(
    pipe: &mut redis::Pipeline,
    keyspace: &RedisGraphKeyspace,
    edge: &EdgeRecord,
) {
    pipe.cmd("SREM")
        .arg(keyspace.edge_type(&edge.edge_type))
        .arg(&edge.id)
        .ignore()
        .cmd("SREM")
        .arg(keyspace.out_adjacency(&edge.from_id, &edge.edge_type))
        .arg(&edge.id)
        .ignore()
        .cmd("SREM")
        .arg(keyspace.in_adjacency(&edge.to_id, &edge.edge_type))
        .arg(&edge.id)
        .ignore();
}

#[cfg(feature = "redis-store")]
fn add_node_to_redis_property_indexes(
    pipe: &mut redis::Pipeline,
    keyspace: &RedisGraphKeyspace,
    node: &NodeRecord,
) {
    for (key, token) in indexed_properties(&node.properties) {
        pipe.cmd("SADD")
            .arg(keyspace.property_index_entries())
            .arg(property_pair(&key, &token))
            .ignore()
            .cmd("SADD")
            .arg(keyspace.property_value(&key, &token))
            .arg(&node.id)
            .ignore();
    }
}

#[cfg(feature = "redis-store")]
fn remove_node_from_redis_property_indexes(
    pipe: &mut redis::Pipeline,
    keyspace: &RedisGraphKeyspace,
    node: &NodeRecord,
) {
    for (key, token) in indexed_properties(&node.properties) {
        pipe.cmd("SREM")
            .arg(keyspace.property_value(&key, &token))
            .arg(&node.id)
            .ignore();
    }
}

#[cfg(feature = "redis-store")]
fn cleanup_empty_redis_set(
    connection: &mut redis::Connection,
    set_key: String,
    catalog_key: String,
    catalog_member: &str,
) -> GraphStoreResult<()> {
    let count = redis::cmd("SCARD")
        .arg(&set_key)
        .query::<usize>(connection)?;
    if count == 0 {
        redis::pipe()
            .atomic()
            .cmd("DEL")
            .arg(set_key)
            .ignore()
            .cmd("SREM")
            .arg(catalog_key)
            .arg(catalog_member)
            .ignore()
            .query::<()>(connection)?;
    }
    Ok(())
}

#[cfg(feature = "redis-store")]
fn graph_event(
    event_type: &str,
    id: &str,
    version: u64,
    checksum: &str,
) -> GraphStoreResult<String> {
    serde_json::to_string(&serde_json::json!({
        "type": event_type,
        "id": id,
        "version": version,
        "checksum": checksum,
    }))
    .map_err(|err| GraphStoreError::invalid_record("event", id, err))
}

#[cfg(feature = "redis-store")]
fn adjacency_pair(node_id: &str, edge_type: &str) -> String {
    serde_json::to_string(&(node_id, edge_type))
        .unwrap_or_else(|_| format!("{node_id}\t{edge_type}"))
}

#[cfg(feature = "redis-store")]
fn decode_adjacency_pair(raw: &str) -> Option<(String, String)> {
    serde_json::from_str::<(String, String)>(raw).ok()
}

#[cfg(feature = "redis-store")]
fn property_pair(key: &str, token: &str) -> String {
    serde_json::to_string(&(key, token)).unwrap_or_else(|_| format!("{key}\t{token}"))
}

#[cfg(feature = "redis-store")]
fn decode_property_pair(raw: &str) -> Option<(String, String)> {
    serde_json::from_str::<(String, String)>(raw).ok()
}

pub fn sanitize_tenant_segment(value: &str) -> String {
    let mut encoded = String::with_capacity("pct_".len() + value.len());
    encoded.push_str("pct_");
    for byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.') {
            encoded.push(*byte as char);
        } else {
            encoded.push('%');
            encoded.push(hex_digit(byte >> 4).to_ascii_uppercase());
            encoded.push(hex_digit(byte & 0x0f).to_ascii_uppercase());
        }
    }
    encoded
}

#[cfg(feature = "redis-store")]
fn encode_key_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(1 + value.len() * 2);
    encoded.push('h');
    for byte in value.as_bytes() {
        encoded.push(hex_digit(byte >> 4));
        encoded.push(hex_digit(byte & 0x0f));
    }
    encoded
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + (value - 10)) as char,
        _ => unreachable!("hex digit nibble is always <= 15"),
    }
}

#[cfg(test)]
#[path = "graph/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "graph_history_tests.rs"]
mod graph_history_tests;

#[cfg(test)]
#[path = "graph_preparation_tests.rs"]
mod graph_preparation_tests;

#[cfg(test)]
impl RedCoreGraphStore {
    /// Republish this store's journal state in the v4 layout, which predates
    /// retained history headers. The next open reports history as
    /// unavailable before this point.
    pub(crate) fn publish_v4_state_layout_for_tests(&self) -> GraphStoreResult<()> {
        let journal = self
            .chunk_store_handle()
            .expect("the v4 layout fixture needs a durable journal");
        let catalog = crate::chunk_store::RootCatalog::from_bytes(
            &journal.get(journal.read_root()?.expect("catalog"))?.unwrap(),
        )?;
        let bytes = journal
            .get(catalog.get(ROOT_NAME_REDCORE).expect("redcore root"))?
            .unwrap();
        let mut state = RedCoreJournalState::from_bytes(&bytes)?;
        state.history_root = None;
        let mut legacy = state.to_bytes()?;
        // v5 inserts its optional root immediately before the unchanged
        // mutation-log tail; removing that one byte restores v4.
        let offset = legacy.len() - state.mutation_log.encode_tail()?.len() - 1;
        assert_eq!(legacy.remove(offset), 0);
        legacy[1] = MUTATION_LOG_REDCORE_JOURNAL_STATE_VERSION;
        let address = Hash::of(&legacy);
        journal.write(&[(address, &legacy)])?;
        journal.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.into(), Some(address))]),
            CommitMeta {
                parents: vec![],
                author: "v4 fixture writer".into(),
                message: "publish preserved v4 state layout".into(),
                timestamp_unix_ms: 1,
                graph_version: state.graph_version,
            },
        )?;
        Ok(())
    }
}
