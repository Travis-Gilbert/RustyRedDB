use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::chunk_store::{ChunkStore, ChunkWriteReport, MemChunkStore};
use crate::graph::error::{is_recoverable_orphan_edge, GraphStoreError, GraphStoreResult};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::residency::{COLD_SNAPSHOT_STUB_LABEL, COLD_SNAPSHOT_STUB_PROPERTY};
use crate::graph::snapshot::GraphSnapshot;
use crate::graph_store::{unix_ms, MAX_RESIDENT_HISTORICAL_NODE_IDS};
use crate::hash::Hash;
use crate::merge_registry::MergeRegistry;
use crate::recovery_profile::RecoveryProfile;
use crate::state::stable_hash;
use crate::symbolic::canonical_json;
use crate::zerocopy::{archive_content_objects, GraphArchiveObjectBytes, GRAPH_ARCHIVE_MIME_TYPE};

pub const VERSIONED_GRAPH_PROTOCOL_VERSION: &str = "rustyred-versioned-graph-v1";
pub const GRAPH_PACK_COMPILER_VERSION: &str = "rustyred-graph-pack-compiler-v1";
pub const DEFAULT_GRAPH_BRANCH: &str = "main";

const PROLLY_MIN_ENTRIES: usize = 24;
const PROLLY_MAX_ENTRIES: usize = 60;
const PROLLY_SPLIT_THRESHOLD: u16 = 8_192;
const MAX_REPOSITORY_TREE_DEPTH: usize = 256;
const DEFAULT_PROLLY_READER_CACHE_CAPACITY: usize = 256;

pub const GRAPH_CHUNK_FORMAT_VERSION: u8 = 2;
pub const LEGACY_GRAPH_CHUNK_FORMAT_VERSION: u8 = 1;
pub const DOMAIN_OBJECT: u8 = 1;
pub const DOMAIN_LEAF: u8 = 2;
pub const DOMAIN_PARENT: u8 = 3;
pub const DOMAIN_COMMIT: u8 = 4;
/// Full-record recovery payloads. Domain 5 is reserved by `RootCatalog`.
pub const DOMAIN_GRAPH_PAYLOAD: u8 = 6;
pub const GRAPH_PAYLOAD_FORMAT_VERSION: u8 = 1;

/// Result of one whole-graph logical-tree parity audit.
///
/// [`Self::parity_holds`] is the oracle's own verdict: the audited tree is the
/// canonical tree for the entries it stores. The counts travel with the verdict
/// so a boundary can cross-check the tree against the manifest that describes it
/// without a second walk.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTreeParityAudit {
    pub tree_root: Hash,
    pub entries_total: usize,
    pub live_nodes: usize,
    pub live_edges: usize,
    pub canonical_root: Hash,
    pub canonical_entries_total: usize,
}

impl GraphTreeParityAudit {
    /// True when the audited tree is exactly the canonical tree for its own
    /// entries: same root, same entry count.
    pub fn parity_holds(&self) -> bool {
        self.tree_root == self.canonical_root && self.entries_total == self.canonical_entries_total
    }
}

/// Process-wide count of whole-graph Prolly parity audits.
///
/// The audit collects every entry of a logical tree, rebuilds the tree from
/// scratch, and compares the two, so it reads and hash-verifies every chunk of
/// the tree: O(stored entries) by construction. That makes it an oracle for a
/// boundary -- a restart, an explicit verify, a deliberate replay of a batch --
/// and never for a write. This counter exists so a test can assert exactly that:
/// read it, perform a bounded write against a large store, and assert it did not
/// move. Constructors, recovery, and explicit audits move it by design, the same
/// way recovery and constructors move `graph_memory_recompute_invocations`.
static GRAPH_PROLLY_PARITY_VALIDATIONS: AtomicU64 = AtomicU64::new(0);

/// How many whole-graph Prolly parity audits have run in this process.
pub fn graph_prolly_parity_validations() -> u64 {
    GRAPH_PROLLY_PARITY_VALIDATIONS.load(Ordering::Relaxed)
}

/// Reset [`graph_prolly_parity_validations`] to zero.
pub fn reset_graph_prolly_parity_validations() {
    GRAPH_PROLLY_PARITY_VALIDATIONS.store(0, Ordering::Relaxed);
}

pub(crate) fn note_prolly_parity_validation() {
    GRAPH_PROLLY_PARITY_VALIDATIONS.fetch_add(1, Ordering::Relaxed);
}

pub const PROLLY_VALIDATION_SAMPLE_SCALE_PPM: u32 = 1_000_000;
pub const DEFAULT_PROLLY_VALIDATION_SAMPLE_PPM: u32 = 10_000;
const PROLLY_VALIDATION_MODE_ENV: &str = "RUSTYRED_PROLLY_VALIDATE";
const PROLLY_VALIDATION_PPM_ENV: &str = "RUSTYRED_PROLLY_VALIDATE_PPM";
#[cfg(test)]
static FORCE_INCREMENTAL_VALIDATION_DIVERGENCE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphObjectKind {
    Node,
    Edge,
}

impl GraphObjectKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
        }
    }

    fn canonical_tag(&self) -> u8 {
        match self {
            Self::Node => 1,
            Self::Edge => 2,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphContentObject {
    pub key: String,
    pub kind: GraphObjectKind,
    pub hash: Hash,
    pub logical_hash: Hash,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_hashes: Vec<Hash>,
    #[serde(default)]
    pub tombstone: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTreeEntry {
    pub key: String,
    pub kind: GraphObjectKind,
    pub object_hash: Hash,
    #[serde(default)]
    pub tombstone: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTreeChild {
    pub hash: Hash,
    pub first_key: String,
    pub last_key: String,
    pub entries_total: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTreeNode {
    pub hash: Hash,
    pub level: u32,
    pub first_key: String,
    pub last_key: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<GraphTreeEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<GraphTreeChild>,
}

/// A tree node decoded from its frozen Phase-C logical preimage.
///
/// Parent preimages deliberately do not encode a level: the domain byte tells
/// readers whether a chunk is a leaf or a parent, while height is a property of
/// the path from a particular root. Keeping the stored form level-free avoids
/// fabricating metadata which did not participate in the address.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoredProllyNode {
    Leaf(Vec<GraphTreeEntry>),
    Parent(Vec<GraphTreeChild>),
}

impl StoredProllyNode {
    pub fn first_key(&self) -> &str {
        match self {
            Self::Leaf(entries) => entries.first().map_or("", |entry| entry.key.as_str()),
            Self::Parent(children) => children
                .first()
                .map_or("", |child| child.first_key.as_str()),
        }
    }

    pub fn last_key(&self) -> &str {
        match self {
            Self::Leaf(entries) => entries.last().map_or("", |entry| entry.key.as_str()),
            Self::Parent(children) => children.last().map_or("", |child| child.last_key.as_str()),
        }
    }

    pub fn entries_total(&self) -> usize {
        match self {
            Self::Leaf(entries) => entries.len(),
            Self::Parent(children) => children.iter().map(|child| child.entries_total).sum(),
        }
    }

    pub fn entries(&self) -> Option<&[GraphTreeEntry]> {
        match self {
            Self::Leaf(entries) => Some(entries),
            Self::Parent(_) => None,
        }
    }

    pub fn children(&self) -> Option<&[GraphTreeChild]> {
        match self {
            Self::Leaf(_) => None,
            Self::Parent(children) => Some(children),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProllyReadMetrics {
    /// Calls which reached the physical store rather than the reader cache.
    pub store_fetches: usize,
    pub cache_hits: usize,
    pub bytes_fetched: usize,
    pub nodes_decoded: usize,
}

#[derive(Clone)]
struct CachedProllyNode {
    /// The LRU owns the verified store bytes so the hot spine remains resident.
    #[allow(dead_code)]
    bytes: Arc<[u8]>,
    node: Arc<StoredProllyNode>,
}

/// On-demand verified Prolly-tree reader with a hard-bounded hot-spine cache.
pub struct ProllyReader<'a, S: ChunkStore + ?Sized> {
    store: &'a S,
    capacity: NonZeroUsize,
    cache: HashMap<Hash, CachedProllyNode>,
    recency: VecDeque<Hash>,
    metrics: ProllyReadMetrics,
}

impl<'a, S: ChunkStore + ?Sized> ProllyReader<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self::with_capacity(
            store,
            NonZeroUsize::new(DEFAULT_PROLLY_READER_CACHE_CAPACITY)
                .expect("default Prolly reader cache capacity is non-zero"),
        )
    }

    pub fn with_capacity(store: &'a S, capacity: NonZeroUsize) -> Self {
        Self {
            store,
            capacity,
            cache: HashMap::with_capacity(capacity.get()),
            recency: VecDeque::with_capacity(capacity.get()),
            metrics: ProllyReadMetrics::default(),
        }
    }

    pub fn read(&mut self, address: Hash) -> GraphStoreResult<Arc<StoredProllyNode>> {
        if let Some(node) = self.cache.get(&address).map(|entry| entry.node.clone()) {
            self.metrics.cache_hits = checked_reader_metric_add(
                self.metrics.cache_hits,
                1,
                "Prolly reader cache-hit count",
            )?;
            self.touch(address);
            return Ok(node);
        }

        self.metrics.store_fetches = checked_reader_metric_add(
            self.metrics.store_fetches,
            1,
            "Prolly reader store-fetch count",
        )?;
        let bytes = self.store.get(address)?.ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_chunk_missing",
                format!("Prolly tree chunk {address} is missing"),
            )
        })?;
        let node = Arc::new(decode_canonical_tree_node(address, &bytes)?);
        self.metrics.bytes_fetched = checked_reader_metric_add(
            self.metrics.bytes_fetched,
            bytes.len(),
            "Prolly reader fetched-byte count",
        )?;
        self.metrics.nodes_decoded = checked_reader_metric_add(
            self.metrics.nodes_decoded,
            1,
            "Prolly reader decoded-node count",
        )?;
        self.insert(address, bytes, node.clone());
        Ok(node)
    }

    pub fn metrics(&self) -> ProllyReadMetrics {
        self.metrics
    }

    pub fn cached_nodes(&self) -> usize {
        self.cache.len()
    }

    fn touch(&mut self, address: Hash) {
        if let Some(index) = self
            .recency
            .iter()
            .position(|candidate| *candidate == address)
        {
            self.recency.remove(index);
        }
        self.recency.push_back(address);
    }

    fn insert(&mut self, address: Hash, bytes: Arc<[u8]>, node: Arc<StoredProllyNode>) {
        while self.cache.len() >= self.capacity.get() {
            let Some(evicted) = self.recency.pop_front() else {
                break;
            };
            self.cache.remove(&evicted);
        }
        self.cache.insert(address, CachedProllyNode { bytes, node });
        self.touch(address);
    }
}

fn checked_reader_metric_add(value: usize, delta: usize, field: &str) -> GraphStoreResult<usize> {
    value.checked_add(delta).ok_or_else(|| {
        GraphStoreError::new(
            "graph_reader_metric_overflow",
            format!("{field} overflowed usize"),
        )
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphProllyTree {
    #[serde(rename = "root_hash", alias = "root")]
    pub root: Hash,
    pub entries_total: usize,
}

/// A complete graph record stored outside the frozen logical tree.
///
/// Logical tree entries keep hashing only the Phase-C semantic projection.
/// This payload is a second, independently addressed recovery record which
/// preserves operational metadata such as record versions, content hashes,
/// parent hashes, confidence, epistemic type, and provenance.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphPayloadRecord {
    Node(NodeRecord),
    Edge(EdgeRecord),
}

impl GraphPayloadRecord {
    pub fn kind(&self) -> GraphObjectKind {
        match self {
            Self::Node(_) => GraphObjectKind::Node,
            Self::Edge(_) => GraphObjectKind::Edge,
        }
    }

    pub fn key(&self) -> String {
        match self {
            Self::Node(node) => format!("node/{}", node.id),
            Self::Edge(edge) => format!("edge/{}", edge.id),
        }
    }

    pub fn version(&self) -> u64 {
        match self {
            Self::Node(node) => node.version,
            Self::Edge(edge) => edge.version,
        }
    }

    pub fn tombstone(&self) -> bool {
        match self {
            Self::Node(node) => node.tombstone,
            Self::Edge(edge) => edge.tombstone,
        }
    }

    pub fn logical_hash(&self) -> GraphStoreResult<Hash> {
        match self {
            Self::Node(node) => canonical_node_preimage(node).map(|bytes| Hash::of(&bytes)),
            Self::Edge(edge) => canonical_edge_preimage(edge).map(|bytes| Hash::of(&bytes)),
        }
    }
}

/// Canonical tree bytes transported with a pack until they are installed in a
/// repository or another [`ChunkStore`]. This is a transfer envelope, not a
/// second in-memory tree representation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTreeChunk {
    pub address: Hash,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphCommit {
    pub commit_hash: Hash,
    pub tree_hash: Hash,
    pub branch: String,
    pub parent_commits: Vec<Hash>,
    pub author: String,
    pub message: String,
    pub timestamp_unix_ms: u128,
    pub graph_version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_registry_hash: Option<Hash>,
    pub objects_total: usize,
    pub compiler_version: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphCompilerCapability {
    pub name: String,
    pub kind: String,
    pub mime_type: String,
    pub content_hash: String,
    pub body: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphPackManifest {
    #[serde(default = "legacy_graph_chunk_format_version")]
    pub chunk_format_version: u8,
    pub protocol_version: String,
    pub compiler_version: String,
    pub name: String,
    pub branch: String,
    pub commit_hash: Hash,
    pub tree_hash: Hash,
    pub graph_version: u64,
    pub nodes_total: usize,
    pub edges_total: usize,
    pub objects_total: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompiledGraphPack {
    #[serde(default = "legacy_graph_chunk_format_version")]
    pub chunk_format_version: u8,
    pub protocol_version: String,
    pub compiler_version: String,
    pub manifest: GraphPackManifest,
    pub commit: GraphCommit,
    pub tree: GraphProllyTree,
    /// True only when `tree_chunks` contains every chunk reachable from
    /// `tree.root`. Incremental packs produced against an external store carry
    /// only their committed delta and therefore require that store to chain.
    #[serde(default = "default_tree_transport_complete")]
    pub tree_transport_complete: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tree_chunks: Vec<GraphTreeChunk>,
    pub capabilities: Vec<GraphCompilerCapability>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<GraphContentObject>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub archive_objects: Vec<GraphArchiveObjectBytes>,
    /// Historical node registry captured with this committed root. This is
    /// metadata for fail-closed checkout consumers, not a graph payload kind.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub historical_node_ids: BTreeSet<String>,
    #[serde(default)]
    pub historical_node_history_complete: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphCompileOptions {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub parent_commits: Vec<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub timestamp_unix_ms: Option<u128>,
    #[serde(default = "default_include_payloads")]
    pub include_payloads: bool,
    /// Exclude `local_only` records at the substrate (SPEC-THEOREM-NODE-SYNC D5).
    /// Defaults true so no pack path can accidentally ship device-private data.
    #[serde(default = "default_exclude_local_only")]
    pub exclude_local_only: bool,
    /// Keep only synced-class domains (accumulated self). Fresh-class connected
    /// domains are stripped (SPEC-THEOREM-NODE-SYNC D1).
    #[serde(default)]
    pub sync_scope_only: bool,
}

fn default_exclude_local_only() -> bool {
    true
}

impl Default for GraphCompileOptions {
    fn default() -> Self {
        Self {
            name: None,
            branch: None,
            parent_commits: Vec::new(),
            author: None,
            message: None,
            timestamp_unix_ms: None,
            include_payloads: true,
            exclude_local_only: true,
            sync_scope_only: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphDiffEntry {
    pub key: String,
    pub kind: GraphObjectKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphVersionDiff {
    pub protocol_version: String,
    pub base_tree_hash: String,
    pub target_tree_hash: String,
    pub added: Vec<GraphDiffEntry>,
    pub removed: Vec<GraphDiffEntry>,
    pub modified: Vec<GraphDiffEntry>,
    pub unchanged: usize,
    /// Changed records as they exist at the target root. Hash-only tree diffs
    /// cannot resolve semantic labels such as terminal outcomes or standing
    /// downgrades, so snapshot diffs carry just the changed target records.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub target_nodes: BTreeMap<String, NodeRecord>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub target_edges: BTreeMap<String, EdgeRecord>,
}

impl GraphVersionDiff {
    /// Paths/keys changed between two graph commits. Git and Records share this set.
    #[must_use]
    pub fn changed_keys(&self) -> BTreeSet<String> {
        self.added
            .iter()
            .chain(self.removed.iter())
            .chain(self.modified.iter())
            .map(|entry| entry.key.clone())
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphVersionRef {
    pub name: String,
    pub commit_hash: Hash,
    pub tree_hash: Hash,
    pub graph_version: u64,
    pub updated_at_unix_ms: u128,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphVersionRepository {
    #[serde(default = "legacy_graph_chunk_format_version")]
    pub chunk_format_version: u8,
    #[serde(default = "versioned_graph_protocol_string")]
    pub protocol_version: String,
    #[serde(default)]
    pub refs: Vec<GraphVersionRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commits: Vec<GraphCommit>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub objects: BTreeMap<Hash, GraphContentObject>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tree_chunks: BTreeMap<Hash, Vec<u8>>,
    #[serde(default)]
    pub packs: Vec<CompiledGraphPack>,
}

impl Default for GraphVersionRepository {
    fn default() -> Self {
        Self {
            chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            refs: Vec::new(),
            commits: Vec::new(),
            objects: BTreeMap::new(),
            tree_chunks: BTreeMap::new(),
            packs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphRefUpdate {
    pub protocol_version: String,
    pub repository: GraphVersionRepository,
    pub reference: GraphVersionRef,
    pub pack: CompiledGraphPack,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphRefDelete {
    pub protocol_version: String,
    pub repository: GraphVersionRepository,
    pub reference: GraphVersionRef,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IncrementalGraphPack {
    pub protocol_version: String,
    pub pack: CompiledGraphPack,
    pub changed_object_keys: Vec<String>,
    pub changed_tree_nodes: usize,
    pub reused_tree_nodes: usize,
    /// Per-commit structural-sharing accounting for this incremental commit
    /// (the genuine O(changed + log n) work + the persisted delta). Defaulted
    /// for backward compatibility with packs serialized before this field.
    #[serde(default)]
    pub commit_cost: CommitCost,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CommitCost {
    pub changed_objects: usize,
    pub changed_tree_nodes: usize,
    pub reused_tree_nodes: usize,
    pub chunks_written: usize,
    pub bytes_written: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphRefConflict {
    pub protocol_version: String,
    pub branch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_commit_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_commit_hash: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphVersionLog {
    pub protocol_version: String,
    pub target: String,
    pub commits: Vec<GraphCommit>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphCheckoutResult {
    pub protocol_version: String,
    pub target: String,
    pub commit: GraphCommit,
    pub tree: GraphProllyTree,
    pub snapshot: GraphSnapshot,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMergeStrategy {
    AutoConfidence,
    PreferOurs,
    PreferTheirs,
    Manual,
}

impl Default for GraphMergeStrategy {
    fn default() -> Self {
        Self::AutoConfidence
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphMergeOptions {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub parent_commits: Vec<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub timestamp_unix_ms: Option<u128>,
    #[serde(default = "default_include_payloads")]
    pub include_payloads: bool,
    #[serde(default)]
    pub strategy: GraphMergeStrategy,
    #[serde(default)]
    pub min_confidence_delta: f64,
}

impl Default for GraphMergeOptions {
    fn default() -> Self {
        Self {
            name: None,
            branch: None,
            parent_commits: Vec::new(),
            author: None,
            message: None,
            timestamp_unix_ms: None,
            include_payloads: true,
            strategy: GraphMergeStrategy::AutoConfidence,
            min_confidence_delta: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMergeSide {
    Ours,
    Theirs,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphMergeResolution {
    pub key: String,
    pub kind: GraphObjectKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<GraphMergeSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ours_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theirs_hash: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphMergeConflict {
    pub key: String,
    pub kind: GraphObjectKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ours_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theirs_hash: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphMergeResult {
    pub protocol_version: String,
    pub status: String,
    pub base_tree_hash: String,
    pub ours_tree_hash: String,
    pub theirs_tree_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_tree_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_snapshot: Option<GraphSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_pack: Option<CompiledGraphPack>,
    pub resolved: Vec<GraphMergeResolution>,
    pub conflicts: Vec<GraphMergeConflict>,
}

fn default_include_payloads() -> bool {
    true
}

fn default_tree_transport_complete() -> bool {
    true
}

fn versioned_graph_protocol_string() -> String {
    VERSIONED_GRAPH_PROTOCOL_VERSION.to_string()
}

fn legacy_graph_chunk_format_version() -> u8 {
    LEGACY_GRAPH_CHUNK_FORMAT_VERSION
}

#[derive(Deserialize, Serialize)]
struct GraphChunkFormatDiscriminator {
    #[serde(default = "legacy_graph_chunk_format_version")]
    chunk_format_version: u8,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphPackV1 {
    #[serde(default = "legacy_graph_chunk_format_version")]
    chunk_format_version: u8,
    protocol_version: String,
    compiler_version: String,
    manifest: LegacyGraphPackManifestV1,
    commit: LegacyGraphCommitV1,
    tree: LegacyGraphProllyTreeV1,
    capabilities: Vec<GraphCompilerCapability>,
    #[serde(default)]
    objects: Vec<LegacyGraphContentObjectV1>,
    #[serde(default)]
    archive_objects: Vec<LegacyGraphArchiveObjectBytesV1>,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphPackManifestV1 {
    protocol_version: String,
    compiler_version: String,
    name: String,
    branch: String,
    commit_hash: String,
    tree_hash: String,
    graph_version: u64,
    nodes_total: usize,
    edges_total: usize,
    objects_total: usize,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphCommitV1 {
    commit_hash: String,
    tree_hash: String,
    branch: String,
    #[serde(default)]
    parent_commits: Vec<String>,
    author: String,
    message: String,
    timestamp_unix_ms: u128,
    graph_version: u64,
    objects_total: usize,
    compiler_version: String,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphProllyTreeV1 {
    root_hash: String,
    root_level: u32,
    entries_total: usize,
    nodes: Vec<LegacyGraphTreeNodeV1>,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphTreeNodeV1 {
    hash: String,
    level: u32,
    first_key: String,
    last_key: String,
    #[serde(default)]
    entries: Vec<LegacyGraphTreeEntryV1>,
    #[serde(default)]
    children: Vec<LegacyGraphTreeChildV1>,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphTreeEntryV1 {
    key: String,
    kind: GraphObjectKind,
    object_hash: String,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphTreeChildV1 {
    hash: String,
    first_key: String,
    last_key: String,
    entries_total: usize,
}

#[derive(Clone, Deserialize, Serialize)]
struct LegacyGraphContentObjectV1 {
    key: String,
    kind: GraphObjectKind,
    hash: String,
    #[serde(default)]
    logical_hash: Option<String>,
    #[serde(default)]
    parent_hashes: Vec<String>,
    payload: Option<Value>,
}

#[derive(Deserialize, Serialize)]
struct LegacyGraphArchiveObjectBytesV1 {
    key: String,
    kind: GraphObjectKind,
    hash: String,
    format_version: u32,
    mime_type: String,
    bytes: Vec<u8>,
}

// Frozen Phase-C v1 rkyv schema. These definitions intentionally duplicate
// the historical writer instead of reusing the current archive types: adding
// tombstone fields changed the binary layout in v2/v3.
#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
struct LegacyGraphArchiveEnvelopeV1 {
    format_version: u32,
    body: LegacyGraphArchiveBodyV1,
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
enum LegacyGraphArchiveBodyV1 {
    ContentObject(LegacyGraphArchiveContentObjectV1),
    TreeNode(LegacyGraphArchiveTreeNodeV1),
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
struct LegacyGraphArchiveContentObjectV1 {
    key: String,
    kind: String,
    logical_hash: String,
    parent_hashes: Vec<String>,
    payload_json: Option<String>,
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
struct LegacyGraphArchiveTreeEntryV1 {
    key: String,
    kind: String,
    object_hash: String,
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
struct LegacyGraphArchiveTreeChildV1 {
    hash: String,
    first_key: String,
    last_key: String,
    entries_total: usize,
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
struct LegacyGraphArchiveTreeNodeV1 {
    hash: String,
    level: u32,
    first_key: String,
    last_key: String,
    entries: Vec<LegacyGraphArchiveTreeEntryV1>,
    children: Vec<LegacyGraphArchiveTreeChildV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyGraphPackMigration {
    pub old_commit: Hash,
    pub new_pack: CompiledGraphPack,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyGraphObjectMigration {
    pub old_hash: Hash,
    pub new_object: GraphContentObject,
}

/// Read the persisted graph-chunk format without decoding the whole pack.
/// Packs written before the discriminator existed are deliberately v1.
pub fn graph_pack_chunk_format(bytes: &[u8]) -> GraphStoreResult<u8> {
    serde_json::from_slice::<GraphChunkFormatDiscriminator>(bytes)
        .map(|probe| probe.chunk_format_version)
        .map_err(|error| {
            GraphStoreError::new(
                "graph_pack_format_decode",
                format!("could not read graph pack format discriminator: {error}"),
            )
        })
}

/// Validate a frozen v1 pack and return its old-format parent identities.
/// This read-only probe lets a cold-store migrator plan the whole parent chain
/// before publishing any replacement pack.
pub fn legacy_graph_pack_v1_parents(
    bytes: &[u8],
    expected_old_commit: Hash,
) -> GraphStoreResult<Vec<Hash>> {
    let legacy: LegacyGraphPackV1 = serde_json::from_slice(bytes).map_err(|error| {
        GraphStoreError::new(
            "legacy_graph_pack_decode",
            format!("could not decode legacy v1 graph pack: {error}"),
        )
    })?;
    if legacy.chunk_format_version != LEGACY_GRAPH_CHUNK_FORMAT_VERSION {
        return Err(GraphStoreError::new(
            "legacy_graph_pack_format_mismatch",
            format!(
                "migration expects graph chunk format {}, found {}",
                LEGACY_GRAPH_CHUNK_FORMAT_VERSION, legacy.chunk_format_version
            ),
        ));
    }
    validate_legacy_graph_pack_identity(&legacy, expected_old_commit)?;
    legacy
        .commit
        .parent_commits
        .iter()
        .map(|parent| {
            parse_legacy_hash(
                parent,
                "legacy_graph_parent_hash_invalid",
                "legacy parent commit",
            )
        })
        .collect()
}

/// Compile a persisted v1 pack into the one canonical v2 format.
///
/// Callers migrating history pass the already-migrated v2 address for each v1
/// parent. This keeps lineage explicit while allowing the intentional format
/// bump to change every commit address exactly once.
pub fn migrate_legacy_graph_pack_v1(
    bytes: &[u8],
    expected_old_commit: Hash,
    migrated_parent_commits: &BTreeMap<String, Hash>,
) -> GraphStoreResult<LegacyGraphPackMigration> {
    let legacy: LegacyGraphPackV1 = serde_json::from_slice(bytes).map_err(|error| {
        GraphStoreError::new(
            "legacy_graph_pack_decode",
            format!("could not decode legacy v1 graph pack: {error}"),
        )
    })?;
    if legacy.chunk_format_version != LEGACY_GRAPH_CHUNK_FORMAT_VERSION {
        return Err(GraphStoreError::new(
            "legacy_graph_pack_format_mismatch",
            format!(
                "migration expects graph chunk format {}, found {}",
                LEGACY_GRAPH_CHUNK_FORMAT_VERSION, legacy.chunk_format_version
            ),
        ));
    }
    validate_legacy_graph_pack_identity(&legacy, expected_old_commit)?;

    let parent_commits = legacy
        .commit
        .parent_commits
        .iter()
        .map(|legacy_parent| {
            migrated_parent_commits
                .get(legacy_parent)
                .copied()
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "legacy_graph_parent_not_migrated",
                        format!("legacy parent {legacy_parent} must be migrated before its child"),
                    )
                })
        })
        .collect::<GraphStoreResult<Vec<_>>>()?;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for object in legacy.objects {
        let payload = object.payload.ok_or_else(|| {
            GraphStoreError::new(
                "legacy_graph_pack_missing_payload",
                format!(
                    "legacy graph object {} has no migration payload",
                    object.key
                ),
            )
        })?;
        match object.kind {
            GraphObjectKind::Node => {
                let node: NodeRecord = serde_json::from_value(payload).map_err(|error| {
                    GraphStoreError::new(
                        "legacy_graph_pack_payload_decode",
                        format!("legacy node {} is invalid: {error}", object.key),
                    )
                })?;
                if object.key != format!("node/{}", node.id) {
                    return Err(GraphStoreError::new(
                        "legacy_graph_pack_payload_mismatch",
                        format!(
                            "legacy node key {} does not match payload {}",
                            object.key, node.id
                        ),
                    ));
                }
                nodes.push(node);
            }
            GraphObjectKind::Edge => {
                let edge: EdgeRecord = serde_json::from_value(payload).map_err(|error| {
                    GraphStoreError::new(
                        "legacy_graph_pack_payload_decode",
                        format!("legacy edge {} is invalid: {error}", object.key),
                    )
                })?;
                if object.key != format!("edge/{}", edge.id) {
                    return Err(GraphStoreError::new(
                        "legacy_graph_pack_payload_mismatch",
                        format!(
                            "legacy edge key {} does not match payload {}",
                            object.key, edge.id
                        ),
                    ));
                }
                edges.push(edge);
            }
        }
    }
    nodes.sort_by(|left, right| left.id.cmp(&right.id));
    edges.sort_by(|left, right| left.id.cmp(&right.id));
    let snapshot = GraphSnapshot {
        version: legacy.commit.graph_version,
        historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
        historical_node_history_complete: false,
        nodes,
        edges,
    };
    let new_pack = compile_graph_pack(
        &snapshot,
        GraphCompileOptions {
            name: Some(legacy.manifest.name),
            branch: Some(legacy.commit.branch),
            parent_commits: parent_commits
                .into_iter()
                .map(|hash| hash.to_prefixed())
                .collect(),
            author: Some(legacy.commit.author),
            message: Some(legacy.commit.message),
            timestamp_unix_ms: Some(legacy.commit.timestamp_unix_ms),
            include_payloads: true,
            ..GraphCompileOptions::default()
        },
    )?;
    Ok(LegacyGraphPackMigration {
        old_commit: expected_old_commit,
        new_pack,
    })
}

/// Recompile one authentic standalone v1 cold object while returning the
/// identity change explicitly. The old address is the SHA-256 of the frozen
/// v1 rkyv envelope, whereas the new address is the canonical logical object
/// preimage introduced by graph chunk format v2.
pub fn migrate_legacy_graph_content_object_v1(
    bytes: &[u8],
    expected_old_hash: Hash,
) -> GraphStoreResult<LegacyGraphObjectMigration> {
    let legacy: LegacyGraphContentObjectV1 = serde_json::from_slice(bytes).map_err(|error| {
        GraphStoreError::new(
            "legacy_graph_object_decode",
            format!("could not decode legacy v1 graph object: {error}"),
        )
    })?;
    validate_legacy_graph_object_identity(&legacy, expected_old_hash, None)?;
    Ok(LegacyGraphObjectMigration {
        old_hash: expected_old_hash,
        new_object: migrate_legacy_graph_object_payload(&legacy)?,
    })
}

fn validate_legacy_graph_pack_identity(
    legacy: &LegacyGraphPackV1,
    expected_old_commit: Hash,
) -> GraphStoreResult<()> {
    if legacy.protocol_version != VERSIONED_GRAPH_PROTOCOL_VERSION
        || legacy.compiler_version != GRAPH_PACK_COMPILER_VERSION
        || legacy.manifest.protocol_version != VERSIONED_GRAPH_PROTOCOL_VERSION
        || legacy.manifest.compiler_version != GRAPH_PACK_COMPILER_VERSION
        || legacy.commit.compiler_version != GRAPH_PACK_COMPILER_VERSION
    {
        return Err(GraphStoreError::new(
            "legacy_graph_pack_protocol_mismatch",
            "legacy pack does not identify the frozen v1 graph protocol and compiler",
        ));
    }

    let stored_commit = parse_legacy_hash(
        &legacy.commit.commit_hash,
        "legacy_graph_commit_hash_invalid",
        "legacy commit hash",
    )?;
    if stored_commit != expected_old_commit {
        return Err(GraphStoreError::new(
            "legacy_graph_commit_hash_mismatch",
            format!(
                "legacy pack stores commit {stored_commit}, but was requested as {expected_old_commit}"
            ),
        ));
    }

    validate_legacy_pack_metadata(legacy)?;
    validate_legacy_graph_tree(legacy)?;
    let recomputed_commit = legacy_v1_commit_hash(&legacy.commit)?;
    if recomputed_commit != stored_commit {
        return Err(GraphStoreError::new(
            "legacy_graph_commit_hash_mismatch",
            format!(
                "legacy commit preimage hashes to {recomputed_commit}, expected {stored_commit}"
            ),
        ));
    }

    if legacy.objects.len() != legacy.archive_objects.len() {
        return Err(GraphStoreError::new(
            "legacy_graph_archive_set_mismatch",
            format!(
                "legacy pack has {} semantic objects but {} physical archives",
                legacy.objects.len(),
                legacy.archive_objects.len()
            ),
        ));
    }

    let mut archives = BTreeMap::<(String, String), &LegacyGraphArchiveObjectBytesV1>::new();
    for archive in &legacy.archive_objects {
        if archive.format_version != 1 || archive.mime_type != GRAPH_ARCHIVE_MIME_TYPE {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_format_mismatch",
                format!(
                    "legacy archive {} must use format 1 and MIME {GRAPH_ARCHIVE_MIME_TYPE}",
                    archive.key
                ),
            ));
        }
        let stored_hash = parse_legacy_hash(
            &archive.hash,
            "legacy_graph_archive_hash_invalid",
            "legacy archive hash",
        )?;
        let actual_hash = Hash::of(&archive.bytes);
        if stored_hash != actual_hash {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_hash_mismatch",
                format!(
                    "legacy archive {} bytes hash to {actual_hash}, expected {stored_hash}",
                    archive.key
                ),
            ));
        }
        let identity = (archive.kind.as_str().to_string(), archive.key.clone());
        if archives.insert(identity, archive).is_some() {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_duplicate",
                format!("legacy pack repeats archive identity {}", archive.key),
            ));
        }
    }

    let mut objects = BTreeMap::new();
    for object in &legacy.objects {
        let identity = (object.kind.as_str().to_string(), object.key.clone());
        let Some(archive) = archives.remove(&identity) else {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_missing",
                format!(
                    "legacy object {} has no matching physical archive",
                    object.key
                ),
            ));
        };
        let object_hash = parse_legacy_hash(
            &object.hash,
            "legacy_graph_object_hash_invalid",
            "legacy object hash",
        )?;
        if object.hash != archive.hash {
            return Err(GraphStoreError::new(
                "legacy_graph_object_archive_mismatch",
                format!(
                    "legacy object {} names physical hash {}, archive names {}",
                    object.key, object.hash, archive.hash
                ),
            ));
        }
        validate_legacy_graph_object_identity(object, object_hash, Some(&archive.bytes))?;
        if objects.insert(identity, object.hash.clone()).is_some() {
            return Err(GraphStoreError::new(
                "legacy_graph_object_duplicate",
                format!("legacy pack repeats object identity {}", object.key),
            ));
        }
    }
    if let Some((_, extra)) = archives.into_iter().next() {
        return Err(GraphStoreError::new(
            "legacy_graph_archive_orphan",
            format!(
                "legacy archive {} has no matching semantic object",
                extra.key
            ),
        ));
    }

    let tree_objects = legacy
        .tree
        .nodes
        .iter()
        .filter(|node| node.level == 0)
        .flat_map(|node| node.entries.iter())
        .map(|entry| {
            (
                (entry.kind.as_str().to_string(), entry.key.clone()),
                entry.object_hash.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if tree_objects != objects {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_object_set_mismatch",
            "legacy tree leaves do not exactly bind the semantic object set",
        ));
    }
    Ok(())
}

fn validate_legacy_pack_metadata(legacy: &LegacyGraphPackV1) -> GraphStoreResult<()> {
    let node_count = legacy
        .objects
        .iter()
        .filter(|object| object.kind == GraphObjectKind::Node)
        .count();
    let edge_count = legacy.objects.len().saturating_sub(node_count);
    if legacy.manifest.branch != legacy.commit.branch
        || legacy.manifest.commit_hash != legacy.commit.commit_hash
        || legacy.manifest.tree_hash != legacy.commit.tree_hash
        || legacy.manifest.tree_hash != legacy.tree.root_hash
        || legacy.manifest.graph_version != legacy.commit.graph_version
        || legacy.manifest.nodes_total != node_count
        || legacy.manifest.edges_total != edge_count
        || legacy.manifest.objects_total != legacy.objects.len()
        || legacy.commit.objects_total != legacy.objects.len()
        || legacy.tree.entries_total != legacy.objects.len()
    {
        return Err(GraphStoreError::new(
            "legacy_graph_pack_metadata_mismatch",
            "legacy manifest, commit, tree, and object counts do not agree",
        ));
    }
    let mut manifest_body = serde_json::to_value(&legacy.manifest).map_err(|error| {
        GraphStoreError::new(
            "legacy_graph_capability_encode",
            format!("could not encode legacy manifest capability: {error}"),
        )
    })?;
    let mut validator_body = json!({
        "validator": "rustyred.verify_tree_root",
        "tree_hash": legacy.tree.root_hash.clone(),
        "objects_total": legacy.objects.len(),
        "protocol_version": VERSIONED_GRAPH_PROTOCOL_VERSION,
    });
    // V1 capabilities hashed Value objects with sorted map keys. Preserve that
    // format even when another consumer enables serde_json/preserve_order;
    // the separately frozen struct-field preimages must keep their own order.
    manifest_body.sort_all_objects();
    validator_body.sort_all_objects();
    let expected_capabilities = [
        (
            format!("{}.manifest", legacy.manifest.name),
            "graph_manifest",
            &manifest_body,
        ),
        (
            format!("{}.verify_tree_root", legacy.manifest.name),
            "validator",
            &validator_body,
        ),
    ];
    if legacy.capabilities.len() != expected_capabilities.len() {
        return Err(GraphStoreError::new(
            "legacy_graph_capability_set_mismatch",
            "legacy pack does not carry the two canonical v1 capabilities",
        ));
    }
    for (actual, (name, kind, body)) in legacy.capabilities.iter().zip(expected_capabilities) {
        let expected_hash = legacy_v1_json_hash(body)?.to_prefixed();
        if actual.name != name
            || actual.kind != kind
            || actual.mime_type != "application/json"
            || &actual.body != body
            || actual.content_hash != expected_hash
        {
            return Err(GraphStoreError::new(
                "legacy_graph_capability_mismatch",
                format!(
                    "legacy capability {} is not authentic v1 output",
                    actual.name
                ),
            ));
        }
    }
    Ok(())
}

fn validate_legacy_graph_tree(legacy: &LegacyGraphPackV1) -> GraphStoreResult<()> {
    let mut nodes = BTreeMap::new();
    for node in &legacy.tree.nodes {
        let recomputed = legacy_v1_tree_node_hash(node)?;
        let stored = parse_legacy_hash(
            &node.hash,
            "legacy_graph_tree_hash_invalid",
            "legacy tree node hash",
        )?;
        if recomputed != stored {
            return Err(GraphStoreError::new(
                "legacy_graph_tree_hash_mismatch",
                format!("legacy tree node {} hashes to {recomputed}", node.hash),
            ));
        }
        if (node.level == 0 && !node.children.is_empty())
            || (node.level > 0 && !node.entries.is_empty())
        {
            return Err(GraphStoreError::new(
                "legacy_graph_tree_shape_invalid",
                format!("legacy tree node {} mixes leaf and parent shape", node.hash),
            ));
        }
        if nodes.insert(node.hash.clone(), node).is_some() {
            return Err(GraphStoreError::new(
                "legacy_graph_tree_duplicate",
                format!("legacy tree repeats node {}", node.hash),
            ));
        }
    }
    let Some(root) = nodes.get(&legacy.tree.root_hash).copied() else {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_root_missing",
            format!("legacy tree root {} is absent", legacy.tree.root_hash),
        ));
    };
    if root.level != legacy.tree.root_level {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_root_mismatch",
            "legacy tree root level does not match its root descriptor",
        ));
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let entries = validate_legacy_tree_subtree(root, &nodes, 0, &mut visiting, &mut visited)?;
    if entries != legacy.tree.entries_total || visited.len() != nodes.len() {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_reachability_mismatch",
            "legacy tree contains an unreachable node or the wrong entry total",
        ));
    }
    Ok(())
}

fn validate_legacy_tree_subtree(
    node: &LegacyGraphTreeNodeV1,
    nodes: &BTreeMap<String, &LegacyGraphTreeNodeV1>,
    depth: usize,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> GraphStoreResult<usize> {
    if depth > MAX_REPOSITORY_TREE_DEPTH {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_depth_exceeded",
            format!(
                "legacy tree depth exceeds {MAX_REPOSITORY_TREE_DEPTH} at {}",
                node.hash
            ),
        ));
    }
    if !visiting.insert(node.hash.clone()) {
        return Err(GraphStoreError::new(
            "legacy_graph_tree_cycle",
            format!("legacy tree cycle reaches {}", node.hash),
        ));
    }
    let entries = if node.level == 0 {
        let first = node.entries.first().map_or("", |entry| entry.key.as_str());
        let last = node.entries.last().map_or("", |entry| entry.key.as_str());
        if first != node.first_key || last != node.last_key {
            return Err(GraphStoreError::new(
                "legacy_graph_tree_range_mismatch",
                format!("legacy leaf {} carries the wrong key range", node.hash),
            ));
        }
        node.entries.len()
    } else {
        let mut total = 0usize;
        for child in &node.children {
            let Some(child_node) = nodes.get(&child.hash).copied() else {
                return Err(GraphStoreError::new(
                    "legacy_graph_tree_child_missing",
                    format!("legacy tree child {} is absent", child.hash),
                ));
            };
            if child_node.level.saturating_add(1) != node.level
                || child.first_key != child_node.first_key
                || child.last_key != child_node.last_key
            {
                return Err(GraphStoreError::new(
                    "legacy_graph_tree_child_mismatch",
                    format!("legacy child descriptor {} is inconsistent", child.hash),
                ));
            }
            let child_entries = validate_legacy_tree_subtree(
                child_node,
                nodes,
                depth.saturating_add(1),
                visiting,
                visited,
            )?;
            if child.entries_total != child_entries {
                return Err(GraphStoreError::new(
                    "legacy_graph_tree_child_mismatch",
                    format!("legacy child {} has the wrong entry total", child.hash),
                ));
            }
            total = total.checked_add(child_entries).ok_or_else(|| {
                GraphStoreError::new(
                    "legacy_graph_tree_count_overflow",
                    "legacy tree entry total overflowed usize",
                )
            })?;
        }
        let first = node
            .children
            .first()
            .map_or("", |child| child.first_key.as_str());
        let last = node
            .children
            .last()
            .map_or("", |child| child.last_key.as_str());
        if first != node.first_key || last != node.last_key {
            return Err(GraphStoreError::new(
                "legacy_graph_tree_range_mismatch",
                format!("legacy parent {} carries the wrong key range", node.hash),
            ));
        }
        total
    };
    visiting.remove(&node.hash);
    visited.insert(node.hash.clone());
    Ok(entries)
}

fn validate_legacy_graph_object_identity(
    object: &LegacyGraphContentObjectV1,
    expected_old_hash: Hash,
    archived_bytes: Option<&[u8]>,
) -> GraphStoreResult<()> {
    let stored_hash = parse_legacy_hash(
        &object.hash,
        "legacy_graph_object_hash_invalid",
        "legacy object hash",
    )?;
    if stored_hash != expected_old_hash {
        return Err(GraphStoreError::new(
            "legacy_graph_object_hash_mismatch",
            format!("legacy object stores {stored_hash}, but was requested as {expected_old_hash}"),
        ));
    }
    let body = legacy_v1_archive_body(object)?;
    let reproduced = rkyv::to_bytes::<rkyv::rancor::Error>(&LegacyGraphArchiveEnvelopeV1 {
        format_version: 1,
        body: LegacyGraphArchiveBodyV1::ContentObject(body),
    })
    .map_err(|error| {
        GraphStoreError::new(
            "legacy_graph_archive_encode",
            format!("could not reproduce frozen v1 archive: {error}"),
        )
    })?;
    if Hash::of(&reproduced) != expected_old_hash {
        return Err(GraphStoreError::new(
            "legacy_graph_archive_hash_mismatch",
            format!(
                "legacy object {} does not reproduce its frozen v1 archive address",
                object.key
            ),
        ));
    }
    if let Some(bytes) = archived_bytes {
        if bytes != reproduced.as_slice() {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_semantic_mismatch",
                format!(
                    "legacy archive {} does not encode the accompanying semantic payload",
                    object.key
                ),
            ));
        }
        let decoded = rkyv::from_bytes::<LegacyGraphArchiveEnvelopeV1, rkyv::rancor::Error>(bytes)
            .map_err(|error| {
                GraphStoreError::new(
                    "legacy_graph_archive_decode",
                    format!(
                        "legacy archive {} is not valid v1 rkyv: {error}",
                        object.key
                    ),
                )
            })?;
        if decoded.format_version != 1
            || !matches!(decoded.body, LegacyGraphArchiveBodyV1::ContentObject(_))
        {
            return Err(GraphStoreError::new(
                "legacy_graph_archive_format_mismatch",
                format!("legacy archive {} is not a v1 content object", object.key),
            ));
        }
    }
    Ok(())
}

fn legacy_v1_archive_body(
    object: &LegacyGraphContentObjectV1,
) -> GraphStoreResult<LegacyGraphArchiveContentObjectV1> {
    let logical_hash = object.logical_hash.clone().ok_or_else(|| {
        GraphStoreError::new(
            "legacy_graph_object_logical_hash_missing",
            format!("legacy object {} has no logical hash", object.key),
        )
    })?;
    let payload_json = object
        .payload
        .as_ref()
        .map(canonical_json)
        .transpose()
        .map_err(|message| GraphStoreError::new("legacy_graph_payload_canonicalize", message))?;
    Ok(LegacyGraphArchiveContentObjectV1 {
        key: object.key.clone(),
        kind: object.kind.as_str().to_string(),
        logical_hash,
        parent_hashes: object.parent_hashes.clone(),
        payload_json,
    })
}

fn migrate_legacy_graph_object_payload(
    object: &LegacyGraphContentObjectV1,
) -> GraphStoreResult<GraphContentObject> {
    let payload = object.payload.clone().ok_or_else(|| {
        GraphStoreError::new(
            "legacy_graph_pack_missing_payload",
            format!(
                "legacy graph object {} has no migration payload",
                object.key
            ),
        )
    })?;
    let migrated = match &object.kind {
        GraphObjectKind::Node => {
            let node: NodeRecord = serde_json::from_value(payload).map_err(|error| {
                GraphStoreError::new(
                    "legacy_graph_pack_payload_decode",
                    format!("legacy node {} is invalid: {error}", object.key),
                )
            })?;
            if object.key != format!("node/{}", node.id) {
                return Err(GraphStoreError::new(
                    "legacy_graph_pack_payload_mismatch",
                    format!(
                        "legacy node key {} does not match payload {}",
                        object.key, node.id
                    ),
                ));
            }
            node_to_content_object(&node)?
        }
        GraphObjectKind::Edge => {
            let edge: EdgeRecord = serde_json::from_value(payload).map_err(|error| {
                GraphStoreError::new(
                    "legacy_graph_pack_payload_decode",
                    format!("legacy edge {} is invalid: {error}", object.key),
                )
            })?;
            if object.key != format!("edge/{}", edge.id) {
                return Err(GraphStoreError::new(
                    "legacy_graph_pack_payload_mismatch",
                    format!(
                        "legacy edge key {} does not match payload {}",
                        object.key, edge.id
                    ),
                ));
            }
            edge_to_content_object(&edge)?
        }
    };
    let legacy_logical = parse_legacy_hash(
        object.logical_hash.as_deref().ok_or_else(|| {
            GraphStoreError::new(
                "legacy_graph_object_logical_hash_missing",
                format!("legacy object {} has no logical hash", object.key),
            )
        })?,
        "legacy_graph_object_logical_hash_invalid",
        "legacy logical hash",
    )?;
    if migrated.logical_hash != legacy_logical {
        return Err(GraphStoreError::new(
            "legacy_graph_object_logical_hash_mismatch",
            format!(
                "legacy object {} names logical hash {legacy_logical}, but its payload yields {}",
                object.key, migrated.logical_hash
            ),
        ));
    }
    Ok(migrated)
}

fn parse_legacy_hash(value: &str, code: &'static str, field: &str) -> GraphStoreResult<Hash> {
    Hash::parse_prefixed(value).map_err(|error| {
        GraphStoreError::new(code, format!("{field} is invalid: {}", error.message))
    })
}

fn legacy_v1_tree_node_hash(node: &LegacyGraphTreeNodeV1) -> GraphStoreResult<Hash> {
    #[derive(Serialize)]
    struct LegacyTreeNodeHashInput<'a> {
        protocol_version: &'static str,
        level: u32,
        first_key: &'a str,
        last_key: &'a str,
        entries: &'a [LegacyGraphTreeEntryV1],
        children: &'a [LegacyGraphTreeChildV1],
    }
    legacy_v1_json_hash(&LegacyTreeNodeHashInput {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION,
        level: node.level,
        first_key: &node.first_key,
        last_key: &node.last_key,
        entries: &node.entries,
        children: &node.children,
    })
}

fn legacy_v1_commit_hash(commit: &LegacyGraphCommitV1) -> GraphStoreResult<Hash> {
    #[derive(Serialize)]
    struct LegacyCommitHashInput<'a> {
        protocol_version: &'static str,
        tree_hash: &'a str,
        branch: &'a str,
        parent_commits: &'a [String],
        author: &'a str,
        message: &'a str,
        timestamp_unix_ms: u128,
        graph_version: u64,
        objects_total: usize,
        compiler_version: &'static str,
    }
    legacy_v1_json_hash(&LegacyCommitHashInput {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION,
        tree_hash: &commit.tree_hash,
        branch: &commit.branch,
        parent_commits: &commit.parent_commits,
        author: &commit.author,
        message: &commit.message,
        timestamp_unix_ms: commit.timestamp_unix_ms,
        graph_version: commit.graph_version,
        objects_total: commit.objects_total,
        compiler_version: GRAPH_PACK_COMPILER_VERSION,
    })
}

fn legacy_v1_json_hash(value: &impl Serialize) -> GraphStoreResult<Hash> {
    serde_json::to_vec(value)
        .map(|bytes| Hash::of(&bytes))
        .map_err(|error| {
            GraphStoreError::new(
                "legacy_graph_hash_serialize",
                format!("could not serialize frozen v1 hash preimage: {error}"),
            )
        })
}

fn checked_u32(value: usize, field: &str) -> GraphStoreResult<u32> {
    u32::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "graph_canonical_length_overflow",
            format!("{field} length {value} exceeds the canonical u32 limit"),
        )
    })
}

fn push_len_prefixed(output: &mut Vec<u8>, value: &str, field: &str) -> GraphStoreResult<()> {
    output.extend_from_slice(&checked_u32(value.len(), field)?.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn canonical_node_projection(node: &NodeRecord) -> GraphStoreResult<Value> {
    #[derive(Serialize)]
    struct NodeProjection<'a> {
        id: &'a str,
        labels: &'a [String],
        properties: &'a Value,
        tombstone: bool,
    }

    serialize_graph_value(
        NodeProjection {
            id: &node.id,
            labels: &node.labels,
            properties: &node.properties,
            tombstone: node.tombstone,
        },
        "graph_object_projection_serialize",
    )
}

fn canonical_edge_projection(edge: &EdgeRecord) -> GraphStoreResult<Value> {
    #[derive(Serialize)]
    struct EdgeProjection<'a> {
        id: &'a str,
        from_id: &'a str,
        to_id: &'a str,
        #[serde(rename = "type")]
        edge_type: &'a str,
        properties: &'a Value,
        tombstone: bool,
        confidence: Option<f64>,
        epistemic_type: &'a Option<crate::graph::record::EpistemicType>,
        provenance: &'a Option<crate::graph::record::Provenance>,
    }

    serialize_graph_value(
        EdgeProjection {
            id: &edge.id,
            from_id: &edge.from_id,
            to_id: &edge.to_id,
            edge_type: &edge.edge_type,
            properties: &edge.properties,
            tombstone: edge.tombstone,
            confidence: edge.confidence,
            epistemic_type: &edge.epistemic_type,
            provenance: &edge.provenance,
        },
        "graph_object_projection_serialize",
    )
}

fn canonical_object_preimage_from_projection(
    kind: &GraphObjectKind,
    key: &str,
    projection: &Value,
) -> GraphStoreResult<Vec<u8>> {
    let logical_json = canonical_json(projection).map_err(|message| {
        GraphStoreError::new(
            "graph_object_projection_serialize",
            format!("could not canonicalize graph object projection: {message}"),
        )
    })?;
    let mut output = Vec::with_capacity(7 + key.len() + logical_json.len());
    output.push(DOMAIN_OBJECT);
    output.push(GRAPH_CHUNK_FORMAT_VERSION);
    output.push(kind.canonical_tag());
    push_len_prefixed(&mut output, key, "graph object key")?;
    output.extend_from_slice(logical_json.as_bytes());
    Ok(output)
}

pub(crate) fn canonical_node_preimage(node: &NodeRecord) -> GraphStoreResult<Vec<u8>> {
    canonical_object_preimage_from_projection(
        &GraphObjectKind::Node,
        &format!("node/{}", node.id),
        &canonical_node_projection(node)?,
    )
}

pub(crate) fn canonical_edge_preimage(edge: &EdgeRecord) -> GraphStoreResult<Vec<u8>> {
    canonical_object_preimage_from_projection(
        &GraphObjectKind::Edge,
        &format!("edge/{}", edge.id),
        &canonical_edge_projection(edge)?,
    )
}

/// Encode one complete recovery record without changing its frozen logical
/// object hash. The explicit framing prevents a payload from being retyped,
/// re-keyed, or assigned a different record version while retaining an
/// otherwise valid JSON body.
pub fn canonical_graph_payload_preimage(record: &GraphPayloadRecord) -> GraphStoreResult<Vec<u8>> {
    let key = record.key();
    validate_graph_payload_key(&record.kind(), &key)?;
    let value = match record {
        GraphPayloadRecord::Node(node) => serialize_graph_value(node, "graph_payload_serialize")?,
        GraphPayloadRecord::Edge(edge) => serialize_graph_value(edge, "graph_payload_serialize")?,
    };
    let body = canonical_json(&value).map_err(|message| {
        GraphStoreError::new(
            "graph_payload_serialize",
            format!("could not canonicalize graph recovery payload: {message}"),
        )
    })?;

    let mut output = Vec::with_capacity(19 + key.len() + body.len());
    output.push(DOMAIN_GRAPH_PAYLOAD);
    output.push(GRAPH_PAYLOAD_FORMAT_VERSION);
    output.push(record.kind().canonical_tag());
    push_len_prefixed(&mut output, &key, "graph payload key")?;
    output.extend_from_slice(&record.version().to_le_bytes());
    push_len_prefixed(&mut output, &body, "graph payload JSON")?;
    Ok(output)
}

pub fn graph_payload_address(record: &GraphPayloadRecord) -> GraphStoreResult<Hash> {
    canonical_graph_payload_preimage(record).map(|bytes| Hash::of(&bytes))
}

/// Decode and verify a complete recovery record.
///
/// Verification is intentionally redundant with `ChunkStore::get`: callers
/// which obtain bytes through a migration, fixture, or remote transport still
/// get address, domain, key, kind, version, canonical-JSON, and trailing-byte
/// checks at this boundary.
pub fn decode_canonical_graph_payload(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<GraphPayloadRecord> {
    let actual_address = Hash::of(bytes);
    if actual_address != expected_address {
        return Err(GraphStoreError::new(
            "graph_payload_address_mismatch",
            format!("graph payload bytes hash to {actual_address}, expected {expected_address}"),
        ));
    }

    let mut decoder = GraphPayloadDecoder::new(bytes);
    decoder.expect_byte(DOMAIN_GRAPH_PAYLOAD, "payload domain")?;
    decoder.expect_byte(GRAPH_PAYLOAD_FORMAT_VERSION, "payload format version")?;
    let kind = match decoder.byte("payload record kind")? {
        1 => GraphObjectKind::Node,
        2 => GraphObjectKind::Edge,
        other => {
            return Err(GraphStoreError::new(
                "graph_payload_kind_invalid",
                format!("graph payload kind tag {other} is invalid"),
            ))
        }
    };
    let key = decoder.string("payload record key")?;
    validate_graph_payload_key(&kind, &key)?;
    let framed_version = decoder.u64("payload record version")?;
    let body = decoder.bytes("payload JSON")?;
    decoder.finish()?;

    let record = match kind {
        GraphObjectKind::Node => {
            GraphPayloadRecord::Node(serde_json::from_slice(body).map_err(|error| {
                GraphStoreError::new(
                    "graph_payload_decode",
                    format!("node payload {expected_address} is invalid JSON: {error}"),
                )
            })?)
        }
        GraphObjectKind::Edge => {
            GraphPayloadRecord::Edge(serde_json::from_slice(body).map_err(|error| {
                GraphStoreError::new(
                    "graph_payload_decode",
                    format!("edge payload {expected_address} is invalid JSON: {error}"),
                )
            })?)
        }
    };
    if record.kind() != kind || record.key() != key {
        return Err(GraphStoreError::new(
            "graph_payload_record_mismatch",
            format!(
                "graph payload frame identifies {} {key:?}, but its record identifies {} {:?}",
                kind.as_str(),
                record.kind().as_str(),
                record.key()
            ),
        ));
    }
    if record.version() != framed_version {
        return Err(GraphStoreError::new(
            "graph_payload_version_mismatch",
            format!(
                "graph payload {key:?} frames version {framed_version}, but its record carries {}",
                record.version()
            ),
        ));
    }
    let canonical = canonical_graph_payload_preimage(&record)?;
    if canonical != bytes {
        return Err(GraphStoreError::new(
            "graph_payload_not_canonical",
            format!("graph payload {expected_address} does not use the canonical encoding"),
        ));
    }
    Ok(record)
}

fn validate_graph_payload_key(kind: &GraphObjectKind, key: &str) -> GraphStoreResult<()> {
    let prefix = match kind {
        GraphObjectKind::Node => "node/",
        GraphObjectKind::Edge => "edge/",
    };
    if key.strip_prefix(prefix).is_none_or(str::is_empty) {
        return Err(GraphStoreError::new(
            "graph_payload_key_invalid",
            format!(
                "{} payload key {key:?} must use the {prefix:?} namespace with a non-empty id",
                kind.as_str()
            ),
        ));
    }
    Ok(())
}

/// Return the frozen logical preimage for a payloadful graph object.
///
/// Payloadless objects can carry an address as a tree/reference optimization,
/// but are not independently verifiable: verification must first recover the
/// logical record payload.
pub fn canonical_content_object_preimage(object: &GraphContentObject) -> GraphStoreResult<Vec<u8>> {
    let payload = object.payload.clone().ok_or_else(|| {
        GraphStoreError::new(
            "graph_object_payload_required",
            format!(
                "graph object {} has no logical payload and cannot be independently verified",
                object.hash
            ),
        )
    })?;
    match object.kind {
        GraphObjectKind::Node => {
            let node: NodeRecord = serde_json::from_value(payload).map_err(|error| {
                GraphStoreError::new(
                    "graph_object_payload_decode",
                    format!("node object {} payload is invalid: {error}", object.hash),
                )
            })?;
            let expected_key = format!("node/{}", node.id);
            if object.key != expected_key || object.tombstone != node.tombstone {
                return Err(GraphStoreError::new(
                    "graph_object_projection_mismatch",
                    format!(
                        "node object metadata does not match payload: expected key {expected_key} and tombstone {}",
                        node.tombstone
                    ),
                ));
            }
            canonical_node_preimage(&node)
        }
        GraphObjectKind::Edge => {
            let edge: EdgeRecord = serde_json::from_value(payload).map_err(|error| {
                GraphStoreError::new(
                    "graph_object_payload_decode",
                    format!("edge object {} payload is invalid: {error}", object.hash),
                )
            })?;
            let expected_key = format!("edge/{}", edge.id);
            if object.key != expected_key || object.tombstone != edge.tombstone {
                return Err(GraphStoreError::new(
                    "graph_object_projection_mismatch",
                    format!(
                        "edge object metadata does not match payload: expected key {expected_key} and tombstone {}",
                        edge.tombstone
                    ),
                ));
            }
            canonical_edge_preimage(&edge)
        }
    }
}

/// Recompute a payloadful object's logical v2 address from its explicit
/// semantic projection. This is the verification seam used by cold storage.
pub fn recompute_content_object_hash(object: &GraphContentObject) -> GraphStoreResult<Hash> {
    canonical_content_object_preimage(object).map(|bytes| Hash::of(&bytes))
}

pub fn canonical_leaf_preimage(entries: &[GraphTreeEntry]) -> GraphStoreResult<Vec<u8>> {
    let mut output = Vec::new();
    output.push(DOMAIN_LEAF);
    output.push(GRAPH_CHUNK_FORMAT_VERSION);
    output.extend_from_slice(&checked_u32(entries.len(), "leaf entry count")?.to_le_bytes());
    for entry in entries {
        output.push(entry.kind.canonical_tag());
        // Reserved in v2 so tombstone semantics do not require another root
        // rebaseline when delete-aware incremental packs land.
        output.push(u8::from(entry.tombstone));
        push_len_prefixed(&mut output, &entry.key, "leaf entry key")?;
        output.extend_from_slice(entry.object_hash.as_bytes());
    }
    Ok(output)
}

pub fn canonical_parent_preimage(children: &[GraphTreeChild]) -> GraphStoreResult<Vec<u8>> {
    let mut output = Vec::new();
    output.push(DOMAIN_PARENT);
    output.push(GRAPH_CHUNK_FORMAT_VERSION);
    output.extend_from_slice(&checked_u32(children.len(), "parent child count")?.to_le_bytes());
    for child in children {
        output.extend_from_slice(child.hash.as_bytes());
        push_len_prefixed(&mut output, &child.first_key, "parent first key")?;
        push_len_prefixed(&mut output, &child.last_key, "parent last key")?;
        let entries_total = u64::try_from(child.entries_total).map_err(|_| {
            GraphStoreError::new(
                "graph_canonical_length_overflow",
                format!(
                    "parent child entries total {} exceeds the canonical u64 limit",
                    child.entries_total
                ),
            )
        })?;
        output.extend_from_slice(&entries_total.to_le_bytes());
    }
    Ok(output)
}

pub fn canonical_tree_node_preimage(node: &GraphTreeNode) -> GraphStoreResult<Vec<u8>> {
    if node.level == 0 {
        if !node.children.is_empty() {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                "leaf tree node cannot carry children",
            ));
        }
        canonical_leaf_preimage(&node.entries)
    } else {
        if !node.entries.is_empty() {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                "parent tree node cannot carry leaf entries",
            ));
        }
        canonical_parent_preimage(&node.children)
    }
}

pub fn recompute_tree_node_hash(node: &GraphTreeNode) -> GraphStoreResult<Hash> {
    canonical_tree_node_preimage(node).map(|bytes| Hash::of(&bytes))
}

/// Strictly decode a verified Phase-C tree-node preimage.
///
/// The store verifies addresses too, but keeping the check at this public
/// decode boundary makes the invariant hold for fixtures and migration tools
/// which receive bytes from elsewhere.
pub fn decode_canonical_tree_node(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<StoredProllyNode> {
    let actual_address = Hash::of(bytes);
    if actual_address != expected_address {
        return Err(GraphStoreError::new(
            "graph_tree_chunk_address_mismatch",
            format!("tree chunk bytes hash to {actual_address}, expected {expected_address}"),
        ));
    }

    let mut decoder = TreeChunkDecoder::new(bytes);
    let domain = decoder.byte("tree node domain")?;
    let version = decoder.byte("tree node format version")?;
    if version != GRAPH_CHUNK_FORMAT_VERSION {
        return Err(GraphStoreError::new(
            "graph_tree_chunk_version_unsupported",
            format!(
                "tree chunk format version {version} is unsupported; expected {GRAPH_CHUNK_FORMAT_VERSION}"
            ),
        ));
    }

    let node = match domain {
        DOMAIN_LEAF => decode_leaf_node(&mut decoder)?,
        DOMAIN_PARENT => decode_parent_node(&mut decoder)?,
        other => {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_domain_invalid",
                format!("tree chunk domain {other} is neither leaf nor parent"),
            ))
        }
    };
    decoder.finish()?;

    let canonical = match &node {
        StoredProllyNode::Leaf(entries) => canonical_leaf_preimage(entries)?,
        StoredProllyNode::Parent(children) => canonical_parent_preimage(children)?,
    };
    if canonical != bytes {
        return Err(GraphStoreError::new(
            "graph_tree_chunk_not_canonical",
            format!("tree chunk {expected_address} does not use the canonical v2 encoding"),
        ));
    }
    Ok(node)
}

fn decode_leaf_node(decoder: &mut TreeChunkDecoder<'_>) -> GraphStoreResult<StoredProllyNode> {
    let count = decoder.count("leaf entry count", 38)?;
    let mut entries = Vec::with_capacity(count);
    let mut previous_key: Option<String> = None;
    for _ in 0..count {
        let kind = match decoder.byte("leaf entry kind")? {
            1 => GraphObjectKind::Node,
            2 => GraphObjectKind::Edge,
            other => {
                return Err(GraphStoreError::new(
                    "graph_tree_chunk_kind_invalid",
                    format!("leaf entry kind tag {other} is invalid"),
                ))
            }
        };
        let tombstone = match decoder.byte("leaf entry tombstone")? {
            0 => false,
            1 => true,
            other => {
                return Err(GraphStoreError::new(
                    "graph_tree_chunk_tombstone_invalid",
                    format!("leaf entry tombstone byte {other} is invalid"),
                ))
            }
        };
        let key = decoder.string("leaf entry key")?;
        if key.is_empty() {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_key_invalid",
                "leaf entry keys cannot be empty",
            ));
        }
        if previous_key
            .as_deref()
            .is_some_and(|previous| previous >= key.as_str())
        {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_order_invalid",
                "leaf entry keys must be strictly increasing",
            ));
        }
        let object_hash = decoder.hash("leaf entry object hash")?;
        previous_key = Some(key.clone());
        entries.push(GraphTreeEntry {
            key,
            kind,
            object_hash,
            tombstone,
        });
    }
    Ok(StoredProllyNode::Leaf(entries))
}

fn decode_parent_node(decoder: &mut TreeChunkDecoder<'_>) -> GraphStoreResult<StoredProllyNode> {
    let count = decoder.count("parent child count", 48)?;
    if count == 0 {
        return Err(GraphStoreError::new(
            "graph_tree_chunk_shape_invalid",
            "parent tree chunks must contain at least one child",
        ));
    }
    let mut children = Vec::with_capacity(count);
    let mut previous_last_key: Option<String> = None;
    for _ in 0..count {
        let hash = decoder.hash("parent child hash")?;
        let first_key = decoder.string("parent child first key")?;
        let last_key = decoder.string("parent child last key")?;
        if first_key.is_empty() || first_key > last_key {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_range_invalid",
                format!("parent child range {first_key:?}..={last_key:?} is invalid"),
            ));
        }
        if previous_last_key
            .as_deref()
            .is_some_and(|previous| previous >= first_key.as_str())
        {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_order_invalid",
                "parent child key ranges must be strictly increasing and non-overlapping",
            ));
        }
        let entries_total_u64 = decoder.u64("parent child entries total")?;
        let entries_total = usize::try_from(entries_total_u64).map_err(|_| {
            GraphStoreError::new(
                "graph_tree_chunk_count_overflow",
                format!(
                    "parent child entry count {entries_total_u64} exceeds this platform's usize"
                ),
            )
        })?;
        if entries_total == 0 {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_count_invalid",
                "parent children must describe at least one leaf entry",
            ));
        }
        previous_last_key = Some(last_key.clone());
        children.push(GraphTreeChild {
            hash,
            first_key,
            last_key,
            entries_total,
        });
    }
    children.iter().try_fold(0usize, |total, child| {
        total.checked_add(child.entries_total).ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_chunk_count_overflow",
                "parent child entry counts overflow usize",
            )
        })
    })?;
    Ok(StoredProllyNode::Parent(children))
}

struct TreeChunkDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> TreeChunkDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("decoder requested four bytes");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("decoder requested eight bytes");
        Ok(u64::from_le_bytes(bytes))
    }

    fn hash(&mut self, field: &str) -> GraphStoreResult<Hash> {
        let bytes: [u8; 32] = self
            .take(32, field)?
            .try_into()
            .expect("decoder requested 32 bytes");
        Ok(Hash::from_bytes(bytes))
    }

    fn string(&mut self, field: &str) -> GraphStoreResult<String> {
        let length = usize::try_from(self.u32(&format!("{field} length"))?).map_err(|_| {
            GraphStoreError::new(
                "graph_tree_chunk_length_overflow",
                format!("{field} length exceeds usize"),
            )
        })?;
        let bytes = self.take(length, field)?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|error| {
                GraphStoreError::new(
                    "graph_tree_chunk_utf8_invalid",
                    format!("{field} is not UTF-8: {error}"),
                )
            })
    }

    fn count(&mut self, field: &str, minimum_item_bytes: usize) -> GraphStoreResult<usize> {
        let count = usize::try_from(self.u32(field)?).map_err(|_| {
            GraphStoreError::new(
                "graph_tree_chunk_count_overflow",
                format!("{field} exceeds usize"),
            )
        })?;
        let required = count.checked_mul(minimum_item_bytes).ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_chunk_count_overflow",
                format!("{field} byte bound overflowed"),
            )
        })?;
        if required > self.remaining() {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_truncated",
                format!(
                    "{field} declares {count} items requiring at least {required} bytes, but only {} remain",
                    self.remaining()
                ),
            ));
        }
        Ok(count)
    }

    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self.position.checked_add(length).ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_chunk_length_overflow",
                format!("{field} offset overflowed"),
            )
        })?;
        let bytes = self.bytes.get(self.position..end).ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_chunk_truncated",
                format!(
                    "{field} needs {length} bytes at offset {}, but the chunk is {} bytes",
                    self.position,
                    self.bytes.len()
                ),
            )
        })?;
        self.position = end;
        Ok(bytes)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn finish(&self) -> GraphStoreResult<()> {
        if self.position != self.bytes.len() {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_trailing_bytes",
                format!(
                    "tree chunk has {} trailing bytes",
                    self.bytes.len() - self.position
                ),
            ));
        }
        Ok(())
    }
}

struct GraphPayloadDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> GraphPayloadDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn expect_byte(&mut self, expected: u8, field: &str) -> GraphStoreResult<()> {
        let actual = self.byte(field)?;
        if actual != expected {
            return Err(GraphStoreError::new(
                "graph_payload_frame_invalid",
                format!("{field} is {actual}, expected {expected}"),
            ));
        }
        Ok(())
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        let raw: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("decoder requested four bytes");
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        let raw: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("decoder requested eight bytes");
        Ok(u64::from_le_bytes(raw))
    }

    fn string(&mut self, field: &str) -> GraphStoreResult<String> {
        let length = self.length(&format!("{field} length"))?;
        std::str::from_utf8(self.take(length, field)?)
            .map(str::to_string)
            .map_err(|error| {
                GraphStoreError::new(
                    "graph_payload_utf8_invalid",
                    format!("{field} is not UTF-8: {error}"),
                )
            })
    }

    fn bytes(&mut self, field: &str) -> GraphStoreResult<&'a [u8]> {
        let length = self.length(&format!("{field} length"))?;
        self.take(length, field)
    }

    fn length(&mut self, field: &str) -> GraphStoreResult<usize> {
        usize::try_from(self.u32(field)?).map_err(|_| {
            GraphStoreError::new(
                "graph_payload_length_overflow",
                format!("{field} exceeds usize"),
            )
        })
    }

    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self.position.checked_add(length).ok_or_else(|| {
            GraphStoreError::new(
                "graph_payload_length_overflow",
                format!("{field} offset overflowed"),
            )
        })?;
        let bytes = self.bytes.get(self.position..end).ok_or_else(|| {
            GraphStoreError::new(
                "graph_payload_truncated",
                format!(
                    "{field} needs {length} bytes at offset {}, but the payload is {} bytes",
                    self.position,
                    self.bytes.len()
                ),
            )
        })?;
        self.position = end;
        Ok(bytes)
    }

    fn finish(&self) -> GraphStoreResult<()> {
        if self.position != self.bytes.len() {
            return Err(GraphStoreError::new(
                "graph_payload_trailing_bytes",
                format!(
                    "graph payload has {} trailing bytes",
                    self.bytes.len() - self.position
                ),
            ));
        }
        Ok(())
    }
}

fn canonical_commit_preimage_from_fields(
    tree_hash: Hash,
    parent_commits: &[Hash],
    author: &str,
    message: &str,
    timestamp_unix_ms: u128,
    graph_version: u64,
    historical_registry_hash: Option<Hash>,
) -> GraphStoreResult<Vec<u8>> {
    let mut output = Vec::new();
    output.push(DOMAIN_COMMIT);
    output.push(GRAPH_CHUNK_FORMAT_VERSION);
    output.extend_from_slice(tree_hash.as_bytes());
    output.extend_from_slice(
        &checked_u32(parent_commits.len(), "commit parent count")?.to_le_bytes(),
    );
    for parent in parent_commits {
        output.extend_from_slice(parent.as_bytes());
    }
    push_len_prefixed(&mut output, author, "commit author")?;
    push_len_prefixed(&mut output, message, "commit message")?;
    output.extend_from_slice(&timestamp_unix_ms.to_le_bytes());
    output.extend_from_slice(&graph_version.to_le_bytes());
    if let Some(history_hash) = historical_registry_hash {
        output.extend_from_slice(b"historical-registry-v1");
        output.extend_from_slice(history_hash.as_bytes());
    }
    Ok(output)
}

pub fn canonical_commit_preimage(commit: &GraphCommit) -> GraphStoreResult<Vec<u8>> {
    canonical_commit_preimage_from_fields(
        commit.tree_hash,
        &commit.parent_commits,
        &commit.author,
        &commit.message,
        commit.timestamp_unix_ms,
        commit.graph_version,
        commit.historical_registry_hash,
    )
}

fn historical_registry_hash(
    node_ids: &BTreeSet<String>,
    history_complete: bool,
) -> GraphStoreResult<Hash> {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"rustyred-historical-registry-v1");
    preimage.push(u8::from(history_complete));
    preimage.extend_from_slice(
        &checked_u32(node_ids.len(), "historical registry node count")?.to_le_bytes(),
    );
    for node_id in node_ids {
        push_len_prefixed(&mut preimage, node_id, "historical registry node id")?;
    }
    Ok(Hash::of(&preimage))
}

fn verify_historical_registry_commitment(pack: &CompiledGraphPack) -> GraphStoreResult<()> {
    let Some(committed_hash) = pack.commit.historical_registry_hash else {
        if pack.historical_node_history_complete || !pack.historical_node_ids.is_empty() {
            return Err(GraphStoreError::new(
                "historical_registry_uncommitted",
                "the pack carries historical registry state without a commit-bound hash",
            ));
        }
        return Ok(());
    };
    let expected_hash = historical_registry_hash(
        &pack.historical_node_ids,
        pack.historical_node_history_complete,
    )?;
    if committed_hash != expected_hash {
        return Err(GraphStoreError::new(
            "historical_registry_hash_mismatch",
            format!(
                "pack historical registry hashes to {expected_hash}, commit declares {committed_hash}"
            ),
        ));
    }
    let expected_commit_hash = recompute_commit_hash(&pack.commit)?;
    if pack.commit.commit_hash != expected_commit_hash {
        return Err(GraphStoreError::new(
            "graph_commit_hash_mismatch",
            format!(
                "pack commit hashes to {expected_commit_hash}, envelope declares {}",
                pack.commit.commit_hash
            ),
        ));
    }
    Ok(())
}

pub fn recompute_commit_hash(commit: &GraphCommit) -> GraphStoreResult<Hash> {
    canonical_commit_preimage(commit).map(|bytes| Hash::of(&bytes))
}

pub fn compile_graph_pack(
    snapshot: &GraphSnapshot,
    options: GraphCompileOptions,
) -> GraphStoreResult<CompiledGraphPack> {
    let scoped = if options.sync_scope_only {
        crate::sync_scope::filter_snapshot_for_sync(snapshot)
    } else if options.exclude_local_only {
        crate::sync_scope::exclude_local_only_from_snapshot(snapshot)
    } else {
        snapshot.clone()
    };
    let snapshot = &scoped;
    let name = clean_or_default(options.name, "graph-pack");
    let branch = clean_or_default(options.branch, DEFAULT_GRAPH_BRANCH);
    let author = clean_or_default(options.author, "rustyred");
    let message = clean_or_default(options.message, "compile graph snapshot");
    let timestamp_unix_ms = options.timestamp_unix_ms.unwrap_or_else(unix_ms);
    let objects = snapshot_content_objects(snapshot, options.include_payloads)?;
    let archive_objects = if options.include_payloads {
        archive_content_objects(&objects).map_err(graph_archive_error)?
    } else {
        Vec::new()
    };
    let tree_store = MemChunkStore::new();
    let entries = objects
        .iter()
        .map(|object| GraphTreeEntry {
            key: object.key.clone(),
            kind: object.kind.clone(),
            object_hash: object.hash,
            tombstone: object.tombstone,
        })
        .collect();
    let StoredTreeBuild {
        tree, tree_chunks, ..
    } = build_prolly_tree_from_entries_with_transport(&tree_store, entries)?;
    let commit = build_commit(
        &tree,
        &branch,
        parse_hash_list(options.parent_commits)?,
        &author,
        &message,
        timestamp_unix_ms,
        snapshot.version,
        Some(historical_registry_hash(
            &snapshot.historical_node_ids,
            snapshot.historical_node_history_complete,
        )?),
    )?;
    let manifest = GraphPackManifest {
        chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        name: name.clone(),
        branch: branch.clone(),
        commit_hash: commit.commit_hash.clone(),
        tree_hash: tree.root,
        graph_version: snapshot.version,
        nodes_total: snapshot.nodes.iter().filter(|node| !node.tombstone).count(),
        edges_total: snapshot.edges.iter().filter(|edge| !edge.tombstone).count(),
        objects_total: objects.len(),
    };
    let manifest_body = serialize_graph_value(&manifest, "graph_manifest_serialize")?;
    let validator_body = json!({
        "validator": "rustyred.verify_tree_root",
        "tree_hash": tree.root,
        "objects_total": objects.len(),
        "protocol_version": VERSIONED_GRAPH_PROTOCOL_VERSION,
    });
    let capabilities = vec![
        GraphCompilerCapability {
            name: format!("{name}.manifest"),
            kind: "graph_manifest".to_string(),
            mime_type: "application/json".to_string(),
            content_hash: stable_hash(&manifest_body)?,
            body: manifest_body,
        },
        GraphCompilerCapability {
            name: format!("{name}.verify_tree_root"),
            kind: "validator".to_string(),
            mime_type: "application/json".to_string(),
            content_hash: stable_hash(&validator_body)?,
            body: validator_body,
        },
    ];

    Ok(CompiledGraphPack {
        chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        manifest,
        commit,
        tree,
        tree_transport_complete: true,
        tree_chunks,
        capabilities,
        objects: if options.include_payloads {
            objects
        } else {
            Vec::new()
        },
        archive_objects,
        historical_node_ids: snapshot.historical_node_ids.clone(),
        historical_node_history_complete: snapshot.historical_node_history_complete,
    })
}

/// Apply a commit batch directly to an owned snapshot mirror.
///
/// The old implementation cloned the whole graph into two maps and then back
/// into vectors. This path only clones records named by the batch; existing
/// records stay in their allocation and deletes use `swap_remove` before the
/// final allocation-free sort restores deterministic order.
pub fn apply_graph_mutation_batch(snapshot: &mut GraphSnapshot, batch: &GraphMutationBatch) {
    while snapshot.historical_node_ids.len() > MAX_RESIDENT_HISTORICAL_NODE_IDS {
        snapshot.historical_node_ids.pop_last();
        snapshot.historical_node_history_complete = false;
    }
    // RedCore defines batch order by mutation class, independent of caller
    // order: deletes first, then upserts, with nodes available before edges.
    // Applying the mirror in the same order makes delete-then-readd
    // deterministic and keeps the versioned tree aligned with the live store.
    for mutation in batch
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::EdgeDelete(_)))
    {
        apply_snapshot_mutation(snapshot, mutation);
    }
    for mutation in batch
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::NodeDelete(_)))
    {
        apply_snapshot_mutation(snapshot, mutation);
    }
    for mutation in batch
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::NodeUpsert(_)))
    {
        apply_snapshot_mutation(snapshot, mutation);
    }
    for mutation in batch
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::EdgeUpsert(_)))
    {
        apply_snapshot_mutation(snapshot, mutation);
    }
    snapshot
        .nodes
        .sort_unstable_by(|left, right| left.id.cmp(&right.id));
    snapshot
        .edges
        .sort_unstable_by(|left, right| left.id.cmp(&right.id));
    snapshot.version = snapshot.version.saturating_add(1);
}

fn apply_snapshot_mutation(snapshot: &mut GraphSnapshot, mutation: &GraphMutation) {
    match mutation {
        GraphMutation::NodeUpsert(node) => {
            record_snapshot_historical_node_id(snapshot, &node.id);
            match snapshot
                .nodes
                .iter()
                .position(|existing| existing.id == node.id)
            {
                Some(index) => snapshot.nodes[index] = node.clone(),
                None => snapshot.nodes.push(node.clone()),
            }
        }
        GraphMutation::EdgeUpsert(edge) => {
            match snapshot
                .edges
                .iter()
                .position(|existing| existing.id == edge.id)
            {
                Some(index) => snapshot.edges[index] = edge.clone(),
                None => snapshot.edges.push(edge.clone()),
            }
        }
        GraphMutation::NodeDelete(id) => {
            if let Some(index) = snapshot.nodes.iter().position(|node| node.id == *id) {
                snapshot.nodes.swap_remove(index);
            }
        }
        GraphMutation::EdgeDelete(id) => {
            if let Some(index) = snapshot.edges.iter().position(|edge| edge.id == *id) {
                snapshot.edges.swap_remove(index);
            }
        }
    }
}

fn record_snapshot_historical_node_id(snapshot: &mut GraphSnapshot, id: &str) {
    if snapshot.historical_node_ids.contains(id) {
        return;
    }
    if snapshot.historical_node_ids.len() >= MAX_RESIDENT_HISTORICAL_NODE_IDS {
        snapshot.historical_node_history_complete = false;
        return;
    }
    snapshot.historical_node_ids.insert(id.to_string());
}

fn checked_graph_count_add(value: usize, delta: usize, field: &str) -> GraphStoreResult<usize> {
    value.checked_add(delta).ok_or_else(|| {
        GraphStoreError::new("graph_count_overflow", format!("{field} overflowed usize"))
    })
}

fn checked_graph_count_sub(value: usize, delta: usize, field: &str) -> GraphStoreResult<usize> {
    value.checked_sub(delta).ok_or_else(|| {
        GraphStoreError::new(
            "graph_count_underflow",
            format!("{field} underflowed usize"),
        )
    })
}

#[derive(Clone, Debug, PartialEq)]
enum NormalizedGraphMutation {
    Upsert(GraphPayloadRecord),
    Delete(GraphObjectKind),
}

/// Collapse a caller-ordered batch using RedCore's class order. Both logical
/// and recovery-payload indexes consume this exact map so delete/re-add
/// ordering cannot drift between the two roots.
fn normalized_graph_mutations(
    batch: &GraphMutationBatch,
) -> BTreeMap<String, NormalizedGraphMutation> {
    let mut normalized = BTreeMap::new();
    for mutation in &batch.mutations {
        if let GraphMutation::EdgeDelete(id) = mutation {
            normalized.insert(
                format!("edge/{id}"),
                NormalizedGraphMutation::Delete(GraphObjectKind::Edge),
            );
        }
    }
    for mutation in &batch.mutations {
        if let GraphMutation::NodeDelete(id) = mutation {
            normalized.insert(
                format!("node/{id}"),
                NormalizedGraphMutation::Delete(GraphObjectKind::Node),
            );
        }
    }
    for mutation in &batch.mutations {
        if let GraphMutation::NodeUpsert(node) = mutation {
            normalized.insert(
                format!("node/{}", node.id),
                NormalizedGraphMutation::Upsert(GraphPayloadRecord::Node(node.clone())),
            );
        }
    }
    for mutation in &batch.mutations {
        if let GraphMutation::EdgeUpsert(edge) = mutation {
            normalized.insert(
                format!("edge/{}", edge.id),
                NormalizedGraphMutation::Upsert(GraphPayloadRecord::Edge(edge.clone())),
            );
        }
    }
    normalized
}

pub fn compile_graph_pack_incremental(
    prior_pack: &CompiledGraphPack,
    batch: &GraphMutationBatch,
    options: GraphCompileOptions,
) -> GraphStoreResult<IncrementalGraphPack> {
    if !prior_pack.tree_transport_complete || prior_pack.tree_chunks.is_empty() {
        return Err(GraphStoreError::new(
            "compacted_prior_pack_requires_chunk_store",
            "incremental compilation of a delta or compact pack requires compile_graph_pack_incremental_with_store and the repository ChunkStore",
        ));
    }
    let store = MemChunkStore::new();
    install_tree_chunks(&store, &prior_pack.tree_chunks)?;
    // A claimed-complete transport is checked before it becomes the source
    // for a convenience compile. This is intentionally the only O(N) path;
    // persistent chaining uses the explicit store seam below.
    collect_tree_entries(&store, &prior_pack.tree)?;
    compile_graph_pack_incremental_with_store(&store, prior_pack, batch, options)
}

pub fn compile_graph_pack_incremental_with_store(
    store: &dyn ChunkStore,
    prior_pack: &CompiledGraphPack,
    batch: &GraphMutationBatch,
    mut options: GraphCompileOptions,
) -> GraphStoreResult<IncrementalGraphPack> {
    let name = clean_or_default(options.name.take(), "graph-pack");
    let branch = clean_or_default(options.branch.take(), DEFAULT_GRAPH_BRANCH);
    let author = clean_or_default(options.author.take(), "rustyred");
    let message = clean_or_default(options.message.take(), "incremental graph mutation batch");
    let timestamp_unix_ms = options.timestamp_unix_ms.unwrap_or_else(unix_ms);
    let parent_commits = if options.parent_commits.is_empty() {
        vec![prior_pack.commit.commit_hash]
    } else {
        parse_hash_list(options.parent_commits)?
    };
    let graph_version = prior_pack
        .commit
        .graph_version
        .checked_add(1)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_version_overflow",
                "incremental graph version overflowed u64",
            )
        })?;
    let mut historical_node_ids = prior_pack.historical_node_ids.clone();
    let mut historical_node_history_complete = prior_pack.historical_node_history_complete;
    while historical_node_ids.len() > MAX_RESIDENT_HISTORICAL_NODE_IDS {
        historical_node_ids.pop_last();
        historical_node_history_complete = false;
    }
    // Match RedCore's deterministic class ordering before collapsing repeated
    // keys: edge delete, node delete, node upsert, edge upsert. An upsert wins
    // over a delete regardless of raw caller order, and only the final state
    // reaches the tree/staging overlay.
    let final_mutations = normalized_graph_mutations(batch);

    let mut upserts: BTreeMap<String, GraphTreeEntry> = BTreeMap::new();
    let mut changed_objects: Vec<GraphContentObject> = Vec::with_capacity(final_mutations.len());
    let mut changed_object_keys: BTreeSet<String> = BTreeSet::new();
    let mut prior_entries = BTreeMap::new();
    let mut reader = ProllyReader::new(store);
    for (key, mutation) in final_mutations {
        let prior_entry = find_tree_entry(&mut reader, &prior_pack.tree, &key)?;
        let (entry, changed_object) = match mutation {
            NormalizedGraphMutation::Upsert(record) => {
                let object = match record {
                    GraphPayloadRecord::Node(node) => {
                        retain_historical_node_id(
                            &mut historical_node_ids,
                            &mut historical_node_history_complete,
                            &node.id,
                            MAX_RESIDENT_HISTORICAL_NODE_IDS,
                        );
                        node_content_object(&node, options.include_payloads)?
                    }
                    GraphPayloadRecord::Edge(edge) => {
                        edge_content_object(&edge, options.include_payloads)?
                    }
                };
                (
                    GraphTreeEntry {
                        key: object.key.clone(),
                        kind: object.kind.clone(),
                        object_hash: object.hash,
                        tombstone: object.tombstone,
                    },
                    Some(object),
                )
            }
            NormalizedGraphMutation::Delete(kind) => {
                let Some(prior) = prior_entry.as_ref() else {
                    continue;
                };
                if prior.kind != kind {
                    return Err(GraphStoreError::new(
                        "graph_tree_entry_kind_mismatch",
                        format!(
                            "delete for {key} expects kind {}, found {}",
                            kind.as_str(),
                            prior.kind.as_str()
                        ),
                    ));
                }
                if prior.tombstone {
                    continue;
                }
                if kind == GraphObjectKind::Node {
                    let node_id = key.strip_prefix("node/").ok_or_else(|| {
                        GraphStoreError::new(
                            "graph_tree_entry_key_invalid",
                            format!("node delete key {key:?} has no node/ prefix"),
                        )
                    })?;
                    retain_historical_node_id(
                        &mut historical_node_ids,
                        &mut historical_node_history_complete,
                        node_id,
                        MAX_RESIDENT_HISTORICAL_NODE_IDS,
                    );
                }
                let mut tombstone = prior.clone();
                tombstone.tombstone = true;
                (tombstone, None)
            }
        };
        if prior_entry
            .as_ref()
            .is_some_and(|prior| prior.kind != entry.kind)
        {
            return Err(GraphStoreError::new(
                "graph_tree_entry_kind_mismatch",
                format!(
                    "mutation for {key} expects kind {}, found a prior {} entry",
                    entry.kind.as_str(),
                    prior_entry
                        .as_ref()
                        .expect("checked as present")
                        .kind
                        .as_str()
                ),
            ));
        }
        if prior_entry.as_ref() != Some(&entry) {
            changed_object_keys.insert(key.clone());
            if let Some(object) = changed_object {
                changed_objects.push(object);
            }
            upserts.insert(key.clone(), entry);
        }
        prior_entries.insert(key, prior_entry);
    }

    // O(changed + log n) build: re-chunk only touched leaves + spines and
    // retain unchanged subtrees solely by their addresses in parent chunks.
    let build = build_prolly_tree_incremental(store, &prior_pack.tree, &upserts)?;
    #[cfg(test)]
    let build = {
        let mut build = build;
        if FORCE_INCREMENTAL_VALIDATION_DIVERGENCE.swap(false, Ordering::SeqCst) {
            // The shape a delta that reached the header but not the spine takes:
            // the published root is the prior root, which still resolves, and the
            // count still includes the delta. Nothing checks a bounded write
            // against the *content* of the tree it publishes, so this shape is
            // published successfully and is exactly what the whole-graph parity
            // audit exists to catch. A broken address would instead be caught by
            // content addressing on the next read, which would prove nothing
            // about the oracle.
            build.tree.root = prior_pack.tree.root;
        }
        build
    };

    // No whole-graph parity validation on this path, deliberately. That audit is
    // O(stored entries) -- it collects every entry of the prior tree and rebuilds
    // the whole tree from scratch -- and it used to run here as a publish
    // precondition on every commit in a debug build. It now runs at the
    // boundaries that already walk the whole graph: journal recovery, and
    // `RedCoreGraphStore::verify_tree_parity`. See
    // `audit_published_prolly_parity` and `graph_prolly_parity_validations`.
    //
    // What stays on this path is the O(changed) structural validation inside
    // `build_prolly_tree_incremental`: the built root must agree with the entry
    // count it declared, the patch windows must tile the tree, and the rechunked
    // parents must agree with their children. What is not asserted here is that
    // the incremental result equals a from-scratch rebuild, because that
    // assertion cannot be made without rebuilding from scratch.

    let tree = build.tree;
    let mut nodes_total = prior_pack.manifest.nodes_total;
    let mut edges_total = prior_pack.manifest.edges_total;
    for (key, entry) in &upserts {
        let prior_entry = prior_entries.get(key).and_then(Option::as_ref);
        let prior_live = prior_entry.is_some_and(|prior| !prior.tombstone);
        let next_live = !entry.tombstone;
        match (prior_live, next_live) {
            (false, true) => match entry.kind {
                GraphObjectKind::Node => {
                    nodes_total = checked_graph_count_add(nodes_total, 1, "manifest node count")?
                }
                GraphObjectKind::Edge => {
                    edges_total = checked_graph_count_add(edges_total, 1, "manifest edge count")?
                }
            },
            (true, false) => match entry.kind {
                GraphObjectKind::Node => {
                    nodes_total = checked_graph_count_sub(nodes_total, 1, "manifest node count")?
                }
                GraphObjectKind::Edge => {
                    edges_total = checked_graph_count_sub(edges_total, 1, "manifest edge count")?
                }
            },
            (false, false) | (true, true) => {
                // The physical entry remains in the tree in both cases; only
                // transitions across the tombstone boundary affect live
                // manifest totals.
            }
        }
    }
    let objects = changed_objects;
    let archive_objects = if options.include_payloads {
        archive_content_objects(&objects).map_err(graph_archive_error)?
    } else {
        Vec::new()
    };
    let commit = build_commit(
        &tree,
        &branch,
        parent_commits,
        &author,
        &message,
        timestamp_unix_ms,
        graph_version,
        Some(historical_registry_hash(
            &historical_node_ids,
            historical_node_history_complete,
        )?),
    )?;
    let manifest = GraphPackManifest {
        chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        name: name.clone(),
        branch: branch.clone(),
        commit_hash: commit.commit_hash.clone(),
        tree_hash: tree.root,
        graph_version,
        nodes_total,
        edges_total,
        objects_total: tree.entries_total,
    };
    let manifest_body = serialize_graph_value(&manifest, "graph_manifest_serialize")?;
    let validator_body = json!({
        "validator": "rustyred.verify_tree_root",
        "tree_hash": tree.root,
        "objects_total": tree.entries_total,
        "protocol_version": VERSIONED_GRAPH_PROTOCOL_VERSION,
    });
    let changed_tree_nodes = build.cost.chunks_written;
    let reused_tree_nodes = build.cost.chunks_reused;
    let commit_cost = CommitCost {
        changed_objects: changed_object_keys.len(),
        changed_tree_nodes,
        reused_tree_nodes,
        chunks_written: build.cost.chunks_written,
        bytes_written: build.cost.bytes_written,
    };

    Ok(IncrementalGraphPack {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        pack: CompiledGraphPack {
            chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
            manifest,
            commit,
            tree,
            tree_transport_complete: false,
            tree_chunks: build.tree_chunks,
            capabilities: vec![
                GraphCompilerCapability {
                    name: format!("{name}.manifest"),
                    kind: "graph_manifest".to_string(),
                    mime_type: "application/json".to_string(),
                    content_hash: stable_hash(&manifest_body)?,
                    body: manifest_body,
                },
                GraphCompilerCapability {
                    name: format!("{name}.verify_tree_root"),
                    kind: "validator".to_string(),
                    mime_type: "application/json".to_string(),
                    content_hash: stable_hash(&validator_body)?,
                    body: validator_body,
                },
            ],
            objects: if options.include_payloads {
                objects
            } else {
                Vec::new()
            },
            archive_objects,
            historical_node_ids,
            historical_node_history_complete,
        },
        changed_object_keys: changed_object_keys.into_iter().collect(),
        changed_tree_nodes,
        reused_tree_nodes,
        commit_cost,
    })
}

fn retain_historical_node_id(
    historical_node_ids: &mut BTreeSet<String>,
    historical_node_history_complete: &mut bool,
    node_id: &str,
    limit: usize,
) {
    if historical_node_ids.contains(node_id) {
        return;
    }
    if historical_node_ids.len() >= limit {
        *historical_node_history_complete = false;
        return;
    }
    historical_node_ids.insert(node_id.to_string());
}

/// Internal per-commit structural-sharing counters for an incremental build.
/// `chunks_written` = tree nodes whose hash is new vs the parent (the genuine
/// O(changed + log n) re-hash/re-serialize work and the persisted delta);
/// `chunks_reused` = nodes carried over by hash; `bytes_written` = serialized
/// byte size of the written nodes. Mapped onto the public `CommitCost`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TreeBuildCost {
    pub(crate) chunks_written: usize,
    /// Maximal unchanged subtree roots referenced by the changed tree. One
    /// count can represent an arbitrarily large untouched subtree; computing
    /// every shared descendant would turn commit accounting back into O(N).
    pub(crate) chunks_reused: usize,
    pub(crate) bytes_written: usize,
}

/// Result of an O(changed + log n) incremental build. `tree` is root-only and
/// `tree_chunks` contains only the final reachable staged delta; unchanged
/// subtrees stay in the backing store and are referenced by address. The root
/// remains byte-identical to a canonical full rebuild.
pub struct IncrementalTreeBuild {
    pub tree: GraphProllyTree,
    pub(crate) cost: TreeBuildCost,
    pub tree_chunks: Vec<GraphTreeChunk>,
    /// All unique chunks allocated in the copy-on-write overlay, including any
    /// unreachable intermediate. The batch builder keeps this at the final
    /// delta (plus at most a collapsed root), which is the allocation oracle
    /// for O(changed set + shared spine) behavior.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) staged_chunks_total: usize,
}

/// Copy-on-write overlay used to keep transient zipper states out of the
/// durable store. Reads fall through to the immutable base; writes land only
/// in memory until the final root's reachable staged delta is known.
struct StagingChunkStore<'a> {
    base: &'a dyn ChunkStore,
    staged: MemChunkStore,
}

impl<'a> StagingChunkStore<'a> {
    fn new(base: &'a dyn ChunkStore) -> Self {
        Self {
            base,
            staged: MemChunkStore::new(),
        }
    }

    fn staged_bytes(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.staged.get(address)
    }

    fn staged_chunk_count(&self) -> GraphStoreResult<usize> {
        self.staged.chunk_count()
    }
}

impl ChunkStore for StagingChunkStore<'_> {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        match self.staged.get(address)? {
            Some(bytes) => Ok(Some(bytes)),
            None => self.base.get(address),
        }
    }

    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        Ok(self.staged.has(address)? || self.base.has(address)?)
    }

    fn contains_address(&self, address: Hash) -> GraphStoreResult<bool> {
        // The overlay answers for the delta it is holding; the base answers from
        // its own index. Both halves are probes, so this does not reintroduce a
        // verified read through the base store.
        Ok(self.staged.contains_address(address)? || self.base.contains_address(address)?)
    }

    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        for address in addresses {
            if seen.insert(*address) && !self.has(*address)? {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.staged.write(chunks)
    }

    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        self.staged.write_accounted(chunks)
    }

    fn commit(&self, _root: Hash, _meta: crate::chunk_store::CommitMeta) -> GraphStoreResult<Hash> {
        Err(GraphStoreError::new(
            "graph_staging_commit_unsupported",
            "the incremental staging overlay cannot publish a root",
        ))
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.base.read_root()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProllyValidationMode {
    Off,
    Always,
    Sampled(u32),
}

/// Compatibility query for callers which only need to know whether the parity
/// audit can run at all. In sampled mode this returns true even though a
/// particular published root may not be selected; use
/// [`prolly_validation_selected`] when a root and graph version are available.
pub fn prolly_validation_enabled() -> bool {
    configured_prolly_validation_mode()
        .map(|mode| {
            !matches!(
                mode,
                ProllyValidationMode::Off | ProllyValidationMode::Sampled(0)
            )
        })
        .unwrap_or(false)
}

/// Select the whole-graph parity audit for one published root.
///
/// This is a boundary selector, not a publish-path one: the audit it gates is
/// O(stored entries), so its only call site is journal recovery, where the tree
/// is audited once before the recovered store serves a read or a write. The
/// decision is stable across process restarts because it depends only on the
/// published root, the graph version, and the configured parts-per-million rate.
pub fn prolly_validation_selected(
    prior_root: Hash,
    next_graph_version: u64,
) -> GraphStoreResult<bool> {
    match configured_prolly_validation_mode()? {
        ProllyValidationMode::Off => Ok(false),
        ProllyValidationMode::Always => Ok(true),
        ProllyValidationMode::Sampled(ppm) => {
            prolly_validation_sample_selected(prior_root, next_graph_version, ppm)
        }
    }
}

/// Pure deterministic sampling seam used by release validation and tests.
pub fn prolly_validation_sample_selected(
    prior_root: Hash,
    next_graph_version: u64,
    ppm: u32,
) -> GraphStoreResult<bool> {
    if ppm > PROLLY_VALIDATION_SAMPLE_SCALE_PPM {
        return Err(GraphStoreError::new(
            "graph_prolly_validation_rate_invalid",
            format!(
                "Prolly validation rate {ppm} ppm exceeds {}",
                PROLLY_VALIDATION_SAMPLE_SCALE_PPM
            ),
        ));
    }
    if ppm == 0 {
        return Ok(false);
    }
    if ppm == PROLLY_VALIDATION_SAMPLE_SCALE_PPM {
        return Ok(true);
    }

    let mut preimage = Vec::with_capacity(70);
    preimage.extend_from_slice(b"rustyred-prolly-validation-v1\0");
    preimage.extend_from_slice(prior_root.as_bytes());
    preimage.extend_from_slice(&next_graph_version.to_le_bytes());
    let sample_hash = Hash::of(&preimage);
    let bucket = u32::from_le_bytes(
        sample_hash.as_bytes()[..4]
            .try_into()
            .expect("SHA-256 has at least four bytes"),
    ) % PROLLY_VALIDATION_SAMPLE_SCALE_PPM;
    Ok(bucket < ppm)
}

fn configured_prolly_validation_mode() -> GraphStoreResult<ProllyValidationMode> {
    let ppm_override = std::env::var(PROLLY_VALIDATION_PPM_ENV)
        .ok()
        .map(|value| parse_prolly_validation_ppm(&value))
        .transpose()?;
    let explicit_mode = std::env::var(PROLLY_VALIDATION_MODE_ENV).ok();
    if let Some(value) = explicit_mode {
        return match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" | "always" => Ok(ProllyValidationMode::Always),
            "0" | "false" | "off" | "no" | "never" => Ok(ProllyValidationMode::Off),
            "sample" | "sampled" => Ok(ProllyValidationMode::Sampled(
                ppm_override.unwrap_or(DEFAULT_PROLLY_VALIDATION_SAMPLE_PPM),
            )),
            other => Err(GraphStoreError::new(
                "graph_prolly_validation_mode_invalid",
                format!(
                    "{PROLLY_VALIDATION_MODE_ENV} value {other:?} must be always, off, or sampled"
                ),
            )),
        };
    }
    if let Some(ppm) = ppm_override {
        return Ok(ProllyValidationMode::Sampled(ppm));
    }
    // The audit is a boundary oracle now, not a per-write one. It runs once per
    // restart -- on a path that already hydrates the whole graph -- so there is
    // no per-write cost left to amortize and no reason to sample it away: a
    // check that silently skips 99 restarts in 100 is not drift detection.
    // `RUSTYRED_PROLLY_VALIDATE=off` (or a zero ppm override) remains the escape
    // hatch, and `sampled`/`RUSTYRED_PROLLY_VALIDATE_PPM` remain honored as an
    // explicit deterministic filter over published roots.
    Ok(ProllyValidationMode::Always)
}

fn parse_prolly_validation_ppm(value: &str) -> GraphStoreResult<u32> {
    let ppm = value.trim().parse::<u32>().map_err(|error| {
        GraphStoreError::new(
            "graph_prolly_validation_rate_invalid",
            format!("{PROLLY_VALIDATION_PPM_ENV} is not a u32 parts-per-million rate: {error}"),
        )
    })?;
    if ppm > PROLLY_VALIDATION_SAMPLE_SCALE_PPM {
        return Err(GraphStoreError::new(
            "graph_prolly_validation_rate_invalid",
            format!(
                "{PROLLY_VALIDATION_PPM_ENV} value {ppm} exceeds {}",
                PROLLY_VALIDATION_SAMPLE_SCALE_PPM
            ),
        ));
    }
    Ok(ppm)
}

/// The from-scratch reference build. Every parity check in this crate compares
/// against this one canonical chunking, so an incremental result is never
/// validated against a second incremental result.
fn canonical_tree_from_entries(entries: Vec<GraphTreeEntry>) -> GraphStoreResult<GraphProllyTree> {
    // A private staging store: the reference build must not publish its chunks
    // into the store the audited tree lives in.
    let reference_store = MemChunkStore::new();
    build_prolly_tree_from_entries(&reference_store, entries)
}

fn prolly_divergence_error(
    subject: &str,
    tree_root: Hash,
    entries_total: usize,
    canonical_root: Hash,
    canonical_entries_total: usize,
) -> GraphStoreError {
    GraphStoreError::new(
        "graph_prolly_validation_divergence",
        format!(
            "{subject} {tree_root} ({entries_total} entries) diverged from the canonical rebuild \
             {canonical_root} ({canonical_entries_total} entries)"
        ),
    )
}

fn parity_audit_from_entries(
    tree: &GraphProllyTree,
    entries: Vec<GraphTreeEntry>,
) -> GraphStoreResult<GraphTreeParityAudit> {
    let live_nodes = entries
        .iter()
        .filter(|entry| !entry.tombstone && entry.kind == GraphObjectKind::Node)
        .count();
    let live_edges = entries
        .iter()
        .filter(|entry| !entry.tombstone && entry.kind == GraphObjectKind::Edge)
        .count();
    let canonical = canonical_tree_from_entries(entries)?;
    Ok(GraphTreeParityAudit {
        tree_root: tree.root,
        entries_total: tree.entries_total,
        live_nodes,
        live_edges,
        canonical_root: canonical.root,
        canonical_entries_total: canonical.entries_total,
    })
}

/// The boundary form of the whole-graph parity oracle: is `tree` the canonical
/// tree for the entries it stores?
///
/// This is the assertion that used to run as a publish precondition on every
/// incremental commit (`compile_graph_pack_incremental_with_store`). It is
/// O(entries) with a hash verification per chunk read, so it belongs where the
/// whole graph is already being walked: journal recovery, and the on-demand
/// verify surface. It counts itself in [`graph_prolly_parity_validations`].
///
/// A published tree that diverged from its canonical shape -- a spine reused
/// past the write that should have replaced it, a stale parent descriptor, a
/// tree that is not the canonical chunking of its own entry set -- is an `Err`
/// with code `graph_prolly_validation_divergence`. An unreachable or corrupt
/// chunk inside the tree surfaces as the chunk store's own read error, which is
/// the same refusal by another route.
///
/// What this form cannot see, stated plainly because a measured test depends on
/// it: the check compares a tree against a rebuild of *its own* entries, so a
/// divergence that leaves both the entry count and the shape canonical is
/// invisible. A delta dropped before it reached the tree and never counted, and
/// a lost in-place modification that changed an object hash without changing the
/// entry count, both leave a tree that is canonical for what it holds. Two other
/// checks cover those, at the boundaries that hold the evidence they need:
///
/// * [`validate_incremental_prolly_parity`] is handed the batch that produced the
///   tree and catches both.
/// * Journal recovery walks the logical and payload indexes together, entry by
///   entry, and compares each record against its logical entry, so a lost
///   modification is caught per key on the durable path before this audit runs.
pub fn audit_published_prolly_parity(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
) -> GraphStoreResult<GraphTreeParityAudit> {
    note_prolly_parity_validation();
    let entries = collect_tree_entries(store, tree)?;
    let audit = parity_audit_from_entries(tree, entries)?;
    if !audit.parity_holds() {
        return Err(prolly_divergence_error(
            "published tree",
            audit.tree_root,
            audit.entries_total,
            audit.canonical_root,
            audit.canonical_entries_total,
        ));
    }
    Ok(audit)
}

/// The replay form of the parity oracle: does building `upserts` incrementally
/// onto `prior_tree` produce the canonical tree for the merged entry stream?
///
/// This is the strongest form of the check and the one the publish path used to
/// run, kept public and exercised for the boundaries that hold the producing
/// batch. A restart does not hold one, so recovery uses
/// [`audit_published_prolly_parity`]; this form is what an audit that records
/// its batches, or a test that plants a divergence, should call.
pub fn validate_incremental_prolly_parity(
    store: &dyn ChunkStore,
    prior_tree: &GraphProllyTree,
    upserts: &BTreeMap<String, GraphTreeEntry>,
    incremental_tree: &GraphProllyTree,
) -> GraphStoreResult<()> {
    note_prolly_parity_validation();
    let prior_entry_stream = collect_tree_entries(store, prior_tree)?;
    let canonical = canonical_tree_from_entries(merge_entries(&prior_entry_stream, upserts))?;
    if incremental_tree.root != canonical.root
        || incremental_tree.entries_total != canonical.entries_total
    {
        return Err(prolly_divergence_error(
            "incremental tree",
            incremental_tree.root,
            incremental_tree.entries_total,
            canonical.root,
            canonical.entries_total,
        ));
    }
    Ok(())
}

fn child_descriptor(node: &GraphTreeNode) -> GraphStoreResult<GraphTreeChild> {
    Ok(GraphTreeChild {
        hash: node.hash.clone(),
        first_key: node.first_key.clone(),
        last_key: node.last_key.clone(),
        entries_total: node.entry_count()?,
    })
}

/// Merge a replacement set into a key-sorted prior entry stream. Logical
/// deletes remain physical replacement entries with `tombstone = true`, so
/// the same two-pointer O(n + k) merge covers upsert and delete validation.
fn merge_entries(
    prior_entries: &[GraphTreeEntry],
    upserts: &BTreeMap<String, GraphTreeEntry>,
) -> Vec<GraphTreeEntry> {
    let mut out = Vec::with_capacity(prior_entries.len() + upserts.len());
    let mut pending = upserts.iter().peekable();
    for entry in prior_entries {
        while let Some((key, value)) = pending.peek() {
            if key.as_str() < entry.key.as_str() {
                out.push((*value).clone());
                pending.next();
            } else {
                break;
            }
        }
        if let Some((key, value)) = pending.peek() {
            if key.as_str() == entry.key.as_str() {
                out.push((*value).clone());
                pending.next();
                continue;
            }
        }
        out.push(entry.clone());
    }
    for (_, value) in pending {
        out.push(value.clone());
    }
    out
}

#[derive(Clone)]
struct CursorFrame {
    height: usize,
    children: Vec<GraphTreeChild>,
    child_index: usize,
}

#[derive(Clone)]
struct LevelCursor {
    address: Hash,
    target_height: usize,
    frames: Vec<CursorFrame>,
}

#[derive(Clone)]
struct LevelPatch {
    start_key: String,
    end_key: Option<String>,
    replacements: Vec<GraphTreeChild>,
}

fn prolly_tree_height(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    root: Hash,
) -> GraphStoreResult<usize> {
    let mut height = 0usize;
    let mut address = root;
    loop {
        if height > MAX_REPOSITORY_TREE_DEPTH {
            return Err(GraphStoreError::new(
                "graph_repository_depth_exceeded",
                format!("Prolly tree depth exceeds {MAX_REPOSITORY_TREE_DEPTH} at {address}"),
            ));
        }
        match reader.read(address)?.as_ref() {
            StoredProllyNode::Leaf(_) => return Ok(height),
            StoredProllyNode::Parent(children) => {
                address = children[0].hash;
                height += 1;
            }
        }
    }
}

fn validate_tree_root_count(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
) -> GraphStoreResult<()> {
    let mut reader = ProllyReader::new(store);
    let declared_by_root = reader.read(tree.root)?.entries_total();
    if declared_by_root != tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "tree {} declares {} entries but its root descriptor declares {declared_by_root}",
                tree.root, tree.entries_total
            ),
        ));
    }
    Ok(())
}

fn seek_level(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    root: Hash,
    root_height: usize,
    target_height: usize,
    key: &str,
) -> GraphStoreResult<LevelCursor> {
    if target_height > root_height {
        return Err(GraphStoreError::new(
            "graph_tree_level_invalid",
            format!("cannot seek height {target_height} from root height {root_height}"),
        ));
    }
    let mut address = root;
    let mut height = root_height;
    let mut frames = Vec::with_capacity(root_height.saturating_sub(target_height));
    while height > target_height {
        let node = reader.read(address)?;
        let StoredProllyNode::Parent(children) = node.as_ref() else {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                format!("tree chunk {address} is a leaf above target height {target_height}"),
            ));
        };
        let child_index = children
            .partition_point(|child| child.first_key.as_str() <= key)
            .saturating_sub(1)
            .min(children.len() - 1);
        frames.push(CursorFrame {
            height,
            children: children.clone(),
            child_index,
        });
        address = children[child_index].hash;
        height -= 1;
    }
    Ok(LevelCursor {
        address,
        target_height,
        frames,
    })
}

fn next_level(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    cursor: &LevelCursor,
) -> GraphStoreResult<Option<LevelCursor>> {
    let mut next = cursor.clone();
    for frame_index in (0..next.frames.len()).rev() {
        if next.frames[frame_index].child_index + 1 >= next.frames[frame_index].children.len() {
            continue;
        }
        next.frames.truncate(frame_index + 1);
        let frame = &mut next.frames[frame_index];
        frame.child_index += 1;
        let mut address = frame.children[frame.child_index].hash;
        let mut height = frame.height - 1;
        while height > next.target_height {
            let node = reader.read(address)?;
            let StoredProllyNode::Parent(children) = node.as_ref() else {
                return Err(GraphStoreError::new(
                    "graph_tree_shape_invalid",
                    format!("tree chunk {address} is a leaf above cursor target"),
                ));
            };
            next.frames.push(CursorFrame {
                height,
                children: children.clone(),
                child_index: 0,
            });
            address = children[0].hash;
            height -= 1;
        }
        next.address = address;
        return Ok(Some(next));
    }
    Ok(None)
}

fn find_tree_entry(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    tree: &GraphProllyTree,
    key: &str,
) -> GraphStoreResult<Option<GraphTreeEntry>> {
    let height = prolly_tree_height(reader, tree.root)?;
    let cursor = seek_level(reader, tree.root, height, 0, key)?;
    let node = reader.read(cursor.address)?;
    let StoredProllyNode::Leaf(entries) = node.as_ref() else {
        return Err(GraphStoreError::new(
            "graph_tree_shape_invalid",
            format!("height-zero tree chunk {} is not a leaf", cursor.address),
        ));
    };
    Ok(entries
        .binary_search_by(|entry| entry.key.as_str().cmp(key))
        .ok()
        .map(|index| entries[index].clone()))
}

/// Verified bounded key-range read over the existing immutable Prolly tree.
/// Only selected spines/leaves are fetched; no graph snapshot is materialized.
pub fn graph_tree_range(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
    start: Option<&str>,
    before: Option<&str>,
    limit: usize,
    descending: bool,
) -> GraphStoreResult<Vec<GraphTreeEntry>> {
    if limit == 0 || limit > 1024 {
        return Err(GraphStoreError::new(
            "graph_tree_page_limit",
            "page limit must be 1..=1024",
        ));
    }
    let mut reader = ProllyReader::new(store);
    let root = reader.read(tree.root)?;
    if root.entries_total() != tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            "root count differs from its descriptor",
        ));
    }
    let mut out = Vec::new();
    let mut visited = HashSet::new();
    graph_tree_range_at(
        &mut reader,
        tree.root,
        0,
        &mut visited,
        start,
        before,
        limit,
        descending,
        &mut out,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn graph_tree_range_at(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    address: Hash,
    depth: usize,
    visited: &mut HashSet<Hash>,
    start: Option<&str>,
    before: Option<&str>,
    limit: usize,
    descending: bool,
    out: &mut Vec<GraphTreeEntry>,
) -> GraphStoreResult<()> {
    guard_repository_tree_visit(&address, depth, visited)?;
    let node = reader.read(address)?;
    match node.as_ref() {
        StoredProllyNode::Leaf(entries) => {
            let range = entries.iter().filter(|entry| {
                start.is_none_or(|key| entry.key.as_str() >= key)
                    && before.is_none_or(|key| entry.key.as_str() < key)
            });
            if descending {
                out.extend(range.rev().take(limit.saturating_sub(out.len())).cloned());
            } else {
                out.extend(range.take(limit.saturating_sub(out.len())).cloned());
            }
        }
        StoredProllyNode::Parent(children) => {
            let indexes: Box<dyn Iterator<Item = usize>> = if descending {
                Box::new((0..children.len()).rev())
            } else {
                Box::new(0..children.len())
            };
            for index in indexes {
                if out.len() >= limit {
                    break;
                }
                let child = &children[index];
                if start.is_some_and(|key| child.last_key.as_str() < key)
                    || before.is_some_and(|key| child.first_key.as_str() >= key)
                {
                    continue;
                }
                let fetched = reader.read(child.hash)?;
                if fetched.first_key() != child.first_key
                    || fetched.last_key() != child.last_key
                    || fetched.entries_total() != child.entries_total
                {
                    return Err(GraphStoreError::new(
                        "graph_tree_child_descriptor_mismatch",
                        "range child does not match its parent descriptor",
                    ));
                }
                graph_tree_range_at(
                    reader,
                    child.hash,
                    depth + 1,
                    visited,
                    start,
                    before,
                    limit,
                    descending,
                    out,
                )?;
            }
        }
    }
    Ok(())
}

pub fn graph_tree_entry(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
    key: &str,
) -> GraphStoreResult<Option<GraphTreeEntry>> {
    let before = format!("{key}\0");
    Ok(graph_tree_range(store, tree, Some(key), Some(&before), 1, false)?.pop())
}

/// The same logical/full-record join as recovery, for one pinned record.
pub fn graph_payload_at(
    store: &dyn ChunkStore,
    logical_tree: &GraphProllyTree,
    payload_tree: &GraphProllyTree,
    graph_version: u64,
    key: &str,
) -> GraphStoreResult<Option<GraphPayloadRecord>> {
    let logical = graph_tree_entry(store, logical_tree, key)?;
    let payload = graph_tree_entry(store, payload_tree, key)?;
    match (logical, payload) {
        (None, None) => Ok(None),
        (Some(logical), Some(payload)) => {
            validated_graph_payload_record(store, &logical, &payload, graph_version).map(Some)
        }
        _ => Err(GraphStoreError::new(
            "graph_payload_index_mismatch",
            "pinned logical and payload key sets differ",
        )),
    }
}

fn push_leaf_item(
    item: GraphTreeEntry,
    chunk: &mut Vec<GraphTreeEntry>,
    nodes: &mut Vec<GraphTreeNode>,
) -> GraphStoreResult<()> {
    let boundary = item.object_hash.trailing_u16();
    chunk.push(item);
    if should_split_chunk(chunk.len(), boundary) {
        nodes.push(make_leaf_node(std::mem::take(chunk))?);
    }
    Ok(())
}

fn persist_tree_nodes_untracked(
    store: &dyn ChunkStore,
    nodes: &[GraphTreeNode],
) -> GraphStoreResult<()> {
    let mut ignored_chunks = Vec::new();
    let mut ignored_cost = TreeBuildCost::default();
    persist_tree_nodes(store, nodes, &mut ignored_chunks, &mut ignored_cost)
}

/// Merge every sorted replacement into affected leaf windows in one pass over
/// each window. Once output and input both land on a stable chunk boundary, a
/// distant replacement starts a new patch and all intervening subtrees remain
/// address-reused.
fn rechunk_leaf_windows(
    store: &dyn ChunkStore,
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    root: Hash,
    root_height: usize,
    upserts: &BTreeMap<String, GraphTreeEntry>,
) -> GraphStoreResult<(Vec<LevelPatch>, usize)> {
    let pending = upserts.values().collect::<Vec<_>>();
    let mut pending_index = 0usize;
    let mut existing_keys = 0usize;
    let mut patches = Vec::new();

    while pending_index < pending.len() {
        let mut cursor = seek_level(reader, root, root_height, 0, &pending[pending_index].key)?;
        let first = reader.read(cursor.address)?;
        let StoredProllyNode::Leaf(first_entries) = first.as_ref() else {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                format!("height-zero tree chunk {} is not a leaf", cursor.address),
            ));
        };
        let start_key = first_entries
            .first()
            .map(|entry| entry.key.clone())
            .unwrap_or_else(|| pending[pending_index].key.clone());
        let mut chunk = Vec::new();
        let mut nodes = Vec::new();
        let end_key;

        loop {
            let node = reader.read(cursor.address)?;
            let StoredProllyNode::Leaf(entries) = node.as_ref() else {
                return Err(GraphStoreError::new(
                    "graph_tree_shape_invalid",
                    format!("height-zero tree chunk {} is not a leaf", cursor.address),
                ));
            };
            for entry in entries {
                while pending_index < pending.len()
                    && pending[pending_index].key.as_str() < entry.key.as_str()
                {
                    push_leaf_item((*pending[pending_index]).clone(), &mut chunk, &mut nodes)?;
                    pending_index += 1;
                }
                if pending_index < pending.len()
                    && pending[pending_index].key.as_str() == entry.key.as_str()
                {
                    push_leaf_item((*pending[pending_index]).clone(), &mut chunk, &mut nodes)?;
                    pending_index += 1;
                    existing_keys = checked_graph_count_add(
                        existing_keys,
                        1,
                        "incremental existing-key count",
                    )?;
                } else {
                    push_leaf_item(entry.clone(), &mut chunk, &mut nodes)?;
                }
            }

            let next = next_level(reader, &cursor)?;
            let next_first_key = match next.as_ref() {
                Some(next_cursor) => reader.read(next_cursor.address)?.first_key().to_string(),
                None => String::new(),
            };
            while pending_index < pending.len()
                && (next.is_none() || pending[pending_index].key.as_str() < next_first_key.as_str())
            {
                push_leaf_item((*pending[pending_index]).clone(), &mut chunk, &mut nodes)?;
                pending_index += 1;
            }

            let Some(next) = next else {
                if !chunk.is_empty() {
                    nodes.push(make_leaf_node(std::mem::take(&mut chunk))?);
                }
                end_key = None;
                break;
            };
            if chunk.is_empty()
                && (pending_index == pending.len()
                    || pending[pending_index].key.as_str() >= next_first_key.as_str())
            {
                end_key = Some(next_first_key);
                break;
            }
            cursor = next;
        }

        persist_tree_nodes_untracked(store, &nodes)?;
        patches.push(LevelPatch {
            start_key,
            end_key,
            replacements: nodes
                .iter()
                .map(child_descriptor)
                .collect::<GraphStoreResult<Vec<_>>>()?,
        });
    }
    Ok((patches, existing_keys))
}

fn push_parent_item(
    item: GraphTreeChild,
    level: u32,
    chunk: &mut Vec<GraphTreeChild>,
    nodes: &mut Vec<GraphTreeNode>,
) -> GraphStoreResult<()> {
    let boundary = item.hash.trailing_u16();
    chunk.push(item);
    if should_split_chunk(chunk.len(), boundary) {
        nodes.push(make_parent_node(level, std::mem::take(chunk))?);
    }
    Ok(())
}

/// Apply every lower-level patch to affected parent windows in one ordered
/// traversal. Multiple edits sharing ancestors rebuild those ancestors once;
/// stable boundaries between distant edits split the work back into independent
/// patches so untouched parent subtrees remain address-reused.
fn rechunk_parent_windows(
    store: &dyn ChunkStore,
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    root: Hash,
    root_height: usize,
    level: usize,
    patches: Vec<LevelPatch>,
) -> GraphStoreResult<Vec<LevelPatch>> {
    let mut patch_index = 0usize;
    let mut output = Vec::new();

    while patch_index < patches.len() {
        let mut cursor = seek_level(
            reader,
            root,
            root_height,
            level,
            &patches[patch_index].start_key,
        )?;
        let first = reader.read(cursor.address)?;
        let start_key = first.first_key().to_string();
        let mut chunk = Vec::new();
        let mut nodes = Vec::new();
        // Some(None) means the active replacement consumes the rest of the
        // level; Some(Some(k)) skips original children until k.
        let mut active_end: Option<Option<String>> = None;
        let end_key;

        loop {
            let node = reader.read(cursor.address)?;
            let StoredProllyNode::Parent(children) = node.as_ref() else {
                return Err(GraphStoreError::new(
                    "graph_tree_shape_invalid",
                    format!(
                        "height-{level} tree chunk {} is not a parent",
                        cursor.address
                    ),
                ));
            };
            for child in children {
                loop {
                    if let Some(end) = active_end.as_ref() {
                        match end {
                            None => break,
                            Some(end) if child.first_key.as_str() < end.as_str() => break,
                            Some(_) => active_end = None,
                        }
                        if active_end.is_some() {
                            break;
                        }
                    }
                    if patch_index < patches.len()
                        && patches[patch_index].start_key.as_str() <= child.first_key.as_str()
                    {
                        for replacement in &patches[patch_index].replacements {
                            push_parent_item(
                                replacement.clone(),
                                level as u32,
                                &mut chunk,
                                &mut nodes,
                            )?;
                        }
                        active_end = Some(patches[patch_index].end_key.clone());
                        patch_index += 1;
                        continue;
                    }
                    push_parent_item(child.clone(), level as u32, &mut chunk, &mut nodes)?;
                    break;
                }
            }

            let next = next_level(reader, &cursor)?;
            let next_first_key = match next.as_ref() {
                Some(next_cursor) => reader.read(next_cursor.address)?.first_key().to_string(),
                None => String::new(),
            };
            if let Some(Some(active)) = active_end.as_ref() {
                if next.is_none() || next_first_key.as_str() >= active.as_str() {
                    active_end = None;
                }
            }

            let Some(next) = next else {
                if patch_index != patches.len() {
                    return Err(GraphStoreError::new(
                        "graph_tree_patch_range_invalid",
                        "parent patch starts beyond the final original child",
                    ));
                }
                if !chunk.is_empty() {
                    nodes.push(make_parent_node(level as u32, std::mem::take(&mut chunk))?);
                }
                end_key = None;
                break;
            };
            if active_end.is_none()
                && chunk.is_empty()
                && (patch_index == patches.len()
                    || patches[patch_index].start_key.as_str() >= next_first_key.as_str())
            {
                end_key = Some(next_first_key);
                break;
            }
            cursor = next;
        }

        persist_tree_nodes_untracked(store, &nodes)?;
        output.push(LevelPatch {
            start_key,
            end_key,
            replacements: nodes
                .iter()
                .map(child_descriptor)
                .collect::<GraphStoreResult<Vec<_>>>()?,
        });
    }
    Ok(output)
}

fn promote_replacements(
    store: &dyn ChunkStore,
    mut replacements: Vec<GraphTreeChild>,
    mut level: usize,
    tree_chunks: &mut Vec<GraphTreeChunk>,
    cost: &mut TreeBuildCost,
) -> GraphStoreResult<Hash> {
    while replacements.len() > 1 {
        let nodes = chunk_by_boundary(
            replacements,
            |child: &GraphTreeChild| child.hash.trailing_u16(),
            |chunk| make_parent_node(level as u32, chunk.to_vec()),
        )?;
        persist_tree_nodes(store, &nodes, tree_chunks, cost)?;
        replacements = nodes
            .iter()
            .map(child_descriptor)
            .collect::<GraphStoreResult<Vec<_>>>()?;
        level += 1;
    }
    replacements
        .first()
        .map(|root| root.hash)
        .ok_or_else(|| GraphStoreError::new("graph_tree_shape_invalid", "tree has no root chunk"))
}

fn collapse_unary_root(store: &dyn ChunkStore, mut root: Hash) -> GraphStoreResult<Hash> {
    let mut reader = ProllyReader::new(store);
    for depth in 0..=MAX_REPOSITORY_TREE_DEPTH {
        match reader.read(root)?.as_ref() {
            StoredProllyNode::Parent(children) if children.len() == 1 => {
                if depth == MAX_REPOSITORY_TREE_DEPTH {
                    return Err(GraphStoreError::new(
                        "graph_repository_depth_exceeded",
                        format!(
                            "unary Prolly root chain exceeds {MAX_REPOSITORY_TREE_DEPTH} at {root}"
                        ),
                    ));
                }
                root = children[0].hash;
            }
            _ => return Ok(root),
        }
    }
    unreachable!("bounded unary-root traversal must return")
}

fn collect_reachable_staged_delta(
    staging: &StagingChunkStore<'_>,
    address: Hash,
    depth: usize,
    visited: &mut HashSet<Hash>,
    chunks: &mut Vec<GraphTreeChunk>,
    reused_subtree_roots: &mut usize,
) -> GraphStoreResult<()> {
    if depth > MAX_REPOSITORY_TREE_DEPTH {
        return Err(GraphStoreError::new(
            "graph_repository_depth_exceeded",
            format!("staged tree depth exceeds {MAX_REPOSITORY_TREE_DEPTH} at {address}"),
        ));
    }
    if !visited.insert(address) {
        return Ok(());
    }

    let Some(bytes) = staging.staged_bytes(address)? else {
        if !staging.base.contains_address(address)? {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_missing",
                format!("final incremental tree references missing base chunk {address}"),
            ));
        }
        *reused_subtree_roots = reused_subtree_roots.checked_add(1).ok_or_else(|| {
            GraphStoreError::new(
                "graph_commit_cost_overflow",
                "reused subtree-root count overflowed usize",
            )
        })?;
        // A base chunk cannot name a staged descendant: its content address
        // existed before this overlay. Pruning here bounds the final-delta walk
        // by the changed set plus spine.
        //
        // The probe is deliberately not `has`: this hint is asked once per base
        // child of every staged parent -- measured at 25 hints for a one-key
        // replacement at 1k records and 56 at 10k -- and a verifying existence
        // check made a bounded write pay a verified read per hint on the journal
        // store. The probe count and the store's physical fetch count are both
        // counted by an oracle, but those counters do not attribute a fetch to a
        // hint, so how many hints were the only read of their chunk is documented
        // in `ChunkStore::contains_address` and asserted by
        // `a_pruned_subtree_this_build_never_reads_is_republished_and_the_boundary_refuses_it`
        // rather than inferred from them.
        return Ok(());
    };

    let node = decode_canonical_tree_node(address, &bytes)?;
    if let StoredProllyNode::Parent(children) = node {
        for child in children {
            collect_reachable_staged_delta(
                staging,
                child.hash,
                depth + 1,
                visited,
                chunks,
                reused_subtree_roots,
            )?;
        }
    }
    chunks.push(GraphTreeChunk {
        address,
        bytes: bytes.as_ref().to_vec(),
    });
    Ok(())
}

/// Apply upserts by fetching only their leaf windows and root spines. Every
/// unchanged subtree is reused by the address already present in its parent
/// descriptor; its chunk is never fetched or cloned.
pub fn build_prolly_tree_incremental(
    store: &dyn ChunkStore,
    prior_tree: &GraphProllyTree,
    upserts: &BTreeMap<String, GraphTreeEntry>,
) -> GraphStoreResult<IncrementalTreeBuild> {
    validate_tree_root_count(store, prior_tree)?;
    let staging = StagingChunkStore::new(store);
    let staging_store: &dyn ChunkStore = &staging;
    let tree = if upserts.is_empty() {
        prior_tree.clone()
    } else {
        let mut reader = ProllyReader::new(staging_store);
        let root_height = prolly_tree_height(&mut reader, prior_tree.root)?;
        let (mut patches, existing_keys) = rechunk_leaf_windows(
            staging_store,
            &mut reader,
            prior_tree.root,
            root_height,
            upserts,
        )?;
        for level in 1..=root_height {
            patches = rechunk_parent_windows(
                staging_store,
                &mut reader,
                prior_tree.root,
                root_height,
                level,
                patches,
            )?;
        }
        if patches.len() != 1 {
            return Err(GraphStoreError::new(
                "graph_tree_patch_root_invalid",
                format!(
                    "incremental batch produced {} disjoint root-level patches",
                    patches.len()
                ),
            ));
        }
        let replacements = patches
            .pop()
            .expect("root patch count was checked")
            .replacements;
        let mut ignored_chunks = Vec::new();
        let mut ignored_cost = TreeBuildCost::default();
        let root = promote_replacements(
            staging_store,
            replacements,
            root_height + 1,
            &mut ignored_chunks,
            &mut ignored_cost,
        )?;
        let root = collapse_unary_root(staging_store, root)?;
        let inserted_keys = upserts.len().checked_sub(existing_keys).ok_or_else(|| {
            GraphStoreError::new(
                "graph_tree_entry_count_invalid",
                format!(
                    "incremental merge found {existing_keys} existing keys for {} replacements",
                    upserts.len()
                ),
            )
        })?;
        let tree = GraphProllyTree {
            root,
            entries_total: checked_graph_count_add(
                prior_tree.entries_total,
                inserted_keys,
                "incremental tree entry count",
            )?,
        };
        validate_tree_root_count(staging_store, &tree)?;
        tree
    };

    let mut tree_chunks = Vec::new();
    let staged_chunks_total = staging.staged_chunk_count()?;
    let mut visited = HashSet::new();
    let mut reused_subtree_roots = 0usize;
    collect_reachable_staged_delta(
        &staging,
        tree.root,
        0,
        &mut visited,
        &mut tree_chunks,
        &mut reused_subtree_roots,
    )?;
    tree_chunks.sort_by_key(|chunk| chunk.address);
    let borrowed = tree_chunks
        .iter()
        .map(|chunk| (chunk.address, chunk.bytes.as_slice()))
        .collect::<Vec<_>>();
    let report = store.write_accounted(&borrowed)?;
    if report.chunks_written > tree_chunks.len() {
        return Err(GraphStoreError::new(
            "graph_commit_cost_invalid",
            format!(
                "store reported {} written chunks for a {}-chunk final delta",
                report.chunks_written,
                tree_chunks.len()
            ),
        ));
    }
    let cost = TreeBuildCost {
        chunks_written: report.chunks_written,
        chunks_reused: reused_subtree_roots,
        bytes_written: report.bytes_written,
    };

    Ok(IncrementalTreeBuild {
        tree,
        cost,
        tree_chunks,
        staged_chunks_total,
    })
}

/// Store-backed adjacent-commit diff. Equal child addresses are pruned from
/// their parent descriptors before either child is fetched.
pub fn diff_graph_trees(
    store: &dyn ChunkStore,
    base: &GraphProllyTree,
    target: &GraphProllyTree,
) -> GraphStoreResult<(GraphVersionDiff, ProllyReadMetrics)> {
    let mut reader = ProllyReader::new(store);
    let base_root = reader.read(base.root)?;
    let target_root = reader.read(target.root)?;
    validate_declared_tree_count(base, &base_root)?;
    validate_declared_tree_count(target, &target_root)?;

    let mut base_changed: BTreeMap<String, GraphTreeEntry> = BTreeMap::new();
    let mut target_changed: BTreeMap<String, GraphTreeEntry> = BTreeMap::new();
    if base.root != target.root {
        let base_height = prolly_tree_height(&mut reader, base.root)?;
        let target_height = prolly_tree_height(&mut reader, target.root)?;
        let base_segments = root_diff_segment(base.root, base_height, &base_root)
            .into_iter()
            .collect();
        let target_segments = root_diff_segment(target.root, target_height, &target_root)
            .into_iter()
            .collect();
        align_diff_segments(
            &mut reader,
            base_segments,
            target_segments,
            &mut base_changed,
            &mut target_changed,
            0,
        )?;
    } else if base.entries_total != target.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "equal tree root {} is paired with conflicting counts {} and {}",
                base.root, base.entries_total, target.entries_total
            ),
        ));
    }

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut modified = Vec::new();
    let mut changed_target_entries = 0usize;
    for (key, target_entry) in &target_changed {
        match base_changed.get(key) {
            None => {
                changed_target_entries = checked_graph_count_add(
                    changed_target_entries,
                    1,
                    "diff changed target entry count",
                )?;
                if !target_entry.tombstone {
                    added.push(GraphDiffEntry {
                        key: key.clone(),
                        kind: target_entry.kind.clone(),
                        old_hash: None,
                        new_hash: Some(target_entry.object_hash.to_prefixed()),
                    });
                }
            }
            Some(base_entry) if base_entry != target_entry => {
                changed_target_entries = checked_graph_count_add(
                    changed_target_entries,
                    1,
                    "diff changed target entry count",
                )?;
                match (base_entry.tombstone, target_entry.tombstone) {
                    (false, true) => removed.push(GraphDiffEntry {
                        key: key.clone(),
                        kind: base_entry.kind.clone(),
                        old_hash: Some(base_entry.object_hash.to_prefixed()),
                        new_hash: None,
                    }),
                    (true, false) => added.push(GraphDiffEntry {
                        key: key.clone(),
                        kind: target_entry.kind.clone(),
                        old_hash: None,
                        new_hash: Some(target_entry.object_hash.to_prefixed()),
                    }),
                    (false, false) => modified.push(GraphDiffEntry {
                        key: key.clone(),
                        kind: target_entry.kind.clone(),
                        old_hash: Some(base_entry.object_hash.to_prefixed()),
                        new_hash: Some(target_entry.object_hash.to_prefixed()),
                    }),
                    (true, true) => {}
                }
            }
            Some(_) => {}
        }
    }
    for (key, base_entry) in &base_changed {
        if !base_entry.tombstone && !target_changed.contains_key(key) {
            removed.push(GraphDiffEntry {
                key: key.clone(),
                kind: base_entry.kind.clone(),
                old_hash: Some(base_entry.object_hash.to_prefixed()),
                new_hash: None,
            });
        }
    }
    let unchanged = target
        .entries_total
        .checked_sub(changed_target_entries)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_diff_count_invalid",
                format!(
                    "diff found {changed_target_entries} changed target entries in a {}-entry tree",
                    target.entries_total
                ),
            )
        })?;

    Ok((
        GraphVersionDiff {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            base_tree_hash: base.root.to_prefixed(),
            target_tree_hash: target.root.to_prefixed(),
            added,
            removed,
            modified,
            unchanged,
            target_nodes: BTreeMap::new(),
            target_edges: BTreeMap::new(),
        },
        reader.metrics(),
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DiffSegment {
    address: Hash,
    height: usize,
    start: String,
    /// Exclusive upper bound; `None` is positive infinity.
    end: Option<String>,
}

fn validate_declared_tree_count(
    tree: &GraphProllyTree,
    root: &StoredProllyNode,
) -> GraphStoreResult<()> {
    let actual = root.entries_total();
    if actual != tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "tree {} declares {} entries but its root descriptor declares {actual}",
                tree.root, tree.entries_total
            ),
        ));
    }
    Ok(())
}

fn root_diff_segment(address: Hash, height: usize, root: &StoredProllyNode) -> Option<DiffSegment> {
    if root.entries_total() == 0 {
        return None;
    }
    Some(DiffSegment {
        address,
        height,
        start: root.first_key().to_string(),
        end: None,
    })
}

fn min_diff_end(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right).clone()),
        (Some(end), None) | (None, Some(end)) => Some(end.clone()),
        (None, None) => None,
    }
}

fn segment_ends_at_or_before(segment: &DiffSegment, start: &str) -> bool {
    segment.end.as_deref().is_some_and(|end| end <= start)
}

fn entry_in_diff_segment(entry: &GraphTreeEntry, segment: &DiffSegment) -> bool {
    entry.key.as_str() >= segment.start.as_str()
        && segment
            .end
            .as_deref()
            .is_none_or(|end| entry.key.as_str() < end)
}

fn diff_node_segments(
    segment: &DiffSegment,
    node: &StoredProllyNode,
) -> GraphStoreResult<VecDeque<DiffSegment>> {
    match (segment.height, node) {
        (0, StoredProllyNode::Leaf(_)) => Ok(VecDeque::from([segment.clone()])),
        (0, StoredProllyNode::Parent(_)) => Err(GraphStoreError::new(
            "graph_tree_shape_invalid",
            format!("height-zero diff chunk {} is a parent", segment.address),
        )),
        (_, StoredProllyNode::Leaf(_)) => Err(GraphStoreError::new(
            "graph_tree_shape_invalid",
            format!(
                "height-{} diff chunk {} is a leaf",
                segment.height, segment.address
            ),
        )),
        (height, StoredProllyNode::Parent(children)) => {
            let mut segments = VecDeque::new();
            for (index, child) in children.iter().enumerate() {
                let start = child.first_key.as_str().max(segment.start.as_str());
                let natural_end = children
                    .get(index + 1)
                    .map(|next| next.first_key.clone())
                    .or_else(|| segment.end.clone());
                let end = min_diff_end(&natural_end, &segment.end);
                if end.as_deref().is_some_and(|end| start >= end) {
                    continue;
                }
                segments.push_back(DiffSegment {
                    address: child.hash,
                    height: height - 1,
                    start: start.to_string(),
                    end,
                });
            }
            Ok(segments)
        }
    }
}

fn collect_diff_segment(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    segment: &DiffSegment,
    out: &mut BTreeMap<String, GraphTreeEntry>,
    depth: usize,
) -> GraphStoreResult<()> {
    if depth > MAX_REPOSITORY_TREE_DEPTH {
        return Err(GraphStoreError::new(
            "graph_repository_depth_exceeded",
            format!(
                "diff tree depth exceeds {MAX_REPOSITORY_TREE_DEPTH} at {}",
                segment.address
            ),
        ));
    }
    let node = reader.read(segment.address)?;
    match (segment.height, node.as_ref()) {
        (0, StoredProllyNode::Leaf(entries)) => {
            for entry in entries
                .iter()
                .filter(|entry| entry_in_diff_segment(entry, segment))
            {
                out.insert(entry.key.clone(), entry.clone());
            }
        }
        (0, StoredProllyNode::Parent(_)) => {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                format!("height-zero diff chunk {} is a parent", segment.address),
            ));
        }
        (_, StoredProllyNode::Leaf(_)) => {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                format!(
                    "height-{} diff chunk {} is a leaf",
                    segment.height, segment.address
                ),
            ));
        }
        (_, StoredProllyNode::Parent(_)) => {
            for child in diff_node_segments(segment, &node)? {
                collect_diff_segment(reader, &child, out, depth + 1)?;
            }
        }
    }
    Ok(())
}

fn diff_segment_pair(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    base: DiffSegment,
    target: DiffSegment,
    base_out: &mut BTreeMap<String, GraphTreeEntry>,
    target_out: &mut BTreeMap<String, GraphTreeEntry>,
    depth: usize,
) -> GraphStoreResult<()> {
    if base.address == target.address {
        if base.height != target.height {
            return Err(GraphStoreError::new(
                "graph_tree_shape_invalid",
                format!(
                    "shared diff chunk {} appears at heights {} and {}",
                    base.address, base.height, target.height
                ),
            ));
        }
        return Ok(());
    }
    if depth > MAX_REPOSITORY_TREE_DEPTH {
        return Err(GraphStoreError::new(
            "graph_repository_depth_exceeded",
            format!("diff tree depth exceeds {MAX_REPOSITORY_TREE_DEPTH}"),
        ));
    }
    let base_node = reader.read(base.address)?;
    let target_node = reader.read(target.address)?;
    match (base_node.as_ref(), target_node.as_ref()) {
        (StoredProllyNode::Leaf(base_entries), StoredProllyNode::Leaf(target_entries)) => {
            if base.height != 0 || target.height != 0 {
                return Err(GraphStoreError::new(
                    "graph_tree_shape_invalid",
                    "leaf diff chunks appeared above height zero",
                ));
            }
            for entry in base_entries
                .iter()
                .filter(|entry| entry_in_diff_segment(entry, &base))
            {
                base_out.insert(entry.key.clone(), entry.clone());
            }
            for entry in target_entries
                .iter()
                .filter(|entry| entry_in_diff_segment(entry, &target))
            {
                target_out.insert(entry.key.clone(), entry.clone());
            }
        }
        _ => {
            let base_segments = diff_node_segments(&base, &base_node)?;
            let target_segments = diff_node_segments(&target, &target_node)?;
            align_diff_segments(
                reader,
                base_segments,
                target_segments,
                base_out,
                target_out,
                depth + 1,
            )?;
        }
    }
    Ok(())
}

fn align_diff_segments(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    mut base: VecDeque<DiffSegment>,
    mut target: VecDeque<DiffSegment>,
    base_out: &mut BTreeMap<String, GraphTreeEntry>,
    target_out: &mut BTreeMap<String, GraphTreeEntry>,
    depth: usize,
) -> GraphStoreResult<()> {
    while !base.is_empty() || !target.is_empty() {
        let Some(base_segment) = base.front().cloned() else {
            for segment in target {
                collect_diff_segment(reader, &segment, target_out, depth)?;
            }
            return Ok(());
        };
        let Some(target_segment) = target.front().cloned() else {
            for segment in base {
                collect_diff_segment(reader, &segment, base_out, depth)?;
            }
            return Ok(());
        };

        if segment_ends_at_or_before(&base_segment, &target_segment.start) {
            collect_diff_segment(reader, &base_segment, base_out, depth)?;
            base.pop_front();
            continue;
        }
        if segment_ends_at_or_before(&target_segment, &base_segment.start) {
            collect_diff_segment(reader, &target_segment, target_out, depth)?;
            target.pop_front();
            continue;
        }
        if base_segment.start < target_segment.start {
            let mut prefix = base_segment.clone();
            prefix.end = Some(target_segment.start.clone());
            collect_diff_segment(reader, &prefix, base_out, depth)?;
            base.front_mut().expect("base front exists").start = target_segment.start;
            continue;
        }
        if target_segment.start < base_segment.start {
            let mut prefix = target_segment.clone();
            prefix.end = Some(base_segment.start.clone());
            collect_diff_segment(reader, &prefix, target_out, depth)?;
            target.front_mut().expect("target front exists").start = base_segment.start;
            continue;
        }

        let overlap_end = min_diff_end(&base_segment.end, &target_segment.end);
        let mut base_overlap = base_segment.clone();
        base_overlap.end = overlap_end.clone();
        let mut target_overlap = target_segment.clone();
        target_overlap.end = overlap_end.clone();
        diff_segment_pair(
            reader,
            base_overlap,
            target_overlap,
            base_out,
            target_out,
            depth,
        )?;

        let base_done = base_segment.end == overlap_end;
        let target_done = target_segment.end == overlap_end;
        if base_done {
            base.pop_front();
        } else {
            base.front_mut().expect("base front exists").start = overlap_end
                .clone()
                .expect("a finite target end split the base segment");
        }
        if target_done {
            target.pop_front();
        } else {
            target.front_mut().expect("target front exists").start =
                overlap_end.expect("a finite base end split the target segment");
        }
    }
    Ok(())
}

pub fn diff_graph_snapshots(
    base: &GraphSnapshot,
    target: &GraphSnapshot,
) -> GraphStoreResult<GraphVersionDiff> {
    let base_objects = snapshot_content_objects(base, false)?;
    let target_objects = snapshot_content_objects(target, false)?;
    let store = MemChunkStore::new();
    let base_tree = build_prolly_tree(&store, &base_objects)?;
    let target_tree = build_prolly_tree(&store, &target_objects)?;
    let mut diff = diff_graph_trees(&store, &base_tree, &target_tree)?.0;
    let changed_keys = diff
        .added
        .iter()
        .chain(&diff.modified)
        .map(|entry| entry.key.as_str())
        .collect::<BTreeSet<_>>();
    let changed_node_ids = changed_keys
        .iter()
        .filter_map(|key| key.strip_prefix("node/"))
        .collect::<BTreeSet<_>>();
    let changed_edge_ids = changed_keys
        .iter()
        .filter_map(|key| key.strip_prefix("edge/"))
        .collect::<BTreeSet<_>>();
    diff.target_nodes = target
        .nodes
        .iter()
        .filter(|node| changed_node_ids.contains(node.id.as_str()))
        .map(|node| (node.id.clone(), node.clone()))
        .collect();
    diff.target_edges = target
        .edges
        .iter()
        .filter(|edge| changed_edge_ids.contains(edge.id.as_str()))
        .map(|edge| (edge.id.clone(), edge.clone()))
        .collect();
    Ok(diff)
}

pub fn update_graph_ref(
    repository: GraphVersionRepository,
    pack: CompiledGraphPack,
    branch: Option<String>,
    updated_at_unix_ms: Option<u128>,
) -> Result<GraphRefUpdate, GraphRefConflict> {
    update_graph_ref_cas(repository, pack, branch, None, updated_at_unix_ms)
}

pub fn update_graph_ref_cas(
    mut repository: GraphVersionRepository,
    pack: CompiledGraphPack,
    branch: Option<String>,
    expected_commit_hash: Option<String>,
    updated_at_unix_ms: Option<u128>,
) -> Result<GraphRefUpdate, GraphRefConflict> {
    repository.chunk_format_version = GRAPH_CHUNK_FORMAT_VERSION;
    repository.protocol_version = VERSIONED_GRAPH_PROTOCOL_VERSION.to_string();
    let ref_name = clean_or_default(
        branch.or_else(|| Some(pack.commit.branch.clone())),
        DEFAULT_GRAPH_BRANCH,
    );
    let actual_commit_hash = repository
        .refs
        .iter()
        .find(|reference| reference.name == ref_name)
        .map(|reference| reference.commit_hash);
    if let Err(error) = verify_historical_registry_commitment(&pack) {
        return Err(GraphRefConflict {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            branch: ref_name,
            expected_commit_hash: None,
            actual_commit_hash: actual_commit_hash.map(|hash| hash.to_prefixed()),
            message: format!("graph pack installation failed: {}", error.message),
        });
    }
    if let Some(expected) = expected_commit_hash.as_ref() {
        if actual_commit_hash.map(|hash| hash.to_prefixed()).as_deref() != Some(expected.as_str()) {
            return Err(GraphRefConflict {
                protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
                branch: ref_name,
                expected_commit_hash,
                actual_commit_hash: actual_commit_hash.map(|hash| hash.to_prefixed()),
                message: "graph ref compare-and-swap failed".to_string(),
            });
        }
    }

    let updated_at_unix_ms = updated_at_unix_ms.unwrap_or_else(unix_ms);
    let reference = GraphVersionRef {
        name: ref_name.clone(),
        commit_hash: pack.commit.commit_hash,
        tree_hash: pack.tree.root,
        graph_version: pack.commit.graph_version,
        updated_at_unix_ms,
    };

    if let Err(error) = index_pack_content(&mut repository, &pack) {
        return Err(GraphRefConflict {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            branch: ref_name,
            expected_commit_hash: None,
            actual_commit_hash: actual_commit_hash.map(|hash| hash.to_prefixed()),
            message: format!("graph pack installation failed: {}", error.message),
        });
    }

    repository
        .refs
        .retain(|reference| reference.name != ref_name);
    repository.refs.push(reference.clone());
    repository.refs.sort_by(|a, b| a.name.cmp(&b.name));

    let stored_pack = compact_repository_pack(&pack);
    repository
        .packs
        .retain(|existing| existing.commit.commit_hash != stored_pack.commit.commit_hash);
    repository.packs.push(stored_pack);
    repository
        .packs
        .sort_by(|a, b| a.commit.commit_hash.cmp(&b.commit.commit_hash));

    Ok(GraphRefUpdate {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        repository,
        reference,
        pack,
    })
}

/// Delete a branch ref only when it still names the expected commit.
///
/// Commits, packs, and content remain in the repository as append-only history;
/// deleting the ref makes the reviewed branch unreachable without erasing its
/// evidence. The compare-and-swap guard prevents a rejection from discarding
/// work that landed on the branch after the review surface was compiled.
pub fn delete_graph_ref_cas(
    mut repository: GraphVersionRepository,
    branch: &str,
    expected_commit_hash: &str,
) -> Result<GraphRefDelete, GraphRefConflict> {
    let branch = clean_or_default(Some(branch.to_string()), DEFAULT_GRAPH_BRANCH);
    let reference = repository
        .refs
        .iter()
        .find(|reference| reference.name == branch)
        .cloned();
    let actual_commit_hash = reference.as_ref().map(|reference| reference.commit_hash);
    if actual_commit_hash.map(|hash| hash.to_prefixed()).as_deref() != Some(expected_commit_hash) {
        return Err(GraphRefConflict {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            branch,
            expected_commit_hash: Some(expected_commit_hash.to_string()),
            actual_commit_hash: actual_commit_hash.map(|hash| hash.to_prefixed()),
            message: "graph ref compare-and-swap delete failed".to_string(),
        });
    }
    let reference = reference.expect("matching commit requires an existing ref");
    repository
        .refs
        .retain(|candidate| candidate.name != reference.name);
    Ok(GraphRefDelete {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        repository,
        reference,
    })
}

pub fn graph_version_log(
    repository: &GraphVersionRepository,
    target: Option<&str>,
) -> GraphVersionLog {
    let target = target
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_GRAPH_BRANCH)
        .to_string();
    let commits_by_hash = repository
        .commits
        .iter()
        .cloned()
        .map(|commit| (commit.commit_hash, commit))
        .chain(
            repository
                .packs
                .iter()
                .map(|pack| (pack.commit.commit_hash, pack.commit.clone())),
        )
        .collect::<BTreeMap<_, _>>();
    let mut commits = Vec::new();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();
    if let Some(head) = resolve_repository_target(repository, &target) {
        queue.push_back(head);
    }

    while let Some(commit_hash) = queue.pop_front() {
        if !seen.insert(commit_hash) {
            continue;
        }
        let Some(commit) = commits_by_hash.get(&commit_hash) else {
            continue;
        };
        commits.push(commit.clone());
        for parent in &commit.parent_commits {
            queue.push_back(*parent);
        }
    }

    GraphVersionLog {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        target,
        commits,
    }
}

fn index_pack_content(
    repository: &mut GraphVersionRepository,
    pack: &CompiledGraphPack,
) -> GraphStoreResult<()> {
    for chunk in &pack.tree_chunks {
        decode_canonical_tree_node(chunk.address, &chunk.bytes)?;
        repository
            .tree_chunks
            .entry(chunk.address)
            .or_insert_with(|| chunk.bytes.clone());
    }
    if !repository.tree_chunks.contains_key(&pack.tree.root) {
        return Err(GraphStoreError::new(
            "graph_checkout_missing_tree_chunk",
            format!(
                "graph pack root {} is absent from both its transport and the repository store",
                pack.tree.root
            ),
        ));
    }
    if !repository
        .commits
        .iter()
        .any(|commit| commit.commit_hash == pack.commit.commit_hash)
    {
        repository.commits.push(pack.commit.clone());
        repository
            .commits
            .sort_by(|a, b| a.commit_hash.cmp(&b.commit_hash));
    }
    for object in &pack.objects {
        repository
            .objects
            .entry(object.hash)
            .or_insert_with(|| object.clone());
    }
    Ok(())
}

fn compact_repository_pack(pack: &CompiledGraphPack) -> CompiledGraphPack {
    let mut compact = pack.clone();
    compact.objects.clear();
    compact.archive_objects.clear();
    compact.tree_chunks.clear();
    compact.tree_transport_complete = false;
    compact
}

pub fn checkout_graph_version(
    repository: &GraphVersionRepository,
    target: &str,
) -> GraphStoreResult<GraphCheckoutResult> {
    let target = target.trim();
    if target.is_empty() {
        return Err(GraphStoreError::new(
            "graph_checkout_target_not_found",
            "graph checkout target is empty",
        ));
    }
    let commit_hash = resolve_repository_target(repository, target).ok_or_else(|| {
        GraphStoreError::new(
            "graph_checkout_target_not_found",
            format!("graph checkout target {target} does not exist"),
        )
    })?;
    let pack = repository
        .packs
        .iter()
        .find(|pack| pack.commit.commit_hash == commit_hash)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_checkout_target_not_found",
                format!("graph checkout commit {commit_hash} has no pack"),
            )
        })?;
    verify_historical_registry_commitment(pack)?;
    let store = RepositoryChunkStore::new(&repository.tree_chunks);
    if !store.has(pack.tree.root)? {
        return Err(GraphStoreError::new(
            "graph_checkout_missing_tree_chunk",
            format!("graph checkout tree {} is incomplete", pack.tree.root),
        ));
    }
    let tree = pack.tree.clone();
    let mut snapshot = graph_snapshot_from_repository(repository, pack)?;
    snapshot.historical_node_ids = pack.historical_node_ids.clone();
    snapshot.historical_node_history_complete = pack.historical_node_history_complete;
    Ok(GraphCheckoutResult {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        target: target.to_string(),
        commit: pack.commit.clone(),
        tree,
        snapshot,
    })
}

pub fn merge_graph_snapshots(
    base: &GraphSnapshot,
    ours: &GraphSnapshot,
    theirs: &GraphSnapshot,
    options: GraphMergeOptions,
) -> GraphStoreResult<GraphMergeResult> {
    merge_graph_snapshots_with_registry(
        base,
        ours,
        theirs,
        options,
        &MergeRegistry::substrate_sync_defaults(),
    )
}

pub fn merge_graph_snapshots_with_registry(
    base: &GraphSnapshot,
    ours: &GraphSnapshot,
    theirs: &GraphSnapshot,
    options: GraphMergeOptions,
    registry: &MergeRegistry,
) -> GraphStoreResult<GraphMergeResult> {
    let base_objects = snapshot_content_objects(base, false)?;
    let ours_objects = snapshot_content_objects(ours, false)?;
    let theirs_objects = snapshot_content_objects(theirs, false)?;
    let tree_store = MemChunkStore::new();
    let base_tree = build_prolly_tree(&tree_store, &base_objects)?;
    let ours_tree = build_prolly_tree(&tree_store, &ours_objects)?;
    let theirs_tree = build_prolly_tree(&tree_store, &theirs_objects)?;

    // The trees are not report-only work: their shared subtree hashes are the
    // merge index. Only keys surfaced by either base-to-side pruned walk need
    // full record materialization and conflict resolution.
    let (ours_diff, _) = diff_graph_trees(&tree_store, &base_tree, &ours_tree)?;
    let (theirs_diff, _) = diff_graph_trees(&tree_store, &base_tree, &theirs_tree)?;
    let mut changed_keys = graph_diff_keys(&ours_diff);
    changed_keys.extend(graph_diff_keys(&theirs_diff));

    // A checkout intentionally omits tombstones. Tree diff reports semantic
    // live-record changes, so removing a base tombstone from the physical side
    // indexes is not present in `GraphVersionDiff`. Promote any base key absent
    // from either side into merge resolution explicitly; otherwise the
    // unchanged fast path would require an object that a checkout cannot carry.
    for base_object in &base_objects {
        if graph_object_by_key(&ours_objects, &base_object.key).is_none()
            || graph_object_by_key(&theirs_objects, &base_object.key).is_none()
        {
            changed_keys.insert(base_object.key.clone());
        }
    }

    let base_records = merge_records_for_keys(base, &changed_keys);
    let ours_records = merge_records_for_keys(ours, &changed_keys);
    let theirs_records = merge_records_for_keys(theirs, &changed_keys);
    let mut keys = changed_keys.clone();
    keys.extend(base_objects.iter().map(|object| object.key.clone()));

    let mut merged_records = Vec::new();
    let mut resolved = Vec::new();
    let mut conflicts = Vec::new();

    for key in keys {
        if !changed_keys.contains(&key) {
            let base_object = required_graph_object_by_key(&base_objects, &key, "base")?;
            let ours_object = required_graph_object_by_key(&ours_objects, &key, "ours")?;
            let theirs_object = required_graph_object_by_key(&theirs_objects, &key, "theirs")?;
            resolved.push(GraphMergeResolution {
                key,
                kind: ours_object.kind.clone(),
                selected: Some(GraphMergeSide::Ours),
                base_hash: Some(base_object.logical_hash.to_prefixed()),
                ours_hash: Some(ours_object.logical_hash.to_prefixed()),
                theirs_hash: Some(theirs_object.logical_hash.to_prefixed()),
                reason: "same_result".to_string(),
            });
            continue;
        }

        let base_record = base_records.get(&key);
        let ours_record = ours_records.get(&key);
        let theirs_record = theirs_records.get(&key);
        let kind = merge_record_kind(base_record, ours_record, theirs_record);
        let base_hash = base_record.map(MergeRecord::hash).transpose()?;
        let ours_hash = ours_record.map(MergeRecord::hash).transpose()?;
        let theirs_hash = theirs_record.map(MergeRecord::hash).transpose()?;

        match resolve_merge_record(
            base_record,
            ours_record,
            theirs_record,
            &options.strategy,
            options.min_confidence_delta,
            registry,
        )? {
            MergeDecision::Resolved {
                selected,
                record,
                reason,
            } => {
                if let Some(record) = record {
                    merged_records.push(record);
                }
                resolved.push(GraphMergeResolution {
                    key,
                    kind,
                    selected,
                    base_hash,
                    ours_hash,
                    theirs_hash,
                    reason,
                });
            }
            MergeDecision::Conflict { reason } => conflicts.push(GraphMergeConflict {
                key,
                kind,
                base_hash,
                ours_hash,
                theirs_hash,
                reason,
            }),
        }
    }

    let (merged_snapshot, merged_pack, merged_tree_hash) = if conflicts.is_empty() {
        append_unchanged_merge_records(ours, &changed_keys, &mut merged_records);
        let snapshot = snapshot_from_merge_records(
            base.version
                .max(ours.version)
                .max(theirs.version)
                .saturating_add(1),
            merged_records,
        );
        let pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                name: options
                    .name
                    .or_else(|| Some("merged-graph-pack".to_string())),
                branch: options.branch,
                parent_commits: options.parent_commits,
                author: options.author,
                message: options
                    .message
                    .or_else(|| Some("merge graph snapshots".to_string())),
                timestamp_unix_ms: options.timestamp_unix_ms,
                include_payloads: options.include_payloads,
                ..GraphCompileOptions::default()
            },
        )?;
        (
            Some(snapshot),
            Some(pack.clone()),
            Some(pack.tree.root.to_prefixed()),
        )
    } else {
        (None, None, None)
    };

    Ok(GraphMergeResult {
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        status: if conflicts.is_empty() {
            "clean".to_string()
        } else {
            "conflicted".to_string()
        },
        base_tree_hash: base_tree.root.to_prefixed(),
        ours_tree_hash: ours_tree.root.to_prefixed(),
        theirs_tree_hash: theirs_tree.root.to_prefixed(),
        merged_tree_hash,
        merged_snapshot,
        merged_pack,
        resolved,
        conflicts,
    })
}

pub fn snapshot_content_objects(
    snapshot: &GraphSnapshot,
    include_payloads: bool,
) -> GraphStoreResult<Vec<GraphContentObject>> {
    let mut objects = Vec::with_capacity(snapshot.nodes.len() + snapshot.edges.len());
    for node in &snapshot.nodes {
        objects.push(node_content_object(node, include_payloads)?);
    }
    for edge in &snapshot.edges {
        objects.push(edge_content_object(edge, include_payloads)?);
    }
    objects.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(objects)
}

/// Build and persist the root-only recovery-payload index for a snapshot.
/// Payload chunks and Prolly nodes are ordinary verified `ChunkStore` chunks;
/// the returned descriptor carries only the root and physical entry count.
pub fn build_graph_payload_index(
    store: &dyn ChunkStore,
    snapshot: &GraphSnapshot,
) -> GraphStoreResult<GraphProllyTree> {
    let mut records = BTreeMap::new();
    for node in &snapshot.nodes {
        let record = GraphPayloadRecord::Node(node.clone());
        insert_unique_payload_record(&mut records, record)?;
    }
    for edge in &snapshot.edges {
        let record = GraphPayloadRecord::Edge(edge.clone());
        insert_unique_payload_record(&mut records, record)?;
    }
    let entries = write_graph_payload_records(store, records.into_values())?;
    build_prolly_tree_from_entries(store, entries)
}

/// Incrementally update a recovery-payload index using the same collapsed
/// RedCore mutation order as logical compilation. Deletes retain the prior
/// payload address and only flip the index tombstone; a later upsert wins over
/// a delete regardless of raw caller order.
pub fn update_graph_payload_index(
    store: &dyn ChunkStore,
    prior_tree: &GraphProllyTree,
    batch: &GraphMutationBatch,
) -> GraphStoreResult<GraphProllyTree> {
    let normalized = normalized_graph_mutations(batch);
    let mut reader = ProllyReader::new(store);
    let mut upserts = BTreeMap::new();
    let mut payloads = Vec::new();
    for (key, mutation) in normalized {
        let prior_entry = find_tree_entry(&mut reader, prior_tree, &key)?;
        let entry = match mutation {
            NormalizedGraphMutation::Upsert(record) => {
                if record.key() != key {
                    return Err(GraphStoreError::new(
                        "graph_payload_record_mismatch",
                        format!(
                            "normalized payload key {key:?} does not match record key {:?}",
                            record.key()
                        ),
                    ));
                }
                let bytes = canonical_graph_payload_preimage(&record)?;
                let object_hash = Hash::of(&bytes);
                payloads.push((object_hash, bytes));
                GraphTreeEntry {
                    key: key.clone(),
                    kind: record.kind(),
                    object_hash,
                    tombstone: record.tombstone(),
                }
            }
            NormalizedGraphMutation::Delete(kind) => {
                let Some(prior) = prior_entry.as_ref() else {
                    continue;
                };
                if prior.kind != kind {
                    return Err(GraphStoreError::new(
                        "graph_payload_entry_kind_mismatch",
                        format!(
                            "delete for {key} expects kind {}, found {}",
                            kind.as_str(),
                            prior.kind.as_str()
                        ),
                    ));
                }
                if prior.tombstone {
                    continue;
                }
                let mut tombstone = prior.clone();
                tombstone.tombstone = true;
                tombstone
            }
        };
        if prior_entry
            .as_ref()
            .is_some_and(|prior| prior.kind != entry.kind)
        {
            return Err(GraphStoreError::new(
                "graph_payload_entry_kind_mismatch",
                format!(
                    "mutation for {key} expects kind {}, found a prior {} entry",
                    entry.kind.as_str(),
                    prior_entry
                        .as_ref()
                        .expect("checked as present")
                        .kind
                        .as_str()
                ),
            ));
        }
        if prior_entry.as_ref() != Some(&entry) {
            upserts.insert(key, entry);
        }
    }

    let borrowed = payloads
        .iter()
        .map(|(address, bytes)| (*address, bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write(&borrowed)?;
    build_prolly_tree_incremental(store, prior_tree, &upserts).map(|build| build.tree)
}

/// Rebuild the operating snapshot by joining the frozen logical tree with the
/// complete-record payload index. Every structural or semantic mismatch is a
/// recovery error. A delete retains the prior live payload address in both
/// indexes, so recovery verifies that record and derives a RAM-only tombstone
/// at `graph_version`. The rooted indexes remain the durable delete marker;
/// the materialized record exists only to reject stale cold-tier readmission.
pub fn hydrate_graph_snapshot_from_payload_index(
    store: &dyn ChunkStore,
    logical_tree: &GraphProllyTree,
    payload_tree: &GraphProllyTree,
    graph_version: u64,
) -> GraphStoreResult<GraphSnapshot> {
    let logical_entries = collect_tree_entries(store, logical_tree)?;
    let payload_entries = collect_tree_entries(store, payload_tree)?;
    if logical_entries.len() != payload_entries.len() {
        return Err(GraphStoreError::new(
            "graph_payload_index_mismatch",
            format!(
                "logical index has {} entries, payload index has {}",
                logical_entries.len(),
                payload_entries.len()
            ),
        ));
    }

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut historical_node_ids = BTreeSet::new();
    for (logical, payload) in logical_entries.iter().zip(&payload_entries) {
        if logical.kind == GraphObjectKind::Node {
            let node_id = logical
                .key
                .strip_prefix("node/")
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_payload_key_invalid",
                        format!("logical node key {:?} is invalid", logical.key),
                    )
                })?;
            if historical_node_ids.len() < MAX_RESIDENT_HISTORICAL_NODE_IDS {
                historical_node_ids.insert(node_id.to_string());
            }
        }
        let record = validated_graph_payload_record(store, logical, payload, graph_version)?;
        match record {
            GraphPayloadRecord::Node(node) => nodes.push(node),
            GraphPayloadRecord::Edge(edge) => edges.push(edge),
        }
    }
    nodes.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    edges.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    Ok(GraphSnapshot {
        version: graph_version,
        nodes,
        edges,
        historical_node_ids,
        // The two root indexes prove every retained node key, including
        // tombstones, but cannot prove that a legacy snapshot never discarded
        // an older id. Fail closed for never-before-used insertion semantics.
        historical_node_history_complete: false,
    })
}

/// Rebuild the operating graph directly from the two rooted indexes without
/// staging a second record-sized `GraphSnapshot`. Recovery used to hold the
/// decoded node/edge vectors while `InMemoryGraphStore::from_snapshot` built
/// another complete resident copy and its indexes. Large tenants could exceed
/// the process limit during that transient duplication even when their final
/// operating graph fit.
pub(crate) fn hydrate_in_memory_graph_from_payload_index(
    store: &dyn ChunkStore,
    logical_tree: &GraphProllyTree,
    payload_tree: &GraphProllyTree,
    graph_version: u64,
    resident_edge_budget_bytes: Option<usize>,
) -> GraphStoreResult<HydratedInMemoryGraph> {
    let mut profile = RecoveryProfile::default();
    hydrate_in_memory_graph_from_payload_index_profiled(
        store,
        logical_tree,
        payload_tree,
        graph_version,
        0,
        resident_edge_budget_bytes,
        &mut profile,
    )
}

pub(crate) fn hydrate_in_memory_graph_from_payload_index_profiled(
    store: &dyn ChunkStore,
    logical_tree: &GraphProllyTree,
    payload_tree: &GraphProllyTree,
    graph_version: u64,
    journal_bytes: u64,
    resident_edge_budget_bytes: Option<usize>,
    profile: &mut RecoveryProfile,
) -> GraphStoreResult<HydratedInMemoryGraph> {
    if logical_tree.entries_total != payload_tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_payload_index_mismatch",
            format!(
                "logical index has {} entries, payload index has {}",
                logical_tree.entries_total, payload_tree.entries_total
            ),
        ));
    }

    let mut recovered = InMemoryGraphStore::new();
    let mut historical_node_ids = BTreeSet::new();
    let mut cold_stub_ids = Vec::new();
    let mut live_nodes = 0usize;
    let mut live_edges = 0usize;
    let mut retained_edge_records = 0usize;
    let mut resident_edge_bytes_estimate = 0usize;
    let mut edges_fully_resident = true;

    // Keys are globally ordered with `edge/` before `node/`. Apply nodes in a
    // first pass so strict edge recovery can validate both live endpoints,
    // then apply edges. Each decoded record is moved into the operating store
    // immediately and dropped before the next record is fetched. Each pass
    // streams one bounded Prolly leaf at a time instead of retaining both
    // complete entry indexes alongside the resident graph.
    for expected_kind in [&GraphObjectKind::Node, &GraphObjectKind::Edge] {
        if expected_kind == &GraphObjectKind::Edge && !edges_fully_resident {
            break;
        }
        let mut logical_entries = ProllyEntryCursor::new(store, logical_tree);
        let mut payload_entries = ProllyEntryCursor::new(store, payload_tree);
        loop {
            let (logical, payload) =
                match (logical_entries.next_entry()?, payload_entries.next_entry()?) {
                    (Some(logical), Some(payload)) => (logical, payload),
                    (None, None) => break,
                    _ => {
                        return Err(GraphStoreError::new(
                            "graph_payload_index_mismatch",
                            "logical and payload indexes ended at different entries",
                        ));
                    }
                };
            if expected_kind == &GraphObjectKind::Node && logical.kind == GraphObjectKind::Edge {
                retained_edge_records = retained_edge_records.checked_add(1).ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_payload_edge_count_overflow",
                        "retained edge count overflowed usize during recovery",
                    )
                })?;
                if edges_fully_resident {
                    if let Some(budget) = resident_edge_budget_bytes {
                        let record = validated_graph_payload_record(
                            store,
                            &logical,
                            &payload,
                            graph_version,
                        )?;
                        let GraphPayloadRecord::Edge(edge) = record else {
                            return Err(GraphStoreError::new(
                                "graph_payload_kind_mismatch",
                                format!("edge key {:?} decoded as a node payload", logical.key),
                            ));
                        };
                        let projected = resident_edge_bytes_estimate
                            .saturating_add(estimated_resident_edge_bytes(&edge)?);
                        if projected > budget {
                            edges_fully_resident = false;
                        } else {
                            resident_edge_bytes_estimate = projected;
                        }
                    }
                }
            }
            if &logical.kind != expected_kind {
                continue;
            }
            let record = validated_graph_payload_record(store, &logical, &payload, graph_version)?;
            match record {
                GraphPayloadRecord::Node(node) => {
                    if historical_node_ids.len() < MAX_RESIDENT_HISTORICAL_NODE_IDS {
                        historical_node_ids.insert(node.id.clone());
                    }
                    if node
                        .labels
                        .iter()
                        .any(|label| label == COLD_SNAPSHOT_STUB_LABEL)
                        && node
                            .properties
                            .get(COLD_SNAPSHOT_STUB_PROPERTY)
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                    {
                        cold_stub_ids.push(node.id.clone());
                    }
                    if !node.tombstone {
                        live_nodes = live_nodes.saturating_add(1);
                    }
                    recovered.apply_recovered_node(node)?;
                }
                GraphPayloadRecord::Edge(edge) => {
                    if !edge.tombstone {
                        live_edges = live_edges.saturating_add(1);
                    }
                    match recovered.apply_recovered_edge(edge) {
                        Ok(()) => {}
                        Err(error) if is_recoverable_orphan_edge(&error) => {}
                        Err(error) => return Err(error),
                    }
                }
            }
        }
        if expected_kind == &GraphObjectKind::Node {
            if edges_fully_resident {
                recovered.reserve_recovered_edges(retained_edge_records);
            }
            profile.capture(
                "live_state_assembly",
                Some("node_label_property_indexes"),
                journal_bytes,
                live_nodes,
                live_edges,
            );
        } else {
            profile.capture(
                "live_state_assembly",
                Some("edge_type_adjacency_indexes"),
                journal_bytes,
                live_nodes,
                if edges_fully_resident { live_edges } else { 0 },
            );
        }
    }

    for id in cold_stub_ids {
        recovered.evict_node(&id)?;
    }
    profile.capture(
        "live_state_assembly",
        Some("cold_residency_filter"),
        journal_bytes,
        live_nodes,
        if edges_fully_resident { live_edges } else { 0 },
    );

    recovered.version = recovered.version.max(graph_version);
    recovered.historical_node_ids = historical_node_ids;
    // The rooted indexes contain every retained key, but do not prove that a
    // legacy snapshot never discarded an older id.
    recovered.historical_node_history_complete = false;
    // Hydration replays records one at a time and then installs the historical
    // registry wholesale, which no per-record delta can express. Re-derive the
    // maintained committed footprint from the graph just assembled so the live
    // store's quota value is exact; recovery may spend O(graph) here.
    recovered.reset_committed_footprint_from_recompute();
    Ok(HydratedInMemoryGraph {
        store: recovered,
        live_nodes,
        live_edges: if edges_fully_resident { live_edges } else { 0 },
        resident_edge_bytes_estimate,
        edges_fully_resident,
    })
}

pub(crate) struct HydratedInMemoryGraph {
    pub(crate) store: InMemoryGraphStore,
    pub(crate) live_nodes: usize,
    pub(crate) live_edges: usize,
    pub(crate) resident_edge_bytes_estimate: usize,
    pub(crate) edges_fully_resident: bool,
}

/// Conservative byte accounting for one recovered edge and its three
/// secondary-index memberships.
///
/// The serialized body captures every owned string and JSON scalar. Multiplying
/// it accounts for `serde_json::Value` containers, B-tree nodes, and allocator
/// metadata that the logical graph-size estimator cannot see. The fixed term
/// covers the inline record plus primary/secondary index entries.
pub(crate) fn estimated_resident_edge_bytes(edge: &EdgeRecord) -> GraphStoreResult<usize> {
    const HEAP_EXPANSION_FACTOR: usize = 4;
    const INDEX_AND_ALLOCATOR_OVERHEAD_BYTES: usize = 512;

    let serialized = serde_json::to_vec(edge).map_err(|error| {
        GraphStoreError::new(
            "graph_payload_edge_budget_encode",
            format!(
                "could not account recovered edge {:?} against the resident byte budget: {error}",
                edge.id
            ),
        )
    })?;
    Ok(serialized
        .len()
        .saturating_mul(HEAP_EXPANSION_FACTOR)
        .saturating_add(std::mem::size_of::<EdgeRecord>())
        .saturating_add(INDEX_AND_ALLOCATOR_OVERHEAD_BYTES))
}
fn validated_graph_payload_record(
    store: &dyn ChunkStore,
    logical: &GraphTreeEntry,
    payload: &GraphTreeEntry,
    graph_version: u64,
) -> GraphStoreResult<GraphPayloadRecord> {
    if logical.key != payload.key
        || logical.kind != payload.kind
        || logical.tombstone != payload.tombstone
    {
        return Err(GraphStoreError::new(
            "graph_payload_index_mismatch",
            format!(
                "logical entry {} {} tombstone={} does not match payload entry {} {} tombstone={}",
                logical.kind.as_str(),
                logical.key,
                logical.tombstone,
                payload.kind.as_str(),
                payload.key,
                payload.tombstone
            ),
        ));
    }
    let bytes = store.get(payload.object_hash)?.ok_or_else(|| {
        GraphStoreError::new(
            "graph_payload_missing",
            format!(
                "payload index entry {:?} points to missing chunk {}",
                payload.key, payload.object_hash
            ),
        )
    })?;
    let record = decode_canonical_graph_payload(payload.object_hash, &bytes)?;
    if record.key() != logical.key || record.kind() != logical.kind {
        return Err(GraphStoreError::new(
            "graph_payload_record_mismatch",
            format!(
                "decoded payload {} {} does not match logical entry {} {}",
                record.kind().as_str(),
                record.key(),
                logical.kind.as_str(),
                logical.key
            ),
        ));
    }
    let logical_hash = record.logical_hash()?;
    if logical_hash != logical.object_hash {
        return Err(GraphStoreError::new(
            "graph_payload_logical_hash_mismatch",
            format!(
                "payload {:?} recomputes logical hash {logical_hash}, expected {}",
                logical.key, logical.object_hash
            ),
        ));
    }
    if logical.tombstone && !record.tombstone() {
        recovery_tombstone(record, graph_version)
    } else if record.tombstone() != logical.tombstone {
        Err(GraphStoreError::new(
            "graph_payload_record_mismatch",
            format!(
                "decoded payload {} {} tombstone={} does not match logical tombstone={}",
                record.kind().as_str(),
                record.key(),
                record.tombstone(),
                logical.tombstone
            ),
        ))
    } else {
        Ok(record)
    }
}

fn recovery_tombstone(
    record: GraphPayloadRecord,
    graph_version: u64,
) -> GraphStoreResult<GraphPayloadRecord> {
    match record {
        GraphPayloadRecord::Node(mut node) => {
            node.tombstone = true;
            node.version = graph_version;
            node.content_hash = None;
            node.content_hash = Some(node.checksum()?);
            Ok(GraphPayloadRecord::Node(node))
        }
        GraphPayloadRecord::Edge(mut edge) => {
            edge.tombstone = true;
            edge.version = graph_version;
            edge.content_hash = None;
            edge.content_hash = Some(edge.checksum()?);
            Ok(GraphPayloadRecord::Edge(edge))
        }
    }
}

/// Construct the compact prior-pack descriptor needed to continue an
/// incremental chain after journal recovery. Only fields consumed by
/// incremental compilation carry recovered state; no JSON object or tree
/// transport is fabricated.
pub fn recovered_graph_pack_descriptor(
    tree: GraphProllyTree,
    graph_version: u64,
    nodes_total: usize,
    edges_total: usize,
    commit_hash: Hash,
) -> GraphStoreResult<CompiledGraphPack> {
    let live_total = checked_graph_count_add(nodes_total, edges_total, "recovered live count")?;
    if live_total > tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_recovered_count_invalid",
            format!(
                "recovered live count {live_total} exceeds physical tree count {}",
                tree.entries_total
            ),
        ));
    }
    let manifest = GraphPackManifest {
        chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        name: "recovered-redcore".to_string(),
        branch: DEFAULT_GRAPH_BRANCH.to_string(),
        commit_hash,
        tree_hash: tree.root,
        graph_version,
        nodes_total,
        edges_total,
        objects_total: tree.entries_total,
    };
    Ok(CompiledGraphPack {
        chunk_format_version: GRAPH_CHUNK_FORMAT_VERSION,
        protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        commit: GraphCommit {
            commit_hash,
            tree_hash: tree.root,
            branch: DEFAULT_GRAPH_BRANCH.to_string(),
            parent_commits: Vec::new(),
            author: String::new(),
            message: String::new(),
            timestamp_unix_ms: 0,
            graph_version,
            historical_registry_hash: None,
            objects_total: tree.entries_total,
            compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        },
        manifest,
        tree,
        tree_transport_complete: false,
        tree_chunks: Vec::new(),
        capabilities: Vec::new(),
        objects: Vec::new(),
        archive_objects: Vec::new(),
        historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: false,
    })
}

/// Retarget a compiled pack to the exact graph version assigned by RedCore.
/// RedCore advances versions per effective mutation, while the standalone
/// versioned-graph API advances once per batch. Journal publication calls this
/// after incremental compilation so commit and manifest metadata match the
/// operating store without rebuilding the already-canonical tree.
pub fn retarget_graph_pack_version(
    mut pack: CompiledGraphPack,
    graph_version: u64,
) -> GraphStoreResult<CompiledGraphPack> {
    if pack.commit.tree_hash != pack.tree.root || pack.manifest.tree_hash != pack.tree.root {
        return Err(GraphStoreError::new(
            "graph_pack_tree_mismatch",
            format!(
                "pack tree root {} does not match commit {} and manifest {}",
                pack.tree.root, pack.commit.tree_hash, pack.manifest.tree_hash
            ),
        ));
    }
    if pack.commit.objects_total != pack.tree.entries_total
        || pack.manifest.objects_total != pack.tree.entries_total
    {
        return Err(GraphStoreError::new(
            "graph_pack_object_count_mismatch",
            format!(
                "pack tree has {} entries, commit declares {}, manifest declares {}",
                pack.tree.entries_total, pack.commit.objects_total, pack.manifest.objects_total
            ),
        ));
    }

    pack.commit.graph_version = graph_version;
    pack.commit.commit_hash = Hash::of(&canonical_commit_preimage_from_fields(
        pack.commit.tree_hash,
        &pack.commit.parent_commits,
        &pack.commit.author,
        &pack.commit.message,
        pack.commit.timestamp_unix_ms,
        graph_version,
        pack.commit.historical_registry_hash,
    )?);
    pack.manifest.graph_version = graph_version;
    pack.manifest.commit_hash = pack.commit.commit_hash;

    let name = pack.manifest.name.clone();
    let manifest_body = serialize_graph_value(&pack.manifest, "graph_manifest_serialize")?;
    let validator_body = json!({
        "validator": "rustyred.verify_tree_root",
        "tree_hash": pack.tree.root,
        "objects_total": pack.tree.entries_total,
        "protocol_version": VERSIONED_GRAPH_PROTOCOL_VERSION,
    });
    pack.capabilities = vec![
        GraphCompilerCapability {
            name: format!("{name}.manifest"),
            kind: "graph_manifest".to_string(),
            mime_type: "application/json".to_string(),
            content_hash: stable_hash(&manifest_body)?,
            body: manifest_body,
        },
        GraphCompilerCapability {
            name: format!("{name}.verify_tree_root"),
            kind: "validator".to_string(),
            mime_type: "application/json".to_string(),
            content_hash: stable_hash(&validator_body)?,
            body: validator_body,
        },
    ];
    Ok(pack)
}

fn insert_unique_payload_record(
    records: &mut BTreeMap<String, GraphPayloadRecord>,
    record: GraphPayloadRecord,
) -> GraphStoreResult<()> {
    let key = record.key();
    validate_graph_payload_key(&record.kind(), &key)?;
    if records.insert(key.clone(), record).is_some() {
        return Err(GraphStoreError::new(
            "graph_payload_duplicate_key",
            format!("snapshot contains duplicate recovery payload key {key:?}"),
        ));
    }
    Ok(())
}

fn write_graph_payload_records(
    store: &dyn ChunkStore,
    records: impl IntoIterator<Item = GraphPayloadRecord>,
) -> GraphStoreResult<Vec<GraphTreeEntry>> {
    let mut encoded = Vec::new();
    let mut entries = Vec::new();
    for record in records {
        let bytes = canonical_graph_payload_preimage(&record)?;
        let address = Hash::of(&bytes);
        entries.push(GraphTreeEntry {
            key: record.key(),
            kind: record.kind(),
            object_hash: address,
            tombstone: record.tombstone(),
        });
        encoded.push((address, bytes));
    }
    let borrowed = encoded
        .iter()
        .map(|(address, bytes)| (*address, bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write(&borrowed)?;
    Ok(entries)
}

pub fn build_prolly_tree(
    store: &dyn ChunkStore,
    objects: &[GraphContentObject],
) -> GraphStoreResult<GraphProllyTree> {
    let entries = objects
        .iter()
        .map(|object| GraphTreeEntry {
            key: object.key.clone(),
            kind: object.kind.clone(),
            object_hash: object.hash,
            tombstone: object.tombstone,
        })
        .collect::<Vec<_>>();
    build_prolly_tree_from_entries(store, entries)
}

/// Full Prolly-tree build from an already-prepared, key-sorted entry list. The
/// reference (oracle) build: `build_prolly_tree` and the dual-commit validation
/// path both route through it so the incremental commit is checked against one
/// canonical chunking. Content-defined boundaries (`should_split_chunk`) give
/// structural sharing across commits via stable per-node hashes.
pub fn build_prolly_tree_from_entries(
    store: &dyn ChunkStore,
    entries: Vec<GraphTreeEntry>,
) -> GraphStoreResult<GraphProllyTree> {
    build_prolly_tree_from_entries_with_transport(store, entries).map(|build| build.tree)
}

struct StoredTreeBuild {
    tree: GraphProllyTree,
    tree_chunks: Vec<GraphTreeChunk>,
}

fn build_prolly_tree_from_entries_with_transport(
    store: &dyn ChunkStore,
    entries: Vec<GraphTreeEntry>,
) -> GraphStoreResult<StoredTreeBuild> {
    let entries_total = entries.len();
    let mut current_level = leaf_nodes(entries)?;
    if current_level.is_empty() {
        current_level.push(empty_leaf_node()?);
    }
    let mut tree_chunks = Vec::new();
    let mut cost = TreeBuildCost::default();
    persist_tree_nodes(store, &current_level, &mut tree_chunks, &mut cost)?;

    let mut level = 1u32;
    while current_level.len() > 1 {
        let parents = parent_nodes(&current_level, level)?;
        persist_tree_nodes(store, &parents, &mut tree_chunks, &mut cost)?;
        current_level = parents;
        level += 1;
    }

    let root = current_level
        .into_iter()
        .next()
        .expect("empty graphs synthesize one canonical leaf before persistence");
    Ok(StoredTreeBuild {
        tree: GraphProllyTree {
            root: root.hash,
            entries_total,
        },
        tree_chunks,
    })
}

fn persist_tree_nodes(
    store: &dyn ChunkStore,
    nodes: &[GraphTreeNode],
    tree_chunks: &mut Vec<GraphTreeChunk>,
    cost: &mut TreeBuildCost,
) -> GraphStoreResult<()> {
    let mut encoded = Vec::with_capacity(nodes.len());
    for node in nodes {
        let bytes = canonical_tree_node_preimage(node)?;
        let address = Hash::of(&bytes);
        if address != node.hash {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_address_mismatch",
                format!(
                    "tree node metadata carries {}, but canonical bytes hash to {address}",
                    node.hash
                ),
            ));
        }
        encoded.push(GraphTreeChunk { address, bytes });
    }

    let addresses = encoded
        .iter()
        .map(|chunk| chunk.address)
        .collect::<Vec<_>>();
    let missing = store
        .missing(&addresses)?
        .into_iter()
        .collect::<HashSet<_>>();
    let novel = encoded
        .into_iter()
        .filter(|chunk| missing.contains(&chunk.address))
        .collect::<Vec<_>>();
    let borrowed = novel
        .iter()
        .map(|chunk| (chunk.address, chunk.bytes.as_slice()))
        .collect::<Vec<_>>();
    let ChunkWriteReport {
        chunks_written,
        bytes_written,
    } = store.write_accounted(&borrowed)?;
    let reused = nodes.len().checked_sub(chunks_written).ok_or_else(|| {
        GraphStoreError::new(
            "graph_commit_cost_invalid",
            format!(
                "store reported {chunks_written} writes for {} input tree nodes",
                nodes.len()
            ),
        )
    })?;
    cost.chunks_written = cost
        .chunks_written
        .checked_add(chunks_written)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_commit_cost_overflow",
                "written chunk count overflowed usize",
            )
        })?;
    cost.bytes_written = cost
        .bytes_written
        .checked_add(bytes_written)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_commit_cost_overflow",
                "written byte count overflowed usize",
            )
        })?;
    cost.chunks_reused = cost.chunks_reused.checked_add(reused).ok_or_else(|| {
        GraphStoreError::new(
            "graph_commit_cost_overflow",
            "reused chunk count overflowed usize",
        )
    })?;
    tree_chunks.extend(novel);
    Ok(())
}

fn install_tree_chunks(
    store: &dyn ChunkStore,
    chunks: &[GraphTreeChunk],
) -> GraphStoreResult<ChunkWriteReport> {
    for chunk in chunks {
        decode_canonical_tree_node(chunk.address, &chunk.bytes)?;
    }
    let borrowed = chunks
        .iter()
        .map(|chunk| (chunk.address, chunk.bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write_accounted(&borrowed)
}

struct RepositoryChunkStore<'a> {
    chunks: &'a BTreeMap<Hash, Vec<u8>>,
}

impl<'a> RepositoryChunkStore<'a> {
    fn new(chunks: &'a BTreeMap<Hash, Vec<u8>>) -> Self {
        Self { chunks }
    }
}

impl ChunkStore for RepositoryChunkStore<'_> {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let Some(bytes) = self.chunks.get(&address) else {
            return Ok(None);
        };
        let actual = Hash::of(bytes);
        if actual != address {
            return Err(GraphStoreError::new(
                "graph_tree_chunk_address_mismatch",
                format!("repository tree chunk bytes hash to {actual}, expected {address}"),
            ));
        }
        Ok(Some(Arc::from(bytes.as_slice())))
    }

    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.get(address).map(|bytes| bytes.is_some())
    }

    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        for address in addresses {
            if seen.insert(*address) && !self.has(*address)? {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, _chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        Err(read_only_repository_store())
    }

    fn write_accounted(&self, _chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        Err(read_only_repository_store())
    }

    fn commit(&self, _root: Hash, _meta: crate::chunk_store::CommitMeta) -> GraphStoreResult<Hash> {
        Err(read_only_repository_store())
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        Ok(None)
    }
}

fn read_only_repository_store() -> GraphStoreError {
    GraphStoreError::new(
        "graph_repository_chunk_store_read_only",
        "serialized graph repository chunks are a read-only ChunkStore view",
    )
}

struct ProllyEntryCursor<'a> {
    reader: ProllyReader<'a, dyn ChunkStore + 'a>,
    stack: Vec<(Hash, usize, Option<GraphTreeChild>)>,
    pending: std::vec::IntoIter<GraphTreeEntry>,
    visited: HashSet<Hash>,
    expected_entries: usize,
    yielded_entries: usize,
    finished: bool,
    #[cfg(test)]
    peak_leaf_entries: usize,
}

impl<'a> ProllyEntryCursor<'a> {
    fn new(store: &'a dyn ChunkStore, tree: &GraphProllyTree) -> Self {
        Self {
            reader: ProllyReader::new(store),
            stack: vec![(tree.root, 0, None)],
            pending: Vec::new().into_iter(),
            visited: HashSet::new(),
            expected_entries: tree.entries_total,
            yielded_entries: 0,
            finished: false,
            #[cfg(test)]
            peak_leaf_entries: 0,
        }
    }

    fn next_entry(&mut self) -> GraphStoreResult<Option<GraphTreeEntry>> {
        if self.finished {
            return Ok(None);
        }
        loop {
            if let Some(entry) = self.pending.next() {
                self.yielded_entries = self.yielded_entries.checked_add(1).ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_tree_chunk_count_overflow",
                        "tree entry count overflowed usize",
                    )
                })?;
                return Ok(Some(entry));
            }
            let Some((address, depth, expected)) = self.stack.pop() else {
                self.finished = true;
                if self.yielded_entries != self.expected_entries {
                    return Err(GraphStoreError::new(
                        "graph_tree_entry_count_mismatch",
                        format!(
                            "tree declares {} entries but traversal found {}",
                            self.expected_entries, self.yielded_entries
                        ),
                    ));
                }
                return Ok(None);
            };
            guard_repository_tree_visit(&address, depth, &mut self.visited)?;
            let node = self.reader.read(address)?;
            if let Some(expected) = expected {
                if node.first_key() != expected.first_key
                    || node.last_key() != expected.last_key
                    || node.entries_total() != expected.entries_total
                {
                    return Err(GraphStoreError::new(
                        "graph_tree_child_descriptor_mismatch",
                        format!(
                            "parent descriptor for {} does not match the fetched child",
                            expected.hash
                        ),
                    ));
                }
            }
            match node.as_ref() {
                StoredProllyNode::Leaf(entries) => {
                    #[cfg(test)]
                    {
                        self.peak_leaf_entries = self.peak_leaf_entries.max(entries.len());
                    }
                    self.pending = entries.clone().into_iter();
                }
                StoredProllyNode::Parent(children) => {
                    for child in children.iter().rev() {
                        self.stack
                            .push((child.hash, depth + 1, Some(child.clone())));
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn peak_leaf_entries(&self) -> usize {
        self.peak_leaf_entries
    }
}
pub(crate) fn collect_tree_entries(
    store: &dyn ChunkStore,
    tree: &GraphProllyTree,
) -> GraphStoreResult<Vec<GraphTreeEntry>> {
    let mut reader = ProllyReader::new(store);
    let mut entries = Vec::with_capacity(tree.entries_total);
    let mut visited = HashSet::new();
    collect_tree_entries_at_depth(&mut reader, tree.root, 0, &mut visited, &mut entries)?;
    if entries.len() != tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            format!(
                "tree {} declares {} entries but traversal found {}",
                tree.root,
                tree.entries_total,
                entries.len()
            ),
        ));
    }
    Ok(entries)
}

fn collect_tree_entries_at_depth(
    reader: &mut ProllyReader<'_, dyn ChunkStore + '_>,
    address: Hash,
    depth: usize,
    visited: &mut HashSet<Hash>,
    out: &mut Vec<GraphTreeEntry>,
) -> GraphStoreResult<(String, String, usize)> {
    guard_repository_tree_visit(&address, depth, visited)?;
    let node = reader.read(address)?;
    match node.as_ref() {
        StoredProllyNode::Leaf(entries) => {
            out.extend(entries.iter().cloned());
            Ok((
                node.first_key().to_string(),
                node.last_key().to_string(),
                entries.len(),
            ))
        }
        StoredProllyNode::Parent(children) => {
            let mut total = 0usize;
            for child in children {
                let (first_key, last_key, entries_total) =
                    collect_tree_entries_at_depth(reader, child.hash, depth + 1, visited, out)?;
                if first_key != child.first_key
                    || last_key != child.last_key
                    || entries_total != child.entries_total
                {
                    return Err(GraphStoreError::new(
                        "graph_tree_child_descriptor_mismatch",
                        format!(
                            "parent descriptor for {} does not match the fetched child",
                            child.hash
                        ),
                    ));
                }
                total = total.checked_add(entries_total).ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_tree_chunk_count_overflow",
                        "tree entry count overflowed usize",
                    )
                })?;
            }
            Ok((
                node.first_key().to_string(),
                node.last_key().to_string(),
                total,
            ))
        }
    }
}

/// Build the content-addressed object for a node, payload included. The cold
/// tier (storage spine, cut 6) commits this on eviction: `key = node/{id}`,
/// `hash` is the node's content hash, and `payload` is the full serialized
/// `NodeRecord` so a later rehydration reconstructs the node exactly. Mirrors
/// what `snapshot_content_objects` produces per node, exposed for the single-
/// node commit-on-evict path.
pub fn node_to_content_object(node: &NodeRecord) -> GraphStoreResult<GraphContentObject> {
    node_content_object(node, true)
}

/// Edge counterpart of [`node_to_content_object`] (warm-tier scope parking
/// commits incident edges alongside their nodes).
pub fn edge_to_content_object(edge: &EdgeRecord) -> GraphStoreResult<GraphContentObject> {
    edge_content_object(edge, true)
}

/// Reconstruct a `NodeRecord` from a content object fetched out of the cold
/// tier. Inverse of [`node_to_content_object`]; returns `None` when the object
/// is not a node or carries no payload.
pub fn node_from_content_object(object: &GraphContentObject) -> Option<NodeRecord> {
    if object.kind != GraphObjectKind::Node {
        return None;
    }
    serde_json::from_value(object.payload.clone()?).ok()
}

/// Reconstruct an `EdgeRecord` from a content object fetched out of the cold
/// tier. Inverse of [`edge_to_content_object`].
pub fn edge_from_content_object(object: &GraphContentObject) -> Option<EdgeRecord> {
    if object.kind != GraphObjectKind::Edge {
        return None;
    }
    serde_json::from_value(object.payload.clone()?).ok()
}

pub fn object_bytes<'a>(pack: &'a CompiledGraphPack, hash: &str) -> Option<&'a [u8]> {
    let hash = Hash::parse_prefixed(hash).ok()?;
    pack.archive_objects
        .iter()
        .find(|object| object.hash == hash)
        .map(|object| object.bytes.as_slice())
}

fn node_content_object(
    node: &NodeRecord,
    include_payload: bool,
) -> GraphStoreResult<GraphContentObject> {
    let logical_hash = match &node.content_hash {
        Some(hash) => Hash::parse_prefixed(hash)?,
        None => Hash::parse_prefixed(&node.checksum()?)?,
    };
    let parent_hashes = node
        .parent_hashes
        .iter()
        .map(|hash| Hash::parse_prefixed(hash))
        .collect::<GraphStoreResult<Vec<_>>>()?;
    let hash = Hash::of(&canonical_node_preimage(node)?);
    let payload = if include_payload {
        Some(serialize_graph_value(node, "graph_node_payload_serialize")?)
    } else {
        None
    };
    Ok(GraphContentObject {
        key: format!("node/{}", node.id),
        kind: GraphObjectKind::Node,
        hash,
        logical_hash,
        parent_hashes,
        tombstone: node.tombstone,
        payload,
    })
}

fn edge_content_object(
    edge: &EdgeRecord,
    include_payload: bool,
) -> GraphStoreResult<GraphContentObject> {
    let logical_hash = match &edge.content_hash {
        Some(hash) => Hash::parse_prefixed(hash)?,
        None => Hash::parse_prefixed(&edge.checksum()?)?,
    };
    let parent_hashes = edge
        .parent_hashes
        .iter()
        .map(|hash| Hash::parse_prefixed(hash))
        .collect::<GraphStoreResult<Vec<_>>>()?;
    let hash = Hash::of(&canonical_edge_preimage(edge)?);
    let payload = if include_payload {
        Some(serialize_graph_value(edge, "graph_edge_payload_serialize")?)
    } else {
        None
    };
    Ok(GraphContentObject {
        key: format!("edge/{}", edge.id),
        kind: GraphObjectKind::Edge,
        hash,
        logical_hash,
        parent_hashes,
        tombstone: edge.tombstone,
        payload,
    })
}

fn leaf_nodes(entries: Vec<GraphTreeEntry>) -> GraphStoreResult<Vec<GraphTreeNode>> {
    if entries.is_empty() {
        return Ok(vec![empty_leaf_node()?]);
    }
    chunk_by_boundary(
        entries,
        |entry: &GraphTreeEntry| entry.object_hash.trailing_u16(),
        |chunk| make_leaf_node(chunk.to_vec()),
    )
}

fn parent_nodes(children: &[GraphTreeNode], level: u32) -> GraphStoreResult<Vec<GraphTreeNode>> {
    let descriptors = children
        .iter()
        .map(child_descriptor)
        .collect::<GraphStoreResult<Vec<_>>>()?;
    chunk_by_boundary(
        descriptors,
        |child: &GraphTreeChild| child.hash.trailing_u16(),
        |chunk| make_parent_node(level, chunk.to_vec()),
    )
}

fn chunk_by_boundary<T: Clone>(
    items: Vec<T>,
    boundary: impl Fn(&T) -> u16,
    make_node: impl Fn(&[T]) -> GraphStoreResult<GraphTreeNode>,
) -> GraphStoreResult<Vec<GraphTreeNode>> {
    let mut nodes = Vec::new();
    let mut chunk = Vec::new();
    for item in items {
        let item_boundary = boundary(&item);
        chunk.push(item);
        if should_split_chunk(chunk.len(), item_boundary) {
            nodes.push(make_node(&chunk)?);
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        nodes.push(make_node(&chunk)?);
    }
    Ok(nodes)
}

fn make_leaf_node(entries: Vec<GraphTreeEntry>) -> GraphStoreResult<GraphTreeNode> {
    let first_key = entries
        .first()
        .map(|entry| entry.key.clone())
        .unwrap_or_default();
    let last_key = entries
        .last()
        .map(|entry| entry.key.clone())
        .unwrap_or_default();
    let hash = Hash::of(&canonical_leaf_preimage(&entries)?);
    Ok(GraphTreeNode {
        hash,
        level: 0,
        first_key,
        last_key,
        entries,
        children: Vec::new(),
    })
}

fn make_parent_node(level: u32, children: Vec<GraphTreeChild>) -> GraphStoreResult<GraphTreeNode> {
    let first_key = children
        .first()
        .map(|child| child.first_key.clone())
        .unwrap_or_default();
    let last_key = children
        .last()
        .map(|child| child.last_key.clone())
        .unwrap_or_default();
    let hash = Hash::of(&canonical_parent_preimage(&children)?);
    Ok(GraphTreeNode {
        hash,
        level,
        first_key,
        last_key,
        entries: Vec::new(),
        children,
    })
}

fn empty_leaf_node() -> GraphStoreResult<GraphTreeNode> {
    make_leaf_node(Vec::new())
}

pub(crate) fn build_commit(
    tree: &GraphProllyTree,
    branch: &str,
    parent_commits: Vec<Hash>,
    author: &str,
    message: &str,
    timestamp_unix_ms: u128,
    graph_version: u64,
    historical_registry_hash: Option<Hash>,
) -> GraphStoreResult<GraphCommit> {
    let preimage = canonical_commit_preimage_from_fields(
        tree.root,
        &parent_commits,
        author,
        message,
        timestamp_unix_ms,
        graph_version,
        historical_registry_hash,
    )?;
    let commit_hash = Hash::of(&preimage);

    Ok(GraphCommit {
        commit_hash,
        tree_hash: tree.root,
        branch: branch.to_string(),
        parent_commits,
        author: author.to_string(),
        message: message.to_string(),
        timestamp_unix_ms,
        graph_version,
        historical_registry_hash,
        objects_total: tree.entries_total,
        compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
    })
}

fn should_split_chunk(len: usize, boundary: u16) -> bool {
    if len >= PROLLY_MAX_ENTRIES {
        return true;
    }
    len >= PROLLY_MIN_ENTRIES && boundary < PROLLY_SPLIT_THRESHOLD
}

fn serialize_graph_value<T: Serialize>(
    value: T,
    error_code: &'static str,
) -> GraphStoreResult<Value> {
    serde_json::to_value(value).map_err(|error| GraphStoreError::new(error_code, error.to_string()))
}

fn graph_archive_error(error: crate::zerocopy::ZeroCopyArchiveError) -> GraphStoreError {
    GraphStoreError::new(error.code, error.message)
}

fn clean_or_default(value: Option<String>, default: &str) -> String {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn parse_hash_list(values: Vec<String>) -> GraphStoreResult<Vec<Hash>> {
    values
        .into_iter()
        .map(|value| Hash::parse_prefixed(&value))
        .collect()
}

impl GraphTreeNode {
    fn entry_count(&self) -> GraphStoreResult<usize> {
        if self.level == 0 {
            Ok(self.entries.len())
        } else {
            self.children.iter().try_fold(0usize, |total, child| {
                total.checked_add(child.entries_total).ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_tree_chunk_count_overflow",
                        "tree-node child entry counts overflowed usize",
                    )
                })
            })
        }
    }
}

fn resolve_repository_target(repository: &GraphVersionRepository, target: &str) -> Option<Hash> {
    let parsed_target = Hash::parse_prefixed(target).ok();
    repository
        .refs
        .iter()
        .find(|reference| reference.name == target)
        .map(|reference| reference.commit_hash)
        .or_else(|| {
            let target = parsed_target?;
            repository
                .packs
                .iter()
                .find(|pack| pack.commit.commit_hash == target)
                .map(|pack| pack.commit.commit_hash)
        })
        .or_else(|| {
            let target = parsed_target?;
            repository
                .commits
                .iter()
                .find(|commit| commit.commit_hash == target)
                .map(|commit| commit.commit_hash)
        })
}

/// Verify one object's membership in an already-retained repository tree.
/// Compacted packs omit their object vectors; the tree's addressed entry is
/// the authority for selecting the matching version from repository.objects.
pub(crate) fn repository_object_at<'a>(
    repository: &'a GraphVersionRepository,
    tree: &GraphProllyTree,
    key: &str,
) -> GraphStoreResult<Option<&'a GraphContentObject>> {
    let store = RepositoryChunkStore::new(&repository.tree_chunks);
    let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
    if reader.read(tree.root)?.entries_total() != tree.entries_total {
        return Err(GraphStoreError::new(
            "graph_tree_entry_count_mismatch",
            "repository tree descriptor differs from its addressed root",
        ));
    }
    let Some(entry) = find_tree_entry(&mut reader, tree, key)? else {
        return Ok(None);
    };
    if entry.tombstone {
        return Ok(None);
    }
    let object = repository.objects.get(&entry.object_hash).ok_or_else(|| {
        GraphStoreError::new(
            "graph_checkout_missing_object",
            format!("missing graph object {}", entry.object_hash),
        )
    })?;
    if object.key != entry.key
        || object.kind != entry.kind
        || object.tombstone != entry.tombstone
        || object.hash != entry.object_hash
        || recompute_content_object_hash(object)? != entry.object_hash
    {
        return Err(GraphStoreError::new(
            "graph_checkout_object_mismatch",
            "retained object does not match its addressed tree entry",
        ));
    }
    Ok(Some(object))
}

fn graph_snapshot_from_repository(
    repository: &GraphVersionRepository,
    pack: &CompiledGraphPack,
) -> GraphStoreResult<GraphSnapshot> {
    if !pack.objects.is_empty() {
        return graph_snapshot_from_pack(pack).ok_or_else(|| {
            GraphStoreError::new(
                "graph_checkout_corrupt_payload",
                format!(
                    "graph pack {} contains an invalid payload",
                    pack.commit.commit_hash
                ),
            )
        });
    }
    let store = RepositoryChunkStore::new(&repository.tree_chunks);
    let entries = collect_tree_entries(&store, &pack.tree)?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for entry in entries {
        if entry.tombstone {
            continue;
        }
        let object = repository.objects.get(&entry.object_hash).ok_or_else(|| {
            GraphStoreError::new(
                "graph_checkout_missing_object",
                format!("missing graph object {}", entry.object_hash),
            )
        })?;
        let payload = object.payload.clone().ok_or_else(|| {
            GraphStoreError::new(
                "graph_checkout_missing_payload",
                format!("graph object {} has no payload", entry.object_hash),
            )
        })?;
        match object.kind {
            GraphObjectKind::Node => {
                nodes.push(serde_json::from_value(payload).map_err(|error| {
                    GraphStoreError::new(
                        "graph_checkout_corrupt_payload",
                        format!("node object {} is corrupt: {error}", entry.object_hash),
                    )
                })?)
            }
            GraphObjectKind::Edge => {
                edges.push(serde_json::from_value(payload).map_err(|error| {
                    GraphStoreError::new(
                        "graph_checkout_corrupt_payload",
                        format!("edge object {} is corrupt: {error}", entry.object_hash),
                    )
                })?)
            }
        }
    }
    nodes.sort_by(|a: &NodeRecord, b| a.id.cmp(&b.id));
    edges.sort_by(|a: &EdgeRecord, b| a.id.cmp(&b.id));
    Ok(GraphSnapshot {
        version: pack.commit.graph_version,
        historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
        historical_node_history_complete: false,
        nodes,
        edges,
    })
}

fn guard_repository_tree_visit(
    hash: &Hash,
    depth: usize,
    visited: &mut HashSet<Hash>,
) -> GraphStoreResult<()> {
    if depth > MAX_REPOSITORY_TREE_DEPTH {
        return Err(GraphStoreError::new(
            "graph_repository_depth_exceeded",
            format!(
                "repository tree depth {depth} exceeds the limit {MAX_REPOSITORY_TREE_DEPTH} at {hash}"
            ),
        ));
    }
    if !visited.insert(*hash) {
        return Err(GraphStoreError::new(
            "graph_repository_cycle_detected",
            format!("repository tree node {hash} was revisited"),
        ));
    }
    Ok(())
}

fn graph_snapshot_from_pack(pack: &CompiledGraphPack) -> Option<GraphSnapshot> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for object in &pack.objects {
        if object.tombstone {
            continue;
        }
        let payload = object.payload.clone()?;
        match object.kind {
            GraphObjectKind::Node => nodes.push(serde_json::from_value(payload).ok()?),
            GraphObjectKind::Edge => edges.push(serde_json::from_value(payload).ok()?),
        }
    }
    nodes.sort_by(|a: &NodeRecord, b| a.id.cmp(&b.id));
    edges.sort_by(|a: &EdgeRecord, b| a.id.cmp(&b.id));
    Some(GraphSnapshot {
        version: pack.commit.graph_version,
        historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
        historical_node_history_complete: false,
        nodes,
        edges,
    })
}

#[derive(Clone, Debug)]
enum MergeRecord {
    Node(NodeRecord),
    Edge(EdgeRecord),
}

impl MergeRecord {
    #[cfg(test)]
    fn key(&self) -> String {
        match self {
            Self::Node(node) => format!("node/{}", node.id),
            Self::Edge(edge) => format!("edge/{}", edge.id),
        }
    }

    fn kind(&self) -> GraphObjectKind {
        match self {
            Self::Node(_) => GraphObjectKind::Node,
            Self::Edge(_) => GraphObjectKind::Edge,
        }
    }

    fn hash(&self) -> GraphStoreResult<String> {
        match self {
            Self::Node(node) => match &node.content_hash {
                Some(hash) => Ok(hash.clone()),
                None => node.checksum(),
            },
            Self::Edge(edge) => match &edge.content_hash {
                Some(hash) => Ok(hash.clone()),
                None => edge.checksum(),
            },
        }
    }
}

enum MergeDecision {
    Resolved {
        selected: Option<GraphMergeSide>,
        record: Option<MergeRecord>,
        reason: String,
    },
    Conflict {
        reason: String,
    },
}

#[cfg(test)]
fn merge_records_by_key(snapshot: &GraphSnapshot) -> BTreeMap<String, MergeRecord> {
    let mut records = BTreeMap::new();
    for node in &snapshot.nodes {
        let record = MergeRecord::Node(node.clone());
        records.insert(record.key(), record);
    }
    for edge in &snapshot.edges {
        let record = MergeRecord::Edge(edge.clone());
        records.insert(record.key(), record);
    }
    records
}

fn graph_diff_keys(diff: &GraphVersionDiff) -> BTreeSet<String> {
    diff.added
        .iter()
        .chain(&diff.removed)
        .chain(&diff.modified)
        .map(|entry| entry.key.clone())
        .collect()
}

fn merge_records_for_keys(
    snapshot: &GraphSnapshot,
    keys: &BTreeSet<String>,
) -> BTreeMap<String, MergeRecord> {
    let mut records = BTreeMap::new();
    for node in &snapshot.nodes {
        let key = format!("node/{}", node.id);
        if keys.contains(&key) {
            records.insert(key, MergeRecord::Node(node.clone()));
        }
    }
    for edge in &snapshot.edges {
        let key = format!("edge/{}", edge.id);
        if keys.contains(&key) {
            records.insert(key, MergeRecord::Edge(edge.clone()));
        }
    }
    records
}

fn append_unchanged_merge_records(
    snapshot: &GraphSnapshot,
    changed_keys: &BTreeSet<String>,
    records: &mut Vec<MergeRecord>,
) {
    records.extend(
        snapshot
            .nodes
            .iter()
            .filter(|node| !changed_keys.contains(&format!("node/{}", node.id)))
            .cloned()
            .map(MergeRecord::Node),
    );
    records.extend(
        snapshot
            .edges
            .iter()
            .filter(|edge| !changed_keys.contains(&format!("edge/{}", edge.id)))
            .cloned()
            .map(MergeRecord::Edge),
    );
}

fn graph_object_by_key<'a>(
    objects: &'a [GraphContentObject],
    key: &str,
) -> Option<&'a GraphContentObject> {
    objects
        .binary_search_by(|object| object.key.as_str().cmp(key))
        .ok()
        .map(|index| &objects[index])
}

fn required_graph_object_by_key<'a>(
    objects: &'a [GraphContentObject],
    key: &str,
    side: &str,
) -> GraphStoreResult<&'a GraphContentObject> {
    graph_object_by_key(objects, key).ok_or_else(|| {
        GraphStoreError::new(
            "graph_merge_tree_index_incomplete",
            format!("unchanged merge key {key} is missing from the {side} snapshot index"),
        )
    })
}

fn merge_record_kind(
    base: Option<&MergeRecord>,
    ours: Option<&MergeRecord>,
    theirs: Option<&MergeRecord>,
) -> GraphObjectKind {
    ours.or(theirs)
        .or(base)
        .map(MergeRecord::kind)
        .unwrap_or(GraphObjectKind::Node)
}

fn resolve_merge_record(
    base: Option<&MergeRecord>,
    ours: Option<&MergeRecord>,
    theirs: Option<&MergeRecord>,
    strategy: &GraphMergeStrategy,
    min_confidence_delta: f64,
    registry: &MergeRegistry,
) -> GraphStoreResult<MergeDecision> {
    let base_hash = base.map(MergeRecord::hash).transpose()?;
    let ours_hash = ours.map(MergeRecord::hash).transpose()?;
    let theirs_hash = theirs.map(MergeRecord::hash).transpose()?;

    if ours_hash == theirs_hash {
        return Ok(MergeDecision::Resolved {
            selected: ours.map(|_| GraphMergeSide::Ours),
            record: ours.cloned(),
            reason: "same_result".to_string(),
        });
    }
    if base_hash == ours_hash {
        return Ok(MergeDecision::Resolved {
            selected: theirs.map(|_| GraphMergeSide::Theirs),
            record: theirs.cloned(),
            reason: "theirs_only_change".to_string(),
        });
    }
    if base_hash == theirs_hash {
        return Ok(MergeDecision::Resolved {
            selected: ours.map(|_| GraphMergeSide::Ours),
            record: ours.cloned(),
            reason: "ours_only_change".to_string(),
        });
    }

    if matches!(strategy, GraphMergeStrategy::AutoConfidence) {
        if let Some(decision) = resolve_registry_merge(base, ours, theirs, registry) {
            return Ok(decision);
        }
    }

    Ok(match strategy {
        GraphMergeStrategy::PreferOurs => MergeDecision::Resolved {
            selected: ours.map(|_| GraphMergeSide::Ours),
            record: ours.cloned(),
            reason: "strategy_prefer_ours".to_string(),
        },
        GraphMergeStrategy::PreferTheirs => MergeDecision::Resolved {
            selected: theirs.map(|_| GraphMergeSide::Theirs),
            record: theirs.cloned(),
            reason: "strategy_prefer_theirs".to_string(),
        },
        GraphMergeStrategy::AutoConfidence => {
            resolve_confidence_merge(base, ours, theirs, min_confidence_delta).unwrap_or_else(
                || MergeDecision::Conflict {
                    reason: "both_sides_changed".to_string(),
                },
            )
        }
        GraphMergeStrategy::Manual => MergeDecision::Conflict {
            reason: "manual_resolution_required".to_string(),
        },
    })
}

fn resolve_registry_merge(
    base: Option<&MergeRecord>,
    ours: Option<&MergeRecord>,
    theirs: Option<&MergeRecord>,
    registry: &MergeRegistry,
) -> Option<MergeDecision> {
    let (Some(MergeRecord::Node(ours)), Some(MergeRecord::Node(theirs))) = (ours, theirs) else {
        return None;
    };
    let base = match base {
        Some(MergeRecord::Node(node)) => Some(node),
        Some(MergeRecord::Edge(_)) => return None,
        None => None,
    };
    let resolution = registry.resolve_node(base, ours, theirs)?;
    Some(MergeDecision::Resolved {
        selected: None,
        record: Some(MergeRecord::Node(resolution.node)),
        reason: resolution.reason,
    })
}

fn resolve_confidence_merge(
    base: Option<&MergeRecord>,
    ours: Option<&MergeRecord>,
    theirs: Option<&MergeRecord>,
    min_confidence_delta: f64,
) -> Option<MergeDecision> {
    let (Some(MergeRecord::Edge(ours)), Some(MergeRecord::Edge(theirs))) = (ours, theirs) else {
        return None;
    };
    let base = match base {
        Some(MergeRecord::Edge(edge)) => Some(edge),
        Some(MergeRecord::Node(_)) => return None,
        None => None,
    };
    let (side, mut edge) = resolve_auto_confidence_edge(ours, theirs, min_confidence_delta)?;
    edge.properties = merge_edge_properties(
        base.map(|edge| &edge.properties),
        &ours.properties,
        &theirs.properties,
    )?;
    Some(MergeDecision::Resolved {
        selected: Some(side),
        record: Some(MergeRecord::Edge(edge)),
        reason: "higher_edge_confidence".to_string(),
    })
}

fn merge_edge_properties(base: Option<&Value>, ours: &Value, theirs: &Value) -> Option<Value> {
    let (Some(ours), Some(theirs)) = (ours.as_object(), theirs.as_object()) else {
        return if ours == theirs {
            Some(ours.clone())
        } else if base == Some(ours) {
            Some(theirs.clone())
        } else if base == Some(theirs) {
            Some(ours.clone())
        } else {
            None
        };
    };
    let base = base.and_then(Value::as_object);
    let keys = base
        .into_iter()
        .flat_map(|map| map.keys())
        .chain(ours.keys())
        .chain(theirs.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut merged = serde_json::Map::new();
    for key in keys {
        let base_value = base.and_then(|map| map.get(&key));
        let ours_value = ours.get(&key);
        let theirs_value = theirs.get(&key);
        let value = if ours_value == theirs_value {
            ours_value
        } else if ours_value == base_value {
            theirs_value
        } else if theirs_value == base_value {
            ours_value
        } else {
            return None;
        };
        if let Some(value) = value {
            merged.insert(key, value.clone());
        }
    }
    Some(Value::Object(merged))
}

pub fn resolve_auto_confidence_edge(
    ours: &EdgeRecord,
    theirs: &EdgeRecord,
    min_confidence_delta: f64,
) -> Option<(GraphMergeSide, EdgeRecord)> {
    let delta = (ours.effective_confidence() - theirs.effective_confidence()).abs();
    if delta <= min_confidence_delta.max(0.0) {
        return None;
    }
    if ours.effective_confidence() > theirs.effective_confidence() {
        Some((GraphMergeSide::Ours, ours.clone()))
    } else {
        Some((GraphMergeSide::Theirs, theirs.clone()))
    }
}

fn snapshot_from_merge_records(version: u64, records: Vec<MergeRecord>) -> GraphSnapshot {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for record in records {
        match record {
            MergeRecord::Node(node) => nodes.push(node),
            MergeRecord::Edge(edge) => edges.push(edge),
        }
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    edges.sort_by(|a, b| a.id.cmp(&b.id));
    GraphSnapshot {
        version,
        historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
        historical_node_history_complete: false,
        nodes,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;
    use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
    use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord, Provenance};
    use crate::graph::snapshot::GraphSnapshot;
    use crate::graph_store::{RedCoreDurability, RedCoreGraphStore, RedCoreOptions};
    use crate::{ActorId, Hlc, MergeRegistry};

    struct CountingChunkStore {
        inner: MemChunkStore,
        gets: AtomicUsize,
        /// Full verified existence checks. On both concrete stores `has` reads,
        /// decompresses and SHA-256s the chunk's bytes, so this counter is a
        /// count of chunk verifications, not of cheap lookups.
        has_checks: AtomicUsize,
        /// Cheap existence probes. Counted separately so a caller can be proven
        /// to be using the probe rather than the verification, in both
        /// directions.
        probes: AtomicUsize,
    }

    impl CountingChunkStore {
        fn new() -> Self {
            Self {
                inner: MemChunkStore::new(),
                gets: AtomicUsize::new(0),
                has_checks: AtomicUsize::new(0),
                probes: AtomicUsize::new(0),
            }
        }

        fn reset_gets(&self) {
            self.gets.store(0, Ordering::Relaxed);
        }

        fn gets(&self) -> usize {
            self.gets.load(Ordering::Relaxed)
        }

        fn reset_existence_checks(&self) {
            self.has_checks.store(0, Ordering::Relaxed);
            self.probes.store(0, Ordering::Relaxed);
        }

        fn verified_existence_checks(&self) -> usize {
            self.has_checks.load(Ordering::Relaxed)
        }

        fn existence_probes(&self) -> usize {
            self.probes.load(Ordering::Relaxed)
        }
    }

    impl ChunkStore for CountingChunkStore {
        fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
            self.gets.fetch_add(1, Ordering::Relaxed);
            self.inner.get(address)
        }

        fn has(&self, address: Hash) -> GraphStoreResult<bool> {
            self.has_checks.fetch_add(1, Ordering::Relaxed);
            self.inner.has(address)
        }

        fn contains_address(&self, address: Hash) -> GraphStoreResult<bool> {
            self.probes.fetch_add(1, Ordering::Relaxed);
            self.inner.contains_address(address)
        }

        fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
            self.inner.missing(addresses)
        }

        fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
            self.inner.write(chunks)
        }

        fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
            self.inner.write_accounted(chunks)
        }

        fn commit(
            &self,
            root: Hash,
            meta: crate::chunk_store::CommitMeta,
        ) -> GraphStoreResult<Hash> {
            self.inner.commit(root, meta)
        }

        fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
            self.inner.read_root()
        }
    }

    /// A base store that can be told one address's bytes are unreadable, so the
    /// probe's blind spot can be asserted rather than assumed.
    ///
    /// `poisoned` is the address whose bytes have gone bad. `probe_lies` selects
    /// which side of the trade is under test: with it false the probe delegates
    /// to `has`, which routes to `get` and refuses; with it true the probe
    /// claims presence without reading, and only `get` refuses.
    struct PoisonedBaseStore {
        inner: MemChunkStore,
        poisoned: std::sync::Mutex<Option<Hash>>,
        probe_lies: AtomicBool,
    }

    impl PoisonedBaseStore {
        fn new() -> Self {
            Self {
                inner: MemChunkStore::new(),
                poisoned: std::sync::Mutex::new(None),
                probe_lies: AtomicBool::new(true),
            }
        }

        fn poison(&self, address: Hash) {
            *self.poisoned.lock().expect("poison slot") = Some(address);
        }

        fn set_probe_lies(&self, lies: bool) {
            self.probe_lies.store(lies, Ordering::Relaxed);
        }

        fn is_poisoned(&self, address: Hash) -> bool {
            *self.poisoned.lock().expect("poison slot") == Some(address)
        }

        fn unreadable(address: Hash) -> GraphStoreError {
            GraphStoreError::new(
                "test_poisoned_base_chunk",
                format!("base chunk {address} is unreadable"),
            )
        }
    }

    impl ChunkStore for PoisonedBaseStore {
        fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
            if self.is_poisoned(address) {
                return Err(Self::unreadable(address));
            }
            self.inner.get(address)
        }

        fn has(&self, address: Hash) -> GraphStoreResult<bool> {
            if self.is_poisoned(address) {
                return Err(Self::unreadable(address));
            }
            self.inner.has(address)
        }

        fn contains_address(&self, address: Hash) -> GraphStoreResult<bool> {
            if self.is_poisoned(address) && self.probe_lies.load(Ordering::Relaxed) {
                // Claims present without reading: exactly what a false positive
                // looks like to the delta walk.
                return Ok(true);
            }
            if self.is_poisoned(address) {
                return self.has(address);
            }
            self.inner.contains_address(address)
        }

        fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
            self.inner.missing(addresses)
        }

        fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
            self.inner.write(chunks)
        }

        fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
            self.inner.write_accounted(chunks)
        }

        fn commit(
            &self,
            root: Hash,
            meta: crate::chunk_store::CommitMeta,
        ) -> GraphStoreResult<Hash> {
            self.inner.commit(root, meta)
        }

        fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
            self.inner.read_root()
        }
    }

    fn synthetic_entries(count: usize) -> Vec<GraphTreeEntry> {
        (0..count)
            .map(|index| GraphTreeEntry {
                key: format!("node/{index:08}"),
                kind: GraphObjectKind::Node,
                object_hash: Hash::of(format!("object:{index}").as_bytes()),
                tombstone: false,
            })
            .collect()
    }

    #[test]
    fn prolly_entry_cursor_keeps_only_one_bounded_leaf_of_entries() -> GraphStoreResult<()> {
        let expected = synthetic_entries(4_096);
        let store = MemChunkStore::new();
        let tree = build_prolly_tree_from_entries(&store, expected.clone())?;
        let mut cursor = ProllyEntryCursor::new(&store, &tree);
        let mut actual = Vec::new();

        while let Some(entry) = cursor.next_entry()? {
            actual.push(entry);
        }

        assert_eq!(actual, expected);
        assert!(
            cursor.peak_leaf_entries() <= PROLLY_MAX_ENTRIES,
            "streaming recovery must not buffer the complete entry index"
        );
        Ok(())
    }

    fn decode_hex(text: &str) -> Vec<u8> {
        assert_eq!(text.len() % 2, 0);
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let digit = |byte| match byte {
                    b'0'..=b'9' => byte - b'0',
                    b'a'..=b'f' => byte - b'a' + 10,
                    _ => panic!("invalid golden hex"),
                };
                (digit(pair[0]) << 4) | digit(pair[1])
            })
            .collect()
    }

    #[test]
    fn redcore_journal_payload_metadata_round_trip_and_hydrate() -> GraphStoreResult<()> {
        let mut node = NodeRecord::new(
            "claim:ada",
            ["Claim", "Person"],
            json!({"nested": {"z": 1, "a": [true, null]}, "name": "Ada"}),
        );
        node.version = 41;
        node.content_hash = Some(Hash::of(b"legacy-node-content-hash").to_prefixed());
        node.parent_hashes = vec![
            Hash::of(b"node-parent-b").to_prefixed(),
            Hash::of(b"node-parent-a").to_prefixed(),
        ];

        let mut edge = EdgeRecord::new(
            "support:ada",
            "claim:ada",
            "SUPPORTS",
            "claim:math",
            json!({"weight": 0.75, "sources": ["paper", "notes"]}),
        )
        .with_confidence(0.875)
        .with_epistemic_type(EpistemicType::Supports)
        .with_provenance(Provenance {
            source_id: Some("source:1".to_string()),
            timestamp: Some("2026-07-18T12:00:00Z".to_string()),
            method: Some("journal-recovery".to_string()),
        });
        edge.version = 42;
        edge.content_hash = Some(Hash::of(b"legacy-edge-content-hash").to_prefixed());
        edge.parent_hashes = vec![Hash::of(b"edge-parent").to_prefixed()];

        for record in [
            GraphPayloadRecord::Node(node.clone()),
            GraphPayloadRecord::Edge(edge.clone()),
        ] {
            let bytes = canonical_graph_payload_preimage(&record)?;
            let address = Hash::of(&bytes);
            assert_eq!(decode_canonical_graph_payload(address, &bytes)?, record);
        }

        let snapshot = GraphSnapshot {
            version: 99,
            nodes: vec![node.clone()],
            edges: vec![edge.clone()],
            historical_node_ids: BTreeSet::from([
                node.id.clone(),
                "claim:historical-only".to_string(),
            ]),
            historical_node_history_complete: true,
        };
        let store = MemChunkStore::new();
        let logical_objects = snapshot_content_objects(&snapshot, false)?;
        let logical_tree = build_prolly_tree(&store, &logical_objects)?;
        let payload_tree = build_graph_payload_index(&store, &snapshot)?;
        let hydrated = hydrate_graph_snapshot_from_payload_index(
            &store,
            &logical_tree,
            &payload_tree,
            snapshot.version,
        )?;
        let direct = hydrate_in_memory_graph_from_payload_index(
            &store,
            &logical_tree,
            &payload_tree,
            snapshot.version,
            None,
        )?;

        assert_eq!(hydrated.version, snapshot.version);
        assert_eq!(hydrated.nodes, vec![node]);
        assert_eq!(hydrated.edges, vec![edge]);
        assert_eq!(
            hydrated.historical_node_ids,
            BTreeSet::from(["claim:ada".to_string()])
        );
        assert!(!hydrated.historical_node_history_complete);
        assert_eq!(direct.live_nodes, 1);
        assert_eq!(direct.live_edges, 1);
        assert!(direct.edges_fully_resident);

        let bounded = hydrate_in_memory_graph_from_payload_index(
            &store,
            &logical_tree,
            &payload_tree,
            snapshot.version,
            Some(0),
        )?;
        assert_eq!(bounded.live_nodes, 1);
        assert_eq!(bounded.live_edges, 0);
        assert_eq!(bounded.resident_edge_bytes_estimate, 0);
        assert!(!bounded.edges_fully_resident);
        assert!(bounded.store.get_node("claim:ada").is_some());
        assert!(bounded.store.get_edge("support:ada").is_none());

        let expected_store = InMemoryGraphStore::from_snapshot(hydrated.clone())?;
        assert_eq!(direct.store.snapshot(), expected_store.snapshot());
        assert_ne!(logical_tree.root, payload_tree.root);
        Ok(())
    }

    #[test]
    fn graph_payload_float_spelling_round_trips_exact_bits() -> GraphStoreResult<()> {
        let confidence = f64::from_bits(0.4_f64.to_bits() - 1);
        assert_eq!(confidence.to_string(), "0.39999999999999997");
        let mut edge = EdgeRecord::new(
            "edge:float-roundtrip",
            "node:a",
            "DERIVES",
            "node:b",
            json!({"confidence": confidence}),
        );
        edge.confidence = Some(confidence);
        edge.version = 7;
        let record = GraphPayloadRecord::Edge(edge);
        let bytes = canonical_graph_payload_preimage(&record)?;
        assert!(bytes
            .windows(b"0.39999999999999997".len())
            .any(|window| window == b"0.39999999999999997"));

        let decoded = decode_canonical_graph_payload(Hash::of(&bytes), &bytes)?;
        assert_eq!(decoded, record);
        Ok(())
    }

    #[test]
    fn redcore_journal_payload_incremental_delete_readd_is_normalized() -> GraphStoreResult<()> {
        let mut original = NodeRecord::new("node:a", ["Claim"], json!({"state": "base"}));
        original.version = 7;
        original.content_hash = Some(Hash::of(b"base-content").to_prefixed());
        let snapshot = GraphSnapshot {
            version: 7,
            nodes: vec![original.clone()],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::from([original.id.clone()]),
            historical_node_history_complete: true,
        };
        let store = MemChunkStore::new();
        let logical_tree = build_prolly_tree(&store, &snapshot_content_objects(&snapshot, false)?)?;
        let payload_tree = build_graph_payload_index(&store, &snapshot)?;
        let prior_payload_entry = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            find_tree_entry(&mut reader, &payload_tree, "node/node:a")?.expect("base payload entry")
        };
        let prior_pack = recovered_graph_pack_descriptor(
            logical_tree,
            snapshot.version,
            1,
            0,
            Hash::of(b"prior logical commit"),
        )?;

        let delete = GraphMutationBatch::new([GraphMutation::NodeDelete(original.id.clone())]);
        let deleted_pack = compile_graph_pack_incremental_with_store(
            &store,
            &prior_pack,
            &delete,
            GraphCompileOptions {
                include_payloads: false,
                ..GraphCompileOptions::default()
            },
        )?;
        let deleted_payload = update_graph_payload_index(&store, &payload_tree, &delete)?;
        let deleted_payload_entry = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            find_tree_entry(&mut reader, &deleted_payload, "node/node:a")?
                .expect("tombstoned payload entry")
        };
        assert!(deleted_payload_entry.tombstone);
        assert_eq!(
            deleted_payload_entry.object_hash, prior_payload_entry.object_hash,
            "delete retains the prior full-record payload address"
        );
        let hydrated_deleted = hydrate_graph_snapshot_from_payload_index(
            &store,
            &deleted_pack.pack.tree,
            &deleted_payload,
            8,
        )?;
        assert_eq!(hydrated_deleted.nodes.len(), 1);
        assert!(hydrated_deleted.nodes[0].tombstone);
        assert_eq!(hydrated_deleted.nodes[0].version, 8);
        assert!(hydrated_deleted.historical_node_ids.contains(&original.id));

        let mut updated = NodeRecord::new("node:a", ["Claim"], json!({"state": "readded"}));
        updated.version = 11;
        updated.parent_hashes = vec![Hash::of(b"base-content").to_prefixed()];
        // Raw caller order says upsert then delete; RedCore class ordering
        // still makes the upsert the final state in both indexes.
        let readd = GraphMutationBatch::new([
            GraphMutation::NodeUpsert(updated.clone()),
            GraphMutation::NodeDelete(updated.id.clone()),
        ]);
        let readded_pack = compile_graph_pack_incremental_with_store(
            &store,
            &deleted_pack.pack,
            &readd,
            GraphCompileOptions {
                include_payloads: false,
                ..GraphCompileOptions::default()
            },
        )?;
        let readded_payload = update_graph_payload_index(&store, &deleted_payload, &readd)?;
        let hydrated = hydrate_graph_snapshot_from_payload_index(
            &store,
            &readded_pack.pack.tree,
            &readded_payload,
            9,
        )?;
        assert_eq!(hydrated.nodes, vec![updated]);
        Ok(())
    }

    #[test]
    fn redcore_journal_full_pack_manifest_counts_only_live_records() -> GraphStoreResult<()> {
        let live_node = NodeRecord::new("node:live", ["Claim"], json!({}));
        let mut deleted_node = NodeRecord::new("node:deleted", ["Claim"], json!({}));
        deleted_node.tombstone = true;
        let live_edge = EdgeRecord::new(
            "edge:live",
            "node:live",
            "RELATED_TO",
            "node:live",
            json!({}),
        );
        let mut deleted_edge = EdgeRecord::new(
            "edge:deleted",
            "node:deleted",
            "RELATED_TO",
            "node:live",
            json!({}),
        );
        deleted_edge.tombstone = true;
        let pack = compile_graph_pack(
            &GraphSnapshot {
                version: 4,
                nodes: vec![live_node, deleted_node],
                edges: vec![live_edge, deleted_edge],
                historical_node_ids: BTreeSet::new(),
                historical_node_history_complete: false,
            },
            GraphCompileOptions::default(),
        )?;

        assert_eq!(pack.manifest.nodes_total, 1);
        assert_eq!(pack.manifest.edges_total, 1);
        assert_eq!(pack.manifest.objects_total, 4);
        Ok(())
    }

    #[test]
    fn redcore_journal_payload_corruption_and_logical_mismatch_fail_closed() -> GraphStoreResult<()>
    {
        let record = GraphPayloadRecord::Node(NodeRecord::new(
            "node:a",
            ["Claim"],
            json!({"state": "payload"}),
        ));
        let bytes = canonical_graph_payload_preimage(&record)?;
        let address = Hash::of(&bytes);
        let mut corrupt = bytes.clone();
        let last = corrupt.last_mut().expect("payload is non-empty");
        *last ^= 0x01;
        assert_eq!(
            decode_canonical_graph_payload(address, &corrupt)
                .unwrap_err()
                .code,
            "graph_payload_address_mismatch"
        );

        let logical_snapshot = GraphSnapshot {
            version: 1,
            nodes: vec![NodeRecord::new(
                "node:a",
                ["Claim"],
                json!({"state": "logical"}),
            )],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: false,
        };
        let payload_snapshot = GraphSnapshot {
            version: 1,
            nodes: vec![NodeRecord::new(
                "node:a",
                ["Claim"],
                json!({"state": "different"}),
            )],
            ..logical_snapshot.clone()
        };
        let store = MemChunkStore::new();
        let logical_tree =
            build_prolly_tree(&store, &snapshot_content_objects(&logical_snapshot, false)?)?;
        let payload_tree = build_graph_payload_index(&store, &payload_snapshot)?;
        assert_eq!(
            hydrate_graph_snapshot_from_payload_index(&store, &logical_tree, &payload_tree, 1,)
                .unwrap_err()
                .code,
            "graph_payload_logical_hash_mismatch"
        );
        Ok(())
    }

    /// The parity oracle's boundary.
    ///
    /// This test used to assert the publish path refused an injected divergence
    /// (`redcore_journal_release_sampled_commit_rejects_injected_divergence`,
    /// ignored in debug because release sampled the check). The oracle is not a
    /// publish precondition any more, so the assertion moves to the boundary and
    /// states the move in the order the cost is actually paid:
    ///
    /// 1. The store-backed incremental compile now *succeeds* and returns a pack
    ///    whose published root is the prior root while its count includes the
    ///    delta. That is the honest cost of moving an O(entries) oracle off a
    ///    write path, and it is asserted here rather than left implied.
    /// 2. The whole-graph parity audit refuses that tree, and the refusal comes
    ///    from inside the audit rather than from content addressing.
    ///
    /// The injected tree's root resolves (it is a real prior root), so this is
    /// the shape a broken address cannot model: the audit's traversal reads the
    /// stale spine, finds one entry where the pack declares two, and refuses.
    #[test]
    fn redcore_journal_injected_incremental_divergence_is_refused_by_the_boundary_audit(
    ) -> GraphStoreResult<()> {
        let snapshot = GraphSnapshot {
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({"value": 1}))],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: false,
        };
        let store = MemChunkStore::new();
        let base = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                include_payloads: false,
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )?;
        install_tree_chunks(&store, &base.tree_chunks)?;
        let prior = retarget_graph_pack_version(base, 0)?;

        let healthy = compile_graph_pack_incremental_with_store(
            &store,
            &prior,
            &GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "node:a",
                ["Claim"],
                json!({"value": 2}),
            ))]),
            GraphCompileOptions {
                include_payloads: false,
                timestamp_unix_ms: Some(2),
                ..GraphCompileOptions::default()
            },
        )?;
        let audit = audit_published_prolly_parity(&store, &healthy.pack.tree)?;
        assert!(
            audit.parity_holds(),
            "a real incremental publish must be canonical for its entries: {audit:?}"
        );
        assert_eq!(audit.live_nodes, 1);

        FORCE_INCREMENTAL_VALIDATION_DIVERGENCE.store(true, Ordering::SeqCst);
        // The delta adds an entry, so the stale spine this injects is visible to
        // the boundary audit: the root still resolves, and the count published
        // for it describes an entry the spine does not hold.
        let diverged = compile_graph_pack_incremental_with_store(
            &store,
            &prior,
            &GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "node:b",
                ["Claim"],
                json!({"value": 3}),
            ))]),
            GraphCompileOptions {
                include_payloads: false,
                timestamp_unix_ms: Some(3),
                ..GraphCompileOptions::default()
            },
        )
        .expect("the publish path no longer runs the whole-graph oracle, so it publishes");
        assert_eq!(
            diverged.pack.tree.root, prior.tree.root,
            "the injected divergence is a stale spine, not a broken address"
        );
        assert!(store.has(diverged.pack.tree.root)?);
        assert_eq!(diverged.pack.tree.entries_total, 2);

        let error = audit_published_prolly_parity(&store, &diverged.pack.tree)
            .expect_err("the boundary audit must refuse a stale published spine");
        // The traversal inside the audit catches this shape by its declared count
        // before the canonical comparison runs: the root's own descriptor says one
        // entry and the pack declares two. Either refusal is the audit refusing;
        // the comparison arm is exercised separately, on a tree that is internally
        // consistent but not canonically chunked.
        assert_eq!(error.code, "graph_tree_entry_count_mismatch");
        Ok(())
    }

    /// A diverged head does not survive a restart, and this pins *which* net
    /// refuses it, because the answer is load-bearing: recovery walks the logical
    /// and payload indexes together entry by entry, so a head whose logical tree
    /// never received a counted delta is refused during hydration -- before the
    /// recovery checkpoint runs. The checkpoint is therefore the second net at the
    /// durable boundary, and the first net is not this test's subject: what this
    /// test establishes is that the diverged head the publish path allowed through
    /// cannot be opened at all.
    #[test]
    fn redcore_journal_injected_incremental_divergence_fails_the_reopen_closed(
    ) -> GraphStoreResult<()> {
        let dir = tempfile::tempdir().expect("a temporary RedCore directory");
        let options = || RedCoreOptions {
            durability: RedCoreDurability::AofAlways,
            snapshot_interval_writes: 0,
            strict_acid: true,
        };
        let node = |id: &str, value: u64| {
            NodeRecord::new(id.to_string(), ["Claim"], json!({"value": value}))
        };

        {
            let mut store = RedCoreGraphStore::open(dir.path(), options())?;
            store.commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(node(
                "node:a", 1,
            ))]))?;
            FORCE_INCREMENTAL_VALIDATION_DIVERGENCE.store(true, Ordering::SeqCst);
            // The delta adds an entry, so the stale spine the seam injects is
            // count-visible and the logical and payload indexes disagree by entry
            // count as well.
            store.commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(node(
                "node:b", 2,
            ))]))?;
        }

        let error = RedCoreGraphStore::open(dir.path(), options())
            .expect_err("a diverged head must not open");
        assert_eq!(error.code, "graph_payload_index_mismatch");
        Ok(())
    }

    #[test]
    fn redcore_journal_prolly_sampling_is_deterministic_and_divergence_is_error(
    ) -> GraphStoreResult<()> {
        let prior_root = Hash::of(b"stable production prior root");
        let decisions = (0..64_u64)
            .map(|version| {
                prolly_validation_sample_selected(prior_root, version, 500_000)
                    .expect("valid sampling rate")
            })
            .collect::<Vec<_>>();
        let repeated = (0..64_u64)
            .map(|version| {
                prolly_validation_sample_selected(prior_root, version, 500_000)
                    .expect("valid sampling rate")
            })
            .collect::<Vec<_>>();
        assert_eq!(decisions, repeated);
        assert!(decisions.iter().any(|selected| *selected));
        assert!(decisions.iter().any(|selected| !*selected));
        assert!(prolly_validation_sample_selected(
            prior_root,
            1,
            PROLLY_VALIDATION_SAMPLE_SCALE_PPM
        )?);
        assert!(!prolly_validation_sample_selected(prior_root, 1, 0)?);
        assert_eq!(
            prolly_validation_sample_selected(
                prior_root,
                1,
                PROLLY_VALIDATION_SAMPLE_SCALE_PPM + 1
            )
            .unwrap_err()
            .code,
            "graph_prolly_validation_rate_invalid"
        );

        let store = MemChunkStore::new();
        let prior_tree = build_prolly_tree_from_entries(&store, synthetic_entries(8))?;
        let replacement = GraphTreeEntry {
            key: "node/00000003".to_string(),
            kind: GraphObjectKind::Node,
            object_hash: Hash::of(b"deliberate divergent replacement"),
            tombstone: false,
        };
        let upserts = BTreeMap::from([(replacement.key.clone(), replacement)]);
        let error = validate_incremental_prolly_parity(&store, &prior_tree, &upserts, &prior_tree)
            .unwrap_err();
        assert_eq!(error.code, "graph_prolly_validation_divergence");
        Ok(())
    }

    #[test]
    fn redcore_journal_empty_graph_root_chunk_is_persisted() -> GraphStoreResult<()> {
        let store = MemChunkStore::new();
        let tree = build_prolly_tree_from_entries(&store, Vec::new())?;
        assert_eq!(tree.entries_total, 0);
        let bytes = store
            .get(tree.root)?
            .expect("empty graph root leaf is persisted");
        assert_eq!(
            decode_canonical_tree_node(tree.root, &bytes)?,
            StoredProllyNode::Leaf(Vec::new())
        );

        let payload_tree = build_graph_payload_index(
            &store,
            &GraphSnapshot {
                version: 0,
                nodes: Vec::new(),
                edges: Vec::new(),
                historical_node_ids: BTreeSet::new(),
                historical_node_history_complete: true,
            },
        )?;
        assert_eq!(payload_tree, tree);
        assert!(store.has(payload_tree.root)?);
        Ok(())
    }

    #[test]
    fn redcore_journal_retarget_pack_version_rehashes_metadata() -> GraphStoreResult<()> {
        let snapshot = GraphSnapshot {
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: false,
        };
        let original = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                timestamp_unix_ms: Some(7),
                ..GraphCompileOptions::default()
            },
        )?;
        let original_commit = original.commit.commit_hash;
        let retargeted = retarget_graph_pack_version(original, 12)?;
        assert_eq!(retargeted.commit.graph_version, 12);
        assert_eq!(retargeted.manifest.graph_version, 12);
        assert_eq!(
            retargeted.manifest.commit_hash,
            retargeted.commit.commit_hash
        );
        assert_ne!(retargeted.commit.commit_hash, original_commit);
        assert_eq!(retargeted.capabilities.len(), 2);
        Ok(())
    }

    #[test]
    fn versioned_graph_v1_empty_leaf_hash_is_frozen() -> GraphStoreResult<()> {
        #[derive(Serialize)]
        struct LegacyTreeNodeHashInput<'a> {
            protocol_version: &'a str,
            level: u32,
            first_key: &'a str,
            last_key: &'a str,
            entries: [u8; 0],
            children: [u8; 0],
        }
        assert_eq!(
            crate::state::stable_hash_graph_v1(LegacyTreeNodeHashInput {
                protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION,
                level: 0,
                first_key: "",
                last_key: "",
                entries: [],
                children: [],
            })?,
            "sha256:e70bdbcacf6dd146f39a28edc8ef8537a5571d7d82f453527616318c205005cf"
        );
        Ok(())
    }

    #[test]
    fn versioned_graph_v2_canonical_preimages_and_hashes_are_frozen() -> GraphStoreResult<()> {
        let node = NodeRecord::new("n1", ["Claim"], json!({"z": 2, "a": 1}));
        let object = node_to_content_object(&node)?;
        let object_preimage = canonical_content_object_preimage(&object)?;
        assert_eq!(
            object_preimage,
            decode_hex(concat!(
                "010201070000006e6f64652f6e31",
                "7b226964223a226e31222c226c6162656c73223a5b22436c61696d225d2c",
                "2270726f70657274696573223a7b2261223a312c227a223a327d2c",
                "22746f6d6273746f6e65223a66616c73657d"
            ))
        );
        assert_eq!(
            object.hash.to_prefixed(),
            "sha256:4e05cd5b12b53f9c98bfcf7a0ba1650a2ca0c580072c437e3764bd455fe7a520"
        );

        let entry = GraphTreeEntry {
            key: object.key.clone(),
            kind: object.kind.clone(),
            object_hash: object.hash,
            tombstone: false,
        };
        let leaf_preimage = canonical_leaf_preimage(std::slice::from_ref(&entry))?;
        assert_eq!(
            leaf_preimage,
            decode_hex(concat!(
                "0202010000000100070000006e6f64652f6e31",
                "4e05cd5b12b53f9c98bfcf7a0ba1650a2ca0c580072c437e3764bd455fe7a520"
            ))
        );
        let leaf_hash = Hash::of(&leaf_preimage);
        assert_eq!(
            leaf_hash.to_prefixed(),
            "sha256:604717a2f95488098df65bf2c8241e95b12ae512f80e7ccdc72e71676aaa2810"
        );

        let child = GraphTreeChild {
            hash: leaf_hash,
            first_key: "node/n1".to_string(),
            last_key: "node/n1".to_string(),
            entries_total: 1,
        };
        let parent_preimage = canonical_parent_preimage(std::slice::from_ref(&child))?;
        assert_eq!(
            parent_preimage,
            decode_hex(concat!(
                "030201000000",
                "604717a2f95488098df65bf2c8241e95b12ae512f80e7ccdc72e71676aaa2810",
                "070000006e6f64652f6e31070000006e6f64652f6e31",
                "0100000000000000"
            ))
        );
        let parent_hash = Hash::of(&parent_preimage);
        assert_eq!(
            parent_hash.to_prefixed(),
            "sha256:49cbe70ae7600f2aa6c1e19d2c0ad14496d3be79bb432d85a0f821d11bd4f850"
        );

        let commit = GraphCommit {
            commit_hash: Hash::of(b"not-in-preimage"),
            tree_hash: parent_hash,
            branch: "main".to_string(),
            parent_commits: vec![Hash::of(b"parent-commit")],
            author: "Ada".to_string(),
            message: "commit".to_string(),
            timestamp_unix_ms: 42,
            graph_version: 7,
            historical_registry_hash: None,
            objects_total: 999,
            compiler_version: "ignored-compiler".to_string(),
        };
        let commit_preimage = canonical_commit_preimage(&commit)?;
        assert_eq!(
            commit_preimage,
            decode_hex(concat!(
                "0402",
                "49cbe70ae7600f2aa6c1e19d2c0ad14496d3be79bb432d85a0f821d11bd4f850",
                "01000000",
                "29446a6e260322cde4a19b6aa781408de855d11752c6adbbc4b5d3377fcbc95e",
                "0300000041646106000000636f6d6d6974",
                "2a0000000000000000000000000000000700000000000000"
            ))
        );
        assert_eq!(
            recompute_commit_hash(&commit)?.to_prefixed(),
            "sha256:374d44653dd778b8a8cf04287e31aa175736187c6c80ad4d41cf10c99ed87f7c"
        );

        let empty_preimage = canonical_leaf_preimage(&[])?;
        assert_eq!(empty_preimage, decode_hex("020200000000"));
        assert_eq!(
            empty_leaf_node()?.hash.to_prefixed(),
            "sha256:0f98732aa9c96a0b12b8f976cf7db5d9c421b426bc464c13b75795401e7ff7de"
        );
        Ok(())
    }

    #[test]
    fn content_address_is_payload_independent_and_tombstone_distinct() -> GraphStoreResult<()> {
        let live = NodeRecord::new("n1", ["Claim"], json!({"claim": "A"}));
        let payloadful = node_content_object(&live, true)?;
        let payloadless = node_content_object(&live, false)?;
        assert_eq!(payloadful.hash, payloadless.hash);
        assert_eq!(recompute_content_object_hash(&payloadful)?, payloadful.hash);
        assert_eq!(
            recompute_content_object_hash(&payloadless)
                .unwrap_err()
                .code,
            "graph_object_payload_required"
        );

        let mut tombstone = live;
        tombstone.tombstone = true;
        let tombstoned = node_to_content_object(&tombstone)?;
        assert_ne!(payloadful.hash, tombstoned.hash);
        let store = MemChunkStore::new();
        let live_tree = build_prolly_tree(&store, std::slice::from_ref(&payloadful))?;
        let tombstone_tree = build_prolly_tree(&store, std::slice::from_ref(&tombstoned))?;
        assert_ne!(live_tree.root, tombstone_tree.root);
        Ok(())
    }

    #[test]
    fn legacy_v1_discriminator_defaults_and_migration_preserves_checkout_graph(
    ) -> GraphStoreResult<()> {
        let node = NodeRecord::new(
            "legacy:n1",
            ["Claim"],
            json!({
                "claim": "preserved",
                "confidence": f64::from_bits(0.4_f64.to_bits() - 1),
            }),
        );
        let (old_commit, legacy_pack) = authentic_legacy_v1_pack(&node, Vec::new())?;
        assert_eq!(
            old_commit.to_prefixed(),
            "sha256:2e8e8fc9604ec441706df97823b49512148d8990b2fe9cd36d54996cbb18eac3"
        );
        let golden: Value = serde_json::from_slice(&legacy_pack)
            .map_err(|error| GraphStoreError::new("test_fixture_decode", error.to_string()))?;
        assert_eq!(
            golden["archive_objects"][0]["hash"],
            json!("sha256:d2504429e2324d825da69ca77556f085d18efa29006ef5f57535145174e9dff9")
        );
        assert_eq!(
            legacy_pack.as_slice(),
            include_bytes!("../tests/fixtures/legacy_graph_v1_nonempty.json")
        );

        assert_eq!(
            graph_pack_chunk_format(&legacy_pack)?,
            LEGACY_GRAPH_CHUNK_FORMAT_VERSION
        );
        let migrated = migrate_legacy_graph_pack_v1(&legacy_pack, old_commit, &BTreeMap::new())?;
        assert_eq!(migrated.old_commit, old_commit);
        assert_eq!(
            migrated.new_pack.chunk_format_version,
            GRAPH_CHUNK_FORMAT_VERSION
        );
        assert_eq!(
            graph_pack_chunk_format(&serde_json::to_vec(&migrated.new_pack).map_err(|error| {
                GraphStoreError::new("test_pack_serialize", error.to_string())
            })?)?,
            GRAPH_CHUNK_FORMAT_VERSION
        );
        assert!(!migrated.new_pack.archive_objects.is_empty());
        crate::zerocopy::access_graph_archive(&migrated.new_pack.archive_objects[0].bytes)
            .map_err(|error| GraphStoreError::new(error.code, error.message))?;

        let repository = update_graph_ref(
            GraphVersionRepository::default(),
            migrated.new_pack,
            Some("main".to_string()),
            Some(124),
        )
        .expect("migrated pack installs")
        .repository;
        let checkout = checkout_graph_version(&repository, "main")?;
        assert_eq!(checkout.snapshot.version, 9);
        assert_eq!(checkout.snapshot.nodes.len(), 1);
        assert_eq!(checkout.snapshot.nodes[0].id, "legacy:n1");
        assert_eq!(
            checkout.snapshot.nodes[0].properties,
            json!({
                "claim": "preserved",
                "confidence": f64::from_bits(0.4_f64.to_bits() - 1),
            })
        );
        assert!(checkout.snapshot.edges.is_empty());
        Ok(())
    }

    fn authentic_legacy_v1_pack(
        node: &NodeRecord,
        parent_commits: Vec<String>,
    ) -> GraphStoreResult<(Hash, Vec<u8>)> {
        let key = format!("node/{}", node.id);
        let logical_hash = node.content_hash.clone().unwrap_or(node.checksum()?);
        let mut object = LegacyGraphContentObjectV1 {
            key: key.clone(),
            kind: GraphObjectKind::Node,
            hash: String::new(),
            logical_hash: Some(logical_hash.clone()),
            parent_hashes: node.parent_hashes.clone(),
            payload: Some(serde_json::to_value(node).map_err(|error| {
                GraphStoreError::new("test_fixture_serialize", error.to_string())
            })?),
        };
        object.payload.as_mut().unwrap().sort_all_objects();
        let archive_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&LegacyGraphArchiveEnvelopeV1 {
            format_version: 1,
            body: LegacyGraphArchiveBodyV1::ContentObject(LegacyGraphArchiveContentObjectV1 {
                key: key.clone(),
                kind: GraphObjectKind::Node.as_str().to_string(),
                logical_hash,
                parent_hashes: node.parent_hashes.clone(),
                payload_json: Some(
                    canonical_json(object.payload.as_ref().expect("fixture payload")).map_err(
                        |message| GraphStoreError::new("test_fixture_canonicalize", message),
                    )?,
                ),
            }),
        })
        .map_err(|error| GraphStoreError::new("test_fixture_archive", error.to_string()))?
        .to_vec();
        let object_hash = Hash::of(&archive_bytes);
        object.hash = object_hash.to_prefixed();

        let mut tree_node = LegacyGraphTreeNodeV1 {
            hash: String::new(),
            level: 0,
            first_key: key.clone(),
            last_key: key.clone(),
            entries: vec![LegacyGraphTreeEntryV1 {
                key: key.clone(),
                kind: GraphObjectKind::Node,
                object_hash: object.hash.clone(),
            }],
            children: Vec::new(),
        };
        tree_node.hash = legacy_v1_tree_node_hash(&tree_node)?.to_prefixed();
        let tree_hash = tree_node.hash.clone();
        let mut commit = LegacyGraphCommitV1 {
            commit_hash: String::new(),
            tree_hash: tree_hash.clone(),
            branch: "main".to_string(),
            parent_commits,
            author: "legacy-writer".to_string(),
            message: "migrate me".to_string(),
            timestamp_unix_ms: 123,
            graph_version: 9,
            objects_total: 1,
            compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
        };
        let commit_hash = legacy_v1_commit_hash(&commit)?;
        commit.commit_hash = commit_hash.to_prefixed();
        let manifest = LegacyGraphPackManifestV1 {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
            name: "legacy-fixture".to_string(),
            branch: commit.branch.clone(),
            commit_hash: commit.commit_hash.clone(),
            tree_hash: tree_hash.clone(),
            graph_version: commit.graph_version,
            nodes_total: 1,
            edges_total: 0,
            objects_total: 1,
        };
        let mut manifest_body = serde_json::to_value(&manifest)
            .map_err(|error| GraphStoreError::new("test_fixture_serialize", error.to_string()))?;
        let mut validator_body = json!({
            "validator": "rustyred.verify_tree_root",
            "tree_hash": tree_hash.clone(),
            "objects_total": 1,
            "protocol_version": VERSIONED_GRAPH_PROTOCOL_VERSION,
        });
        manifest_body.sort_all_objects();
        validator_body.sort_all_objects();
        let manifest_capability_hash = legacy_v1_json_hash(&manifest_body)?.to_prefixed();
        let validator_capability_hash = legacy_v1_json_hash(&validator_body)?.to_prefixed();
        assert_eq!(
            manifest_capability_hash,
            "sha256:d2f180896a86d7db5fac89391a9eefbb2af182a6298bc9cbf48a9d0a19008418"
        );
        assert_eq!(
            validator_capability_hash,
            "sha256:b11c345c0b00d08a06bd3e8ec821161d4562ac23283fbc71177cfc2fd00a07ea"
        );
        let capabilities = vec![
            GraphCompilerCapability {
                name: "legacy-fixture.manifest".to_string(),
                kind: "graph_manifest".to_string(),
                mime_type: "application/json".to_string(),
                content_hash: manifest_capability_hash,
                body: manifest_body,
            },
            GraphCompilerCapability {
                name: "legacy-fixture.verify_tree_root".to_string(),
                kind: "validator".to_string(),
                mime_type: "application/json".to_string(),
                content_hash: validator_capability_hash,
                body: validator_body,
            },
        ];
        let pack = LegacyGraphPackV1 {
            chunk_format_version: LEGACY_GRAPH_CHUNK_FORMAT_VERSION,
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            compiler_version: GRAPH_PACK_COMPILER_VERSION.to_string(),
            manifest,
            commit,
            tree: LegacyGraphProllyTreeV1 {
                root_hash: tree_hash,
                root_level: 0,
                entries_total: 1,
                nodes: vec![tree_node],
            },
            capabilities,
            objects: vec![object.clone()],
            archive_objects: vec![LegacyGraphArchiveObjectBytesV1 {
                key,
                kind: GraphObjectKind::Node,
                hash: object.hash,
                format_version: 1,
                mime_type: GRAPH_ARCHIVE_MIME_TYPE.to_string(),
                bytes: archive_bytes,
            }],
        };
        #[derive(Serialize)]
        struct LegacyGraphPackWireV1<'a> {
            protocol_version: &'a str,
            compiler_version: &'a str,
            manifest: &'a LegacyGraphPackManifestV1,
            commit: &'a LegacyGraphCommitV1,
            tree: &'a LegacyGraphProllyTreeV1,
            capabilities: &'a [GraphCompilerCapability],
            objects: &'a [LegacyGraphContentObjectV1],
            archive_objects: &'a [LegacyGraphArchiveObjectBytesV1],
        }
        let bytes = serde_json::to_vec(&LegacyGraphPackWireV1 {
            protocol_version: &pack.protocol_version,
            compiler_version: &pack.compiler_version,
            manifest: &pack.manifest,
            commit: &pack.commit,
            tree: &pack.tree,
            capabilities: &pack.capabilities,
            objects: &pack.objects,
            archive_objects: &pack.archive_objects,
        })
        .map_err(|error| GraphStoreError::new("test_fixture_serialize", error.to_string()))?;
        Ok((commit_hash, bytes))
    }

    #[test]
    fn commit_address_excludes_branch_object_count_and_compiler_metadata() -> GraphStoreResult<()> {
        let store = MemChunkStore::new();
        let tree = build_prolly_tree(&store, &[])?;
        let original = build_commit(&tree, "main", Vec::new(), "Ada", "message", 7, 3, None)?;
        let mut metadata_only = original.clone();
        metadata_only.branch = "release".to_string();
        metadata_only.objects_total = usize::MAX;
        metadata_only.compiler_version = "future-compiler".to_string();
        assert_eq!(
            recompute_commit_hash(&original)?,
            recompute_commit_hash(&metadata_only)?
        );
        Ok(())
    }

    #[test]
    fn boundary_chunker_accepts_items_without_serialize() -> GraphStoreResult<()> {
        #[derive(Clone)]
        struct BoundaryOnly {
            address: Hash,
        }
        let items = (0_u64..61)
            .map(|index| BoundaryOnly {
                address: Hash::of(&index.to_le_bytes()),
            })
            .collect();
        let chunks = chunk_by_boundary(
            items,
            |item| item.address.trailing_u16(),
            |_| empty_leaf_node(),
        )?;
        assert!(!chunks.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "one-million-address deterministic boundary distribution"]
    fn million_boundary_distribution_targets_mean_31_and_forced_split_below_one_percent() {
        let mut chunk_len = 0_usize;
        let mut chunks = 0_usize;
        let mut forced = 0_usize;
        for index in 0_u64..1_000_000 {
            let boundary = Hash::of(&index.to_le_bytes()).trailing_u16();
            chunk_len += 1;
            if should_split_chunk(chunk_len, boundary) {
                if chunk_len == PROLLY_MAX_ENTRIES && boundary >= PROLLY_SPLIT_THRESHOLD {
                    forced += 1;
                }
                chunks += 1;
                chunk_len = 0;
            }
        }
        if chunk_len > 0 {
            chunks += 1;
        }
        let mean = 1_000_000_f64 / chunks as f64;
        let forced_fraction = forced as f64 / chunks as f64;
        assert!((30.0..=32.0).contains(&mean), "mean={mean}");
        assert!(forced_fraction < 0.01, "forced={forced_fraction}");
    }

    fn hlc(actor: &str, physical_ms: i64) -> Hlc {
        Hlc::new(physical_ms, 0, ActorId::from_label(actor))
    }

    fn memory_node(status: &str, tags: &[&str], stamp: Hlc) -> NodeRecord {
        NodeRecord::new(
            "mem:a",
            ["MemoryDocument"],
            json!({
                "status": status,
                "confidence": 0.5,
                "tags": tags,
                "_crdt_hlc": {
                    "record": stamp,
                    "properties": {
                        "status": stamp,
                        "confidence": stamp,
                        "tags": stamp
                    }
                }
            }),
        )
    }

    fn legacy_diff_graph_snapshots(
        base: &GraphSnapshot,
        target: &GraphSnapshot,
    ) -> GraphStoreResult<GraphVersionDiff> {
        let base_objects = snapshot_content_objects(base, false)?;
        let target_objects = snapshot_content_objects(target, false)?;
        let store = MemChunkStore::new();
        let base_tree = build_prolly_tree(&store, &base_objects)?;
        let target_tree = build_prolly_tree(&store, &target_objects)?;
        let base_map = base_objects
            .into_iter()
            .map(|object| (object.key.clone(), object))
            .collect::<BTreeMap<_, _>>();
        let target_map = target_objects
            .into_iter()
            .map(|object| (object.key.clone(), object))
            .collect::<BTreeMap<_, _>>();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged = 0usize;

        for (key, target_object) in &target_map {
            match base_map.get(key) {
                None => added.push(GraphDiffEntry {
                    key: key.clone(),
                    kind: target_object.kind.clone(),
                    old_hash: None,
                    new_hash: Some(target_object.hash.to_prefixed()),
                }),
                Some(base_object) if base_object.hash != target_object.hash => {
                    modified.push(GraphDiffEntry {
                        key: key.clone(),
                        kind: target_object.kind.clone(),
                        old_hash: Some(base_object.hash.to_prefixed()),
                        new_hash: Some(target_object.hash.to_prefixed()),
                    });
                }
                Some(_) => unchanged += 1,
            }
        }

        for (key, base_object) in &base_map {
            if !target_map.contains_key(key) {
                removed.push(GraphDiffEntry {
                    key: key.clone(),
                    kind: base_object.kind.clone(),
                    old_hash: Some(base_object.hash.to_prefixed()),
                    new_hash: None,
                });
            }
        }

        let changed = added
            .iter()
            .chain(&modified)
            .map(|entry| entry.key.clone())
            .collect::<BTreeSet<_>>();
        Ok(GraphVersionDiff {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            base_tree_hash: base_tree.root.to_prefixed(),
            target_tree_hash: target_tree.root.to_prefixed(),
            added,
            removed,
            modified,
            unchanged,
            target_nodes: target
                .nodes
                .iter()
                .filter(|node| changed.contains(&format!("node/{}", node.id)))
                .map(|node| (node.id.clone(), node.clone()))
                .collect(),
            target_edges: target
                .edges
                .iter()
                .filter(|edge| changed.contains(&format!("edge/{}", edge.id)))
                .map(|edge| (edge.id.clone(), edge.clone()))
                .collect(),
        })
    }

    fn legacy_merge_graph_snapshots_with_registry(
        base: &GraphSnapshot,
        ours: &GraphSnapshot,
        theirs: &GraphSnapshot,
        options: GraphMergeOptions,
        registry: &MergeRegistry,
    ) -> GraphStoreResult<GraphMergeResult> {
        let store = MemChunkStore::new();
        let base_tree = build_prolly_tree(&store, &snapshot_content_objects(base, false)?)?;
        let ours_tree = build_prolly_tree(&store, &snapshot_content_objects(ours, false)?)?;
        let theirs_tree = build_prolly_tree(&store, &snapshot_content_objects(theirs, false)?)?;
        let base_records = merge_records_by_key(base);
        let ours_records = merge_records_by_key(ours);
        let theirs_records = merge_records_by_key(theirs);
        let mut keys = BTreeSet::new();
        keys.extend(base_records.keys().cloned());
        keys.extend(ours_records.keys().cloned());
        keys.extend(theirs_records.keys().cloned());

        let mut merged_records = Vec::new();
        let mut resolved = Vec::new();
        let mut conflicts = Vec::new();

        for key in keys {
            let base_record = base_records.get(&key);
            let ours_record = ours_records.get(&key);
            let theirs_record = theirs_records.get(&key);
            let kind = merge_record_kind(base_record, ours_record, theirs_record);
            let base_hash = base_record.map(MergeRecord::hash).transpose()?;
            let ours_hash = ours_record.map(MergeRecord::hash).transpose()?;
            let theirs_hash = theirs_record.map(MergeRecord::hash).transpose()?;

            match resolve_merge_record(
                base_record,
                ours_record,
                theirs_record,
                &options.strategy,
                options.min_confidence_delta,
                registry,
            )? {
                MergeDecision::Resolved {
                    selected,
                    record,
                    reason,
                } => {
                    if let Some(record) = record {
                        merged_records.push(record);
                    }
                    resolved.push(GraphMergeResolution {
                        key,
                        kind,
                        selected,
                        base_hash,
                        ours_hash,
                        theirs_hash,
                        reason,
                    });
                }
                MergeDecision::Conflict { reason } => conflicts.push(GraphMergeConflict {
                    key,
                    kind,
                    base_hash,
                    ours_hash,
                    theirs_hash,
                    reason,
                }),
            }
        }

        let (merged_snapshot, merged_pack, merged_tree_hash) = if conflicts.is_empty() {
            let snapshot = snapshot_from_merge_records(
                base.version
                    .max(ours.version)
                    .max(theirs.version)
                    .saturating_add(1),
                merged_records,
            );
            let pack = compile_graph_pack(
                &snapshot,
                GraphCompileOptions {
                    name: options
                        .name
                        .or_else(|| Some("merged-graph-pack".to_string())),
                    branch: options.branch,
                    parent_commits: options.parent_commits,
                    author: options.author,
                    message: options
                        .message
                        .or_else(|| Some("merge graph snapshots".to_string())),
                    timestamp_unix_ms: options.timestamp_unix_ms,
                    include_payloads: options.include_payloads,
                    ..GraphCompileOptions::default()
                },
            )?;
            (
                Some(snapshot),
                Some(pack.clone()),
                Some(pack.tree.root.to_prefixed()),
            )
        } else {
            (None, None, None)
        };

        Ok(GraphMergeResult {
            protocol_version: VERSIONED_GRAPH_PROTOCOL_VERSION.to_string(),
            status: if conflicts.is_empty() {
                "clean".to_string()
            } else {
                "conflicted".to_string()
            },
            base_tree_hash: base_tree.root.to_prefixed(),
            ours_tree_hash: ours_tree.root.to_prefixed(),
            theirs_tree_hash: theirs_tree.root.to_prefixed(),
            merged_tree_hash,
            merged_snapshot,
            merged_pack,
            resolved,
            conflicts,
        })
    }

    fn assert_serialized_parity<T: Serialize>(actual: &T, legacy: &T) {
        assert_eq!(
            serde_json::to_vec(actual).expect("serialize tree-walk result"),
            serde_json::to_vec(legacy).expect("serialize legacy result")
        );
    }

    fn compacted_repository_with_synthetic_tree(
        root: Hash,
        entries_total: usize,
        tree_chunks: BTreeMap<Hash, Vec<u8>>,
    ) -> GraphStoreResult<GraphVersionRepository> {
        let snapshot = GraphSnapshot {
            historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
        };
        let pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )?;
        let mut repository = update_graph_ref(
            GraphVersionRepository::default(),
            pack,
            Some("main".to_string()),
            Some(2),
        )
        .expect("fixture pack installs")
        .repository;
        repository.tree_chunks = tree_chunks;
        let stored_pack = repository.packs.first_mut().ok_or_else(|| {
            GraphStoreError::new(
                "test_compacted_repository_missing_pack",
                "synthetic compacted repository has no stored pack",
            )
        })?;
        debug_assert!(stored_pack.tree_chunks.is_empty());
        debug_assert!(stored_pack.objects.is_empty());
        stored_pack.tree.root = root;
        stored_pack.tree.entries_total = entries_total;
        Ok(repository)
    }

    #[test]
    fn compacted_repository_corrupt_tree_chunk_is_rejected() -> GraphStoreResult<()> {
        let root = Hash::of(b"synthetic:claimed-address");
        let repository = compacted_repository_with_synthetic_tree(
            root,
            0,
            BTreeMap::from([(root, canonical_leaf_preimage(&[])?)]),
        )?;
        let pack = repository.packs[0].clone();
        let store = RepositoryChunkStore::new(&repository.tree_chunks);

        let collect_error = collect_tree_entries(&store, &pack.tree).unwrap_err();
        assert_eq!(collect_error.code, "graph_tree_chunk_address_mismatch");
        let snapshot_error = graph_snapshot_from_repository(&repository, &pack).unwrap_err();
        assert_eq!(snapshot_error.code, "graph_tree_chunk_address_mismatch");
        let checkout_error = checkout_graph_version(&repository, "main").unwrap_err();
        assert_eq!(checkout_error.code, "graph_tree_chunk_address_mismatch");
        Ok(())
    }

    #[test]
    fn compacted_repository_depth_is_bounded() -> GraphStoreResult<()> {
        let entry = GraphTreeEntry {
            key: "node/node:a".to_string(),
            kind: GraphObjectKind::Node,
            object_hash: Hash::of(b"synthetic:object"),
            tombstone: false,
        };
        let leaf_bytes = canonical_leaf_preimage(std::slice::from_ref(&entry))?;
        let mut root = Hash::of(&leaf_bytes);
        let mut tree_chunks = BTreeMap::from([(root, leaf_bytes)]);
        for _ in 0..=MAX_REPOSITORY_TREE_DEPTH {
            let parent_bytes = canonical_parent_preimage(&[GraphTreeChild {
                hash: root,
                first_key: entry.key.clone(),
                last_key: entry.key.clone(),
                entries_total: 1,
            }])?;
            root = Hash::of(&parent_bytes);
            tree_chunks.insert(root, parent_bytes);
        }
        let repository = compacted_repository_with_synthetic_tree(root, 1, tree_chunks)?;
        let pack = repository.packs[0].clone();
        let store = RepositoryChunkStore::new(&repository.tree_chunks);

        let collect_error = collect_tree_entries(&store, &pack.tree).unwrap_err();
        assert_eq!(collect_error.code, "graph_repository_depth_exceeded");
        let snapshot_error = graph_snapshot_from_repository(&repository, &pack).unwrap_err();
        assert_eq!(snapshot_error.code, "graph_repository_depth_exceeded");
        let checkout_error = checkout_graph_version(&repository, "main").unwrap_err();
        assert_eq!(checkout_error.code, "graph_repository_depth_exceeded");
        Ok(())
    }

    #[test]
    fn compiler_builds_stable_tree_independent_of_record_order() {
        let ada = NodeRecord::new("node:ada", ["Person"], json!({"name": "Ada"}));
        let bob = NodeRecord::new("node:bob", ["Person"], json!({"name": "Bob"}));
        let edge = EdgeRecord::new("edge:knows", "node:ada", "KNOWS", "node:bob", json!({}));
        let first = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![ada.clone(), bob.clone()],
            edges: vec![edge.clone()],
        };
        let second = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![bob, ada],
            edges: vec![edge],
        };
        let opts = GraphCompileOptions {
            timestamp_unix_ms: Some(1),
            ..GraphCompileOptions::default()
        };
        let first_pack = compile_graph_pack(&first, opts.clone()).unwrap();
        let second_pack = compile_graph_pack(&second, opts).unwrap();

        assert_eq!(first_pack.tree.root, second_pack.tree.root);
        assert_eq!(
            first_pack.commit.commit_hash,
            second_pack.commit.commit_hash
        );
        assert_eq!(first_pack.manifest.objects_total, 3);
        assert_eq!(first_pack.capabilities.len(), 2);
    }

    #[test]
    fn diff_reports_added_removed_and_modified_records() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new(
                "node:ada",
                ["Person"],
                json!({"name": "Ada"}),
            )],
            edges: Vec::new(),
        };
        let target = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![
                NodeRecord::new("node:ada", ["Person"], json!({"name": "Ada Lovelace"})),
                NodeRecord::new("node:bob", ["Person"], json!({"name": "Bob"})),
            ],
            edges: Vec::new(),
        };
        let diff = diff_graph_snapshots(&base, &target).unwrap();

        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.removed.len(), 0);
        assert_eq!(diff.unchanged, 0);
    }

    #[test]
    fn snapshot_diff_tree_walk_is_byte_identical_to_legacy_map_diff() {
        let nodes = (0..48)
            .map(|index| {
                NodeRecord::new(
                    format!("node:{index:03}"),
                    ["Fixture"],
                    json!({"value": index}),
                )
            })
            .collect::<Vec<_>>();
        let base = GraphSnapshot {
            historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
            historical_node_history_complete: true,
            version: 7,
            nodes,
            edges: vec![EdgeRecord::new(
                "edge:stable",
                "node:000",
                "LINKS",
                "node:001",
                json!({"weight": 1}),
            )],
        };
        let mut target = base.clone();
        target.version = 8;
        target.nodes[23].properties = json!({"value": 23, "changed": true});
        target.nodes.remove(37);
        target.nodes.push(NodeRecord::new(
            "node:added",
            ["Fixture"],
            json!({"value": "new"}),
        ));

        let actual = diff_graph_snapshots(&base, &target).unwrap();
        let legacy = legacy_diff_graph_snapshots(&base, &target).unwrap();

        assert_serialized_parity(&actual, &legacy);
        assert_eq!(actual.added.len(), 1);
        assert_eq!(actual.removed.len(), 1);
        assert_eq!(actual.modified.len(), 1);
        assert_eq!(actual.unchanged, 47);
    }

    #[test]
    fn graph_prolly_tree_descriptor_is_root_and_count_only() -> GraphStoreResult<()> {
        let store = MemChunkStore::new();
        let tree = build_prolly_tree_from_entries(&store, synthetic_entries(3))?;
        let encoded = serde_json::to_value(&tree).expect("serialize root-only tree descriptor");
        let object = encoded.as_object().expect("tree descriptor object");

        assert_eq!(object.len(), 2);
        assert_eq!(object.get("root_hash"), Some(&json!(tree.root)));
        assert_eq!(object.get("entries_total"), Some(&json!(3)));
        assert!(!object.contains_key("nodes"));
        assert!(!object.contains_key("root_level"));
        Ok(())
    }

    #[test]
    fn prolly_reader_lru_never_exceeds_configured_capacity() -> GraphStoreResult<()> {
        let store = MemChunkStore::new();
        let build = build_prolly_tree_from_entries_with_transport(&store, synthetic_entries(256))?;
        assert!(build.tree_chunks.len() >= 3, "fixture needs three chunks");
        let addresses = build
            .tree_chunks
            .iter()
            .take(3)
            .map(|chunk| chunk.address)
            .collect::<Vec<_>>();
        let mut reader = ProllyReader::with_capacity(
            &store,
            NonZeroUsize::new(2).expect("non-zero test cache capacity"),
        );

        for address in &addresses {
            reader.read(*address)?;
            assert!(reader.cached_nodes() <= 2);
        }
        let fetches_before = reader.metrics().store_fetches;
        reader.read(addresses[0])?;
        assert_eq!(reader.cached_nodes(), 2);
        assert_eq!(reader.metrics().store_fetches, fetches_before + 1);
        Ok(())
    }

    #[test]
    fn store_backed_incremental_working_set_is_height_bounded() -> GraphStoreResult<()> {
        let store = CountingChunkStore::new();
        let entries = synthetic_entries(10_000);
        let prior = build_prolly_tree_from_entries(&store, entries)?;
        let mut height_reader = ProllyReader::new(&store as &dyn ChunkStore);
        let height = prolly_tree_height(&mut height_reader, prior.root)?;
        store.reset_gets();

        let key = "node/00005000".to_string();
        let upserts = BTreeMap::from([(
            key.clone(),
            GraphTreeEntry {
                key,
                kind: GraphObjectKind::Node,
                object_hash: Hash::of(b"object:changed"),
                tombstone: false,
            },
        )]);
        let build = build_prolly_tree_incremental(&store, &prior, &upserts)?;
        let bound = 16 * (height + 1);

        assert!(store.gets() <= bound, "gets={} bound={bound}", store.gets());
        assert!(
            build.tree_chunks.len() <= bound,
            "delta={} bound={bound}",
            build.tree_chunks.len()
        );
        assert_eq!(build.cost.chunks_written, build.tree_chunks.len());
        assert_eq!(
            build.cost.bytes_written,
            build
                .tree_chunks
                .iter()
                .map(|chunk| chunk.bytes.len())
                .sum::<usize>()
        );
        Ok(())
    }

    #[test]
    fn incremental_batch_allocates_only_changed_windows_and_shared_spines() -> GraphStoreResult<()>
    {
        let store = CountingChunkStore::new();
        let entries = synthetic_entries(10_000);
        let prior = build_prolly_tree_from_entries(&store, entries.clone())?;
        let mut height_reader = ProllyReader::new(&store as &dyn ChunkStore);
        let height = prolly_tree_height(&mut height_reader, prior.root)?;
        let replacement_count = 64usize;
        let upserts = (0..replacement_count)
            .map(|offset| {
                let index = (offset + 1) * entries.len() / (replacement_count + 1);
                let key = format!("node/{index:08}");
                (
                    key.clone(),
                    GraphTreeEntry {
                        key,
                        kind: GraphObjectKind::Node,
                        object_hash: Hash::of(format!("batch-replacement:{offset}").as_bytes()),
                        tombstone: false,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        store.reset_gets();

        let build = build_prolly_tree_incremental(&store, &prior, &upserts)?;
        let full_store = MemChunkStore::new();
        let full = build_prolly_tree_from_entries(&full_store, merge_entries(&entries, &upserts))?;

        assert_eq!(build.tree, full);
        assert_eq!(build.cost.chunks_written, build.tree_chunks.len());
        assert!(
            build.staged_chunks_total <= build.tree_chunks.len() + 1,
            "staged={} reachable_delta={}; a per-key zipper leaves O(k*h) transient chunks",
            build.staged_chunks_total,
            build.tree_chunks.len()
        );
        let changed_spine_bound = 8 * replacement_count + 16 * (height + 1);
        assert!(
            build.tree_chunks.len() <= changed_spine_bound,
            "delta={} bound={changed_spine_bound}",
            build.tree_chunks.len()
        );
        let fetch_bound = 8 * replacement_count + 16 * (height + 1);
        assert!(
            store.gets() <= fetch_bound,
            "gets={} bound={fetch_bound}",
            store.gets()
        );
        Ok(())
    }

    #[test]
    fn a_bounded_tree_build_probes_base_chunks_instead_of_verifying_them() -> GraphStoreResult<()>
    {
        // The publish path asks "may I stop walking this subtree?" once per base
        // child of every staged parent. Answering that with `has` bought a seek,
        // a crc32, a zstd decode and a SHA-256 per hint, which made a bounded
        // write pay a verified read for every prune instead of for every chunk it
        // actually wrote. The probe answers the same question from the admission
        // index without reading bytes.
        //
        // Asserted by counted work, not wall clock, and the bounds are the ones
        // that are actually true and actually matter:
        //
        //   * the probe is live (`probes >= 1`), so the bounds below are not vacuous;
        //   * verified existence checks stay bounded by the commit's own size and
        //     not by the graph -- and on revert they move with the prune count
        //     instead, 28 at 1k and 59 at 10k, which is what this catches;
        //   * the delta and the verified count stay flat across a ten-fold larger
        //     store, while the probe count grows 25 -> 56. Only the per-hint cost
        //     changed; the number of hints is still a (sublinear) function of the
        //     tree, which is the honest shape of the win and the reason the append
        //     cost curve is flattened but not flattened *to a constant*.
        //
        // The residual verified checks are `persist_tree_nodes`' admission probe
        // ("which of these do I still owe the store?"), which must stay verified:
        // a false positive there would skip a write. It is bounded by the nodes
        // offered for persistence, and on the journal store a novel chunk's miss
        // is one index lookup.
        let mut observed = Vec::new();
        for tier in [1_000_usize, 10_000] {
            let store = CountingChunkStore::new();
            let entries = synthetic_entries(tier);
            let prior = build_prolly_tree_from_entries(&store, entries.clone())?;

            // One replacement in the middle: a bounded write, always a base-chunk
            // prune at every level above the changed leaf.
            let index = tier / 2;
            let key = format!("node/{index:08}");
            let upserts = BTreeMap::from([(
                key.clone(),
                GraphTreeEntry {
                    key,
                    kind: GraphObjectKind::Node,
                    object_hash: Hash::of(format!("probe-tier:{tier}").as_bytes()),
                    tombstone: false,
                },
            )]);
            store.reset_existence_checks();
            store.reset_gets();

            let build = build_prolly_tree_incremental(&store, &prior, &upserts)?;
            observed.push((
                tier,
                store.existence_probes(),
                store.verified_existence_checks(),
                build.tree_chunks.len(),
                store.gets(),
            ));
        }

        for &(tier, probes, has_checks, delta_chunks, gets) in &observed {
            println!(
                "existence-check oracle | tier {tier} | probes {probes} | verified {has_checks} | \
                 delta chunks {delta_chunks} | reads {gets}"
            );
            assert!(
                probes >= 1,
                "tier {tier}: the walk must actually reach a base-chunk prune, otherwise the \
                 bound below is vacuous"
            );
            assert!(
                has_checks <= delta_chunks + 4,
                "tier {tier}: verified existence checks ({has_checks}) must be bounded by the \
                 nodes this commit offers for persistence, which this fixture keeps at or just \
                 above the published delta ({delta_chunks}); the slack covers a dedup hit, where \
                 a node is offered that is already present and so is not published. Graph size \
                 must not enter this bound; probes {probes}"
            );
        }

        let (_, first_probes, first_has, first_delta, _) = observed[0];
        let (_, second_probes, second_has, second_delta, second_reads) = observed[1];

        // The commit's own size must stay flat under a ten-fold larger store. The
        // per-tier `has_checks <= delta_chunks` bound above then carries that
        // flatness to the verified checks, which is the property that matters:
        // neither the work a bounded write does nor the work it verifies is a
        // function of stored history. The slack is one tree level's worth of
        // spine, since 10k records is one level taller than 1k.
        assert!(
            second_delta <= first_delta + 2,
            "the delta a bounded write publishes must stay flat under a ten-fold larger store: \
             {first_delta} chunks at 1k against {second_delta} at 10k"
        );
        assert!(
            second_has <= first_has + 2,
            "verified existence checks must stay flat under a ten-fold larger store: {first_has} \
             at 1k against {second_has} at 10k"
        );
        // The residual, quantified as far as the counters honestly reach.
        // `reads` is `get` below the Prolly reader cache, so it counts physical
        // base-store fetches that missed that cache -- not "chunks this build
        // read", since cache hits are invisible here and the same chunk can be
        // fetched once per reader. `probes` is the prunes the walk took. The two
        // are not disjoint and nothing attributes a fetch to a hint, so this
        // assertion claims only what it can: the probe is not redundant with the
        // reads, i.e. most hints cost no read. The class that makes that matter --
        // a hint that was the *only* read of its chunk -- is asserted directly by
        // the sibling poison test, not inferred from these two numbers.
        assert!(
            second_probes > second_reads,
            "expected most prunes to avoid a read: probes {second_probes} against physical \
             fetches {second_reads}"
        );
        assert!(
            second_probes > first_probes,
            "a larger store prunes more base subtrees, so the probe count must grow with it: \
             {first_probes} at 1k against {second_probes} at 10k"
        );

        // Liveness, so the counter above cannot be dead: a caller that genuinely
        // needs a verified existence check still pays for one.
        let store = CountingChunkStore::new();
        let entries = synthetic_entries(64);
        let prior = build_prolly_tree_from_entries(&store, entries)?;
        store.reset_existence_checks();
        assert!(store.has(prior.root)?);
        assert_eq!(
            store.verified_existence_checks(),
            1,
            "`has` must still be the verifying existence check"
        );
        Ok(())
    }

    #[test]
    fn a_bounded_tree_build_still_refuses_an_unreadable_base_chunk() -> GraphStoreResult<()> {
        // Class one of the probe's exposure, asserted so it cannot silently
        // weaken: a pruned base subtree that this build *does* read -- here the
        // root child immediately after the changed subtree, which the window
        // walk's next-unit lookahead fetches -- is still refused when its bytes
        // are unreadable, because that fetch verifies.
        //
        // The read is the lookahead at `versioned_graph.rs`'s level pass
        // (`reader.read(next_cursor.address)?.first_key()`), NOT the parent
        // rewrite: rewriting a parent reads the parent's own bytes and takes its
        // children's descriptors from them, so it never fetches a child chunk.
        //
        // Both arms below therefore reach the same refusal by the same route --
        // the lookahead fetch -- and `probe_lies` does not change the outcome,
        // because the hint is never consulted for this address. The arm that
        // distinguishes the two probe behaviours is the class-two sibling,
        // `a_pruned_subtree_this_build_never_reads_is_republished_and_the_boundary_refuses_it`.
        let store = PoisonedBaseStore::new();
        let entries = synthetic_entries(10_000);
        let prior = build_prolly_tree_from_entries(&store, entries)?;

        // The fixture must have a root with at least two children, so that a
        // mutation inside one subtree leaves another subtree reused (and thus
        // pruned by the delta walk rather than rewritten).
        let root_bytes = store
            .get(prior.root)?
            .expect("the prior root must be readable before poisoning");
        let children = match decode_canonical_tree_node(prior.root, &root_bytes)? {
            StoredProllyNode::Parent(children) => children,
            StoredProllyNode::Leaf(_) => {
                panic!("fixture must build a multi-level tree, not a single leaf")
            }
        };
        assert!(
            children.len() >= 2,
            "fixture needs at least two root children, got {}",
            children.len()
        );

        // Poison a subtree the mutation does not touch: the walk will prune it.
        let poisoned = children[1].hash;
        store.poison(poisoned);

        // A replacement, not an insert: the entry count per subtree is
        // unchanged, so no sibling subtree is pulled into a rebounding merge.
        let key = children[0].first_key.clone();
        let upserts = BTreeMap::from([(
            key.clone(),
            GraphTreeEntry {
                key,
                kind: GraphObjectKind::Node,
                object_hash: Hash::of(b"poisoned-probe-replacement"),
                tombstone: false,
            },
        )]);

        // Direction A -- the probe delegates to the verifying existence check.
        // The error code is asserted rather than just "it errored", so this cannot
        // pass because the fixture broke somewhere else.
        store.set_probe_lies(false);
        let refused = match build_prolly_tree_incremental(&store, &prior, &upserts) {
            Ok(_) => panic!(
                "with a verifying existence check the build must refuse an unreadable base chunk"
            ),
            Err(error) => error,
        };
        assert_eq!(
            refused.code, "test_poisoned_base_chunk",
            "the refusal must be the poisoned base chunk's, proving the build reached it: {refused}"
        );

        // Direction B -- the probe claims presence without reading. The build must
        // still refuse, by the lookahead fetch described above. If this arm ever
        // starts succeeding, the adjacent-child read has moved and class one has
        // collapsed into class two, so `ChunkStore::contains_address`'s exposure
        // note needs re-deriving rather than inheriting.
        store.set_probe_lies(true);
        let refused_with_probe = match build_prolly_tree_incremental(&store, &prior, &upserts) {
            Ok(_) => panic!(
                "the probe let a bounded build publish a root referencing an unreadable base \
                 chunk. The adjacent-child read no longer covers it, so this case has moved into \
                 the deferred-detection class and the note on `ChunkStore::contains_address` is \
                 now wrong"
            ),
            Err(error) => error,
        };
        assert_eq!(
            refused_with_probe.code, "test_poisoned_base_chunk",
            "the refusal must still be the poisoned chunk's with the probe in place, not an \
             unrelated failure: {refused_with_probe}"
        );
        assert!(
            store.get(poisoned).is_err(),
            "the poisoned chunk must genuinely be unreadable, otherwise both arms prove nothing"
        );

        // The boundary refuses such a root independently of the commit path:
        // `collect_tree_entries` walks every chunk reachable from a root with a
        // verified read. The tree audited here is `prior`, not a fresh build,
        // because both arms above failed and so there is no fresh root; `prior`
        // carries the same poisoned reference and is enough for this claim. The
        // successful-build case, where the freshly built root is the thing
        // audited, is the sibling class-two test.
        let audit = match audit_published_prolly_parity(&store, &prior) {
            Ok(_) => panic!(
                "the boundary audit must refuse a root that references an unreadable chunk"
            ),
            Err(error) => error,
        };
        assert_eq!(
            audit.code, "test_poisoned_base_chunk",
            "the boundary refusal must be the unreadable chunk's, not an unrelated failure: {audit}"
        );
        Ok(())
    }

    #[test]
    fn a_pruned_subtree_this_build_never_reads_is_republished_and_the_boundary_refuses_it(
    ) -> GraphStoreResult<()> {
        // Class two, and the real exposure of the probe: a pruned base subtree
        // beyond the window walk's next-unit lookahead is never fetched by this
        // build at all, so the probe removes the only commit-time read there was
        // and the chunk is republished by reference. This test is what makes that
        // a measured claim rather than an argument from two counters -- the
        // counters count hints and physical fetches, and neither attributes a
        // fetch to a hint.
        //
        // The pair with its sibling is the point: there the poisoned child is the
        // one the lookahead fetches and the build refuses; here it is one the
        // build never touches and the build succeeds. Same store, same overlay,
        // same poison -- only the position differs.
        let store = PoisonedBaseStore::new();
        let entries = synthetic_entries(10_000);
        let prior = build_prolly_tree_from_entries(&store, entries)?;

        let root_bytes = store
            .get(prior.root)?
            .expect("the prior root must be readable before poisoning");
        let children = match decode_canonical_tree_node(prior.root, &root_bytes)? {
            StoredProllyNode::Parent(children) => children,
            StoredProllyNode::Leaf(_) => {
                panic!("fixture must build a multi-level tree, not a single leaf")
            }
        };
        assert!(
            children.len() >= 3,
            "fixture needs a root child beyond the one the lookahead fetches, so that at least \
             one subtree is genuinely never read; got {}",
            children.len()
        );

        // The mutation lands in the leftmost subtree, so the walk fetches the child
        // after it. The last child is beyond both the mutation and the lookahead.
        let poisoned = children[children.len() - 1].hash;
        store.poison(poisoned);

        let key = children[0].first_key.clone();
        let upserts = BTreeMap::from([(
            key.clone(),
            GraphTreeEntry {
                key,
                kind: GraphObjectKind::Node,
                object_hash: Hash::of(b"class-two-replacement"),
                tombstone: false,
            },
        )]);

        store.set_probe_lies(true);
        let build = build_prolly_tree_incremental(&store, &prior, &upserts)?;

        // The poison is real, so the build succeeding is the exposure and not a
        // fixture that forgot to break anything.
        assert!(
            store.get(poisoned).is_err(),
            "the poisoned chunk must be genuinely unreadable, otherwise this proves nothing"
        );
        // And it is reused by reference rather than rewritten: a pruned subtree is
        // by definition not part of the staged delta, so the published root points
        // at a chunk this build could not read.
        assert!(
            !build
                .tree_chunks
                .iter()
                .any(|chunk| chunk.address == poisoned),
            "a pruned subtree must not appear in the delta; if it does, it was rewritten and this \
             test is measuring the wrong chunk"
        );

        // The boundary is what refuses it, on the freshly built root rather than on
        // the prior tree.
        let audit = match audit_published_prolly_parity(&store, &build.tree) {
            Ok(_) => panic!(
                "the boundary audit must refuse a freshly published root that references an \
                 unreadable chunk, otherwise the deferred-detection note on \
                 `ChunkStore::contains_address` is false"
            ),
            Err(error) => error,
        };
        assert_eq!(
            audit.code, "test_poisoned_base_chunk",
            "the boundary refusal must be the unreadable chunk's, not an unrelated failure: {audit}"
        );
        Ok(())
    }

    #[test]
    fn prolly_incremental_root_height_collapse_matches_full_tree() -> GraphStoreResult<()> {
        let entry = |index: u8, boundary: u16| {
            let mut raw = [0_u8; 32];
            raw[0] = index;
            raw[30..].copy_from_slice(&boundary.to_le_bytes());
            GraphTreeEntry {
                key: format!("node/{index:02}"),
                kind: GraphObjectKind::Node,
                object_hash: Hash::from_bytes(raw),
                tombstone: false,
            }
        };
        let entries = (0_u8..25)
            .map(|index| entry(index, if index == 23 { 0 } else { u16::MAX }))
            .collect::<Vec<_>>();
        let store = MemChunkStore::new();
        let prior = build_prolly_tree_from_entries(&store, entries.clone())?;
        let prior_height = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            prolly_tree_height(&mut reader, prior.root)?
        };
        assert_eq!(prior_height, 1, "fixture must start as a 24+1 tree");

        let replacement = entry(23, u16::MAX);
        let incremental = build_prolly_tree_incremental(
            &store,
            &prior,
            &BTreeMap::from([(replacement.key.clone(), replacement.clone())]),
        )?;
        let canonical_entries = entries
            .into_iter()
            .map(|candidate| {
                if candidate.key == replacement.key {
                    replacement.clone()
                } else {
                    candidate
                }
            })
            .collect::<Vec<_>>();
        let canonical_store = MemChunkStore::new();
        let canonical = build_prolly_tree_from_entries(&canonical_store, canonical_entries)?;
        let incremental_height = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            prolly_tree_height(&mut reader, incremental.tree.root)?
        };

        assert_eq!(incremental_height, 0);
        assert_eq!(incremental.tree, canonical);
        assert_eq!(incremental.tree.entries_total, 25);
        assert_eq!(incremental.tree_chunks.len(), 1);
        assert_eq!(incremental.cost.chunks_written, 1);
        assert_eq!(
            incremental.cost.bytes_written,
            incremental.tree_chunks[0].bytes.len()
        );
        Ok(())
    }

    #[test]
    fn diff_aligns_height_transition_and_compares_full_entry_semantics() -> GraphStoreResult<()> {
        let store = MemChunkStore::new();
        let mut base_tree = build_prolly_tree_from_entries(&store, synthetic_entries(1))?;
        let mut base_count = 1usize;
        let mut base_height = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            prolly_tree_height(&mut reader, base_tree.root)?
        };
        let (target_tree, target_count) = loop {
            let candidate_count = base_count + 1;
            let candidate =
                build_prolly_tree_from_entries(&store, synthetic_entries(candidate_count))?;
            let candidate_height = {
                let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
                prolly_tree_height(&mut reader, candidate.root)?
            };
            if candidate_height > base_height {
                break (candidate, candidate_count);
            }
            base_tree = candidate;
            base_count = candidate_count;
            base_height = candidate_height;
            assert!(base_count <= PROLLY_MAX_ENTRIES + 1);
        };
        let (transition_diff, metrics) = diff_graph_trees(&store, &base_tree, &target_tree)?;
        assert_eq!(target_count, base_count + 1);
        assert_eq!(transition_diff.added.len(), 1);
        assert!(metrics.store_fetches <= 16);

        let object_hash = Hash::of(b"same-object-address");
        let live = GraphTreeEntry {
            key: "node/semantic".to_string(),
            kind: GraphObjectKind::Node,
            object_hash,
            tombstone: false,
        };
        let mut tombstone = live.clone();
        tombstone.tombstone = true;
        let live_tree = build_prolly_tree_from_entries(&store, vec![live])?;
        let tombstone_tree = build_prolly_tree_from_entries(&store, vec![tombstone])?;
        let (semantic_diff, _) = diff_graph_trees(&store, &live_tree, &tombstone_tree)?;
        assert_eq!(semantic_diff.removed.len(), 1);
        assert_eq!(semantic_diff.unchanged, 0);

        let mut inconsistent = live_tree.clone();
        inconsistent.entries_total = 2;
        assert_eq!(
            diff_graph_trees(&store, &live_tree, &inconsistent)
                .unwrap_err()
                .code,
            "graph_tree_entry_count_mismatch"
        );
        let incremental_error =
            match build_prolly_tree_incremental(&store, &inconsistent, &BTreeMap::new()) {
                Ok(_) => panic!("inconsistent prior tree count must be rejected"),
                Err(error) => error,
            };
        assert_eq!(incremental_error.code, "graph_tree_entry_count_mismatch");
        Ok(())
    }

    #[test]
    fn in_place_mutation_retains_existing_mirror_vector_allocations() {
        let mut nodes = Vec::with_capacity(256);
        nodes.extend((0..128).map(|index| {
            NodeRecord::new(format!("node:{index:03}"), ["Fixture"], json!({"v": index}))
        }));
        let mut edges = Vec::with_capacity(8);
        edges.push(EdgeRecord::new(
            "edge:0",
            "node:000",
            "LINKS",
            "node:001",
            json!({}),
        ));
        let mut snapshot = GraphSnapshot {
            version: 1,
            historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
            historical_node_history_complete: true,
            nodes,
            edges,
        };
        let nodes_ptr = snapshot.nodes.as_ptr();
        let nodes_capacity = snapshot.nodes.capacity();
        let edges_ptr = snapshot.edges.as_ptr();
        let edges_capacity = snapshot.edges.capacity();
        let batch = GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "node:064",
                ["Fixture"],
                json!({"v": "changed"}),
            )),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:0",
                "node:000",
                "LINKS",
                "node:002",
                json!({"changed": true}),
            )),
        ]);

        apply_graph_mutation_batch(&mut snapshot, &batch);
        assert_eq!(snapshot.nodes.as_ptr(), nodes_ptr);
        assert_eq!(snapshot.nodes.capacity(), nodes_capacity);
        assert_eq!(snapshot.edges.as_ptr(), edges_ptr);
        assert_eq!(snapshot.edges.capacity(), edges_capacity);
    }

    #[test]
    fn store_backed_compile_collapses_repeated_mutations_and_marks_delta_transport(
    ) -> GraphStoreResult<()> {
        let snapshot = GraphSnapshot {
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({"v": 1}))],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::from(["node:a".to_string()]),
            historical_node_history_complete: true,
        };
        let prior = compile_graph_pack(&snapshot, GraphCompileOptions::default())?;
        let store = MemChunkStore::new();
        install_tree_chunks(&store, &prior.tree_chunks)?;
        let batch = GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "node:a",
                ["Claim"],
                json!({"v": "transient"}),
            )),
            GraphMutation::NodeUpsert(snapshot.nodes[0].clone()),
        ]);
        let incremental = compile_graph_pack_incremental_with_store(
            &store,
            &prior,
            &batch,
            GraphCompileOptions::default(),
        )?;

        assert!(incremental.changed_object_keys.is_empty());
        assert!(incremental.pack.objects.is_empty());
        assert!(incremental.pack.tree_chunks.is_empty());
        assert_eq!(incremental.commit_cost.chunks_written, 0);
        assert!(!incremental.pack.tree_transport_complete);
        assert_eq!(
            compile_graph_pack_incremental(
                &incremental.pack,
                &GraphMutationBatch::default(),
                GraphCompileOptions::default(),
            )
            .unwrap_err()
            .code,
            "compacted_prior_pack_requires_chunk_store"
        );
        Ok(())
    }

    #[test]
    fn delete_incremental_node_writes_tombstone_and_checkout_omits_it() -> GraphStoreResult<()> {
        let base = GraphSnapshot {
            version: 1,
            nodes: vec![
                NodeRecord::new("node:a", ["Claim"], json!({"v": 1})),
                NodeRecord::new("node:b", ["Claim"], json!({"v": 2})),
            ],
            edges: Vec::new(),
            historical_node_ids: BTreeSet::from(["node:a".to_string(), "node:b".to_string()]),
            historical_node_history_complete: true,
        };
        let base_pack = compile_graph_pack(
            &base,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )?;
        let store = MemChunkStore::new();
        install_tree_chunks(&store, &base_pack.tree_chunks)?;
        let prior_entry = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            find_tree_entry(&mut reader, &base_pack.tree, "node/node:a")?
                .expect("base node is indexed")
        };
        let deleted = compile_graph_pack_incremental_with_store(
            &store,
            &base_pack,
            &GraphMutationBatch::new([GraphMutation::NodeDelete("node:a".to_string())]),
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(2),
                ..GraphCompileOptions::default()
            },
        )?;

        assert_ne!(deleted.pack.tree.root, base_pack.tree.root);
        assert_eq!(deleted.changed_object_keys, vec!["node/node:a"]);
        assert!(deleted.pack.objects.is_empty());
        assert!(deleted.pack.archive_objects.is_empty());
        assert_eq!(deleted.pack.manifest.nodes_total, 1);
        assert_eq!(deleted.pack.manifest.objects_total, 2);
        let tombstone = {
            let mut reader = ProllyReader::new(&store as &dyn ChunkStore);
            find_tree_entry(&mut reader, &deleted.pack.tree, "node/node:a")?
                .expect("delete retains a physical tree entry")
        };
        assert!(tombstone.tombstone);
        assert_eq!(tombstone.object_hash, prior_entry.object_hash);

        let (diff, _) = diff_graph_trees(&store, &base_pack.tree, &deleted.pack.tree)?;
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].key, "node/node:a");
        assert_eq!(diff.unchanged, 1);

        let repository = update_graph_ref(
            GraphVersionRepository::default(),
            base_pack.clone(),
            Some("main".to_string()),
            Some(1),
        )
        .expect("base pack installs")
        .repository;
        let repository = update_graph_ref(
            repository,
            deleted.pack.clone(),
            Some("main".to_string()),
            Some(2),
        )
        .expect("delete pack installs")
        .repository;
        let checkout = checkout_graph_version(&repository, "main")?;
        assert_eq!(checkout.snapshot.nodes.len(), 1);
        assert_eq!(checkout.snapshot.nodes[0].id, "node:b");
        let historical =
            checkout_graph_version(&repository, &base_pack.commit.commit_hash.to_prefixed())?;
        assert_eq!(historical.snapshot.nodes.len(), 2);
        Ok(())
    }

    #[test]
    fn delete_live_edge_then_readd_resolves_present_in_tree_mirror_and_checkout(
    ) -> GraphStoreResult<()> {
        let edge = EdgeRecord::new("edge:ab", "node:a", "LINKS", "node:b", json!({"weight": 1}));
        let base = GraphSnapshot {
            version: 1,
            nodes: vec![
                NodeRecord::new("node:a", ["Claim"], json!({})),
                NodeRecord::new("node:b", ["Claim"], json!({})),
            ],
            edges: vec![edge.clone()],
            historical_node_ids: BTreeSet::from(["node:a".to_string(), "node:b".to_string()]),
            historical_node_history_complete: true,
        };
        let base_pack = compile_graph_pack(
            &base,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(10),
                ..GraphCompileOptions::default()
            },
        )?;
        let store = MemChunkStore::new();
        install_tree_chunks(&store, &base_pack.tree_chunks)?;
        let deleted = compile_graph_pack_incremental_with_store(
            &store,
            &base_pack,
            &GraphMutationBatch::new([GraphMutation::EdgeDelete("edge:ab".to_string())]),
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(11),
                ..GraphCompileOptions::default()
            },
        )?;
        let readded = compile_graph_pack_incremental_with_store(
            &store,
            &deleted.pack,
            &GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge.clone())]),
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(12),
                ..GraphCompileOptions::default()
            },
        )?;

        assert_eq!(deleted.pack.manifest.edges_total, 0);
        assert_eq!(readded.pack.manifest.edges_total, 1);
        assert_eq!(readded.pack.tree.root, base_pack.tree.root);
        assert_eq!(
            readded.pack.tree.entries_total,
            base_pack.tree.entries_total
        );
        let (diff, _) = diff_graph_trees(&store, &deleted.pack.tree, &readded.pack.tree)?;
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].key, "edge/edge:ab");

        let raw_order_batch = GraphMutationBatch::new([
            GraphMutation::EdgeUpsert(edge.clone()),
            GraphMutation::EdgeDelete(edge.id.clone()),
        ]);
        let normalized = compile_graph_pack_incremental_with_store(
            &store,
            &base_pack,
            &raw_order_batch,
            GraphCompileOptions::default(),
        )?;
        assert_eq!(normalized.pack.tree.root, base_pack.tree.root);
        assert!(normalized.changed_object_keys.is_empty());
        let mut mirror = base.clone();
        apply_graph_mutation_batch(&mut mirror, &raw_order_batch);
        assert_eq!(mirror.edges, vec![edge.clone()]);

        let repository = update_graph_ref(
            GraphVersionRepository::default(),
            base_pack,
            Some("main".to_string()),
            Some(10),
        )
        .expect("base pack installs")
        .repository;
        let repository =
            update_graph_ref(repository, deleted.pack, Some("main".to_string()), Some(11))
                .expect("delete pack installs")
                .repository;
        let repository =
            update_graph_ref(repository, readded.pack, Some("main".to_string()), Some(12))
                .expect("re-add pack installs")
                .repository;
        let checkout = checkout_graph_version(&repository, "main")?;
        assert_eq!(checkout.snapshot.edges, vec![edge]);
        Ok(())
    }

    #[test]
    fn incremental_pack_matches_full_recompile_and_reuses_tree_nodes() {
        let base_nodes = (0..96)
            .map(|idx| {
                NodeRecord::new(
                    format!("node:{idx:03}"),
                    ["Person"],
                    json!({ "name": format!("person-{idx:03}") }),
                )
            })
            .collect::<Vec<_>>();
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 7,
            nodes: base_nodes,
            edges: Vec::new(),
        };
        let base_pack = compile_graph_pack(
            &base,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "node:095",
            ["Person"],
            json!({ "name": "person-095", "status": "updated" }),
        ))]);
        let mut final_snapshot = base.clone();
        apply_graph_mutation_batch(&mut final_snapshot, &batch);
        let incremental = compile_graph_pack_incremental(
            &base_pack,
            &batch,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                parent_commits: vec![base_pack.commit.commit_hash.to_prefixed()],
                timestamp_unix_ms: Some(2),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let full = compile_graph_pack(
            &final_snapshot,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                parent_commits: vec![base_pack.commit.commit_hash.to_prefixed()],
                timestamp_unix_ms: Some(2),
                message: Some("incremental graph mutation batch".to_string()),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();

        assert_eq!(incremental.changed_object_keys, vec!["node/node:095"]);
        assert_eq!(incremental.pack.tree.root, full.tree.root);
        assert_eq!(incremental.pack.commit.commit_hash, full.commit.commit_hash);
        assert!(incremental.changed_tree_nodes > 0);
        assert!(
            incremental.changed_tree_nodes <= 16,
            "single-record persisted delta should be height-bounded: changed={}",
            incremental.changed_tree_nodes
        );
        assert_eq!(
            incremental.commit_cost.chunks_written,
            incremental.changed_tree_nodes
        );
    }

    #[test]
    fn refs_log_and_checkout_round_trip_snapshot_payloads() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 7,
            nodes: vec![NodeRecord::new(
                "node:ada",
                ["Person"],
                json!({"name": "Ada"}),
            )],
            edges: Vec::new(),
        };
        let pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let update = update_graph_ref(
            GraphVersionRepository::default(),
            pack.clone(),
            Some("main".to_string()),
            Some(2),
        )
        .expect("pack installs");

        assert_eq!(update.reference.commit_hash, pack.commit.commit_hash);
        let log = graph_version_log(&update.repository, Some("main"));
        assert_eq!(log.commits.len(), 1);

        let checkout = checkout_graph_version(&update.repository, "main").unwrap();
        assert_eq!(checkout.commit.commit_hash, pack.commit.commit_hash);
        assert_eq!(checkout.snapshot.nodes[0].id, "node:ada");
    }

    #[test]
    fn repository_deduplicates_objects_and_tree_nodes_across_commits() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 7,
            nodes: vec![NodeRecord::new(
                "node:ada",
                ["Person"],
                json!({"name": "Ada"}),
            )],
            edges: Vec::new(),
        };
        let first_pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let first = update_graph_ref(
            GraphVersionRepository::default(),
            first_pack,
            Some("main".to_string()),
            Some(2),
        )
        .expect("first pack installs");
        let object_count = first.repository.objects.len();
        let tree_chunk_count = first.repository.tree_chunks.len();
        assert_eq!(object_count, 1);
        assert!(tree_chunk_count > 0);
        assert!(first.repository.packs[0].objects.is_empty());
        assert!(first.repository.packs[0].tree_chunks.is_empty());

        let second_pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                parent_commits: vec![first.reference.commit_hash.to_prefixed()],
                timestamp_unix_ms: Some(3),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let second = update_graph_ref(
            first.repository,
            second_pack,
            Some("main".to_string()),
            Some(4),
        )
        .expect("second pack installs");

        assert_eq!(second.repository.objects.len(), object_count);
        assert_eq!(second.repository.tree_chunks.len(), tree_chunk_count);
        let checkout = checkout_graph_version(&second.repository, "main").unwrap();
        assert_eq!(checkout.snapshot.nodes[0].id, "node:ada");
    }

    #[test]
    fn graph_ref_cas_rejects_stale_expected_commit() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:ada", ["Person"], json!({}))],
            edges: Vec::new(),
        };
        let first_pack = compile_graph_pack(
            &snapshot,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                timestamp_unix_ms: Some(1),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();
        let first = update_graph_ref(
            GraphVersionRepository::default(),
            first_pack,
            Some("main".to_string()),
            Some(2),
        )
        .expect("first pack installs");
        let mut changed = snapshot.clone();
        changed.version = 2;
        changed
            .nodes
            .push(NodeRecord::new("node:bob", ["Person"], json!({})));
        let second_pack = compile_graph_pack(
            &changed,
            GraphCompileOptions {
                branch: Some("main".to_string()),
                parent_commits: vec![first.reference.commit_hash.to_prefixed()],
                timestamp_unix_ms: Some(3),
                ..GraphCompileOptions::default()
            },
        )
        .unwrap();

        let conflict = update_graph_ref_cas(
            first.repository,
            second_pack,
            Some("main".to_string()),
            Some("sha256:stale".to_string()),
            Some(4),
        )
        .unwrap_err();

        assert_eq!(conflict.branch, "main");
        let first_commit_hash = first.reference.commit_hash.to_prefixed();
        assert_eq!(
            conflict.actual_commit_hash.as_deref(),
            Some(first_commit_hash.as_str())
        );
    }

    #[test]
    fn three_way_merge_resolves_non_overlapping_changes() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new(
                "node:ada",
                ["Person"],
                json!({"name": "Ada"}),
            )],
            edges: Vec::new(),
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![
                NodeRecord::new("node:ada", ["Person"], json!({"name": "Ada Lovelace"})),
                NodeRecord::new("node:ours", ["Note"], json!({})),
            ],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![
                NodeRecord::new("node:ada", ["Person"], json!({"name": "Ada"})),
                NodeRecord::new("node:theirs", ["Note"], json!({})),
            ],
            edges: Vec::new(),
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "clean");
        let snapshot = merged.merged_snapshot.unwrap();
        let ids = snapshot
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["node:ada", "node:ours", "node:theirs"]);
        assert!(merged.conflicts.is_empty());
    }

    #[test]
    fn tombstone_bearing_base_merges_with_checkout_shaped_sides() {
        let live = NodeRecord::new("node:live", ["Claim"], json!({"value": "retained"}));
        let mut tombstone = NodeRecord::new(
            "node:historical-delete",
            ["Claim"],
            json!({"value": "deleted"}),
        );
        tombstone.tombstone = true;
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![live.clone(), tombstone],
            edges: Vec::new(),
        };
        // checkout_graph_version intentionally returns only live records.
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![live.clone()],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![live],
            edges: Vec::new(),
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "clean");
        assert!(merged.conflicts.is_empty());
        let snapshot = merged.merged_snapshot.unwrap();
        assert_eq!(snapshot.nodes.len(), 1);
        assert_eq!(snapshot.nodes[0].id, "node:live");
        assert!(merged.resolved.iter().any(|resolution| {
            resolution.key == "node/node:historical-delete"
                && resolution.selected.is_none()
                && resolution.reason == "same_result"
        }));
    }

    #[test]
    fn three_way_merge_confidence_resolves_edge_conflicts() {
        let base_edge = EdgeRecord::new("edge:e", "a", "SUPPORTS", "b", json!({"shared": 0}))
            .with_confidence(0.4);
        let ours_edge = EdgeRecord::new(
            "edge:e",
            "a",
            "SUPPORTS",
            "b",
            json!({"shared": 0, "ours": 1}),
        )
        .with_confidence(0.9);
        let theirs_edge = EdgeRecord::new(
            "edge:e",
            "a",
            "SUPPORTS",
            "b",
            json!({"shared": 0, "theirs": 2}),
        )
        .with_confidence(0.6);
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: Vec::new(),
            edges: vec![base_edge],
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: Vec::new(),
            edges: vec![ours_edge],
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: Vec::new(),
            edges: vec![theirs_edge],
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "clean");
        let edge = &merged.merged_snapshot.unwrap().edges[0];
        assert_eq!(edge.confidence, Some(0.9));
        assert_eq!(edge.properties["ours"], 1);
        assert_eq!(edge.properties["theirs"], 2);
        assert_eq!(merged.resolved[0].reason, "higher_edge_confidence");
    }

    #[test]
    fn pruned_merge_is_byte_identical_to_legacy_across_clean_resolution_paths() {
        let mut unchanged =
            NodeRecord::new("node:00-unchanged", ["Fixture"], json!({"value": "same"}));
        unchanged.version = 1;
        unchanged.parent_hashes = vec![Hash::of(b"history-unchanged").to_prefixed()];
        let mut history = NodeRecord::new(
            "node:01-history",
            ["Fixture"],
            json!({"value": "same-logical-record"}),
        );
        history.parent_hashes = vec![Hash::of(b"history-base").to_prefixed()];
        let base_edge = EdgeRecord::new(
            "edge:confidence",
            "node:00-unchanged",
            "SUPPORTS",
            "node:02-ours",
            json!({"shared": 0}),
        )
        .with_confidence(0.4);
        let filler = (0..32)
            .map(|index| {
                NodeRecord::new(
                    format!("node:filler-{index:02}"),
                    ["Fixture"],
                    json!({"value": index}),
                )
            })
            .collect::<Vec<_>>();
        let mut base_nodes = vec![
            unchanged,
            history,
            NodeRecord::new("node:02-ours", ["Fixture"], json!({"value": "base"})),
            NodeRecord::new("node:03-theirs", ["Fixture"], json!({"value": "base"})),
            NodeRecord::new("node:04-same-result", ["Fixture"], json!({"value": "base"})),
            NodeRecord::new("node:05-delete-both", ["Fixture"], json!({"value": "base"})),
            NodeRecord::new("node:06-delete-ours", ["Fixture"], json!({"value": "base"})),
            NodeRecord::new(
                "node:07-delete-theirs",
                ["Fixture"],
                json!({"value": "base"}),
            ),
            memory_node("active", &[], hlc("base", 10)),
        ];
        base_nodes.extend(filler);
        let base = GraphSnapshot {
            historical_node_ids: base_nodes.iter().map(|node| node.id.clone()).collect(),
            historical_node_history_complete: true,
            version: 20,
            nodes: base_nodes,
            edges: vec![base_edge],
        };
        let mut ours = base.clone();
        ours.version = 21;
        ours.nodes
            .iter_mut()
            .find(|node| node.id == "node:00-unchanged")
            .unwrap()
            .version = 21;
        ours.nodes
            .iter_mut()
            .find(|node| node.id == "node:01-history")
            .unwrap()
            .parent_hashes = vec![Hash::of(b"history-ours").to_prefixed()];
        ours.nodes
            .iter_mut()
            .find(|node| node.id == "node:02-ours")
            .unwrap()
            .properties = json!({"value": "ours"});
        ours.nodes
            .iter_mut()
            .find(|node| node.id == "node:04-same-result")
            .unwrap()
            .properties = json!({"value": "joint"});
        ours.nodes.retain(|node| {
            !matches!(
                node.id.as_str(),
                "node:05-delete-both" | "node:06-delete-ours"
            )
        });
        *ours
            .nodes
            .iter_mut()
            .find(|node| node.id == "mem:a")
            .unwrap() = memory_node("archived", &[], hlc("local", 20));
        ours.nodes.push(NodeRecord::new(
            "node:add-ours",
            ["Fixture"],
            json!({"value": "ours"}),
        ));
        ours.nodes.push(NodeRecord::new(
            "node:add-both",
            ["Fixture"],
            json!({"value": "joint"}),
        ));
        ours.edges[0] = EdgeRecord::new(
            "edge:confidence",
            "node:00-unchanged",
            "SUPPORTS",
            "node:02-ours",
            json!({"shared": 0, "ours": 1}),
        )
        .with_confidence(0.9);

        let mut theirs = base.clone();
        theirs.version = 22;
        theirs
            .nodes
            .iter_mut()
            .find(|node| node.id == "node:00-unchanged")
            .unwrap()
            .version = 22;
        theirs
            .nodes
            .iter_mut()
            .find(|node| node.id == "node:03-theirs")
            .unwrap()
            .properties = json!({"value": "theirs"});
        theirs
            .nodes
            .iter_mut()
            .find(|node| node.id == "node:04-same-result")
            .unwrap()
            .properties = json!({"value": "joint"});
        theirs.nodes.retain(|node| {
            !matches!(
                node.id.as_str(),
                "node:05-delete-both" | "node:07-delete-theirs"
            )
        });
        *theirs
            .nodes
            .iter_mut()
            .find(|node| node.id == "mem:a")
            .unwrap() = memory_node("deleted", &[], hlc("railway", 30));
        theirs.nodes.push(NodeRecord::new(
            "node:add-theirs",
            ["Fixture"],
            json!({"value": "theirs"}),
        ));
        theirs.nodes.push(NodeRecord::new(
            "node:add-both",
            ["Fixture"],
            json!({"value": "joint"}),
        ));
        theirs.edges[0] = EdgeRecord::new(
            "edge:confidence",
            "node:00-unchanged",
            "SUPPORTS",
            "node:02-ours",
            json!({"shared": 0, "theirs": 2}),
        )
        .with_confidence(0.6);

        let options = GraphMergeOptions {
            name: Some("parity-pack".to_string()),
            branch: Some("parity".to_string()),
            parent_commits: vec![
                Hash::of(b"parent-a").to_prefixed(),
                Hash::of(b"parent-b").to_prefixed(),
            ],
            author: Some("parity-test".to_string()),
            message: Some("tree-walk parity".to_string()),
            timestamp_unix_ms: Some(4_242),
            include_payloads: true,
            strategy: GraphMergeStrategy::AutoConfidence,
            min_confidence_delta: 0.0,
        };
        let registry = MergeRegistry::substrate_sync_defaults();

        let actual =
            merge_graph_snapshots_with_registry(&base, &ours, &theirs, options.clone(), &registry)
                .unwrap();
        let legacy =
            legacy_merge_graph_snapshots_with_registry(&base, &ours, &theirs, options, &registry)
                .unwrap();

        assert_serialized_parity(&actual, &legacy);
        assert_eq!(actual.status, "clean");
        let reasons = actual
            .resolved
            .iter()
            .map(|resolution| resolution.reason.as_str())
            .collect::<BTreeSet<_>>();
        assert!(reasons.contains("same_result"));
        assert!(reasons.contains("ours_only_change"));
        assert!(reasons.contains("theirs_only_change"));
        assert!(reasons.contains("higher_edge_confidence"));
        assert!(actual
            .resolved
            .iter()
            .any(|resolution| resolution.reason.contains("status:lww")));
    }

    #[test]
    fn pruned_merge_is_byte_identical_to_legacy_for_conflicts_and_side_preferences() {
        let base = GraphSnapshot {
            historical_node_ids: BTreeSet::from(["node:conflict".to_string()]),
            historical_node_history_complete: true,
            version: 30,
            nodes: vec![NodeRecord::new(
                "node:conflict",
                ["Doc"],
                json!({"title": "base"}),
            )],
            edges: Vec::new(),
        };
        let mut ours = base.clone();
        ours.version = 31;
        ours.nodes[0].properties = json!({"title": "ours"});
        let mut theirs = base.clone();
        theirs.version = 32;
        theirs.nodes[0].properties = json!({"title": "theirs"});
        let registry = MergeRegistry::default();

        for strategy in [
            GraphMergeStrategy::Manual,
            GraphMergeStrategy::PreferOurs,
            GraphMergeStrategy::PreferTheirs,
        ] {
            let options = GraphMergeOptions {
                timestamp_unix_ms: Some(9_999),
                strategy,
                ..GraphMergeOptions::default()
            };
            let actual = merge_graph_snapshots_with_registry(
                &base,
                &ours,
                &theirs,
                options.clone(),
                &registry,
            )
            .unwrap();
            let legacy = legacy_merge_graph_snapshots_with_registry(
                &base, &ours, &theirs, options, &registry,
            )
            .unwrap();

            assert_serialized_parity(&actual, &legacy);
        }
    }

    #[test]
    fn compiled_pack_indexes_derived_archive_bytes_by_logical_object_hash() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({"claim": "A"}))],
            edges: vec![EdgeRecord::new(
                "edge:a",
                "node:a",
                "SUPPORTS",
                "node:a",
                json!({}),
            )],
        };
        let pack = compile_graph_pack(&snapshot, GraphCompileOptions::default()).unwrap();

        assert_eq!(pack.archive_objects.len(), pack.objects.len());
        for object in &pack.objects {
            let bytes = object_bytes(&pack, &object.hash.to_prefixed())
                .expect("archive bytes by object hash");
            let archive = pack
                .archive_objects
                .iter()
                .find(|archive| archive.hash == object.hash)
                .expect("archive index by logical object hash");
            assert_eq!(crate::zerocopy::archive_hash(bytes), archive.archive_hash);
            assert_ne!(archive.archive_hash, object.hash);
            crate::zerocopy::access_graph_archive(bytes).expect("checked archive access");
        }
    }

    #[test]
    fn incremental_pack_carries_only_changed_archive_bytes() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({"claim": "A"}))],
            edges: Vec::new(),
        };
        let base_pack = compile_graph_pack(&base, GraphCompileOptions::default()).unwrap();
        let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "node:b",
            ["Claim"],
            json!({"claim": "B"}),
        ))]);
        let incremental =
            compile_graph_pack_incremental(&base_pack, &batch, GraphCompileOptions::default())
                .unwrap();

        assert_eq!(incremental.changed_object_keys, vec!["node/node:b"]);
        assert_eq!(incremental.pack.objects.len(), 1);
        assert_eq!(incremental.pack.archive_objects.len(), 1);
        let changed = &incremental.pack.objects[0];
        assert!(object_bytes(&base_pack, &changed.hash.to_prefixed()).is_none());
        assert!(object_bytes(&incremental.pack, &changed.hash.to_prefixed()).is_some());
    }

    #[test]
    fn three_way_merge_reports_manual_node_conflict() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Doc"], json!({"title": "base"}))],
            edges: Vec::new(),
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![NodeRecord::new("node:a", ["Doc"], json!({"title": "ours"}))],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![NodeRecord::new(
                "node:a",
                ["Doc"],
                json!({"title": "theirs"}),
            )],
            edges: Vec::new(),
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "conflicted");
        assert!(merged.merged_snapshot.is_none());
        assert_eq!(merged.conflicts.len(), 1);
    }

    #[test]
    fn empty_merge_registry_preserves_predecessor_node_conflict() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new(
                "mem:a",
                ["MemoryDocument"],
                json!({"status": "active"}),
            )],
            edges: Vec::new(),
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![NodeRecord::new(
                "mem:a",
                ["MemoryDocument"],
                json!({"status": "archived"}),
            )],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![NodeRecord::new(
                "mem:a",
                ["MemoryDocument"],
                json!({"status": "deleted"}),
            )],
            edges: Vec::new(),
        };

        let merged = merge_graph_snapshots_with_registry(
            &base,
            &ours,
            &theirs,
            GraphMergeOptions::default(),
            &MergeRegistry::default(),
        )
        .unwrap();

        assert_eq!(merged.status, "conflicted");
        assert_eq!(merged.conflicts[0].reason, "both_sides_changed");
    }

    #[test]
    fn merge_registry_lww_resolves_memory_status_by_latest_hlc() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![memory_node("active", &[], hlc("base", 10))],
            edges: Vec::new(),
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![memory_node("archived", &[], hlc("local", 20))],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![memory_node("deleted", &[], hlc("railway", 30))],
            edges: Vec::new(),
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "clean");
        assert!(merged.resolved[0].reason.contains("status:lww"));
        let node = &merged.merged_snapshot.unwrap().nodes[0];
        assert_eq!(node.properties["status"].as_str(), Some("deleted"));
    }

    #[test]
    fn merge_registry_or_set_keeps_concurrent_tag_readd() {
        let base = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![memory_node("active", &["A"], hlc("base", 10))],
            edges: Vec::new(),
        };
        let ours = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 2,
            nodes: vec![memory_node("active", &[], hlc("local", 20))],
            edges: Vec::new(),
        };
        let theirs = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 3,
            nodes: vec![memory_node("active", &["A"], hlc("railway", 30))],
            edges: Vec::new(),
        };

        let merged =
            merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default()).unwrap();

        assert_eq!(merged.status, "clean");
        let node = &merged.merged_snapshot.unwrap().nodes[0];
        let tags = node.properties["tags"].as_array().expect("tags array");
        assert_eq!(tags, &vec![json!("A")]);
    }

    #[test]
    fn incremental_compile_rejects_compacted_prior_without_tree_chunks() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
        };
        let mut compacted = compile_graph_pack(&snapshot, GraphCompileOptions::default()).unwrap();
        compacted.tree_chunks.clear();
        let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "node:b",
            ["Claim"],
            json!({}),
        ))]);
        let error =
            compile_graph_pack_incremental(&compacted, &batch, GraphCompileOptions::default())
                .unwrap_err();
        assert_eq!(error.code, "compacted_prior_pack_requires_chunk_store");
    }

    #[test]
    fn historical_registry_marks_incomplete_at_resident_bound() {
        let mut ids = BTreeSet::from(["node:a".to_string(), "node:b".to_string()]);
        let mut complete = true;
        retain_historical_node_id(&mut ids, &mut complete, "node:c", 2);
        assert_eq!(ids.len(), 2);
        assert!(!ids.contains("node:c"));
        assert!(!complete);
    }

    #[test]
    fn incremental_noop_delete_does_not_pollute_historical_registry() {
        let snapshot = GraphSnapshot {
            historical_node_ids: BTreeSet::from(["node:a".to_string()]),
            historical_node_history_complete: true,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
        };
        let pack = compile_graph_pack(&snapshot, GraphCompileOptions::default()).unwrap();
        let incremental = compile_graph_pack_incremental(
            &pack,
            &GraphMutationBatch::new([GraphMutation::NodeDelete("node:missing".to_string())]),
            GraphCompileOptions::default(),
        )
        .unwrap();
        assert_eq!(
            incremental.pack.historical_node_ids,
            snapshot.historical_node_ids
        );
        assert!(incremental.pack.historical_node_history_complete);
    }

    #[test]
    fn repository_refuses_tampered_historical_registry_metadata() {
        let snapshot = GraphSnapshot {
            historical_node_ids: BTreeSet::from(["node:a".to_string()]),
            historical_node_history_complete: true,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
        };
        let mut pack = compile_graph_pack(&snapshot, GraphCompileOptions::default()).unwrap();
        pack.historical_node_ids.insert("node:forged".to_string());
        let error =
            update_graph_ref(GraphVersionRepository::default(), pack, None, None).unwrap_err();
        assert!(error.message.contains("historical registry hashes to"));
    }

    #[test]
    fn checkout_distinguishes_missing_payload_from_unknown_target() {
        let snapshot = GraphSnapshot {
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            version: 1,
            nodes: vec![NodeRecord::new("node:a", ["Claim"], json!({}))],
            edges: Vec::new(),
        };
        let pack = compile_graph_pack(&snapshot, GraphCompileOptions::default()).unwrap();
        let mut update = update_graph_ref(
            GraphVersionRepository::default(),
            pack,
            Some("main".to_string()),
            None,
        )
        .expect("pack installs");
        update
            .repository
            .objects
            .values_mut()
            .for_each(|object| object.payload = None);
        assert_eq!(
            checkout_graph_version(&update.repository, "main")
                .unwrap_err()
                .code,
            "graph_checkout_missing_payload"
        );
        assert_eq!(
            checkout_graph_version(&update.repository, "missing")
                .unwrap_err()
                .code,
            "graph_checkout_target_not_found"
        );
    }
}
