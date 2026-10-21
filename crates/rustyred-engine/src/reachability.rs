//! Typed, fail-closed reachability over the committed chunk-store catalog.
//!
//! Reachability starts exclusively at [`ChunkStore::read_root`]. A chunk is
//! decoded only as the type carried by the edge which reached it; arbitrary
//! body bytes are never inspected to guess a domain. Reclamation runs only
//! after the complete typed walk succeeds.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use serde_json::Value;

use crate::browser_storage::{
    decode_browser_database, decode_browser_origin, decode_browser_partition, decode_browser_root,
    ROOT_NAME_BROWSER,
};
use crate::chunk_store::{
    ChunkStore, ChunkStoreCompaction, CompactionReport, RootCatalog, ROOT_NAME_DOCUMENTS,
    ROOT_NAME_GRAPH, ROOT_NAME_OBJECTS, ROOT_NAME_REDCORE,
};
use crate::doc_tree::{decode_doc_reachability, DocReachability};
use crate::file_payload::{
    decode_file_manifest, FileManifestRef, FILE_CONTENT_MANIFEST_PROPERTY,
    FILE_HISTORY_ELIDED_COUNT_PROPERTY, FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY,
    MAX_PREVIOUS_FILE_MANIFESTS,
};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;
use crate::graph::vector_index::VectorDesignation;
use crate::graph_history::{
    decode_legacy_roots, GraphHistoryEntry, GraphHistoryRoot, GraphRevisionRoots,
};
use crate::graph_store::{
    decode_redcore_history_root, decode_redcore_reachability_roots,
    decode_redcore_reachability_state,
};
use crate::hash::Hash;
use crate::object_store::{decode_object_store_reachability, ObjectStoreReachability};
use crate::vector_payload::{vector_payload_references, VectorPropertyRef};
use crate::versioned_graph::{
    decode_canonical_graph_payload, decode_canonical_tree_node, GraphPayloadRecord,
    StoredProllyNode,
};

/// Hard cap on unique chunks decoded by one committed-root traversal.
pub const MAX_REACHABILITY_VISITS: usize = 1_000_000;
/// Hard cap on a typed path from the committed catalog to a terminal chunk.
pub const MAX_REACHABILITY_DEPTH: usize = 256;

/// The only chunk interpretations accepted by the reachability queue.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReachabilityKind {
    Catalog,
    LogicalProlly,
    PayloadProlly,
    HistoricalProlly,
    RedCoreState,
    HistoryRoot,
    HistoryIndex,
    HistoryIndexArchive,
    HistoryEntry,
    HistoryLegacy,
    DocRoot,
    DocEntry,
    ObjectIndex,
    ObjectPayload,
    BrowserRoot,
    BrowserPartition,
    BrowserOrigin,
    BrowserDatabase,
    PackEnvelope,
    PrivateManifestEnvelope,
    FileManifest,
    Terminal,
}

impl ReachabilityKind {
    fn is_terminal(self) -> bool {
        self == Self::Terminal
    }

    fn is_prolly(self) -> bool {
        matches!(
            self,
            Self::LogicalProlly
                | Self::PayloadProlly
                | Self::HistoricalProlly
                | Self::HistoryIndex
                | Self::HistoryIndexArchive
        )
    }
}

/// A child returned by a strict object-index or pack-envelope codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReachabilityChild {
    pub address: Hash,
    pub kind: ReachabilityKind,
}

impl ReachabilityChild {
    pub const fn new(address: Hash, kind: ReachabilityKind) -> Self {
        Self { address, kind }
    }
}

/// Strict decoding seam for typed descendant formats outside the Prolly spine.
///
/// Implementations must verify `expected_address`, domain, version, canonical
/// encoding, bounds, and trailing bytes. Separate methods make the expected
/// type positional; implementations must not select a decoder by sniffing the
/// supplied bytes.
pub trait ReachabilityCodec: Send + Sync {
    fn decode_file_manifest(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        decode_file_manifest(expected_address, bytes).map(|manifest| {
            manifest
                .chunks
                .into_iter()
                .map(|chunk| chunk.address)
                .collect()
        })
    }

    fn decode_doc_root(&self, expected_address: Hash, bytes: &[u8]) -> GraphStoreResult<Vec<Hash>>;

    fn decode_doc_entry(&self, expected_address: Hash, bytes: &[u8])
        -> GraphStoreResult<Vec<Hash>>;

    fn decode_object_index(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<ReachabilityChild>>;

    fn decode_pack_envelope(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<ReachabilityChild>>;

    fn decode_private_manifest_envelope(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>>;

    fn decode_browser_root(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>>;

    fn decode_browser_partition(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>>;

    fn decode_browser_origin(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>>;

    fn decode_browser_database(&self, expected_address: Hash, bytes: &[u8])
        -> GraphStoreResult<()>;
}

/// Strict codecs for the chunk formats owned by this crate.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreReachabilityCodec;

impl ReachabilityCodec for CoreReachabilityCodec {
    fn decode_doc_root(&self, expected_address: Hash, bytes: &[u8]) -> GraphStoreResult<Vec<Hash>> {
        match decode_doc_reachability(expected_address, bytes)? {
            DocReachability::Root { entries } => Ok(entries),
            DocReachability::Entry { .. } => Err(reachability_codec_type_mismatch(
                "document root",
                expected_address,
                "document entry",
            )),
        }
    }

    fn decode_doc_entry(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        match decode_doc_reachability(expected_address, bytes)? {
            DocReachability::Entry { bodies } => Ok(bodies),
            DocReachability::Root { .. } => Err(reachability_codec_type_mismatch(
                "document entry",
                expected_address,
                "document root",
            )),
        }
    }

    fn decode_object_index(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<ReachabilityChild>> {
        match decode_object_store_reachability(expected_address, bytes)? {
            ObjectStoreReachability::Index {
                object_payloads,
                pack_envelopes,
                private_manifest_envelopes,
            } => {
                Ok(object_payloads
                    .into_iter()
                    .map(|address| ReachabilityChild::new(address, ReachabilityKind::ObjectPayload))
                    .chain(pack_envelopes.into_iter().map(|address| {
                        ReachabilityChild::new(address, ReachabilityKind::PackEnvelope)
                    }))
                    .chain(private_manifest_envelopes.into_iter().map(|address| {
                        ReachabilityChild::new(address, ReachabilityKind::PrivateManifestEnvelope)
                    }))
                    .collect())
            }
            ObjectStoreReachability::PackEnvelope { .. }
            | ObjectStoreReachability::PrivateManifestEnvelope { .. } => {
                Err(reachability_codec_type_mismatch(
                    "object index",
                    expected_address,
                    "non-index envelope",
                ))
            }
        }
    }

    fn decode_pack_envelope(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<ReachabilityChild>> {
        match decode_object_store_reachability(expected_address, bytes)? {
            ObjectStoreReachability::PackEnvelope {
                logical_roots,
                payload_roots,
                object_payloads,
                file_manifests,
                terminals,
            } => {
                Ok(logical_roots
                    .into_iter()
                    .map(|address| ReachabilityChild::new(address, ReachabilityKind::LogicalProlly))
                    .chain(payload_roots.into_iter().map(|address| {
                        ReachabilityChild::new(address, ReachabilityKind::PayloadProlly)
                    }))
                    .chain(object_payloads.into_iter().map(|address| {
                        ReachabilityChild::new(address, ReachabilityKind::ObjectPayload)
                    }))
                    .chain(file_manifests.into_iter().map(|address| {
                        ReachabilityChild::new(address, ReachabilityKind::FileManifest)
                    }))
                    .chain(
                        terminals.into_iter().map(|address| {
                            ReachabilityChild::new(address, ReachabilityKind::Terminal)
                        }),
                    )
                    .collect())
            }
            ObjectStoreReachability::Index { .. }
            | ObjectStoreReachability::PrivateManifestEnvelope { .. } => {
                Err(reachability_codec_type_mismatch(
                    "pack envelope",
                    expected_address,
                    "non-pack object-store record",
                ))
            }
        }
    }

    fn decode_private_manifest_envelope(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        match decode_object_store_reachability(expected_address, bytes)? {
            ObjectStoreReachability::PrivateManifestEnvelope { manifest } => Ok(vec![manifest]),
            ObjectStoreReachability::Index { .. }
            | ObjectStoreReachability::PackEnvelope { .. } => {
                Err(reachability_codec_type_mismatch(
                    "private manifest envelope",
                    expected_address,
                    "different object-store record",
                ))
            }
        }
    }

    fn decode_browser_root(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        decode_browser_root(expected_address, bytes)
    }

    fn decode_browser_partition(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        decode_browser_partition(expected_address, bytes)
    }

    fn decode_browser_origin(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<Vec<Hash>> {
        decode_browser_origin(expected_address, bytes)
    }

    fn decode_browser_database(
        &self,
        expected_address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<()> {
        decode_browser_database(expected_address, bytes)
    }
}

fn reachability_codec_type_mismatch(
    expected_kind: &str,
    address: Hash,
    actual_kind: &str,
) -> GraphStoreError {
    GraphStoreError::new(
        "reachability_codec_type_mismatch",
        format!("expected {expected_kind} at {address}, decoded {actual_kind}"),
    )
}

/// Object-safe store surface required by committed reachability compaction.
pub trait ReachabilityStore: ChunkStore + ChunkStoreCompaction {
    /// Rust 1.85 does not yet support trait-object upcasting, so expose the
    /// read-only supertrait explicitly while retaining an object-safe API.
    fn as_chunk_store(&self) -> &dyn ChunkStore;
}

impl<T> ReachabilityStore for T
where
    T: ChunkStore + ChunkStoreCompaction,
{
    fn as_chunk_store(&self) -> &dyn ChunkStore {
        self
    }
}

/// Complete typed walk of one committed catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReachabilityReport {
    pub catalog_root: Hash,
    pub reachable: BTreeSet<Hash>,
    pub visits: usize,
    pub max_depth: usize,
}

/// Traversal evidence paired with the underlying store's reclamation report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReachabilityCompactionReport {
    pub reachability: ReachabilityReport,
    pub compaction: CompactionReport,
}

/// Cheap gates evaluated before an automatic maintenance invocation is
/// allowed to traverse or compact a committed store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionPolicy {
    /// Minimum journal growth since the last successful compaction.
    pub min_growth_bytes: u64,
    /// Minimum elapsed time between successful compactions.
    pub min_interval_seconds: u64,
}

/// Durable baseline from the last successful policy-driven compaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionPolicyState {
    pub last_compacted_file_bytes: u64,
    pub last_compacted_unix_seconds: u64,
}

/// Auditable reason a policy-driven maintenance invocation ran or declined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactionPolicyDecision {
    BelowGrowthThreshold {
        observed_growth_bytes: u64,
        required_growth_bytes: u64,
    },
    CadenceNotElapsed {
        observed_growth_bytes: u64,
        elapsed_seconds: u64,
        required_interval_seconds: u64,
    },
    Eligible {
        observed_growth_bytes: u64,
        elapsed_seconds: Option<u64>,
    },
}

impl CompactionPolicyDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BelowGrowthThreshold { .. } => "skipped_below_growth_threshold",
            Self::CadenceNotElapsed { .. } => "skipped_cadence_not_elapsed",
            Self::Eligible { .. } => "eligible",
        }
    }

    pub fn observed_growth_bytes(self) -> u64 {
        match self {
            Self::BelowGrowthThreshold {
                observed_growth_bytes,
                ..
            }
            | Self::CadenceNotElapsed {
                observed_growth_bytes,
                ..
            }
            | Self::Eligible {
                observed_growth_bytes,
                ..
            } => observed_growth_bytes,
        }
    }

    pub fn elapsed_seconds(self) -> Option<u64> {
        match self {
            Self::BelowGrowthThreshold { .. } => None,
            Self::CadenceNotElapsed {
                elapsed_seconds, ..
            } => Some(elapsed_seconds),
            Self::Eligible {
                elapsed_seconds, ..
            } => elapsed_seconds,
        }
    }

    pub fn is_eligible(self) -> bool {
        matches!(self, Self::Eligible { .. })
    }
}

/// Result of a policy invocation. A skipped decision always carries no report
/// because the committed store was not traversed or mutated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyCompactionOutcome {
    pub decision: CompactionPolicyDecision,
    pub report: Option<ReachabilityCompactionReport>,
}

/// Evaluate byte growth first, then cadence. A missing state is treated as a
/// zero-byte baseline with no prior compaction, which lets a large legacy
/// journal cross the threshold on its first maintenance invocation.
pub fn evaluate_compaction_policy(
    policy: CompactionPolicy,
    previous: Option<CompactionPolicyState>,
    current_file_bytes: u64,
    now_unix_seconds: u64,
) -> GraphStoreResult<CompactionPolicyDecision> {
    if policy.min_growth_bytes == 0 {
        return Err(GraphStoreError::new(
            "compaction_policy_invalid_growth_threshold",
            "automatic compaction requires a positive growth threshold",
        ));
    }

    let baseline_bytes = previous
        .map(|state| state.last_compacted_file_bytes)
        .unwrap_or(0);
    let observed_growth_bytes = current_file_bytes.saturating_sub(baseline_bytes);
    if observed_growth_bytes < policy.min_growth_bytes {
        return Ok(CompactionPolicyDecision::BelowGrowthThreshold {
            observed_growth_bytes,
            required_growth_bytes: policy.min_growth_bytes,
        });
    }

    let elapsed_seconds =
        previous.map(|state| now_unix_seconds.saturating_sub(state.last_compacted_unix_seconds));
    if let Some(elapsed_seconds) = elapsed_seconds {
        if elapsed_seconds < policy.min_interval_seconds {
            return Ok(CompactionPolicyDecision::CadenceNotElapsed {
                observed_growth_bytes,
                elapsed_seconds,
                required_interval_seconds: policy.min_interval_seconds,
            });
        }
    }

    Ok(CompactionPolicyDecision::Eligible {
        observed_growth_bytes,
        elapsed_seconds,
    })
}

/// Run a complete typed walk and compaction only when the cheap policy gates
/// admit this maintenance invocation. Callers should invoke this from a safe
/// lifecycle point which already excludes a live process owning the journal.
pub fn compact_committed_reachable_if_due(
    store: &dyn ReachabilityStore,
    codec: &dyn ReachabilityCodec,
    policy: CompactionPolicy,
    previous: Option<CompactionPolicyState>,
    current_file_bytes: u64,
    now_unix_seconds: u64,
) -> GraphStoreResult<PolicyCompactionOutcome> {
    let decision =
        evaluate_compaction_policy(policy, previous, current_file_bytes, now_unix_seconds)?;
    let report = if decision.is_eligible() {
        Some(compact_committed_reachable(store, codec)?)
    } else {
        None
    };
    Ok(PolicyCompactionOutcome { decision, report })
}

/// Per-invocation safety budget for a committed-root reachability walk.
///
/// The defaults bound untrusted/automatic maintenance. A trusted operator may
/// raise either value for a larger admitted store without disabling cycle or
/// depth protection; limit refusal happens before compaction and deletes
/// nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReachabilityLimits {
    pub max_visits: usize,
    pub max_depth: usize,
}

impl ReachabilityLimits {
    pub const fn new(max_visits: usize, max_depth: usize) -> Self {
        Self {
            max_visits,
            max_depth,
        }
    }
}

pub const PRODUCTION_REACHABILITY_LIMITS: ReachabilityLimits = ReachabilityLimits {
    max_visits: MAX_REACHABILITY_VISITS,
    max_depth: MAX_REACHABILITY_DEPTH,
};

#[derive(Debug)]
struct PathNode {
    address: Hash,
    kind: ReachabilityKind,
    parent: Option<Arc<PathNode>>,
}

#[derive(Debug)]
struct Pending {
    vector_designations: Arc<Vec<VectorDesignation>>,
    context: Hash,
    address: Hash,
    kind: ReachabilityKind,
    depth: usize,
    path: Arc<PathNode>,
}

/// Walk every chunk reachable from the store's current durable catalog.
pub fn walk_committed_reachable(
    store: &dyn ChunkStore,
    codec: &dyn ReachabilityCodec,
) -> GraphStoreResult<ReachabilityReport> {
    walk_committed_reachable_with_limits(store, codec, PRODUCTION_REACHABILITY_LIMITS)
}

/// Walk the committed catalog with an explicit trusted-operator budget.
pub fn walk_committed_reachable_with_limits(
    store: &dyn ChunkStore,
    codec: &dyn ReachabilityCodec,
    limits: ReachabilityLimits,
) -> GraphStoreResult<ReachabilityReport> {
    walk_committed_reachable_impl(store, codec, limits, &mut |_, _, _| Ok(()))
}

/// Reclaim unreachable chunks only after a complete typed walk succeeds.
///
/// The compaction implementation compares its current committed root to the
/// exact root observed by this walk while holding its mutation lock. If a
/// publisher advances the catalog after the walk, that check refuses the stale
/// reachable set and its error propagates.
pub fn compact_committed_reachable(
    store: &dyn ReachabilityStore,
    codec: &dyn ReachabilityCodec,
) -> GraphStoreResult<ReachabilityCompactionReport> {
    compact_committed_reachable_with_limits(store, codec, PRODUCTION_REACHABILITY_LIMITS)
}

/// Compact with an explicit traversal budget. A limit error is returned before
/// `compact_reachable` is invoked, leaving the physical store untouched.
pub fn compact_committed_reachable_with_limits(
    store: &dyn ReachabilityStore,
    codec: &dyn ReachabilityCodec,
    limits: ReachabilityLimits,
) -> GraphStoreResult<ReachabilityCompactionReport> {
    let reachability = walk_committed_reachable_with_limits(store.as_chunk_store(), codec, limits)?;
    let addresses = reachability.reachable.iter().copied().collect::<Vec<_>>();
    let compaction = store.compact_reachable(Some(reachability.catalog_root), &addresses)?;
    Ok(ReachabilityCompactionReport {
        reachability,
        compaction,
    })
}

/// Internal typed observation after canonical decoding, before any caller's
/// destination mutation. Existing maintenance walks keep their no-op observer.
pub(crate) fn walk_committed_reachable_with_observer(
    store: &dyn ChunkStore,
    codec: &dyn ReachabilityCodec,
    limits: ReachabilityLimits,
    observer: &mut dyn FnMut(ReachabilityKind, Hash, &[u8]) -> GraphStoreResult<()>,
) -> GraphStoreResult<ReachabilityReport> {
    walk_committed_reachable_impl(store, codec, limits, observer)
}

fn walk_committed_reachable_impl(
    store: &dyn ChunkStore,
    codec: &dyn ReachabilityCodec,
    limits: ReachabilityLimits,
    observer: &mut dyn FnMut(ReachabilityKind, Hash, &[u8]) -> GraphStoreResult<()>,
) -> GraphStoreResult<ReachabilityReport> {
    let catalog_root = store.read_root()?.ok_or_else(|| {
        GraphStoreError::new(
            "reachability_committed_root_missing",
            "cannot traverse or compact a chunk store without a committed root",
        )
    })?;
    let vector_designations = Arc::new(committed_vector_designations(store, catalog_root)?);
    let context = designation_context(&vector_designations)?;

    let root_path = Arc::new(PathNode {
        address: catalog_root,
        kind: ReachabilityKind::Catalog,
        parent: None,
    });
    let mut queue = VecDeque::from([Pending {
        address: catalog_root,
        kind: ReachabilityKind::Catalog,
        depth: 0,
        path: root_path,
        vector_designations: Arc::clone(&vector_designations),
        context,
    }]);
    let mut assigned = HashMap::from([(catalog_root, HashSet::from([ReachabilityKind::Catalog]))]);
    let mut visited = HashSet::new();
    let mut scheduled = HashSet::from([(catalog_root, ReachabilityKind::Catalog, context)]);
    let mut reachable = BTreeSet::new();
    let mut max_depth = 0;

    while let Some(pending) = queue.pop_front() {
        if visited.contains(&(pending.address, pending.kind, pending.context)) {
            continue;
        }
        if !reachable.contains(&pending.address) && reachable.len() >= limits.max_visits {
            return Err(GraphStoreError::new(
                "reachability_visit_limit_exceeded",
                format!(
                    "typed walk exceeds the configured {} unique-chunk limit",
                    limits.max_visits
                ),
            ));
        }

        let bytes = load_exact(store, pending.address)?;
        // Payload interpretation follows the recorded revision, including
        // designations subsequently removed or changed in the live graph.
        let child_designations = match pending.kind {
            ReachabilityKind::HistoryEntry => Arc::new(
                GraphHistoryEntry::from_bytes(pending.address, &bytes)?
                    .roots
                    .vector_designations,
            ),
            ReachabilityKind::HistoryLegacy => {
                Arc::new(decode_legacy_roots(pending.address, &bytes)?.vector_designations)
            }
            _ => Arc::clone(&pending.vector_designations),
        };
        let child_context = designation_context(&child_designations)?;
        let children = decode_children(
            codec,
            pending.kind,
            pending.address,
            &bytes,
            &pending.vector_designations,
        )?;

        observer(pending.kind, pending.address, &bytes)?;

        visited.insert((pending.address, pending.kind, pending.context));
        reachable.insert(pending.address);
        max_depth = max_depth.max(pending.depth);

        for child in children {
            enqueue_child(
                child,
                &pending,
                limits,
                &mut assigned,
                &mut queue,
                &mut scheduled,
                Arc::clone(&child_designations),
                child_context,
            )?;
        }
    }

    Ok(ReachabilityReport {
        catalog_root,
        visits: reachable.len(),
        reachable,
        max_depth,
    })
}

fn designation_context(designations: &[VectorDesignation]) -> GraphStoreResult<Hash> {
    serde_json::to_vec(designations)
        .map(|bytes| Hash::of(&bytes))
        .map_err(|error| {
            GraphStoreError::new("reachability_designation_encoding", error.to_string())
        })
}

fn history_revision_children(roots: &GraphRevisionRoots) -> Vec<ReachabilityChild> {
    let mut children = vec![
        ReachabilityChild::new(roots.logical_tree.root, ReachabilityKind::LogicalProlly),
        ReachabilityChild::new(roots.payload_tree.root, ReachabilityKind::PayloadProlly),
    ];
    if let Some(tree) = &roots.historical_node_tree {
        children.push(ReachabilityChild::new(
            tree.root,
            ReachabilityKind::HistoricalProlly,
        ));
    }
    children
}

fn load_exact(store: &dyn ChunkStore, address: Hash) -> GraphStoreResult<Arc<[u8]>> {
    let bytes = store.get(address)?.ok_or_else(|| {
        GraphStoreError::new(
            "reachability_chunk_missing",
            format!("typed reachable chunk {address} is missing"),
        )
    })?;
    let actual = Hash::of(&bytes);
    if actual != address {
        return Err(GraphStoreError::new(
            "reachability_chunk_address_mismatch",
            format!("reachable chunk bytes hash to {actual}, expected {address}"),
        ));
    }
    Ok(bytes)
}

fn decode_children(
    codec: &dyn ReachabilityCodec,
    kind: ReachabilityKind,
    address: Hash,
    bytes: &[u8],
    vector_designations: &[VectorDesignation],
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    match kind {
        ReachabilityKind::Catalog => decode_catalog_children(address, bytes),
        ReachabilityKind::LogicalProlly => {
            decode_prolly_children(address, bytes, ReachabilityKind::LogicalProlly)
        }
        ReachabilityKind::PayloadProlly => {
            decode_prolly_children(address, bytes, ReachabilityKind::PayloadProlly)
        }
        ReachabilityKind::HistoricalProlly => {
            decode_prolly_children(address, bytes, ReachabilityKind::HistoricalProlly)
        }
        ReachabilityKind::HistoryIndex => {
            decode_prolly_children(address, bytes, ReachabilityKind::HistoryIndex)
        }
        ReachabilityKind::HistoryIndexArchive => {
            decode_prolly_children(address, bytes, ReachabilityKind::HistoryIndexArchive)
        }
        ReachabilityKind::HistoryRoot => {
            let root = GraphHistoryRoot::from_bytes(address, bytes)?;
            let mut children = vec![
                ReachabilityChild::new(root.by_commit.root, ReachabilityKind::HistoryIndex),
                ReachabilityChild::new(root.by_transaction.root, ReachabilityKind::HistoryIndex),
                ReachabilityChild::new(
                    root.retained_indexes.root,
                    ReachabilityKind::HistoryIndexArchive,
                ),
            ];
            if let Some(legacy) = root.legacy_boundary {
                children.push(ReachabilityChild::new(
                    legacy,
                    ReachabilityKind::HistoryLegacy,
                ));
            }
            Ok(children)
        }
        ReachabilityKind::HistoryEntry => Ok(history_revision_children(
            &GraphHistoryEntry::from_bytes(address, bytes)?.roots,
        )),
        ReachabilityKind::HistoryLegacy => Ok(history_revision_children(&decode_legacy_roots(
            address, bytes,
        )?)),
        ReachabilityKind::RedCoreState => {
            let (logical_root, payload_root, historical_root) =
                decode_redcore_reachability_roots(address, bytes)?;
            let mut children = vec![
                ReachabilityChild::new(logical_root, ReachabilityKind::LogicalProlly),
                ReachabilityChild::new(payload_root, ReachabilityKind::PayloadProlly),
            ];
            if let Some(root) = historical_root {
                children.push(ReachabilityChild::new(
                    root,
                    ReachabilityKind::HistoricalProlly,
                ));
            }
            if let Some(root) = decode_redcore_history_root(address, bytes)? {
                children.push(ReachabilityChild::new(root, ReachabilityKind::HistoryRoot));
            }
            Ok(children)
        }
        ReachabilityKind::DocRoot => codec
            .decode_doc_root(address, bytes)
            .map(|entries| typed_children(entries, ReachabilityKind::DocEntry)),
        ReachabilityKind::DocEntry => codec
            .decode_doc_entry(address, bytes)
            .map(|bodies| typed_children(bodies, ReachabilityKind::Terminal)),
        ReachabilityKind::ObjectIndex => codec
            .decode_object_index(address, bytes)
            .and_then(|children| validate_object_index_children(address, children)),
        ReachabilityKind::PackEnvelope => codec
            .decode_pack_envelope(address, bytes)
            .and_then(|children| validate_pack_children(address, children)),
        ReachabilityKind::PrivateManifestEnvelope => codec
            .decode_private_manifest_envelope(address, bytes)
            .map(|manifests| typed_children(manifests, ReachabilityKind::FileManifest)),
        ReachabilityKind::ObjectPayload => {
            let record = decode_canonical_graph_payload(address, bytes)?;
            graph_payload_children(&record, vector_designations)
        }
        ReachabilityKind::BrowserRoot => codec
            .decode_browser_root(address, bytes)
            .map(|partitions| typed_children(partitions, ReachabilityKind::BrowserPartition)),
        ReachabilityKind::BrowserPartition => codec
            .decode_browser_partition(address, bytes)
            .map(|origins| typed_children(origins, ReachabilityKind::BrowserOrigin)),
        ReachabilityKind::BrowserOrigin => codec
            .decode_browser_origin(address, bytes)
            .map(|databases| typed_children(databases, ReachabilityKind::BrowserDatabase)),
        ReachabilityKind::BrowserDatabase => codec
            .decode_browser_database(address, bytes)
            .map(|()| Vec::new()),
        ReachabilityKind::FileManifest => codec
            .decode_file_manifest(address, bytes)
            .map(|chunks| typed_children(chunks, ReachabilityKind::Terminal)),
        ReachabilityKind::Terminal => Ok(Vec::new()),
    }
}

fn committed_vector_designations(
    store: &dyn ChunkStore,
    catalog_address: Hash,
) -> GraphStoreResult<Vec<VectorDesignation>> {
    let catalog_bytes = load_exact(store, catalog_address)?;
    let catalog = RootCatalog::from_bytes(&catalog_bytes)?;
    if catalog.address()? != catalog_address {
        return Err(GraphStoreError::new(
            "reachability_catalog_address_mismatch",
            format!("root catalog hashes to a different address than {catalog_address}"),
        ));
    }
    let Some(state_address) = catalog.get(ROOT_NAME_REDCORE) else {
        return Ok(Vec::new());
    };
    let state_bytes = load_exact(store, state_address)?;
    let (logical_root, _payload_root, _historical_root, designations) =
        decode_redcore_reachability_state(state_address, &state_bytes)?;
    if Some(logical_root) != catalog.get(ROOT_NAME_GRAPH) {
        return Err(GraphStoreError::new(
            "reachability_redcore_graph_root_mismatch",
            "RedCore reachability state does not match the catalog graph root",
        ));
    }
    Ok(designations)
}

/// Every committed catalog name bound to the typed decoder that interprets it.
///
/// Registration is the only way a name becomes traversable. A name that is not
/// registered is refused by name rather than assumed reachable or assumed
/// dead, so adding an authoritative product root is one declarative line here
/// plus the decoder it points at.
const REGISTERED_ROOT_DECODERS: &[(&str, ReachabilityKind)] = &[
    (ROOT_NAME_GRAPH, ReachabilityKind::LogicalProlly),
    (ROOT_NAME_REDCORE, ReachabilityKind::RedCoreState),
    (ROOT_NAME_DOCUMENTS, ReachabilityKind::DocRoot),
    (ROOT_NAME_OBJECTS, ReachabilityKind::ObjectIndex),
    (ROOT_NAME_BROWSER, ReachabilityKind::BrowserRoot),
];

fn registered_root_decoder(name: &str) -> Option<ReachabilityKind> {
    REGISTERED_ROOT_DECODERS
        .iter()
        .find(|(registered, _)| *registered == name)
        .map(|(_, kind)| *kind)
}

fn decode_catalog_children(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    let catalog = RootCatalog::from_bytes(bytes)?;
    let actual_address = catalog.address()?;
    if actual_address != expected_address {
        return Err(GraphStoreError::new(
            "reachability_catalog_address_mismatch",
            format!("root catalog hashes to {actual_address}, expected {expected_address}"),
        ));
    }
    catalog.graph_root()?;

    catalog
        .roots()
        .iter()
        .map(|(name, address)| {
            let kind = registered_root_decoder(name).ok_or_else(|| {
                GraphStoreError::new(
                    "reachability_catalog_name_unsupported",
                    format!(
                        "committed catalog name {name:?} has no registered typed reachability decoder"
                    ),
                )
            })?;
            Ok(ReachabilityChild::new(*address, kind))
        })
        .collect()
}

fn decode_prolly_children(
    address: Hash,
    bytes: &[u8],
    tree_kind: ReachabilityKind,
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    let node = decode_canonical_tree_node(address, bytes)?;
    match node {
        StoredProllyNode::Parent(children) => Ok(children
            .into_iter()
            .map(|child| ReachabilityChild::new(child.hash, tree_kind))
            .collect()),
        StoredProllyNode::Leaf(entries) => {
            let entry_kind = match tree_kind {
                ReachabilityKind::LogicalProlly => ReachabilityKind::Terminal,
                ReachabilityKind::PayloadProlly => ReachabilityKind::ObjectPayload,
                ReachabilityKind::HistoricalProlly => ReachabilityKind::Terminal,
                ReachabilityKind::HistoryIndex => ReachabilityKind::HistoryEntry,
                ReachabilityKind::HistoryIndexArchive => ReachabilityKind::HistoryIndex,
                _ => {
                    return Err(GraphStoreError::new(
                        "reachability_internal_prolly_type_invalid",
                        format!("{tree_kind:?} is not a Prolly-tree queue type"),
                    ))
                }
            };
            Ok(entries
                .into_iter()
                .map(|entry| ReachabilityChild::new(entry.object_hash, entry_kind))
                .collect())
        }
    }
}

fn typed_children(addresses: Vec<Hash>, kind: ReachabilityKind) -> Vec<ReachabilityChild> {
    addresses
        .into_iter()
        .map(|address| ReachabilityChild::new(address, kind))
        .collect()
}

fn validate_object_index_children(
    parent: Hash,
    children: Vec<ReachabilityChild>,
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    validate_extension_children(
        "object index",
        parent,
        children,
        &[
            ReachabilityKind::ObjectPayload,
            ReachabilityKind::PackEnvelope,
            ReachabilityKind::PrivateManifestEnvelope,
            ReachabilityKind::Terminal,
        ],
    )
}

fn validate_pack_children(
    parent: Hash,
    children: Vec<ReachabilityChild>,
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    validate_extension_children(
        "pack envelope",
        parent,
        children,
        &[
            ReachabilityKind::LogicalProlly,
            ReachabilityKind::PayloadProlly,
            ReachabilityKind::ObjectPayload,
            ReachabilityKind::FileManifest,
            ReachabilityKind::Terminal,
        ],
    )
}

fn validate_extension_children(
    parent_kind: &str,
    parent: Hash,
    children: Vec<ReachabilityChild>,
    allowed: &[ReachabilityKind],
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    if let Some(child) = children.iter().find(|child| !allowed.contains(&child.kind)) {
        return Err(GraphStoreError::new(
            "reachability_extension_child_type_invalid",
            format!(
                "{parent_kind} {parent} returned unsupported {:?} child {}",
                child.kind, child.address
            ),
        ));
    }
    Ok(children)
}

/// Future packed-property formats must add explicit, typed references here.
/// Deliberately do not search node/edge JSON strings for hash-shaped values.
fn graph_payload_children(
    record: &GraphPayloadRecord,
    vector_designations: &[VectorDesignation],
) -> GraphStoreResult<Vec<ReachabilityChild>> {
    let children =
        match record {
            GraphPayloadRecord::Node(node) => {
                let mut children = vector_payload_references(node, vector_designations)?
                    .into_iter()
                    .map(|address| ReachabilityChild::new(address, ReachabilityKind::Terminal))
                    .collect::<Vec<_>>();
                children.extend(
                    tensor_block_payload_references(node)?
                        .into_iter()
                        .map(|address| ReachabilityChild::new(address, ReachabilityKind::Terminal)),
                );
                children.extend(file_manifest_payload_references(node)?.into_iter().map(
                    |address| ReachabilityChild::new(address, ReachabilityKind::FileManifest),
                ));
                children.extend(
                    raw_payload_references(node)?
                        .into_iter()
                        .map(|address| ReachabilityChild::new(address, ReachabilityKind::Terminal)),
                );
                children
            }
            GraphPayloadRecord::Edge(_) => Vec::new(),
        };
    Ok(children)
}

/// Decode every schema-owned external payload reference carried by a full
/// node record embedded in an immutable parked pack.
///
/// Unlike the live graph, a parked pack does not carry the vector-designation
/// registry which normally types vector properties. The strict two-field
/// [`VectorPropertyRef`] representation is therefore the positional type
/// marker for vectors inside the pack. All other owners reuse the same
/// label/property decoders as live graph reachability.
pub(crate) fn packed_graph_node_payload_references(
    node: &NodeRecord,
) -> GraphStoreResult<(Vec<Hash>, Vec<Hash>)> {
    let mut terminals = BTreeSet::new();
    if let Some(properties) = node.properties.as_object() {
        for value in properties.values() {
            let Some(reference) = value.as_object() else {
                continue;
            };
            if reference.len() == 2
                && reference.contains_key("content_id")
                && reference.contains_key("dimension")
            {
                let reference = VectorPropertyRef::from_value(value)?;
                terminals.insert(Hash::parse_prefixed(&reference.content_id)?);
            }
        }
    }
    terminals.extend(tensor_block_payload_references(node)?);
    terminals.extend(raw_payload_references(node)?);

    let file_manifests = file_manifest_payload_references(node)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    Ok((
        file_manifests.into_iter().collect(),
        terminals.into_iter().collect(),
    ))
}

/// Raw payload addresses are recognized only at schema-owned label/property
/// positions. This deliberately does not scan arbitrary strings for hashes:
/// adding another raw-payload owner requires adding its exact typed decoder.
fn raw_payload_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    let mut addresses = Vec::new();

    if node.labels.iter().any(|label| label == "FsInode") {
        addresses.extend(agentfs_chunk_references(node)?);
    }

    if node.labels.iter().any(|label| label == "Item") {
        addresses.extend(commonplace_item_blob_references(node)?);
    }

    if node.labels.iter().any(|label| label == "Page") {
        addresses.extend(rustyweb_evidence_references(node)?);
    }

    if node.labels.iter().any(|label| label == "TheoremChatUpload") {
        addresses.extend(chat_upload_payload_references(node)?);
    }

    if node.labels.iter().any(|label| label == "TopicPreviewAsset") {
        let content_hash = required_string_property(
            node,
            "content_hash",
            "topic_preview_payload_reference_invalid",
        )?;
        addresses.push(parse_raw_sha256(
            node,
            "content_hash",
            content_hash,
            "topic_preview_payload_reference_invalid",
        )?);
    }

    if node
        .labels
        .iter()
        .any(|label| label == "ColdMultiVectorArtifact")
    {
        if node.properties.get("vectors").is_some() {
            return Err(GraphStoreError::new(
                "legacy_multivector_payload_requires_reupsert",
                format!(
                    "ColdMultiVectorArtifact node {} stores vectors in graph properties; re-upsert the embedding set",
                    node.id
                ),
            ));
        }
        let exact_object_ref = required_string_property(
            node,
            "exact_object_ref",
            "multivector_payload_reference_invalid",
        )?;
        let exact_address =
            parse_multivector_object_reference(node, "exact_object_ref", exact_object_ref)?;
        let cold_address = node
            .properties
            .get("cold_object_hash")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    GraphStoreError::new(
                        "multivector_payload_reference_invalid",
                        format!(
                            "ColdMultiVectorArtifact node {} property cold_object_hash must be a string",
                            node.id
                        ),
                    )
                })
            })
            .transpose()?
            .map(|value| {
                parse_raw_sha256(
                    node,
                    "cold_object_hash",
                    value,
                    "multivector_payload_reference_invalid",
                )
            })
            .transpose()?;

        match (exact_address, cold_address) {
            (Some(exact), Some(cold)) if exact != cold => {
                return Err(GraphStoreError::new(
                    "multivector_payload_reference_mismatch",
                    format!(
                        "ColdMultiVectorArtifact node {} exact_object_ref resolves to {exact}, but cold_object_hash is {cold}",
                        node.id
                    ),
                ));
            }
            (Some(address), _) => addresses.push(address),
            (None, Some(cold)) => {
                return Err(GraphStoreError::new(
                    "multivector_payload_reference_mismatch",
                    format!(
                        "ColdMultiVectorArtifact node {} has local cold_object_hash {cold}, but exact_object_ref is external",
                        node.id
                    ),
                ));
            }
            (None, None) => {}
        }
    }

    if node
        .labels
        .iter()
        .any(|label| label == "HotMultiVectorProjection")
    {
        if node.properties.get("words").is_some() || node.properties.get("words_hex").is_some() {
            return Err(GraphStoreError::new(
                "legacy_multivector_payload_requires_reupsert",
                format!(
                    "HotMultiVectorProjection node {} stores packed words in graph properties; re-upsert the embedding set",
                    node.id
                ),
            ));
        }
        let binary_projection_ref = required_string_property(
            node,
            "binary_projection_ref",
            "multivector_payload_reference_invalid",
        )?;
        let projection_address = parse_multivector_object_reference(
            node,
            "binary_projection_ref",
            binary_projection_ref,
        )?;
        let binary_address = node
            .properties
            .get("binary_object_hash")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    GraphStoreError::new(
                        "multivector_payload_reference_invalid",
                        format!(
                            "HotMultiVectorProjection node {} property binary_object_hash must be a string",
                            node.id
                        ),
                    )
                })
            })
            .transpose()?
            .map(|value| {
                parse_raw_sha256(
                    node,
                    "binary_object_hash",
                    value,
                    "multivector_payload_reference_invalid",
                )
            })
            .transpose()?;

        match (projection_address, binary_address) {
            (Some(projection), Some(binary)) if projection != binary => {
                return Err(GraphStoreError::new(
                    "multivector_payload_reference_mismatch",
                    format!(
                        "HotMultiVectorProjection node {} binary_projection_ref resolves to {projection}, but binary_object_hash is {binary}",
                        node.id
                    ),
                ));
            }
            (Some(address), _) => addresses.push(address),
            (None, Some(binary)) => {
                return Err(GraphStoreError::new(
                    "multivector_payload_reference_mismatch",
                    format!(
                        "HotMultiVectorProjection node {} has local binary_object_hash {binary}, but binary_projection_ref is external",
                        node.id
                    ),
                ));
            }
            (None, None) => {}
        }
    }

    if node
        .labels
        .iter()
        .any(|label| label == "MultiVectorManifest")
    {
        let exact_object_ref = required_string_property(
            node,
            "exact_object_ref",
            "multivector_payload_reference_invalid",
        )?;
        if let Some(address) =
            parse_multivector_object_reference(node, "exact_object_ref", exact_object_ref)?
        {
            addresses.push(address);
        }
        let binary_projection_ref = required_string_property(
            node,
            "binary_projection_ref",
            "multivector_payload_reference_invalid",
        )?;
        if let Some(address) = parse_multivector_object_reference(
            node,
            "binary_projection_ref",
            binary_projection_ref,
        )? {
            addresses.push(address);
        }
    }

    let mut seen = BTreeSet::new();
    addresses.retain(|address| seen.insert(*address));
    Ok(addresses)
}

fn chat_upload_payload_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    const ERROR_CODE: &str = "chat_upload_payload_reference_invalid";
    let state = required_string_property(node, "state", ERROR_CODE)?;
    if matches!(state, "removed" | "failed" | "canceled") {
        // These are the graph-level release states. Removal is a tombstone a
        // message-referenced upload never enters, so its bytes stay reachable
        // until message retention releases them. A failed or canceled upload
        // never became admissible context: its quarantined source bytes, and
        // whatever extraction had produced before it stopped, are collectible
        // on the next global GC instead of holding quota until expiry.
        return Ok(Vec::new());
    }
    if !matches!(
        state,
        "pending" | "uploading" | "received" | "scanning" | "extracting" | "indexing" | "ready"
    ) {
        return Err(GraphStoreError::new(
            ERROR_CODE,
            format!(
                "TheoremChatUpload node {} has unknown state {state}",
                node.id
            ),
        ));
    }

    let mut addresses = Vec::with_capacity(3);
    let content = required_string_property(node, "contentRef", ERROR_CODE)?;
    addresses.push(parse_raw_sha256(node, "contentRef", content, ERROR_CODE)?);
    // The context document and the extracted text exist only from the
    // extraction stage on. A received or scanning upload owns its source bytes
    // alone; once either reference is written it is retained like the source.
    for property in ["contextDocumentRef", "extractedContentRef"] {
        let Some(value) = node
            .properties
            .get(property)
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        let value = value
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                GraphStoreError::new(
                    ERROR_CODE,
                    format!(
                        "TheoremChatUpload node {} property {property} must be a non-empty string",
                        node.id
                    ),
                )
            })?;
        addresses.push(parse_raw_sha256(node, property, value, ERROR_CODE)?);
    }
    let mut seen = BTreeSet::new();
    addresses.retain(|address| seen.insert(*address));
    Ok(addresses)
}

fn agentfs_chunk_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    let Some(chunks) = node.properties.get("chunks") else {
        return Ok(Vec::new());
    };
    let chunks = chunks.as_array().ok_or_else(|| {
        GraphStoreError::new(
            "agentfs_chunk_reference_invalid",
            format!("FsInode node {} property chunks must be an array", node.id),
        )
    })?;
    chunks
        .iter()
        .enumerate()
        .filter_map(|(index, value)| match value.as_str() {
            // AgentFS represents an unmaterialized sparse-file chunk with the
            // empty string. It owns no payload and therefore contributes no
            // reachability edge.
            Some("") => None,
            Some(value) => Some(parse_raw_sha256(
                node,
                &format!("chunks[{index}]"),
                value,
                "agentfs_chunk_reference_invalid",
            )),
            None => Some(Err(GraphStoreError::new(
                "agentfs_chunk_reference_invalid",
                format!(
                    "FsInode node {} property chunks[{index}] must be a string",
                    node.id
                ),
            ))),
        })
        .collect()
}

fn commonplace_item_blob_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    let mut addresses = Vec::new();
    if let Some(body) = node.properties.get("body").and_then(Value::as_object) {
        if body.get("body_kind").and_then(Value::as_str) == Some("blob") {
            addresses.push(parse_nested_raw_sha256(
                node,
                body,
                "body.content_hash",
                "content_hash",
                "commonplace_blob_reference_invalid",
            )?);
        }
    }
    if let Some(extra) = node.properties.get("extra").and_then(Value::as_object) {
        for property in ["original_blob", "extracted_markdown_blob"] {
            let Some(blob) = extra.get(property) else {
                continue;
            };
            let blob = blob.as_object().ok_or_else(|| {
                GraphStoreError::new(
                    "commonplace_blob_reference_invalid",
                    format!("Item node {} extra.{property} must be an object", node.id),
                )
            })?;
            addresses.push(parse_nested_raw_sha256(
                node,
                blob,
                &format!("extra.{property}.content_hash"),
                "content_hash",
                "commonplace_blob_reference_invalid",
            )?);
        }
    }
    Ok(addresses)
}

fn rustyweb_evidence_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    let Some(package_hash) = node.properties.get("evidence_package_hash") else {
        return Ok(Vec::new());
    };
    let package_hash = package_hash.as_str().ok_or_else(|| {
        GraphStoreError::new(
            "evidence_payload_reference_invalid",
            format!(
                "Page node {} property evidence_package_hash must be a string",
                node.id
            ),
        )
    })?;
    let package_address = parse_raw_sha256(
        node,
        "evidence_package_hash",
        package_hash,
        "evidence_payload_reference_invalid",
    )?;
    let members = node
        .properties
        .get("evidence_member_hashes")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GraphStoreError::new(
                "evidence_payload_manifest_required_for_gc",
                format!(
                    "Page node {} names an evidence package without the typed evidence_member_hashes closure",
                    node.id
                ),
            )
    })?;
    let mut addresses = Vec::with_capacity(members.len().saturating_add(1));
    addresses.push(package_address);
    let mut member_includes_package = false;
    for (index, member) in members.iter().enumerate() {
        let member = member.as_str().ok_or_else(|| {
            GraphStoreError::new(
                "evidence_payload_reference_invalid",
                format!(
                    "Page node {} property evidence_member_hashes[{index}] must be a string",
                    node.id
                ),
            )
        })?;
        let address = parse_raw_sha256(
            node,
            &format!("evidence_member_hashes[{index}]"),
            member,
            "evidence_payload_reference_invalid",
        )?;
        member_includes_package |= address == package_address;
        addresses.push(address);
    }
    if !member_includes_package {
        return Err(GraphStoreError::new(
            "evidence_payload_manifest_required_for_gc",
            format!(
                "Page node {} evidence_member_hashes must include its package index identity",
                node.id
            ),
        ));
    }
    Ok(addresses)
}

fn parse_nested_raw_sha256(
    node: &NodeRecord,
    object: &serde_json::Map<String, Value>,
    path: &str,
    field: &str,
    error_code: &'static str,
) -> GraphStoreResult<Hash> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            GraphStoreError::new(
                error_code,
                format!("node {} requires non-empty string property {path}", node.id),
            )
        })?;
    parse_raw_sha256(node, path, value, error_code)
}

fn required_string_property<'a>(
    node: &'a NodeRecord,
    property: &str,
    error_code: &'static str,
) -> GraphStoreResult<&'a str> {
    node.properties
        .get(property)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            GraphStoreError::new(
                error_code,
                format!(
                    "node {} requires non-empty string property {property}",
                    node.id
                ),
            )
        })
}

fn parse_raw_sha256(
    node: &NodeRecord,
    property: &str,
    value: &str,
    error_code: &'static str,
) -> GraphStoreResult<Hash> {
    Hash::parse_prefixed(value).map_err(|error| {
        GraphStoreError::new(
            error_code,
            format!(
                "node {} property {property} is not a canonical raw SHA-256 address: {}",
                node.id, error.message
            ),
        )
    })
}

/// A multi-vector exact object may be held by the local raw-document tier or
/// by an explicit external pointer. Only the two canonical local forms create
/// GC descendants; malformed local forms fail instead of being treated as an
/// unrelated external reference.
fn parse_multivector_object_reference(
    node: &NodeRecord,
    property: &str,
    object_ref: &str,
) -> GraphStoreResult<Option<Hash>> {
    if let Some(content_id) = object_ref.strip_prefix("cold://document/") {
        return parse_raw_sha256(
            node,
            property,
            content_id,
            "multivector_payload_reference_invalid",
        )
        .map(Some);
    }
    if object_ref.starts_with("sha256:") {
        return parse_raw_sha256(
            node,
            property,
            object_ref,
            "multivector_payload_reference_invalid",
        )
        .map(Some);
    }
    if object_ref.starts_with("cold://") || !object_ref.contains("://") {
        return Err(GraphStoreError::new(
            "multivector_payload_reference_invalid",
            format!(
                "node {} property {property} is neither a canonical local SHA-256 reference nor an explicit external URI",
                node.id,
            ),
        ));
    }
    Ok(None)
}

/// File-manifest addresses are positional graph references. Unrelated
/// hash-shaped metadata and file nodes without a body reference are ignored;
/// once the exact property is present its one-field shape is strict.
fn file_manifest_payload_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    const FILE_LABELS: [&str; 2] = ["File", "CodeFile"];
    if !node
        .labels
        .iter()
        .any(|label| FILE_LABELS.contains(&label.as_str()))
    {
        return Ok(Vec::new());
    }
    let Some(current) = node.properties.get(FILE_CONTENT_MANIFEST_PROPERTY) else {
        if node
            .properties
            .get(FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY)
            .is_some()
            || node
                .properties
                .get(FILE_HISTORY_ELIDED_COUNT_PROPERTY)
                .is_some()
        {
            return Err(GraphStoreError::new(
                "file_manifest_history_invalid",
                "File history properties require a current content_manifest",
            ));
        }
        return Ok(Vec::new());
    };
    let mut references = vec![FileManifestRef::from_value(current)?];
    if let Some(history) = node
        .properties
        .get(FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY)
    {
        let history = history.as_array().ok_or_else(|| {
            GraphStoreError::new(
                "file_manifest_history_invalid",
                "previous_content_manifests must be an array of strict manifest references",
            )
        })?;
        if history.len() > MAX_PREVIOUS_FILE_MANIFESTS {
            return Err(GraphStoreError::new(
                "file_manifest_history_invalid",
                format!(
                    "previous_content_manifests has {} entries; maximum is {MAX_PREVIOUS_FILE_MANIFESTS}",
                    history.len()
                ),
            ));
        }
        for value in history {
            references.push(FileManifestRef::from_value(value)?);
        }
    }
    if let Some(elided) = node.properties.get(FILE_HISTORY_ELIDED_COUNT_PROPERTY) {
        if elided.as_u64().is_none_or(|count| count == 0) {
            return Err(GraphStoreError::new(
                "file_manifest_history_invalid",
                "history_elided_count must be a positive integer when present",
            ));
        }
    }

    let mut seen = BTreeSet::new();
    let mut addresses = Vec::with_capacity(references.len());
    for reference in references {
        if !seen.insert(reference.content_id.clone()) {
            return Err(GraphStoreError::new(
                "file_manifest_history_invalid",
                "current and previous file manifests must be unique",
            ));
        }
        addresses.push(reference.address()?);
    }
    Ok(addresses)
}

/// TensorBlock addresses are positional graph references, just like vector
/// designations. Restricting recognition to the two canonical TensorBlock
/// labels prevents unrelated hash-shaped strings from becoming GC roots.
fn tensor_block_payload_references(node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
    const TENSOR_LABELS: [&str; 2] = ["TensorBlock", "TensorBlockGeneration"];
    if !node
        .labels
        .iter()
        .any(|label| TENSOR_LABELS.contains(&label.as_str()))
    {
        return Ok(Vec::new());
    }

    ["manifest_content_id", "payload_content_id"]
        .into_iter()
        .map(|property| {
            let content_id = node
                .properties
                .get(property)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "tensor_payload_reference_invalid",
                        format!(
                            "TensorBlock node {} requires string property {property}",
                            node.id
                        ),
                    )
                })?;
            Hash::parse_prefixed(content_id).map_err(|error| {
                GraphStoreError::new(
                    "tensor_payload_reference_invalid",
                    format!(
                        "TensorBlock node {} property {property} is not canonical: {}",
                        node.id, error.message
                    ),
                )
            })
        })
        .collect()
}

fn enqueue_child(
    child: ReachabilityChild,
    parent: &Pending,
    limits: ReachabilityLimits,
    assigned: &mut HashMap<Hash, HashSet<ReachabilityKind>>,
    queue: &mut VecDeque<Pending>,
    scheduled: &mut HashSet<(Hash, ReachabilityKind, Hash)>,
    vector_designations: Arc<Vec<VectorDesignation>>,
    context: Hash,
) -> GraphStoreResult<()> {
    let depth = parent.depth.checked_add(1).ok_or_else(|| {
        GraphStoreError::new(
            "reachability_depth_overflow",
            "typed reachability depth overflowed usize",
        )
    })?;
    if depth > limits.max_depth {
        return Err(GraphStoreError::new(
            "reachability_depth_limit_exceeded",
            format!(
                "typed path to {} exceeds the configured depth limit {}",
                child.address, limits.max_depth
            ),
        ));
    }
    if path_contains(&parent.path, child.address, child.kind) {
        return Err(GraphStoreError::new(
            "reachability_cycle_detected",
            format!(
                "typed edge from {} to {} forms a reachability cycle",
                parent.address, child.address
            ),
        ));
    }

    if let Some(existing_kinds) = assigned.get_mut(&child.address) {
        if let Some(existing) = existing_kinds
            .iter()
            .copied()
            .find(|existing| !reachability_kinds_compatible(*existing, child.kind))
        {
            return Err(GraphStoreError::new(
                "reachability_type_conflict",
                format!(
                    "chunk {} is reachable as incompatible nonterminal types {:?} and {:?}",
                    child.address, existing, child.kind
                ),
            ));
        }
        existing_kinds.insert(child.kind);
    } else {
        if assigned.len() >= limits.max_visits {
            return Err(GraphStoreError::new(
                "reachability_visit_limit_exceeded",
                format!(
                    "typed walk exceeds the configured {} unique-chunk limit",
                    limits.max_visits
                ),
            ));
        }
        assigned.insert(child.address, HashSet::from([child.kind]));
    }

    if !scheduled.insert((child.address, child.kind, context)) {
        return Ok(());
    }
    if scheduled.len() > limits.max_visits.saturating_mul(32) {
        return Err(GraphStoreError::new(
            "reachability_context_limit_exceeded",
            "revision-specific payload interpretations exceed traversal budget",
        ));
    }
    let path = Arc::new(PathNode {
        address: child.address,
        kind: child.kind,
        parent: Some(Arc::clone(&parent.path)),
    });
    queue.push_back(Pending {
        address: child.address,
        kind: child.kind,
        depth,
        path,
        vector_designations,
        context,
    });
    Ok(())
}

fn reachability_kinds_compatible(left: ReachabilityKind, right: ReachabilityKind) -> bool {
    left == right
        || left.is_terminal()
        || right.is_terminal()
        || (left.is_prolly() && right.is_prolly())
}

fn path_contains(path: &Arc<PathNode>, candidate: Hash, kind: ReachabilityKind) -> bool {
    let mut cursor = Some(path.as_ref());
    while let Some(node) = cursor {
        if node.address == candidate && node.kind == kind {
            return true;
        }
        cursor = node.parent.as_deref();
    }
    false
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    use super::{
        compact_committed_reachable, compact_committed_reachable_if_due,
        compact_committed_reachable_with_limits, evaluate_compaction_policy,
        graph_payload_children, walk_committed_reachable, walk_committed_reachable_with_limits,
        CompactionPolicy, CompactionPolicyDecision, CompactionPolicyState, CoreReachabilityCodec,
        ReachabilityChild, ReachabilityCodec, ReachabilityKind, ReachabilityLimits,
        ReachabilityStore,
    };
    use crate::chunk_store::{
        validate_catalog_root, ChunkStore, ChunkStoreCompaction, CommitMeta, CompactionReport,
        MemChunkStore, RootCatalog, ROOT_NAME_DOCUMENTS, ROOT_NAME_GRAPH, ROOT_NAME_OBJECTS,
        ROOT_NAME_REDCORE,
    };
    use crate::cold_index::ColdTierKind;
    use crate::doc_tree::{LegacyDocTree, LegacyDocTreeFixture, PathKey};
    use crate::file_payload::{
        stage_file_payload, FileChunkRef, FileManifest, FileManifestRef,
        MAX_PREVIOUS_FILE_MANIFESTS,
    };
    use crate::graph::error::{GraphStoreError, GraphStoreResult};
    use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
    use crate::graph::record::NodeRecord;
    use crate::graph::snapshot::GraphSnapshot;
    use crate::graph::vector_index::VectorDesignation;
    use crate::graph_store::{RedCoreDurability, RedCoreGraphStore, RedCoreOptions};
    use crate::hash::Hash;
    use crate::object_store::{ColdObjectStore, DiskObjectStore, UnifiedChunkStore};
    use crate::vector_payload::{pack_vector, write_vector_property, VectorPropertyRef};
    use crate::versioned_graph::{
        canonical_leaf_preimage, canonical_parent_preimage, compile_graph_pack,
        node_to_content_object, GraphCompileOptions, GraphObjectKind, GraphPayloadRecord,
        GraphTreeChild, GraphTreeEntry,
    };
    use serde_json::json;

    static NEXT_GLOBAL_GC_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct RejectExternalCodec;

    impl ReachabilityCodec for RejectExternalCodec {
        fn decode_file_manifest(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_doc_root(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_doc_entry(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_object_index(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<ReachabilityChild>> {
            Err(unexpected_external_codec())
        }

        fn decode_pack_envelope(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<ReachabilityChild>> {
            Err(unexpected_external_codec())
        }

        fn decode_private_manifest_envelope(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_browser_root(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_browser_partition(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_browser_origin(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<Vec<Hash>> {
            Err(unexpected_external_codec())
        }

        fn decode_browser_database(
            &self,
            _expected_address: Hash,
            _bytes: &[u8],
        ) -> GraphStoreResult<()> {
            Err(unexpected_external_codec())
        }
    }

    fn unexpected_external_codec() -> GraphStoreError {
        GraphStoreError::new(
            "test_external_codec_unexpected",
            "fixture unexpectedly reached an external codec",
        )
    }

    fn write_exact(store: &dyn ChunkStore, bytes: &[u8]) -> Hash {
        let address = Hash::of(bytes);
        store.write(&[(address, bytes)]).unwrap();
        address
    }

    fn leaf_with_terminal(store: &dyn ChunkStore, body: &[u8]) -> (Hash, Hash) {
        let body_address = write_exact(store, body);
        let leaf = canonical_leaf_preimage(&[GraphTreeEntry {
            key: "node/reachable".to_string(),
            kind: GraphObjectKind::Node,
            object_hash: body_address,
            tombstone: false,
        }])
        .unwrap();
        (write_exact(store, &leaf), body_address)
    }

    fn publish_catalog(store: &MemChunkStore, roots: BTreeMap<String, Hash>) -> Hash {
        let catalog = RootCatalog::new(roots).unwrap();
        let catalog_address = catalog.write_to(store).unwrap();
        store
            .commit(
                catalog_address,
                CommitMeta {
                    parents: Vec::new(),
                    author: "reachability-test".to_string(),
                    message: "publish fixture".to_string(),
                    timestamp_unix_ms: 1,
                    graph_version: 1,
                },
            )
            .unwrap();
        catalog_address
    }

    fn global_gc_meta(version: u64) -> CommitMeta {
        CommitMeta {
            parents: Vec::new(),
            author: "global-gc-test".to_string(),
            message: format!("publish global GC fixture {version}"),
            timestamp_unix_ms: u128::from(version),
            graph_version: version,
        }
    }

    fn global_gc_options() -> RedCoreOptions {
        RedCoreOptions {
            durability: RedCoreDurability::AofAlways,
            snapshot_interval_writes: 1,
            strict_acid: true,
        }
    }

    #[test]
    fn compact_removes_orphan_and_retains_all_typed_descendants() {
        let store = MemChunkStore::new();
        let raw_body = [2, 2, 0, 0, 0, 0, 0xff];
        let (leaf_address, body_address) = leaf_with_terminal(&store, &raw_body);
        let parent = canonical_parent_preimage(&[GraphTreeChild {
            hash: leaf_address,
            first_key: "node/reachable".to_string(),
            last_key: "node/reachable".to_string(),
            entries_total: 1,
        }])
        .unwrap();
        let parent_address = write_exact(&store, &parent);
        let orphan_address = write_exact(&store, b"unreferenced chunk");
        let catalog_address = publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), parent_address)]),
        );

        let report = compact_committed_reachable(&store, &RejectExternalCodec).unwrap();

        assert_eq!(report.compaction.chunks_removed, 1);
        assert_eq!(
            report.reachability.reachable,
            BTreeSet::from([catalog_address, parent_address, leaf_address, body_address])
        );
        assert!(!store.has(orphan_address).unwrap());
        for address in &report.reachability.reachable {
            assert!(store.has(*address).unwrap());
        }
    }

    #[test]
    fn parked_pack_vector_and_file_payloads_survive_compaction_and_unpark() -> GraphStoreResult<()>
    {
        let backend = Arc::new(MemChunkStore::new());
        let unified: Arc<dyn UnifiedChunkStore> = backend.clone();
        let facade = DiskObjectStore::from_chunk_store(unified)?;

        let file_body = vec![b'f'; 70_000];
        let file_staging = stage_file_payload(
            backend.as_ref(),
            "pack-reachability:file:content_manifest",
            &file_body,
        )?;
        let manifest_address = file_staging.reference().address()?;
        let file_chunk_addresses = file_staging
            .manifest()
            .chunks
            .iter()
            .map(|chunk| chunk.address)
            .collect::<Vec<_>>();

        let vector = [1.0_f32, -0.0, 0.5];
        let vector_bytes = pack_vector(&vector);
        let vector_staging = facade.stage_raw_payload(&vector_bytes)?;
        let vector_address = vector_staging.address();
        let vector_reference = VectorPropertyRef {
            content_id: vector_address.to_prefixed(),
            dimension: vector.len(),
        };
        let file_node = NodeRecord::new(
            "file:parked-pack",
            ["File"],
            json!({
                "content_manifest": file_staging.reference().to_value()?,
                "embedding": vector_reference.to_value()?,
            }),
        );
        let pack = compile_graph_pack(
            &GraphSnapshot {
                version: 1,
                nodes: vec![file_node],
                ..GraphSnapshot::default()
            },
            GraphCompileOptions::default(),
        )?;
        let commit_hash = pack.commit.commit_hash;
        facade.put_pack(&pack)?;
        file_staging.finish_after_publication();
        vector_staging.finish_after_publication();

        let report = compact_committed_reachable(backend.as_ref(), &CoreReachabilityCodec)?;
        assert!(report.reachability.reachable.contains(&vector_address));
        assert!(report.reachability.reachable.contains(&manifest_address));
        assert_eq!(
            backend.get(vector_address)?.as_deref(),
            Some(vector_bytes.as_slice())
        );
        for address in &file_chunk_addresses {
            assert!(report.reachability.reachable.contains(address));
            assert!(backend.has(*address)?);
        }
        assert_eq!(facade.get_pack(commit_hash)?, Some(pack));
        Ok(())
    }

    #[test]
    fn journal_global_gc_preserves_legacy_documents_root_and_every_named_root(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-global-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let options = global_gc_options();

        let mut redcore = RedCoreGraphStore::open(&data_dir, options.clone())?;
        redcore.upsert_node(NodeRecord::new(
            "graph:gc-live",
            ["Claim"],
            json!({"status": "rooted"}),
        ))?;
        redcore.designate_vector_property("Doc", "embedding", 3)?;
        redcore.upsert_node(NodeRecord::new(
            "graph:gc-vector",
            ["Doc"],
            json!({"embedding": [1.0, -0.0, 0.5]}),
        ))?;
        let vector_node = redcore
            .get_node("graph:gc-vector")?
            .expect("vector node was published");
        let vector_address = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&vector_node.properties["embedding"])?.content_id,
        )?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let shared: Arc<dyn UnifiedChunkStore> = journal.clone();
        let objects = DiskObjectStore::from_chunk_store(shared)?;

        let small_path = PathKey::from_slash_path("tenant/project/doc/inline")?;
        let history_path = PathKey::from_slash_path("tenant/project/doc/history")?;
        let small_body = b"tiny";
        let mut documents = LegacyDocTreeFixture::new(16);
        let small_entry = documents.put_body(
            small_path.clone(),
            small_body,
            ColdTierKind::Cold,
            1,
            None,
            &objects,
        )?;
        assert!(small_entry.is_inline());
        let small_address = Hash::parse_prefixed(
            small_entry
                .content_hash
                .as_deref()
                .expect("inline body retains its raw address"),
        )?;

        let mut history_addresses = Vec::new();
        let mut latest_history_body = Vec::new();
        let mut latest_history_entry = None;
        for version in 0..66_u64 {
            let body = format!("history-version-{version:03}-overflow-body").into_bytes();
            history_addresses.push(Hash::of(&body));
            latest_history_body = body.clone();
            latest_history_entry = Some(documents.put_body(
                history_path.clone(),
                &body,
                ColdTierKind::Cold,
                i64::try_from(version).expect("bounded fixture version"),
                None,
                &objects,
            )?);
        }
        let latest_history_entry = latest_history_entry.expect("history fixture has versions");
        assert!(!latest_history_entry.is_inline());
        assert_eq!(latest_history_entry.previous_hashes.len(), 64);
        assert_eq!(latest_history_entry.history_elided_count, 1);
        let elided_address = history_addresses[0];
        let retained_history = std::iter::once(
            latest_history_entry
                .content_hash
                .as_deref()
                .expect("history current address"),
        )
        .chain(
            latest_history_entry
                .previous_hashes
                .iter()
                .map(String::as_str),
        )
        .map(Hash::parse_prefixed)
        .collect::<GraphStoreResult<BTreeSet<_>>>()?;
        assert_eq!(retained_history.len(), 65);
        assert!(!retained_history.contains(&elided_address));
        documents.publish(&objects, global_gc_meta(2))?;

        let archived_node = NodeRecord::new(
            "object:gc-live",
            ["Memory"],
            json!({"body": "rooted object payload"}),
        );
        let archived_object = node_to_content_object(&archived_node)?;
        objects.put_object(&archived_object)?;

        let orphan_address =
            Hash::parse_prefixed(&objects.put_document_bytes(b"global GC orphan")?)?;
        assert!(journal.has(orphan_address)?);
        assert!(journal.has(elided_address)?);

        let catalog_address = journal.read_root()?.expect("committed root catalog");
        let catalog = validate_catalog_root(journal.as_ref(), catalog_address)?;
        for root_name in [
            ROOT_NAME_GRAPH,
            ROOT_NAME_REDCORE,
            ROOT_NAME_DOCUMENTS,
            ROOT_NAME_OBJECTS,
        ] {
            assert!(catalog.get(root_name).is_some(), "missing {root_name} root");
        }

        let report = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(report.compaction.chunks_removed >= 2);
        assert!(!journal.has(orphan_address)?);
        assert!(!journal.has(elided_address)?);
        assert!(journal.has(small_address)?);
        assert!(journal.has(vector_address)?);
        assert!(report.reachability.reachable.contains(&vector_address));
        assert!(report.reachability.reachable.contains(&small_address));
        for address in &retained_history {
            assert!(
                journal.has(*address)?,
                "retained body {address} was removed"
            );
            assert!(report.reachability.reachable.contains(address));
        }
        assert_eq!(
            documents.resolve_body(&small_path, &objects)?,
            Some(small_body.to_vec())
        );
        assert_eq!(
            documents.resolve_body(&history_path, &objects)?,
            Some(latest_history_body.clone())
        );

        drop(documents);
        drop(objects);
        drop(journal);
        drop(redcore);

        let reopened_redcore = RedCoreGraphStore::open(&data_dir, options)?;
        assert!(reopened_redcore.get_node("graph:gc-live")?.is_some());
        assert_eq!(
            reopened_redcore.vector_search(Some("Doc"), "embedding", &[1.0, -0.0, 0.5], 1)?[0].0,
            "graph:gc-vector"
        );
        let reopened_journal = reopened_redcore
            .chunk_store_handle()
            .expect("reopened RedCore exposes its journal");
        let reopened_shared: Arc<dyn UnifiedChunkStore> = reopened_journal.clone();
        let reopened_objects = DiskObjectStore::from_chunk_store(reopened_shared)?;
        let reopened_documents = LegacyDocTree::open_legacy_committed(&reopened_objects)?
            .expect("documents root survives compaction and reopen");
        let reopened_history = reopened_documents
            .get(&history_path)
            .expect("history entry survives reopen");
        assert_eq!(reopened_history.previous_hashes.len(), 64);
        assert_eq!(reopened_history.history_elided_count, 1);
        assert_eq!(
            reopened_documents.resolve_body(&small_path, &reopened_objects)?,
            Some(small_body.to_vec())
        );
        assert_eq!(
            reopened_documents.resolve_body(&history_path, &reopened_objects)?,
            Some(latest_history_body)
        );
        assert_eq!(
            reopened_objects.get_object(archived_object.hash)?,
            Some(archived_object)
        );
        assert!(!reopened_journal.has(orphan_address)?);
        assert!(!reopened_journal.has(elided_address)?);
        for address in retained_history {
            assert!(reopened_journal.has(address)?);
        }

        drop(reopened_documents);
        drop(reopened_objects);
        drop(reopened_journal);
        drop(reopened_redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn vector_payload_pin_closes_publication_vs_compaction_window() -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-vector-pin-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        redcore.designate_vector_property("Doc", "embedding", 3)?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&redcore)?;
        let mut node = NodeRecord::new("doc:pin-window", ["Doc"], json!({}));
        write_vector_property(
            &mut node,
            "embedding",
            &[1.0, -0.0, 0.5],
            payload_store.as_ref(),
        )?;
        let payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&node.properties["embedding"])?.content_id,
        )?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 1);

        // Deterministic interleaving: the payload is verified, but the graph
        // node has not been committed. It is absent from ordinary reachability
        // and survives this compaction solely because of the in-flight pin.
        let before_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(!before_publication.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);

        redcore.upsert_node(node)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        let historical = redcore.pin_graph_history()?.head_revision()?;
        let after_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(after_publication.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);

        // A delete tombstone intentionally retains the prior full-record
        // payload address, so committed history keeps the vector reachable.
        assert!(redcore.delete_node("doc:pin-window")?);
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        let after_delete = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(
            after_delete.reachability.reachable.contains(&payload),
            "the tombstone must retain its prior record payload and descendants"
        );
        assert!(journal.has(payload)?);

        // Re-adding replaces the live payload. The transient publication pin
        // is released, while durable history must retain the prior vector.
        let mut replacement = NodeRecord::new("doc:pin-window", ["Doc"], json!({}));
        write_vector_property(
            &mut replacement,
            "embedding",
            &[0.0, 1.0, 0.0],
            payload_store.as_ref(),
        )?;
        let replacement_payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&replacement.properties["embedding"])?.content_id,
        )?;
        redcore.upsert_node(replacement)?;
        let after_replacement =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        assert_eq!(journal.in_flight_pin_count(replacement_payload)?, 0);
        assert!(after_replacement.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);
        assert!(journal.has(replacement_payload)?);
        let historical_node = historical.get_node("doc:pin-window")?.unwrap();
        let historical_payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&historical_node.properties["embedding"])?.content_id,
        )?;
        assert_eq!(historical_payload, payload);
        assert_ne!(historical_payload, replacement_payload);

        drop(payload_store);
        drop(journal);
        drop(redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn raw_preview_and_multivector_pins_bridge_publication_then_typed_gc_preserves_payloads(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-raw-payload-pin-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");

        let preview_bytes = b"tenant-held preview bytes";
        let exact_bytes = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let binary_bytes = [1_u64, 2_u64]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let preview_receipt = redcore
            .stage_cold_document_bytes(preview_bytes)?
            .expect("durable RedCore stages preview bytes");
        let exact_receipt = redcore
            .stage_cold_document_bytes(&exact_bytes)?
            .expect("durable RedCore stages exact multi-vector bytes");
        let binary_receipt = redcore
            .stage_cold_document_bytes(&binary_bytes)?
            .expect("durable RedCore stages binary multi-vector bytes");
        let preview_address = preview_receipt.address();
        let exact_address = exact_receipt.address();
        let binary_address = binary_receipt.address();
        assert_eq!(journal.in_flight_pin_count(preview_address)?, 1);
        assert_eq!(journal.in_flight_pin_count(exact_address)?, 1);
        assert_eq!(journal.in_flight_pin_count(binary_address)?, 1);

        // Deterministically compact after physical writes but before graph
        // publication. Neither address is reachable yet; the scoped receipts
        // are the sole reason reclamation cannot delete the bytes.
        let before_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(!before_publication
            .reachability
            .reachable
            .contains(&preview_address));
        assert!(!before_publication
            .reachability
            .reachable
            .contains(&exact_address));
        assert!(!before_publication
            .reachability
            .reachable
            .contains(&binary_address));
        assert!(journal.has(preview_address)?);
        assert!(journal.has(exact_address)?);
        assert!(journal.has(binary_address)?);

        redcore.commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "topic-preview:pin-window",
                ["TopicPreviewAsset"],
                json!({"content_hash": preview_receipt.content_id()}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "multivector:exact:pin-window",
                ["ColdMultiVectorArtifact"],
                json!({
                    "exact_object_ref": format!("cold://document/{}", exact_receipt.content_id()),
                    "cold_object_hash": exact_receipt.content_id(),
                }),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "multivector:binary:pin-window",
                ["HotMultiVectorProjection"],
                json!({
                    "binary_projection_ref": format!(
                        "cold://document/{}",
                        binary_receipt.content_id()
                    ),
                    "binary_object_hash": binary_receipt.content_id(),
                }),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "multivector:manifest:pin-window",
                ["MultiVectorManifest"],
                json!({
                    "exact_object_ref": exact_receipt.content_id(),
                    "binary_projection_ref": binary_receipt.content_id(),
                }),
            )),
        ]))?;
        preview_receipt.finish_after_publication();
        exact_receipt.finish_after_publication();
        binary_receipt.finish_after_publication();
        assert_eq!(journal.in_flight_pin_count(preview_address)?, 0);
        assert_eq!(journal.in_flight_pin_count(exact_address)?, 0);
        assert_eq!(journal.in_flight_pin_count(binary_address)?, 0);

        let after_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(after_publication
            .reachability
            .reachable
            .contains(&preview_address));
        assert!(after_publication
            .reachability
            .reachable
            .contains(&exact_address));
        assert!(after_publication
            .reachability
            .reachable
            .contains(&binary_address));
        assert_eq!(
            redcore.get_cold_document_bytes(&preview_address.to_prefixed())?,
            Some(preview_bytes.to_vec())
        );
        assert_eq!(
            redcore.get_cold_document_bytes(&exact_address.to_prefixed())?,
            Some(exact_bytes.clone())
        );
        assert_eq!(
            redcore.get_cold_document_bytes(&binary_address.to_prefixed())?,
            Some(binary_bytes.clone())
        );

        drop(journal);
        drop(redcore);
        let reopened = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        assert_eq!(
            reopened.get_cold_document_bytes(&preview_address.to_prefixed())?,
            Some(preview_bytes.to_vec())
        );
        assert_eq!(
            reopened.get_cold_document_bytes(&exact_address.to_prefixed())?,
            Some(exact_bytes)
        );
        assert_eq!(
            reopened.get_cold_document_bytes(&binary_address.to_prefixed())?,
            Some(binary_bytes)
        );
        drop(reopened);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn agentfs_and_commonplace_raw_references_survive_publication_gc_and_restart(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-producer-raw-payload-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");

        let agentfs_receipt = redcore
            .stage_cold_document_bytes(b"agentfs chunk")?
            .expect("stage AgentFS chunk");
        let item_body_receipt = redcore
            .stage_cold_document_bytes(b"commonplace file body")?
            .expect("stage CommonPlace body");
        let original_receipt = redcore
            .stage_cold_document_bytes(b"commonplace original")?
            .expect("stage CommonPlace original");
        let markdown_receipt = redcore
            .stage_cold_document_bytes(b"commonplace markdown")?
            .expect("stage CommonPlace markdown");
        let addresses = [
            agentfs_receipt.address(),
            item_body_receipt.address(),
            original_receipt.address(),
            markdown_receipt.address(),
        ];

        let before_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        for address in addresses {
            assert!(!before_publication.reachability.reachable.contains(&address));
            assert!(journal.has(address)?, "publication pin retains {address}");
        }

        redcore.commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "fs:inode:producer-gc",
                ["FsInode"],
                json!({
                    "chunks": [agentfs_receipt.content_id(), ""],
                }),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "item:producer-gc",
                ["Item"],
                json!({
                    "body": {
                        "body_kind": "blob",
                        "content_hash": item_body_receipt.content_id(),
                        "byte_len": 21,
                    },
                    "extra": {
                        "original_blob": {
                            "content_hash": original_receipt.content_id(),
                            "byte_len": 20,
                        },
                        "extracted_markdown_blob": {
                            "content_hash": markdown_receipt.content_id(),
                            "byte_len": 21,
                        },
                    },
                }),
            )),
        ]))?;
        agentfs_receipt.finish_after_publication();
        item_body_receipt.finish_after_publication();
        original_receipt.finish_after_publication();
        markdown_receipt.finish_after_publication();

        let after_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        for address in addresses {
            assert!(after_publication.reachability.reachable.contains(&address));
            assert!(journal.has(address)?, "typed graph edge retains {address}");
        }

        drop(journal);
        drop(redcore);
        let reopened = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        for address in addresses {
            assert!(reopened
                .get_cold_document_bytes(&address.to_prefixed())?
                .is_some());
        }
        drop(reopened);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn evidence_package_without_typed_member_manifest_refuses_gc_without_deleting_bytes(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-evidence-gc-refusal-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let package_receipt = redcore
            .stage_cold_document_bytes(b"evidence package index")?
            .expect("stage evidence package");
        let member_receipt = redcore
            .stage_cold_document_bytes(b"evidence package member")?
            .expect("stage evidence member");
        let package_address = package_receipt.address();
        let member_address = member_receipt.address();
        redcore.upsert_node(NodeRecord::new(
            "page:evidence-gc-refusal",
            ["Page"],
            json!({"evidence_package_hash": package_receipt.content_id()}),
        ))?;
        package_receipt.finish_after_publication();
        member_receipt.finish_after_publication();

        let error = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)
            .expect_err("an evidence package without member closure must block compaction");
        assert_eq!(error.code, "evidence_payload_manifest_required_for_gc");
        assert!(journal.has(package_address)?);
        assert!(journal.has(member_address)?);

        drop(journal);
        drop(redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn rustyweb_evidence_member_manifest_survives_gc_and_restart() -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-evidence-manifest-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let package_receipt = redcore
            .stage_cold_document_bytes(b"canonical SHA-256 evidence package index")?
            .expect("stage evidence package");
        let member_receipt = redcore
            .stage_cold_document_bytes(b"canonical SHA-256 evidence package member")?
            .expect("stage evidence member");
        let package_address = package_receipt.address();
        let member_address = member_receipt.address();
        redcore.upsert_node(NodeRecord::new(
            "page:evidence-manifest-gc",
            ["Page"],
            json!({
                "evidence_package_hash": package_receipt.content_id(),
                "evidence_member_hashes": [
                    package_receipt.content_id(),
                    member_receipt.content_id(),
                ],
            }),
        ))?;
        package_receipt.finish_after_publication();
        member_receipt.finish_after_publication();

        let report = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(report.reachability.reachable.contains(&package_address));
        assert!(report.reachability.reachable.contains(&member_address));
        assert!(journal.has(package_address)?);
        assert!(journal.has(member_address)?);

        drop(journal);
        drop(redcore);
        let reopened = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        assert!(reopened
            .get_cold_document_bytes(&package_address.to_prefixed())?
            .is_some());
        assert!(reopened
            .get_cold_document_bytes(&member_address.to_prefixed())?
            .is_some());
        drop(reopened);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn builtin_vector_payload_position_is_claimed_and_reachable_without_an_index_designation(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-builtin-vector-pin-gc-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        assert!(redcore.vector_designations().is_empty());
        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&redcore)?;
        let mut shadow = NodeRecord::new(
            "epistemic-shadow:pin-window",
            ["EpistemicShadow"],
            json!({}),
        );
        write_vector_property(
            &mut shadow,
            "structural_role_vector",
            &[0.25, -0.0, 1.0],
            payload_store.as_ref(),
        )?;
        let payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&shadow.properties["structural_role_vector"])?
                .content_id,
        )?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 1);

        redcore.upsert_node(shadow)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        let historical = redcore.pin_graph_history()?.head_revision()?;
        let after_publication =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(after_publication.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);

        assert!(redcore.delete_node("epistemic-shadow:pin-window")?);
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        let after_delete = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(after_delete.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);

        let mut replacement = NodeRecord::new(
            "epistemic-shadow:pin-window",
            ["EpistemicShadow"],
            json!({}),
        );
        write_vector_property(
            &mut replacement,
            "structural_role_vector",
            &[0.0, 1.0, 0.0],
            payload_store.as_ref(),
        )?;
        let replacement_payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&replacement.properties["structural_role_vector"])?
                .content_id,
        )?;
        redcore.upsert_node(replacement)?;
        let after_replacement =
            compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        assert_eq!(journal.in_flight_pin_count(replacement_payload)?, 0);
        assert!(after_replacement.reachability.reachable.contains(&payload));
        assert!(journal.has(payload)?);
        assert!(journal.has(replacement_payload)?);
        let historical_node = historical.get_node("epistemic-shadow:pin-window")?.unwrap();
        let historical_payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&historical_node.properties["structural_role_vector"])?
                .content_id,
        )?;
        assert_eq!(historical_payload, payload);
        assert_ne!(historical_payload, replacement_payload);

        drop(payload_store);
        drop(journal);
        drop(redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn vector_publication_cannot_steal_another_nodes_deduplicated_pending_pin(
    ) -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-vector-pin-owner-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        redcore.designate_vector_property("Doc", "embedding", 3)?;
        redcore.upsert_node(NodeRecord::new(
            "doc:published-a",
            ["Doc"],
            json!({"embedding": [1.0, -0.0, 0.5]}),
        ))?;

        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&redcore)?;
        let published_a = redcore
            .get_node("doc:published-a")?
            .expect("first node is published");
        let mut pending_b = NodeRecord::new("doc:pending-b", ["Doc"], json!({}));
        write_vector_property(
            &mut pending_b,
            "embedding",
            &[1.0, -0.0, 0.5],
            payload_store.as_ref(),
        )?;
        let payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&pending_b.properties["embedding"])?.content_id,
        )?;
        assert_eq!(journal.bound_pending_pin_count(payload)?, 1);

        // Republishing A claims its own transaction pin. It must not consume
        // B's keyed producer receipt merely because both references dedupe to
        // the same physical content address.
        redcore.upsert_node(published_a)?;
        assert_eq!(journal.bound_pending_pin_count(payload)?, 1);
        assert_eq!(journal.claimed_pin_count(payload)?, 0);

        assert!(redcore.delete_node("doc:published-a")?);
        compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(
            journal.has(payload)?,
            "B's pending receipt must protect the deduplicated payload"
        );

        redcore.upsert_node(pending_b)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(journal.has(payload)?);

        drop(payload_store);
        drop(journal);
        drop(redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn identical_vector_rewrite_preserves_the_existing_ownership_domain() -> GraphStoreResult<()> {
        let data_dir = std::env::temp_dir().join(format!(
            "rustyred-vector-pin-identical-{}-{}",
            std::process::id(),
            NEXT_GLOBAL_GC_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&data_dir);
        let mut redcore = RedCoreGraphStore::open(&data_dir, global_gc_options())?;
        redcore.designate_vector_property("Doc", "embedding", 3)?;
        redcore.upsert_node(NodeRecord::new(
            "doc:committed-identical",
            ["Doc"],
            json!({"embedding": [1.0, -0.0, 0.5]}),
        ))?;

        let journal = redcore
            .chunk_store_handle()
            .expect("durable RedCore exposes its journal");
        let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&redcore)?;
        let mut committed = redcore
            .get_node("doc:committed-identical")?
            .expect("node is committed");
        let payload = Hash::parse_prefixed(
            &VectorPropertyRef::from_value(&committed.properties["embedding"])?.content_id,
        )?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);

        // The durable root already owns these exact bytes. Re-running the
        // accessor must verify and reuse that reference without manufacturing
        // then cancelling a fresh pending receipt.
        write_vector_property(
            &mut committed,
            "embedding",
            &[1.0, -0.0, 0.5],
            payload_store.as_ref(),
        )?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);
        compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(journal.has(payload)?);
        redcore.upsert_node(committed)?;

        // A detached writer, by contrast, owns one pending receipt. An exact
        // repeated accessor call preserves that receipt for the later upsert.
        let mut detached = NodeRecord::new("doc:detached-identical", ["Doc"], json!({}));
        write_vector_property(
            &mut detached,
            "embedding",
            &[1.0, -0.0, 0.5],
            payload_store.as_ref(),
        )?;
        assert_eq!(journal.bound_pending_pin_count(payload)?, 1);
        write_vector_property(
            &mut detached,
            "embedding",
            &[1.0, -0.0, 0.5],
            payload_store.as_ref(),
        )?;
        assert_eq!(journal.bound_pending_pin_count(payload)?, 1);
        redcore.upsert_node(detached)?;
        assert_eq!(journal.in_flight_pin_count(payload)?, 0);

        drop(payload_store);
        drop(journal);
        drop(redcore);
        let _ = std::fs::remove_dir_all(data_dir);
        Ok(())
    }

    #[test]
    fn terminal_body_is_not_sniffed_as_a_tree_domain() {
        let store = MemChunkStore::new();
        let body = [2, 2, 0, 0, 0, 0, 0xff];
        let (leaf_address, body_address) = leaf_with_terminal(&store, &body);
        publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), leaf_address)]),
        );

        let report = walk_committed_reachable(&store, &RejectExternalCodec).unwrap();

        assert!(report.reachable.contains(&body_address));
        assert_eq!(store.get(body_address).unwrap().unwrap().as_ref(), body);
    }

    #[test]
    fn vector_property_reachability_is_designation_positional_and_strict() {
        let payload = Hash::of(b"packed vector payload");
        let designation = VectorDesignation {
            label: "Doc".to_string(),
            property: "embedding".to_string(),
            dimension: 2,
        };
        let record = crate::versioned_graph::GraphPayloadRecord::Node(NodeRecord::new(
            "doc:vector-reachability",
            ["Doc"],
            json!({
                "embedding": {"content_id": payload.to_prefixed(), "dimension": 2},
                "unrelated": {"content_id": "not-a-hash", "dimension": "metadata"}
            }),
        ));
        assert_eq!(
            graph_payload_children(&record, std::slice::from_ref(&designation)).unwrap(),
            vec![ReachabilityChild::new(payload, ReachabilityKind::Terminal)]
        );

        let malformed = crate::versioned_graph::GraphPayloadRecord::Node(NodeRecord::new(
            "doc:malformed-vector-reachability",
            ["Doc"],
            json!({
                "embedding": {
                    "content_id": payload.to_prefixed(),
                    "dimension": 2,
                    "extra": true
                }
            }),
        ));
        assert_eq!(
            graph_payload_children(&malformed, &[designation])
                .unwrap_err()
                .code,
            "vector_property_reference_invalid"
        );

        let scalar = crate::versioned_graph::GraphPayloadRecord::Node(NodeRecord::new(
            "doc:scalar-vector-reachability",
            ["Doc"],
            json!({"embedding": "not-a-vector-reference"}),
        ));
        assert_eq!(
            graph_payload_children(
                &scalar,
                &[VectorDesignation {
                    label: "Doc".to_string(),
                    property: "embedding".to_string(),
                    dimension: 2,
                }]
            )
            .unwrap_err()
            .code,
            "vector_property_reference_invalid"
        );
    }

    #[test]
    fn builtin_vector_property_reachability_is_schema_positional_and_strict() {
        let payload = Hash::of(b"epistemic structural role");
        let record = GraphPayloadRecord::Node(NodeRecord::new(
            "epistemic-shadow:typed-vector",
            ["EpistemicShadow"],
            json!({
                "structural_role_vector": {
                    "content_id": payload.to_prefixed(),
                    "dimension": 3
                },
                "unrelated": {"content_id": "not-a-hash", "dimension": "metadata"}
            }),
        ));
        assert_eq!(
            graph_payload_children(&record, &[]).unwrap(),
            vec![ReachabilityChild::new(payload, ReachabilityKind::Terminal)]
        );

        let malformed = GraphPayloadRecord::Node(NodeRecord::new(
            "epistemic-shadow:malformed-vector",
            ["EpistemicShadow"],
            json!({
                "structural_role_vector": {
                    "content_id": payload.to_prefixed(),
                    "dimension": 3,
                    "extra": true
                }
            }),
        ));
        assert_eq!(
            graph_payload_children(&malformed, &[]).unwrap_err().code,
            "vector_property_reference_invalid"
        );

        let unrelated = GraphPayloadRecord::Node(NodeRecord::new(
            "metadata:looks-like-vector",
            ["Metadata"],
            json!({
                "structural_role_vector": {
                    "content_id": payload.to_prefixed(),
                    "dimension": 3
                }
            }),
        ));
        assert!(graph_payload_children(&unrelated, &[]).unwrap().is_empty());
    }

    #[test]
    fn raw_preview_and_multivector_reachability_is_label_positional_and_strict() {
        let preview = Hash::of(b"held preview bytes");
        let exact = Hash::of(b"exact multi-vector bytes");
        let binary = Hash::of(b"binary multi-vector bytes");
        let preview_record = GraphPayloadRecord::Node(NodeRecord::new(
            "topic-preview:typed",
            ["TopicPreviewAsset"],
            json!({
                "content_hash": preview.to_prefixed(),
                "unrelated_hash": exact.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&preview_record, &[]).unwrap(),
            vec![ReachabilityChild::new(preview, ReachabilityKind::Terminal)]
        );

        let exact_record = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:exact:typed",
            ["ColdMultiVectorArtifact"],
            json!({
                "exact_object_ref": format!("cold://document/{exact}"),
                "cold_object_hash": exact.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&exact_record, &[]).unwrap(),
            vec![ReachabilityChild::new(exact, ReachabilityKind::Terminal)]
        );

        let binary_record = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:binary:typed",
            ["HotMultiVectorProjection"],
            json!({
                "binary_projection_ref": format!("cold://document/{binary}"),
                "binary_object_hash": binary.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&binary_record, &[]).unwrap(),
            vec![ReachabilityChild::new(binary, ReachabilityKind::Terminal)]
        );

        let raw_manifest_record = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:manifest:typed",
            ["MultiVectorManifest"],
            json!({
                "exact_object_ref": exact.to_prefixed(),
                "binary_projection_ref": binary.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&raw_manifest_record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(exact, ReachabilityKind::Terminal),
                ReachabilityChild::new(binary, ReachabilityKind::Terminal),
            ]
        );

        let external_record = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:exact:external",
            ["ColdMultiVectorArtifact"],
            json!({"exact_object_ref": "s3://fixture/exact.bin"}),
        ));
        assert!(graph_payload_children(&external_record, &[])
            .unwrap()
            .is_empty());

        let malformed_preview = GraphPayloadRecord::Node(NodeRecord::new(
            "topic-preview:malformed",
            ["TopicPreviewAsset"],
            json!({"content_hash": "sha256:not-canonical"}),
        ));
        assert_eq!(
            graph_payload_children(&malformed_preview, &[])
                .unwrap_err()
                .code,
            "topic_preview_payload_reference_invalid"
        );

        let mismatched_exact = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:exact:mismatch",
            ["ColdMultiVectorArtifact"],
            json!({
                "exact_object_ref": format!("cold://document/{exact}"),
                "cold_object_hash": preview.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&mismatched_exact, &[])
                .unwrap_err()
                .code,
            "multivector_payload_reference_mismatch"
        );

        let mismatched_binary = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:binary:mismatch",
            ["HotMultiVectorProjection"],
            json!({
                "binary_projection_ref": format!("cold://document/{binary}"),
                "binary_object_hash": preview.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&mismatched_binary, &[])
                .unwrap_err()
                .code,
            "multivector_payload_reference_mismatch"
        );

        let malformed_exact = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:exact:malformed",
            ["ColdMultiVectorArtifact"],
            json!({"exact_object_ref": "cold://document/sha256:not-canonical"}),
        ));
        assert_eq!(
            graph_payload_children(&malformed_exact, &[])
                .unwrap_err()
                .code,
            "multivector_payload_reference_invalid"
        );

        let legacy_exact = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:exact:legacy",
            ["ColdMultiVectorArtifact"],
            json!({
                "vectors": [[1.0, 0.0]],
                "exact_object_ref": exact.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&legacy_exact, &[]).unwrap_err().code,
            "legacy_multivector_payload_requires_reupsert"
        );

        let legacy_binary = GraphPayloadRecord::Node(NodeRecord::new(
            "multivector:binary:legacy",
            ["HotMultiVectorProjection"],
            json!({
                "words_hex": ["0000000000000001"],
                "binary_projection_ref": binary.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&legacy_binary, &[])
                .unwrap_err()
                .code,
            "legacy_multivector_payload_requires_reupsert"
        );

        let agentfs_record = GraphPayloadRecord::Node(NodeRecord::new(
            "fs:inode:7",
            ["FsInode"],
            json!({
                "chunks": [preview.to_prefixed(), "", exact.to_prefixed()],
            }),
        ));
        assert_eq!(
            graph_payload_children(&agentfs_record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(preview, ReachabilityKind::Terminal),
                ReachabilityChild::new(exact, ReachabilityKind::Terminal),
            ]
        );

        let commonplace_record = GraphPayloadRecord::Node(NodeRecord::new(
            "item:typed-blobs",
            ["Item"],
            json!({
                "body": {
                    "body_kind": "blob",
                    "content_hash": preview.to_prefixed(),
                    "byte_len": 18,
                },
                "extra": {
                    "original_blob": {
                        "content_hash": exact.to_prefixed(),
                        "byte_len": 24,
                    },
                    "extracted_markdown_blob": {
                        "content_hash": binary.to_prefixed(),
                        "byte_len": 25,
                    },
                },
            }),
        ));
        assert_eq!(
            graph_payload_children(&commonplace_record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(preview, ReachabilityKind::Terminal),
                ReachabilityChild::new(exact, ReachabilityKind::Terminal),
                ReachabilityChild::new(binary, ReachabilityKind::Terminal),
            ]
        );

        let evidence_record = GraphPayloadRecord::Node(NodeRecord::new(
            "page:evidence-typed",
            ["Page"],
            json!({
                "evidence_package_hash": preview.to_prefixed(),
                "evidence_member_hashes": [preview.to_prefixed(), exact.to_prefixed()],
            }),
        ));
        assert_eq!(
            graph_payload_children(&evidence_record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(preview, ReachabilityKind::Terminal),
                ReachabilityChild::new(exact, ReachabilityKind::Terminal),
            ]
        );

        let evidence_without_manifest = GraphPayloadRecord::Node(NodeRecord::new(
            "page:evidence",
            ["Page"],
            json!({"evidence_package_hash": preview.to_prefixed()}),
        ));
        assert_eq!(
            graph_payload_children(&evidence_without_manifest, &[])
                .unwrap_err()
                .code,
            "evidence_payload_manifest_required_for_gc"
        );

        let unrelated = GraphPayloadRecord::Node(NodeRecord::new(
            "metadata:raw-hash",
            ["Metadata"],
            json!({
                "content_hash": preview.to_prefixed(),
                "cold_object_hash": exact.to_prefixed(),
                "exact_object_ref": exact.to_prefixed(),
                "binary_object_hash": binary.to_prefixed(),
                "binary_projection_ref": binary.to_prefixed(),
            }),
        ));
        assert!(graph_payload_children(&unrelated, &[]).unwrap().is_empty());
    }

    #[test]
    fn chat_upload_reachability_retains_active_context_and_releases_removed_bytes() {
        let source = Hash::of(b"chat upload source");
        let context = Hash::of(b"chat upload context document");
        let extracted = Hash::of(b"chat upload extracted text");
        let active = GraphPayloadRecord::Node(NodeRecord::new(
            "chatupload_active",
            ["TheoremChatUpload"],
            json!({
                "state": "ready",
                "contentRef": source.to_prefixed(),
                "contextDocumentRef": context.to_prefixed(),
                "extractedContentRef": extracted.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&active, &[]).unwrap(),
            vec![
                ReachabilityChild::new(source, ReachabilityKind::Terminal),
                ReachabilityChild::new(context, ReachabilityKind::Terminal),
                ReachabilityChild::new(extracted, ReachabilityKind::Terminal),
            ]
        );

        let removed = GraphPayloadRecord::Node(NodeRecord::new(
            "chatupload_removed",
            ["TheoremChatUpload"],
            json!({
                "state": "removed",
                "contentRef": source.to_prefixed(),
                "contextDocumentRef": context.to_prefixed(),
                "extractedContentRef": extracted.to_prefixed(),
            }),
        ));
        assert!(graph_payload_children(&removed, &[]).unwrap().is_empty());

        // A received upload owns only its quarantined source bytes; the
        // context document does not exist before extraction.
        let received = GraphPayloadRecord::Node(NodeRecord::new(
            "chatupload_received",
            ["TheoremChatUpload"],
            json!({
                "state": "received",
                "contentRef": source.to_prefixed(),
                "contextDocumentRef": null,
            }),
        ));
        assert_eq!(
            graph_payload_children(&received, &[]).unwrap(),
            vec![ReachabilityChild::new(source, ReachabilityKind::Terminal)]
        );
        let indexing = GraphPayloadRecord::Node(NodeRecord::new(
            "chatupload_indexing",
            ["TheoremChatUpload"],
            json!({
                "state": "indexing",
                "contentRef": source.to_prefixed(),
                "contextDocumentRef": context.to_prefixed(),
            }),
        ));
        assert_eq!(graph_payload_children(&indexing, &[]).unwrap().len(), 2);
        for released in ["failed", "canceled"] {
            let record = GraphPayloadRecord::Node(NodeRecord::new(
                format!("chatupload_{released}"),
                ["TheoremChatUpload"],
                json!({
                    "state": released,
                    "contentRef": source.to_prefixed(),
                }),
            ));
            assert!(
                graph_payload_children(&record, &[]).unwrap().is_empty(),
                "{released} must release its quarantined bytes"
            );
        }

        let malformed = GraphPayloadRecord::Node(NodeRecord::new(
            "chatupload_malformed",
            ["TheoremChatUpload"],
            json!({
                "state": "ready",
                "contentRef": "sha256:not-canonical",
                "contextDocumentRef": context.to_prefixed(),
            }),
        ));
        assert_eq!(
            graph_payload_children(&malformed, &[]).unwrap_err().code,
            "chat_upload_payload_reference_invalid"
        );
    }

    #[test]
    fn tensor_block_reachability_is_label_positional_and_strict() {
        let manifest = Hash::of(b"tensor manifest");
        let payload = Hash::of(b"tensor payload");
        let record = GraphPayloadRecord::Node(NodeRecord::new(
            "value:tensor",
            ["Value", "TensorBlock"],
            json!({
                "manifest_content_id": manifest.to_prefixed(),
                "payload_content_id": payload.to_prefixed(),
                "unrelated": "sha256:not-an-address"
            }),
        ));
        assert_eq!(
            graph_payload_children(&record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(manifest, ReachabilityKind::Terminal),
                ReachabilityChild::new(payload, ReachabilityKind::Terminal),
            ]
        );

        let malformed = GraphPayloadRecord::Node(NodeRecord::new(
            "value:tensor-malformed",
            ["TensorBlockGeneration"],
            json!({"manifest_content_id": manifest.to_prefixed()}),
        ));
        assert_eq!(
            graph_payload_children(&malformed, &[]).unwrap_err().code,
            "tensor_payload_reference_invalid"
        );

        let unrelated = GraphPayloadRecord::Node(NodeRecord::new(
            "node:metadata",
            ["Metadata"],
            json!({
                "manifest_content_id": manifest.to_prefixed(),
                "payload_content_id": payload.to_prefixed()
            }),
        ));
        assert!(graph_payload_children(&unrelated, &[]).unwrap().is_empty());
    }

    #[test]
    fn file_manifest_reachability_is_positional_strict_and_typed() {
        let terminal_bytes = b"file manifest reachability terminal";
        let terminal = Hash::of(terminal_bytes);
        let manifest = FileManifest {
            total_length: u64::try_from(terminal_bytes.len()).unwrap(),
            whole_file_sha256: terminal,
            chunks: vec![FileChunkRef {
                address: terminal,
                length: u32::try_from(terminal_bytes.len()).unwrap(),
            }],
        };
        let manifest_bytes = manifest.to_bytes().unwrap();
        let manifest_address = Hash::of(&manifest_bytes);
        let reference = FileManifestRef::new(manifest_address);
        let previous_manifest = Hash::of(b"previous file manifest");
        let record = GraphPayloadRecord::Node(NodeRecord::new(
            "file:typed-manifest",
            ["File"],
            json!({
                "content_manifest": reference.to_value().unwrap(),
                "previous_content_manifests": [
                    FileManifestRef::new(previous_manifest).to_value().unwrap()
                ],
                "history_elided_count": 2,
                "metadata_hash": terminal,
            }),
        ));
        assert_eq!(
            graph_payload_children(&record, &[]).unwrap(),
            vec![
                ReachabilityChild::new(manifest_address, ReachabilityKind::FileManifest),
                ReachabilityChild::new(previous_manifest, ReachabilityKind::FileManifest),
            ]
        );
        assert_eq!(
            super::decode_children(
                &CoreReachabilityCodec,
                ReachabilityKind::FileManifest,
                manifest_address,
                &manifest_bytes,
                &[],
            )
            .unwrap(),
            vec![ReachabilityChild::new(terminal, ReachabilityKind::Terminal,)]
        );

        let malformed = GraphPayloadRecord::Node(NodeRecord::new(
            "code-file:malformed-manifest",
            ["CodeFile"],
            json!({
                "content_manifest": {
                    "content_id": manifest_address,
                    "body": "inline bodies are forbidden",
                }
            }),
        ));
        assert_eq!(
            graph_payload_children(&malformed, &[]).unwrap_err().code,
            "file_manifest_reference_invalid"
        );

        let overlong_history = GraphPayloadRecord::Node(NodeRecord::new(
            "file:overlong-history",
            ["File"],
            json!({
                "content_manifest": reference.to_value().unwrap(),
                "previous_content_manifests": (0..=MAX_PREVIOUS_FILE_MANIFESTS)
                    .map(|index| FileManifestRef::new(Hash::of(format!("old-{index}").as_bytes())).to_value().unwrap())
                    .collect::<Vec<_>>()
            }),
        ));
        assert_eq!(
            graph_payload_children(&overlong_history, &[])
                .unwrap_err()
                .code,
            "file_manifest_history_invalid"
        );

        let unrelated = GraphPayloadRecord::Node(NodeRecord::new(
            "metadata:file-shaped",
            ["Metadata"],
            json!({"content_manifest": reference.to_value().unwrap()}),
        ));
        assert!(graph_payload_children(&unrelated, &[]).unwrap().is_empty());
    }

    #[test]
    fn missing_descendant_refuses_before_any_deletion() {
        let store = MemChunkStore::new();
        let missing = Hash::of(b"missing leaf");
        let parent = canonical_parent_preimage(&[GraphTreeChild {
            hash: missing,
            first_key: "node/missing".to_string(),
            last_key: "node/missing".to_string(),
            entries_total: 1,
        }])
        .unwrap();
        let parent_address = write_exact(&store, &parent);
        let orphan_address = write_exact(&store, b"must survive refused compaction");
        publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), parent_address)]),
        );

        let error = compact_committed_reachable(&store, &RejectExternalCodec).unwrap_err();

        assert_eq!(error.code, "reachability_chunk_missing");
        assert!(store.has(orphan_address).unwrap());
    }

    #[test]
    fn unknown_catalog_name_and_type_conflict_fail_closed() {
        let unknown_store = MemChunkStore::new();
        let empty_leaf = canonical_leaf_preimage(&[]).unwrap();
        let leaf_address = write_exact(&unknown_store, &empty_leaf);
        let unknown_address = write_exact(&unknown_store, b"future subsystem root");
        let orphan_address = write_exact(&unknown_store, b"unknown-name orphan");
        publish_catalog(
            &unknown_store,
            BTreeMap::from([
                (ROOT_NAME_GRAPH.to_string(), leaf_address),
                ("future".to_string(), unknown_address),
            ]),
        );

        let error = compact_committed_reachable(&unknown_store, &RejectExternalCodec).unwrap_err();
        assert_eq!(error.code, "reachability_catalog_name_unsupported");
        assert!(unknown_store.has(orphan_address).unwrap());

        let conflict_store = MemChunkStore::new();
        let leaf_address = write_exact(&conflict_store, &empty_leaf);
        publish_catalog(
            &conflict_store,
            BTreeMap::from([
                (ROOT_NAME_DOCUMENTS.to_string(), leaf_address),
                (ROOT_NAME_GRAPH.to_string(), leaf_address),
            ]),
        );
        let error = walk_committed_reachable(&conflict_store, &RejectExternalCodec).unwrap_err();
        assert_eq!(error.code, "reachability_type_conflict");
    }

    #[test]
    fn cycle_and_configured_bounds_abort_the_walk() {
        let cycle_address = Hash::of(b"typed cycle fixture");
        let cycle_kind = super::ReachabilityKind::PackEnvelope;
        let path = Arc::new(super::PathNode {
            address: cycle_address,
            kind: cycle_kind,
            parent: None,
        });
        let parent = super::Pending {
            address: cycle_address,
            kind: cycle_kind,
            depth: 1,
            path,
            vector_designations: Arc::new(Vec::new()),
            context: super::designation_context(&[]).unwrap(),
        };
        let mut assigned = HashMap::from([(cycle_address, HashSet::from([cycle_kind]))]);
        let mut queue = VecDeque::new();
        let cycle = super::enqueue_child(
            ReachabilityChild::new(cycle_address, cycle_kind),
            &parent,
            ReachabilityLimits {
                max_visits: 10,
                max_depth: 256,
            },
            &mut assigned,
            &mut queue,
            &mut HashSet::new(),
            Arc::new(Vec::new()),
            super::designation_context(&[]).unwrap(),
        )
        .unwrap_err();
        assert_eq!(cycle.code, "reachability_cycle_detected");

        let store = MemChunkStore::new();
        let empty_leaf = canonical_leaf_preimage(&[]).unwrap();
        let graph_root = write_exact(&store, &empty_leaf);
        publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), graph_root)]),
        );

        let visit_limit = walk_committed_reachable_with_limits(
            &store,
            &RejectExternalCodec,
            ReachabilityLimits {
                max_visits: 1,
                max_depth: 256,
            },
        )
        .unwrap_err();
        assert_eq!(visit_limit.code, "reachability_visit_limit_exceeded");

        let orphan = write_exact(&store, b"operator-budget orphan");
        let compact_limit = compact_committed_reachable_with_limits(
            &store,
            &RejectExternalCodec,
            ReachabilityLimits::new(1, 256),
        )
        .unwrap_err();
        assert_eq!(compact_limit.code, "reachability_visit_limit_exceeded");
        assert!(
            store.has(orphan).unwrap(),
            "budget refusal must happen before physical compaction"
        );
    }

    #[derive(Clone)]
    struct CatalogRaceStore {
        inner: MemChunkStore,
        future_catalog_root: Hash,
        advanced: Arc<AtomicBool>,
    }

    impl CatalogRaceStore {
        fn new(inner: MemChunkStore, future_catalog_root: Hash) -> Self {
            Self {
                inner,
                future_catalog_root,
                advanced: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl ChunkStore for CatalogRaceStore {
        fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
            self.inner.get(addr)
        }

        fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
            self.inner.has(addr)
        }

        fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
            self.inner.missing(addrs)
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

    impl ChunkStoreCompaction for CatalogRaceStore {
        fn compact_reachable(
            &self,
            expected_catalog_root: Option<Hash>,
            reachable: &[Hash],
        ) -> GraphStoreResult<CompactionReport> {
            if !self.advanced.swap(true, Ordering::SeqCst) {
                self.inner
                    .commit(self.future_catalog_root, global_gc_meta(2))?;
            }
            self.inner
                .compact_reachable(expected_catalog_root, reachable)
        }
    }

    fn assert_reachability_store(_store: &dyn ReachabilityStore) {}

    #[test]
    fn catalog_advance_to_old_terminal_addresses_refuses_without_deletion() {
        let inner = MemChunkStore::new();

        // The future graph is valid and has a descendant that the old walk
        // must not decode because it reaches this address as a terminal.
        let future_descendant = write_exact(&inner, b"future graph descendant");
        let future_graph_bytes = canonical_leaf_preimage(&[GraphTreeEntry {
            key: "node/future-descendant".to_string(),
            kind: GraphObjectKind::Node,
            object_hash: future_descendant,
            tombstone: false,
        }])
        .unwrap();
        let future_graph_root = write_exact(&inner, &future_graph_bytes);
        let future_catalog = RootCatalog::new(BTreeMap::from([(
            ROOT_NAME_GRAPH.to_string(),
            future_graph_root,
        )]))
        .unwrap();
        let future_catalog_bytes = future_catalog.to_bytes().unwrap();
        let future_catalog_root = write_exact(&inner, &future_catalog_bytes);

        // Both future roots are already reachable under catalog A, but only
        // as terminal object hashes. The old membership-based guard therefore
        // admitted catalog B while omitting its descendant.
        let old_graph_bytes = canonical_leaf_preimage(&[
            GraphTreeEntry {
                key: "node/future-catalog-terminal".to_string(),
                kind: GraphObjectKind::Node,
                object_hash: future_catalog_root,
                tombstone: false,
            },
            GraphTreeEntry {
                key: "node/future-graph-terminal".to_string(),
                kind: GraphObjectKind::Node,
                object_hash: future_graph_root,
                tombstone: false,
            },
        ])
        .unwrap();
        let old_graph_root = write_exact(&inner, &old_graph_bytes);
        let orphan = write_exact(&inner, b"race orphan");
        let old_catalog_root = publish_catalog(
            &inner,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), old_graph_root)]),
        );

        let old_walk = walk_committed_reachable(&inner, &RejectExternalCodec).unwrap();
        assert_eq!(old_walk.catalog_root, old_catalog_root);
        assert!(old_walk.reachable.contains(&future_catalog_root));
        assert!(old_walk.reachable.contains(&future_graph_root));
        assert!(!old_walk.reachable.contains(&future_descendant));

        let all_chunks = [
            old_catalog_root,
            old_graph_root,
            future_catalog_root,
            future_graph_root,
            future_descendant,
            orphan,
        ];
        let store = CatalogRaceStore::new(inner, future_catalog_root);
        assert_reachability_store(&store);

        let error = compact_committed_reachable(&store, &RejectExternalCodec).unwrap_err();

        assert_eq!(error.code, "chunk_store_compaction_root_changed");
        assert_eq!(store.read_root().unwrap(), Some(future_catalog_root));
        for address in all_chunks {
            assert!(
                store.has(address).unwrap(),
                "root mismatch must refuse before deleting {address}"
            );
        }
    }

    #[test]
    fn depth_limit_refuses_before_descendant_fetch() {
        let store = MemChunkStore::new();
        let (leaf, _) = leaf_with_terminal(&store, b"depth body");
        publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), leaf)]),
        );

        let error = walk_committed_reachable_with_limits(
            &store,
            &RejectExternalCodec,
            ReachabilityLimits {
                max_visits: 10,
                max_depth: 0,
            },
        )
        .unwrap_err();
        assert_eq!(error.code, "reachability_depth_limit_exceeded");
    }

    #[test]
    fn compaction_policy_skips_without_walking_below_growth_or_cadence_thresholds() {
        let uncommitted = MemChunkStore::new();
        let policy = CompactionPolicy {
            min_growth_bytes: 1_000,
            min_interval_seconds: 3_600,
        };

        let below_growth = compact_committed_reachable_if_due(
            &uncommitted,
            &RejectExternalCodec,
            policy,
            Some(CompactionPolicyState {
                last_compacted_file_bytes: 10_000,
                last_compacted_unix_seconds: 1_000,
            }),
            10_999,
            10_000,
        )
        .expect("below-growth policy must not touch the uncommitted store");
        assert_eq!(
            below_growth.decision,
            CompactionPolicyDecision::BelowGrowthThreshold {
                observed_growth_bytes: 999,
                required_growth_bytes: 1_000,
            }
        );
        assert!(below_growth.report.is_none());

        let cadence = compact_committed_reachable_if_due(
            &uncommitted,
            &RejectExternalCodec,
            policy,
            Some(CompactionPolicyState {
                last_compacted_file_bytes: 10_000,
                last_compacted_unix_seconds: 9_500,
            }),
            11_000,
            10_000,
        )
        .expect("cadence policy must not touch the uncommitted store");
        assert_eq!(
            cadence.decision,
            CompactionPolicyDecision::CadenceNotElapsed {
                observed_growth_bytes: 1_000,
                elapsed_seconds: 500,
                required_interval_seconds: 3_600,
            }
        );
        assert!(cadence.report.is_none());

        let error = compact_committed_reachable_if_due(
            &uncommitted,
            &RejectExternalCodec,
            policy,
            None,
            1_000,
            10_000,
        )
        .expect_err("eligible policy must attempt the typed walk");
        assert_eq!(error.code, "reachability_committed_root_missing");
    }

    #[test]
    fn compaction_policy_threshold_crossing_compacts_and_reports() {
        let store = MemChunkStore::new();
        let (leaf, terminal) = leaf_with_terminal(&store, b"policy live body");
        let orphan = write_exact(&store, b"policy orphan");
        let catalog = publish_catalog(
            &store,
            BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), leaf)]),
        );

        let outcome = compact_committed_reachable_if_due(
            &store,
            &RejectExternalCodec,
            CompactionPolicy {
                min_growth_bytes: 1,
                min_interval_seconds: 86_400,
            },
            None,
            1,
            10_000,
        )
        .unwrap();

        assert_eq!(
            outcome.decision,
            CompactionPolicyDecision::Eligible {
                observed_growth_bytes: 1,
                elapsed_seconds: None,
            }
        );
        let report = outcome.report.expect("eligible policy compacts");
        assert_eq!(report.reachability.catalog_root, catalog);
        assert!(report.compaction.chunks_removed >= 1);
        for address in [catalog, leaf, terminal] {
            assert!(store.has(address).unwrap());
        }
        assert!(!store.has(orphan).unwrap());
    }

    #[test]
    fn compaction_policy_rejects_zero_growth_threshold() {
        let error = evaluate_compaction_policy(
            CompactionPolicy {
                min_growth_bytes: 0,
                min_interval_seconds: 0,
            },
            None,
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(error.code, "compaction_policy_invalid_growth_threshold");
    }
}
