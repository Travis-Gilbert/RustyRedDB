//! Durable logical graph history in RedCore's existing chunk store.
//!
//! Complete compiler headers and their full-record roots publish with the
//! RedCore catalog barrier. Legacy roots are readable but never fabricated
//! into commit headers. Retention is all recorded revisions, including index
//! spines held by immutable reader cursors.

use crate::chunk_store::ChunkStore;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::mutation::GraphTransaction;
use crate::graph::record::{Direction, EdgeRecord, NodeRecord};
use crate::graph::vector_index::VectorDesignation;
use crate::hash::Hash;
use crate::object_store::UnifiedChunkStore;
use crate::ordered::OrderedDesignation;
use crate::versioned_graph::{
    build_prolly_tree_from_entries, build_prolly_tree_incremental, canonical_commit_preimage,
    graph_payload_at, graph_tree_entry, graph_tree_range, GraphCommit, GraphObjectKind,
    GraphPayloadRecord, GraphProllyTree, GraphTreeEntry,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub mod article;

const HISTORY_ROOT_DOMAIN: u8 = 16;
const HISTORY_ENTRY_DOMAIN: u8 = 17;
const HISTORY_LEGACY_DOMAIN: u8 = 18;
const HISTORY_VERSION: u8 = 1;
const MAX_HISTORY_RECORD_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_HISTORY_PAGE: usize = 512;
pub const MAX_HISTORY_SCAN: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphCommitMetadata {
    pub author: String,
    pub message: String,
    pub timestamp_unix_ms: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphCommitReceipt {
    pub transaction: GraphTransaction,
    pub commit: GraphCommit,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum GraphHistoryCoverage {
    Complete,
    Since {
        first_transaction: u64,
        prior_history_unavailable: bool,
    },
    Unavailable,
}

/// Exact physical and logical state, including a legitimate header-less
/// legacy head. Designations belong to this revision's GC interpretation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphRevisionRoots {
    pub transaction: u64,
    pub graph_version: u64,
    pub logical_commit: Hash,
    pub logical_tree: GraphProllyTree,
    pub payload_tree: GraphProllyTree,
    pub historical_node_tree: Option<GraphProllyTree>,
    pub historical_node_history_complete: bool,
    pub vector_designations: Vec<VectorDesignation>,
    pub ordered_designations: Vec<OrderedDesignation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphHistoryEntry {
    pub commit: GraphCommit,
    pub roots: GraphRevisionRoots,
    pub changed_object_keys: Vec<String>,
}

/// Exact notification material prepared from the same accepted roots before
/// publication. It is delivered only after their catalog barrier succeeds.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphCommitPublication {
    pub commit: GraphCommit,
    pub transaction: u64,
    pub changed_object_keys: Vec<String>,
    /// Both prior and current endpoints, including deleted and rewired edges.
    pub changed_edge_endpoints: Vec<String>,
}

impl GraphCommitPublication {
    pub(crate) fn prepare(
        store: &dyn ChunkStore,
        entry: &GraphHistoryEntry,
        prior: Option<&GraphRevisionRoots>,
    ) -> GraphStoreResult<Self> {
        entry.validate()?;
        let mut endpoints = BTreeSet::new();
        for key in &entry.changed_object_keys {
            if !key.starts_with("edge/") {
                continue;
            }
            for roots in prior.into_iter().chain(std::iter::once(&entry.roots)) {
                if let Some(record) = graph_payload_at(
                    store,
                    &roots.logical_tree,
                    &roots.payload_tree,
                    roots.graph_version,
                    key,
                )? {
                    let GraphPayloadRecord::Edge(edge) = record else {
                        return Err(invalid("changed edge notification resolved to a node"));
                    };
                    endpoints.insert(edge.from_id);
                    endpoints.insert(edge.to_id);
                }
            }
        }
        Ok(Self {
            commit: entry.commit.clone(),
            transaction: entry.roots.transaction,
            changed_object_keys: entry.changed_object_keys.clone(),
            changed_edge_endpoints: endpoints.into_iter().collect(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphHistoryRoot {
    pub by_commit: GraphProllyTree,
    pub by_transaction: GraphProllyTree,
    /// Old pinned cursors retain their original immutable index spines.
    pub retained_indexes: GraphProllyTree,
    pub head_commit: Hash,
    pub head_transaction: u64,
    pub coverage: GraphHistoryCoverage,
    pub legacy_boundary: Option<Hash>,
}

fn invalid(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("graph_history_integrity", message)
}

fn encode<T: Serialize>(domain: u8, value: &T) -> GraphStoreResult<Vec<u8>> {
    let body = serde_json::to_vec(value).map_err(|error| invalid(error.to_string()))?;
    if body.len() > MAX_HISTORY_RECORD_BYTES {
        return Err(invalid("history record exceeds byte limit"));
    }
    let mut bytes = vec![domain, HISTORY_VERSION];
    bytes.extend(body);
    Ok(bytes)
}

fn decode<T: for<'de> Deserialize<'de> + Serialize>(
    domain: u8,
    address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<T> {
    if bytes.len() > MAX_HISTORY_RECORD_BYTES + 2
        || bytes.get(..2) != Some(&[domain, HISTORY_VERSION])
    {
        return Err(invalid("history domain, version, or byte bound is invalid"));
    }
    if Hash::of(bytes) != address {
        return Err(invalid("history content address mismatch"));
    }
    let value: T =
        serde_json::from_slice(&bytes[2..]).map_err(|error| invalid(error.to_string()))?;
    if encode(domain, &value)? != bytes {
        return Err(invalid("history record is not canonical"));
    }
    Ok(value)
}

fn load(store: &dyn ChunkStore, address: Hash) -> GraphStoreResult<Arc<[u8]>> {
    store
        .get(address)?
        .ok_or_else(|| invalid(format!("retained history chunk {address} is missing")))
}

fn validate_roots(roots: &GraphRevisionRoots) -> GraphStoreResult<()> {
    if roots.logical_tree.entries_total != roots.payload_tree.entries_total {
        return Err(invalid("history logical and payload counts differ"));
    }
    if !roots.vector_designations.windows(2).all(|p| {
        (&p[0].label, &p[0].property, p[0].dimension)
            < (&p[1].label, &p[1].property, p[1].dimension)
    }) || !roots
        .ordered_designations
        .windows(2)
        .all(|p| (&p[0].label, &p[0].property) < (&p[1].label, &p[1].property))
    {
        return Err(invalid("history designations must be sorted and unique"));
    }
    Ok(())
}

impl GraphHistoryEntry {
    pub fn validate(&self) -> GraphStoreResult<()> {
        validate_roots(&self.roots)?;
        if self.commit.commit_hash != self.roots.logical_commit
            || self.commit.tree_hash != self.roots.logical_tree.root
            || self.commit.graph_version != self.roots.graph_version
            || Hash::of(&canonical_commit_preimage(&self.commit)?) != self.commit.commit_hash
        {
            return Err(invalid(
                "full history header does not match its canonical commitment and roots",
            ));
        }
        if !self.changed_object_keys.windows(2).all(|p| p[0] < p[1])
            || self.changed_object_keys.iter().any(|key| {
                !(key.starts_with("node/") || key.starts_with("edge/")) || key.ends_with('/')
            })
        {
            return Err(invalid(
                "changed keys must be sorted unique graph-object paths",
            ));
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        self.validate()?;
        encode(HISTORY_ENTRY_DOMAIN, self)
    }
    pub fn from_bytes(address: Hash, bytes: &[u8]) -> GraphStoreResult<Self> {
        let value: Self = decode(HISTORY_ENTRY_DOMAIN, address, bytes)?;
        value.validate()?;
        Ok(value)
    }
}

impl GraphHistoryRoot {
    pub fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        if self.by_commit.entries_total != self.by_transaction.entries_total
            || self.by_commit.entries_total == 0
        {
            return Err(invalid("history index counts differ or are empty"));
        }
        if matches!(self.coverage, GraphHistoryCoverage::Unavailable) {
            return Err(invalid("recorded history must declare coverage"));
        }
        encode(HISTORY_ROOT_DOMAIN, self)
    }
    pub fn from_bytes(address: Hash, bytes: &[u8]) -> GraphStoreResult<Self> {
        let value: Self = decode(HISTORY_ROOT_DOMAIN, address, bytes)?;
        value.to_bytes()?;
        Ok(value)
    }
}

pub(crate) fn decode_legacy_roots(
    address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<GraphRevisionRoots> {
    let roots = decode(HISTORY_LEGACY_DOMAIN, address, bytes)?;
    validate_roots(&roots)?;
    Ok(roots)
}

pub(crate) fn load_history_root(
    store: &dyn ChunkStore,
    address: Hash,
) -> GraphStoreResult<GraphHistoryRoot> {
    GraphHistoryRoot::from_bytes(address, &load(store, address)?)
}

fn index_entry(key: String, object_hash: Hash) -> GraphTreeEntry {
    GraphTreeEntry {
        key,
        kind: GraphObjectKind::Node,
        object_hash,
        tombstone: false,
    }
}
fn transaction_key(transaction: u64) -> String {
    format!("{transaction:020}")
}

/// Stage one actual accepted commit and its indexes in the caller's existing
/// publication window. This function never changes the committed catalog.
pub(crate) fn append_history(
    store: &dyn ChunkStore,
    prior: Option<&GraphHistoryRoot>,
    legacy: Option<&GraphRevisionRoots>,
    entry: &GraphHistoryEntry,
) -> GraphStoreResult<(Hash, GraphHistoryRoot)> {
    let bytes = entry.to_bytes()?;
    let address = Hash::of(&bytes);
    store.write(&[(address, &bytes)])?;
    let commit_key = entry.commit.commit_hash.to_string();
    let txn_key = transaction_key(entry.roots.transaction);
    let by_commit_entry = index_entry(commit_key.clone(), address);
    let by_txn_entry = index_entry(txn_key.clone(), address);
    let root = if let Some(prior) = prior {
        if entry.roots.transaction <= prior.head_transaction
            || entry.commit.parent_commits.first() != Some(&prior.head_commit)
        {
            return Err(invalid(
                "history transaction or parent does not continue the captured head",
            ));
        }
        if graph_tree_entry(store, &prior.by_commit, &commit_key)?.is_some() {
            return Err(invalid("logical commit already appears in history"));
        }
        let archived = BTreeMap::from([
            (
                format!("{txn_key}/commit"),
                index_entry(format!("{txn_key}/commit"), prior.by_commit.root),
            ),
            (
                format!("{txn_key}/transaction"),
                index_entry(format!("{txn_key}/transaction"), prior.by_transaction.root),
            ),
        ]);
        GraphHistoryRoot {
            by_commit: build_prolly_tree_incremental(
                store,
                &prior.by_commit,
                &BTreeMap::from([(commit_key, by_commit_entry)]),
            )?
            .tree,
            by_transaction: build_prolly_tree_incremental(
                store,
                &prior.by_transaction,
                &BTreeMap::from([(txn_key, by_txn_entry)]),
            )?
            .tree,
            retained_indexes: build_prolly_tree_incremental(
                store,
                &prior.retained_indexes,
                &archived,
            )?
            .tree,
            head_commit: entry.commit.commit_hash,
            head_transaction: entry.roots.transaction,
            coverage: prior.coverage.clone(),
            legacy_boundary: prior.legacy_boundary,
        }
    } else {
        let legacy_boundary = legacy
            .map(|roots| {
                validate_roots(roots)?;
                let bytes = encode(HISTORY_LEGACY_DOMAIN, roots)?;
                let address = Hash::of(&bytes);
                store.write(&[(address, &bytes)])?;
                Ok::<_, GraphStoreError>(address)
            })
            .transpose()?;
        GraphHistoryRoot {
            by_commit: build_prolly_tree_from_entries(store, vec![by_commit_entry])?,
            by_transaction: build_prolly_tree_from_entries(store, vec![by_txn_entry])?,
            retained_indexes: build_prolly_tree_from_entries(store, vec![])?,
            head_commit: entry.commit.commit_hash,
            head_transaction: entry.roots.transaction,
            coverage: if legacy.is_some() {
                GraphHistoryCoverage::Since {
                    first_transaction: entry.roots.transaction,
                    prior_history_unavailable: true,
                }
            } else {
                GraphHistoryCoverage::Complete
            },
            legacy_boundary,
        }
    };
    let bytes = root.to_bytes()?;
    let address = Hash::of(&bytes);
    store.write(&[(address, &bytes)])?;
    Ok((address, root))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphHistoryCursor {
    pub head: Hash,
    pub before_transaction: u64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct GraphHistoryPage {
    pub entries: Vec<GraphHistoryEntry>,
    pub next: Option<GraphHistoryCursor>,
    pub coverage: GraphHistoryCoverage,
}

#[derive(Clone)]
pub struct PinnedGraphHistory {
    store: Arc<dyn UnifiedChunkStore>,
    root: Option<GraphHistoryRoot>,
    head: GraphRevisionRoots,
}
impl std::fmt::Debug for PinnedGraphHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedGraphHistory")
            .field("head", &self.head)
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}
impl PinnedGraphHistory {
    pub(crate) fn new(
        store: Arc<dyn UnifiedChunkStore>,
        history: Option<Hash>,
        head: GraphRevisionRoots,
    ) -> GraphStoreResult<Self> {
        validate_roots(&head)?;
        let root = history
            .map(|address| load_history_root(store.as_chunk_store(), address))
            .transpose()?;
        let pinned = Self { store, root, head };
        if let Some(root) = &pinned.root {
            pinned.entry(root.head_commit)?;
        }
        Ok(pinned)
    }
    pub fn coverage(&self) -> GraphHistoryCoverage {
        self.root
            .as_ref()
            .map_or(GraphHistoryCoverage::Unavailable, |root| {
                root.coverage.clone()
            })
    }
    pub fn head_roots(&self) -> &GraphRevisionRoots {
        &self.head
    }
    pub fn entry(&self, commit: Hash) -> GraphStoreResult<GraphHistoryEntry> {
        let root = self.root.as_ref().ok_or_else(|| {
            GraphStoreError::new(
                "graph_history_unavailable",
                "this store predates retained logical headers",
            )
        })?;
        let index = graph_tree_entry(
            self.store.as_chunk_store(),
            &root.by_commit,
            &commit.to_string(),
        )?
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_history_unavailable",
                "commit is outside the pinned retained history",
            )
        })?;
        let entry = self.entry_at(&index)?;
        if entry.roots.transaction > root.head_transaction {
            return Err(GraphStoreError::new(
                "graph_history_unavailable",
                "commit is newer than the pinned head",
            ));
        }
        if entry.commit.commit_hash != commit {
            return Err(invalid("commit index points at a different header"));
        }
        Ok(entry)
    }
    fn entry_at(&self, index: &GraphTreeEntry) -> GraphStoreResult<GraphHistoryEntry> {
        if index.tombstone || index.kind != GraphObjectKind::Node {
            return Err(invalid("invalid history index tag"));
        }
        GraphHistoryEntry::from_bytes(
            index.object_hash,
            &load(self.store.as_chunk_store(), index.object_hash)?,
        )
    }
    pub fn page(
        &self,
        cursor: Option<GraphHistoryCursor>,
        limit: usize,
    ) -> GraphStoreResult<GraphHistoryPage> {
        if limit == 0 || limit > MAX_HISTORY_PAGE {
            return Err(GraphStoreError::new(
                "graph_history_page_limit",
                "history page limit must be 1..=512",
            ));
        }
        let Some(root) = &self.root else {
            return Ok(GraphHistoryPage {
                entries: vec![],
                next: None,
                coverage: self.coverage(),
            });
        };
        if cursor
            .as_ref()
            .is_some_and(|cursor| cursor.head != root.head_commit)
        {
            return Err(GraphStoreError::new(
                "graph_history_cursor_mismatch",
                "cursor belongs to a different pinned head",
            ));
        }
        let cutoff = root.head_transaction.checked_add(1);
        if cursor
            .as_ref()
            .is_some_and(|cursor| cutoff.is_some_and(|end| cursor.before_transaction > end))
        {
            return Err(GraphStoreError::new(
                "graph_history_cursor_mismatch",
                "cursor exceeds pinned history cutoff",
            ));
        }
        let before = cursor
            .as_ref()
            .map(|cursor| cursor.before_transaction)
            .or(cutoff)
            .map(transaction_key);
        let indexes = graph_tree_range(
            self.store.as_chunk_store(),
            &root.by_transaction,
            None,
            before.as_deref(),
            limit + 1,
            true,
        )?;
        let has_more = indexes.len() > limit;
        let entries = indexes
            .iter()
            .take(limit)
            .map(|index| {
                let entry = self.entry_at(index)?;
                if index.key != transaction_key(entry.roots.transaction) {
                    return Err(invalid("transaction index/header mismatch"));
                }
                Ok(entry)
            })
            .collect::<GraphStoreResult<Vec<_>>>()?;
        let next = if has_more {
            entries.last().map(|entry| GraphHistoryCursor {
                head: root.head_commit,
                before_transaction: entry.roots.transaction,
            })
        } else {
            None
        };
        Ok(GraphHistoryPage {
            entries,
            next,
            coverage: self.coverage(),
        })
    }
    /// Read the next retained commits in transaction order. A durable worker
    /// advances its cursor from the final returned entry. Missing pre-upgrade
    /// history is an explicit refusal, never an empty successful catchup.
    pub fn entries_after(
        &self,
        after_transaction: u64,
        limit: usize,
    ) -> GraphStoreResult<Vec<GraphHistoryEntry>> {
        if limit == 0 || limit > MAX_HISTORY_PAGE {
            return Err(GraphStoreError::new(
                "graph_history_page_limit",
                "page limit must be 1..=512",
            ));
        }
        let root = self.root.as_ref().ok_or_else(|| {
            GraphStoreError::new(
                "graph_history_unavailable",
                "this store predates retained logical headers",
            )
        })?;
        if after_transaction > self.head.transaction {
            return Err(GraphStoreError::new(
                "graph_history_cursor_mismatch",
                "cursor exceeds pinned history cutoff",
            ));
        }
        if let GraphHistoryCoverage::Since {
            first_transaction,
            prior_history_unavailable: true,
        } = &root.coverage
        {
            if after_transaction < first_transaction.saturating_sub(1) {
                return Err(GraphStoreError::new(
                    "graph_history_unavailable",
                    "requested catchup crosses unavailable pre-upgrade history",
                ));
            }
        }
        if after_transaction == self.head.transaction {
            return Ok(Vec::new());
        }
        let start = transaction_key(after_transaction + 1);
        let before = self.head.transaction.checked_add(1).map(transaction_key);
        graph_tree_range(
            self.store.as_chunk_store(),
            &root.by_transaction,
            Some(&start),
            before.as_deref(),
            limit,
            false,
        )?
        .iter()
        .map(|index| {
            let entry = self.entry_at(index)?;
            if index.key != transaction_key(entry.roots.transaction)
                || entry.roots.transaction <= after_transaction
                || entry.roots.transaction > self.head.transaction
            {
                return Err(invalid("forward transaction index/header mismatch"));
            }
            Ok(entry)
        })
        .collect()
    }
    pub fn revision(&self, commit: Hash) -> GraphStoreResult<PinnedGraphRevision> {
        let entry = self.entry(commit)?;
        Ok(PinnedGraphRevision {
            store: Arc::clone(&self.store),
            roots: entry.roots,
            commit: Some(entry.commit),
        })
    }
    pub fn head_revision(&self) -> GraphStoreResult<PinnedGraphRevision> {
        let commit = if self.root.is_some() {
            Some(self.entry(self.head.logical_commit)?.commit)
        } else {
            None
        };
        Ok(PinnedGraphRevision {
            store: Arc::clone(&self.store),
            roots: self.head.clone(),
            commit,
        })
    }
    pub fn legacy_revision(&self) -> GraphStoreResult<Option<PinnedGraphRevision>> {
        let Some(address) = self.root.as_ref().and_then(|root| root.legacy_boundary) else {
            return Ok(None);
        };
        let roots = decode_legacy_roots(address, &load(self.store.as_chunk_store(), address)?)?;
        Ok(Some(PinnedGraphRevision {
            store: Arc::clone(&self.store),
            roots,
            commit: None,
        }))
    }
    /// Restrict the already captured history indexes to an actual older head.
    pub fn view_at(&self, commit: Hash) -> GraphStoreResult<Self> {
        let entry = self.entry(commit)?;
        let mut root = self.root.clone().expect("entry requires retained history");
        root.head_commit = commit;
        root.head_transaction = entry.roots.transaction;
        Ok(Self {
            store: Arc::clone(&self.store),
            root: Some(root),
            head: entry.roots,
        })
    }

    pub fn before_commit(&self, commit: Hash) -> GraphStoreResult<Option<Self>> {
        let entry = self.entry(commit)?;
        let Some(parent) = entry.commit.parent_commits.first().copied() else {
            return Ok(None);
        };
        if let Some(legacy) = self.legacy_revision()? {
            if legacy.roots.logical_commit == parent {
                return Ok(Some(Self {
                    store: Arc::clone(&self.store),
                    root: None,
                    head: legacy.roots,
                }));
            }
        }
        self.view_at(parent).map(Some)
    }

    /// Resolve deleted/rewired adjacency from actual before and after records,
    /// so a removed edge is still a change to its former article endpoint.
    pub fn touches(
        &self,
        entry: &GraphHistoryEntry,
        keys: &BTreeSet<String>,
    ) -> GraphStoreResult<bool> {
        if entry
            .changed_object_keys
            .iter()
            .any(|key| keys.contains(key))
        {
            return Ok(true);
        }
        let changed_edges = entry
            .changed_object_keys
            .iter()
            .filter_map(|key| key.strip_prefix("edge/"))
            .collect::<Vec<_>>();
        if changed_edges.is_empty() {
            return Ok(false);
        }
        let after = self.revision(entry.commit.commit_hash)?;
        let before = self
            .before_commit(entry.commit.commit_hash)?
            .map(|pin| pin.head_revision())
            .transpose()?;
        for id in changed_edges {
            let prior = before
                .as_ref()
                .map(|revision| revision.get_edge(id))
                .transpose()?
                .flatten();
            for edge in [prior, after.get_edge(id)?].into_iter().flatten() {
                if keys.contains(&format!("node/{}", edge.from_id))
                    || keys.contains(&format!("node/{}", edge.to_id))
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub fn newest_touching(
        &self,
        keys: &BTreeSet<String>,
    ) -> GraphStoreResult<Option<GraphHistoryEntry>> {
        let mut cursor = None;
        let mut scanned = 0;
        loop {
            let page = self.page(cursor, MAX_HISTORY_PAGE)?;
            for entry in page.entries {
                scanned += 1;
                if scanned > MAX_HISTORY_SCAN {
                    return Err(GraphStoreError::new(
                        "graph_history_scan_limit",
                        "history dependency scan exceeded its work budget",
                    ));
                }
                if self.touches(&entry, keys)? {
                    return Ok(Some(entry));
                }
            }
            cursor = page.next;
            if cursor.is_none() {
                return Ok(None);
            }
        }
    }
}

#[derive(Clone)]
pub struct PinnedGraphRevision {
    store: Arc<dyn UnifiedChunkStore>,
    pub roots: GraphRevisionRoots,
    commit: Option<GraphCommit>,
}
impl std::fmt::Debug for PinnedGraphRevision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedGraphRevision")
            .field("roots", &self.roots)
            .field("commit", &self.commit)
            .finish_non_exhaustive()
    }
}
impl PinnedGraphRevision {
    pub fn commit(&self) -> Option<&GraphCommit> {
        self.commit.as_ref()
    }
    /// None is explicit for legacy roots; no timestamp is guessed.
    pub fn timestamp_unix_ms(&self) -> Option<u128> {
        self.commit.as_ref().map(|commit| commit.timestamp_unix_ms)
    }
    pub fn get_node(&self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        match self.record(&format!("node/{id}"))? {
            Some(GraphPayloadRecord::Node(node)) => Ok(Some(node)),
            None => Ok(None),
            _ => Err(invalid("node key returned edge payload")),
        }
    }
    pub fn get_edge(&self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        match self.record(&format!("edge/{id}"))? {
            Some(GraphPayloadRecord::Edge(edge)) => Ok(Some(edge)),
            None => Ok(None),
            _ => Err(invalid("edge key returned node payload")),
        }
    }
    fn record(&self, key: &str) -> GraphStoreResult<Option<GraphPayloadRecord>> {
        graph_payload_at(
            self.store.as_chunk_store(),
            &self.roots.logical_tree,
            &self.roots.payload_tree,
            self.roots.graph_version,
            key,
        )
    }
    pub fn records_page(
        &self,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> GraphStoreResult<Vec<GraphPayloadRecord>> {
        if prefix != "node/" && prefix != "edge/" {
            return Err(invalid(
                "record scan must name the node/ or edge/ key namespace",
            ));
        }
        self.records_prefix_page(prefix, after, limit)
    }

    /// Read at most `limit` records under one key prefix from this retained
    /// ordered index. Entries outside the prefix are never hydrated.
    pub fn records_prefix_page(
        &self,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> GraphStoreResult<Vec<GraphPayloadRecord>> {
        let before = if prefix.starts_with("node/") {
            "node0"
        } else if prefix.starts_with("edge/") {
            "edge0"
        } else {
            return Err(invalid(
                "record prefix is outside the node/ and edge/ namespaces",
            ));
        };
        if after.is_some_and(|key| !key.starts_with(prefix)) {
            return Err(invalid("record cursor is outside its namespace"));
        }
        let start = after.map_or_else(|| prefix.to_owned(), |key| format!("{key}\0"));
        graph_tree_range(
            self.store.as_chunk_store(),
            &self.roots.logical_tree,
            Some(&start),
            Some(before),
            limit,
            false,
        )?
        .into_iter()
        .take_while(|entry| entry.key.starts_with(prefix))
        .map(|entry| {
            self.record(&entry.key)?
                .ok_or_else(|| invalid("pinned record vanished"))
        })
        .collect()
    }
    /// Complete incident adjacency, including dangling endpoints. Records are
    /// paged from storage; only the requested node's matching edges accumulate.
    pub fn incident_edges(
        &self,
        id: &str,
        direction: Direction,
    ) -> GraphStoreResult<Vec<EdgeRecord>> {
        let mut after = None;
        let mut edges = Vec::new();
        let mut scanned = 0;
        loop {
            let page = self.records_page("edge/", after.as_deref(), 256)?;
            if page.is_empty() {
                break;
            }
            for record in page {
                after = Some(record.key());
                scanned += 1;
                if scanned > MAX_HISTORY_SCAN {
                    return Err(GraphStoreError::new(
                        "graph_history_scan_limit",
                        "incident-edge scan exceeded its work budget",
                    ));
                }
                let GraphPayloadRecord::Edge(edge) = record else {
                    return Err(invalid("edge page contains a node"));
                };
                let matches = match direction {
                    Direction::Out => edge.from_id == id,
                    Direction::In => edge.to_id == id,
                };
                if matches {
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }
}
