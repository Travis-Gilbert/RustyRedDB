//! The cold index (storage spine, cut 6): which node and edge ids live in which
//! tier, and at which object hash.
//!
//! This is the catalog's load-bearing structure for tiering. It makes
//! rehydration a KEYED lookup (id -> object hash -> object), never a scan of the
//! versioned-graph repository -- otherwise eviction would just trade a RAM scan
//! for a disk scan. It is deliberately a catalog, not a re-encoding of the
//! graph: it stores addresses and tier state, not node/edge payloads (those live
//! in the [`ColdObjectStore`](crate::object_store::ColdObjectStore)).
//!
//! The trait is synchronous so the synchronous eviction path
//! (`rustyred-thg-memory::decay`) can use it directly. The in-RAM and disk impls
//! here cover the hot path and the durability acceptance test; the
//! `rustyred-thg-catalog` crate provides the sqlx/Postgres impl named by the
//! spec (tenants, projects, billing, auth, and this same cold index).

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use serde::{Deserialize, Serialize};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::object_store::safe_filename;
use crate::ordered::{OrderedIndex, OrderedMode};
use crate::query_receipt::{ConsumerOverlapTrace, PinnedReadSet};
use crate::state::stable_hash;
use theorem_oplog::WriteSet;

/// Which tier a record currently lives in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ColdTierKind {
    /// A single node evicted from RAM; its content object is durable in the
    /// cold object store, addressed by `object_hash`.
    Cold,
    /// A member of a whole scope (repo_id / tenant) parked as a
    /// `CompiledGraphPack`; rehydrated as a unit on first access.
    Warm,
}

/// One cold-tier residency record. Keyed by `id` (the rehydration key).
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ColdIndexEntry {
    pub id: String,
    pub scope: String,
    pub tier: ColdTierKind,
    /// Address in the cold object store (the content hash of the committed
    /// object). The `id -> object_hash -> object` chain is the keyed lookup.
    pub object_hash: String,
    /// Set for warm-scope members: the parked pack's commit hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_hash: Option<String>,
}

impl ColdIndexEntry {
    /// A cold (single-node) record.
    pub fn cold(
        id: impl Into<String>,
        scope: impl Into<String>,
        object_hash: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            scope: scope.into(),
            tier: ColdTierKind::Cold,
            object_hash: object_hash.into(),
            commit_hash: None,
        }
    }
}

/// A parked warm scope's catalog record: the head commit plus the node/edge ids
/// belonging to the scope, so unpark can rehydrate the whole subgraph.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ColdScopeEntry {
    pub scope: String,
    pub commit_hash: String,
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
    /// True while the scope is parked (absent from the operating store).
    pub parked: bool,
}

/// Query-driven hotness for fold materializations.
///
/// Each materialization supplies the same canonical write footprint used by
/// OCC and invalidation. A count advances only when the pinned query reads
/// overlap that footprint.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FoldHotness {
    counts: BTreeMap<String, u64>,
}

impl FoldHotness {
    pub fn observe(
        &mut self,
        materialization: impl Into<String>,
        reads: &PinnedReadSet,
        footprint: &WriteSet,
    ) -> ConsumerOverlapTrace {
        let materialization = materialization.into();
        let trace = reads.hotness_trace(footprint);
        if trace.overlaps {
            let count = self.counts.entry(materialization).or_default();
            *count = count.saturating_add(1);
        }
        trace
    }

    pub fn count(&self, materialization: &str) -> u64 {
        self.counts.get(materialization).copied().unwrap_or(0)
    }

    pub fn counts(&self) -> &BTreeMap<String, u64> {
        &self.counts
    }

    pub fn forget(&mut self, materialization: &str) -> bool {
        self.counts.remove(materialization).is_some()
    }

    pub(crate) fn retained_identifier_capacity_bytes(&self) -> usize {
        self.counts.keys().map(String::capacity).sum()
    }
}

/// The cold index contract. Synchronous (the eviction path is sync). Interior
/// mutability (`&self`) so a single index can be shared behind an `Arc` by the
/// cold tier without `&mut` borrow juggling.
pub trait ColdIndex: Send + Sync {
    /// Record (or overwrite) a node's cold residency.
    fn record(&self, entry: ColdIndexEntry) -> GraphStoreResult<()>;
    /// Look up a node's cold residency by id. `Ok(None)` when not cold.
    fn lookup(&self, id: &str) -> GraphStoreResult<Option<ColdIndexEntry>>;
    /// Drop a node's cold record (it was rehydrated back into the operating store).
    fn remove(&self, id: &str) -> GraphStoreResult<()>;
    /// Record (or overwrite) a parked scope.
    fn record_scope(&self, entry: ColdScopeEntry) -> GraphStoreResult<()>;
    /// Look up a parked scope by name.
    fn scope(&self, scope: &str) -> GraphStoreResult<Option<ColdScopeEntry>>;
    /// Drop a scope record (it was unparked).
    fn remove_scope(&self, scope: &str) -> GraphStoreResult<()>;
    /// Compare-and-remap one standalone cold object's address after an
    /// authentic v1 object has been durably published and read back.
    fn remap_object_hash(
        &self,
        id: &str,
        expected_old: Hash,
        new_hash: Hash,
    ) -> GraphStoreResult<ColdIndexEntry> {
        let mut entry = self.lookup(id)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_remap_entry_missing",
                format!("cold object index entry {id} is absent"),
            )
        })?;
        if entry.tier != ColdTierKind::Cold {
            return Err(GraphStoreError::new(
                "cold_object_remap_tier_mismatch",
                format!("cold object index entry {id} is not in the cold tier"),
            ));
        }
        let expected = expected_old.to_prefixed();
        let new = new_hash.to_prefixed();
        if entry.object_hash != expected && entry.object_hash != new {
            return Err(GraphStoreError::new(
                "cold_object_remap_conflict",
                format!(
                    "cold object index entry {id} names {}, expected {expected}",
                    entry.object_hash
                ),
            ));
        }
        if entry.object_hash == expected {
            entry.object_hash = new.clone();
            self.record(entry.clone())?;
        }
        let verified = self.lookup(id)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_remap_verify_missing",
                format!("cold object index entry {id} disappeared during remap"),
            )
        })?;
        if verified.object_hash != new {
            return Err(GraphStoreError::new(
                "cold_object_remap_verify_failed",
                format!(
                    "cold object index entry {id} still names {}",
                    verified.object_hash
                ),
            ));
        }
        Ok(verified)
    }

    /// Compare-and-remap a parked scope and every authoritative warm member.
    /// All rows are validated before member-first publication begins. The
    /// authoritative scope pointer advances only after every member is new.
    fn remap_scope_commit(
        &self,
        scope: &str,
        expected_old: Hash,
        new_commit: Hash,
    ) -> GraphStoreResult<ColdScopeEntry> {
        let mut scope_entry = self.scope(scope)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_scope_remap_entry_missing",
                format!("cold scope index entry {scope} is absent"),
            )
        })?;
        let expected = expected_old.to_prefixed();
        let new = new_commit.to_prefixed();
        if scope_entry.commit_hash != expected && scope_entry.commit_hash != new {
            return Err(GraphStoreError::new(
                "cold_scope_remap_conflict",
                format!(
                    "cold scope {scope} names {}, expected {expected}",
                    scope_entry.commit_hash
                ),
            ));
        }

        let mut members = Vec::with_capacity(scope_entry.node_ids.len());
        for id in &scope_entry.node_ids {
            let entry = self.lookup(id)?.ok_or_else(|| {
                GraphStoreError::new(
                    "cold_scope_remap_member_missing",
                    format!("warm member {id} of scope {scope} is absent"),
                )
            })?;
            let member_commit_valid = entry
                .commit_hash
                .as_deref()
                .is_some_and(|hash| hash == expected || hash == new);
            if entry.tier != ColdTierKind::Warm || entry.scope != scope || !member_commit_valid {
                return Err(GraphStoreError::new(
                    "cold_scope_remap_member_conflict",
                    format!("warm member {id} does not name scope {scope} at {expected}"),
                ));
            }
            members.push(entry);
        }

        let mut members_published = 0usize;
        for mut member in members {
            if member.commit_hash.as_deref() == Some(expected.as_str()) {
                member.commit_hash = Some(new.clone());
                self.record(member)?;
                members_published = members_published.saturating_add(1);
                #[cfg(test)]
                if members_published == 1 {
                    self.after_first_scope_member_publication_for_test()?;
                }
            }
        }
        #[cfg(test)]
        self.after_all_scope_members_publication_for_test()?;
        if scope_entry.commit_hash == expected {
            scope_entry.commit_hash = new.clone();
            self.record_scope(scope_entry.clone())?;
        }

        let verified_scope = self.scope(scope)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_scope_remap_verify_missing",
                format!("cold scope index entry {scope} disappeared during remap"),
            )
        })?;
        if verified_scope.commit_hash != new {
            return Err(GraphStoreError::new(
                "cold_scope_remap_verify_failed",
                format!(
                    "cold scope {scope} still names {}",
                    verified_scope.commit_hash
                ),
            ));
        }
        for id in &verified_scope.node_ids {
            let member = self.lookup(id)?.ok_or_else(|| {
                GraphStoreError::new(
                    "cold_scope_remap_verify_missing",
                    format!("warm member {id} disappeared during scope remap"),
                )
            })?;
            if member.commit_hash.as_deref() != Some(new.as_str()) {
                return Err(GraphStoreError::new(
                    "cold_scope_remap_verify_failed",
                    format!("warm member {id} still names {:?}", member.commit_hash),
                ));
            }
        }
        Ok(verified_scope)
    }

    #[cfg(test)]
    fn after_first_scope_member_publication_for_test(&self) -> GraphStoreResult<()> {
        Ok(())
    }

    #[cfg(test)]
    fn after_all_scope_members_publication_for_test(&self) -> GraphStoreResult<()> {
        Ok(())
    }

    /// Number of cold node records (diagnostics).
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-RAM cold index. For tests and the in-process default -- it does NOT
/// survive a restart, so the durable acceptance path uses [`DiskColdIndex`] or
/// the Postgres catalog.
#[derive(Clone, Debug, Default)]
pub struct InMemoryColdIndex {
    entries: Arc<Mutex<BTreeMap<String, ColdIndexEntry>>>,
    scopes: Arc<Mutex<BTreeMap<String, ColdScopeEntry>>>,
}

impl InMemoryColdIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ColdIndex for InMemoryColdIndex {
    fn record(&self, entry: ColdIndexEntry) -> GraphStoreResult<()> {
        self.entries
            .lock()
            .map_err(poisoned)?
            .insert(entry.id.clone(), entry);
        Ok(())
    }

    fn lookup(&self, id: &str) -> GraphStoreResult<Option<ColdIndexEntry>> {
        Ok(self.entries.lock().map_err(poisoned)?.get(id).cloned())
    }

    fn remove(&self, id: &str) -> GraphStoreResult<()> {
        self.entries.lock().map_err(poisoned)?.remove(id);
        Ok(())
    }

    fn record_scope(&self, entry: ColdScopeEntry) -> GraphStoreResult<()> {
        self.scopes
            .lock()
            .map_err(poisoned)?
            .insert(entry.scope.clone(), entry);
        Ok(())
    }

    fn scope(&self, scope: &str) -> GraphStoreResult<Option<ColdScopeEntry>> {
        Ok(self.scopes.lock().map_err(poisoned)?.get(scope).cloned())
    }

    fn remove_scope(&self, scope: &str) -> GraphStoreResult<()> {
        self.scopes.lock().map_err(poisoned)?.remove(scope);
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.lock().map(|map| map.len()).unwrap_or(0)
    }
}

/// Native ordered-index-backed cold residency catalog. This is the in-process
/// replacement for the Postgres hot-path adapter: id lookups stay keyed in RAM,
/// while scope membership is maintained through [`OrderedIndex`] so promotion,
/// rehydration, and scope scans do not leave the process.
#[derive(Clone, Debug, Default)]
pub struct OrderedColdIndex {
    state: Arc<Mutex<OrderedColdIndexState>>,
}

#[derive(Clone, Debug, Default)]
struct OrderedColdIndexState {
    entries: BTreeMap<String, ColdIndexEntry>,
    scopes: BTreeMap<String, ColdScopeEntry>,
    scope_members: BTreeMap<String, OrderedIndex>,
    seq: u64,
}

impl OrderedColdIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ids_for_scope(&self, scope: &str, limit: usize) -> GraphStoreResult<Vec<String>> {
        let state = self.state.lock().map_err(poisoned)?;
        let Some(index) = state.scope_members.get(scope) else {
            return Ok(Vec::new());
        };
        let cap = if limit == 0 { usize::MAX } else { limit };
        index
            .entries()
            .into_iter()
            .take(cap)
            .map(|(member, _)| {
                String::from_utf8(member)
                    .map_err(|error| GraphStoreError::new("cold_index_member", error.to_string()))
            })
            .collect()
    }
}

impl ColdIndex for OrderedColdIndex {
    fn record(&self, entry: ColdIndexEntry) -> GraphStoreResult<()> {
        let mut state = self.state.lock().map_err(poisoned)?;
        let previous_scope = state
            .entries
            .get(&entry.id)
            .map(|previous| previous.scope.clone());
        if let Some(previous_scope) = previous_scope {
            if previous_scope != entry.scope {
                if let Some(index) = state.scope_members.get_mut(&previous_scope) {
                    index.zrem(entry.id.as_bytes());
                }
            }
        }
        state.seq = state.seq.saturating_add(1);
        let score = state.seq as f64;
        state
            .scope_members
            .entry(entry.scope.clone())
            .or_insert_with(|| OrderedIndex::new(OrderedMode::Persistent))
            .zadd(entry.id.as_bytes().to_vec(), score)?;
        state.entries.insert(entry.id.clone(), entry);
        Ok(())
    }

    fn lookup(&self, id: &str) -> GraphStoreResult<Option<ColdIndexEntry>> {
        Ok(self
            .state
            .lock()
            .map_err(poisoned)?
            .entries
            .get(id)
            .cloned())
    }

    fn remove(&self, id: &str) -> GraphStoreResult<()> {
        let mut state = self.state.lock().map_err(poisoned)?;
        if let Some(entry) = state.entries.remove(id) {
            if let Some(index) = state.scope_members.get_mut(&entry.scope) {
                index.zrem(id.as_bytes());
            }
        }
        Ok(())
    }

    fn record_scope(&self, entry: ColdScopeEntry) -> GraphStoreResult<()> {
        let mut state = self.state.lock().map_err(poisoned)?;
        state.scopes.insert(entry.scope.clone(), entry);
        Ok(())
    }

    fn scope(&self, scope: &str) -> GraphStoreResult<Option<ColdScopeEntry>> {
        Ok(self
            .state
            .lock()
            .map_err(poisoned)?
            .scopes
            .get(scope)
            .cloned())
    }

    fn remove_scope(&self, scope: &str) -> GraphStoreResult<()> {
        let mut state = self.state.lock().map_err(poisoned)?;
        state.scopes.remove(scope);
        state.scope_members.remove(scope);
        Ok(())
    }

    fn len(&self) -> usize {
        self.state
            .lock()
            .map(|state| state.entries.len())
            .unwrap_or(0)
    }
}

/// Disk-backed cold index: one JSON file per record, so a write is O(1) and the
/// index survives a restart (acceptance #3 needs BOTH the object and its
/// `id -> hash` mapping to be durable). Filenames are `stable_hash(id)` to avoid
/// collisions from arbitrary node ids (which carry `:` and other separators);
/// the full id is stored inside the record.
///
/// ```text
/// <root>/cold_index/<hash(id)>.json     # one ColdIndexEntry per node
/// <root>/cold_scopes/<hash(scope)>.json # one ColdScopeEntry per parked scope
/// ```
#[derive(Clone, Debug)]
pub struct DiskColdIndex {
    root: PathBuf,
    entry_count: Arc<AtomicUsize>,
    #[cfg(test)]
    fail_after_first_scope_member: Arc<AtomicUsize>,
    #[cfg(test)]
    fail_after_all_scope_members: Arc<AtomicUsize>,
}

impl DiskColdIndex {
    pub fn open(root: impl AsRef<Path>) -> GraphStoreResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("cold_index")).map_err(io_err("create cold_index dir"))?;
        fs::create_dir_all(root.join("cold_scopes")).map_err(io_err("create cold_scopes dir"))?;
        let entry_count = fs::read_dir(root.join("cold_index"))
            .map_err(io_err("count cold index entries"))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .count();
        Ok(Self {
            root,
            entry_count: Arc::new(AtomicUsize::new(entry_count)),
            #[cfg(test)]
            fail_after_first_scope_member: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            fail_after_all_scope_members: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[cfg(test)]
    pub fn fail_after_first_scope_member_once(&self) {
        self.fail_after_first_scope_member
            .store(1, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub fn fail_after_all_scope_members_once(&self) {
        self.fail_after_all_scope_members.store(1, Ordering::SeqCst);
    }

    fn entry_path(&self, id: &str) -> GraphStoreResult<PathBuf> {
        Ok(self
            .root
            .join("cold_index")
            .join(format!("{}.json", safe_filename(&stable_hash(id)?))))
    }

    fn scope_path(&self, scope: &str) -> GraphStoreResult<PathBuf> {
        Ok(self
            .root
            .join("cold_scopes")
            .join(format!("{}.json", safe_filename(&stable_hash(scope)?))))
    }

    fn write_atomic(path: &Path, bytes: &[u8]) -> GraphStoreResult<()> {
        let tmp = path.with_extension("json.tmp");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .map_err(io_err("open cold index tmp"))?;
        file.write_all(bytes)
            .map_err(io_err("write cold index tmp"))?;
        file.sync_all().map_err(io_err("sync cold index tmp"))?;
        #[cfg(windows)]
        let _ = fs::remove_file(path);
        fs::rename(&tmp, path).map_err(io_err("rename cold index"))?;
        sync_parent_directory(path)?;
        Ok(())
    }

    fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> GraphStoreResult<Option<T>> {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|error| GraphStoreError::new("cold_index_decode", error.to_string())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(io_err("read cold index")(error)),
        }
    }

    /// Stream durable entries one at a time. This restores cold residency
    /// after RedCore AOF replay without allocating a vector proportional to
    /// the cold catalog.
    pub fn for_each_entry(
        &self,
        mut visit: impl FnMut(ColdIndexEntry) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<()> {
        for entry in fs::read_dir(self.root.join("cold_index"))
            .map_err(io_err("read cold index directory"))?
        {
            let path = entry
                .map_err(io_err("read cold index directory entry"))?
                .path();
            if !path.extension().is_some_and(|ext| ext == "json") {
                continue;
            }
            if let Some(entry) = Self::read_json(&path)? {
                visit(entry)?;
            }
        }
        Ok(())
    }

    /// Stream durable parked-scope records one at a time.
    pub fn for_each_scope(
        &self,
        mut visit: impl FnMut(ColdScopeEntry) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<()> {
        for entry in fs::read_dir(self.root.join("cold_scopes"))
            .map_err(io_err("read cold scope directory"))?
        {
            let path = entry
                .map_err(io_err("read cold scope directory entry"))?
                .path();
            if !path.extension().is_some_and(|ext| ext == "json") {
                continue;
            }
            if let Some(entry) = Self::read_json(&path)? {
                visit(entry)?;
            }
        }
        Ok(())
    }
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> GraphStoreResult<()> {
    let parent = path.parent().ok_or_else(|| {
        GraphStoreError::new(
            "cold_index_io",
            format!("cold index path has no parent: {}", path.display()),
        )
    })?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(io_err("sync cold index directory"))
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> GraphStoreResult<()> {
    Ok(())
}

impl ColdIndex for DiskColdIndex {
    fn record(&self, entry: ColdIndexEntry) -> GraphStoreResult<()> {
        let path = self.entry_path(&entry.id)?;
        let existed = path.exists();
        let bytes = serde_json::to_vec(&entry)
            .map_err(|error| GraphStoreError::new("cold_index_encode", error.to_string()))?;
        Self::write_atomic(&path, &bytes)?;
        if !existed {
            self.entry_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn lookup(&self, id: &str) -> GraphStoreResult<Option<ColdIndexEntry>> {
        Self::read_json(&self.entry_path(id)?)
    }

    fn remove(&self, id: &str) -> GraphStoreResult<()> {
        match fs::remove_file(self.entry_path(id)?) {
            Ok(()) => {
                let _ =
                    self.entry_count
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                            Some(count.saturating_sub(1))
                        });
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_err("remove cold index")(error)),
        }
    }

    fn record_scope(&self, entry: ColdScopeEntry) -> GraphStoreResult<()> {
        let bytes = serde_json::to_vec(&entry)
            .map_err(|error| GraphStoreError::new("cold_scope_encode", error.to_string()))?;
        Self::write_atomic(&self.scope_path(&entry.scope)?, &bytes)
    }

    fn scope(&self, scope: &str) -> GraphStoreResult<Option<ColdScopeEntry>> {
        Self::read_json(&self.scope_path(scope)?)
    }

    fn remove_scope(&self, scope: &str) -> GraphStoreResult<()> {
        match fs::remove_file(self.scope_path(scope)?) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_err("remove cold scope")(error)),
        }
    }

    #[cfg(test)]
    fn after_first_scope_member_publication_for_test(&self) -> GraphStoreResult<()> {
        if self
            .fail_after_first_scope_member
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(GraphStoreError::new(
                "cold_scope_remap_first_member_injected_failure",
                "injected failure after first scope-member publication",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn after_all_scope_members_publication_for_test(&self) -> GraphStoreResult<()> {
        if self
            .fail_after_all_scope_members
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(GraphStoreError::new(
                "cold_scope_remap_all_members_injected_failure",
                "injected failure after all scope-member publications",
            ));
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }
}

fn io_err(context: &'static str) -> impl Fn(std::io::Error) -> GraphStoreError {
    move |error| GraphStoreError::new("cold_index_io", format!("{context}: {error}"))
}

fn poisoned<T>(_: T) -> GraphStoreError {
    GraphStoreError::new(
        "cold_index_poisoned",
        "cold index mutex was poisoned".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_records_and_removes() {
        let index = InMemoryColdIndex::new();
        index
            .record(ColdIndexEntry::cold("mem:a", "theorem", "sha256:deadbeef"))
            .unwrap();
        assert_eq!(index.len(), 1);
        let entry = index.lookup("mem:a").unwrap().unwrap();
        assert_eq!(entry.object_hash, "sha256:deadbeef");
        assert_eq!(entry.tier, ColdTierKind::Cold);
        index.remove("mem:a").unwrap();
        assert_eq!(index.lookup("mem:a").unwrap(), None);
    }

    #[test]
    fn disk_index_survives_reopen_with_separator_ids() {
        let dir = std::env::temp_dir().join(format!("cold-index-{}", std::process::id()));
        let id = "mem:project:theorem:alpha"; // ':' separators -> hashed filename
        {
            let index = DiskColdIndex::open(&dir).unwrap();
            index
                .record(ColdIndexEntry::cold(id, "theorem", "sha256:abc123"))
                .unwrap();
        }
        let reopened = DiskColdIndex::open(&dir).unwrap();
        let entry = reopened.lookup(id).unwrap().unwrap();
        assert_eq!(entry.id, id);
        assert_eq!(entry.object_hash, "sha256:abc123");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scope_records_round_trip() {
        let index = InMemoryColdIndex::new();
        index
            .record_scope(ColdScopeEntry {
                scope: "repo:theorem".to_string(),
                commit_hash: "sha256:commit".to_string(),
                node_ids: vec!["n1".to_string(), "n2".to_string()],
                edge_ids: vec!["e1".to_string()],
                parked: true,
            })
            .unwrap();
        let scope = index.scope("repo:theorem").unwrap().unwrap();
        assert!(scope.parked);
        assert_eq!(scope.node_ids.len(), 2);
        index.remove_scope("repo:theorem").unwrap();
        assert_eq!(index.scope("repo:theorem").unwrap(), None);
    }

    #[test]
    fn ordered_cold_index_keeps_hot_path_native_and_scope_ordered() {
        let index = OrderedColdIndex::new();
        index
            .record(ColdIndexEntry::cold("mem:a", "tenant:a", "sha256:a"))
            .unwrap();
        index
            .record(ColdIndexEntry::cold("mem:b", "tenant:a", "sha256:b"))
            .unwrap();
        index
            .record(ColdIndexEntry::cold("mem:c", "tenant:b", "sha256:c"))
            .unwrap();

        assert_eq!(
            index.lookup("mem:a").unwrap().unwrap().object_hash,
            "sha256:a"
        );
        assert_eq!(
            index.ids_for_scope("tenant:a", 0).unwrap(),
            vec!["mem:a".to_string(), "mem:b".to_string()]
        );
        index.remove("mem:a").unwrap();
        assert_eq!(index.lookup("mem:a").unwrap(), None);
        assert_eq!(
            index.ids_for_scope("tenant:a", 0).unwrap(),
            vec!["mem:b".to_string()]
        );
    }

    #[test]
    fn disk_scope_remap_restart_heals_partial_member_publication() {
        let old = Hash::of(b"legacy scope commit");
        let new = Hash::of(b"current scope commit");
        let old_text = old.to_prefixed();
        let new_text = new.to_prefixed();
        for fail_after_all in [false, true] {
            let dir = std::env::temp_dir().join(format!(
                "cold-index-remap-restart-{}-{fail_after_all}-{old}",
                std::process::id(),
            ));
            let _ = std::fs::remove_dir_all(&dir);
            {
                let index = DiskColdIndex::open(&dir).unwrap();
                index
                    .record_scope(ColdScopeEntry {
                        scope: "scope:a".to_string(),
                        commit_hash: old_text.clone(),
                        node_ids: vec!["node:a".to_string(), "node:b".to_string()],
                        edge_ids: Vec::new(),
                        parked: true,
                    })
                    .unwrap();
                for id in ["node:a", "node:b"] {
                    index
                        .record(ColdIndexEntry {
                            id: id.to_string(),
                            scope: "scope:a".to_string(),
                            tier: ColdTierKind::Warm,
                            object_hash: String::new(),
                            commit_hash: Some(old_text.clone()),
                        })
                        .unwrap();
                }
                if fail_after_all {
                    index.fail_after_all_scope_members_once();
                } else {
                    index.fail_after_first_scope_member_once();
                }
                let error = index.remap_scope_commit("scope:a", old, new).unwrap_err();
                assert_eq!(
                    error.code,
                    if fail_after_all {
                        "cold_scope_remap_all_members_injected_failure"
                    } else {
                        "cold_scope_remap_first_member_injected_failure"
                    }
                );
                assert_eq!(
                    index.scope("scope:a").unwrap().unwrap().commit_hash,
                    old_text
                );
                assert_eq!(
                    index
                        .lookup("node:a")
                        .unwrap()
                        .unwrap()
                        .commit_hash
                        .as_deref(),
                    Some(new_text.as_str())
                );
                assert_eq!(
                    index
                        .lookup("node:b")
                        .unwrap()
                        .unwrap()
                        .commit_hash
                        .as_deref(),
                    Some(if fail_after_all {
                        new_text.as_str()
                    } else {
                        old_text.as_str()
                    })
                );
            }

            let reopened = DiskColdIndex::open(&dir).unwrap();
            reopened.remap_scope_commit("scope:a", old, new).unwrap();
            assert_eq!(
                reopened.scope("scope:a").unwrap().unwrap().commit_hash,
                new_text
            );
            for id in ["node:a", "node:b"] {
                assert_eq!(
                    reopened.lookup(id).unwrap().unwrap().commit_hash.as_deref(),
                    Some(new_text.as_str())
                );
            }
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
