//! Verified content-addressed chunk storage.
//!
//! [`ChunkStore`] deliberately contains only the operations shared by every
//! physical store. Reachability-based reclamation is a separate extension so
//! callers cannot mistake deletion for part of the durability contract.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::versioned_graph::{DOMAIN_COMMIT, GRAPH_CHUNK_FORMAT_VERSION};

/// Canonical domain byte for a named-root catalog.
pub const DOMAIN_ROOT_CATALOG: u8 = 5;
/// Stable name of the graph root in a [`RootCatalog`].
pub const ROOT_NAME_GRAPH: &str = "graph";
/// Stable name of the RedCore operating-state root in a [`RootCatalog`].
pub const ROOT_NAME_REDCORE: &str = "redcore";
/// Stable name of the transitional document root in a [`RootCatalog`].
pub const ROOT_NAME_DOCUMENTS: &str = "documents";
/// Stable name of the cold object/pack semantic-to-physical index.
pub const ROOT_NAME_OBJECTS: &str = "objects";
/// Stable name of RedCore's full-text index root.
pub const ROOT_NAME_FULLTEXT: &str = "fulltext";

/// Metadata which participates in the durable commit address.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitMeta {
    pub parents: Vec<Hash>,
    pub author: String,
    pub message: String,
    pub timestamp_unix_ms: u128,
    pub graph_version: u64,
}

/// The latest durable root together with the address of its commit record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedRoot {
    pub root: Hash,
    pub commit: Hash,
    pub meta: CommitMeta,
}

impl CommittedRoot {
    pub fn new(root: Hash, meta: CommitMeta) -> GraphStoreResult<Self> {
        let commit = canonical_chunk_store_commit_hash(root, &meta)?;
        Ok(Self { root, commit, meta })
    }

    /// Encode the exact Phase-C commit preimage used as a journal root payload.
    ///
    /// Refuses to serialize a root whose `commit` no longer matches its payload.
    /// [`Self::new`] derives `commit` from `root` and `meta`, and [`Self::from_bytes`]
    /// routes through `Self::new` rather than trusting a payload — but the fields
    /// are public, so a holder can mutate `meta` afterwards and leave `commit`
    /// stale. This call is the only thing standing between that state and a
    /// persisted root whose record address lies about its content, so it stays.
    /// `chunk_store_commit_preimage_cross_matches_phase_c_and_golden` pins it
    /// (`chunk_store.rs`, `assert!(tampered.to_bytes().is_err())`).
    ///
    /// The cost is one SHA-256 over a ~150-byte preimage, which is not worth
    /// trading for; the per-commit byte volume on this path is the journal state
    /// chunk, not this.
    pub fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        self.verify_commit(self.commit)?;
        canonical_chunk_store_commit_preimage(self.root, &self.meta)
    }

    /// Strictly decode a canonical journal root payload.
    pub fn from_bytes(bytes: &[u8]) -> GraphStoreResult<Self> {
        let mut decoder = CanonicalDecoder::new(bytes, "chunk store commit");
        decoder.expect_byte(DOMAIN_COMMIT, "commit domain")?;
        decoder.expect_byte(GRAPH_CHUNK_FORMAT_VERSION, "commit format version")?;
        let root = decoder.hash("commit root")?;
        let parent_count = decoder.count("commit parent count", 32)?;
        let mut parents = Vec::with_capacity(parent_count);
        for _ in 0..parent_count {
            parents.push(decoder.hash("commit parent")?);
        }
        let author = decoder.string("commit author")?;
        let message = decoder.string("commit message")?;
        let timestamp_unix_ms = decoder.u128("commit timestamp")?;
        let graph_version = decoder.u64("commit graph version")?;
        decoder.finish()?;

        Self::new(
            root,
            CommitMeta {
                parents,
                author,
                message,
                timestamp_unix_ms,
                graph_version,
            },
        )
    }

    pub fn verify_commit(&self, expected: Hash) -> GraphStoreResult<()> {
        let actual = canonical_chunk_store_commit_hash(self.root, &self.meta)?;
        if actual != self.commit {
            return Err(GraphStoreError::new(
                "chunk_store_commit_payload_mismatch",
                format!(
                    "commit fields hash to {actual}, but committed root carries {}",
                    self.commit
                ),
            ));
        }
        if actual != expected {
            return Err(GraphStoreError::new(
                "chunk_store_commit_address_mismatch",
                format!(
                    "root payload hashes to {}, but record address is {expected}",
                    actual
                ),
            ));
        }
        Ok(())
    }
}

/// A canonical sorted map from stable subsystem names to their current roots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootCatalog {
    roots: BTreeMap<String, Hash>,
}

impl RootCatalog {
    pub fn new(roots: BTreeMap<String, Hash>) -> GraphStoreResult<Self> {
        for name in roots.keys() {
            validate_root_name(name)?;
        }
        Ok(Self { roots })
    }

    pub fn graph(root: Hash) -> Self {
        Self {
            roots: BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), root)]),
        }
    }

    pub fn roots(&self) -> &BTreeMap<String, Hash> {
        &self.roots
    }

    pub fn get(&self, name: &str) -> Option<Hash> {
        self.roots.get(name).copied()
    }

    pub fn insert(
        &mut self,
        name: impl Into<String>,
        root: Hash,
    ) -> GraphStoreResult<Option<Hash>> {
        let name = name.into();
        validate_root_name(&name)?;
        Ok(self.roots.insert(name, root))
    }

    /// Encode the frozen catalog layout: domain, version, count, then sorted
    /// length-prefixed UTF-8 names and raw SHA-256 addresses.
    pub fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        let mut output = Vec::new();
        output.push(DOMAIN_ROOT_CATALOG);
        output.push(GRAPH_CHUNK_FORMAT_VERSION);
        output.extend_from_slice(
            &checked_u32(self.roots.len(), "root catalog entry count")?.to_le_bytes(),
        );
        for (name, root) in &self.roots {
            validate_root_name(name)?;
            push_string(&mut output, name, "root catalog name")?;
            output.extend_from_slice(root.as_bytes());
        }
        Ok(output)
    }

    pub fn from_bytes(bytes: &[u8]) -> GraphStoreResult<Self> {
        let mut decoder = CanonicalDecoder::new(bytes, "root catalog");
        decoder.expect_byte(DOMAIN_ROOT_CATALOG, "root catalog domain")?;
        decoder.expect_byte(GRAPH_CHUNK_FORMAT_VERSION, "root catalog format version")?;
        let root_count = decoder.count("root catalog entry count", 37)?;
        let mut roots = BTreeMap::new();
        let mut previous_name: Option<String> = None;
        for _ in 0..root_count {
            let name = decoder.string("root catalog name")?;
            validate_root_name(&name)?;
            if previous_name
                .as_deref()
                .is_some_and(|previous| previous >= name.as_str())
            {
                return Err(GraphStoreError::new(
                    "chunk_store_catalog_not_canonical",
                    "root catalog names must be strictly increasing",
                ));
            }
            let root = decoder.hash("named root")?;
            previous_name = Some(name.clone());
            roots.insert(name, root);
        }
        decoder.finish()?;
        Ok(Self { roots })
    }

    pub fn address(&self) -> GraphStoreResult<Hash> {
        self.to_bytes().map(|bytes| Hash::of(&bytes))
    }

    /// Write this catalog as an ordinary verified chunk and return its address.
    pub fn write_to(&self, store: &dyn ChunkStore) -> GraphStoreResult<Hash> {
        let bytes = self.to_bytes()?;
        let address = Hash::of(&bytes);
        store.write(&[(address, bytes.as_slice())])?;
        Ok(address)
    }

    /// Verify that every root directly named by this catalog exists.
    pub fn validate_named_roots(&self, store: &dyn ChunkStore) -> GraphStoreResult<()> {
        for (name, root) in &self.roots {
            if store.get(*root)?.is_none() {
                return Err(GraphStoreError::new(
                    "chunk_store_named_root_missing",
                    format!("root catalog entry {name:?} points to missing chunk {root}"),
                ));
            }
        }
        Ok(())
    }

    /// Return the mandatory graph root for a committable catalog.
    pub fn graph_root(&self) -> GraphStoreResult<Hash> {
        self.get(ROOT_NAME_GRAPH).ok_or_else(|| {
            GraphStoreError::new(
                "chunk_store_catalog_graph_root_missing",
                format!("committed root catalog must contain {ROOT_NAME_GRAPH:?}"),
            )
        })
    }
}

/// Load and validate the catalog chunk and all of its immediate named roots.
pub fn validate_catalog_root(
    store: &dyn ChunkStore,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    let bytes = store.get(catalog_address)?.ok_or_else(|| {
        GraphStoreError::new(
            "chunk_store_catalog_missing",
            format!("committed root catalog {catalog_address} is missing"),
        )
    })?;
    let catalog = RootCatalog::from_bytes(&bytes)?;
    let actual = catalog.address()?;
    if actual != catalog_address {
        return Err(GraphStoreError::new(
            "chunk_store_catalog_address_mismatch",
            format!("root catalog hashes to {actual}, expected {catalog_address}"),
        ));
    }
    catalog.graph_root()?;
    catalog.validate_named_roots(store)?;
    Ok(catalog)
}

/// Frozen Phase-C commit preimage shared by graph commits and store roots.
pub fn canonical_chunk_store_commit_preimage(
    root: Hash,
    meta: &CommitMeta,
) -> GraphStoreResult<Vec<u8>> {
    let mut output = Vec::new();
    output.push(DOMAIN_COMMIT);
    output.push(GRAPH_CHUNK_FORMAT_VERSION);
    output.extend_from_slice(root.as_bytes());
    output
        .extend_from_slice(&checked_u32(meta.parents.len(), "commit parent count")?.to_le_bytes());
    for parent in &meta.parents {
        output.extend_from_slice(parent.as_bytes());
    }
    push_string(&mut output, &meta.author, "commit author")?;
    push_string(&mut output, &meta.message, "commit message")?;
    output.extend_from_slice(&meta.timestamp_unix_ms.to_le_bytes());
    output.extend_from_slice(&meta.graph_version.to_le_bytes());
    Ok(output)
}

pub fn canonical_chunk_store_commit_hash(root: Hash, meta: &CommitMeta) -> GraphStoreResult<Hash> {
    canonical_chunk_store_commit_preimage(root, meta).map(|bytes| Hash::of(&bytes))
}

/// Verified chunk storage with one committed named-root catalog.
pub trait ChunkStore: Send + Sync {
    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>>;
    /// Physical payload admission before allocation/decompression. A backend
    /// without this capability refuses; calling ordinary get first is unsafe.
    fn get_bounded(&self, _addr: Hash, _max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        Err(GraphStoreError::new(
            "chunk_bounded_read_unsupported",
            "backend has no bounded chunk read",
        ))
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool>;

    /// Cheap existence probe: is this address present in the store?
    ///
    /// [`ChunkStore::has`] is a *verified* existence check on both concrete
    /// stores: `MemChunkStore` routes it through `verified_chunk` and
    /// `JournalChunkStore` through `load_verified_chunk`. On the journal store,
    /// where the publish cost lives, answering "is it there?" therefore costs a
    /// seek, a frame read, a crc32, a zstd decode and a SHA-256 over the chunk's
    /// bytes; on the memory store it is a map lookup plus the SHA-256.
    ///
    /// That is the right price for an admission question. It is the wrong price
    /// for a *prune hint* on the publish path, where the question is only "may I
    /// stop walking this subtree?" over a chunk reachable from an
    /// already-published root. The hint is asked once per base child of every
    /// staged parent, so a verifying check made a bounded write pay a verified
    /// read per hint: measured at 25 hints for a one-key replacement at 1k
    /// records and 56 at 10k.
    ///
    /// This method answers that weaker question. A store with an index may
    /// answer it from the index alone, without reading or verifying bytes.
    ///
    /// The default delegates to [`ChunkStore::has`], so a store without an
    /// index keeps its current verifying behaviour. `MemChunkStore` and
    /// `JournalChunkStore` override it; `RepositoryChunkStore` and
    /// `ProjectionChunkStore` inherit the verifying default, which is correct
    /// today because neither is handed to the incremental tree build.
    ///
    /// Two prohibitions. Do not use this to decide whether to *skip a write*:
    /// a false positive there would drop bytes the caller still owes the store,
    /// so admission checks stay on `has`/`missing`. And do not treat `true` as
    /// evidence the bytes are intact; reads still go through `get`, which
    /// verifies.
    ///
    /// ## What this gives up, and where it is caught
    ///
    /// On the publish path `has` was also the only refusal a corrupted base
    /// chunk would meet before being republished by reference. The reach of that
    /// loss is store-specific, and it is narrower than "the check is gone":
    ///
    /// * **`MemChunkStore` loses nothing.** Its values are immutable `Arc<[u8]>`
    ///   inserted only after `ensure_address` proved them, so an indexed address
    ///   always hashes to itself and the probe is outcome-equivalent to `has`.
    /// * **`JournalChunkStore` loses detection for one class.** The index is
    ///   populated after the bytes are verified, both on append and by
    ///   `scan_journal` on open, so an entry can only disagree with its bytes if
    ///   the backing file is corrupted *after* that open. For a pruned subtree the
    ///   build never reads, such a chunk is republished by reference rather than
    ///   raising here.
    ///
    /// That second class is real and is asserted, not argued, by
    /// `a_pruned_subtree_this_build_never_reads_is_republished_and_the_boundary_refuses_it`.
    /// Detection for it is the boundary, and each of these is a verified read:
    /// hydration walks the tree with `get` on the next open, `recover` runs the
    /// parity audit by default, `verify_tree_parity` is available on demand,
    /// `scan_journal` SHA-256s every frame before indexing it, and
    /// `ChunkStoreCompaction::compact_reachable` verifies every reachable
    /// address. The window is "published and not yet reopened or verified".
    ///
    /// A third class is unchanged and worth naming so it is not mistaken for new:
    /// a descendant of a *reused* subtree root was never read by either path, so
    /// no refusal existed there to lose.
    fn contains_address(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.has(addr)
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>>;
    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()>;
    /// Write a batch and report the chunks this call physically admitted.
    ///
    /// The default keeps third-party stores source-compatible, but concrete
    /// stores should override it so admission and accounting share one lock.
    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        let addresses = chunks
            .iter()
            .map(|(address, _)| *address)
            .collect::<Vec<_>>();
        let missing = self
            .missing(&addresses)?
            .into_iter()
            .collect::<HashSet<_>>();
        self.write(chunks)?;

        let mut seen = HashSet::new();
        let mut report = ChunkWriteReport::default();
        for (address, bytes) in chunks {
            if missing.contains(address) && seen.insert(*address) {
                report.chunks_written = report
                    .chunks_written
                    .checked_add(1)
                    .ok_or_else(chunk_accounting_overflow)?;
                report.bytes_written = report
                    .bytes_written
                    .checked_add(bytes.len())
                    .ok_or_else(chunk_accounting_overflow)?;
            }
        }
        Ok(report)
    }
    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash>;
    fn read_root(&self) -> GraphStoreResult<Option<Hash>>;

    /// Protect one content address while bytes are being staged for a future
    /// root publication. The default is a compatibility no-op for stores
    /// without physical reclamation.
    fn pin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let _ = address;
        Ok(())
    }

    /// Bind one anonymous producer receipt to its exact logical publication.
    /// Repeated bindings of the same key/address are reference counted; using
    /// the same key for a different address fails without consuming either.
    fn bind_pending_in_flight(&self, address: Hash, publication_key: &str) -> GraphStoreResult<()> {
        let _ = (address, publication_key);
        Ok(())
    }

    /// Release only the pending receipt owned by this exact key/address pair.
    /// Returns false when the key has no bound receipt.
    fn release_bound_pending_in_flight(
        &self,
        address: Hash,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        let _ = (address, publication_key);
        Ok(false)
    }

    /// Create a fresh transaction-owned publication claim. Claims never
    /// consume anonymous or keyed producer receipts.
    fn claim_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let _ = address;
        Ok(())
    }

    /// Release one publication claim after its referencing root is durable.
    fn release_claimed_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let _ = address;
        Ok(())
    }

    /// Abort one pending producer receipt which will not be published. This
    /// never consumes an active publication claim.
    fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let _ = address;
        Ok(())
    }
}

static PUBLICATION_WINDOW_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Process-local pins protecting chunks between physical admission and the
/// durable root publication which first makes them reachable.
///
/// The wrapper implements [`ChunkStore`] so tree builders can write through it
/// without learning about reclamation. Every distinct address receives one
/// collision-safe keyed receipt. [`Self::finish_after_publication`] releases
/// those receipts only after the caller's root commit succeeds; dropping an
/// unfinished window aborts them best-effort.
pub(crate) struct ChunkPublicationWindow<S: ChunkStore + ?Sized> {
    store: Arc<S>,
    key_prefix: String,
    receipts: Mutex<BTreeMap<Hash, String>>,
    armed: bool,
}

impl<S: ChunkStore + ?Sized> ChunkPublicationWindow<S> {
    pub(crate) fn new(store: Arc<S>, scope: &str) -> Self {
        let scope_address = Hash::of(scope.as_bytes());
        let sequence = PUBLICATION_WINDOW_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        Self {
            store,
            key_prefix: format!("chunk-publication:{scope_address}:{sequence}"),
            receipts: Mutex::new(BTreeMap::new()),
            armed: true,
        }
    }

    fn pin_once(&self, address: Hash) -> GraphStoreResult<()> {
        let mut receipts = self.receipts.lock().map_err(|_| {
            GraphStoreError::new(
                "chunk_store_publication_window_poisoned",
                "publication-window receipt lock is poisoned",
            )
        })?;
        if receipts.contains_key(&address) {
            return Ok(());
        }
        let publication_key = format!("{}:{address}", self.key_prefix);
        self.store.pin_in_flight(address)?;
        if let Err(bind_error) = self.store.bind_pending_in_flight(address, &publication_key) {
            return match self.store.unpin_in_flight(address) {
                Ok(()) => Err(bind_error),
                Err(release_error) => Err(publication_window_cleanup_error(
                    "binding publication-window pin",
                    bind_error,
                    release_error,
                )),
            };
        }
        receipts.insert(address, publication_key);
        Ok(())
    }

    fn pin_batch(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        for (address, _) in chunks {
            self.pin_once(*address)?;
        }
        Ok(())
    }

    fn release_all(&self) -> GraphStoreResult<()> {
        loop {
            let receipt = {
                self.receipts
                    .lock()
                    .map_err(|_| {
                        GraphStoreError::new(
                            "chunk_store_publication_window_poisoned",
                            "publication-window receipt lock is poisoned",
                        )
                    })?
                    .iter()
                    .next()
                    .map(|(address, key)| (*address, key.clone()))
            };
            let Some((address, publication_key)) = receipt else {
                return Ok(());
            };
            if !self
                .store
                .release_bound_pending_in_flight(address, &publication_key)?
            {
                return Err(GraphStoreError::new(
                    "chunk_store_publication_window_pin_missing",
                    format!(
                        "publication-window receipt {publication_key:?} for {address} is missing"
                    ),
                ));
            }
            self.receipts
                .lock()
                .map_err(|_| {
                    GraphStoreError::new(
                        "chunk_store_publication_window_poisoned",
                        "publication-window receipt lock is poisoned",
                    )
                })?
                .remove(&address);
        }
    }

    /// Pin an address already present in the store, so it cannot be
    /// reclaimed before the publication that references it.
    pub(crate) fn protect(&self, address: Hash) -> GraphStoreResult<()> {
        self.pin_once(address)
    }

    pub(crate) fn finish_after_publication(mut self) {
        // The root barrier has already succeeded, so pin cleanup can no longer
        // change the transaction outcome. Keep the guard armed on a transient
        // cleanup error so Drop retries once; never turn a durable commit into
        // a caller-visible Err.
        if self.release_all().is_ok() {
            self.armed = false;
        }
    }
}

impl<S: ChunkStore + ?Sized> Drop for ChunkPublicationWindow<S> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.release_all();
        }
    }
}

impl<S: ChunkStore + ?Sized> ChunkStore for ChunkPublicationWindow<S> {
    fn get_bounded(&self, addr: Hash, max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.store.get_bounded(addr, max_bytes)
    }

    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.store.get(addr)
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.store.has(addr)
    }

    fn contains_address(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.store.contains_address(addr)
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.store.missing(addrs)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.pin_batch(chunks)?;
        self.store.write(chunks)
    }

    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        self.pin_batch(chunks)?;
        self.store.write_accounted(chunks)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.store.commit(root, meta)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.store.read_root()
    }

    fn pin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        self.store.pin_in_flight(address)
    }

    fn bind_pending_in_flight(&self, address: Hash, publication_key: &str) -> GraphStoreResult<()> {
        self.store.bind_pending_in_flight(address, publication_key)
    }

    fn release_bound_pending_in_flight(
        &self,
        address: Hash,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        self.store
            .release_bound_pending_in_flight(address, publication_key)
    }

    fn claim_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        self.store.claim_in_flight(address)
    }

    fn release_claimed_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        self.store.release_claimed_in_flight(address)
    }

    fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        self.store.unpin_in_flight(address)
    }
}

fn publication_window_cleanup_error(
    action: &str,
    original: GraphStoreError,
    cleanup: GraphStoreError,
) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_publication_window_cleanup_failed",
        format!(
            "{action} failed ({}: {}); cleanup also failed ({}: {})",
            original.code, original.message, cleanup.code, cleanup.message
        ),
    )
}

/// Addresses published by one atomic named-root catalog update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogCommitResult {
    /// Address of the new canonical [`RootCatalog`] chunk.
    pub catalog_root: Hash,
    /// Address of the commit record which durably publishes `catalog_root`.
    pub commit: Hash,
}

/// Optional extension for atomically publishing updates to named roots.
///
/// `Some(root)` sets a name and `None` removes it. Implementations merge the
/// updates into the latest committed catalog while holding their root-state
/// lock, so concurrent callers cannot overwrite unrelated names. The new
/// catalog chunk and its commit become visible together or neither does. The
/// implementation also derives the commit's single physical parent from that
/// same locked state; caller-supplied parents cannot race another publisher.
pub trait ChunkStoreCatalog: ChunkStore {
    fn commit_catalog_updates(
        &self,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult>;

    /// Whether this backend can compare and publish one catalog atomically.
    /// Importers check this before staging canonical destination chunks.
    fn supports_catalog_compare_and_swap(&self) -> bool {
        false
    }

    /// Compare the accepted catalog while holding the same publication lock
    /// used by every other writer. A mismatch publishes nothing; staged chunks
    /// remain unreachable. `None` requires an uncommitted destination.
    fn commit_catalog_updates_if_root(
        &self,
        _expected: Option<Hash>,
        _updates: &BTreeMap<String, Option<Hash>>,
        _meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        Err(GraphStoreError::new(
            "chunk_store_catalog_precondition_unsupported",
            "backend cannot atomically fence a changed catalog",
        ))
    }
}

/// Exact physical delta admitted by one [`ChunkStore::write_accounted`] call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChunkWriteReport {
    pub chunks_written: usize,
    pub bytes_written: usize,
}

/// Outcome of an explicit reachability compaction pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionReport {
    pub chunks_before: usize,
    pub chunks_after: usize,
    pub chunks_removed: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub bytes_removed: u64,
}

/// Optional extension for stores capable of reclaiming unreachable chunks.
///
/// `expected_catalog_root` is the exact committed root observed by the
/// reachability walk which produced `reachable`. Implementations must compare
/// it with the current committed root while holding the same mutation lock
/// used for publication and reclamation, and refuse compaction on any mismatch.
/// `None` asserts that the store was uncommitted when the reachable set was
/// assembled.
pub trait ChunkStoreCompaction: ChunkStore {
    fn compact_reachable(
        &self,
        expected_catalog_root: Option<Hash>,
        reachable: &[Hash],
    ) -> GraphStoreResult<CompactionReport>;
}

#[derive(Clone, Debug, Default)]
pub struct MemChunkStore {
    state: Arc<RwLock<MemChunkStoreState>>,
}

#[derive(Debug, Default)]
struct MemChunkStoreState {
    chunks: HashMap<Hash, Arc<[u8]>>,
    committed: Option<CommittedRoot>,
    pending_pins: HashMap<Hash, usize>,
    bound_pending_pins: HashMap<String, BoundPendingPin>,
    claimed_pins: HashMap<Hash, usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundPendingPin {
    address: Hash,
    count: usize,
}

impl MemChunkStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct physical chunks currently admitted by this store.
    ///
    /// This is an observability seam for deduplication and compaction oracles;
    /// it is deliberately not part of the base [`ChunkStore`] contract.
    pub fn chunk_count(&self) -> GraphStoreResult<usize> {
        Ok(read_state(&self.state)?.chunks.len())
    }

    pub fn committed_root(&self) -> GraphStoreResult<Option<CommittedRoot>> {
        Ok(read_state(&self.state)?.committed.clone())
    }

    /// Current process-local in-flight reference count for one address.
    pub fn in_flight_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = read_state(&self.state)?;
        let pending =
            pending_pin_count_in_state(&state.pending_pins, &state.bound_pending_pins, address)?;
        pending
            .checked_add(state.claimed_pins.get(&address).copied().unwrap_or(0))
            .ok_or_else(pin_count_overflow)
    }

    pub fn pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = read_state(&self.state)?;
        pending_pin_count_in_state(&state.pending_pins, &state.bound_pending_pins, address)
    }

    pub fn anonymous_pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        Ok(read_state(&self.state)?
            .pending_pins
            .get(&address)
            .copied()
            .unwrap_or(0))
    }

    pub fn bound_pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = read_state(&self.state)?;
        bound_pending_pin_count_in_state(&state.bound_pending_pins, address)
    }

    pub fn claimed_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        Ok(read_state(&self.state)?
            .claimed_pins
            .get(&address)
            .copied()
            .unwrap_or(0))
    }

    fn verified_chunk(
        state: &MemChunkStoreState,
        address: Hash,
    ) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let Some(bytes) = state.chunks.get(&address).cloned() else {
            return Ok(None);
        };
        ensure_address(address, &bytes)?;
        Ok(Some(bytes))
    }

    fn catalog_from_state(
        state: &MemChunkStoreState,
        catalog_address: Hash,
    ) -> GraphStoreResult<RootCatalog> {
        let bytes = Self::verified_chunk(state, catalog_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "chunk_store_catalog_missing",
                format!("committed root catalog {catalog_address} is missing"),
            )
        })?;
        let catalog = RootCatalog::from_bytes(&bytes)?;
        if catalog.address()? != catalog_address {
            return Err(GraphStoreError::new(
                "chunk_store_catalog_address_mismatch",
                format!("root catalog does not match address {catalog_address}"),
            ));
        }
        catalog.graph_root()?;
        for (name, named_root) in catalog.roots() {
            if Self::verified_chunk(state, *named_root)?.is_none() {
                return Err(GraphStoreError::new(
                    "chunk_store_named_root_missing",
                    format!("root catalog entry {name:?} points to missing chunk {named_root}"),
                ));
            }
        }
        Ok(catalog)
    }
}

impl ChunkStore for MemChunkStore {
    fn get_bounded(&self, addr: Hash, max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let state = read_state(&self.state)?;
        if state
            .chunks
            .get(&addr)
            .is_some_and(|bytes| bytes.len() > max_bytes)
        {
            return Err(GraphStoreError::new(
                "chunk_read_bound_refused",
                "chunk exceeds admitted read bytes",
            ));
        }
        Self::verified_chunk(&state, addr)
    }

    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let state = read_state(&self.state)?;
        Self::verified_chunk(&state, addr)
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.get(addr).map(|bytes| bytes.is_some())
    }

    fn contains_address(&self, addr: Hash) -> GraphStoreResult<bool> {
        // Presence in `chunks` is the admission record: `write_accounted`
        // verifies the address before it inserts, so an indexed address has
        // already been proven to hash to itself. No bytes are read here.
        let state = read_state(&self.state)?;
        Ok(state.chunks.contains_key(&addr))
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let state = read_state(&self.state)?;
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        for address in addrs {
            if seen.insert(*address) && Self::verified_chunk(&state, *address)?.is_none() {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.write_accounted(chunks).map(|_| ())
    }

    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        // Validate and own the complete batch before mutating shared state.
        let mut staged = Vec::with_capacity(chunks.len());
        let mut seen = HashSet::with_capacity(chunks.len());
        for (address, bytes) in chunks {
            ensure_address(*address, bytes)?;
            if seen.insert(*address) {
                staged.push((*address, Arc::<[u8]>::from(*bytes)));
            }
        }

        let mut state = self.state.write().map_err(|_| poisoned())?;
        let mut report = ChunkWriteReport::default();
        for (address, bytes) in staged {
            if let Some(existing) = state.chunks.get(&address) {
                ensure_address(address, existing)?;
                continue;
            }
            report.chunks_written = report
                .chunks_written
                .checked_add(1)
                .ok_or_else(chunk_accounting_overflow)?;
            report.bytes_written = report
                .bytes_written
                .checked_add(bytes.len())
                .ok_or_else(chunk_accounting_overflow)?;
            state.chunks.insert(address, bytes);
        }
        Ok(report)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        Self::catalog_from_state(&state, root)?;
        let mut meta = meta;
        meta.parents = state
            .committed
            .as_ref()
            .map(|committed| committed.commit)
            .into_iter()
            .collect();
        let committed = CommittedRoot::new(root, meta)?;
        state.committed = Some(committed.clone());
        Ok(committed.commit)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        let state = read_state(&self.state)?;
        let Some(committed) = &state.committed else {
            return Ok(None);
        };
        committed.verify_commit(committed.commit)?;
        Self::catalog_from_state(&state, committed.root)?;
        Ok(Some(committed.root))
    }

    fn pin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        increment_pin_count(&mut state.pending_pins, address)?;
        Ok(())
    }

    fn claim_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        increment_pin_count(&mut state.claimed_pins, address)?;
        Ok(())
    }

    fn bind_pending_in_flight(&self, address: Hash, publication_key: &str) -> GraphStoreResult<()> {
        validate_publication_key(publication_key)?;
        let mut state = self.state.write().map_err(|_| poisoned())?;
        if let Some(bound) = state.bound_pending_pins.get(publication_key) {
            if bound.address != address {
                return Err(publication_key_address_conflict(
                    publication_key,
                    bound.address,
                    address,
                ));
            }
        }
        if state.pending_pins.get(&address).copied().unwrap_or(0) == 0 {
            return Err(pending_pin_missing(address, publication_key));
        }
        let incremented_count = state
            .bound_pending_pins
            .get(publication_key)
            .map(|bound| bound.count.checked_add(1).ok_or_else(pin_count_overflow))
            .transpose()?;
        decrement_pin_count(&mut state.pending_pins, address, "pending")?;
        if let Some(count) = incremented_count {
            state
                .bound_pending_pins
                .get_mut(publication_key)
                .expect("validated bound pending receipt")
                .count = count;
        } else {
            state.bound_pending_pins.insert(
                publication_key.to_string(),
                BoundPendingPin { address, count: 1 },
            );
        }
        Ok(())
    }

    fn release_bound_pending_in_flight(
        &self,
        address: Hash,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        validate_publication_key(publication_key)?;
        let mut state = self.state.write().map_err(|_| poisoned())?;
        release_bound_pending(&mut state.bound_pending_pins, address, publication_key)
    }

    fn release_claimed_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        decrement_pin_count(&mut state.claimed_pins, address, "claimed")?;
        Ok(())
    }

    fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        decrement_pin_count(&mut state.pending_pins, address, "pending")?;
        Ok(())
    }
}

impl ChunkStoreCatalog for MemChunkStore {
    fn commit_catalog_updates(
        &self,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        self.commit_catalog_updates_checked(None, updates, meta)
    }
    fn supports_catalog_compare_and_swap(&self) -> bool {
        true
    }
    fn commit_catalog_updates_if_root(
        &self,
        expected: Option<Hash>,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        self.commit_catalog_updates_checked(Some(expected), updates, meta)
    }
}

impl MemChunkStore {
    fn commit_catalog_updates_checked(
        &self,
        expected: Option<Option<Hash>>,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        let mut state = self.state.write().map_err(|_| poisoned())?;
        if expected.is_some_and(|expected| {
            expected != state.committed.as_ref().map(|committed| committed.root)
        }) {
            return Err(GraphStoreError::new(
                "chunk_store_catalog_precondition_failed",
                "accepted catalog changed before publication",
            ));
        }

        let (current, current_parent) = match state.committed.clone() {
            Some(committed) => {
                committed.verify_commit(committed.commit)?;
                (
                    Some(Self::catalog_from_state(&state, committed.root)?),
                    Some(committed.commit),
                )
            }
            None => (None, None),
        };
        let catalog = merge_catalog_updates(current, updates)?;
        catalog.graph_root()?;
        for (name, root) in catalog.roots() {
            if Self::verified_chunk(&state, *root)?.is_none() {
                return Err(GraphStoreError::new(
                    "chunk_store_named_root_missing",
                    format!("root catalog entry {name:?} points to missing chunk {root}"),
                ));
            }
        }

        let catalog_bytes = Arc::<[u8]>::from(catalog.to_bytes()?);
        let catalog_root = Hash::of(&catalog_bytes);
        if let Some(existing) = Self::verified_chunk(&state, catalog_root)? {
            if existing.as_ref() != catalog_bytes.as_ref() {
                return Err(GraphStoreError::new(
                    "chunk_store_catalog_address_collision",
                    format!("existing catalog chunk {catalog_root} has different bytes"),
                ));
            }
        }
        let mut meta = meta;
        meta.parents = current_parent.into_iter().collect();
        let committed = CommittedRoot::new(catalog_root, meta)?;

        // All fallible validation happens before either shared-state update.
        state.chunks.entry(catalog_root).or_insert(catalog_bytes);
        state.committed = Some(committed.clone());
        Ok(CatalogCommitResult {
            catalog_root,
            commit: committed.commit,
        })
    }
}

impl ChunkStoreCompaction for MemChunkStore {
    fn compact_reachable(
        &self,
        expected_catalog_root: Option<Hash>,
        reachable: &[Hash],
    ) -> GraphStoreResult<CompactionReport> {
        let requested: HashSet<Hash> = reachable.iter().copied().collect();
        let mut state = self.state.write().map_err(|_| poisoned())?;

        let current_catalog_root = state.committed.as_ref().map(|committed| committed.root);
        if current_catalog_root != expected_catalog_root {
            return Err(compaction_root_changed(
                expected_catalog_root,
                current_catalog_root,
            ));
        }

        for address in &requested {
            if Self::verified_chunk(&state, *address)?.is_none() {
                return Err(GraphStoreError::new(
                    "chunk_store_compaction_reachable_missing",
                    format!("reachable chunk {address} is missing"),
                ));
            }
        }

        // Snapshot pins while holding the same mutation lock used by writes,
        // unpins, and reclamation. A pin may precede its raw write, so only
        // caller-declared reachable addresses are required to exist here.
        let mut keep = requested;
        keep.extend(state.pending_pins.keys().copied());
        keep.extend(state.bound_pending_pins.values().map(|bound| bound.address));
        keep.extend(state.claimed_pins.keys().copied());

        if let Some(committed) = &state.committed {
            let catalog = Self::catalog_from_state(&state, committed.root)?;
            if !keep.contains(&committed.root) {
                return Err(GraphStoreError::new(
                    "chunk_store_compaction_live_root_omitted",
                    format!(
                        "reachable set omits current root catalog {}",
                        committed.root
                    ),
                ));
            }
            for (name, root) in catalog.roots() {
                if !keep.contains(root) {
                    return Err(GraphStoreError::new(
                        "chunk_store_compaction_live_root_omitted",
                        format!("reachable set omits current named root {name:?} at {root}"),
                    ));
                }
            }
        }

        let chunks_before = state.chunks.len();
        let bytes_before = total_bytes(state.chunks.values())?;
        state.chunks.retain(|address, _| keep.contains(address));
        let chunks_after = state.chunks.len();
        let bytes_after = total_bytes(state.chunks.values())?;
        Ok(CompactionReport {
            chunks_before,
            chunks_after,
            chunks_removed: chunks_before - chunks_after,
            bytes_before,
            bytes_after,
            bytes_removed: bytes_before - bytes_after,
        })
    }
}

pub(crate) fn compaction_root_changed(
    expected_catalog_root: Option<Hash>,
    current_catalog_root: Option<Hash>,
) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_compaction_root_changed",
        format!(
            "committed catalog changed before compaction: expected {expected_catalog_root:?}, current {current_catalog_root:?}"
        ),
    )
}

fn pin_count_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_pin_count_overflow",
        "in-flight chunk pin reference count overflowed",
    )
}

fn increment_pin_count(pins: &mut HashMap<Hash, usize>, address: Hash) -> GraphStoreResult<()> {
    let count = pins.entry(address).or_insert(0);
    *count = count.checked_add(1).ok_or_else(pin_count_overflow)?;
    Ok(())
}

fn decrement_pin_count(
    pins: &mut HashMap<Hash, usize>,
    address: Hash,
    state: &str,
) -> GraphStoreResult<()> {
    let count = pins.get_mut(&address).ok_or_else(|| {
        GraphStoreError::new(
            "chunk_store_pin_underflow",
            format!("address {address} has no {state} in-flight pin to release"),
        )
    })?;
    *count -= 1;
    if *count == 0 {
        pins.remove(&address);
    }
    Ok(())
}

fn pending_pin_count_in_state(
    pending: &HashMap<Hash, usize>,
    bound: &HashMap<String, BoundPendingPin>,
    address: Hash,
) -> GraphStoreResult<usize> {
    pending
        .get(&address)
        .copied()
        .unwrap_or(0)
        .checked_add(bound_pending_pin_count_in_state(bound, address)?)
        .ok_or_else(pin_count_overflow)
}

fn bound_pending_pin_count_in_state(
    bound: &HashMap<String, BoundPendingPin>,
    address: Hash,
) -> GraphStoreResult<usize> {
    bound
        .values()
        .filter(|candidate| candidate.address == address)
        .try_fold(0_usize, |total, candidate| {
            total
                .checked_add(candidate.count)
                .ok_or_else(pin_count_overflow)
        })
}

fn validate_publication_key(publication_key: &str) -> GraphStoreResult<()> {
    if publication_key.trim().is_empty() {
        return Err(GraphStoreError::new(
            "chunk_store_publication_key_invalid",
            "in-flight publication key must be non-empty",
        ));
    }
    Ok(())
}

fn pending_pin_missing(address: Hash, publication_key: &str) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_pending_pin_missing",
        format!("address {address} has no anonymous pending pin to bind as {publication_key:?}"),
    )
}

fn publication_key_address_conflict(
    publication_key: &str,
    bound_address: Hash,
    requested_address: Hash,
) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_publication_key_address_conflict",
        format!(
            "publication key {publication_key:?} is already bound to {bound_address}, not {requested_address}"
        ),
    )
}

fn release_bound_pending(
    bound: &mut HashMap<String, BoundPendingPin>,
    address: Hash,
    publication_key: &str,
) -> GraphStoreResult<bool> {
    let Some(receipt) = bound.get_mut(publication_key) else {
        return Ok(false);
    };
    if receipt.address != address {
        return Err(GraphStoreError::new(
            "chunk_store_pending_pin_address_mismatch",
            format!(
                "publication key {publication_key:?} is bound to {}, not {address}",
                receipt.address
            ),
        ));
    }
    receipt.count -= 1;
    if receipt.count == 0 {
        bound.remove(publication_key);
    }
    Ok(true)
}

fn checked_u32(value: usize, field: &str) -> GraphStoreResult<u32> {
    u32::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "chunk_store_canonical_length_overflow",
            format!("{field} length {value} exceeds the canonical u32 limit"),
        )
    })
}

pub(crate) fn merge_catalog_updates(
    current: Option<RootCatalog>,
    updates: &BTreeMap<String, Option<Hash>>,
) -> GraphStoreResult<RootCatalog> {
    let mut roots = current.map(|catalog| catalog.roots).unwrap_or_default();
    for (name, update) in updates {
        validate_root_name(name)?;
        match update {
            Some(root) => {
                roots.insert(name.clone(), *root);
            }
            None => {
                roots.remove(name);
            }
        }
    }
    RootCatalog::new(roots)
}

fn push_string(output: &mut Vec<u8>, value: &str, field: &str) -> GraphStoreResult<()> {
    output.extend_from_slice(&checked_u32(value.len(), field)?.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn validate_root_name(name: &str) -> GraphStoreResult<()> {
    if name.is_empty() {
        return Err(GraphStoreError::new(
            "chunk_store_catalog_name_invalid",
            "root catalog names cannot be empty",
        ));
    }
    checked_u32(name.len(), "root catalog name")?;
    Ok(())
}

fn ensure_address(expected: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual != expected {
        return Err(GraphStoreError::new(
            "chunk_store_address_mismatch",
            format!("chunk bytes hash to {actual}, expected {expected}"),
        ));
    }
    Ok(())
}

fn poisoned() -> GraphStoreError {
    GraphStoreError::new("chunk_store_poisoned", "chunk store lock poisoned")
}

fn chunk_accounting_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_size_overflow",
        "chunk write accounting exceeded the canonical usize limit",
    )
}

fn read_state(
    state: &RwLock<MemChunkStoreState>,
) -> GraphStoreResult<RwLockReadGuard<'_, MemChunkStoreState>> {
    state.read().map_err(|_| poisoned())
}

fn total_bytes<'a>(mut chunks: impl Iterator<Item = &'a Arc<[u8]>>) -> GraphStoreResult<u64> {
    chunks.try_fold(0_u64, |total, chunk| {
        let length = u64::try_from(chunk.len()).map_err(|_| {
            GraphStoreError::new(
                "chunk_store_size_overflow",
                "chunk length exceeds the canonical u64 accounting limit",
            )
        })?;
        total.checked_add(length).ok_or_else(|| {
            GraphStoreError::new(
                "chunk_store_size_overflow",
                "chunk byte accounting exceeded the canonical u64 limit",
            )
        })
    })
}

struct CanonicalDecoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    context: &'static str,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::{
        canonical_chunk_store_commit_hash, canonical_chunk_store_commit_preimage,
        ChunkPublicationWindow, ChunkStore, ChunkStoreCatalog, ChunkStoreCompaction,
        ChunkWriteReport, CommitMeta, CommittedRoot, MemChunkStore, RootCatalog,
        DOMAIN_ROOT_CATALOG, ROOT_NAME_DOCUMENTS, ROOT_NAME_GRAPH, ROOT_NAME_REDCORE,
    };
    use crate::graph::error::{GraphStoreError, GraphStoreResult};
    use crate::hash::Hash;
    use crate::versioned_graph::{
        canonical_commit_preimage, GraphCommit, GRAPH_CHUNK_FORMAT_VERSION,
    };

    fn chunk_store_test_meta() -> CommitMeta {
        CommitMeta {
            parents: vec![Hash::of(b"parent-1"), Hash::of(b"parent-2")],
            author: "Ada".to_string(),
            message: "persist roots".to_string(),
            timestamp_unix_ms: 0x1020_3040_5060_7080_90a0_b0c0_d0e0_f000,
            graph_version: 42,
        }
    }

    fn chunk_store_test_write(store: &dyn ChunkStore, bytes: &[u8]) -> Hash {
        let address = Hash::of(bytes);
        store.write(&[(address, bytes)]).unwrap();
        address
    }

    fn chunk_store_test_catalog(store: &dyn ChunkStore) -> (RootCatalog, Hash, Hash, Hash) {
        let graph = chunk_store_test_write(store, b"graph root bytes");
        let documents = chunk_store_test_write(store, b"document root bytes");
        let catalog = RootCatalog::new(BTreeMap::from([
            (ROOT_NAME_DOCUMENTS.to_string(), documents),
            (ROOT_NAME_GRAPH.to_string(), graph),
        ]))
        .unwrap();
        let catalog_address = catalog.write_to(store).unwrap();
        (catalog, catalog_address, graph, documents)
    }

    struct FailOnceReleaseStore {
        inner: MemChunkStore,
        fail_next_release: AtomicBool,
    }

    impl FailOnceReleaseStore {
        fn new() -> Self {
            Self {
                inner: MemChunkStore::new(),
                fail_next_release: AtomicBool::new(true),
            }
        }
    }

    impl ChunkStore for FailOnceReleaseStore {
        fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
            self.inner.get(address)
        }

        fn has(&self, address: Hash) -> GraphStoreResult<bool> {
            self.inner.has(address)
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

        fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
            self.inner.commit(root, meta)
        }

        fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
            self.inner.read_root()
        }

        fn pin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
            self.inner.pin_in_flight(address)
        }

        fn bind_pending_in_flight(
            &self,
            address: Hash,
            publication_key: &str,
        ) -> GraphStoreResult<()> {
            self.inner.bind_pending_in_flight(address, publication_key)
        }

        fn release_bound_pending_in_flight(
            &self,
            address: Hash,
            publication_key: &str,
        ) -> GraphStoreResult<bool> {
            if self.fail_next_release.swap(false, Ordering::SeqCst) {
                return Err(GraphStoreError::new(
                    "chunk_store_test_release_failure",
                    "injected post-publication receipt release failure",
                ));
            }
            self.inner
                .release_bound_pending_in_flight(address, publication_key)
        }

        fn claim_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
            self.inner.claim_in_flight(address)
        }

        fn release_claimed_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
            self.inner.release_claimed_in_flight(address)
        }

        fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
            self.inner.unpin_in_flight(address)
        }
    }

    #[test]
    fn publication_window_cleanup_failure_after_durable_root_is_retried_without_failing_commit() {
        let store = Arc::new(FailOnceReleaseStore::new());
        let publication =
            ChunkPublicationWindow::new(Arc::clone(&store), "post-commit-cleanup-oracle");
        let graph_bytes = b"durably published graph";
        let graph = Hash::of(graph_bytes);
        publication.write(&[(graph, graph_bytes)]).unwrap();
        let catalog = RootCatalog::graph(graph);
        let catalog_bytes = catalog.to_bytes().unwrap();
        let catalog_root = Hash::of(&catalog_bytes);
        publication
            .write(&[(catalog_root, catalog_bytes.as_slice())])
            .unwrap();

        let commit = store.commit(catalog_root, chunk_store_test_meta()).unwrap();
        assert_eq!(
            store.inner.committed_root().unwrap().unwrap().commit,
            commit
        );
        assert_eq!(store.inner.in_flight_pin_count(graph).unwrap(), 1);
        assert_eq!(store.inner.in_flight_pin_count(catalog_root).unwrap(), 1);

        // The first cleanup attempt fails after the root is durable. The
        // infallible finish keeps the publication outcome successful and Drop
        // immediately retries the still-armed receipt set.
        publication.finish_after_publication();

        assert_eq!(store.read_root().unwrap(), Some(catalog_root));
        assert_eq!(store.inner.in_flight_pin_count(graph).unwrap(), 0);
        assert_eq!(store.inner.in_flight_pin_count(catalog_root).unwrap(), 0);
    }

    #[test]
    fn chunk_store_commit_preimage_cross_matches_phase_c_and_golden() {
        let root = Hash::of(b"root catalog");
        let meta = chunk_store_test_meta();
        let preimage = canonical_chunk_store_commit_preimage(root, &meta).unwrap();
        let graph_commit = GraphCommit {
            commit_hash: Hash::of(b"ignored commit address"),
            tree_hash: root,
            branch: "ignored-branch".to_string(),
            parent_commits: meta.parents.clone(),
            author: meta.author.clone(),
            message: meta.message.clone(),
            timestamp_unix_ms: meta.timestamp_unix_ms,
            graph_version: meta.graph_version,
            historical_registry_hash: None,
            objects_total: usize::MAX,
            compiler_version: "ignored-compiler".to_string(),
        };

        assert_eq!(preimage, canonical_commit_preimage(&graph_commit).unwrap());
        assert_eq!(preimage[0], 4);
        assert_eq!(preimage[1], GRAPH_CHUNK_FORMAT_VERSION);
        assert_eq!(
            canonical_chunk_store_commit_hash(root, &meta)
                .unwrap()
                .to_prefixed(),
            "sha256:f98c1cb88fa5a729a943229e3efe29e299a40ae592841eb7cc72552f072d29de"
        );

        let committed = CommittedRoot::new(root, meta).unwrap();
        assert_eq!(CommittedRoot::from_bytes(&preimage).unwrap(), committed);
        committed.verify_commit(committed.commit).unwrap();
        assert!(committed.verify_commit(Hash::of(b"wrong commit")).is_err());
        let mut tampered = committed.clone();
        tampered.meta.message.push_str(" after address");
        assert_eq!(
            tampered.verify_commit(tampered.commit).unwrap_err().code,
            "chunk_store_commit_payload_mismatch"
        );
        assert!(tampered.to_bytes().is_err());
    }

    #[test]
    fn chunk_store_root_catalog_is_sorted_strict_and_golden() {
        let graph = Hash::of(b"graph");
        let documents = Hash::of(b"documents");
        let catalog = RootCatalog::new(BTreeMap::from([
            (ROOT_NAME_GRAPH.to_string(), graph),
            (ROOT_NAME_DOCUMENTS.to_string(), documents),
        ]))
        .unwrap();
        let bytes = catalog.to_bytes().unwrap();

        assert_eq!(bytes[0], DOMAIN_ROOT_CATALOG);
        assert_eq!(bytes[1], GRAPH_CHUNK_FORMAT_VERSION);
        assert_eq!(RootCatalog::from_bytes(&bytes).unwrap(), catalog);
        assert_eq!(
            catalog.address().unwrap().to_prefixed(),
            "sha256:ffec51267ca417d41a27d067a06d2152ebcb2154dd241778afe731702fb84ed6"
        );

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(RootCatalog::from_bytes(&trailing).is_err());

        // Swap the two complete entries to prove the decoder rejects a valid
        // but non-canonical ordering rather than silently re-sorting it.
        let header_len = 6;
        let first_len = 4 + ROOT_NAME_DOCUMENTS.len() + 32;
        let mut unsorted = bytes[..header_len].to_vec();
        unsorted.extend_from_slice(&bytes[header_len + first_len..]);
        unsorted.extend_from_slice(&bytes[header_len..header_len + first_len]);
        assert!(RootCatalog::from_bytes(&unsorted).is_err());
    }

    #[test]
    fn chunk_store_batch_mismatch_is_atomic_and_reads_are_verified() {
        let store = MemChunkStore::new();
        let good_bytes = b"good bytes";
        let good = Hash::of(good_bytes);
        let bad_bytes = b"bad bytes";
        let stated_bad = Hash::of(b"different bytes");

        let error = store
            .write(&[(good, good_bytes), (stated_bad, bad_bytes)])
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_address_mismatch");
        assert!(!store.has(good).unwrap());

        store.write(&[(good, good_bytes)]).unwrap();
        store
            .state
            .write()
            .unwrap()
            .chunks
            .insert(good, Arc::from(b"corrupt".as_slice()));
        assert_eq!(
            store.get(good).unwrap_err().code,
            "chunk_store_address_mismatch"
        );
    }

    #[test]
    fn chunk_store_missing_deduplicates_and_preserves_caller_order() {
        let store = MemChunkStore::new();
        let present = chunk_store_test_write(&store, b"present");
        let missing_one = Hash::of(b"missing one");
        let missing_two = Hash::of(b"missing two");

        assert_eq!(
            store
                .missing(&[missing_two, present, missing_one, missing_two, missing_one])
                .unwrap(),
            vec![missing_two, missing_one]
        );
    }

    #[test]
    fn chunk_store_accounted_write_reports_only_new_unique_bytes() {
        let store = MemChunkStore::new();
        let first_bytes = b"accounted first";
        let second_bytes = b"accounted second";
        let first = Hash::of(first_bytes);
        let second = Hash::of(second_bytes);

        let initial = store
            .write_accounted(&[(first, first_bytes), (first, first_bytes)])
            .unwrap();
        assert_eq!(initial.chunks_written, 1);
        assert_eq!(initial.bytes_written, first_bytes.len());

        let delta = store
            .write_accounted(&[(first, first_bytes), (second, second_bytes)])
            .unwrap();
        assert_eq!(delta.chunks_written, 1);
        assert_eq!(delta.bytes_written, second_bytes.len());
    }

    #[test]
    fn chunk_store_commit_rejects_missing_or_graphless_catalog_roots() {
        let store = MemChunkStore::new();
        let missing_catalog = Hash::of(b"missing catalog");
        assert_eq!(
            store
                .commit(missing_catalog, chunk_store_test_meta())
                .unwrap_err()
                .code,
            "chunk_store_catalog_missing"
        );

        let documents = chunk_store_test_write(&store, b"documents only");
        let graphless = RootCatalog::new(BTreeMap::from([(
            ROOT_NAME_DOCUMENTS.to_string(),
            documents,
        )]))
        .unwrap();
        let graphless_address = graphless.write_to(&store).unwrap();
        assert_eq!(
            store
                .commit(graphless_address, chunk_store_test_meta())
                .unwrap_err()
                .code,
            "chunk_store_catalog_graph_root_missing"
        );

        let missing_graph = Hash::of(b"absent graph root");
        let catalog = RootCatalog::graph(missing_graph);
        let catalog_address = catalog.write_to(&store).unwrap();
        assert_eq!(
            store
                .commit(catalog_address, chunk_store_test_meta())
                .unwrap_err()
                .code,
            "chunk_store_named_root_missing"
        );
        assert_eq!(store.read_root().unwrap(), None);
    }

    #[test]
    fn chunk_store_arc_dyn_supports_concurrent_writes_and_commit() {
        let store: Arc<dyn ChunkStore> = Arc::new(MemChunkStore::new());
        let graph_bytes: Arc<[u8]> = Arc::from(b"thread graph root".as_slice());
        let document_bytes: Arc<[u8]> = Arc::from(b"thread document root".as_slice());
        let graph = Hash::of(&graph_bytes);
        let documents = Hash::of(&document_bytes);

        let graph_writer = {
            let store = Arc::clone(&store);
            let graph_bytes = Arc::clone(&graph_bytes);
            thread::spawn(move || store.write(&[(graph, &graph_bytes)]))
        };
        let document_writer = {
            let store = Arc::clone(&store);
            let document_bytes = Arc::clone(&document_bytes);
            thread::spawn(move || store.write(&[(documents, &document_bytes)]))
        };
        graph_writer.join().unwrap().unwrap();
        document_writer.join().unwrap().unwrap();

        let catalog = RootCatalog::new(BTreeMap::from([
            (ROOT_NAME_DOCUMENTS.to_string(), documents),
            (ROOT_NAME_GRAPH.to_string(), graph),
        ]))
        .unwrap();
        let catalog_address = catalog.write_to(store.as_ref()).unwrap();
        let commit = store
            .commit(catalog_address, chunk_store_test_meta())
            .unwrap();

        assert_eq!(store.read_root().unwrap(), Some(catalog_address));
        assert_eq!(
            store.get(graph).unwrap().unwrap().as_ref(),
            graph_bytes.as_ref()
        );
        assert_eq!(
            store.get(documents).unwrap().unwrap().as_ref(),
            document_bytes.as_ref()
        );
        // commit() derives lineage from the prior committed root and ignores
        // caller-supplied parents, so the first commit in a fresh store is
        // parentless regardless of what the caller passed.
        let mut committed_meta = chunk_store_test_meta();
        committed_meta.parents = Vec::new();
        assert_eq!(
            commit,
            canonical_chunk_store_commit_hash(catalog_address, &committed_meta).unwrap()
        );
    }

    #[test]
    fn redcore_journal_catalog_extension_merges_mem_updates_without_lost_names() {
        let store = Arc::new(MemChunkStore::new());
        let graph = chunk_store_test_write(store.as_ref(), b"mem graph root");
        let documents = chunk_store_test_write(store.as_ref(), b"mem document root");
        let redcore = chunk_store_test_write(store.as_ref(), b"mem redcore root");
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_test_meta(),
            )
            .unwrap();

        let document_store = Arc::clone(&store);
        let document_thread = thread::spawn(move || {
            document_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_DOCUMENTS.to_string(), Some(documents))]),
                chunk_store_test_meta(),
            )
        });
        let redcore_store = Arc::clone(&store);
        let redcore_thread = thread::spawn(move || {
            redcore_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(redcore))]),
                chunk_store_test_meta(),
            )
        });
        let document_commit = document_thread.join().unwrap().unwrap();
        let redcore_commit = redcore_thread.join().unwrap().unwrap();

        let root = store.read_root().unwrap().unwrap();
        let committed = store.committed_root().unwrap().unwrap();
        let immediately_previous = if committed.commit == document_commit.commit {
            redcore_commit.commit
        } else {
            assert_eq!(committed.commit, redcore_commit.commit);
            document_commit.commit
        };
        assert_eq!(committed.meta.parents, vec![immediately_previous]);
        let catalog = RootCatalog::from_bytes(&store.get(root).unwrap().unwrap()).unwrap();
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph));
        assert_eq!(catalog.get(ROOT_NAME_DOCUMENTS), Some(documents));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), Some(redcore));

        let removal = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), None)]),
                chunk_store_test_meta(),
            )
            .unwrap();
        let catalog = RootCatalog::from_bytes(
            &store
                .get(removal.catalog_root)
                .unwrap()
                .expect("removed catalog"),
        )
        .unwrap();
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph));
        assert_eq!(catalog.get(ROOT_NAME_DOCUMENTS), Some(documents));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), None);
    }

    #[test]
    fn chunk_store_compaction_refuses_current_catalog_or_named_root_omission() {
        let store = MemChunkStore::new();
        let (_catalog, catalog_address, graph, documents) = chunk_store_test_catalog(&store);
        let orphan = chunk_store_test_write(&store, b"unreachable orphan");
        store
            .commit(catalog_address, chunk_store_test_meta())
            .unwrap();

        for incomplete in [vec![graph, documents], vec![catalog_address, graph]] {
            assert_eq!(
                store
                    .compact_reachable(Some(catalog_address), &incomplete)
                    .unwrap_err()
                    .code,
                "chunk_store_compaction_live_root_omitted"
            );
            assert!(store.has(orphan).unwrap());
        }

        let report = store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert_eq!(report.chunks_before, 4);
        assert_eq!(report.chunks_after, 3);
        assert_eq!(report.chunks_removed, 1);
        assert!(!store.has(orphan).unwrap());
        assert_eq!(store.read_root().unwrap(), Some(catalog_address));
    }

    #[test]
    fn mem_chunk_store_compaction_honors_keyed_ref_counted_in_flight_pins() {
        let store = MemChunkStore::new();
        let (_catalog, catalog_address, graph, documents) = chunk_store_test_catalog(&store);
        store
            .commit(catalog_address, chunk_store_test_meta())
            .unwrap();
        let payload_bytes = b"payload awaiting graph publication";
        let payload = Hash::of(payload_bytes);
        let node_a_key = "node-a:tensor_payload:sha256:shared";
        let node_b_key = "node-b:tensor_payload:sha256:shared";

        // Pinning precedes the raw write. Compaction in this exact interval is
        // valid and must not reject a pin whose bytes have not arrived yet.
        store.pin_in_flight(payload).unwrap();
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        store.write(&[(payload, payload_bytes)]).unwrap();
        store.pin_in_flight(payload).unwrap();
        store.bind_pending_in_flight(payload, node_b_key).unwrap();
        store.bind_pending_in_flight(payload, node_b_key).unwrap();
        assert_eq!(store.pending_pin_count(payload).unwrap(), 2);
        assert_eq!(store.anonymous_pending_pin_count(payload).unwrap(), 0);
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 2);
        assert_eq!(store.claimed_pin_count(payload).unwrap(), 0);
        assert_eq!(
            store.unpin_in_flight(payload).unwrap_err().code,
            "chunk_store_pin_underflow"
        );
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 2);

        // An idempotent upsert for node A creates and releases its own claim;
        // it cannot consume node B's pending receipts despite sharing a hash.
        store.claim_in_flight(payload).unwrap();
        assert_eq!(store.pending_pin_count(payload).unwrap(), 2);
        assert_eq!(store.claimed_pin_count(payload).unwrap(), 1);
        store.release_claimed_in_flight(payload).unwrap();
        assert!(!store
            .release_bound_pending_in_flight(payload, node_a_key)
            .unwrap());
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 2);

        // Releasing one of two identical logical writes preserves the other.
        assert!(store
            .release_bound_pending_in_flight(payload, node_b_key)
            .unwrap());
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 1);
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(store.has(payload).unwrap());
        assert!(store
            .release_bound_pending_in_flight(payload, node_b_key)
            .unwrap());
        assert_eq!(store.in_flight_pin_count(payload).unwrap(), 0);
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(!store.has(payload).unwrap());

        // An external reference with no producer receipt creates a claim. A
        // later pending producer can abort without consuming that claim.
        let external_bytes = b"externally supplied payload reference";
        let external = Hash::of(external_bytes);
        store.write(&[(external, external_bytes)]).unwrap();
        store.claim_in_flight(external).unwrap();
        store.pin_in_flight(external).unwrap();
        store.unpin_in_flight(external).unwrap();
        assert_eq!(store.pending_pin_count(external).unwrap(), 0);
        assert_eq!(store.claimed_pin_count(external).unwrap(), 1);
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(store.has(external).unwrap());
        store.release_claimed_in_flight(external).unwrap();
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(!store.has(external).unwrap());

        // A logical publication key cannot be rebound to another address. The
        // failed attempt preserves both the old bound receipts and the new
        // writer's still-anonymous receipt for explicit cleanup.
        let old_bytes = b"superseded logical publication";
        let old = Hash::of(old_bytes);
        let new_bytes = b"replacement logical publication";
        let new = Hash::of(new_bytes);
        let rebinding_key = "node-c:tensor_payload:replacement";
        store.write(&[(old, old_bytes), (new, new_bytes)]).unwrap();
        store.pin_in_flight(old).unwrap();
        store.bind_pending_in_flight(old, rebinding_key).unwrap();
        store.pin_in_flight(old).unwrap();
        store.bind_pending_in_flight(old, rebinding_key).unwrap();
        assert_eq!(store.bound_pending_pin_count(old).unwrap(), 2);
        store.pin_in_flight(new).unwrap();
        assert_eq!(
            store
                .bind_pending_in_flight(new, rebinding_key)
                .unwrap_err()
                .code,
            "chunk_store_publication_key_address_conflict"
        );
        assert_eq!(store.bound_pending_pin_count(old).unwrap(), 2);
        assert_eq!(store.anonymous_pending_pin_count(new).unwrap(), 1);
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(store.has(old).unwrap());
        assert!(store.has(new).unwrap());
        store.unpin_in_flight(new).unwrap();
        store
            .release_bound_pending_in_flight(old, rebinding_key)
            .unwrap();
        store
            .release_bound_pending_in_flight(old, rebinding_key)
            .unwrap();
        store
            .compact_reachable(Some(catalog_address), &[catalog_address, graph, documents])
            .unwrap();
        assert!(!store.has(old).unwrap());
        assert!(!store.has(new).unwrap());
    }
}

impl<'a> CanonicalDecoder<'a> {
    fn new(bytes: &'a [u8], context: &'static str) -> Self {
        Self {
            bytes,
            offset: 0,
            context,
        }
    }

    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| self.invalid(field))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| self.invalid(field))?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn expect_byte(&mut self, expected: u8, field: &str) -> GraphStoreResult<()> {
        let actual = self.byte(field)?;
        if actual != expected {
            return Err(GraphStoreError::new(
                "chunk_store_canonical_decode",
                format!(
                    "{} has invalid {field}: expected {expected}, found {actual}",
                    self.context
                ),
            ));
        }
        Ok(())
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(self.take(4, field)?);
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(self.take(8, field)?);
        Ok(u64::from_le_bytes(raw))
    }

    fn u128(&mut self, field: &str) -> GraphStoreResult<u128> {
        let mut raw = [0_u8; 16];
        raw.copy_from_slice(self.take(16, field)?);
        Ok(u128::from_le_bytes(raw))
    }

    fn hash(&mut self, field: &str) -> GraphStoreResult<Hash> {
        let mut raw = [0_u8; 32];
        raw.copy_from_slice(self.take(32, field)?);
        Ok(Hash::from_bytes(raw))
    }

    fn count(&mut self, field: &str, minimum_item_bytes: usize) -> GraphStoreResult<usize> {
        let count = usize::try_from(self.u32(field)?).map_err(|_| self.invalid(field))?;
        if count > self.remaining() / minimum_item_bytes {
            return Err(self.invalid(field));
        }
        Ok(count)
    }

    fn string(&mut self, field: &str) -> GraphStoreResult<String> {
        let length = usize::try_from(self.u32(field)?).map_err(|_| self.invalid(field))?;
        let bytes = self.take(length, field)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| self.invalid(field))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn finish(self) -> GraphStoreResult<()> {
        if self.offset != self.bytes.len() {
            return Err(GraphStoreError::new(
                "chunk_store_canonical_decode",
                format!("{} has trailing bytes", self.context),
            ));
        }
        Ok(())
    }

    fn invalid(&self, field: &str) -> GraphStoreError {
        GraphStoreError::new(
            "chunk_store_canonical_decode",
            format!("{} is truncated or invalid at {field}", self.context),
        )
    }
}
