//! Typed lenses over the verified content-addressed chunk store.
//!
//! Documents, filesystem chunks, graph recovery payloads, and parked pack
//! envelopes all live in one physical [`ChunkStore`] address space. The
//! filesystem layout retained by [`DiskObjectStore`] is read-only migration
//! input; current writes never create per-tier object files.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::chunk_store::{
    validate_catalog_root, CatalogCommitResult, ChunkPublicationWindow, ChunkStore,
    ChunkStoreCatalog, CommitMeta, ROOT_NAME_DOCUMENTS, ROOT_NAME_GRAPH, ROOT_NAME_OBJECTS,
};
use crate::file_payload::{read_file_payload, FileManifestRef};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::journal::JournalChunkStore;
use crate::reachability::packed_graph_node_payload_references;
use crate::symbolic::canonical_json;
use crate::versioned_graph::{
    canonical_graph_payload_preimage, canonical_leaf_preimage, decode_canonical_graph_payload,
    edge_to_content_object, graph_pack_chunk_format, legacy_graph_pack_v1_parents,
    migrate_legacy_graph_content_object_v1, migrate_legacy_graph_pack_v1, node_to_content_object,
    recompute_commit_hash, recompute_content_object_hash, CompiledGraphPack, GraphContentObject,
    GraphObjectKind, GraphPayloadRecord, LegacyGraphObjectMigration, LegacyGraphPackMigration,
    GRAPH_CHUNK_FORMAT_VERSION, LEGACY_GRAPH_CHUNK_FORMAT_VERSION,
};

const ADDRESS_HEX_LENGTH: usize = 64;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];
const UNIFIED_JOURNAL_FILE: &str = "chunk-store.journal";
const DOMAIN_OBJECT_STORE_INDEX: u8 = 10;
const LEGACY_OBJECT_STORE_INDEX_VERSION: u8 = 1;
const BLAKE3_OBJECT_STORE_INDEX_VERSION: u8 = 2;
const OBJECT_STORE_INDEX_VERSION: u8 = 3;
const DOMAIN_PACK_ENVELOPE: u8 = 11;
const PACK_ENVELOPE_VERSION: u8 = 1;
const DOMAIN_BLAKE3_ALIAS: u8 = 12;
const BLAKE3_ALIAS_VERSION: u8 = 1;
const DOMAIN_PRIVATE_MANIFEST_SLOT: u8 = 14;
const DOMAIN_PRIVATE_MANIFEST_SLOT_IDENTITY: u8 = 15;
const PRIVATE_MANIFEST_SLOT_VERSION: u8 = 1;
const MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES: usize = 64 * 1024;
const MAX_LEGACY_IMPORT_ADDRESSES: usize = 1_000_000;
const MAX_LEGACY_PACK_PARENT_DEPTH: usize = 256;

/// Object-safe backend required by the typed object-store facade.
///
/// `as_chunk_store` avoids relying on trait-object upcasting when calling
/// helpers which intentionally accept the minimal [`ChunkStore`] contract.
pub trait UnifiedChunkStore: ChunkStore + ChunkStoreCatalog {
    fn as_chunk_store(&self) -> &dyn ChunkStore;
}

impl<T> UnifiedChunkStore for T
where
    T: ChunkStore + ChunkStoreCatalog + 'static,
{
    fn as_chunk_store(&self) -> &dyn ChunkStore {
        self
    }
}

/// Durable, content-addressed home for the cold tail.
///
/// Keys are object hashes (cold per-node tier) and commit hashes (warm per-scope
/// tier). Implementations must survive an operating-store restart -- that is the
/// whole point of the tier -- so a purely in-RAM impl is for tests only.
pub trait ColdObjectStore: Send + Sync {
    /// Persist a content object under its hash. Idempotent: writing the same
    /// hash twice is a no-op (content addressing -> identical bytes).
    fn put_object(&self, object: &GraphContentObject) -> GraphStoreResult<()>;

    /// Fetch a content object by hash. `Ok(None)` when absent.
    fn get_object(&self, hash: Hash) -> GraphStoreResult<Option<GraphContentObject>>;

    /// Load an object while preserving an explicit old->new identity receipt
    /// when a standalone v1 cold object is re-encoded in the current format.
    fn load_object(&self, hash: Hash) -> GraphStoreResult<Option<LoadedColdObject>> {
        self.get_object(hash).map(|object| {
            object.map(|object| LoadedColdObject {
                object,
                migrated_from: None,
            })
        })
    }

    /// Persist a compiled pack (a parked warm scope) under its commit hash.
    fn put_pack(&self, pack: &CompiledGraphPack) -> GraphStoreResult<()>;

    /// Fetch a parked pack by commit hash. `Ok(None)` when absent.
    fn get_pack(&self, commit_hash: Hash) -> GraphStoreResult<Option<CompiledGraphPack>>;

    /// Load a pack while preserving an explicit old->new identity receipt when
    /// a durable v1 cold pack is semantically recompiled to the current format.
    /// Ordinary stores delegate to `get_pack`; the disk facade overrides this
    /// for its read-only legacy source.
    fn load_pack(&self, commit_hash: Hash) -> GraphStoreResult<Option<LoadedColdPack>> {
        self.get_pack(commit_hash).map(|pack| {
            pack.map(|pack| LoadedColdPack {
                pack,
                migrated_from: None,
            })
        })
    }

    /// Number of distinct content objects held (diagnostics / RAM accounting).
    fn object_count(&self) -> GraphStoreResult<usize>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedColdPack {
    pub pack: CompiledGraphPack,
    pub migrated_from: Option<Hash>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedColdObject {
    pub object: GraphContentObject,
    pub migrated_from: Option<Hash>,
}

/// In-RAM cold store. For tests and the in-process default only -- it does NOT
/// survive a restart, so it cannot satisfy the durability acceptance criterion;
/// use [`DiskObjectStore`] (or the Postgres catalog) for a real cold tier.
#[derive(Clone, Debug, Default)]
pub struct InMemoryObjectStore {
    objects: Arc<Mutex<BTreeMap<Hash, GraphContentObject>>>,
    packs: Arc<Mutex<BTreeMap<Hash, CompiledGraphPack>>>,
    chunks: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
}

impl InMemoryObjectStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a content chunk under `sha256:<hex>`. Mirrors [`DiskObjectStore::put_chunk`]
    /// for in-process tests (no durability across process restart).
    pub fn put_chunk(&self, bytes: &[u8]) -> GraphStoreResult<String> {
        let digest = content_hash_bytes(bytes);
        self.chunks
            .lock()
            .map_err(poisoned)?
            .entry(digest.clone())
            .or_insert_with(|| bytes.to_vec());
        Ok(digest)
    }

    /// Fetch a chunk by its `sha256:<hex>` ref, or `None` if absent.
    pub fn get_chunk(&self, digest: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        if address_hex(digest, AddressAlgorithm::Sha256).is_none() {
            return Ok(None);
        }
        let Some(bytes) = self.chunks.lock().map_err(poisoned)?.get(digest).cloned() else {
            return Ok(None);
        };
        verify_bytes_address(digest, &content_hash_bytes(&bytes), "chunk")?;
        Ok(Some(bytes))
    }

    /// Number of distinct in-memory chunks (diagnostics / dedup oracles).
    pub fn chunk_count(&self) -> GraphStoreResult<usize> {
        Ok(self.chunks.lock().map_err(poisoned)?.len())
    }
}

impl ColdObjectStore for InMemoryObjectStore {
    fn put_object(&self, object: &GraphContentObject) -> GraphStoreResult<()> {
        verify_object_address(object.hash, object)?;
        self.objects
            .lock()
            .map_err(poisoned)?
            .insert(object.hash, object.clone());
        Ok(())
    }

    fn get_object(&self, hash: Hash) -> GraphStoreResult<Option<GraphContentObject>> {
        let Some(object) = self.objects.lock().map_err(poisoned)?.get(&hash).cloned() else {
            return Ok(None);
        };
        verify_object_address(hash, &object)?;
        Ok(Some(object))
    }

    fn put_pack(&self, pack: &CompiledGraphPack) -> GraphStoreResult<()> {
        verify_pack_address(pack.commit.commit_hash, pack)?;
        self.packs
            .lock()
            .map_err(poisoned)?
            .insert(pack.commit.commit_hash, pack.clone());
        Ok(())
    }

    fn get_pack(&self, commit_hash: Hash) -> GraphStoreResult<Option<CompiledGraphPack>> {
        let Some(pack) = self
            .packs
            .lock()
            .map_err(poisoned)?
            .get(&commit_hash)
            .cloned()
        else {
            return Ok(None);
        };
        verify_pack_address(commit_hash, &pack)?;
        Ok(Some(pack))
    }

    fn object_count(&self) -> GraphStoreResult<usize> {
        Ok(self.objects.lock().map_err(poisoned)?.len())
    }
}

/// Arc-backed typed facade over one physical chunk store.
#[derive(Clone)]
pub struct DiskObjectStore {
    store: Arc<dyn UnifiedChunkStore>,
    legacy_root: Option<PathBuf>,
    diagnostic_root: PathBuf,
    index_update: Arc<Mutex<()>>,
    observed_raw: Arc<Mutex<BTreeMap<Hash, ()>>>,
    #[cfg(test)]
    before_catalog_commit: Arc<Mutex<Option<BeforeCatalogCommitHook>>>,
}

#[cfg(test)]
type BeforeCatalogCommitHook = Box<dyn FnOnce() -> GraphStoreResult<()> + Send>;

/// Scoped ownership of raw bytes staged for a graph/catalog publication.
///
/// The payload stays pinned against concurrent reachability compaction until
/// [`Self::finish_after_publication`] is called after the owning transaction's
/// durable root commit. Dropping the receipt before then aborts the pin.
pub struct StagedRawPayload {
    address: Hash,
    publication: Option<ChunkPublicationWindow<dyn UnifiedChunkStore>>,
}

impl std::fmt::Debug for StagedRawPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StagedRawPayload")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

impl StagedRawPayload {
    pub fn address(&self) -> Hash {
        self.address
    }

    pub fn content_id(&self) -> String {
        self.address.to_prefixed()
    }

    pub fn finish_after_publication(mut self) {
        self.publication
            .take()
            .expect("staged raw payload owns one publication window")
            .finish_after_publication();
    }
}

impl std::fmt::Debug for DiskObjectStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DiskObjectStore")
            .field("legacy_root", &self.legacy_root)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexedPayloadKind {
    Object = 1,
    Pack = 2,
    PrivateManifestSlot = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexedPayload {
    kind: IndexedPayloadKind,
    physical: Hash,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Blake3Digest([u8; 32]);

impl Blake3Digest {
    fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Blake3Alias {
    legacy: Blake3Digest,
    canonical: Hash,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ObjectStoreIndex {
    payloads: BTreeMap<Hash, IndexedPayload>,
    /// Blake3 aliases have their own key namespace so a legacy digest cannot
    /// collide with a semantic SHA-256 object or pack address.
    blake3_aliases: BTreeMap<Blake3Digest, Hash>,
}

/// One stable private semantic slot whose current immutable value references a
/// strict file manifest.
///
/// `domain`, `scope`, and `key` define the stable semantic identity. Publishing
/// a new `manifest_address` atomically retargets that identity in the object
/// index while leaving the old physical envelope immutable and unreachable.
/// This is intentionally not a graph node or filesystem namespace entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateManifestSlot {
    pub domain: String,
    pub scope: String,
    pub key: String,
    pub manifest_address: Hash,
}

impl PrivateManifestSlot {
    pub fn new(
        domain: impl Into<String>,
        scope: impl Into<String>,
        key: impl Into<String>,
        manifest_address: Hash,
    ) -> GraphStoreResult<Self> {
        let slot = Self {
            domain: domain.into(),
            scope: scope.into(),
            key: key.into(),
            manifest_address,
        };
        slot.validate()?;
        Ok(slot)
    }

    /// Content address of the stable `(domain, scope, key)` identity. The
    /// current manifest deliberately does not participate in this address.
    pub fn semantic_address(&self) -> GraphStoreResult<Hash> {
        encode_private_manifest_slot_identity(&self.domain, &self.scope, &self.key)
            .map(|bytes| Hash::of(&bytes))
    }

    fn validate(&self) -> GraphStoreResult<()> {
        for (field, value) in [
            ("domain", self.domain.as_str()),
            ("scope", self.scope.as_str()),
            ("key", self.key.as_str()),
        ] {
            if value.is_empty() || value.len() > MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES {
                return Err(GraphStoreError::new(
                    "cold_private_manifest_slot_invalid",
                    format!(
                        "private manifest slot {field} must contain 1..={MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES} UTF-8 bytes"
                    ),
                ));
            }
        }
        Ok(())
    }
}

type SharedIndexUpdate = Arc<Mutex<()>>;
type WeakIndexUpdate = Weak<Mutex<()>>;

static INDEX_UPDATE_LOCKS: OnceLock<Mutex<BTreeMap<usize, WeakIndexUpdate>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ObjectStoreReachability {
    Index {
        object_payloads: Vec<Hash>,
        pack_envelopes: Vec<Hash>,
        private_manifest_envelopes: Vec<Hash>,
    },
    PackEnvelope {
        logical_roots: Vec<Hash>,
        payload_roots: Vec<Hash>,
        object_payloads: Vec<Hash>,
        file_manifests: Vec<Hash>,
        terminals: Vec<Hash>,
    },
    PrivateManifestEnvelope {
        manifest: Hash,
    },
}

impl DiskObjectStore {
    /// Open an owned journal inside `root`. Existing per-tier directories are
    /// treated solely as read-only migration input.
    pub fn open(root: impl AsRef<Path>) -> GraphStoreResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io_err("create unified object-store root"))?;
        let journal: Arc<dyn UnifiedChunkStore> =
            Arc::new(JournalChunkStore::open(root.join(UNIFIED_JOURNAL_FILE))?);
        Self::from_parts(journal, Some(root.clone()), root, true)
    }

    /// Open the former standalone object journal only as one-time migration
    /// input. Returns `Ok(None)` when the historical journal is absent.
    ///
    /// This does not initialize or advance its root catalog. New filesystem
    /// writes must target a facade built from RedCore's
    /// `unified_chunk_store()` handle instead.
    pub fn open_legacy_journal_for_import(
        root: impl AsRef<Path>,
    ) -> GraphStoreResult<Option<Self>> {
        let root = root.as_ref().to_path_buf();
        let journal_path = root.join(UNIFIED_JOURNAL_FILE);
        if !journal_path.is_file() {
            return Ok(None);
        }
        let journal: Arc<dyn UnifiedChunkStore> = Arc::new(JournalChunkStore::open(journal_path)?);
        Self::from_parts(journal, Some(root.clone()), root, false).map(Some)
    }

    /// Remove only the retired `documents` named root after its files have
    /// become durable graph `File` nodes in `unified`.
    ///
    /// The destination catalog must still name `expected_graph_root`, and each
    /// imported file-manifest address must be readable there. The legacy
    /// journal's graph/object/RedCore roots and every physical chunk are left
    /// untouched; only `ROOT_NAME_DOCUMENTS` is removed by a merged catalog
    /// update. `Ok(None)` means that root was already absent.
    pub fn clear_legacy_documents_root_after_graph_import(
        &self,
        unified: &dyn UnifiedChunkStore,
        expected_graph_root: Hash,
        imported_manifest_addresses: &[Hash],
        meta: CommitMeta,
    ) -> GraphStoreResult<Option<CatalogCommitResult>> {
        prove_unified_graph_import(unified, expected_graph_root, imported_manifest_addresses)?;

        let Some(legacy_catalog_address) = self.store.read_root()? else {
            return Ok(None);
        };
        let legacy_catalog =
            validate_catalog_root(self.store.as_chunk_store(), legacy_catalog_address)?;
        if legacy_catalog.get(ROOT_NAME_DOCUMENTS).is_none() {
            return Ok(None);
        }
        self.store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_DOCUMENTS.to_string(), None)]),
                meta,
            )
            .map(Some)
    }

    /// Recoverably retire the former standalone object journal after an import.
    ///
    /// The legacy journal has its own unrelated root, so the gate is the current
    /// root of the destination `unified` store: it must exactly equal the graph
    /// root the importer durably published, and every imported address supplied
    /// by the caller must be readable there. The source is renamed to
    /// `chunk-store.journal.migrated`, never deleted or overwritten.
    ///
    /// This is only valid after a complete journal migration. A DocTree-only
    /// import must use [`Self::clear_legacy_documents_root_after_graph_import`]
    /// so unrelated object, pack, and named-root state remains live.
    ///
    /// Callers must drop every facade returned by
    /// [`Self::open_legacy_journal_for_import`] before invoking this function;
    /// the exclusive journal probe refuses retirement while another owner is
    /// still reading the source.
    pub fn retire_legacy_journal_after_graph_import(
        legacy_root: impl AsRef<Path>,
        unified: &dyn UnifiedChunkStore,
        expected_graph_root: Hash,
        imported_addresses: &[Hash],
    ) -> GraphStoreResult<Option<PathBuf>> {
        prove_unified_graph_import(unified, expected_graph_root, imported_addresses)?;

        let legacy_root = legacy_root.as_ref();
        let source = legacy_root.join(UNIFIED_JOURNAL_FILE);
        let retired = legacy_root.join(format!("{UNIFIED_JOURNAL_FILE}.migrated"));
        if !source.exists() {
            return Ok(retired.exists().then_some(retired));
        }
        if retired.exists() {
            return Err(GraphStoreError::new(
                "legacy_object_journal_retirement_collision",
                format!(
                    "refusing to overwrite existing retired journal {}",
                    retired.display()
                ),
            ));
        }

        // The probe verifies the source framing and proves there is no live
        // owner. Keep its lock until the recoverable rename is complete.
        let probe = JournalChunkStore::open(&source)?;
        fs::rename(&source, &retired).map_err(io_err("retire legacy object journal"))?;
        File::open(legacy_root)
            .and_then(|directory| directory.sync_all())
            .map_err(io_err("fsync legacy object directory after retirement"))?;
        drop(probe);
        Ok(Some(retired))
    }

    /// Build a facade over an already-open store. This never reopens or locks
    /// the backend path.
    pub fn from_chunk_store(store: Arc<dyn UnifiedChunkStore>) -> GraphStoreResult<Self> {
        Self::from_parts(store, None, PathBuf::new(), true)
    }

    /// Build a facade over an already-open store with a read-only legacy root.
    pub fn from_chunk_store_with_legacy_root(
        store: Arc<dyn UnifiedChunkStore>,
        legacy_root: impl AsRef<Path>,
    ) -> GraphStoreResult<Self> {
        let legacy_root = legacy_root.as_ref().to_path_buf();
        Self::from_parts(store, Some(legacy_root.clone()), legacy_root, true)
    }

    /// Build a facade over a store whose catalog a RedCore recovery owns. Unlike
    /// [`Self::from_chunk_store`] it publishes no empty catalog on open: the
    /// first publication belongs to the graph commit, as it does for a
    /// journal-backed store. It has no legacy root, so it reads no path.
    pub(crate) fn from_recovered_chunk_store(
        store: Arc<dyn UnifiedChunkStore>,
    ) -> GraphStoreResult<Self> {
        Self::from_parts(store, None, PathBuf::new(), false)
    }

    /// Build the payload facade used by RedCore before graph recovery. It must
    /// not synthesize an empty root catalog: doing so would mask a legacy
    /// snapshot/AOF that recovery still needs to import.
    pub(crate) fn from_redcore_chunk_store(
        store: Arc<dyn UnifiedChunkStore>,
        legacy_root: impl AsRef<Path>,
    ) -> GraphStoreResult<Self> {
        let legacy_root = legacy_root.as_ref().to_path_buf();
        Self::from_parts(store, Some(legacy_root.clone()), legacy_root, false)
    }

    fn from_parts(
        store: Arc<dyn UnifiedChunkStore>,
        legacy_root: Option<PathBuf>,
        diagnostic_root: PathBuf,
        initialize_catalog: bool,
    ) -> GraphStoreResult<Self> {
        let index_update = shared_index_update(&store)?;
        let facade = Self {
            store,
            legacy_root,
            diagnostic_root,
            index_update,
            observed_raw: Arc::new(Mutex::new(BTreeMap::new())),
            #[cfg(test)]
            before_catalog_commit: Arc::new(Mutex::new(None)),
        };
        if initialize_catalog {
            facade.ensure_root_catalog()?;
        }
        Ok(facade)
    }

    /// Clone the already-open physical backend used by this typed facade.
    /// Consumers must share this handle rather than reopening a second journal.
    pub fn chunk_store(&self) -> Arc<dyn UnifiedChunkStore> {
        Arc::clone(&self.store)
    }

    /// Resolve an object's semantic graph hash to the physical payload chunk
    /// named by the current committed object index.
    ///
    /// This is a typed address lookup: a pack entry at the same semantic key
    /// is rejected, and a dangling physical address fails closed. Payload
    /// bytes are verified by the chunk store but are not returned or decoded.
    pub fn object_payload_address(&self, semantic: Hash) -> GraphStoreResult<Option<Hash>> {
        self.indexed_payload_address(semantic, IndexedPayloadKind::Object)
    }

    /// Resolve a pack's semantic commit hash to the physical envelope chunk
    /// named by the current committed object index.
    ///
    /// This is a typed address lookup: an object entry at the same semantic key
    /// is rejected, and a dangling physical address fails closed. Payload
    /// bytes are verified by the chunk store but are not returned or decoded.
    pub fn pack_payload_address(&self, commit: Hash) -> GraphStoreResult<Option<Hash>> {
        self.indexed_payload_address(commit, IndexedPayloadKind::Pack)
    }

    /// Compatibility/diagnostic root. Current payload writes do not use it.
    pub fn root_dir(&self) -> &Path {
        &self.diagnostic_root
    }

    fn ensure_root_catalog(&self) -> GraphStoreResult<()> {
        let _guard = self.lock_index_updates()?;
        if self.store.read_root()?.is_some() {
            return Ok(());
        }
        let empty_graph = canonical_leaf_preimage(&[])?;
        let empty_graph_root = Hash::of(&empty_graph);
        let publication =
            ChunkPublicationWindow::new(Arc::clone(&self.store), "initialize-unified-object-store");
        self.write_verified_to(&publication, empty_graph_root, &empty_graph)?;
        self.store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(empty_graph_root))]),
            internal_commit_meta("initialize unified object store"),
        )?;
        publication.finish_after_publication();
        Ok(())
    }

    fn lock_index_updates(&self) -> GraphStoreResult<std::sync::MutexGuard<'_, ()>> {
        self.index_update.lock().map_err(|_| {
            GraphStoreError::new(
                "cold_object_store_poisoned",
                "object index update mutex was poisoned",
            )
        })
    }

    #[cfg(test)]
    fn install_before_catalog_commit_hook(
        &self,
        hook: impl FnOnce() -> GraphStoreResult<()> + Send + 'static,
    ) -> GraphStoreResult<()> {
        *self.before_catalog_commit.lock().map_err(poisoned)? = Some(Box::new(hook));
        Ok(())
    }

    #[cfg(test)]
    fn run_before_catalog_commit_hook(&self) -> GraphStoreResult<()> {
        let hook = self.before_catalog_commit.lock().map_err(poisoned)?.take();
        hook.map_or(Ok(()), |hook| hook())
    }

    fn write_verified(&self, address: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
        self.write_verified_to(self.store.as_chunk_store(), address, bytes)
    }

    fn write_verified_to(
        &self,
        store: &dyn ChunkStore,
        address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<()> {
        store.write(&[(address, bytes)])?;
        let stored = store.get(address)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_write_missing",
                format!("chunk store accepted {address} but cannot read it back"),
            )
        })?;
        if stored.as_ref() != bytes {
            return Err(typed_hash_mismatch(
                "chunk-store readback",
                address,
                Hash::of(stored.as_ref()),
            ));
        }
        Ok(())
    }

    fn write_raw(&self, bytes: &[u8], message: &'static str) -> GraphStoreResult<Hash> {
        let staged = self.stage_raw_payload(bytes)?;
        let address = staged.address();
        // An empty catalog update is a root barrier which merges against the
        // latest catalog and therefore cannot overwrite another subsystem.
        self.store
            .commit_catalog_updates(&BTreeMap::new(), internal_commit_meta(message))?;
        staged.finish_after_publication();
        Ok(address)
    }

    /// Stage raw bytes for a later graph/catalog publication and retain a
    /// scoped GC pin until the returned receipt is finished or dropped.
    pub fn stage_raw_payload(&self, bytes: &[u8]) -> GraphStoreResult<StagedRawPayload> {
        let address = Hash::of(bytes);
        let publication =
            ChunkPublicationWindow::new(Arc::clone(&self.store), &format!("raw-payload:{address}"));
        self.write_verified_to(&publication, address, bytes)?;
        self.observed_raw
            .lock()
            .map_err(poisoned)?
            .insert(address, ());
        Ok(StagedRawPayload {
            address,
            publication: Some(publication),
        })
    }

    /// Document-lens name for [`Self::stage_raw_payload`].
    pub fn stage_document_bytes(&self, body: &[u8]) -> GraphStoreResult<StagedRawPayload> {
        self.stage_raw_payload(body)
    }

    /// Stage verified raw bytes without advancing the root catalog. RedCore
    /// vector publication holds an in-flight pin across this write and uses
    /// the subsequent graph transaction's catalog commit as the durability
    /// and visibility barrier.
    pub(crate) fn put_unpublished_bytes(&self, bytes: &[u8]) -> GraphStoreResult<String> {
        self.write_raw_unpublished(bytes)
            .map(|address| address.to_prefixed())
    }

    fn write_raw_unpublished(&self, bytes: &[u8]) -> GraphStoreResult<Hash> {
        let address = Hash::of(bytes);
        self.write_verified(address, bytes)?;
        self.observed_raw
            .lock()
            .map_err(poisoned)?
            .insert(address, ());
        Ok(address)
    }

    /// Store arbitrary document bytes in the unified address space.
    pub fn put_document_bytes(&self, body: &[u8]) -> GraphStoreResult<String> {
        self.write_raw(body, "persist unified document body")
            .map(|address| address.to_prefixed())
    }

    pub fn get_document_bytes(&self, content_hash: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(bytes) = self.get_raw_or_legacy_document(content_hash)? else {
            return Ok(None);
        };
        let address = Hash::of(&bytes);
        if !self.store.has(address)? {
            self.write_raw(&bytes, "migrate legacy document body")?;
        }
        Ok(Some(bytes))
    }

    /// Resolve historical document bytes without writing them into this
    /// backend or advancing any catalog. Importers copy the returned bytes into
    /// RedCore's unified store before publishing graph references.
    pub fn get_document_bytes_read_only(
        &self,
        content_hash: &str,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        self.get_raw_or_legacy_document(content_hash)
    }

    /// Resolve raw payload bytes without creating or advancing a catalog root.
    /// Legacy payload materialization remains protected by the caller's pin
    /// until the owning RedCore graph transaction publishes its root.
    pub(crate) fn get_unpublished_bytes(
        &self,
        content_hash: &str,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(bytes) = self.get_raw_or_legacy_document(content_hash)? else {
            return Ok(None);
        };
        let address = Hash::of(&bytes);
        if !self.store.has(address)? {
            self.write_raw_unpublished(&bytes)?;
        }
        Ok(Some(bytes))
    }

    fn get_raw_or_legacy_document(&self, content_hash: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(address) = parse_sha256_address(content_hash) else {
            return Ok(None);
        };
        if let Some(bytes) = self.store.get(address)? {
            self.observed_raw
                .lock()
                .map_err(poisoned)?
                .insert(address, ());
            return Ok(Some(bytes.to_vec()));
        }
        let Some(bytes) = self.read_legacy_document(content_hash)? else {
            return Ok(None);
        };
        Ok(Some(bytes))
    }

    /// Legacy/diagnostic path only. Current documents live in the journal.
    pub fn document_path(&self, content_hash: &str) -> PathBuf {
        self.address_path("documents", AddressAlgorithm::Sha256, content_hash, "zst")
            .unwrap_or_else(|| self.invalid_diagnostic_path("documents", content_hash, "zst"))
    }

    /// Store a content chunk in exactly the same address space as documents.
    pub fn put_chunk(&self, bytes: &[u8]) -> GraphStoreResult<String> {
        self.write_raw(bytes, "persist unified content chunk")
            .map(|address| address.to_prefixed())
    }

    pub fn get_chunk(&self, digest: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        if let Some(legacy) = parse_blake3_digest(digest) {
            if let Some(bytes) = self.resolve_blake3_alias(legacy)? {
                return Ok(Some(bytes));
            }
            let Some(bytes) = self.read_legacy_blake3_chunk(digest)? else {
                return Ok(None);
            };
            return self
                .publish_blake3_alias(legacy, &bytes)
                .map(|bytes| Some(bytes.to_vec()));
        }
        let Some(address) = parse_sha256_address(digest) else {
            return Ok(None);
        };
        if let Some(bytes) = self.store.get(address)? {
            self.observed_raw
                .lock()
                .map_err(poisoned)?
                .insert(address, ());
            return Ok(Some(bytes.to_vec()));
        }
        let Some(bytes) = self.read_legacy_sha256_chunk(digest)? else {
            return Ok(None);
        };
        self.write_raw(&bytes, "migrate legacy sha256 chunk")?;
        Ok(Some(bytes))
    }

    /// Legacy/diagnostic path only. Current chunks live in the journal.
    pub fn chunk_path(&self, digest: &str) -> PathBuf {
        self.address_path("chunks", AddressAlgorithm::Sha256, digest, "zst")
            .unwrap_or_else(|| self.invalid_diagnostic_path("chunks", digest, "zst"))
    }

    pub fn chunk_count(&self) -> GraphStoreResult<usize> {
        Ok(self.observed_raw.lock().map_err(poisoned)?.len())
    }

    /// Atomically publish the latest immutable manifest envelope for one
    /// stable private semantic slot.
    ///
    /// The manifest and every terminal must already be present and valid in
    /// this same unified store. Callers retain their staging pins until this
    /// method returns, because the object-index catalog commit is the first
    /// durable root which makes the manifest reachable.
    pub fn publish_private_manifest_slot(
        &self,
        slot: &PrivateManifestSlot,
    ) -> GraphStoreResult<Hash> {
        slot.validate()?;
        read_file_payload(
            self.store.as_chunk_store(),
            &FileManifestRef::new(slot.manifest_address),
        )?;
        let semantic = slot.semantic_address()?;
        let bytes = encode_private_manifest_slot(slot)?;
        self.publish_indexed_payload(
            semantic,
            IndexedPayloadKind::PrivateManifestSlot,
            &bytes,
            "publish private manifest slot",
        )
    }

    /// Resolve the current immutable value of one stable private semantic
    /// address. Superseded envelopes are deliberately invisible here.
    pub fn private_manifest_slot(
        &self,
        semantic: Hash,
    ) -> GraphStoreResult<Option<PrivateManifestSlot>> {
        let Some((physical, bytes)) =
            self.indexed_payload(semantic, IndexedPayloadKind::PrivateManifestSlot)?
        else {
            return Ok(None);
        };
        let slot = decode_private_manifest_slot(physical, &bytes)?;
        if slot.semantic_address()? != semantic {
            return Err(GraphStoreError::new(
                "cold_private_manifest_slot_identity_mismatch",
                format!(
                    "private manifest envelope {physical} does not match semantic slot {semantic}"
                ),
            ));
        }
        Ok(Some(slot))
    }

    /// Enumerate the current values for one private manifest domain in stable
    /// semantic-address order.
    pub fn private_manifest_slots(
        &self,
        domain: &str,
    ) -> GraphStoreResult<Vec<(Hash, PrivateManifestSlot)>> {
        validate_private_manifest_slot_string("domain", domain)?;
        let index = self.load_index()?;
        let mut slots = Vec::new();
        for (semantic, indexed) in index.payloads {
            if indexed.kind != IndexedPayloadKind::PrivateManifestSlot {
                continue;
            }
            let bytes = self.store.get(indexed.physical)?.ok_or_else(|| {
                GraphStoreError::new(
                    "cold_private_manifest_slot_missing",
                    format!(
                        "semantic slot {semantic} points to missing private manifest envelope {}",
                        indexed.physical
                    ),
                )
            })?;
            let slot = decode_private_manifest_slot(indexed.physical, &bytes)?;
            if slot.semantic_address()? != semantic {
                return Err(GraphStoreError::new(
                    "cold_private_manifest_slot_identity_mismatch",
                    format!(
                        "private manifest envelope {} does not match semantic slot {semantic}",
                        indexed.physical
                    ),
                ));
            }
            if slot.domain == domain {
                slots.push((semantic, slot));
            }
        }
        Ok(slots)
    }

    fn load_index(&self) -> GraphStoreResult<ObjectStoreIndex> {
        let Some(catalog_address) = self.store.read_root()? else {
            return Err(GraphStoreError::new(
                "cold_object_catalog_missing",
                "unified object store has no committed root catalog",
            ));
        };
        let catalog = validate_catalog_root(self.store.as_chunk_store(), catalog_address)?;
        let Some(index_address) = catalog.get(ROOT_NAME_OBJECTS) else {
            return Ok(ObjectStoreIndex::default());
        };
        let bytes = self.store.get(index_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_index_missing",
                format!("object index root {index_address} is absent"),
            )
        })?;
        decode_object_store_index(index_address, &bytes)
    }

    fn publish_indexed_payload(
        &self,
        semantic: Hash,
        kind: IndexedPayloadKind,
        bytes: &[u8],
        message: &'static str,
    ) -> GraphStoreResult<Hash> {
        let _guard = self.lock_index_updates()?;
        let physical = Hash::of(bytes);
        let publication = ChunkPublicationWindow::new(
            Arc::clone(&self.store),
            &format!("indexed-payload:{semantic}:{physical}"),
        );
        self.write_verified_to(&publication, physical, bytes)?;
        let mut index = self.load_index()?;
        let next = IndexedPayload { kind, physical };
        if index.payloads.get(&semantic) == Some(&next) {
            publication.finish_after_publication();
            return Ok(physical);
        }
        index.payloads.insert(semantic, next);
        let index_bytes = encode_object_store_index(&index)?;
        let index_address = Hash::of(&index_bytes);
        self.write_verified_to(&publication, index_address, &index_bytes)?;
        #[cfg(test)]
        self.run_before_catalog_commit_hook()?;
        self.store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_OBJECTS.to_string(), Some(index_address))]),
            internal_commit_meta(message),
        )?;
        publication.finish_after_publication();
        Ok(physical)
    }

    fn indexed_payload(
        &self,
        semantic: Hash,
        expected_kind: IndexedPayloadKind,
    ) -> GraphStoreResult<Option<(Hash, Arc<[u8]>)>> {
        let Some(indexed) = self.load_index()?.payloads.get(&semantic).copied() else {
            return Ok(None);
        };
        if indexed.kind != expected_kind {
            return Err(GraphStoreError::new(
                "cold_object_index_kind_mismatch",
                format!(
                    "semantic address {semantic} is indexed as {:?}, expected {expected_kind:?}",
                    indexed.kind
                ),
            ));
        }
        let bytes = self.store.get(indexed.physical)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_payload_missing",
                format!(
                    "semantic address {semantic} points to missing payload {}",
                    indexed.physical
                ),
            )
        })?;
        Ok(Some((indexed.physical, bytes)))
    }

    fn indexed_payload_address(
        &self,
        semantic: Hash,
        expected_kind: IndexedPayloadKind,
    ) -> GraphStoreResult<Option<Hash>> {
        let _guard = self.lock_index_updates()?;
        let Some(indexed) = self.load_index()?.payloads.get(&semantic).copied() else {
            return Ok(None);
        };
        if indexed.kind != expected_kind {
            return Err(GraphStoreError::new(
                "cold_object_index_kind_mismatch",
                format!(
                    "semantic address {semantic} is indexed as {:?}, expected {expected_kind:?}",
                    indexed.kind
                ),
            ));
        }
        if !self.store.has(indexed.physical)? {
            return Err(GraphStoreError::new(
                "cold_object_payload_missing",
                format!(
                    "semantic address {semantic} points to missing payload {}",
                    indexed.physical
                ),
            ));
        }
        Ok(Some(indexed.physical))
    }

    fn resolve_blake3_alias(&self, legacy: Blake3Digest) -> GraphStoreResult<Option<Vec<u8>>> {
        let index = self.load_index()?;
        self.resolve_blake3_alias_from_index(&index, legacy)
    }

    fn resolve_blake3_alias_from_index(
        &self,
        index: &ObjectStoreIndex,
        legacy: Blake3Digest,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(alias_address) = index.blake3_aliases.get(&legacy).copied() else {
            return Ok(None);
        };
        let alias_bytes = self.store.get(alias_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_alias_missing",
                format!("Blake3 alias record {alias_address} is absent"),
            )
        })?;
        let alias = decode_blake3_alias(alias_address, &alias_bytes)?;
        if alias.legacy != legacy {
            return Err(GraphStoreError::new(
                "cold_object_alias_mismatch",
                "Blake3 alias index key does not match its rooted alias record",
            ));
        }
        let body = self.store.get(alias.canonical)?.ok_or_else(|| {
            GraphStoreError::new(
                "cold_object_alias_target_missing",
                format!(
                    "Blake3 alias record {alias_address} points to missing canonical payload {}",
                    alias.canonical
                ),
            )
        })?;
        verify_physical_address(alias.canonical, &body, "Blake3 alias target")?;
        verify_blake3_digest(legacy, &body, "Blake3 alias target")?;
        self.observed_raw
            .lock()
            .map_err(poisoned)?
            .insert(alias.canonical, ());
        Ok(Some(body.to_vec()))
    }

    fn publish_blake3_alias<'a>(
        &self,
        legacy: Blake3Digest,
        bytes: &'a [u8],
    ) -> GraphStoreResult<&'a [u8]> {
        verify_blake3_digest(legacy, bytes, "legacy blake3 chunk")?;
        let _guard = self.lock_index_updates()?;
        let mut index = self.load_index()?;
        if let Some(existing) = self.resolve_blake3_alias_from_index(&index, legacy)? {
            if existing.as_slice() != bytes {
                return Err(GraphStoreError::new(
                    "cold_object_alias_collision",
                    "legacy Blake3 bytes disagree with the already committed alias target",
                ));
            }
            return Ok(bytes);
        }

        let canonical = Hash::of(bytes);
        let publication = ChunkPublicationWindow::new(
            Arc::clone(&self.store),
            &format!(
                "blake3-alias:{}:{canonical}",
                encode_lower_hex(legacy.as_bytes())
            ),
        );
        self.write_verified_to(&publication, canonical, bytes)?;
        let alias_bytes = encode_blake3_alias(Blake3Alias { legacy, canonical });
        let alias_address = Hash::of(&alias_bytes);
        self.write_verified_to(&publication, alias_address, &alias_bytes)?;
        index.blake3_aliases.insert(legacy, alias_address);
        let index_bytes = encode_object_store_index(&index)?;
        let index_address = Hash::of(&index_bytes);
        self.write_verified_to(&publication, index_address, &index_bytes)?;
        #[cfg(test)]
        self.run_before_catalog_commit_hook()?;
        self.store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_OBJECTS.to_string(), Some(index_address))]),
            internal_commit_meta("publish legacy blake3 alias"),
        )?;
        publication.finish_after_publication();
        self.observed_raw
            .lock()
            .map_err(poisoned)?
            .insert(canonical, ());
        Ok(bytes)
    }

    fn hash_path(&self, kind: &str, hash: Hash, extension: &str) -> PathBuf {
        let address = hash.to_prefixed();
        let hex = address
            .strip_prefix("sha256:")
            .expect("Hash renders with a canonical sha256 prefix");
        self.diagnostic_root
            .join(kind)
            .join("sha256")
            .join(format!("{hex}.{extension}"))
    }

    fn address_path(
        &self,
        kind: &str,
        algorithm: AddressAlgorithm,
        address: &str,
        extension: &str,
    ) -> Option<PathBuf> {
        address_hex(address, algorithm).map(|hex| {
            self.diagnostic_root
                .join(kind)
                .join(algorithm.as_str())
                .join(format!("{hex}.{extension}"))
        })
    }

    fn invalid_diagnostic_path(&self, kind: &str, address: &str, extension: &str) -> PathBuf {
        let digest = Hash::of(address.as_bytes()).to_prefixed();
        let hex = digest
            .strip_prefix("sha256:")
            .expect("Hash renders with a canonical sha256 prefix");
        self.diagnostic_root
            .join("invalid-address")
            .join(kind)
            .join(format!("{hex}.{extension}"))
    }

    fn legacy_path(&self, kind: &str, address: &str, extension: &str) -> Option<PathBuf> {
        self.legacy_root.as_ref().map(|root| {
            root.join(kind)
                .join(format!("{}.{}", safe_filename(address), extension))
        })
    }

    fn read_legacy_document(&self, address: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        self.read_legacy_sha256_blob("documents", address, "legacy cold document")
    }

    fn read_legacy_sha256_chunk(&self, address: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        self.read_legacy_sha256_blob("chunks", address, "legacy sha256 chunk")
    }

    fn read_legacy_sha256_blob(
        &self,
        kind: &str,
        address: &str,
        context: &'static str,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(root) = self.legacy_root.as_ref() else {
            return Ok(None);
        };
        let qualified = address_hex(address, AddressAlgorithm::Sha256).map(|hex| {
            root.join(kind)
                .join(AddressAlgorithm::Sha256.as_str())
                .join(format!("{hex}.zst"))
        });
        let decoded = match qualified.as_ref().map(|path| fs::read(path)).transpose() {
            Ok(Some(bytes)) => Some(decompress_cold_bytes(&bytes)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(io_err(context)(error)),
            Ok(None) => None,
        };
        let decoded = match decoded {
            Some(bytes) => bytes,
            None => {
                let Some(flat) = self.legacy_path(kind, address, "zst") else {
                    return Ok(None);
                };
                let bytes = match fs::read(flat) {
                    Ok(bytes) => bytes,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(error) => return Err(io_err(context)(error)),
                };
                decode_legacy_bytes(&bytes)?
            }
        };
        verify_bytes_address(address, &content_hash_bytes(&decoded), context)?;
        Ok(Some(decoded))
    }

    fn read_legacy_blake3_chunk(&self, address: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(root) = self.legacy_root.as_ref() else {
            return Ok(None);
        };
        let hex = require_address(address, AddressAlgorithm::Blake3)?;
        let qualified = root
            .join("chunks")
            .join("blake3")
            .join(format!("{hex}.zst"));
        let decoded = match fs::read(&qualified) {
            Ok(bytes) => decompress_cold_bytes(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(flat) = self.legacy_path("chunks", address, "zst") else {
                    return Ok(None);
                };
                let bytes = match fs::read(flat) {
                    Ok(bytes) => bytes,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(error) => return Err(io_err("read flat legacy blake3 chunk")(error)),
                };
                decode_legacy_bytes(&bytes)?
            }
            Err(error) => return Err(io_err("read qualified legacy blake3 chunk")(error)),
        };
        verify_bytes_address(address, &blake3_address(&decoded), "legacy blake3 chunk")?;
        Ok(Some(decoded))
    }

    fn read_legacy_object(&self, hash: Hash) -> GraphStoreResult<Option<GraphContentObject>> {
        let Some(bytes) = self.read_legacy_record_bytes("objects", hash)? else {
            return Ok(None);
        };
        let object: GraphContentObject = serde_json::from_slice(&bytes)
            .map_err(|error| GraphStoreError::new("cold_object_decode", error.to_string()))?;
        verify_object_address(hash, &object)?;
        Ok(Some(object))
    }

    fn read_legacy_pack(&self, hash: Hash) -> GraphStoreResult<Option<CompiledGraphPack>> {
        let Some(bytes) = self.read_legacy_record_bytes("packs", hash)? else {
            return Ok(None);
        };
        match graph_pack_chunk_format(&bytes)? {
            GRAPH_CHUNK_FORMAT_VERSION => {
                let pack: CompiledGraphPack = serde_json::from_slice(&bytes)
                    .map_err(|error| GraphStoreError::new("cold_pack_decode", error.to_string()))?;
                verify_pack_address(hash, &pack)?;
                Ok(Some(pack))
            }
            LEGACY_GRAPH_CHUNK_FORMAT_VERSION => Err(GraphStoreError::new(
                "legacy_graph_pack_migration_context_required",
                format!("legacy pack {hash} changes identity and must be loaded through load_pack"),
            )),
            version => Err(GraphStoreError::new(
                "graph_pack_format_unsupported",
                format!("graph pack {hash} uses unsupported chunk format {version}"),
            )),
        }
    }

    /// Resolve an object from this facade without materializing legacy bytes
    /// into its backing journal or advancing its catalog. Upgrade importers use
    /// this while the former standalone cold journal is rollback-only input.
    pub fn get_object_read_only(&self, hash: Hash) -> GraphStoreResult<Option<GraphContentObject>> {
        if let Some((physical, bytes)) = self.indexed_payload(hash, IndexedPayloadKind::Object)? {
            let record = decode_canonical_graph_payload(physical, &bytes)?;
            let object = content_object_from_payload(record)?;
            verify_object_address(hash, &object)?;
            return Ok(Some(object));
        }
        let Some(object) = self.read_legacy_object(hash)? else {
            return Ok(None);
        };
        verify_object_address(hash, &object)?;
        Ok(Some(object))
    }

    /// Resolve a compiled pack without writing to this facade. This is the
    /// pack counterpart to [`Self::get_object_read_only`].
    pub fn get_pack_read_only(
        &self,
        commit_hash: Hash,
    ) -> GraphStoreResult<Option<CompiledGraphPack>> {
        if let Some((physical, bytes)) =
            self.indexed_payload(commit_hash, IndexedPayloadKind::Pack)?
        {
            let pack = decode_pack_envelope(physical, &bytes)?;
            verify_pack_address(commit_hash, &pack)?;
            return Ok(Some(pack));
        }
        let Some(pack) = self.read_legacy_pack(commit_hash)? else {
            return Ok(None);
        };
        verify_pack_address(commit_hash, &pack)?;
        Ok(Some(pack))
    }

    /// Validate every byte that [`ColdObjectStore::load_object`] would consume
    /// without publishing a migrated payload or advancing the unified catalog.
    /// Startup migration uses this to prove a whole legacy input set before its
    /// first destination write.
    pub fn preflight_object_load(&self, hash: Hash) -> GraphStoreResult<bool> {
        if let Some((physical, bytes)) = self.indexed_payload(hash, IndexedPayloadKind::Object)? {
            let record = decode_canonical_graph_payload(physical, &bytes)?;
            let object = content_object_from_payload(record)?;
            verify_object_address(hash, &object)?;
            return Ok(true);
        }
        let Some(bytes) = self.read_legacy_record_bytes("objects", hash)? else {
            return Ok(false);
        };
        if let Ok(object) = serde_json::from_slice::<GraphContentObject>(&bytes) {
            if verify_object_address(hash, &object).is_ok() {
                return Ok(true);
            }
        }
        migrate_legacy_graph_content_object_v1(&bytes, hash)?;
        Ok(true)
    }

    /// Validate the requested pack and its complete bounded v1 parent chain
    /// without publishing any replacement packs or advancing the catalog.
    pub fn preflight_pack_load(&self, commit_hash: Hash) -> GraphStoreResult<bool> {
        if let Some((physical, bytes)) =
            self.indexed_payload(commit_hash, IndexedPayloadKind::Pack)?
        {
            let pack = decode_pack_envelope(physical, &bytes)?;
            verify_pack_address(commit_hash, &pack)?;
            return Ok(true);
        }
        let Some(bytes) = self.read_legacy_record_bytes("packs", commit_hash)? else {
            return Ok(false);
        };
        match graph_pack_chunk_format(&bytes)? {
            GRAPH_CHUNK_FORMAT_VERSION => {
                let pack: CompiledGraphPack = serde_json::from_slice(&bytes)
                    .map_err(|error| GraphStoreError::new("cold_pack_decode", error.to_string()))?;
                verify_pack_address(commit_hash, &pack)?;
            }
            LEGACY_GRAPH_CHUNK_FORMAT_VERSION => {
                let mut parent_map = BTreeMap::new();
                let mut visiting = BTreeSet::new();
                let mut plans = Vec::new();
                self.plan_legacy_pack_migration(
                    commit_hash,
                    0,
                    &mut parent_map,
                    &mut visiting,
                    &mut plans,
                )?;
                if !parent_map.contains_key(&commit_hash.to_prefixed()) {
                    return Err(GraphStoreError::new(
                        "legacy_graph_pack_plan_incomplete",
                        format!("migration plan omitted requested pack {commit_hash}"),
                    ));
                }
            }
            version => {
                return Err(GraphStoreError::new(
                    "graph_pack_format_unsupported",
                    format!("graph pack {commit_hash} uses unsupported chunk format {version}"),
                ));
            }
        }
        Ok(true)
    }

    fn read_legacy_record_bytes(
        &self,
        kind: &str,
        hash: Hash,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(_root) = self.legacy_root.as_ref() else {
            return Ok(None);
        };
        let qualified = self.hash_path(kind, hash, "json");
        match fs::read(&qualified) {
            Ok(bytes) => decompress_cold_bytes(&bytes).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let address = hash.to_prefixed();
                let Some(flat) = self.legacy_path(kind, &address, "json") else {
                    return Ok(None);
                };
                match fs::read(flat) {
                    Ok(bytes) => decode_legacy_bytes(&bytes).map(Some),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                    Err(error) => Err(io_err("read flat legacy object-store record")(error)),
                }
            }
            Err(error) => Err(io_err("read qualified legacy object-store record")(error)),
        }
    }

    fn plan_legacy_pack_migration(
        &self,
        old_commit: Hash,
        depth: usize,
        migrated_parents: &mut BTreeMap<String, Hash>,
        visiting: &mut BTreeSet<Hash>,
        plans: &mut Vec<LegacyGraphPackMigration>,
    ) -> GraphStoreResult<()> {
        if depth > MAX_LEGACY_PACK_PARENT_DEPTH {
            return Err(GraphStoreError::new(
                "legacy_graph_parent_depth_exceeded",
                format!(
                    "legacy graph parent depth exceeds {MAX_LEGACY_PACK_PARENT_DEPTH} at {old_commit}"
                ),
            ));
        }
        if migrated_parents.contains_key(&old_commit.to_prefixed()) {
            return Ok(());
        }
        if !visiting.insert(old_commit) {
            return Err(GraphStoreError::new(
                "legacy_graph_parent_cycle",
                format!("legacy graph parent cycle reaches {old_commit}"),
            ));
        }
        let bytes = self
            .read_legacy_record_bytes("packs", old_commit)?
            .ok_or_else(|| {
                GraphStoreError::new(
                    "legacy_graph_parent_missing",
                    format!("legacy parent pack {old_commit} is absent"),
                )
            })?;
        if graph_pack_chunk_format(&bytes)? != LEGACY_GRAPH_CHUNK_FORMAT_VERSION {
            return Err(GraphStoreError::new(
                "legacy_graph_parent_format_mismatch",
                format!("legacy parent {old_commit} is not graph chunk format v1"),
            ));
        }
        for parent in legacy_graph_pack_v1_parents(&bytes, old_commit)? {
            self.plan_legacy_pack_migration(
                parent,
                depth.saturating_add(1),
                migrated_parents,
                visiting,
                plans,
            )?;
        }
        let migration = migrate_legacy_graph_pack_v1(&bytes, old_commit, migrated_parents)?;
        migrated_parents.insert(
            migration.old_commit.to_prefixed(),
            migration.new_pack.commit.commit_hash,
        );
        visiting.remove(&old_commit);
        plans.push(migration);
        Ok(())
    }

    fn publish_legacy_object_migration(
        &self,
        migration: LegacyGraphObjectMigration,
    ) -> GraphStoreResult<LoadedColdObject> {
        self.put_object(&migration.new_object)?;
        let new_hash = migration.new_object.hash;
        let read_back = self.get_object(new_hash)?.ok_or_else(|| {
            GraphStoreError::new(
                "legacy_graph_object_readback_missing",
                format!("migrated graph object {new_hash} was not readable after publication"),
            )
        })?;
        if read_back != migration.new_object {
            return Err(GraphStoreError::new(
                "legacy_graph_object_readback_mismatch",
                format!("migrated graph object {new_hash} changed during publication"),
            ));
        }
        Ok(LoadedColdObject {
            object: read_back,
            migrated_from: Some(migration.old_hash),
        })
    }
}

impl ColdObjectStore for DiskObjectStore {
    fn put_object(&self, object: &GraphContentObject) -> GraphStoreResult<()> {
        let record = graph_payload_record(object)?;
        let bytes = canonical_graph_payload_preimage(&record)?;
        self.publish_indexed_payload(
            object.hash,
            IndexedPayloadKind::Object,
            &bytes,
            "publish unified graph object",
        )?;
        Ok(())
    }

    fn get_object(&self, hash: Hash) -> GraphStoreResult<Option<GraphContentObject>> {
        if let Some((physical, bytes)) = self.indexed_payload(hash, IndexedPayloadKind::Object)? {
            let record = decode_canonical_graph_payload(physical, &bytes)?;
            let object = content_object_from_payload(record)?;
            verify_object_address(hash, &object)?;
            return Ok(Some(object));
        }
        let Some(object) = self.read_legacy_object(hash)? else {
            return Ok(None);
        };
        verify_object_address(hash, &object)?;
        self.put_object(&object)?;
        Ok(Some(object))
    }

    fn load_object(&self, hash: Hash) -> GraphStoreResult<Option<LoadedColdObject>> {
        if let Some((physical, bytes)) = self.indexed_payload(hash, IndexedPayloadKind::Object)? {
            let record = decode_canonical_graph_payload(physical, &bytes)?;
            let object = content_object_from_payload(record)?;
            verify_object_address(hash, &object)?;
            return Ok(Some(LoadedColdObject {
                object,
                migrated_from: None,
            }));
        }
        let Some(bytes) = self.read_legacy_record_bytes("objects", hash)? else {
            return Ok(None);
        };
        if let Ok(object) = serde_json::from_slice::<GraphContentObject>(&bytes) {
            if verify_object_address(hash, &object).is_ok() {
                self.put_object(&object)?;
                let read_back = self.get_object(hash)?.ok_or_else(|| {
                    GraphStoreError::new(
                        "legacy_graph_object_readback_missing",
                        format!("legacy graph object {hash} was not readable after publication"),
                    )
                })?;
                if read_back != object {
                    return Err(GraphStoreError::new(
                        "legacy_graph_object_readback_mismatch",
                        format!("legacy graph object {hash} changed during publication"),
                    ));
                }
                return Ok(Some(LoadedColdObject {
                    object: read_back,
                    migrated_from: None,
                }));
            }
        }
        let migration = migrate_legacy_graph_content_object_v1(&bytes, hash)?;
        self.publish_legacy_object_migration(migration).map(Some)
    }

    fn put_pack(&self, pack: &CompiledGraphPack) -> GraphStoreResult<()> {
        verify_pack_address(pack.commit.commit_hash, pack)?;
        let bytes = encode_pack_envelope(pack)?;
        self.publish_indexed_payload(
            pack.commit.commit_hash,
            IndexedPayloadKind::Pack,
            &bytes,
            "publish unified graph pack",
        )?;
        Ok(())
    }

    fn get_pack(&self, commit_hash: Hash) -> GraphStoreResult<Option<CompiledGraphPack>> {
        if let Some((physical, bytes)) =
            self.indexed_payload(commit_hash, IndexedPayloadKind::Pack)?
        {
            let pack = decode_pack_envelope(physical, &bytes)?;
            verify_pack_address(commit_hash, &pack)?;
            return Ok(Some(pack));
        }
        let Some(pack) = self.read_legacy_pack(commit_hash)? else {
            return Ok(None);
        };
        verify_pack_address(commit_hash, &pack)?;
        self.put_pack(&pack)?;
        Ok(Some(pack))
    }

    fn load_pack(&self, commit_hash: Hash) -> GraphStoreResult<Option<LoadedColdPack>> {
        if let Some((physical, bytes)) =
            self.indexed_payload(commit_hash, IndexedPayloadKind::Pack)?
        {
            let pack = decode_pack_envelope(physical, &bytes)?;
            verify_pack_address(commit_hash, &pack)?;
            return Ok(Some(LoadedColdPack {
                pack,
                migrated_from: None,
            }));
        }
        let Some(bytes) = self.read_legacy_record_bytes("packs", commit_hash)? else {
            return Ok(None);
        };
        match graph_pack_chunk_format(&bytes)? {
            GRAPH_CHUNK_FORMAT_VERSION => {
                let pack: CompiledGraphPack = serde_json::from_slice(&bytes)
                    .map_err(|error| GraphStoreError::new("cold_pack_decode", error.to_string()))?;
                verify_pack_address(commit_hash, &pack)?;
                self.put_pack(&pack)?;
                let read_back = self.get_pack(commit_hash)?.ok_or_else(|| {
                    GraphStoreError::new(
                        "legacy_graph_pack_readback_missing",
                        format!(
                            "legacy graph pack {commit_hash} was not readable after publication"
                        ),
                    )
                })?;
                if read_back != pack {
                    return Err(GraphStoreError::new(
                        "legacy_graph_pack_readback_mismatch",
                        format!("legacy graph pack {commit_hash} changed during publication"),
                    ));
                }
                Ok(Some(LoadedColdPack {
                    pack: read_back,
                    migrated_from: None,
                }))
            }
            LEGACY_GRAPH_CHUNK_FORMAT_VERSION => {
                let mut parent_map = BTreeMap::new();
                let mut visiting = BTreeSet::new();
                let mut plans = Vec::new();
                self.plan_legacy_pack_migration(
                    commit_hash,
                    0,
                    &mut parent_map,
                    &mut visiting,
                    &mut plans,
                )?;
                let root_new_commit = parent_map
                    .get(&commit_hash.to_prefixed())
                    .copied()
                    .ok_or_else(|| {
                        GraphStoreError::new(
                            "legacy_graph_pack_plan_incomplete",
                            format!("migration plan omitted requested pack {commit_hash}"),
                        )
                    })?;
                for migration in plans {
                    self.put_pack(&migration.new_pack)?;
                    let new_hash = migration.new_pack.commit.commit_hash;
                    let read_back = self.get_pack(new_hash)?.ok_or_else(|| {
                        GraphStoreError::new(
                            "legacy_graph_pack_readback_missing",
                            format!(
                                "migrated graph pack {new_hash} was not readable after publication"
                            ),
                        )
                    })?;
                    if read_back != migration.new_pack {
                        return Err(GraphStoreError::new(
                            "legacy_graph_pack_readback_mismatch",
                            format!("migrated graph pack {new_hash} changed during publication"),
                        ));
                    }
                }
                let pack = self.get_pack(root_new_commit)?.ok_or_else(|| {
                    GraphStoreError::new(
                        "legacy_graph_pack_readback_missing",
                        format!("migrated root graph pack {root_new_commit} is absent"),
                    )
                })?;
                Ok(Some(LoadedColdPack {
                    pack,
                    migrated_from: Some(commit_hash),
                }))
            }
            version => Err(GraphStoreError::new(
                "graph_pack_format_unsupported",
                format!("graph pack {commit_hash} uses unsupported chunk format {version}"),
            )),
        }
    }

    fn object_count(&self) -> GraphStoreResult<usize> {
        Ok(self
            .load_index()?
            .payloads
            .values()
            .filter(|entry| entry.kind == IndexedPayloadKind::Object)
            .count())
    }
}

fn shared_index_update(store: &Arc<dyn UnifiedChunkStore>) -> GraphStoreResult<SharedIndexUpdate> {
    let identity = Arc::as_ptr(store) as *const () as usize;
    let registry = INDEX_UPDATE_LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut registry = registry.lock().map_err(|_| {
        GraphStoreError::new(
            "cold_object_store_poisoned",
            "object index lock registry was poisoned",
        )
    })?;
    registry.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = registry.get(&identity).and_then(Weak::upgrade) {
        return Ok(lock);
    }
    let lock = Arc::new(Mutex::new(()));
    registry.insert(identity, Arc::downgrade(&lock));
    Ok(lock)
}

fn prove_unified_graph_import(
    unified: &dyn UnifiedChunkStore,
    expected_graph_root: Hash,
    imported_addresses: &[Hash],
) -> GraphStoreResult<()> {
    if imported_addresses.len() > MAX_LEGACY_IMPORT_ADDRESSES {
        return Err(GraphStoreError::new(
            "legacy_object_import_address_count_too_large",
            format!(
                "legacy import supplied {} addresses; limit is {MAX_LEGACY_IMPORT_ADDRESSES}",
                imported_addresses.len()
            ),
        ));
    }
    let catalog_address = unified.read_root()?.ok_or_else(|| {
        GraphStoreError::new(
            "legacy_object_import_catalog_missing",
            "cannot clean up legacy object state without a unified committed catalog",
        )
    })?;
    let catalog = validate_catalog_root(unified.as_chunk_store(), catalog_address)?;
    let actual_graph_root = catalog.graph_root()?;
    if actual_graph_root != expected_graph_root {
        return Err(GraphStoreError::new(
            "legacy_object_graph_import_not_durable",
            format!(
                "unified catalog graph root is {actual_graph_root}, expected imported root {expected_graph_root}"
            ),
        ));
    }
    let missing = unified.missing(imported_addresses)?;
    if !missing.is_empty() {
        return Err(GraphStoreError::new(
            "legacy_object_import_payload_missing",
            format!(
                "unified store is missing {} imported payload addresses (first: {})",
                missing.len(),
                missing[0]
            ),
        ));
    }
    Ok(())
}

/// Legacy-only mapper for the pre-qualified flat layout.
///
/// This mapping is collision-prone by design and must never select a current
/// object path. Callers reach it only after strict canonical address validation
/// and migrate verified bytes into the algorithm-qualified layout.
pub(crate) fn safe_filename(hash: &str) -> String {
    hash.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

pub fn content_hash_bytes(bytes: &[u8]) -> String {
    Hash::of(bytes).to_prefixed()
}

#[cfg(test)]
pub(crate) fn compress_cold_bytes(bytes: &[u8]) -> GraphStoreResult<Vec<u8>> {
    zstd::stream::encode_all(bytes, 0)
        .map_err(|error| GraphStoreError::new("cold_object_compress", error.to_string()))
}

pub(crate) fn decompress_cold_bytes(bytes: &[u8]) -> GraphStoreResult<Vec<u8>> {
    zstd::stream::decode_all(bytes)
        .map_err(|error| GraphStoreError::new("cold_object_decompress", error.to_string()))
}

/// Decode a file from the validated pre-qualified flat layout. Raw fallback is
/// deliberately confined here: current qualified paths are always strict zstd.
fn decode_legacy_bytes(bytes: &[u8]) -> GraphStoreResult<Vec<u8>> {
    match decompress_cold_bytes(bytes) {
        Ok(decoded) => Ok(decoded),
        Err(error) if bytes.starts_with(&ZSTD_MAGIC) => Err(error),
        Err(_) => Ok(bytes.to_vec()),
    }
}

fn graph_payload_record(object: &GraphContentObject) -> GraphStoreResult<GraphPayloadRecord> {
    verify_object_address(object.hash, object)?;
    let payload = object.payload.clone().ok_or_else(|| {
        GraphStoreError::new(
            "graph_object_payload_required",
            format!("graph object {} has no full recovery payload", object.hash),
        )
    })?;
    let record = match object.kind {
        GraphObjectKind::Node => GraphPayloadRecord::Node(
            serde_json::from_value(payload)
                .map_err(|error| GraphStoreError::new("cold_object_decode", error.to_string()))?,
        ),
        GraphObjectKind::Edge => GraphPayloadRecord::Edge(
            serde_json::from_value(payload)
                .map_err(|error| GraphStoreError::new("cold_object_decode", error.to_string()))?,
        ),
    };
    let reconstructed = content_object_from_payload(record.clone())?;
    if &reconstructed != object {
        return Err(GraphStoreError::new(
            "cold_object_payload_metadata_mismatch",
            format!(
                "full recovery payload does not reconstruct graph object {} verbatim",
                object.hash
            ),
        ));
    }
    Ok(record)
}

fn content_object_from_payload(record: GraphPayloadRecord) -> GraphStoreResult<GraphContentObject> {
    match record {
        GraphPayloadRecord::Node(node) => node_to_content_object(&node),
        GraphPayloadRecord::Edge(edge) => edge_to_content_object(&edge),
    }
}

fn encode_object_store_index(index: &ObjectStoreIndex) -> GraphStoreResult<Vec<u8>> {
    encode_object_store_index_version(index, OBJECT_STORE_INDEX_VERSION)
}

fn encode_object_store_index_version(
    index: &ObjectStoreIndex,
    version: u8,
) -> GraphStoreResult<Vec<u8>> {
    let payload_count = object_index_count(index.payloads.len())?;
    let alias_count = object_index_count(index.blake3_aliases.len())?;
    let mut bytes = Vec::with_capacity(
        10_usize
            .saturating_add(index.payloads.len().saturating_mul(65))
            .saturating_add(index.blake3_aliases.len().saturating_mul(64)),
    );
    bytes.push(DOMAIN_OBJECT_STORE_INDEX);
    bytes.push(version);
    bytes.extend_from_slice(&payload_count.to_le_bytes());
    for (semantic, payload) in &index.payloads {
        bytes.extend_from_slice(semantic.as_bytes());
        bytes.push(payload.kind as u8);
        bytes.extend_from_slice(payload.physical.as_bytes());
    }
    bytes.extend_from_slice(&alias_count.to_le_bytes());
    for (legacy, alias_record) in &index.blake3_aliases {
        bytes.extend_from_slice(legacy.as_bytes());
        bytes.extend_from_slice(alias_record.as_bytes());
    }
    Ok(bytes)
}

fn decode_object_store_index(expected: Hash, bytes: &[u8]) -> GraphStoreResult<ObjectStoreIndex> {
    verify_physical_address(expected, bytes, "object index")?;
    if bytes.len() < 6 || bytes[0] != DOMAIN_OBJECT_STORE_INDEX {
        return Err(GraphStoreError::new(
            "cold_object_index_invalid",
            "object index has an invalid domain or version",
        ));
    }
    let index = match bytes[1] {
        LEGACY_OBJECT_STORE_INDEX_VERSION => decode_legacy_object_store_index(bytes)?,
        BLAKE3_OBJECT_STORE_INDEX_VERSION => decode_current_object_store_index(bytes, false)?,
        OBJECT_STORE_INDEX_VERSION => decode_current_object_store_index(bytes, true)?,
        _ => {
            return Err(GraphStoreError::new(
                "cold_object_index_invalid",
                "object index has an invalid domain or version",
            ))
        }
    };
    let canonical = if bytes[1] == LEGACY_OBJECT_STORE_INDEX_VERSION {
        encode_legacy_object_store_index(&index.payloads)?
    } else {
        encode_object_store_index_version(&index, bytes[1])?
    };
    if canonical != bytes {
        return Err(GraphStoreError::new(
            "cold_object_index_not_canonical",
            "object index does not use the canonical encoding",
        ));
    }
    Ok(index)
}

fn object_index_count(count: usize) -> GraphStoreResult<u32> {
    u32::try_from(count).map_err(|_| {
        GraphStoreError::new(
            "cold_object_index_too_large",
            "object index entry count exceeds the canonical u32 limit",
        )
    })
}

fn encode_legacy_object_store_index(
    payloads: &BTreeMap<Hash, IndexedPayload>,
) -> GraphStoreResult<Vec<u8>> {
    let count = object_index_count(payloads.len())?;
    let mut bytes = Vec::with_capacity(6 + payloads.len().saturating_mul(65));
    bytes.push(DOMAIN_OBJECT_STORE_INDEX);
    bytes.push(LEGACY_OBJECT_STORE_INDEX_VERSION);
    bytes.extend_from_slice(&count.to_le_bytes());
    for (semantic, payload) in payloads {
        bytes.extend_from_slice(semantic.as_bytes());
        bytes.push(payload.kind as u8);
        bytes.extend_from_slice(payload.physical.as_bytes());
    }
    Ok(bytes)
}

fn decode_legacy_object_store_index(bytes: &[u8]) -> GraphStoreResult<ObjectStoreIndex> {
    let count = u32::from_le_bytes(bytes[2..6].try_into().expect("four-byte count")) as usize;
    let expected_len = count
        .checked_mul(65)
        .and_then(|entries| entries.checked_add(6))
        .ok_or_else(|| {
            GraphStoreError::new("cold_object_index_invalid", "object index length overflow")
        })?;
    if bytes.len() != expected_len {
        return Err(GraphStoreError::new(
            "cold_object_index_invalid",
            format!(
                "object index declares {count} entries but has {} bytes",
                bytes.len()
            ),
        ));
    }
    let mut offset = 6;
    let payloads = decode_indexed_payloads(bytes, &mut offset, count, false)?;
    Ok(ObjectStoreIndex {
        payloads,
        blake3_aliases: BTreeMap::new(),
    })
}

fn decode_current_object_store_index(
    bytes: &[u8],
    allow_private_manifest_slots: bool,
) -> GraphStoreResult<ObjectStoreIndex> {
    let payload_count =
        u32::from_le_bytes(bytes[2..6].try_into().expect("four-byte count")) as usize;
    let payload_bytes = payload_count.checked_mul(65).ok_or_else(|| {
        GraphStoreError::new("cold_object_index_invalid", "object index length overflow")
    })?;
    let alias_count_offset = 6_usize.checked_add(payload_bytes).ok_or_else(|| {
        GraphStoreError::new("cold_object_index_invalid", "object index length overflow")
    })?;
    let alias_count_end = alias_count_offset.checked_add(4).ok_or_else(|| {
        GraphStoreError::new("cold_object_index_invalid", "object index length overflow")
    })?;
    if bytes.len() < alias_count_end {
        return Err(GraphStoreError::new(
            "cold_object_index_invalid",
            "object index is truncated before the Blake3 alias count",
        ));
    }
    let alias_count = u32::from_le_bytes(
        bytes[alias_count_offset..alias_count_end]
            .try_into()
            .expect("four-byte alias count"),
    ) as usize;
    let expected_len = alias_count
        .checked_mul(64)
        .and_then(|aliases| aliases.checked_add(alias_count_end))
        .ok_or_else(|| {
            GraphStoreError::new("cold_object_index_invalid", "object index length overflow")
        })?;
    if bytes.len() != expected_len {
        return Err(GraphStoreError::new(
            "cold_object_index_invalid",
            format!(
                "object index declares {payload_count} payloads and {alias_count} aliases but has {} bytes",
                bytes.len()
            ),
        ));
    }

    let mut offset = 6;
    let payloads = decode_indexed_payloads(
        bytes,
        &mut offset,
        payload_count,
        allow_private_manifest_slots,
    )?;
    offset += 4;
    let mut blake3_aliases = BTreeMap::new();
    let mut previous = None;
    for _ in 0..alias_count {
        let legacy = Blake3Digest(
            bytes[offset..offset + 32]
                .try_into()
                .expect("validated Blake3 digest width"),
        );
        offset += 32;
        if previous.is_some_and(|prior| prior >= legacy) {
            return Err(GraphStoreError::new(
                "cold_object_index_not_canonical",
                "Blake3 alias digests must be strictly increasing",
            ));
        }
        let alias_record = Hash::from_bytes(
            bytes[offset..offset + 32]
                .try_into()
                .expect("validated alias record address width"),
        );
        offset += 32;
        previous = Some(legacy);
        blake3_aliases.insert(legacy, alias_record);
    }
    Ok(ObjectStoreIndex {
        payloads,
        blake3_aliases,
    })
}

fn decode_indexed_payloads(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
    allow_private_manifest_slots: bool,
) -> GraphStoreResult<BTreeMap<Hash, IndexedPayload>> {
    let mut payloads = BTreeMap::new();
    let mut previous = None;
    for _ in 0..count {
        let semantic = Hash::from_bytes(
            bytes[*offset..*offset + 32]
                .try_into()
                .expect("validated index address width"),
        );
        *offset += 32;
        if previous.is_some_and(|prior| prior >= semantic) {
            return Err(GraphStoreError::new(
                "cold_object_index_not_canonical",
                "object index semantic addresses must be strictly increasing",
            ));
        }
        let kind = match bytes[*offset] {
            1 => IndexedPayloadKind::Object,
            2 => IndexedPayloadKind::Pack,
            3 if allow_private_manifest_slots => IndexedPayloadKind::PrivateManifestSlot,
            tag => {
                return Err(GraphStoreError::new(
                    "cold_object_index_invalid",
                    format!("object index payload tag {tag} is invalid"),
                ))
            }
        };
        *offset += 1;
        let physical = Hash::from_bytes(
            bytes[*offset..*offset + 32]
                .try_into()
                .expect("validated index payload address width"),
        );
        *offset += 32;
        previous = Some(semantic);
        payloads.insert(semantic, IndexedPayload { kind, physical });
    }
    Ok(payloads)
}

fn encode_private_manifest_slot_identity(
    domain: &str,
    scope: &str,
    key: &str,
) -> GraphStoreResult<Vec<u8>> {
    validate_private_manifest_slot_string("domain", domain)?;
    validate_private_manifest_slot_string("scope", scope)?;
    validate_private_manifest_slot_string("key", key)?;
    let mut bytes = Vec::with_capacity(14 + domain.len() + scope.len() + key.len());
    bytes.push(DOMAIN_PRIVATE_MANIFEST_SLOT_IDENTITY);
    bytes.push(PRIVATE_MANIFEST_SLOT_VERSION);
    push_private_manifest_slot_string(&mut bytes, domain, "domain")?;
    push_private_manifest_slot_string(&mut bytes, scope, "scope")?;
    push_private_manifest_slot_string(&mut bytes, key, "key")?;
    Ok(bytes)
}

fn encode_private_manifest_slot(slot: &PrivateManifestSlot) -> GraphStoreResult<Vec<u8>> {
    slot.validate()?;
    let semantic = slot.semantic_address()?;
    let mut bytes = Vec::with_capacity(78 + slot.domain.len() + slot.scope.len() + slot.key.len());
    bytes.push(DOMAIN_PRIVATE_MANIFEST_SLOT);
    bytes.push(PRIVATE_MANIFEST_SLOT_VERSION);
    bytes.extend_from_slice(semantic.as_bytes());
    push_private_manifest_slot_string(&mut bytes, &slot.domain, "domain")?;
    push_private_manifest_slot_string(&mut bytes, &slot.scope, "scope")?;
    push_private_manifest_slot_string(&mut bytes, &slot.key, "key")?;
    bytes.extend_from_slice(slot.manifest_address.as_bytes());
    Ok(bytes)
}

fn decode_private_manifest_slot(
    expected: Hash,
    bytes: &[u8],
) -> GraphStoreResult<PrivateManifestSlot> {
    verify_physical_address(expected, bytes, "private manifest slot")?;
    if bytes.len() < 81
        || bytes[0] != DOMAIN_PRIVATE_MANIFEST_SLOT
        || bytes[1] != PRIVATE_MANIFEST_SLOT_VERSION
    {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            "private manifest slot has an invalid domain, version, or truncated header",
        ));
    }
    let semantic = Hash::from_bytes(
        bytes[2..34]
            .try_into()
            .expect("private slot minimum length was checked"),
    );
    let mut offset = 34;
    let domain = read_private_manifest_slot_string(bytes, &mut offset, "domain")?;
    let scope = read_private_manifest_slot_string(bytes, &mut offset, "scope")?;
    let key = read_private_manifest_slot_string(bytes, &mut offset, "key")?;
    let manifest_end = offset.checked_add(32).ok_or_else(|| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            "private manifest slot manifest offset overflow",
        )
    })?;
    if manifest_end != bytes.len() {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            "private manifest slot has a truncated manifest or trailing bytes",
        ));
    }
    let manifest_address = Hash::from_bytes(
        bytes[offset..manifest_end]
            .try_into()
            .expect("private slot manifest width was checked"),
    );
    let slot = PrivateManifestSlot::new(domain, scope, key, manifest_address)?;
    if slot.semantic_address()? != semantic {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_identity_mismatch",
            format!("private manifest envelope {expected} carries the wrong semantic identity"),
        ));
    }
    if encode_private_manifest_slot(&slot)?.as_slice() != bytes {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_not_canonical",
            format!("private manifest envelope {expected} is not canonical"),
        ));
    }
    Ok(slot)
}

fn validate_private_manifest_slot_string(field: &str, value: &str) -> GraphStoreResult<()> {
    if value.is_empty() || value.len() > MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!(
                "private manifest slot {field} must contain 1..={MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES} UTF-8 bytes"
            ),
        ));
    }
    Ok(())
}

fn push_private_manifest_slot_string(
    output: &mut Vec<u8>,
    value: &str,
    field: &str,
) -> GraphStoreResult<()> {
    validate_private_manifest_slot_string(field, value)?;
    let length = u32::try_from(value.len()).map_err(|_| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot {field} length exceeds u32"),
        )
    })?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn read_private_manifest_slot_string(
    bytes: &[u8],
    offset: &mut usize,
    field: &str,
) -> GraphStoreResult<String> {
    let length_end = offset.checked_add(4).ok_or_else(|| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot {field} length offset overflow"),
        )
    })?;
    let length_bytes = bytes.get(*offset..length_end).ok_or_else(|| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot is truncated before {field} length"),
        )
    })?;
    let length = usize::try_from(u32::from_le_bytes(
        length_bytes
            .try_into()
            .expect("private slot string length width was checked"),
    ))
    .expect("u32 fits every supported usize");
    if length == 0 || length > MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES {
        return Err(GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!(
                "private manifest slot {field} length must be 1..={MAX_PRIVATE_MANIFEST_SLOT_STRING_BYTES}"
            ),
        ));
    }
    let value_end = length_end.checked_add(length).ok_or_else(|| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot {field} length overflow"),
        )
    })?;
    let value_bytes = bytes.get(length_end..value_end).ok_or_else(|| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot is truncated in {field}"),
        )
    })?;
    let value = std::str::from_utf8(value_bytes).map_err(|error| {
        GraphStoreError::new(
            "cold_private_manifest_slot_invalid",
            format!("private manifest slot {field} is not UTF-8: {error}"),
        )
    })?;
    *offset = value_end;
    Ok(value.to_string())
}

fn encode_blake3_alias(alias: Blake3Alias) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(66);
    bytes.push(DOMAIN_BLAKE3_ALIAS);
    bytes.push(BLAKE3_ALIAS_VERSION);
    bytes.extend_from_slice(alias.legacy.as_bytes());
    bytes.extend_from_slice(alias.canonical.as_bytes());
    bytes
}

fn decode_blake3_alias(expected: Hash, bytes: &[u8]) -> GraphStoreResult<Blake3Alias> {
    verify_physical_address(expected, bytes, "Blake3 alias record")?;
    if bytes.len() != 66 || bytes[0] != DOMAIN_BLAKE3_ALIAS || bytes[1] != BLAKE3_ALIAS_VERSION {
        return Err(GraphStoreError::new(
            "cold_object_alias_invalid",
            "Blake3 alias has an invalid domain, version, length, or trailing bytes",
        ));
    }
    let alias = Blake3Alias {
        legacy: Blake3Digest(
            bytes[2..34]
                .try_into()
                .expect("validated Blake3 alias digest width"),
        ),
        canonical: Hash::from_bytes(
            bytes[34..66]
                .try_into()
                .expect("validated Blake3 alias target width"),
        ),
    };
    if encode_blake3_alias(alias) != bytes {
        return Err(GraphStoreError::new(
            "cold_object_alias_not_canonical",
            "Blake3 alias does not use the canonical encoding",
        ));
    }
    Ok(alias)
}

fn encode_pack_envelope(pack: &CompiledGraphPack) -> GraphStoreResult<Vec<u8>> {
    verify_pack_address(pack.commit.commit_hash, pack)?;
    let value = serde_json::to_value(pack)
        .map_err(|error| GraphStoreError::new("cold_pack_encode", error.to_string()))?;
    let body =
        canonical_json(&value).map_err(|error| GraphStoreError::new("cold_pack_encode", error))?;
    let body_len = u32::try_from(body.len()).map_err(|_| {
        GraphStoreError::new(
            "cold_pack_too_large",
            "canonical pack envelope exceeds the u32 framing limit",
        )
    })?;
    let mut bytes = Vec::with_capacity(6 + body.len());
    bytes.push(DOMAIN_PACK_ENVELOPE);
    bytes.push(PACK_ENVELOPE_VERSION);
    bytes.extend_from_slice(&body_len.to_le_bytes());
    bytes.extend_from_slice(body.as_bytes());
    Ok(bytes)
}

fn decode_pack_envelope(expected: Hash, bytes: &[u8]) -> GraphStoreResult<CompiledGraphPack> {
    verify_physical_address(expected, bytes, "pack envelope")?;
    if bytes.len() < 6 || bytes[0] != DOMAIN_PACK_ENVELOPE || bytes[1] != PACK_ENVELOPE_VERSION {
        return Err(GraphStoreError::new(
            "cold_pack_decode",
            "pack envelope has an invalid domain or version",
        ));
    }
    let body_len =
        u32::from_le_bytes(bytes[2..6].try_into().expect("four-byte pack length")) as usize;
    if bytes.len() != body_len.saturating_add(6) {
        return Err(GraphStoreError::new(
            "cold_pack_decode",
            "pack envelope length or trailing bytes are invalid",
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes[6..])
        .map_err(|error| GraphStoreError::new("cold_pack_decode", error.to_string()))?;
    let canonical =
        canonical_json(&value).map_err(|error| GraphStoreError::new("cold_pack_decode", error))?;
    if canonical.as_bytes() != &bytes[6..] {
        return Err(GraphStoreError::new(
            "cold_pack_not_canonical",
            "pack envelope JSON is not canonical",
        ));
    }
    let pack: CompiledGraphPack = serde_json::from_value(value)
        .map_err(|error| GraphStoreError::new("cold_pack_decode", error.to_string()))?;
    verify_pack_address(pack.commit.commit_hash, &pack)?;
    Ok(pack)
}

pub(crate) fn decode_object_store_reachability(
    expected: Hash,
    bytes: &[u8],
) -> GraphStoreResult<ObjectStoreReachability> {
    verify_physical_address(expected, bytes, "object-store reachability record")?;
    match bytes.first().copied() {
        Some(DOMAIN_OBJECT_STORE_INDEX) => {
            let index = decode_object_store_index(expected, bytes)?;
            let mut object_payloads = Vec::new();
            let mut pack_envelopes = index.blake3_aliases.values().copied().collect::<Vec<_>>();
            let mut private_manifest_envelopes = Vec::new();
            for payload in index.payloads.values() {
                match payload.kind {
                    IndexedPayloadKind::Object => object_payloads.push(payload.physical),
                    IndexedPayloadKind::Pack => pack_envelopes.push(payload.physical),
                    IndexedPayloadKind::PrivateManifestSlot => {
                        private_manifest_envelopes.push(payload.physical)
                    }
                }
            }
            Ok(ObjectStoreReachability::Index {
                object_payloads,
                pack_envelopes,
                private_manifest_envelopes,
            })
        }
        Some(DOMAIN_PACK_ENVELOPE) => {
            let pack = decode_pack_envelope(expected, bytes)?;
            let (file_manifests, terminals) = pack_embedded_references(&pack)?;
            // The canonical envelope is self-contained. Its embedded logical
            // roots and object representations are not independent chunks,
            // but typed references inside full node payloads still own their
            // external manifests and terminal bytes.
            Ok(ObjectStoreReachability::PackEnvelope {
                logical_roots: Vec::new(),
                payload_roots: Vec::new(),
                object_payloads: Vec::new(),
                file_manifests,
                terminals,
            })
        }
        Some(DOMAIN_BLAKE3_ALIAS) => {
            let alias = decode_blake3_alias(expected, bytes)?;
            Ok(ObjectStoreReachability::PackEnvelope {
                logical_roots: Vec::new(),
                payload_roots: Vec::new(),
                object_payloads: Vec::new(),
                file_manifests: Vec::new(),
                terminals: vec![alias.canonical],
            })
        }
        Some(DOMAIN_PRIVATE_MANIFEST_SLOT) => {
            let slot = decode_private_manifest_slot(expected, bytes)?;
            Ok(ObjectStoreReachability::PrivateManifestEnvelope {
                manifest: slot.manifest_address,
            })
        }
        Some(domain) => Err(GraphStoreError::new(
            "cold_object_reachability_domain_invalid",
            format!("object-store reachability domain {domain} is invalid"),
        )),
        None => Err(GraphStoreError::new(
            "cold_object_reachability_domain_invalid",
            "object-store reachability record is empty",
        )),
    }
}

fn pack_embedded_references(pack: &CompiledGraphPack) -> GraphStoreResult<(Vec<Hash>, Vec<Hash>)> {
    let mut file_manifests = BTreeSet::new();
    let mut terminals = BTreeSet::new();
    for object in &pack.objects {
        let GraphPayloadRecord::Node(node) = graph_payload_record(object)? else {
            continue;
        };
        let (node_file_manifests, node_terminals) = packed_graph_node_payload_references(&node)?;
        file_manifests.extend(node_file_manifests);
        terminals.extend(node_terminals);
    }
    Ok((
        file_manifests.into_iter().collect(),
        terminals.into_iter().collect(),
    ))
}

fn verify_physical_address(expected: Hash, bytes: &[u8], kind: &str) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual != expected {
        return Err(typed_hash_mismatch(kind, expected, actual));
    }
    Ok(())
}

fn parse_sha256_address(address: &str) -> Option<Hash> {
    Hash::parse_prefixed(address).ok()
}

fn parse_blake3_digest(address: &str) -> Option<Blake3Digest> {
    let encoded = address_hex(address, AddressAlgorithm::Blake3)?.as_bytes();
    let mut raw = [0_u8; 32];
    for (index, pair) in encoded.chunks_exact(2).enumerate() {
        let high = decode_lower_hex(pair[0])?;
        let low = decode_lower_hex(pair[1])?;
        raw[index] = (high << 4) | low;
    }
    Some(Blake3Digest(raw))
}

fn decode_lower_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn verify_blake3_digest(expected: Blake3Digest, bytes: &[u8], kind: &str) -> GraphStoreResult<()> {
    let actual = Blake3Digest(*blake3::hash(bytes).as_bytes());
    if actual != expected {
        return Err(GraphStoreError::new(
            "cold_object_hash_mismatch",
            format!(
                "{kind} address mismatch: requested blake3:{}, computed {}",
                encode_lower_hex(expected.as_bytes()),
                blake3_address(bytes)
            ),
        ));
    }
    Ok(())
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn internal_commit_meta(message: &'static str) -> CommitMeta {
    CommitMeta {
        parents: Vec::new(),
        author: "rustyred-object-store".to_string(),
        message: message.to_string(),
        timestamp_unix_ms: 0,
        graph_version: 0,
    }
}

fn io_err(context: &'static str) -> impl Fn(std::io::Error) -> GraphStoreError {
    move |error| GraphStoreError::new("cold_object_store_io", format!("{context}: {error}"))
}

fn poisoned<T>(_: T) -> GraphStoreError {
    GraphStoreError::new(
        "cold_object_store_poisoned",
        "cold object store mutex was poisoned".to_string(),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AddressAlgorithm {
    Sha256,
    Blake3,
}

impl AddressAlgorithm {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            Self::Blake3 => "blake3",
        }
    }
}

fn address_hex(address: &str, algorithm: AddressAlgorithm) -> Option<&str> {
    let hex = address
        .strip_prefix(algorithm.as_str())?
        .strip_prefix(':')?;
    (hex.len() == ADDRESS_HEX_LENGTH
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then_some(hex)
}

fn require_address(address: &str, algorithm: AddressAlgorithm) -> GraphStoreResult<&str> {
    address_hex(address, algorithm).ok_or_else(|| {
        GraphStoreError::new(
            "cold_object_address_invalid",
            format!(
                "expected canonical {}:<64 lowercase hex> address, got {address}",
                algorithm.as_str()
            ),
        )
    })
}

fn blake3_address(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn verify_bytes_address(requested: &str, actual: &str, kind: &str) -> GraphStoreResult<()> {
    if requested != actual {
        return Err(hash_mismatch(kind, requested, actual));
    }
    Ok(())
}

fn verify_object_address(requested: Hash, object: &GraphContentObject) -> GraphStoreResult<()> {
    if object.hash != requested {
        return Err(typed_hash_mismatch("object field", requested, object.hash));
    }
    let recomputed = recompute_content_object_hash(object)?;
    if recomputed != requested {
        return Err(typed_hash_mismatch(
            "canonical object",
            requested,
            recomputed,
        ));
    }
    Ok(())
}

fn verify_pack_address(requested: Hash, pack: &CompiledGraphPack) -> GraphStoreResult<()> {
    if pack.commit.commit_hash != requested {
        return Err(typed_hash_mismatch(
            "pack commit",
            requested,
            pack.commit.commit_hash,
        ));
    }
    let recomputed = recompute_commit_hash(&pack.commit)?;
    if recomputed != requested {
        return Err(typed_hash_mismatch(
            "canonical pack commit",
            requested,
            recomputed,
        ));
    }
    Ok(())
}

fn typed_hash_mismatch(kind: &str, requested: Hash, actual: Hash) -> GraphStoreError {
    GraphStoreError::new(
        "cold_object_hash_mismatch",
        format!("{kind} address mismatch: requested {requested}, computed {actual}"),
    )
}

fn hash_mismatch(kind: &str, requested: &str, actual: &str) -> GraphStoreError {
    GraphStoreError::new(
        "cold_object_hash_mismatch",
        format!("{kind} address mismatch: requested {requested}, computed {actual}"),
    )
}

#[cfg(test)]
mod unified_tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Barrier;

    use serde_json::json;

    use super::*;
    use crate::chunk_store::{ChunkStoreCompaction, MemChunkStore};
    use crate::file_payload::stage_file_payload;
    use crate::graph::record::NodeRecord;
    use crate::graph::snapshot::GraphSnapshot;
    use crate::reachability::{walk_committed_reachable, CoreReachabilityCodec};
    use crate::tensor_derivation::TensorBlockPayloadStore;
    use crate::versioned_graph::{
        compile_graph_pack, node_from_content_object, node_to_content_object, GraphCompileOptions,
        DOMAIN_GRAPH_PAYLOAD,
    };

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn unique_temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{name}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn memory_facade() -> GraphStoreResult<(Arc<MemChunkStore>, DiskObjectStore)> {
        let concrete = Arc::new(MemChunkStore::new());
        let backend: Arc<dyn UnifiedChunkStore> = concrete.clone();
        let facade = DiskObjectStore::from_chunk_store(backend)?;
        Ok((concrete, facade))
    }

    fn install_old_root_compaction(
        backend: &Arc<MemChunkStore>,
        store: &DiskObjectStore,
    ) -> GraphStoreResult<(
        usize,
        Arc<Mutex<Option<crate::chunk_store::CompactionReport>>>,
    )> {
        let expected_root = backend.read_root()?;
        let reachable = walk_committed_reachable(backend.as_ref(), &CoreReachabilityCodec)?
            .reachable
            .into_iter()
            .collect::<Vec<_>>();
        let prior_reachable_count = reachable.len();
        let report = Arc::new(Mutex::new(None));
        let hook_report = Arc::clone(&report);
        let hook_backend = Arc::clone(backend);
        store.install_before_catalog_commit_hook(move || {
            let compaction = hook_backend.compact_reachable(expected_root, &reachable)?;
            *hook_report.lock().map_err(poisoned)? = Some(compaction);
            Ok(())
        })?;
        Ok((prior_reachable_count, report))
    }

    fn versioned_object() -> GraphStoreResult<GraphContentObject> {
        let mut node = NodeRecord::new(
            "mem:versioned",
            ["Memory", "Episode"],
            json!({"body": "full fidelity"}),
        );
        node.version = 17;
        node.content_hash = Some(node.checksum()?);
        node.parent_hashes = vec![Hash::of(b"parent-a").to_prefixed()];
        node_to_content_object(&node)
    }

    fn object_for(id: &str) -> GraphStoreResult<GraphContentObject> {
        let mut node = NodeRecord::new(id, ["Memory"], json!({"id": id}));
        node.content_hash = Some(node.checksum()?);
        node_to_content_object(&node)
    }

    fn pack_for(id: &str) -> GraphStoreResult<CompiledGraphPack> {
        compile_graph_pack(
            &GraphSnapshot {
                version: 9,
                nodes: vec![NodeRecord::new(id, ["Memory"], json!({"name": "pack"}))],
                ..GraphSnapshot::default()
            },
            GraphCompileOptions {
                author: Some("object-store-test".to_string()),
                message: Some("canonical parked pack".to_string()),
                timestamp_unix_ms: Some(7),
                ..GraphCompileOptions::default()
            },
        )
    }

    #[test]
    fn disk_object_store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DiskObjectStore>();
        assert_send_sync::<InMemoryObjectStore>();
    }

    #[test]
    fn tensor_payload_facade_keys_pending_receipts_without_cross_node_stealing(
    ) -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let body = b"identical tensor payload producers";
        let address = Hash::of(body);
        let content_id = address.to_prefixed();
        let node_a_key = "node-a:tensor_payload:shared";
        let node_b_key = "node-b:tensor_payload:shared";

        store.put_tensor_block_bytes(&content_id, body)?;
        store.put_tensor_block_bytes(&content_id, body)?;
        assert_eq!(backend.pending_pin_count(address)?, 2);
        store.bind_tensor_block_pending(&content_id, node_b_key)?;
        store.bind_tensor_block_pending(&content_id, node_b_key)?;
        assert_eq!(backend.anonymous_pending_pin_count(address)?, 0);
        assert_eq!(backend.bound_pending_pin_count(address)?, 2);

        // Node A's idempotent graph publication owns a fresh claim and cannot
        // release node B's pending writer receipts for the shared hash.
        store.claim_tensor_block_pin(&content_id)?;
        assert_eq!(backend.pending_pin_count(address)?, 2);
        assert_eq!(backend.claimed_pin_count(address)?, 1);
        store.release_claimed_tensor_block_pin(&content_id)?;
        assert!(!store.release_bound_tensor_block_pending(&content_id, node_a_key)?);
        assert_eq!(backend.bound_pending_pin_count(address)?, 2);

        // Repeated writes for one logical key remain independently owned.
        assert!(store.release_bound_tensor_block_pending(&content_id, node_b_key)?);
        assert_eq!(backend.bound_pending_pin_count(address)?, 1);
        assert!(store.release_bound_tensor_block_pending(&content_id, node_b_key)?);
        assert_eq!(backend.in_flight_pin_count(address)?, 0);

        let external_body = b"external tensor payload reference";
        let external = Hash::of(external_body);
        let external_id = external.to_prefixed();
        backend.write(&[(external, external_body)])?;
        store.claim_tensor_block_pin(&external_id)?;
        store.put_tensor_block_bytes(&external_id, external_body)?;
        store.release_tensor_block_pin(&external_id)?;
        assert_eq!(backend.pending_pin_count(external)?, 0);
        assert_eq!(backend.claimed_pin_count(external)?, 1);
        store.release_claimed_tensor_block_pin(&external_id)?;
        assert_eq!(backend.in_flight_pin_count(external)?, 0);
        Ok(())
    }

    #[test]
    fn document_and_chunk_share_one_physical_chunk() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let body = b"one logical payload across both lenses";
        let address = Hash::of(body);
        assert_eq!(backend.missing(&[address])?, vec![address]);

        let document = store.put_document_bytes(body)?;
        assert_eq!(document, address.to_prefixed());
        assert!(backend.has(address)?);
        assert_eq!(
            backend.write_accounted(&[(address, body.as_slice())])?,
            crate::chunk_store::ChunkWriteReport::default()
        );

        let chunk = store.put_chunk(body)?;
        assert_eq!(chunk, document);
        assert_eq!(store.chunk_count()?, 1);
        assert_eq!(
            store.get_document_bytes(&document)?.as_deref(),
            Some(body.as_slice())
        );
        assert_eq!(store.get_chunk(&chunk)?.as_deref(), Some(body.as_slice()));
        Ok(())
    }

    #[test]
    fn staged_raw_payload_pin_spans_compaction_and_releases_without_leaking() -> GraphStoreResult<()>
    {
        let (backend, store) = memory_facade()?;
        let body = b"raw payload awaiting graph publication";
        let address = Hash::of(body);
        let expected_root = backend.read_root()?;
        let reachable = walk_committed_reachable(backend.as_ref(), &CoreReachabilityCodec)?
            .reachable
            .into_iter()
            .collect::<Vec<_>>();

        let staged = store.stage_document_bytes(body)?;
        assert_eq!(staged.address(), address);
        assert_eq!(backend.in_flight_pin_count(address)?, 1);
        backend.compact_reachable(expected_root, &reachable)?;
        assert!(backend.has(address)?);

        backend.commit_catalog_updates(
            &BTreeMap::new(),
            internal_commit_meta("test staged raw publication barrier"),
        )?;
        staged.finish_after_publication();
        assert_eq!(backend.in_flight_pin_count(address)?, 0);

        let current_root = backend.read_root()?;
        let current = walk_committed_reachable(backend.as_ref(), &CoreReachabilityCodec)?
            .reachable
            .into_iter()
            .collect::<Vec<_>>();
        backend.compact_reachable(current_root, &current)?;
        assert!(!backend.has(address)?, "unowned raw bytes are reclaimable");
        Ok(())
    }

    #[test]
    fn indexed_payload_publication_pins_survive_old_root_compaction() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let (prior_reachable_count, hook_report) = install_old_root_compaction(&backend, &store)?;
        let object = versioned_object()?;

        store.put_object(&object)?;

        let report = hook_report
            .lock()
            .map_err(poisoned)?
            .take()
            .expect("indexed publication compaction hook ran");
        assert!(report.chunks_after > prior_reachable_count);
        assert_eq!(store.get_object(object.hash)?, Some(object.clone()));
        let index = store.load_index()?;
        let physical = index.payloads[&object.hash].physical;
        let catalog_address = backend.read_root()?.expect("catalog root");
        let catalog = validate_catalog_root(backend.as_ref(), catalog_address)?;
        let index_address = catalog.get(ROOT_NAME_OBJECTS).expect("objects root");
        assert_eq!(backend.in_flight_pin_count(physical)?, 0);
        assert_eq!(backend.in_flight_pin_count(index_address)?, 0);
        Ok(())
    }

    #[test]
    fn blake3_alias_publication_pins_survive_old_root_compaction() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let (prior_reachable_count, hook_report) = install_old_root_compaction(&backend, &store)?;
        let body = b"legacy alias bytes protected during publication";
        let legacy = Blake3Digest(*blake3::hash(body).as_bytes());

        store.publish_blake3_alias(legacy, body)?;

        let report = hook_report
            .lock()
            .map_err(poisoned)?
            .take()
            .expect("alias publication compaction hook ran");
        assert!(report.chunks_after > prior_reachable_count);
        assert_eq!(
            store.get_chunk(&format!("blake3:{}", encode_lower_hex(legacy.as_bytes())))?,
            Some(body.to_vec())
        );
        let index = store.load_index()?;
        let alias_address = index.blake3_aliases[&legacy];
        let canonical = Hash::of(body);
        let catalog_address = backend.read_root()?.expect("catalog root");
        let catalog = validate_catalog_root(backend.as_ref(), catalog_address)?;
        let index_address = catalog.get(ROOT_NAME_OBJECTS).expect("objects root");
        for address in [canonical, alias_address, index_address] {
            assert_eq!(backend.in_flight_pin_count(address)?, 0);
        }
        Ok(())
    }

    #[test]
    fn full_object_payload_preserves_version_and_lineage() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let object = versioned_object()?;
        store.put_object(&object)?;

        let indexed = store
            .load_index()?
            .payloads
            .get(&object.hash)
            .copied()
            .expect("object index entry");
        assert_eq!(indexed.kind, IndexedPayloadKind::Object);
        assert_ne!(indexed.physical, object.hash);
        assert_eq!(
            store.object_payload_address(object.hash)?,
            Some(indexed.physical)
        );
        assert_eq!(
            store.pack_payload_address(object.hash).unwrap_err().code,
            "cold_object_index_kind_mismatch"
        );
        let physical = backend.get(indexed.physical)?.expect("payload chunk");
        assert_eq!(physical.first(), Some(&DOMAIN_GRAPH_PAYLOAD));

        let fetched = store.get_object(object.hash)?.expect("stored object");
        assert_eq!(fetched, object);
        let node = node_from_content_object(&fetched).expect("node payload");
        assert_eq!(node.version, 17);
        assert_eq!(
            node.parent_hashes,
            vec![Hash::of(b"parent-a").to_prefixed()]
        );
        Ok(())
    }

    #[test]
    fn pack_semantic_hash_maps_to_a_distinct_physical_envelope() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let pack = pack_for("mem:pack")?;
        let semantic = pack.commit.commit_hash;
        store.put_pack(&pack)?;

        let indexed = store
            .load_index()?
            .payloads
            .get(&semantic)
            .copied()
            .expect("pack index entry");
        assert_eq!(indexed.kind, IndexedPayloadKind::Pack);
        assert_ne!(indexed.physical, semantic);
        assert_eq!(
            store.pack_payload_address(semantic)?,
            Some(indexed.physical)
        );
        assert_eq!(
            store.object_payload_address(semantic).unwrap_err().code,
            "cold_object_index_kind_mismatch"
        );
        let envelope = backend.get(indexed.physical)?.expect("pack envelope");
        assert_eq!(envelope.first(), Some(&DOMAIN_PACK_ENVELOPE));
        assert_eq!(store.get_pack(semantic)?, Some(pack));
        Ok(())
    }

    #[test]
    fn journal_facade_survives_reopen() -> GraphStoreResult<()> {
        let root = unique_temp("unified-object-reopen");
        let _ = fs::remove_dir_all(&root);
        let body = b"durable unified body";
        let object = versioned_object()?;
        {
            let store = DiskObjectStore::open(&root)?;
            store.put_document_bytes(body)?;
            store.put_object(&object)?;
        }
        {
            let reopened = DiskObjectStore::open(&root)?;
            assert_eq!(
                reopened
                    .get_document_bytes(&content_hash_bytes(body))?
                    .as_deref(),
                Some(body.as_slice())
            );
            assert_eq!(reopened.get_object(object.hash)?, Some(object));
            assert_eq!(reopened.object_count()?, 1);
        }
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn legacy_object_journal_import_read_does_not_advance_catalog() -> GraphStoreResult<()> {
        let root = unique_temp("legacy-object-journal-read-only");
        let _ = fs::remove_dir_all(&root);
        let body = b"legacy overflow body";
        let content_id;
        {
            let store = DiskObjectStore::open(&root)?;
            content_id = store.put_document_bytes(body)?;
        }
        let journal_path = root.join(UNIFIED_JOURNAL_FILE);
        let length_before = fs::metadata(&journal_path)
            .map_err(io_err("stat legacy object journal"))?
            .len();
        {
            let legacy = DiskObjectStore::open_legacy_journal_for_import(&root)?
                .expect("legacy object journal");
            let root_before = legacy.store.read_root()?;
            assert_eq!(
                legacy.get_document_bytes_read_only(&content_id)?.as_deref(),
                Some(body.as_slice())
            );
            assert_eq!(legacy.store.read_root()?, root_before);
        }
        assert_eq!(
            fs::metadata(&journal_path)
                .map_err(io_err("restat legacy object journal"))?
                .len(),
            length_before
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn legacy_document_cleanup_preserves_unrelated_roots_and_chunks() -> GraphStoreResult<()> {
        let legacy_root = unique_temp("legacy-document-root-cleanup");
        let _ = fs::remove_dir_all(&legacy_root);
        let legacy = DiskObjectStore::open(&legacy_root)?;
        let document_root_bytes = b"legacy document tree root";
        let object_root_bytes = b"unrelated legacy object index root";
        let document_root = Hash::of(document_root_bytes);
        let object_root = Hash::of(object_root_bytes);
        legacy.store.write(&[
            (document_root, &document_root_bytes[..]),
            (object_root, &object_root_bytes[..]),
        ])?;
        legacy.store.commit_catalog_updates(
            &BTreeMap::from([
                (ROOT_NAME_DOCUMENTS.to_string(), Some(document_root)),
                (ROOT_NAME_OBJECTS.to_string(), Some(object_root)),
            ]),
            internal_commit_meta("install legacy roots for cleanup test"),
        )?;
        let legacy_before_address = legacy.store.read_root()?.expect("legacy catalog");
        let legacy_before =
            validate_catalog_root(legacy.store.as_chunk_store(), legacy_before_address)?;
        let legacy_graph_root = legacy_before.graph_root()?;

        let (unified, destination) = memory_facade()?;
        let manifest_id = destination.put_document_bytes(b"imported file manifest")?;
        let manifest = Hash::parse_prefixed(&manifest_id)?;
        let unified_catalog_address = unified.read_root()?.expect("destination catalog");
        let expected_graph_root =
            validate_catalog_root(unified.as_ref(), unified_catalog_address)?.graph_root()?;

        let wrong_graph = legacy
            .clear_legacy_documents_root_after_graph_import(
                unified.as_ref(),
                Hash::of(b"stale graph root"),
                &[manifest],
                internal_commit_meta("must not clear with stale graph proof"),
            )
            .unwrap_err();
        assert_eq!(wrong_graph.code, "legacy_object_graph_import_not_durable");
        let missing_manifest = legacy
            .clear_legacy_documents_root_after_graph_import(
                unified.as_ref(),
                expected_graph_root,
                &[Hash::of(b"missing manifest")],
                internal_commit_meta("must not clear with missing manifest proof"),
            )
            .unwrap_err();
        assert_eq!(
            missing_manifest.code,
            "legacy_object_import_payload_missing"
        );

        assert!(legacy
            .clear_legacy_documents_root_after_graph_import(
                unified.as_ref(),
                expected_graph_root,
                &[manifest],
                internal_commit_meta("retire only legacy documents root"),
            )?
            .is_some());
        let legacy_after_address = legacy.store.read_root()?.expect("updated legacy catalog");
        let legacy_after =
            validate_catalog_root(legacy.store.as_chunk_store(), legacy_after_address)?;
        assert_eq!(legacy_after.get(ROOT_NAME_DOCUMENTS), None);
        assert_eq!(legacy_after.get(ROOT_NAME_OBJECTS), Some(object_root));
        assert_eq!(legacy_after.graph_root()?, legacy_graph_root);
        assert!(legacy.store.has(document_root)?);
        assert!(legacy.store.has(object_root)?);
        assert!(legacy_root.join(UNIFIED_JOURNAL_FILE).is_file());
        assert_eq!(
            legacy.clear_legacy_documents_root_after_graph_import(
                unified.as_ref(),
                expected_graph_root,
                &[manifest],
                internal_commit_meta("idempotent document-root cleanup"),
            )?,
            None
        );

        drop(legacy);
        let _ = fs::remove_dir_all(legacy_root);
        Ok(())
    }

    #[test]
    fn legacy_object_journal_retirement_requires_unified_graph_and_payload_proof(
    ) -> GraphStoreResult<()> {
        let legacy_root = unique_temp("legacy-object-journal-retirement");
        let _ = fs::remove_dir_all(&legacy_root);
        let body = b"body copied from retired journal";
        {
            let legacy = DiskObjectStore::open(&legacy_root)?;
            legacy.put_document_bytes(body)?;
        }

        let (unified, destination) = memory_facade()?;
        let content_id = destination.put_document_bytes(body)?;
        let imported = Hash::parse_prefixed(&content_id)?;
        let catalog_address = unified.read_root()?.expect("destination catalog");
        let graph_root = validate_catalog_root(unified.as_ref(), catalog_address)?.graph_root()?;

        let wrong_root = DiskObjectStore::retire_legacy_journal_after_graph_import(
            &legacy_root,
            unified.as_ref(),
            Hash::of(b"stale graph root"),
            &[imported],
        )
        .unwrap_err();
        assert_eq!(wrong_root.code, "legacy_object_graph_import_not_durable");
        let missing = DiskObjectStore::retire_legacy_journal_after_graph_import(
            &legacy_root,
            unified.as_ref(),
            graph_root,
            &[Hash::of(b"missing imported payload")],
        )
        .unwrap_err();
        assert_eq!(missing.code, "legacy_object_import_payload_missing");

        let retired = DiskObjectStore::retire_legacy_journal_after_graph_import(
            &legacy_root,
            unified.as_ref(),
            graph_root,
            &[imported],
        )?
        .expect("legacy journal should retire once");
        assert_eq!(
            retired,
            legacy_root.join(format!("{UNIFIED_JOURNAL_FILE}.migrated"))
        );
        assert!(!legacy_root.join(UNIFIED_JOURNAL_FILE).exists());
        assert!(retired.exists());
        assert_eq!(fs::read(&retired).is_ok(), true);
        assert_eq!(
            DiskObjectStore::retire_legacy_journal_after_graph_import(
                &legacy_root,
                unified.as_ref(),
                graph_root,
                &[imported],
            )?,
            Some(retired)
        );
        let _ = fs::remove_dir_all(legacy_root);
        Ok(())
    }

    #[test]
    fn legacy_blake3_alias_survives_source_removal_and_reopen_once() -> GraphStoreResult<()> {
        let root = unique_temp("unified-object-legacy-blake3-alias");
        let _ = fs::remove_dir_all(&root);
        let body = b"legacy blake3 payload with a durable canonical alias";
        let address = blake3_address(body);
        let legacy_digest = parse_blake3_digest(&address).expect("canonical Blake3 digest");
        let hex = address.strip_prefix("blake3:").expect("Blake3 address");
        let legacy = root
            .join("chunks")
            .join("blake3")
            .join(format!("{hex}.zst"));
        fs::create_dir_all(legacy.parent().expect("legacy parent"))
            .map_err(io_err("create legacy Blake3 fixture"))?;
        let fixture = compress_cold_bytes(body)?;
        fs::write(&legacy, &fixture).map_err(io_err("write legacy Blake3 fixture"))?;

        let alias_address;
        let objects_root;
        {
            let store = DiskObjectStore::open(&root)?;
            assert_eq!(store.get_chunk(&address)?.as_deref(), Some(body.as_slice()));
            assert_eq!(
                fs::read(&legacy).map_err(io_err("reread legacy Blake3 fixture"))?,
                fixture
            );
            let index = store.load_index()?;
            assert_eq!(index.blake3_aliases.len(), 1);
            alias_address = index.blake3_aliases[&legacy_digest];
            let alias_bytes = store.store.get(alias_address)?.expect("alias record");
            let alias = decode_blake3_alias(alias_address, &alias_bytes)?;
            assert_eq!(alias.legacy, legacy_digest);
            assert_eq!(alias.canonical, Hash::of(body));
            let catalog_address = store.store.read_root()?.expect("catalog root");
            objects_root = validate_catalog_root(store.store.as_chunk_store(), catalog_address)?
                .get(ROOT_NAME_OBJECTS)
                .expect("objects root");
            let index_bytes = store.store.get(objects_root)?.expect("objects index");
            let ObjectStoreReachability::Index { pack_envelopes, .. } =
                decode_object_store_reachability(objects_root, &index_bytes)?
            else {
                panic!("expected object index reachability");
            };
            assert!(pack_envelopes.contains(&alias_address));
            let ObjectStoreReachability::PackEnvelope { terminals, .. } =
                decode_object_store_reachability(alias_address, &alias_bytes)?
            else {
                panic!("expected alias envelope reachability");
            };
            assert_eq!(terminals, vec![Hash::of(body)]);

            assert_eq!(store.get_chunk(&address)?.as_deref(), Some(body.as_slice()));
            let repeated_catalog = store.store.read_root()?.expect("catalog root");
            assert_eq!(
                validate_catalog_root(store.store.as_chunk_store(), repeated_catalog)?
                    .get(ROOT_NAME_OBJECTS),
                Some(objects_root)
            );
            assert_eq!(store.load_index()?.blake3_aliases.len(), 1);
        }

        fs::remove_dir_all(root.join("chunks")).map_err(io_err("remove legacy Blake3 source"))?;
        let journal = root.join(UNIFIED_JOURNAL_FILE);
        let journal_len = fs::metadata(&journal)
            .map_err(io_err("stat unified journal"))?
            .len();
        {
            let reopened = DiskObjectStore::open(&root)?;
            assert_eq!(
                reopened.get_chunk(&address)?.as_deref(),
                Some(body.as_slice())
            );
            assert_eq!(
                reopened.get_chunk(&address)?.as_deref(),
                Some(body.as_slice())
            );
            let index = reopened.load_index()?;
            assert_eq!(index.blake3_aliases.len(), 1);
            assert_eq!(index.blake3_aliases[&legacy_digest], alias_address);
            let catalog_address = reopened.store.read_root()?.expect("catalog root");
            assert_eq!(
                validate_catalog_root(reopened.store.as_chunk_store(), catalog_address)?
                    .get(ROOT_NAME_OBJECTS),
                Some(objects_root)
            );
            assert_eq!(reopened.chunk_count()?, 1);
        }
        assert_eq!(
            fs::metadata(&journal)
                .map_err(io_err("restat unified journal"))?
                .len(),
            journal_len,
            "alias reads must not append chunks, aliases, or catalog commits"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn independent_facades_share_the_backend_index_mutation_lock() -> GraphStoreResult<()> {
        let backend = Arc::new(MemChunkStore::new());
        let first_backend: Arc<dyn UnifiedChunkStore> = backend.clone();
        let second_backend: Arc<dyn UnifiedChunkStore> = backend.clone();
        let first = DiskObjectStore::from_chunk_store(first_backend)?;
        let second = DiskObjectStore::from_chunk_store(second_backend)?;
        assert!(Arc::ptr_eq(&first.index_update, &second.index_update));

        let first_object = object_for("mem:independent-facade-a")?;
        let second_object = object_for("mem:independent-facade-b")?;
        let barrier = Arc::new(Barrier::new(3));
        let first_start = Arc::clone(&barrier);
        let first_expected = first_object.clone();
        let first_writer = std::thread::spawn(move || {
            first_start.wait();
            first.put_object(&first_object)
        });
        let second_start = Arc::clone(&barrier);
        let second_expected = second_object.clone();
        let second_writer = std::thread::spawn(move || {
            second_start.wait();
            second.put_object(&second_object)
        });
        barrier.wait();
        first_writer.join().expect("first writer panicked")?;
        second_writer.join().expect("second writer panicked")?;

        let verifier_backend: Arc<dyn UnifiedChunkStore> = backend;
        let verifier = DiskObjectStore::from_chunk_store(verifier_backend)?;
        assert_eq!(
            verifier.get_object(first_expected.hash)?,
            Some(first_expected)
        );
        assert_eq!(
            verifier.get_object(second_expected.hash)?,
            Some(second_expected)
        );
        assert_eq!(verifier.object_count()?, 2);
        Ok(())
    }

    #[test]
    fn current_index_decoder_accepts_the_strict_version_one_fixture() -> GraphStoreResult<()> {
        let semantic = Hash::of(b"legacy index semantic");
        let payload = IndexedPayload {
            kind: IndexedPayloadKind::Object,
            physical: Hash::of(b"legacy index physical"),
        };
        let payloads = BTreeMap::from([(semantic, payload)]);
        let bytes = encode_legacy_object_store_index(&payloads)?;
        let decoded = decode_object_store_index(Hash::of(&bytes), &bytes)?;
        assert_eq!(decoded.payloads, payloads);
        assert!(decoded.blake3_aliases.is_empty());

        let version_two = ObjectStoreIndex {
            payloads: decoded.payloads.clone(),
            blake3_aliases: BTreeMap::new(),
        };
        let version_two_bytes =
            encode_object_store_index_version(&version_two, BLAKE3_OBJECT_STORE_INDEX_VERSION)?;
        assert_eq!(
            decode_object_store_index(Hash::of(&version_two_bytes), &version_two_bytes)?,
            version_two
        );
        Ok(())
    }

    #[test]
    fn legacy_document_migration_is_read_only_and_idempotent() -> GraphStoreResult<()> {
        let root = unique_temp("unified-object-legacy-document");
        let _ = fs::remove_dir_all(&root);
        let body = b"legacy compressed document";
        let address = content_hash_bytes(body);
        let hex = address.strip_prefix("sha256:").expect("sha address");
        let legacy = root
            .join("documents")
            .join("sha256")
            .join(format!("{hex}.zst"));
        fs::create_dir_all(legacy.parent().expect("legacy parent"))
            .map_err(io_err("create legacy fixture"))?;
        let fixture = compress_cold_bytes(body)?;
        fs::write(&legacy, &fixture).map_err(io_err("write legacy fixture"))?;

        let store = DiskObjectStore::open(&root)?;
        assert_eq!(
            store.get_document_bytes(&address)?.as_deref(),
            Some(body.as_slice())
        );
        assert_eq!(
            fs::read(&legacy).map_err(io_err("reread legacy fixture"))?,
            fixture
        );
        fs::remove_file(&legacy).map_err(io_err("remove legacy fixture"))?;
        assert_eq!(
            store.get_document_bytes(&address)?.as_deref(),
            Some(body.as_slice())
        );
        drop(store);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn legacy_object_migrates_to_the_rooted_index_once() -> GraphStoreResult<()> {
        let root = unique_temp("unified-object-legacy-object");
        let _ = fs::remove_dir_all(&root);
        let object = versioned_object()?;
        let address = object.hash.to_prefixed();
        let legacy = root
            .join("objects")
            .join(format!("{}.json", safe_filename(&address)));
        fs::create_dir_all(legacy.parent().expect("legacy parent"))
            .map_err(io_err("create legacy object fixture"))?;
        let fixture = compress_cold_bytes(
            &serde_json::to_vec(&object)
                .map_err(|error| GraphStoreError::new("test_encode", error.to_string()))?,
        )?;
        fs::write(&legacy, &fixture).map_err(io_err("write legacy object fixture"))?;

        let store = DiskObjectStore::open(&root)?;
        assert_eq!(store.get_object(object.hash)?, Some(object.clone()));
        assert_eq!(store.object_count()?, 1);
        fs::remove_file(&legacy).map_err(io_err("remove legacy object fixture"))?;
        assert_eq!(store.get_object(object.hash)?, Some(object));
        drop(store);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn private_manifest_slot_retargets_one_stable_semantic_identity() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let first = stage_file_payload(backend.as_ref(), "private-slot:first", b"first snapshot")?;
        let first_manifest = first.reference().address()?;
        let first_slot = PrivateManifestSlot::new(
            "test.private-manifest.v1",
            "scope:a",
            "region:body",
            first_manifest,
        )?;
        let semantic = first_slot.semantic_address()?;
        let first_envelope = store.publish_private_manifest_slot(&first_slot)?;
        first.finish_after_publication();

        let second =
            stage_file_payload(backend.as_ref(), "private-slot:second", b"second snapshot")?;
        let second_manifest = second.reference().address()?;
        let second_slot = PrivateManifestSlot::new(
            "test.private-manifest.v1",
            "scope:a",
            "region:body",
            second_manifest,
        )?;
        assert_eq!(second_slot.semantic_address()?, semantic);
        let second_envelope = store.publish_private_manifest_slot(&second_slot)?;
        second.finish_after_publication();
        assert_ne!(first_envelope, second_envelope);

        assert_eq!(
            store.private_manifest_slot(semantic)?,
            Some(second_slot.clone())
        );
        assert_eq!(
            store.private_manifest_slots("test.private-manifest.v1")?,
            vec![(semantic, second_slot)]
        );

        let catalog_address = backend.read_root()?.expect("catalog root");
        let catalog = validate_catalog_root(backend.as_ref(), catalog_address)?;
        let index_address = catalog.get(ROOT_NAME_OBJECTS).expect("objects root");
        let index_bytes = backend.get(index_address)?.expect("index bytes");
        let ObjectStoreReachability::Index {
            private_manifest_envelopes,
            ..
        } = decode_object_store_reachability(index_address, &index_bytes)?
        else {
            panic!("expected object index reachability");
        };
        assert_eq!(private_manifest_envelopes, vec![second_envelope]);
        let envelope_bytes = backend.get(second_envelope)?.expect("slot envelope");
        assert_eq!(
            decode_object_store_reachability(second_envelope, &envelope_bytes)?,
            ObjectStoreReachability::PrivateManifestEnvelope {
                manifest: second_manifest,
            }
        );
        assert!(backend.has(first_manifest)?);
        Ok(())
    }

    #[test]
    fn corrupt_private_manifest_slot_fails_closed() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let semantic = Hash::of(b"corrupt private semantic slot");
        let malformed = [
            DOMAIN_PRIVATE_MANIFEST_SLOT,
            PRIVATE_MANIFEST_SLOT_VERSION,
            0,
        ];
        store.publish_indexed_payload(
            semantic,
            IndexedPayloadKind::PrivateManifestSlot,
            &malformed,
            "publish corrupt private manifest fixture",
        )?;

        assert_eq!(
            store
                .private_manifest_slots("test.private-manifest.v1")
                .unwrap_err()
                .code,
            "cold_private_manifest_slot_invalid"
        );
        let catalog_address = backend.read_root()?.expect("catalog root");
        let catalog = validate_catalog_root(backend.as_ref(), catalog_address)?;
        let index_address = catalog.get(ROOT_NAME_OBJECTS).expect("objects root");
        let index_bytes = backend.get(index_address)?.expect("index bytes");
        let ObjectStoreReachability::Index {
            private_manifest_envelopes,
            ..
        } = decode_object_store_reachability(index_address, &index_bytes)?
        else {
            panic!("expected object index reachability");
        };
        let corrupt_envelope = private_manifest_envelopes[0];
        let corrupt_bytes = backend.get(corrupt_envelope)?.expect("corrupt envelope");
        assert_eq!(
            decode_object_store_reachability(corrupt_envelope, &corrupt_bytes)
                .unwrap_err()
                .code,
            "cold_private_manifest_slot_invalid"
        );
        Ok(())
    }

    #[test]
    fn reachability_codec_is_typed_and_rejects_trailing_bytes() -> GraphStoreResult<()> {
        let (backend, store) = memory_facade()?;
        let object = versioned_object()?;
        let pack = pack_for("mem:reachability-pack")?;
        store.put_object(&object)?;
        store.put_pack(&pack)?;

        let catalog_address = backend.read_root()?.expect("catalog root");
        let catalog = validate_catalog_root(backend.as_ref(), catalog_address)?;
        let index_address = catalog.get(ROOT_NAME_OBJECTS).expect("objects root");
        let index_bytes = backend.get(index_address)?.expect("index bytes");
        let reachability = decode_object_store_reachability(index_address, &index_bytes)?;
        let ObjectStoreReachability::Index {
            object_payloads,
            pack_envelopes,
            private_manifest_envelopes,
        } = reachability
        else {
            panic!("expected index reachability");
        };
        assert_eq!(object_payloads.len(), 1);
        assert_eq!(pack_envelopes.len(), 1);
        assert!(private_manifest_envelopes.is_empty());

        let pack_bytes = backend.get(pack_envelopes[0])?.expect("pack bytes");
        assert!(matches!(
            decode_object_store_reachability(pack_envelopes[0], &pack_bytes)?,
            ObjectStoreReachability::PackEnvelope { .. }
        ));
        let mut trailing = pack_bytes.to_vec();
        trailing.push(0);
        let trailing_address = Hash::of(&trailing);
        assert_eq!(
            decode_object_store_reachability(trailing_address, &trailing)
                .unwrap_err()
                .code,
            "cold_pack_decode"
        );
        Ok(())
    }

    #[test]
    fn parked_pack_reachability_preserves_embedded_vector_and_file_references(
    ) -> GraphStoreResult<()> {
        let vector = Hash::of(b"parked pack vector payload");
        let manifest = Hash::of(b"parked pack file manifest");
        let node = NodeRecord::new(
            "file:parked-pack",
            ["File"],
            json!({
                "content_manifest": FileManifestRef::new(manifest).to_value()?,
                "embedding": {
                    "content_id": vector.to_prefixed(),
                    "dimension": 3
                }
            }),
        );
        let pack = compile_graph_pack(
            &GraphSnapshot {
                version: 1,
                nodes: vec![node],
                ..GraphSnapshot::default()
            },
            GraphCompileOptions {
                author: Some("object-store-test".to_string()),
                message: Some("pack embedded reachability".to_string()),
                timestamp_unix_ms: Some(11),
                ..GraphCompileOptions::default()
            },
        )?;
        let bytes = encode_pack_envelope(&pack)?;
        let address = Hash::of(&bytes);

        let ObjectStoreReachability::PackEnvelope {
            file_manifests,
            terminals,
            ..
        } = decode_object_store_reachability(address, &bytes)?
        else {
            panic!("expected pack envelope reachability");
        };
        assert_eq!(file_manifests, vec![manifest]);
        assert_eq!(terminals, vec![vector]);
        Ok(())
    }
}
