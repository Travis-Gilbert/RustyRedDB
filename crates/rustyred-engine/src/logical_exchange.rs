//! Bounded, versioned logical exchange. Physical journal/header bytes are never
//! transport. A complete typed reachable set retains graph history and payloads.
//! The host supplies current authority separately from untrusted archive claims.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::chunk_store::{ChunkStore, ChunkStoreCatalog, CommitMeta, RootCatalog};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph_store::RedCoreGraphStore;
use crate::hash::Hash;
use crate::object_store::UnifiedChunkStore;
use crate::reachability::{
    walk_committed_reachable_with_observer, CoreReachabilityCodec, ReachabilityKind,
    ReachabilityLimits,
};
use crate::versioned_graph::{
    decode_canonical_graph_payload, recompute_commit_hash, CompiledGraphPack, GraphPayloadRecord,
    GRAPH_CHUNK_FORMAT_VERSION, GRAPH_PACK_COMPILER_VERSION, VERSIONED_GRAPH_PROTOCOL_VERSION,
};

const MAGIC: &[u8; 8] = b"THGLEX01";
const ROOTS: &[&str] = &["graph", "redcore", "documents", "objects"];

/// Trusted per-operation capacity, never read from archive metadata. Finite
/// budgets cover framing, canonical bytes, chunk count, and typed traversal.
#[derive(Clone, Copy, Debug)]
pub struct ExchangeLimits {
    pub max_archive_bytes: usize,
    pub max_header_bytes: usize,
    pub max_chunk_bytes: usize,
    pub max_chunks: usize,
    pub max_depth: usize,
}
impl ExchangeLimits {
    fn validate(self) -> GraphStoreResult<()> {
        if self.max_archive_bytes == 0
            || self.max_header_bytes == 0
            || self.max_chunk_bytes == 0
            || self.max_chunks == 0
            || self.max_depth == 0
            || self.max_header_bytes > self.max_archive_bytes
            || self.max_chunk_bytes > self.max_archive_bytes
        {
            return Err(refusal(
                "exchange_limits_invalid",
                "finite positive exchange capacities are required",
            ));
        }
        Ok(())
    }
}

/// Advisory source claims must exactly match independently admitted host
/// identity. This is not an ownership grant and is never sufficient to write.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExchangeIdentity {
    pub tenant: String,
    pub principal: String,
}

/// Domain owners verify every canonically decoded historical payload, then
/// resolve cross-record references before an archive can be published/applied.
/// Stateless component validators can use a closure; product owners implement
/// the finalization method for their actual tenant and shared-schema rules.
pub trait ExchangeRecordAdmission {
    fn admit_record(&mut self, record: &GraphPayloadRecord) -> GraphStoreResult<()>;
    fn finish_admission(&mut self) -> GraphStoreResult<()> {
        Ok(())
    }
    /// Called only after typed reachability and exact retained head binding.
    /// Product owners admit historical cross-record contracts here, before
    /// export frames or a privately constructed ValidatedExchange can escape.
    fn admit_history(&mut self, _context: &ExchangeHistoryContext) -> GraphStoreResult<()> {
        Ok(())
    }
}

/// Canonical retained read context, constructed solely by exchange validation.
/// It grants no destination writes or authority. Consumers supply trusted
/// finite read/working-set budgets to the retained revision APIs.
pub struct ExchangeHistoryContext {
    history: crate::graph_history::PinnedGraphHistory,
    identity: ExchangeIdentity,
    limits: ExchangeLimits,
}
impl ExchangeHistoryContext {
    pub fn history(&self) -> &crate::graph_history::PinnedGraphHistory {
        &self.history
    }
    pub fn identity(&self) -> &ExchangeIdentity {
        &self.identity
    }
    pub fn limits(&self) -> ExchangeLimits {
        self.limits
    }
}
impl<F: FnMut(&GraphPayloadRecord) -> GraphStoreResult<()>> ExchangeRecordAdmission for F {
    fn admit_record(&mut self, record: &GraphPayloadRecord) -> GraphStoreResult<()> {
        self(record)
    }
}
impl ExchangeIdentity {
    fn validate(&self) -> GraphStoreResult<()> {
        if self.tenant.trim().is_empty()
            || self.principal.trim().is_empty()
            || self.tenant.len() > 1024
            || self.principal.len() > 1024
            || self.principal.eq_ignore_ascii_case("anonymous")
        {
            return Err(refusal(
                "exchange_identity_refused",
                "an admitted tenant and durable principal are required",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    identity: ExchangeIdentity,
    catalog: Hash,
    pack: CompiledGraphPack,
}

/// Validated staged bytes. Its constructor is private: apply cannot receive an
/// unchecked envelope. Staging uses an indexed read-only temporary archive, never the destination.
pub struct ValidatedExchange {
    header: Header,
    store: Arc<ArchiveStore>,
    addresses: BTreeSet<Hash>,
    bytes: usize,
    limits: ExchangeLimits,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExchangeReceipt {
    pub catalog: Hash,
    pub logical_commit: Hash,
    pub chunks: usize,
    pub canonical_bytes: usize,
    pub source_preserved: bool,
}

fn refusal(code: &'static str, message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new(code, message)
}
fn meta() -> CommitMeta {
    CommitMeta {
        parents: vec![],
        author: "logical-exchange".into(),
        message: "validated logical archive publication".into(),
        timestamp_unix_ms: 0,
        graph_version: 0,
    }
}
fn catalog(store: &dyn ChunkStore, address: Hash, maximum: usize) -> GraphStoreResult<RootCatalog> {
    let bytes = store
        .get_bounded(address, maximum)?
        .ok_or_else(|| refusal("exchange_catalog_missing", "committed catalog is absent"))?;
    if Hash::of(&bytes) != address {
        return Err(refusal(
            "exchange_catalog_hash",
            "catalog address differs from its bytes",
        ));
    }
    let catalog = RootCatalog::from_bytes(&bytes)?;
    if catalog
        .roots()
        .keys()
        .any(|name| !ROOTS.contains(&name.as_str()))
    {
        return Err(refusal(
            "exchange_catalog_scope_refused",
            "archive contains a catalog domain without an exchange contract",
        ));
    }
    Ok(catalog)
}

/// Bounded random-access archive storage. Implementations wrap a read-only
/// native file or an already closed/uploaded temporary OPFS file; callers never
/// hand this interface their canonical destination journal.
pub trait ExchangeReader: Send + Sync {
    fn len(&self) -> GraphStoreResult<u64>;
    fn read_exact(&self, offset: u64, length: usize) -> GraphStoreResult<Vec<u8>>;
}
/// A private temporary export receives frames. Expose the file only after a
/// successful receipt and durable close, because a failed source fence can
/// leave an incomplete temporary stream. The source always remains available.
pub trait ExchangeWriter {
    fn append(&mut self, bytes: &[u8]) -> GraphStoreResult<()>;
}

/// Read-only native archive adapter. This opens an exchange stream, never a
/// native database or a writable source journal.
#[cfg(not(target_arch = "wasm32"))]
pub struct FileExchangeReader {
    file: Mutex<std::fs::File>,
}
#[cfg(not(target_arch = "wasm32"))]
impl FileExchangeReader {
    pub fn open(path: impl AsRef<std::path::Path>) -> GraphStoreResult<Self> {
        Ok(Self {
            file: Mutex::new(
                std::fs::File::open(path).map_err(|error| {
                    GraphStoreError::io("open read-only logical archive", error)
                })?,
            ),
        })
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl ExchangeReader for FileExchangeReader {
    fn len(&self) -> GraphStoreResult<u64> {
        self.file
            .lock()
            .map_err(|_| {
                refusal(
                    "exchange_archive_lock_poisoned",
                    "archive reader lock poisoned",
                )
            })?
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(|error| GraphStoreError::io("inspect logical archive", error))
    }
    fn read_exact(&self, offset: u64, length: usize) -> GraphStoreResult<Vec<u8>> {
        use std::io::{Read, Seek};
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| {
            refusal(
                "exchange_read_capacity_refused",
                "cannot reserve bounded archive frame",
            )
        })?;
        bytes.resize(length, 0);
        let mut file = self.file.lock().map_err(|_| {
            refusal(
                "exchange_archive_lock_poisoned",
                "archive reader lock poisoned",
            )
        })?;
        file.seek(std::io::SeekFrom::Start(offset))
            .map_err(|error| GraphStoreError::io("seek logical archive", error))?;
        file.read_exact(&mut bytes)
            .map_err(|error| GraphStoreError::io("read bounded logical archive frame", error))?;
        Ok(bytes)
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl ExchangeWriter for std::fs::File {
    fn append(&mut self, bytes: &[u8]) -> GraphStoreResult<()> {
        use std::io::Write;
        self.write_all(bytes)
            .map_err(|error| GraphStoreError::io("write logical archive frame", error))
    }
}

/// Snapshot immutable canonical chunks under the exact committed catalog, with
/// one chunk resident at a time. No all-history graph snapshot is hydrated.
pub fn export_logical_exchange(
    source: &RedCoreGraphStore,
    identity: &ExchangeIdentity,
    limits: ExchangeLimits,
    writer: &mut dyn ExchangeWriter,
    record_admission: &mut impl ExchangeRecordAdmission,
) -> GraphStoreResult<ExchangeReceipt> {
    export_logical_exchange_from_store(
        source.unified_chunk_store(),
        identity,
        limits,
        writer,
        record_admission,
    )
}

/// Export an already accepted native or browser chunk source without opening a
/// graph facade or hydrating its full records. The descriptor is derived from
/// bounded canonical retained state, never from caller supplied pack metadata.
pub fn export_logical_exchange_from_store(
    source_store: Arc<dyn UnifiedChunkStore>,
    identity: &ExchangeIdentity,
    limits: ExchangeLimits,
    writer: &mut dyn ExchangeWriter,
    record_admission: &mut impl ExchangeRecordAdmission,
) -> GraphStoreResult<ExchangeReceipt> {
    limits.validate()?;
    identity.validate()?;
    let root = source_store.read_root()?.ok_or_else(|| {
        refusal(
            "exchange_source_uncommitted",
            "source has no accepted catalog",
        )
    })?;
    let capture = Arc::new(Capture::new(Arc::clone(&source_store), root, limits));
    let source_catalog = catalog(capture.as_ref(), root, limits.max_header_bytes)?;
    let bounded_source: Arc<dyn UnifiedChunkStore> = capture.clone();
    let pack = crate::graph_store::compiled_graph_head_from_store_for_exchange(Arc::clone(
        &bounded_source,
    ))?;
    if source_catalog.get("graph") != Some(pack.tree.root)
        || source_catalog.get("redcore").is_none()
    {
        return Err(refusal(
            "exchange_pack_catalog_mismatch",
            "compiled graph head must name the source catalog graph and retained state",
        ));
    }
    validate_pack(&pack)?;
    let mut seen_records = BTreeSet::new();
    let report = walk_committed_reachable_with_observer(
        capture.as_ref(),
        &CoreReachabilityCodec,
        ReachabilityLimits::new(limits.max_chunks, limits.max_depth),
        &mut |kind, address, bytes| {
            if kind == ReachabilityKind::ObjectPayload && seen_records.insert(address) {
                record_admission.admit_record(&decode_canonical_graph_payload(address, bytes)?)?;
            }
            Ok(())
        },
    )?;
    record_admission.finish_admission()?;
    let history =
        crate::graph_store::validate_compiled_graph_head_for_exchange(bounded_source, &pack)?;
    record_admission.admit_history(&ExchangeHistoryContext {
        history,
        identity: identity.clone(),
        limits,
    })?;
    let header = Header {
        format: "theorem-logical-exchange/1".into(),
        identity: identity.clone(),
        catalog: root,
        pack,
    };
    let encoded_header = serde_json::to_vec(&header)
        .map_err(|error| refusal("exchange_header_encode", error.to_string()))?;
    if encoded_header.len() > limits.max_header_bytes {
        return Err(refusal(
            "exchange_header_limit",
            "compiled graph header exceeds its admitted size",
        ));
    }
    let captured = capture
        .state
        .lock()
        .map_err(|_| refusal("exchange_capture_poisoned", "capture lock poisoned"))?;
    let chunks = &captured.lengths;
    let canonical_bytes = captured.total;
    let total = 16usize
        .checked_add(encoded_header.len())
        .and_then(|size| {
            chunks.values().try_fold(size, |size, bytes| {
                size.checked_add(36)?.checked_add(*bytes)
            })
        })
        .ok_or_else(|| refusal("exchange_size_overflow", "archive size overflow"))?;
    if total > limits.max_archive_bytes {
        return Err(refusal(
            "exchange_archive_limit",
            "archive framing and canonical bytes exceed capacity",
        ));
    }
    if report.reachable != chunks.keys().copied().collect() {
        return Err(refusal(
            "exchange_capture_incomplete",
            "captured set differs from complete typed reachability",
        ));
    }
    if source_store.read_root()? != Some(root) {
        return Err(refusal(
            "exchange_source_changed",
            "source publication advanced during validation",
        ));
    }
    writer.append(MAGIC)?;
    writer.append(
        &u32::try_from(encoded_header.len())
            .map_err(|_| refusal("exchange_header_limit", "header exceeds u32 framing"))?
            .to_le_bytes(),
    )?;
    writer.append(&encoded_header)?;
    writer.append(
        &u32::try_from(chunks.len())
            .map_err(|_| {
                refusal(
                    "exchange_chunk_count_limit",
                    "chunk count exceeds u32 framing",
                )
            })?
            .to_le_bytes(),
    )?;
    for (address, length) in chunks.iter() {
        let bytes = source_store
            .get_bounded(*address, limits.max_chunk_bytes)?
            .ok_or_else(|| refusal("exchange_source_missing", "source chunk disappeared"))?;
        if bytes.len() != *length || Hash::of(&bytes) != *address {
            return Err(refusal(
                "exchange_source_changed",
                "source immutable chunk bytes changed",
            ));
        }
        writer.append(address.as_bytes())?;
        writer.append(
            &u32::try_from(bytes.len())
                .map_err(|_| refusal("exchange_chunk_limit", "chunk exceeds u32 framing"))?
                .to_le_bytes(),
        )?;
        writer.append(&bytes)?;
    }
    if source_store.read_root()? != Some(root) {
        return Err(refusal(
            "exchange_source_changed",
            "source publication advanced during capture",
        ));
    }
    Ok(ExchangeReceipt {
        catalog: root,
        logical_commit: header.pack.commit.commit_hash,
        chunks: chunks.len(),
        canonical_bytes,
        source_preserved: true,
    })
}

/// Index framed offsets, then validate every typed reachable byte before the
/// destination is touched. Resident memory is the bounded index plus one chunk;
/// body bytes remain in the caller's read-only temporary archive.
pub fn validate_logical_exchange(
    reader: Arc<dyn ExchangeReader>,
    admitted: &ExchangeIdentity,
    limits: ExchangeLimits,
    record_admission: &mut impl ExchangeRecordAdmission,
) -> GraphStoreResult<ValidatedExchange> {
    limits.validate()?;
    admitted.validate()?;
    let length = reader.len()?;
    if length > limits.max_archive_bytes as u64 {
        return Err(refusal(
            "exchange_archive_limit",
            "archive exceeds admitted capacity",
        ));
    }
    let mut decoder = Decoder {
        reader: Arc::clone(&reader),
        cursor: 0,
        length,
    };
    if decoder.take(8)?.as_slice() != MAGIC {
        return Err(refusal(
            "exchange_version_unsupported",
            "unsupported logical exchange version; retain the original",
        ));
    }
    let header_len = decoder.u32()?;
    if header_len > limits.max_header_bytes {
        return Err(refusal(
            "exchange_header_limit",
            "archive header exceeds admitted capacity",
        ));
    }
    let header: Header = serde_json::from_slice(&decoder.take(header_len)?)
        .map_err(|error| refusal("exchange_header_invalid", error.to_string()))?;
    if header.format != "theorem-logical-exchange/1" {
        return Err(refusal(
            "exchange_version_unsupported",
            "unsupported logical exchange schema",
        ));
    }
    if &header.identity != admitted {
        return Err(refusal(
            "exchange_identity_refused",
            "archive identity differs from the host's admitted tenant and principal",
        ));
    }
    validate_pack(&header.pack)?;
    let count = decoder.u32()?;
    if count == 0 || count > limits.max_chunks {
        return Err(refusal(
            "exchange_chunk_count_limit",
            "archive chunk count exceeds admitted capacity",
        ));
    }
    let mut index = BTreeMap::new();
    let mut previous = None;
    let mut total = 0usize;
    for _ in 0..count {
        let address = Hash::from_bytes(
            decoder
                .take(32)?
                .as_slice()
                .try_into()
                .expect("32-byte frame"),
        );
        if previous.is_some_and(|prior| prior >= address) {
            return Err(refusal(
                "exchange_noncanonical_chunks",
                "chunk addresses must be unique and sorted",
            ));
        }
        previous = Some(address);
        let chunk_length = decoder.u32()?;
        if chunk_length > limits.max_chunk_bytes {
            return Err(refusal(
                "exchange_chunk_limit",
                "chunk exceeds admitted capacity",
            ));
        }
        let offset = decoder.cursor;
        decoder.skip(chunk_length)?;
        total = total
            .checked_add(chunk_length)
            .ok_or_else(|| refusal("exchange_size_overflow", "canonical byte count overflow"))?;
        index.insert(address, (offset, chunk_length));
    }
    if decoder.cursor != length {
        return Err(refusal(
            "exchange_trailing_bytes",
            "archive has trailing bytes",
        ));
    }
    let addresses: BTreeSet<Hash> = index.keys().copied().collect();
    let store = Arc::new(ArchiveStore {
        reader,
        index,
        root: header.catalog,
        limits,
    });
    let roots = catalog(store.as_ref(), header.catalog, limits.max_header_bytes)?;
    if roots.get("graph") != Some(header.pack.tree.root) || roots.get("redcore").is_none() {
        return Err(refusal(
            "exchange_pack_catalog_mismatch",
            "pack does not name the archive's graph root and retained state",
        ));
    }
    let mut seen = BTreeSet::new();
    let reachable = walk_committed_reachable_with_observer(
        store.as_ref(),
        &CoreReachabilityCodec,
        ReachabilityLimits::new(limits.max_chunks, limits.max_depth),
        &mut |kind, address, bytes| {
            if kind == ReachabilityKind::ObjectPayload && seen.insert(address) {
                record_admission.admit_record(&decode_canonical_graph_payload(address, bytes)?)?;
            }
            Ok(())
        },
    )?;
    record_admission.finish_admission()?;
    if reachable.reachable != addresses {
        return Err(refusal(
            "exchange_reachable_set_mismatch",
            "archive must contain exactly the complete reachable logical set",
        ));
    }
    let unified: Arc<dyn UnifiedChunkStore> = store.clone();
    let history =
        crate::graph_store::validate_compiled_graph_head_for_exchange(unified, &header.pack)?;
    record_admission.admit_history(&ExchangeHistoryContext {
        history,
        identity: admitted.clone(),
        limits,
    })?;
    if store.reader.len()? != length {
        return Err(refusal(
            "exchange_archive_changed",
            "archive size changed during validation",
        ));
    }
    Ok(ValidatedExchange {
        header,
        store,
        addresses,
        bytes: total,
        limits,
    })
}

impl ValidatedExchange {
    pub fn receipt(&self) -> ExchangeReceipt {
        ExchangeReceipt {
            catalog: self.header.catalog,
            logical_commit: self.header.pack.commit.commit_hash,
            chunks: self.addresses.len(),
            canonical_bytes: self.bytes,
            source_preserved: true,
        }
    }
    /// Stream immutable validated chunks for physical-capacity admission. The
    /// callback may refuse; this method never writes to a canonical store.
    pub fn for_each_chunk(
        &self,
        visitor: &mut impl FnMut(Hash, &[u8]) -> GraphStoreResult<()>,
    ) -> GraphStoreResult<()> {
        for address in &self.addresses {
            let bytes = self.store.get(*address)?.ok_or_else(|| {
                refusal(
                    "exchange_stage_missing",
                    "validated staged chunk disappeared",
                )
            })?;
            visitor(*address, &bytes)?;
        }
        Ok(())
    }
    pub fn publication_metadata(&self) -> CommitMeta {
        meta()
    }
    pub fn require_empty_destination(
        &self,
        destination: &dyn ChunkStore,
        expected: Option<Hash>,
    ) -> GraphStoreResult<()> {
        require_empty(destination, expected, self.limits.max_header_bytes)
    }
    /// Called on an independently reopened native/browser backend by its host.
    /// Physical reopen is the host's responsibility and is separately observed
    /// by the portability oracle; this verifies exact canonical data identity.
    pub fn verify_destination(
        &self,
        destination: Arc<dyn UnifiedChunkStore>,
    ) -> GraphStoreResult<ExchangeReceipt> {
        if destination.read_root()? != Some(self.header.catalog) {
            return Err(refusal(
                "exchange_reopen_catalog_mismatch",
                "reopened destination differs from imported catalog",
            ));
        }
        let report = crate::reachability::walk_committed_reachable_with_limits(
            destination.as_chunk_store(),
            &CoreReachabilityCodec,
            ReachabilityLimits::new(self.limits.max_chunks, self.limits.max_depth),
        )?;
        if report.reachable != self.addresses {
            return Err(refusal(
                "exchange_reopen_reachable_mismatch",
                "reopened destination has different reachable data",
            ));
        }
        crate::graph_store::validate_compiled_graph_head_for_exchange(
            destination,
            &self.header.pack,
        )?;
        Ok(self.receipt())
    }
    /// Caller owns the physical exclusion and supplies the exact current empty
    /// catalog after checking its durable identity. This method rechecks that
    /// catalog immediately before staging and publication; it never clears or
    /// overwrites nonempty data. The caller independently reopens before success.
    pub fn apply_to_empty(
        &self,
        destination: &dyn ChunkStoreCatalog,
        expected_empty_catalog: Option<Hash>,
    ) -> GraphStoreResult<ExchangeReceipt> {
        self.apply_to_empty_with_admission(destination, expected_empty_catalog, &mut || Ok(()))
    }

    /// Product hosts recheck their existing admitted identity/owner generation
    /// before staging and immediately before the atomic catalog publication.
    /// This callback carries host authority; archive claims never provide it.
    pub fn apply_to_empty_with_admission(
        &self,
        destination: &dyn ChunkStoreCatalog,
        expected_empty_catalog: Option<Hash>,
        revalidate_admission: &mut impl FnMut() -> GraphStoreResult<()>,
    ) -> GraphStoreResult<ExchangeReceipt> {
        revalidate_admission()?;
        if !destination.supports_catalog_compare_and_swap() {
            return Err(refusal(
                "exchange_atomic_publication_unsupported",
                "destination cannot fence a changed catalog; no canonical chunks staged",
            ));
        }
        require_empty(
            destination,
            expected_empty_catalog,
            self.limits.max_header_bytes,
        )?;
        for address in &self.addresses {
            let bytes = self.store.get(*address)?.ok_or_else(|| {
                refusal(
                    "exchange_stage_missing",
                    "validated staged chunk disappeared",
                )
            })?;
            destination.write(&[(*address, &bytes)])?;
            let stored = destination
                .get_bounded(*address, self.limits.max_chunk_bytes)?
                .ok_or_else(|| {
                    refusal(
                        "exchange_destination_chunk_missing",
                        "staged destination chunk is absent",
                    )
                })?;
            if stored.as_ref() != bytes.as_ref() || Hash::of(&stored) != *address {
                return Err(refusal(
                    "exchange_destination_chunk_mismatch",
                    "destination chunk differs from validated canonical bytes",
                ));
            }
        }
        require_empty(
            destination,
            expected_empty_catalog,
            self.limits.max_header_bytes,
        )?;
        let updates = catalog(
            self.store.as_ref(),
            self.header.catalog,
            self.limits.max_header_bytes,
        )?
        .roots()
        .iter()
        .map(|(name, address)| (name.clone(), Some(*address)))
        .collect();
        revalidate_admission()?;
        let published =
            destination.commit_catalog_updates_if_root(expected_empty_catalog, &updates, meta())?;
        if published.catalog_root != self.header.catalog {
            return Err(refusal(
                "exchange_publication_mismatch",
                "destination published a different logical catalog",
            ));
        }
        Ok(self.receipt())
    }
}

/// An uncommitted store or the exact graph-only empty Prolly leaf catalog
/// initialized by DiskObjectStore is admissible. Retained RedCore state, any
/// other named root, and the native DesktopRuntime genesis are nonempty.
fn require_empty(
    destination: &dyn ChunkStore,
    expected: Option<Hash>,
    maximum_catalog_bytes: usize,
) -> GraphStoreResult<()> {
    if destination.read_root()? != expected {
        return Err(refusal(
            "exchange_destination_changed",
            "destination catalog differs from the host's current empty grant",
        ));
    }
    if let Some(root) = expected {
        let empty_leaf = crate::versioned_graph::canonical_leaf_preimage(&[])?;
        let empty_leaf_address = Hash::of(&empty_leaf);
        let expected_facade = RootCatalog::graph(empty_leaf_address);
        if catalog(destination, root, maximum_catalog_bytes)? != expected_facade
            || destination
                .get_bounded(empty_leaf_address, empty_leaf.len())?
                .as_deref()
                != Some(empty_leaf.as_slice())
        {
            return Err(refusal("exchange_destination_not_empty", "import requires an uncommitted store or exact empty facade catalog; preserve existing data"));
        }
    }
    Ok(())
}
fn validate_pack(pack: &CompiledGraphPack) -> GraphStoreResult<()> {
    if pack
        .capabilities
        .iter()
        .any(|capability| !matches!(capability.kind.as_str(), "graph_manifest" | "validator"))
    {
        return Err(refusal(
            "exchange_capability_kind_unsupported",
            "compiled pack contains a metadata capability without a recognized schema",
        ));
    }
    if pack.chunk_format_version != GRAPH_CHUNK_FORMAT_VERSION
        || pack.protocol_version != VERSIONED_GRAPH_PROTOCOL_VERSION
        || pack.compiler_version != GRAPH_PACK_COMPILER_VERSION
        || pack.manifest.chunk_format_version != GRAPH_CHUNK_FORMAT_VERSION
        || pack.manifest.protocol_version != VERSIONED_GRAPH_PROTOCOL_VERSION
        || pack.manifest.compiler_version != GRAPH_PACK_COMPILER_VERSION
    {
        return Err(refusal(
            "exchange_graph_version_unsupported",
            "unsupported compiled graph protocol, compiler or chunk version",
        ));
    }
    if !pack.tree_chunks.is_empty()
        || !pack.objects.is_empty()
        || !pack.archive_objects.is_empty()
        || !pack.historical_node_ids.is_empty()
        || pack.tree_transport_complete
    {
        return Err(refusal("exchange_pack_descriptor_invalid", "exchange head uses the compact descriptor; complete chunks have one canonical frame set"));
    }
    if recompute_commit_hash(&pack.commit)? != pack.commit.commit_hash
        || pack.manifest.commit_hash != pack.commit.commit_hash
        || pack.tree.root != pack.manifest.tree_hash
        || pack.commit.tree_hash != pack.tree.root
        || pack.commit.graph_version != pack.manifest.graph_version
        || pack.commit.compiler_version != pack.compiler_version
        || pack.commit.branch != pack.manifest.branch
        || pack.commit.objects_total != pack.manifest.objects_total
    {
        return Err(refusal(
            "exchange_pack_identity_mismatch",
            "compiled graph commit and manifest identities differ",
        ));
    }
    // Reuse the real compiler's exact manifest/validator constructors, including
    // stable content hashes, names, MIME types, and body-to-head bindings.
    let expected = crate::versioned_graph::retarget_graph_pack_version(
        pack.clone(),
        pack.commit.graph_version,
    )?;
    if pack.capabilities != expected.capabilities {
        return Err(refusal(
            "exchange_capability_invalid",
            "compiled metadata capabilities differ from canonical compiler output",
        ));
    }
    Ok(())
}

struct Decoder {
    reader: Arc<dyn ExchangeReader>,
    cursor: u64,
    length: u64,
}
impl Decoder {
    fn skip(&mut self, length: usize) -> GraphStoreResult<()> {
        let end = self
            .cursor
            .checked_add(length as u64)
            .ok_or_else(|| refusal("exchange_frame_overflow", "archive frame length overflow"))?;
        if end > self.length {
            return Err(refusal(
                "exchange_frame_truncated",
                "archive frame is truncated",
            ));
        }
        self.cursor = end;
        Ok(())
    }
    fn take(&mut self, length: usize) -> GraphStoreResult<Vec<u8>> {
        let offset = self.cursor;
        self.skip(length)?;
        let bytes = self.reader.read_exact(offset, length)?;
        if bytes.len() != length {
            return Err(refusal(
                "exchange_reader_short_read",
                "archive reader did not return the exact bounded frame",
            ));
        }
        Ok(bytes)
    }
    fn u32(&mut self) -> GraphStoreResult<usize> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .as_slice()
                .try_into()
                .expect("four-byte frame"),
        ) as usize)
    }
}

struct ArchiveStore {
    reader: Arc<dyn ExchangeReader>,
    index: BTreeMap<Hash, (u64, usize)>,
    root: Hash,
    limits: ExchangeLimits,
}
impl ChunkStore for ArchiveStore {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.get_bounded(address, self.limits.max_chunk_bytes)
    }
    fn get_bounded(&self, address: Hash, maximum: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let Some((offset, length)) = self.index.get(&address) else {
            return Ok(None);
        };
        if *length > maximum.min(self.limits.max_chunk_bytes) {
            return Err(refusal(
                "exchange_chunk_limit",
                "archive chunk exceeds bounded read",
            ));
        }
        let bytes = self.reader.read_exact(*offset, *length)?;
        if bytes.len() != *length || Hash::of(&bytes) != address {
            return Err(refusal(
                "exchange_chunk_hash",
                "archive chunk differs from its indexed length or address",
            ));
        }
        Ok(Some(Arc::from(bytes)))
    }
    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        Ok(self.get(address)?.is_some())
    }
    fn contains_address(&self, address: Hash) -> GraphStoreResult<bool> {
        Ok(self.index.contains_key(&address))
    }
    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        Ok(addresses
            .iter()
            .filter(|address| !self.index.contains_key(address))
            .copied()
            .collect())
    }
    fn write(&self, _: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        Err(refusal(
            "exchange_archive_read_only",
            "staged archive never writes canonical storage",
        ))
    }
    fn commit(&self, _: Hash, _: CommitMeta) -> GraphStoreResult<Hash> {
        Err(refusal(
            "exchange_archive_read_only",
            "staged archive never publishes",
        ))
    }
    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        Ok(Some(self.root))
    }
}
impl ChunkStoreCatalog for ArchiveStore {
    fn commit_catalog_updates(
        &self,
        _: &BTreeMap<String, Option<Hash>>,
        _: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        Err(refusal(
            "exchange_archive_read_only",
            "staged archive never publishes",
        ))
    }
}

struct CapturedLengths {
    lengths: BTreeMap<Hash, usize>,
    total: usize,
}
struct Capture {
    source: Arc<dyn UnifiedChunkStore>,
    root: Hash,
    limits: ExchangeLimits,
    state: Mutex<CapturedLengths>,
}
impl Capture {
    fn new(source: Arc<dyn UnifiedChunkStore>, root: Hash, limits: ExchangeLimits) -> Self {
        Self {
            source,
            root,
            limits,
            state: Mutex::new(CapturedLengths {
                lengths: BTreeMap::new(),
                total: 0,
            }),
        }
    }
}
impl ChunkStore for Capture {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.get_bounded(address, self.limits.max_chunk_bytes)
    }
    fn get_bounded(&self, address: Hash, maximum: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let mut captured = self
            .state
            .lock()
            .map_err(|_| refusal("exchange_capture_poisoned", "capture lock poisoned"))?;
        let known = captured.lengths.contains_key(&address);
        if !known && captured.lengths.len() >= self.limits.max_chunks {
            return Err(refusal(
                "exchange_chunk_count_limit",
                "source exceeds admitted chunk capacity",
            ));
        }
        let used = captured.total;
        let available = if known {
            self.limits.max_chunk_bytes
        } else {
            self.limits.max_archive_bytes.saturating_sub(used)
        };
        let bytes = self.source.get_bounded(
            address,
            maximum.min(self.limits.max_chunk_bytes).min(available),
        )?;
        if let Some(bytes) = &bytes {
            if Hash::of(bytes) != address {
                return Err(refusal(
                    "exchange_source_chunk_hash",
                    "source chunk differs from its address",
                ));
            }
            if !known {
                captured.total = captured
                    .total
                    .checked_add(bytes.len())
                    .ok_or_else(|| refusal("exchange_size_overflow", "capture size overflow"))?;
            }
            if captured
                .lengths
                .insert(address, bytes.len())
                .is_some_and(|previous| previous != bytes.len())
            {
                return Err(refusal(
                    "exchange_source_changed",
                    "source immutable chunk length changed",
                ));
            }
        }
        Ok(bytes)
    }
    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        Ok(self.get(address)?.is_some())
    }
    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        addresses
            .iter()
            .filter_map(|address| match self.has(*address) {
                Ok(true) => None,
                Ok(false) => Some(Ok(*address)),
                Err(error) => Some(Err(error)),
            })
            .collect()
    }
    fn write(&self, _: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        Err(refusal(
            "exchange_source_read_only",
            "source exchange capture never writes",
        ))
    }
    fn commit(&self, _: Hash, _: CommitMeta) -> GraphStoreResult<Hash> {
        Err(refusal(
            "exchange_source_read_only",
            "source exchange capture never publishes",
        ))
    }
    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        Ok(Some(self.root))
    }
}

#[cfg(test)]
#[path = "logical_exchange_tests.rs"]
mod tests;

impl ChunkStoreCatalog for Capture {
    fn commit_catalog_updates(
        &self,
        _: &BTreeMap<String, Option<Hash>>,
        _: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        Err(refusal(
            "exchange_source_read_only",
            "bounded export source never publishes",
        ))
    }
}
