//! Content-defined file bodies over the unified chunk store.
//!
//! A graph `File` or `CodeFile` node carries only a [`FileManifestRef`]. The
//! manifest and its ordered terminal chunks are canonical SHA-256 objects in
//! the one physical store. Staging owns exact keyed in-flight pins until the
//! referencing graph root is durable, so concurrent reclamation cannot expose
//! a manifest whose body chunks disappeared between preparation and commit.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::chunk_store::{ChunkStore, ChunkWriteReport};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::object_store::UnifiedChunkStore;

/// Exact graph-property position holding a file-manifest reference.
pub const FILE_CONTENT_MANIFEST_PROPERTY: &str = "content_manifest";
/// Bounded, newest-first manifest history retained by graph File nodes.
pub const FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY: &str = "previous_content_manifests";
/// Exact count of older manifests intentionally omitted from the bounded list.
pub const FILE_HISTORY_ELIDED_COUNT_PROPERTY: &str = "history_elided_count";
pub const MAX_PREVIOUS_FILE_MANIFESTS: usize = 64;
/// Smallest non-final content-defined chunk.
pub const FILE_CHUNK_MIN_BYTES: usize = 4 * 1024;
/// Mean content-defined chunk target.
pub const FILE_CHUNK_TARGET_BYTES: usize = 16 * 1024;
/// Forced boundary for adversarial or low-entropy input.
pub const FILE_CHUNK_MAX_BYTES: usize = 64 * 1024;
/// Property names that would make a graph File/CodeFile own inline body bytes.
/// Payload-producing crates share this set so retiring one legacy spelling
/// cannot leave another hidden byte owner in the graph namespace.
pub const INLINE_FILE_BODY_PROPERTIES: [&str; 5] = ["text", "body", "content", "bytes", "inline"];
/// Hard decoder/allocation bound for one manifest.
pub const MAX_FILE_MANIFEST_CHUNKS: usize = 1_000_000;

const DOMAIN_FILE_MANIFEST: u8 = 13;
const FILE_MANIFEST_FORMAT_VERSION: u8 = 1;
const FILE_MANIFEST_HEADER_BYTES: usize = 1 + 1 + 8 + 32 + 4;
const FILE_MANIFEST_CHUNK_BYTES: usize = 32 + 4;
const FILE_CHUNK_TARGET_MASK: u64 = (FILE_CHUNK_TARGET_BYTES as u64) - 1;

/// The only accepted JSON shape at `File.content_manifest` and
/// `CodeFile.content_manifest`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileManifestRef {
    pub content_id: String,
}

impl FileManifestRef {
    pub fn new(address: Hash) -> Self {
        Self {
            content_id: address.to_prefixed(),
        }
    }

    /// Strictly decode `{ "content_id": "sha256:<lower hex>" }`.
    pub fn from_value(value: &Value) -> GraphStoreResult<Self> {
        let object = value.as_object().ok_or_else(|| {
            GraphStoreError::new(
                "file_manifest_reference_invalid",
                "file manifest reference must be a JSON object",
            )
        })?;
        if object.len() != 1 || !object.contains_key("content_id") {
            return Err(GraphStoreError::new(
                "file_manifest_reference_invalid",
                "file manifest reference must contain only content_id",
            ));
        }
        let reference: Self = serde_json::from_value(value.clone()).map_err(|error| {
            GraphStoreError::new("file_manifest_reference_invalid", error.to_string())
        })?;
        reference.address()?;
        Ok(reference)
    }

    pub fn to_value(&self) -> GraphStoreResult<Value> {
        self.address()?;
        let mut object = Map::new();
        object.insert(
            "content_id".to_string(),
            Value::String(self.content_id.clone()),
        );
        Ok(Value::Object(object))
    }

    pub fn address(&self) -> GraphStoreResult<Hash> {
        Hash::parse_prefixed(&self.content_id).map_err(|error| {
            GraphStoreError::new(
                "file_manifest_reference_invalid",
                format!("content_id is not canonical: {}", error.message),
            )
        })
    }
}

/// Replace a node's current file manifest while retaining bounded, typed
/// history. Every dropped legacy/current reference increments the explicit
/// elision counter, so rewrite history is never lost silently.
pub fn install_file_manifest_reference(
    properties: &mut Map<String, Value>,
    next: &FileManifestRef,
    additional_elided: u64,
) -> GraphStoreResult<()> {
    let mut previous = Vec::<FileManifestRef>::new();
    if let Some(current) = properties.get(FILE_CONTENT_MANIFEST_PROPERTY) {
        let current = FileManifestRef::from_value(current)?;
        if current != *next {
            previous.push(current);
        }
    }
    if let Some(history) = properties.get(FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY) {
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
            previous.push(FileManifestRef::from_value(value)?);
        }
    }

    let mut seen = std::collections::BTreeSet::new();
    seen.insert(next.content_id.clone());
    previous.retain(|reference| seen.insert(reference.content_id.clone()));
    let dropped = previous.len().saturating_sub(MAX_PREVIOUS_FILE_MANIFESTS);
    previous.truncate(MAX_PREVIOUS_FILE_MANIFESTS);
    let prior_elided = match properties.get(FILE_HISTORY_ELIDED_COUNT_PROPERTY) {
        None => 0,
        Some(value) => value.as_u64().ok_or_else(|| {
            GraphStoreError::new(
                "file_manifest_history_invalid",
                "history_elided_count must be a non-negative integer",
            )
        })?,
    };
    let elided = prior_elided
        .saturating_add(additional_elided)
        .saturating_add(u64::try_from(dropped).unwrap_or(u64::MAX));

    properties.insert(FILE_CONTENT_MANIFEST_PROPERTY.to_string(), next.to_value()?);
    if previous.is_empty() {
        properties.remove(FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY);
    } else {
        properties.insert(
            FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY.to_string(),
            Value::Array(
                previous
                    .into_iter()
                    .map(|reference| reference.to_value())
                    .collect::<GraphStoreResult<Vec<_>>>()?,
            ),
        );
    }
    if elided == 0 {
        properties.remove(FILE_HISTORY_ELIDED_COUNT_PROPERTY);
    } else {
        properties.insert(
            FILE_HISTORY_ELIDED_COUNT_PROPERTY.to_string(),
            Value::Number(elided.into()),
        );
    }
    Ok(())
}

/// One ordered terminal in a canonical file manifest.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileChunkRef {
    pub address: Hash,
    pub length: u32,
}

/// Canonical description of one complete file body.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileManifest {
    pub total_length: u64,
    pub whole_file_sha256: Hash,
    pub chunks: Vec<FileChunkRef>,
}

impl FileManifest {
    pub fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        self.validate()?;
        let chunk_count = u32::try_from(self.chunks.len()).map_err(|_| {
            GraphStoreError::new(
                "file_manifest_chunk_count_overflow",
                "file manifest chunk count exceeds the canonical u32 limit",
            )
        })?;
        let capacity = FILE_MANIFEST_HEADER_BYTES
            .checked_add(
                self.chunks
                    .len()
                    .checked_mul(FILE_MANIFEST_CHUNK_BYTES)
                    .ok_or_else(manifest_size_overflow)?,
            )
            .ok_or_else(manifest_size_overflow)?;
        let mut output = Vec::with_capacity(capacity);
        output.push(DOMAIN_FILE_MANIFEST);
        output.push(FILE_MANIFEST_FORMAT_VERSION);
        output.extend_from_slice(&self.total_length.to_le_bytes());
        output.extend_from_slice(self.whole_file_sha256.as_bytes());
        output.extend_from_slice(&chunk_count.to_le_bytes());
        for chunk in &self.chunks {
            output.extend_from_slice(chunk.address.as_bytes());
            output.extend_from_slice(&chunk.length.to_le_bytes());
        }
        Ok(output)
    }

    pub fn address(&self) -> GraphStoreResult<Hash> {
        self.to_bytes().map(|bytes| Hash::of(&bytes))
    }

    fn validate(&self) -> GraphStoreResult<()> {
        if self.chunks.len() > MAX_FILE_MANIFEST_CHUNKS {
            return Err(GraphStoreError::new(
                "file_manifest_chunk_limit_exceeded",
                format!(
                    "file manifest has {} chunks, exceeding the {} chunk limit",
                    self.chunks.len(),
                    MAX_FILE_MANIFEST_CHUNKS
                ),
            ));
        }
        if self.total_length == 0 {
            if !self.chunks.is_empty() {
                return Err(GraphStoreError::new(
                    "file_manifest_empty_body_has_chunks",
                    "an empty file manifest must not reference terminal chunks",
                ));
            }
            if self.whole_file_sha256 != Hash::of(&[]) {
                return Err(GraphStoreError::new(
                    "file_manifest_empty_hash_mismatch",
                    "an empty file manifest must carry the SHA-256 of empty bytes",
                ));
            }
            return Ok(());
        }
        if self.chunks.is_empty() {
            return Err(GraphStoreError::new(
                "file_manifest_nonempty_body_missing_chunks",
                "a non-empty file manifest must reference at least one chunk",
            ));
        }

        let mut total = 0_u64;
        for (index, chunk) in self.chunks.iter().enumerate() {
            let length = usize::try_from(chunk.length).map_err(|_| {
                GraphStoreError::new(
                    "file_manifest_chunk_length_overflow",
                    "file chunk length does not fit usize",
                )
            })?;
            if length == 0 || length > FILE_CHUNK_MAX_BYTES {
                return Err(GraphStoreError::new(
                    "file_manifest_chunk_length_invalid",
                    format!(
                        "file manifest chunk {index} has length {length}; expected 1..={FILE_CHUNK_MAX_BYTES}"
                    ),
                ));
            }
            if index + 1 != self.chunks.len() && length < FILE_CHUNK_MIN_BYTES {
                return Err(GraphStoreError::new(
                    "file_manifest_nonfinal_chunk_too_short",
                    format!(
                        "non-final file manifest chunk {index} has length {length}, below {FILE_CHUNK_MIN_BYTES}"
                    ),
                ));
            }
            total = total.checked_add(u64::from(chunk.length)).ok_or_else(|| {
                GraphStoreError::new(
                    "file_manifest_total_length_overflow",
                    "sum of file chunk lengths exceeds u64",
                )
            })?;
        }
        if total != self.total_length {
            return Err(GraphStoreError::new(
                "file_manifest_total_length_mismatch",
                format!(
                    "file manifest chunk lengths sum to {total}, declared total is {}",
                    self.total_length
                ),
            ));
        }
        Ok(())
    }
}

/// Exact physical delta admitted while staging one file body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilePayloadWriteReport {
    pub manifest_address: Hash,
    /// Number of terminal references in the manifest, including duplicates.
    pub total_chunks: usize,
    /// Novel physical chunks admitted, including the manifest itself.
    pub chunks_written: usize,
    /// Novel physical bytes admitted, including the manifest itself.
    pub bytes_written: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingReceiptState {
    Anonymous,
    Bound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingReceipt {
    address: Hash,
    publication_key: String,
    state: PendingReceiptState,
}

/// Owns all pins installed while one file body awaits graph publication.
#[must_use = "staged file payloads must remain pinned through graph publication"]
pub struct FilePayloadStagingGuard<'a> {
    store: &'a dyn UnifiedChunkStore,
    reference: FileManifestRef,
    manifest: FileManifest,
    report: FilePayloadWriteReport,
    pending: Vec<PendingReceipt>,
}

impl fmt::Debug for FilePayloadStagingGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FilePayloadStagingGuard")
            .field("reference", &self.reference)
            .field("manifest", &self.manifest)
            .field("report", &self.report)
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl FilePayloadStagingGuard<'_> {
    pub fn reference(&self) -> &FileManifestRef {
        &self.reference
    }

    pub fn manifest(&self) -> &FileManifest {
        &self.manifest
    }

    pub fn report(&self) -> FilePayloadWriteReport {
        self.report
    }

    /// Release every receipt still owned by this exact publication. Failures
    /// are aggregated after all receipts are attempted and retained for retry.
    pub fn cancel(&mut self) -> GraphStoreResult<()> {
        let mut failed = Vec::new();
        let mut errors = Vec::new();
        for receipt in std::mem::take(&mut self.pending).into_iter().rev() {
            let release = match receipt.state {
                PendingReceiptState::Anonymous => self.store.unpin_in_flight(receipt.address),
                PendingReceiptState::Bound => self
                    .store
                    .release_bound_pending_in_flight(receipt.address, &receipt.publication_key)
                    .map(|_| ()),
            };
            if let Err(error) = release {
                failed.push(receipt);
                errors.push(error);
            }
        }
        failed.reverse();
        self.pending = failed;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphStoreError::new(
                "file_payload_pin_cleanup_failed",
                errors
                    .into_iter()
                    .map(|error| format!("{}: {}", error.code, error.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            ))
        }
    }

    /// Release staging pins only after the referencing graph root is durable.
    ///
    /// Cleanup is deliberately best-effort at this point: publication already
    /// committed, so a pin-release failure must not make the caller observe a
    /// failed transaction and retry durable side effects. `cancel` retains any
    /// failed receipts and `Drop` immediately retries them once.
    pub fn finish_after_publication(mut self) {
        let _ = self.cancel();
    }
}

impl Drop for FilePayloadStagingGuard<'_> {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}

/// Chunk, pin, write, and verify one complete body without publishing a graph
/// reference. `publication_key` identifies the exact logical node/property
/// write; it is folded into every terminal receipt so concurrent duplicates do
/// not consume one another's pins.
pub fn stage_file_payload<'a>(
    store: &'a dyn UnifiedChunkStore,
    publication_key: &str,
    body: &[u8],
) -> GraphStoreResult<FilePayloadStagingGuard<'a>> {
    validate_publication_key(publication_key)?;
    let boundaries = chunk_boundaries(body)?;
    let chunks = boundaries
        .iter()
        .map(|boundary| FileChunkRef {
            address: Hash::of(&body[boundary.start..boundary.end]),
            length: u32::try_from(boundary.end - boundary.start)
                .expect("file chunks never exceed the 64 KiB canonical maximum"),
        })
        .collect::<Vec<_>>();
    let manifest = FileManifest {
        total_length: u64::try_from(body.len()).map_err(|_| {
            GraphStoreError::new(
                "file_payload_length_overflow",
                "file body length exceeds the canonical u64 range",
            )
        })?,
        whole_file_sha256: Hash::of(body),
        chunks,
    };
    let manifest_bytes = manifest.to_bytes()?;
    let manifest_address = Hash::of(&manifest_bytes);
    let reference = FileManifestRef::new(manifest_address);
    let mut guard = FilePayloadStagingGuard {
        store,
        reference,
        manifest,
        report: FilePayloadWriteReport {
            manifest_address,
            total_chunks: boundaries.len(),
            chunks_written: 0,
            bytes_written: 0,
        },
        pending: Vec::with_capacity(boundaries.len().saturating_add(1)),
    };

    let stage_result = (|| {
        guard.pin_and_bind(
            manifest_address,
            file_receipt_key(publication_key, manifest_address, "manifest", 0),
        )?;
        let terminal_addresses = guard
            .manifest
            .chunks
            .iter()
            .map(|chunk| chunk.address)
            .collect::<Vec<_>>();
        for (index, address) in terminal_addresses.into_iter().enumerate() {
            guard.pin_and_bind(
                address,
                file_receipt_key(publication_key, manifest_address, "chunk", index),
            )?;
        }

        let mut unique = BTreeMap::<Hash, &[u8]>::new();
        for (boundary, chunk) in boundaries.iter().zip(&guard.manifest.chunks) {
            let bytes = &body[boundary.start..boundary.end];
            if let Some(existing) = unique.insert(chunk.address, bytes) {
                if existing != bytes {
                    return Err(GraphStoreError::new(
                        "file_payload_sha256_collision",
                        format!("distinct terminal bytes share address {}", chunk.address),
                    ));
                }
            }
        }
        if let Some(existing) = unique.insert(manifest_address, manifest_bytes.as_slice()) {
            if existing != manifest_bytes.as_slice() {
                return Err(GraphStoreError::new(
                    "file_payload_sha256_collision",
                    format!("terminal bytes collide with manifest {manifest_address}"),
                ));
            }
        }
        let batch = unique
            .iter()
            .map(|(address, bytes)| (*address, *bytes))
            .collect::<Vec<_>>();
        let ChunkWriteReport {
            chunks_written,
            bytes_written,
        } = store.write_accounted(&batch)?;
        guard.report.chunks_written = chunks_written;
        guard.report.bytes_written = bytes_written;

        for (address, expected) in &batch {
            let readback = store.get(*address)?.ok_or_else(|| {
                GraphStoreError::new(
                    "file_payload_readback_missing",
                    format!("chunk store cannot read back staged file chunk {address}"),
                )
            })?;
            if readback.as_ref() != *expected {
                return Err(GraphStoreError::new(
                    "file_payload_readback_mismatch",
                    format!("chunk store returned different bytes for {address}"),
                ));
            }
        }
        let reconstructed = read_file_payload(store.as_chunk_store(), &guard.reference)?;
        if reconstructed != body {
            return Err(GraphStoreError::new(
                "file_payload_verified_roundtrip_mismatch",
                "staged file body did not survive its verified manifest round-trip",
            ));
        }
        Ok(())
    })();

    if let Err(write_error) = stage_result {
        return match guard.cancel() {
            Ok(()) => Err(write_error),
            Err(cleanup_error) => Err(GraphStoreError::new(
                "file_payload_stage_and_cleanup_failed",
                format!(
                    "staging failed ({}: {}); cleanup failed ({}: {})",
                    write_error.code,
                    write_error.message,
                    cleanup_error.code,
                    cleanup_error.message
                ),
            )),
        };
    }
    Ok(guard)
}

impl FilePayloadStagingGuard<'_> {
    fn pin_and_bind(&mut self, address: Hash, publication_key: String) -> GraphStoreResult<()> {
        self.store.pin_in_flight(address)?;
        self.pending.push(PendingReceipt {
            address,
            publication_key,
            state: PendingReceiptState::Anonymous,
        });
        let receipt = self
            .pending
            .last_mut()
            .expect("receipt was appended before binding");
        self.store
            .bind_pending_in_flight(address, &receipt.publication_key)?;
        receipt.state = PendingReceiptState::Bound;
        Ok(())
    }
}

/// Strictly decode one canonical manifest at its expected content address.
pub fn decode_file_manifest(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<FileManifest> {
    if bytes.len() < FILE_MANIFEST_HEADER_BYTES {
        return Err(GraphStoreError::new(
            "file_manifest_truncated",
            "file manifest is shorter than its canonical header",
        ));
    }
    if bytes[0] != DOMAIN_FILE_MANIFEST {
        return Err(GraphStoreError::new(
            "file_manifest_domain_invalid",
            format!(
                "file manifest domain {} does not match {DOMAIN_FILE_MANIFEST}",
                bytes[0]
            ),
        ));
    }
    if bytes[1] != FILE_MANIFEST_FORMAT_VERSION {
        return Err(GraphStoreError::new(
            "file_manifest_version_unsupported",
            format!("file manifest version {} is not supported", bytes[1]),
        ));
    }
    let actual_address = Hash::of(bytes);
    if actual_address != expected_address {
        return Err(GraphStoreError::new(
            "file_manifest_address_mismatch",
            format!("file manifest hashes to {actual_address}, expected {expected_address}"),
        ));
    }

    let total_length = u64::from_le_bytes(
        bytes[2..10]
            .try_into()
            .expect("manifest header length was checked"),
    );
    let whole_file_sha256 = Hash::from_bytes(
        bytes[10..42]
            .try_into()
            .expect("manifest header length was checked"),
    );
    let chunk_count = usize::try_from(u32::from_le_bytes(
        bytes[42..46]
            .try_into()
            .expect("manifest header length was checked"),
    ))
    .expect("u32 always fits supported usize targets");
    if chunk_count > MAX_FILE_MANIFEST_CHUNKS {
        return Err(GraphStoreError::new(
            "file_manifest_chunk_limit_exceeded",
            format!(
                "file manifest declares {chunk_count} chunks, exceeding the {MAX_FILE_MANIFEST_CHUNKS} chunk limit"
            ),
        ));
    }
    let expected_length = FILE_MANIFEST_HEADER_BYTES
        .checked_add(
            chunk_count
                .checked_mul(FILE_MANIFEST_CHUNK_BYTES)
                .ok_or_else(manifest_size_overflow)?,
        )
        .ok_or_else(manifest_size_overflow)?;
    if bytes.len() != expected_length {
        return Err(GraphStoreError::new(
            "file_manifest_length_noncanonical",
            format!(
                "file manifest has {} bytes, canonical count requires {expected_length}",
                bytes.len()
            ),
        ));
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    let mut offset = FILE_MANIFEST_HEADER_BYTES;
    for _ in 0..chunk_count {
        let address = Hash::from_bytes(
            bytes[offset..offset + 32]
                .try_into()
                .expect("canonical manifest length was checked"),
        );
        offset += 32;
        let length = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("canonical manifest length was checked"),
        );
        offset += 4;
        chunks.push(FileChunkRef { address, length });
    }
    let manifest = FileManifest {
        total_length,
        whole_file_sha256,
        chunks,
    };
    manifest.validate()?;
    if manifest.to_bytes()?.as_slice() != bytes {
        return Err(GraphStoreError::new(
            "file_manifest_not_canonical",
            "decoded file manifest does not reproduce its exact canonical bytes",
        ));
    }
    Ok(manifest)
}

/// Resolve every terminal in order and verify its address, declared length,
/// total length, and whole-file digest before returning bytes.
pub fn read_file_payload(
    store: &dyn ChunkStore,
    reference: &FileManifestRef,
) -> GraphStoreResult<Vec<u8>> {
    let manifest_address = reference.address()?;
    let manifest_bytes = store.get(manifest_address)?.ok_or_else(|| {
        GraphStoreError::new(
            "file_manifest_missing",
            format!("file manifest {manifest_address} is missing"),
        )
    })?;
    if Hash::of(&manifest_bytes) != manifest_address {
        return Err(GraphStoreError::new(
            "file_manifest_address_mismatch",
            format!("stored bytes do not match file manifest {manifest_address}"),
        ));
    }
    let manifest = decode_file_manifest(manifest_address, &manifest_bytes)?;
    let capacity = usize::try_from(manifest.total_length).map_err(|_| {
        GraphStoreError::new(
            "file_payload_length_overflow",
            "file manifest total length cannot fit in process memory",
        )
    })?;
    let mut output = Vec::new();
    for (index, chunk) in manifest.chunks.iter().enumerate() {
        let bytes = store.get(chunk.address)?.ok_or_else(|| {
            GraphStoreError::new(
                "file_payload_chunk_missing",
                format!(
                    "file manifest {manifest_address} terminal {index} at {} is missing",
                    chunk.address
                ),
            )
        })?;
        let actual_address = Hash::of(&bytes);
        if actual_address != chunk.address {
            return Err(GraphStoreError::new(
                "file_payload_chunk_address_mismatch",
                format!(
                    "file terminal {index} hashes to {actual_address}, expected {}",
                    chunk.address
                ),
            ));
        }
        if bytes.len() != usize::try_from(chunk.length).expect("u32 fits usize") {
            return Err(GraphStoreError::new(
                "file_payload_chunk_length_mismatch",
                format!(
                    "file terminal {index} has {} bytes, manifest declares {}",
                    bytes.len(),
                    chunk.length
                ),
            ));
        }
        let next_length = output.len().checked_add(bytes.len()).ok_or_else(|| {
            GraphStoreError::new(
                "file_payload_length_overflow",
                "reconstructed file length exceeds usize",
            )
        })?;
        if next_length > capacity {
            return Err(GraphStoreError::new(
                "file_payload_total_length_mismatch",
                "file terminals exceed the manifest total length",
            ));
        }
        output.try_reserve_exact(bytes.len()).map_err(|error| {
            GraphStoreError::new(
                "file_payload_allocation_refused",
                format!("cannot reserve {} more file bytes: {error}", bytes.len()),
            )
        })?;
        output.extend_from_slice(&bytes);
    }
    if output.len() != capacity {
        return Err(GraphStoreError::new(
            "file_payload_total_length_mismatch",
            format!(
                "file terminals reconstruct {} bytes, manifest declares {}",
                output.len(),
                manifest.total_length
            ),
        ));
    }
    let actual_whole = Hash::of(&output);
    if actual_whole != manifest.whole_file_sha256 {
        return Err(GraphStoreError::new(
            "file_payload_whole_hash_mismatch",
            format!(
                "reconstructed file hashes to {actual_whole}, expected {}",
                manifest.whole_file_sha256
            ),
        ));
    }
    let canonical_boundaries = chunk_boundaries(&output)?;
    let canonical_chunks = canonical_boundaries
        .into_iter()
        .map(|boundary| FileChunkRef {
            address: Hash::of(&output[boundary.start..boundary.end]),
            length: u32::try_from(boundary.end - boundary.start)
                .expect("file chunks never exceed the 64 KiB canonical maximum"),
        })
        .collect::<Vec<_>>();
    if canonical_chunks != manifest.chunks {
        return Err(GraphStoreError::new(
            "file_manifest_chunking_noncanonical",
            "file terminals reconstruct valid bytes but do not match canonical Gear boundaries",
        ));
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkBoundary {
    start: usize,
    end: usize,
}

fn chunk_boundaries(body: &[u8]) -> GraphStoreResult<Vec<ChunkBoundary>> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < body.len() {
        if chunks.len() >= MAX_FILE_MANIFEST_CHUNKS {
            return Err(GraphStoreError::new(
                "file_manifest_chunk_limit_exceeded",
                format!("file body requires more than {MAX_FILE_MANIFEST_CHUNKS} canonical chunks"),
            ));
        }
        let hard_end = start.saturating_add(FILE_CHUNK_MAX_BYTES).min(body.len());
        let search_start = start.saturating_add(FILE_CHUNK_MIN_BYTES).min(hard_end);
        let mut fingerprint = 0_u64;
        let mut cursor = start;
        let mut end = hard_end;
        while cursor < hard_end {
            fingerprint = fingerprint
                .wrapping_shl(1)
                .wrapping_add(GEAR_TABLE[usize::from(body[cursor])]);
            cursor += 1;
            if cursor >= search_start && fingerprint & FILE_CHUNK_TARGET_MASK == 0 {
                end = cursor;
                break;
            }
        }
        chunks.push(ChunkBoundary { start, end });
        start = end;
    }
    Ok(chunks)
}

const fn make_gear_table() -> [u64; 256] {
    let mut table = [0_u64; 256];
    let mut index = 0_usize;
    while index < table.len() {
        // Fixed SplitMix64 expansion gives every address byte one stable Gear
        // word without runtime hashing or platform-dependent randomness.
        let mut value = (index as u64).wrapping_add(0x4649_4c45_4745_4152);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        table[index] = value ^ (value >> 31);
        index += 1;
    }
    table
}

const GEAR_TABLE: [u64; 256] = make_gear_table();

fn file_receipt_key(
    publication_key: &str,
    manifest_address: Hash,
    role: &str,
    ordinal: usize,
) -> String {
    let mut preimage = Vec::with_capacity(
        publication_key
            .len()
            .saturating_add(role.len())
            .saturating_add(80),
    );
    for field in [
        publication_key.as_bytes(),
        manifest_address.as_bytes(),
        role.as_bytes(),
    ] {
        preimage.extend_from_slice(&(field.len() as u64).to_le_bytes());
        preimage.extend_from_slice(field);
    }
    preimage.extend_from_slice(&(ordinal as u64).to_le_bytes());
    format!("file-payload:{}", Hash::of(&preimage).to_prefixed())
}

fn validate_publication_key(publication_key: &str) -> GraphStoreResult<()> {
    if publication_key.trim().is_empty() {
        return Err(GraphStoreError::new(
            "file_payload_publication_key_invalid",
            "file payload publication key must be non-empty",
        ));
    }
    Ok(())
}

fn manifest_size_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "file_manifest_size_overflow",
        "file manifest encoded size exceeds usize",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use serde_json::json;

    use super::{
        chunk_boundaries, decode_file_manifest, install_file_manifest_reference, read_file_payload,
        stage_file_payload, FileChunkRef, FileManifest, FileManifestRef, FILE_CHUNK_MAX_BYTES,
        FILE_CHUNK_MIN_BYTES, FILE_HISTORY_ELIDED_COUNT_PROPERTY,
        FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY, MAX_FILE_MANIFEST_CHUNKS,
        MAX_PREVIOUS_FILE_MANIFESTS,
    };
    use crate::chunk_store::{
        CatalogCommitResult, ChunkStore, ChunkStoreCatalog, ChunkStoreCompaction, ChunkWriteReport,
        CommitMeta, MemChunkStore,
    };
    use crate::graph::error::{GraphStoreError, GraphStoreResult};
    use crate::hash::Hash;
    use crate::object_store::UnifiedChunkStore;

    fn fixture_bytes(length: usize) -> Vec<u8> {
        let mut state = 0x7c2d_91a4_e6b8_350f_u64;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    fn as_unified(store: &Arc<MemChunkStore>) -> Arc<dyn UnifiedChunkStore> {
        store.clone()
    }

    #[test]
    fn manifest_history_is_bounded_typed_and_counts_every_elided_version() {
        let mut properties = serde_json::Map::new();
        for version in 0_u8..67 {
            install_file_manifest_reference(
                &mut properties,
                &FileManifestRef::new(Hash::of(&[version])),
                0,
            )
            .unwrap();
        }
        let history = properties[FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY]
            .as_array()
            .unwrap();
        assert_eq!(history.len(), MAX_PREVIOUS_FILE_MANIFESTS);
        assert!(history
            .iter()
            .all(|value| FileManifestRef::from_value(value).is_ok()));
        assert_eq!(
            properties[FILE_HISTORY_ELIDED_COUNT_PROPERTY],
            json!(2),
            "67 distinct versions retain current + 64 previous and count two elisions"
        );
    }

    #[test]
    fn content_defined_boundaries_are_deterministic_bounded_and_golden() {
        let body = fixture_bytes(256 * 1024);
        let first = chunk_boundaries(&body).unwrap();
        let second = chunk_boundaries(&body).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first.iter().map(|chunk| chunk.end).collect::<Vec<_>>(),
            vec![
                27473, 83236, 94382, 104401, 116518, 182054, 201849, 210321, 245093, 258035,
                262144,
            ]
        );
        for (index, chunk) in first.iter().enumerate() {
            let length = chunk.end - chunk.start;
            assert!(length <= FILE_CHUNK_MAX_BYTES);
            if index + 1 != first.len() {
                assert!(length >= FILE_CHUNK_MIN_BYTES);
            }
        }

        let sample = fixture_bytes(8 * 1024 * 1024);
        let sample_chunks = chunk_boundaries(&sample).unwrap();
        let mean = sample.len() / sample_chunks.len();
        assert!((12 * 1024..=24 * 1024).contains(&mean), "mean={mean}");
    }

    #[test]
    fn roundtrip_dedup_and_local_edit_delta_are_measured() -> GraphStoreResult<()> {
        let body = fixture_bytes(2 * 1024 * 1024);
        let backend = Arc::new(MemChunkStore::new());
        let store = as_unified(&backend);
        let empty = stage_file_payload(store.as_ref(), "file:empty/content_manifest", &[])?;
        assert!(empty.manifest().chunks.is_empty());
        assert_eq!(
            read_file_payload(store.as_chunk_store(), empty.reference())?,
            Vec::<u8>::new()
        );
        empty.finish_after_publication();

        let original = stage_file_payload(store.as_ref(), "file:original/content_manifest", &body)?;
        let original_reference = original.reference().clone();
        let original_manifest = original.manifest().clone();
        let original_report = original.report();
        assert_eq!(
            read_file_payload(store.as_chunk_store(), &original_reference)?,
            body
        );
        original.finish_after_publication();

        let duplicate =
            stage_file_payload(store.as_ref(), "file:duplicate/content_manifest", &body)?;
        assert_eq!(duplicate.reference(), &original_reference);
        assert_eq!(duplicate.report().chunks_written, 0);
        assert_eq!(duplicate.report().bytes_written, 0);
        duplicate.finish_after_publication();

        let mut edited = body.clone();
        let middle = edited.len() / 2;
        edited[middle - 257..middle + 257].fill(0xa5);
        let delta = stage_file_payload(store.as_ref(), "file:edited/content_manifest", &edited)?;
        let delta_report = delta.report();
        let reused = delta
            .manifest()
            .chunks
            .iter()
            .filter(|chunk| original_manifest.chunks.contains(chunk))
            .count();
        assert!(reused * 4 > original_manifest.chunks.len() * 3);

        let fresh_backend = Arc::new(MemChunkStore::new());
        let fresh_store = as_unified(&fresh_backend);
        let full = stage_file_payload(
            fresh_store.as_ref(),
            "file:edited-fresh/content_manifest",
            &edited,
        )?;
        assert!(delta_report.chunks_written * 4 < full.report().chunks_written);
        assert!(delta_report.bytes_written * 4 < full.report().bytes_written);
        assert_eq!(
            read_file_payload(store.as_chunk_store(), delta.reference())?,
            edited
        );
        delta.finish_after_publication();
        full.finish_after_publication();

        assert!(original_report.chunks_written > 1);
        Ok(())
    }

    #[test]
    fn insertion_and_deletion_resynchronize_to_existing_addresses() -> GraphStoreResult<()> {
        let original = fixture_bytes(1024 * 1024);
        let backend = Arc::new(MemChunkStore::new());
        let store = as_unified(&backend);
        let base = stage_file_payload(store.as_ref(), "file:base/content_manifest", &original)?;
        let base_chunks = base.manifest().chunks.clone();

        let insert_at = original.len() / 3;
        let mut inserted = original[..insert_at].to_vec();
        inserted.extend_from_slice(&fixture_bytes(937));
        inserted.extend_from_slice(&original[insert_at..]);
        let insertion =
            stage_file_payload(store.as_ref(), "file:inserted/content_manifest", &inserted)?;

        let delete_start = original.len() * 2 / 3;
        let mut deleted = original[..delete_start].to_vec();
        deleted.extend_from_slice(&original[delete_start + 1231..]);
        let deletion =
            stage_file_payload(store.as_ref(), "file:deleted/content_manifest", &deleted)?;

        for manifest in [insertion.manifest(), deletion.manifest()] {
            let reused = manifest
                .chunks
                .iter()
                .filter(|chunk| base_chunks.contains(chunk))
                .count();
            assert!(reused * 2 > base_chunks.len());
        }
        assert_eq!(
            read_file_payload(store.as_chunk_store(), insertion.reference())?,
            inserted
        );
        assert_eq!(
            read_file_payload(store.as_chunk_store(), deletion.reference())?,
            deleted
        );
        base.finish_after_publication();
        insertion.finish_after_publication();
        deletion.finish_after_publication();
        Ok(())
    }

    #[test]
    fn malformed_oversized_missing_and_corrupt_payloads_fail_closed() -> GraphStoreResult<()> {
        let body = fixture_bytes(96 * 1024);
        let backend = Arc::new(MemChunkStore::new());
        let store = as_unified(&backend);
        let staged = stage_file_payload(store.as_ref(), "file:strict/content_manifest", &body)?;
        let reference = staged.reference().clone();
        let manifest_address = reference.address()?;
        let manifest_bytes = store
            .get(manifest_address)?
            .expect("staged manifest is readable");
        assert_eq!(
            decode_file_manifest(Hash::of(b"wrong manifest address"), &manifest_bytes)
                .unwrap_err()
                .code,
            "file_manifest_address_mismatch"
        );

        let mut trailing = manifest_bytes.to_vec();
        trailing.push(0);
        let trailing_address = Hash::of(&trailing);
        assert_eq!(
            decode_file_manifest(trailing_address, &trailing)
                .unwrap_err()
                .code,
            "file_manifest_length_noncanonical"
        );

        let mut oversized = vec![0_u8; 46];
        oversized[0] = 13;
        oversized[1] = 1;
        oversized[42..46]
            .copy_from_slice(&(u32::try_from(MAX_FILE_MANIFEST_CHUNKS).unwrap() + 1).to_le_bytes());
        let oversized_address = Hash::of(&oversized);
        assert_eq!(
            decode_file_manifest(oversized_address, &oversized)
                .unwrap_err()
                .code,
            "file_manifest_chunk_limit_exceeded"
        );

        let missing = MemChunkStore::new();
        missing.write(&[(manifest_address, manifest_bytes.as_ref())])?;
        assert_eq!(
            read_file_payload(&missing, &reference).unwrap_err().code,
            "file_payload_chunk_missing"
        );

        let corrupt_target = staged.manifest().chunks[0].address;
        let corrupt = CorruptReadStore {
            inner: backend.as_ref(),
            corrupt_target,
        };
        assert_eq!(
            read_file_payload(&corrupt, &reference).unwrap_err().code,
            "file_payload_chunk_address_mismatch"
        );

        let mut wrong_length = staged.manifest().clone();
        wrong_length.chunks[0].length -= 1;
        wrong_length.total_length -= 1;
        let wrong_length_bytes = wrong_length.to_bytes()?;
        let wrong_length_address = Hash::of(&wrong_length_bytes);
        backend.write(&[(wrong_length_address, &wrong_length_bytes)])?;
        assert_eq!(
            read_file_payload(
                backend.as_ref(),
                &FileManifestRef::new(wrong_length_address),
            )
            .unwrap_err()
            .code,
            "file_payload_chunk_length_mismatch"
        );

        let wrong_hash = FileManifest {
            whole_file_sha256: Hash::of(b"wrong whole-file digest"),
            ..staged.manifest().clone()
        };
        let wrong_hash_bytes = wrong_hash.to_bytes()?;
        let wrong_hash_address = Hash::of(&wrong_hash_bytes);
        backend.write(&[(wrong_hash_address, &wrong_hash_bytes)])?;
        assert_eq!(
            read_file_payload(backend.as_ref(), &FileManifestRef::new(wrong_hash_address),)
                .unwrap_err()
                .code,
            "file_payload_whole_hash_mismatch"
        );

        let mut fixed_chunks = Vec::new();
        for bytes in body.chunks(32 * 1024) {
            let address = Hash::of(bytes);
            backend.write(&[(address, bytes)])?;
            fixed_chunks.push(FileChunkRef {
                address,
                length: u32::try_from(bytes.len()).unwrap(),
            });
        }
        let noncanonical = FileManifest {
            total_length: u64::try_from(body.len()).unwrap(),
            whole_file_sha256: Hash::of(&body),
            chunks: fixed_chunks,
        };
        let noncanonical_bytes = noncanonical.to_bytes()?;
        let noncanonical_address = Hash::of(&noncanonical_bytes);
        backend.write(&[(noncanonical_address, &noncanonical_bytes)])?;
        assert_eq!(
            read_file_payload(
                backend.as_ref(),
                &FileManifestRef::new(noncanonical_address),
            )
            .unwrap_err()
            .code,
            "file_manifest_chunking_noncanonical"
        );
        staged.finish_after_publication();
        Ok(())
    }

    #[test]
    fn graph_reference_shape_is_strict_and_never_contains_body_text() {
        let address = Hash::of(b"manifest");
        let reference = FileManifestRef::new(address);
        assert_eq!(
            reference.to_value().unwrap(),
            json!({"content_id": address})
        );
        assert_eq!(
            FileManifestRef::from_value(&reference.to_value().unwrap()).unwrap(),
            reference
        );
        for malformed in [
            json!(address.to_prefixed()),
            json!({"content_id": address, "body": "forbidden"}),
            json!({"content_id": address.to_prefixed().to_uppercase()}),
        ] {
            assert_eq!(
                FileManifestRef::from_value(&malformed).unwrap_err().code,
                "file_manifest_reference_invalid"
            );
        }
    }

    #[test]
    fn staging_pins_survive_gc_and_exact_cleanup_attempts_every_receipt() -> GraphStoreResult<()> {
        let body = fixture_bytes(200 * 1024);
        let backend = Arc::new(MemChunkStore::new());
        let store = as_unified(&backend);
        let mut staged = stage_file_payload(store.as_ref(), "file:gc/content_manifest", &body)?;
        let mut duplicate =
            stage_file_payload(store.as_ref(), "file:gc-duplicate/content_manifest", &body)?;
        let addresses = std::iter::once(staged.report().manifest_address)
            .chain(staged.manifest().chunks.iter().map(|chunk| chunk.address))
            .collect::<Vec<_>>();

        backend.compact_reachable(None, &[])?;
        for address in &addresses {
            assert!(backend.has(*address)?);
            assert!(backend.bound_pending_pin_count(*address)? > 0);
        }
        staged.cancel()?;
        backend.compact_reachable(None, &[])?;
        for address in &addresses {
            assert!(backend.has(*address)?);
            assert!(backend.bound_pending_pin_count(*address)? > 0);
        }
        duplicate.cancel()?;
        backend.compact_reachable(None, &[])?;
        for address in &addresses {
            assert!(!backend.has(*address)?);
        }

        let failures = Arc::new(ReleaseFailureStore::new(2));
        let unified: Arc<dyn UnifiedChunkStore> = failures.clone();
        let mut staged = stage_file_payload(
            unified.as_ref(),
            "file:partial-cleanup/content_manifest",
            &body,
        )?;
        let receipt_count = staged.manifest().chunks.len() + 1;
        let error = staged.cancel().unwrap_err();
        assert_eq!(error.code, "file_payload_pin_cleanup_failed");
        assert_eq!(
            failures.release_attempts.load(Ordering::SeqCst),
            receipt_count
        );
        staged.cancel()?;
        assert_eq!(
            failures.release_attempts.load(Ordering::SeqCst),
            receipt_count + 2
        );
        Ok(())
    }

    #[test]
    fn postpublication_cleanup_failure_cannot_change_commit_outcome() -> GraphStoreResult<()> {
        let failures = Arc::new(ReleaseFailureStore::new(1));
        let unified: Arc<dyn UnifiedChunkStore> = failures.clone();
        let staged = stage_file_payload(
            unified.as_ref(),
            "file:postcommit-cleanup/content_manifest",
            &fixture_bytes(200 * 1024),
        )?;
        let addresses = std::iter::once(staged.report().manifest_address)
            .chain(staged.manifest().chunks.iter().map(|chunk| chunk.address))
            .collect::<Vec<_>>();

        // Publication is already durable at this point. The first release is
        // faulted, finish remains infallible, and the guard's Drop retry drains
        // the retained receipt before returning to the caller.
        staged.finish_after_publication();

        for address in addresses {
            assert_eq!(failures.inner.bound_pending_pin_count(address)?, 0);
        }
        Ok(())
    }

    struct CorruptReadStore<'a> {
        inner: &'a MemChunkStore,
        corrupt_target: Hash,
    }

    impl ChunkStore for CorruptReadStore<'_> {
        fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
            if address == self.corrupt_target {
                return Ok(Some(Arc::from(b"corrupt terminal".as_slice())));
            }
            self.inner.get(address)
        }

        fn has(&self, address: Hash) -> GraphStoreResult<bool> {
            self.get(address).map(|value| value.is_some())
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

    struct ReleaseFailureStore {
        inner: MemChunkStore,
        failures_remaining: AtomicUsize,
        release_attempts: AtomicUsize,
    }

    impl ReleaseFailureStore {
        fn new(failures: usize) -> Self {
            Self {
                inner: MemChunkStore::new(),
                failures_remaining: AtomicUsize::new(failures),
                release_attempts: AtomicUsize::new(0),
            }
        }
    }

    impl ChunkStore for ReleaseFailureStore {
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
            self.release_attempts.fetch_add(1, Ordering::SeqCst);
            let failure = self.failures_remaining.fetch_update(
                Ordering::SeqCst,
                Ordering::SeqCst,
                |remaining| remaining.checked_sub(1),
            );
            if failure.is_ok() {
                return Err(GraphStoreError::new(
                    "test_release_failure",
                    format!("injected release failure for {publication_key}"),
                ));
            }
            self.inner
                .release_bound_pending_in_flight(address, publication_key)
        }

        fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
            self.inner.unpin_in_flight(address)
        }
    }

    impl ChunkStoreCatalog for ReleaseFailureStore {
        fn commit_catalog_updates(
            &self,
            updates: &BTreeMap<String, Option<Hash>>,
            meta: CommitMeta,
        ) -> GraphStoreResult<CatalogCommitResult> {
            self.inner.commit_catalog_updates(updates, meta)
        }
    }
}
