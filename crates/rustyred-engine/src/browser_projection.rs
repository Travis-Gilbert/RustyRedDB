//! The browser projection's storage half: durable local state as a
//! [`ChunkStore`] over the log's six-operation storage seam.
//!
//! SPEC-THEOREMWEB-PWA-HTML-CANVAS-1.1 section 32 puts browser-local durable
//! application data in RustyRed and forbids a TheoremWeb-specific database;
//! section 33 draws the ownership as `rustyredcore_THG -> browser storage
//! projection -> OPFS`. This module is that projection's storage: it owns
//! chunks and their committed named-root catalog inside one append-only log
//! file, reached only through [`LogFile`] -- `getSize`, `read` at an offset,
//! `append`, `truncate`, `flush`, and the directory sync that is honestly a
//! no-op on OPFS.
//!
//! # Why this is one more `ChunkStore` and not a second store
//!
//! [`RedCoreGraphStore`](crate::RedCoreGraphStore) already treats the chunk
//! store as a seam: its field is an `Arc<dyn UnifiedChunkStore>`, the memory
//! constructor already ships [`MemChunkStore`](crate::chunk_store::MemChunkStore)
//! in production, and [`UnifiedChunkStore`](crate::object_store::UnifiedChunkStore)
//! is a blanket implementation over `ChunkStore + ChunkStoreCatalog`. So the
//! browser composition is one more implementation of those traits beside the
//! memory one, and [`DiskObjectStore::from_chunk_store`] consumes it with no
//! change of its own. Section 36's single store holds: there is no IndexedDB
//! database here, and nothing in this module is Theorem-specific.
//!
//! # The commit protocol, re-specified for a store with no `rename`
//!
//! The native journal commits by writing a root frame after its catalog chunk
//! and issuing exactly one fsync; compaction is the only operation that
//! rewrites by temp file plus `fs::rename` (`journal.rs`). OPFS has neither an
//! atomic rename nor a directory entry to fsync, so the question this module
//! answers is what [`ChunkStoreCatalog::commit_catalog_updates`] means without
//! them.
//!
//! The log is a header followed by self-delimiting frames: `kind`,
//! `payload_len`, a CRC over the payload, then the payload. The CRC does not cover
//! the declared length. V1 stores canonical bytes directly; V2 may store a
//! smaller zstd envelope with bounded decoded length and canonical hash. Every
//! address is *recomputed from verified decoded bytes*, and V2 checks it against
//! the envelope's hash before admission. Existing V1 journals remain V1; only
//! an explicitly selected empty-store opener creates V2. Opening scans from the first
//! frame and builds the **verified prefix**. A short final header/payload, or a
//! complete corrupt chunk suffix with no later valid commit root, is uncommitted
//! and may be truncated and synced. A length that runs past EOF triggers a
//! bounded rescan of at most the existing 64 MiB frame-payload limit; a later
//! self-checking root is committed-data corruption, while an inconclusive budget
//! is an explicit unknown outcome. Neither diagnosis is repairable. A true final
//! partial frame with no verifiable later root remains repairable.
//! [`ProjectionChunkStore::inspect`] reports these same classifications without
//! mutation. The committed root is the last verified root frame in the prefix.
//!
//! With that rule, a catalog publication is:
//!
//! 1. merge the updates into the current catalog while holding the root-state
//!    lock, and derive the new commit's single physical parent from that same
//!    locked state, so concurrent publishers cannot race each other;
//! 2. append the catalog chunk when it is new, then append the root frame,
//!    then issue exactly one `flush` -- the only durability barrier in the
//!    operation, and the point at which the commit becomes visible;
//! 3. only then publish the new chunk and root into the in-memory index.
//!
//! A crash between step 2's two appends leaves a complete, verified catalog
//! chunk in the log with the *previous* root frame still last, so the visible
//! state is exactly the previous commit and the orphan chunk is unreferenced
//! bytes that reclamation would collect. A crash inside either frame leaves a
//! frame that fails its CRC or its length, so the prefix ends before it and the
//! truncation removes it. The native contract -- the new catalog chunk and its
//! commit become visible together or neither does -- therefore holds without
//! `rename`: the root frame's single flush *is* the visibility point, and the
//! only atomicity the operation needs is "one append either landed whole or
//! did not", which the frame's own length and CRC decide on the way back in.
//!
//! # What this store refuses, and why
//!
//! [`ChunkStoreCompaction::compact_reachable`] is refused by name. Reclamation
//! is the one operation whose crash consistency genuinely needs a `rename`:
//! the compaction image is a *different* byte layout, so it cannot be appended
//! beside the old one in the same file, and a rewrite that starts by truncating
//! destroys the committed root before the replacement exists -- a crash there
//! loses the store rather than a tail. Refusing loudly is the honest answer;
//! the rename-free route is a two-file generation switch (write the new
//! generation to a second named log, flush it, then append one switch frame to
//! the first and flush that), which the named openers on [`LogFile`] now
//! support but which is not built here.
//!
//! The in-flight pin methods keep their trait defaults for a reason: a pin
//! protects staged bytes from reclamation, and this store cannot reclaim, so a
//! pin would guard nothing. `ChunkStore` documents the defaults as the
//! compatibility path for exactly that case. The visible consequence is that
//! the `ChunkPublicationWindow` `DiskObjectStore::from_chunk_store` opens
//! around its empty-catalog write reports its release as missing a receipt and
//! retries once in `Drop`; nothing reads that report, and nothing can be lost
//! by it, because there is no reclamation for the receipt to have protected. A
//! store that does implement reclamation must implement them with it.
//!
//! # What runs on the browser path
//!
//! [`browser_journal_commit_policy`] maps a [`RedCoreDurability`] onto the
//! journal's commit policy and **refuses `AofEverysec` by name**, because that
//! arm selects `JournalCommitPolicy::Grouped`, which constructs a
//! `std::thread` group-commit worker that a `wasm32` build has no spawner for.
//! `RedCoreOptions::default()` is `AofEverysec`, so the hazard is the default
//! one, and [`browser_red_core_options`] is the composition that cannot reach
//! it. `Immediate` needs no browser arm: the flush it issues is
//! [`LogFile::sync`], which is `FileSystemSyncAccessHandle::flush` on OPFS and
//! a real durability barrier there.
//!
//! # Testing discipline
//!
//! This module is compiled on every target on purpose. Its store is written
//! against the seam, so the host runs the same commit protocol, recovery and
//! refusal paths over a real file through
//! [`NativeLogFile`](theorem_oplog::NativeLogFile) that the browser runs over
//! OPFS -- which is what makes the protocol testable where a browser is not
//! available. A passing host test is still not a browser run: nothing here
//! proves OPFS storage or retrieval.

use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::Serialize;
use theorem_oplog::{LogFile, PhysicalLogError};

use crate::chunk_store::{
    merge_catalog_updates, CatalogCommitResult, ChunkStore, ChunkStoreCatalog,
    ChunkStoreCompaction, ChunkWriteReport, CommitMeta, CommittedRoot, CompactionReport,
    RootCatalog,
};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::journal::JournalCommitPolicy;
use crate::{RedCoreDurability, RedCoreOptions};

/// The file the projection's chunk journal owns inside a store directory.
///
/// A second name beside the ledger's `theorem.oplog` on purpose: the ledger and
/// this store are two logs, and neither opens the other's file. Both names are
/// resolved by their own backend's opener, so this one is a fragment below the
/// OPFS root in a browser and an ordinary file name on a host.
pub const BROWSER_CHUNK_JOURNAL_FILE: &str = "redcore.chunks";

/// The durability a browser composition runs unless it is told otherwise.
///
/// `RedCoreOptions::default()` is `AofEverysec`, which the browser cannot run;
/// `AofAlways` selects `JournalCommitPolicy::Immediate` and costs one flush per
/// root commit, which is what the commit protocol above already performs.
pub const BROWSER_DEFAULT_DURABILITY: RedCoreDurability = RedCoreDurability::AofAlways;

/// Identifies this journal as the projection's chunk log rather than a ledger.
const JOURNAL_MAGIC: [u8; 8] = *b"RRBPCJ01";
const JOURNAL_VERSION: u8 = 1;
const COMPRESSED_JOURNAL_VERSION: u8 = 2;

#[path = "browser_projection_codec.rs"]
mod codec;
const JOURNAL_HEADER_BYTES: u64 = 9;

/// `kind`, `payload_len`, `payload_crc` -- no address, deliberately.
const FRAME_HEADER_BYTES: u64 = 9;

/// The same bound the native journal puts on a decoded payload, for the same
/// reason: a corrupt or torn length must not become an unbounded allocation.
/// Encoded and decoded chunks each retain this ceiling. V2 compression is used
/// only when the envelope is smaller than the canonical bytes. Damaged-length
/// rescans retain this same bound.
const MAX_FRAME_PAYLOAD: u32 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum FrameKind {
    Chunk = 1,
    Root = 2,
    /// Payload-free sticky metadata; it never changes the committed catalog.
    PressureMarker = 3,
    /// V2 only: decoded length, canonical hash, then one zstd payload.
    CompressedChunk = 4,
}

impl FrameKind {
    fn from_byte(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::Chunk),
            2 => Some(Self::Root),
            3 => Some(Self::PressureMarker),
            4 => Some(Self::CompressedChunk),
            _ => None,
        }
    }
}

/// One journal frame, before it is encoded.
#[derive(Debug)]
struct Frame {
    kind: FrameKind,
    payload: Vec<u8>,
}

/// Where one verified chunk frame's payload sits in the log file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkLocation {
    payload_offset: u64,
    /// Canonical decoded length used by pre-allocation admission.
    payload_len: u32,
    encoded_len: u32,
    compressed: bool,
}

struct ProjectionState {
    file: Box<dyn LogFile>,
    format_version: u8,
    /// Verified frames only. A recovered scan populates this from the verified
    /// prefix, and a write populates it only after the frame that carries the
    /// bytes is complete.
    index: HashMap<Hash, ChunkLocation>,
    /// The last verified root frame in the verified prefix.
    committed: Option<CommittedRoot>,
    /// Set when a failed append could not be rolled back. A handle in this
    /// state refuses further use by name rather than appending behind bytes
    /// whose interpretation it can no longer vouch for.
    poisoned: Option<String>,
    /// Optional first-release safe-growth bounds. Existing openers remain
    /// unbounded for compatibility; product owners opt in at their composition.
    bounds: Option<ProjectionStorageBounds>,
    /// A verified frame records that a real adapter failure occurred.
    pressure_marker_observed: bool,
    /// A real failure was observed but could not be recorded durably. This is
    /// handle-local unless an incomplete/corrupt marker is recovered later.
    pressure_unknown: bool,
    /// Safe-bound refusals are estimates, not physical adapter failures.
    bound_refusal_observed: bool,
    #[cfg(any(test, feature = "test-fault-injection"))]
    fail_next_publication_flush: bool,
}

/// RustyRed's durable browser-local state, over the log's storage seam.
///
/// The store is the browser projection's physical half: chunks and one
/// committed named-root catalog, in one append-only log file reached through
/// [`LogFile`]. It implements [`ChunkStore`], [`ChunkStoreCatalog`] and
/// [`ChunkStoreCompaction`], so it satisfies
/// [`UnifiedChunkStore`](crate::object_store::UnifiedChunkStore) through the
/// blanket implementation and can be handed to
/// [`DiskObjectStore::from_chunk_store`](crate::object_store::DiskObjectStore::from_chunk_store)
/// without either side changing.
///
/// Every operation takes one lock. That lock is the root-state lock the catalog
/// contract requires: it spans reading the current commit, merging the update,
/// deriving the new parent, appending, flushing and publishing, so two
/// publishers cannot produce a root that lost the other's names.
pub struct ProjectionChunkStore {
    state: Mutex<ProjectionState>,
}

/// How an inspection classifies bytes after the last verified frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionTailState {
    /// The file ends exactly after the verified prefix.
    Clean,
    /// A short final header or payload has no verifiable frame after it and is
    /// therefore an incomplete, uncommitted append.
    Incomplete,
    /// A complete but invalid chunk frame has no later commit root and is
    /// therefore an uncommitted suffix that recovery may truncate.
    UncommittedCorruption,
    /// Accepted data is damaged before a later valid commit root. Recovery must
    /// refuse without changing bytes; inspection reports this state read-only.
    CommittedCorruption,
    /// A root frame or bounded rescan cannot establish the publication outcome.
    /// Recovery must refuse without changing bytes.
    CommitStateUnknown,
}

/// Bounds for append-only projection growth. `commit_headroom_bytes` is
/// available to a publication frame after staged user bytes reach their lower
/// admission limit. `recovery_headroom_bytes` is held for bounded recovery
/// bookkeeping such as the sticky pressure marker; commits and user chunks do
/// not consume it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionStorageBounds {
    pub max_log_bytes: u64,
    pub commit_headroom_bytes: u64,
    pub recovery_headroom_bytes: u64,
}

impl ProjectionStorageBounds {
    pub fn validate(self) -> GraphStoreResult<Self> {
        let reserved = self
            .commit_headroom_bytes
            .checked_add(self.recovery_headroom_bytes)
            .and_then(|reserved| JOURNAL_HEADER_BYTES.checked_add(reserved))
            .ok_or_else(storage_bounds_invalid)?;
        if self.max_log_bytes <= reserved {
            return Err(storage_bounds_invalid());
        }
        Ok(self)
    }

    #[must_use]
    pub const fn user_growth_limit_bytes(self) -> u64 {
        self.max_log_bytes
            .saturating_sub(self.commit_headroom_bytes)
            .saturating_sub(self.recovery_headroom_bytes)
    }

    #[must_use]
    pub const fn publication_limit_bytes(self) -> u64 {
        self.max_log_bytes
            .saturating_sub(self.recovery_headroom_bytes)
    }
}

/// Pressure knowledge for a projection journal, preserving the distinction
/// between an observed condition and missing durable evidence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProjectionPressure {
    Clear,
    Observed { reason: &'static str },
    Unknown { reason: &'static str },
}

/// Payload-free accounting facts for a projection journal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionStorageSnapshot {
    pub log_bytes: u64,
    pub chunk_count: usize,
    pub committed_catalog_roots: usize,
    pub bounds: Option<ProjectionStorageBounds>,
    pub pressure: ProjectionPressure,
}

/// Read-only result for the browser projection's existing journal format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionInspection {
    /// None for an empty file; otherwise its verified physical header version.
    pub format_version: Option<u8>,
    pub file_len: u64,
    pub verified_prefix_len: u64,
    pub committed: Option<CommittedRoot>,
    pub tail: ProjectionTailState,
    /// Detail accompanying `CommittedCorruption`, `CommitStateUnknown`, or an
    /// uncommitted corrupt suffix diagnosis.
    pub diagnosis: Option<String>,
}

impl std::fmt::Debug for ProjectionChunkStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProjectionChunkStore")
            .finish_non_exhaustive()
    }
}

impl ProjectionChunkStore {
    /// Size from the admitted frame index, without loading the payload. This
    /// is an allocation-admission hint only: callers must still use `get` to
    /// verify content integrity before returning any bytes. Addresses are
    /// immutable and this append-only backend cannot replace their length.
    pub fn indexed_chunk_len(&self, address: Hash) -> GraphStoreResult<Option<u64>> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        Ok(state
            .index
            .get(&address)
            .map(|location| u64::from(location.payload_len)))
    }

    /// Inspect the existing journal without truncating or otherwise mutating it.
    ///
    /// Incomplete uncommitted tails and committed-corruption/unknown diagnoses
    /// are returned in [`ProjectionInspection::tail`]; inspection never mutates
    /// bytes. Recovery refuses the latter two states by name before any write.
    pub fn inspect(file: &mut dyn LogFile) -> GraphStoreResult<ProjectionInspection> {
        inspect_file(file)
    }

    /// Recovers the journal behind `file` and returns a store over it.
    ///
    /// Recovery repairs only an incomplete or demonstrably uncommitted suffix.
    /// Committed corruption and unknown commit outcomes are named refusals; no
    /// truncate or sync is attempted for either diagnosis.
    /// An absent (zero-length) file is a fresh journal: the header is written
    /// and made durable before the first frame.
    pub fn open(file: Box<dyn LogFile>) -> GraphStoreResult<Self> {
        Self::open_with_storage_bounds(file, None)
    }

    /// Open with a declared append-only growth bound. Existing bytes are never
    /// truncated because they exceed the new bound: reads remain available and
    /// subsequent growth is refused by name. Core failure/refusal observations
    /// are handle-local and reset on open; only a verified pressure-marker frame
    /// is recovered here. The local-runtime layer also merges its durable
    /// `redcore.pressure` sidecar. If both channels fail before writing any bytes,
    /// process death necessarily removes the only observation; the next adapter
    /// write remains authoritative.
    pub fn open_with_storage_bounds(
        file: Box<dyn LogFile>,
        bounds: Option<ProjectionStorageBounds>,
    ) -> GraphStoreResult<Self> {
        Self::open_with_initial_version(file, bounds, JOURNAL_VERSION)
    }

    /// Opt in to V2 compression for a fresh, empty journal only. An existing
    /// V1 journal retains its header and uncompressed writer; no bytes migrate
    /// or get reclaimed during this open. Ordinary new readers support both.
    pub fn open_with_compressed_chunks(
        file: Box<dyn LogFile>,
        bounds: Option<ProjectionStorageBounds>,
    ) -> GraphStoreResult<Self> {
        Self::open_with_initial_version(file, bounds, COMPRESSED_JOURNAL_VERSION)
    }

    fn open_with_initial_version(
        file: Box<dyn LogFile>,
        bounds: Option<ProjectionStorageBounds>,
        initial_version: u8,
    ) -> GraphStoreResult<Self> {
        let bounds = bounds.map(ProjectionStorageBounds::validate).transpose()?;
        let mut file = file;
        let recovered = recover(&mut *file, initial_version)?;
        let store = Self {
            state: Mutex::new(ProjectionState {
                file,
                format_version: recovered.format_version,
                index: recovered.index,
                committed: recovered.committed,
                poisoned: None,
                bounds,
                pressure_marker_observed: recovered.pressure_marker_observed,
                pressure_unknown: recovered.pressure_unknown,
                bound_refusal_observed: false,
                #[cfg(any(test, feature = "test-fault-injection"))]
                fail_next_publication_flush: false,
            }),
        };
        Ok(store)
    }

    /// Opens the projection's chunk journal in `directory` on a host.
    ///
    /// The native arm exists so the same commit protocol is exercised by real
    /// files in this crate's tests. The browser builds its handle through
    /// [`Self::open_in_worker`] instead, and the two hand the same
    /// [`LogFile`] to [`Self::open`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_at(directory: impl AsRef<Path>) -> GraphStoreResult<Self> {
        let file = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.as_ref(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        Self::open(Box::new(file))
    }

    /// Opens the projection's chunk journal in OPFS, from a dedicated worker.
    ///
    /// `createSyncAccessHandle` returns a promise and exists only in a worker,
    /// so a browser cannot open this store synchronously; everything after the
    /// handle exists is the synchronous code above.
    #[cfg(target_arch = "wasm32")]
    pub async fn open_in_worker(directory: &str) -> GraphStoreResult<Self> {
        let file =
            theorem_oplog::OpfsLogFile::open_named_in_worker(directory, BROWSER_CHUNK_JOURNAL_FILE)
                .await
                .map_err(storage_failure)?;
        Self::open(Box::new(file))
    }

    /// Number of distinct chunks this store currently holds.
    ///
    /// Observability only, and deliberately not part of the [`ChunkStore`]
    /// contract -- the same seam [`MemChunkStore`](crate::chunk_store::MemChunkStore)
    /// exposes under the same name.
    pub fn chunk_count(&self) -> GraphStoreResult<usize> {
        Ok(self.lock_state()?.index.len())
    }

    /// Return physical log use and the durable chunk/catalog facts without
    /// exposing payloads. Uncommitted valid frames are conservatively counted
    /// as retained bytes because this backend has no safe compaction path.
    pub fn storage_snapshot(&self) -> GraphStoreResult<ProjectionStorageSnapshot> {
        let state = self.lock_state()?;
        let log_bytes = state.file.len().map_err(storage_failure)?;
        let limit_pressure = state
            .bounds
            .is_some_and(|bounds| log_bytes >= bounds.user_growth_limit_bytes());
        let pressure = if state.pressure_marker_observed {
            ProjectionPressure::Observed {
                reason: "persisted_adapter_failure_marker",
            }
        } else if state.pressure_unknown {
            ProjectionPressure::Unknown {
                reason: "adapter_failure_marker_unavailable",
            }
        } else if state.bound_refusal_observed {
            ProjectionPressure::Observed {
                reason: "safe_growth_refused",
            }
        } else if limit_pressure {
            ProjectionPressure::Observed {
                reason: "reserved_headroom_active",
            }
        } else {
            ProjectionPressure::Clear
        };
        Ok(ProjectionStorageSnapshot {
            log_bytes,
            chunk_count: state.index.len(),
            committed_catalog_roots: if state.committed.is_some() { 1 } else { 0 },
            bounds: state.bounds,
            pressure,
        })
    }

    fn lock_state(&self) -> GraphStoreResult<MutexGuard<'_, ProjectionState>> {
        self.state
            .lock()
            .map_err(|_| GraphStoreError::new("chunk_store_poisoned", "chunk store lock poisoned"))
    }

    /// Fail the next catalog publication flush through the normal rollback path.
    /// This hook is compiled only for tests and downstream failure oracles.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn fail_next_publication_flush_for_test(&self) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        state.fail_next_publication_flush = true;
        Ok(())
    }
}

impl ChunkStore for ProjectionChunkStore {
    fn get_bounded(&self, addr: Hash, max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        if state
            .index
            .get(&addr)
            .is_some_and(|location| u64::from(location.payload_len) > max_bytes as u64)
        {
            return Err(GraphStoreError::new(
                "chunk_read_bound_refused",
                "chunk exceeds admitted read bytes",
            ));
        }
        chunk_bytes(&mut state, addr)
    }

    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        chunk_bytes(&mut state, addr)
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.get(addr).map(|bytes| bytes.is_some())
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        for address in addrs {
            if seen.insert(*address) && chunk_bytes(&mut state, *address)?.is_none() {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.write_accounted(chunks).map(|_| ())
    }

    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        // Validate and own the whole batch before the first append, so a
        // malformed later entry cannot leave earlier entries looking accepted.
        let mut staged = Vec::with_capacity(chunks.len());
        let mut seen = HashSet::with_capacity(chunks.len());
        for (address, bytes) in chunks {
            verify_address(*address, bytes)?;
            ensure_payload_bound(bytes.len())?;
            if seen.insert(*address) {
                staged.push((*address, Arc::<[u8]>::from(*bytes)));
            }
        }

        let mut state = self.lock_state()?;
        ensure_usable(&state)?;

        let mut frames = Vec::new();
        let mut admitted = Vec::new();
        let mut report = ChunkWriteReport::default();
        for (address, bytes) in staged {
            if state.index.contains_key(&address) {
                continue;
            }
            let payload_len = u32::try_from(bytes.len()).map_err(|_| size_overflow())?;
            report.chunks_written = report
                .chunks_written
                .checked_add(1)
                .ok_or_else(size_overflow)?;
            report.bytes_written = report
                .bytes_written
                .checked_add(bytes.len())
                .ok_or_else(size_overflow)?;
            let frame = prepare_chunk_frame(state.format_version, &bytes)?;
            let encoded_len = u32::try_from(frame.payload.len()).map_err(|_| size_overflow())?;
            admitted.push((
                address,
                payload_len,
                encoded_len,
                frame.kind == FrameKind::CompressedChunk,
            ));
            frames.push(frame);
        }

        if frames.is_empty() {
            return Ok(report);
        }

        // Chunk frames are not a durability barrier on their own: the root
        // frame's flush is what makes them reachable, and a chunk that never
        // gets one is recovered as an unreferenced frame in the prefix. The
        // index changes only after every frame in the batch is complete, so a
        // partial later frame cannot leave a hole in the middle of a log this
        // process can keep using.
        let offsets = append_frames(
            &mut state,
            &frames,
            false,
            GrowthReservation::StagedUserData,
        )?;
        for ((address, payload_len, encoded_len, compressed), payload_offset) in
            admitted.into_iter().zip(offsets)
        {
            state.index.insert(
                address,
                ChunkLocation {
                    payload_offset,
                    payload_len,
                    encoded_len,
                    compressed,
                },
            );
        }
        Ok(report)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let _catalog = catalog_in_state(&mut state, root)?;
        let mut meta = meta;
        meta.parents = state
            .committed
            .as_ref()
            .map(|committed| committed.commit)
            .into_iter()
            .collect();
        let committed = CommittedRoot::new(root, meta)?;
        publish_root(&mut state, &committed)?;
        Ok(committed.commit)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let Some(committed) = state.committed.clone() else {
            return Ok(None);
        };
        committed.verify_commit(committed.commit)?;
        catalog_in_state(&mut state, committed.root)?;
        Ok(Some(committed.root))
    }
}

impl ChunkStoreCatalog for ProjectionChunkStore {
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

impl ProjectionChunkStore {
    fn commit_catalog_updates_checked(
        &self,
        expected: Option<Option<Hash>>,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        // The lock spans merge, parent derivation, append, flush and
        // publication: this is the root-state lock the trait contract names.
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        if expected.is_some_and(|expected| {
            expected != state.committed.as_ref().map(|committed| committed.root)
        }) {
            return Err(GraphStoreError::new(
                "chunk_store_catalog_precondition_failed",
                "accepted catalog changed before publication",
            ));
        }

        let current = match state.committed.clone() {
            Some(committed) => {
                committed.verify_commit(committed.commit)?;
                Some(catalog_in_state(&mut state, committed.root)?)
            }
            None => None,
        };
        let catalog = merge_catalog_updates(current, updates)?;
        catalog.graph_root()?;
        for (name, root) in catalog.roots() {
            if chunk_bytes(&mut state, *root)?.is_none() {
                return Err(named_root_missing(name, *root));
            }
        }

        let catalog_bytes = catalog.to_bytes()?;
        let catalog_root = Hash::of(&catalog_bytes);
        if let Some(existing) = chunk_bytes(&mut state, catalog_root)? {
            if existing.as_ref() != catalog_bytes.as_slice() {
                return Err(GraphStoreError::new(
                    "chunk_store_catalog_address_collision",
                    format!("existing catalog chunk {catalog_root} has different bytes"),
                ));
            }
        }

        let catalog_is_new = !state.index.contains_key(&catalog_root);
        let (committed, frames) = prepare_catalog_frames(
            catalog_root,
            &catalog_bytes,
            state.committed.as_ref(),
            meta,
            catalog_is_new,
        )?;

        let offsets = append_frames(&mut state, &frames, true, GrowthReservation::Publication)?;
        if catalog_is_new {
            state.index.insert(
                catalog_root,
                ChunkLocation {
                    payload_offset: offsets[0],
                    payload_len: u32::try_from(catalog_bytes.len()).map_err(|_| size_overflow())?,
                    encoded_len: u32::try_from(catalog_bytes.len()).map_err(|_| size_overflow())?,
                    compressed: false,
                },
            );
        }
        state.committed = Some(committed.clone());

        Ok(CatalogCommitResult {
            catalog_root,
            commit: committed.commit,
        })
    }
}

impl ChunkStoreCompaction for ProjectionChunkStore {
    /// Refused by name, deliberately.
    ///
    /// Reclamation is the one operation of this store whose crash consistency
    /// a single append-only file cannot carry: see the module documentation.
    /// Every argument is ignored -- the refusal is a property of the store, not
    /// of the request -- and it is a named error rather than a silent `Ok`, so
    /// a caller that reached it learns which case it is in.
    fn compact_reachable(
        &self,
        expected_catalog_root: Option<Hash>,
        reachable: &[Hash],
    ) -> GraphStoreResult<CompactionReport> {
        let _ = (expected_catalog_root, reachable);
        Err(GraphStoreError::new(
            "chunk_store_browser_compaction_refused",
            "the browser projection refuses reachability reclamation: it needs an atomic \
             replacement of the whole file, and OPFS has neither a rename nor a directory \
             entry to make one durable. A rewrite that truncates before it writes destroys \
             the committed root instead of a torn tail, so reclamation is refused rather \
             than left unsafe. The rename-free route is a two-file generation switch: write \
             the compacted image to a second named log, flush it, then append one switch \
             record to this file and flush that",
        ))
    }
}

/// The commit policy a browser composition may run, or a named refusal.
///
/// Mirrors [`RedCoreDurability`](crate::RedCoreDurability)'s own mapping onto
/// the journal's policy, with one arm replaced: `AofEverysec` asks for
/// `JournalCommitPolicy::Grouped`, whose runtime constructs a `std::thread`
/// group-commit worker, and a `wasm32` build has no thread spawner -- the call
/// compiles and fails when it executes. Rather than let that reach a browser,
/// the arm is refused here by name, which is the shape
/// [`crate::file_lock`] uses for the other wasm32 boundary.
///
/// `None` is not a refusal: a memory-only `RedCoreDurability::None` composes no
/// durable journal at all, and `Ok(None)` says exactly that.
pub fn browser_journal_commit_policy(
    durability: RedCoreDurability,
) -> GraphStoreResult<Option<JournalCommitPolicy>> {
    match durability {
        RedCoreDurability::None => Ok(None),
        RedCoreDurability::AofAlways | RedCoreDurability::SnapshotOnly => {
            // `Immediate` needs no browser arm. Its flush is `LogFile::sync`,
            // which is `FileSystemSyncAccessHandle::flush` on OPFS: one append
            // and one barrier, the same protocol this module's commit uses.
            Ok(Some(JournalCommitPolicy::Immediate))
        }
        RedCoreDurability::AofEverysec => Err(GraphStoreError::new(
            "redcore_browser_durability_refused",
            "the browser projection refuses aof_everysec by name: it selects \
             JournalCommitPolicy::Grouped, whose commit runtime starts a std::thread \
             group-commit worker, and a wasm32 build has no thread spawner. Select \
             aof_always or snapshot_only, both of which select Immediate, or compose no \
             durable journal with durability none",
        )),
    }
}

/// The options a browser composition runs: `RedCoreOptions::default()` with the
/// durability the browser can actually execute.
///
/// Every other field keeps its default, because none of them changes shape on
/// this target. The one that does is the durability, and its default
/// (`aof_everysec`) is the value
/// [`browser_journal_commit_policy`] refuses, so a browser must not reach for
/// `RedCoreOptions::default()` on its own.
pub fn browser_red_core_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: BROWSER_DEFAULT_DURABILITY,
        ..RedCoreOptions::default()
    }
}

struct Recovered {
    format_version: u8,
    index: HashMap<Hash, ChunkLocation>,
    committed: Option<CommittedRoot>,
    pressure_marker_observed: bool,
    pressure_unknown: bool,
}

struct Scanned {
    format_version: u8,
    inspection: ProjectionInspection,
    index: HashMap<Hash, ChunkLocation>,
    pressure_marker_observed: bool,
    pressure_unknown: bool,
}

enum FrameRead {
    Valid(Frame),
    Incomplete,
    IncompletePayload,
    CorruptChunk { next_offset: u64, detail: String },
    CorruptPressureMarker { next_offset: u64 },
    CorruptRoot(String),
    Unsupported(String),
}

enum RootRescan {
    NoVerifiableRoot,
    VerifiableRoot,
    BudgetExceeded(String),
}

/// A read-only scan shared by inspection and recovery.
fn inspect_file(file: &mut dyn LogFile) -> GraphStoreResult<ProjectionInspection> {
    scan_file(file).map(|scanned| scanned.inspection)
}

fn scan_file(file: &mut dyn LogFile) -> GraphStoreResult<Scanned> {
    let length = file.len().map_err(storage_failure)?;
    if length == 0 {
        return Ok(Scanned {
            format_version: JOURNAL_VERSION,
            inspection: ProjectionInspection {
                format_version: None,
                file_len: 0,
                verified_prefix_len: 0,
                committed: None,
                tail: ProjectionTailState::Clean,
                diagnosis: None,
            },
            index: HashMap::new(),
            pressure_marker_observed: false,
            pressure_unknown: false,
        });
    }
    if length < JOURNAL_HEADER_BYTES {
        return Err(foreign_journal(format!(
            "the journal is {length} bytes, shorter than its {JOURNAL_HEADER_BYTES}-byte header"
        )));
    }
    let mut header = [0_u8; JOURNAL_HEADER_BYTES as usize];
    file.read_exact_at(0, &mut header)
        .map_err(storage_failure)?;
    if header[..JOURNAL_MAGIC.len()] != JOURNAL_MAGIC {
        return Err(foreign_journal(format!(
            "the journal does not begin with this store's magic: {:?}",
            &header[..JOURNAL_MAGIC.len()]
        )));
    }
    let format_version = header[JOURNAL_MAGIC.len()];
    if !matches!(format_version, JOURNAL_VERSION | COMPRESSED_JOURNAL_VERSION) {
        return Err(unsupported_format(format!(
            "the journal carries format version {}, and this build supports {JOURNAL_VERSION}/{COMPRESSED_JOURNAL_VERSION}",
            header[JOURNAL_MAGIC.len()]
        )));
    }

    let mut index = HashMap::new();
    let mut committed = None;
    let mut offset = JOURNAL_HEADER_BYTES;
    let mut tail = ProjectionTailState::Clean;
    let mut diagnosis = None;
    let mut pressure_marker_observed = false;
    let mut pressure_unknown = false;
    while offset < length {
        match read_frame_in_version(file, offset, length, format_version)? {
            FrameRead::Valid(frame) => {
                let payload_len =
                    u32::try_from(frame.payload.len()).map_err(|_| size_overflow())?;
                match frame.kind {
                    FrameKind::Chunk | FrameKind::CompressedChunk => {
                        let compressed = frame.kind == FrameKind::CompressedChunk;
                        if compressed && format_version != COMPRESSED_JOURNAL_VERSION {
                            return Err(unsupported_format("compressed chunk in a V1 journal"));
                        }
                        let decoded = if compressed {
                            Some(codec::decode(&frame.payload)?)
                        } else {
                            None
                        };
                        let bytes = decoded.as_deref().unwrap_or(&frame.payload);
                        // Identity belongs to canonical bytes, never their envelope.
                        let address = Hash::of(bytes);
                        index.insert(
                            address,
                            ChunkLocation {
                                payload_offset: offset + FRAME_HEADER_BYTES,
                                payload_len: u32::try_from(bytes.len())
                                    .map_err(|_| size_overflow())?,
                                encoded_len: payload_len,
                                compressed,
                            },
                        );
                    }
                    FrameKind::Root => {
                        let root = CommittedRoot::from_bytes(&frame.payload)
                            .map_err(|error| unknown_commit_outcome(offset, error.message))?;
                        root.verify_commit(Hash::of(&frame.payload))
                            .map_err(|error| unknown_commit_outcome(offset, error.message))?;
                        committed = Some(root);
                    }
                    FrameKind::PressureMarker => {
                        if frame.payload.is_empty() {
                            pressure_marker_observed = true;
                        } else {
                            pressure_unknown = true;
                        }
                    }
                }
                offset += FRAME_HEADER_BYTES + u64::from(payload_len);
            }
            FrameRead::Incomplete => {
                pressure_unknown |= incomplete_frame_is_pressure_marker(file, offset, length)?;
                tail = ProjectionTailState::Incomplete;
                break;
            }
            FrameRead::IncompletePayload => {
                pressure_unknown |= incomplete_frame_is_pressure_marker(file, offset, length)?;
                match suffix_has_verifiable_commit_root(file, offset + FRAME_HEADER_BYTES, length)?
                {
                    RootRescan::VerifiableRoot => {
                        tail = ProjectionTailState::CommittedCorruption;
                        diagnosis = Some(
                            "declared payload length runs past EOF before a later valid commit root"
                                .to_string(),
                        );
                    }
                    RootRescan::NoVerifiableRoot => tail = ProjectionTailState::Incomplete,
                    RootRescan::BudgetExceeded(detail) => {
                        tail = ProjectionTailState::CommitStateUnknown;
                        diagnosis = Some(detail);
                    }
                }
                break;
            }
            FrameRead::CorruptChunk {
                next_offset,
                detail,
            } => {
                match suffix_has_commit_root(file, next_offset, length, format_version) {
                    Ok(true) => {
                        tail = ProjectionTailState::CommittedCorruption;
                        diagnosis = Some(detail);
                    }
                    Ok(false) => {
                        tail = ProjectionTailState::UncommittedCorruption;
                        diagnosis = Some(detail);
                    }
                    Err(error) if error.code == "chunk_store_browser_commit_state_unknown" => {
                        tail = ProjectionTailState::CommitStateUnknown;
                        diagnosis = Some(error.message);
                    }
                    Err(error) => return Err(error),
                }
                break;
            }
            FrameRead::CorruptPressureMarker { next_offset } => {
                pressure_unknown = true;
                offset = next_offset;
            }
            FrameRead::CorruptRoot(detail) => {
                tail = ProjectionTailState::CommitStateUnknown;
                diagnosis = Some(detail);
                break;
            }
            FrameRead::Unsupported(detail) => return Err(unsupported_format(detail)),
        }
    }

    if let Some(root) = committed.as_ref() {
        catalog_from_index(file, &index, root.root)?;
    }

    Ok(Scanned {
        format_version,
        inspection: ProjectionInspection {
            format_version: Some(format_version),
            file_len: length,
            verified_prefix_len: offset,
            committed,
            tail,
            diagnosis,
        },
        index,
        pressure_marker_observed,
        pressure_unknown: pressure_unknown && !pressure_marker_observed,
    })
}

/// Scans and only truncates a suffix that inspection proved uncommitted.
fn recover(file: &mut dyn LogFile, initial_version: u8) -> GraphStoreResult<Recovered> {
    let scanned = scan_file(file)?;
    if scanned.inspection.file_len == 0 {
        // A fresh journal. The header goes in and is made durable here, before
        // the first frame, so a log with frames in it always has an identity to
        // check; `sync_directory` is honestly a no-op on OPFS.
        file.append(&journal_header(initial_version))
            .map_err(storage_failure)?;
        file.sync().map_err(storage_failure)?;
        file.sync_directory().map_err(storage_failure)?;
        return Ok(Recovered {
            format_version: initial_version,
            index: HashMap::new(),
            committed: None,
            pressure_marker_observed: false,
            pressure_unknown: false,
        });
    }
    match scanned.inspection.tail {
        ProjectionTailState::CommittedCorruption => {
            return Err(corrupt_committed_data(
                scanned.inspection.verified_prefix_len,
                scanned
                    .inspection
                    .diagnosis
                    .as_deref()
                    .unwrap_or("committed data is corrupt"),
            ));
        }
        ProjectionTailState::CommitStateUnknown => {
            return Err(unknown_commit_outcome(
                scanned.inspection.verified_prefix_len,
                scanned
                    .inspection
                    .diagnosis
                    .as_deref()
                    .unwrap_or("the commit outcome cannot be established"),
            ));
        }
        ProjectionTailState::Clean => {}
        ProjectionTailState::Incomplete | ProjectionTailState::UncommittedCorruption => {
            file.truncate(scanned.inspection.verified_prefix_len)
                .map_err(storage_failure)?;
            file.sync().map_err(storage_failure)?;
        }
    }
    Ok(Recovered {
        format_version: scanned.format_version,
        index: scanned.index,
        committed: scanned.inspection.committed,
        pressure_marker_observed: scanned.pressure_marker_observed,
        pressure_unknown: scanned.pressure_unknown,
    })
}

/// Reads and fully verifies one frame. Only a short final header/payload is an
/// incomplete write; complete invalid frames keep their own named diagnosis.
fn read_frame_in_version(
    file: &mut dyn LogFile,
    offset: u64,
    length: u64,
    version: u8,
) -> GraphStoreResult<FrameRead> {
    if version == JOURNAL_VERSION && offset < length {
        let mut kind = [0u8; 1];
        file.read_exact_at(offset, &mut kind)
            .map_err(storage_failure)?;
        if kind[0] == FrameKind::CompressedChunk as u8 {
            return Ok(FrameRead::Unsupported(
                "V2 compressed frame in a V1 journal".into(),
            ));
        }
    }
    read_frame(file, offset, length)
}

fn read_frame(file: &mut dyn LogFile, offset: u64, length: u64) -> GraphStoreResult<FrameRead> {
    if length.saturating_sub(offset) < FRAME_HEADER_BYTES as u64 {
        return Ok(FrameRead::Incomplete);
    }
    let mut raw = [0_u8; FRAME_HEADER_BYTES as usize];
    file.read_exact_at(offset, &mut raw)
        .map_err(storage_failure)?;
    let Some(kind) = FrameKind::from_byte(raw[0]) else {
        return Ok(FrameRead::Unsupported(format!(
            "unsupported frame kind {} at byte {offset}",
            raw[0]
        )));
    };
    let payload_len = u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]);
    let expected_crc = u32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]]);
    if payload_len > MAX_FRAME_PAYLOAD {
        return Ok(FrameRead::Unsupported(format!(
            "frame at byte {offset} declares {} payload bytes, above the {MAX_FRAME_PAYLOAD}-byte limit",
            payload_len
        )));
    }
    let payload_end = offset + FRAME_HEADER_BYTES + u64::from(payload_len);
    if payload_end > length {
        return Ok(FrameRead::IncompletePayload);
    }
    let mut payload = vec![0_u8; payload_len as usize];
    file.read_exact_at(offset + FRAME_HEADER_BYTES, &mut payload)
        .map_err(storage_failure)?;
    if crc32fast::hash(&payload) != expected_crc {
        let detail = format!("frame payload checksum mismatch at byte {offset}");
        return Ok(match kind {
            FrameKind::Chunk | FrameKind::CompressedChunk => FrameRead::CorruptChunk {
                next_offset: payload_end,
                detail,
            },
            FrameKind::Root => FrameRead::CorruptRoot(detail),
            FrameKind::PressureMarker => FrameRead::CorruptPressureMarker {
                next_offset: payload_end,
            },
        });
    }
    if kind == FrameKind::CompressedChunk {
        if let Err(error) = codec::decode(&payload) {
            return Ok(FrameRead::CorruptChunk {
                next_offset: payload_end,
                detail: error.message,
            });
        }
    }
    if kind == FrameKind::Root {
        let committed = match CommittedRoot::from_bytes(&payload) {
            Ok(committed) => committed,
            Err(error) => return Ok(FrameRead::CorruptRoot(error.message)),
        };
        if let Err(error) = committed.verify_commit(Hash::of(&payload)) {
            return Ok(FrameRead::CorruptRoot(error.message));
        }
    }
    Ok(FrameRead::Valid(Frame { kind, payload }))
}

/// A length that claims bytes beyond EOF loses the only frame boundary, so
/// rescan the bounded bytes after its header for a self-checking commit root.
/// The length is not covered by the frame CRC; only a verifiable root is enough
/// evidence to refuse rather than mistake accepted data for a torn final write.
fn incomplete_frame_is_pressure_marker(
    file: &mut dyn LogFile,
    offset: u64,
    length: u64,
) -> GraphStoreResult<bool> {
    if offset >= length {
        return Ok(false);
    }
    let mut kind = [0_u8; 1];
    file.read_exact_at(offset, &mut kind)
        .map_err(storage_failure)?;
    Ok(kind[0] == FrameKind::PressureMarker as u8)
}

fn suffix_has_verifiable_commit_root(
    file: &mut dyn LogFile,
    offset: u64,
    length: u64,
) -> GraphStoreResult<RootRescan> {
    let suffix_len = length.saturating_sub(offset);
    if suffix_len > u64::from(MAX_FRAME_PAYLOAD) {
        return Ok(RootRescan::BudgetExceeded(format!(
            "damaged-length rescan exceeds the existing {MAX_FRAME_PAYLOAD}-byte frame bound"
        )));
    }
    let mut suffix = Vec::new();
    if suffix.try_reserve_exact(suffix_len as usize).is_err() {
        return Ok(RootRescan::BudgetExceeded(format!(
            "unable to allocate the bounded {suffix_len}-byte damaged-length rescan"
        )));
    }
    suffix.resize(suffix_len as usize, 0);
    if !suffix.is_empty() {
        file.read_exact_at(offset, &mut suffix)
            .map_err(storage_failure)?;
    }

    let header_len = FRAME_HEADER_BYTES as usize;
    if suffix.len() < header_len {
        return Ok(RootRescan::NoVerifiableRoot);
    }

    // The scan span is at most MAX_FRAME_PAYLOAD because the overrun frame's
    // accepted declared length itself is bounded by that existing limit. The
    // aggregate candidate payload work has the same ceiling; there is no second
    // format or rescan-size constant.
    let mut checked_payload_bytes = 0_u64;
    for start in 0..=suffix.len() - header_len {
        if suffix[start] != FrameKind::Root as u8 {
            continue;
        }
        let payload_len = u32::from_le_bytes([
            suffix[start + 1],
            suffix[start + 2],
            suffix[start + 3],
            suffix[start + 4],
        ]);
        if payload_len > MAX_FRAME_PAYLOAD {
            continue;
        }
        let payload_start = start + header_len;
        let payload_end = payload_start + payload_len as usize;
        if payload_end > suffix.len() {
            continue;
        }
        checked_payload_bytes = checked_payload_bytes.saturating_add(u64::from(payload_len));
        if checked_payload_bytes > u64::from(MAX_FRAME_PAYLOAD) {
            return Ok(RootRescan::BudgetExceeded(format!(
                "damaged-length root candidates exceed the existing {MAX_FRAME_PAYLOAD}-byte frame bound"
            )));
        }

        let expected_crc = u32::from_le_bytes([
            suffix[start + 5],
            suffix[start + 6],
            suffix[start + 7],
            suffix[start + 8],
        ]);
        let payload = &suffix[payload_start..payload_end];
        if crc32fast::hash(payload) != expected_crc {
            continue;
        }
        let Ok(root) = CommittedRoot::from_bytes(payload) else {
            continue;
        };
        if root.verify_commit(Hash::of(payload)).is_ok() {
            return Ok(RootRescan::VerifiableRoot);
        }
    }
    Ok(RootRescan::NoVerifiableRoot)
}

/// If a valid later root exists, a preceding bad chunk is accepted-data
/// corruption, not an uncommitted suffix. Root payloads are self-checking.
fn suffix_has_commit_root(
    file: &mut dyn LogFile,
    mut offset: u64,
    length: u64,
    version: u8,
) -> GraphStoreResult<bool> {
    while offset < length {
        match read_frame_in_version(file, offset, length, version)? {
            FrameRead::Valid(frame) => {
                if frame.kind == FrameKind::Root {
                    return Ok(true);
                }
                offset += FRAME_HEADER_BYTES + frame.payload.len() as u64;
            }
            FrameRead::Incomplete => return Ok(false),
            FrameRead::IncompletePayload => {
                return match suffix_has_verifiable_commit_root(
                    file,
                    offset + FRAME_HEADER_BYTES,
                    length,
                )? {
                    RootRescan::VerifiableRoot => Ok(true),
                    RootRescan::NoVerifiableRoot => Ok(false),
                    RootRescan::BudgetExceeded(detail) => {
                        Err(unknown_commit_outcome(offset, detail))
                    }
                };
            }
            FrameRead::CorruptChunk {
                next_offset,
                detail: _,
            }
            | FrameRead::CorruptPressureMarker { next_offset } => offset = next_offset,
            FrameRead::CorruptRoot(detail) => {
                return Err(unknown_commit_outcome(offset, detail));
            }
            FrameRead::Unsupported(detail) => return Err(unsupported_format(detail)),
        }
    }
    Ok(false)
}

/// Appends `frames` in order, optionally behind one durability barrier, and
/// returns each frame's payload offset.
///
/// The checkpoint is the pre-append length: a failed append is rolled back to
/// it and synced, so this process never keeps using a log with a partial frame
/// in it. A rollback that itself fails poisons the handle, because the bytes on
/// the far side can no longer be interpreted by this process.
#[derive(Clone, Copy, Eq, PartialEq)]
enum GrowthReservation {
    StagedUserData,
    Publication,
    PressureMarker,
}

fn append_frames(
    state: &mut ProjectionState,
    frames: &[Frame],
    barrier: bool,
    reservation: GrowthReservation,
) -> GraphStoreResult<Vec<u64>> {
    debug_assert!(!frames.is_empty());
    let checkpoint = match state.file.len() {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let failure = storage_failure(error);
            if reservation != GrowthReservation::PressureMarker {
                observe_adapter_failure(state);
            }
            return Err(failure);
        }
    };
    if let Some(bounds) = state.bounds {
        let allowed_end = match reservation {
            GrowthReservation::StagedUserData => bounds.user_growth_limit_bytes(),
            GrowthReservation::Publication => bounds.publication_limit_bytes(),
            GrowthReservation::PressureMarker => bounds.max_log_bytes,
        };
        let append_len = encoded_frames_len(frames)?;
        let projected_end = checkpoint
            .checked_add(append_len)
            .ok_or_else(size_overflow)?;
        if projected_end > allowed_end {
            if reservation != GrowthReservation::PressureMarker {
                state.bound_refusal_observed = true;
            }
            return Err(GraphStoreError::new(
                "local_storage_pressure_refused",
                format!(
                    "growth refused before append: {projected_end} bytes would exceed the {allowed_end}-byte safe limit; export supported local data or clear/rebuild derived caches, then retry. Retained user data was not changed."
                ),
            ));
        }
    }
    let mut payload_offsets = Vec::with_capacity(frames.len());
    for frame in frames {
        let start = match state.file.len() {
            Ok(start) => start,
            Err(error) => {
                let failure = rollback_or_poison(state, checkpoint, storage_failure(error));
                if reservation != GrowthReservation::PressureMarker {
                    observe_adapter_failure(state);
                }
                return Err(failure);
            }
        };
        if let Err(error) = state.file.append(&frame.encode()) {
            let failure = rollback_or_poison(state, checkpoint, storage_failure(error));
            if reservation != GrowthReservation::PressureMarker {
                observe_adapter_failure(state);
            }
            return Err(failure);
        }
        payload_offsets.push(start + FRAME_HEADER_BYTES);
    }
    if barrier {
        #[cfg(any(test, feature = "test-fault-injection"))]
        if reservation == GrowthReservation::Publication && state.fail_next_publication_flush {
            state.fail_next_publication_flush = false;
            let failure = rollback_or_poison(
                state,
                checkpoint,
                GraphStoreError::new(
                    "chunk_store_browser_storage",
                    "test fault: publication flush was refused",
                ),
            );
            observe_adapter_failure(state);
            return Err(failure);
        }
        if let Err(error) = state.file.sync() {
            let failure = rollback_or_poison(state, checkpoint, storage_failure(error));
            if reservation != GrowthReservation::PressureMarker {
                observe_adapter_failure(state);
            }
            return Err(failure);
        }
    }
    Ok(payload_offsets)
}

/// Record a physical adapter failure in the reserved journal headroom. If the
/// marker cannot be appended and synced, retain `Unknown` for this handle rather
/// than claiming a clean store.
fn observe_adapter_failure(state: &mut ProjectionState) {
    if state.pressure_marker_observed {
        return;
    }
    state.pressure_unknown = true;
    if state.poisoned.is_some() {
        return;
    }
    let marker = Frame {
        kind: FrameKind::PressureMarker,
        payload: Vec::new(),
    };
    if append_frames(state, &[marker], true, GrowthReservation::PressureMarker).is_ok() {
        state.pressure_marker_observed = true;
        state.pressure_unknown = false;
    }
}

fn storage_bounds_invalid() -> GraphStoreError {
    GraphStoreError::new(
        "local_storage_bounds_invalid",
        "the storage bound must leave space for a journal header and bounded commit/recovery headroom",
    )
}

/// Appends one root frame behind one flush, then publishes it in memory.
fn publish_root(state: &mut ProjectionState, committed: &CommittedRoot) -> GraphStoreResult<()> {
    let frame = Frame {
        kind: FrameKind::Root,
        payload: committed.to_bytes()?,
    };
    append_frames(state, &[frame], true, GrowthReservation::Publication)?;

    state.committed = Some(committed.clone());
    Ok(())
}

fn rollback_or_poison(
    state: &mut ProjectionState,
    checkpoint: u64,
    original: GraphStoreError,
) -> GraphStoreError {
    if state.poisoned.is_none() {
        let rollback = state
            .file
            .truncate(checkpoint)
            .and_then(|()| state.file.sync());
        if let Err(error) = rollback {
            state.poisoned = Some(format!(
                "rollback to byte {checkpoint} failed: {}",
                storage_failure(error)
            ));
        }
    }
    match &state.poisoned {
        Some(reason) => GraphStoreError::new(
            "chunk_store_journal_poisoned",
            format!("{reason}; original append failure: {}", original.message),
        ),
        None => original,
    }
}

/// Reads a chunk by address, re-verifying its bytes against that address.
///
/// The index only ever holds verified frames, so an address that is in it was
/// checked when it went in: recovery recomputes every chunk address from the
/// bytes it read, and [`ChunkStore::write_accounted`] checks incoming bytes
/// before it appends them. This re-check on the way out is a second line over
/// the frame CRC, and with this format it should never fire -- a frame whose
/// payload did not hash to its stored address could not have passed recovery.
/// It is here because the cost is one hash on a read path that already copies
/// the bytes, and because handing back bytes that hash to something else as
/// content is the failure this store exists to prevent. It is the discipline
/// [`MemChunkStore`](crate::chunk_store::MemChunkStore) applies on every read.
fn chunk_bytes(state: &mut ProjectionState, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
    chunk_bytes_from_index(&mut *state.file, &state.index, address)
}

fn chunk_bytes_from_index(
    file: &mut dyn LogFile,
    index: &HashMap<Hash, ChunkLocation>,
    address: Hash,
) -> GraphStoreResult<Option<Arc<[u8]>>> {
    let Some(location) = index.get(&address).copied() else {
        return Ok(None);
    };
    let mut buffer = vec![0_u8; location.encoded_len as usize];
    file.read_exact_at(location.payload_offset, &mut buffer)
        .map_err(storage_failure)?;
    if location.compressed && codec::decoded_len(&buffer)? != location.payload_len {
        return Err(GraphStoreError::new(
            "chunk_store_browser_chunk_length_mismatch",
            "compressed length changed after index admission",
        ));
    }
    let bytes = if location.compressed {
        codec::decode(&buffer)?
    } else {
        buffer
    };
    if bytes.len() != location.payload_len as usize {
        return Err(GraphStoreError::new(
            "chunk_store_browser_chunk_length_mismatch",
            "decoded chunk changed length",
        ));
    }
    verify_address(address, &bytes)?;
    Ok(Some(Arc::from(bytes)))
}

/// Loads the committed catalog chunk and checks the graph root it must carry
/// and every named root it points at.
///
/// The same checks `MemChunkStore::catalog_from_state` performs, so a catalog
/// this store accepts is one the memory store would accept.
fn catalog_in_state(
    state: &mut ProjectionState,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    catalog_from_index(&mut *state.file, &state.index, catalog_address)
}

fn catalog_from_index(
    file: &mut dyn LogFile,
    index: &HashMap<Hash, ChunkLocation>,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    let bytes = chunk_bytes_from_index(file, index, catalog_address)?.ok_or_else(|| {
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
    for (name, root) in catalog.roots() {
        if chunk_bytes_from_index(file, index, *root)?.is_none() {
            return Err(named_root_missing(name, *root));
        }
    }
    Ok(catalog)
}

fn named_root_missing(name: &str, address: Hash) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_named_root_missing",
        format!("root catalog entry {name:?} points to missing chunk {address}"),
    )
}

fn ensure_usable(state: &ProjectionState) -> GraphStoreResult<()> {
    match &state.poisoned {
        Some(reason) => Err(GraphStoreError::new(
            "chunk_store_journal_poisoned",
            reason.clone(),
        )),
        None => Ok(()),
    }
}

fn verify_address(expected: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual != expected {
        return Err(GraphStoreError::new(
            "chunk_store_address_mismatch",
            format!("chunk bytes hash to {actual}, expected {expected}"),
        ));
    }
    Ok(())
}

fn ensure_payload_bound(len: usize) -> GraphStoreResult<()> {
    if len > MAX_FRAME_PAYLOAD as usize {
        return Err(GraphStoreError::new(
            "chunk_store_browser_payload_too_large",
            format!("chunk of {len} bytes exceeds the {MAX_FRAME_PAYLOAD}-byte frame bound"),
        ));
    }
    Ok(())
}

fn journal_header(version: u8) -> Vec<u8> {
    let mut header = Vec::with_capacity(JOURNAL_HEADER_BYTES as usize);
    header.extend_from_slice(&JOURNAL_MAGIC);
    header.push(version);
    header
}

impl Frame {
    fn encode(&self) -> Vec<u8> {
        let payload_len = u32::try_from(self.payload.len())
            .expect("frame payloads are bounded by ensure_payload_bound before they are encoded");
        let mut encoded = Vec::with_capacity(FRAME_HEADER_BYTES as usize + self.payload.len());
        encoded.push(self.kind as u8);
        encoded.extend_from_slice(&payload_len.to_le_bytes());
        encoded.extend_from_slice(&crc32fast::hash(&self.payload).to_le_bytes());
        encoded.extend_from_slice(&self.payload);
        encoded
    }
}

/// The seam's failure, named as this store's.
///
/// The browser's refusals arrive here as ordinary storage failures with the
/// browser's own message inside them, and the host's arrive as I/O failures;
/// both keep the detail rather than collapsing to a generic I/O error.
fn storage_failure(error: PhysicalLogError) -> GraphStoreError {
    GraphStoreError::new("chunk_store_browser_storage", format!("{error}"))
}

fn foreign_journal(detail: impl std::fmt::Display) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_browser_foreign_journal",
        format!("the file is not this store's journal: {detail}"),
    )
}

fn unsupported_format(detail: impl std::fmt::Display) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_browser_unsupported_format",
        format!("the browser projection cannot interpret this journal format: {detail}"),
    )
}

fn corrupt_committed_data(offset: u64, detail: impl std::fmt::Display) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_browser_corrupt_committed_data",
        format!("committed journal data is corrupt at or after byte {offset}: {detail}"),
    )
}

fn unknown_commit_outcome(offset: u64, detail: impl std::fmt::Display) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_browser_commit_state_unknown",
        format!("the commit record at byte {offset} is invalid; publication outcome is unknown: {detail}"),
    )
}

fn size_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_size_overflow",
        "chunk write accounting exceeded the canonical limit",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{ROOT_NAME_GRAPH, ROOT_NAME_REDCORE};
    use crate::object_store::{DiskObjectStore, UnifiedChunkStore};
    use std::path::PathBuf;

    fn commit_meta(message: &str) -> CommitMeta {
        CommitMeta {
            parents: Vec::new(),
            author: "rustyred-browser-projection-test".to_string(),
            message: message.to_string(),
            timestamp_unix_ms: 1_700_000_000_000,
            graph_version: 1,
        }
    }

    fn journal_path(directory: &std::path::Path) -> PathBuf {
        directory.join(BROWSER_CHUNK_JOURNAL_FILE)
    }

    fn journal_len(directory: &std::path::Path) -> u64 {
        std::fs::metadata(journal_path(directory))
            .expect("journal metadata")
            .len()
    }

    /// Appends raw bytes to the journal behind an open store's back, so a torn
    /// tail and a torn frame can be produced exactly.
    fn append_raw(directory: &std::path::Path, bytes: &[u8]) {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(journal_path(directory))
            .expect("open journal for raw append");
        file.write_all(bytes).expect("raw append");
        file.sync_all().expect("raw append sync");
    }

    fn overwrite_raw(directory: &std::path::Path, offset: u64, bytes: &[u8]) {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(journal_path(directory))
            .expect("open journal for raw overwrite");
        file.seek(SeekFrom::Start(offset)).expect("seek");
        file.write_all(bytes).expect("raw overwrite");
        file.sync_all().expect("raw overwrite sync");
    }

    fn tempdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("rr-browser-projection-")
            .tempdir()
            .expect("temp dir")
    }

    #[derive(Debug)]
    struct QuotaCeilingLogFile {
        inner: theorem_oplog::NativeLogFile,
        ceiling_bytes: u64,
    }

    impl LogFile for QuotaCeilingLogFile {
        fn len(&self) -> Result<u64, PhysicalLogError> {
            self.inner.len()
        }

        fn read_exact_at(
            &mut self,
            offset: u64,
            buffer: &mut [u8],
        ) -> Result<(), PhysicalLogError> {
            self.inner.read_exact_at(offset, buffer)
        }

        fn append(&mut self, bytes: &[u8]) -> Result<(), PhysicalLogError> {
            let projected = self
                .inner
                .len()?
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| {
                    PhysicalLogError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "injected OPFS quota exceeded: length overflow",
                    ))
                })?;
            if projected > self.ceiling_bytes {
                return Err(PhysicalLogError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "injected OPFS quota exceeded at LogFile::append byte ceiling",
                )));
            }
            self.inner.append(bytes)
        }

        fn truncate(&mut self, length: u64) -> Result<(), PhysicalLogError> {
            self.inner.truncate(length)
        }

        fn sync(&mut self) -> Result<(), PhysicalLogError> {
            self.inner.sync()
        }

        fn sync_directory(&mut self) -> Result<(), PhysicalLogError> {
            self.inner.sync_directory()
        }
    }

    #[test]
    fn lfr_storage_pressure_below_bound_survives_reopen_and_preserves_reads() -> GraphStoreResult<()>
    {
        let directory = tempdir();
        let bounds = ProjectionStorageBounds {
            max_log_bytes: 4096,
            commit_headroom_bytes: 512,
            recovery_headroom_bytes: 128,
        };
        let quota_ceiling = bounds.user_growth_limit_bytes() - 64;
        let log = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.path(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        let store = ProjectionChunkStore::open_with_storage_bounds(
            Box::new(QuotaCeilingLogFile {
                inner: log,
                ceiling_bytes: quota_ceiling,
            }),
            Some(bounds),
        )?;

        let acknowledged = b"acknowledged graph remains readable after quota pressure";
        let acknowledged_root = Hash::of(acknowledged);
        store.write(&[(acknowledged_root, acknowledged)])?;
        let committed = store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(acknowledged_root))]),
            commit_meta("acknowledged before below-bound quota failure"),
        )?;

        let before_fill = store.storage_snapshot()?;
        let marker_frame_bytes = FRAME_HEADER_BYTES;
        let fill_target = quota_ceiling - marker_frame_bytes;
        let fill_len = usize::try_from(fill_target - before_fill.log_bytes - FRAME_HEADER_BYTES)
            .expect("bounded test payload fits usize");
        let filler = vec![0x5a; fill_len];
        let filler_root = Hash::of(&filler);
        store.write(&[(filler_root, &filler)])?;
        assert_eq!(journal_len(directory.path()), fill_target);
        assert!(fill_target < bounds.user_growth_limit_bytes());

        let unfit = b"x";
        let failure = store
            .write(&[(Hash::of(unfit), unfit)])
            .expect_err("the adapter ceiling is lower than the declared growth limit");
        assert_eq!(failure.code, "chunk_store_browser_storage");
        assert!(failure.message.contains("injected OPFS quota exceeded"));
        assert_eq!(store.read_root()?, Some(committed.catalog_root));
        assert_eq!(
            store.get(acknowledged_root)?.as_deref(),
            Some(acknowledged.as_slice())
        );
        assert_eq!(store.get(filler_root)?.as_deref(), Some(filler.as_slice()));
        assert_eq!(
            store.storage_snapshot()?.pressure,
            ProjectionPressure::Observed {
                reason: "persisted_adapter_failure_marker",
            }
        );
        assert_eq!(journal_len(directory.path()), quota_ceiling);
        drop(store);

        let reopened_log = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.path(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        let reopened =
            ProjectionChunkStore::open_with_storage_bounds(Box::new(reopened_log), Some(bounds))?;
        let reopened_snapshot = reopened.storage_snapshot()?;
        assert_eq!(
            reopened_snapshot.pressure,
            ProjectionPressure::Observed {
                reason: "persisted_adapter_failure_marker",
            },
            "real quota failure cannot disappear on reopen"
        );
        assert_eq!(reopened.read_root()?, Some(committed.catalog_root));
        assert_eq!(
            reopened.get(acknowledged_root)?.as_deref(),
            Some(acknowledged.as_slice())
        );
        assert_eq!(
            reopened.get(filler_root)?.as_deref(),
            Some(filler.as_slice())
        );
        Ok(())
    }

    #[test]
    fn a_fresh_journal_opens_with_no_root_and_a_durable_header() -> GraphStoreResult<()> {
        let directory = tempdir();
        let store = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(store.read_root()?, None);
        assert_eq!(store.chunk_count()?, 0);
        assert_eq!(journal_len(directory.path()), JOURNAL_HEADER_BYTES);
        Ok(())
    }

    #[test]
    fn lfr_storage_pressure_projection_bound_refuses_growth_without_changing_reads(
    ) -> GraphStoreResult<()> {
        let directory = tempdir();
        let bounds = ProjectionStorageBounds {
            max_log_bytes: 2048,
            commit_headroom_bytes: 768,
            recovery_headroom_bytes: 128,
        };
        let log = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.path(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        let store = ProjectionChunkStore::open_with_storage_bounds(Box::new(log), Some(bounds))?;
        let payload = vec![
            0x6d;
            (bounds.user_growth_limit_bytes() - JOURNAL_HEADER_BYTES - FRAME_HEADER_BYTES)
                as usize
        ];
        let address = Hash::of(&payload);
        store.write(&[(address, &payload)])?;
        let commit = store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(address))]),
            commit_meta("bounded publication"),
        )?;
        let before_refusal = store.storage_snapshot()?;
        assert_eq!(
            before_refusal.pressure,
            ProjectionPressure::Observed {
                reason: "reserved_headroom_active",
            }
        );
        assert!(before_refusal.log_bytes > bounds.user_growth_limit_bytes());
        assert!(before_refusal.log_bytes <= bounds.publication_limit_bytes());
        let bytes_before_refusal = journal_len(directory.path());

        let extra = b"growth refused before unsafe append";
        let refusal = store
            .write(&[(Hash::of(extra), extra)])
            .expect_err("append-only user data cannot consume reserved headroom");
        assert_eq!(refusal.code, "local_storage_pressure_refused");
        assert_eq!(journal_len(directory.path()), bytes_before_refusal);
        assert_eq!(store.read_root()?, Some(commit.catalog_root));
        assert_eq!(store.get(address)?.as_deref(), Some(payload.as_slice()));
        let after_refusal = store.storage_snapshot()?;
        assert_eq!(
            after_refusal.pressure,
            ProjectionPressure::Observed {
                reason: "safe_growth_refused",
            }
        );
        assert_eq!(after_refusal.log_bytes, bytes_before_refusal);
        Ok(())
    }

    #[test]
    fn a_committed_catalog_survives_reopen_over_a_real_file() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"a graph root standing in for a committed tree".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let catalog_root = {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            let report = store.write_accounted(&[(graph_root, graph_bytes.as_slice())])?;
            assert_eq!(report.chunks_written, 1);
            assert_eq!(report.bytes_written, graph_bytes.len());
            let result = store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
                commit_meta("publish the browser projection's first root"),
            )?;
            assert_eq!(store.read_root()?, Some(result.catalog_root));
            result.catalog_root
        };

        // A second handle on the same directory is what a reload is.
        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.read_root()?, Some(catalog_root));
        assert_eq!(
            reopened.get(graph_root)?.as_deref(),
            Some(graph_bytes.as_slice()),
            "the named root's bytes come back from the file"
        );
        assert_eq!(reopened.chunk_count()?, 2);
        Ok(())
    }

    #[test]
    fn a_second_commit_extends_the_same_log_and_keeps_both_names() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"graph root".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let state_bytes = b"redcore operating state".to_vec();
        let state_root = Hash::of(&state_bytes);

        let store = ProjectionChunkStore::open_at(directory.path())?;
        store.write(&[(graph_root, graph_bytes.as_slice())])?;
        let first = store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
            commit_meta("first"),
        )?;
        store.write(&[(state_root, state_bytes.as_slice())])?;
        let second = store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(state_root))]),
            commit_meta("second"),
        )?;
        assert_ne!(first.catalog_root, second.catalog_root);
        // A reopened native log models a process restart, so release the
        // exclusive writer handle before acquiring the replacement handle.
        drop(store);

        // The second catalog merged the first commit's names rather than
        // replacing them, and its parent is the first commit.
        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.read_root()?, Some(second.catalog_root));
        let mut state = reopened.lock_state()?;
        let catalog = catalog_in_state(&mut state, second.catalog_root)?;
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph_root));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), Some(state_root));
        Ok(())
    }

    #[test]
    fn lfr_storage_corrupted_committed_length_never_erases_later_root() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"committed graph behind a damaged earlier length".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let catalog_root = {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(graph_root, graph_bytes.as_slice())])?;
            store
                .commit_catalog_updates(
                    &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
                    commit_meta("root after the damaged frame"),
                )?
                .catalog_root
        };
        let committed_len = journal_len(directory.path());
        assert!(committed_len > JOURNAL_HEADER_BYTES + FRAME_HEADER_BYTES);

        // payload_len is not covered by payload_crc. Change the first chunk's
        // length to the largest accepted value, so it claims to extend past EOF
        // even though a verified catalog and root follow in the file.
        overwrite_raw(
            directory.path(),
            JOURNAL_HEADER_BYTES + 1,
            &MAX_FRAME_PAYLOAD.to_le_bytes(),
        );
        let corrupted_len = journal_len(directory.path());
        assert_eq!(corrupted_len, committed_len);

        match ProjectionChunkStore::open_at(directory.path()) {
            Ok(reopened) => {
                assert_eq!(
                    reopened.read_root()?,
                    Some(catalog_root),
                    "recovery may not truncate an earlier frame over a later committed root"
                );
                drop(reopened);
            }
            Err(error) => {
                assert_eq!(error.code, "chunk_store_browser_corrupt_committed_data");
            }
        }
        assert_eq!(
            journal_len(directory.path()),
            corrupted_len,
            "either recovery retained the root or named refusal preserved every byte"
        );

        let bytes_before_inspect =
            std::fs::read(journal_path(directory.path())).expect("snapshot damaged journal");
        let mut file = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.path(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        let inspection = ProjectionChunkStore::inspect(&mut file)
            .expect("read-only inspection returns a diagnosis without mutation");
        assert_eq!(inspection.tail, ProjectionTailState::CommittedCorruption);
        assert!(inspection
            .diagnosis
            .as_deref()
            .is_some_and(|detail| detail.contains("later valid commit root")));
        drop(file);
        assert_eq!(
            std::fs::read(journal_path(directory.path())).expect("read inspected journal"),
            bytes_before_inspect,
            "inspection must not mutate the damaged committed log"
        );
        Ok(())
    }

    #[test]
    fn lfr_storage_length_overrun_on_final_root_repairs_only_to_prior_commit(
    ) -> GraphStoreResult<()> {
        let directory = tempdir();
        let prior_graph = b"prior graph before the final root".to_vec();
        let prior_graph_root = Hash::of(&prior_graph);
        let latest_graph = b"new graph behind the torn final root".to_vec();
        let latest_graph_root = Hash::of(&latest_graph);
        let (prior_catalog_root, latest_catalog_root) = {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(prior_graph_root, prior_graph.as_slice())])?;
            let prior = store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(prior_graph_root))]),
                commit_meta("prior committed root"),
            )?;
            store.write(&[(latest_graph_root, latest_graph.as_slice())])?;
            let latest = store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(latest_graph_root))]),
                commit_meta("final root that will be torn"),
            )?;
            (prior.catalog_root, latest.catalog_root)
        };
        assert_ne!(prior_catalog_root, latest_catalog_root);

        let bytes = std::fs::read(journal_path(directory.path())).expect("read committed log");
        let mut offset = JOURNAL_HEADER_BYTES as usize;
        let mut final_root_offset = None;
        while offset + FRAME_HEADER_BYTES as usize <= bytes.len() {
            let payload_len = u32::from_le_bytes([
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
            ]) as usize;
            if bytes[offset] == FrameKind::Root as u8 {
                final_root_offset = Some(offset as u64);
            }
            offset += FRAME_HEADER_BYTES as usize + payload_len;
        }
        assert_eq!(offset, bytes.len(), "fixture consists of complete frames");
        let final_root_offset = final_root_offset.expect("latest root frame exists");
        overwrite_raw(
            directory.path(),
            final_root_offset + 1,
            &MAX_FRAME_PAYLOAD.to_le_bytes(),
        );

        let mut file = theorem_oplog::NativeLogFile::create_or_open_named(
            directory.path(),
            BROWSER_CHUNK_JOURNAL_FILE,
        )
        .map_err(storage_failure)?;
        let inspection = ProjectionChunkStore::inspect(&mut file)?;
        assert_eq!(inspection.tail, ProjectionTailState::Incomplete);
        assert_eq!(
            inspection.committed.as_ref().map(|root| root.root),
            Some(prior_catalog_root),
            "the prior verified root remains the visible state"
        );
        drop(file);

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.read_root()?, Some(prior_catalog_root));
        assert_eq!(
            journal_len(directory.path()),
            inspection.verified_prefix_len,
            "a length-overrun final root with no later root is a repairable tail"
        );
        Ok(())
    }

    #[test]
    fn a_torn_tail_is_truncated_to_the_verified_prefix() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"graph root".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let catalog_root = {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(graph_root, graph_bytes.as_slice())])?;
            store
                .commit_catalog_updates(
                    &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
                    commit_meta("before the tear"),
                )?
                .catalog_root
        };
        let verified_len = journal_len(directory.path());

        // A half-written frame: a complete frame header whose payload never
        // arrived. This is the shape a crash between the length and the bytes
        // leaves behind.
        let mut torn = Frame {
            kind: FrameKind::Root,
            payload: b"a root frame whose payload was cut off".to_vec(),
        }
        .encode();
        torn.truncate(FRAME_HEADER_BYTES as usize + 4);
        append_raw(directory.path(), &torn);

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(
            reopened.read_root()?,
            Some(catalog_root),
            "the previous commit is the visible state"
        );
        assert_eq!(
            journal_len(directory.path()),
            verified_len,
            "recovery truncated the torn tail and synced the repair"
        );

        // The repaired log still appends: a torn tail is not a poisoned store.
        let next_bytes = b"a chunk after the repair".to_vec();
        let next_root = Hash::of(&next_bytes);
        reopened.write(&[(next_root, next_bytes.as_slice())])?;
        drop(reopened);
        let third = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(
            third.get(next_root)?.as_deref(),
            Some(next_bytes.as_slice())
        );
        Ok(())
    }

    #[test]
    fn a_crash_between_the_catalog_append_and_the_root_append_keeps_the_previous_commit(
    ) -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"graph root".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let first_catalog = {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(graph_root, graph_bytes.as_slice())])?;
            store
                .commit_catalog_updates(
                    &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
                    commit_meta("the commit that must survive"),
                )?
                .catalog_root
        };

        // Exactly what the second step of a catalog publication leaves if the
        // process dies between its two appends: a complete, verified catalog
        // chunk frame, and a root frame that never finished arriving.
        let orphan_bytes = b"a catalog chunk staged for a commit that never landed".to_vec();
        let orphan_root = Hash::of(&orphan_bytes);
        append_raw(
            directory.path(),
            &Frame {
                kind: FrameKind::Chunk,
                payload: orphan_bytes.clone(),
            }
            .encode(),
        );
        let mut torn_root = Frame {
            kind: FrameKind::Root,
            payload: CommittedRoot::new(orphan_root, commit_meta("never published"))?.to_bytes()?,
        }
        .encode();
        torn_root.truncate(FRAME_HEADER_BYTES as usize + 6);
        append_raw(directory.path(), &torn_root);

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(
            reopened.read_root()?,
            Some(first_catalog),
            "an unpublished catalog chunk does not become the committed root"
        );
        assert_eq!(
            reopened.get(orphan_root)?.as_deref(),
            Some(orphan_bytes.as_slice()),
            "the staged chunk is a member of the verified prefix, so it is readable"
        );
        Ok(())
    }

    #[test]
    fn lfr_storage_flush_failure_never_acknowledges_save() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"graph whose first publication must fail".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let store = ProjectionChunkStore::open_at(directory.path())?;
        store.write(&[(graph_root, graph_bytes.as_slice())])?;
        store.fail_next_publication_flush_for_test()?;
        let error = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
                commit_meta("injected barrier failure"),
            )
            .expect_err("failed publication flush must not acknowledge save");
        assert_eq!(error.code, "chunk_store_browser_storage");
        drop(store);

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(
            reopened.read_root()?,
            None,
            "reopen proves the failed publication did not become committed"
        );
        Ok(())
    }

    #[test]
    fn lfr_storage_failed_publication_preserves_prior_committed_catalog() -> GraphStoreResult<()> {
        let directory = tempdir();
        let graph_bytes = b"prior accepted graph".to_vec();
        let graph_root = Hash::of(&graph_bytes);
        let state_bytes = b"state staged for a failed publication".to_vec();
        let state_root = Hash::of(&state_bytes);
        let store = ProjectionChunkStore::open_at(directory.path())?;
        store.write(&[(graph_root, graph_bytes.as_slice())])?;
        let first = store.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph_root))]),
            commit_meta("prior accepted catalog"),
        )?;
        store.write(&[(state_root, state_bytes.as_slice())])?;
        store.fail_next_publication_flush_for_test()?;
        let error = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(state_root))]),
                commit_meta("publication rejected at flush"),
            )
            .expect_err("failed barrier cannot replace the earlier catalog");
        assert_eq!(error.code, "chunk_store_browser_storage");
        drop(store);

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.read_root()?, Some(first.catalog_root));
        let mut state = reopened.lock_state()?;
        let catalog = catalog_in_state(&mut state, first.catalog_root)?;
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph_root));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), None);
        Ok(())
    }

    #[test]
    fn corruption_inside_the_prefix_ends_it_rather_than_being_trusted() -> GraphStoreResult<()> {
        let directory = tempdir();
        let first = b"the first chunk".to_vec();
        let second = b"the second chunk".to_vec();
        {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(Hash::of(&first), first.as_slice())])?;
            store.write(&[(Hash::of(&second), second.as_slice())])?;
        }
        // Flip one payload byte of the first chunk in place. Its CRC no longer
        // matches, so the verified prefix ends before it and nothing after it
        // counts either.
        overwrite_raw(
            directory.path(),
            JOURNAL_HEADER_BYTES + FRAME_HEADER_BYTES,
            b"T",
        );

        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.get(Hash::of(&first))?, None);
        assert_eq!(reopened.get(Hash::of(&second))?, None);
        assert_eq!(reopened.chunk_count()?, 0);
        assert_eq!(journal_len(directory.path()), JOURNAL_HEADER_BYTES);
        Ok(())
    }

    #[test]
    fn a_foreign_file_is_refused_by_name() {
        let directory = tempdir();
        std::fs::write(
            journal_path(directory.path()),
            b"not this store's journal at all",
        )
        .expect("write a foreign file");
        let error = ProjectionChunkStore::open_at(directory.path())
            .expect_err("a foreign file must not open");
        assert_eq!(error.code, "chunk_store_browser_foreign_journal");
        assert!(
            error.message.contains("magic"),
            "the refusal names what it found: {}",
            error.message
        );
    }

    #[test]
    fn a_rotted_chunk_frame_is_dropped_by_recovery_rather_than_returned() -> GraphStoreResult<()> {
        let directory = tempdir();
        let bytes = b"a chunk that will rot".to_vec();
        let address = Hash::of(&bytes);
        {
            let store = ProjectionChunkStore::open_at(directory.path())?;
            store.write(&[(address, bytes.as_slice())])?;
            assert_eq!(store.get(address)?.as_deref(), Some(bytes.as_slice()));
        }
        // Rot one payload byte in place, leaving the frame's length and CRC
        // untouched. Recovery reads the frame, the CRC no longer matches, so
        // the prefix ends before it and the repair truncates it away: a caller
        // is told the chunk is absent rather than handed bytes that are not
        // what their address says they are.
        let offset = JOURNAL_HEADER_BYTES + FRAME_HEADER_BYTES;
        overwrite_raw(directory.path(), offset, b"R");
        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert_eq!(reopened.get(address)?, None);
        assert_eq!(reopened.chunk_count()?, 0);
        assert_eq!(journal_len(directory.path()), JOURNAL_HEADER_BYTES);
        Ok(())
    }

    #[test]
    fn reclamation_is_refused_by_name_with_its_reason() -> GraphStoreResult<()> {
        let directory = tempdir();
        let store = ProjectionChunkStore::open_at(directory.path())?;
        let error = store
            .compact_reachable(None, &[])
            .expect_err("the browser refuses reclamation");
        assert_eq!(error.code, "chunk_store_browser_compaction_refused");
        assert!(
            error.message.contains("rename"),
            "the refusal names the missing primitive: {}",
            error.message
        );
        assert!(
            error.message.contains("generation switch"),
            "the refusal names the route that would work: {}",
            error.message
        );
        Ok(())
    }

    #[test]
    fn writing_the_same_chunk_twice_admits_it_once() -> GraphStoreResult<()> {
        let directory = tempdir();
        let bytes = b"deduplicated".to_vec();
        let address = Hash::of(&bytes);
        let store = ProjectionChunkStore::open_at(directory.path())?;
        let first = store.write_accounted(&[(address, bytes.as_slice())])?;
        assert_eq!(first.chunks_written, 1);
        let second = store.write_accounted(&[(address, bytes.as_slice())])?;
        assert_eq!(second.chunks_written, 0);
        assert_eq!(second.bytes_written, 0);
        assert_eq!(store.chunk_count()?, 1);
        Ok(())
    }

    #[test]
    fn the_store_initializes_a_real_object_store_facade() -> GraphStoreResult<()> {
        // The consumption path: `UnifiedChunkStore`'s blanket implementation
        // accepts this store because it implements `ChunkStore` and
        // `ChunkStoreCatalog`, and `DiskObjectStore::from_chunk_store` then
        // initializes a root catalog through `commit_catalog_updates` -- which
        // is the operation this module re-specified -- without either side
        // knowing what is underneath.
        let directory = tempdir();
        let store = Arc::new(ProjectionChunkStore::open_at(directory.path())?);
        let unified: Arc<dyn UnifiedChunkStore> = store.clone();
        assert_eq!(store.read_root()?, None);
        let _facade = DiskObjectStore::from_chunk_store(unified)?;
        assert!(
            store.read_root()?.is_some(),
            "the facade published its empty graph root through this store"
        );

        // Both facade and caller own store references whose NativeLogFile
        // handle carries the exclusive writer lock until dropped.
        drop(_facade);
        drop(store);
        let reopened = ProjectionChunkStore::open_at(directory.path())?;
        assert!(
            reopened.read_root()?.is_some(),
            "and that publication is in the file"
        );
        Ok(())
    }

    #[test]
    fn the_browser_commit_policy_takes_the_immediate_arms_and_refuses_the_grouped_one() {
        assert_eq!(
            browser_journal_commit_policy(RedCoreDurability::AofAlways).expect("aof_always"),
            Some(JournalCommitPolicy::Immediate)
        );
        assert_eq!(
            browser_journal_commit_policy(RedCoreDurability::SnapshotOnly).expect("snapshot_only"),
            Some(JournalCommitPolicy::Immediate)
        );
        assert_eq!(
            browser_journal_commit_policy(RedCoreDurability::None).expect("none"),
            None
        );

        let error = browser_journal_commit_policy(RedCoreDurability::AofEverysec)
            .expect_err("aof_everysec selects the grouped worker and must be refused");
        assert_eq!(error.code, "redcore_browser_durability_refused");
        assert!(
            error.message.contains("Grouped"),
            "the refusal names the policy it refuses: {}",
            error.message
        );
        assert!(
            error.message.contains("std::thread"),
            "the refusal names the worker it would start: {}",
            error.message
        );

        // The hazard is the default, which is why the browser has its own.
        assert_eq!(
            RedCoreOptions::default().durability,
            RedCoreDurability::AofEverysec
        );
        assert_eq!(
            browser_red_core_options().durability,
            BROWSER_DEFAULT_DURABILITY
        );
        assert_ne!(BROWSER_DEFAULT_DURABILITY, RedCoreDurability::AofEverysec);
    }
}

#[cfg(test)]
#[path = "browser_projection_codec_tests.rs"]
mod codec_tests;

/// Exact physical bytes admitted by one preflight over a validated logical
/// archive. Estimates are source/version bound, not an actual write receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProjectionExchangeAdmission {
    pub format_version: u8,
    pub current_log_bytes: u64,
    pub staged_chunk_bytes: u64,
    pub publication_bytes: u64,
    pub projected_log_bytes: u64,
}

impl ProjectionChunkStore {
    /// Preflight every unique archive address against this physical index and
    /// the exact V1/V2 encoder used by write_accounted. No state or pressure
    /// marker is changed. Final publication still needs the atomic root fence.
    pub fn preflight_logical_exchange(
        &self,
        archive: &crate::logical_exchange::ValidatedExchange,
        expected_catalog: Option<Hash>,
    ) -> GraphStoreResult<ProjectionExchangeAdmission> {
        archive.require_empty_destination(self, expected_catalog)?;
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        if state.committed.as_ref().map(|committed| committed.root) != expected_catalog {
            return Err(GraphStoreError::new(
                "chunk_store_catalog_precondition_failed",
                "accepted catalog changed before capacity admission",
            ));
        }
        let mut staged_chunk_bytes = 0_u64;
        let root = archive.receipt().catalog;
        let mut catalog_bytes = None;
        archive.for_each_chunk(&mut |address, bytes| {
            if address == root {
                catalog_bytes = Some(bytes.to_vec());
            }
            if !state.index.contains_key(&address) {
                let frame = prepare_chunk_frame(state.format_version, bytes)?;
                staged_chunk_bytes = staged_chunk_bytes
                    .checked_add(encoded_frames_len(&[frame])?)
                    .ok_or_else(size_overflow)?;
            }
            Ok(())
        })?;
        let catalog_bytes = catalog_bytes.ok_or_else(|| {
            GraphStoreError::new(
                "exchange_catalog_missing",
                "validated archive does not contain its catalog",
            )
        })?;
        // The complete archive stages the catalog itself, so it is present by
        // publication. The shared constructor still handles a missing raw
        // catalog frame for the ordinary writer's un-staged catalog path.
        let (_, publication_frames) = prepare_catalog_frames(
            root,
            &catalog_bytes,
            state.committed.as_ref(),
            archive.publication_metadata(),
            false,
        )?;
        let publication_bytes = encoded_frames_len(&publication_frames)?;
        let current_log_bytes = state.file.len().map_err(storage_failure)?;
        let staged_end = current_log_bytes
            .checked_add(staged_chunk_bytes)
            .ok_or_else(size_overflow)?;
        let projected_log_bytes = staged_end
            .checked_add(publication_bytes)
            .ok_or_else(size_overflow)?;
        if state.bounds.is_some_and(|bounds| {
            staged_end > bounds.user_growth_limit_bytes()
                || projected_log_bytes > bounds.publication_limit_bytes()
        }) {
            return Err(GraphStoreError::new("local_storage_pressure_refused", "complete exchange cannot fit the current journal's admitted staging and publication capacities; original archive preserved"));
        }
        Ok(ProjectionExchangeAdmission {
            format_version: state.format_version,
            current_log_bytes,
            staged_chunk_bytes,
            publication_bytes,
            projected_log_bytes,
        })
    }
}

fn prepare_chunk_frame(format_version: u8, bytes: &[u8]) -> GraphStoreResult<Frame> {
    ensure_payload_bound(bytes.len())?;
    if format_version == COMPRESSED_JOURNAL_VERSION {
        if let Some(payload) = codec::encode(bytes)? {
            return Ok(Frame {
                kind: FrameKind::CompressedChunk,
                payload,
            });
        }
    }
    Ok(Frame {
        kind: FrameKind::Chunk,
        payload: bytes.to_vec(),
    })
}

fn prepare_catalog_frames(
    catalog_root: Hash,
    catalog_bytes: &[u8],
    prior: Option<&CommittedRoot>,
    mut meta: CommitMeta,
    catalog_is_new: bool,
) -> GraphStoreResult<(CommittedRoot, Vec<Frame>)> {
    meta.parents = prior
        .map(|committed| committed.commit)
        .into_iter()
        .collect();
    let committed = CommittedRoot::new(catalog_root, meta)?;
    let mut frames = Vec::new();
    if catalog_is_new {
        frames.push(Frame {
            kind: FrameKind::Chunk,
            payload: catalog_bytes.to_vec(),
        });
    }
    frames.push(Frame {
        kind: FrameKind::Root,
        payload: committed.to_bytes()?,
    });
    Ok((committed, frames))
}

fn encoded_frames_len(frames: &[Frame]) -> GraphStoreResult<u64> {
    frames.iter().try_fold(0_u64, |total, frame| {
        total
            .checked_add(FRAME_HEADER_BYTES)
            .and_then(|total| total.checked_add(frame.payload.len() as u64))
            .ok_or_else(size_overflow)
    })
}
