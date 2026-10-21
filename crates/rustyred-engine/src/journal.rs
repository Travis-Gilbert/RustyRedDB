//! Crash-safe append-only storage for verified content-addressed chunks.
//!
//! Chunk records are intentionally not synced as they are appended. A root
//! record is the durability barrier: it is appended only after its catalog and
//! named roots have been verified, then the journal is synced exactly once.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::chunk_store::{
    compaction_root_changed, merge_catalog_updates, CatalogCommitResult, ChunkStore,
    ChunkStoreCatalog, ChunkStoreCompaction, ChunkWriteReport, CommitMeta, CommittedRoot,
    CompactionReport, RootCatalog,
};
use crate::file_lock::FileExt;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;

const JOURNAL_MAGIC: [u8; 4] = *b"RRCJ";
const JOURNAL_VERSION: u8 = 1;
const RECORD_HEADER_LEN: usize = 4 + 1 + 1 + 32 + 4 + 1;
const RECORD_ENCODED_LEN_LEN: usize = 4;
const RECORD_CRC_LEN: usize = 4;
const RAW_PAYLOAD_THRESHOLD: usize = 256;
const MAX_DECODED_PAYLOAD: usize = 64 * 1024 * 1024;
// Zstd's incompressible-frame overhead is small. This bound leaves room for it
// while preventing a corrupt u32 length from causing an unbounded allocation.
const MAX_ENCODED_PAYLOAD: usize = MAX_DECODED_PAYLOAD + 1024 * 1024;

static JOURNAL_PROCESS_LOCKS: OnceLock<Mutex<BTreeSet<PathBuf>>> = OnceLock::new();
static COMPACTION_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Synchronous durability policy for root commits.
///
/// Both policies preserve the [`ChunkStore::commit`] contract: a successful
/// call returns only after its root is durable. `Grouped` changes only how
/// concurrent callers share that durability barrier.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JournalCommitPolicy {
    /// Append one root and issue its fsync before returning.
    #[default]
    Immediate,
    /// Hold the first concurrent caller for at most `max_wait`, then append all
    /// queued roots in journal order and issue one shared fsync.
    Grouped { max_wait: Duration },
}

#[derive(Clone)]
pub struct JournalChunkStore {
    inner: Arc<JournalInner>,
}

impl std::fmt::Debug for JournalChunkStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JournalChunkStore")
            .field("path", &self.inner.path)
            .finish_non_exhaustive()
    }
}

struct JournalInner {
    path: PathBuf,
    /// `None` on a read-only handle, which takes no exclusive claim so any
    /// number of readers may sit beside the one writer (S30).
    _lock: Option<JournalFileLock>,
    state: Arc<Mutex<JournalState>>,
    syncer: Arc<dyn SyncStrategy>,
    commits: CommitRuntime,
    /// Present when this journal is somebody else's body and that somebody
    /// holds the write claim, which today means an `.RR` file.
    ///
    /// It lives here, inside the `Arc` that `JournalChunkStore::clone` shares,
    /// for the same reason `_lock` does: a handle that can write has to carry
    /// the claim that authorises it, or a clone of the handle outlives the
    /// claim and a second writer walks in behind it. V01 refuted exactly that.
    ///
    /// Declared last so it is dropped last. Releasing the claim before the
    /// group-commit worker has stopped and the file has closed would open the
    /// file to another process while this one is still finishing a flush.
    caller_claim: Option<CallerClaim>,
}

enum CommitRuntime {
    Immediate,
    Grouped(GroupCommitRuntime),
}

struct GroupCommitRuntime {
    shared: Arc<GroupCommitShared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct GroupCommitShared {
    state: Arc<Mutex<JournalState>>,
    syncer: Arc<dyn SyncStrategy>,
    max_wait: Duration,
    queue: Mutex<GroupCommitQueue>,
    ready: Condvar,
    #[cfg(test)]
    stopped: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct GroupCommitQueue {
    requests: VecDeque<CommitRequest>,
    shutdown: bool,
}

struct CommitRequest {
    operation: CommitOperation,
    response: SyncSender<GraphStoreResult<CommitOutcome>>,
}

#[derive(Clone)]
enum CommitOperation {
    Root {
        root: Hash,
        meta: CommitMeta,
    },
    Catalog {
        updates: BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
        expected: Option<Option<Hash>>,
    },
}

#[derive(Clone, Copy)]
struct CommitOutcome {
    catalog_root: Hash,
    commit: Hash,
}

struct JournalState {
    file: File,
    /// First byte of the frame stream. Zero for a bare `redcore.journal`, and
    /// `RR_HEADER_LEN` for an `.RR` file whose header block precedes the
    /// frames. Recovery must never truncate below it or it would eat the
    /// header that identifies the database.
    body_offset: u64,
    /// A read-only handle never repairs a tail and never appends.
    read_only: bool,
    index: HashMap<Hash, RecordLocation>,
    chunk_lengths: HashMap<Hash, u64>,
    committed: Option<CommittedRoot>,
    pending_pins: HashMap<Hash, usize>,
    bound_pending_pins: HashMap<String, JournalBoundPendingPin>,
    claimed_pins: HashMap<Hash, usize>,
    poisoned: Option<String>,
    #[cfg(any(test, feature = "test-fault-injection"))]
    injected_append_failure: Option<InjectedAppendFailure>,
    #[cfg(any(test, feature = "test-fault-injection"))]
    injected_root_failure: Option<InjectedRootFailure>,
    #[cfg(any(test, feature = "test-fault-injection"))]
    injected_sync_failure: Option<(bool, bool)>,
    #[cfg(any(test, feature = "test-fault-injection"))]
    injected_claimed_release_failure: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct JournalBoundPendingPin {
    address: Hash,
    count: usize,
}

#[derive(Clone, Copy)]
struct RecordLocation {
    offset: u64,
}

struct StagedChunkAppend {
    address: Hash,
    logical_len: u64,
    frame: Vec<u8>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Copy)]
struct InjectedRootFailure {
    after_bytes: usize,
    fail_rollback: bool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Copy)]
struct InjectedAppendFailure {
    successful_appends_before_failure: usize,
    after_bytes: usize,
    fail_rollback: bool,
}

struct JournalFileLock {
    file: File,
    process_key: PathBuf,
}

impl Drop for JournalFileLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
        release_process_lock(&self.process_key);
    }
}

/// Where every fsync in this module goes.
///
/// Crate-visible rather than private so `rr_file` can hand the native `.RR`
/// open path a fault-injecting syncer under `#[cfg(test)]`. It is not public:
/// callers outside the crate never choose how the journal syncs.
pub(crate) trait SyncStrategy: Send + Sync {
    fn sync_all(&self, file: &File) -> io::Result<()>;
}

pub(crate) struct SystemSync;

impl SyncStrategy for SystemSync {
    fn sync_all(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum RecordKind {
    Chunk = 1,
    Root = 2,
}

impl RecordKind {
    fn from_byte(raw: u8) -> GraphStoreResult<Self> {
        match raw {
            1 => Ok(Self::Chunk),
            2 => Ok(Self::Root),
            _ => Err(corrupt(format!("unknown journal record kind {raw}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum PayloadEncoding {
    Raw = 0,
    Zstd = 1,
}

impl PayloadEncoding {
    fn from_byte(raw: u8) -> GraphStoreResult<Self> {
        match raw {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Zstd),
            _ => Err(corrupt(format!("unknown journal payload encoding {raw}"))),
        }
    }
}

struct DecodedFrame {
    kind: RecordKind,
    address: Hash,
    payload: Vec<u8>,
    frame_len: u64,
}

enum FrameReadError {
    Partial,
    BadCrc { frame_len: u64 },
    ZeroedHeader,
    Semantic(GraphStoreError),
    Io(GraphStoreError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TailChainProbe {
    UncommittedOnly,
    ContainsLineageRoot,
    Invalid,
}

impl JournalChunkStore {
    /// Open or create a journal at `path`, exclusively locking its sibling
    /// `.lock` file until the last clone of this store is dropped.
    pub fn open(path: impl AsRef<Path>) -> GraphStoreResult<Self> {
        Self::open_with_commit_policy(path, JournalCommitPolicy::Immediate)
    }

    /// Open a journal with an explicit synchronous root-commit policy.
    pub fn open_with_commit_policy(
        path: impl AsRef<Path>,
        policy: JournalCommitPolicy,
    ) -> GraphStoreResult<Self> {
        Self::open_with_policy_and_syncer(path.as_ref(), policy, Arc::new(SystemSync))
    }

    #[cfg(test)]
    fn open_with_syncer(path: &Path, syncer: Arc<dyn SyncStrategy>) -> GraphStoreResult<Self> {
        Self::open_with_policy_and_syncer(path, JournalCommitPolicy::Immediate, syncer)
    }

    fn open_with_policy_and_syncer(
        path: &Path,
        policy: JournalCommitPolicy,
        syncer: Arc<dyn SyncStrategy>,
    ) -> GraphStoreResult<Self> {
        Self::open_writable_at(path, policy, syncer, 0, JournalLockOwner::Journal)
    }

    /// Open a writable journal whose frame stream starts at `body_offset`.
    ///
    /// An `.RR` file passes its header length so the scanner begins after the
    /// identity block; a bare `redcore.journal` passes zero and behaves exactly
    /// as before.
    fn open_writable_at(
        path: &Path,
        policy: JournalCommitPolicy,
        syncer: Arc<dyn SyncStrategy>,
        body_offset: u64,
        lock_owner: JournalLockOwner,
    ) -> GraphStoreResult<Self> {
        let path = canonical_journal_path(path)?;
        let (journal_lock, caller_claim) = match lock_owner {
            JournalLockOwner::Journal => (Some(acquire_journal_lock(&path)?), None),
            JournalLockOwner::Caller(claim) => (None, Some(claim)),
        };
        let newly_created = !path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| journal_io("open journal", error))?;
        if newly_created {
            sync_parent_directory(&path, syncer.as_ref(), "journal creation")?;
        }

        let (index, chunk_lengths, committed) = scan_journal(
            &mut file,
            &path,
            syncer.as_ref(),
            body_offset,
            ScanMode::Repair,
        )?;
        let mut state = JournalState {
            file,
            body_offset,
            read_only: false,
            index,
            chunk_lengths,
            committed,
            pending_pins: HashMap::new(),
            bound_pending_pins: HashMap::new(),
            claimed_pins: HashMap::new(),
            poisoned: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_append_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_root_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_sync_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_claimed_release_failure: false,
        };
        if let Some(committed) = state.committed.clone() {
            validate_catalog_in_state(&mut state, committed.root)?;
        }

        let state = Arc::new(Mutex::new(state));
        let commits = match policy {
            JournalCommitPolicy::Immediate => CommitRuntime::Immediate,
            JournalCommitPolicy::Grouped { max_wait } => CommitRuntime::Grouped(
                GroupCommitRuntime::start(Arc::clone(&state), Arc::clone(&syncer), max_wait)?,
            ),
        };

        Ok(Self {
            inner: Arc::new(JournalInner {
                path,
                _lock: journal_lock,
                state,
                syncer,
                commits,
                caller_claim,
            }),
        })
    }

    /// Open the body of an `.RR` file writable, taking custody of its claim.
    ///
    /// The caller has already taken the single-writer claim and hands it over
    /// rather than keeping it, because this handle is `Clone` and every clone
    /// can write. A claim held beside the handle instead of inside it is
    /// released by whichever copy happens to be dropped first.
    ///
    /// The platform syncer is supplied here so product callers never choose
    /// how the body syncs. The syncer-taking form below exists for the
    /// fault-injection tests that cover the replacement window.
    pub(crate) fn open_rr_body(
        path: &Path,
        policy: JournalCommitPolicy,
        body_offset: u64,
        claim: CallerClaim,
    ) -> GraphStoreResult<Self> {
        Self::open_rr_body_with_syncer(path, policy, body_offset, claim, Arc::new(SystemSync))
    }

    /// [`JournalChunkStore::open_rr_body`] with an explicit syncer.
    ///
    /// Only `rr_file` reaches this, and only from its `#[cfg(test)]` open, so
    /// a test can make one specific fsync of a native `.RR` compaction fail
    /// and observe what the file looks like on the far side.
    pub(crate) fn open_rr_body_with_syncer(
        path: &Path,
        policy: JournalCommitPolicy,
        body_offset: u64,
        claim: CallerClaim,
        syncer: Arc<dyn SyncStrategy>,
    ) -> GraphStoreResult<Self> {
        Self::open_writable_at(
            path,
            policy,
            syncer,
            body_offset,
            JournalLockOwner::Caller(claim),
        )
    }

    /// Open a journal read-only.
    ///
    /// Takes no exclusive claim, never repairs a torn tail, and refuses every
    /// mutation by name. S30 permits read-only secondary access "where safe",
    /// and declining to repair is what makes it safe: a reader that truncated a
    /// tail would be writing to a file another process owns.
    pub(crate) fn open_read_only_at(path: &Path, body_offset: u64) -> GraphStoreResult<Self> {
        let path = path
            .canonicalize()
            .map_err(|error| journal_io("resolve read-only journal path", error))?;
        let syncer: Arc<dyn SyncStrategy> = Arc::new(SystemSync);
        let mut file = OpenOptions::new()
            .read(true)
            .open(&path)
            .map_err(|error| journal_io("open read-only journal", error))?;

        let (index, chunk_lengths, committed) = scan_journal(
            &mut file,
            &path,
            syncer.as_ref(),
            body_offset,
            ScanMode::ReadOnly,
        )?;
        let mut state = JournalState {
            file,
            body_offset,
            read_only: true,
            index,
            chunk_lengths,
            committed,
            pending_pins: HashMap::new(),
            bound_pending_pins: HashMap::new(),
            claimed_pins: HashMap::new(),
            poisoned: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_append_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_root_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_sync_failure: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            injected_claimed_release_failure: false,
        };
        if let Some(committed) = state.committed.clone() {
            validate_catalog_in_state(&mut state, committed.root)?;
        }

        Ok(Self {
            inner: Arc::new(JournalInner {
                path,
                _lock: None,
                state: Arc::new(Mutex::new(state)),
                syncer,
                commits: CommitRuntime::Immediate,
                caller_claim: None,
            }),
        })
    }

    /// Open an existing bare journal without claiming ownership, repairing,
    /// creating directories, or invoking graph recovery/migrations.
    pub fn open_read_only(path: impl AsRef<Path>) -> GraphStoreResult<Self> {
        Self::open_read_only_at(path.as_ref(), 0)
    }

    /// All distinct addressable physical chunks, including unreferenced chunks.
    /// Lengths are decompressed payload lengths; root publication frames are not
    /// chunks. A strict diagnostic rescan refuses an ignored or torn tail; the
    /// ordinary secondary reader retains its committed-prefix behavior.
    pub fn physical_chunks(&self) -> GraphStoreResult<Vec<(Hash, u64)>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let body_offset = state.body_offset;
        let (_, lengths, _) = scan_journal(
            &mut state.file,
            &self.inner.path,
            self.inner.syncer.as_ref(),
            body_offset,
            ScanMode::InventoryCheck,
        )?;
        let mut chunks: Vec<_> = lengths.into_iter().collect();
        chunks.sort_unstable_by_key(|(address, _)| *address);
        Ok(chunks)
    }

    /// Already observed prefix only; callers must report physical completeness
    /// as false when strict enumeration fails. This grants no deletion authority.
    pub(crate) fn observed_physical_chunks(&self) -> GraphStoreResult<Vec<(Hash, u64)>> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        let mut chunks: Vec<_> = state
            .chunk_lengths
            .iter()
            .map(|(address, length)| (*address, *length))
            .collect();
        chunks.sort_unstable_by_key(|(address, _)| *address);
        Ok(chunks)
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Return the effective root-commit policy owned by this journal.
    pub fn commit_policy(&self) -> JournalCommitPolicy {
        match &self.inner.commits {
            CommitRuntime::Immediate => JournalCommitPolicy::Immediate,
            CommitRuntime::Grouped(runtime) => JournalCommitPolicy::Grouped {
                max_wait: runtime.shared.max_wait,
            },
        }
    }

    /// Number of distinct physical chunk records addressable in the journal.
    /// Root frames are excluded because they publish catalog state rather than
    /// store content bytes.
    pub fn chunk_count(&self) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        Ok(state.index.len())
    }

    pub fn committed_root(&self) -> GraphStoreResult<Option<CommittedRoot>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        if let Some(committed) = state.committed.clone() {
            validate_catalog_in_state(&mut state, committed.root)?;
        }
        Ok(state.committed.clone())
    }

    /// Verify that the resident journal is usable and still attached to its
    /// configured filesystem paths.
    pub fn readiness_check(&self) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        if let Some(committed) = state.committed.clone() {
            validate_catalog_in_state(&mut state, committed.root)?;
        }
        verify_open_file_identity(&state.file, &self.inner.path, "journal")?;
        match &self.inner._lock {
            Some(lock) => verify_open_file_identity(&lock.file, &lock.process_key, "journal lock"),
            // A read-only handle holds no lock file, so there is no second
            // identity to reconcile.
            None => Ok(()),
        }
    }

    /// Current process-local in-flight reference count for one address.
    pub fn in_flight_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        let pending =
            journal_pending_pin_count(&state.pending_pins, &state.bound_pending_pins, address)?;
        pending
            .checked_add(state.claimed_pins.get(&address).copied().unwrap_or(0))
            .ok_or_else(journal_pin_count_overflow)
    }

    pub fn pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        journal_pending_pin_count(&state.pending_pins, &state.bound_pending_pins, address)
    }

    pub fn anonymous_pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        Ok(state.pending_pins.get(&address).copied().unwrap_or(0))
    }

    pub fn bound_pending_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        journal_bound_pending_pin_count(&state.bound_pending_pins, address)
    }

    pub fn claimed_pin_count(&self, address: Hash) -> GraphStoreResult<usize> {
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        Ok(state.claimed_pins.get(&address).copied().unwrap_or(0))
    }

    fn lock_state(&self) -> GraphStoreResult<MutexGuard<'_, JournalState>> {
        self.inner.state.lock().map_err(|_| {
            GraphStoreError::new(
                "chunk_store_journal_lock_poisoned",
                "journal state lock is poisoned",
            )
        })
    }

    fn submit_commit(&self, operation: CommitOperation) -> GraphStoreResult<CommitOutcome> {
        match &self.inner.commits {
            CommitRuntime::Immediate => execute_commit_batch(
                &self.inner.state,
                self.inner.syncer.as_ref(),
                vec![operation],
            )
            .into_iter()
            .next()
            .expect("a one-operation commit batch returns one outcome"),
            CommitRuntime::Grouped(runtime) => runtime.submit(operation),
        }
    }

    #[cfg(test)]
    fn inject_append_failure(&self, after_bytes: usize, fail_rollback: bool) {
        self.inject_append_failure_after(0, after_bytes, fail_rollback);
    }

    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn inject_append_failure_after(
        &self,
        successful_appends_before_failure: usize,
        after_bytes: usize,
        fail_rollback: bool,
    ) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.injected_append_failure = Some(InjectedAppendFailure {
            successful_appends_before_failure,
            after_bytes,
            fail_rollback,
        });
    }

    /// Fail an actual root frame append, after all ordinary chunk appends.
    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn inject_root_append_failure_for_test(&self, after_bytes: usize, fail_rollback: bool) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.injected_root_failure = Some(InjectedRootFailure {
            after_bytes,
            fail_rollback,
        });
    }

    /// Fail the real root durability barrier, optionally after syncing its bytes.
    /// A rollback refusal models publication whose outcome is indeterminate.
    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn inject_root_sync_failure_for_test(&self, after_sync: bool, fail_rollback: bool) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.injected_sync_failure = Some((after_sync, fail_rollback));
    }

    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn inject_claimed_release_failure_for_test(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.injected_claimed_release_failure = true;
    }

    #[cfg(any(test, feature = "test-fault-injection"))]
    #[doc(hidden)]
    pub fn claimed_release_failure_pending_for_test(&self) -> bool {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .injected_claimed_release_failure
    }

    #[cfg(test)]
    fn is_poisoned(&self) -> bool {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .poisoned
            .is_some()
    }

    #[cfg(test)]
    fn grouped_worker_stopped(&self) -> bool {
        match &self.inner.commits {
            CommitRuntime::Immediate => true,
            CommitRuntime::Grouped(runtime) => runtime.shared.stopped.load(Ordering::SeqCst),
        }
    }
}

impl Drop for GroupCommitRuntime {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.shared.queue.lock() {
            queue.shutdown = true;
            self.shared.ready.notify_all();
        }
        if let Ok(worker) = self.worker.get_mut() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }
}

impl GroupCommitRuntime {
    fn start(
        state: Arc<Mutex<JournalState>>,
        syncer: Arc<dyn SyncStrategy>,
        max_wait: Duration,
    ) -> GraphStoreResult<Self> {
        let shared = Arc::new(GroupCommitShared {
            state,
            syncer,
            max_wait,
            queue: Mutex::new(GroupCommitQueue::default()),
            ready: Condvar::new(),
            #[cfg(test)]
            stopped: std::sync::atomic::AtomicBool::new(false),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("rustyred-journal-group-commit".to_string())
            .spawn(move || group_commit_worker(worker_shared))
            .map_err(|error| journal_io("start grouped-commit worker", error))?;
        Ok(Self {
            shared,
            worker: Mutex::new(Some(worker)),
        })
    }

    fn submit(&self, operation: CommitOperation) -> GraphStoreResult<CommitOutcome> {
        let (sender, receiver) = mpsc::sync_channel(1);
        {
            let mut queue = self.shared.queue.lock().map_err(|_| {
                GraphStoreError::new(
                    "chunk_store_group_commit_lock_poisoned",
                    "grouped-commit queue lock is poisoned",
                )
            })?;
            if queue.shutdown {
                return Err(GraphStoreError::new(
                    "chunk_store_group_commit_stopped",
                    "grouped-commit worker has stopped",
                ));
            }
            queue.requests.push_back(CommitRequest {
                operation,
                response: sender,
            });
            self.shared.ready.notify_one();
        }
        receiver.recv().map_err(|_| {
            GraphStoreError::new(
                "chunk_store_group_commit_stopped",
                "grouped-commit worker stopped before responding",
            )
        })?
    }
}

fn group_commit_worker(shared: Arc<GroupCommitShared>) {
    loop {
        let requests = {
            let mut queue = shared
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while queue.requests.is_empty() && !queue.shutdown {
                queue = shared
                    .ready
                    .wait(queue)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if queue.shutdown && queue.requests.is_empty() {
                break;
            }

            let deadline = Instant::now() + shared.max_wait;
            loop {
                if queue.shutdown {
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let timeout = deadline.saturating_duration_since(now);
                let (next_queue, wait) = shared
                    .ready
                    .wait_timeout(queue, timeout)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                queue = next_queue;
                if wait.timed_out() {
                    break;
                }
            }
            queue.requests.drain(..).collect::<Vec<_>>()
        };

        let operations = requests
            .iter()
            .map(|request| &request.operation)
            .cloned()
            .collect::<Vec<_>>();
        let outcomes = execute_commit_batch(&shared.state, shared.syncer.as_ref(), operations);
        for (request, outcome) in requests.into_iter().zip(outcomes) {
            let _ = request.response.send(outcome);
        }
    }

    #[cfg(test)]
    shared.stopped.store(true, Ordering::SeqCst);
}

impl ChunkStore for JournalChunkStore {
    fn get_bounded(&self, addr: Hash, max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let Some(location) = state.index.get(&addr).copied() else {
            return Ok(None);
        };
        let bytes = load_chunk_at_bounded(&mut state.file, location.offset, Some(max_bytes))?;
        ensure_address(addr, &bytes)?;
        Ok(Some(bytes))
    }

    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        load_verified_chunk(&mut state, addr)
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.get(addr).map(|bytes| bytes.is_some())
    }

    fn contains_address(&self, addr: Hash) -> GraphStoreResult<bool> {
        // `index` is written only after `ensure_address` proved the chunk hashes
        // to its own address, so a hit is an admission record, not a guess. The
        // bytes are deliberately not read: this exists so the publish path's
        // prune hints stop paying a seek plus a crc32 plus a zstd decode plus a
        // SHA-256 each.
        let state = self.lock_state()?;
        ensure_usable(&state)?;
        Ok(state.index.contains_key(&addr))
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let mut missing = Vec::new();
        let mut seen_missing = HashSet::new();
        for address in addrs {
            if load_verified_chunk(&mut state, *address)?.is_none() && seen_missing.insert(*address)
            {
                missing.push(*address);
            }
        }
        Ok(missing)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.write_accounted(chunks).map(|_| ())
    }

    fn write_accounted(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<ChunkWriteReport> {
        // Validate and own the whole batch before the first append. A malformed
        // later entry must not leave earlier entries looking accepted.
        let mut staged = Vec::with_capacity(chunks.len());
        let mut seen = HashSet::with_capacity(chunks.len());
        for (address, bytes) in chunks {
            ensure_address(*address, bytes)?;
            ensure_decoded_bound(bytes.len())?;
            if seen.insert(*address) {
                staged.push((*address, Arc::<[u8]>::from(*bytes)));
            }
        }

        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        ensure_writable(&state)?;
        let mut report = ChunkWriteReport::default();
        let mut appends = Vec::new();
        for (address, bytes) in staged {
            if load_verified_chunk(&mut state, address)?.is_some() {
                continue;
            }
            let frame = encode_frame(RecordKind::Chunk, address, &bytes)?;
            let logical_len = u64::try_from(bytes.len()).map_err(|_| size_overflow())?;
            report.chunks_written = report
                .chunks_written
                .checked_add(1)
                .ok_or_else(size_overflow)?;
            report.bytes_written = report
                .bytes_written
                .checked_add(bytes.len())
                .ok_or_else(size_overflow)?;
            appends.push(StagedChunkAppend {
                address,
                logical_len,
                frame,
            });
        }

        // A caller may stage several chunks before publishing a root. Treat
        // that admission as one append transaction: a partial later frame must
        // not remain in the middle of a journal which this process can keep
        // using. The address index changes only after every frame is complete.
        let frames = appends
            .iter()
            .map(|append| append.frame.as_slice())
            .collect::<Vec<_>>();
        let offsets = append_frames_with_checkpoint(&mut state, &frames, None)?;
        for (append, offset) in appends.into_iter().zip(offsets) {
            state
                .index
                .insert(append.address, RecordLocation { offset });
            state
                .chunk_lengths
                .insert(append.address, append.logical_len);
        }
        Ok(report)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.submit_commit(CommitOperation::Root { root, meta })
            .map(|outcome| outcome.commit)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        let Some(committed) = state.committed.clone() else {
            return Ok(None);
        };
        validate_catalog_in_state(&mut state, committed.root)?;
        Ok(Some(committed.root))
    }

    fn pin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        increment_journal_pin_count(&mut state.pending_pins, address)?;
        Ok(())
    }

    fn bind_pending_in_flight(&self, address: Hash, publication_key: &str) -> GraphStoreResult<()> {
        validate_journal_publication_key(publication_key)?;
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        if let Some(bound) = state.bound_pending_pins.get(publication_key) {
            if bound.address != address {
                return Err(journal_publication_key_address_conflict(
                    publication_key,
                    bound.address,
                    address,
                ));
            }
        }
        if state.pending_pins.get(&address).copied().unwrap_or(0) == 0 {
            return Err(journal_pending_pin_missing(address, publication_key));
        }
        let incremented_count = state
            .bound_pending_pins
            .get(publication_key)
            .map(|bound| {
                bound
                    .count
                    .checked_add(1)
                    .ok_or_else(journal_pin_count_overflow)
            })
            .transpose()?;
        decrement_journal_pin_count(&mut state.pending_pins, address, "pending")?;
        if let Some(count) = incremented_count {
            state
                .bound_pending_pins
                .get_mut(publication_key)
                .expect("validated bound pending receipt")
                .count = count;
        } else {
            state.bound_pending_pins.insert(
                publication_key.to_string(),
                JournalBoundPendingPin { address, count: 1 },
            );
        }
        Ok(())
    }

    fn release_bound_pending_in_flight(
        &self,
        address: Hash,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        validate_journal_publication_key(publication_key)?;
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        release_journal_bound_pending(&mut state.bound_pending_pins, address, publication_key)
    }

    fn claim_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        increment_journal_pin_count(&mut state.claimed_pins, address)?;
        Ok(())
    }

    fn release_claimed_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        #[cfg(any(test, feature = "test-fault-injection"))]
        if std::mem::take(&mut state.injected_claimed_release_failure) {
            return Err(GraphStoreError::new(
                "chunk_store_injected_claimed_release_failure",
                "injected claimed pin cleanup failure",
            ));
        }
        decrement_journal_pin_count(&mut state.claimed_pins, address, "claimed")?;
        Ok(())
    }

    fn unpin_in_flight(&self, address: Hash) -> GraphStoreResult<()> {
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        decrement_journal_pin_count(&mut state.pending_pins, address, "pending")?;
        Ok(())
    }
}

impl ChunkStoreCatalog for JournalChunkStore {
    fn commit_catalog_updates(
        &self,
        updates: &std::collections::BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        self.submit_commit(CommitOperation::Catalog {
            updates: updates.clone(),
            meta,
            expected: None,
        })
        .map(|outcome| CatalogCommitResult {
            catalog_root: outcome.catalog_root,
            commit: outcome.commit,
        })
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
        self.submit_commit(CommitOperation::Catalog {
            updates: updates.clone(),
            meta,
            expected: Some(expected),
        })
        .map(|outcome| CatalogCommitResult {
            catalog_root: outcome.catalog_root,
            commit: outcome.commit,
        })
    }
}

impl ChunkStoreCompaction for JournalChunkStore {
    fn compact_reachable(
        &self,
        expected_catalog_root: Option<Hash>,
        reachable: &[Hash],
    ) -> GraphStoreResult<CompactionReport> {
        let requested: BTreeSet<Hash> = reachable.iter().copied().collect();
        let mut state = self.lock_state()?;
        ensure_usable(&state)?;
        ensure_writable(&state)?;

        let current_catalog_root = state.committed.as_ref().map(|committed| committed.root);
        if current_catalog_root != expected_catalog_root {
            return Err(compaction_root_changed(
                expected_catalog_root,
                current_catalog_root,
            ));
        }

        for address in &requested {
            if load_verified_chunk(&mut state, *address)?.is_none() {
                return Err(GraphStoreError::new(
                    "chunk_store_compaction_reachable_missing",
                    format!("reachable chunk {address} is missing"),
                ));
            }
        }
        // Pins and compaction share this state mutex. Pins may be installed
        // before their raw chunk exists, so absence is valid for pin-only
        // addresses while the writer is between those two locked operations.
        let mut keep = requested;
        keep.extend(state.pending_pins.keys().copied());
        keep.extend(state.bound_pending_pins.values().map(|bound| bound.address));
        keep.extend(state.claimed_pins.keys().copied());
        if let Some(committed) = state.committed.clone() {
            let catalog = validate_catalog_in_state(&mut state, committed.root)?;
            if !keep.contains(&committed.root) {
                return Err(live_root_omitted(format!(
                    "reachable set omits current root catalog {}",
                    committed.root
                )));
            }
            for (name, address) in catalog.roots() {
                if !keep.contains(address) {
                    return Err(live_root_omitted(format!(
                        "reachable set omits current named root {name:?} at {address}"
                    )));
                }
            }
        }

        let chunks_before = state.index.len();
        let bytes_before = total_logical_bytes(state.chunk_lengths.values().copied())?;
        let temp_path = compaction_temp_path(&self.inner.path);
        let rewrite_result =
            rewrite_compacted_journal(&mut state, &temp_path, &keep, self.inner.syncer.as_ref());
        let (mut replacement, replacement_index, replacement_lengths) = match rewrite_result {
            Ok(replacement) => replacement,
            Err(error) => {
                let _ = fs::remove_file(&temp_path);
                return Err(error);
            }
        };

        // Before the rename, never after, and reported either way. See
        // `CallerWriteClaim`.
        let claim = self.inner.caller_claim.as_ref();
        if let Some(claim) = claim {
            if let Err(error) = claim.adopt_replacement(&replacement) {
                drop(replacement);
                let _ = fs::remove_file(&temp_path);
                return Err(error);
            }
        }

        if let Err(error) = fs::rename(&temp_path, &self.inner.path) {
            if let Some(claim) = claim {
                claim.publish_failed();
            }
            let _ = fs::remove_file(&temp_path);
            return Err(journal_io("replace journal during compaction", error));
        }
        if let Some(claim) = claim {
            claim.publish_committed();
        }
        replacement
            .seek(SeekFrom::End(0))
            .map_err(|error| journal_io("seek compacted journal", error))?;
        state.file = replacement;
        state.index = replacement_index;
        state.chunk_lengths = replacement_lengths;

        sync_parent_directory(
            &self.inner.path,
            self.inner.syncer.as_ref(),
            "journal compaction",
        )?;

        let chunks_after = state.index.len();
        let bytes_after = total_logical_bytes(state.chunk_lengths.values().copied())?;
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

fn rewrite_compacted_journal(
    state: &mut JournalState,
    temp_path: &Path,
    keep: &BTreeSet<Hash>,
    syncer: &dyn SyncStrategy,
) -> GraphStoreResult<(File, HashMap<Hash, RecordLocation>, HashMap<Hash, u64>)> {
    let mut replacement = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(temp_path)
        .map_err(|error| journal_io("create compacted journal", error))?;
    let mut replacement_index = HashMap::new();
    let mut replacement_lengths = HashMap::new();

    let rewrite_result = (|| {
        // The replacement is a fresh file, so anything living before the frame
        // stream has to be copied across first. For an `.RR` file that is the
        // header block, and losing it would leave a valid frame stream that is
        // no longer a database anyone can identify. Copied verbatim from the
        // live file rather than re-encoded, so the digest stays correct without
        // this function knowing the header's layout.
        if state.body_offset > 0 {
            let mut prefix =
                vec![0_u8; usize::try_from(state.body_offset).map_err(|_| size_overflow())?];
            state
                .file
                .seek(SeekFrom::Start(0))
                .map_err(|error| journal_io("seek journal header", error))?;
            state
                .file
                .read_exact(&mut prefix)
                .map_err(|error| journal_io("read journal header", error))?;
            replacement
                .write_all(&prefix)
                .map_err(|error| journal_io("write compacted journal header", error))?;
        }
        let mut ordered = keep.iter().copied().collect::<Vec<_>>();
        ordered.sort_by_key(|address| {
            state
                .index
                .get(address)
                .map(|location| location.offset)
                .unwrap_or(u64::MAX)
        });
        for address in ordered {
            let Some(bytes) = load_verified_chunk(state, address)? else {
                continue;
            };
            let frame = encode_frame(RecordKind::Chunk, address, &bytes)?;
            let offset = replacement
                .seek(SeekFrom::End(0))
                .map_err(|error| journal_io("seek compacted journal", error))?;
            replacement
                .write_all(&frame)
                .map_err(|error| journal_io("write compacted journal", error))?;
            replacement_index.insert(address, RecordLocation { offset });
            replacement_lengths.insert(
                address,
                u64::try_from(bytes.len()).map_err(|_| size_overflow())?,
            );
        }

        if let Some(committed) = &state.committed {
            let payload = committed.to_bytes()?;
            let frame = encode_frame(RecordKind::Root, committed.commit, &payload)?;
            replacement
                .write_all(&frame)
                .map_err(|error| journal_io("write compacted root", error))?;
        }
        syncer
            .sync_all(&replacement)
            .map_err(|error| journal_io("sync compacted journal", error))?;
        Ok(())
    })();

    if let Err(error) = rewrite_result {
        drop(replacement);
        let _ = fs::remove_file(temp_path);
        return Err(error);
    }
    Ok((replacement, replacement_index, replacement_lengths))
}

fn execute_commit_batch(
    state: &Arc<Mutex<JournalState>>,
    syncer: &dyn SyncStrategy,
    operations: Vec<CommitOperation>,
) -> Vec<GraphStoreResult<CommitOutcome>> {
    let mut state = match state.lock() {
        Ok(state) => state,
        Err(_) => {
            return operations
                .into_iter()
                .map(|_| {
                    Err(GraphStoreError::new(
                        "chunk_store_journal_lock_poisoned",
                        "journal state lock is poisoned",
                    ))
                })
                .collect()
        }
    };
    if let Err(error) = ensure_usable(&state) {
        return operations.into_iter().map(|_| Err(error.clone())).collect();
    }
    if let Err(error) = ensure_writable(&state) {
        return operations.into_iter().map(|_| Err(error.clone())).collect();
    }

    let mut staged_committed = state.committed.clone();
    let mut staged_chunks = HashMap::new();
    let mut chunk_appends = Vec::new();
    let mut frames = Vec::with_capacity(operations.len());
    let mut outcomes = Vec::with_capacity(operations.len());
    let mut first_error: Option<GraphStoreError> = None;
    for operation in operations {
        if let Some(error) = first_error.as_ref() {
            outcomes.push(Err(error.clone()));
            continue;
        }
        // Refuse a stale compare-and-swap before entering the operation
        // preparer: no bytes/catalog overlay for this request can be staged.
        // It loses independently while earlier valid requests retain their
        // order. Ordinary preparation/validation/IO errors still abort all.
        if let CommitOperation::Catalog {
            expected: Some(expected),
            ..
        } = &operation
        {
            if *expected != staged_committed.as_ref().map(|committed| committed.root) {
                outcomes.push(Err(GraphStoreError::new(
                    "chunk_store_catalog_precondition_failed",
                    "accepted catalog changed before publication",
                )));
                continue;
            }
        }
        match prepare_commit_operation(
            &mut state,
            &mut staged_chunks,
            &mut chunk_appends,
            staged_committed.as_ref(),
            operation,
        ) {
            Ok((frame, committed)) => {
                staged_committed = Some(committed.clone());
                frames.push((frame, committed.clone()));
                outcomes.push(Ok(CommitOutcome {
                    catalog_root: committed.root,
                    commit: committed.commit,
                }));
            }
            Err(error) => {
                first_error = Some(error.clone());
                outcomes.push(Err(error));
            }
        }
    }

    if let Some(error) = first_error {
        // No root frame is written for a partially prepared grouped batch, so
        // no caller in that batch may observe a successful acknowledgement.
        return outcomes.into_iter().map(|_| Err(error.clone())).collect();
    }

    // Catalog chunks generated by this batch are an overlay until the complete
    // append plus root fsync succeeds. This lets later operations in the group
    // validate an earlier staged catalog without exposing it through the live
    // journal index prematurely.
    let append_frames = chunk_appends
        .iter()
        .map(|append| append.frame.as_slice())
        .chain(frames.iter().map(|(frame, _)| frame.as_slice()))
        .collect::<Vec<_>>();
    let chunk_append_count = chunk_appends.len();
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(injection) = state.injected_root_failure.take() {
        state.injected_append_failure = Some(InjectedAppendFailure {
            successful_appends_before_failure: chunk_append_count,
            after_bytes: injection.after_bytes,
            fail_rollback: injection.fail_rollback,
        });
    }
    let offsets = match append_frames_with_checkpoint(
        &mut state,
        &append_frames,
        Some((syncer, "sync journal root batch")),
    ) {
        Ok(offsets) => offsets,
        Err(error) => {
            return outcomes.into_iter().map(|_| Err(error.clone())).collect();
        }
    };

    for (append, offset) in chunk_appends
        .into_iter()
        .zip(offsets.into_iter().take(chunk_append_count))
    {
        state
            .index
            .insert(append.address, RecordLocation { offset });
        state
            .chunk_lengths
            .insert(append.address, append.logical_len);
    }
    state.committed = staged_committed;
    outcomes
}

#[cfg(test)]
mod exchange_grouped_cas_tests {
    use super::test_support::CountingSync;
    use super::*;
    fn operation(graph: Hash, expected: Option<Option<Hash>>) -> CommitOperation {
        CommitOperation::Catalog {
            updates: BTreeMap::from([("graph".into(), Some(graph))]),
            expected,
            meta: CommitMeta {
                parents: vec![],
                author: "exchange-cas-test".into(),
                message: "actual native grouped publication".into(),
                timestamp_unix_ms: 0,
                graph_version: 0,
            },
        }
    }
    #[test]
    fn logical_exchange_grouped_cas_refusal_does_not_abort_valid_request() {
        let directory = tempfile::tempdir().unwrap();
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(
            &directory.path().join("actual.journal"),
            syncer.clone(),
        )
        .unwrap();
        let graph = Hash::of(b"actual graph");
        store.write(&[(graph, b"actual graph")]).unwrap();
        let syncs = syncer.calls();
        let results = execute_commit_batch(
            &store.inner.state,
            syncer.as_ref(),
            vec![operation(graph, Some(None)), operation(graph, Some(None))],
        );
        assert!(results[0].is_ok());
        assert_eq!(
            results[1].as_ref().err().unwrap().code,
            "chunk_store_catalog_precondition_failed"
        );
        assert_eq!(
            store.read_root().unwrap(),
            Some(results[0].as_ref().unwrap().catalog_root)
        );
        assert_eq!(syncer.calls(), syncs + 1);
    }
    #[test]
    fn logical_exchange_grouped_cas_loser_plus_ordinary_invalid_request_aborts_all() {
        let directory = tempfile::tempdir().unwrap();
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(
            &directory.path().join("actual.journal"),
            syncer.clone(),
        )
        .unwrap();
        let graph = Hash::of(b"actual graph");
        store.write(&[(graph, b"actual graph")]).unwrap();
        let syncs = syncer.calls();
        let results = execute_commit_batch(
            &store.inner.state,
            syncer.as_ref(),
            vec![
                operation(graph, Some(None)),
                operation(graph, Some(None)),
                operation(Hash::of(b"missing"), None),
            ],
        );
        assert!(results.iter().all(Result::is_err));
        assert_eq!(store.read_root().unwrap(), None);
        assert_eq!(
            syncer.calls(),
            syncs,
            "ordinary validation failure never appends a partially prepared group"
        );
    }
    #[test]
    fn logical_exchange_grouped_cas_post_sync_failure_never_acknowledges() {
        let directory = tempfile::tempdir().unwrap();
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(
            &directory.path().join("actual.journal"),
            syncer.clone(),
        )
        .unwrap();
        let graph = Hash::of(b"actual graph");
        store.write(&[(graph, b"actual graph")]).unwrap();
        syncer.fail_after_sync_on_call(syncer.calls() + 1);
        let results = execute_commit_batch(
            &store.inner.state,
            syncer.as_ref(),
            vec![operation(graph, Some(None)), operation(graph, Some(None))],
        );
        assert!(results.iter().all(Result::is_err));
        assert_ne!(
            results[0].as_ref().err().unwrap().code,
            "chunk_store_catalog_precondition_failed",
            "a valid request's failed barrier cannot be reported as a stale precondition"
        );
        assert_eq!(store.read_root().unwrap(), None);
    }
}

fn prepare_commit_operation(
    state: &mut JournalState,
    staged_chunks: &mut HashMap<Hash, Arc<[u8]>>,
    chunk_appends: &mut Vec<StagedChunkAppend>,
    prior: Option<&CommittedRoot>,
    operation: CommitOperation,
) -> GraphStoreResult<(Vec<u8>, CommittedRoot)> {
    let committed = match operation {
        CommitOperation::Root { root, mut meta } => {
            validate_catalog_with_staged(state, staged_chunks, root)?;
            meta.parents = prior.into_iter().map(|root| root.commit).collect();
            CommittedRoot::new(root, meta)?
        }
        CommitOperation::Catalog {
            updates,
            mut meta,
            expected: _,
        } => {
            let current = prior
                .map(|committed| validate_catalog_with_staged(state, staged_chunks, committed.root))
                .transpose()?;
            let catalog = merge_catalog_updates(current, &updates)?;
            catalog.graph_root()?;
            for (name, address) in catalog.roots() {
                if !contains_address_with_staged(state, staged_chunks, *address) {
                    return Err(GraphStoreError::new(
                        "chunk_store_named_root_missing",
                        format!("root catalog entry {name:?} points to missing chunk {address}"),
                    ));
                }
            }
            let catalog_bytes = catalog.to_bytes()?;
            let catalog_root = Hash::of(&catalog_bytes);
            if !contains_address_with_staged(state, staged_chunks, catalog_root) {
                let bytes = Arc::<[u8]>::from(catalog_bytes);
                let frame = encode_frame(RecordKind::Chunk, catalog_root, &bytes)?;
                let logical_len = u64::try_from(bytes.len()).map_err(|_| size_overflow())?;
                staged_chunks.insert(catalog_root, Arc::clone(&bytes));
                chunk_appends.push(StagedChunkAppend {
                    address: catalog_root,
                    logical_len,
                    frame,
                });
            }
            meta.parents = prior.into_iter().map(|root| root.commit).collect();
            CommittedRoot::new(catalog_root, meta)?
        }
    };
    let payload = committed.to_bytes()?;
    let frame = encode_frame(RecordKind::Root, committed.commit, &payload)?;
    Ok((frame, committed))
}

/// Who holds the single-writer claim on a journal file.
///
/// A bare `redcore.journal` locks its own sibling `.lock`. An `.RR` file does
/// not, because D5 and S30 put the single-writer boundary one layer up, where
/// the refusal can name the device that holds it. A second lock file beside the
/// RR one would be redundant and would collide with any sibling sharing the
/// stem, since `Path::with_extension` turns both `Travis.RR` and `Travis.db`
/// into `Travis.lock`.
pub(crate) enum JournalLockOwner {
    /// The journal acquires and holds its own lock.
    Journal,
    /// The caller already holds the claim and hands it over to be held for as
    /// long as any clone of the resulting handle is alive.
    Caller(CallerClaim),
}

/// A write claim taken by whoever owns the file this journal is the body of.
///
/// Opaque on purpose: the journal must not have to know what an `.RR` file is
/// to hold its claim, and `rr_file` already depends on this module, so naming
/// `RrWriteLock` here would point the dependency both ways.
pub(crate) type CallerClaim = Box<dyn CallerWriteClaim>;

/// What the journal has to be able to ask of somebody else's write claim.
///
/// Three calls rather than one because publishing a compacted body is a
/// three-state operation, and a claim keyed on the file has to be told which
/// state it ended in. Compaction publishes by rename, which changes the inode.
/// Adopting after the rename leaves the new inode reachable and unclaimed for
/// the length of a re-acquisition; moving the claim at adoption time leaves the
/// *old* inode, which the path still resolves to, unclaimed for the length of
/// the rename. Neither end is safe alone, so the claim holds both inodes across
/// the rename and lets go of one when it is told which.
pub(crate) trait CallerWriteClaim: Send + Sync {
    /// Take a claim on `replacement` as well, before it is renamed over this
    /// journal's file. Reversible: the file this journal is the body of is
    /// still the old inode until the rename lands.
    fn adopt_replacement(&self, replacement: &File) -> GraphStoreResult<()>;

    /// The rename landed. The superseded inode is unreachable and its claim
    /// can be released.
    fn publish_committed(&self);

    /// The rename did not land. The superseded inode is still the database and
    /// the claim taken on the replacement can be released.
    fn publish_failed(&self);
}

/// Whether a scan may repair the file it is reading.
///
/// Recovery truncates a torn tail, which is a write. A read-only opener must
/// reach the same in-memory view without performing it. Diagnostic inventory
/// additionally requires complete framing; all modes share one scanner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanMode {
    /// Truncate a torn or uncommitted tail and fsync the repair.
    Repair,
    /// Read the same committed state and leave the bytes alone.
    ReadOnly,
    /// Validate every physical frame without repairing or ignoring a suffix.
    InventoryCheck,
}

fn scan_journal(
    file: &mut File,
    path: &Path,
    syncer: &dyn SyncStrategy,
    body_offset: u64,
    mode: ScanMode,
) -> GraphStoreResult<(
    HashMap<Hash, RecordLocation>,
    HashMap<Hash, u64>,
    Option<CommittedRoot>,
)> {
    let file_len = file
        .metadata()
        .map_err(|error| journal_io("stat journal", error))?
        .len();
    let mut offset = body_offset;
    let mut index: HashMap<Hash, RecordLocation> = HashMap::new();
    let mut chunk_lengths = HashMap::new();
    let mut committed: Option<CommittedRoot> = None;
    let mut committed_end = body_offset;
    while offset < file_len {
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| journal_io("seek journal scan", error))?;
        let frame_result = {
            let mut reader = BufReader::new(&mut *file);
            read_frame(&mut reader)
        };
        match frame_result {
            Ok(frame) => {
                let next = offset
                    .checked_add(frame.frame_len)
                    .ok_or_else(size_overflow)?;
                match frame.kind {
                    RecordKind::Chunk => {
                        ensure_address(frame.address, &frame.payload)?;
                        if let Some(existing) = index.get(&frame.address).copied() {
                            let existing_bytes = load_chunk_at(file, existing.offset)?;
                            if existing_bytes.as_ref() != frame.payload.as_slice() {
                                return Err(corrupt(format!(
                                    "duplicate chunk address {} has different payload bytes",
                                    frame.address
                                )));
                            }
                        } else {
                            index.insert(frame.address, RecordLocation { offset });
                            chunk_lengths.insert(
                                frame.address,
                                u64::try_from(frame.payload.len()).map_err(|_| size_overflow())?,
                            );
                        }
                    }
                    RecordKind::Root => {
                        let root = CommittedRoot::from_bytes(&frame.payload)?;
                        root.verify_commit(frame.address)?;
                        validate_catalog_against_index(file, &index, root.root)?;
                        if let Some(previous) = &committed {
                            if root.meta.parents != vec![previous.commit] {
                                return Err(corrupt(format!(
                                    "root commit {} does not name immediately previous root {} as its sole parent",
                                    root.commit, previous.commit
                                )));
                            }
                        } else if root.meta.parents.len() > 1 {
                            return Err(corrupt(format!(
                                "first journal root {} names {} parents; an uncompacted root or compacted checkpoint may name at most one",
                                root.commit,
                                root.meta.parents.len()
                            )));
                        }
                        // Compaction retains the latest self-verifying root
                        // record and may elide its immediately preceding root
                        // frame. In that case the first retained record keeps
                        // one externalized parent as its checkpoint lineage.
                        // Every root that follows is still required above to
                        // name the immediately previous in-file commit.
                        committed = Some(root);
                        committed_end = next;
                    }
                }
                offset = next;
            }
            Err(FrameReadError::Partial) => {
                if mode == ScanMode::InventoryCheck {
                    return Err(GraphStoreError::new(
                        "chunk_store_inventory_torn_tail",
                        format!("physical frame at offset {offset} is incomplete"),
                    ));
                }
                if mode == ScanMode::Repair {
                    file.set_len(offset)
                        .map_err(|error| journal_io("truncate torn journal tail", error))?;
                    syncer
                        .sync_all(file)
                        .map_err(|error| journal_io("sync recovered journal tail", error))?;
                    sync_parent_directory(path, syncer, "journal tail recovery")?;
                }
                break;
            }
            Err(FrameReadError::BadCrc { frame_len }) => {
                if mode == ScanMode::InventoryCheck {
                    return Err(GraphStoreError::new(
                        "chunk_store_inventory_bad_crc_tail",
                        format!("physical frame at offset {offset} has a bad CRC"),
                    ));
                }
                let frame_end = offset.checked_add(frame_len).ok_or_else(size_overflow)?;
                if frame_end > file_len {
                    return Err(corrupt(format!(
                        "bad-CRC journal record at offset {offset} extends past the file tail"
                    )));
                }
                let tail = probe_tail_chain(file, frame_end, file_len, committed.as_ref())?;
                if tail != TailChainProbe::UncommittedOnly {
                    return Err(corrupt(format!(
                        "journal record at offset {offset} has a bad CRC before a non-recoverable suffix"
                    )));
                }
                if mode == ScanMode::Repair {
                    file.set_len(committed_end)
                        .map_err(|error| journal_io("truncate bad-CRC journal tail", error))?;
                    syncer.sync_all(file).map_err(|error| {
                        journal_io("sync recovered bad-CRC journal tail", error)
                    })?;
                    sync_parent_directory(path, syncer, "bad-CRC journal tail recovery")?;
                }
                retain_index_before(&mut index, &mut chunk_lengths, committed_end);
                break;
            }
            Err(FrameReadError::ZeroedHeader) => {
                if mode == ScanMode::InventoryCheck {
                    return Err(GraphStoreError::new(
                        "chunk_store_inventory_zero_hole_tail",
                        format!("physical frame at offset {offset} has a zeroed header"),
                    ));
                }
                let Some(suffix_start) = zero_hole_suffix_start(file, offset, file_len)? else {
                    return Err(corrupt(format!(
                        "zeroed journal record at offset {offset} is not followed by a canonical frame boundary"
                    )));
                };
                let tail = probe_tail_chain(file, suffix_start, file_len, committed.as_ref())?;
                if tail != TailChainProbe::UncommittedOnly {
                    return Err(corrupt(format!(
                        "zeroed journal record at offset {offset} precedes a non-recoverable suffix"
                    )));
                }
                if mode == ScanMode::Repair {
                    file.set_len(committed_end)
                        .map_err(|error| journal_io("truncate zero-hole journal tail", error))?;
                    syncer.sync_all(file).map_err(|error| {
                        journal_io("sync recovered zero-hole journal tail", error)
                    })?;
                    sync_parent_directory(path, syncer, "zero-hole journal tail recovery")?;
                }
                retain_index_before(&mut index, &mut chunk_lengths, committed_end);
                break;
            }
            Err(FrameReadError::Semantic(error) | FrameReadError::Io(error)) => return Err(error),
        }
    }
    file.seek(SeekFrom::End(0))
        .map_err(|error| journal_io("seek journal end", error))?;
    Ok((index, chunk_lengths, committed))
}

fn retain_index_before(
    index: &mut HashMap<Hash, RecordLocation>,
    chunk_lengths: &mut HashMap<Hash, u64>,
    boundary: u64,
) {
    index.retain(|address, location| {
        let keep = location.offset < boundary;
        if !keep {
            chunk_lengths.remove(address);
        }
        keep
    });
}

/// Locate the first canonical frame boundary after a zero-filled persistence
/// hole. The bytes between `offset` and the boundary must be entirely zero, so
/// arbitrary magic embedded in a damaged payload cannot be mistaken for an
/// independently framed suffix. An all-zero physical tail has no suffix and is
/// represented by `file_len`.
fn zero_hole_suffix_start(
    file: &mut File,
    offset: u64,
    file_len: u64,
) -> GraphStoreResult<Option<u64>> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| journal_io("seek zero-hole journal tail", error))?;
    let mut reader = BufReader::new(&mut *file);
    let mut position = offset;
    let mut byte = [0_u8; 1];
    while position < file_len {
        reader
            .read_exact(&mut byte)
            .map_err(|error| journal_io("read zero-hole journal tail", error))?;
        if byte[0] != 0 {
            if byte[0] != JOURNAL_MAGIC[0] {
                return Ok(None);
            }
            let mut magic_tail = [0_u8; JOURNAL_MAGIC.len() - 1];
            if reader.read_exact(&mut magic_tail).is_err() || magic_tail != JOURNAL_MAGIC[1..] {
                return Ok(None);
            }
            return Ok(Some(position));
        }
        position = position.checked_add(1).ok_or_else(size_overflow)?;
    }
    Ok(Some(file_len))
}

/// Classify a suffix beginning at a proven frame boundary. A suffix containing
/// a self-verifying root whose parent is the last committed root is publication
/// evidence and must never be truncated, even when intact orphan chunks follow
/// it. A chain containing only chunks (plus an optional torn final frame) is an
/// uncommitted tail and may be discarded after power loss.
fn probe_tail_chain(
    file: &mut File,
    start: u64,
    file_len: u64,
    committed: Option<&CommittedRoot>,
) -> GraphStoreResult<TailChainProbe> {
    if start > file_len {
        return Ok(TailChainProbe::Invalid);
    }
    let mut cursor = start;
    let mut expected_parent = committed.map(|root| root.commit);
    let mut saw_lineage_root = false;
    while cursor < file_len {
        file.seek(SeekFrom::Start(cursor))
            .map_err(|error| journal_io("seek journal tail probe", error))?;
        let frame_result = {
            let mut reader = BufReader::new(&mut *file);
            read_frame(&mut reader)
        };
        match frame_result {
            Ok(frame) => {
                let next = cursor
                    .checked_add(frame.frame_len)
                    .ok_or_else(size_overflow)?;
                if next > file_len {
                    return Ok(TailChainProbe::Invalid);
                }
                match frame.kind {
                    RecordKind::Chunk => {
                        if ensure_address(frame.address, &frame.payload).is_err() {
                            return Ok(TailChainProbe::Invalid);
                        }
                    }
                    RecordKind::Root => {
                        let Ok(root) = CommittedRoot::from_bytes(&frame.payload) else {
                            return Ok(TailChainProbe::Invalid);
                        };
                        if root.verify_commit(frame.address).is_err() {
                            return Ok(TailChainProbe::Invalid);
                        }
                        let lineage_matches = match expected_parent {
                            Some(parent) => root.meta.parents == vec![parent],
                            None => root.meta.parents.len() <= 1,
                        };
                        if !lineage_matches {
                            return Ok(TailChainProbe::Invalid);
                        }
                        expected_parent = Some(root.commit);
                        saw_lineage_root = true;
                    }
                }
                cursor = next;
            }
            Err(FrameReadError::Partial) => break,
            Err(FrameReadError::BadCrc { frame_len }) => {
                let Some(frame_end) = cursor.checked_add(frame_len) else {
                    return Ok(TailChainProbe::Invalid);
                };
                if frame_end != file_len {
                    return Ok(TailChainProbe::Invalid);
                }
                break;
            }
            Err(FrameReadError::ZeroedHeader | FrameReadError::Semantic(_)) => {
                return Ok(TailChainProbe::Invalid);
            }
            Err(FrameReadError::Io(error)) => return Err(error),
        }
    }
    Ok(if saw_lineage_root {
        TailChainProbe::ContainsLineageRoot
    } else {
        TailChainProbe::UncommittedOnly
    })
}

fn load_verified_chunk(
    state: &mut JournalState,
    address: Hash,
) -> GraphStoreResult<Option<Arc<[u8]>>> {
    let Some(location) = state.index.get(&address).copied() else {
        return Ok(None);
    };
    let bytes = load_chunk_at(&mut state.file, location.offset)?;
    ensure_address(address, &bytes)?;
    Ok(Some(bytes))
}

fn load_verified_chunk_with_staged(
    state: &mut JournalState,
    staged: &HashMap<Hash, Arc<[u8]>>,
    address: Hash,
) -> GraphStoreResult<Option<Arc<[u8]>>> {
    if let Some(bytes) = staged.get(&address) {
        ensure_address(address, bytes)?;
        return Ok(Some(Arc::clone(bytes)));
    }
    load_verified_chunk(state, address)
}

/// Existence probe for a chunk that is either staged in this commit or already
/// admitted to the journal index, without reading or re-hashing it.
///
/// Both provenances are already verified: a staged entry passed `ensure_address`
/// when it was staged, and an index entry passed `ensure_address` when it was
/// admitted ([`load_verified_chunk`] is the only writer of `state.index`). So a
/// hit here is an admission record, not a guess, and answering it costs a hash
/// lookup instead of a seek, a crc32, a zstd decode and a SHA-256.
///
/// The commit path uses this for its named-root and catalog existence checks,
/// which need reachability — "is this chunk present?" — not byte integrity of
/// bytes this same commit or an earlier one already verified. Integrity of
/// everything reachable from a durable head is re-established at every boundary
/// that reads one: open/recovery (`scan_journal`), compaction, `verify_tree_parity`,
/// and every ordinary `get`.
fn contains_address_with_staged(
    state: &JournalState,
    staged: &HashMap<Hash, Arc<[u8]>>,
    address: Hash,
) -> bool {
    staged.contains_key(&address) || state.index.contains_key(&address)
}

fn load_chunk_at(file: &mut File, offset: u64) -> GraphStoreResult<Arc<[u8]>> {
    load_chunk_at_bounded(file, offset, None)
}
fn load_chunk_at_bounded(
    file: &mut File,
    offset: u64,
    max_bytes: Option<usize>,
) -> GraphStoreResult<Arc<[u8]>> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| journal_io("seek chunk", error))?;
    let mut reader = BufReader::new(file);
    let frame = read_frame_bounded(&mut reader, max_bytes).map_err(|error| match error {
        FrameReadError::Partial | FrameReadError::BadCrc { .. } | FrameReadError::ZeroedHeader => {
            corrupt(format!("stored chunk frame at offset {offset} is torn"))
        }
        FrameReadError::Semantic(error) | FrameReadError::Io(error) => error,
    })?;
    if frame.kind != RecordKind::Chunk {
        return Err(corrupt(format!(
            "index offset {offset} points to a root record"
        )));
    }
    Ok(Arc::from(frame.payload))
}

fn validate_catalog_in_state(
    state: &mut JournalState,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    let bytes = load_verified_chunk(state, catalog_address)?.ok_or_else(|| {
        GraphStoreError::new(
            "chunk_store_catalog_missing",
            format!("root catalog {catalog_address} is missing"),
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
    for (name, address) in catalog.roots() {
        if load_verified_chunk(state, *address)?.is_none() {
            return Err(GraphStoreError::new(
                "chunk_store_named_root_missing",
                format!("root catalog entry {name:?} points to missing chunk {address}"),
            ));
        }
    }
    Ok(catalog)
}

fn validate_catalog_with_staged(
    state: &mut JournalState,
    staged: &HashMap<Hash, Arc<[u8]>>,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    let bytes =
        load_verified_chunk_with_staged(state, staged, catalog_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "chunk_store_catalog_missing",
                format!("root catalog {catalog_address} is missing"),
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
    for (name, address) in catalog.roots() {
        if !contains_address_with_staged(state, staged, *address) {
            return Err(GraphStoreError::new(
                "chunk_store_named_root_missing",
                format!("root catalog entry {name:?} points to missing chunk {address}"),
            ));
        }
    }
    Ok(catalog)
}

fn validate_catalog_against_index(
    file: &mut File,
    index: &HashMap<Hash, RecordLocation>,
    catalog_address: Hash,
) -> GraphStoreResult<RootCatalog> {
    let location = index.get(&catalog_address).copied().ok_or_else(|| {
        GraphStoreError::new(
            "chunk_store_catalog_missing",
            format!("root catalog {catalog_address} is missing"),
        )
    })?;
    let bytes = load_chunk_at(file, location.offset)?;
    ensure_address(catalog_address, &bytes)?;
    let catalog = RootCatalog::from_bytes(&bytes)?;
    if catalog.address()? != catalog_address {
        return Err(GraphStoreError::new(
            "chunk_store_catalog_address_mismatch",
            format!("root catalog does not match address {catalog_address}"),
        ));
    }
    catalog.graph_root()?;
    for (name, address) in catalog.roots() {
        let location = index.get(address).copied().ok_or_else(|| {
            GraphStoreError::new(
                "chunk_store_named_root_missing",
                format!("root catalog entry {name:?} points to missing chunk {address}"),
            )
        })?;
        let bytes = load_chunk_at(file, location.offset)?;
        ensure_address(*address, &bytes)?;
    }
    Ok(catalog)
}

fn encode_frame(kind: RecordKind, address: Hash, payload: &[u8]) -> GraphStoreResult<Vec<u8>> {
    ensure_decoded_bound(payload.len())?;
    let (encoding, encoded) = encode_payload(payload)?;
    let decoded_len = u32::try_from(payload.len()).map_err(|_| size_overflow())?;
    let encoded_len = u32::try_from(encoded.len()).map_err(|_| size_overflow())?;
    let mut frame = Vec::with_capacity(
        RECORD_HEADER_LEN
            .checked_add(RECORD_ENCODED_LEN_LEN)
            .and_then(|length| length.checked_add(encoded.len()))
            .and_then(|length| length.checked_add(RECORD_CRC_LEN))
            .ok_or_else(size_overflow)?,
    );
    frame.extend_from_slice(&JOURNAL_MAGIC);
    frame.push(JOURNAL_VERSION);
    frame.push(kind as u8);
    frame.extend_from_slice(address.as_bytes());
    frame.extend_from_slice(&decoded_len.to_le_bytes());
    frame.push(encoding as u8);
    frame.extend_from_slice(&encoded_len.to_le_bytes());
    frame.extend_from_slice(&encoded);
    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_le_bytes());
    Ok(frame)
}

fn read_frame(reader: &mut impl Read) -> Result<DecodedFrame, FrameReadError> {
    read_frame_bounded(reader, None)
}
fn read_frame_bounded(
    reader: &mut impl Read,
    max_bytes: Option<usize>,
) -> Result<DecodedFrame, FrameReadError> {
    let mut fixed = [0_u8; RECORD_HEADER_LEN];
    read_exact_classified(reader, &mut fixed)?;
    if fixed[..4] != JOURNAL_MAGIC {
        if fixed.iter().all(|byte| *byte == 0) {
            return Err(FrameReadError::ZeroedHeader);
        }
        return Err(FrameReadError::Semantic(corrupt(
            "journal record magic mismatch".to_string(),
        )));
    }
    if fixed[4] != JOURNAL_VERSION {
        return Err(FrameReadError::Semantic(corrupt(format!(
            "unsupported journal version {}",
            fixed[4]
        ))));
    }
    let kind = RecordKind::from_byte(fixed[5]).map_err(FrameReadError::Semantic)?;
    let mut raw_address = [0_u8; 32];
    raw_address.copy_from_slice(&fixed[6..38]);
    let address = Hash::from_bytes(raw_address);
    let decoded_len = u32::from_le_bytes(
        fixed[38..42]
            .try_into()
            .expect("fixed decoded length range"),
    ) as usize;
    let encoding = PayloadEncoding::from_byte(fixed[42]).map_err(FrameReadError::Semantic)?;
    let mut encoded_len_bytes = [0_u8; 4];
    read_exact_classified(reader, &mut encoded_len_bytes)?;
    let encoded_len = u32::from_le_bytes(encoded_len_bytes) as usize;
    ensure_decoded_bound(decoded_len).map_err(FrameReadError::Semantic)?;
    ensure_encoded_bound(encoded_len).map_err(FrameReadError::Semantic)?;
    match encoding {
        PayloadEncoding::Raw if encoded_len != decoded_len => {
            return Err(FrameReadError::Semantic(corrupt(format!(
                "raw journal payload declares decoded length {decoded_len} but encoded length {encoded_len}"
            ))));
        }
        PayloadEncoding::Zstd if encoded_len >= decoded_len => {
            return Err(FrameReadError::Semantic(corrupt(format!(
                "zstd journal payload length {encoded_len} is not smaller than decoded length {decoded_len}"
            ))));
        }
        PayloadEncoding::Raw | PayloadEncoding::Zstd => {}
    }
    if max_bytes.is_some_and(|bound| decoded_len > bound || encoded_len > bound) {
        return Err(FrameReadError::Semantic(GraphStoreError::new(
            "chunk_read_bound_refused",
            "chunk frame exceeds admitted read bytes",
        )));
    }
    let mut encoded = Vec::with_capacity(encoded_len);
    (&mut *reader)
        .take(u64::try_from(encoded_len).map_err(|_| FrameReadError::Semantic(size_overflow()))?)
        .read_to_end(&mut encoded)
        .map_err(|error| FrameReadError::Io(journal_io("read journal payload", error)))?;
    if encoding == PayloadEncoding::Zstd {
        // The zstd frame carries its own end marker. Bind the journal's outer
        // encoded_len to that independently derivable size so a corrupted
        // length cannot swallow later Root frames while still satisfying the
        // weaker encoded_len < decoded_len relation.
        if let Ok(frame_size) = zstd::zstd_safe::find_frame_compressed_size(&encoded) {
            if frame_size != encoded_len {
                return Err(FrameReadError::Semantic(corrupt(format!(
                    "zstd frame occupies {frame_size} bytes but journal header declares {encoded_len}"
                ))));
            }
        }
    }
    if encoded.len() != encoded_len {
        return Err(FrameReadError::Partial);
    }
    let mut crc_bytes = [0_u8; RECORD_CRC_LEN];
    read_exact_classified(reader, &mut crc_bytes)?;
    let expected_crc = u32::from_le_bytes(crc_bytes);
    let mut crc_input = fixed.to_vec();
    crc_input.extend_from_slice(&encoded_len_bytes);
    crc_input.extend_from_slice(&encoded);
    let frame_len = u64::try_from(
        RECORD_HEADER_LEN
            .checked_add(RECORD_ENCODED_LEN_LEN)
            .and_then(|length| length.checked_add(encoded_len))
            .and_then(|length| length.checked_add(RECORD_CRC_LEN))
            .ok_or_else(size_overflow)
            .map_err(FrameReadError::Semantic)?,
    )
    .map_err(|_| FrameReadError::Semantic(size_overflow()))?;
    if crc32fast::hash(&crc_input) != expected_crc {
        return Err(FrameReadError::BadCrc { frame_len });
    }
    let payload =
        decode_payload(encoding, &encoded, decoded_len).map_err(FrameReadError::Semantic)?;
    Ok(DecodedFrame {
        kind,
        address,
        payload,
        frame_len,
    })
}

fn encode_payload(payload: &[u8]) -> GraphStoreResult<(PayloadEncoding, Vec<u8>)> {
    if payload.len() < RAW_PAYLOAD_THRESHOLD {
        return Ok((PayloadEncoding::Raw, payload.to_vec()));
    }
    let compressed = zstd::stream::encode_all(payload, 3)
        .map_err(|error| journal_io("compress journal payload", error))?;
    if compressed.len() >= payload.len() {
        Ok((PayloadEncoding::Raw, payload.to_vec()))
    } else {
        Ok((PayloadEncoding::Zstd, compressed))
    }
}

fn decode_payload(
    encoding: PayloadEncoding,
    encoded: &[u8],
    decoded_len: usize,
) -> GraphStoreResult<Vec<u8>> {
    let payload = match encoding {
        PayloadEncoding::Raw => {
            if encoded.len() != decoded_len {
                return Err(corrupt(format!(
                    "raw payload length {} does not match declared decoded length {decoded_len}",
                    encoded.len()
                )));
            }
            encoded.to_vec()
        }
        PayloadEncoding::Zstd => {
            let decoder = zstd::stream::read::Decoder::new(encoded)
                .map_err(|error| corrupt(format!("invalid zstd payload: {error}")))?;
            let take_limit = u64::try_from(decoded_len)
                .map_err(|_| size_overflow())?
                .checked_add(1)
                .ok_or_else(size_overflow)?;
            let mut payload = Vec::with_capacity(decoded_len);
            decoder
                .take(take_limit)
                .read_to_end(&mut payload)
                .map_err(|error| corrupt(format!("invalid zstd payload: {error}")))?;
            if payload.len() != decoded_len {
                return Err(corrupt(format!(
                    "decoded payload length {} does not match declared length {decoded_len}",
                    payload.len()
                )));
            }
            payload
        }
    };
    Ok(payload)
}

fn read_exact_classified(reader: &mut impl Read, buffer: &mut [u8]) -> Result<(), FrameReadError> {
    match reader.read_exact(buffer) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Err(FrameReadError::Partial),
        Err(error) => Err(FrameReadError::Io(journal_io("read journal", error))),
    }
}

/// Append a complete logical operation behind one rollback checkpoint.
///
/// The caller deliberately receives offsets only after every frame (and, for a
/// root publication, the durability sync) succeeds. It can therefore update
/// address indexes and the committed root without ever exposing a torn middle
/// frame to subsequent operations in this process.
fn append_frames_with_checkpoint(
    state: &mut JournalState,
    frames: &[&[u8]],
    sync: Option<(&dyn SyncStrategy, &str)>,
) -> GraphStoreResult<Vec<u64>> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    let checkpoint = state
        .file
        .seek(SeekFrom::End(0))
        .map_err(|error| journal_io("seek before journal append transaction", error))?;
    let mut offsets = Vec::with_capacity(frames.len());
    for frame in frames {
        match append_frame(state, frame) {
            Ok(offset) => offsets.push(offset),
            Err(error) => {
                let rollback_sync = sync.map(|(syncer, _)| syncer);
                return Err(rollback_or_poison(state, checkpoint, error, rollback_sync));
            }
        }
    }
    if let Some((syncer, action)) = sync {
        #[cfg(any(test, feature = "test-fault-injection"))]
        let sync_result =
            if let Some((after_sync, fail_rollback)) = state.injected_sync_failure.take() {
                let result = if after_sync {
                    syncer.sync_all(&state.file)
                } else {
                    Ok(())
                };
                if fail_rollback {
                    state.poisoned = Some("injected durability rollback refusal".to_string());
                }
                result.and(Err(io::Error::other("injected root sync failure")))
            } else {
                syncer.sync_all(&state.file)
            };
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let sync_result = syncer.sync_all(&state.file);
        if let Err(error) = sync_result {
            return Err(rollback_or_poison(
                state,
                checkpoint,
                journal_io(action, error),
                Some(syncer),
            ));
        }
    }
    Ok(offsets)
}

fn append_frame(state: &mut JournalState, frame: &[u8]) -> GraphStoreResult<u64> {
    let offset = state
        .file
        .seek(SeekFrom::End(0))
        .map_err(|error| journal_io("seek journal append", error))?;
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(mut injection) = state.injected_append_failure.take() {
        if injection.successful_appends_before_failure > 0 {
            injection.successful_appends_before_failure -= 1;
            state.injected_append_failure = Some(injection);
        } else {
            let partial = injection.after_bytes.min(frame.len());
            if partial > 0 {
                state
                    .file
                    .write_all(&frame[..partial])
                    .map_err(|error| journal_io("write injected partial frame", error))?;
            }
            if injection.fail_rollback {
                state.poisoned = Some("injected rollback failure".to_string());
            }
            return Err(GraphStoreError::new(
                "chunk_store_injected_append_failure",
                "injected journal append failure",
            ));
        }
    }
    state
        .file
        .write_all(frame)
        .map_err(|error| journal_io("append journal record", error))?;
    Ok(offset)
}

fn rollback_or_poison(
    state: &mut JournalState,
    pre_append_len: u64,
    original: GraphStoreError,
    syncer: Option<&dyn SyncStrategy>,
) -> GraphStoreError {
    if state.poisoned.is_none() {
        let rollback = state
            .file
            .set_len(pre_append_len)
            .and_then(|_| state.file.seek(SeekFrom::End(0)).map(|_| ()))
            .and_then(|_| match syncer {
                Some(syncer) => syncer.sync_all(&state.file),
                None => Ok(()),
            });
        if let Err(error) = rollback {
            state.poisoned = Some(format!("journal rollback failed: {error}"));
        }
    }
    if let Some(reason) = &state.poisoned {
        GraphStoreError::new(
            "chunk_store_journal_poisoned",
            format!("{reason}; original append failure: {}", original.message),
        )
    } else {
        original
    }
}

fn ensure_address(address: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual != address {
        return Err(GraphStoreError::new(
            "chunk_store_hash_mismatch",
            format!("chunk address {address} does not match payload hash {actual}"),
        ));
    }
    Ok(())
}

/// Refuse a mutation on a read-only handle.
///
/// Separate from [`ensure_usable`] because that one guards reads as well, and a
/// read-only handle must still serve every read.
fn ensure_writable(state: &JournalState) -> GraphStoreResult<()> {
    if state.read_only {
        return Err(GraphStoreError::new(
            "rr_read_only_database",
            "database is open read-only and refuses writes",
        ));
    }
    Ok(())
}

fn ensure_usable(state: &JournalState) -> GraphStoreResult<()> {
    if let Some(reason) = &state.poisoned {
        return Err(GraphStoreError::new(
            "chunk_store_journal_poisoned",
            reason.clone(),
        ));
    }
    Ok(())
}

fn ensure_decoded_bound(length: usize) -> GraphStoreResult<()> {
    if length > MAX_DECODED_PAYLOAD {
        return Err(corrupt(format!(
            "decoded payload length {length} exceeds {MAX_DECODED_PAYLOAD}"
        )));
    }
    Ok(())
}

fn ensure_encoded_bound(length: usize) -> GraphStoreResult<()> {
    if length > MAX_ENCODED_PAYLOAD {
        return Err(corrupt(format!(
            "encoded payload length {length} exceeds {MAX_ENCODED_PAYLOAD}"
        )));
    }
    Ok(())
}

fn compaction_temp_path(path: &Path) -> PathBuf {
    let sequence = COMPACTION_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("compact-{}-{sequence}.tmp", std::process::id()))
}

fn verify_open_file_identity(file: &File, path: &Path, kind: &str) -> GraphStoreResult<()> {
    let open_handle = same_file::Handle::from_file(
        file.try_clone()
            .map_err(|error| journal_io("clone resident journal file", error))?,
    )
    .map_err(|error| journal_io("inspect resident journal file", error))?;
    let configured_handle = same_file::Handle::from_path(path)
        .map_err(|error| journal_io("inspect configured journal path", error))?;
    if open_handle != configured_handle {
        return Err(GraphStoreError::new(
            "chunk_store_journal_replaced",
            format!(
                "resident {kind} is detached from configured path {}",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn canonical_journal_path(path: &Path) -> GraphStoreResult<PathBuf> {
    // `Path::new("Travis.RR").parent()` is `Some("")`, not `None`, so an
    // `unwrap_or` on its own leaves a bare relative name canonicalizing the
    // empty path and failing with ENOENT. That surfaced as an `.RR` database
    // created in the working directory taking its claim, writing its header and
    // then dying here, leaving a path no later `create` could reuse.
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::create_dir_all(parent).map_err(|error| journal_io("create journal directory", error))?;
    let canonical_parent = parent
        .canonicalize()
        .map_err(|error| journal_io("canonicalize journal directory", error))?;
    let file_name = path.file_name().ok_or_else(|| {
        GraphStoreError::new(
            "chunk_store_journal_path_invalid",
            format!("journal path {} has no file name", path.display()),
        )
    })?;
    Ok(canonical_parent.join(file_name))
}

fn acquire_journal_lock(path: &Path) -> GraphStoreResult<JournalFileLock> {
    let lock_path = path.with_extension("lock");
    let process_key = lock_path.clone();
    let process_locks = JOURNAL_PROCESS_LOCKS.get_or_init(|| Mutex::new(BTreeSet::new()));
    {
        let mut locks = process_locks.lock().map_err(|_| {
            GraphStoreError::new(
                "chunk_store_journal_lock_poisoned",
                "journal process lock registry is poisoned",
            )
        })?;
        if !locks.insert(process_key.clone()) {
            return Err(GraphStoreError::new(
                "chunk_store_journal_lock_unavailable",
                format!("journal {} is already open in this process", path.display()),
            ));
        }
    }

    let result = (|| {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| journal_io("open journal lock", error))?;
        file.try_lock_exclusive().map_err(|error| {
            let message = if error.kind() == ErrorKind::WouldBlock {
                format!("journal {} is locked by another process", path.display())
            } else {
                format!("lock journal {}: {error}", path.display())
            };
            GraphStoreError::new("chunk_store_journal_lock_unavailable", message)
        })?;
        Ok(JournalFileLock { file, process_key })
    })();
    if result.is_err() {
        release_process_lock(&lock_path);
    }
    result
}

fn release_process_lock(process_key: &Path) {
    if let Some(locks) = JOURNAL_PROCESS_LOCKS.get() {
        if let Ok(mut locks) = locks.lock() {
            locks.remove(process_key);
        }
    }
}

fn sync_parent_directory(
    path: &Path,
    syncer: &dyn SyncStrategy,
    action: &str,
) -> GraphStoreResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let directory = File::open(parent).map_err(|error| journal_io(action, error))?;
    syncer
        .sync_all(&directory)
        .map_err(|error| journal_io(action, error))
}

fn live_root_omitted(message: String) -> GraphStoreError {
    GraphStoreError::new("chunk_store_compaction_live_root_omitted", message)
}

fn corrupt(message: String) -> GraphStoreError {
    GraphStoreError::new("chunk_store_journal_corrupt", message)
}

fn journal_io(action: &str, error: impl std::fmt::Display) -> GraphStoreError {
    GraphStoreError::new("chunk_store_journal_io", format!("{action}: {error}"))
}

fn size_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_journal_size_overflow",
        "journal size accounting overflowed",
    )
}

fn journal_pin_count_overflow() -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_pin_count_overflow",
        "in-flight chunk pin reference count overflowed",
    )
}

fn increment_journal_pin_count(
    pins: &mut HashMap<Hash, usize>,
    address: Hash,
) -> GraphStoreResult<()> {
    let count = pins.entry(address).or_insert(0);
    *count = count
        .checked_add(1)
        .ok_or_else(journal_pin_count_overflow)?;
    Ok(())
}

fn decrement_journal_pin_count(
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

fn journal_pending_pin_count(
    pending: &HashMap<Hash, usize>,
    bound: &HashMap<String, JournalBoundPendingPin>,
    address: Hash,
) -> GraphStoreResult<usize> {
    pending
        .get(&address)
        .copied()
        .unwrap_or(0)
        .checked_add(journal_bound_pending_pin_count(bound, address)?)
        .ok_or_else(journal_pin_count_overflow)
}

fn journal_bound_pending_pin_count(
    bound: &HashMap<String, JournalBoundPendingPin>,
    address: Hash,
) -> GraphStoreResult<usize> {
    bound
        .values()
        .filter(|candidate| candidate.address == address)
        .try_fold(0_usize, |total, candidate| {
            total
                .checked_add(candidate.count)
                .ok_or_else(journal_pin_count_overflow)
        })
}

fn validate_journal_publication_key(publication_key: &str) -> GraphStoreResult<()> {
    if publication_key.trim().is_empty() {
        return Err(GraphStoreError::new(
            "chunk_store_publication_key_invalid",
            "in-flight publication key must be non-empty",
        ));
    }
    Ok(())
}

fn journal_pending_pin_missing(address: Hash, publication_key: &str) -> GraphStoreError {
    GraphStoreError::new(
        "chunk_store_pending_pin_missing",
        format!("address {address} has no anonymous pending pin to bind as {publication_key:?}"),
    )
}

fn journal_publication_key_address_conflict(
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

fn release_journal_bound_pending(
    bound: &mut HashMap<String, JournalBoundPendingPin>,
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

fn total_logical_bytes(mut lengths: impl Iterator<Item = u64>) -> GraphStoreResult<u64> {
    lengths.try_fold(0_u64, |total, length| {
        total.checked_add(length).ok_or_else(size_overflow)
    })
}

/// Fault injection shared by this module's tests and by the `.RR` oracles under
/// `graph/tests/rr_migration.rs`, which reach it through the crate rather than
/// carrying a second syncer of their own.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::SyncStrategy;

    /// A real syncer that counts its calls and fails the ones it is told to.
    ///
    /// `fail_on_call` refuses before touching the platter, which is a sync
    /// that never happened. `fail_after_sync_on_call` syncs and then reports
    /// failure, which is a sync whose result the process never learned.
    #[derive(Default)]
    pub(crate) struct CountingSync {
        calls: AtomicUsize,
        fail_at: Mutex<BTreeSet<usize>>,
        fail_after_sync_at: Mutex<BTreeSet<usize>>,
    }

    impl CountingSync {
        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        pub(crate) fn fail_on_call(&self, call: usize) {
            self.fail_at.lock().unwrap().insert(call);
        }

        pub(crate) fn fail_after_sync_on_call(&self, call: usize) {
            self.fail_after_sync_at.lock().unwrap().insert(call);
        }
    }

    impl SyncStrategy for CountingSync {
        fn sync_all(&self, file: &File) -> io::Result<()> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_after_sync_at.lock().unwrap().remove(&call) {
                file.sync_all()?;
                return Err(io::Error::other("injected post-sync failure"));
            }
            if self.fail_at.lock().unwrap().remove(&call) {
                return Err(io::Error::other("injected sync failure"));
            }
            file.sync_all()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::Barrier;

    use super::test_support::CountingSync;
    use super::*;
    use crate::chunk_store::{
        validate_catalog_root, MemChunkStore, RootCatalog, ROOT_NAME_GRAPH, ROOT_NAME_REDCORE,
    };

    static NEXT_TEST_JOURNAL: AtomicU64 = AtomicU64::new(0);

    struct TestJournal {
        path: PathBuf,
    }

    impl TestJournal {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rustyred-journal-{name}-{}-{}.rrcj",
                std::process::id(),
                NEXT_TEST_JOURNAL.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(path.with_extension("lock"));
            Self { path }
        }
    }

    impl Drop for TestJournal {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_file(self.path.with_extension("lock"));
        }
    }

    fn chunk_store_meta(version: u64) -> CommitMeta {
        CommitMeta {
            parents: Vec::new(),
            author: "journal-test".to_string(),
            message: format!("commit {version}"),
            timestamp_unix_ms: u128::from(version),
            graph_version: version,
        }
    }

    fn chunk_store_write_graph_catalog(
        store: &JournalChunkStore,
        graph_bytes: &[u8],
    ) -> (Hash, Hash) {
        let graph = Hash::of(graph_bytes);
        store.write(&[(graph, graph_bytes)]).unwrap();
        let catalog =
            RootCatalog::new(BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), graph)])).unwrap();
        let catalog_bytes = catalog.to_bytes().unwrap();
        let catalog_address = Hash::of(&catalog_bytes);
        store.write(&[(catalog_address, &catalog_bytes)]).unwrap();
        (graph, catalog_address)
    }

    #[test]
    fn chunk_store_journal_round_trip_and_reopen() {
        let fixture = TestJournal::new("roundtrip");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph bytes");
        let commit = store.commit(catalog, chunk_store_meta(1)).unwrap();
        assert_eq!(store.read_root().unwrap(), Some(catalog));
        assert_eq!(&*store.get(graph).unwrap().unwrap(), b"graph bytes");
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(reopened.committed_root().unwrap().unwrap().commit, commit);
        assert_eq!(&*reopened.get(graph).unwrap().unwrap(), b"graph bytes");
    }

    #[test]
    fn mem_and_journal_raw_commit_derive_identical_parent_lineage() {
        let fixture = TestJournal::new("mem-journal-parent-parity");
        let journal = JournalChunkStore::open(&fixture.path).unwrap();
        let memory = MemChunkStore::new();

        let write_catalog = |store: &dyn ChunkStore, graph_bytes: &[u8]| {
            let graph = Hash::of(graph_bytes);
            store.write(&[(graph, graph_bytes)]).unwrap();
            let catalog = RootCatalog::graph(graph);
            let bytes = catalog.to_bytes().unwrap();
            let address = Hash::of(&bytes);
            store.write(&[(address, bytes.as_slice())]).unwrap();
            address
        };

        let first_memory_root = write_catalog(&memory, b"parent parity graph one");
        let first_journal_root = write_catalog(&journal, b"parent parity graph one");
        assert_eq!(first_memory_root, first_journal_root);
        let mut first_meta = chunk_store_meta(1);
        first_meta.parents = vec![Hash::of(b"caller-supplied parent must be ignored")];
        let first_memory_commit = memory
            .commit(first_memory_root, first_meta.clone())
            .unwrap();
        let first_journal_commit = journal.commit(first_journal_root, first_meta).unwrap();
        assert_eq!(first_memory_commit, first_journal_commit);
        assert!(memory
            .committed_root()
            .unwrap()
            .unwrap()
            .meta
            .parents
            .is_empty());
        assert!(journal
            .committed_root()
            .unwrap()
            .unwrap()
            .meta
            .parents
            .is_empty());

        let second_memory_root = write_catalog(&memory, b"parent parity graph two");
        let second_journal_root = write_catalog(&journal, b"parent parity graph two");
        assert_eq!(second_memory_root, second_journal_root);
        let mut second_meta = chunk_store_meta(2);
        second_meta.parents = vec![Hash::of(b"another caller parent")];
        let second_memory_commit = memory
            .commit(second_memory_root, second_meta.clone())
            .unwrap();
        let second_journal_commit = journal.commit(second_journal_root, second_meta).unwrap();
        assert_eq!(second_memory_commit, second_journal_commit);
        assert_eq!(
            memory.committed_root().unwrap().unwrap().meta.parents,
            vec![first_memory_commit]
        );
        assert_eq!(
            journal.committed_root().unwrap().unwrap().meta.parents,
            vec![first_journal_commit]
        );
    }

    #[test]
    fn chunk_store_journal_rejects_hash_mismatch_before_append() {
        let fixture = TestJournal::new("hash-mismatch");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let wrong = Hash::of(b"other");
        let error = store.write(&[(wrong, b"payload")]).unwrap_err();
        assert_eq!(error.code, "chunk_store_hash_mismatch");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), 0);
    }

    #[test]
    fn chunk_store_journal_batch_is_all_or_nothing_on_validation() {
        let fixture = TestJournal::new("batch-validation");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let valid = b"valid";
        let valid_address = Hash::of(valid);
        let error = store
            .write(&[(valid_address, valid), (Hash::of(b"wrong"), b"invalid")])
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_hash_mismatch");
        assert!(!store.has(valid_address).unwrap());
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), 0);
    }

    #[test]
    fn chunk_store_journal_torn_tail_is_truncated() {
        let fixture = TestJournal::new("torn-tail");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let before = fs::metadata(&fixture.path).unwrap().len();
        let torn = encode_frame(RecordKind::Chunk, Hash::of(b"torn"), b"torn").unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&torn[..torn.len() - 2]).unwrap();
        file.sync_all().unwrap();
        drop(file);
        assert!(fs::metadata(&fixture.path).unwrap().len() > before);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(&*reopened.get(graph).unwrap().unwrap(), b"live graph");
    }

    #[test]
    fn chunk_store_journal_torn_tail_ignores_nested_chunk_frame_bytes() {
        let fixture = TestJournal::new("torn-tail-nested-frame");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let before = fs::metadata(&fixture.path).unwrap().len();
        let nested_payload = b"self-verifying nested chunk";
        let nested =
            encode_frame(RecordKind::Chunk, Hash::of(nested_payload), nested_payload).unwrap();
        let prefix = b"outer-prefix:";
        let mut outer_payload = prefix.to_vec();
        outer_payload.extend_from_slice(&nested);
        outer_payload.extend_from_slice(b":outer-suffix");
        assert!(outer_payload.len() < RAW_PAYLOAD_THRESHOLD);
        let outer =
            encode_frame(RecordKind::Chunk, Hash::of(&outer_payload), &outer_payload).unwrap();
        let nested_end = RECORD_HEADER_LEN + RECORD_ENCODED_LEN_LEN + prefix.len() + nested.len();
        assert!(nested_end < outer.len());
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&outer[..nested_end]).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"live graph"
        );
    }

    #[test]
    fn chunk_store_journal_torn_tail_respects_outer_frame_over_nested_valid_root() {
        let fixture = TestJournal::new("torn-tail-nested-root");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let prior = store.committed_root().unwrap().unwrap();
        drop(store);

        let before = fs::metadata(&fixture.path).unwrap().len();
        let mut nested_meta = chunk_store_meta(2);
        nested_meta.parents = vec![prior.commit];
        let nested_root = CommittedRoot::new(catalog, nested_meta).unwrap();
        let nested = encode_frame(
            RecordKind::Root,
            nested_root.commit,
            &nested_root.to_bytes().unwrap(),
        )
        .unwrap();
        let prefix = b"outer-prefix:";
        let mut outer_payload = prefix.to_vec();
        outer_payload.extend_from_slice(&nested);
        outer_payload.extend_from_slice(b":outer-suffix");
        assert!(outer_payload.len() < RAW_PAYLOAD_THRESHOLD);
        let outer =
            encode_frame(RecordKind::Chunk, Hash::of(&outer_payload), &outer_payload).unwrap();
        let nested_end = RECORD_HEADER_LEN + RECORD_ENCODED_LEN_LEN + prefix.len() + nested.len();
        assert!(nested_end < outer.len());
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&outer[..nested_end]).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(reopened.committed_root().unwrap(), Some(prior));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"live graph"
        );
    }

    #[test]
    fn chunk_store_journal_exact_tail_bad_crc_is_truncated() {
        let fixture = TestJournal::new("bad-crc-tail");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let before = fs::metadata(&fixture.path).unwrap().len();
        let mut bad_crc = encode_frame(
            RecordKind::Chunk,
            Hash::of(b"complete tail with bad crc"),
            b"complete tail with bad crc",
        )
        .unwrap();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xff;
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&bad_crc).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"live graph"
        );
    }

    #[test]
    fn chunk_store_journal_zero_hole_before_intact_uncommitted_tail_recovers_prior_root() {
        let fixture = TestJournal::new("zero-hole-intact-uncommitted-tail");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let committed_end = fs::metadata(&fixture.path).unwrap().len();
        let orphan_payload = b"intact orphan after zero hole";
        let orphan =
            encode_frame(RecordKind::Chunk, Hash::of(orphan_payload), orphan_payload).unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&[0_u8; 128]).unwrap();
        file.write_all(&orphan).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), committed_end);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"live graph"
        );
        assert!(!reopened.has(Hash::of(orphan_payload)).unwrap());
    }

    #[test]
    fn chunk_store_journal_bad_crc_before_intact_uncommitted_tail_recovers_prior_root() {
        let fixture = TestJournal::new("bad-crc-intact-uncommitted-tail");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let committed_end = fs::metadata(&fixture.path).unwrap().len();
        let mut bad_crc = encode_frame(
            RecordKind::Chunk,
            Hash::of(b"bad crc uncommitted chunk"),
            b"bad crc uncommitted chunk",
        )
        .unwrap();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xff;
        let orphan_payload = b"intact orphan after bad crc";
        let orphan =
            encode_frame(RecordKind::Chunk, Hash::of(orphan_payload), orphan_payload).unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&bad_crc).unwrap();
        file.write_all(&orphan).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), committed_end);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"live graph"
        );
        assert!(!reopened.has(Hash::of(orphan_payload)).unwrap());
    }

    #[test]
    fn chunk_store_journal_zero_hole_before_later_lineage_root_is_fatal() {
        let fixture = TestJournal::new("zero-hole-before-lineage-root");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let prior = store.committed_root().unwrap().unwrap();
        drop(store);

        let mut next_meta = chunk_store_meta(2);
        next_meta.parents = vec![prior.commit];
        let next_root = CommittedRoot::new(catalog, next_meta).unwrap();
        let next_root_frame = encode_frame(
            RecordKind::Root,
            next_root.commit,
            &next_root.to_bytes().unwrap(),
        )
        .unwrap();
        let orphan_payload = b"orphan after hidden lineage root";
        let orphan =
            encode_frame(RecordKind::Chunk, Hash::of(orphan_payload), orphan_payload).unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&[0_u8; 128]).unwrap();
        file.write_all(&next_root_frame).unwrap();
        file.write_all(&orphan).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let damaged_len = fs::metadata(&fixture.path).unwrap().len();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), damaged_len);
    }

    #[test]
    fn chunk_store_journal_bad_crc_before_later_lineage_root_is_fatal() {
        let fixture = TestJournal::new("bad-crc-before-lineage-root");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let prior = store.committed_root().unwrap().unwrap();
        drop(store);

        let mut bad_crc = encode_frame(
            RecordKind::Chunk,
            Hash::of(b"bad crc before lineage root"),
            b"bad crc before lineage root",
        )
        .unwrap();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xff;
        let mut next_meta = chunk_store_meta(2);
        next_meta.parents = vec![prior.commit];
        let next_root = CommittedRoot::new(catalog, next_meta).unwrap();
        let next_root_frame = encode_frame(
            RecordKind::Root,
            next_root.commit,
            &next_root.to_bytes().unwrap(),
        )
        .unwrap();
        let orphan_payload = b"orphan after hidden lineage root";
        let orphan =
            encode_frame(RecordKind::Chunk, Hash::of(orphan_payload), orphan_payload).unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&bad_crc).unwrap();
        file.write_all(&next_root_frame).unwrap();
        file.write_all(&orphan).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let damaged_len = fs::metadata(&fixture.path).unwrap().len();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), damaged_len);
    }

    #[test]
    fn chunk_store_journal_mid_log_corruption_is_fatal() {
        let fixture = TestJournal::new("mid-corruption");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        bytes[RECORD_HEADER_LEN + 1] ^= 0xff;
        fs::write(&fixture.path, bytes).unwrap();
        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        let _ = graph;
    }

    /// Damage an already-committed chunk on disk while the write handle is still
    /// open, and pin where the damage is caught.
    ///
    /// This is the oracle for the commit path's verification boundary. The commit
    /// path answers named-root existence from the admission index (a chunk is in
    /// `state.index` only after `ensure_address` proved it hashes to its own
    /// address), so a commit proves *reachability and continuity*, not byte
    /// integrity of bytes an earlier commit already verified. Integrity of a
    /// durable head is re-established at every boundary that reads one, and this
    /// is the one that matters: a fresh open must refuse the damaged file rather
    /// than hydrate it.
    ///
    /// If a future change removes open-time verification, this fails — which is
    /// the point. Detection at the next commit is deliberately NOT asserted here,
    /// because it is not part of the contract.
    #[test]
    fn chunk_store_journal_corruption_under_a_live_handle_is_caught_at_the_boundary() {
        let fixture = TestJournal::new("live-handle-corruption-boundary");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let committed_len = fs::metadata(&fixture.path).unwrap().len();

        // Corrupt in place, with the handle still open. Nothing between here and
        // the boundary re-reads this chunk for its bytes.
        let mut bytes = fs::read(&fixture.path).unwrap();
        bytes[RECORD_HEADER_LEN + 1] ^= 0xff;
        fs::write(&fixture.path, bytes).unwrap();
        drop(store);

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            committed_len,
            "a corrupt committed frame is fatal, not a torn tail to be truncated: \
             the refusal must not silently drop committed data"
        );
        let _ = graph;
    }

    #[test]
    fn chunk_store_journal_mid_log_length_corruption_cannot_masquerade_as_tail() {
        let fixture = TestJournal::new("mid-length-corruption");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        let original_len = bytes.len();
        let declared = u32::try_from(original_len + 128).unwrap().to_le_bytes();
        let encoded_len_offset = RECORD_HEADER_LEN;
        bytes[encoded_len_offset..encoded_len_offset + RECORD_ENCODED_LEN_LEN]
            .copy_from_slice(&declared);
        fs::write(&fixture.path, bytes).unwrap();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            original_len as u64
        );
    }

    #[test]
    fn chunk_store_journal_length_corruption_before_committed_root_and_orphan_is_fatal() {
        let fixture = TestJournal::new("mid-length-before-root-and-orphan");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let first_graph = Hash::of(b"first durable graph");
        store
            .write(&[(first_graph, b"first durable graph")])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(first_graph))]),
                chunk_store_meta(1),
            )
            .unwrap();

        let corrupt_offset = fs::metadata(&fixture.path).unwrap().len();
        let second_graph = Hash::of(b"second durable graph");
        store
            .write(&[(second_graph, b"second durable graph")])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(second_graph))]),
                chunk_store_meta(2),
            )
            .unwrap();
        let orphan = Hash::of(b"complete orphan after durable root");
        store
            .write(&[(orphan, b"complete orphan after durable root")])
            .unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        let original_len = bytes.len();
        let corrupt_offset = usize::try_from(corrupt_offset).unwrap();
        let declared = u32::try_from(original_len - corrupt_offset + 128)
            .unwrap()
            .to_le_bytes();
        let encoded_len_offset = corrupt_offset + RECORD_HEADER_LEN;
        bytes[encoded_len_offset..encoded_len_offset + RECORD_ENCODED_LEN_LEN]
            .copy_from_slice(&declared);
        fs::write(&fixture.path, bytes).unwrap();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            original_len as u64
        );
    }

    #[test]
    fn chunk_store_journal_length_corruption_before_later_root_chain_is_fatal() {
        let fixture = TestJournal::new("mid-length-before-root-chain");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let first_graph = Hash::of(b"root-chain graph one");
        store
            .write(&[(first_graph, b"root-chain graph one")])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(first_graph))]),
                chunk_store_meta(1),
            )
            .unwrap();

        let corrupt_offset = fs::metadata(&fixture.path).unwrap().len();
        let second_graph = Hash::of(b"root-chain graph two");
        store
            .write(&[(second_graph, b"root-chain graph two")])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(second_graph))]),
                chunk_store_meta(2),
            )
            .unwrap();
        let third_graph = Hash::of(b"root-chain graph three");
        store
            .write(&[(third_graph, b"root-chain graph three")])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(third_graph))]),
                chunk_store_meta(3),
            )
            .unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        let original_len = bytes.len();
        let corrupt_offset = usize::try_from(corrupt_offset).unwrap();
        let declared = u32::try_from(original_len - corrupt_offset + 128)
            .unwrap()
            .to_le_bytes();
        let encoded_len_offset = corrupt_offset + RECORD_HEADER_LEN;
        bytes[encoded_len_offset..encoded_len_offset + RECORD_ENCODED_LEN_LEN]
            .copy_from_slice(&declared);
        fs::write(&fixture.path, bytes).unwrap();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            original_len as u64
        );
    }

    #[test]
    fn chunk_store_journal_zstd_length_cannot_swallow_later_root_and_orphan() {
        let fixture = TestJournal::new("zstd-length-before-root-and-orphan");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let first_graph = Hash::of(b"zstd prior graph");
        store.write(&[(first_graph, b"zstd prior graph")]).unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(first_graph))]),
                chunk_store_meta(1),
            )
            .unwrap();

        let corrupt_offset = fs::metadata(&fixture.path).unwrap().len();
        let compressed_graph = vec![b'z'; 1024 * 1024];
        let compressed_graph_address = Hash::of(&compressed_graph);
        store
            .write(&[(compressed_graph_address, compressed_graph.as_slice())])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(compressed_graph_address))]),
                chunk_store_meta(2),
            )
            .unwrap();
        let orphan = Hash::of(b"zstd suffix orphan");
        store.write(&[(orphan, b"zstd suffix orphan")]).unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        let original_len = bytes.len();
        let corrupt_offset = usize::try_from(corrupt_offset).unwrap();
        assert_eq!(
            bytes[corrupt_offset + 42],
            PayloadEncoding::Zstd as u8,
            "the adversarial frame must use compressed framing"
        );
        let framing_without_payload = RECORD_HEADER_LEN + RECORD_ENCODED_LEN_LEN + RECORD_CRC_LEN;
        let swallowed_suffix_len = original_len - corrupt_offset - framing_without_payload;
        assert!(swallowed_suffix_len < compressed_graph.len());
        let declared = u32::try_from(swallowed_suffix_len).unwrap().to_le_bytes();
        let encoded_len_offset = corrupt_offset + RECORD_HEADER_LEN;
        bytes[encoded_len_offset..encoded_len_offset + RECORD_ENCODED_LEN_LEN]
            .copy_from_slice(&declared);
        fs::write(&fixture.path, bytes).unwrap();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert!(error.message.contains("zstd frame occupies"));
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            original_len as u64
        );
    }

    #[test]
    fn chunk_store_journal_mid_length_corruption_consuming_exact_eof_is_fatal() {
        let fixture = TestJournal::new("mid-length-consumes-eof");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);

        let mut bytes = fs::read(&fixture.path).unwrap();
        let original_len = bytes.len();
        let framing_without_payload = RECORD_HEADER_LEN + RECORD_ENCODED_LEN_LEN + RECORD_CRC_LEN;
        let declared = u32::try_from(original_len - framing_without_payload)
            .unwrap()
            .to_le_bytes();
        let encoded_len_offset = RECORD_HEADER_LEN;
        bytes[encoded_len_offset..encoded_len_offset + RECORD_ENCODED_LEN_LEN]
            .copy_from_slice(&declared);
        fs::write(&fixture.path, bytes).unwrap();

        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(
            fs::metadata(&fixture.path).unwrap().len(),
            original_len as u64
        );
    }

    #[test]
    fn chunk_store_journal_lock_is_exclusive() {
        let fixture = TestJournal::new("lock");
        let first = JournalChunkStore::open(&fixture.path).unwrap();
        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_lock_unavailable");
        drop(first);
        JournalChunkStore::open(&fixture.path).unwrap();
    }

    #[test]
    fn chunk_store_journal_concurrent_writes_share_one_index() {
        let fixture = TestJournal::new("concurrent");
        let store = Arc::new(JournalChunkStore::open(&fixture.path).unwrap());
        let left = b"left thread chunk".to_vec();
        let right = b"right thread chunk".to_vec();
        let left_address = Hash::of(&left);
        let right_address = Hash::of(&right);
        let left_store = Arc::clone(&store);
        let right_store = Arc::clone(&store);
        let left_thread =
            thread::spawn(move || left_store.write(&[(left_address, left.as_slice())]));
        let right_thread =
            thread::spawn(move || right_store.write(&[(right_address, right.as_slice())]));
        left_thread.join().unwrap().unwrap();
        right_thread.join().unwrap().unwrap();

        let catalog = RootCatalog::new(BTreeMap::from([
            ("graph".to_string(), left_address),
            ("secondary".to_string(), right_address),
        ]))
        .unwrap();
        let catalog_bytes = catalog.to_bytes().unwrap();
        let catalog_address = Hash::of(&catalog_bytes);
        store.write(&[(catalog_address, &catalog_bytes)]).unwrap();
        store.commit(catalog_address, chunk_store_meta(1)).unwrap();
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(catalog_address));
        assert_eq!(
            &*reopened.get(left_address).unwrap().unwrap(),
            b"left thread chunk"
        );
        assert_eq!(
            &*reopened.get(right_address).unwrap().unwrap(),
            b"right thread chunk"
        );
    }

    #[test]
    fn chunk_store_journal_compaction_preserves_catalog_and_named_roots() {
        let fixture = TestJournal::new("compaction");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        let orphan_bytes = b"unreachable orphan";
        let orphan = Hash::of(orphan_bytes);
        store.write(&[(orphan, orphan_bytes)]).unwrap();
        store.commit(catalog, chunk_store_meta(1)).unwrap();

        let omission = store
            .compact_reachable(Some(catalog), &[catalog])
            .unwrap_err();
        assert_eq!(omission.code, "chunk_store_compaction_live_root_omitted");
        assert!(store.has(orphan).unwrap());

        let report = store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert_eq!(report.chunks_before, 3);
        assert_eq!(report.chunks_after, 2);
        assert_eq!(report.chunks_removed, 1);
        assert!(!store.has(orphan).unwrap());
        assert_eq!(store.read_root().unwrap(), Some(catalog));
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(&*reopened.get(graph).unwrap().unwrap(), b"live graph");
        assert!(!reopened.has(orphan).unwrap());
    }

    #[test]
    fn chunk_store_journal_multi_commit_compaction_reopens_from_checkpoint() {
        let fixture = TestJournal::new("multi-commit-compaction-reopen");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (first_graph, first_catalog) =
            chunk_store_write_graph_catalog(&store, b"first live graph");
        store.commit(first_catalog, chunk_store_meta(1)).unwrap();

        let second_graph_bytes = b"second live graph";
        let second_graph = Hash::of(second_graph_bytes);
        store.write(&[(second_graph, second_graph_bytes)]).unwrap();
        let second = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(second_graph))]),
                chunk_store_meta(2),
            )
            .unwrap();
        let committed_before = store.committed_root().unwrap().unwrap();
        assert_eq!(committed_before.root, second.catalog_root);

        let report = store
            .compact_reachable(
                Some(second.catalog_root),
                &[second.catalog_root, second_graph],
            )
            .unwrap();
        assert!(report.chunks_removed >= 2);
        assert!(!store.has(first_catalog).unwrap());
        assert!(!store.has(first_graph).unwrap());
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.committed_root().unwrap(), Some(committed_before));
        assert_eq!(reopened.read_root().unwrap(), Some(second.catalog_root));
        assert_eq!(
            &*reopened.get(second_graph).unwrap().unwrap(),
            second_graph_bytes
        );
    }

    #[test]
    fn chunk_store_journal_compaction_honors_keyed_ref_counted_in_flight_pins() {
        let fixture = TestJournal::new("compaction-pins");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let payload_bytes = b"payload awaiting durable graph pointer";
        let payload = Hash::of(payload_bytes);
        let node_a_key = "node-a:tensor_payload:sha256:shared";
        let node_b_key = "node-b:tensor_payload:sha256:shared";

        store.pin_in_flight(payload).unwrap();
        store
            .compact_reachable(Some(catalog), &[catalog, graph])
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

        // An idempotent upsert for node A creates and releases a fresh claim;
        // it cannot consume node B's keyed receipts for the same address.
        store.claim_in_flight(payload).unwrap();
        assert_eq!(store.pending_pin_count(payload).unwrap(), 2);
        assert_eq!(store.claimed_pin_count(payload).unwrap(), 1);
        store.release_claimed_in_flight(payload).unwrap();
        assert!(!store
            .release_bound_pending_in_flight(payload, node_a_key)
            .unwrap());
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 2);

        assert!(store
            .release_bound_pending_in_flight(payload, node_b_key)
            .unwrap());
        assert_eq!(store.bound_pending_pin_count(payload).unwrap(), 1);
        store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert!(store.has(payload).unwrap());
        assert!(store
            .release_bound_pending_in_flight(payload, node_b_key)
            .unwrap());
        assert_eq!(store.in_flight_pin_count(payload).unwrap(), 0);
        store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert!(!store.has(payload).unwrap());

        let external_bytes = b"externally supplied journal payload";
        let external = Hash::of(external_bytes);
        store.write(&[(external, external_bytes)]).unwrap();
        store.claim_in_flight(external).unwrap();
        store.pin_in_flight(external).unwrap();
        store.unpin_in_flight(external).unwrap();
        assert_eq!(store.pending_pin_count(external).unwrap(), 0);
        assert_eq!(store.claimed_pin_count(external).unwrap(), 1);
        store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert!(store.has(external).unwrap());
        store.release_claimed_in_flight(external).unwrap();
        store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert!(!store.has(external).unwrap());

        // A logical publication key cannot be rebound to another address. The
        // failed attempt preserves both sides for explicit cleanup.
        let old_bytes = b"superseded journal publication";
        let old = Hash::of(old_bytes);
        let new_bytes = b"replacement journal publication";
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
            .compact_reachable(Some(catalog), &[catalog, graph])
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
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        assert!(!store.has(old).unwrap());
        assert!(!store.has(new).unwrap());
    }

    #[test]
    fn redcore_journal_commits_new_graph_and_redcore_roots() {
        let fixture = TestJournal::new("redcore-first-commit");
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(&fixture.path, syncer.clone()).unwrap();
        let baseline_syncs = syncer.calls();
        let graph_bytes = b"redcore journal graph";
        let redcore_bytes = b"redcore journal operating state";
        let graph = Hash::of(graph_bytes);
        let redcore = Hash::of(redcore_bytes);
        store
            .write(&[(graph, graph_bytes), (redcore, redcore_bytes)])
            .unwrap();

        let result = store
            .commit_catalog_updates(
                &BTreeMap::from([
                    (crate::chunk_store::ROOT_NAME_GRAPH.to_string(), Some(graph)),
                    (
                        crate::chunk_store::ROOT_NAME_REDCORE.to_string(),
                        Some(redcore),
                    ),
                ]),
                chunk_store_meta(1),
            )
            .unwrap();
        assert_eq!(syncer.calls(), baseline_syncs + 1);
        assert_eq!(store.read_root().unwrap(), Some(result.catalog_root));
        assert_eq!(
            store.committed_root().unwrap().unwrap().commit,
            result.commit
        );
        let catalog = validate_catalog_root(&store, result.catalog_root).unwrap();
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), Some(redcore));
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(result.catalog_root));
        assert_eq!(
            reopened.committed_root().unwrap().unwrap().commit,
            result.commit
        );
    }

    #[test]
    fn chunk_store_journal_catalog_updates_merge_under_one_lock() {
        let fixture = TestJournal::new("catalog-merge");
        let store = Arc::new(JournalChunkStore::open(&fixture.path).unwrap());
        let graph = Hash::of(b"graph root");
        let documents = Hash::of(b"document root");
        let redcore = Hash::of(b"redcore root");
        store
            .write(&[
                (graph, b"graph root"),
                (documents, b"document root"),
                (redcore, b"redcore root"),
            ])
            .unwrap();
        let initial = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let left_store = Arc::clone(&store);
        let left_barrier = Arc::clone(&barrier);
        let left = thread::spawn(move || {
            left_barrier.wait();
            left_store.commit_catalog_updates(
                &BTreeMap::from([("documents".to_string(), Some(documents))]),
                chunk_store_meta(2),
            )
        });
        let right_store = Arc::clone(&store);
        let right_barrier = Arc::clone(&barrier);
        let right = thread::spawn(move || {
            right_barrier.wait();
            right_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(redcore))]),
                chunk_store_meta(3),
            )
        });
        barrier.wait();
        let left_result = left.join().unwrap().unwrap();
        let right_result = right.join().unwrap().unwrap();

        let root = store.read_root().unwrap().unwrap();
        let catalog = validate_catalog_root(store.as_ref(), root).unwrap();
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph));
        assert_eq!(catalog.get("documents"), Some(documents));
        assert_eq!(catalog.get(ROOT_NAME_REDCORE), Some(redcore));
        let committed = store.committed_root().unwrap().unwrap();
        let immediately_previous = if committed.commit == left_result.commit {
            right_result.commit
        } else {
            assert_eq!(committed.commit, right_result.commit);
            left_result.commit
        };
        assert_eq!(committed.meta.parents, vec![immediately_previous]);
        assert_ne!(immediately_previous, initial.commit);
    }

    #[test]
    fn chunk_store_journal_grouped_commit_shares_one_sync_barrier() {
        let fixture = TestJournal::new("grouped-commit");
        let syncer = Arc::new(CountingSync::default());
        let store = Arc::new(
            JournalChunkStore::open_with_policy_and_syncer(
                &fixture.path,
                JournalCommitPolicy::Grouped {
                    max_wait: Duration::from_millis(100),
                },
                syncer.clone(),
            )
            .unwrap(),
        );
        let graph = Hash::of(b"group graph");
        let documents = Hash::of(b"group documents");
        store
            .write(&[(graph, b"group graph"), (documents, b"group documents")])
            .unwrap();
        let initial = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        let baseline_syncs = syncer.calls();
        let barrier = Arc::new(Barrier::new(3));
        let first_store = Arc::clone(&store);
        let first_barrier = Arc::clone(&barrier);
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_store.commit_catalog_updates(
                &BTreeMap::from([("documents".to_string(), Some(documents))]),
                chunk_store_meta(2),
            )
        });
        let second_store = Arc::clone(&store);
        let second_barrier = Arc::clone(&barrier);
        let second = thread::spawn(move || {
            second_barrier.wait();
            second_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(documents))]),
                chunk_store_meta(3),
            )
        });
        barrier.wait();
        let first_result = first.join().unwrap().unwrap();
        let second_result = second.join().unwrap().unwrap();
        assert_eq!(syncer.calls(), baseline_syncs + 1);

        let committed = store.committed_root().unwrap().unwrap();
        let immediately_previous = if committed.commit == first_result.commit {
            second_result.commit
        } else {
            assert_eq!(committed.commit, second_result.commit);
            first_result.commit
        };
        assert_eq!(committed.meta.parents, vec![immediately_previous]);
        assert_ne!(immediately_previous, initial.commit);
        assert_eq!(store.read_root().unwrap(), Some(committed.root));
    }

    #[test]
    fn chunk_store_journal_grouped_commit_failure_refuses_entire_batch() {
        let fixture = TestJournal::new("grouped-failure");
        let syncer = Arc::new(CountingSync::default());
        let store = Arc::new(
            JournalChunkStore::open_with_policy_and_syncer(
                &fixture.path,
                JournalCommitPolicy::Grouped {
                    max_wait: Duration::from_millis(100),
                },
                syncer.clone(),
            )
            .unwrap(),
        );
        let graph = Hash::of(b"stable graph");
        let documents = Hash::of(b"candidate documents");
        store
            .write(&[
                (graph, b"stable graph"),
                (documents, b"candidate documents"),
            ])
            .unwrap();
        let initial = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        let baseline_len = fs::metadata(&fixture.path).unwrap().len();
        syncer.fail_on_call(syncer.calls() + 1);
        let barrier = Arc::new(Barrier::new(3));
        let first_store = Arc::clone(&store);
        let first_barrier = Arc::clone(&barrier);
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_store.commit_catalog_updates(
                &BTreeMap::from([("documents".to_string(), Some(documents))]),
                chunk_store_meta(2),
            )
        });
        let second_store = Arc::clone(&store);
        let second_barrier = Arc::clone(&barrier);
        let second = thread::spawn(move || {
            second_barrier.wait();
            second_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(documents))]),
                chunk_store_meta(3),
            )
        });
        barrier.wait();
        let first_error = first.join().unwrap().unwrap_err();
        let second_error = second.join().unwrap().unwrap_err();
        assert_eq!(first_error.code, "chunk_store_journal_io");
        assert_eq!(second_error.code, "chunk_store_journal_io");
        assert_eq!(store.read_root().unwrap(), Some(initial.catalog_root));
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), baseline_len);
    }

    #[test]
    fn chunk_store_journal_grouped_runtime_stops_with_last_clone() {
        let fixture = TestJournal::new("grouped-stop");
        let store = JournalChunkStore::open_with_commit_policy(
            &fixture.path,
            JournalCommitPolicy::Grouped {
                max_wait: Duration::from_millis(1),
            },
        )
        .unwrap();
        let shared = match &store.inner.commits {
            CommitRuntime::Grouped(runtime) => Arc::clone(&runtime.shared),
            CommitRuntime::Immediate => panic!("expected grouped runtime"),
        };
        drop(store);
        let started = Instant::now();
        while !shared.stopped.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(1) {
            thread::yield_now();
        }
        assert!(shared.stopped.load(Ordering::SeqCst));
    }

    #[test]
    fn chunk_store_journal_refuses_root_with_missing_named_entry() {
        let fixture = TestJournal::new("missing-root");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let missing = Hash::of(b"missing");
        let catalog =
            RootCatalog::new(BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), missing)])).unwrap();
        let catalog_bytes = catalog.to_bytes().unwrap();
        let catalog_address = Hash::of(&catalog_bytes);
        store.write(&[(catalog_address, &catalog_bytes)]).unwrap();
        let error = store
            .commit(catalog_address, chunk_store_meta(1))
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_named_root_missing");
        assert_eq!(store.read_root().unwrap(), None);
    }

    #[test]
    fn chunk_store_journal_sync_failure_does_not_publish_root() {
        let fixture = TestJournal::new("sync-failure");
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(&fixture.path, syncer.clone()).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        let pre_commit_len = fs::metadata(&fixture.path).unwrap().len();
        // Model an fsync which made the bytes durable but reported failure.
        // Recovery must durably sync the subsequent truncation before returning.
        syncer.fail_after_sync_on_call(syncer.calls() + 1);
        let error = store.commit(catalog, chunk_store_meta(1)).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_io");
        assert_eq!(store.read_root().unwrap(), None);
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), pre_commit_len);
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), None);
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), pre_commit_len);
    }

    #[test]
    fn chunk_store_journal_rollback_sync_failure_poisons_store() {
        let fixture = TestJournal::new("rollback-sync-poison");
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(&fixture.path, syncer.clone()).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        let commit_sync = syncer.calls() + 1;
        syncer.fail_after_sync_on_call(commit_sync);
        syncer.fail_on_call(commit_sync + 1);

        let error = store.commit(catalog, chunk_store_meta(1)).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_poisoned");
        assert!(store.is_poisoned());
        assert_eq!(
            store.read_root().unwrap_err().code,
            "chunk_store_journal_poisoned"
        );
    }

    #[test]
    fn chunk_store_journal_injected_append_failure_rolls_back() {
        let fixture = TestJournal::new("append-rollback");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        let pre_commit_len = fs::metadata(&fixture.path).unwrap().len();
        store.inject_append_failure(3, false);
        let error = store.commit(catalog, chunk_store_meta(1)).unwrap_err();
        assert_eq!(error.code, "chunk_store_injected_append_failure");
        assert!(!store.is_poisoned());
        assert_eq!(store.read_root().unwrap(), None);
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), pre_commit_len);
    }

    #[test]
    fn chunk_store_journal_failed_rollback_poisons_store() {
        let fixture = TestJournal::new("append-poison");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        store.inject_append_failure(2, true);
        let error = store.commit(catalog, chunk_store_meta(1)).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_poisoned");
        assert!(store.is_poisoned());
        assert_eq!(
            store.read_root().unwrap_err().code,
            "chunk_store_journal_poisoned"
        );
    }

    #[test]
    fn chunk_store_journal_compresses_large_payloads() {
        let fixture = TestJournal::new("compression");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let payload = vec![b'x'; 1024 * 1024];
        let address = Hash::of(&payload);
        store.write(&[(address, &payload)]).unwrap();
        assert!(fs::metadata(&fixture.path).unwrap().len() < 16 * 1024);
        assert_eq!(&*store.get(address).unwrap().unwrap(), payload.as_slice());
    }

    #[test]
    fn chunk_store_journal_torn_compressed_tail_recovers() {
        let fixture = TestJournal::new("torn-compressed");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        drop(store);
        let before = fs::metadata(&fixture.path).unwrap().len();

        let payload = vec![b'z'; 1024 * 1024];
        let frame = encode_frame(RecordKind::Chunk, Hash::of(&payload), &payload).unwrap();
        let mut file = OpenOptions::new().append(true).open(&fixture.path).unwrap();
        file.write_all(&frame[..frame.len() - 1]).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(reopened.read_root().unwrap(), Some(catalog));
        assert_eq!(&*reopened.get(graph).unwrap().unwrap(), b"graph");
    }

    #[test]
    fn chunk_store_journal_concurrent_catalog_updates_preserve_parent_chain() {
        let fixture = TestJournal::new("parent-chain");
        let store = Arc::new(JournalChunkStore::open(&fixture.path).unwrap());
        let graph = Hash::of(b"graph");
        let documents = Hash::of(b"documents");
        let redcore = Hash::of(b"redcore");
        store
            .write(&[
                (graph, b"graph"),
                (documents, b"documents"),
                (redcore, b"redcore"),
            ])
            .unwrap();
        store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let left_store = Arc::clone(&store);
        let left_barrier = Arc::clone(&barrier);
        let left = thread::spawn(move || {
            left_barrier.wait();
            left_store.commit_catalog_updates(
                &BTreeMap::from([("documents".to_string(), Some(documents))]),
                chunk_store_meta(2),
            )
        });
        let right_store = Arc::clone(&store);
        let right_barrier = Arc::clone(&barrier);
        let right = thread::spawn(move || {
            right_barrier.wait();
            right_store.commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_REDCORE.to_string(), Some(redcore))]),
                chunk_store_meta(3),
            )
        });
        barrier.wait();
        left.join().unwrap().unwrap();
        right.join().unwrap().unwrap();
        let final_committed = store.committed_root().unwrap().unwrap();
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.committed_root().unwrap(), Some(final_committed));
    }

    #[test]
    fn chunk_store_journal_compaction_rewrites_only_reachable_chunks() {
        let fixture = TestJournal::new("compaction-rewrite");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"live graph");
        let orphan = Hash::of(b"large unreachable payload");
        store
            .write(&[(orphan, b"large unreachable payload")])
            .unwrap();
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        let before = fs::metadata(&fixture.path).unwrap().len();

        let report = store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap();
        let after = fs::metadata(&fixture.path).unwrap().len();
        assert_eq!(report.chunks_removed, 1);
        assert!(!store.has(orphan).unwrap());
        assert!(after < before);
        assert_eq!(store.read_root().unwrap(), Some(catalog));
    }

    #[test]
    fn chunk_store_journal_compaction_refuses_stale_root_set() {
        let fixture = TestJournal::new("stale-compaction");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (graph, first_catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        store.commit(first_catalog, chunk_store_meta(1)).unwrap();
        let second_root = Hash::of(b"second root");
        store.write(&[(second_root, b"second root")]).unwrap();
        let second = store
            .commit_catalog_updates(
                &BTreeMap::from([("secondary".to_string(), Some(second_root))]),
                chunk_store_meta(2),
            )
            .unwrap();

        let error = store
            .compact_reachable(Some(first_catalog), &[first_catalog, graph])
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_compaction_root_changed");
        assert!(store.has(second.catalog_root).unwrap());
        assert!(store.has(second_root).unwrap());
    }

    #[test]
    fn chunk_store_journal_failed_append_recovers_or_poison_refuses() {
        let fixture = TestJournal::new("append-recovery");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let (_graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        let before = fs::metadata(&fixture.path).unwrap().len();
        store.inject_append_failure(7, false);
        assert!(store.commit(catalog, chunk_store_meta(1)).is_err());
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), None);
    }

    #[test]
    fn chunk_store_journal_partial_chunk_batch_rolls_back_before_later_commit() {
        let fixture = TestJournal::new("partial-chunk-then-commit");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let first = Hash::of(b"first staged chunk");
        let second = Hash::of(b"second staged chunk");
        let before = fs::metadata(&fixture.path).unwrap().len();

        // Admit the first frame, then tear the second. The entire write batch
        // must return to its checkpoint, including the in-memory address index.
        store.inject_append_failure_after(1, 5, false);
        let error = store
            .write(&[
                (first, b"first staged chunk"),
                (second, b"second staged chunk"),
            ])
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_injected_append_failure");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert!(!store.has(first).unwrap());
        assert!(!store.has(second).unwrap());

        let graph = Hash::of(b"durable graph after failed chunk batch");
        store
            .write(&[(graph, b"durable graph after failed chunk batch")])
            .unwrap();
        let committed = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(committed.catalog_root));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"durable graph after failed chunk batch"
        );
    }

    #[test]
    fn chunk_store_journal_partial_catalog_rolls_back_before_later_commit() {
        let fixture = TestJournal::new("partial-catalog-then-commit");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let graph = Hash::of(b"graph survives partial catalog");
        store
            .write(&[(graph, b"graph survives partial catalog")])
            .unwrap();
        let before = fs::metadata(&fixture.path).unwrap().len();
        let updates = BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]);

        // Fail inside the newly generated catalog frame, before its root frame
        // exists. The next successful commit must not land after a torn middle
        // record which recovery would later truncate.
        store.inject_append_failure(7, false);
        let error = store
            .commit_catalog_updates(&updates, chunk_store_meta(1))
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_injected_append_failure");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
        assert_eq!(store.read_root().unwrap(), None);

        let committed = store
            .commit_catalog_updates(&updates, chunk_store_meta(2))
            .unwrap();
        drop(store);

        let reopened = JournalChunkStore::open(&fixture.path).unwrap();
        assert_eq!(reopened.read_root().unwrap(), Some(committed.catalog_root));
        assert_eq!(
            reopened.get(graph).unwrap().unwrap().as_ref(),
            b"graph survives partial catalog"
        );
    }

    #[test]
    fn chunk_store_journal_append_failure_after_catalog_chunk_rolls_back_all() {
        let fixture = TestJournal::new("catalog-append-rollback");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let graph = Hash::of(b"graph");
        store.write(&[(graph, b"graph")]).unwrap();
        let before = fs::metadata(&fixture.path).unwrap().len();
        store.inject_append_failure_after(1, 4, false);
        let error = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_injected_append_failure");
        assert_eq!(store.read_root().unwrap(), None);
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), before);
    }

    #[test]
    fn chunk_store_journal_catalog_root_chunk_is_addressable_after_commit() {
        let fixture = TestJournal::new("catalog-addressable");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let graph = Hash::of(b"graph");
        store.write(&[(graph, b"graph")]).unwrap();
        let result = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap();
        let bytes = store
            .get(result.catalog_root)
            .unwrap()
            .expect("catalog chunk");
        let catalog = RootCatalog::from_bytes(&bytes).unwrap();
        assert_eq!(catalog.get(ROOT_NAME_GRAPH), Some(graph));
    }

    #[test]
    fn chunk_store_journal_unsupported_encoding_is_fatal() {
        let fixture = TestJournal::new("bad-encoding");
        let address = Hash::of(b"payload");
        let mut frame = encode_frame(RecordKind::Chunk, address, b"payload").unwrap();
        frame[42] = 99;
        let crc_offset = frame.len() - RECORD_CRC_LEN;
        let crc = crc32fast::hash(&frame[..crc_offset]);
        frame[crc_offset..].copy_from_slice(&crc.to_le_bytes());
        fs::write(&fixture.path, frame).unwrap();
        let error = JournalChunkStore::open(&fixture.path).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
    }

    #[test]
    fn chunk_store_journal_payload_bound_is_enforced() {
        let fixture = TestJournal::new("payload-bound");
        let store = JournalChunkStore::open(&fixture.path).unwrap();
        let payload = vec![0_u8; MAX_DECODED_PAYLOAD + 1];
        let address = Hash::of(&payload);
        let error = store.write(&[(address, &payload)]).unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_corrupt");
        assert_eq!(fs::metadata(&fixture.path).unwrap().len(), 0);
    }

    #[test]
    fn chunk_store_journal_compaction_sync_failure_preserves_old_handle() {
        let fixture = TestJournal::new("compaction-sync-failure");
        let syncer = Arc::new(CountingSync::default());
        let store = JournalChunkStore::open_with_syncer(&fixture.path, syncer.clone()).unwrap();
        let (graph, catalog) = chunk_store_write_graph_catalog(&store, b"graph");
        let orphan = Hash::of(b"orphan");
        store.write(&[(orphan, b"orphan")]).unwrap();
        store.commit(catalog, chunk_store_meta(1)).unwrap();
        syncer.fail_on_call(syncer.calls() + 1);

        let error = store
            .compact_reachable(Some(catalog), &[catalog, graph])
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_journal_io");
        assert!(store.has(orphan).unwrap());
        assert_eq!(store.read_root().unwrap(), Some(catalog));
    }

    #[test]
    fn grouped_worker_shutdown_wakes_pending_requests() {
        let fixture = TestJournal::new("grouped-shutdown");
        let store = JournalChunkStore::open_with_commit_policy(
            &fixture.path,
            JournalCommitPolicy::Grouped {
                max_wait: Duration::from_secs(10),
            },
        )
        .unwrap();
        let graph = Hash::of(b"graph");
        store.write(&[(graph, b"graph")]).unwrap();
        let runtime = match &store.inner.commits {
            CommitRuntime::Grouped(runtime) => runtime,
            CommitRuntime::Immediate => panic!("expected grouped runtime"),
        };
        {
            let mut queue = runtime.shared.queue.lock().unwrap();
            queue.shutdown = true;
            runtime.shared.ready.notify_all();
        }
        let started = Instant::now();
        while !store.grouped_worker_stopped() && started.elapsed() < Duration::from_secs(1) {
            thread::yield_now();
        }
        assert!(store.grouped_worker_stopped());
        let error = store
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_GRAPH.to_string(), Some(graph))]),
                chunk_store_meta(1),
            )
            .unwrap_err();
        assert_eq!(error.code, "chunk_store_group_commit_stopped");
    }

    /// A bare relative name has an *empty* parent, not an absent one.
    ///
    /// `Path::new("Travis.RR").parent()` is `Some("")`, so an `unwrap_or` on
    /// its own leaves the empty path being canonicalized, which fails with
    /// ENOENT. It surfaced one layer up as an `.RR` database that took its
    /// claim, wrote its header and then died here, leaving a path no later
    /// `create` could reuse. Asserted directly rather than through a test that
    /// changes the working directory, which is process-global and would race
    /// every other test in this binary.
    #[test]
    fn canonical_journal_path_resolves_a_bare_relative_name() {
        let here = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(
            canonical_journal_path(Path::new("Travis.RR")).unwrap(),
            here.join("Travis.RR")
        );
        assert_eq!(
            canonical_journal_path(Path::new("./Travis.RR")).unwrap(),
            here.join("Travis.RR")
        );
        assert!(
            !here.join("Travis.RR").exists(),
            "resolving a path must not create it"
        );
    }
}

#[cfg(test)]
mod bounded_frame_admission_tests {
    use super::*;
    #[test]
    fn refuses_decoded_size_before_payload_read_or_decompression() {
        // A real compressed frame with a tiny encoded body must still be
        // refused on its declared decoded size before that body is read.
        let body = vec![b'x'; 64 * 1024];
        let frame = encode_frame(RecordKind::Chunk, Hash::of(&body), &body).unwrap();
        let mut source = std::io::Cursor::new(frame);
        let result = read_frame_bounded(&mut source, Some(1024));
        assert!(
            matches!(result,Err(FrameReadError::Semantic(ref e)) if e.code=="chunk_read_bound_refused")
        );
        assert_eq!(
            source.position(),
            (RECORD_HEADER_LEN + RECORD_ENCODED_LEN_LEN) as u64
        );
    }
}
