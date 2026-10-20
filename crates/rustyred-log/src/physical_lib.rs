//! The physical append-only operation log.
//!
//! This crate is the sole durable writer below every RustyRed fold. The
//! temporary GraphStore-backed API lives in `theorem-oplog-graph-compat` and
//! is not a second authority.

pub mod format;
pub mod overlap;
pub mod physical;
pub mod proposal;
pub mod provenance;
pub mod schedule;

// The file layer beneath the physical log. Private module, public seam: storage
// backends are built by the crate, not injected by callers, and the browser
// backend (OPFS) implements it inside this crate -- but the layer above the
// log is written against the seam, so `LogFile` and each backend's opener are
// re-exported below.
mod storage;

// The browser storage backend, and the probe surface its oracle drives. It
// needs bindings that only exist on wasm32 and a Worker to hold a synchronous
// access handle, so it is off unless both the target and the feature ask for
// it.
#[cfg(all(target_arch = "wasm32", feature = "browser-opfs"))]
mod opfs;

// The six-operation storage seam and its two openers.
//
// The ledger reaches storage only through `LogFile`, and so does everything
// built above the ledger: RustyRed's browser projection owns durable local
// state as a `ChunkStore` over this seam (`browser_projection` in
// `rustyred-thg-core`, SPEC-THEOREMWEB-PWA-HTML-CANVAS-1.1 section 33).
// Exposing the trait is what lets that layer be written once against the six
// operations instead of once per backend, and exposing each backend's opener
// is what lets it be constructed on either target.
pub use storage::LogFile;
#[cfg(not(target_arch = "wasm32"))]
pub use storage::native::NativeLogFile;
#[cfg(all(target_arch = "wasm32", feature = "browser-opfs"))]
pub use opfs::OpfsLogFile;

pub use format::{
    ContentHash, DeploymentId, IntentKind, LogId, LogIntent, LogPosition, LogRecord, OpDigest,
    RecordId, PHYSICAL_LOG_PROTOCOL_V2, PHYSICAL_LOG_PROTOCOL_V3, PHYSICAL_LOG_PROTOCOL_VERSION,
};
pub use overlap::{read_write_overlap, AccessPath, OverlapError, ReadSet, WriteSet};
pub use physical::{
    canonical_record_hash, AppendReceipt, CrashInjection, PhysicalLog, PhysicalLogError,
    PhysicalLogOptions, PinnedPrefix, VerifyReport,
};
pub use proposal::{
    CommitError, CommitOutcome, CommitReceipt, Proposal, ProposalCommitter, RefusalCode,
    RefusalReceipt,
};
pub use provenance::{
    BridgeTransport, CommitAuthority, EntryOrigin, EntryProvenance, EvidenceGrade,
    VerificationOutcome, VerificationState, WriterId,
};
pub use schedule::{
    wrap_scheduled_mutation, AtomicScheduledMutation, BusinessOutcome, BusinessOutcomeKind,
    HeldSchedule, ScheduleChange, ScheduleError, ScheduleHold, ScheduleHoldReason, ScheduleId,
    ScheduleJournal, ScheduleProjection, ScheduleRow, ScheduleSource, ScheduleTransaction,
    ScheduledFunctionKind, SCHEDULED_MUTATION_INTENT_NAME, SCHEDULE_INTENT_NAME,
    SCHEDULE_INTENT_NAMESPACE, SCHEDULE_INTENT_VERSION,
};
