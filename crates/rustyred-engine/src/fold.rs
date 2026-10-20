//! Deterministic fold ABI over a pinned `theorem-oplog` prefix.
//!
//! The log remains the sole durable authority. [`FoldRuntime`] owns replay
//! position, checkpoint identity, and quarantine receipts so individual fold
//! implementations cannot accidentally turn derived state into a second
//! writer.
//!
//! # The client fold ABI (`LogFold`)
//!
//! SPEC-THEOREM-APP-HOST-2.0 declares one client read protocol. The declared
//! subset is [`FoldId::CLIENT_SUBSET`] — that constant is the single
//! declaration, and `theorem-doc-store::client` re-exports it for the written
//! surface rather than restating a list. A client (a browser tab, or any reader
//! that is not the writer) has exactly four operations, all over the same
//! [`FoldRuntime`] the server uses:
//!
//! 1. [`LogFold::load_checkpoint`] — adopt a hash-validated checkpoint and
//!    apply its tail; a rejected checkpoint rebuilds and returns the rejection
//!    as a receipt, never silently.
//! 2. [`LogFold::apply_tail`] — extend the applied prefix to the log head.
//! 3. [`LogFold::upquery_by_hash`] — answer a keyed query at a prefix the
//!    client pinned by log-content hash; a mismatch refuses rather than
//!    answering from a different prefix.
//! 4. [`LogFold::submit_proposal`] — hand a proposal to the one committer; the
//!    log is the sole writer and both commit and refusal return receipts.
//!
//! A browser tab is a reader of the one fold, never a second implementation.

pub mod algebra;
pub mod documents;
pub mod history;

use crate::fold_quarantine::{FoldConsumeError, QuarantineReceipt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use theorem_oplog::{
    CommitOutcome, ContentHash, LogId, LogPosition, LogRecord, PhysicalLog, Proposal,
    ProposalCommitter,
};

pub const FOLD_CHECKPOINT_PROTOCOL_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FoldId {
    Facts,
    Graph,
    Algebra,
    History,
    Documents,
}

impl FoldId {
    pub const ALL: [Self; 5] = [
        Self::Facts,
        Self::Graph,
        Self::Algebra,
        Self::History,
        Self::Documents,
    ];

    /// APP-HOST-2.0 declared client fold subset. One read protocol; documents
    /// are named like Facts and Graph rather than a second sync surface.
    pub const CLIENT_SUBSET: [Self; 5] = Self::ALL;

    pub fn surfaces(self) -> &'static [crate::partial_fold::FoldSurface] {
        match self {
            Self::Facts => crate::partial_fold::FACTS_SURFACES,
            Self::Graph => crate::partial_fold::GRAPH_SURFACES,
            Self::Algebra => crate::partial_fold::ALGEBRA_SURFACES,
            Self::History => crate::partial_fold::HISTORY_SURFACES,
            Self::Documents => crate::partial_fold::DOCUMENTS_SURFACES,
        }
    }
}

impl fmt::Display for FoldId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Facts => "facts",
            Self::Graph => "graph",
            Self::Algebra => "algebra",
            Self::History => "history",
            Self::Documents => "documents",
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct FoldVersion(u32);

impl FoldVersion {
    pub fn new(version: u32) -> Result<Self, FoldError> {
        if version == 0 {
            Err(FoldError::InvalidFoldVersion)
        } else {
            Ok(Self(version))
        }
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for FoldVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// The law shared by every fold: ordered entries denote composable state
/// transitions. This does not imply that two final snapshot states can merge.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionComposition {
    OrderedStateTransitions,
}

/// Identity of an optional sufficient segment summary.
///
/// A descriptor is present only when a fold has an implementation whose
/// identity, combination, and application laws are tested. Raw snapshot state
/// is never treated as a summary by default.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SegmentSummaryDescriptor {
    pub summary_id: String,
    pub version: u32,
}

impl SegmentSummaryDescriptor {
    pub fn new(summary_id: impl Into<String>, version: u32) -> Result<Self, FoldError> {
        let summary_id = summary_id.into();
        if summary_id.trim().is_empty() || version == 0 {
            return Err(FoldError::InvalidCapability(
                "segment summary requires a non-blank id and non-zero version".to_string(),
            ));
        }
        Ok(Self {
            summary_id,
            version,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetractionStrategy {
    Inverse,
    SupportCounted,
    BoundedReplay,
}

impl Default for RetractionStrategy {
    fn default() -> Self {
        Self::BoundedReplay
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FoldCapabilities {
    pub transition_composition: TransitionComposition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_summary: Option<SegmentSummaryDescriptor>,
    #[serde(default)]
    pub retraction: RetractionStrategy,
}

impl FoldCapabilities {
    pub const fn ordered_transitions_only() -> Self {
        Self {
            transition_composition: TransitionComposition::OrderedStateTransitions,
            segment_summary: None,
            retraction: RetractionStrategy::BoundedReplay,
        }
    }
}

impl Default for FoldCapabilities {
    fn default() -> Self {
        Self::ordered_transitions_only()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FoldDescriptor {
    pub fold_id: FoldId,
    pub fold_version: FoldVersion,
    pub capabilities: FoldCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valuation_profile: Option<algebra::AlgebraValuationProfile>,
}

impl FoldDescriptor {
    pub fn new(fold_id: FoldId, fold_version: FoldVersion, capabilities: FoldCapabilities) -> Self {
        Self {
            fold_id,
            fold_version,
            capabilities,
            valuation_profile: None,
        }
    }

    pub fn with_valuation_profile(
        mut self,
        valuation_profile: algebra::AlgebraValuationProfile,
    ) -> Self {
        self.valuation_profile = Some(valuation_profile);
        self
    }
}

/// A sufficient, optionally compact representation of an ordered segment.
///
/// Implementing this trait is an earned capability. Implementations must prove
/// identity and associativity for `combine`, plus equivalence between
/// `apply` and replaying the summarized records against any admitted prefix.
pub trait SegmentSummary<F: Fold>: Clone {
    fn descriptor() -> SegmentSummaryDescriptor;
    fn identity() -> Self;
    fn summarize(records: &[LogRecord]) -> Result<Self, FoldError>;
    fn combine(&self, later: &Self) -> Result<Self, FoldError>;
    fn apply(&self, fold: &mut F) -> Result<(), FoldError>;
}

/// A hash-bound interval whose entries act as one state transition.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SegmentTransition {
    pub log_id: LogId,
    pub from_exclusive: LogPosition,
    pub through: LogPosition,
    pub from_content_hash: ContentHash,
    pub through_content_hash: ContentHash,
}

impl SegmentTransition {
    pub fn from_log(
        log: &PhysicalLog,
        from_exclusive: LogPosition,
        through: LogPosition,
    ) -> Result<Self, FoldError> {
        if from_exclusive > through {
            return Err(FoldError::TransitionMismatch(format!(
                "segment starts after its end: {from_exclusive} > {through}"
            )));
        }
        let from_content_hash = log
            .pinned_prefix(from_exclusive)
            .map_err(|error| FoldError::Physical(error.to_string()))?
            .content_hash;
        let through_content_hash = log
            .pinned_prefix(through)
            .map_err(|error| FoldError::Physical(error.to_string()))?
            .content_hash;
        Ok(Self {
            log_id: log.log_id(),
            from_exclusive,
            through,
            from_content_hash,
            through_content_hash,
        })
    }

    pub fn identity(log: &PhysicalLog, at: LogPosition) -> Result<Self, FoldError> {
        Self::from_log(log, at, at)
    }

    pub fn compose(&self, later: &Self) -> Result<Self, FoldError> {
        self.validate_shape()?;
        later.validate_shape()?;
        if self.log_id != later.log_id
            || self.through != later.from_exclusive
            || self.through_content_hash != later.from_content_hash
        {
            return Err(FoldError::TransitionMismatch(
                "segment transitions do not meet at the same pinned log boundary".to_string(),
            ));
        }
        Ok(Self {
            log_id: self.log_id,
            from_exclusive: self.from_exclusive,
            through: later.through,
            from_content_hash: self.from_content_hash.clone(),
            through_content_hash: later.through_content_hash.clone(),
        })
    }

    fn validate_shape(&self) -> Result<(), FoldError> {
        if self.from_exclusive > self.through {
            return Err(FoldError::TransitionMismatch(format!(
                "segment starts after its end: {} > {}",
                self.from_exclusive, self.through
            )));
        }
        if self.from_exclusive == self.through
            && self.from_content_hash != self.through_content_hash
        {
            return Err(FoldError::TransitionMismatch(
                "identity segment has different boundary hashes".to_string(),
            ));
        }
        Ok(())
    }

    fn validate(&self, log: &PhysicalLog) -> Result<(), FoldError> {
        self.validate_shape()?;
        let canonical = Self::from_log(log, self.from_exclusive, self.through)?;
        if &canonical == self {
            Ok(())
        } else {
            Err(FoldError::TransitionMismatch(
                "segment transition does not match the pinned physical log".to_string(),
            ))
        }
    }
}

/// State-specific behavior implemented by Facts, Graph, Algebra, History, and Documents.
///
/// Implementations validate an entry completely before mutating state. A
/// schema refusal is returned as [`FoldConsumeError::Refused`]; the runtime
/// advances past and quarantines that entry. Fatal errors stop replay.
pub trait Fold {
    fn id(&self) -> FoldId;
    fn version(&self) -> FoldVersion;
    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }
    fn descriptor(&self) -> FoldDescriptor {
        FoldDescriptor::new(self.id(), self.version(), self.capabilities())
    }
    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError>;
    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError>;
    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError>;
    fn checkpoint(&self) -> Result<Vec<u8>, FoldError>;
    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError>;
    fn reset(&mut self) -> Result<(), FoldError>;
    fn advance_without_absorb(&mut self, _entry: &LogRecord) -> Result<(), FoldError> {
        Ok(())
    }
}

/// Half-open physical-log interval `(from_exclusive, through]`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LogInterval {
    pub from_exclusive: LogPosition,
    pub through: LogPosition,
}

impl LogInterval {
    pub fn new(from_exclusive: LogPosition, through: LogPosition) -> Result<Self, FoldError> {
        if from_exclusive > through {
            return Err(FoldError::InvalidRetraction(format!(
                "interval starts after {from_exclusive} but ends at {through}"
            )));
        }
        Ok(Self {
            from_exclusive,
            through,
        })
    }

    pub fn contains(self, position: LogPosition) -> bool {
        self.from_exclusive < position && position <= self.through
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetractOutcome {
    Retracted,
    RequiresRecompute { from_interval: LogInterval },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetractionReceipt {
    pub position: LogPosition,
    pub strategy: RetractionStrategy,
    pub outcome: RetractOutcome,
    pub replay: ApplyReport,
    pub newly_retracted: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FoldCheckpoint {
    pub protocol_version: u32,
    pub log_id: LogId,
    pub fold_id: FoldId,
    pub fold_version: FoldVersion,
    pub fold_descriptor: FoldDescriptor,
    pub through: LogPosition,
    pub log_content_hash: ContentHash,
    pub state: Vec<u8>,
    pub quarantine: Vec<QuarantineReceipt>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub retracted_positions: BTreeSet<LogPosition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retraction_receipts: Vec<RetractionReceipt>,
    pub checkpoint_hash: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointRejectCode {
    ProtocolVersion,
    LogIdentity,
    FoldIdentity,
    FoldVersion,
    FoldCapabilities,
    Position,
    LogContentHash,
    CheckpointHash,
    QuarantineMetadata,
    RetractionMetadata,
    StateDecode,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CheckpointRejectionReceipt {
    pub code: CheckpointRejectCode,
    pub fold_id: FoldId,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApplyReport {
    pub applied: usize,
    pub quarantined: usize,
    pub retracted: usize,
    pub through: LogPosition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RestoreDisposition {
    Restored {
        tail: ApplyReport,
    },
    Rebuilt {
        rejection: CheckpointRejectionReceipt,
        replay: ApplyReport,
    },
}

#[derive(Debug)]
pub enum FoldError {
    InvalidFoldVersion,
    InvalidCapability(String),
    LogIdentity {
        expected: LogId,
        actual: LogId,
    },
    PositionAhead {
        requested: LogPosition,
        available: LogPosition,
    },
    PositionRegression {
        requested: LogPosition,
        current: LogPosition,
    },
    PositionGap {
        expected: LogPosition,
        actual: LogPosition,
    },
    TransitionMismatch(String),
    InvalidRetraction(String),
    QueryAhead {
        requested: LogPosition,
        current: LogPosition,
    },
    /// A client pin names a log prefix the log does not produce.
    PinHashMismatch {
        through: LogPosition,
        expected: String,
        actual: String,
    },
    FatalConsume(String),
    Codec(String),
    Physical(String),
}

impl fmt::Display for FoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFoldVersion => formatter.write_str("fold version must be non-zero"),
            Self::InvalidCapability(reason) => {
                write!(formatter, "fold capability invalid: {reason}")
            }
            Self::LogIdentity { expected, actual } => {
                write!(formatter, "fold is bound to log {expected}, not {actual}")
            }
            Self::PositionAhead {
                requested,
                available,
            } => write!(
                formatter,
                "fold position {requested} is beyond available log position {available}"
            ),
            Self::PositionRegression { requested, current } => write!(
                formatter,
                "fold cannot move backward from position {current} to {requested}"
            ),
            Self::PositionGap { expected, actual } => write!(
                formatter,
                "fold expected contiguous position {expected}, found {actual}"
            ),
            Self::TransitionMismatch(reason) => {
                write!(formatter, "fold transition mismatch: {reason}")
            }
            Self::InvalidRetraction(reason) => {
                write!(formatter, "fold retraction invalid: {reason}")
            }
            Self::QueryAhead { requested, current } => write!(
                formatter,
                "fold cannot answer position {requested}; it is applied through {current}"
            ),
            Self::PinHashMismatch {
                through,
                expected,
                actual,
            } => write!(
                formatter,
                "client pin for prefix {through} names hash {expected}, but the log produces {actual}"
            ),
            Self::FatalConsume(reason) => write!(formatter, "fold replay stopped: {reason}"),
            Self::Codec(reason) => write!(formatter, "fold checkpoint codec failed: {reason}"),
            Self::Physical(reason) => write!(formatter, "fold log access failed: {reason}"),
        }
    }
}

impl Error for FoldError {}

#[derive(Debug)]
pub struct FoldRuntime<F> {
    fold: F,
    log_id: LogId,
    through: LogPosition,
    quarantine: Vec<QuarantineReceipt>,
    retracted_positions: BTreeSet<LogPosition>,
    retraction_receipts: BTreeMap<LogPosition, RetractionReceipt>,
}

impl<F: Fold> FoldRuntime<F> {
    pub fn new(fold: F, log_id: LogId) -> Result<Self, FoldError> {
        if fold.version().get() == 0 {
            return Err(FoldError::InvalidFoldVersion);
        }
        Ok(Self {
            fold,
            log_id,
            through: LogPosition::ZERO,
            quarantine: Vec::new(),
            retracted_positions: BTreeSet::new(),
            retraction_receipts: BTreeMap::new(),
        })
    }

    pub fn for_log(fold: F, log: &PhysicalLog) -> Result<Self, FoldError> {
        Self::new(fold, log.log_id())
    }

    pub fn id(&self) -> FoldId {
        self.fold.id()
    }

    pub fn version(&self) -> FoldVersion {
        self.fold.version()
    }

    pub fn descriptor(&self) -> FoldDescriptor {
        self.fold.descriptor()
    }

    pub fn current_position(&self) -> LogPosition {
        self.through
    }

    pub fn quarantine_receipts(&self) -> &[QuarantineReceipt] {
        &self.quarantine
    }

    pub fn retracted_positions(&self) -> &BTreeSet<LogPosition> {
        &self.retracted_positions
    }

    pub fn retraction_receipts(&self) -> &BTreeMap<LogPosition, RetractionReceipt> {
        &self.retraction_receipts
    }

    pub fn inner(&self) -> &F {
        &self.fold
    }

    pub fn inner_mut(&mut self) -> &mut F {
        &mut self.fold
    }

    pub fn log_id(&self) -> theorem_oplog::LogId {
        self.log_id
    }

    pub fn into_inner(self) -> F {
        self.fold
    }

    pub fn apply_range(
        &mut self,
        log: &PhysicalLog,
        through: LogPosition,
    ) -> Result<ApplyReport, FoldError> {
        self.require_log(log)?;
        let available = log.latest_position();
        if through > available {
            return Err(FoldError::PositionAhead {
                requested: through,
                available,
            });
        }
        if through < self.through {
            return Err(FoldError::PositionRegression {
                requested: through,
                current: self.through,
            });
        }

        let mut report = ApplyReport {
            through: self.through,
            ..ApplyReport::default()
        };
        for record in log
            .records_after(self.through)
            .iter()
            .take_while(|record| record.position <= through)
        {
            let expected = self
                .through
                .next()
                .ok_or_else(|| FoldError::FatalConsume("position space exhausted".to_string()))?;
            if record.log_id != self.log_id {
                return Err(FoldError::LogIdentity {
                    expected: self.log_id,
                    actual: record.log_id,
                });
            }
            if record.position != expected {
                return Err(FoldError::PositionGap {
                    expected,
                    actual: record.position,
                });
            }

            if self.retracted_positions.contains(&record.position) {
                self.fold.advance_without_absorb(record)?;
                report.retracted += 1;
            } else {
                match self.fold.apply(record) {
                    Ok(()) => report.applied += 1,
                    Err(FoldConsumeError::Refused(refusal)) => {
                        self.quarantine.push(QuarantineReceipt::from_refusal(
                            self.id(),
                            self.version(),
                            record,
                            refusal,
                        ));
                        report.quarantined += 1;
                    }
                    Err(FoldConsumeError::Fatal(reason)) => {
                        return Err(FoldError::FatalConsume(reason));
                    }
                }
            }
            self.through = record.position;
            report.through = self.through;
        }
        if self.through != through {
            return Err(FoldError::PositionGap {
                expected: through,
                actual: self.through,
            });
        }
        Ok(report)
    }

    pub fn apply_all(&mut self, log: &PhysicalLog) -> Result<ApplyReport, FoldError> {
        self.apply_range(log, log.latest_position())
    }

    /// Apply one hash-bound ordered segment from the runtime's current point.
    pub fn apply_transition(
        &mut self,
        log: &PhysicalLog,
        transition: &SegmentTransition,
    ) -> Result<ApplyReport, FoldError> {
        self.require_log(log)?;
        transition.validate(log)?;
        if transition.from_exclusive != self.through {
            return Err(FoldError::TransitionMismatch(format!(
                "runtime is at {}, but segment starts after {}",
                self.through, transition.from_exclusive
            )));
        }
        self.apply_range(log, transition.through)
    }

    pub fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        self.require_applied(position)?;
        self.fold.query(query, position)
    }

    pub fn upquery(
        &mut self,
        log: &PhysicalLog,
        key: &Value,
        position: LogPosition,
    ) -> Result<Value, FoldError> {
        self.require_log(log)?;
        if position > self.through {
            self.apply_range(log, position)?;
        }
        let count =
            usize::try_from(position.0).map_err(|error| FoldError::Physical(error.to_string()))?;
        let pinned = &log.records()[..count];
        if self.retracted_positions.is_empty() {
            self.fold.upquery(key, position, pinned)
        } else {
            let retained = pinned
                .iter()
                .filter(|record| !self.retracted_positions.contains(&record.position))
                .cloned()
                .collect::<Vec<_>>();
            self.fold.upquery(key, position, &retained)
        }
    }

    pub fn checkpoint(&self, log: &PhysicalLog) -> Result<FoldCheckpoint, FoldError> {
        self.require_log(log)?;
        let prefix = log
            .pinned_prefix(self.through)
            .map_err(|error| FoldError::Physical(error.to_string()))?;
        let mut checkpoint = FoldCheckpoint {
            protocol_version: FOLD_CHECKPOINT_PROTOCOL_VERSION,
            log_id: self.log_id,
            fold_id: self.id(),
            fold_version: self.version(),
            fold_descriptor: self.descriptor(),
            through: self.through,
            log_content_hash: prefix.content_hash,
            state: self.fold.checkpoint()?,
            quarantine: self.quarantine.clone(),
            retracted_positions: self.retracted_positions.clone(),
            retraction_receipts: self.retraction_receipts.values().cloned().collect(),
            checkpoint_hash: String::new(),
        };
        checkpoint.checkpoint_hash = checkpoint_hash(&checkpoint)?;
        Ok(checkpoint)
    }

    pub fn restore_or_rebuild(
        &mut self,
        checkpoint: &FoldCheckpoint,
        log: &PhysicalLog,
    ) -> Result<RestoreDisposition, FoldError> {
        let rejection = self.validate_checkpoint(checkpoint, log).err();
        if let Some(rejection) = rejection {
            let replay = self.full_rebuild(log)?;
            return Ok(RestoreDisposition::Rebuilt { rejection, replay });
        }

        if let Err(error) = self.fold.restore(&checkpoint.state) {
            let rejection = self.rejection(
                CheckpointRejectCode::StateDecode,
                format!("fold state decode failed: {error}"),
            );
            let replay = self.full_rebuild(log)?;
            return Ok(RestoreDisposition::Rebuilt { rejection, replay });
        }
        self.through = checkpoint.through;
        self.quarantine.clone_from(&checkpoint.quarantine);
        self.retracted_positions
            .clone_from(&checkpoint.retracted_positions);
        self.retraction_receipts = checkpoint
            .retraction_receipts
            .iter()
            .cloned()
            .map(|receipt| (receipt.position, receipt))
            .collect();
        let tail = self.apply_all(log)?;
        Ok(RestoreDisposition::Restored { tail })
    }

    pub fn full_rebuild(&mut self, log: &PhysicalLog) -> Result<ApplyReport, FoldError> {
        self.require_log(log)?;
        self.fold.reset()?;
        self.through = LogPosition::ZERO;
        self.quarantine.clear();
        self.apply_all(log)
    }

    pub fn retract_position(
        &mut self,
        log: &PhysicalLog,
        position: LogPosition,
        checkpoints: &[FoldCheckpoint],
    ) -> Result<RetractionReceipt, FoldError> {
        self.require_log(log)?;
        if position == LogPosition::ZERO || position > self.through {
            return Err(FoldError::InvalidRetraction(format!(
                "position {position} is outside the applied interval (0, {}]",
                self.through
            )));
        }
        let strategy = self.fold.capabilities().retraction;
        if strategy != RetractionStrategy::BoundedReplay {
            return Err(FoldError::InvalidRetraction(format!(
                "replay retraction cannot execute a {strategy:?} fold strategy"
            )));
        }
        if let Some(previous) = self.retraction_receipts.get(&position) {
            let mut receipt = previous.clone();
            receipt.newly_retracted = false;
            receipt.replay = ApplyReport {
                through: self.through,
                ..ApplyReport::default()
            };
            return Ok(receipt);
        }

        let through = self.through;
        let mut overlay = self.retracted_positions.clone();
        overlay.insert(position);
        let mut candidates = checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.through < position)
            .filter(|checkpoint| self.validate_checkpoint(checkpoint, log).is_ok())
            .filter(|checkpoint| checkpoint_overlay_matches(checkpoint, &overlay))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|checkpoint| std::cmp::Reverse(checkpoint.through));

        let mut from_exclusive = LogPosition::ZERO;
        let mut restored = false;
        for checkpoint in candidates {
            self.fold.reset()?;
            if self.fold.restore(&checkpoint.state).is_ok() {
                from_exclusive = checkpoint.through;
                self.through = checkpoint.through;
                self.quarantine.clone_from(&checkpoint.quarantine);
                restored = true;
                break;
            }
        }
        if !restored {
            self.fold.reset()?;
            self.through = LogPosition::ZERO;
            self.quarantine.clear();
        }
        self.retracted_positions = overlay;
        let replay = self.apply_range(log, through)?;
        let interval = LogInterval::new(from_exclusive, through)?;
        debug_assert!(interval.contains(position));
        let receipt = RetractionReceipt {
            position,
            strategy,
            outcome: RetractOutcome::RequiresRecompute {
                from_interval: interval,
            },
            replay,
            newly_retracted: true,
        };
        self.retraction_receipts.insert(position, receipt.clone());
        Ok(receipt)
    }

    fn validate_checkpoint(
        &self,
        checkpoint: &FoldCheckpoint,
        log: &PhysicalLog,
    ) -> Result<(), CheckpointRejectionReceipt> {
        if checkpoint.protocol_version != FOLD_CHECKPOINT_PROTOCOL_VERSION {
            return Err(self.rejection(
                CheckpointRejectCode::ProtocolVersion,
                format!(
                    "checkpoint protocol {} is not {}",
                    checkpoint.protocol_version, FOLD_CHECKPOINT_PROTOCOL_VERSION
                ),
            ));
        }
        if log.log_id() != self.log_id || checkpoint.log_id != self.log_id {
            return Err(self.rejection(
                CheckpointRejectCode::LogIdentity,
                format!(
                    "checkpoint/log identities {}/{} do not match runtime {}",
                    checkpoint.log_id,
                    log.log_id(),
                    self.log_id
                ),
            ));
        }
        if checkpoint.fold_id != self.id() {
            return Err(self.rejection(
                CheckpointRejectCode::FoldIdentity,
                format!(
                    "checkpoint fold {} does not match {}",
                    checkpoint.fold_id,
                    self.id()
                ),
            ));
        }
        if checkpoint.fold_version != self.version() {
            return Err(self.rejection(
                CheckpointRejectCode::FoldVersion,
                format!(
                    "checkpoint fold version {} does not match {}",
                    checkpoint.fold_version,
                    self.version()
                ),
            ));
        }
        if checkpoint.fold_descriptor != self.descriptor() {
            return Err(self.rejection(
                CheckpointRejectCode::FoldCapabilities,
                "checkpoint fold descriptor does not match runtime capabilities",
            ));
        }
        if checkpoint.through > log.latest_position() {
            return Err(self.rejection(
                CheckpointRejectCode::Position,
                format!(
                    "checkpoint position {} is beyond log position {}",
                    checkpoint.through,
                    log.latest_position()
                ),
            ));
        }
        if checkpoint
            .retracted_positions
            .iter()
            .any(|position| *position == LogPosition::ZERO || *position > checkpoint.through)
        {
            return Err(self.rejection(
                CheckpointRejectCode::Position,
                "checkpoint retraction overlay is outside its pinned prefix",
            ));
        }
        let mut receipt_positions = BTreeSet::new();
        for receipt in &checkpoint.retraction_receipts {
            let position_valid = receipt.position > LogPosition::ZERO
                && receipt.position <= checkpoint.through
                && checkpoint.retracted_positions.contains(&receipt.position)
                && receipt_positions.insert(receipt.position);
            let strategy_valid =
                receipt.strategy == checkpoint.fold_descriptor.capabilities.retraction;
            let outcome_valid = match receipt.outcome {
                RetractOutcome::Retracted => receipt.strategy != RetractionStrategy::BoundedReplay,
                RetractOutcome::RequiresRecompute { from_interval } => {
                    receipt.strategy == RetractionStrategy::BoundedReplay
                        && from_interval.contains(receipt.position)
                        && from_interval.through <= checkpoint.through
                }
            };
            let replay_valid = receipt.replay.through <= checkpoint.through;
            if !position_valid || !strategy_valid || !outcome_valid || !replay_valid {
                return Err(self.rejection(
                    CheckpointRejectCode::RetractionMetadata,
                    format!(
                        "retraction receipt at position {} is incompatible with its checkpoint",
                        receipt.position
                    ),
                ));
            }
        }
        if receipt_positions != checkpoint.retracted_positions {
            return Err(self.rejection(
                CheckpointRejectCode::RetractionMetadata,
                "checkpoint retraction receipts do not exactly cover the overlay",
            ));
        }
        let prefix = log
            .pinned_prefix(checkpoint.through)
            .map_err(|error| self.rejection(CheckpointRejectCode::Position, error.to_string()))?;
        if checkpoint.log_content_hash != prefix.content_hash {
            return Err(self.rejection(
                CheckpointRejectCode::LogContentHash,
                "checkpoint is not anchored to the requested log prefix",
            ));
        }
        let expected_hash = checkpoint_hash(checkpoint).map_err(|error| {
            self.rejection(CheckpointRejectCode::CheckpointHash, error.to_string())
        })?;
        if checkpoint.checkpoint_hash != expected_hash {
            return Err(self.rejection(
                CheckpointRejectCode::CheckpointHash,
                "checkpoint envelope checksum mismatch",
            ));
        }
        for receipt in &checkpoint.quarantine {
            let valid_position =
                receipt.position > LogPosition::ZERO && receipt.position <= checkpoint.through;
            let record = valid_position
                .then(|| log.records().get((receipt.position.0 - 1) as usize))
                .flatten();
            let valid_record = record.is_some_and(|record| {
                receipt.log_id == self.log_id
                    && receipt.fold_id == self.id()
                    && receipt.fold_version == self.version()
                    && receipt.record_id == record.record_id
                    && receipt.record_hash == record.record_hash
                    && receipt.op_digest == record.op_digest
            });
            if !valid_record || checkpoint.retracted_positions.contains(&receipt.position) {
                return Err(self.rejection(
                    CheckpointRejectCode::QuarantineMetadata,
                    format!(
                        "quarantine receipt at position {} does not match its log record",
                        receipt.position
                    ),
                ));
            }
        }
        Ok(())
    }

    fn require_log(&self, log: &PhysicalLog) -> Result<(), FoldError> {
        if log.log_id() == self.log_id {
            Ok(())
        } else {
            Err(FoldError::LogIdentity {
                expected: self.log_id,
                actual: log.log_id(),
            })
        }
    }

    fn require_applied(&self, position: LogPosition) -> Result<(), FoldError> {
        if position <= self.through {
            Ok(())
        } else {
            Err(FoldError::QueryAhead {
                requested: position,
                current: self.through,
            })
        }
    }

    fn rejection(
        &self,
        code: CheckpointRejectCode,
        detail: impl Into<String>,
    ) -> CheckpointRejectionReceipt {
        CheckpointRejectionReceipt {
            code,
            fold_id: self.id(),
            detail: detail.into(),
        }
    }
}

/// The log prefix a client's answer must come from.
///
/// Clients hold hashes, not positions, across a sync: a [`FoldCheckpoint`]
/// carries `through` plus `log_content_hash`, and the FLEET-RECEIPTS object
/// (`docs/plans/HANDOFF-FLEET-RECEIPTS-1.0.md`) names the same pair. A pin is
/// the client's handle on a prefix, so a by-hash read can refuse to answer from
/// a prefix the client did not name.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ClientPin {
    pub through: LogPosition,
    pub log_content_hash: ContentHash,
}

impl ClientPin {
    pub fn new(through: LogPosition, log_content_hash: ContentHash) -> Self {
        Self {
            through,
            log_content_hash,
        }
    }

    /// The pin a checkpoint names: what a client resumes from.
    pub fn of_checkpoint(checkpoint: &FoldCheckpoint) -> Self {
        Self {
            through: checkpoint.through,
            log_content_hash: checkpoint.log_content_hash.clone(),
        }
    }
}

/// The client fold ABI (SPEC-THEOREM-APP-HOST-2.0). See the module docs for
/// the declared subset and the four operations; this type adds no fold state of
/// its own and wraps the one [`FoldRuntime`] so the client and the server share
/// replay, checkpoint, and quarantine semantics by construction.
#[derive(Debug)]
pub struct LogFold<F: Fold> {
    runtime: FoldRuntime<F>,
}

impl<F: Fold> LogFold<F> {
    pub fn new(fold: F, log_id: LogId) -> Result<Self, FoldError> {
        Ok(Self {
            runtime: FoldRuntime::new(fold, log_id)?,
        })
    }

    pub fn for_log(fold: F, log: &PhysicalLog) -> Result<Self, FoldError> {
        Ok(Self {
            runtime: FoldRuntime::for_log(fold, log)?,
        })
    }

    /// The underlying runtime. Reads and retraction stay on it; the client ABI
    /// only names the four client operations, it does not hide the runtime.
    pub fn runtime(&self) -> &FoldRuntime<F> {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut FoldRuntime<F> {
        &mut self.runtime
    }

    pub fn fold_id(&self) -> FoldId {
        self.runtime.id()
    }

    pub fn position(&self) -> LogPosition {
        self.runtime.current_position()
    }

    /// Checkpoint load: adopt `checkpoint` and apply its tail in one step.
    ///
    /// A checkpoint that fails validation or state decode is rebuilt from the
    /// log, and the [`RestoreDisposition`] carries the rejection receipt.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &FoldCheckpoint,
        log: &PhysicalLog,
    ) -> Result<RestoreDisposition, FoldError> {
        self.runtime.restore_or_rebuild(checkpoint, log)
    }

    /// Tail apply: extend the applied prefix to the log head.
    pub fn apply_tail(&mut self, log: &PhysicalLog) -> Result<ApplyReport, FoldError> {
        self.runtime.apply_all(log)
    }

    /// Upquery by hash: answer `key` at the prefix `pin` names.
    ///
    /// Refuses with [`FoldError::PinHashMismatch`] when the log does not
    /// produce the pinned prefix hash, so an answer can only come from the
    /// prefix the client holds a hash for.
    pub fn upquery_by_hash(
        &mut self,
        log: &PhysicalLog,
        key: &Value,
        pin: &ClientPin,
    ) -> Result<Value, FoldError> {
        self.require_pin(log, pin)?;
        self.runtime.upquery(log, key, pin.through)
    }

    /// Refuse unless `log` really produces the prefix hash `pin` names.
    pub fn require_pin(&self, log: &PhysicalLog, pin: &ClientPin) -> Result<(), FoldError> {
        let prefix = log
            .pinned_prefix(pin.through)
            .map_err(|error| FoldError::Physical(error.to_string()))?;
        if prefix.content_hash != pin.log_content_hash {
            return Err(FoldError::PinHashMismatch {
                through: pin.through,
                expected: pin.log_content_hash.to_string(),
                actual: prefix.content_hash.to_string(),
            });
        }
        Ok(())
    }

    /// Proposal submission: hand `proposal` to the one committer.
    ///
    /// Both outcomes are receipts — a commit extends the log (apply the tail to
    /// read it) and a refusal carries the conflicting positions for the retry.
    /// The committer must hold this runtime's log; a mismatch refuses with
    /// [`FoldError::LogIdentity`] rather than submitting a proposal built
    /// against a different prefix.
    pub fn submit_proposal(
        &mut self,
        committer: &mut ProposalCommitter,
        proposal: Proposal,
    ) -> Result<CommitOutcome, FoldError> {
        if committer.log().log_id() != self.runtime.log_id() {
            return Err(FoldError::LogIdentity {
                expected: self.runtime.log_id(),
                actual: committer.log().log_id(),
            });
        }
        committer
            .commit(proposal)
            .map_err(|error| FoldError::Physical(error.to_string()))
    }
}

#[derive(Serialize)]
struct CheckpointHashInput<'a> {
    protocol_version: u32,
    log_id: LogId,
    fold_id: FoldId,
    fold_version: FoldVersion,
    fold_descriptor: &'a FoldDescriptor,
    through: LogPosition,
    log_content_hash: &'a ContentHash,
    state: &'a [u8],
    quarantine: &'a [QuarantineReceipt],
    retracted_positions: &'a BTreeSet<LogPosition>,
    retraction_receipts: &'a [RetractionReceipt],
}

fn checkpoint_hash(checkpoint: &FoldCheckpoint) -> Result<String, FoldError> {
    let payload = serde_json::to_vec(&CheckpointHashInput {
        protocol_version: checkpoint.protocol_version,
        log_id: checkpoint.log_id,
        fold_id: checkpoint.fold_id,
        fold_version: checkpoint.fold_version,
        fold_descriptor: &checkpoint.fold_descriptor,
        through: checkpoint.through,
        log_content_hash: &checkpoint.log_content_hash,
        state: &checkpoint.state,
        quarantine: &checkpoint.quarantine,
        retracted_positions: &checkpoint.retracted_positions,
        retraction_receipts: &checkpoint.retraction_receipts,
    })
    .map_err(|error| FoldError::Codec(error.to_string()))?;
    Ok(blake3::hash(&payload).to_hex().to_string())
}

fn checkpoint_overlay_matches(
    checkpoint: &FoldCheckpoint,
    overlay: &BTreeSet<LogPosition>,
) -> bool {
    checkpoint.retracted_positions
        == overlay
            .iter()
            .copied()
            .filter(|position| *position <= checkpoint.through)
            .collect()
}
