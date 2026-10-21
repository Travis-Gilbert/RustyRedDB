use crate::physical::{internal_provenance, validate_v3_provenance};
use crate::{
    read_write_overlap, AppendReceipt, DeploymentId, EntryProvenance, LogIntent, LogPosition,
    OpDigest, PhysicalLog, PhysicalLogError, ReadSet, WriteSet, PHYSICAL_LOG_PROTOCOL_V2,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// One evaluation of the canonical read/write overlap predicate.
///
/// The references point at already-normalized sets. Consumers may attach
/// their own action label or pinned position without reinterpreting either
/// set. Keeping this view borrowed avoids cloning access paths on the OCC hot
/// path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OverlapEvaluation<'a> {
    pub read_set: &'a ReadSet,
    pub write_set: &'a WriteSet,
    pub overlaps: bool,
}

/// Evaluate normalized access sets through the single overlap kernel.
pub fn evaluate_overlap<'a>(
    read_set: &'a ReadSet,
    write_set: &'a WriteSet,
) -> OverlapEvaluation<'a> {
    OverlapEvaluation {
        read_set,
        write_set,
        overlaps: read_write_overlap(read_set, write_set),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Proposal {
    pub begin_position: LogPosition,
    pub read_set: ReadSet,
    pub write_set: WriteSet,
    pub op_digest: OpDigest,
    pub intent: LogIntent,
}

impl Proposal {
    pub fn new(
        begin_position: LogPosition,
        read_set: ReadSet,
        write_set: WriteSet,
        op_digest: OpDigest,
        intent: LogIntent,
    ) -> Self {
        Self {
            begin_position,
            read_set,
            write_set,
            op_digest,
            intent,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalCode {
    OccConflict,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RefusalReceipt {
    pub deployment_id: DeploymentId,
    pub code: RefusalCode,
    pub proposal_digest: OpDigest,
    pub begin_position: LogPosition,
    pub observed_through: LogPosition,
    pub conflicting_positions: Vec<LogPosition>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitReceipt {
    pub deployment_id: DeploymentId,
    pub append: AppendReceipt,
    pub attempts: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitOutcome {
    Committed(CommitReceipt),
    Refused(RefusalReceipt),
}

#[derive(Debug)]
pub struct ProposalCommitter {
    log: PhysicalLog,
}

impl ProposalCommitter {
    pub fn new(log: PhysicalLog) -> Self {
        Self { log }
    }

    pub fn log(&self) -> &PhysicalLog {
        &self.log
    }

    pub fn into_log(self) -> PhysicalLog {
        self.log
    }

    /// Evaluate the exact predicate used by optimistic concurrency control.
    ///
    /// Exposing this borrowed decision lets differential tests compare OCC to
    /// cache, subscription, and hotness consumers without duplicating OCC's
    /// interpretation of access sets.
    pub fn evaluate_occ<'a>(
        read_set: &'a ReadSet,
        write_set: &'a WriteSet,
    ) -> OverlapEvaluation<'a> {
        evaluate_overlap(read_set, write_set)
    }

    pub fn commit(&mut self, proposal: Proposal) -> Result<CommitOutcome, CommitError> {
        self.commit_with_provenance(proposal, internal_provenance())
    }

    /// Commit through an already-admitted, typed provenance context.
    ///
    /// The proposal remains untrusted semantic data. Runtime capability
    /// adapters construct `provenance` separately from their admitted
    /// function envelope and writer identity.
    pub fn commit_with_provenance(
        &mut self,
        proposal: Proposal,
        provenance: EntryProvenance,
    ) -> Result<CommitOutcome, CommitError> {
        self.commit_once(proposal, &provenance, 1)
    }

    /// Re-run a deterministic mutation after a typed OCC refusal.
    ///
    /// `rerun` receives the newest committed position and the refusal that
    /// invalidated the previous proposal. It must rebuild the proposal from
    /// that position; no stale proposal is silently resubmitted.
    pub fn commit_with_retry<F>(
        &mut self,
        proposal: Proposal,
        max_retries: u32,
        rerun: F,
    ) -> Result<CommitOutcome, CommitError>
    where
        F: FnMut(LogPosition, &RefusalReceipt) -> Result<Proposal, CommitError>,
    {
        self.commit_with_retry_and_provenance(proposal, internal_provenance(), max_retries, rerun)
    }

    pub fn commit_with_retry_and_provenance<F>(
        &mut self,
        mut proposal: Proposal,
        provenance: EntryProvenance,
        max_retries: u32,
        mut rerun: F,
    ) -> Result<CommitOutcome, CommitError>
    where
        F: FnMut(LogPosition, &RefusalReceipt) -> Result<Proposal, CommitError>,
    {
        let mut attempts = 1;
        loop {
            match self.commit_once(proposal, &provenance, attempts)? {
                CommitOutcome::Committed(receipt) => {
                    return Ok(CommitOutcome::Committed(receipt));
                }
                CommitOutcome::Refused(refusal) if attempts <= max_retries => {
                    let latest = self.log.latest_position();
                    proposal = rerun(latest, &refusal)?;
                    if proposal.begin_position != latest {
                        return Err(CommitError::RetryBeginMismatch {
                            proposed: proposal.begin_position,
                            expected: latest,
                        });
                    }
                    attempts += 1;
                }
                CommitOutcome::Refused(refusal) => {
                    return Ok(CommitOutcome::Refused(refusal));
                }
            }
        }
    }

    fn commit_once(
        &mut self,
        proposal: Proposal,
        provenance: &EntryProvenance,
        attempts: u32,
    ) -> Result<CommitOutcome, CommitError> {
        self.log.ensure_writable()?;
        validate_v3_provenance(provenance)?;
        if let Some(existing) = self.log.record_for_digest(&proposal.op_digest) {
            let provenance_matches = existing.protocol_version == PHYSICAL_LOG_PROTOCOL_V2
                || &existing.provenance == provenance;
            if existing.intent != proposal.intent
                || existing.write_set != proposal.write_set
                || !provenance_matches
            {
                return Err(PhysicalLogError::DuplicateDigestMismatch(proposal.op_digest).into());
            }
            let append = self
                .log
                .receipt_for_digest(&existing.op_digest)
                .expect("digest index and record lookup must agree");
            return Ok(CommitOutcome::Committed(CommitReceipt {
                deployment_id: append.deployment_id.clone(),
                append,
                attempts,
            }));
        }

        let latest = self.log.latest_position();
        if proposal.begin_position > latest {
            return Err(CommitError::BeginAhead {
                proposed: proposal.begin_position,
                latest,
            });
        }
        let conflicting_positions = self
            .log
            .records_after(proposal.begin_position)
            .iter()
            .filter(|record| Self::evaluate_occ(&proposal.read_set, &record.write_set).overlaps)
            .map(|record| record.position)
            .collect::<Vec<_>>();
        if !conflicting_positions.is_empty() {
            return Ok(CommitOutcome::Refused(RefusalReceipt {
                deployment_id: self.log.deployment_id().clone(),
                code: RefusalCode::OccConflict,
                proposal_digest: proposal.op_digest,
                begin_position: proposal.begin_position,
                observed_through: latest,
                conflicting_positions,
            }));
        }

        let append = self.log.append_with_provenance(
            proposal.intent,
            proposal.write_set,
            proposal.op_digest,
            provenance.clone(),
        )?;
        Ok(CommitOutcome::Committed(CommitReceipt {
            deployment_id: append.deployment_id.clone(),
            append,
            attempts,
        }))
    }
}

#[derive(Debug)]
pub enum CommitError {
    Physical(PhysicalLogError),
    BeginAhead {
        proposed: LogPosition,
        latest: LogPosition,
    },
    RetryBeginMismatch {
        proposed: LogPosition,
        expected: LogPosition,
    },
}

impl fmt::Display for CommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Physical(error) => write!(formatter, "proposal commit failed: {error}"),
            Self::BeginAhead { proposed, latest } => write!(
                formatter,
                "proposal begins at future position {proposed}; latest is {latest}"
            ),
            Self::RetryBeginMismatch { proposed, expected } => write!(
                formatter,
                "retried proposal begins at {proposed}; deterministic rerun must begin at {expected}"
            ),
        }
    }
}

impl Error for CommitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Physical(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PhysicalLogError> for CommitError {
    fn from(error: PhysicalLogError) -> Self {
        Self::Physical(error)
    }
}
