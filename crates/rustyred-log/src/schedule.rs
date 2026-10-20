//! Durable transactional scheduling over the physical operation log.
//!
//! A schedule transaction is one physical log record. Business outcomes and
//! every schedule change caused by that outcome therefore share the same
//! checksum and fsync boundary. Replay rebuilds runnable and held work solely
//! from those records; there is no sidecar queue or crontab.

use crate::{
    AccessPath, AppendReceipt, CommitOutcome, IntentKind, LogIntent, LogRecord, OpDigest, Proposal,
    ProposalCommitter, ReadSet, WriteSet,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const SCHEDULE_INTENT_NAMESPACE: &str = "theorem";
pub const SCHEDULE_INTENT_NAME: &str = "schedule_transaction";
pub const SCHEDULE_INTENT_VERSION: u32 = 1;
pub const SCHEDULED_MUTATION_INTENT_NAME: &str = "scheduled_mutation";

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ScheduleId(String);

impl ScheduleId {
    pub fn new(value: impl Into<String>) -> Result<Self, ScheduleError> {
        let value = value.into();
        require_identifier(&value, "schedule_id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ScheduleId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledFunctionKind {
    Mutation,
    Action,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ScheduleSource {
    Once,
    Loop { iteration: u64 },
    Cron { expression: String, occurrence: u64 },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleRow {
    pub schedule_id: ScheduleId,
    pub due_at_ms: u64,
    pub function_kind: ScheduledFunctionKind,
    pub function_json: String,
    pub input: Value,
    pub logical_key: String,
    pub source: ScheduleSource,
}

impl ScheduleRow {
    pub fn validate(&self) -> Result<(), ScheduleError> {
        require_identifier(self.schedule_id.as_str(), "schedule_id")?;
        require_identifier(&self.logical_key, "logical_key")?;
        if self.function_json.trim().is_empty() {
            return Err(ScheduleError::Invalid(
                "function_json must not be empty".to_string(),
            ));
        }
        if let ScheduleSource::Cron { expression, .. } = &self.source {
            if expression.trim().is_empty() {
                return Err(ScheduleError::Invalid(
                    "cron expression must not be empty".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BusinessOutcomeKind {
    Succeeded,
    Failed,
    EffectUnknown,
    Reconciled,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BusinessOutcome {
    pub subject_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule_id: Option<ScheduleId>,
    pub kind: BusinessOutcomeKind,
    pub detail: Value,
}

impl BusinessOutcome {
    fn validate(&self) -> Result<(), ScheduleError> {
        require_identifier(&self.subject_id, "business_outcome.subject_id")?;
        if let Some(schedule_id) = &self.schedule_id {
            require_identifier(schedule_id.as_str(), "business_outcome.schedule_id")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleHoldReason {
    ExternalActionReconciliation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleHold {
    pub hold_id: String,
    pub reason: ScheduleHoldReason,
    pub detail: Value,
}

impl ScheduleHold {
    fn validate(&self) -> Result<(), ScheduleError> {
        require_identifier(&self.hold_id, "schedule_hold.hold_id")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum ScheduleChange {
    Enqueue {
        row: ScheduleRow,
    },
    Hold {
        schedule_id: ScheduleId,
        hold: ScheduleHold,
    },
    Complete {
        schedule_id: ScheduleId,
    },
}

impl ScheduleChange {
    fn schedule_id(&self) -> &ScheduleId {
        match self {
            Self::Enqueue { row } => &row.schedule_id,
            Self::Hold { schedule_id, .. } | Self::Complete { schedule_id } => schedule_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleTransaction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<BusinessOutcome>,
    pub changes: Vec<ScheduleChange>,
}

/// One physical intent containing both the semantic mutation and the schedule
/// transaction caused by that mutation. The outer proposal retains the
/// mutation's digest and read set while its write set is widened with schedule
/// addresses.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AtomicScheduledMutation {
    pub business_intent: LogIntent,
    pub schedule: ScheduleTransaction,
}

impl AtomicScheduledMutation {
    fn validate(&self) -> Result<(), ScheduleError> {
        self.business_intent
            .validate()
            .map_err(|error| ScheduleError::Invalid(error.to_string()))?;
        if self.business_intent.kind.namespace() == SCHEDULE_INTENT_NAMESPACE
            && matches!(
                self.business_intent.kind.name(),
                SCHEDULE_INTENT_NAME | SCHEDULED_MUTATION_INTENT_NAME
            )
        {
            return Err(ScheduleError::Invalid(
                "a scheduled mutation cannot nest another schedule envelope".to_string(),
            ));
        }
        self.schedule.validate()
    }
}

impl ScheduleTransaction {
    pub fn new(
        outcome: Option<BusinessOutcome>,
        changes: Vec<ScheduleChange>,
    ) -> Result<Self, ScheduleError> {
        let transaction = Self { outcome, changes };
        transaction.validate()?;
        Ok(transaction)
    }

    pub fn outcome_and_next(
        outcome: BusinessOutcome,
        completed: Option<ScheduleId>,
        next: impl IntoIterator<Item = ScheduleRow>,
    ) -> Result<Self, ScheduleError> {
        let mut changes = Vec::new();
        if let Some(schedule_id) = completed {
            changes.push(ScheduleChange::Complete { schedule_id });
        }
        changes.extend(next.into_iter().map(|row| ScheduleChange::Enqueue { row }));
        Self::new(Some(outcome), changes)
    }

    pub fn validate(&self) -> Result<(), ScheduleError> {
        if self.outcome.is_none() && self.changes.is_empty() {
            return Err(ScheduleError::Invalid(
                "schedule transaction must contain an outcome or a change".to_string(),
            ));
        }
        if let Some(outcome) = &self.outcome {
            outcome.validate()?;
        }
        let mut changed_ids = BTreeSet::new();
        for change in &self.changes {
            require_identifier(change.schedule_id().as_str(), "schedule_change.schedule_id")?;
            if !changed_ids.insert(change.schedule_id().clone()) {
                return Err(ScheduleError::Invalid(format!(
                    "schedule {} changes more than once in one transaction",
                    change.schedule_id()
                )));
            }
            match change {
                ScheduleChange::Enqueue { row } => row.validate()?,
                ScheduleChange::Hold { hold, .. } => hold.validate()?,
                ScheduleChange::Complete { .. } => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeldSchedule {
    pub row: ScheduleRow,
    pub hold: ScheduleHold,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScheduleProjection {
    pending: BTreeMap<ScheduleId, ScheduleRow>,
    held: BTreeMap<ScheduleId, HeldSchedule>,
    completed: BTreeSet<ScheduleId>,
    outcomes: Vec<BusinessOutcome>,
}

impl ScheduleProjection {
    pub fn replay(records: &[LogRecord]) -> Result<Self, ScheduleError> {
        let mut projection = Self::default();
        for record in records {
            let Some(transaction) = schedule_transaction_from_record(record)? else {
                continue;
            };
            transaction.validate()?;
            projection.apply(&transaction)?;
        }
        Ok(projection)
    }

    pub fn apply(&mut self, transaction: &ScheduleTransaction) -> Result<(), ScheduleError> {
        transaction.validate()?;
        let mut next = self.clone();
        if let Some(outcome) = &transaction.outcome {
            next.outcomes.push(outcome.clone());
        }
        for change in &transaction.changes {
            match change {
                ScheduleChange::Enqueue { row } => {
                    if next.pending.contains_key(&row.schedule_id)
                        || next.held.contains_key(&row.schedule_id)
                        || next.completed.contains(&row.schedule_id)
                    {
                        return Err(ScheduleError::Transition(format!(
                            "schedule {} was already observed",
                            row.schedule_id
                        )));
                    }
                    next.pending.insert(row.schedule_id.clone(), row.clone());
                }
                ScheduleChange::Hold { schedule_id, hold } => {
                    let row = if let Some(row) = next.pending.remove(schedule_id) {
                        row
                    } else if let Some(existing) = next.held.remove(schedule_id) {
                        if existing.hold.hold_id != hold.hold_id {
                            return Err(ScheduleError::Transition(format!(
                                "schedule {schedule_id} changed hold identity"
                            )));
                        }
                        existing.row
                    } else {
                        return Err(ScheduleError::Transition(format!(
                            "cannot hold unknown schedule {schedule_id}"
                        )));
                    };
                    next.held.insert(
                        schedule_id.clone(),
                        HeldSchedule {
                            row,
                            hold: hold.clone(),
                        },
                    );
                }
                ScheduleChange::Complete { schedule_id } => {
                    let existed = next.pending.remove(schedule_id).is_some()
                        || next.held.remove(schedule_id).is_some();
                    if !existed {
                        return Err(ScheduleError::Transition(format!(
                            "cannot complete non-runnable schedule {schedule_id}"
                        )));
                    }
                    next.completed.insert(schedule_id.clone());
                }
            }
        }
        *self = next;
        Ok(())
    }

    pub fn due(&self, now_ms: u64) -> Vec<ScheduleRow> {
        let mut due = self
            .pending
            .values()
            .filter(|row| row.due_at_ms <= now_ms)
            .cloned()
            .collect::<Vec<_>>();
        due.sort_by(|left, right| {
            left.due_at_ms
                .cmp(&right.due_at_ms)
                .then_with(|| left.schedule_id.cmp(&right.schedule_id))
        });
        due
    }

    pub fn pending(&self) -> &BTreeMap<ScheduleId, ScheduleRow> {
        &self.pending
    }

    pub fn held(&self) -> &BTreeMap<ScheduleId, HeldSchedule> {
        &self.held
    }

    pub fn completed(&self) -> &BTreeSet<ScheduleId> {
        &self.completed
    }

    pub fn outcomes(&self) -> &[BusinessOutcome] {
        &self.outcomes
    }
}

pub fn wrap_scheduled_mutation(
    proposal: Proposal,
    schedule: ScheduleTransaction,
) -> Result<Proposal, ScheduleError> {
    schedule.validate()?;
    if proposal.intent.kind.namespace() == SCHEDULE_INTENT_NAMESPACE
        && matches!(
            proposal.intent.kind.name(),
            SCHEDULE_INTENT_NAME | SCHEDULED_MUTATION_INTENT_NAME
        )
    {
        return Err(ScheduleError::Invalid(
            "a scheduled mutation cannot wrap another schedule envelope".to_string(),
        ));
    }

    let mut paths = proposal.write_set.entries().to_vec();
    paths.extend(schedule_write_paths(&schedule));
    let write_set =
        WriteSet::new(paths).map_err(|error| ScheduleError::Encode(error.to_string()))?;
    let envelope = AtomicScheduledMutation {
        business_intent: proposal.intent,
        schedule,
    };
    envelope.validate()?;
    let intent = LogIntent::new(
        IntentKind::new(
            SCHEDULE_INTENT_NAMESPACE,
            SCHEDULED_MUTATION_INTENT_NAME,
            SCHEDULE_INTENT_VERSION,
        )
        .map_err(|error| ScheduleError::Encode(error.to_string()))?,
        serde_json::to_value(envelope).map_err(|error| ScheduleError::Encode(error.to_string()))?,
    )
    .map_err(|error| ScheduleError::Encode(error.to_string()))?;
    Ok(Proposal::new(
        proposal.begin_position,
        proposal.read_set,
        write_set,
        proposal.op_digest,
        intent,
    ))
}

pub struct ScheduleJournal<'a> {
    committer: &'a mut ProposalCommitter,
}

impl<'a> ScheduleJournal<'a> {
    pub fn new(committer: &'a mut ProposalCommitter) -> Self {
        Self { committer }
    }

    pub fn projection(&self) -> Result<ScheduleProjection, ScheduleError> {
        ScheduleProjection::replay(self.committer.log().records())
    }

    pub fn commit(
        &mut self,
        transaction: ScheduleTransaction,
        op_digest: OpDigest,
    ) -> Result<AppendReceipt, ScheduleError> {
        transaction.validate()?;
        let paths = schedule_write_paths(&transaction);
        let write_set =
            WriteSet::new(paths).map_err(|error| ScheduleError::Encode(error.to_string()))?;
        let intent = LogIntent::new(
            IntentKind::new(
                SCHEDULE_INTENT_NAMESPACE,
                SCHEDULE_INTENT_NAME,
                SCHEDULE_INTENT_VERSION,
            )
            .map_err(|error| ScheduleError::Encode(error.to_string()))?,
            serde_json::to_value(&transaction)
                .map_err(|error| ScheduleError::Encode(error.to_string()))?,
        )
        .map_err(|error| ScheduleError::Encode(error.to_string()))?;
        let proposal = Proposal::new(
            self.committer.log().latest_position(),
            ReadSet::default(),
            write_set,
            op_digest,
            intent,
        );
        if self
            .committer
            .log()
            .record_for_digest(&proposal.op_digest)
            .is_none()
        {
            let mut projection = self.projection()?;
            projection.apply(&transaction)?;
        }
        match self
            .committer
            .commit(proposal)
            .map_err(|error| ScheduleError::Commit(error.to_string()))?
        {
            CommitOutcome::Committed(receipt) => Ok(receipt.append),
            CommitOutcome::Refused(refusal) => Err(ScheduleError::Commit(format!(
                "schedule proposal conflicted at positions {:?}",
                refusal
                    .conflicting_positions
                    .iter()
                    .map(|position| position.0)
                    .collect::<Vec<_>>()
            ))),
        }
    }
}

fn schedule_transaction_from_record(
    record: &LogRecord,
) -> Result<Option<ScheduleTransaction>, ScheduleError> {
    if record.intent.kind.namespace() != SCHEDULE_INTENT_NAMESPACE
        || record.intent.kind.version() != SCHEDULE_INTENT_VERSION
    {
        return Ok(None);
    }
    match record.intent.kind.name() {
        SCHEDULE_INTENT_NAME => serde_json::from_value(record.intent.payload.clone())
            .map(Some)
            .map_err(|error| {
                ScheduleError::Decode(format!(
                    "schedule record at position {} is invalid: {error}",
                    record.position
                ))
            }),
        SCHEDULED_MUTATION_INTENT_NAME => {
            let envelope: AtomicScheduledMutation =
                serde_json::from_value(record.intent.payload.clone()).map_err(|error| {
                    ScheduleError::Decode(format!(
                        "scheduled mutation at position {} is invalid: {error}",
                        record.position
                    ))
                })?;
            envelope.validate()?;
            Ok(Some(envelope.schedule))
        }
        _ => Ok(None),
    }
}

fn schedule_write_paths(transaction: &ScheduleTransaction) -> Vec<AccessPath> {
    let mut paths = transaction
        .changes
        .iter()
        .map(|change| AccessPath::Entity {
            entity: format!("schedule:{}", change.schedule_id()),
        })
        .collect::<Vec<_>>();
    if let Some(outcome) = &transaction.outcome {
        paths.push(AccessPath::Entity {
            entity: format!("business-outcome:{}", outcome.subject_id),
        });
    }
    paths
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), ScheduleError> {
    if value.trim().is_empty() {
        Err(ScheduleError::Invalid(format!("{field} must not be empty")))
    } else if value != value.trim() {
        Err(ScheduleError::Invalid(format!(
            "{field} must not contain surrounding whitespace"
        )))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    Invalid(String),
    Encode(String),
    Decode(String),
    Transition(String),
    Commit(String),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(detail) => write!(formatter, "schedule invalid: {detail}"),
            Self::Encode(detail) => write!(formatter, "schedule encoding failed: {detail}"),
            Self::Decode(detail) => write!(formatter, "schedule replay failed: {detail}"),
            Self::Transition(detail) => write!(formatter, "schedule transition refused: {detail}"),
            Self::Commit(detail) => write!(formatter, "schedule commit failed: {detail}"),
        }
    }
}

impl Error for ScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scheduled_mutation_revalidates_its_nested_business_intent() {
        let forged_business_intent: LogIntent = serde_json::from_value(json!({
            "kind": {
                "namespace": "graph",
                "name": "node_upsert",
                "version": 1
            },
            "payload": {"value": "forged"}
        }))
        .unwrap();
        let envelope = AtomicScheduledMutation {
            business_intent: forged_business_intent,
            schedule: ScheduleTransaction::new(
                Some(BusinessOutcome {
                    subject_id: "mutation:forged".to_string(),
                    schedule_id: None,
                    kind: BusinessOutcomeKind::Failed,
                    detail: json!({}),
                }),
                Vec::new(),
            )
            .unwrap(),
        };

        let error = envelope.validate().unwrap_err();
        assert!(error.to_string().contains("generic CRUD-diff intent"));
    }
}
