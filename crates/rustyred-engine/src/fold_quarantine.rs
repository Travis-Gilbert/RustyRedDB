//! Typed refusal receipts for schemaless entries rejected by a fold.

use crate::fold::{FoldId, FoldVersion};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use theorem_oplog::{ContentHash, LogId, LogPosition, LogRecord, OpDigest, RecordId};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FoldRefusalCode {
    InvalidPayload,
    SchemaViolation,
    UnsupportedIntent,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FoldRefusal {
    pub code: FoldRefusalCode,
    pub reason: String,
}

impl FoldRefusal {
    pub fn new(code: FoldRefusalCode, reason: impl Into<String>) -> Self {
        Self {
            code,
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FoldConsumeError {
    Refused(FoldRefusal),
    Fatal(String),
}

impl FoldConsumeError {
    pub fn refused(code: FoldRefusalCode, reason: impl Into<String>) -> Self {
        Self::Refused(FoldRefusal::new(code, reason))
    }

    pub fn fatal(reason: impl Into<String>) -> Self {
        Self::Fatal(reason.into())
    }
}

impl fmt::Display for FoldConsumeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(refusal) => write!(
                formatter,
                "fold refused entry with {:?}: {}",
                refusal.code, refusal.reason
            ),
            Self::Fatal(reason) => write!(formatter, "fold consume failed: {reason}"),
        }
    }
}

impl Error for FoldConsumeError {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QuarantineReceipt {
    pub fold_id: FoldId,
    pub fold_version: FoldVersion,
    pub log_id: LogId,
    pub position: LogPosition,
    pub record_id: RecordId,
    pub record_hash: ContentHash,
    pub op_digest: OpDigest,
    pub code: FoldRefusalCode,
    pub reason: String,
}

impl QuarantineReceipt {
    pub(crate) fn from_refusal(
        fold_id: FoldId,
        fold_version: FoldVersion,
        record: &LogRecord,
        refusal: FoldRefusal,
    ) -> Self {
        Self {
            fold_id,
            fold_version,
            log_id: record.log_id,
            position: record.position,
            record_id: record.record_id,
            record_hash: record.record_hash.clone(),
            op_digest: record.op_digest.clone(),
            code: refusal.code,
            reason: refusal.reason,
        }
    }
}
