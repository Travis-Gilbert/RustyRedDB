//! Scoped Link service wire records. Generator execution receipts stay native:
//! their index-wide work counters are not a public evidence surface.
use crate::{
    article::ArticleCommit,
    link::{Justification, LinkCandidate, LinkKind, LinkReadStamp},
    wiki::WikiRefusal,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkCandidatesQuery {
    pub scope: String,
    pub kind: LinkKind,
    pub limit: usize,
    #[serde(default)]
    pub include_below: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkSuggestion {
    pub candidate: LinkCandidate,
    pub target_title: String,
    pub target_slug: String,
    pub score: f64,
    pub model_version: String,
    pub model_artifact_id: Option<String>,
    pub threshold: f64,
    pub threshold_revision: u64,
    pub above_threshold: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkCandidates {
    pub source_id: String,
    pub graph_root: String,
    pub candidates: Vec<LinkSuggestion>,
    pub receipt_id: String,
    pub expires_at_ms: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkDecision {
    Accept,
    Keep,
    Reject,
    Retract,
    Mute,
}

/// A decision names server-issued evidence. Feature blobs, endpoint IDs and
/// scorer versions cannot be supplied or overwritten by the caller.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkDecide {
    pub candidate_id: String,
    pub decision: LinkDecision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkDecisionResult {
    pub candidate_id: String,
    pub decision: LinkDecision,
    pub edge_id: String,
    pub receipt_id: String,
    pub signal_id: String,
    pub commit: ArticleCommit,
}

/// Only evidence admitted to this reader is presented. This intentionally has
/// no open JSON payload through which executor or index counters could escape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkEvidence {
    pub candidate_id: String,
    pub generator: String,
    pub justification: Justification,
    pub read_stamp: LinkReadStamp,
    pub model_version: String,
    pub model_artifact_id: Option<String>,
    pub threshold_revision: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkReceiptView {
    pub receipt_id: String,
    pub evidence: Vec<LinkEvidence>,
}

/// Editable per-tenant policy. Values are read from an actual registry fact;
/// an uninitialized policy is a refusal, never a code-defined threshold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkThresholds {
    pub inline: f64,
    pub see_also: f64,
    pub cross_scope: f64,
    pub typed: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkThresholdPolicy {
    pub thresholds: LinkThresholds,
    pub revision: u64,
    pub modified_at_ms: u64,
    pub actor: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkThresholdUpdate {
    pub thresholds: LinkThresholds,
    pub expected_revision: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LinkRefusal {
    Admission { refusal: WikiRefusal },
    ThresholdsNotConfigured,
    PublicAuthorityUnavailable,
    StaleCandidate,
    ProjectionUnavailable,
    Invalid { message: String },
    Unavailable,
}

/// Explicit authenticated import of minimal, independently public repository
/// identity. This never accepts or publishes captured document prose.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkRepositoryImport {
    pub repository: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkRepositoryImported {
    pub node_id: String,
    pub repository: String,
    pub title: String,
    pub receipt_id: String,
    pub commit: ArticleCommit,
}

impl From<WikiRefusal> for LinkRefusal {
    fn from(refusal: WikiRefusal) -> Self {
        Self::Admission { refusal }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_cannot_override_server_candidate_or_scorer() {
        for extra in [
            "features",
            "src",
            "dst",
            "score",
            "model_version",
            "threshold_revision",
        ] {
            let mut wire = serde_json::json!({"candidate_id":"issued", "decision":"accept"});
            wire[extra] = serde_json::json!("forged");
            assert!(
                serde_json::from_value::<LinkDecide>(wire).is_err(),
                "{extra}"
            );
        }
        assert_eq!(
            serde_json::from_value::<LinkDecide>(
                serde_json::json!({"candidate_id":"issued", "decision":"reject"})
            )
            .unwrap()
            .decision,
            LinkDecision::Reject
        );
    }
}
