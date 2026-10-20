//! Engine-neutral standing-query objects.
//!
//! A standing query owns the durable product contract: tenant scope, bounded
//! definition, current result, last-evaluated marker, and delivery policy. An
//! evaluator owns *how* that definition is maintained. The first evaluator can
//! rescan a bounded scope; a future DBSP/Feldera circuit can implement the same
//! trait without changing consumers of a [`StandingQuery`].

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where a standing query delivers a firing. The delivery is part of the
/// standing object rather than an evaluator concern, so changing from a rescan
/// to an incremental engine cannot change product behavior.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StandingDelivery {
    BriefingItem {
        title: String,
        body: String,
    },
    Notification {
        channel: String,
        title: String,
        body: String,
    },
    AgentActionProposal {
        action_kind: String,
        summary: String,
    },
}

/// A bounded scope an evaluator must respect while it maintains a query.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingScope {
    /// Stable, user-facing name for the scope the query watches.
    pub id: String,
    /// Hard cap on observations the evaluator may include in one result.
    pub max_items: usize,
}

/// A typed definition for a standing query. `C` is deliberately evaluator
/// specific: the stable query object does not need to learn DBSP's circuit AST
/// in order to start with a bounded rescan condition today.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingDefinition<C> {
    pub scope: StandingScope,
    pub condition: C,
    pub delivery: StandingDelivery,
}

/// One bounded piece of evidence that contributed to a standing result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingMatch {
    /// Stable match identity. Evaluators must keep this deterministic so output
    /// changes, rather than evaluation timestamps, drive firings.
    pub key: String,
    /// Evaluator-specific evidence retained for delivery and audit.
    pub evidence: Value,
}

/// The current materialized answer to a standing query. Evaluation time lives
/// in [`StandingEvaluationMarker`], not here, so equality means the answer
/// changed rather than merely being recomputed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingResult {
    pub matched: bool,
    pub matches: Vec<StandingMatch>,
    pub summary: String,
    #[serde(default)]
    pub detail: Value,
}

impl StandingResult {
    pub fn no_match(summary: impl Into<String>) -> Self {
        Self {
            matched: false,
            matches: Vec::new(),
            summary: summary.into(),
            detail: Value::Null,
        }
    }

    pub fn matched(summary: impl Into<String>, matches: Vec<StandingMatch>, detail: Value) -> Self {
        Self {
            matched: true,
            matches,
            summary: summary.into(),
            detail,
        }
    }

    /// Canonicalize evidence order before comparing materialized results. The
    /// evaluator contract names `key` as stable identity; the evidence JSON
    /// disambiguates two distinct observations with the same key. This keeps a
    /// reordered batch from becoming a spurious `Updated` firing.
    pub fn canonicalize_matches(&mut self) {
        let mut keyed = self
            .matches
            .drain(..)
            .map(|standing_match| {
                let evidence_key = serde_json::to_string(&standing_match.evidence)
                    .expect("serde_json::Value always serializes");
                ((standing_match.key.clone(), evidence_key), standing_match)
            })
            .collect::<Vec<_>>();
        keyed.sort_by(|left, right| left.0.cmp(&right.0));

        let mut seen = BTreeSet::new();
        self.matches = keyed
            .into_iter()
            .filter_map(|(key, standing_match)| seen.insert(key).then_some(standing_match))
            .collect();
    }
}

/// The evaluator and sequence that produced a materialized answer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingEvaluationMarker {
    pub evaluator: String,
    pub sequence: u64,
    pub evaluated_at_ms: i64,
}

/// The semantic reason a changed result delivered a firing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandingFiringKind {
    Raised,
    Updated,
    Resolved,
}

/// A delivery-ready transition. Proactive surfaces subscribe to this stable
/// payload and do not know whether it came from a rescan or an incremental
/// circuit.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingFiring {
    pub query_id: String,
    pub tenant_slug: String,
    pub kind: StandingFiringKind,
    pub delivery: StandingDelivery,
    pub previous_result: Option<StandingResult>,
    pub current_result: StandingResult,
}

/// Receipt for every evaluation. A firing is optional because a stable answer
/// is still an auditable re-evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingEvaluationReceipt {
    pub query_id: String,
    pub tenant_slug: String,
    pub marker: StandingEvaluationMarker,
    pub previous_result: Option<StandingResult>,
    pub current_result: StandingResult,
    pub firing: Option<StandingFiring>,
}

/// The typed standing object consumed by proactive features.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StandingQuery<C> {
    pub id: String,
    pub tenant_slug: String,
    pub definition: StandingDefinition<C>,
    pub current_result: Option<StandingResult>,
    pub last_evaluated: Option<StandingEvaluationMarker>,
}

/// Validation failures for a standing-query contract. These are intentionally
/// local and dependency-free so a caller cannot create an unscoped sentinel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StandingQueryError {
    EmptyId,
    EmptyTenant,
    EmptyScope,
    UnboundedScope,
}

impl fmt::Display for StandingQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyId => "standing query id must not be empty",
            Self::EmptyTenant => "standing query tenant_slug must not be empty",
            Self::EmptyScope => "standing query scope id must not be empty",
            Self::UnboundedScope => "standing query scope max_items must be greater than zero",
        };
        formatter.write_str(message)
    }
}

impl Error for StandingQueryError {}

/// An evaluator or contract failure while staging a result. A staged result
/// never mutates the standing object's materialized state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StandingEvaluationError<E> {
    Evaluator(E),
    ResultExceedsScope { actual: usize, max_items: usize },
    Commit(StandingCommitError),
}

impl<E: fmt::Display> fmt::Display for StandingEvaluationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => write!(formatter, "standing evaluator failed: {error}"),
            Self::ResultExceedsScope { actual, max_items } => write!(
                formatter,
                "standing evaluator returned {actual} matches for a scope capped at {max_items}"
            ),
            Self::Commit(error) => write!(formatter, "standing evaluation commit failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for StandingEvaluationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Evaluator(error) => Some(error),
            Self::Commit(error) => Some(error),
            Self::ResultExceedsScope { .. } => None,
        }
    }
}

/// A receipt that cannot be committed to this standing object's current state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StandingCommitError {
    QueryIdMismatch,
    TenantMismatch,
    PreviousResultMismatch,
    SequenceMismatch { expected: u64, actual: u64 },
    ConflictingMarker,
}

impl fmt::Display for StandingCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::QueryIdMismatch => "standing evaluation belongs to a different query",
            Self::TenantMismatch => "standing evaluation belongs to a different tenant",
            Self::PreviousResultMismatch => {
                "standing evaluation does not continue the current materialized result"
            }
            Self::SequenceMismatch { expected, actual } => {
                return write!(
                    formatter,
                    "standing evaluation sequence mismatch: expected {expected}, got {actual}"
                );
            }
            Self::ConflictingMarker => {
                "standing evaluation reuses a sequence with a conflicting materialized result"
            }
        };
        formatter.write_str(message)
    }
}

impl Error for StandingCommitError {}

/// Pluggable evaluator for a typed standing definition and its current input.
/// A DBSP/Feldera adapter implements this same seam after the decision gates in
/// the integration plan pass; it does not change the query or its consumers.
pub trait StandingEvaluator<C, Input> {
    type Error;

    fn evaluator_id(&self) -> &str;

    fn evaluate(
        &self,
        definition: &StandingDefinition<C>,
        input: &Input,
    ) -> Result<StandingResult, Self::Error>;
}

impl<C> StandingQuery<C> {
    pub fn new(
        id: impl Into<String>,
        tenant_slug: impl Into<String>,
        definition: StandingDefinition<C>,
    ) -> Result<Self, StandingQueryError> {
        let id = id.into().trim().to_string();
        let tenant_slug = tenant_slug.into().trim().to_string();
        if id.is_empty() {
            return Err(StandingQueryError::EmptyId);
        }
        if tenant_slug.is_empty() {
            return Err(StandingQueryError::EmptyTenant);
        }
        if definition.scope.id.trim().is_empty() {
            return Err(StandingQueryError::EmptyScope);
        }
        if definition.scope.max_items == 0 {
            return Err(StandingQueryError::UnboundedScope);
        }

        Ok(Self {
            id,
            tenant_slug,
            definition,
            current_result: None,
            last_evaluated: None,
        })
    }

    /// Evaluate without advancing the materialized result. Persist the returned
    /// receipt (and any delivery it contains) before calling
    /// [`commit_evaluation`](Self::commit_evaluation). This creates an explicit
    /// audit-before-state boundary for durable proactive surfaces.
    pub fn stage_evaluation<Input, Evaluator>(
        &self,
        evaluator: &Evaluator,
        input: &Input,
        evaluated_at_ms: i64,
    ) -> Result<StandingEvaluationReceipt, StandingEvaluationError<Evaluator::Error>>
    where
        Evaluator: StandingEvaluator<C, Input>,
    {
        let previous_result = self.current_result.clone();
        let mut current_result = evaluator
            .evaluate(&self.definition, input)
            .map_err(StandingEvaluationError::Evaluator)?;
        current_result.canonicalize_matches();
        if current_result.matches.len() > self.definition.scope.max_items {
            return Err(StandingEvaluationError::ResultExceedsScope {
                actual: current_result.matches.len(),
                max_items: self.definition.scope.max_items,
            });
        }
        let marker = StandingEvaluationMarker {
            evaluator: evaluator.evaluator_id().to_string(),
            sequence: self
                .last_evaluated
                .as_ref()
                .map_or(1, |previous| previous.sequence.saturating_add(1)),
            evaluated_at_ms,
        };

        let firing =
            firing_kind(previous_result.as_ref(), &current_result).map(|kind| StandingFiring {
                query_id: self.id.clone(),
                tenant_slug: self.tenant_slug.clone(),
                kind,
                delivery: self.definition.delivery.clone(),
                previous_result: previous_result.clone(),
                current_result: current_result.clone(),
            });

        Ok(StandingEvaluationReceipt {
            query_id: self.id.clone(),
            tenant_slug: self.tenant_slug.clone(),
            marker,
            previous_result,
            current_result,
            firing,
        })
    }

    /// Advance the materialized result after its receipt and delivery have been
    /// made durable. Receipts must continue exactly from the current state, so a
    /// failed write cannot be hidden by a later stable re-evaluation.
    pub fn commit_evaluation(
        &mut self,
        evaluation: &StandingEvaluationReceipt,
    ) -> Result<(), StandingCommitError> {
        self.validate_receipt_identity(evaluation)?;
        if evaluation.previous_result != self.current_result {
            return Err(StandingCommitError::PreviousResultMismatch);
        }
        let expected = self
            .last_evaluated
            .as_ref()
            .map_or(1, |marker| marker.sequence.saturating_add(1));
        if evaluation.marker.sequence != expected {
            return Err(StandingCommitError::SequenceMismatch {
                expected,
                actual: evaluation.marker.sequence,
            });
        }
        self.current_result = Some(evaluation.current_result.clone());
        self.last_evaluated = Some(evaluation.marker.clone());
        Ok(())
    }

    /// Restore the latest durable evaluation after a process restart. Unlike a
    /// live commit, restoration may begin from a non-empty prior result because
    /// the preceding receipts are already durable in the sidecar graph.
    pub fn hydrate_from_receipt(
        &mut self,
        evaluation: &StandingEvaluationReceipt,
    ) -> Result<(), StandingCommitError> {
        self.validate_receipt_identity(evaluation)?;
        if let Some(existing) = &self.last_evaluated {
            if existing.sequence > evaluation.marker.sequence {
                return Ok(());
            }
            if existing.sequence == evaluation.marker.sequence {
                if self.current_result == Some(evaluation.current_result.clone())
                    && existing == &evaluation.marker
                {
                    return Ok(());
                }
                return Err(StandingCommitError::ConflictingMarker);
            }
        }
        self.current_result = Some(evaluation.current_result.clone());
        self.last_evaluated = Some(evaluation.marker.clone());
        Ok(())
    }

    /// Convenience for in-memory callers that do not need a separate durable
    /// receipt boundary. Durable runtimes should use `stage_evaluation` followed
    /// by `commit_evaluation` after persistence succeeds.
    pub fn evaluate<Input, Evaluator>(
        &mut self,
        evaluator: &Evaluator,
        input: &Input,
        evaluated_at_ms: i64,
    ) -> Result<StandingEvaluationReceipt, StandingEvaluationError<Evaluator::Error>>
    where
        Evaluator: StandingEvaluator<C, Input>,
    {
        let evaluation = self.stage_evaluation(evaluator, input, evaluated_at_ms)?;
        self.commit_evaluation(&evaluation)
            .map_err(StandingEvaluationError::Commit)?;
        Ok(evaluation)
    }

    fn validate_receipt_identity(
        &self,
        evaluation: &StandingEvaluationReceipt,
    ) -> Result<(), StandingCommitError> {
        if evaluation.query_id != self.id {
            return Err(StandingCommitError::QueryIdMismatch);
        }
        if evaluation.tenant_slug != self.tenant_slug {
            return Err(StandingCommitError::TenantMismatch);
        }
        Ok(())
    }
}

fn firing_kind(
    previous: Option<&StandingResult>,
    current: &StandingResult,
) -> Option<StandingFiringKind> {
    match previous {
        None if current.matched => Some(StandingFiringKind::Raised),
        None => None,
        Some(previous) if previous == current => None,
        Some(previous) if !previous.matched && !current.matched => None,
        Some(previous) if !previous.matched && current.matched => Some(StandingFiringKind::Raised),
        Some(previous) if previous.matched && !current.matched => {
            Some(StandingFiringKind::Resolved)
        }
        Some(_) => Some(StandingFiringKind::Updated),
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use serde_json::json;

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    struct TestCondition;

    struct InputEvaluator;

    impl StandingEvaluator<TestCondition, (&str, bool)> for InputEvaluator {
        type Error = Infallible;

        fn evaluator_id(&self) -> &str {
            "test-input-rescan"
        }

        fn evaluate(
            &self,
            _definition: &StandingDefinition<TestCondition>,
            input: &(&str, bool),
        ) -> Result<StandingResult, Self::Error> {
            if input.1 {
                Ok(StandingResult::matched(
                    "input matched",
                    vec![StandingMatch {
                        key: input.0.to_string(),
                        evidence: json!({ "matched": true }),
                    }],
                    json!({ "source": "test" }),
                ))
            } else {
                Ok(StandingResult::no_match("input did not match"))
            }
        }
    }

    struct AlwaysMatchEvaluator;

    impl StandingEvaluator<TestCondition, ()> for AlwaysMatchEvaluator {
        type Error = Infallible;

        fn evaluator_id(&self) -> &str {
            "replacement-incremental-engine"
        }

        fn evaluate(
            &self,
            _definition: &StandingDefinition<TestCondition>,
            _input: &(),
        ) -> Result<StandingResult, Self::Error> {
            Ok(StandingResult::matched(
                "replacement engine matched",
                vec![StandingMatch {
                    key: "replacement".to_string(),
                    evidence: json!({ "engine": "replacement" }),
                }],
                json!({}),
            ))
        }
    }

    fn query() -> StandingQuery<TestCondition> {
        StandingQuery::new(
            "sentinel:workspace-change",
            "Travis-Gilbert",
            StandingDefinition {
                scope: StandingScope {
                    id: "workspace".to_string(),
                    max_items: 16,
                },
                condition: TestCondition,
                delivery: StandingDelivery::BriefingItem {
                    title: "Workspace changed".to_string(),
                    body: "A watched input changed.".to_string(),
                },
            },
        )
        .unwrap()
    }

    #[test]
    fn only_changed_materialized_results_fire() {
        let mut query = query();
        let evaluator = InputEvaluator;

        let initial = query.evaluate(&evaluator, &("first", false), 10).unwrap();
        assert!(initial.firing.is_none());
        assert_eq!(initial.marker.sequence, 1);

        let raised = query.evaluate(&evaluator, &("first", true), 20).unwrap();
        assert_eq!(
            raised.firing.as_ref().map(|firing| firing.kind),
            Some(StandingFiringKind::Raised)
        );

        let stable = query.evaluate(&evaluator, &("first", true), 30).unwrap();
        assert!(stable.firing.is_none(), "timestamps do not create firings");
        assert_eq!(stable.marker.sequence, 3);

        let updated = query.evaluate(&evaluator, &("second", true), 40).unwrap();
        assert_eq!(
            updated.firing.as_ref().map(|firing| firing.kind),
            Some(StandingFiringKind::Updated)
        );

        let resolved = query.evaluate(&evaluator, &("second", false), 50).unwrap();
        assert_eq!(
            resolved.firing.as_ref().map(|firing| firing.kind),
            Some(StandingFiringKind::Resolved)
        );
    }

    #[test]
    fn evaluator_can_be_replaced_without_changing_the_standing_object() {
        let mut query = query();
        let initial = query
            .evaluate(&InputEvaluator, &("first", false), 10)
            .unwrap();
        assert!(initial.firing.is_none());

        let replacement = query.evaluate(&AlwaysMatchEvaluator, &(), 20).unwrap();
        assert_eq!(
            replacement.marker.evaluator,
            "replacement-incremental-engine"
        );
        assert_eq!(
            replacement.firing.as_ref().map(|firing| firing.kind),
            Some(StandingFiringKind::Raised)
        );
        assert_eq!(query.tenant_slug, "Travis-Gilbert");
    }

    #[test]
    fn staged_evaluation_does_not_advance_state_until_committed() {
        let mut query = query();
        let staged = query
            .stage_evaluation(&InputEvaluator, &("first", true), 10)
            .unwrap();
        assert!(query.current_result.is_none());
        assert!(query.last_evaluated.is_none());

        query.commit_evaluation(&staged).unwrap();
        assert_eq!(query.current_result, Some(staged.current_result));
        assert_eq!(query.last_evaluated, Some(staged.marker));
    }

    #[test]
    fn evaluator_output_cannot_exceed_the_bounded_scope() {
        struct UnboundedEvaluator;

        impl StandingEvaluator<TestCondition, ()> for UnboundedEvaluator {
            type Error = Infallible;

            fn evaluator_id(&self) -> &str {
                "unbounded-test"
            }

            fn evaluate(
                &self,
                _definition: &StandingDefinition<TestCondition>,
                _input: &(),
            ) -> Result<StandingResult, Self::Error> {
                Ok(StandingResult::matched(
                    "too many",
                    vec![
                        StandingMatch {
                            key: "one".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "two".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "three".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "four".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "five".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "six".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "seven".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "eight".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "nine".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "ten".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "eleven".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "twelve".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "thirteen".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "fourteen".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "fifteen".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "sixteen".to_string(),
                            evidence: json!({}),
                        },
                        StandingMatch {
                            key: "seventeen".to_string(),
                            evidence: json!({}),
                        },
                    ],
                    json!({}),
                ))
            }
        }

        let mut query = query();
        query.definition.scope.max_items = 16;
        let error = query
            .stage_evaluation(&UnboundedEvaluator, &(), 10)
            .unwrap_err();
        assert_eq!(
            error,
            StandingEvaluationError::ResultExceedsScope {
                actual: 17,
                max_items: 16,
            }
        );
    }

    #[test]
    fn standing_queries_require_a_tenant_and_bounded_scope() {
        let definition = StandingDefinition {
            scope: StandingScope {
                id: "scope".to_string(),
                max_items: 1,
            },
            condition: TestCondition,
            delivery: StandingDelivery::AgentActionProposal {
                action_kind: "review".to_string(),
                summary: "Review changed evidence".to_string(),
            },
        };
        assert_eq!(
            StandingQuery::new("query", "", definition.clone()).unwrap_err(),
            StandingQueryError::EmptyTenant
        );

        let unbounded = StandingDefinition {
            scope: StandingScope {
                id: "scope".to_string(),
                max_items: 0,
            },
            ..definition
        };
        assert_eq!(
            StandingQuery::new("query", "Travis-Gilbert", unbounded).unwrap_err(),
            StandingQueryError::UnboundedScope
        );
    }
}
