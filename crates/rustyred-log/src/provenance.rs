//! Dependency-minimal wire vocabulary for operation provenance.
//!
//! These types live beside the durable log so higher layers can share their
//! serialized tokens without making the log depend on execution or program
//! runtimes. Origin describes where work began; authority describes who was
//! permitted to commit it. They are intentionally not the same field.

use serde::{Deserialize, Serialize};

/// Capability context in which an operation originated.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryOrigin {
    #[default]
    LegacyUnknown,
    Query,
    Mutation,
    Action,
    Internal,
}

/// Authority that admitted a durable append.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitAuthority {
    #[default]
    LegacyUnknown,
    Mutation,
    Internal,
}

impl CommitAuthority {
    /// Whether this authority can truthfully commit an operation of `origin`.
    pub const fn permits(self, origin: EntryOrigin) -> bool {
        matches!(
            (self, origin),
            (Self::LegacyUnknown, EntryOrigin::LegacyUnknown)
                | (Self::Mutation, EntryOrigin::Mutation | EntryOrigin::Action)
                | (Self::Internal, EntryOrigin::Internal)
        )
    }
}

/// Transport used by the executor that produced an operation result.
///
/// Tokens and the `InProcess` default are retained from the offload receipt
/// surface. A legacy log entry represents an unknown transport with `None`
/// rather than manufacturing this default inside [`EntryProvenance`].
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BridgeTransport {
    #[default]
    InProcess,
    PyO3,
    Service,
}

/// Normalized verification state, preserving the historical offload tokens.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    #[default]
    Unverified,
    FeasibilityVerified,
    SolverAttested,
    GroundingVerified,
    SourceAttested,
    Failed,
}

impl VerificationState {
    /// Presentation order only. This is not an evidence lattice operation.
    pub const fn display_rank(self) -> u8 {
        match self {
            Self::Failed => 0,
            Self::Unverified => 1,
            Self::SourceAttested => 2,
            Self::SolverAttested => 3,
            Self::FeasibilityVerified => 4,
            Self::GroundingVerified => 5,
        }
    }

    /// Separate success/refutation from the strength of supporting evidence.
    pub const fn outcome(self) -> VerificationOutcome {
        match self {
            Self::Unverified => VerificationOutcome::Unverified,
            Self::Failed => VerificationOutcome::Refuted,
            Self::FeasibilityVerified
            | Self::SolverAttested
            | Self::GroundingVerified
            | Self::SourceAttested => VerificationOutcome::Supported,
        }
    }

    /// Project supporting states into an explicit assurance grade.
    ///
    /// `None` means there is no supporting evidence grade: an unverified or
    /// refuted outcome must not be mistaken for the bottom of a proof lattice.
    pub const fn evidence_grade(self) -> Option<EvidenceGrade> {
        match self {
            Self::SourceAttested | Self::SolverAttested => Some(EvidenceGrade::Attested),
            Self::FeasibilityVerified | Self::GroundingVerified => {
                Some(EvidenceGrade::IndependentlyVerified)
            }
            Self::Unverified | Self::Failed => None,
        }
    }
}

/// Whether verification supports, refutes, or has not evaluated a claim.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    #[default]
    Unverified,
    Supported,
    Refuted,
}

/// Assurance grade for supporting evidence.
///
/// Deliberately lacks `Ord`: a UI presentation rank is not automatically a
/// lawful meet/join operation for the future evidence fold.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceGrade {
    Attested,
    IndependentlyVerified,
}

/// Typed writer identity. Absence is represented by `Option<WriterId>`.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct WriterId(String);

impl WriterId {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err("writer identity must not be blank");
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for WriterId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Provenance that will be hash-bound to a physical log entry in protocol v3.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EntryProvenance {
    #[serde(default)]
    pub origin: EntryOrigin,
    #[serde(default)]
    pub authority: CommitAuthority,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<BridgeTransport>,
    #[serde(default)]
    pub verification: VerificationState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writer: Option<WriterId>,
}

impl EntryProvenance {
    pub fn is_authority_consistent(&self) -> bool {
        self.authority.permits(self.origin)
    }

    pub fn is_explicit_legacy(&self) -> bool {
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_defaults_are_explicit_and_do_not_invent_identity() {
        let provenance: EntryProvenance = serde_json::from_str("{}").unwrap();
        assert!(provenance.is_explicit_legacy());
        assert_eq!(provenance.origin, EntryOrigin::LegacyUnknown);
        assert_eq!(provenance.authority, CommitAuthority::LegacyUnknown);
        assert_eq!(provenance.transport, None);
        assert_eq!(provenance.verification, VerificationState::Unverified);
        assert_eq!(provenance.writer, None);
    }

    #[test]
    fn authority_keeps_origin_and_commit_capability_distinct() {
        assert!(CommitAuthority::Mutation.permits(EntryOrigin::Mutation));
        assert!(CommitAuthority::Mutation.permits(EntryOrigin::Action));
        assert!(!CommitAuthority::Mutation.permits(EntryOrigin::Query));
        assert!(!CommitAuthority::LegacyUnknown.permits(EntryOrigin::Mutation));
    }

    #[test]
    fn verification_outcome_is_not_the_evidence_grade() {
        assert_eq!(
            VerificationState::GroundingVerified.outcome(),
            VerificationOutcome::Supported
        );
        assert_eq!(
            VerificationState::GroundingVerified.evidence_grade(),
            Some(EvidenceGrade::IndependentlyVerified)
        );
        assert_eq!(
            VerificationState::Failed.outcome(),
            VerificationOutcome::Refuted
        );
        assert_eq!(VerificationState::Failed.evidence_grade(), None);
        assert_eq!(VerificationState::Failed.display_rank(), 0);
        assert_eq!(VerificationState::GroundingVerified.display_rank(), 5);
    }

    #[test]
    fn writer_id_refuses_blank_values() {
        assert!(WriterId::new("  ").is_err());
        assert_eq!(WriterId::new("head:codex").unwrap().as_str(), "head:codex");
    }
}
