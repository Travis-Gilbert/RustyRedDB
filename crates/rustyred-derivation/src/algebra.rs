//! Runtime-readable algebra capabilities and their law profiles.
//!
//! The operational [`crate::Semiring`] trait predates these descriptors. It is
//! retained as a compatibility ABI, while [`MathematicalSemiring`] is the
//! compile-time certificate for consumers that rely on the full semiring laws.

use serde::{Deserialize, Serialize};

use crate::semiring::Semiring;

/// The algebraic object a descriptor actually claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgebraClass {
    StateTransitionMonoid,
    SegmentSummaryMonoid,
    AbelianChangeGroup,
    MathematicalSemiring,
    GraphBlasOperatorPair,
    EvidenceLattice,
    IndependentNoisyOr,
}

/// Laws for one binary operation and its optional identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationLaws {
    pub closed: bool,
    pub associative: bool,
    pub commutative: bool,
    pub has_identity: bool,
    pub has_inverse: bool,
    pub idempotent: bool,
}

impl OperationLaws {
    pub const MONOID: Self = Self {
        closed: true,
        associative: true,
        commutative: false,
        has_identity: true,
        has_inverse: false,
        idempotent: false,
    };

    pub const COMMUTATIVE_MONOID: Self = Self {
        commutative: true,
        ..Self::MONOID
    };

    pub const ABELIAN_GROUP: Self = Self {
        has_inverse: true,
        ..Self::COMMUTATIVE_MONOID
    };

    pub const IDEMPOTENT_COMMUTATIVE_MONOID: Self = Self {
        idempotent: true,
        ..Self::COMMUTATIVE_MONOID
    };

    pub const CLOSED_BINARY_OPERATION: Self = Self {
        closed: true,
        associative: false,
        commutative: false,
        has_identity: false,
        has_inverse: false,
        idempotent: false,
    };
}

/// Complete law profile for one- or two-operation algebras.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LawProfile {
    pub primary: OperationLaws,
    pub secondary: Option<OperationLaws>,
    pub left_distributive: bool,
    pub right_distributive: bool,
    pub primary_identity_annihilates_secondary: bool,
    pub absorption: bool,
}

impl LawProfile {
    pub const TRANSITION_MONOID: Self = Self::single(OperationLaws::MONOID);
    pub const SUMMARY_MONOID: Self = Self::single(OperationLaws::MONOID);
    pub const ABELIAN_CHANGE_GROUP: Self = Self::single(OperationLaws::ABELIAN_GROUP);

    pub const MATHEMATICAL_SEMIRING: Self = Self {
        primary: OperationLaws::COMMUTATIVE_MONOID,
        secondary: Some(OperationLaws::MONOID),
        left_distributive: true,
        right_distributive: true,
        primary_identity_annihilates_secondary: true,
        absorption: false,
    };

    pub const GRAPHBLAS_OPERATOR_PAIR: Self = Self {
        primary: OperationLaws::COMMUTATIVE_MONOID,
        secondary: Some(OperationLaws::CLOSED_BINARY_OPERATION),
        left_distributive: false,
        right_distributive: false,
        primary_identity_annihilates_secondary: false,
        absorption: false,
    };

    pub const EVIDENCE_LATTICE: Self = Self {
        primary: OperationLaws::IDEMPOTENT_COMMUTATIVE_MONOID,
        secondary: Some(OperationLaws::IDEMPOTENT_COMMUTATIVE_MONOID),
        left_distributive: false,
        right_distributive: false,
        primary_identity_annihilates_secondary: false,
        absorption: true,
    };

    pub const INDEPENDENT_NOISY_OR: Self = Self {
        primary: OperationLaws::COMMUTATIVE_MONOID,
        secondary: Some(OperationLaws::COMMUTATIVE_MONOID),
        left_distributive: false,
        right_distributive: false,
        primary_identity_annihilates_secondary: true,
        absorption: false,
    };

    pub const fn single(primary: OperationLaws) -> Self {
        Self {
            primary,
            secondary: None,
            left_distributive: false,
            right_distributive: false,
            primary_identity_annihilates_secondary: false,
            absorption: false,
        }
    }

    pub const fn is_mathematical_semiring(self) -> bool {
        self.primary.closed
            && self.primary.associative
            && self.primary.commutative
            && self.primary.has_identity
            && matches!(self.secondary, Some(operation) if operation.closed && operation.associative && operation.has_identity)
            && self.left_distributive
            && self.right_distributive
            && self.primary_identity_annihilates_secondary
    }
}

/// Equality relation under which the declared laws are checked.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EqualityPolicy {
    Exact,
    CanonicalFloat,
    ApproximateFloat { absolute_tolerance: f64 },
}

/// Reduction-order requirement behind a repeatable result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismPolicy {
    OrderIndependent,
    CanonicalOperandOrder,
    FixedReductionOrder,
}

/// A named, serializable capability claim.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct AlgebraDescriptor {
    pub name: &'static str,
    pub compatibility_tokens: &'static [&'static str],
    pub class: AlgebraClass,
    pub laws: LawProfile,
    pub equality: EqualityPolicy,
    pub determinism: DeterminismPolicy,
    pub content_identity_safe: bool,
}

impl AlgebraDescriptor {
    pub const fn claims_mathematical_semiring(self) -> bool {
        matches!(self.class, AlgebraClass::MathematicalSemiring)
            && self.laws.is_mathematical_semiring()
    }

    pub fn require_mathematical_semiring(
        &'static self,
    ) -> Result<&'static Self, AlgebraRequirementError> {
        if self.claims_mathematical_semiring() {
            Ok(self)
        } else {
            Err(AlgebraRequirementError {
                algebra: self.name,
                required: AlgebraClass::MathematicalSemiring,
                actual: self.class,
            })
        }
    }
}

/// A failed attempt to use a weaker algebra at a law-dependent boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("algebra {algebra:?} is {actual:?}, but {required:?} is required")]
pub struct AlgebraRequirementError {
    pub algebra: &'static str,
    pub required: AlgebraClass,
    pub actual: AlgebraClass,
}

/// Marker implemented only after the complete mathematical-semiring law matrix passes.
pub trait MathematicalSemiring: Semiring {}

const APPROXIMATE_FLOAT: EqualityPolicy = EqualityPolicy::ApproximateFloat {
    absolute_tolerance: 1.0e-9,
};

pub static BOOLEAN_SEMIRING_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "boolean",
    compatibility_tokens: &["boolean"],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: EqualityPolicy::Exact,
    determinism: DeterminismPolicy::OrderIndependent,
    content_identity_safe: true,
};

pub static COUNTING_SEMIRING_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "counting",
    compatibility_tokens: &["counting"],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: EqualityPolicy::Exact,
    determinism: DeterminismPolicy::OrderIndependent,
    content_identity_safe: true,
};

pub static TROPICAL_SEMIRING_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "tropical",
    compatibility_tokens: &["tropical"],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: APPROXIMATE_FLOAT,
    determinism: DeterminismPolicy::FixedReductionOrder,
    content_identity_safe: false,
};

pub static VITERBI_SEMIRING_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "viterbi",
    compatibility_tokens: &["viterbi"],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: APPROXIMATE_FLOAT,
    determinism: DeterminismPolicy::FixedReductionOrder,
    content_identity_safe: false,
};

pub static CORROBORATION_SEMIRING_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "corroboration",
    compatibility_tokens: &["corroboration"],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: EqualityPolicy::CanonicalFloat,
    determinism: DeterminismPolicy::OrderIndependent,
    content_identity_safe: false,
};

pub static INDEPENDENT_NOISY_OR_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "independent_noisy_or",
    compatibility_tokens: &["probability", "independent_noisy_or"],
    class: AlgebraClass::IndependentNoisyOr,
    laws: LawProfile::INDEPENDENT_NOISY_OR,
    equality: APPROXIMATE_FLOAT,
    determinism: DeterminismPolicy::FixedReductionOrder,
    content_identity_safe: false,
};

static REGISTERED_ALGEBRAS: [&AlgebraDescriptor; 6] = [
    &BOOLEAN_SEMIRING_DESCRIPTOR,
    &COUNTING_SEMIRING_DESCRIPTOR,
    &TROPICAL_SEMIRING_DESCRIPTOR,
    &VITERBI_SEMIRING_DESCRIPTOR,
    &CORROBORATION_SEMIRING_DESCRIPTOR,
    &INDEPENDENT_NOISY_OR_DESCRIPTOR,
];

pub fn registered_algebras() -> &'static [&'static AlgebraDescriptor] {
    &REGISTERED_ALGEBRAS
}

pub fn resolve_algebra_token(token: &str) -> Option<&'static AlgebraDescriptor> {
    registered_algebras()
        .iter()
        .copied()
        .find(|descriptor| descriptor.compatibility_tokens.contains(&token))
}
