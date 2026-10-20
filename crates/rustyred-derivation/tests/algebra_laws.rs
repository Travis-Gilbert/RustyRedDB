use std::fmt::Debug;

use proptest::{
    prelude::*,
    test_runner::{Config, TestError, TestRunner},
};
use rustyred_thg_derivation::{
    registered_algebras, resolve_algebra_token, AlgebraClass, BooleanSemiring,
    CorroborationSemiring, CountingSemiring, DeterminismPolicy, EqualityPolicy,
    IndependentNoisyOrAlgebra, LawProfile, MathematicalSemiring, Semiring, TropicalSemiring,
    ViterbiSemiring,
};

fn assert_mathematical_semiring<S: MathematicalSemiring>() {}

fn check_semiring_laws<S, Equal>(a: S::Element, b: S::Element, c: S::Element, equal: Equal)
where
    S: MathematicalSemiring,
    S::Element: Debug,
    Equal: Fn(&S::Element, &S::Element) -> bool,
{
    let descriptor = S::descriptor();
    assert!(descriptor.claims_mathematical_semiring());

    assert!(
        equal(&S::sum(&a, &S::zero()), &a),
        "additive right identity"
    );
    assert!(equal(&S::sum(&S::zero(), &a), &a), "additive left identity");
    assert!(
        equal(&S::product(&a, &S::one()), &a),
        "multiplicative right identity"
    );
    assert!(
        equal(&S::product(&S::one(), &a), &a),
        "multiplicative left identity"
    );

    assert!(
        equal(&S::sum(&S::sum(&a, &b), &c), &S::sum(&a, &S::sum(&b, &c))),
        "additive associativity"
    );
    assert!(
        equal(
            &S::product(&S::product(&a, &b), &c),
            &S::product(&a, &S::product(&b, &c)),
        ),
        "multiplicative associativity"
    );
    assert!(
        equal(&S::sum(&a, &b), &S::sum(&b, &a)),
        "additive commutativity"
    );

    if descriptor
        .laws
        .secondary
        .expect("mathematical semiring has a secondary operation")
        .commutative
    {
        assert!(
            equal(&S::product(&a, &b), &S::product(&b, &a)),
            "declared multiplicative commutativity"
        );
    }

    assert!(
        equal(
            &S::product(&a, &S::sum(&b, &c)),
            &S::sum(&S::product(&a, &b), &S::product(&a, &c)),
        ),
        "left distributivity"
    );
    assert!(
        equal(
            &S::product(&S::sum(&a, &b), &c),
            &S::sum(&S::product(&a, &c), &S::product(&b, &c)),
        ),
        "right distributivity"
    );
    assert!(
        equal(&S::product(&S::zero(), &a), &S::zero()),
        "left zero annihilation"
    );
    assert!(
        equal(&S::product(&a, &S::zero()), &S::zero()),
        "right zero annihilation"
    );
}

fn exact<T: PartialEq>(left: &T, right: &T) -> bool {
    left == right
}

fn approximate(left: &f64, right: &f64) -> bool {
    if left.is_infinite() || right.is_infinite() {
        left == right
    } else {
        (left - right).abs() <= 1.0e-9
    }
}

#[test]
fn registry_separates_orthogonal_algebra_classes() {
    assert_eq!(LawProfile::TRANSITION_MONOID.secondary, None);
    assert_eq!(LawProfile::SUMMARY_MONOID.secondary, None);
    assert!(LawProfile::ABELIAN_CHANGE_GROUP.primary.has_inverse);
    assert!(LawProfile::MATHEMATICAL_SEMIRING.is_mathematical_semiring());
    assert!(!LawProfile::GRAPHBLAS_OPERATOR_PAIR.is_mathematical_semiring());
    assert!(LawProfile::EVIDENCE_LATTICE.absorption);

    let registry = registered_algebras();
    assert_eq!(registry.len(), 6);
    assert_eq!(
        registry
            .iter()
            .filter(|descriptor| descriptor.claims_mathematical_semiring())
            .count(),
        5
    );

    assert_mathematical_semiring::<BooleanSemiring>();
    assert_mathematical_semiring::<CountingSemiring>();
    assert_mathematical_semiring::<TropicalSemiring>();
    assert_mathematical_semiring::<ViterbiSemiring>();
    assert_mathematical_semiring::<CorroborationSemiring>();
}

#[test]
fn legacy_probability_token_resolves_but_semiring_requirement_refuses_it() {
    let descriptor = resolve_algebra_token("probability").expect("legacy token remains readable");
    assert_eq!(descriptor, IndependentNoisyOrAlgebra::descriptor());
    assert_eq!(descriptor.class, AlgebraClass::IndependentNoisyOr);
    assert!(!descriptor.laws.left_distributive);
    assert!(!descriptor.laws.right_distributive);

    let refusal = descriptor
        .require_mathematical_semiring()
        .expect_err("noisy-OR must not cross a semiring-only boundary");
    assert_eq!(refusal.required, AlgebraClass::MathematicalSemiring);
    assert_eq!(refusal.actual, AlgebraClass::IndependentNoisyOr);

    let left =
        IndependentNoisyOrAlgebra::product(&0.5, &IndependentNoisyOrAlgebra::sum(&0.5, &0.5));
    let right = IndependentNoisyOrAlgebra::sum(
        &IndependentNoisyOrAlgebra::product(&0.5, &0.5),
        &IndependentNoisyOrAlgebra::product(&0.5, &0.5),
    );
    assert_eq!(left, 0.375);
    assert_eq!(right, 0.4375);
}

#[test]
fn float_descriptors_cannot_become_content_identity_from_approximate_equality() {
    for descriptor in registered_algebras() {
        if matches!(descriptor.equality, EqualityPolicy::ApproximateFloat { .. }) {
            assert!(!descriptor.content_identity_safe, "{}", descriptor.name);
            assert_eq!(
                descriptor.determinism,
                DeterminismPolicy::FixedReductionOrder
            );
        }
    }
}

proptest! {
    #![proptest_config(Config {
        failure_persistence: None,
        ..Config::default()
    })]

    #[test]
    fn boolean_law_matrix(a in any::<bool>(), b in any::<bool>(), c in any::<bool>()) {
        check_semiring_laws::<BooleanSemiring, _>(a, b, c, exact);
    }

    #[test]
    fn counting_law_matrix(
        a in prop_oneof![any::<u64>(), Just(u64::MAX), Just(u64::MAX - 1)],
        b in prop_oneof![any::<u64>(), Just(u64::MAX), Just(u64::MAX - 1)],
        c in prop_oneof![any::<u64>(), Just(u64::MAX), Just(u64::MAX - 1)],
    ) {
        check_semiring_laws::<CountingSemiring, _>(a, b, c, exact);
    }

    #[test]
    fn tropical_law_matrix(a in 0.0f64..1_000.0, b in 0.0f64..1_000.0, c in 0.0f64..1_000.0) {
        check_semiring_laws::<TropicalSemiring, _>(a, b, c, approximate);
    }

    #[test]
    fn viterbi_law_matrix(a in 0.0f64..=1.0, b in 0.0f64..=1.0, c in 0.0f64..=1.0) {
        check_semiring_laws::<ViterbiSemiring, _>(a, b, c, approximate);
    }

    #[test]
    fn corroboration_law_matrix(a in 0.0f64..=1.0, b in 0.0f64..=1.0, c in 0.0f64..=1.0) {
        check_semiring_laws::<CorroborationSemiring, _>(a, b, c, exact);
    }
}

#[derive(Clone, Copy, Debug)]
enum BrokenLaw {
    Closure,
    AdditiveIdentity,
    MultiplicativeIdentity,
    AdditiveAssociativity,
    MultiplicativeAssociativity,
    AdditiveCommutativity,
    MultiplicativeCommutativity,
    LeftDistributivity,
    RightDistributivity,
    ZeroAnnihilation,
}

impl BrokenLaw {
    const ALL: [Self; 10] = [
        Self::Closure,
        Self::AdditiveIdentity,
        Self::MultiplicativeIdentity,
        Self::AdditiveAssociativity,
        Self::MultiplicativeAssociativity,
        Self::AdditiveCommutativity,
        Self::MultiplicativeCommutativity,
        Self::LeftDistributivity,
        Self::RightDistributivity,
        Self::ZeroAnnihilation,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Closure => "closure",
            Self::AdditiveIdentity => "additive_identity",
            Self::MultiplicativeIdentity => "multiplicative_identity",
            Self::AdditiveAssociativity => "additive_associativity",
            Self::MultiplicativeAssociativity => "multiplicative_associativity",
            Self::AdditiveCommutativity => "additive_commutativity",
            Self::MultiplicativeCommutativity => "multiplicative_commutativity",
            Self::LeftDistributivity => "left_distributivity",
            Self::RightDistributivity => "right_distributivity",
            Self::ZeroAnnihilation => "zero_annihilation",
        }
    }
}

#[test]
fn deliberately_broken_fixtures_shrink_and_name_each_violated_law() {
    for law in BrokenLaw::ALL {
        let mut runner = TestRunner::new(Config {
            cases: 128,
            failure_persistence: None,
            ..Config::default()
        });
        let result = runner.run(&(0u8..=10, 0u8..=10, 0u8..=10), |(a, b, c)| {
            let holds = match law {
                BrokenLaw::Closure => u16::from(a) + 11 <= 10,
                BrokenLaw::AdditiveIdentity => a.saturating_add(1) == a,
                BrokenLaw::MultiplicativeIdentity => a.saturating_add(1) == a,
                BrokenLaw::AdditiveAssociativity => {
                    a.saturating_sub(b).saturating_sub(c) == a.saturating_sub(b.saturating_sub(c))
                }
                BrokenLaw::MultiplicativeAssociativity => {
                    a.saturating_sub(b).saturating_sub(c) == a.saturating_sub(b.saturating_sub(c))
                }
                BrokenLaw::AdditiveCommutativity => a.saturating_sub(b) == b.saturating_sub(a),
                BrokenLaw::MultiplicativeCommutativity => {
                    a.saturating_sub(b) == b.saturating_sub(a)
                }
                BrokenLaw::LeftDistributivity => {
                    a.saturating_add(b.saturating_add(c))
                        == a.saturating_add(b).saturating_add(a.saturating_add(c))
                }
                BrokenLaw::RightDistributivity => {
                    a.saturating_add(b).saturating_add(c)
                        == a.saturating_add(c).saturating_add(b.saturating_add(c))
                }
                BrokenLaw::ZeroAnnihilation => a.saturating_add(1) == 0,
            };
            prop_assert!(holds, "violated law: {}", law.name());
            Ok(())
        });

        match result {
            Err(TestError::Fail(reason, minimized)) => {
                assert!(reason.to_string().contains(law.name()), "{reason}");
                assert!(minimized.0 <= 10 && minimized.1 <= 10 && minimized.2 <= 10);
            }
            other => panic!("broken {} fixture was not rejected: {other:?}", law.name()),
        }
    }
}
