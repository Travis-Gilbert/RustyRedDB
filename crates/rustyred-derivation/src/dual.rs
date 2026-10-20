//! Dual-indeterminate provenance polynomials: the free layer above the six
//! semirings.
//!
//! Semiring addition and multiplication are monotone and there is no
//! subtraction, so no semiring can express attack. Before this module the code
//! compensated by routing defeat around the fold entirely, into the Dung
//! grounded-extension fixpoint. Two valuation systems existed de facto —
//! positive composition through semirings, defeat through argumentation
//! semantics — with no shared vocabulary.
//!
//! `N[X, Xbar]` closes that gap. One fold of a derivation DAG produces a
//! dual-indeterminate provenance polynomial:
//! [`AlgebraicClass::MonotonePositive`] edges mint positive generators,
//! [`AlgebraicClass::Defeating`] edges mint negative ones, and
//! [`AlgebraicClass::Equivalence`] / [`AlgebraicClass::Structural`] edges mint
//! nothing and pass through as the multiplicative identity. The six semirings
//! then become homomorphic readouts of that one polynomial
//! ([`crate::interpret`](mod@crate::interpret)) rather than six separate folds. The fold that mints
//! those generators is [`crate::fold::fold_dual`], next to the scalar fold it
//! parallels.
//!
//! The quotient by the nilpotent-pair ideal is Graedel's: for one underlying
//! edge token `e`, `x_e * xbar_e` is the zero polynomial, because no derivation
//! uses a single edge both for and against its conclusion. That set of
//! monomials is closed under addition and under multiplication by anything, so
//! it is an ideal and the quotient is still a commutative semiring.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::algebra::{
    AlgebraClass, AlgebraDescriptor, DeterminismPolicy, EqualityPolicy, LawProfile,
};
use crate::ontology::AlgebraicClass;
use crate::semiring::Semiring;

/// A provenance token: the identity of one edge use.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EdgeToken(pub String);

impl EdgeToken {
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    /// The polarity an [`AlgebraicClass`] mints, or `None` for classes that are
    /// never valued.
    pub fn for_class(class: AlgebraicClass) -> Option<Self> {
        match class {
            AlgebraicClass::MonotonePositive => Some(Self::Positive),
            AlgebraicClass::Defeating => Some(Self::Negative),
            AlgebraicClass::Equivalence | AlgebraicClass::Structural => None,
        }
    }
}

/// One indeterminate: `x_e` ([`Polarity::Positive`]) or `xbar_e`
/// ([`Polarity::Negative`]), tagged with the canonical edge type so readouts
/// can distinguish `Undercuts` from `Contradicts` from `Tension` without
/// re-consulting the graph.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Generator {
    pub token: EdgeToken,
    pub polarity: Polarity,
    pub canonical_type: String,
}

impl Generator {
    pub fn new(
        token: impl Into<String>,
        polarity: Polarity,
        canonical_type: impl Into<String>,
    ) -> Self {
        Self {
            token: EdgeToken::new(token),
            polarity,
            canonical_type: canonical_type.into(),
        }
    }

    pub fn is_negative(&self) -> bool {
        self.polarity == Polarity::Negative
    }
}

/// A monomial: a sorted multiset of generators with an `N` coefficient.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Monomial {
    pub generators: Vec<(Generator, u32)>,
    pub coefficient: u64,
}

impl Monomial {
    /// The multiplicative unit monomial, coefficient `1`, no generators.
    pub fn unit() -> Self {
        Self {
            generators: Vec::new(),
            coefficient: 1,
        }
    }

    pub fn is_positive_only(&self) -> bool {
        !self.generators.iter().any(|(g, _)| g.is_negative())
    }

    /// Distinct generators, exponents dropped. Readouts that value a shared
    /// premise once work over this set rather than the multiset.
    pub fn support(&self) -> BTreeSet<&Generator> {
        self.generators.iter().map(|(g, _)| g).collect()
    }

    /// Canonicalize in place: merge repeated generators, drop zero exponents,
    /// and reject the monomial entirely when it carries a nilpotent pair.
    fn canonicalize(mut self) -> Option<Self> {
        if self.coefficient == 0 {
            return None;
        }
        self.generators.sort_by(|left, right| left.0.cmp(&right.0));

        let mut merged: Vec<(Generator, u32)> = Vec::with_capacity(self.generators.len());
        for (generator, exponent) in self.generators.into_iter() {
            if exponent == 0 {
                continue;
            }
            match merged.last_mut() {
                Some((last, last_exponent)) if *last == generator => {
                    *last_exponent = last_exponent.saturating_add(exponent);
                }
                _ => merged.push((generator, exponent)),
            }
        }

        // Graedel's nilpotent-pair quotient, keyed on the token alone: no
        // derivation uses one edge both for and against its conclusion.
        let mut positive_tokens: HashSet<&EdgeToken> = HashSet::new();
        let mut negative_tokens: HashSet<&EdgeToken> = HashSet::new();
        for (generator, _) in &merged {
            match generator.polarity {
                Polarity::Positive => positive_tokens.insert(&generator.token),
                Polarity::Negative => negative_tokens.insert(&generator.token),
            };
        }
        if positive_tokens
            .intersection(&negative_tokens)
            .next()
            .is_some()
        {
            return None;
        }

        self.generators = merged;
        Some(self)
    }
}

/// An element of `N[X, Xbar]` modulo the nilpotent-pair ideal.
///
/// The monomial list is kept in a unique canonical form, so equal polynomials
/// compare equal structurally and serialize identically. That is what lets
/// `DualPolynomial` take part in the receipts and content-address discipline
/// the rest of this crate uses.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(from = "RawDualPolynomial")]
pub struct DualPolynomial {
    monomials: Vec<Monomial>,
}

/// Deserialization shim.
///
/// The derived `Deserialize` would write `monomials` straight into the struct,
/// so a persisted payload could carry duplicate or unsorted generators, a zero
/// coefficient, or an `x_e * xbar_e` pair that should have normalized to zero.
/// Any of those breaks the uniqueness the type promises, and with it the
/// equality, serialization, and content-address discipline built on top.
/// Routing every load through [`DualPolynomial::from_monomials`] means a
/// polynomial read back from disk is canonical by construction.
#[derive(Deserialize)]
struct RawDualPolynomial {
    #[serde(default)]
    monomials: Vec<Monomial>,
}

impl From<RawDualPolynomial> for DualPolynomial {
    fn from(raw: RawDualPolynomial) -> Self {
        Self::from_monomials(raw.monomials)
    }
}

/// Largest monomial count [`crate::fold::fold_dual`] will accumulate.
///
/// One monomial is retained per complete derivation path, so a layered
/// branching DAG expands multiplicatively: a few dozen binary layers reach
/// billions of terms and exhaust memory. The fold refuses past this bound
/// instead of trying to materialize them. The bound is on the free layer
/// itself, so it binds before any readout runs.
pub const MAX_DUAL_MONOMIALS: usize = 4096;

impl DualPolynomial {
    pub fn zero() -> Self {
        Self {
            monomials: Vec::new(),
        }
    }

    pub fn one() -> Self {
        Self {
            monomials: vec![Monomial::unit()],
        }
    }

    pub fn generator(g: Generator) -> Self {
        Self::from_monomials(vec![Monomial {
            generators: vec![(g, 1)],
            coefficient: 1,
        }])
    }

    /// Build from raw monomials, canonicalizing.
    pub fn from_monomials(monomials: Vec<Monomial>) -> Self {
        let mut merged: BTreeMap<Vec<(Generator, u32)>, u64> = BTreeMap::new();
        for monomial in monomials {
            let Some(monomial) = monomial.canonicalize() else {
                continue;
            };
            let slot = merged.entry(monomial.generators).or_insert(0);
            *slot = slot.saturating_add(monomial.coefficient);
        }
        Self {
            monomials: merged
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != 0)
                .map(|(generators, coefficient)| Monomial {
                    generators,
                    coefficient,
                })
                .collect(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }

    pub fn monomials(&self) -> &[Monomial] {
        &self.monomials
    }

    /// The sub-polynomial whose monomials carry no negative generator.
    pub fn positive_fragment(&self) -> Self {
        Self {
            monomials: self
                .monomials
                .iter()
                .filter(|monomial| monomial.is_positive_only())
                .cloned()
                .collect(),
        }
    }

    /// The sub-polynomial whose monomials carry at least one negative
    /// generator.
    pub fn defeating_fragment(&self) -> Self {
        Self {
            monomials: self
                .monomials
                .iter()
                .filter(|monomial| !monomial.is_positive_only())
                .cloned()
                .collect(),
        }
    }

    /// Every distinct generator appearing anywhere in the polynomial.
    pub fn generators(&self) -> BTreeSet<&Generator> {
        self.monomials
            .iter()
            .flat_map(|monomial| monomial.generators.iter().map(|(g, _)| g))
            .collect()
    }

    pub fn plus(&self, other: &Self) -> Self {
        let mut monomials = self.monomials.clone();
        monomials.extend(other.monomials.iter().cloned());
        Self::from_monomials(monomials)
    }

    pub fn times(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut products = Vec::with_capacity(self.monomials.len() * other.monomials.len());
        for left in &self.monomials {
            for right in &other.monomials {
                let mut generators = left.generators.clone();
                generators.extend(right.generators.iter().cloned());
                products.push(Monomial {
                    generators,
                    coefficient: left.coefficient.saturating_mul(right.coefficient),
                });
            }
        }
        Self::from_monomials(products)
    }
}

// ---------------------------------------------------------------------------
// Semiring ABI
// ---------------------------------------------------------------------------

/// Law claim for the free dual layer.
///
/// Deliberately absent from [`crate::algebra::registered_algebras`]: that
/// registry is the catalog of *scalar readout* algebras a caller selects by
/// token (`fold_semiring`, the MCP surface). The free layer is the object those
/// readouts specialize from, not one of them, and its `Element` is a polynomial
/// rather than a scalar. Keeping it out of the token registry means
/// `resolve_algebra_token("dual_provenance")` stays `None`, so no caller can
/// route a scalar fold into it by accident.
pub static DUAL_PROVENANCE_DESCRIPTOR: AlgebraDescriptor = AlgebraDescriptor {
    name: "dual_provenance",
    compatibility_tokens: &[],
    class: AlgebraClass::MathematicalSemiring,
    laws: LawProfile::MATHEMATICAL_SEMIRING,
    equality: EqualityPolicy::Exact,
    determinism: DeterminismPolicy::OrderIndependent,
    content_identity_safe: true,
};

/// egglog sort name reported by [`DualPolynomial`].
///
/// The free layer has no egglog sort: its carrier is a polynomial, not one of
/// egglog's primitive sorts. This sentinel is returned instead of a plausible
/// but wrong sort so that any program built for it fails loudly at parse time
/// with the reason in the message, rather than silently saturating over the
/// wrong carrier.
pub const DUAL_NO_EGGLOG_SORT: &str = "dual-provenance-has-no-egglog-sort";

impl Semiring for DualPolynomial {
    type Element = DualPolynomial;

    fn zero() -> Self::Element {
        DualPolynomial::zero()
    }

    fn one() -> Self::Element {
        DualPolynomial::one()
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.plus(b)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.times(b)
    }

    /// Not meaningful for the free layer: a raw `f64` carries no edge identity,
    /// so it cannot name an indeterminate. Generator minting happens in
    /// [`fold_dual`](crate::fold::fold_dual), which sees the edge. Annotations re-enter at readout time
    /// through [`crate::interpret::AnnotationSource`].
    fn from_annotation(_f: f64) -> Self::Element {
        DualPolynomial::one()
    }

    fn name() -> &'static str {
        "dual_provenance"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &DUAL_PROVENANCE_DESCRIPTOR
    }

    /// `N[X, Xbar]` is not absorptive: `p + p*q != p` in general, exactly as in
    /// the counting semiring. Recursive folds therefore need a depth bound.
    fn is_absorptive() -> bool {
        false
    }

    fn egglog_merge() -> &'static str {
        DUAL_NO_EGGLOG_SORT
    }

    fn egglog_product(_left: &str, _right: &str) -> String {
        DUAL_NO_EGGLOG_SORT.to_string()
    }

    fn egglog_sort() -> &'static str {
        DUAL_NO_EGGLOG_SORT
    }

    fn egglog_zero() -> String {
        DUAL_NO_EGGLOG_SORT.to_string()
    }

    fn egglog_one() -> String {
        DUAL_NO_EGGLOG_SORT.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn pos(token: &str) -> Generator {
        Generator::new(token, Polarity::Positive, "Supports")
    }

    fn neg(token: &str) -> Generator {
        Generator::new(token, Polarity::Negative, "Undercuts")
    }

    #[test]
    fn identities_hold() {
        let x = DualPolynomial::generator(pos("e1"));
        assert_eq!(x.plus(&DualPolynomial::zero()), x);
        assert_eq!(DualPolynomial::zero().plus(&x), x);
        assert_eq!(x.times(&DualPolynomial::one()), x);
        assert_eq!(DualPolynomial::one().times(&x), x);
        assert_eq!(x.times(&DualPolynomial::zero()), DualPolynomial::zero());
        assert_eq!(DualPolynomial::zero().times(&x), DualPolynomial::zero());
    }

    #[test]
    fn nilpotent_pairs_normalize_to_zero() {
        let x = DualPolynomial::generator(pos("e1"));
        let x_bar = DualPolynomial::generator(neg("e1"));
        assert_eq!(x.times(&x_bar), DualPolynomial::zero());
        assert_eq!(x_bar.times(&x), DualPolynomial::zero());
    }

    #[test]
    fn distinct_tokens_are_not_nilpotent() {
        let x = DualPolynomial::generator(pos("e1"));
        let y_bar = DualPolynomial::generator(neg("e2"));
        let product = x.times(&y_bar);
        assert!(!product.is_zero());
        assert_eq!(product.monomials().len(), 1);
        assert_eq!(product.monomials()[0].generators.len(), 2);
    }

    #[test]
    fn nilpotency_survives_distribution() {
        // (x_e + x_f) * xbar_e == x_f * xbar_e: the x_e term is annihilated,
        // the x_f term is not.
        let sum = DualPolynomial::generator(pos("e")).plus(&DualPolynomial::generator(pos("f")));
        let got = sum.times(&DualPolynomial::generator(neg("e")));
        let want = DualPolynomial::generator(pos("f")).times(&DualPolynomial::generator(neg("e")));
        assert_eq!(got, want);
    }

    #[test]
    fn like_monomials_merge_their_coefficients() {
        let x = DualPolynomial::generator(pos("e1"));
        let doubled = x.plus(&x);
        assert_eq!(doubled.monomials().len(), 1);
        assert_eq!(doubled.monomials()[0].coefficient, 2);
    }

    #[test]
    fn repeated_generators_merge_their_exponents() {
        let x = DualPolynomial::generator(pos("e1"));
        let squared = x.times(&x);
        assert_eq!(squared.monomials().len(), 1);
        assert_eq!(squared.monomials()[0].generators.len(), 1);
        assert_eq!(squared.monomials()[0].generators[0].1, 2);
    }

    #[test]
    fn canonical_form_is_order_independent() {
        let x = DualPolynomial::generator(pos("e1"));
        let y = DualPolynomial::generator(pos("e2"));
        assert_eq!(x.plus(&y), y.plus(&x));
        assert_eq!(x.times(&y), y.times(&x));
        assert_eq!(
            serde_json::to_string(&x.times(&y)).unwrap(),
            serde_json::to_string(&y.times(&x)).unwrap(),
            "canonical form must serialize identically for content addressing"
        );
    }

    #[test]
    fn fragments_partition_the_polynomial() {
        let poly = DualPolynomial::generator(pos("e1"))
            .plus(&DualPolynomial::generator(neg("e2")))
            .plus(
                &DualPolynomial::generator(pos("e3")).times(&DualPolynomial::generator(neg("e4"))),
            );
        assert_eq!(poly.positive_fragment().monomials().len(), 1);
        assert_eq!(poly.defeating_fragment().monomials().len(), 2);
        assert_eq!(
            poly.positive_fragment().monomials().len()
                + poly.defeating_fragment().monomials().len(),
            poly.monomials().len()
        );
    }

    #[test]
    fn deserialization_canonicalizes_a_noncanonical_payload() {
        // Hand-written payload with everything canonicalization must repair:
        // an unsorted duplicate pair, a zero coefficient, and a nilpotent pair.
        let payload = r#"{"monomials":[
            {"generators":[["e2",1],["e1",1],["e1",1]],"coefficient":2},
            {"generators":[],"coefficient":0},
            {"generators":[["e3p",1],["e3n",1]],"coefficient":5}
        ]}"#
        .replace(
            r#"["e2",1]"#,
            r#"[{"token":"e2","polarity":"positive","canonical_type":"Supports"},1]"#,
        )
        .replace(
            r#"["e1",1]"#,
            r#"[{"token":"e1","polarity":"positive","canonical_type":"Supports"},1]"#,
        )
        .replace(
            r#"["e3p",1]"#,
            r#"[{"token":"e3","polarity":"positive","canonical_type":"Supports"},1]"#,
        )
        .replace(
            r#"["e3n",1]"#,
            r#"[{"token":"e3","polarity":"negative","canonical_type":"Undercuts"},1]"#,
        );

        let loaded: DualPolynomial = serde_json::from_str(&payload).expect("deserialize");

        // The zero-coefficient and nilpotent monomials are gone; the survivor
        // has merged exponents and sorted generators.
        assert_eq!(loaded.monomials().len(), 1);
        let monomial = &loaded.monomials()[0];
        assert_eq!(monomial.coefficient, 2);
        assert_eq!(monomial.generators.len(), 2);
        assert_eq!(monomial.generators[0].0.token, EdgeToken::new("e1"));
        assert_eq!(monomial.generators[0].1, 2, "duplicate exponents merge");
        assert_eq!(monomial.generators[1].0.token, EdgeToken::new("e2"));

        // Canonical by construction: re-canonicalizing is a no-op, and the
        // loaded value equals the same polynomial built through the API.
        assert_eq!(
            DualPolynomial::from_monomials(loaded.monomials().to_vec()),
            loaded
        );
    }

    #[test]
    fn round_tripping_through_serde_is_stable() {
        let poly = DualPolynomial::generator(pos("e1"))
            .times(&DualPolynomial::generator(pos("e2")))
            .plus(&DualPolynomial::generator(neg("e3")));
        let json = serde_json::to_string(&poly).expect("serialize");
        let back: DualPolynomial = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, poly);
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn descriptor_is_a_semiring_but_is_not_token_selectable() {
        assert!(DualPolynomial::descriptor().claims_mathematical_semiring());
        assert_eq!(
            crate::algebra::resolve_algebra_token("dual_provenance"),
            None,
            "the free layer must not be selectable as a scalar readout"
        );
        assert_eq!(crate::algebra::registered_algebras().len(), 6);
    }

    #[test]
    fn egglog_surface_refuses_rather_than_guessing_a_sort() {
        assert_eq!(DualPolynomial::egglog_sort(), DUAL_NO_EGGLOG_SORT);
        for sort in ["bool", "i64", "f64"] {
            assert_ne!(DualPolynomial::egglog_sort(), sort);
        }
    }

    // -- property tests -----------------------------------------------------

    fn arb_generator() -> impl Strategy<Value = Generator> {
        (0usize..4, prop::bool::ANY).prop_map(|(index, positive)| Generator {
            token: EdgeToken::new(format!("e{index}")),
            polarity: if positive {
                Polarity::Positive
            } else {
                Polarity::Negative
            },
            canonical_type: if positive { "Supports" } else { "Undercuts" }.to_string(),
        })
    }

    fn arb_monomial() -> impl Strategy<Value = Monomial> {
        (
            prop::collection::vec((arb_generator(), 1u32..3), 0..3),
            1u64..4,
        )
            .prop_map(|(generators, coefficient)| Monomial {
                generators,
                coefficient,
            })
    }

    fn arb_polynomial() -> impl Strategy<Value = DualPolynomial> {
        prop::collection::vec(arb_monomial(), 0..4).prop_map(DualPolynomial::from_monomials)
    }

    proptest! {
        #![proptest_config(ProptestConfig { failure_persistence: None, ..ProptestConfig::default() })]

        #[test]
        fn plus_is_associative(a in arb_polynomial(), b in arb_polynomial(), c in arb_polynomial()) {
            prop_assert_eq!(a.plus(&b).plus(&c), a.plus(&b.plus(&c)));
        }

        #[test]
        fn plus_is_commutative(a in arb_polynomial(), b in arb_polynomial()) {
            prop_assert_eq!(a.plus(&b), b.plus(&a));
        }

        #[test]
        fn times_is_associative(a in arb_polynomial(), b in arb_polynomial(), c in arb_polynomial()) {
            prop_assert_eq!(a.times(&b).times(&c), a.times(&b.times(&c)));
        }

        #[test]
        fn times_is_commutative(a in arb_polynomial(), b in arb_polynomial()) {
            prop_assert_eq!(a.times(&b), b.times(&a));
        }

        #[test]
        fn identities(a in arb_polynomial()) {
            prop_assert_eq!(a.plus(&DualPolynomial::zero()), a.clone());
            prop_assert_eq!(a.times(&DualPolynomial::one()), a.clone());
        }

        #[test]
        fn zero_annihilates(a in arb_polynomial()) {
            prop_assert_eq!(a.times(&DualPolynomial::zero()), DualPolynomial::zero());
        }

        #[test]
        fn times_distributes_over_plus(
            a in arb_polynomial(),
            b in arb_polynomial(),
            c in arb_polynomial(),
        ) {
            prop_assert_eq!(a.times(&b.plus(&c)), a.times(&b).plus(&a.times(&c)));
        }

        #[test]
        fn canonical_form_is_unique(a in arb_polynomial()) {
            // Re-canonicalizing an already canonical polynomial is a no-op, and
            // every monomial that survives is nilpotent-free with a nonzero
            // coefficient and strictly increasing generators.
            prop_assert_eq!(DualPolynomial::from_monomials(a.monomials().to_vec()), a.clone());
            for monomial in a.monomials() {
                prop_assert!(monomial.coefficient > 0);
                for window in monomial.generators.windows(2) {
                    prop_assert!(window[0].0 < window[1].0);
                }
                for (_, exponent) in &monomial.generators {
                    prop_assert!(*exponent > 0);
                }
                // No surviving monomial carries a nilpotent pair.
                for (generator, _) in &monomial.generators {
                    let paired = monomial.generators.iter().any(|(other, _)| {
                        other.token == generator.token && other.polarity != generator.polarity
                    });
                    prop_assert!(!paired);
                }
            }
        }
    }
}
