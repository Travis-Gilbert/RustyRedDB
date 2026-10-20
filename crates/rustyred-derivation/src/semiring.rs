use std::fmt::Debug;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::algebra::{
    AlgebraDescriptor, MathematicalSemiring, BOOLEAN_SEMIRING_DESCRIPTOR,
    CORROBORATION_SEMIRING_DESCRIPTOR, COUNTING_SEMIRING_DESCRIPTOR,
    INDEPENDENT_NOISY_OR_DESCRIPTOR, TROPICAL_SEMIRING_DESCRIPTOR, VITERBI_SEMIRING_DESCRIPTOR,
};

/// Compatibility ABI for a two-operation derivation algebra.
///
/// This trait predates runtime law descriptors, so implementing it is not by
/// itself a mathematical-semiring certificate. Consumers that depend on the
/// full laws must additionally require [`MathematicalSemiring`] or call
/// [`AlgebraDescriptor::require_mathematical_semiring`].
///
/// A mathematical semiring has two associative binary operations where:
/// - `sum` has identity `zero` (the additive monoid)
/// - `product` has identity `one` (the multiplicative monoid)
/// - `product` distributes over `sum`
/// - `zero` is annihilating for `product`
///
/// The four required instances match the PRD spec:
/// - **Boolean** — whether at least one derivation exists
/// - **Counting** — how many distinct derivations (capped by u64)
/// - **Tropical** — minimum-cost derivation (shortest proof)
/// - **Independent noisy-OR** — legacy probability aggregation, which is not
///   distributive and therefore does not implement [`MathematicalSemiring`]
pub trait Semiring {
    /// The element type carried by this semiring (e.g., `bool`, `u64`, `f64`).
    type Element: Clone + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync;

    /// Additive identity: `sum(zero(), a) == a` and `sum(a, zero()) == a`
    fn zero() -> Self::Element;

    /// Multiplicative identity: `product(one(), a) == a` and `product(a, one()) == a`
    fn one() -> Self::Element;

    /// Semiring addition: associative and commutative.
    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Semiring multiplication: associative.
    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Convert a raw `f64` edge annotation into this semiring's element type.
    ///
    /// Edge annotations are always stored as `f64` in the graph. This method
    /// lifts the annotation into the semiring domain for use in `product`.
    fn from_annotation(f: f64) -> Self::Element;

    /// A human-readable name for this semiring (e.g., `"boolean"`).
    fn name() -> &'static str;

    /// Runtime-readable law claim for this operational implementation.
    fn descriptor() -> &'static AlgebraDescriptor;

    /// Whether iterative fixed-point computation converges under this semiring.
    /// Absorptive semirings (e.g., Tropical, Boolean) support unbounded
    /// recursion; non-absorptive ones (e.g., Counting) require a depth bound.
    fn is_absorptive() -> bool {
        false
    }

    /// egglog `:merge` body, in terms of `old` and `new`. This is semiring sum.
    fn egglog_merge() -> &'static str;

    /// egglog expression combining two accumulated values. This is semiring product.
    /// Rendered with the caller's variable names.
    fn egglog_product(left: &str, right: &str) -> String;

    /// egglog sort name for `Element` ("f64", "i64", "bool").
    fn egglog_sort() -> &'static str;

    /// Literal rendering of `zero()` / `one()` in egglog source.
    fn egglog_zero() -> String;
    fn egglog_one() -> String;
}

// ---------------------------------------------------------------------------
// Boolean semiring
// ---------------------------------------------------------------------------

/// Boolean semiring: sum = OR, product = AND, zero = false, one = true.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanSemiring;

impl Semiring for BooleanSemiring {
    type Element = bool;

    fn zero() -> Self::Element {
        false
    }

    fn one() -> Self::Element {
        true
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        *a || *b
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        *a && *b
    }

    fn from_annotation(f: f64) -> Self::Element {
        f != 0.0
    }

    fn name() -> &'static str {
        "boolean"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &BOOLEAN_SEMIRING_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        true
    }

    fn egglog_merge() -> &'static str {
        "(or old new)"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(and {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "bool"
    }

    fn egglog_zero() -> String {
        "false".to_string()
    }

    fn egglog_one() -> String {
        "true".to_string()
    }
}

impl MathematicalSemiring for BooleanSemiring {}

// ---------------------------------------------------------------------------
// Counting semiring
// ---------------------------------------------------------------------------

/// Counting semiring: sum = saturated add, product = saturated mul,
/// zero = 0, one = 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountingSemiring;

impl Semiring for CountingSemiring {
    type Element = u64;

    fn zero() -> Self::Element {
        0
    }

    fn one() -> Self::Element {
        1
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.saturating_add(*b)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.saturating_mul(*b)
    }

    fn from_annotation(_f: f64) -> Self::Element {
        1 // each derivation step counts as 1
    }

    fn name() -> &'static str {
        "counting"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &COUNTING_SEMIRING_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        false
    }

    fn egglog_merge() -> &'static str {
        "(+ old new)"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(* {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "i64"
    }

    fn egglog_zero() -> String {
        "0".to_string()
    }

    fn egglog_one() -> String {
        "1".to_string()
    }
}

impl MathematicalSemiring for CountingSemiring {}

// ---------------------------------------------------------------------------
// Tropical semiring (min-plus)
// ---------------------------------------------------------------------------

/// Tropical semiring: sum = min, product = +, zero = +∞, one = 0.
///
/// Used for shortest-path / minimum-cost derivations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TropicalSemiring;

impl Semiring for TropicalSemiring {
    type Element = f64;

    fn zero() -> Self::Element {
        f64::INFINITY
    }

    fn one() -> Self::Element {
        0.0
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.min(*b)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn from_annotation(f: f64) -> Self::Element {
        f
    }

    fn name() -> &'static str {
        "tropical"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &TROPICAL_SEMIRING_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        true
    }

    fn egglog_merge() -> &'static str {
        "(min old new)"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(+ {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "f64"
    }

    fn egglog_zero() -> String {
        "1e308".to_string()
    }

    fn egglog_one() -> String {
        "0.0".to_string()
    }
}

impl MathematicalSemiring for TropicalSemiring {}

// ---------------------------------------------------------------------------
// Independent noisy-OR compatibility algebra
// ---------------------------------------------------------------------------

/// Independent noisy-OR algebra: sum = P(A ∪ B) = a + b - a·b,
/// product = P(A ∩ B) = a · b (independent events),
/// zero = 0, one = 1.
///
/// Elements are clamped to [0, 1] on construction. Noisy-OR with ordinary
/// multiplication is not distributive, so this type deliberately does not
/// implement [`MathematicalSemiring`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IndependentNoisyOrAlgebra;

impl Semiring for IndependentNoisyOrAlgebra {
    type Element = f64;

    fn zero() -> Self::Element {
        0.0
    }

    fn one() -> Self::Element {
        1.0
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        // P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        (a + b - a * b).clamp(0.0, 1.0)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        // P(A ∩ B) = P(A) * P(B) for independent events
        (a * b).clamp(0.0, 1.0)
    }

    fn from_annotation(f: f64) -> Self::Element {
        f.clamp(0.0, 1.0)
    }

    fn name() -> &'static str {
        "probability"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &INDEPENDENT_NOISY_OR_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        true
    }

    fn egglog_merge() -> &'static str {
        // egglog `+`/`*` are binary. Noisy-OR a+b-ab as nested binary ops.
        "(- (+ old new) (* old new))"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(* {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "f64"
    }

    fn egglog_zero() -> String {
        "0.0".to_string()
    }

    fn egglog_one() -> String {
        "1.0".to_string()
    }
}

/// Backward-compatible Rust name for the historical `"probability"` token.
///
/// The alias preserves source compatibility but does not grant the
/// [`MathematicalSemiring`] marker.
pub type ProbabilitySemiring = IndependentNoisyOrAlgebra;

// ---------------------------------------------------------------------------
// Viterbi semiring (max-times) — alias surface for ProvenanceSemiringKind::Viterbi
// ---------------------------------------------------------------------------

/// Viterbi / max-times: sum = max, product = ×, on `[0, 1]`.
///
/// This is what `ProvenanceSemiringKind::Viterbi` implements. MCP historically
/// advertised `"probability"` for fold_semiring, but that name was bound to
/// [`ProbabilitySemiring`] (noisy-OR). Prefer `"viterbi"` for max-times; keep
/// `"probability"` as the noisy-OR kind for backward compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ViterbiSemiring;

impl Semiring for ViterbiSemiring {
    type Element = f64;

    fn zero() -> Self::Element {
        0.0
    }

    fn one() -> Self::Element {
        1.0
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.max(*b).clamp(0.0, 1.0)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a.clamp(0.0, 1.0) * b.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }

    fn from_annotation(f: f64) -> Self::Element {
        f.clamp(0.0, 1.0)
    }

    fn name() -> &'static str {
        "viterbi"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &VITERBI_SEMIRING_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        true
    }

    fn egglog_merge() -> &'static str {
        "(max old new)"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(* {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "f64"
    }

    fn egglog_zero() -> String {
        "0.0".to_string()
    }

    fn egglog_one() -> String {
        "1.0".to_string()
    }
}

impl MathematicalSemiring for ViterbiSemiring {}

// ---------------------------------------------------------------------------
// Corroboration semiring (max-min bottleneck)
// ---------------------------------------------------------------------------

/// Corroboration / max-min: sum = max, product = min, on `[0, 1]`.
///
/// Bottleneck semantics: a derivation is as strong as its weakest premise;
/// multiple derivations take the best. Length-neutral (unlike Viterbi).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CorroborationSemiring;

impl Semiring for CorroborationSemiring {
    type Element = f64;

    fn zero() -> Self::Element {
        0.0
    }

    fn one() -> Self::Element {
        1.0
    }

    fn sum(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.max(*b).clamp(0.0, 1.0)
    }

    fn product(a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clamp(0.0, 1.0).min(b.clamp(0.0, 1.0))
    }

    fn from_annotation(f: f64) -> Self::Element {
        f.clamp(0.0, 1.0)
    }

    fn name() -> &'static str {
        "corroboration"
    }

    fn descriptor() -> &'static AlgebraDescriptor {
        &CORROBORATION_SEMIRING_DESCRIPTOR
    }

    fn is_absorptive() -> bool {
        true
    }

    fn egglog_merge() -> &'static str {
        "(max old new)"
    }

    fn egglog_product(left: &str, right: &str) -> String {
        format!("(min {left} {right})")
    }

    fn egglog_sort() -> &'static str {
        "f64"
    }

    fn egglog_zero() -> String {
        "0.0".to_string()
    }

    fn egglog_one() -> String {
        "1.0".to_string()
    }
}

impl MathematicalSemiring for CorroborationSemiring {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        (a - b).abs() < 1e-9
    }

    /// Verify identity laws: sum(a, zero) == a, product(a, one) == a
    fn assert_identity_laws<S: Semiring>(values: &[S::Element]) {
        for a in values {
            assert_eq!(S::sum(a, &S::zero()), *a, "sum(a, zero) != a");
            assert_eq!(S::sum(&S::zero(), a), *a, "sum(zero, a) != a");
            assert_eq!(S::product(a, &S::one()), *a, "product(a, one) != a");
            assert_eq!(S::product(&S::one(), a), *a, "product(one, a) != a");
        }
    }

    /// Verify associativity: sum(sum(a,b),c) == sum(a,sum(b,c))
    fn assert_associativity<S: Semiring + PartialEq>(values: &[S::Element])
    where
        S::Element: PartialEq,
    {
        for a in values {
            for b in values {
                for c in values {
                    assert_eq!(
                        S::sum(&S::sum(a, b), c),
                        S::sum(a, &S::sum(b, c)),
                        "sum associativity failed"
                    );
                    assert_eq!(
                        S::product(&S::product(a, b), c),
                        S::product(a, &S::product(b, c)),
                        "product associativity failed"
                    );
                }
            }
        }
    }

    /// Verify zero is annihilating for product: product(zero, a) == zero
    fn assert_annihilation<S: Semiring>(values: &[S::Element]) {
        let z = S::zero();
        for a in values {
            assert_eq!(S::product(&z, a), z, "product(zero, a) != zero");
            assert_eq!(S::product(a, &z), z, "product(a, zero) != zero");
        }
    }

    #[test]
    fn boolean_semiring_laws() {
        let values = vec![false, true];
        assert_identity_laws::<BooleanSemiring>(&values);
        assert_associativity::<BooleanSemiring>(&values);
        assert_annihilation::<BooleanSemiring>(&values);
        assert!(BooleanSemiring::is_absorptive());
    }

    #[test]
    fn counting_semiring_laws() {
        let values = vec![0u64, 1, 2, 5, u64::MAX];
        assert_identity_laws::<CountingSemiring>(&values);
        assert_associativity::<CountingSemiring>(&values);
        assert_annihilation::<CountingSemiring>(&values);
        assert!(!CountingSemiring::is_absorptive());
    }

    #[test]
    fn counting_sum_is_saturating() {
        assert_eq!(
            CountingSemiring::sum(&u64::MAX, &1),
            u64::MAX,
            "counting sum should saturate"
        );
    }

    #[test]
    fn tropical_semiring_laws() {
        let values = vec![0.0, 1.0, 2.0, 5.0, f64::INFINITY];
        // Use approximate comparison for f64
        let z = TropicalSemiring::zero();
        let o = TropicalSemiring::one();
        for &a in &values {
            assert!(close(TropicalSemiring::sum(&a, &z), a));
            assert!(close(TropicalSemiring::sum(&z, &a), a));
            assert!(close(TropicalSemiring::product(&a, &o), a));
            assert!(close(TropicalSemiring::product(&o, &a), a));
        }
        assert!(TropicalSemiring::is_absorptive());
    }

    #[test]
    fn tropical_min_plus_example() {
        // a --1--> b --1--> c  (cost 2)
        // a --------5-------> c  (cost 5)
        // tropical fold should pick cost 2
        let path1 = TropicalSemiring::product(&1.0, &1.0); // 1+1 = 2
        let path2 = 5.0;
        let best = TropicalSemiring::sum(&path1, &path2); // min(2, 5) = 2
        assert!(close(best, 2.0));
    }

    #[test]
    fn probability_semiring_laws() {
        let values = vec![0.0, 0.2, 0.5, 0.9, 1.0];
        let z = ProbabilitySemiring::zero();
        let o = ProbabilitySemiring::one();
        for &a in &values {
            assert!(close(ProbabilitySemiring::sum(&a, &z), a));
            assert!(close(ProbabilitySemiring::sum(&z, &a), a));
            assert!(close(ProbabilitySemiring::product(&a, &o), a));
            assert!(close(ProbabilitySemiring::product(&o, &a), a));
        }
        assert!(ProbabilitySemiring::is_absorptive());
    }

    #[test]
    fn probability_union_example() {
        // Two independent derivations, each with probability 0.5
        // P(at least one) = 0.5 + 0.5 - 0.5*0.5 = 0.75
        let p = ProbabilitySemiring::sum(&0.5, &0.5);
        assert!(close(p, 0.75));
    }

    #[test]
    fn probability_independent_conjunction() {
        // Two independent steps, each 0.8
        // P(both) = 0.8 * 0.8 = 0.64
        let p = ProbabilitySemiring::product(&0.8, &0.8);
        assert!(close(p, 0.64));
    }

    #[test]
    fn corroboration_semiring_laws() {
        let values = vec![0.0, 0.2, 0.5, 0.9, 1.0];
        let z = CorroborationSemiring::zero();
        let o = CorroborationSemiring::one();
        for &a in &values {
            assert!(close(CorroborationSemiring::sum(&a, &z), a));
            assert!(close(CorroborationSemiring::sum(&z, &a), a));
            assert!(close(CorroborationSemiring::product(&a, &o), a));
            assert!(close(CorroborationSemiring::product(&o, &a), a));
            for &b in &values {
                // Absorption: max(a, min(a, b)) = a
                assert!(close(
                    CorroborationSemiring::sum(&a, &CorroborationSemiring::product(&a, &b)),
                    a
                ));
            }
        }
        assert!(CorroborationSemiring::is_absorptive());
    }

    #[test]
    fn corroboration_is_length_neutral() {
        // Five 0.9 links: bottleneck stays 0.9 (unlike probability product decay).
        let mut acc = 1.0;
        for _ in 0..5 {
            acc = CorroborationSemiring::product(&acc, &0.9);
        }
        assert!(close(acc, 0.9));
        let short = CorroborationSemiring::product(&1.0, &0.9);
        assert!(close(acc, short));
    }

    fn render_f64(value: f64) -> String {
        let mut rendered = format!("{value:.17}");
        while rendered.contains('.') && rendered.ends_with('0') {
            rendered.pop();
        }
        if rendered.ends_with('.') {
            rendered.push('0');
        }
        rendered
    }

    fn egglog_product_parity_f64<S: Semiring<Element = f64>>() {
        let grid = [0.0_f64, 0.2, 0.5, 0.9, 1.0];
        for &a in &grid {
            for &b in &grid {
                let product_expr = S::egglog_product(&render_f64(a), &render_f64(b));
                let program = format!(
                    "(function acc () {sort} :merge {merge})\n(set (acc) {product})\n(run 1)\n",
                    sort = S::egglog_sort(),
                    merge = S::egglog_merge(),
                    product = product_expr,
                );
                let egraph = egraph::Engine::new(egraph::RulePack::new(
                    program.clone(),
                    egraph::CostModel::default(),
                ))
                .unwrap_or_else(|err| panic!("{} program failed: {err}\n{program}", S::name()));
                let output = egraph.function_rows("acc").unwrap();
                assert_eq!(output.rows.len(), 1);
                let value_term = &output.rows[0].output.value;
                let egraph::TermValue::Atom(egraph::Atom::F64(got)) = value_term else {
                    panic!("expected float lit, got {value_term:?}");
                };
                let expected = S::product(&a, &b);
                assert!(
                    close(*got, expected),
                    "{} product mismatch for ({a},{b}): egglog={} rust={}",
                    S::name(),
                    *got,
                    expected
                );
            }
        }
    }

    #[test]
    fn egglog_product_matches_rust_viterbi() {
        egglog_product_parity_f64::<ViterbiSemiring>();
    }

    #[test]
    fn egglog_product_matches_rust_tropical() {
        egglog_product_parity_f64::<TropicalSemiring>();
    }

    #[test]
    fn egglog_product_matches_rust_probability() {
        egglog_product_parity_f64::<ProbabilitySemiring>();
    }

    #[test]
    fn egglog_product_matches_rust_corroboration() {
        egglog_product_parity_f64::<CorroborationSemiring>();
    }

    #[test]
    fn egglog_product_matches_rust_boolean() {
        for a in [false, true] {
            for b in [false, true] {
                let left = if a { "true" } else { "false" };
                let right = if b { "true" } else { "false" };
                let product_expr = BooleanSemiring::egglog_product(left, right);
                let program = format!(
                    "(function acc () bool :merge {})\n(set (acc) {product_expr})\n(run 1)\n",
                    BooleanSemiring::egglog_merge()
                );
                let egraph = egraph::Engine::new(egraph::RulePack::new(
                    program,
                    egraph::CostModel::default(),
                ))
                .unwrap();
                let output = egraph.function_rows("acc").unwrap();
                assert_eq!(output.rows.len(), 1);
                let egraph::TermValue::Atom(egraph::Atom::Bool(got)) = &output.rows[0].output.value
                else {
                    panic!("expected bool");
                };
                assert_eq!(*got, BooleanSemiring::product(&a, &b));
            }
        }
    }

    #[test]
    fn egglog_product_matches_rust_counting() {
        for a in [0i64, 1, 2, 5] {
            for b in [0i64, 1, 2, 5] {
                let product_expr = CountingSemiring::egglog_product(&a.to_string(), &b.to_string());
                let program = format!(
                    "(function acc () i64 :merge {})\n(set (acc) {product_expr})\n(run 1)\n",
                    CountingSemiring::egglog_merge()
                );
                let egraph = egraph::Engine::new(egraph::RulePack::new(
                    program,
                    egraph::CostModel::default(),
                ))
                .unwrap();
                let output = egraph.function_rows("acc").unwrap();
                assert_eq!(output.rows.len(), 1);
                let egraph::TermValue::Atom(egraph::Atom::I64(got)) = &output.rows[0].output.value
                else {
                    panic!("expected int");
                };
                assert_eq!(
                    *got as u64,
                    CountingSemiring::product(&(a as u64), &(b as u64))
                );
            }
        }
    }

    #[test]
    fn egglog_surface_contract() {
        assert_eq!(ViterbiSemiring::egglog_merge(), "(max old new)");
        assert_eq!(TropicalSemiring::egglog_merge(), "(min old new)");
        assert_eq!(CorroborationSemiring::egglog_product("a", "b"), "(min a b)");
        assert_eq!(BooleanSemiring::egglog_sort(), "bool");
    }

    #[test]
    fn all_semirings_are_named() {
        assert_eq!(BooleanSemiring::name(), "boolean");
        assert_eq!(CountingSemiring::name(), "counting");
        assert_eq!(TropicalSemiring::name(), "tropical");
        assert_eq!(ProbabilitySemiring::name(), "probability");
        assert_eq!(CorroborationSemiring::name(), "corroboration");
        assert_eq!(ViterbiSemiring::name(), "viterbi");
    }
}
