//! Per-semiring readouts of a dual-indeterminate provenance polynomial.
//!
//! One polynomial, six readouts. Specializing a positive-only polynomial
//! through a semiring's valuation reproduces that semiring's direct fold
//! exactly — the Green, Karvounarakis, Tannen universality property, extended
//! over the dual generators per Graedel.
//!
//! Every readout returns a pair, never a premature scalar. An element carries
//! `for` and `against` independently; collapsing them into one number is a
//! caller decision, because the right collapse depends on what the caller is
//! deciding. [`net_probability`] ships the common one and is documented as
//! noisy-and-not.
//!
//! | semiring | `for_value` | `against_value` |
//! |---|---|---|
//! | boolean | a supporting derivation exists | an attacking derivation exists |
//! | counting | count of distinct supporting derivations | count of distinct attacking derivations |
//! | tropical | cost of the cheapest supporting derivation | cost of the cheapest attacking derivation |
//! | probability | noisy-or of independent supporting derivations | noisy-or of independent attacking derivations |
//! | viterbi | best single supporting path | best single attacking path |
//! | corroboration | bottleneck strength of the best supporting derivation | bottleneck strength of the best attacking derivation |
//!
//! The tropical row reads *cost*, and cost runs opposite to confidence. Wrap
//! the annotation source in [`CostAnnotations`] for that readout; the other
//! five take confidence as it is stored.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::dual::{DualPolynomial, EdgeToken, Generator, Monomial};
use crate::semiring::Semiring;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Standing {
    In,
    Out,
    Undecided,
}

impl Standing {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Undecided => "undecided",
        }
    }
}

/// The primary result: for and against, valued in one semiring's carrier.
///
/// Collapsing to a scalar is the caller's decision, not the readout's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DualReadout<E> {
    pub for_value: E,
    pub against_value: E,
    pub standing: Standing,
}

impl<E: PartialEq> DualReadout<E> {
    /// Build a readout, deriving [`Standing`] from the pair and the carrier's
    /// zero.
    ///
    /// `In` when nothing attacks and something supports, `Out` when something
    /// attacks and nothing supports, `Undecided` when both hold or neither
    /// does. A caller holding a grounded-extension result may override
    /// `standing` with the fixpoint's answer.
    pub fn new(for_value: E, against_value: E, zero: &E) -> Self {
        let supported = for_value != *zero;
        let attacked = against_value != *zero;
        let standing = match (supported, attacked) {
            (true, false) => Standing::In,
            (false, true) => Standing::Out,
            _ => Standing::Undecided,
        };
        Self {
            for_value,
            against_value,
            standing,
        }
    }

    /// Re-derive [`Standing`] under the one-level defeat rule that the
    /// structural epistemic pass applies: any surviving attack puts the node
    /// `Out`, however well supported it is.
    ///
    /// [`DualReadout::new`] answers `Undecided` when a node is both supported
    /// and attacked, which is the honest reading of the pair on its own. The
    /// pass resolves that case to `Out`, so a caller comparing against a
    /// stored grounded status asks for this rule instead.
    ///
    /// Keeping the rule here is what lets a node's polynomial carry its
    /// supporting fragment intact. Folding defeat into every monomial also
    /// forces `Out`, but only by emptying the positive fragment, which drives
    /// `for_value` to zero and takes any usable [`net_probability`] with it.
    pub fn into_grounded(mut self, zero: &E) -> Self {
        self.standing = if self.against_value != *zero {
            Standing::Out
        } else if self.for_value != *zero {
            Standing::In
        } else {
            Standing::Undecided
        };
        self
    }
}

/// Annotation lookup: token to stored `f64` edge annotation (effective
/// confidence in practice).
pub trait AnnotationSource {
    fn annotation(&self, token: &EdgeToken) -> f64;
}

/// A map-backed [`AnnotationSource`]. Tokens absent from the map take
/// `default`, which callers set to the annotation an unweighted edge carries
/// (`1.0` for the graph's own convention).
#[derive(Clone, Debug)]
pub struct MapAnnotations {
    annotations: HashMap<EdgeToken, f64>,
    default: f64,
}

impl MapAnnotations {
    pub fn new(default: f64) -> Self {
        Self {
            annotations: HashMap::new(),
            default,
        }
    }

    pub fn with(mut self, token: impl Into<String>, annotation: f64) -> Self {
        self.annotations.insert(EdgeToken::new(token), annotation);
        self
    }

    pub fn insert(&mut self, token: impl Into<String>, annotation: f64) {
        self.annotations.insert(EdgeToken::new(token), annotation);
    }
}

impl Default for MapAnnotations {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl AnnotationSource for MapAnnotations {
    fn annotation(&self, token: &EdgeToken) -> f64 {
        self.annotations.get(token).copied().unwrap_or(self.default)
    }
}

/// Converts a confidence-valued [`AnnotationSource`] into a cost-valued one for
/// [`crate::TropicalSemiring`] readouts.
///
/// Tropical is min-plus: a smaller annotation is a cheaper and therefore better
/// derivation. Feeding it a confidence inverts the ordering, so an edge at
/// `0.1` would outrank one at `0.9` and lowering a source's reliability would
/// improve the result. Wrapping the source maps confidence `p` to cost
/// `-ln(p)`, which restores the ordering and lands the identities exactly:
/// `p = 1` becomes cost `0`, tropical's `one`, and `p = 0` becomes `+inf`,
/// tropical's `zero`. Summing costs along a path is then the log of the viterbi
/// product along it.
///
/// Use it for tropical readouts only. The other five carry confidence directly.
#[derive(Clone, Copy, Debug)]
pub struct CostAnnotations<A>(pub A);

impl<A: AnnotationSource> AnnotationSource for CostAnnotations<A> {
    fn annotation(&self, token: &EdgeToken) -> f64 {
        let confidence = self.0.annotation(token).clamp(0.0, 1.0);
        if confidence <= 0.0 {
            f64::INFINITY
        } else {
            -confidence.ln()
        }
    }
}

/// `base` multiplied by itself `exponent` times, by binary exponentiation.
fn power<S: Semiring>(base: &S::Element, exponent: u32) -> S::Element {
    let mut result = S::one();
    let mut accumulator = base.clone();
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = S::product(&result, &accumulator);
        }
        remaining >>= 1;
        if remaining > 0 {
            accumulator = S::product(&accumulator, &accumulator);
        }
    }
    result
}

/// `value` added to itself `count` times, by binary doubling.
///
/// A monomial's `N` coefficient counts derivations sharing one variable
/// multiset, so its readout is the semiring sum of that many copies: `k` under
/// counting, idempotent under boolean, tropical, viterbi, and corroboration.
fn scale<S: Semiring>(value: &S::Element, count: u64) -> S::Element {
    let mut result = S::zero();
    let mut accumulator = value.clone();
    let mut remaining = count;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = S::sum(&result, &accumulator);
        }
        remaining >>= 1;
        if remaining > 0 {
            accumulator = S::sum(&accumulator, &accumulator);
        }
    }
    result
}

fn valuate_monomial<S: Semiring>(
    monomial: &Monomial,
    annotations: &dyn AnnotationSource,
) -> S::Element {
    let mut product = S::one();
    for (generator, exponent) in &monomial.generators {
        let lifted = S::from_annotation(annotations.annotation(&generator.token));
        product = S::product(&product, &power::<S>(&lifted, *exponent));
    }
    scale::<S>(&product, monomial.coefficient)
}

fn valuate_fragment<S: Semiring>(
    poly: &DualPolynomial,
    annotations: &dyn AnnotationSource,
) -> S::Element {
    poly.monomials()
        .iter()
        .fold(S::zero(), |accumulator, monomial| {
            S::sum(&accumulator, &valuate_monomial::<S>(monomial, annotations))
        })
}

/// Specialize a dual polynomial into semiring `S` by homomorphism.
///
/// Positive generators valuate through `S::from_annotation` on the token's
/// annotation. The positive-only fragment yields `for_value`; the fragment
/// whose monomials contain negative generators yields `against_value`, with
/// each negative generator valuated through `S::from_annotation` on its own
/// annotation — the attack's strength composes like any derivation.
///
/// This is the semiring homomorphism, so it is exact for the five algebras that
/// carry the [`crate::MathematicalSemiring`] certificate. It is *not* exact for
/// the independent noisy-OR algebra, which is not distributive: on a DAG where
/// two derivations share a premise, expanding to sum-of-products counts the
/// shared premise twice. [`probability_readout`] is the corrected valuation for
/// that case.
pub fn interpret<S: Semiring>(
    poly: &DualPolynomial,
    annotations: &dyn AnnotationSource,
) -> DualReadout<S::Element> {
    DualReadout::new(
        valuate_fragment::<S>(&poly.positive_fragment(), annotations),
        valuate_fragment::<S>(&poly.defeating_fragment(), annotations),
        &S::zero(),
    )
}

// ---------------------------------------------------------------------------
// Probability: the one readout the homomorphism cannot serve
// ---------------------------------------------------------------------------

/// The largest number of monomials [`probability_readout`] will value exactly.
///
/// The probability of a disjunction of conjunctions over shared variables is
/// the DNF probability problem, and inclusion-exclusion over `n` monomials
/// costs `2^n`. Above this bound the readout refuses rather than returning an
/// approximation the caller cannot distinguish from an exact answer.
pub const MAX_INCLUSION_EXCLUSION_MONOMIALS: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error(
    "probability readout needs inclusion-exclusion over {monomials} monomials, above the exact \
     bound of {limit}"
)]
pub struct ProbabilityReadoutError {
    pub monomials: usize,
    pub limit: usize,
}

fn union_probability(
    poly: &DualPolynomial,
    annotations: &dyn AnnotationSource,
) -> Result<f64, ProbabilityReadoutError> {
    // Two derivations over the same variable set are the same event, and a
    // variable used twice in one derivation is still one event, so the
    // valuation works over supports: coefficients and exponents drop out.
    let supports: Vec<BTreeSet<&Generator>> = {
        let mut supports: Vec<BTreeSet<&Generator>> =
            poly.monomials().iter().map(Monomial::support).collect();
        supports.sort();
        supports.dedup();
        supports
    };
    if supports.is_empty() {
        return Ok(0.0);
    }
    if supports.len() > MAX_INCLUSION_EXCLUSION_MONOMIALS {
        return Err(ProbabilityReadoutError {
            monomials: supports.len(),
            limit: MAX_INCLUSION_EXCLUSION_MONOMIALS,
        });
    }

    let mut total = 0.0_f64;
    for mask in 1u32..(1u32 << supports.len()) {
        let mut union: BTreeSet<&Generator> = BTreeSet::new();
        for (index, support) in supports.iter().enumerate() {
            if mask & (1 << index) != 0 {
                union.extend(support.iter().copied());
            }
        }
        let joint = union.iter().fold(1.0_f64, |accumulator, generator| {
            accumulator * annotations.annotation(&generator.token).clamp(0.0, 1.0)
        });
        if mask.count_ones() % 2 == 1 {
            total += joint;
        } else {
            total -= joint;
        }
    }
    Ok(total.clamp(0.0, 1.0))
}

/// Probability readout that values a shared premise once.
///
/// Unlike [`interpret`], this does not assume the monomials are independent
/// events. Two derivations that share an edge share that edge's provenance
/// variable, so their joint probability is the product over the *union* of
/// their variables, not the product of their probabilities. Inclusion-exclusion
/// over the monomial supports is what makes that exact.
///
/// This corrects the double count the independent-sum path commits on any DAG
/// with a shared premise. On a DAG whose derivations share no edge the two
/// agree, because there the independence assumption happens to be true.
pub fn probability_readout(
    poly: &DualPolynomial,
    annotations: &dyn AnnotationSource,
) -> Result<DualReadout<f64>, ProbabilityReadoutError> {
    let for_value = union_probability(&poly.positive_fragment(), annotations)?;
    let against_value = union_probability(&poly.defeating_fragment(), annotations)?;
    Ok(DualReadout::new(for_value, against_value, &0.0))
}

/// The common scalar collapse for the probability readout: noisy-and-not.
///
/// `for_value * (1 - against_value)` — the conclusion stands to the degree it
/// is supported *and* not defeated. Other collapses are left to callers, which
/// is why only this one ships.
///
/// # This assumes the two fragments are independent
///
/// A [`DualReadout`] holds two marginals and no longer knows which generators
/// produced them, so multiplying them is only correct when the supporting and
/// defeating derivations share no generator. When they share a premise the
/// answer is wrong: with a positive `p * s`, a defeating `p * abar`, and
/// annotations `p = 0.5`, `s = 1`, `a = 0.5`, this returns
/// `0.5 * (1 - 0.25) = 0.375`, while the event `p and s and not a` has
/// probability `0.25`.
///
/// Use [`net_probability_of`] when the fragments may share generators. It takes
/// the polynomial, so it can keep the dependency this one has already lost.
pub fn net_probability(readout: &DualReadout<f64>) -> f64 {
    (readout.for_value * (1.0 - readout.against_value)).clamp(0.0, 1.0)
}

/// The noisy-and-not collapse computed jointly from the polynomial.
///
/// `P(supported and not defeated)`, evaluated as
/// `P(positive) - P(positive and defeating)`. Both terms go through the same
/// inclusion-exclusion over monomial supports that [`probability_readout`]
/// uses, so a generator shared between the fragments is counted once rather
/// than twice. Multiplying the two fragments also drops any monomial carrying
/// both polarities of one token, which is the algebra's way of saying that
/// combination cannot happen.
///
/// Prefer this to [`net_probability`] whenever the fragments may share a
/// generator, which is the ordinary case for a node whose support and attack
/// rest on a common premise.
pub fn net_probability_of(
    poly: &DualPolynomial,
    annotations: &dyn AnnotationSource,
) -> Result<f64, ProbabilityReadoutError> {
    let positive = poly.positive_fragment();
    let defeating = poly.defeating_fragment();
    let supported = union_probability(&positive, annotations)?;

    // Refuse before materializing. The defeating fragment can hold up to
    // MAX_DUAL_MONOMIALS terms, so building the product first and letting
    // union_probability reject it would allocate tens of thousands of
    // monomials, each cloning every generator on both sides, only to return
    // this same error. The bound is conservative: deduplication can only be
    // measured after the product exists, so a product whose distinct supports
    // would have fitted is refused too. A typed refusal beats the allocation.
    let product_terms = positive
        .monomials()
        .len()
        .saturating_mul(defeating.monomials().len());
    if product_terms > MAX_INCLUSION_EXCLUSION_MONOMIALS {
        return Err(ProbabilityReadoutError {
            monomials: product_terms,
            limit: MAX_INCLUSION_EXCLUSION_MONOMIALS,
        });
    }

    let both = union_probability(&positive.times(&defeating), annotations)?;
    Ok((supported - both).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Polarity;
    use crate::fold::fold;
    use crate::fold::{derivation_dag_ontology, fold_dual};
    use crate::semiring::{
        BooleanSemiring, CorroborationSemiring, CountingSemiring, ProbabilitySemiring,
        TropicalSemiring, ViterbiSemiring,
    };
    use crate::store::{
        DerivationNeighbor, DerivationStore, ASSUMPTION_LABEL, DERIVATION_ASSUMES_EDGE,
        DERIVATION_LABEL,
    };

    fn close(a: f64, b: f64) -> bool {
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        (a - b).abs() < 1e-9
    }

    /// Minimal [`DerivationStore`] over plain data, so these tests do not need
    /// `rustyred-thg-core`.
    #[derive(Default)]
    struct TestStore {
        labels: HashMap<String, Vec<String>>,
        /// edge id -> (from, to, edge type, annotation)
        edges: Vec<(String, String, String, String, f64)>,
    }

    impl TestStore {
        fn node(mut self, id: &str, label: &str) -> Self {
            self.labels.insert(id.to_string(), vec![label.to_string()]);
            self
        }

        fn edge(mut self, id: &str, from: &str, to: &str, annotation: f64) -> Self {
            self.edges.push((
                id.to_string(),
                from.to_string(),
                to.to_string(),
                DERIVATION_ASSUMES_EDGE.to_string(),
                annotation,
            ));
            self
        }

        /// An edge whose type is something other than the derivation
        /// relation, the way a defeating edge actually sits in a graph.
        fn typed_edge(
            mut self,
            id: &str,
            from: &str,
            to: &str,
            edge_type: &str,
            annotation: f64,
        ) -> Self {
            self.edges.push((
                id.to_string(),
                from.to_string(),
                to.to_string(),
                edge_type.to_string(),
                annotation,
            ));
            self
        }

        fn annotations(&self) -> MapAnnotations {
            let mut annotations = MapAnnotations::new(1.0);
            for (id, _, _, _, annotation) in &self.edges {
                annotations.insert(id.clone(), *annotation);
            }
            annotations
        }
    }

    impl DerivationStore for TestStore {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            self.labels.get(node_id).cloned()
        }

        fn edge_annotation(&self, edge_id: &str, _property: &str) -> Option<f64> {
            self.edges
                .iter()
                .find(|(id, ..)| id == edge_id)
                .map(|(_, _, _, _, annotation)| *annotation)
        }

        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            self.edges
                .iter()
                .filter(|(_, _, to, kind, _)| to == node_id && kind == edge_type)
                .map(|(id, from, _, _, _)| DerivationNeighbor {
                    node_id: from.clone(),
                    edge_id: id.clone(),
                })
                .collect()
        }

        fn edge_type(&self, edge_id: &str) -> Option<String> {
            self.edges
                .iter()
                .find(|(id, ..)| id == edge_id)
                .map(|(_, _, _, kind, _)| kind.clone())
        }
    }

    /// Mirrors `fold.rs`'s `build_three_layer_chain`.
    fn chain() -> TestStore {
        TestStore::default()
            .node("A", ASSUMPTION_LABEL)
            .node("B", DERIVATION_LABEL)
            .node("C", DERIVATION_LABEL)
            .edge("e_ab", "A", "B", 0.9)
            .edge("e_bc", "B", "C", 0.8)
    }

    /// Mirrors `fold.rs`'s `build_diamond`.
    fn diamond() -> TestStore {
        TestStore::default()
            .node("A", ASSUMPTION_LABEL)
            .node("B1", DERIVATION_LABEL)
            .node("B2", DERIVATION_LABEL)
            .node("C", DERIVATION_LABEL)
            .edge("e_ab1", "A", "B1", 0.9)
            .edge("e_ab2", "A", "B2", 0.7)
            .edge("e_b1c", "B1", "C", 0.8)
            .edge("e_b2c", "B2", "C", 0.6)
    }

    fn assert_parity<S: Semiring>(store: &TestStore, target: &str) {
        let poly = fold_dual(store, target, derivation_dag_ontology()).expect("dual fold");
        let readout = interpret::<S>(&poly, &store.annotations());
        assert_eq!(
            readout.for_value,
            fold::<S>(store, target),
            "{} readout diverged from its direct fold at {target}",
            S::name()
        );
        assert_eq!(
            readout.against_value,
            S::zero(),
            "{} found an attack in a positive-only DAG",
            S::name()
        );
        assert_eq!(readout.standing, Standing::In);
    }

    // -- section 8.3: universality parity, all six semirings ----------------

    #[test]
    fn universality_parity_on_the_chain_fixture() {
        let store = chain();
        assert_parity::<BooleanSemiring>(&store, "C");
        assert_parity::<CountingSemiring>(&store, "C");
        assert_parity::<TropicalSemiring>(&store, "C");
        assert_parity::<ProbabilitySemiring>(&store, "C");
        assert_parity::<ViterbiSemiring>(&store, "C");
        assert_parity::<CorroborationSemiring>(&store, "C");
    }

    #[test]
    fn universality_parity_on_the_diamond_fixture() {
        let store = diamond();
        assert_parity::<BooleanSemiring>(&store, "C");
        assert_parity::<CountingSemiring>(&store, "C");
        assert_parity::<TropicalSemiring>(&store, "C");
        assert_parity::<ProbabilitySemiring>(&store, "C");
        assert_parity::<ViterbiSemiring>(&store, "C");
        assert_parity::<CorroborationSemiring>(&store, "C");
    }

    #[test]
    fn universality_parity_on_assumption_and_missing_nodes() {
        let store = chain();
        for target in ["A", "missing"] {
            assert_parity_or_zero::<BooleanSemiring>(&store, target);
            assert_parity_or_zero::<CountingSemiring>(&store, target);
            assert_parity_or_zero::<TropicalSemiring>(&store, target);
            assert_parity_or_zero::<ProbabilitySemiring>(&store, target);
            assert_parity_or_zero::<ViterbiSemiring>(&store, target);
            assert_parity_or_zero::<CorroborationSemiring>(&store, target);
        }
    }

    fn assert_parity_or_zero<S: Semiring>(store: &TestStore, target: &str) {
        let poly = fold_dual(store, target, derivation_dag_ontology()).expect("dual fold");
        let readout = interpret::<S>(&poly, &store.annotations());
        assert_eq!(readout.for_value, fold::<S>(store, target), "{target}");
    }

    #[test]
    fn readout_values_match_the_documented_table() {
        let store = diamond();
        let poly = fold_dual(&store, "C", derivation_dag_ontology()).expect("dual fold");
        let annotations = store.annotations();

        assert!(interpret::<BooleanSemiring>(&poly, &annotations).for_value);
        assert_eq!(
            interpret::<CountingSemiring>(&poly, &annotations).for_value,
            2,
            "two distinct supporting derivations"
        );
        // Tropical: min(0.8 + 0.9, 0.6 + 0.7) = 1.3
        assert!(close(
            interpret::<TropicalSemiring>(&poly, &annotations).for_value,
            1.3
        ));
        // Viterbi: max(0.8 * 0.9, 0.6 * 0.7) = 0.72
        assert!(close(
            interpret::<ViterbiSemiring>(&poly, &annotations).for_value,
            0.72
        ));
        // Corroboration: max(min(0.8, 0.9), min(0.6, 0.7)) = 0.8
        assert!(close(
            interpret::<CorroborationSemiring>(&poly, &annotations).for_value,
            0.8
        ));
    }

    // -- section 8.4: diamond double-count correction ----------------------

    /// The spec's diamond with `a`'s own provenance made explicit.
    ///
    /// `EdgeToken` is the identity of one *edge use*, so a premise contributes a
    /// variable through the edge that establishes it. Give `a` a ground edge and
    /// both derivations of `d` carry `x_ga`; that shared variable is what the
    /// independent-sum path double counts.
    fn shared_premise_diamond() -> TestStore {
        TestStore::default()
            .node("g", ASSUMPTION_LABEL)
            .node("a", DERIVATION_LABEL)
            .node("b", DERIVATION_LABEL)
            .node("c", DERIVATION_LABEL)
            .node("d", DERIVATION_LABEL)
            .edge("e_ga", "g", "a", 0.8)
            .edge("e_ab", "a", "b", 0.8)
            .edge("e_ac", "a", "c", 0.8)
            .edge("e_bd", "b", "d", 0.8)
            .edge("e_cd", "c", "d", 0.8)
    }

    #[test]
    fn diamond_probability_values_the_shared_premise_once() {
        let store = shared_premise_diamond();
        let annotations = store.annotations();
        let poly = fold_dual(&store, "d", derivation_dag_ontology()).expect("dual fold");

        // d = x_bd·x_ab·x_ga + x_cd·x_ac·x_ga
        assert_eq!(poly.monomials().len(), 2);
        for monomial in poly.monomials() {
            assert_eq!(monomial.generators.len(), 3);
            assert_eq!(monomial.coefficient, 1);
        }

        // Hand-computed: P(m1) + P(m2) - P(m1 ∧ m2) with the union of variables
        // in the joint term, so x_ga is counted once.
        //   P(m1) = P(m2) = 0.8^3          = 0.512
        //   P(m1 ∧ m2) = 0.8^5             = 0.32768   (five distinct edges)
        //   P = 0.512 + 0.512 - 0.32768    = 0.69632
        let corrected = probability_readout(&poly, &annotations).expect("within the exact bound");
        assert!(
            close(corrected.for_value, 0.69632),
            "corrected probability was {}",
            corrected.for_value
        );
        assert_eq!(corrected.standing, Standing::In);

        // The current independent-sum path on the same fixture:
        //   noisy-or(0.8 * 0.64, 0.8 * 0.64) = 0.512 + 0.512 - 0.512^2 = 0.761856
        let independent_sum = fold::<ProbabilitySemiring>(&store, "d");
        assert!(
            close(independent_sum, 0.761856),
            "independent-sum path was {independent_sum}"
        );
        assert!(
            !close(corrected.for_value, independent_sum),
            "the corrected value must differ from the double-counting path"
        );
        assert!(
            corrected.for_value < independent_sum,
            "double counting inflates confidence"
        );

        // The naive homomorphic readout reproduces the old path exactly, which
        // is what localizes the defect: it is in the independence assumption,
        // not in the polynomial.
        let naive = interpret::<ProbabilitySemiring>(&poly, &annotations);
        assert!(close(naive.for_value, independent_sum));
    }

    #[test]
    fn diamond_without_a_shared_edge_needs_no_correction() {
        // The spec's four-edge sketch: a is itself the assumption, so the two
        // derivations of d share no provenance variable and independence is
        // true. Both paths agree, which is exactly the point.
        let store = TestStore::default()
            .node("a", ASSUMPTION_LABEL)
            .node("b", DERIVATION_LABEL)
            .node("c", DERIVATION_LABEL)
            .node("d", DERIVATION_LABEL)
            .edge("e_ab", "a", "b", 0.8)
            .edge("e_ac", "a", "c", 0.8)
            .edge("e_bd", "b", "d", 0.8)
            .edge("e_cd", "c", "d", 0.8);
        let poly = fold_dual(&store, "d", derivation_dag_ontology()).expect("dual fold");
        let corrected = probability_readout(&poly, &store.annotations()).expect("exact");
        let independent_sum = fold::<ProbabilitySemiring>(&store, "d");
        assert!(close(corrected.for_value, 0.8704));
        assert!(close(independent_sum, 0.8704));
    }

    #[test]
    fn probability_readout_refuses_past_the_exact_bound() {
        let mut poly = DualPolynomial::zero();
        for index in 0..(MAX_INCLUSION_EXCLUSION_MONOMIALS + 1) {
            poly = poly.plus(&DualPolynomial::generator(Generator::new(
                format!("e{index}"),
                Polarity::Positive,
                "Supports",
            )));
        }
        let err = probability_readout(&poly, &MapAnnotations::new(0.5)).unwrap_err();
        assert_eq!(err.limit, MAX_INCLUSION_EXCLUSION_MONOMIALS);
        assert_eq!(err.monomials, MAX_INCLUSION_EXCLUSION_MONOMIALS + 1);
    }

    // -- standing and defeat ------------------------------------------------

    fn support(token: &str) -> DualPolynomial {
        DualPolynomial::generator(Generator::new(token, Polarity::Positive, "Supports"))
    }

    fn attack(token: &str) -> DualPolynomial {
        DualPolynomial::generator(Generator::new(token, Polarity::Negative, "Undercuts"))
    }

    #[test]
    fn standing_is_in_out_or_undecided() {
        let annotations = MapAnnotations::new(0.5);

        let supported = interpret::<BooleanSemiring>(&support("e1"), &annotations);
        assert_eq!(supported.standing, Standing::In);
        assert!(supported.for_value && !supported.against_value);

        let attacked = interpret::<BooleanSemiring>(&attack("e2"), &annotations);
        assert_eq!(attacked.standing, Standing::Out);
        assert!(!attacked.for_value && attacked.against_value);

        let both = interpret::<BooleanSemiring>(&support("e1").plus(&attack("e2")), &annotations);
        assert_eq!(both.standing, Standing::Undecided);

        let neither = interpret::<BooleanSemiring>(&DualPolynomial::zero(), &annotations);
        assert_eq!(neither.standing, Standing::Undecided);
    }

    #[test]
    fn defeating_generators_valuate_through_their_own_annotation() {
        let annotations = MapAnnotations::new(1.0).with("e_attack", 0.25);
        let readout = interpret::<ViterbiSemiring>(&attack("e_attack"), &annotations);
        assert!(close(readout.against_value, 0.25));
        assert!(close(readout.for_value, 0.0));
    }

    #[test]
    fn a_defeated_monomial_lands_in_the_against_fragment_whole() {
        // x_s · xbar_a is one attacking derivation whose strength composes from
        // both edges.
        let annotations = MapAnnotations::new(1.0).with("s", 0.5).with("a", 0.5);
        let poly = support("s").times(&attack("a"));
        let readout = interpret::<ViterbiSemiring>(&poly, &annotations);
        assert!(close(readout.for_value, 0.0));
        assert!(close(readout.against_value, 0.25));
        assert_eq!(readout.standing, Standing::Out);
    }

    #[test]
    fn cost_annotations_restore_the_tropical_ordering() {
        // Confidence fed straight to tropical inverts the ranking: the weaker
        // edge looks cheaper and therefore better.
        let raw = MapAnnotations::new(1.0)
            .with("strong", 0.9)
            .with("weak", 0.1);
        let strong = support("strong");
        let weak = support("weak");
        assert!(
            interpret::<TropicalSemiring>(&weak, &raw).for_value
                < interpret::<TropicalSemiring>(&strong, &raw).for_value,
            "raw confidence ranks the weak edge as cheaper"
        );

        // Wrapped, the ordering is right: the strong edge costs less.
        let cost = CostAnnotations(raw);
        let strong_cost = interpret::<TropicalSemiring>(&strong, &cost).for_value;
        let weak_cost = interpret::<TropicalSemiring>(&weak, &cost).for_value;
        assert!(strong_cost < weak_cost, "{strong_cost} !< {weak_cost}");
        assert!(close(strong_cost, -0.9_f64.ln()));

        // The identities land exactly on tropical's one and zero.
        let ends = CostAnnotations(
            MapAnnotations::new(1.0)
                .with("certain", 1.0)
                .with("impossible", 0.0),
        );
        assert!(close(
            interpret::<TropicalSemiring>(&support("certain"), &ends).for_value,
            TropicalSemiring::one()
        ));
        assert!(close(
            interpret::<TropicalSemiring>(&support("impossible"), &ends).for_value,
            TropicalSemiring::zero()
        ));
    }

    #[test]
    fn tropical_over_costs_is_viterbi_in_log_space() {
        // Summing costs along a path is the log of the viterbi product.
        let raw = MapAnnotations::new(1.0).with("a", 0.5).with("b", 0.8);
        let path = support("a").times(&support("b"));
        let cost = interpret::<TropicalSemiring>(&path, &CostAnnotations(raw.clone())).for_value;
        let viterbi = interpret::<ViterbiSemiring>(&path, &raw).for_value;
        assert!(close(cost, -viterbi.ln()), "{cost} vs {}", -viterbi.ln());
    }

    #[test]
    fn a_shared_premise_makes_the_marginal_collapse_wrong() {
        // positive p*s and defeating p*abar share the premise p. Multiplying
        // the two marginals double counts it. The event p and s and not a has
        // probability 0.5 * 1 * 0.5 = 0.25.
        let annotations = MapAnnotations::new(1.0)
            .with("p", 0.5)
            .with("s", 1.0)
            .with("a", 0.5);

        let positive = support("p").times(&support("s"));
        let defeating = support("p").times(&attack("a"));
        let poly = positive.plus(&defeating);

        let readout = probability_readout(&poly, &annotations).expect("well inside the bound");
        assert!(close(readout.for_value, 0.5), "for {}", readout.for_value);
        assert!(
            close(readout.against_value, 0.25),
            "against {}",
            readout.against_value
        );

        // The marginal collapse assumes independence and overstates.
        assert!(
            close(net_probability(&readout), 0.375),
            "marginal {}",
            net_probability(&readout)
        );

        // The joint collapse keeps the shared premise.
        let joint = net_probability_of(&poly, &annotations).expect("well inside the bound");
        assert!(close(joint, 0.25), "joint {joint}");
    }

    #[test]
    fn the_two_collapses_agree_when_the_fragments_share_nothing() {
        let annotations = MapAnnotations::new(1.0).with("s", 0.8).with("a", 0.25);
        let poly = support("s").plus(&attack("a"));

        let readout = probability_readout(&poly, &annotations).expect("well inside the bound");
        let joint = net_probability_of(&poly, &annotations).expect("well inside the bound");
        assert!(
            close(net_probability(&readout), joint),
            "disjoint fragments must agree: {} vs {joint}",
            net_probability(&readout)
        );
        assert!(close(joint, 0.6), "joint {joint}");
    }

    #[test]
    fn net_probability_is_noisy_and_not() {
        let readout = DualReadout {
            for_value: 0.8,
            against_value: 0.25,
            standing: Standing::Undecided,
        };
        assert!(close(net_probability(&readout), 0.8 * 0.75));
    }

    #[test]
    fn coefficients_scale_by_repeated_semiring_sum() {
        let annotations = MapAnnotations::new(1.0);
        let doubled = support("e1").plus(&support("e1"));
        assert_eq!(doubled.monomials()[0].coefficient, 2);
        assert_eq!(
            interpret::<CountingSemiring>(&doubled, &annotations).for_value,
            2
        );
        // Idempotent carriers are unmoved by the coefficient.
        assert!(interpret::<BooleanSemiring>(&doubled, &annotations).for_value);
        assert!(close(
            interpret::<ViterbiSemiring>(&doubled, &annotations).for_value,
            1.0
        ));
    }

    #[test]
    fn exponents_apply_through_repeated_semiring_product() {
        let annotations = MapAnnotations::new(1.0).with("e1", 0.5);
        let squared = support("e1").times(&support("e1"));
        assert_eq!(squared.monomials()[0].generators[0].1, 2);
        assert!(close(
            interpret::<ViterbiSemiring>(&squared, &annotations).for_value,
            0.25
        ));
    }

    #[test]
    fn nilpotent_pairs_never_reach_a_readout() {
        let annotations = MapAnnotations::new(1.0);
        let poly = support("e1").times(&attack("e1"));
        let readout = interpret::<BooleanSemiring>(&poly, &annotations);
        assert!(!readout.for_value);
        assert!(!readout.against_value);
        assert_eq!(readout.standing, Standing::Undecided);
    }

    #[test]
    fn a_defeating_edge_in_the_graph_is_reachable_and_mints_a_negative_generator() {
        // An edge carries one type. A defeating edge is stored as `Undercuts`,
        // not as a derivation relation, so the traversal has to ask for the
        // defeating spellings the ontology declares or it can never see one.
        // Asking only for derivation_ancestor/derivation_assumes made the
        // production fold positive-only.
        let store = chain().typed_edge("e_attack", "A", "C", "Undercuts", 1.0);
        let poly = fold_dual(&store, "C", derivation_dag_ontology()).expect("dual fold");

        let negatives: Vec<_> = poly
            .generators()
            .into_iter()
            .filter(|generator| generator.polarity == Polarity::Negative)
            .cloned()
            .collect();
        assert_eq!(negatives.len(), 1, "the Undercuts edge must be reached");
        assert_eq!(negatives[0].canonical_type, "Undercuts");
        assert_eq!(negatives[0].token.as_str(), "e_attack");

        let mut annotations = store.annotations();
        annotations.insert("e_attack", 1.0);
        let readout = interpret::<BooleanSemiring>(&poly, &annotations);
        assert!(readout.against_value, "the attack must reach the readout");
    }

    #[test]
    fn an_incidental_positive_edge_is_not_counted_as_an_alternative_proof() {
        // A derivation node in a real graph often carries unrelated links. A
        // Supports edge into C is not a derivation of C, and the scalar fold
        // never sees it, so the dual fold must not either. Traversing every
        // valued class pulled these in and inflated counting and probability
        // against the scalar fold on any mixed-purpose graph.
        let store = chain().typed_edge("e_incidental", "A", "C", "Supports", 1.0);
        let poly = fold_dual(&store, "C", derivation_dag_ontology()).expect("dual fold");
        let annotations = store.annotations();

        assert_eq!(
            interpret::<CountingSemiring>(&poly, &annotations).for_value,
            fold::<CountingSemiring>(&store, "C"),
            "an incidental Supports edge must not add a derivation"
        );
        assert_eq!(
            interpret::<CountingSemiring>(&poly, &annotations).for_value,
            1
        );
        for (label, got, want) in [
            (
                "probability",
                interpret::<ProbabilitySemiring>(&poly, &annotations).for_value,
                fold::<ProbabilitySemiring>(&store, "C"),
            ),
            (
                "viterbi",
                interpret::<ViterbiSemiring>(&poly, &annotations).for_value,
                fold::<ViterbiSemiring>(&store, "C"),
            ),
        ] {
            assert!(close(got, want), "{label}: {got} != {want}");
        }

        // A defeating edge on the same node is still reached.
        let store = store.typed_edge("e_attack", "A", "C", "Undercuts", 1.0);
        let poly = fold_dual(&store, "C", derivation_dag_ontology()).expect("dual fold");
        assert_eq!(
            poly.generators()
                .iter()
                .filter(|generator| generator.polarity == Polarity::Negative)
                .count(),
            1
        );
    }

    #[test]
    fn defeating_spellings_are_all_reachable() {
        // Neighbor queries match exactly, so every declared spelling of a
        // defeating type has to be asked for, not just the canonical one.
        for spelling in [
            "Undercuts",
            "undercuts",
            "CONTRADICTS",
            "attacks",
            "Tension",
        ] {
            let store = chain().typed_edge("e_attack", "A", "C", spelling, 1.0);
            let poly = fold_dual(&store, "C", derivation_dag_ontology()).expect("dual fold");
            assert_eq!(
                poly.generators()
                    .iter()
                    .filter(|generator| generator.polarity == Polarity::Negative)
                    .count(),
                1,
                "spelling {spelling} was not reached"
            );
        }
    }

    #[test]
    fn unclassified_edge_types_refuse_rather_than_defaulting() {
        // default_registry() does not declare the derivation-DAG edge types, so
        // folding a derivation DAG through it reaches an edge it cannot class.
        let err = fold_dual(&chain(), "C", crate::ontology::default_registry()).unwrap_err();
        assert_eq!(
            err,
            crate::fold::DualFoldError::UnclassifiedEdge {
                edge_id: "e_ab".to_string(),
                edge_type: crate::store::DERIVATION_ASSUMES_EDGE.to_string(),
            }
        );
    }

    #[test]
    fn a_store_that_records_no_edge_types_refuses_rather_than_defaulting() {
        struct Untyped(TestStore);
        impl DerivationStore for Untyped {
            fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
                self.0.node_labels(node_id)
            }
            fn edge_annotation(&self, edge_id: &str, property: &str) -> Option<f64> {
                self.0.edge_annotation(edge_id, property)
            }
            fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
                self.0.neighbors_in(node_id, edge_type)
            }
            // `edge_type` left at the trait default, which returns None.
        }
        // The fold recurses before it mints, so the refusal names the deepest
        // untyped edge it reached. Either edge is a correct refusal; what
        // matters is that it refuses instead of assuming a polarity.
        let err = fold_dual(&Untyped(chain()), "C", derivation_dag_ontology()).unwrap_err();
        assert_eq!(
            err,
            crate::fold::DualFoldError::UntypedEdge {
                edge_id: "e_ab".to_string()
            }
        );
    }
}
