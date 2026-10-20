//! Thin adapter from storage records into the derivation crate's vocabulary.
//!
//! The dependency direction is fixed by the workspace: core depends on
//! `rustyred-thg-derivation`, and derivation does not see [`EdgeRecord`]. The
//! registry therefore speaks in edge-type strings, epistemic-type strings, and
//! `f64` annotations, and this module is the only place that translates
//! storage records into them.

use std::collections::BTreeSet;

use rustyred_thg_derivation::dual::{DualPolynomial, EdgeToken, Generator, Polarity};
use rustyred_thg_derivation::interpret::AnnotationSource;
use rustyred_thg_derivation::ontology::{EdgeOntology, EdgeTypeSpec};

use rustyred_thg_derivation::store::ANNOTATION_PROPERTY;

use crate::epistemic::{
    effective_confidence, normalized_confidence, EpistemicRelationInput, SourceReliability,
};
use crate::graph::capability::GraphStore;
use crate::graph::record::EdgeRecord;

/// The generator one stored edge mints, if any.
///
/// `None` covers both an edge type the ontology does not classify and an edge
/// type whose class is never valued (`Equivalence`, `Structural`). Neither
/// grants trust: an unclassified edge contributes nothing rather than acquiring
/// a default polarity.
pub fn lift_edge(edge: &EdgeRecord, ontology: &EdgeOntology) -> Option<Generator> {
    let spec = ontology.resolve(&edge.edge_type)?;
    Polarity::for_class(spec.class).map(|polarity| Generator {
        token: EdgeToken::new(edge.id.clone()),
        polarity,
        canonical_type: spec.canonical.clone(),
    })
}

/// [`AnnotationSource`] backed by a [`GraphStore`]'s edges.
///
/// The annotation is the edge's effective confidence, weighted by the source
/// reliability recorded on the edge when there is one. A token naming no edge
/// in the store annotates as `1.0`, the graph's convention for an unweighted
/// edge.
///
/// The `annotation` edge property wins when present. That is the property
/// `DerivationStore::edge_annotation` reads, so a derivation-DAG edge carries
/// its weight there and normally leaves `confidence` unset. Reading only
/// `confidence` gave every such edge `1.0`, which silently broke agreement
/// between a `fold_dual_graph` readout and the scalar fold over the same graph
/// for viterbi, probability, tropical, and corroboration.
///
/// This yields *confidence*, where larger is stronger. The tropical readout is
/// min-plus and reads its annotation as a *cost*, where smaller is better, so
/// handing it confidence directly inverts the ranking. Wrap this in
/// [`rustyred_thg_derivation::interpret::CostAnnotations`] for tropical
/// readouts. The other five semirings take confidence as it is stored.
pub struct StoreAnnotations<'a, S: GraphStore>(pub &'a S);

/// The weight one edge carries for a derivation readout: the `annotation`
/// property when it holds a number, otherwise the edge's effective confidence
/// weighted by any recorded source reliability.
///
/// Shared so the two paths over the same graph cannot drift. [`StoreAnnotations`]
/// feeds the dual readout and `GraphDerivationStore::edge_annotation` feeds the
/// fold. While only the former consulted `confidence`, an edge carrying its
/// weight there and not in `annotation` valued 0.5 on the dual side and 1.0 on
/// the scalar side, so probability, viterbi, tropical, and corroboration
/// disagreed on the same graph.
///
/// An edge with neither reads 1.0, the graph's convention for an unweighted
/// edge, because [`EdgeRecord::effective_confidence`] defaults to 1.0.
pub fn edge_annotation_weight(edge: &EdgeRecord) -> f64 {
    if let Some(annotation) = edge
        .properties
        .get(ANNOTATION_PROPERTY)
        .and_then(serde_json::Value::as_f64)
    {
        return annotation;
    }
    let reliability = edge
        .properties
        .get("source_reliability")
        .and_then(|value| serde_json::from_value::<SourceReliability>(value.clone()).ok());
    effective_confidence(edge.effective_confidence(), reliability.as_ref())
}

impl<S: GraphStore> AnnotationSource for StoreAnnotations<'_, S> {
    fn annotation(&self, token: &EdgeToken) -> f64 {
        self.0
            .get_edge(token.as_str())
            .map(edge_annotation_weight)
            .unwrap_or(1.0)
    }
}

/// Deterministic provenance token for an epistemic relation.
///
/// Relations reach the pass from three places — induced store edges, inferred
/// candidate pairs, and explicit input — and only the first has an edge id, so
/// the token is derived from the relation itself. Distinct relation kinds
/// between the same pair get distinct tokens.
///
/// Each content id is length-prefixed rather than joined by a delimiter.
/// Concatenating raw ids lets distinct relations collide whenever an id
/// contains the separator: `from="a->b", to="c"` and `from="a", to="b->c"`
/// would both encode as `Supports:a->b->c`, and colliding relations would then
/// share one generator and one annotation, corrupting coefficients and the
/// probability readout's dependency structure. A length prefix makes the
/// encoding injective for any id content.
/// The canonical row a relation lands on: its stored `canonical_edge_type` when
/// that resolves in the same algebraic class as the relation kind, otherwise the
/// row the kind itself resolves to.
///
/// Shared so identity and polarity cannot disagree. [`relation_token`] embeds
/// this and [`lift_relation`] mints its generator from it. While only the latter
/// resolved, `Contradicts` and `CONTRADICTS` produced one generator but two
/// tokens, so one relation became two provenance variables and
/// `probability_readout`, which is inclusion-exclusion over monomial supports,
/// counted duplicate evidence as independent and inflated the result. A value
/// that fails to resolve, or resolves into another class, changed identity here
/// while `lift_relation` ignored it.
fn canonical_row<'a>(
    relation: &EpistemicRelationInput,
    ontology: &'a EdgeOntology,
) -> Option<&'a EdgeTypeSpec> {
    let by_kind = ontology.resolve(relation.kind.edge_type())?;
    Some(
        relation
            .canonical_edge_type
            .as_deref()
            .and_then(|token| ontology.resolve(token))
            .filter(|stored| stored.class == by_kind.class)
            .unwrap_or(by_kind),
    )
}

pub fn relation_token(relation: &EpistemicRelationInput, ontology: &EdgeOntology) -> String {
    let kind = relation.kind.edge_type();
    let from = &relation.from_content_id;
    let to = &relation.to_content_id;
    // The stored row belongs in the identity. Every defeating relation
    // collapses to EpistemicRelationKind::Undercuts, so parallel Contradicts
    // and Tension rows on the same endpoints shared one token. Their
    // generators stayed distinct because canonical_type differs, but
    // RelationAnnotations::annotation looks the token up with `.find()`, so
    // both were valued from whichever relation came first, silently corrupting
    // the probability, Viterbi and corroboration readouts.
    //
    // Resolved, not raw: the spelling a caller happened to store must not split
    // one relation into two provenance variables.
    let canonical = canonical_row(relation, ontology)
        .map(|spec| spec.canonical.as_str())
        .unwrap_or("");
    format!(
        "{}:{}:{}:{}:{}:{}:{}",
        kind,
        canonical.len(),
        canonical,
        from.len(),
        from,
        to.len(),
        to
    )
}

/// The generator an epistemic relation mints, resolved through `ontology`.
pub fn lift_relation(
    relation: &EpistemicRelationInput,
    ontology: &EdgeOntology,
) -> Option<Generator> {
    // Prefer the stored row this relation came from. Every defeating relation
    // collapses to EpistemicRelationKind::Undercuts, so resolving the kind
    // alone tags a Contradicts or Tension edge as Undercuts, and
    // Generator::canonical_type is exactly what a caller reads to tell the
    // three apart. The stored string wins only when it lands in the same
    // algebraic class, so a wrong value cannot flip a defeat into a support.
    let spec = canonical_row(relation, ontology)?;
    Polarity::for_class(spec.class).map(|polarity| Generator {
        token: EdgeToken::new(relation_token(relation, ontology)),
        polarity,
        canonical_type: spec.canonical.clone(),
    })
}

/// The dual polynomial for one content node, built from the relations incident
/// on it.
///
/// ```text
///   base(n)  = 1                       when nothing supports n
///            = sum of x_s over supporting relations s -> n
///   poly(n)  = 0                       when n sits in a contradiction cycle
///            = base(n) (+) sum of xbar_a over attacking relations a -> n
/// ```
///
/// Three things follow:
///
/// - A node nothing supports stands on its own, which is the `1`. That matches
///   Dung's reading that an argument holds unless defeated, and the derivation
///   fold where a premise-free node is `one()`. A node something supports is
///   worth exactly its supports, again matching the fold. Keeping the `1`
///   alongside real supports would leave an empty monomial in the positive
///   fragment, and an empty monomial reads as certainty under every semiring,
///   which would pin `for_value` at the carrier's one however weak the support.
/// - Defeat is summed beside the support, never multiplied into it. Multiplying
///   also forces the boolean standing to `Out`, but only by leaving no
///   positive-only monomial, which drives `for_value` to zero for every
///   attacked node and takes any usable
///   [`rustyred_thg_derivation::interpret::net_probability`] with it. The
///   fragments stay separate, and the one-level defeat rule that
///   [`crate::epistemic::structural_epistemic_pass`] applies lives in
///   [`rustyred_thg_derivation::interpret::DualReadout::into_grounded`], which
///   is what a caller comparing against a stored
///   [`crate::epistemic::GroundedExtensionStatus`] asks for.
/// - A node in a contradiction cycle contributes zero on both sides, so it
///   reads `Undecided` under either rule, which is the answer the fixpoint
///   gives and the convention the recursive fold uses for a cycle.
///
/// Relations whose normalized confidence is `0.0` are skipped. Such a relation
/// asserts nothing, and admitting one would put a generator carrying no weight
/// on the against side, flipping the node `Out` on an attack that asserts
/// nothing.
///
/// Note the residual divergence this leaves with
/// [`crate::epistemic::structural_epistemic_pass`], which counts a relation in
/// `attack_in_degree` on existence alone and would record `Out` for that same
/// zero-confidence attack. The grounded-parity claim therefore holds for
/// relations carrying nonzero confidence. Making the pass ignore
/// zero-confidence relations too would close the gap, but that changes existing
/// pass behavior and belongs in its own change.
pub fn epistemic_node_polynomial(
    node_id: &str,
    relations: &[EpistemicRelationInput],
    cycle_members: &BTreeSet<String>,
    ontology: &EdgeOntology,
) -> DualPolynomial {
    if cycle_members.contains(node_id) {
        return DualPolynomial::zero();
    }

    let mut supports = DualPolynomial::zero();
    let mut attacks = DualPolynomial::zero();
    let mut supported = false;
    for relation in relations.iter().filter(|relation| {
        // normalized_confidence is what structural_epistemic_pass applies, and
        // it reads a negative or non-finite value as 1.0. Clamping instead
        // dropped relations the pass still counts, and a NaN comparison is
        // false, so both diverged from the stored grounded status.
        relation.to_content_id == node_id && normalized_confidence(relation.confidence) > 0.0
    }) {
        let Some(generator) = lift_relation(relation, ontology) else {
            continue;
        };
        let term = DualPolynomial::generator(generator.clone());
        match generator.polarity {
            Polarity::Positive => {
                supports = supports.plus(&term);
                supported = true;
            }
            Polarity::Negative => attacks = attacks.plus(&term),
        }
    }

    // Same convention fold_dual uses: a node nothing derives stands on its own
    // like an assumption, and a node something derives is worth exactly those
    // derivations. Keeping `one()` in the sum alongside real supports would
    // leave an empty monomial in the positive fragment, and an empty monomial
    // reads as certainty under every semiring, pinning `for_value` at the
    // carrier's one no matter how weak the support.
    let base = if supported {
        supports
    } else {
        DualPolynomial::one()
    };

    // Sum, never product. Multiplying defeat into the base does force the
    // Boolean standing to Out, but it does so by leaving no positive-only
    // monomial at all, so `for_value` collapses to zero for every attacked
    // node and `net_probability` reads zero however weak the attack. The
    // fragments stay separate here and the one-level defeat rule lives in
    // `DualReadout::into_grounded`.
    base.plus(&attacks)
}

/// [`AnnotationSource`] over a relation set, keyed by [`relation_token`].
pub struct RelationAnnotations<'a>(pub &'a [EpistemicRelationInput], pub &'a EdgeOntology);

impl AnnotationSource for RelationAnnotations<'_> {
    /// Folds every relation sharing the token rather than taking the first
    /// match.
    ///
    /// Two relations can carry the same endpoints, kind and canonical row while
    /// differing in confidence or evidence, and they mint one token by
    /// construction. A `.find()` then read whichever the caller happened to
    /// list first, so reversing the input changed the probability, Viterbi and
    /// corroboration readouts for the same relation multiset.
    ///
    /// The fold is the maximum. These are duplicate assertions of one relation,
    /// and the polynomial has already merged them into a single generator, so
    /// one number has to stand for all of them; the strongest stated confidence
    /// is that number. Averaging is the defensible alternative and would let a
    /// weak restatement dilute a strong assertion, so it is not the default
    /// here. Either way the answer no longer depends on input order, which was
    /// the defect.
    fn annotation(&self, token: &EdgeToken) -> f64 {
        self.0
            .iter()
            .filter(|relation| relation_token(relation, self.1) == token.0)
            .map(|relation| normalized_confidence(relation.confidence))
            .fold(None::<f64>, |best, value| {
                Some(best.map_or(value, |best| best.max(value)))
            })
            .unwrap_or(1.0)
    }
}
