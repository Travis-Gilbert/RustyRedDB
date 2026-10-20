//! Bridge [`GraphStore`] to [`rustyred_thg_derivation::DerivationStore`].

use std::collections::BTreeSet;

use rustyred_thg_derivation::dual::DualPolynomial;
use rustyred_thg_derivation::fold::{fold_dual, DualFoldError};
use rustyred_thg_derivation::ontology::EdgeOntology;
use rustyred_thg_derivation::store::ANNOTATION_PROPERTY;
use rustyred_thg_derivation::{fold, why, DerivationNeighbor, DerivationStore, Semiring, WhyTrace};

use crate::graph::capability::GraphStore;
use crate::graph::ontology_adapter::edge_annotation_weight;
use crate::graph::query::NeighborQuery;

/// Local newtype so the orphan rule allows implementing foreign [`DerivationStore`].
pub struct GraphDerivationStore<'a, G: GraphStore>(&'a G);

impl<'a, G: GraphStore> GraphDerivationStore<'a, G> {
    pub fn new(store: &'a G) -> Self {
        Self(store)
    }
}

impl<G: GraphStore> DerivationStore for GraphDerivationStore<'_, G> {
    fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
        self.0.get_node(node_id).map(|node| node.labels.clone())
    }

    /// Reads the same weight [`StoreAnnotations`] reads, through the shared
    /// [`edge_annotation_weight`]. Returning only the raw property left an edge
    /// carrying its weight in `confidence` or `source_reliability` valued 1.0
    /// here and at its real confidence in the dual readout, so `fold_graph` and
    /// `fold_dual_graph` disagreed on the same graph.
    ///
    /// Only the annotation property falls back. Any other property is still
    /// answered from the edge's own properties or not at all.
    fn edge_annotation(&self, edge_id: &str, property: &str) -> Option<f64> {
        let edge = self.0.get_edge(edge_id)?;
        if property != ANNOTATION_PROPERTY {
            return edge.properties.get(property)?.as_f64();
        }
        Some(edge_annotation_weight(edge))
    }

    fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
        self.0
            .neighbors(NeighborQuery::in_(node_id).with_edge_type(edge_type))
            .into_iter()
            .map(|hit| DerivationNeighbor {
                node_id: hit.node_id,
                edge_id: hit.edge_id,
            })
            .collect()
    }

    /// Required by [`fold_dual`], which decides a generator's polarity from the
    /// edge type. Without it the trait default returns `None` and every
    /// nontrivial fold through this adapter refuses with `UntypedEdge`.
    fn edge_type(&self, edge_id: &str) -> Option<String> {
        self.0.get_edge(edge_id).map(|edge| edge.edge_type.clone())
    }

    /// A `GraphStore` can list incoming edges without filtering by type, so it
    /// can report the spellings actually stored. That lets `fold_dual` reach a
    /// defeating edge whose case differs from the registry's spellings, which a
    /// traversal built only from those spellings would never fetch.
    fn incoming_edge_types(&self, node_id: &str) -> Option<Vec<String>> {
        let mut types = Vec::new();
        let mut seen = BTreeSet::new();
        for hit in self.0.neighbors(NeighborQuery::in_(node_id)) {
            if let Some(edge) = self.0.get_edge(&hit.edge_id) {
                if seen.insert(edge.edge_type.clone()) {
                    types.push(edge.edge_type.clone());
                }
            }
        }
        Some(types)
    }
}

/// Fold over a [`GraphStore`] using the derivation-crate semiring.
pub fn fold_graph<S: Semiring>(store: &impl GraphStore, node_id: &str) -> S::Element {
    fold::<S>(&GraphDerivationStore::new(store), node_id)
}

/// Fold a derivation DAG in a [`GraphStore`] into a dual provenance polynomial.
///
/// The companion to [`fold_graph`] for the free layer. `ontology` classifies
/// each edge type; pass
/// [`rustyred_thg_derivation::fold::derivation_dag_ontology`] for the ordinary
/// derivation-DAG edge types.
pub fn fold_dual_graph(
    store: &impl GraphStore,
    node_id: &str,
    ontology: &EdgeOntology,
) -> Result<DualPolynomial, DualFoldError> {
    fold_dual(&GraphDerivationStore::new(store), node_id, ontology)
}

/// Why-trace over a [`GraphStore`].
pub fn why_graph(store: &impl GraphStore, node_id: &str) -> Option<WhyTrace> {
    why(&GraphDerivationStore::new(store), node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::ontology_adapter::StoreAnnotations;
    use crate::graph::record::{EdgeRecord, NodeRecord};
    use rustyred_thg_derivation::dual::{EdgeToken, Polarity};
    use rustyred_thg_derivation::fold::derivation_dag_ontology;
    use rustyred_thg_derivation::interpret::AnnotationSource;
    use rustyred_thg_derivation::interpret::{interpret, MapAnnotations, Standing};
    use rustyred_thg_derivation::store::{
        ANNOTATION_PROPERTY, ASSUMPTION_LABEL, DERIVATION_ASSUMES_EDGE, DERIVATION_LABEL,
    };
    use rustyred_thg_derivation::{
        BooleanSemiring, CorroborationSemiring, CountingSemiring, ProbabilitySemiring,
        ViterbiSemiring,
    };
    use serde_json::json;

    /// A --e_ab--> B --e_bc--> C, the shape `fold.rs` uses as its chain fixture.
    fn chain() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("A", [ASSUMPTION_LABEL], json!({})))
            .unwrap();
        for id in ["B", "C"] {
            store
                .upsert_node(NodeRecord::new(id, [DERIVATION_LABEL], json!({})))
                .unwrap();
        }
        for (id, from, to, annotation) in [("e_ab", "A", "B", 0.9_f64), ("e_bc", "B", "C", 0.8)] {
            store
                .upsert_edge(EdgeRecord::new(
                    id,
                    from,
                    DERIVATION_ASSUMES_EDGE,
                    to,
                    json!({ ANNOTATION_PROPERTY: annotation }),
                ))
                .unwrap();
        }
        store
    }

    #[test]
    fn a_defeating_edge_stored_in_an_unregistered_case_still_attacks() {
        // `EdgeOntology::resolve` is case-insensitive, but `neighbors_in`
        // matches exactly, so a traversal built only from the spellings the
        // registry lists never fetched `UNDERCUTS`. The edge classified as
        // Defeating on inspection and vanished from the fold, leaving an
        // attacked node reading as supported.
        let mut store = chain();
        store
            .upsert_node(NodeRecord::new("D", [ASSUMPTION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "e_attack",
                "D",
                "UNDERCUTS",
                "C",
                json!({ ANNOTATION_PROPERTY: 0.7_f64 }),
            ))
            .unwrap();

        let poly = fold_dual_graph(&store, "C", derivation_dag_ontology())
            .expect("an unregistered case is still a classified edge type");

        let negatives = poly
            .monomials()
            .iter()
            .flat_map(|monomial| monomial.generators.iter())
            .filter(|(generator, _)| generator.polarity == Polarity::Negative)
            .count();
        assert_eq!(negatives, 1, "the attack must reach the fold");

        let annotations = StoreAnnotations(&store);
        let readout = interpret::<BooleanSemiring>(&poly, &annotations);
        assert_ne!(
            readout.standing,
            Standing::In,
            "an attacked node must not read as supported"
        );
    }

    #[test]
    fn a_confidence_valued_edge_reads_the_same_on_both_paths() {
        // The weight may live in `EdgeRecord.confidence` instead of the
        // `annotation` property. `StoreAnnotations` always read that fallback;
        // `GraphDerivationStore::edge_annotation` did not, so the scalar fold
        // defaulted the same edge to 1.0 and the two paths valued one graph
        // differently under probability, viterbi, tropical, and corroboration.
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("A", [ASSUMPTION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("B", [DERIVATION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_edge(
                EdgeRecord::new("e_ab", "A", DERIVATION_ASSUMES_EDGE, "B", json!({}))
                    .with_confidence(0.5),
            )
            .unwrap();

        let bridged = GraphDerivationStore::new(&store)
            .edge_annotation("e_ab", ANNOTATION_PROPERTY)
            .expect("the edge carries a weight");
        let direct = StoreAnnotations(&store).annotation(&EdgeToken::new("e_ab"));
        assert_eq!(
            bridged, direct,
            "the fold and the dual readout must read one edge the same way"
        );
        assert_eq!(bridged, 0.5, "and both must read the stored confidence");

        let scalar = fold_graph::<ProbabilitySemiring>(&store, "B");
        let dual = interpret::<ProbabilitySemiring>(
            &fold_dual_graph(&store, "B", derivation_dag_ontology()).expect("dual fold"),
            &StoreAnnotations(&store),
        )
        .for_value;
        assert!(
            (scalar - dual).abs() < 1e-12,
            "scalar {scalar} and dual {dual} must agree on a confidence-valued edge"
        );
    }

    #[test]
    fn dual_fold_works_through_the_canonical_adapter() {
        // Regression: the adapter used to leave `edge_type` at the trait
        // default, so every nontrivial fold through it refused with
        // UntypedEdge and the dual layer was unusable on a real graph.
        let store = chain();
        let poly = fold_dual_graph(&store, "C", derivation_dag_ontology())
            .expect("the canonical adapter must supply edge types");

        assert_eq!(poly.monomials().len(), 1);
        assert_eq!(poly.monomials()[0].generators.len(), 2);
        for (generator, _) in &poly.monomials()[0].generators {
            assert_eq!(generator.polarity, Polarity::Positive);
        }

        let annotations = MapAnnotations::new(1.0).with("e_ab", 0.9).with("e_bc", 0.8);
        let readout = interpret::<BooleanSemiring>(&poly, &annotations);
        assert!(readout.for_value);
        assert_eq!(readout.standing, Standing::In);
    }

    #[test]
    fn a_defeating_edge_in_a_real_store_reaches_the_dual_fold() {
        // The decisive case for the free layer: an edge stored in an actual
        // GraphStore with edge_type "Undercuts". A GraphStore edge has exactly
        // one type, so a traversal asking only for the derivation relations
        // could never return this edge, and the production fold was
        // positive-only no matter what the ontology declared.
        let mut store = chain();
        store
            .upsert_edge(EdgeRecord::new(
                "e_attack",
                "A",
                "Undercuts",
                "C",
                json!({ ANNOTATION_PROPERTY: 1.0 }),
            ))
            .unwrap();

        let poly = fold_dual_graph(&store, "C", derivation_dag_ontology()).expect("dual fold");
        let negatives: Vec<_> = poly
            .generators()
            .into_iter()
            .filter(|generator| generator.polarity == Polarity::Negative)
            .cloned()
            .collect();
        assert_eq!(
            negatives.len(),
            1,
            "the stored Undercuts edge must be reached"
        );
        assert_eq!(negatives[0].canonical_type, "Undercuts");
        assert_eq!(negatives[0].token.as_str(), "e_attack");

        let annotations = MapAnnotations::new(1.0)
            .with("e_ab", 0.9)
            .with("e_bc", 0.8)
            .with("e_attack", 1.0);
        let readout = interpret::<BooleanSemiring>(&poly, &annotations);
        assert!(readout.against_value, "defeat must reach the readout");
    }

    #[test]
    fn dual_readout_through_the_adapter_matches_the_scalar_fold() {
        let store = chain();
        let annotations = MapAnnotations::new(1.0).with("e_ab", 0.9).with("e_bc", 0.8);
        let poly = fold_dual_graph(&store, "C", derivation_dag_ontology()).expect("dual fold");
        assert_eq!(
            interpret::<CountingSemiring>(&poly, &annotations).for_value,
            fold_graph::<CountingSemiring>(&store, "C")
        );
        assert_eq!(
            interpret::<BooleanSemiring>(&poly, &annotations).for_value,
            fold_graph::<BooleanSemiring>(&store, "C")
        );
    }

    #[test]
    fn store_annotations_reproduce_the_scalar_fold_on_a_weighted_dag() {
        // The production pairing: fold_dual_graph plus StoreAnnotations, with
        // no hand-supplied weights. Derivation edges carry their weight in the
        // `annotation` property and leave `confidence` unset, so an adapter
        // reading only `confidence` scored every edge 1.0 and silently
        // disagreed with the scalar fold on every weighted semiring.
        let store = chain();
        let annotations = StoreAnnotations(&store);
        let poly = fold_dual_graph(&store, "C", derivation_dag_ontology()).expect("dual fold");

        assert!(
            (annotations.annotation(&EdgeToken::new("e_ab")) - 0.9).abs() < 1e-9,
            "the annotation property must win over the unset confidence"
        );

        for (label, got, want) in [
            (
                "viterbi",
                interpret::<ViterbiSemiring>(&poly, &annotations).for_value,
                fold_graph::<ViterbiSemiring>(&store, "C"),
            ),
            (
                "probability",
                interpret::<ProbabilitySemiring>(&poly, &annotations).for_value,
                fold_graph::<ProbabilitySemiring>(&store, "C"),
            ),
            (
                "corroboration",
                interpret::<CorroborationSemiring>(&poly, &annotations).for_value,
                fold_graph::<CorroborationSemiring>(&store, "C"),
            ),
        ] {
            assert!((got - want).abs() < 1e-9, "{label}: {got} != {want}");
        }
    }
}
