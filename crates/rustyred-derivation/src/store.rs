//! Minimal graph read surface for fold/why without depending on core.

/// Edge-neighbor hit returned by [`DerivationStore::neighbors_in`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivationNeighbor {
    pub node_id: String,
    pub edge_id: String,
}

/// Read-only graph access needed by [`crate::fold`](mod@crate::fold) and [`crate::why`](mod@crate::why).
///
/// Kept in this crate so `rustyred-thg-core` can depend on derivation for
/// [`crate::Semiring`] without a Cargo cycle.
pub trait DerivationStore {
    fn node_labels(&self, node_id: &str) -> Option<Vec<String>>;
    fn edge_annotation(&self, edge_id: &str, property: &str) -> Option<f64>;
    fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor>;

    /// Edge type of `edge_id`, for stores that record one.
    ///
    /// The scalar [`crate::fold`](fn@crate::fold) never needs this: it lifts the `f64`
    /// annotation and nothing else. [`crate::fold::fold_dual`] does, because
    /// the edge type is what decides a generator's polarity.
    ///
    /// The default returns `None`, which keeps every existing implementor
    /// compiling. A store that does not record edge types cannot mint dual
    /// generators, and `fold_dual` refuses with
    /// [`crate::fold::DualFoldError::UntypedEdge`] rather than guessing a
    /// polarity.
    fn edge_type(&self, _edge_id: &str) -> Option<String> {
        None
    }

    /// Distinct edge types on the incoming edges of `node_id`, for stores that
    /// can enumerate them.
    ///
    /// [`Self::neighbors_in`] matches an edge type exactly, but
    /// [`crate::ontology::EdgeOntology::resolve`] is case-insensitive. A
    /// traversal that asks only for the spellings the registry lists therefore
    /// misses a stored `UNDERCUTS`, which classifies as `Defeating` when
    /// inspected yet is never fetched, so the attack is dropped and the readout
    /// can report an attacked node as supported. A store that can enumerate its
    /// own edge types closes that gap.
    ///
    /// The default returns `None`, and the traversal falls back to the
    /// registered spellings.
    fn incoming_edge_types(&self, _node_id: &str) -> Option<Vec<String>> {
        None
    }
}

pub const DERIVATION_LABEL: &str = "Derivation";
pub const ASSUMPTION_LABEL: &str = "Assumption";
pub const DERIVATION_ANCESTOR_EDGE: &str = "derivation_ancestor";
pub const DERIVATION_ASSUMES_EDGE: &str = "derivation_assumes";
pub const ANNOTATION_PROPERTY: &str = "annotation";
pub const MAX_DERIVATION_DEPTH: usize = 256;
