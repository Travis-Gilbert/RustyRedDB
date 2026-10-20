//! Derivation substrate for the Theorem Harness Experience Compiler.
//!
//! Provides:
//! - runtime algebra descriptors plus the compatibility [`Semiring`] ABI
//! - law-certified Boolean, Counting, Tropical, Viterbi, and Corroboration semirings
//! - legacy independent noisy-OR probability aggregation without a false semiring claim
//! - [`ontology`] — the edge-type signature every valuation interprets
//! - [`dual`] — free dual-indeterminate provenance polynomials `N[X, Xbar]`
//! - [`interpret`](mod@crate::interpret) — the six semirings as homomorphic readouts of one polynomial
//! - [`fold`](mod@crate::fold) — accumulate derivation scores over a derivation graph
//! - [`why`](mod@crate::why) — trace derivation ancestors from a target node

pub mod algebra;
pub mod atms;
pub mod dual;
pub mod fold;
pub mod interpret;
pub mod ontology;
pub mod semiring;
pub mod store;
pub mod why;

pub use algebra::{
    registered_algebras, resolve_algebra_token, AlgebraClass, AlgebraDescriptor,
    AlgebraRequirementError, DeterminismPolicy, EqualityPolicy, LawProfile, MathematicalSemiring,
    OperationLaws,
};
pub use atms::{
    environment_is_valid, minimize_nogoods, AssumptionEnvironment, AssumptionId, AtmsLabelRecord,
    AtmsRecalculation, BooleanProvenance, IncrementalAtms,
};
pub use dual::{
    DualPolynomial, EdgeToken, Generator, Monomial, Polarity, DUAL_PROVENANCE_DESCRIPTOR,
    MAX_DUAL_MONOMIALS,
};
pub use fold::{
    derivation_dag_ontology, fold, fold_code_edges, fold_code_incoming, fold_dual, DualFoldError,
};
pub use interpret::{
    interpret, net_probability, probability_readout, AnnotationSource, CostAnnotations,
    DualReadout, MapAnnotations, ProbabilityReadoutError, Standing,
};
pub use ontology::{
    code_edge_specs, code_graph_ontology, default_registry, default_specs, AlgebraicClass,
    EdgeDirection, EdgeOntology, EdgeTypeSpec, OntologyError, CALLS_SYMBOL_EDGE, CHANGED_FILE_EDGE,
    CONTAINS_EDGE, DEPENDS_ON_SYMBOL_EDGE, MENTIONS_LITERAL_EDGE, READS_FIELD_EDGE,
    RENDEZVOUS_EDGE, SCIP_DEFINES_EDGE, SCIP_JOINS_CODE_SYMBOL_EDGE, SCIP_REFERENCES_EDGE,
    WRITES_FIELD_EDGE,
};
pub use semiring::{
    BooleanSemiring, CorroborationSemiring, CountingSemiring, IndependentNoisyOrAlgebra,
    ProbabilitySemiring, Semiring, TropicalSemiring, ViterbiSemiring,
};
pub use store::{DerivationNeighbor, DerivationStore};
pub use why::{why, why_code_fact, CodeWhyEdge, CodeWhyTrace, WhyNode, WhyTrace};
