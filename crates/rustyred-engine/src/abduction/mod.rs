//! Abduction dual-space surface (assumption retraction over incremental derivation).

pub mod retract;

pub use retract::{
    get_hypothesis_space, put_explanation, put_hypothesis_space, renormalize_over_survivors,
    retract_assumption, Assumable, Explanation, ExplanationStatus, HypothesisSpace,
    RetractionReport, EXPLANATION_LABEL, HYPOTHESIS_SPACE_LABEL, HYPOTHESIS_SPACE_NODE_ID,
};
