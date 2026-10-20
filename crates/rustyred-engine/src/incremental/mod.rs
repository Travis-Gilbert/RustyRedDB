//! Incremental derivation maintenance (DBSP delta rule, native saturation path).
//!
//! Does not depend on the `dbsp` crate. Maintains support multiplicity under
//! addition and retraction; recursive predicates use delete-and-rederive (DRed).

pub mod bridge;
pub mod delta;
pub mod rules;
pub mod support;

pub use bridge::{
    epistemic_support_dependents, epistemic_support_edges, project_epistemic_justifications,
    record_derivation_outputs, record_saturation_statement, retract_assumption_facts,
    surviving_support, sync_saturation_materialize,
};
pub use delta::{
    apply_delta, apply_distinct_for_predicates, derivation_delta_consumer_shape_note,
    drop_derived_facts, edge_fact, ensure_base_fact, ensure_observational_or_active_premise,
    install_rules, install_rules_with_semiring, last_derivation_receipt, path_fact, path_program,
    present_facts, record_justified_fact, record_justified_fact_with_relation,
    register_derivation_subscriber, supported_fact, DerivationDelta, DerivationSubscriber,
    FactDelta, PredicateStrategy, INCREMENTAL_INDEX_LABEL, INCREMENTAL_INDEX_NODE_ID,
};
pub use rules::{
    fact_predicate, format_fact, parse_fact, stratify_recursive_predicates, Atom, Rule, RuleProgram,
};
pub use support::{distinct, distinct_delta, Justification, Multiplicity, SupportedFact};
