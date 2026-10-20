//! Consolidated integration-test binary for `rustyred-thg-core`.
//!
//! One binary instead of one per file: every file in `tests/` used to link the
//! crate's full dependency closure separately. Module declarations only live
//! here; test bodies are unchanged in their per-file modules. See
//! `CONVENTIONS.md` (build-performance BP10) for the target-selection ladder.

mod accounting_scaling;
mod adaptive_index_phase6_tests;
mod aof_checksum_compat;
mod assert_claim_acceptance;
mod commit_scaling;
mod composite_partial_covering_tests;
mod context_training_view_tests;
mod derivation_acceptance;
mod document_tier_acceptance;
mod epistemic_acceptance;
mod epistemic_columnar_acceptance;
mod fulltext_phase3_tests;
mod graph_decomposition_acceptance;
mod hook_dispatch;
mod lfr_chunk_store_injection;
mod identity_index_tests;
mod improvement_plan_acceptance;
mod incremental_derivation_acceptance;
mod index_manifest_tests;
mod map_artifact_phase8_tests;
mod multiplex_layers;
mod ordered_index_acceptance;
mod planner_multimodal_acceptance;
mod ppr_csr_acceptance;
mod prolly_incremental_commit;
mod prolly_parity_boundary;
mod query_receipt_tests;
mod redcore_snapshot_replay_parity;
mod relational_copy_gate;
mod relational_core_acceptance;
mod relational_lens_acceptance;
mod saturation_acceptance;
mod scoped_ordered_index_tests;
mod semiring_annotated_joins_acceptance;
mod statement_acceptance;
mod tensor_derivation_acceptance;
mod ua15_connection_native;
mod vector_index_update_acceptance;
mod vector_phase4_tests;
