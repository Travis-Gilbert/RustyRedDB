use std::time::Duration;

use egg::{EGraph, Extractor, Runner};
use rustyred_thg_core::{
    AccessMethodRegistry, GraphStoreError, GraphStoreResult, LayerRegistry, QueryIr as QueryInput,
    RankingRegistry,
};
use serde::{Deserialize, Serialize};

pub mod cost;
pub mod cypher;
pub mod desugar;
pub mod lang;
pub mod proof;
pub mod rules;
pub mod structured;

pub use cost::{ensure_snapshot_current, CostSnapshot, QueryCostFunction};
pub use desugar::{desugar_exists, ExistsDesugar, EXISTS_TO_SEMI, NOT_EXISTS_TO_ANTI};
pub use lang::{
    expr_to_query_input, expr_to_query_ir, query_input_to_expr, query_ir_to_expr, rank_term,
    EncodedAttr, QueryIr, QueryLang,
};
pub use proof::{
    replay_query_proof, ExtractionStrategy, ProofReplayReport, ProofStep, QueryPlanProof,
};
pub use rules::rewrite_rules_v1;

use cost::SaturationAnalysis;
use proof::proof_steps_from_explanation;

pub struct RegistryHandles<'a> {
    pub access_methods: &'a AccessMethodRegistry,
    pub ranking_methods: &'a RankingRegistry,
    pub layers: &'a LayerRegistry,
    pub current_graph_version: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SaturationLimits {
    pub node_limit: usize,
    pub iter_limit: usize,
    pub time_limit_ms: u64,
}

impl Default for SaturationLimits {
    fn default() -> Self {
        Self {
            node_limit: 100_000,
            iter_limit: 20,
            time_limit_ms: 1_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SaturateConfig {
    pub enabled: bool,
    #[serde(default)]
    pub saturation: SaturationLimits,
    #[serde(default)]
    pub extraction: ExtractionStrategy,
}

impl Default for SaturateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            saturation: SaturationLimits::default(),
            extraction: ExtractionStrategy::DagAware,
        }
    }
}

impl SaturateConfig {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SaturatedPlan {
    pub query: QueryInput,
    pub proof: QueryPlanProof,
    pub extracted_cost: f64,
    pub greedy_cost: f64,
    pub extracted_term: String,
}

pub fn saturate_query(
    query: &QueryInput,
    registries: &RegistryHandles<'_>,
    config: &SaturateConfig,
) -> GraphStoreResult<SaturatedPlan> {
    rustyred_thg_core::validate_query_ir(query)?;
    let snapshot = CostSnapshot::capture(
        query,
        registries.access_methods,
        registries.ranking_methods,
        registries.layers,
    )?;
    ensure_snapshot_current(&snapshot, registries.access_methods)?;
    if snapshot.indexed_graph_version != registries.current_graph_version {
        return Err(GraphStoreError::new(
            "stale_cost_snapshot",
            format!(
                "snapshot watermark {} does not match requested graph version {}",
                snapshot.indexed_graph_version, registries.current_graph_version
            ),
        ));
    }
    let input = query_input_to_expr(
        query,
        registries.current_graph_version,
        snapshot.snapshot_ref.clone(),
    )?;
    let greedy_cost = expression_cost(&input, &snapshot);
    let rules = if config.enabled {
        rewrite_rules_v1()
    } else {
        Vec::new()
    };
    let mut runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_explanations_enabled()
        .with_expr(&input)
        .with_iter_limit(config.saturation.iter_limit)
        .with_node_limit(config.saturation.node_limit);
    runner = if config.saturation.time_limit_ms == 0 {
        runner.with_time_limit(Duration::from_secs(24 * 60 * 60))
    } else {
        runner.with_time_limit(Duration::from_millis(config.saturation.time_limit_ms))
    };
    let mut runner = runner.run(&rules);
    let root = runner.roots[0];
    let extractor = Extractor::new(
        &runner.egraph,
        QueryCostFunction {
            snapshot: snapshot.clone(),
        },
    );
    let (extracted_cost, extracted) = extractor.find_best(root);
    let extracted_term = extracted.to_string();
    let rewritten = expr_to_query_input(&extracted)?;
    ensure_snapshot_current(&snapshot, registries.access_methods)?;

    let steps = proof_steps_from_explanation(&mut runner, &input, &extracted);
    let stop_reason = runner
        .stop_reason
        .as_ref()
        .map(|reason| format!("{reason:?}"))
        .unwrap_or_else(|| "none".to_string());
    let mut proof = QueryPlanProof {
        library: "query-rewrite-library-v1".to_string(),
        input_term: input.to_string(),
        output_term: extracted_term.clone(),
        extraction_strategy: config.extraction,
        runner_stop_reason: stop_reason,
        steps,
        verified: false,
        analysis: runner.egraph[root].data.clone(),
    };
    let replay = replay_query_proof(&proof);
    if !replay.verified {
        return Err(GraphStoreError::new(
            "query_proof_replay_failed",
            replay.message,
        ));
    }
    proof.verified = true;
    Ok(SaturatedPlan {
        query: rewritten,
        proof,
        extracted_cost,
        greedy_cost,
        extracted_term,
    })
}

fn expression_cost(expr: &egg::RecExpr<QueryLang>, snapshot: &CostSnapshot) -> f64 {
    let mut egraph: EGraph<QueryLang, SaturationAnalysis> = EGraph::new(SaturationAnalysis);
    let root = egraph.add_expr(expr);
    egraph.rebuild();
    Extractor::new(
        &egraph,
        QueryCostFunction {
            snapshot: snapshot.clone(),
        },
    )
    .find_best(root)
    .0
}
