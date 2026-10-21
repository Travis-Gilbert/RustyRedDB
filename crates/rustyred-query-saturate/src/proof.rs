use std::collections::BTreeSet;
use std::str::FromStr;

use egg::{FlatTerm, RecExpr, Runner};
use rustyred_thg_core::{GraphStoreError, GraphStoreResult, Predicate, PredicateMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cost::{QueryAnalysisData, SaturationAnalysis};
use crate::lang::{expr_to_query_ir, PredicateAttrs, QueryLang};
use crate::rules::{canonical_algebra_predicate, known_rule_ids, rewrite_rules_v1};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionStrategy {
    #[default]
    DagAware,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProofStep {
    pub rule_id: String,
    pub before_term: String,
    pub after_term: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QueryPlanProof {
    pub library: String,
    pub input_term: String,
    pub output_term: String,
    pub extraction_strategy: ExtractionStrategy,
    pub runner_stop_reason: String,
    pub steps: Vec<ProofStep>,
    pub verified: bool,
    pub analysis: QueryAnalysisData,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProofReplayReport {
    pub verified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_step: Option<usize>,
    pub message: String,
    pub terminal_term: String,
}

pub(crate) fn proof_steps_from_explanation(
    runner: &mut Runner<QueryLang, SaturationAnalysis, ()>,
    input: &RecExpr<QueryLang>,
    output: &RecExpr<QueryLang>,
) -> Vec<ProofStep> {
    let mut explanation = runner.explain_equivalence(input, output);
    let flat = explanation.make_flat_explanation().clone();
    let Some(first) = flat.first() else {
        return Vec::new();
    };
    let mut current = first.remove_rewrites().get_string();
    let mut steps = Vec::new();
    for term in flat.iter().skip(1) {
        let Some(rule_id) = rewrite_name(term) else {
            continue;
        };
        let after = term.remove_rewrites().get_string();
        steps.push(ProofStep {
            rule_id,
            before_term: current,
            after_term: after.clone(),
        });
        current = after;
    }
    steps
}

fn rewrite_name(term: &FlatTerm<QueryLang>) -> Option<String> {
    term.forward_rule
        .or(term.backward_rule)
        .map(|rule| rule.to_string())
        .or_else(|| term.children.iter().find_map(rewrite_name))
}

pub fn replay_query_proof(proof: &QueryPlanProof) -> ProofReplayReport {
    let known = known_rule_ids().iter().copied().collect::<BTreeSet<_>>();
    let mut current = proof.input_term.clone();
    if let Err(error) = validate_term(&current) {
        return failure(None, error.message, current);
    }
    for (index, step) in proof.steps.iter().enumerate() {
        if !known.contains(step.rule_id.as_str()) {
            return failure(
                Some(index),
                format!("unknown rewrite rule {}", step.rule_id),
                current,
            );
        }
        if step.before_term != current {
            return failure(
                Some(index),
                "proof step does not start from the previous terminal term".to_string(),
                current,
            );
        }
        if let Err(error) = validate_term(&step.after_term) {
            return failure(Some(index), error.message, current);
        }
        let before = match semantic_fingerprint(&step.before_term) {
            Ok(value) => value,
            Err(error) => return failure(Some(index), error.message, current),
        };
        let after = match semantic_fingerprint(&step.after_term) {
            Ok(value) => value,
            Err(error) => return failure(Some(index), error.message, current),
        };
        if before != after {
            return failure(
                Some(index),
                format!("rewrite {} changed the semantic fingerprint", step.rule_id),
                current,
            );
        }
        match rule_library_reaches(&step.before_term, &step.after_term, &step.rule_id) {
            Ok(true) => {}
            Ok(false) => {
                return failure(
                    Some(index),
                    format!(
                        "rewrite {} does not connect the recorded proof terms",
                        step.rule_id
                    ),
                    current,
                )
            }
            Err(error) => return failure(Some(index), error.message, current),
        }
        current = step.after_term.clone();
    }
    if current != proof.output_term {
        return failure(
            None,
            "proof terminal term does not match output term".to_string(),
            current,
        );
    }
    let input = match semantic_fingerprint(&proof.input_term) {
        Ok(value) => value,
        Err(error) => return failure(None, error.message, current),
    };
    let output = match semantic_fingerprint(&proof.output_term) {
        Ok(value) => value,
        Err(error) => return failure(None, error.message, current),
    };
    if input != output {
        return failure(
            None,
            "proof output changed the input semantic fingerprint".to_string(),
            current,
        );
    }
    ProofReplayReport {
        verified: true,
        failed_step: None,
        message: "proof replayed".to_string(),
        terminal_term: current,
    }
}

fn validate_term(term: &str) -> GraphStoreResult<()> {
    let expr = RecExpr::<QueryLang>::from_str(term).map_err(|source| {
        GraphStoreError::new(
            "invalid_query_proof_term",
            format!("failed to parse proof term: {source}"),
        )
    })?;
    for node in expr.as_ref() {
        let QueryLang::Attr(attr) = node else {
            continue;
        };
        let Ok(predicate) = attr.decode::<PredicateAttrs>() else {
            continue;
        };
        if let Predicate::Expand {
            layers,
            mode: PredicateMode::Rank,
            ..
        } = predicate.predicate
        {
            if layers.len() != 1 {
                return Err(GraphStoreError::new(
                    "multi_layer_rank_undefined",
                    format!(
                        "rank mode requires exactly one layer; got {} layers",
                        layers.len()
                    ),
                ));
            }
        }
    }
    expr_to_query_ir(&expr)?;
    Ok(())
}

pub fn semantic_fingerprint(term: &str) -> GraphStoreResult<String> {
    let expr = RecExpr::<QueryLang>::from_str(term).map_err(|source| {
        GraphStoreError::new(
            "invalid_query_proof_term",
            format!("failed to parse proof term: {source}"),
        )
    })?;
    let mut ir = expr_to_query_ir(&expr)?;
    for relation in &mut ir.body.relations {
        for predicate in &mut relation.predicates {
            normalize_predicate(predicate);
        }
        relation.predicates.sort_by_key(stable_predicate);
    }
    ir.body.relations.sort_by(|left, right| {
        left.alias
            .cmp(&right.alias)
            .then_with(|| left.relation.cmp(&right.relation))
    });
    let payload = serde_json::to_vec(&(ir.graph_version, ir.body)).map_err(|source| {
        GraphStoreError::new(
            "query_semantic_fingerprint",
            format!("failed to encode semantic fingerprint: {source}"),
        )
    })?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

fn normalize_predicate(predicate: &mut Predicate) {
    let canonical = canonical_algebra_predicate(predicate);
    if canonical != *predicate {
        *predicate = canonical;
        return;
    }
    match predicate {
        Predicate::Range { column, lo, hi } if lo == hi => {
            if let rustyred_thg_core::ScalarBound::Included(value) = lo {
                if value.as_f64().is_some() {
                    *predicate = Predicate::Equals {
                        column: column.clone(),
                        value: value.clone(),
                    };
                }
            }
        }
        Predicate::Expand { layers, .. } => {
            layers.sort();
            layers.dedup();
        }
        _ => {}
    }
}

fn stable_predicate(predicate: &Predicate) -> String {
    serde_json::to_string(predicate).expect("predicate serialization is infallible")
}

fn failure(
    failed_step: Option<usize>,
    message: String,
    terminal_term: String,
) -> ProofReplayReport {
    ProofReplayReport {
        verified: false,
        failed_step,
        message,
        terminal_term,
    }
}

pub fn rule_library_reaches(before: &str, after: &str, rule_id: &str) -> GraphStoreResult<bool> {
    let before = RecExpr::<QueryLang>::from_str(before)
        .map_err(|source| GraphStoreError::new("invalid_query_proof_term", source.to_string()))?;
    let after = RecExpr::<QueryLang>::from_str(after)
        .map_err(|source| GraphStoreError::new("invalid_query_proof_term", source.to_string()))?;
    let rules = rewrite_rules_v1()
        .into_iter()
        .filter(|rule| rule.name.to_string() == rule_id)
        .collect::<Vec<_>>();
    if rules.is_empty() {
        return Ok(false);
    }
    let mut runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_expr(&before)
        .with_iter_limit(3)
        .run(&rules);
    let before_id = runner.egraph.add_expr(&before);
    let after_id = runner.egraph.add_expr(&after);
    runner.egraph.rebuild();
    Ok(runner.egraph.find(before_id) == runner.egraph.find(after_id))
}
