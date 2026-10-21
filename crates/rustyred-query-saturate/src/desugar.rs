//! Proof-carrying EXISTS / NOT EXISTS desugar (SPEC-THEOREM-RELATIONAL-OPERATORS-1.0).
//!
//! Logical rewrites live in query-saturate's egg rewrite library, not in pg-wire
//! lowering. EXISTS becomes a semi join; NOT EXISTS becomes an anti join.

use egg::{CostFunction, Extractor, Id, Language, RecExpr, Runner};
use rustyred_thg_core::{
    GraphStoreError, GraphStoreResult, JoinKind, JoinPredicate, QueryRelation,
};
use serde::{Deserialize, Serialize};

use crate::cost::SaturationAnalysis;
use crate::lang::{EncodedAttr, QueryLang};
use crate::proof::{proof_steps_from_explanation, ExtractionStrategy, QueryPlanProof};
use crate::rules::rewrite_rules_v1;

pub use crate::rules::{EXISTS_TO_SEMI, NOT_EXISTS_TO_ANTI};

struct ExistsDesugarCost;

impl CostFunction<QueryLang> for ExistsDesugarCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &QueryLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let children: usize = enode.children().iter().copied().map(&mut costs).sum();
        let node = match enode {
            QueryLang::Exists(_) | QueryLang::NotExists(_) => 8,
            _ => 1,
        };
        children.saturating_add(node)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExistsDesugar {
    pub join: JoinPredicate,
    pub subquery_relation: QueryRelation,
    pub proof: QueryPlanProof,
}

#[derive(Clone, Debug, Serialize)]
struct ExistsCorr {
    negated: bool,
    outer_alias: String,
    outer_column: String,
    subquery_alias: String,
    subquery_column: String,
}

pub fn desugar_exists(
    outer_alias: impl Into<String>,
    outer_column: impl Into<String>,
    subquery_alias: impl Into<String>,
    subquery_relation: impl Into<String>,
    subquery_column: impl Into<String>,
    negated: bool,
) -> GraphStoreResult<ExistsDesugar> {
    let outer_alias = outer_alias.into();
    let outer_column = outer_column.into();
    let subquery_alias = subquery_alias.into();
    let subquery_relation_name = subquery_relation.into();
    let subquery_column = subquery_column.into();
    if outer_alias.is_empty()
        || outer_column.is_empty()
        || subquery_alias.is_empty()
        || subquery_relation_name.is_empty()
        || subquery_column.is_empty()
    {
        return Err(GraphStoreError::new(
            "exists_desugar_incomplete",
            "EXISTS desugar requires correlated outer and subquery columns",
        ));
    }
    let rule_id = if negated {
        NOT_EXISTS_TO_ANTI
    } else {
        EXISTS_TO_SEMI
    };
    let kind = if negated {
        JoinKind::Anti
    } else {
        JoinKind::Semi
    };
    let corr = ExistsCorr {
        negated,
        outer_alias: outer_alias.clone(),
        outer_column: outer_column.clone(),
        subquery_alias: subquery_alias.clone(),
        subquery_column: subquery_column.clone(),
    };
    let encoded = EncodedAttr::from_json(&corr)?;
    let mut input = RecExpr::default();
    let attr = input.add(QueryLang::Attr(encoded));
    if negated {
        input.add(QueryLang::NotExists([attr]));
    } else {
        input.add(QueryLang::Exists([attr]));
    }
    let mut runner = Runner::<QueryLang, SaturationAnalysis, ()>::default()
        .with_explanations_enabled()
        .with_iter_limit(8)
        .with_node_limit(64)
        .with_expr(&input)
        .run(&rewrite_rules_v1());
    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, ExistsDesugarCost);
    let (_cost, output) = extractor.find_best(root);
    let steps = proof_steps_from_explanation(&mut runner, &input, &output);
    let verified = steps.iter().any(|step| step.rule_id == rule_id);
    let proof = QueryPlanProof {
        library: "query-rewrite-library-v1".to_string(),
        input_term: input.to_string(),
        output_term: output.to_string(),
        extraction_strategy: ExtractionStrategy::DagAware,
        runner_stop_reason: format!("{:?}", runner.stop_reason),
        steps,
        verified,
        analysis: runner.egraph[root].data.clone(),
    };
    Ok(ExistsDesugar {
        join: JoinPredicate {
            left_alias: outer_alias,
            left_column: outer_column,
            right_alias: subquery_alias.clone(),
            right_column: subquery_column,
            kind,
        },
        subquery_relation: QueryRelation {
            alias: subquery_alias,
            relation: subquery_relation_name,
            predicates: Vec::new(),
        },
        proof,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exists_becomes_semi_and_not_exists_becomes_anti() {
        let exists = desugar_exists("m", "id", "e", "epistemic", "content_id", false).unwrap();
        assert_eq!(exists.join.kind, JoinKind::Semi);
        assert!(exists.proof.verified, "egg must extract exists-to-semi");
        assert!(exists
            .proof
            .steps
            .iter()
            .any(|step| step.rule_id == EXISTS_TO_SEMI));
        let not_exists = desugar_exists("m", "id", "e", "epistemic", "content_id", true).unwrap();
        assert_eq!(not_exists.join.kind, JoinKind::Anti);
        assert!(
            not_exists.proof.verified,
            "egg must extract not-exists-to-anti"
        );
        assert!(not_exists
            .proof
            .steps
            .iter()
            .any(|step| step.rule_id == NOT_EXISTS_TO_ANTI));
    }
}
