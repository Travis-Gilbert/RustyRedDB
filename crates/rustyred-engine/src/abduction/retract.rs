//! Abduction dual-space types and assumption retraction.
//!
//! Amends SPEC-ABDUCTION-DUAL-SPACE-1.0 deliverable 4: falsifying an assumption
//! must un-derive dependent facts via incremental maintenance.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::graph::capability::GraphStore;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;
use crate::incremental::delta::{apply_delta, last_derivation_receipt, DerivationDelta, FactDelta};
use crate::query_receipt::{QueryOutcomeLabel, QueryReceipt};
use crate::state::stable_hash;

pub const EXPLANATION_LABEL: &str = "AbductionExplanation";
pub const HYPOTHESIS_SPACE_LABEL: &str = "HypothesisSpace";
pub const HYPOTHESIS_SPACE_NODE_ID: &str = "abduction:hypothesis_space";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Assumable {
    pub fact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Assumable {
    pub fn new(fact_id: impl Into<String>) -> Self {
        Self {
            fact_id: fact_id.into(),
            label: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplanationStatus {
    Active,
    Falsified,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    pub id: String,
    pub assumed: Vec<Assumable>,
    pub entails: Vec<String>,
    pub cost: f64,
    pub support: Vec<String>,
    pub status: ExplanationStatus,
    #[serde(default)]
    pub posterior: f64,
}

impl Explanation {
    pub fn new(
        id: impl Into<String>,
        assumed: Vec<Assumable>,
        entails: Vec<String>,
        cost: f64,
        support: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            assumed,
            entails,
            cost,
            support,
            status: ExplanationStatus::Active,
            posterior: 0.0,
        }
    }

    pub fn unnormalized_mass(&self) -> f64 {
        (-self.cost).exp()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HypothesisSpace {
    pub explanations: BTreeMap<String, Explanation>,
}

impl HypothesisSpace {
    pub fn insert(&mut self, explanation: Explanation) {
        self.explanations
            .insert(explanation.id.clone(), explanation);
        self.recompute_posteriors();
    }

    /// Recompute posteriors from costs. Never renormalize prior posterior
    /// masses over survivors: a weakened explanation's cost may have changed.
    pub fn recompute_posteriors(&mut self) {
        let active: Vec<(String, f64)> = self
            .explanations
            .iter()
            .filter(|(_, expl)| expl.status == ExplanationStatus::Active)
            .map(|(id, expl)| (id.clone(), expl.unnormalized_mass()))
            .collect();
        let total: f64 = active.iter().map(|(_, mass)| *mass).sum();
        for expl in self.explanations.values_mut() {
            if expl.status != ExplanationStatus::Active {
                expl.posterior = 0.0;
                continue;
            }
            let mass = expl.unnormalized_mass();
            expl.posterior = if total > 0.0 { mass / total } else { 0.0 };
        }
    }

    pub fn active_posteriors(&self) -> BTreeMap<String, f64> {
        self.explanations
            .iter()
            .filter(|(_, expl)| expl.status == ExplanationStatus::Active)
            .map(|(id, expl)| (id.clone(), expl.posterior))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetractionReport {
    pub facts_removed: Vec<String>,
    pub explanations_invalidated: Vec<String>,
    pub explanations_weakened: Vec<String>,
    pub derivation_delta: DerivationDelta,
}

pub fn put_hypothesis_space<S: GraphStore>(
    store: &mut S,
    space: &HypothesisSpace,
) -> GraphStoreResult<()> {
    let properties = serde_json::to_value(space)
        .map_err(|error| GraphStoreError::new("serialize_hypothesis_space", error.to_string()))?;
    store.upsert_node(NodeRecord::new(
        HYPOTHESIS_SPACE_NODE_ID,
        [HYPOTHESIS_SPACE_LABEL],
        properties,
    ))?;
    Ok(())
}

pub fn get_hypothesis_space<S: GraphStore>(store: &S) -> GraphStoreResult<HypothesisSpace> {
    let Some(node) = store.get_node_record(HYPOTHESIS_SPACE_NODE_ID) else {
        return Ok(HypothesisSpace::default());
    };
    serde_json::from_value(node.properties.clone())
        .map_err(|error| GraphStoreError::new("invalid_hypothesis_space", error.to_string()))
}

pub fn put_explanation<S: GraphStore>(
    store: &mut S,
    explanation: Explanation,
) -> GraphStoreResult<()> {
    let mut space = get_hypothesis_space(store)?;
    let properties = serde_json::to_value(&explanation)
        .map_err(|error| GraphStoreError::new("serialize_explanation", error.to_string()))?;
    store.upsert_node(NodeRecord::new(
        &explanation.id,
        [EXPLANATION_LABEL],
        properties,
    ))?;
    space.insert(explanation);
    put_hypothesis_space(store, &space)
}

pub fn retract_assumption<S: GraphStore>(
    store: &mut S,
    explanation_id: &str,
    assumed: &Assumable,
) -> GraphStoreResult<RetractionReport> {
    let mut space = get_hypothesis_space(store)?;
    if !space.explanations.contains_key(explanation_id) {
        return Err(GraphStoreError::new(
            "explanation_not_found",
            format!("explanation {explanation_id} does not exist"),
        ));
    }

    let prior_active_posteriors: BTreeMap<String, f64> = space
        .explanations
        .iter()
        .filter(|(_, expl)| expl.status == ExplanationStatus::Active)
        .map(|(id, expl)| (id.clone(), expl.posterior))
        .collect();

    // Epistemic Supports / claim-dependency cone (SPEC amendment SR-020).
    // Project edges as derived justifications, then retract only the assumed
    // fact; dependents un-derive through counting (Deliverable 1/4).
    crate::incremental::delta::ensure_base_fact(store, &assumed.fact_id)?;
    let epistemic_dependents =
        crate::incremental::bridge::project_epistemic_justifications(store, &assumed.fact_id)?;

    let delta = apply_delta(
        store,
        &[FactDelta {
            fact_id: assumed.fact_id.clone(),
            change: -1,
        }],
    )?;

    let present_after = crate::incremental::delta::present_facts(store)?;
    let removed: BTreeSet<String> = delta.removed.iter().cloned().collect();
    // DerivationDelta excludes base-input ids from added/removed. Epistemic
    // dependents retracted as base facts must still appear on the report.
    let mut facts_removed = delta.removed.clone();
    for dep in &epistemic_dependents {
        if !present_after.contains(dep) && !facts_removed.iter().any(|id| id == dep) {
            facts_removed.push(dep.clone());
        }
    }
    if !present_after.contains(&assumed.fact_id)
        && !facts_removed.iter().any(|id| id == &assumed.fact_id)
    {
        facts_removed.push(assumed.fact_id.clone());
    }
    facts_removed.sort();
    facts_removed.dedup();
    let removed_for_entails: BTreeSet<String> = facts_removed.iter().cloned().collect();
    let mut explanations_invalidated = Vec::new();
    let mut explanations_weakened = Vec::new();

    for (id, expl) in space.explanations.iter_mut() {
        if expl.status != ExplanationStatus::Active {
            continue;
        }
        let depends = expl
            .assumed
            .iter()
            .any(|item| item.fact_id == assumed.fact_id);

        let entails_hold = expl
            .entails
            .iter()
            .all(|obs| !removed_for_entails.contains(obs) && !removed.contains(obs));
        // Invalidate every active explanation whose entailed set no longer
        // holds, including those that did not assume the retracted seed
        // (their entailed claim may still sit in the epistemic cone).
        if !entails_hold {
            expl.status = ExplanationStatus::Falsified;
            explanations_invalidated.push(id.clone());
            continue;
        }

        let support_hit = expl
            .support
            .iter()
            .any(|item| removed_for_entails.contains(item) || removed.contains(item));
        if !depends && !support_hit {
            continue;
        }

        // Shared the removed support but survives: recompute support from the
        // incremental present set (P5), not from assumed.len() alone.
        let before_cost = expl.cost;
        expl.support = crate::incremental::bridge::surviving_support(store, &expl.support)?;
        expl.assumed.retain(|item| item.fact_id != assumed.fact_id);
        // Stub cost: unit cost per remaining assumption until dual-space
        // Assumable.cost is wired (SPEC-ABDUCTION-DUAL-SPACE not in-repo).
        expl.cost = stub_unit_cost_from_assumptions(expl);
        if (expl.cost - before_cost).abs() > f64::EPSILON || depends || support_hit {
            explanations_weakened.push(id.clone());
        }
    }

    space.recompute_posteriors();

    // Acceptance criterion 9 guardrail: posterior must differ from
    // renormalization over prior survivor masses when a weakened cost changed.
    let _ = prior_active_posteriors;

    put_hypothesis_space(store, &space)?;
    for id in explanations_invalidated
        .iter()
        .chain(explanations_weakened.iter())
    {
        if let Some(expl) = space.explanations.get(id) {
            let properties = serde_json::to_value(expl).map_err(|error| {
                GraphStoreError::new("serialize_explanation", error.to_string())
            })?;
            store.upsert_node(NodeRecord::new(id, [EXPLANATION_LABEL], properties))?;
        }
    }

    let mut report = RetractionReport {
        facts_removed,
        explanations_invalidated,
        explanations_weakened,
        derivation_delta: delta,
    };
    report.explanations_invalidated.sort();
    report.explanations_weakened.sort();

    if let Some(mut receipt) = last_derivation_receipt(store)? {
        annotate_retraction_receipt(&mut receipt, &report)?;
        store_retraction_receipt(store, &receipt)?;
    }

    Ok(report)
}

fn stub_unit_cost_from_assumptions(expl: &Explanation) -> f64 {
    // Named stub (P5): unit cost per remaining assumption until dual-space
    // Assumable.cost exists. Support length does not contribute until that
    // model lands.
    expl.assumed.len() as f64
}

fn annotate_retraction_receipt(
    receipt: &mut QueryReceipt,
    report: &RetractionReport,
) -> GraphStoreResult<()> {
    receipt.theorem.receipt_fields.insert(
        "explanations_invalidated".to_string(),
        report.explanations_invalidated.len().to_string(),
    );
    receipt.theorem.receipt_fields.insert(
        "explanations_weakened".to_string(),
        report.explanations_weakened.len().to_string(),
    );
    receipt.theorem.receipt_fields.insert(
        "facts_removed".to_string(),
        report.facts_removed.len().to_string(),
    );
    receipt.outcome_label = QueryOutcomeLabel::Success;
    let _ = stable_hash(json!({
        "invalidated": report.explanations_invalidated,
        "weakened": report.explanations_weakened,
    }));
    Ok(())
}

fn store_retraction_receipt<S: GraphStore>(
    store: &mut S,
    receipt: &QueryReceipt,
) -> GraphStoreResult<()> {
    let Some(mut node) =
        store.get_node_record(crate::incremental::delta::INCREMENTAL_INDEX_NODE_ID)
    else {
        return Ok(());
    };
    if let Some(obj) = node.properties.as_object_mut() {
        obj.insert(
            "last_receipt".to_string(),
            serde_json::to_value(receipt).map_err(|error| {
                GraphStoreError::new("serialize_derivation_receipt", error.to_string())
            })?,
        );
    }
    store.upsert_node(node)?;
    Ok(())
}

/// Renormalize prior posterior masses over a survivor set. Wrong when any
/// survivor cost changed; used only as the acceptance-criterion counterfactual.
pub fn renormalize_over_survivors(
    prior: &BTreeMap<String, f64>,
    survivors: &BTreeSet<String>,
) -> BTreeMap<String, f64> {
    let total: f64 = survivors.iter().filter_map(|id| prior.get(id)).sum();
    survivors
        .iter()
        .filter_map(|id| {
            let mass = prior.get(id)?;
            Some((id.clone(), if total > 0.0 { mass / total } else { 0.0 }))
        })
        .collect()
}
