//! Intrinsic judgment and sensemaking signals.
//!
//! Interestingness is intentionally separate from goal utility. A thing can be
//! valuable to resurface because it is surprising, structurally distant, or
//! information-dense even before a planner has decided it serves an active goal.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::state::stable_hash;
use crate::GraphStoreResult;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgmentItem {
    pub item_id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub surprise: f64,
    #[serde(default)]
    pub novelty: f64,
    #[serde(default)]
    pub value_density: f64,
    #[serde(default)]
    pub analogy_distance: f64,
    #[serde(default)]
    pub redundancy: f64,
    #[serde(default)]
    pub goal_utility: f64,
    #[serde(default)]
    pub evidence_item_ids: Vec<String>,
}

impl JudgmentItem {
    pub fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            title: String::new(),
            surprise: 0.0,
            novelty: 0.0,
            value_density: 0.0,
            analogy_distance: 0.0,
            redundancy: 0.0,
            goal_utility: 0.0,
            evidence_item_ids: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterestingnessConfig {
    pub surprise_weight: f64,
    pub novelty_weight: f64,
    pub value_density_weight: f64,
    pub analogy_distance_weight: f64,
    pub redundancy_penalty: f64,
}

impl Default for InterestingnessConfig {
    fn default() -> Self {
        Self {
            surprise_weight: 0.34,
            novelty_weight: 0.22,
            value_density_weight: 0.22,
            analogy_distance_weight: 0.22,
            redundancy_penalty: 0.35,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterestingnessScore {
    pub item_id: String,
    pub intrinsic_score: f64,
    pub salience_score: f64,
    pub surprise: f64,
    pub novelty: f64,
    pub value_density: f64,
    pub analogy_distance: f64,
    pub redundancy: f64,
    pub goal_utility: f64,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposedGoal {
    pub goal_id: String,
    pub title: String,
    pub source_item_id: String,
    pub rationale: String,
    pub evidence_item_ids: Vec<String>,
}

pub fn score_interestingness(
    items: &[JudgmentItem],
    config: InterestingnessConfig,
) -> Vec<InterestingnessScore> {
    let mut scored = items
        .iter()
        .map(|item| interestingness_score(item, config))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .salience_score
            .partial_cmp(&left.salience_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.item_id.cmp(&right.item_id))
    });
    scored
}

pub fn propose_goal_for_surprise_gap(
    items: &[JudgmentItem],
    scored: &[InterestingnessScore],
    min_top_surprise: f64,
    min_gap: f64,
) -> GraphStoreResult<Option<ProposedGoal>> {
    let Some(top) = scored.first() else {
        return Ok(None);
    };
    if top.surprise < min_top_surprise.clamp(0.0, 1.0) {
        return Ok(None);
    }
    let runner_up = scored
        .get(1)
        .map(|score| score.salience_score)
        .unwrap_or(0.0);
    if top.salience_score - runner_up < min_gap.max(0.0) {
        return Ok(None);
    }
    let Some(item) = items.iter().find(|item| item.item_id == top.item_id) else {
        return Ok(None);
    };
    let title = if item.title.trim().is_empty() {
        format!("Investigate {}", item.item_id)
    } else {
        format!("Investigate {}", item.title)
    };
    Ok(Some(ProposedGoal {
        goal_id: proposed_goal_id(&item.item_id, top.salience_score)?,
        title,
        source_item_id: item.item_id.clone(),
        rationale: format!(
            "High-surprise item scored {:.3} salience with {:.3} surprise and {:.3} analogy distance.",
            top.salience_score, top.surprise, top.analogy_distance
        ),
        evidence_item_ids: item.evidence_item_ids.clone(),
    }))
}

fn interestingness_score(
    item: &JudgmentItem,
    config: InterestingnessConfig,
) -> InterestingnessScore {
    let surprise = clamp01(item.surprise);
    let novelty = clamp01(item.novelty);
    let value_density = clamp01(item.value_density);
    let analogy_distance = clamp01(item.analogy_distance);
    let redundancy = clamp01(item.redundancy);
    let goal_utility = clamp01(item.goal_utility);
    let intrinsic = (config.surprise_weight * surprise
        + config.novelty_weight * novelty
        + config.value_density_weight * value_density
        + config.analogy_distance_weight * analogy_distance
        - config.redundancy_penalty * redundancy)
        .clamp(0.0, 1.0);
    let salience = (intrinsic + 0.15 * goal_utility).clamp(0.0, 1.0);
    InterestingnessScore {
        item_id: item.item_id.clone(),
        intrinsic_score: round6(intrinsic),
        salience_score: round6(salience),
        surprise,
        novelty,
        value_density,
        analogy_distance,
        redundancy,
        goal_utility,
        reasons: interestingness_reasons(surprise, novelty, value_density, analogy_distance),
    }
}

fn interestingness_reasons(
    surprise: f64,
    novelty: f64,
    value_density: f64,
    analogy_distance: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if surprise >= 0.7 {
        reasons.push("surprise".to_string());
    }
    if novelty >= 0.7 {
        reasons.push("novelty".to_string());
    }
    if value_density >= 0.7 {
        reasons.push("value_density".to_string());
    }
    if analogy_distance >= 0.7 {
        reasons.push("far_analogy".to_string());
    }
    reasons
}

fn proposed_goal_id(item_id: &str, salience: f64) -> GraphStoreResult<String> {
    Ok(format!(
        "goal:proposed:{}",
        stable_hash(json!({
            "item_id": item_id,
            "salience": round6(salience),
            "kind": "interestingness_gap",
        }))?
    ))
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprising_far_analogy_beats_redundant_item_and_emits_goal() {
        let surprising = JudgmentItem {
            item_id: "item:far-analogy".to_string(),
            title: "atom orbit pattern explains scheduler contention".to_string(),
            surprise: 0.94,
            novelty: 0.88,
            value_density: 0.81,
            analogy_distance: 0.92,
            redundancy: 0.05,
            goal_utility: 0.10,
            evidence_item_ids: vec!["atom".to_string(), "scheduler".to_string()],
        };
        let redundant = JudgmentItem {
            item_id: "item:redundant".to_string(),
            title: "known scheduler retry note".to_string(),
            surprise: 0.08,
            novelty: 0.10,
            value_density: 0.40,
            analogy_distance: 0.02,
            redundancy: 0.92,
            goal_utility: 0.95,
            evidence_item_ids: vec!["scheduler".to_string()],
        };

        let scored = score_interestingness(
            &[redundant.clone(), surprising.clone()],
            InterestingnessConfig::default(),
        );

        assert_eq!(scored[0].item_id, surprising.item_id);
        assert!(
            scored[0].intrinsic_score > scored[1].intrinsic_score,
            "intrinsic interestingness should beat redundant utility"
        );
        assert!(scored[0].reasons.contains(&"surprise".to_string()));
        assert!(scored[0].reasons.contains(&"far_analogy".to_string()));

        let proposed = propose_goal_for_surprise_gap(&[redundant, surprising], &scored, 0.8, 0.15)
            .expect("goal id hashing")
            .expect("goal proposal");
        assert_eq!(proposed.source_item_id, "item:far-analogy");
        assert!(proposed.title.starts_with("Investigate atom orbit pattern"));
        assert_eq!(
            proposed.evidence_item_ids,
            vec!["atom".to_string(), "scheduler".to_string()]
        );
    }
}
