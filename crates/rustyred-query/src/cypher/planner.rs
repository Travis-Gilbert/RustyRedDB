//! Row-shape operators for the WITH/ORDER BY/SKIP pipeline.
//!
//! `aggregate` groups rows by N keys and finalizes SUM/AVG/MIN/MAX/COUNT;
//! `sort_rows` orders rows by one or more `OrderBy` clauses (numeric and
//! string-aware); `skip_rows` drops a leading prefix. `execute_read` pipes
//! MATCH bindings through these operators after the pattern phase.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::ast::{AggOp, OrderBy};

#[derive(Clone, Debug)]
pub struct AggregateOutput {
    pub alias: String,
    pub op: AggOp,
    /// Source column key inside each input row. `None` is the COUNT(*) case:
    /// every non-null row contributes one to the count and no value is read.
    pub source_key: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AggregateSpec {
    pub group_keys: Vec<String>,
    pub aggs: Vec<AggregateOutput>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlanCandidate {
    pub candidate_id: String,
    pub summary: String,
    pub native_rank: u32,
    pub estimated_cost_units: f64,
    pub hints: Vec<String>,
}

pub const PLAN_CANDIDATE_EXPAND_LEFT_OUT: &str = "expand_left_out";
pub const PLAN_CANDIDATE_EXPAND_RIGHT_IN: &str = "expand_right_in";

/// Enumerate the native candidate set for a single-hop relationship MATCH.
/// The native rule plan anchors on the left node and expands outward; when
/// the right node is independently anchorable (label or property filter),
/// the reversed anchor is a second enumerated candidate. The learned ranker
/// may only pick among these.
pub fn enumerate_edge_pattern_candidates(right_anchorable: bool) -> Vec<PlanCandidate> {
    let mut candidates = vec![PlanCandidate {
        candidate_id: PLAN_CANDIDATE_EXPAND_LEFT_OUT.to_string(),
        summary: "scan left anchor, expand outgoing edges".to_string(),
        native_rank: 0,
        estimated_cost_units: 100.0,
        hints: vec![],
    }];
    if right_anchorable {
        candidates.push(PlanCandidate {
            candidate_id: PLAN_CANDIDATE_EXPAND_RIGHT_IN.to_string(),
            summary: "scan right anchor, expand incoming edges".to_string(),
            native_rank: 1,
            estimated_cost_units: 100.0,
            hints: vec!["anchor_right".to_string()],
        });
    }
    candidates
}

/// Stable key for "queries of this shape": structure only, never parameter
/// values, so observations pool across runs of the same plan family.
pub fn edge_pattern_shape_key(
    left_label: Option<&str>,
    left_property_keys: &[String],
    edge_type: &str,
    right_label: Option<&str>,
    right_property_keys: &[String],
) -> String {
    format!(
        "edge_pattern|l:{}|lp:{}|e:{}|r:{}|rp:{}",
        left_label.unwrap_or("*"),
        left_property_keys.join(","),
        edge_type,
        right_label.unwrap_or("*"),
        right_property_keys.join(","),
    )
}

#[allow(unused_imports)] // re-export for RO6 consumers
pub use rustyred_thg_core::join_shape_key;

#[derive(Clone, Debug, Default)]
struct PlanObservationAccumulator {
    observations: u32,
    successes: u32,
    total_cost: f64,
    total_cost_squared: f64,
}

/// Durable slot for one (shape, candidate) observation accumulator.
/// MCP writes these as `PlanSteeringObservation` nodes; the process copy is
/// only a reconstruction of the store.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct PlanSteerSlot {
    pub shape_key: String,
    pub candidate_id: String,
    pub observations: u32,
    pub successes: u32,
    pub total_cost: f64,
    pub total_cost_squared: f64,
}

/// Observation accumulator for the steered optimizer: per query shape and
/// candidate, the measured cost units of real executions. MCP loads this
/// from the store at the start of every query. Until it accumulates past
/// the cold-start floor, the native plan runs unconditionally.
#[derive(Debug, Default)]
pub struct PlanSteeringState {
    observations: std::sync::Mutex<BTreeMap<(String, String), PlanObservationAccumulator>>,
    trial_counter: std::sync::atomic::AtomicU32,
}

impl PlanSteeringState {
    pub fn next_trial_nonce(&self) -> u32 {
        self.trial_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn record(&self, shape_key: &str, candidate_id: &str, cost_units: f64, success: bool) {
        if !cost_units.is_finite() || cost_units < 0.0 {
            return;
        }
        let mut observations = self
            .observations
            .lock()
            .expect("plan steering observations lock");
        let slot = observations
            .entry((shape_key.to_string(), candidate_id.to_string()))
            .or_default();
        slot.observations = slot.observations.saturating_add(1);
        if success {
            slot.successes = slot.successes.saturating_add(1);
        }
        slot.total_cost += cost_units;
        slot.total_cost_squared += cost_units * cost_units;
    }

    /// Snapshot metrics for one query shape. Uncertainty is the standard
    /// error of the mean cost: it shrinks as observations accumulate, which
    /// is what the exploration term in the steering score expects.
    pub fn metrics_for(&self, shape_key: &str) -> Vec<PlanObservationMetrics> {
        let observations = self
            .observations
            .lock()
            .expect("plan steering observations lock");
        observations
            .iter()
            .filter(|((shape, _), accumulator)| shape == shape_key && accumulator.observations > 0)
            .map(|((_, candidate_id), accumulator)| {
                let count = accumulator.observations as f64;
                let mean = accumulator.total_cost / count;
                let variance = (accumulator.total_cost_squared / count - mean * mean).max(0.0);
                PlanObservationMetrics {
                    candidate_id: candidate_id.clone(),
                    observations: accumulator.observations,
                    mean_cost_units: mean,
                    success_rate: accumulator.successes as f64 / count,
                    uncertainty: (variance / count).sqrt(),
                }
            })
            .collect()
    }

    /// Replace one slot with a previously persisted accumulator.
    pub fn load_slot(&self, slot: PlanSteerSlot) {
        if slot.observations == 0 {
            return;
        }
        let mut observations = self
            .observations
            .lock()
            .expect("plan steering observations lock");
        observations.insert(
            (slot.shape_key, slot.candidate_id),
            PlanObservationAccumulator {
                observations: slot.observations,
                successes: slot.successes,
                total_cost: slot.total_cost,
                total_cost_squared: slot.total_cost_squared,
            },
        );
    }

    pub fn slots(&self) -> Vec<PlanSteerSlot> {
        let observations = self
            .observations
            .lock()
            .expect("plan steering observations lock");
        observations
            .iter()
            .map(|((shape_key, candidate_id), accumulator)| PlanSteerSlot {
                shape_key: shape_key.clone(),
                candidate_id: candidate_id.clone(),
                observations: accumulator.observations,
                successes: accumulator.successes,
                total_cost: accumulator.total_cost,
                total_cost_squared: accumulator.total_cost_squared,
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlanObservationMetrics {
    pub candidate_id: String,
    pub observations: u32,
    pub mean_cost_units: f64,
    pub success_rate: f64,
    pub uncertainty: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlannerSteeringPolicy {
    pub min_observations: u32,
    pub exploration_weight: f64,
    pub max_tail_risk_multiplier: f64,
    #[serde(default)]
    pub trial_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_nonce: Option<u32>,
}

impl Default for PlannerSteeringPolicy {
    fn default() -> Self {
        Self {
            min_observations: 20,
            exploration_weight: 0.25,
            max_tail_risk_multiplier: 2.0,
            trial_rate: 0.1,
            trial_nonce: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlannerSteeringDecision {
    pub selected_candidate_id: String,
    pub native_candidate_id: String,
    pub abstained: bool,
    pub reason: String,
    pub score: f64,
    pub observed_candidate_count: usize,
}

/// Bao-style steering over a bounded candidate set. The rule planner proposes
/// candidates; this ranker may choose among them only after enough observations
/// exist. It never creates a plan that is absent from `candidates`.
pub fn steer_plan_candidates(
    candidates: &[PlanCandidate],
    metrics: &[PlanObservationMetrics],
    policy: PlannerSteeringPolicy,
) -> Option<PlannerSteeringDecision> {
    let native = candidates
        .iter()
        .min_by_key(|candidate| candidate.native_rank)?;
    if candidates.len() <= 1 {
        return Some(native_decision(native, "single_candidate"));
    }

    let policy = normalize_policy(policy);
    let metrics_by_id = metrics
        .iter()
        .filter(|item| item.observations > 0)
        .map(|item| (item.candidate_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let total_observations = candidates
        .iter()
        .filter_map(|candidate| metrics_by_id.get(candidate.candidate_id.as_str()))
        .map(|item| item.observations)
        .sum::<u32>();
    if total_observations < policy.min_observations {
        if let Some(nonce) = policy.trial_nonce {
            let trial_bucket = ((policy.trial_rate.clamp(0.0, 1.0) * 10.0).round() as u32).max(0);
            if trial_bucket > 0 && nonce > 0 && nonce % 10 < trial_bucket {
                let unobserved = candidates.iter().find(|candidate| {
                    candidate.candidate_id != native.candidate_id
                        && !metrics_by_id.contains_key(candidate.candidate_id.as_str())
                });
                if let Some(candidate) = unobserved {
                    return Some(PlannerSteeringDecision {
                        selected_candidate_id: candidate.candidate_id.clone(),
                        native_candidate_id: native.candidate_id.clone(),
                        abstained: false,
                        reason: "bounded_trial_unobserved_candidate".to_string(),
                        score: candidate.estimated_cost_units,
                        observed_candidate_count: metrics_by_id.len(),
                    });
                }
            }
        }
        return Some(native_decision(native, "cold_start_native_floor"));
    }

    let native_mean = metrics_by_id
        .get(native.candidate_id.as_str())
        .map(|item| item.mean_cost_units)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(native.estimated_cost_units.max(1.0));
    let tail_risk_ceiling = native_mean * policy.max_tail_risk_multiplier;

    let mut best: Option<(&PlanCandidate, f64)> = None;
    for candidate in candidates {
        let Some(observed) = metrics_by_id.get(candidate.candidate_id.as_str()) else {
            continue;
        };
        if observed.mean_cost_units <= 0.0
            || !observed.mean_cost_units.is_finite()
            || observed.success_rate <= 0.0
            || observed.mean_cost_units > tail_risk_ceiling
        {
            continue;
        }
        let score = observed.mean_cost_units / observed.success_rate.clamp(0.01, 1.0)
            - policy.exploration_weight * observed.uncertainty.max(0.0);
        match best {
            Some((prior, prior_score))
                if prior_score < score
                    || (prior_score == score && prior.native_rank <= candidate.native_rank) => {}
            _ => best = Some((candidate, score)),
        }
    }

    let Some((candidate, score)) = best else {
        return Some(native_decision(native, "all_learned_candidates_too_risky"));
    };
    Some(PlannerSteeringDecision {
        selected_candidate_id: candidate.candidate_id.clone(),
        native_candidate_id: native.candidate_id.clone(),
        abstained: candidate.candidate_id == native.candidate_id,
        reason: if candidate.candidate_id == native.candidate_id {
            "native_candidate_ranked_best".to_string()
        } else {
            "learned_ranker_selected_enumerated_candidate".to_string()
        },
        score,
        observed_candidate_count: metrics_by_id.len(),
    })
}

pub fn aggregate(rows: &[Map<String, Value>], spec: &AggregateSpec) -> Vec<Map<String, Value>> {
    // Two parallel BTreeMaps keyed by the same group-string so the deterministic
    // emit-order matches what tests assert against.
    let mut groups: BTreeMap<String, (Map<String, Value>, BTreeMap<String, AggAccumulator>)> =
        BTreeMap::new();
    for row in rows {
        let key = make_group_key(&spec.group_keys, row);
        let entry = groups.entry(key).or_insert_with(|| {
            let mut group_row = Map::new();
            for key_name in &spec.group_keys {
                if let Some(v) = row.get(key_name) {
                    group_row.insert(key_name.clone(), v.clone());
                }
            }
            let accs: BTreeMap<String, AggAccumulator> = spec
                .aggs
                .iter()
                .map(|out| (out.alias.clone(), AggAccumulator::default()))
                .collect();
            (group_row, accs)
        });
        for out in &spec.aggs {
            let source = match &out.source_key {
                Some(k) => row.get(k).cloned().unwrap_or(Value::Null),
                None => Value::Null,
            };
            let acc = entry
                .1
                .get_mut(&out.alias)
                .expect("accumulator initialised in or_insert_with");
            acc.observe(out.op, &source);
        }
    }
    let mut out_rows: Vec<Map<String, Value>> = Vec::with_capacity(groups.len());
    for (_, (group_row, accs)) in groups {
        let mut row = group_row;
        for out in &spec.aggs {
            let acc = accs
                .get(&out.alias)
                .expect("accumulator present at finalize");
            row.insert(out.alias.clone(), acc.finalize(out.op));
        }
        out_rows.push(row);
    }
    out_rows
}

fn normalize_policy(mut policy: PlannerSteeringPolicy) -> PlannerSteeringPolicy {
    if policy.min_observations == 0 {
        policy.min_observations = 20;
    }
    if !policy.exploration_weight.is_finite() || policy.exploration_weight < 0.0 {
        policy.exploration_weight = 0.0;
    }
    if !policy.max_tail_risk_multiplier.is_finite() || policy.max_tail_risk_multiplier < 1.0 {
        policy.max_tail_risk_multiplier = 1.0;
    }
    policy
}

fn native_decision(native: &PlanCandidate, reason: &str) -> PlannerSteeringDecision {
    PlannerSteeringDecision {
        selected_candidate_id: native.candidate_id.clone(),
        native_candidate_id: native.candidate_id.clone(),
        abstained: true,
        reason: reason.to_string(),
        score: native.estimated_cost_units,
        observed_candidate_count: 0,
    }
}

pub fn sort_rows(rows: &mut [Map<String, Value>], order: &[OrderBy]) {
    if order.is_empty() {
        return;
    }
    rows.sort_by(|a, b| {
        for clause in order {
            let av = a.get(&clause.expression).cloned().unwrap_or(Value::Null);
            let bv = b.get(&clause.expression).cloned().unwrap_or(Value::Null);
            let ord = value_cmp(&av, &bv);
            if ord != std::cmp::Ordering::Equal {
                return if clause.descending {
                    ord.reverse()
                } else {
                    ord
                };
            }
        }
        std::cmp::Ordering::Equal
    });
}

pub fn skip_rows(rows: Vec<Map<String, Value>>, skip: usize) -> Vec<Map<String, Value>> {
    if skip == 0 {
        return rows;
    }
    rows.into_iter().skip(skip).collect()
}

fn value_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Number(an), Value::Number(bn)) => an
            .as_f64()
            .unwrap_or(f64::NAN)
            .partial_cmp(&bn.as_f64().unwrap_or(f64::NAN))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(av), Value::String(bv)) => av.cmp(bv),
        (Value::Bool(av), Value::Bool(bv)) => av.cmp(bv),
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Less,
        (_, Value::Null) => std::cmp::Ordering::Greater,
        _ => format!("{a}").cmp(&format!("{b}")),
    }
}

fn make_group_key(keys: &[String], row: &Map<String, Value>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(keys.len());
    for k in keys {
        parts.push(format!(
            "{}={}",
            k,
            row.get(k).cloned().unwrap_or(Value::Null)
        ));
    }
    parts.join("|")
}

#[derive(Debug, Clone, Default)]
struct AggAccumulator {
    /// Total rows observed; this is what COUNT emits.
    row_count: u64,
    /// Count of rows whose source value parsed as a finite f64; AVG divides by this.
    numeric_count: u64,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl AggAccumulator {
    fn observe(&mut self, _op: AggOp, v: &Value) {
        // Every observed row contributes to the row count. COUNT(*) and
        // count(n) both reach this path with v = Null because no source column
        // is attached, but the matched row itself is what's being counted.
        self.row_count += 1;
        if let Some(num) = v.as_f64() {
            self.sum += num;
            self.numeric_count += 1;
            self.min = Some(self.min.map_or(num, |m| m.min(num)));
            self.max = Some(self.max.map_or(num, |m| m.max(num)));
        }
    }

    fn finalize(&self, op: AggOp) -> Value {
        match op {
            AggOp::Count => json!(self.row_count),
            AggOp::Sum => {
                // Emit integer JSON when the running sum is a whole number; this
                // keeps `sum(integer_column)` round-tripping cleanly in JSON.
                if self.sum.fract() == 0.0 && self.sum.abs() < (i64::MAX as f64) {
                    json!(self.sum as i64)
                } else {
                    json!(self.sum)
                }
            }
            AggOp::Avg => {
                if self.numeric_count == 0 {
                    Value::Null
                } else {
                    json!(self.sum / self.numeric_count as f64)
                }
            }
            AggOp::Min => self.min.map(|v| json!(v)).unwrap_or(Value::Null),
            AggOp::Max => self.max.map(|v| json!(v)).unwrap_or(Value::Null),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cypher::ast::{AggOp, OrderBy};
    use serde_json::json;

    fn rows_sample() -> Vec<serde_json::Map<String, serde_json::Value>> {
        vec![
            {
                let mut m = serde_json::Map::new();
                m.insert("cat".into(), json!("blue"));
                m.insert("n.score".into(), json!(5));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("cat".into(), json!("blue"));
                m.insert("n.score".into(), json!(7));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("cat".into(), json!("red"));
                m.insert("n.score".into(), json!(3));
                m
            },
        ]
    }

    #[test]
    fn aggregate_sum_groups_rows_by_key() {
        let rows = rows_sample();
        let spec = AggregateSpec {
            group_keys: vec!["cat".into()],
            aggs: vec![AggregateOutput {
                alias: "total".into(),
                op: AggOp::Sum,
                source_key: Some("n.score".into()),
            }],
        };
        let out = aggregate(&rows, &spec);
        assert_eq!(out.len(), 2);
        let blue = out.iter().find(|r| r["cat"] == "blue").unwrap();
        assert_eq!(blue["total"], json!(12));
        let red = out.iter().find(|r| r["cat"] == "red").unwrap();
        assert_eq!(red["total"], json!(3));
    }

    #[test]
    fn sort_orders_rows_by_key_desc() {
        let mut rows = vec![
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(1));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(3));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(2));
                m
            },
        ];
        sort_rows(
            &mut rows,
            &[OrderBy {
                expression: "c".into(),
                descending: true,
            }],
        );
        assert_eq!(rows[0]["c"], json!(3));
        assert_eq!(rows[2]["c"], json!(1));
    }

    #[test]
    fn skip_drops_leading_rows() {
        let rows = vec![
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(1));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(2));
                m
            },
            {
                let mut m = serde_json::Map::new();
                m.insert("c".into(), json!(3));
                m
            },
        ];
        let trimmed = skip_rows(rows, 2);
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0]["c"], json!(3));
    }

    #[test]
    fn aggregate_avg_returns_mean() {
        let rows = rows_sample();
        let spec = AggregateSpec {
            group_keys: vec!["cat".into()],
            aggs: vec![AggregateOutput {
                alias: "mean".into(),
                op: AggOp::Avg,
                source_key: Some("n.score".into()),
            }],
        };
        let out = aggregate(&rows, &spec);
        let blue = out.iter().find(|r| r["cat"] == "blue").unwrap();
        assert_eq!(blue["mean"], json!(6.0));
        let red = out.iter().find(|r| r["cat"] == "red").unwrap();
        assert_eq!(red["mean"], json!(3.0));
    }

    #[test]
    fn aggregate_min_max_track_extrema() {
        let rows = rows_sample();
        let spec = AggregateSpec {
            group_keys: vec!["cat".into()],
            aggs: vec![
                AggregateOutput {
                    alias: "lo".into(),
                    op: AggOp::Min,
                    source_key: Some("n.score".into()),
                },
                AggregateOutput {
                    alias: "hi".into(),
                    op: AggOp::Max,
                    source_key: Some("n.score".into()),
                },
            ],
        };
        let out = aggregate(&rows, &spec);
        let blue = out.iter().find(|r| r["cat"] == "blue").unwrap();
        assert_eq!(blue["lo"], json!(5.0));
        assert_eq!(blue["hi"], json!(7.0));
    }

    #[test]
    fn steering_abstains_to_native_until_enough_observations_exist() {
        let candidates = plan_candidates();
        let decision = steer_plan_candidates(
            &candidates,
            &[PlanObservationMetrics {
                candidate_id: "hinted-index-first".to_string(),
                observations: 3,
                mean_cost_units: 20.0,
                success_rate: 1.0,
                uncertainty: 1.0,
            }],
            PlannerSteeringPolicy::default(),
        )
        .unwrap();

        assert!(decision.abstained);
        assert_eq!(decision.selected_candidate_id, "native");
        assert_eq!(decision.reason, "cold_start_native_floor");
    }

    #[test]
    fn steering_bounded_trial_selects_unobserved_candidate_when_nonce_hits_bucket() {
        let candidates = plan_candidates();
        let trial = steer_plan_candidates(
            &candidates,
            &[],
            PlannerSteeringPolicy {
                trial_rate: 0.1,
                trial_nonce: Some(10),
                ..PlannerSteeringPolicy::default()
            },
        )
        .unwrap();
        assert!(!trial.abstained);
        assert_eq!(trial.selected_candidate_id, "hinted-index-first");
        assert_eq!(trial.reason, "bounded_trial_unobserved_candidate");

        let first_query = steer_plan_candidates(
            &candidates,
            &[],
            PlannerSteeringPolicy {
                trial_rate: 0.1,
                trial_nonce: Some(0),
                ..PlannerSteeringPolicy::default()
            },
        )
        .unwrap();
        assert!(first_query.abstained);
        assert_eq!(first_query.selected_candidate_id, "native");
        assert_eq!(first_query.reason, "cold_start_native_floor");
    }

    #[test]
    fn steering_selects_only_enumerated_lower_cost_candidate() {
        let candidates = plan_candidates();
        let decision = steer_plan_candidates(
            &candidates,
            &[
                PlanObservationMetrics {
                    candidate_id: "native".to_string(),
                    observations: 24,
                    mean_cost_units: 100.0,
                    success_rate: 1.0,
                    uncertainty: 0.0,
                },
                PlanObservationMetrics {
                    candidate_id: "hinted-index-first".to_string(),
                    observations: 24,
                    mean_cost_units: 40.0,
                    success_rate: 0.95,
                    uncertainty: 1.0,
                },
                PlanObservationMetrics {
                    candidate_id: "not-enumerated".to_string(),
                    observations: 100,
                    mean_cost_units: 1.0,
                    success_rate: 1.0,
                    uncertainty: 0.0,
                },
            ],
            PlannerSteeringPolicy::default(),
        )
        .unwrap();

        assert!(!decision.abstained);
        assert_eq!(decision.selected_candidate_id, "hinted-index-first");
    }

    #[test]
    fn steering_rejects_tail_risk_even_when_mean_looks_learned() {
        let candidates = plan_candidates();
        let decision = steer_plan_candidates(
            &candidates,
            &[
                PlanObservationMetrics {
                    candidate_id: "native".to_string(),
                    observations: 24,
                    mean_cost_units: 100.0,
                    success_rate: 1.0,
                    uncertainty: 0.0,
                },
                PlanObservationMetrics {
                    candidate_id: "hinted-index-first".to_string(),
                    observations: 24,
                    mean_cost_units: 500.0,
                    success_rate: 1.0,
                    uncertainty: 0.0,
                },
            ],
            PlannerSteeringPolicy {
                max_tail_risk_multiplier: 2.0,
                ..PlannerSteeringPolicy::default()
            },
        )
        .unwrap();

        assert_eq!(decision.selected_candidate_id, "native");
        assert_eq!(decision.reason, "native_candidate_ranked_best");
    }

    fn plan_candidates() -> Vec<PlanCandidate> {
        vec![
            PlanCandidate {
                candidate_id: "native".to_string(),
                summary: "native rule plan".to_string(),
                native_rank: 0,
                estimated_cost_units: 100.0,
                hints: vec![],
            },
            PlanCandidate {
                candidate_id: "hinted-index-first".to_string(),
                summary: "index-first hinted plan".to_string(),
                native_rank: 1,
                estimated_cost_units: 80.0,
                hints: vec!["index_first".to_string()],
            },
        ]
    }
}
