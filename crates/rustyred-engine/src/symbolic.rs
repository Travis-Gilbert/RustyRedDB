use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub use datalog::DATALOG_RULE_IDS;

/// Bounded adapters between a retained graph revision and the pure engine
/// crates. Formerly `rustyredcore_THG/src/symbolic/graph.rs`, which lived in
/// the `theseus_native` PyO3 package until that door was retired.
pub mod graph;
pub use graph::{
    apply_derivation, apply_extraction, fact_pack_from_graph, GraphScope, SymbolicGraph,
};

pub fn canonical_json(value: &Value) -> Result<String, String> {
    serde_json::to_string(&canonicalize_json(value))
        .map_err(|err| format!("could not serialize JSON: {err}"))
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            let mut canonical = Map::new();
            for (key, child) in entries {
                canonical.insert(key.clone(), canonicalize_json(child));
            }
            Value::Object(canonical)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonicalize_json).collect()),
        _ => value.clone(),
    }
}

pub fn stable_hash_value(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(&canonical_json(value)?))
}

pub fn stable_hash_json(payload_json: &str) -> Result<String, String> {
    let payload = parse_json(payload_json)?;
    stable_hash_value(&payload)
}

pub fn derive_datalog_receipt_from_json(payload_json: &str) -> Result<Value, String> {
    let payload = parse_json(payload_json)?;
    derive_datalog_receipt(&payload)
}

/// Compatibility adapter; the fixed rules live only in theorem-datalog.
pub fn derive_datalog_receipt(payload: &Value) -> Result<Value, String> {
    datalog::derive_legacy_receipt(payload).map_err(|error| error.to_string())
}

pub fn probabilistic_source_reliability_from_json(payload_json: &str) -> Result<Value, String> {
    let payload = parse_json(payload_json)?;
    probabilistic_source_reliability(&payload)
}

pub fn probabilistic_source_reliability(payload: &Value) -> Result<Value, String> {
    let source_id = arg_string(payload, "source_id", "source");
    let prior_alpha = numeric_arg(payload, "prior_alpha", 1.0);
    let prior_beta = numeric_arg(payload, "prior_beta", 1.0);
    let corroborated = integer_arg(payload, "corroborated", 0).max(0) as f64;
    let contradicted = integer_arg(payload, "contradicted", 0).max(0) as f64;
    let alpha = prior_alpha + corroborated;
    let beta = prior_beta + contradicted;
    let total = alpha + beta;
    let mean = if total != 0.0 { alpha / total } else { 0.5 };
    let variance = if total > 0.0 {
        (alpha * beta) / ((total * total) * (total + 1.0))
    } else {
        0.0
    };
    posterior_receipt(
        "source-reliability",
        &source_id,
        json!({"alpha": prior_alpha, "beta": prior_beta}),
        json!({"corroborated": integer_arg(payload, "corroborated", 0), "contradicted": integer_arg(payload, "contradicted", 0)}),
        json!({"alpha": alpha, "beta": beta, "mean": mean, "variance": variance}),
        json!({"source_id": source_id, "distribution": "beta"}),
    )
}

pub fn probabilistic_expected_value_from_json(payload_json: &str) -> Result<Value, String> {
    let payload = parse_json(payload_json)?;
    probabilistic_expected_value(&payload)
}

pub fn probabilistic_expected_value(payload: &Value) -> Result<Value, String> {
    let current_uncertainty = numeric_arg(payload, "current_uncertainty", 0.0);
    let expected_uncertainty_after = numeric_arg(payload, "expected_uncertainty_after", 0.0);
    let decision_value = numeric_arg(payload, "decision_value", 1.0);
    let validator_cost = numeric_arg(payload, "validator_cost", 0.0);
    let selected_verifier_id = arg_string(payload, "selected_verifier_id", "unspecified");
    let result = arg_string(payload, "result", "pending");
    let uncertainty_reduction = (current_uncertainty - expected_uncertainty_after).max(0.0);
    let expected_value = (uncertainty_reduction * decision_value) - validator_cost;
    posterior_receipt(
        "expected-value-of-information",
        "",
        json!({"current_uncertainty": current_uncertainty}),
        json!({
            "expected_uncertainty_after": expected_uncertainty_after,
            "validator_cost": validator_cost,
            "selected_verifier_id": selected_verifier_id,
            "result": result,
        }),
        json!({
            "expected_value": expected_value,
            "uncertainty_reduction": uncertainty_reduction,
        }),
        json!({"decision_value": decision_value}),
    )
}

pub fn evolution_archive_from_json(payload_json: &str) -> Result<Value, String> {
    let payload = parse_json(payload_json)?;
    evolution_archive(payload)
}

pub fn evolution_archive(payload: Value) -> Result<Value, String> {
    let candidates = evolution_candidates_from_payload(&payload)?;
    let elites_per_niche = integer_arg(&payload, "elites_per_niche", 2);
    let mut niches: BTreeMap<String, Vec<EvolutionCandidateRecord>> = BTreeMap::new();
    for candidate in candidates {
        niches
            .entry(candidate.niche.clone())
            .or_default()
            .push(candidate);
    }

    let mut elites_by_niche: Map<String, Value> = Map::new();
    let mut selected: Vec<EvolutionCandidateRecord> = Vec::new();
    let mut candidate_count = 0usize;
    for (niche, mut niche_candidates) in niches {
        candidate_count += niche_candidates.len();
        niche_candidates.sort_by(compare_evolution_elite_rank);
        let take_count = python_slice_stop_len(niche_candidates.len(), elites_per_niche);
        let ranked = niche_candidates
            .into_iter()
            .take(take_count)
            .collect::<Vec<_>>();
        selected.extend(ranked.clone());
        elites_by_niche.insert(
            niche,
            Value::Array(
                ranked
                    .iter()
                    .map(EvolutionCandidateRecord::to_value)
                    .collect(),
            ),
        );
    }

    let archive_hash = evolution_archive_hash(&selected)?;
    let rejected_count = candidate_count.saturating_sub(selected.len());
    Ok(json!({
        "engine": "quality-diversity-python-fallback",
        "archive_hash": archive_hash,
        "elites_by_niche": Value::Object(elites_by_niche),
        "rejected_count": rejected_count,
        "writeback_policy": "read-only",
    }))
}

fn posterior_receipt(
    kind: &str,
    source_id: &str,
    prior: Value,
    observations: Value,
    posterior: Value,
    metadata: Value,
) -> Result<Value, String> {
    let model_id = if kind == "source-reliability" {
        format!("source-reliability:{source_id}")
    } else {
        kind.to_string()
    };
    let engine = "beta-binomial-python-fallback";
    let hash_payload = json!({
        "engine": engine,
        "model_id": model_id,
        "prior": prior,
        "observations": observations,
        "posterior": posterior,
        "metadata": metadata,
    });
    let receipt_hash = stable_hash_value(&hash_payload)?;
    Ok(json!({
        "engine": engine,
        "model_id": hash_payload["model_id"].clone(),
        "prior": hash_payload["prior"].clone(),
        "observations": hash_payload["observations"].clone(),
        "posterior": hash_payload["posterior"].clone(),
        "metadata": hash_payload["metadata"].clone(),
        "receipt_hash": receipt_hash,
        "writeback_policy": "read-only",
    }))
}

#[derive(Clone)]
struct EvolutionCandidateRecord {
    candidate_id: String,
    niche: String,
    score: f64,
    novelty: f64,
    payload: Value,
}

impl EvolutionCandidateRecord {
    fn to_value(&self) -> Value {
        json!({
            "candidate_id": self.candidate_id,
            "niche": self.niche,
            "score": self.score,
            "novelty": self.novelty,
            "payload": self.payload.clone(),
        })
    }
}

fn evolution_candidates_from_payload(
    payload: &Value,
) -> Result<Vec<EvolutionCandidateRecord>, String> {
    let candidates = payload
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "evolution archive payload expected candidates array".to_string())?;
    candidates
        .iter()
        .map(evolution_candidate_from_value)
        .collect()
}

fn evolution_candidate_from_value(value: &Value) -> Result<EvolutionCandidateRecord, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "evolution candidate expected JSON object".to_string())?;
    let candidate_id = object
        .get("candidate_id")
        .map(value_to_string)
        .unwrap_or_default();
    let niche = object
        .get("niche")
        .map(value_to_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let payload = object
        .get("payload")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    Ok(EvolutionCandidateRecord {
        candidate_id,
        niche,
        score: object.get("score").and_then(value_to_f64).unwrap_or(0.0),
        novelty: object.get("novelty").and_then(value_to_f64).unwrap_or(0.0),
        payload,
    })
}

fn compare_evolution_elite_rank(
    left: &EvolutionCandidateRecord,
    right: &EvolutionCandidateRecord,
) -> Ordering {
    compare_f64_desc(left.score, right.score)
        .then_with(|| compare_f64_desc(left.novelty, right.novelty))
        .then_with(|| right.candidate_id.cmp(&left.candidate_id))
}

fn compare_evolution_hash_rank(
    left: &EvolutionCandidateRecord,
    right: &EvolutionCandidateRecord,
) -> Ordering {
    left.niche
        .cmp(&right.niche)
        .then_with(|| compare_f64_desc(left.score, right.score))
        .then_with(|| left.candidate_id.cmp(&right.candidate_id))
}

fn compare_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn python_slice_stop_len(len: usize, stop: i64) -> usize {
    if stop >= 0 {
        return len.min(stop as usize);
    }
    len.saturating_sub(stop.unsigned_abs() as usize)
}

fn evolution_archive_hash(candidates: &[EvolutionCandidateRecord]) -> Result<String, String> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(compare_evolution_hash_rank);
    stable_hash_value(&Value::Array(
        sorted
            .iter()
            .map(EvolutionCandidateRecord::to_value)
            .collect(),
    ))
}

fn parse_json(text: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|err| format!("expected valid JSON: {err}"))
}

fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn object_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn numeric_arg(payload: &Value, key: &str, default: f64) -> f64 {
    payload.get(key).and_then(value_to_f64).unwrap_or(default)
}

fn integer_arg(payload: &Value, key: &str, default: i64) -> i64 {
    payload.get(key).map(value_to_i64).unwrap_or(default)
}

fn arg_string(payload: &Value, key: &str, default: &str) -> String {
    payload
        .get(key)
        .map(value_to_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(value) => value.to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    if let Some(value) = value.as_f64() {
        return Some(value);
    }
    value.as_str().and_then(|value| value.parse::<f64>().ok())
}

fn value_to_i64(value: &Value) -> i64 {
    if let Some(value) = value.as_i64() {
        return value;
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value).unwrap_or(i64::MAX);
    }
    if let Some(value) = value.as_f64() {
        return value as i64;
    }
    value
        .as_str()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn datalog_receipt_matches_reference_shape_for_all_rules() {
        let payload = json!([
            {"relation": "claim", "entity_id": "claim-1", "attributes": {"status": "proposed"}, "source_ref": "u", "fact_id": "f1"},
            {"relation": "object", "entity_id": "obj-1", "attributes": {"title": "Same", "properties": {"visibility": "private"}}, "source_ref": "u", "fact_id": "f2"},
            {"relation": "object", "entity_id": "obj-2", "attributes": {"title": "same"}, "source_ref": "u", "fact_id": "f3"},
            {"relation": "context_atom", "entity_id": "atom-1", "attributes": {"object_pk": "obj-1", "artifact_id": "ctx-1", "metadata": {"export_candidate": true}}, "source_ref": "u", "fact_id": "f4"}
        ]);

        let receipt = derive_datalog_receipt(&payload).unwrap();
        let relations: BTreeSet<String> = receipt["derived_facts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|fact| object_field(fact, "relation").to_string())
            .collect();

        assert!(relations.contains("unsupported_claim"));
        assert!(relations.contains("likely_duplicate_entity"));
        assert!(relations.contains("claim_has_no_independent_support"));
        assert!(relations.contains("private_source_reaches_export_candidate"));
        assert_eq!(receipt["engine"], "python-reference-datalog");
        assert_eq!(
            receipt["rule_ids"].as_array().unwrap().len(),
            DATALOG_RULE_IDS.len()
        );
    }

    #[test]
    fn probabilistic_receipt_hashes_are_content_addressed() {
        let receipt = probabilistic_source_reliability(&json!({
            "source_id": "source-a",
            "prior_alpha": 2.0,
            "prior_beta": 2.0,
            "corroborated": 6,
            "contradicted": 2,
        }))
        .unwrap();

        assert_eq!(receipt["posterior"]["alpha"], 8.0);
        assert_eq!(receipt["posterior"]["beta"], 4.0);
        assert!(receipt["receipt_hash"].as_str().unwrap().len() == 64);
    }

    #[test]
    fn expected_value_receipt_records_verifier_and_result() {
        let receipt = probabilistic_expected_value(&json!({
            "current_uncertainty": 0.7,
            "expected_uncertainty_after": 0.2,
            "decision_value": 2.0,
            "validator_cost": 0.4,
            "selected_verifier_id": "deepseek",
            "result": "verify",
        }))
        .unwrap();

        assert_eq!(receipt["observations"]["selected_verifier_id"], "deepseek");
        assert_eq!(receipt["observations"]["result"], "verify");
        assert_eq!(receipt["posterior"]["expected_value"], 0.5999999999999999);
    }

    #[test]
    fn evolution_archive_preserves_python_ranking_shape() {
        let receipt = evolution_archive(json!({
            "candidates": [
                {"candidate_id": "a", "niche": "n1", "score": 0.8, "novelty": 0.1, "payload": {"k": 1}},
                {"candidate_id": "b", "niche": "n1", "score": 0.8, "novelty": 0.9, "payload": {"k": 2}},
                {"candidate_id": "c", "niche": "n2", "score": 0.4, "novelty": 0.2, "payload": {}}
            ],
            "elites_per_niche": 1,
        }))
        .unwrap();

        assert_eq!(receipt["engine"], "quality-diversity-python-fallback");
        assert_eq!(receipt["elites_by_niche"]["n1"][0]["candidate_id"], "b");
        assert_eq!(receipt["elites_by_niche"]["n2"][0]["candidate_id"], "c");
        assert_eq!(receipt["rejected_count"], 1);
        assert!(receipt["archive_hash"].as_str().unwrap().len() == 64);
    }
}

#[cfg(test)]
mod graph_tests;
