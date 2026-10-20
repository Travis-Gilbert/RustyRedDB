use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

use rustyred_thg_core::ppr_cache::{
    cached_csr_personalized_pagerank, cached_personalized_pagerank, clear_scoped_ppr_cache,
    reset_scoped_ppr_execution_counters, scoped_ppr_execution_counters,
};
use rustyred_thg_core::{epistemic_shadow_ppr, CsrGraph, InMemoryGraphStore, NodeRecord};
use serde_json::json;

static PPR_COUNTER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn counter_guard() -> std::sync::MutexGuard<'static, ()> {
    PPR_COUNTER_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap()
}

fn weighted_fixture() -> HashMap<String, Vec<(String, f64)>> {
    HashMap::from([
        (
            "a".to_string(),
            vec![("b".to_string(), 3.0), ("c".to_string(), 1.0)],
        ),
        (
            "b".to_string(),
            vec![("c".to_string(), 2.0), ("d".to_string(), 1.0)],
        ),
        ("c".to_string(), vec![("a".to_string(), 1.0)]),
        ("d".to_string(), vec![]),
        ("isolated".to_string(), vec![]),
    ])
}

fn cosine(left: &HashMap<String, f64>, right: &HashMap<String, f64>) -> f64 {
    let keys = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for key in keys {
        let l = left.get(&key).copied().unwrap_or(0.0);
        let r = right.get(&key).copied().unwrap_or(0.0);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

#[test]
fn csr_local_push_matches_graph_rs_ppr_with_cosine_tolerance() {
    let adjacency = weighted_fixture();
    let seeds = HashMap::from([("a".to_string(), 0.75), ("isolated".to_string(), 0.25)]);
    let legacy = rustyred_thg_core::personalized_pagerank(&adjacency, &seeds, 0.15, 1e-7, 200_000);
    let csr = CsrGraph::from_weighted_adjacency(&adjacency, seeds.keys());
    let csr_scores = csr.personalized_pagerank_local_push_named(&seeds, 0.15, 1e-7, 200_000);

    assert!(
        cosine(&legacy, &csr_scores) > 0.999_999,
        "CSR local-push PPR should match graph.rs PPR: legacy={legacy:?}, csr={csr_scores:?}"
    );
}

#[test]
fn scoped_ppr_cache_miss_routes_through_csr_and_hit_reuses_it() {
    let _guard = counter_guard();
    clear_scoped_ppr_cache();
    reset_scoped_ppr_execution_counters();
    let adjacency = weighted_fixture();
    let seeds = HashMap::from([("a".to_string(), 1.0)]);

    let first = cached_personalized_pagerank("ppr-csr", 1, &adjacency, &seeds, 0.15, 1e-6, 5000)
        .expect("first PPR cache key should be content-addressable");
    let after_first = scoped_ppr_execution_counters();
    let second = cached_personalized_pagerank("ppr-csr", 1, &adjacency, &seeds, 0.15, 1e-6, 5000)
        .expect("second PPR cache key should be content-addressable");
    let after_second = scoped_ppr_execution_counters();

    assert_eq!(first, second);
    assert_eq!(after_first.csr_ppr_runs, 1);
    assert_eq!(after_first.csr_builds_from_adjacency, 1);
    assert_eq!(
        after_second, after_first,
        "cache hit should not rebuild CSR or rerun PPR"
    );
}

#[test]
fn csr_cache_entrypoint_uses_prebuilt_csr_without_adjacency_rebuild() {
    let _guard = counter_guard();
    clear_scoped_ppr_cache();
    reset_scoped_ppr_execution_counters();
    let adjacency = weighted_fixture();
    let seeds = HashMap::from([("a".to_string(), 1.0)]);
    let csr = CsrGraph::from_weighted_adjacency(&adjacency, seeds.keys());

    let scores =
        cached_csr_personalized_pagerank("ppr-csr-direct", 7, &csr, &seeds, 0.15, 1e-6, 5000)
            .expect("PPR cache key should be content-addressable");
    let counters = scoped_ppr_execution_counters();

    assert!(!scores.is_empty());
    assert_eq!(counters.csr_ppr_runs, 1);
    assert_eq!(
        counters.csr_builds_from_adjacency, 0,
        "CSR-native hot path should not rebuild legacy adjacency maps"
    );
}

#[test]
fn epistemic_shadow_ppr_uses_csr_without_adjacency_rebuild() {
    let _guard = counter_guard();
    clear_scoped_ppr_cache();
    reset_scoped_ppr_execution_counters();

    let mut store = InMemoryGraphStore::new();
    for id in ["a", "b", "c"] {
        store
            .upsert_node(NodeRecord::new(id, ["Claim"], json!({ "content": id })))
            .unwrap();
    }
    let batch = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    rustyred_thg_core::structural_epistemic_pass(
        &mut store,
        rustyred_thg_core::epistemic::StructuralEpistemicInput {
            batch_node_ids: batch,
            candidate_pairs: Vec::new(),
            explicit_relations: vec![
                rustyred_thg_core::epistemic::EpistemicRelationInput {
                    from_content_id: "a".to_string(),
                    to_content_id: "b".to_string(),
                    kind: rustyred_thg_core::epistemic::EpistemicRelationKind::Supports,
                    confidence: 1.0,
                    evidence: "fixture".to_string(),
                    source_kind: rustyred_thg_core::epistemic::EpistemicSourceKind::Structural,
                    score: None,
                    model_id: None,
                    calibration_version: None,
                    feature_version: None,
                    connection_features: None,
                    canonical_edge_type: None,
                },
                rustyred_thg_core::epistemic::EpistemicRelationInput {
                    from_content_id: "b".to_string(),
                    to_content_id: "c".to_string(),
                    kind: rustyred_thg_core::epistemic::EpistemicRelationKind::Supports,
                    confidence: 1.0,
                    evidence: "fixture".to_string(),
                    source_kind: rustyred_thg_core::epistemic::EpistemicSourceKind::Structural,
                    score: None,
                    model_id: None,
                    calibration_version: None,
                    feature_version: None,
                    connection_features: None,
                    canonical_edge_type: None,
                },
            ],
            config: rustyred_thg_core::epistemic::StructuralEpistemicConfig::default(),
        },
    )
    .unwrap();

    let ranking = epistemic_shadow_ppr(
        &store,
        &HashMap::from([("a".to_string(), 1.0)]),
        10,
        0.15,
        1e-6,
        5000,
    )
    .expect("shadow PPR cache key should be content-addressable");
    let counters = scoped_ppr_execution_counters();

    assert!(!ranking.is_empty());
    assert_eq!(counters.csr_ppr_runs, 1);
    assert_eq!(
        counters.csr_builds_from_adjacency, 0,
        "epistemic shadow PPR should build CSR directly from shadow edges"
    );
}
