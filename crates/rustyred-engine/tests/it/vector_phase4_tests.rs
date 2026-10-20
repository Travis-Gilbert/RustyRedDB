use rustyred_thg_core::{
    filter_vector_candidates, vector_recall_against_exact, IndexBackend, IndexCreatedBy, IndexKind,
    IndexScope, ReceiptScope, VectorFilterPolicy, VectorIndexDefinition, VectorSearchBackend,
    VectorSearchCandidate,
};

fn scope(project: &str) -> ReceiptScope {
    ReceiptScope::from([
        ("tenant".to_string(), "Travis-Gilbert".to_string()),
        ("project".to_string(), project.to_string()),
    ])
}

#[test]
fn vector_definition_projects_to_first_class_manifest() {
    let definition = VectorIndexDefinition::new(
        "vector:context-atom:embedding",
        "ContextAtom",
        "embedding",
        384,
        VectorSearchBackend::ExactCosine,
    )
    .with_scope_fields(["tenant", "project"]);

    let manifest = definition
        .to_manifest(IndexScope::Project, IndexCreatedBy::System)
        .expect("vector manifest should be content-addressable");

    assert_eq!(manifest.id, "vector:context-atom:embedding");
    assert_eq!(manifest.kind, IndexKind::Vector);
    assert_eq!(manifest.backend, IndexBackend::RustyredCore);
    assert_eq!(manifest.target_properties, vec!["embedding"]);
    assert!(manifest.memory_bytes > 0);
}

#[test]
fn vector_filters_enforce_scope_policy_tombstone_ttl_and_metadata() {
    let policy = VectorFilterPolicy::scoped(scope("Theorem"))
        .require_labels(["ContextAtom"])
        .allow_trust_statuses(["trusted", "accepted"])
        .allow_freshness_statuses(["fresh"])
        .for_repo("Travis-Gilbert/Theorem")
        .for_user("travis");

    let accepted = VectorSearchCandidate::new("atom:accepted", 0.01, scope("Theorem"))
        .with_labels(["ContextAtom"])
        .with_repo("Travis-Gilbert/Theorem")
        .with_user("travis")
        .with_trust_status("trusted")
        .with_freshness_status("fresh");
    let wrong_scope = VectorSearchCandidate::new("atom:wrong-scope", 0.0, scope("Other"))
        .with_labels(["ContextAtom"])
        .with_repo("Travis-Gilbert/Theorem")
        .with_user("travis")
        .with_trust_status("trusted")
        .with_freshness_status("fresh");
    let mut denied = accepted.clone();
    denied.object_id = "atom:denied".to_string();
    denied.policy_allowed = false;
    let mut tombstone = accepted.clone();
    tombstone.object_id = "atom:tombstone".to_string();
    tombstone.tombstone = true;
    let mut expired = accepted.clone();
    expired.object_id = "atom:expired".to_string();
    expired.ttl_expired = true;

    let filtered = filter_vector_candidates(
        &[wrong_scope, denied, tombstone, expired, accepted.clone()],
        &policy,
        10,
    );

    assert_eq!(filtered, vec![accepted]);
}

#[test]
fn vector_filters_drop_non_finite_distances_before_ranking() {
    let policy = VectorFilterPolicy::scoped(scope("Theorem"));
    let accepted = VectorSearchCandidate::new("atom:accepted", 0.01, scope("Theorem"));
    let nan = VectorSearchCandidate::new("atom:nan", f32::NAN, scope("Theorem"));

    let filtered = filter_vector_candidates(&[nan, accepted.clone()], &policy, 10);

    assert_eq!(filtered, vec![accepted]);
}

#[test]
fn vector_recall_report_measures_turbovec_candidates_against_exact_oracle() {
    let exact = vec![
        VectorSearchCandidate::new("atom:a", 0.01, scope("Theorem")),
        VectorSearchCandidate::new("atom:b", 0.02, scope("Theorem")),
        VectorSearchCandidate::new("atom:c", 0.03, scope("Theorem")),
    ];
    let accelerated = vec![
        VectorSearchCandidate::new("atom:b", 0.02, scope("Theorem")),
        VectorSearchCandidate::new("atom:d", 0.04, scope("Theorem")),
        VectorSearchCandidate::new("atom:c", 0.03, scope("Theorem")),
    ];

    let report = vector_recall_against_exact(&exact, &accelerated, 3, 3);

    assert_eq!(report.exact_count, 3);
    assert_eq!(report.candidate_count, 3);
    assert_eq!(report.overlap_count, 2);
    assert_eq!(report.missing_object_ids, vec!["atom:a"]);
    assert!((report.recall - 0.666_666_7).abs() < 0.000_001);
}
