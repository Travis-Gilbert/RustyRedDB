use rustyred_thg_core::{
    IndexBackend, IndexBuildStatus, IndexCreatedBy, IndexEvalReceipt, IndexKind, IndexManifest,
    IndexRegistry, IndexScope,
};

fn lifecycle_manifest() -> IndexManifest {
    IndexManifest::new(
        "idx:web-page",
        "Web page",
        IndexKind::FullText,
        IndexBackend::Tantivy,
        IndexScope::Tenant,
        "Page",
        IndexCreatedBy::System,
    )
    .expect("manifest")
}

#[test]
fn legacy_shadow_transition_populates_activation_identity() {
    let mut manifest = lifecycle_manifest();
    manifest.mark_shadow(7).expect("shadow transition");

    assert_eq!(manifest.generation_id.as_deref(), Some("idx:web-page"));
    assert_eq!(manifest.graph_watermark.as_deref(), Some("graph:7"));
}

#[test]
fn shadow_eval_refuses_a_missing_manifest_watermark() {
    let mut manifest = lifecycle_manifest();
    manifest.build_status = IndexBuildStatus::Shadow;
    manifest.generation_id = Some("generation-7".to_string());
    manifest.graph_watermark = None;
    let receipt = IndexEvalReceipt {
        receipt_id: "eval-7".to_string(),
        generation_id: "generation-7".to_string(),
        manifest_id: manifest.id.clone(),
        graph_watermark: "graph:7".to_string(),
        passed: true,
        recall_at_k: None,
        ndcg_at_10: None,
        notes: None,
    };

    let error = manifest
        .attach_eval_receipt(receipt)
        .expect_err("an unbound watermark must not promote");
    assert_eq!(error.code, "index_watermark_missing");
}

#[test]
fn manifest_lifecycle_updates_hash_and_retirement_reason() {
    let mut manifest = IndexManifest::new(
        "idx:artifact-card",
        "Artifact card",
        IndexKind::Covering,
        IndexBackend::RustyredCore,
        IndexScope::Project,
        "ContextArtifact",
        IndexCreatedBy::Manual,
    )
    .expect("manifest should be content-addressable")
    .with_target_properties(["scope", "artifact_type", "updated_at"])
    .expect("target properties should update the manifest hash")
    .with_covering_fields(["id", "title", "status", "outcome"])
    .expect("covering fields should update the manifest hash");

    manifest.validate().unwrap();
    let proposed_hash = manifest.state_hash.clone();

    manifest
        .activate(7)
        .expect("activation should update the manifest hash");
    assert_eq!(manifest.build_status, IndexBuildStatus::Active);
    assert_eq!(manifest.graph_version, 7);
    assert_ne!(manifest.state_hash, proposed_hash);

    manifest
        .record_hit(12.0, 30.0)
        .expect("hit accounting should update the manifest hash");
    assert_eq!(manifest.hit_count, 1);
    assert_eq!(manifest.avg_latency_saved_ms, 30.0);

    manifest
        .retire("covered by wider project artifact index")
        .expect("retirement should update the manifest hash");
    assert_eq!(manifest.build_status, IndexBuildStatus::Retired);
    assert_eq!(
        manifest.retirement_reason.as_deref(),
        Some("covered by wider project artifact index")
    );
}

#[test]
fn registry_registers_lists_and_retires_manifests() {
    let manifest = IndexManifest::new(
        "idx:run-list",
        "Run list",
        IndexKind::Composite,
        IndexBackend::RustyredCore,
        IndexScope::Tenant,
        "HarnessRun",
        IndexCreatedBy::Migration,
    )
    .expect("manifest should be content-addressable")
    .with_target_properties(["scope", "run_status", "updated_at"])
    .expect("target properties should update the manifest hash");
    let mut registry = IndexRegistry::new();

    registry.register_manifest(manifest.clone()).unwrap();
    assert_eq!(registry.list_manifests().len(), 1);
    assert_eq!(
        registry.get_manifest("idx:run-list").unwrap().name,
        "Run list"
    );

    let duplicate = registry.register_manifest(manifest);
    assert_eq!(duplicate.unwrap_err().code, "index_manifest_exists");

    registry
        .retire_manifest("idx:run-list", "replaced by covering run-card index")
        .unwrap();
    let retired = registry.get_manifest("idx:run-list").unwrap();
    assert_eq!(retired.build_status, IndexBuildStatus::Retired);
}
