use std::collections::BTreeMap;

use rustyred_thg_core::{
    reconstruct_temporal_context, ColdFragment, ColdFragmentSkipMetadata, ColdSkipIndexDefinition,
    GraphStructuralIndexDefinition, GraphStructuralIndexKind, GraphStructuralIndexReceipt,
    IndexBackend, IndexCreatedBy, IndexKind, IndexScope, ResolvedRow, ScalarBound, ScalarValue,
    SpatialDesignation, SpatialIndexDefinition, TemporalFact, TemporalIndexDefinition,
};
use serde_json::json;

fn row(id: &str, ts: i64) -> ResolvedRow {
    ResolvedRow::new(
        "memory",
        id,
        BTreeMap::from([("t_ms".to_string(), ScalarValue::I64(ts))]),
    )
}

#[test]
fn graph_structural_manifest_records_graph_version_and_staleness() {
    let definition = GraphStructuralIndexDefinition::new(
        "structural:ppr:claims",
        GraphStructuralIndexKind::PersonalizedPagerank,
        "Claim",
        42,
    )
    .with_edge_type("supports")
    .with_seed_properties(["claim_id"]);

    let manifest = definition
        .to_manifest(IndexScope::Project, IndexCreatedBy::System)
        .expect("graph structural manifest should be content-addressable");
    let receipt = GraphStructuralIndexReceipt {
        manifest_id: manifest.id.clone(),
        graph_version: manifest.graph_version,
        indexed_node_count: 10,
        indexed_edge_count: 12,
        problem_count: 0,
    };

    assert_eq!(manifest.kind, IndexKind::GraphStructural);
    assert_eq!(manifest.backend, IndexBackend::RustyredCore);
    assert_eq!(manifest.graph_version, 42);
    assert_eq!(manifest.target_edge_type.as_deref(), Some("supports"));
    assert!(!receipt.stale_for_graph_version(42));
    assert!(receipt.stale_for_graph_version(43));
}

#[test]
fn temporal_index_reconstructs_past_context_without_losing_history() {
    let definition = TemporalIndexDefinition::new(
        "temporal:claims",
        "Claim",
        "t_valid",
        "t_invalid",
        "t_created",
    )
    .with_ttl_property("expires_at_ms");
    let manifest = definition
        .to_manifest(IndexScope::Project, IndexCreatedBy::System)
        .expect("temporal manifest should be content-addressable");
    assert_eq!(manifest.kind, IndexKind::Temporal);
    assert!(manifest
        .target_properties
        .contains(&"expires_at_ms".to_string()));

    let accepted = TemporalFact::new("fact:accepted", 100, json!({ "claim": "accepted" }));
    let stale = TemporalFact::new("fact:stale", 50, json!({ "claim": "old" })).invalidate(150, 200);
    let future = TemporalFact::new("fact:future", 300, json!({ "claim": "future" }));
    let slice = reconstruct_temporal_context(&[accepted, stale, future], 125);

    assert_eq!(slice.fact_ids, vec!["fact:accepted", "fact:stale"]);
    assert_eq!(slice.as_of_ms, 125);
}

#[test]
fn spatial_manifest_defaults_to_h3_and_keeps_s2_optional() {
    let definition = SpatialIndexDefinition::h3(
        "spatial:places:h3",
        SpatialDesignation {
            label: "Place".to_string(),
            lat_property: "lat".to_string(),
            lon_property: "lon".to_string(),
            resolution: 8,
        },
    );

    let manifest = definition
        .to_manifest(IndexScope::Project, IndexCreatedBy::System)
        .expect("spatial manifest should be content-addressable");

    assert_eq!(manifest.kind, IndexKind::Spatial);
    assert_eq!(manifest.backend, IndexBackend::H3);
    assert_eq!(manifest.target_properties, vec!["lat", "lon"]);
}

#[test]
fn cold_skip_metadata_has_no_false_negative_for_zone_map_pruning() {
    let fragment =
        ColdFragment::from_rows("frag:memory:1", "memory", &[row("m1", 10), row("m2", 20)]);
    let definition = ColdSkipIndexDefinition::new("cold:memory:t-ms", "memory", ["t_ms"]);
    let manifest = definition
        .to_manifest(IndexScope::Project, IndexCreatedBy::System)
        .expect("cold skip manifest should be content-addressable");
    let metadata = ColdFragmentSkipMetadata::from_fragment(&fragment);

    assert_eq!(manifest.kind, IndexKind::ColdSkip);
    assert_eq!(manifest.backend, IndexBackend::ColdFragment);
    assert!(metadata.excludes_range(
        "t_ms",
        &ScalarBound::Included(ScalarValue::I64(100)),
        &ScalarBound::Included(ScalarValue::I64(200)),
    ));
    assert!(metadata.no_false_negative_for_range(
        &fragment,
        "t_ms",
        ScalarBound::Included(ScalarValue::I64(100)),
        ScalarBound::Included(ScalarValue::I64(200)),
    ));
    assert!(!metadata.excludes_range(
        "t_ms",
        &ScalarBound::Included(ScalarValue::I64(15)),
        &ScalarBound::Included(ScalarValue::I64(25)),
    ));
}
