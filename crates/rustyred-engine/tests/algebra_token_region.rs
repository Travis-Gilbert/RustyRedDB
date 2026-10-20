//! Ungated Algebra token-region acceptance for SPEC-THEOREM-MEMTENSOR-ALGEBRA-1.1 D2–D4.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use rustyred_thg_core::{
    AlgebraFold, AlgebraValuationProfile, CalibratedEncoder, CalibrationEpoch,
    CheckpointRejectCode, DeploymentId, Fold, FoldGraphStore, FoldRuntime, FoldVersion, GraphWrite,
    NodeRecord, RestoreDisposition, RowEmbedding, TOKEN_BIT_WIDTH,
};
use serde_json::json;
use std::error::Error;
use theorem_oplog::PhysicalLog;

const DIM: usize = 128;

fn unit_rows(rows: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut vector = Vec::with_capacity(DIM);
        for col in 0..DIM {
            let raw = ((row as u32 * 31 + col as u32 * 17 + seed) % 97) as f32 / 97.0 - 0.5;
            vector.push(raw);
        }
        let norm = vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        out.push(vector.into_iter().map(|value| value / norm).collect());
    }
    out
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

fn open_store(label: &str) -> (tempfile::TempDir, FoldGraphStore, DeploymentId) {
    let dir = tempfile::tempdir().expect("tempdir");
    let deployment = DeploymentId::new(label).expect("deployment");
    let store = FoldGraphStore::open(dir.path(), deployment.clone()).expect("store");
    (dir, store, deployment)
}

fn fit_epoch() -> CalibrationEpoch {
    CalibrationEpoch::fit(
        &flatten(&unit_rows(8, 1)),
        DIM,
        TOKEN_BIT_WIDTH,
        "jina-embeddings-v4",
        "2026-05",
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
    )
    .expect("fit")
}

fn embedding_property(
    encoder: &mut CalibratedEncoder,
    node_id: &str,
    epoch: &CalibrationEpoch,
    rows: &[Vec<f32>],
) -> serde_json::Value {
    let encoded = encoder.encode(&flatten(rows), DIM).expect("encode");
    json!({
        "embedding_set_id": format!("mv:{node_id}"),
        "content_id": node_id,
        "model_id": epoch.model_id,
        "model_version": epoch.model_version,
        "epoch": epoch.id,
        "dim": epoch.dim,
        "bit_width": epoch.bit_width,
        "vector_count": encoded.vector_count,
        "packed_codes": STANDARD.encode(&encoded.packed_codes),
        "scales": encoded.scales,
        "exact_object_ref": format!("cold://tenant/mv/{node_id}.f32"),
    })
}

fn schema_failed(error: &rustyred_thg_core::GraphStoreError) -> bool {
    error.code == "algebra_fold_validation_failed"
        && (error.message.contains("SchemaViolation")
            || error.message.to_ascii_lowercase().contains("schema"))
}

#[test]
fn embedding_before_epoch_is_schema_violation() {
    let (_dir, mut store, _) = open_store("algebra-embed-before-epoch");
    let epoch = fit_epoch();
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    let property = embedding_property(&mut encoder, "memory:early", &epoch, &unit_rows(4, 3));
    let error = GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:early",
            ["Memory"],
            json!({
                "_valuation": { "activation/score": 1.0 },
                "_embedding": property
            }),
        ),
    )
    .expect_err("embedding before an active epoch must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(
        error.message.contains("before any active calibration epoch"),
        "{}",
        error.message
    );
}

#[test]
fn embedding_under_stale_epoch_is_schema_violation() {
    let (_dir, mut store, _) = open_store("algebra-stale-epoch");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    let mut property = embedding_property(&mut encoder, "memory:stale", &epoch, &unit_rows(4, 4));
    property["epoch"] = json!("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    let error = GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:stale",
            ["Memory"],
            json!({
                "_valuation": { "activation/score": 1.0 },
                "_embedding": property
            }),
        ),
    )
    .expect_err("stale epoch id must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(
        error.message.contains("stale calibration epoch"),
        "{}",
        error.message
    );
}

#[test]
fn packed_codes_wrong_length_is_schema_violation() {
    let (_dir, mut store, _) = open_store("algebra-bad-codes");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    let mut property = embedding_property(&mut encoder, "memory:len", &epoch, &unit_rows(4, 5));
    property["packed_codes"] = json!(STANDARD.encode(&[0u8; 8]));
    let error = GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:len",
            ["Memory"],
            json!({
                "_valuation": { "activation/score": 1.0 },
                "_embedding": property
            }),
        ),
    )
    .expect_err("wrong packed_codes length must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(
        error.message.contains("packed_codes has the wrong length"),
        "{}",
        error.message
    );
}

#[test]
fn epoch_with_legacy_turbovec_version_is_schema_violation() {
    let (_dir, mut store, _) = open_store("algebra-legacy-epoch");
    let mut node = fit_epoch().to_node();
    node.properties["turbovec_version"] = json!("0.8.0");
    let error = GraphWrite::upsert_node(&mut store, node).expect_err("0.8.0 epoch must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(
        error.message.contains("0.8.0"),
        "{}",
        error.message
    );
}

#[test]
fn new_active_epoch_demotes_prior_sets() {
    let (_dir, mut store, _) = open_store("algebra-epoch-demote");
    let first = fit_epoch();
    GraphWrite::upsert_node(&mut store, first.to_node()).expect("first epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&first).expect("encoder");
    for (id, seed) in [("memory:a", 6u32), ("memory:b", 7)] {
        let property = embedding_property(&mut encoder, id, &first, &unit_rows(4, seed));
        GraphWrite::upsert_node(
            &mut store,
            NodeRecord::new(
                id,
                ["Memory"],
                json!({
                    "_valuation": { "activation/score": 1.0 },
                    "_embedding": property
                }),
            ),
        )
        .expect("memory");
    }
    let second = CalibrationEpoch::fit(
        &flatten(&unit_rows(8, 2)),
        DIM,
        TOKEN_BIT_WIDTH,
        "jina-embeddings-v4",
        "2026-06",
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
    )
    .expect("second fit");
    GraphWrite::upsert_node(&mut store, second.to_node()).expect("second epoch");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");
    let projection = store.algebra().project().expect("project");
    for id in ["memory:a", "memory:b"] {
        let index = projection.node_index(id).expect("indexed").0 as usize;
        assert!(
            matches!(projection.embedding_state[index], RowEmbedding::Pending),
            "{id} must demote to Pending"
        );
    }
    assert!(projection.tokens.slot_node.is_empty());
    assert!(store.algebra().pending_embedding_count() >= 2);
}

#[test]
fn memory_upsert_materializes_matching_packed_codes() {
    let (_dir, mut store, _) = open_store("algebra-thirty-two");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    let rows = unit_rows(32, 9);
    let encoded = encoder.encode(&flatten(&rows), DIM).expect("encode");
    let property = json!({
        "embedding_set_id": "mv:memory:wide",
        "content_id": "memory:wide",
        "model_id": epoch.model_id,
        "model_version": epoch.model_version,
        "epoch": epoch.id,
        "dim": epoch.dim,
        "bit_width": epoch.bit_width,
        "vector_count": encoded.vector_count,
        "packed_codes": STANDARD.encode(&encoded.packed_codes),
        "scales": encoded.scales,
        "exact_object_ref": "cold://tenant/mv/memory-wide.f32",
    });
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:wide",
            ["Memory"],
            json!({
                "_valuation": { "activation/score": 0.5 },
                "_embedding": property
            }),
        ),
    )
    .expect("memory");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");
    let projection = store.algebra().project().expect("project");
    let index = projection.node_index("memory:wide").expect("indexed").0 as usize;
    match projection.leaf("multivector/count") {
        Some(rustyred_thg_core::LeafValues::U32(values)) => {
            assert_eq!(values[index], 32);
        }
        other => panic!("expected u32 count leaf, got {other:?}"),
    }
    let start = projection.tokens.offsets[index] as usize;
    let end = projection.tokens.offsets[index + 1] as usize;
    assert_eq!(end - start, 32);
    assert_eq!(
        &projection.tokens.index.packed_codes()[start * 64..end * 64],
        encoded.packed_codes.as_slice()
    );
}

#[test]
fn checkpoint_restore_preserves_packed_codes() {
    let (_dir, mut store, _) = open_store("algebra-checkpoint-codes");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    let property = embedding_property(&mut encoder, "memory:ckpt", &epoch, &unit_rows(8, 10));
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:ckpt",
            ["Memory"],
            json!({
                "_valuation": { "activation/score": 1.0 },
                "_embedding": property
            }),
        ),
    )
    .expect("memory");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");
    let before = store
        .algebra()
        .project()
        .expect("project")
        .tokens
        .index
        .packed_codes()
        .to_vec();
    let checkpoint = Fold::checkpoint(store.algebra()).expect("checkpoint");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .restore(&checkpoint)
        .expect("restore");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize after restore");
    let after = store
        .algebra()
        .project()
        .expect("project")
        .tokens
        .index
        .packed_codes()
        .to_vec();
    assert_eq!(before, after);
}

#[test]
fn version_two_checkpoint_rebuilds_with_fold_version() -> Result<(), Box<dyn Error>> {
    let (dir, mut store, deployment) = open_store("algebra-v2-checkpoint");
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("memory:v2", ["Memory"], json!({})),
    )?;
    drop(store);
    let log = PhysicalLog::open(dir.path(), deployment)?;
    let mut source = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    source.apply_all(&log)?;
    let mut checkpoint = source.checkpoint(&log)?;
    checkpoint.fold_version = FoldVersion::new(2)?;
    let mut target = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    let RestoreDisposition::Rebuilt { rejection, .. } =
        target.restore_or_rebuild(&checkpoint, &log)?
    else {
        panic!("version-2 Algebra checkpoint must rebuild");
    };
    assert_eq!(rejection.code, CheckpointRejectCode::FoldVersion);
    Ok(())
}

#[test]
fn reachability_only_descriptor_rebuilds_with_fold_capabilities() -> Result<(), Box<dyn Error>> {
    let (dir, mut store, deployment) = open_store("algebra-reachability-descriptor");
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("memory:cap", ["Memory"], json!({})),
    )?;
    drop(store);
    let log = PhysicalLog::open(dir.path(), deployment)?;
    let mut source = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    source.apply_all(&log)?;
    let mut checkpoint = source.checkpoint(&log)?;
    checkpoint.fold_descriptor.valuation_profile = Some(AlgebraValuationProfile::reachability());
    let mut target = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    let RestoreDisposition::Rebuilt { rejection, .. } =
        target.restore_or_rebuild(&checkpoint, &log)?
    else {
        panic!("reachability-only descriptor must rebuild");
    };
    assert_eq!(rejection.code, CheckpointRejectCode::FoldCapabilities);
    Ok(())
}

#[test]
fn algebra_fold_version_is_three() {
    assert_eq!(AlgebraFold::default().version().get(), 3);
    assert_eq!(
        AlgebraFold::default()
            .descriptor()
            .valuation_profile
            .expect("v3 profile")
            .operations
            .iter()
            .map(|op| op.operation_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "reachability",
            "activation_spread",
            "competitive_activation",
            "memtensor_compose"
        ]
    );
}
