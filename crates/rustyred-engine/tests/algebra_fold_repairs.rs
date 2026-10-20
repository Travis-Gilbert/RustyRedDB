use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use rustyred_thg_core::{
    ensure_assumption_for_epoch, AlgebraProjection, CalibratedEncoder, CalibrationEpoch,
    DeploymentId, EdgeRecord, FoldGraphStore, GraphRead, GraphWrite, InMemoryGraphStore,
    LeafValues, NodeQuery, NodeRecord, RowEmbedding, ASSUMPTION_LABEL, TOKEN_BIT_WIDTH,
};
use serde_json::json;

const DIM: usize = 128;

fn unit_rows(rows: usize, seed: u32) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..DIM)
                .map(|col| ((row as u32 * 31 + col as u32 * 17 + seed) % 97) as f32 / 97.0 - 0.5)
                .collect()
        })
        .collect()
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

fn fit_epoch() -> CalibrationEpoch {
    CalibrationEpoch::fit(
        &flatten(&unit_rows(8, 1)),
        DIM,
        TOKEN_BIT_WIDTH,
        "fixture-model",
        "2026-08",
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
    )
    .expect("fit epoch")
}

fn embedding_property(
    encoder: &mut CalibratedEncoder,
    node_id: &str,
    epoch: &CalibrationEpoch,
) -> serde_json::Value {
    let encoded = encoder
        .encode(&flatten(&unit_rows(4, 3)), DIM)
        .expect("encode");
    json!({
        "embedding_set_id": format!("mv:{node_id}"),
        "content_id": node_id,
        "model_id": epoch.model_id,
        "model_version": epoch.model_version,
        "epoch": epoch.id,
        "dim": epoch.dim,
        "bit_width": epoch.bit_width,
        "vector_count": encoded.vector_count,
        "packed_codes": STANDARD.encode(encoded.packed_codes),
        "scales": encoded.scales,
        "exact_object_ref": format!("cold://tenant/mv/{node_id}.f32"),
    })
}

fn open_store(label: &str) -> (tempfile::TempDir, FoldGraphStore) {
    let directory = tempfile::tempdir().expect("tempdir");
    let deployment = DeploymentId::new(label).expect("deployment");
    let store = FoldGraphStore::open(directory.path(), deployment).expect("store");
    (directory, store)
}

fn node_index(projection: &AlgebraProjection<'_>, id: &str) -> usize {
    projection
        .node_index(id)
        .unwrap_or_else(|| panic!("missing node {id}"))
        .0 as usize
}

fn f32_leaf(projection: &AlgebraProjection<'_>, key: &str, index: usize) -> f32 {
    match projection.leaf(key) {
        Some(LeafValues::F32(values)) => values[index],
        other => panic!("expected f32 leaf {key}, got {other:?}"),
    }
}

fn u32_leaf(projection: &AlgebraProjection<'_>, key: &str, index: usize) -> u32 {
    match projection.leaf(key) {
        Some(LeafValues::U32(values)) => values[index],
        other => panic!("expected u32 leaf {key}, got {other:?}"),
    }
}

fn schema_failed(error: &rustyred_thg_core::GraphStoreError) -> bool {
    error.code == "algebra_fold_validation_failed"
        && error.message.to_ascii_lowercase().contains("schema")
}

#[test]
fn replacement_omission_clears_valuation_leaves_and_embedding_state() {
    let (_directory, mut store) = open_store("algebra-replacement-omission");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:replace",
            ["Memory"],
            json!({
                "_valuation": {
                    "activation/score": 0.75,
                    "retrieval/recency": 0.5,
                },
                "_embedding": embedding_property(&mut encoder, "memory:replace", &epoch),
            }),
        ),
    )
    .expect("initial memory");

    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:replace",
            ["Memory"],
            json!({"_valuation": {"activation/score": 0.25}}),
        ),
    )
    .expect("replacement");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");
    {
        let projection = store.algebra().project().expect("project");
        let index = node_index(&projection, "memory:replace");
        assert_eq!(f32_leaf(&projection, "activation/score", index), 0.25);
        assert_eq!(f32_leaf(&projection, "retrieval/recency", index), 0.0);
        assert_eq!(u32_leaf(&projection, "multivector/count", index), 0);
        assert!(matches!(
            projection.embedding(index),
            Some(RowEmbedding::Pending)
        ));
        assert_eq!(
            projection.tokens.offsets[index],
            projection.tokens.offsets[index + 1]
        );
    }

    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("memory:replace", ["Memory"], json!({})),
    )
    .expect("replacement omitting valuation");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("rematerialize");
    let projection = store.algebra().project().expect("project");
    let index = node_index(&projection, "memory:replace");
    assert_eq!(f32_leaf(&projection, "activation/score", index), 0.0);
}

#[test]
fn active_epoch_deactivation_demotes_and_discards_existing_codes() {
    let (_directory, mut store) = open_store("algebra-epoch-deactivation");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    let mut encoder = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:epoch",
            ["Memory"],
            json!({"_embedding": embedding_property(&mut encoder, "memory:epoch", &epoch)}),
        ),
    )
    .expect("embedded memory");

    let mut inactive = epoch.clone();
    inactive.active = false;
    GraphWrite::upsert_node(&mut store, inactive.to_node()).expect("deactivate epoch");
    assert!(store.algebra().epoch().is_none());
    assert!(store.algebra().pending_embedding_count() >= 1);

    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("reactivate epoch");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");
    let projection = store.algebra().project().expect("project");
    let index = node_index(&projection, "memory:epoch");
    assert!(matches!(
        projection.embedding(index),
        Some(RowEmbedding::Pending)
    ));
    assert_eq!(u32_leaf(&projection, "multivector/count", index), 0);
    assert!(projection.tokens.slot_node.is_empty());
}

#[test]
fn canonical_epoch_identity_refuses_aliases_and_same_id_body_mutation() {
    let (_directory, mut store) = open_store("algebra-canonical-epoch");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");

    let mut mutated = epoch.to_node();
    mutated.properties["model_version"] = json!("mutated-under-same-id");
    let error =
        GraphWrite::upsert_node(&mut store, mutated).expect_err("same-id body mutation must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(error.message.contains("canonical content id"));

    let mut aliased = epoch.to_node();
    aliased.id = "calibration-epoch:alias".into();
    let error =
        GraphWrite::upsert_node(&mut store, aliased).expect_err("non-canonical node id must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(error.message.contains("canonical node id"));

    assert_eq!(store.algebra().epoch(), Some(&epoch));
    assert_eq!(
        store
            .get_node(&epoch.node_id)
            .and_then(|node| node.properties.get("model_version"))
            .and_then(serde_json::Value::as_str),
        Some(epoch.model_version.as_str())
    );
}

#[test]
fn oversized_numeric_inputs_are_schema_refusals_without_wrapping() {
    let (_directory, mut store) = open_store("algebra-numeric-ranges");
    let f32_error = GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "memory:overflow",
            ["Memory"],
            json!({"_valuation": {"activation/score": f64::MAX}}),
        ),
    )
    .expect_err("f64 outside f32 range must fail");
    assert!(schema_failed(&f32_error), "{f32_error:?}");
    assert!(f32_error.message.contains("finite f32 range"));

    let u32_error = GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new(
            "claim:overflow",
            ["Claim"],
            json!({"_valuation": {"belief/status_code": u64::from(u32::MAX) + 1}}),
        ),
    )
    .expect_err("u64 outside u32 range must fail");
    assert!(schema_failed(&u32_error), "{u32_error:?}");
    assert!(u32_error.message.contains("exceeds u32"));

    let epoch = fit_epoch();
    let mut oversized_epoch = epoch.to_node();
    oversized_epoch.properties["bit_width"] = json!(260_u64);
    let epoch_error = GraphWrite::upsert_node(&mut store, oversized_epoch)
        .expect_err("u64 outside u8 range must fail");
    assert!(schema_failed(&epoch_error), "{epoch_error:?}");
    assert!(epoch_error.message.contains("exceeds u8"));

    assert!(store.get_node("memory:overflow").is_none());
    assert!(store.get_node("claim:overflow").is_none());
}

#[test]
fn csr_payload_uses_node_index_order_and_last_edge_clears_counts() {
    let (_directory, mut store) = open_store("algebra-csr-order");
    let epoch = fit_epoch();
    GraphWrite::upsert_node(&mut store, epoch.to_node()).expect("epoch");
    for id in ["memory:z", "memory:a", "target:z", "target:a"] {
        GraphWrite::upsert_node(&mut store, NodeRecord::new(id, ["Memory"], json!({})))
            .expect("node");
    }
    GraphWrite::upsert_edge(
        &mut store,
        EdgeRecord::new("edge:z", "memory:z", "supports", "target:z", json!({})),
    )
    .expect("z edge");
    GraphWrite::upsert_edge(
        &mut store,
        EdgeRecord::new("edge:a", "memory:a", "supports", "target:a", json!({})),
    )
    .expect("a edge");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("materialize");

    {
        let projection = store.algebra().project().expect("project");
        let z = node_index(&projection, "memory:z");
        let a = node_index(&projection, "memory:a");
        let target_z = node_index(&projection, "target:z") as u64;
        let target_a = node_index(&projection, "target:a") as u64;
        let branch = projection
            .branches
            .get("supports")
            .expect("supports branch");
        assert_eq!(branch.counts[z], 1);
        assert_eq!(branch.counts[a], 1);
        let z_offset = branch.counts[..z]
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        let a_offset = branch.counts[..a]
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        assert_eq!(branch.targets[z_offset], target_z);
        assert_eq!(branch.edge_ids[z_offset], "edge:z");
        assert_eq!(branch.targets[a_offset], target_a);
        assert_eq!(branch.edge_ids[a_offset], "edge:a");
    }

    store.delete_edge("edge:z").expect("delete z");
    store.delete_edge("edge:a").expect("delete a");
    store
        .algebra_runtime_mut()
        .inner_mut()
        .materialize_tokens()
        .expect("rematerialize after delete");
    let projection = store.algebra().project().expect("project after delete");
    assert!(!projection.branches.contains_key("supports"));
    match projection.leaf("branch/supports/count") {
        Some(LeafValues::U32(values)) => assert!(values.iter().all(|count| *count == 0)),
        other => panic!("expected retained zero count leaf, got {other:?}"),
    }
}

#[test]
fn invalid_edge_type_is_a_typed_refusal() {
    let (_directory, mut store) = open_store("algebra-invalid-edge-type");
    for id in ["node:a", "node:b"] {
        GraphWrite::upsert_node(&mut store, NodeRecord::new(id, ["Memory"], json!({})))
            .expect("node");
    }
    let error = GraphWrite::upsert_edge(
        &mut store,
        EdgeRecord::new("edge:bad", "node:a", "..", "node:b", json!({})),
    )
    .expect_err("invalid branch leaf path must fail");
    assert!(schema_failed(&error), "{error:?}");
    assert!(error.message.contains("invalid edge type"));
    assert!(store.get_edge("edge:bad").is_none());
}

#[test]
fn assumption_family_deactivation_scans_beyond_default_query_limit() {
    let mut graph = InMemoryGraphStore::new();
    for index in 0..101 {
        GraphWrite::upsert_node(
            &mut graph,
            NodeRecord::new(
                format!("assumption:old:{index:03}"),
                [ASSUMPTION_LABEL],
                json!({
                    "name": format!("old-{index}"),
                    "active": true,
                    "family": "calibration-epoch",
                    "token_region": "algebra/tokens",
                }),
            ),
        )
        .expect("old assumption");
    }

    let keep = ensure_assumption_for_epoch(&mut graph, "epoch:new").expect("new assumption");
    let assumptions = graph.query_nodes(NodeQuery::label(ASSUMPTION_LABEL).with_limit(usize::MAX));
    assert_eq!(assumptions.len(), 102);
    for assumption in assumptions {
        let active = assumption
            .properties
            .get("active")
            .and_then(serde_json::Value::as_bool)
            .expect("active");
        assert_eq!(active, assumption.id == keep, "{}", assumption.id);
    }
}

#[cfg(feature = "graphblas")]
#[test]
fn activation_spread_follows_the_forward_edge_direction() {
    let (_directory, mut store) = open_store("algebra-forward-spread");
    for id in ["node:source", "node:target"] {
        GraphWrite::upsert_node(&mut store, NodeRecord::new(id, ["Memory"], json!({})))
            .expect("node");
    }
    GraphWrite::upsert_edge(
        &mut store,
        EdgeRecord::new(
            "edge:forward",
            "node:source",
            "supports",
            "node:target",
            json!({}),
        ),
    )
    .expect("edge");

    let spread = store
        .algebra_runtime_mut()
        .inner_mut()
        .activation_spread("supports", &[1.0, 0.0], 1)
        .expect("spread");
    assert_eq!(spread, vec![0.0, 1.0]);
}
