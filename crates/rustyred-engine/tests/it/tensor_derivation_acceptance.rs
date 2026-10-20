use rustyred_thg_core::{
    content_hash_bytes, get_derivation, publish_tensor_block, put_assumption, AssumptionRecord,
    DerivationKind, DiskObjectStore, GraphMutation, GraphMutationBatch, InMemoryGraphStore,
    InMemoryTensorBlockPayloadStore, NodeRecord, RedCoreGraphStore, TensorBlockAssumptionRefs,
    TensorBlockDerivationStore, TensorBlockLineageInput, TensorBlockManifestLineage,
    TensorBlockPublicationInput, TENSOR_BLOCK_POINTER_EDGE, TENSOR_BLOCK_POINTER_LABEL,
    TENSOR_BLOCK_VALUE_LABEL, VALUE_LABEL,
};

fn assumptions(graph: &mut InMemoryGraphStore) -> TensorBlockAssumptionRefs {
    for (id, name) in [
        ("assumption:commit", "graph commit"),
        ("assumption:schema", "tensor schema"),
        ("assumption:model", "embedding model"),
        ("assumption:calibration", "calibration epoch"),
    ] {
        put_assumption(graph, AssumptionRecord::active(id, name)).expect("seed assumption");
    }
    TensorBlockAssumptionRefs {
        graph_commit: "assumption:commit".to_string(),
        tensor_schema: "assumption:schema".to_string(),
        embedding_model: "assumption:model".to_string(),
        calibration_epoch: "assumption:calibration".to_string(),
    }
}

#[test]
fn durable_bytes_precede_an_atomic_functional_generation_pointer() {
    let source_bytes = b"lineage source";
    let source_content_id = content_hash_bytes(source_bytes);
    let manifest_bytes = b"canonical manifest".to_vec();
    let payload_bytes = b"safetensors payload".to_vec();
    let manifest_content_id = content_hash_bytes(&manifest_bytes);
    let payload_content_id = content_hash_bytes(&payload_bytes);
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "value:lineage",
            [VALUE_LABEL],
            serde_json::json!({"content_id": source_content_id}),
        ))
        .expect("seed lineage Value");
    let assumptions = assumptions(&mut graph);
    let root =
        std::env::temp_dir().join(format!("theorem-tensor-derivation-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let payload_store = DiskObjectStore::open(&root).expect("open durable payload store");

    let receipt = publish_tensor_block(
        &mut graph,
        &payload_store,
        TensorBlockPublicationInput {
            pointer_id: "memtensor-generation:Travis-Gilbert:active".to_string(),
            manifest_content_id: manifest_content_id.clone(),
            manifest_bytes: manifest_bytes.clone(),
            payload_content_id: payload_content_id.clone(),
            payload_bytes: payload_bytes.clone(),
            manifest_lineage: vec![TensorBlockManifestLineage {
                content_id: source_content_id.clone(),
                kind: "explicit_block".to_string(),
                relation: "fixture/source".to_string(),
            }],
            lineage_inputs: vec![TensorBlockLineageInput {
                content_id: source_content_id,
                value_ref: "value:lineage".to_string(),
            }],
            assumptions: assumptions.clone(),
            operator_id: "operator:tensor-block-compiler:fixture".to_string(),
            environment_version: "fixture-v1".to_string(),
            annotation: Default::default(),
        },
    )
    .expect("atomic TensorBlock publication");

    drop(payload_store);
    let reopened = DiskObjectStore::open(&root).expect("reopen durable payload store");
    assert_eq!(
        reopened
            .get_document_bytes(&manifest_content_id)
            .expect("read manifest")
            .expect("manifest is durable"),
        manifest_bytes
    );
    assert_eq!(
        reopened
            .get_document_bytes(&payload_content_id)
            .expect("read payload")
            .expect("payload is durable"),
        payload_bytes
    );
    let value = graph.get_node(&receipt.value_id).expect("output Value");
    assert!(value
        .labels
        .iter()
        .any(|label| label == TENSOR_BLOCK_VALUE_LABEL));
    let derivation = get_derivation(&graph, &receipt.derivation_id).expect("functional derivation");
    assert_eq!(derivation.spec.kind, DerivationKind::Functional);
    assert_eq!(derivation.spec.assumption_refs, assumptions.all());
    assert!(graph
        .neighbors(
            rustyred_thg_core::NeighborQuery::out(&receipt.pointer_id)
                .with_edge_type(TENSOR_BLOCK_POINTER_EDGE),
        )
        .iter()
        .any(|hit| hit.node_id == receipt.value_id));
    assert!(graph
        .get_node(&receipt.pointer_id)
        .expect("generation pointer")
        .labels
        .iter()
        .any(|label| label == TENSOR_BLOCK_POINTER_LABEL));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn generation_pointer_replaces_its_target_on_a_later_compilation() {
    let mut graph = InMemoryGraphStore::new();
    for (value_ref, content_id) in [
        ("value:first", content_hash_bytes(b"first lineage")),
        ("value:second", content_hash_bytes(b"second lineage")),
    ] {
        graph
            .upsert_node(NodeRecord::new(
                value_ref,
                [VALUE_LABEL],
                serde_json::json!({"content_id": content_id}),
            ))
            .expect("seed lineage Value");
    }
    let assumptions = assumptions(&mut graph);
    let payloads = InMemoryTensorBlockPayloadStore::new();
    let pointer_id = "memtensor-generation:Travis-Gilbert:active";

    let first = publish_fixture(
        &mut graph,
        &payloads,
        pointer_id,
        "value:first",
        content_hash_bytes(b"first lineage"),
        b"first manifest",
        b"first payload",
        assumptions.clone(),
    )
    .expect("first generation publishes");
    let second = publish_fixture(
        &mut graph,
        &payloads,
        pointer_id,
        "value:second",
        content_hash_bytes(b"second lineage"),
        b"second manifest",
        b"second payload",
        assumptions,
    )
    .expect("second generation publishes");

    assert_ne!(first.value_id, second.value_id);
    let current = graph
        .neighbors(
            rustyred_thg_core::NeighborQuery::out(pointer_id)
                .with_edge_type(TENSOR_BLOCK_POINTER_EDGE),
        )
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(current.len(), 1, "a generation pointer has one target");
    assert_eq!(current[0].node_id, second.value_id);
}

#[test]
fn tensor_block_batches_reject_deletes_on_every_backend() {
    let mut in_memory = InMemoryGraphStore::new();
    assert_tensor_delete_rejected(
        &mut in_memory,
        GraphMutation::NodeDelete("value:stale".to_string()),
    );
    assert_tensor_delete_rejected(
        &mut in_memory,
        GraphMutation::EdgeDelete("edge:stale".to_string()),
    );

    let mut redcore = RedCoreGraphStore::memory();
    assert_tensor_delete_rejected(
        &mut redcore,
        GraphMutation::NodeDelete("value:stale".to_string()),
    );
    assert_tensor_delete_rejected(
        &mut redcore,
        GraphMutation::EdgeDelete("edge:stale".to_string()),
    );
}

fn assert_tensor_delete_rejected(
    store: &mut impl TensorBlockDerivationStore,
    mutation: GraphMutation,
) {
    let error = store
        .commit_tensor_block_batch(GraphMutationBatch::new(vec![mutation]))
        .expect_err("TensorBlock publication must reject destructive batches");
    assert_eq!(error.code, "tensor_block_delete_unsupported");
}

fn publish_fixture(
    graph: &mut InMemoryGraphStore,
    payloads: &InMemoryTensorBlockPayloadStore,
    pointer_id: &str,
    value_ref: &str,
    source_content_id: String,
    manifest_bytes: &[u8],
    payload_bytes: &[u8],
    assumptions: TensorBlockAssumptionRefs,
) -> Result<rustyred_thg_core::TensorBlockPublicationReceipt, rustyred_thg_core::GraphStoreError> {
    let manifest_content_id = content_hash_bytes(manifest_bytes);
    let payload_content_id = content_hash_bytes(payload_bytes);
    publish_tensor_block(
        graph,
        payloads,
        TensorBlockPublicationInput {
            pointer_id: pointer_id.to_string(),
            manifest_content_id,
            manifest_bytes: manifest_bytes.to_vec(),
            payload_content_id,
            payload_bytes: payload_bytes.to_vec(),
            manifest_lineage: vec![TensorBlockManifestLineage {
                content_id: source_content_id.clone(),
                kind: "explicit_block".to_string(),
                relation: "fixture/source".to_string(),
            }],
            lineage_inputs: vec![TensorBlockLineageInput {
                content_id: source_content_id,
                value_ref: value_ref.to_string(),
            }],
            assumptions,
            operator_id: "operator:tensor-block-compiler:fixture".to_string(),
            environment_version: "fixture-v1".to_string(),
            annotation: Default::default(),
        },
    )
}
