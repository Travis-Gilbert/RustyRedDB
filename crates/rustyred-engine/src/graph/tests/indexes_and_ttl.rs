#[test]
fn in_memory_stats_version_increments_per_upsert() {
    let mut store = InMemoryGraphStore::default();
    assert_eq!(store.stats().version, 0);

    let node_write = store
        .upsert_node(NodeRecord::new("node:a", ["Entity"], json!({})))
        .unwrap();
    assert_eq!(node_write.version, 1);
    assert_eq!(store.stats().version, 1);

    let node_update = store
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Entity"],
            json!({ "name": "A" }),
        ))
        .unwrap();
    assert_eq!(node_update.version, 2);
    assert_eq!(store.stats().version, 2);

    store
        .upsert_node(NodeRecord::new("node:b", ["Entity"], json!({})))
        .unwrap();
    let edge_write = store
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "RELATED",
            "node:b",
            json!({}),
        ))
        .unwrap();
    assert_eq!(edge_write.version, 4);
    assert_eq!(store.stats().version, 4);
}

#[test]
fn vector_designate_and_search_returns_nearest() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();

    for (id, vec) in [
        ("doc:1", vec![1.0_f32, 0.0, 0.0]),
        ("doc:2", vec![0.0, 1.0, 0.0]),
        ("doc:3", vec![0.7, 0.7, 0.0]),
    ] {
        store
            .upsert_node(NodeRecord::new(id, ["Doc"], json!({ "embedding": vec })))
            .unwrap();
    }

    let results = store
        .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 2)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "doc:1");
    assert!(
        results[0].1 < 0.01,
        "exact match should have near-zero distance"
    );
    assert_eq!(results[1].0, "doc:3");
}

#[test]
fn vector_index_shape_includes_index_only_points() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:index-only",
            ["Doc"],
            json!({"title": "index only"}),
        ))
        .unwrap();
    store
        .index_vector("doc:index-only", "embedding", &[1.0, 0.0])
        .unwrap();

    assert_eq!(
        store.vector_index_shape("Doc", "embedding"),
        Some(VectorIndexShape {
            dimension: 2,
            point_count: 1,
        })
    );

    store
        .upsert_node(NodeRecord::new(
            "doc:index-only",
            ["Doc"],
            json!({"title": "index point removed"}),
        ))
        .unwrap();
    assert_eq!(
        store.vector_index_shape("Doc", "embedding"),
        Some(VectorIndexShape {
            dimension: 2,
            point_count: 0,
        })
    );
}

#[test]
fn vector_index_removes_node_when_embedding_property_is_removed() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:1",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0, 0.0] }),
        ))
        .unwrap();
    assert_eq!(
        store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 1)
            .unwrap()[0]
            .0,
        "doc:1"
    );

    store
        .upsert_node(NodeRecord::new(
            "doc:1",
            ["Doc"],
            json!({ "title": "no vector" }),
        ))
        .unwrap();

    let results = store
        .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 1)
        .unwrap();
    assert!(
        results.is_empty(),
        "node without embedding should not remain in the vector index"
    );
}

#[test]
fn exact_vector_search_caps_heap_capacity_to_index_size() {
    let mut index = VectorIndex::new(2, 1);
    index.insert("doc:only", &[1.0, 0.0]);

    let receipt = index.exact_search_with_receipt(&[1.0, 0.0], usize::MAX);

    assert_eq!(receipt.hits.len(), 1);
    assert_eq!(receipt.hits[0].0, "doc:only");
    assert_eq!(receipt.work.retained_candidates, 1);
}

#[test]
fn packed_vector_exact_search_preserves_legacy_f32_rounding_order() {
    fn pack(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    let query = [-0.541_903_85_f32, -0.935_799_54];
    let mut index = VectorIndex::new(2, 1);
    index
        .insert_packed("a", pack(&[-0.542_219_46, -0.935_075_9]))
        .unwrap();
    index
        .insert_packed("b", pack(&[-0.541_323_07, -0.933_478_1]))
        .unwrap();

    let receipt = index.exact_search_with_receipt(&query, 2);
    assert_eq!(receipt.hits[0].0, "a");
    assert_eq!(receipt.hits[1].0, "b");
    assert_eq!(receipt.hits[0].1.to_bits(), receipt.hits[1].1.to_bits());
}

#[test]
fn vector_index_uses_turbovec_for_supported_embedding_dimension() {
    let mut index = VectorIndex::new(128, 1);
    let mut alpha = vec![0.0_f32; 128];
    alpha[0] = 1.0;
    let mut beta = vec![0.0_f32; 128];
    beta[1] = 1.0;

    index.insert("doc:alpha", &alpha);
    index.insert("doc:beta", &beta);
    for id in 0..62 {
        index.insert(&format!("doc:filler:{id}"), &beta);
    }

    assert_eq!(index.resident_len(), Some(64));
    assert_eq!(index.full_builds(), 1);
    let results = index.search_with_receipt(&alpha, 1).hits;
    assert_eq!(results[0].0, "doc:alpha");
    assert!(results[0].1 < 0.01);
}

#[test]
fn vector_receipt_reports_below_acceleration_minimum_under_threshold() {
    let mut index = VectorIndex::new(128, 7);
    let mut alpha = vec![0.0_f32; 128];
    alpha[0] = 1.0;
    for id in 0..63 {
        index.insert(&format!("doc:{id}"), &alpha);
    }
    let receipt = index.search_with_receipt(&alpha, 1);
    assert_eq!(receipt.work.backend, VectorSearchBackend::ExactCosine);
    assert_eq!(
        receipt.work.fallback_reason.as_deref(),
        Some("below_acceleration_minimum")
    );
    assert!(index.resident_len().is_none());
}

#[test]
fn vector_fallback_reason_serialization_preserves_single_reason_bytes() {
    let reason = r"turbovec_rebuild_failed:fixture;path\detail";

    assert_eq!(
        serialize_vector_fallback_reasons(vec![reason.to_string()]).as_deref(),
        Some(reason)
    );
    assert_eq!(
        serialize_vector_fallback_reasons(vec![
            r"zeta\detail".to_string(),
            "alpha;detail".to_string(),
            "alpha;detail".to_string(),
        ])
        .as_deref(),
        Some(r"alpha\;detail;zeta\\detail")
    );
}

#[test]
fn vector_search_dimension_mismatch_errors() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:1",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0, 0.0] }),
        ))
        .unwrap();

    let err = store
        .vector_search(Some("Doc"), "embedding", &[1.0, 0.0], 2)
        .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("dimension"),
        "error should mention dimension: {msg}"
    );
}

#[test]
fn vector_redesignate_rebuilds_index_for_new_dimension() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:old",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0, 0.0] }),
        ))
        .unwrap();
    assert_eq!(
        store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 1)
            .unwrap()[0]
            .0,
        "doc:old"
    );

    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();
    assert!(
        store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0], 1)
            .unwrap()
            .is_empty(),
        "old 3-d vector must not survive the 2-d designation rebuild"
    );
    store
        .upsert_node(NodeRecord::new(
            "doc:new",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0] }),
        ))
        .unwrap();
    assert_eq!(
        store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0], 1)
            .unwrap()[0]
            .0,
        "doc:new"
    );
}

#[test]
fn vector_auto_index_on_upsert() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();

    store
        .upsert_node(NodeRecord::new(
            "doc:a",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0] }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:b",
            ["Doc"],
            json!({ "embedding": [0.0, 1.0] }),
        ))
        .unwrap();

    let results = store
        .vector_search(Some("Doc"), "embedding", &[0.0, 1.0], 1)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc:b");
}

#[test]
fn vector_property_designation_migrates_deduplicates_and_preserves_neighbors() {
    let payloads = Arc::new(InMemoryTensorBlockPayloadStore::new());
    let payload_handle: Arc<dyn crate::TensorBlockPayloadStore> = payloads.clone();
    let mut store = InMemoryGraphStore::with_vector_payload_store(payload_handle);
    let vector = vec![0.25_f32, -0.0, 0.75];
    store
        .upsert_node(NodeRecord::new(
            "doc:vector-a",
            ["Doc"],
            json!({"embedding": vector}),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:vector-b",
            ["Doc"],
            json!({"embedding": [0.25, -0.0, 0.75]}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:vector-neighbor",
            "doc:vector-a",
            "RELATED",
            "doc:vector-b",
            json!({}),
        ))
        .unwrap();
    let before_neighbors = store.neighbors(NeighborQuery::out("doc:vector-a"));

    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();

    let first = store.get_node("doc:vector-a").unwrap();
    let second = store.get_node("doc:vector-b").unwrap();
    assert!(!first.properties["embedding"].is_array());
    assert!(!second.properties["embedding"].is_array());
    assert_eq!(
        VectorPropertyRef::from_value(&first.properties["embedding"]).unwrap(),
        VectorPropertyRef::from_value(&second.properties["embedding"]).unwrap()
    );
    assert_eq!(payloads.object_count().unwrap(), 1);
    let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&store).unwrap();
    let round_trip = read_vector_property(first, "embedding", payload_store.as_ref())
        .unwrap()
        .unwrap();
    assert_eq!(
        round_trip
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0.25_f32, -0.0, 0.75]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        store.neighbors(NeighborQuery::out("doc:vector-a")),
        before_neighbors
    );
    assert_eq!(
        store
            .vector_search(Some("Doc"), "embedding", &[0.25, -0.0, 0.75], 2)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["doc:vector-a".to_string(), "doc:vector-b".to_string()])
    );

    let versions = (
        store.get_node("doc:vector-a").unwrap().version,
        store.get_node("doc:vector-b").unwrap().version,
    );
    store
        .designate_vector_property("Doc", "embedding", 3)
        .unwrap();
    assert_eq!(
        versions,
        (
            store.get_node("doc:vector-a").unwrap().version,
            store.get_node("doc:vector-b").unwrap().version,
        ),
        "reference-form replay must be idempotent"
    );
    assert_eq!(payloads.object_count().unwrap(), 1);
}

#[test]
fn vector_property_reference_reduces_serialized_and_mirror_ram() {
    let payloads = Arc::new(InMemoryTensorBlockPayloadStore::new());
    let payload_handle: Arc<dyn crate::TensorBlockPayloadStore> = payloads.clone();
    let mut store = InMemoryGraphStore::with_vector_payload_store(payload_handle);
    let vector = (0..128).map(|lane| lane as f32 / 128.0).collect::<Vec<_>>();
    for index in 0..64 {
        store
            .upsert_node(NodeRecord::new(
                format!("doc:vector-ram-{index:03}"),
                ["Doc"],
                json!({"embedding": vector}),
            ))
            .unwrap();
    }
    let before_memory = store.memory_attribution();
    let before_size = serde_json::to_vec(store.get_node("doc:vector-ram-000").unwrap())
        .unwrap()
        .len();

    store
        .designate_vector_property("Doc", "embedding", 128)
        .unwrap();
    let after_memory = store.memory_attribution();
    let node = store.get_node("doc:vector-ram-000").unwrap();
    let after_size = serde_json::to_vec(node).unwrap().len();
    assert!(
        after_size < before_size / 2,
        "{after_size} vs {before_size}"
    );
    assert!(
        after_memory.accounted_capacity_lower_bound_bytes
            < before_memory.accounted_capacity_lower_bound_bytes,
        "{} vs {}",
        after_memory.accounted_capacity_lower_bound_bytes,
        before_memory.accounted_capacity_lower_bound_bytes,
    );
    assert_eq!(payloads.object_count().unwrap(), 1);
    assert_eq!(payloads.stored_bytes().unwrap(), 128 * 4);
    let vector_index = after_memory
        .vector_indexes
        .iter()
        .find(|index| index.label == "Doc" && index.property == "embedding")
        .expect("designated vector index attribution");
    assert_eq!(vector_index.point_count, 64);
    assert_eq!(vector_index.point_values_capacity_bytes, 64 * 128 * 4);
    assert!(after_memory
        .exclusions
        .contains(&"vector payload-store heap (reported by the owning payload store)"));
    assert!(store
        .snapshot()
        .nodes
        .iter()
        .all(|node| !node.properties["embedding"].is_array()));
}

#[test]
fn hybrid_search_blends_vector_and_graph() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();

    store
        .upsert_node(NodeRecord::new(
            "doc:a",
            ["Doc"],
            json!({ "embedding": [1.0, 0.0] }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:b",
            ["Doc"],
            json!({ "embedding": [0.9, 0.1] }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "doc:c",
            ["Doc"],
            json!({ "embedding": [0.0, 1.0] }),
        ))
        .unwrap();

    store
        .upsert_edge(EdgeRecord::new("e1", "doc:c", "LINKS", "doc:a", json!({})))
        .unwrap();

    let results = store
        .hybrid_search(
            Some("Doc"),
            "embedding",
            &[1.0, 0.0],
            3,
            &["doc:c".to_string()],
            2,
            0.8,
        )
        .unwrap();

    assert!(!results.is_empty());
    let ids: Vec<&str> = results.iter().map(|r| r.0.as_str()).collect();
    assert!(
        ids.contains(&"doc:a"),
        "doc:a should appear (near vector + reachable from seed)"
    );
}

#[test]
fn hybrid_search_can_penalize_contradicting_edges() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Claim", "embedding", 2)
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "claim:seed",
            ["Claim"],
            json!({ "embedding": [1.0, 0.0] }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "claim:support",
            ["Claim"],
            json!({ "embedding": [0.95, 0.05] }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "claim:against",
            ["Claim"],
            json!({ "embedding": [0.95, 0.05] }),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:support",
            "claim:seed",
            "SUPPORTS",
            "claim:support",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(
            EdgeRecord::new(
                "edge:against",
                "claim:seed",
                "CONTRADICTS",
                "claim:against",
                json!({}),
            )
            .with_confidence(1.0),
        )
        .unwrap();

    let results = store
        .hybrid_search(
            Some("Claim"),
            "embedding",
            &[1.0, 0.0],
            3,
            &["claim:seed".to_string()],
            1,
            0.9,
        )
        .unwrap();
    let support = results
        .iter()
        .find(|(id, _)| id == "claim:support")
        .map(|(_, score)| *score)
        .unwrap();
    let against = results
        .iter()
        .find(|(id, _)| id == "claim:against")
        .map(|(_, score)| *score)
        .unwrap();

    assert!(support > against);
}

#[test]
fn vector_designations_list() {
    let mut store = InMemoryGraphStore::default();
    store
        .designate_vector_property("Doc", "embedding", 384)
        .unwrap();
    store
        .designate_vector_property("Claim", "vector", 128)
        .unwrap();

    let desigs = store.vector_designations();
    assert_eq!(desigs.len(), 2);
    let labels: Vec<&str> = desigs.iter().map(|d| d.label.as_str()).collect();
    assert!(labels.contains(&"Doc"));
    assert!(labels.contains(&"Claim"));
}

#[test]
fn ordered_designation_tracks_numeric_property_updates() {
    let mut store = InMemoryGraphStore::default();
    store
        .upsert_node(NodeRecord::new("doc:b", ["Doc"], json!({ "score": 1.0 })))
        .unwrap();
    store
        .upsert_node(NodeRecord::new("doc:a", ["Doc"], json!({ "score": 1.0 })))
        .unwrap();
    store.designate_ordered_property("Doc", "score").unwrap();

    assert_eq!(
        store.ordered_range_by_score("Doc", "score", 0.0, 10.0, None),
        Ok(vec![("doc:a".to_string(), 1.0), ("doc:b".to_string(), 1.0)])
    );

    store
        .upsert_node(NodeRecord::new("doc:b", ["Doc"], json!({ "score": 9.0 })))
        .unwrap();

    assert_eq!(store.ordered_score("Doc", "score", "doc:b"), Some(9.0));
    assert_eq!(
        store.ordered_range_by_score("Doc", "score", 0.0, 10.0, None),
        Ok(vec![("doc:a".to_string(), 1.0), ("doc:b".to_string(), 9.0)])
    );
}

#[test]
fn redcore_persists_ordered_designation_through_reopen() {
    let data_dir = unique_test_dir("ordered-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new("role:1", ["Role"], json!({ "rank": 2.0 })))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("role:2", ["Role"], json!({ "rank": 1.0 })))
            .unwrap();
        store.designate_ordered_property("Role", "rank").unwrap();
        assert_eq!(
            store
                .ordered_range_by_score("Role", "rank", 0.0, 10.0, None)
                .unwrap(),
            vec![("role:2".to_string(), 1.0), ("role:1".to_string(), 2.0)]
        );
    }

    {
        let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
        assert_eq!(store.ordered_designations().len(), 1);
        assert_eq!(
            store
                .ordered_range_by_score("Role", "rank", 0.0, 10.0, None)
                .unwrap(),
            vec![("role:2".to_string(), 1.0), ("role:1".to_string(), 2.0)]
        );
    }

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn transient_ordered_set_does_not_advance_redcore_commit_log() {
    let mut store = RedCoreGraphStore::memory();
    let before = store.status();

    assert!(store
        .transient_ordered_zadd("frontier:test", b"url:b".to_vec(), 2.0)
        .unwrap());
    assert!(store
        .transient_ordered_zadd("frontier:test", b"url:a".to_vec(), 1.0)
        .unwrap());

    assert_eq!(store.transient_ordered_zcard("frontier:test"), 2);
    assert_eq!(
        store.transient_ordered_zpop_max("frontier:test"),
        Some((b"url:b".to_vec(), 2.0))
    );
    assert_eq!(store.status().last_txn_id, before.last_txn_id);
    assert_eq!(store.status().graph_version, before.graph_version);
}

#[test]
fn redcore_persists_vector_designation_through_reopen() {
    let data_dir = unique_test_dir("vec-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .designate_vector_property("Doc", "embedding", 3)
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "doc:1",
                ["Doc"],
                json!({ "embedding": [1.0, 0.0, 0.0] }),
            ))
            .unwrap();
        let results = store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc:1");
    }

    {
        let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
        let desigs = store.vector_designations();
        assert_eq!(desigs.len(), 1);
        assert_eq!(desigs[0].label, "Doc");
        assert_eq!(desigs[0].dimension, 3);

        let results = store
            .vector_search(Some("Doc"), "embedding", &[1.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc:1");
    }

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_persists_cleared_designations_through_reopen() {
    let data_dir = unique_test_dir("cleared-designations-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .designate_vector_property("Doc", "embedding", 3)
            .unwrap();
        store.designate_ordered_property("Doc", "score").unwrap();

        store.clear_vector_property("Doc", "embedding").unwrap();
        store.clear_ordered_property("Doc", "score").unwrap();

        assert!(store.vector_designations().is_empty());
        assert!(store.ordered_designations().is_empty());
        assert_eq!(
            store
                .clear_vector_property("", "embedding")
                .unwrap_err()
                .code,
            "invalid_vector_designation"
        );
        assert_eq!(
            store.clear_ordered_property("Doc", " ").unwrap_err().code,
            "invalid_ordered_designation"
        );
    }

    {
        let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
        assert!(store.vector_designations().is_empty());
        assert!(store.ordered_designations().is_empty());
    }

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_vector_property_journal_is_reference_only_and_reopens() {
    let data_dir = unique_test_dir("vector-property-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1,
        strict_acid: true,
    };
    let reference;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .designate_vector_property("Doc", "embedding", 3)
            .unwrap();
        let legacy = NodeRecord::new(
            "doc:vector-durable",
            ["Doc"],
            json!({"embedding": [1.0, -0.0, 0.5]}),
        );
        store.upsert_node(legacy).unwrap();
        store.snapshot_now().unwrap();

        let node = store.get_node("doc:vector-durable").unwrap().unwrap();
        assert!(!node.properties["embedding"].is_array());
        reference = VectorPropertyRef::from_value(&node.properties["embedding"]).unwrap();
        let snapshot = store.graph_snapshot();
        assert!(snapshot
            .nodes
            .iter()
            .all(|node| !node.properties["embedding"].is_array()));
        let journal = store.chunk_store_handle().expect("RedCore journal handle");
        let catalog_address = journal.read_root().unwrap().expect("committed catalog");
        let catalog =
            crate::chunk_store::validate_catalog_root(journal.as_ref(), catalog_address).unwrap();
        let state_address = catalog
            .get(crate::chunk_store::ROOT_NAME_REDCORE)
            .expect("RedCore state root");
        let state_bytes = journal
            .get(state_address)
            .unwrap()
            .expect("RedCore state chunk");
        let state = super::RedCoreJournalState::from_bytes(&state_bytes).unwrap();
        let persisted = super::hydrate_graph_snapshot_from_payload_index(
            journal.as_ref(),
            &state.logical_tree,
            &state.payload_tree,
            state.graph_version,
        )
        .unwrap();
        let persisted_node = persisted
            .nodes
            .iter()
            .find(|node| node.id == "doc:vector-durable")
            .expect("persisted vector node");
        assert_eq!(
            VectorPropertyRef::from_value(&persisted_node.properties["embedding"]).unwrap(),
            reference
        );
        assert!(
            !persisted_node.properties["embedding"].is_array(),
            "the journal-rooted payload record must contain only the typed reference"
        );
        let payload_store = crate::GraphVectorPayloadAccess::vector_payload_store(&store).unwrap();
        let vector = read_vector_property(&node, "embedding", payload_store.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(
            vector
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [1.0_f32, -0.0, 0.5]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(store
            .chunk_store_handle()
            .unwrap()
            .has(crate::Hash::parse_prefixed(&reference.content_id).unwrap())
            .unwrap());
        for legacy_file in [
            super::REDCORE_AOF_FILE,
            super::REDCORE_SNAPSHOT_FILE,
            super::REDCORE_PREVIOUS_SNAPSHOT_FILE,
        ] {
            assert!(
                !data_dir.join(legacy_file).exists(),
                "vector publication must not revive legacy persistence file {legacy_file}"
            );
        }
    }

    {
        let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
        let node = store.get_node("doc:vector-durable").unwrap().unwrap();
        assert_eq!(
            VectorPropertyRef::from_value(&node.properties["embedding"]).unwrap(),
            reference
        );
        assert_eq!(
            store
                .vector_search(Some("Doc"), "embedding", &[1.0, -0.0, 0.5], 1)
                .unwrap()[0]
                .0,
            "doc:vector-durable"
        );
    }
    std::fs::remove_dir_all(data_dir).ok();
}

#[cfg(feature = "redis-store")]
#[test]
fn redis_keyspace_uses_deployment_hash_tagged_graph_keys() {
    let deployment_id = crate::DeploymentId::new("deployment:redis-fixture").unwrap();
    let prefix = super::RedisGraphKeyspace::deployment_prefix("rrgdb", &deployment_id);
    let keyspace = super::RedisGraphKeyspace::new(prefix);

    assert_eq!(
        keyspace.prefix(),
        "rrgdb:{deployment:52a3d6aba8920b180bc177cd16462043fea15723336e7c7dbbf315b9793e4606}:graph:v1"
    );
    assert_eq!(
        keyspace.node("node:a"),
        "rrgdb:{deployment:52a3d6aba8920b180bc177cd16462043fea15723336e7c7dbbf315b9793e4606}:graph:v1:node:h6e6f64653a61"
    );
    assert_eq!(
        keyspace.out_adjacency("node:a", "LINKS"),
        "rrgdb:{deployment:52a3d6aba8920b180bc177cd16462043fea15723336e7c7dbbf315b9793e4606}:graph:v1:adj:out:h6e6f64653a61:h4c494e4b53"
    );
    assert_eq!(
        keyspace.property_value("path", "s:src/lib.rs"),
        "rrgdb:{tenant:pct_Tenant%20One%21}:graph:v1:property:h70617468:h733a7372632f6c69622e7273"
    );
    assert_eq!(
        keyspace.events(),
        "rrgdb:{tenant:pct_Tenant%20One%21}:graph:v1:events"
    );
}

#[test]
fn redcore_journal_catalog_records_graph_and_state_roots() {
    let data_dir = unique_test_dir("redcore-journal-catalog-format");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1,
        strict_acid: true,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let node = NodeRecord::new("n1", ["Doc"], json!({"title": "x"}));
    store.upsert_node(node).unwrap();
    store.snapshot_now().unwrap();
    let journal = store.journal.as_ref().unwrap();
    let catalog_address = journal.read_root().unwrap().unwrap();
    let catalog =
        crate::chunk_store::validate_catalog_root(journal.as_ref(), catalog_address).unwrap();
    assert!(catalog.get(crate::chunk_store::ROOT_NAME_GRAPH).is_some());
    assert!(catalog.get(crate::chunk_store::ROOT_NAME_REDCORE).is_some());
    assert!(
        journal
            .committed_root()
            .unwrap()
            .unwrap()
            .meta
            .graph_version
            > 0
    );
    assert!(!data_dir.join(super::REDCORE_MANIFEST_FILE).exists());

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_refuses_too_new_manifest() {
    let data_dir = unique_test_dir("redcore-manifest-too-new");
    std::fs::create_dir_all(&data_dir).unwrap();
    let manifest_path = data_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        r#"{
                "version": 9999,
                "graph_version": 0,
                "last_txn_id": 0,
                "snapshot_txn_id": 0,
                "durability": "none",
                "snapshot_file": "graph.snapshot.current",
                "aof_file": "graph.aof",
                "updated_at_unix_ms": 0
            }"#,
    )
    .unwrap();

    let result = RedCoreGraphStore::open(
        &data_dir,
        RedCoreOptions {
            durability: RedCoreDurability::None,
            snapshot_interval_writes: 0,
            strict_acid: false,
        },
    );

    let error = result.unwrap_err();
    assert_eq!(error.code, "redcore_format_too_new");
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_legacy_manifest_without_format_kind_loads() {
    let data_dir = unique_test_dir("redcore-manifest-legacy");
    std::fs::create_dir_all(&data_dir).unwrap();
    let manifest_path = data_dir.join("manifest.json");
    // Legacy: no format_kind, crate_version, created_at_unix_ms.
    std::fs::write(
        &manifest_path,
        r#"{
                "version": 1,
                "graph_version": 0,
                "last_txn_id": 0,
                "snapshot_txn_id": 0,
                "durability": "none",
                "snapshot_file": "graph.snapshot.current",
                "aof_file": "graph.aof",
                "updated_at_unix_ms": 0
            }"#,
    )
    .unwrap();

    let store = RedCoreGraphStore::open(
        &data_dir,
        RedCoreOptions {
            durability: RedCoreDurability::None,
            snapshot_interval_writes: 0,
            strict_acid: false,
        },
    )
    .unwrap();
    drop(store);
    std::fs::remove_dir_all(data_dir).ok();
}

#[cfg(feature = "redis-store")]
#[test]
fn redis_keyspace_hashes_deployments_and_encodes_dynamic_key_segments() {
    let deployment_id = crate::DeploymentId::new("deployment:redis-normalization").unwrap();
    assert_eq!(
        super::RedisGraphKeyspace::deployment_prefix("rrgdb", &deployment_id),
        "rrgdb:{deployment:1eefcdc0312af222707d1e740a9110edb996b959e1c5b26ad20a2724be777f86}:graph:v1"
    );

    let keyspace = super::RedisGraphKeyspace::new("rrgdb:{deployment:D}:graph:v1");
    assert_ne!(
        keyspace.out_adjacency("a:b", "c"),
        keyspace.out_adjacency("a", "b:c")
    );
    assert_ne!(
        keyspace.in_adjacency("a:b", "c"),
        keyspace.in_adjacency("a", "b:c")
    );
}

// ---- TTL primitive tests (TTL-03 design v2) -----------------------
//
// Each test constructs an InMemoryGraphStore and exercises one TTL
// contract from docs/plans/rustyred-thg-ttl-primitive/design.md.
// Naming follows the design's acceptance criteria.
//
// Note: `GraphStore`, `InMemoryGraphStore`, `NodeQuery`, `NeighborQuery`,
// `Direction`, `NodeRecord`, `EdgeRecord`, and `json` are already
// imported at the top of this test module by earlier test cohorts;
// only the TTL-specific symbols need explicit import here.
use super::{node_is_expired, node_ttl_expires_at_ms, now_ms, TTL_PROPERTY};

fn ttl_node(id: &str, expires_at_ms: i64) -> NodeRecord {
    NodeRecord::new(
        id,
        ["MemoryAtom"],
        json!({ TTL_PROPERTY: expires_at_ms, "title": id }),
    )
}

fn plain_node(id: &str) -> NodeRecord {
    NodeRecord::new(id, ["MemoryAtom"], json!({ "title": id }))
}

#[test]
fn graph_snapshot_reads_filter_expired_nodes_unless_explicitly_included() {
    let live = plain_node("memory:live");
    let expired = ttl_node("memory:expired", 1);
    let snapshot = GraphSnapshot {
        version: 3,
        nodes: vec![live, expired],
        edges: vec![EdgeRecord::new(
            "edge:expired",
            "memory:live",
            "RELATED_TO",
            "memory:expired",
            json!({}),
        )],
        historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: true,
    };

    assert!(snapshot.get_node("memory:expired").is_none());
    assert!(snapshot.get_node_record("memory:expired").is_none());
    assert_eq!(snapshot.live_node_records().len(), 1);
    assert_eq!(
        snapshot.node_ids_for_label("MemoryAtom"),
        vec!["memory:live".to_string()]
    );
    assert_eq!(
        snapshot.query_nodes(NodeQuery::label("MemoryAtom")).len(),
        1
    );
    assert_eq!(
        snapshot
            .query_nodes(NodeQuery::label("MemoryAtom").with_include_expired(true))
            .len(),
        2
    );
    assert!(snapshot
        .neighbors(NeighborQuery::out("memory:live"))
        .is_empty());
    assert_eq!(
        snapshot
            .neighbors(
                NeighborQuery::out("memory:live")
                    .with_edge_type("RELATED_TO")
                    .with_include_expired(true),
            )
            .len(),
        1
    );
}

fn memory_doc(id: &str, tenant_slug: &str, status: &str, updated_at: &str) -> NodeRecord {
    NodeRecord::new(
        format!("memory:document:{tenant_slug}:{id}"),
        ["HarnessMemory", "MemoryAtom", "MemoryDocument"],
        json!({
            "tenant_slug": tenant_slug,
            "doc_id": id,
            "kind": "note",
            "title": id,
            "content": format!("body {id}"),
            "status": status,
            "updated_at": updated_at,
            "created_at": updated_at,
        }),
    )
}

#[test]
fn memory_documents_by_updated_at_pages_from_tenant_status_index() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(memory_doc(
            "older-active",
            "Tenant-A",
            "active",
            "2026-06-27T00:00:01Z",
        ))
        .unwrap();
    store
        .upsert_node(memory_doc(
            "archived",
            "Tenant-A",
            "archived",
            "2026-06-27T00:00:02Z",
        ))
        .unwrap();
    store
        .upsert_node(memory_doc(
            "newer-active",
            "Tenant-A",
            "active",
            "2026-06-27T00:00:03Z",
        ))
        .unwrap();
    store
        .upsert_node(memory_doc(
            "deleted",
            "Tenant-A",
            "deleted",
            "2026-06-27T00:00:04Z",
        ))
        .unwrap();
    store
        .upsert_node(memory_doc(
            "other-tenant",
            "Tenant-B",
            "active",
            "2026-06-27T00:00:05Z",
        ))
        .unwrap();

    let active = store.memory_documents_by_updated_at(
        MemoryDocumentQuery::tenant("Tenant-A")
            .with_status("active")
            .with_limit(10),
    );
    let active_ids: Vec<_> = active.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(
        active_ids,
        vec![
            "memory:document:Tenant-A:newer-active",
            "memory:document:Tenant-A:older-active"
        ]
    );

    let include_inactive = store.memory_documents_by_updated_at(
        MemoryDocumentQuery::tenant("Tenant-A")
            .with_before("2026-06-27T00:00:03Z")
            .with_limit(10),
    );
    let doc_ids: Vec<_> = include_inactive
        .iter()
        .map(|node| node.properties["doc_id"].as_str().unwrap())
        .collect();
    assert_eq!(doc_ids, vec!["archived", "older-active"]);
}

#[test]
fn memory_documents_by_updated_at_excludes_nodes_that_expire_after_indexing() {
    let mut store = InMemoryGraphStore::new();
    let expires_at_ms = now_ms().saturating_add(250);
    let mut expiring = memory_doc("expiring", "Tenant-A", "active", "2026-06-27T00:00:01Z");
    expiring.properties[TTL_PROPERTY] = json!(expires_at_ms);
    store.upsert_node(expiring).unwrap();

    let lens = crate::MemoryDocumentLens::bind(&store).unwrap();
    let query = MemoryDocumentQuery::tenant("Tenant-A").with_limit(10);
    assert_eq!(lens.query(query.clone()).unwrap().len(), 1);
    while now_ms() <= expires_at_ms {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    assert!(lens.query(query).unwrap().is_empty());
}

#[test]
fn memory_document_index_updates_on_upsert_and_rebuild() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(memory_doc(
            "note",
            "Tenant-A",
            "active",
            "2026-06-27T00:00:01Z",
        ))
        .unwrap();
    store
        .upsert_node(memory_doc(
            "note",
            "Tenant-A",
            "archived",
            "2026-06-27T00:00:03Z",
        ))
        .unwrap();

    assert!(store
        .memory_documents_by_updated_at(
            MemoryDocumentQuery::tenant("Tenant-A")
                .with_status("active")
                .with_limit(10),
        )
        .is_empty());

    let archived = store.memory_documents_by_updated_at(
        MemoryDocumentQuery::tenant("Tenant-A")
            .with_status("archived")
            .with_limit(10),
    );
    assert_eq!(archived.len(), 1);
    assert_eq!(archived[0].properties["doc_id"], json!("note"));
    assert!(store.verify().ok);

    store.rebuild_indexes().unwrap();
    let rebuilt = store.memory_documents_by_updated_at(
        MemoryDocumentQuery::tenant("Tenant-A")
            .with_status("archived")
            .with_limit(10),
    );
    assert_eq!(rebuilt.len(), 1);
    assert_eq!(
        rebuilt[0].properties["updated_at"],
        json!("2026-06-27T00:00:03Z")
    );
    assert!(store.verify().ok);
}

#[test]
fn memory_document_query_latency_fixture_records_delta() {
    const DOCUMENTS: usize = 512;
    const ITERATIONS: usize = 250;
    // Same optimized test profile and fixture immediately before G7 removed
    // the two bespoke store maps (2026-07-22, Apple Silicon macOS).
    const PRE_G7_PER_QUERY_NS: [u128; 2] = [47_828, 43_397];

    let mut store = InMemoryGraphStore::new();
    for index in 0..DOCUMENTS {
        let tenant = if index % 5 == 0 {
            "Tenant-B"
        } else {
            "Tenant-A"
        };
        let status = if index % 2 == 0 { "active" } else { "archived" };
        let updated_at = format!(
            "2026-06-27T{:02}:{:02}:{:02}Z",
            index / 3_600,
            (index / 60) % 60,
            index % 60
        );
        store
            .upsert_node(memory_doc(
                &format!("latency-{index:04}"),
                tenant,
                status,
                &updated_at,
            ))
            .unwrap();
    }

    let query = MemoryDocumentQuery::tenant("Tenant-A")
        .with_status("active")
        .with_limit(50);
    let expected = store.memory_documents_by_updated_at(query.clone());
    assert_eq!(expected.len(), 50);

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(store.memory_documents_by_updated_at(query.clone()));
    }
    let elapsed = started.elapsed();
    let post_g7_per_query_ns = elapsed.as_nanos() / ITERATIONS as u128;
    let regression_budget_ns = PRE_G7_PER_QUERY_NS.iter().copied().max().unwrap() * 4;
    eprintln!(
        "memory_document_query_latency_fixture documents={DOCUMENTS} iterations={ITERATIONS} pre_g7_per_query_ns={PRE_G7_PER_QUERY_NS:?} elapsed_ns={} post_g7_per_query_ns={}",
        elapsed.as_nanos(),
        post_g7_per_query_ns
    );
    assert!(
        post_g7_per_query_ns <= regression_budget_ns,
        "cached graph-derived query exceeded the 4x pre-G7 regression budget: post={post_g7_per_query_ns}ns budget={regression_budget_ns}ns"
    );
}

#[test]
fn node_ttl_property_is_extracted_correctly() {
    let node = ttl_node("atom-1", 12345);
    assert_eq!(node_ttl_expires_at_ms(&node), Some(12345));
}

#[test]
fn node_ttl_absent_means_no_expiration() {
    let node = plain_node("atom-1");
    assert_eq!(node_ttl_expires_at_ms(&node), None);
    assert!(!node_is_expired(&node, now_ms() + 1_000_000));
}

#[test]
fn node_ttl_zero_or_negative_means_no_expiration() {
    let zero = ttl_node("a", 0);
    let neg = ttl_node("b", -1);
    assert_eq!(node_ttl_expires_at_ms(&zero), None);
    assert_eq!(node_ttl_expires_at_ms(&neg), None);
    assert!(!node_is_expired(&zero, 99_999_999_999));
    assert!(!node_is_expired(&neg, 99_999_999_999));
}

#[test]
fn get_node_returns_none_for_expired_node() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    store.upsert_node(ttl_node("expired", past)).unwrap();
    assert!(store.get_node("expired").is_none());
}

#[test]
fn live_node_records_and_relational_discovery_exclude_expired_nodes() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "live",
            ["LiveOnly"],
            json!({ "title": "live" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "expired",
            ["ExpiredOnly", "LiveOnly"],
            json!({ TTL_PROPERTY: now_ms() - 60_000 }),
        ))
        .unwrap();

    assert_eq!(
        GraphRead::live_node_records(&store)
            .into_iter()
            .map(|node| node.id)
            .collect::<Vec<_>>(),
        vec!["live"]
    );

    let lens = crate::RelationalLens::bind(&store).unwrap();
    assert_eq!(
        lens.relation("liveonly").unwrap().row_ids(),
        vec!["live".to_string()]
    );
    assert!(lens.relation("expiredonly").is_none());
}

#[test]
fn get_node_returns_node_within_ttl_window() {
    let mut store = InMemoryGraphStore::new();
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("alive", future)).unwrap();
    assert!(store.get_node("alive").is_some());
}

#[test]
fn get_node_including_expired_returns_expired_node() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    store.upsert_node(ttl_node("expired", past)).unwrap();
    assert!(store.get_node("expired").is_none());
    assert!(store.get_node_including_expired("expired").is_some());
}

#[test]
fn set_node_ttl_extends_existing_node() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 1_000;
    store.upsert_node(ttl_node("a", past)).unwrap();
    assert!(store.get_node("a").is_none()); // already expired
    let future = now_ms() + 60_000;
    store.set_node_ttl("a", Some(future)).unwrap();
    assert!(store.get_node("a").is_some());
    let refreshed = store.get_node("a").unwrap();
    assert_eq!(node_ttl_expires_at_ms(refreshed), Some(future));
}

#[test]
fn set_node_ttl_none_clears_ttl() {
    let mut store = InMemoryGraphStore::new();
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("a", future)).unwrap();
    store.set_node_ttl("a", None).unwrap();
    let node = store.get_node("a").unwrap();
    assert_eq!(node_ttl_expires_at_ms(node), None);
    assert_eq!(store.ttl_active_count(), 0);
}

#[test]
fn set_node_ttl_on_missing_node_errors() {
    let mut store = InMemoryGraphStore::new();
    let err = store
        .set_node_ttl("nonexistent", Some(now_ms() + 60_000))
        .unwrap_err();
    assert_eq!(err.code, "missing_graph_node");
}

#[test]
fn guarded_insert_returns_none_for_existing_node() {
    let mut store = InMemoryGraphStore::new();
    let first = store
        .insert_node_if_absent(NodeRecord::new(
            "node:url",
            ["url"],
            json!({ "state": "frontier" }),
        ))
        .unwrap();
    let second = store
        .insert_node_if_absent(NodeRecord::new(
            "node:url",
            ["url"],
            json!({ "state": "frontier" }),
        ))
        .unwrap();
    assert!(first.is_some());
    assert!(second.is_none());
}

#[test]
fn guarded_insert_replaces_expired_node_before_purge() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(ttl_node("node:url", now_ms() - 60_000))
        .unwrap();

    let replacement = store
        .insert_node_if_absent(NodeRecord::new(
            "node:url",
            ["url"],
            json!({ "state": "refreshed" }),
        ))
        .unwrap();

    assert!(replacement.is_some());
    let node = store.get_node("node:url").unwrap();
    assert_eq!(node.properties["state"], "refreshed");
    assert_eq!(node_ttl_expires_at_ms(node), None);
    assert_eq!(store.ttl_active_count(), 0);
}

#[test]
fn compare_and_set_node_property_returns_none_on_stale_state() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "node:url",
            ["url"],
            json!({ "state": "frontier" }),
        ))
        .unwrap();
    let first = store
        .compare_and_set_node_property("node:url", "state", &json!("frontier"), json!("in_flight"))
        .unwrap();
    let second = store
        .compare_and_set_node_property("node:url", "state", &json!("frontier"), json!("in_flight"))
        .unwrap();
    assert!(first.is_some());
    assert!(second.is_none());
    assert_eq!(
        store
            .get_node("node:url")
            .and_then(|node| node.properties.get("state")),
        Some(&json!("in_flight"))
    );
}

#[test]
fn query_nodes_excludes_expired() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("expired", past)).unwrap();
    store.upsert_node(ttl_node("alive", future)).unwrap();
    let results = store.query_nodes(NodeQuery::label("MemoryAtom"));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "alive");
}

#[test]
fn query_nodes_includes_expired_when_flag_set() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("expired", past)).unwrap();
    store.upsert_node(ttl_node("alive", future)).unwrap();
    let results = store.query_nodes(NodeQuery::label("MemoryAtom").with_include_expired(true));
    assert_eq!(results.len(), 2);
}

#[test]
fn neighbors_excludes_expired_target() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    let future = now_ms() + 60_000;
    store.upsert_node(plain_node("A")).unwrap();
    store.upsert_node(ttl_node("B-expired", past)).unwrap();
    store.upsert_node(ttl_node("B-alive", future)).unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e1",
            "A",
            "MENTIONS",
            "B-expired",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new("e2", "A", "MENTIONS", "B-alive", json!({})))
        .unwrap();
    let hits = store.neighbors(NeighborQuery {
        node_id: "A".into(),
        direction: Direction::Out,
        edge_type: None,
        include_expired: false,
    });
    let target_ids: Vec<_> = hits.into_iter().map(|h| h.node_id).collect();
    assert_eq!(target_ids, vec!["B-alive"]);
}

#[test]
fn nodes_expiring_before_returns_index_ordered() {
    let mut store = InMemoryGraphStore::new();
    let base = now_ms() + 60_000;
    store.upsert_node(ttl_node("third", base + 30_000)).unwrap();
    store.upsert_node(ttl_node("first", base + 10_000)).unwrap();
    store
        .upsert_node(ttl_node("second", base + 20_000))
        .unwrap();
    let results = store.nodes_expiring_before(base + 50_000, 10);
    let ids: Vec<_> = results.into_iter().map(|n| n.id).collect();
    assert_eq!(ids, vec!["first", "second", "third"]);
}

#[test]
fn expiration_index_updates_on_upsert() {
    let mut store = InMemoryGraphStore::new();
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("a", future)).unwrap();
    assert_eq!(store.ttl_active_count(), 1);
    // Re-upsert with new TTL: old index entry should be removed,
    // new one inserted (count stays 1, not 2).
    let new_future = now_ms() + 120_000;
    store.upsert_node(ttl_node("a", new_future)).unwrap();
    assert_eq!(store.ttl_active_count(), 1);
    // Re-upsert without TTL: index entry should be evicted.
    store.upsert_node(plain_node("a")).unwrap();
    assert_eq!(store.ttl_active_count(), 0);
}

#[test]
fn expiration_index_updates_on_set_node_ttl() {
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(plain_node("a")).unwrap();
    assert_eq!(store.ttl_active_count(), 0);
    store.set_node_ttl("a", Some(now_ms() + 60_000)).unwrap();
    assert_eq!(store.ttl_active_count(), 1);
    store.set_node_ttl("a", None).unwrap();
    assert_eq!(store.ttl_active_count(), 0);
}

#[test]
fn expiration_index_rebuilt_on_startup_scan() {
    // Simulate a fresh store loaded from a snapshot by calling
    // rebuild_indexes, which clears all secondary indexes and
    // repopulates them from records. TTL index should rebuild
    // automatically because add_node_indexes handles it.
    let mut store = InMemoryGraphStore::new();
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("a", future)).unwrap();
    store.upsert_node(ttl_node("b", future + 1)).unwrap();
    assert_eq!(store.ttl_active_count(), 2);
    store.rebuild_indexes().unwrap();
    assert_eq!(store.ttl_active_count(), 2);
}

#[test]
fn purge_expired_nodes_returns_count_and_clears_storage() {
    let mut store = InMemoryGraphStore::new();
    let past = now_ms() - 60_000;
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("expired-1", past)).unwrap();
    store
        .upsert_node(ttl_node("expired-2", past - 100))
        .unwrap();
    store.upsert_node(ttl_node("alive", future)).unwrap();
    store.upsert_node(plain_node("permanent")).unwrap();
    let purged = store.purge_expired_nodes().unwrap();
    assert_eq!(purged, 2);
    // Use include_expired to verify the storage really is cleared
    // (not just hidden by the read-time filter).
    assert!(store.get_node_including_expired("expired-1").is_none());
    assert!(store.get_node_including_expired("expired-2").is_none());
    assert!(store.get_node("alive").is_some());
    assert!(store.get_node("permanent").is_some());
    assert_eq!(store.ttl_active_count(), 1); // only "alive"
}

#[test]
fn purge_expired_nodes_removes_incident_edges_and_preserves_graph_invariants() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(ttl_node("expired", now_ms() - 60_000))
        .unwrap();
    store.upsert_node(plain_node("live")).unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:expired-live",
            "expired",
            "LINKS",
            "live",
            json!({}),
        ))
        .unwrap();

    assert_eq!(store.purge_expired_nodes().unwrap(), 1);
    assert!(store.get_node_including_expired("expired").is_none());
    assert!(store.get_edge("edge:expired-live").is_none());
    assert!(store.verify().ok);
}

#[test]
fn purge_with_no_expired_returns_zero() {
    let mut store = InMemoryGraphStore::new();
    let future = now_ms() + 60_000;
    store.upsert_node(ttl_node("alive", future)).unwrap();
    store.upsert_node(plain_node("permanent")).unwrap();
    assert!(!store.has_expired_nodes_before(now_ms()));
    let purged = store.purge_expired_nodes().unwrap();
    assert_eq!(purged, 0);
    assert_eq!(store.ttl_active_count(), 1);
}

#[test]
fn historical_registry_round_trips_and_legacy_snapshots_fail_closed() {
    let mut store = InMemoryGraphStore::new();
    let node = plain_node("historical:evicted");
    store.upsert_node(node.clone()).unwrap();
    store.evict_node(&node.id).unwrap();
    let snapshot = store.snapshot();
    assert!(snapshot.historical_node_history_complete);
    let mut restored = InMemoryGraphStore::from_snapshot(snapshot).unwrap();
    assert!(restored
        .insert_node_if_never_existed(node.clone())
        .unwrap()
        .is_none());

    let legacy_snapshot: GraphSnapshot = serde_json::from_value(json!({
        "version": 7,
        "nodes": [],
        "edges": []
    }))
    .unwrap();
    assert!(!legacy_snapshot.historical_node_history_complete);
    let mut legacy = InMemoryGraphStore::from_snapshot(legacy_snapshot).unwrap();
    let error = legacy
        .insert_node_if_never_existed(plain_node("unknown:legacy"))
        .unwrap_err();
    assert_eq!(error.code, "historical_registry_incomplete");

    let envelope: RedCoreSnapshotEnvelope = serde_json::from_value(json!({
        "version": REDCORE_MANIFEST_VERSION,
        "txn_id": 1,
        "graph": {"version": 1, "nodes": [], "edges": []}
    }))
    .unwrap();
    assert_eq!(envelope.historical_node_history_complete, None);

    let empty = InMemoryGraphStore::new().snapshot();
    assert!(empty.historical_node_history_complete);
    let mut empty = InMemoryGraphStore::from_snapshot(empty).unwrap();
    assert!(empty
        .insert_node_if_never_existed(plain_node("first:new"))
        .unwrap()
        .is_some());
}

#[test]
fn historical_registry_stops_growing_at_limit_and_fails_closed() {
    let mut store = InMemoryGraphStore::new();
    store.historical_node_id_limit = 2;

    store.upsert_node(plain_node("history:a")).unwrap();
    store.upsert_node(plain_node("history:b")).unwrap();
    assert!(store.historical_node_history_complete);
    assert_eq!(store.historical_node_ids.len(), 2);

    // Crossing the cap does not reject an ordinary graph write. It marks
    // the duplicate history registry incomplete and stops its growth.
    store.upsert_node(plain_node("history:c")).unwrap();
    assert!(store.get_node("history:c").is_some());
    assert!(!store.historical_node_history_complete);
    assert_eq!(store.historical_node_ids.len(), 2);
    assert!(!store.has_historical_node_id("history:c"));

    store
        .apply_recovered_delete("history:replayed-delete")
        .unwrap();
    store.upsert_node(plain_node("history:d")).unwrap();
    assert_eq!(store.historical_node_ids.len(), 2);

    let error = store
        .insert_node_if_never_existed(plain_node("history:unknown"))
        .unwrap_err();
    assert_eq!(error.code, "historical_registry_incomplete");
    let snapshot = store.snapshot();
    assert_eq!(snapshot.historical_node_ids.len(), 2);
    assert!(!snapshot.historical_node_history_complete);
}

#[test]
fn historical_registry_recovery_is_bounded_deterministically() {
    let snapshot = GraphSnapshot {
        version: 7,
        historical_node_ids: BTreeSet::from([
            "history:a".to_string(),
            "history:b".to_string(),
            "history:c".to_string(),
        ]),
        historical_node_history_complete: true,
        nodes: vec![plain_node("history:live")],
        edges: Vec::new(),
    };

    let restored = InMemoryGraphStore::from_snapshot_with_historical_limit(snapshot, 2)
        .expect("bounded historical recovery");

    assert_eq!(
        restored.historical_node_ids,
        BTreeSet::from(["history:a".to_string(), "history:b".to_string()])
    );
    assert!(!restored.historical_node_history_complete);
    assert!(restored.get_node("history:live").is_some());
    assert_eq!(restored.snapshot().historical_node_ids.len(), 2);
}
