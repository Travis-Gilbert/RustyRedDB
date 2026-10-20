use proptest::prelude::*;
use rustyred_thg_core::{
    search_scoped_vector_batch, EdgeRecord, GraphMutation, GraphMutationBatch, InMemoryGraphStore,
    NodeRecord, ReceiptScope, RedCoreDurability, RedCoreGraphStore, RedCoreOptions,
    ScopedVectorQuery, VectorFilterPolicy, VectorIndexDefinition, VectorSearchBackend,
};
use serde_json::json;

fn point(id: usize) -> NodeRecord {
    NodeRecord::new(
        format!("point:{id:04}"),
        ["VectorFixture"],
        json!({"embedding":[id as f32 + 1.0, 1.0]}),
    )
}

fn scope(tenant: &str, repo: &str, generation: u64) -> ReceiptScope {
    ReceiptScope::from([
        ("tenant_id".to_string(), tenant.to_string()),
        ("repo_id".to_string(), repo.to_string()),
        ("generation".to_string(), generation.to_string()),
    ])
}

fn scoped_definition() -> VectorIndexDefinition {
    VectorIndexDefinition::new(
        "code-symbol-epistemic",
        "CodeSymbol",
        "embedding",
        2,
        Default::default(),
    )
    .with_scope_fields(["tenant_id", "repo_id", "generation"])
}

fn scoped_policy() -> VectorFilterPolicy {
    VectorFilterPolicy::scoped(scope("Travis-Gilbert", "repo:a", 9))
        .require_labels(["CodeSymbol"])
        .for_repo("repo:a")
}

fn scoped_node(
    id: &str,
    tenant: &str,
    repo: &str,
    generation: u64,
    vector: [f32; 2],
) -> NodeRecord {
    NodeRecord::new(
        id,
        ["CodeSymbol"],
        json!({
            "tenant_id": tenant,
            "repo_id": repo,
            "generation": generation,
            "embedding": vector,
        }),
    )
}

fn persistent_scoped_store(nodes: impl IntoIterator<Item = NodeRecord>) -> RedCoreGraphStore {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("CodeSymbol", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            nodes.into_iter().map(GraphMutation::NodeUpsert),
        ))
        .unwrap();
    store
}

#[test]
fn persistent_scoped_batch_rejects_foreign_points_without_rebuilding_an_index() {
    let store = persistent_scoped_store([
        scoped_node("a", "Travis-Gilbert", "repo:a", 9, [1.0, 0.0]),
        scoped_node("foreign", "other", "repo:b", 9, [1.0, 0.0]),
        scoped_node("incomplete", "Travis-Gilbert", "repo:a", 8, [1.0, 0.0]),
    ]);
    let queries = [
        ScopedVectorQuery::new("query", vec![1.0, 0.0]),
        ScopedVectorQuery::new("a", vec![1.0, 0.0]),
    ];
    let receipt =
        search_scoped_vector_batch(&store, &scoped_definition(), &scoped_policy(), &queries, 8)
            .unwrap();

    assert_eq!(receipt.admitted_points, 1);
    assert_eq!(receipt.rejected_points, 2);
    assert_eq!(receipt.work.indexed_points, 3);
    assert_eq!(receipt.work.index_rebuilds, 0);
    assert_eq!(
        receipt
            .matches
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["a", "query"]
    );
    assert!(receipt.matches["a"].is_empty());
    assert_eq!(receipt.matches["query"][0].object_id, "a");
}

#[test]
fn persistent_scoped_batch_rejects_an_incomplete_policy_before_searching() {
    let store = persistent_scoped_store([scoped_node(
        "point",
        "Travis-Gilbert",
        "repo:a",
        9,
        [1.0, 0.0],
    )]);
    let policy = VectorFilterPolicy::scoped(ReceiptScope::from([
        ("tenant_id".to_string(), "Travis-Gilbert".to_string()),
        ("repo_id".to_string(), "repo:a".to_string()),
    ]))
    .require_labels(["CodeSymbol"])
    .for_repo("repo:a");

    let error = search_scoped_vector_batch(
        &store,
        &scoped_definition(),
        &policy,
        &[ScopedVectorQuery::new("query", vec![1.0, 0.0])],
        8,
    )
    .unwrap_err();

    assert_eq!(error.code, "incomplete_vector_scope_policy");
}

#[test]
fn persistent_scoped_batch_rejects_duplicate_query_ids_before_searching() {
    let store = persistent_scoped_store([scoped_node(
        "point",
        "Travis-Gilbert",
        "repo:a",
        9,
        [1.0, 0.0],
    )]);
    let error = search_scoped_vector_batch(
        &store,
        &scoped_definition(),
        &scoped_policy(),
        &[
            ScopedVectorQuery::new("duplicate", vec![1.0, 0.0]),
            ScopedVectorQuery::new("duplicate", vec![0.0, 1.0]),
        ],
        8,
    )
    .unwrap_err();

    assert_eq!(error.code, "duplicate_vector_query_id");
}

#[test]
fn persistent_scoped_batch_with_no_queries_reports_admission_without_index_work() {
    let mut rejected = scoped_node("rejected", "Travis-Gilbert", "repo:a", 9, [0.0, 1.0]);
    rejected
        .properties
        .as_object_mut()
        .unwrap()
        .insert("policy_allowed".to_string(), json!(false));
    let store = persistent_scoped_store([
        scoped_node("admitted", "Travis-Gilbert", "repo:a", 9, [1.0, 0.0]),
        rejected,
    ]);

    let receipt =
        search_scoped_vector_batch(&store, &scoped_definition(), &scoped_policy(), &[], 8).unwrap();

    assert_eq!(receipt.admitted_points, 1);
    assert_eq!(receipt.rejected_points, 1);
    assert!(receipt.matches.is_empty());
    assert_eq!(receipt.work, Default::default());
}

#[test]
fn persistent_scoped_batch_results_are_deterministic_across_insertion_order() {
    let nodes = vec![
        scoped_node("a", "Travis-Gilbert", "repo:a", 9, [1.0, 0.0]),
        scoped_node("b", "Travis-Gilbert", "repo:a", 9, [0.75, 0.25]),
        scoped_node("c", "Travis-Gilbert", "repo:a", 9, [0.0, 1.0]),
    ];
    let first = persistent_scoped_store(nodes.clone());
    let second = persistent_scoped_store(nodes.into_iter().rev());
    let queries = [
        ScopedVectorQuery::new("query:x", vec![1.0, 0.0]),
        ScopedVectorQuery::new("query:y", vec![0.0, 1.0]),
    ];
    let expected =
        search_scoped_vector_batch(&first, &scoped_definition(), &scoped_policy(), &queries, 2)
            .unwrap();
    let actual =
        search_scoped_vector_batch(&second, &scoped_definition(), &scoped_policy(), &queries, 2)
            .unwrap();

    assert_eq!(actual.matches, expected.matches);
    assert_eq!(actual.work, expected.work);
    assert_eq!(actual.admitted_points, expected.admitted_points);
    assert_eq!(actual.rejected_points, expected.rejected_points);
}

#[test]
fn exact_search_retains_only_k_candidates() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            (0..63).map(|id| GraphMutation::NodeUpsert(point(id))),
        ))
        .unwrap();

    let receipt = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();

    assert_eq!(receipt.hits.len(), 8);
    assert_eq!(receipt.work.backend, VectorSearchBackend::ExactCosine);
    assert_eq!(receipt.work.distance_evaluations, 63);
    assert!(receipt.work.retained_candidates <= 8);
}

#[test]
fn bulk_commit_builds_turbovec_once_at_the_acceleration_threshold() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            (0..256).map(|id| GraphMutation::NodeUpsert(point(id))),
        ))
        .unwrap();

    let receipt = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();

    assert_eq!(
        receipt.work.backend,
        rustyred_thg_core::VectorSearchBackend::Turbovec,
        "unexpected work receipt: {:?}",
        receipt.work
    );
    assert_eq!(receipt.work.index_rebuilds, 1);
    assert_eq!(receipt.work.fallback_reason, None);
}

#[test]
fn small_indexes_remain_exact() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            (0..63).map(|id| GraphMutation::NodeUpsert(point(id))),
        ))
        .unwrap();
    let receipt = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(
        receipt.work.backend,
        rustyred_thg_core::VectorSearchBackend::ExactCosine
    );
    assert_eq!(receipt.work.index_rebuilds, 0);
    assert_eq!(
        receipt.work.fallback_reason.as_deref(),
        Some("below_acceleration_minimum")
    );
}

#[test]
fn incremental_updates_do_not_rebuild_the_resident_index() {
    let mut store = InMemoryGraphStore::new();
    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    for id in 0..64 {
        store.upsert_node(point(id)).unwrap();
    }

    let receipt = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();

    assert_eq!(
        receipt.work.backend,
        rustyred_thg_core::VectorSearchBackend::Turbovec
    );
    assert_eq!(receipt.work.index_rebuilds, 1);
    assert_eq!(receipt.work.fallback_reason, None);

    store.upsert_node(point(0)).unwrap();
    let after_update = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(
        after_update.work.backend,
        rustyred_thg_core::VectorSearchBackend::Turbovec
    );
    assert_eq!(after_update.work.index_rebuilds, 1);
}

#[test]
fn failed_batch_does_not_publish_rebuilt_vector_state() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            (0..256).map(|id| GraphMutation::NodeUpsert(point(id))),
        ))
        .unwrap();

    let error = store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(point(256)),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:missing",
                "point:0256",
                "SIMILAR_TO",
                "point:missing",
                json!({}),
            )),
        ]))
        .unwrap_err();
    assert_eq!(error.code, "missing_graph_endpoint");

    let receipt = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(
        receipt.work.backend,
        rustyred_thg_core::VectorSearchBackend::Turbovec
    );
    assert_eq!(receipt.work.indexed_points, 256);
    assert_eq!(receipt.work.index_rebuilds, 1);
    assert_eq!(receipt.work.fallback_reason, None);
}

#[test]
fn snapshot_only_commits_preserve_untouched_indexes() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let data_dir = std::env::temp_dir().join(format!(
        "theorem-vector-snapshot-clone-{}-{stamp}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&data_dir).ok();
    let mut store = RedCoreGraphStore::open(
        &data_dir,
        RedCoreOptions {
            durability: RedCoreDurability::SnapshotOnly,
            snapshot_interval_writes: 1,
            strict_acid: false,
        },
    )
    .unwrap();
    store
        .commit_batch(GraphMutationBatch::new(
            (0..256).map(|id| GraphMutation::NodeUpsert(point(id))),
        ))
        .unwrap();

    store
        .designate_vector_property("VectorFixture", "embedding", 2)
        .unwrap();
    let designated = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(designated.work.backend, VectorSearchBackend::Turbovec);
    assert_eq!(designated.work.index_rebuilds, 1);
    assert_eq!(designated.work.fallback_reason, None);

    store
        .commit_batch(GraphMutationBatch::new([GraphMutation::EdgeUpsert(
            EdgeRecord::new(
                "edge:untouched-vector-index",
                "point:0000",
                "SIMILAR_TO",
                "point:0001",
                json!({}),
            ),
        )]))
        .unwrap();
    let after_edge = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(after_edge.work.backend, VectorSearchBackend::Turbovec);
    assert_eq!(after_edge.work.index_rebuilds, 1);
    assert_eq!(after_edge.work.fallback_reason, None);

    store
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new(
                "other:0000",
                ["OtherFixture"],
                json!({"embedding":[1.0, 0.0]}),
            ),
        )]))
        .unwrap();
    let after_other_label = store
        .vector_search_with_receipt(Some("VectorFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(
        after_other_label.work.backend,
        VectorSearchBackend::Turbovec
    );
    assert_eq!(after_other_label.work.index_rebuilds, 1);
    assert_eq!(after_other_label.work.fallback_reason, None);

    drop(store);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn mixed_exact_and_turbovec_indexes_report_conservative_backend() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("LargeFixture", "embedding", 2)
        .unwrap();
    store
        .designate_vector_property("SmallFixture", "embedding", 2)
        .unwrap();
    let large = (0..65).map(|id| {
        GraphMutation::NodeUpsert(NodeRecord::new(
            format!("large:{id:04}"),
            ["LargeFixture"],
            json!({"embedding":[id as f32 + 1.0, 1.0]}),
        ))
    });
    let small = (0..63).map(|id| {
        GraphMutation::NodeUpsert(NodeRecord::new(
            format!("small:{id:04}"),
            ["SmallFixture"],
            json!({"embedding":[id as f32 + 1.0, 1.0]}),
        ))
    });
    store
        .commit_batch(GraphMutationBatch::new(large.chain(small)))
        .unwrap();

    let receipt = store
        .vector_search_with_receipt(None, "embedding", &[1.0, 0.0], 8)
        .unwrap();

    let large_receipt = store
        .vector_search_with_receipt(Some("LargeFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(large_receipt.work.backend, VectorSearchBackend::Turbovec);
    assert_eq!(large_receipt.work.fallback_reason, None);
    let small_receipt = store
        .vector_search_with_receipt(Some("SmallFixture"), "embedding", &[1.0, 0.0], 8)
        .unwrap();
    assert_eq!(small_receipt.work.backend, VectorSearchBackend::ExactCosine);
    assert_eq!(
        small_receipt.work.fallback_reason.as_deref(),
        Some("below_acceleration_minimum")
    );

    assert_eq!(receipt.work.backend, VectorSearchBackend::ExactCosine);
    assert_eq!(receipt.work.indexed_points, 128);
    assert_eq!(receipt.work.index_rebuilds, 1);
    assert_eq!(
        receipt.work.fallback_reason, small_receipt.work.fallback_reason,
        "the aggregate must preserve the exact index's threshold fallback"
    );
}

#[test]
fn receipt_aggregates_matching_indexes_and_orders_ties_by_node_id() {
    let mut store = RedCoreGraphStore::memory();
    store
        .designate_vector_property("AlphaFixture", "embedding", 2)
        .unwrap();
    store
        .designate_vector_property("BetaFixture", "embedding", 2)
        .unwrap();
    store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "point:a",
                ["AlphaFixture"],
                json!({"embedding":[1.0, 0.0]}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "point:b",
                ["BetaFixture"],
                json!({"embedding":[1.0, 0.0]}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "point:c",
                ["AlphaFixture"],
                json!({"embedding":[0.0, 1.0]}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "point:d",
                ["BetaFixture"],
                json!({"embedding":[-1.0, 0.0]}),
            )),
        ]))
        .unwrap();

    let receipt = store
        .vector_search_with_receipt(None, "embedding", &[1.0, 0.0], 3)
        .unwrap();

    assert_eq!(
        receipt
            .hits
            .iter()
            .map(|(node_id, _)| node_id.as_str())
            .collect::<Vec<_>>(),
        ["point:a", "point:b", "point:c"]
    );
    assert_eq!(receipt.work.indexed_points, 4);
    assert_eq!(receipt.work.distance_evaluations, 4);
    assert_eq!(receipt.work.retained_candidates, 3);
}

proptest! {
    #[test]
    fn updates_and_removals_match_the_full_sort_oracle(
        raw in prop::collection::vec((-100i16..100, -100i16..100, any::<bool>()), 1..96)
    ) {
        let mut store = RedCoreGraphStore::memory();
        store
            .designate_vector_property("VectorFixture", "embedding", 2)
            .unwrap();
        let mut live = Vec::new();
        for (index, (x, y, remove)) in raw.iter().copied().enumerate() {
            let id = format!("point:{index:04}");
            store
                .upsert_node(NodeRecord::new(
                    id.clone(),
                    ["VectorFixture"],
                    json!({"embedding":[x as f32, y as f32]}),
                ))
                .unwrap();
            if remove {
                store.delete_node(&id).unwrap();
            } else {
                live.push((id, [x as f32, y as f32]));
            }
        }
        let query = [0.25f32, 0.75f32];
        let mut expected = live
            .into_iter()
            .map(|(id, vector)| {
                let norm = (vector[0] * vector[0] + vector[1] * vector[1]).sqrt();
                let normalized = if norm < 1e-10 {
                    vector
                } else {
                    [vector[0] / norm, vector[1] / norm]
                };
                let query_norm = (query[0] * query[0] + query[1] * query[1]).sqrt();
                let distance = 1.0
                    - ((query[0] / query_norm) * normalized[0]
                        + (query[1] / query_norm) * normalized[1])
                        .clamp(-1.0, 1.0);
                (id, distance)
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        expected.truncate(8);

        let actual = store
            .vector_search(Some("VectorFixture"), "embedding", &query, 8)
            .unwrap();
        prop_assert_eq!(actual, expected);
    }
}
