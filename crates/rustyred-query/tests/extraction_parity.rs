//! Synthetic fixture executed against the exact plan pin and the extracted engine.
use rustyred_thg_core::{
    edge_time_interval, EdgeRecord, FullTextDesignation, FullTextIndex, GraphMutation,
    GraphMutationBatch, InMemoryGraphStore, NodeRecord, RedCoreDurability, RedCoreGraphStore,
    RedCoreOptions, SpatialDesignation, SpatialIndex,
};
use rustyred_thg_query::{
    ast_hash, execute_read, params_hash, parse_cypher_pest, PlanSteeringState,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[test]
fn source_fixture_preserves_records_queries_indexes_payloads_and_history() {
    let directory = std::env::temp_dir().join(format!(
        "rustyred-parity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        ..Default::default()
    };
    let mut store = RedCoreGraphStore::open(&directory, options.clone()).unwrap();
    store
        .designate_vector_property("Part", "embedding", 2)
        .unwrap();
    let first = NodeRecord::new(
        "part:a",
        ["Part"],
        json!({"name":"steel bolt", "embedding":[1.0,0.0]}),
    );
    let second = NodeRecord::new(
        "part:b",
        ["Part"],
        json!({"name":"copper ring", "embedding":[0.0,1.0]}),
    );
    let edge = EdgeRecord::new(
        "edge:a",
        "part:a",
        "RELATED",
        "part:b",
        json!({"t_start_ms":100,"t_end_ms":200}),
    );
    let batch = GraphMutationBatch::new([
        GraphMutation::NodeUpsert(first.clone()),
        GraphMutation::NodeUpsert(second.clone()),
        GraphMutation::EdgeUpsert(edge.clone()),
    ]);
    let serialized_batch = serde_json::to_value(&batch).unwrap();
    store.commit_batch(batch).unwrap();
    let old_head = store
        .pin_graph_history()
        .unwrap()
        .head_roots()
        .logical_commit;
    let bytes = b"synthetic-payload\x00\xff";
    let payload_hash = store.put_cold_document_bytes(bytes).unwrap().unwrap();
    store
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new(
                "part:a",
                ["Part"],
                json!({"name":"updated bolt", "embedding":[1.0,0.0]}),
            ),
        )]))
        .unwrap();
    drop(store);
    let store = RedCoreGraphStore::open(&directory, options).unwrap();
    assert_eq!(
        store
            .get_cold_document_bytes(&payload_hash)
            .unwrap()
            .unwrap(),
        bytes
    );
    let historical = store
        .pin_graph_history()
        .unwrap()
        .revision(old_head)
        .unwrap()
        .get_node("part:a")
        .unwrap()
        .unwrap();
    assert_eq!(historical.properties["name"], "steel bolt");
    assert_eq!(historical.version, 1);
    assert!(historical.content_hash.is_some());
    let hits = store
        .vector_search(Some("Part"), "embedding", &[1.0, 0.0], 2)
        .unwrap();
    assert_eq!(hits[0].0, "part:a");

    // The query crate already implements its graph trait for this existing owner.
    let mut graph = InMemoryGraphStore::new();
    graph.upsert_node(first.clone()).unwrap();
    graph.upsert_node(second).unwrap();
    let params = BTreeMap::from([("name".to_owned(), json!("steel bolt"))]);
    let parsed = parse_cypher_pest("MATCH (p:Part {name: $name}) RETURN p", &params).unwrap();
    let query = execute_read(&graph, &parsed, None::<&PlanSteeringState>, 100).unwrap();
    assert_eq!(query.rows.len(), 1);
    let mut text = FullTextIndex::for_designation(FullTextDesignation {
        label: "Part".into(),
        property: "name".into(),
    });
    text.upsert("part:a", "steel bolt");
    text.upsert("part:b", "copper ring");
    let mut spatial = SpatialIndex::for_designation(SpatialDesignation {
        label: "Place".into(),
        lat_property: "lat".into(),
        lon_property: "lon".into(),
        resolution: 8,
    });
    spatial.upsert("flint", 43.0125, -83.6875).unwrap();
    spatial.upsert("detroit", 42.3314, -83.0458).unwrap();
    let interval = edge_time_interval(&edge).unwrap();
    let actual = json!({
        "typed_mutation_serialization":serialized_batch,
        "reopened_node":store.get_node("part:a").unwrap(),
        "reopened_edge":store.get_edge("edge:a").unwrap(),
        "historical_node":historical,
        "vector_hits":hits,
        "payload_hash":payload_hash,
        "payload_bytes":store.get_cold_document_bytes(&payload_hash).unwrap(),
        "ast_hash":ast_hash(&parsed), "params_hash":params_hash(&json!(params)),
        "query_columns":query.columns, "query_rows":query.rows,
        "fulltext_hits":text.search("bolt", 2),
        "spatial_radius":spatial.radius_search(43.0125, -83.6875, 1.0).unwrap(),
        "spatial_bbox":spatial.bbox_search(43.0,-83.7,43.1,-83.6),
        "temporal_membership":[interval.contains_ms(99),interval.contains_ms(150),interval.contains_ms(201)]
    });
    drop(store);
    std::fs::remove_dir_all(directory).unwrap();
    if let Some(output) = std::env::var_os("RUSTYRED_CAPTURE_SOURCE_FIXTURE") {
        std::fs::write(output, serde_json::to_vec_pretty(&actual).unwrap()).unwrap();
    } else {
        let expected: Value =
            serde_json::from_str(include_str!("fixtures/plan-pin-parity.json")).unwrap();
        assert_eq!(
            actual, expected,
            "exact plan-pin synthetic behavior changed"
        );
    }
}
