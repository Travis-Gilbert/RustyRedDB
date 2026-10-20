use rustyred_thg_core::graph::error::GraphStoreResult;
use rustyred_thg_core::graph::mutation::{GraphMutation, GraphMutationBatch};
use rustyred_thg_core::graph::query::{NeighborQuery, NodeQuery};
use rustyred_thg_core::graph::record::{Direction, EdgeRecord, NodeRecord};
use rustyred_thg_core::graph::temporal::node_time_interval;
use rustyred_thg_core::{Hash, InMemoryGraphStore};
use serde_json::json;

#[test]
fn records_queries_and_mutations_preserve_fixture_outputs() -> GraphStoreResult<()> {
    assert_eq!(Direction::Out, Direction::Out);

    let mut store = InMemoryGraphStore::new();
    store.upsert_node(NodeRecord::new(
        "person:ada",
        ["Researcher", "Person", "Person"],
        json!({
            "name": "Ada",
            "rank": 1,
            "t_start_ms": 10,
            "t_end_ms": 20
        }),
    ))?;
    store.upsert_node(NodeRecord::new(
        "paper:engine",
        ["Paper"],
        json!({"title": "Analytical Engine"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:wrote",
        "person:ada",
        "WROTE",
        "paper:engine",
        json!({"year": 1843}),
    ))?;

    let matches = store.query_nodes(
        NodeQuery::label("Person")
            .with_property("name", json!("Ada"))
            .with_limit(10),
    );
    assert_eq!(
        matches
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["person:ada"]
    );
    assert_eq!(
        node_time_interval(&matches[0]).map(|interval| (interval.start_ms, interval.end_ms)),
        Some((Some(10), Some(20)))
    );

    let neighbors = store.neighbors(NeighborQuery::out("person:ada").with_edge_type("WROTE"));
    assert_eq!(
        neighbors
            .iter()
            .map(|hit| (hit.edge_id.as_str(), hit.node_id.as_str()))
            .collect::<Vec<_>>(),
        vec![("edge:wrote", "paper:engine")]
    );

    let snapshot_bytes =
        serde_json::to_vec(&store.snapshot()).expect("fixture snapshot serializes");
    assert_eq!(
        Hash::of(&snapshot_bytes).to_prefixed(),
        "sha256:6e713828855678ecb413ffa39d9e7dafb94913fe5f5a02aab0f6d373919661ea"
    );

    let mutation_bytes = serde_json::to_vec(&GraphMutationBatch::new([
        GraphMutation::NodeUpsert(matches[0].clone()),
        GraphMutation::EdgeDelete("edge:wrote".to_string()),
    ]))
    .expect("fixture mutation serializes");
    assert_eq!(
        Hash::of(&mutation_bytes).to_prefixed(),
        "sha256:b1f70ffbe9768027b6b5f3223af13215168b5050b0906cd4f25ce50bed8cd446"
    );
    Ok(())
}

#[test]
fn subsystem_extraction_preserves_fixture_outputs() -> GraphStoreResult<()> {
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(NodeRecord::new(
        "person:ada",
        ["Researcher", "Person", "Person"],
        json!({
            "name": "Ada",
            "rank": 1,
            "t_start_ms": 10,
            "t_end_ms": 20
        }),
    ))?;
    store.upsert_node(NodeRecord::new(
        "paper:engine",
        ["Paper"],
        json!({"title": "Analytical Engine"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:wrote",
        "person:ada",
        "WROTE",
        "paper:engine",
        json!({"year": 1843}),
    ))?;

    let snapshot_bytes = serde_json::to_vec(&store.snapshot()).expect("snapshot serializes");
    let verify_bytes = serde_json::to_vec(&store.verify()).expect("verify report serializes");
    let attribution_bytes =
        serde_json::to_vec(&store.memory_attribution()).expect("attribution serializes");

    assert_eq!(
        Hash::of(&snapshot_bytes).to_prefixed(),
        "sha256:6e713828855678ecb413ffa39d9e7dafb94913fe5f5a02aab0f6d373919661ea"
    );
    assert_eq!(
        Hash::of(&verify_bytes).to_prefixed(),
        "sha256:10fdf311153c40fb6897c1110881250813233b2df1b83c63df91907ea93cf217"
    );
    assert_eq!(
        Hash::of(&attribution_bytes).to_prefixed(),
        "sha256:6a94f4faf3e3fd28d184ba78b2ef9db78ce02d50de12cc4c27af40cb87b3311e"
    );
    Ok(())
}
