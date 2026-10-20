use rustyred_thg_core::{
    EdgeRecord, FoldGraphStore, GraphRead, GraphWrite, NeighborQuery, NodeQuery, NodeRecord,
    PointReadExecution,
};
use serde_json::json;
use std::error::Error;
use theorem_oplog::LogPosition;

#[test]
fn pinned_point_read_uses_direct_graph_fold_index_without_appending() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    store.upsert_node(NodeRecord::new(
        "claim:direct",
        ["Claim"],
        json!({"status": "accepted"}),
    ))?;

    let before_position = store.latest_position();
    let before_records = store.log().len();
    let result = store.pin_current().get_node("claim:direct");

    assert_eq!(result.position, before_position);
    assert_eq!(result.execution, PointReadExecution::DirectGraphFoldIndex);
    assert_eq!(result.value, store.get_node_record("claim:direct"));
    assert_eq!(store.latest_position(), before_position);
    assert_eq!(store.log().len(), before_records);
    assert!(store.pin(LogPosition::ZERO).is_err());
    Ok(())
}

#[test]
fn every_primary_pinned_query_reports_one_position_without_appending() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    store.upsert_node(NodeRecord::new(
        "claim:a",
        ["Claim"],
        json!({"status": "accepted"}),
    ))?;
    store.upsert_node(NodeRecord::new(
        "claim:b",
        ["Claim"],
        json!({"status": "accepted"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "claim:a",
        "SUPPORTS",
        "claim:b",
        json!({}),
    ))?;

    let before_position = store.latest_position();
    let before_records = store.log().len();
    let pinned = store.pin_current();
    let node = pinned.get_node("claim:a");
    let edge = pinned.get_edge("edge:ab");
    let nodes = pinned.query_nodes(NodeQuery::label("Claim"));
    let neighbors = pinned.neighbors(NeighborQuery::out("claim:a").with_edge_type("SUPPORTS"));

    for (position, execution) in [
        (node.position, node.execution),
        (edge.position, edge.execution),
        (nodes.position, nodes.execution),
        (neighbors.position, neighbors.execution),
    ] {
        assert_eq!(position, before_position);
        assert_eq!(execution, PointReadExecution::DirectGraphFoldIndex);
    }
    assert_eq!(node.value.expect("node exists").id, "claim:a");
    assert_eq!(edge.value.expect("edge exists").id, "edge:ab");
    assert_eq!(nodes.value.len(), 2);
    assert_eq!(neighbors.value.len(), 1);
    assert_eq!(neighbors.value[0].node_id, "claim:b");
    assert_eq!(store.latest_position(), before_position);
    assert_eq!(store.log().len(), before_records);
    Ok(())
}

#[test]
fn fold_graph_facade_persists_only_through_the_proposal_committer() -> Result<(), Box<dyn Error>> {
    let source = include_str!("../src/graph/fold_store.rs");
    assert!(source.contains("ProposalCommitter"));
    assert!(source.contains("self.committer.commit(proposal)"));
    for forbidden in [
        "RedCoreGraphStore",
        "JournalChunkStore",
        "redcore.journal",
        "OpenOptions",
        "File::create",
        "write_all(",
        "sync_all(",
        "PhysicalLog::append",
    ] {
        assert!(
            !source.contains(forbidden),
            "fold facade contains direct persistence symbol {forbidden}"
        );
    }

    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    store.upsert_node(NodeRecord::new(
        "claim:writer-gate",
        ["Claim"],
        json!({"status": "accepted"}),
    ))?;
    assert_eq!(store.log().len(), 1);
    drop(store);

    let artifacts = std::fs::read_dir(directory.path())?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(artifacts, vec!["theorem.oplog"]);
    assert!(!directory.path().join("redcore.journal").exists());
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:point-read-no-interpreter").unwrap()
}

#[test]
fn fold_graph_module_has_no_interpreter_dependency_or_dispatch_path() {
    let manifest = include_str!("../Cargo.toml");
    let source = include_str!("../src/graph/fold_store.rs");

    for forbidden in [
        "rustyred-thg-programmable-graph",
        "rustyred_thg_programmable_graph",
        "graph-lisp",
        "graph_lisp",
    ] {
        assert!(
            !manifest.contains(forbidden),
            "manifest contains {forbidden}"
        );
        assert!(
            !source.contains(forbidden),
            "fold store contains {forbidden}"
        );
    }
    for forbidden_dispatch in ["eval_program(", "execute_program(", "dispatch_program("] {
        assert!(
            !source.contains(forbidden_dispatch),
            "point-read module dispatches through {forbidden_dispatch}"
        );
    }
}
