use rustyred_thg_core::{DeploymentId, FoldGraphStore, GraphRead, GraphWrite, NodeRecord};
use serde_json::json;

#[test]
fn deployment_roots_are_isolated_and_hash_bound() {
    let first_root = tempfile::tempdir().unwrap();
    let second_root = tempfile::tempdir().unwrap();
    let first_id = DeploymentId::new("deployment:first").unwrap();
    let second_id = DeploymentId::new("deployment:second").unwrap();

    let mut first = FoldGraphStore::open(first_root.path(), first_id.clone()).unwrap();
    let mut second = FoldGraphStore::open(second_root.path(), second_id.clone()).unwrap();
    let node = NodeRecord::new("logical:key", ["Fixture"], json!({"value": "same"}));
    first.upsert_node(node.clone()).unwrap();
    second.upsert_node(node).unwrap();

    assert!(first.get_node("logical:key").is_some());
    assert!(second.get_node("logical:key").is_some());
    assert_eq!(first.log().records()[0].deployment_id, first_id);
    assert_eq!(second.log().records()[0].deployment_id, second_id);
    assert_ne!(
        first.log().records()[0].record_hash,
        second.log().records()[0].record_hash
    );

    drop(first);
    let error = FoldGraphStore::open(first_root.path(), second_id).unwrap_err();
    assert!(error.message.contains("does not belong to deployment"));
}
