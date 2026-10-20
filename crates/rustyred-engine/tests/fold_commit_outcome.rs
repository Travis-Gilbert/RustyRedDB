//! Known physical commits must not become apparent transaction refusals when a fold fails.
#![cfg(feature = "test-fault-injection")]
use rustyred_thg_core::{
    FoldGraphStore, GraphAtomic, GraphMutation, GraphMutationBatch, GraphRead, GraphVerify,
    GraphWrite, NodeRecord,
};
use serde_json::json;
use theorem_oplog::{DeploymentId, LogPosition};

fn batch(id: &str) -> GraphMutationBatch {
    GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
        id,
        ["Record"],
        json!({"value": id}),
    ))])
}

#[test]
fn physical_commit_receipt_retains_position_when_fold_maintenance_fails() {
    for after_graph in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let deployment = DeploymentId::new("test:committed-fold-maintenance").unwrap();
        let mut store = FoldGraphStore::open(dir.path(), deployment.clone()).unwrap();
        store.inject_fold_maintenance_failure_for_test(after_graph);
        let receipt = store.commit_batch(batch("committed")).unwrap();
        assert_eq!(receipt.position, LogPosition(1));
        let failure = receipt.maintenance_failure.unwrap();
        assert_eq!(failure.committed_position, receipt.position);
        assert_eq!(failure.cause.code, "injected_fold_maintenance_failure");
        assert_eq!(store.latest_position(), receipt.position);
        assert!(store.pin(receipt.position).is_err());
        let verification = store.verify();
        assert!(!verification.ok);
        assert!(verification
            .problems
            .iter()
            .any(|problem| problem.kind == "committed_fold_maintenance_failed"));
        assert_eq!(
            store.pin_current().get_node("committed").position,
            if after_graph {
                receipt.position
            } else {
                LogPosition::ZERO
            }
        );
        let error = store.commit_batch(batch("do-not-append")).unwrap_err();
        assert_eq!(error.code, "committed_fold_maintenance_failed");
        assert!(error.message.contains("log position 1"));
        assert_eq!(store.latest_position(), receipt.position);
        drop(store);
        let reopened = FoldGraphStore::open(dir.path(), deployment).unwrap();
        assert_eq!(reopened.latest_position(), receipt.position);
        assert!(reopened.get_node("committed").is_some());
        assert!(reopened.get_node("do-not-append").is_none());
        assert!(reopened.maintenance_failure().is_none());
        assert!(reopened.verify().ok);
        assert!(reopened.pin(receipt.position).is_ok());
    }
}

#[test]
fn trait_adapters_report_known_commit_instead_of_missing_write_or_success() {
    for conditional in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let deployment = DeploymentId::new("test:committed-fold-adapter").unwrap();
        let mut store = FoldGraphStore::open(dir.path(), deployment.clone()).unwrap();
        store.inject_fold_maintenance_failure_for_test(false);
        let error = if conditional {
            store
                .commit_conditional_batch_atomic(vec![], batch("committed"))
                .unwrap_err()
        } else {
            GraphWrite::upsert_node(
                &mut store,
                NodeRecord::new("committed", ["Record"], json!({})),
            )
            .unwrap_err()
        };
        assert_eq!(error.code, "committed_fold_maintenance_failed");
        let failure = store.maintenance_failure().unwrap();
        assert_eq!(failure.committed_position, LogPosition(1));
        assert!(error.message.contains("log position 1"));
        drop(store);
        let reopened = FoldGraphStore::open(dir.path(), deployment).unwrap();
        assert!(reopened.get_node("committed").is_some());
        assert_eq!(reopened.latest_position(), LogPosition(1));
    }
}
