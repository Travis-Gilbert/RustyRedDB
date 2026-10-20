use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

fn append_derived(calls: Arc<AtomicUsize>) -> GraphBatchPreparer {
    GraphBatchPreparer::new(move |_, original| {
        calls.fetch_add(1, Ordering::SeqCst);
        let mut batch = original.clone();
        batch
            .mutations
            .push(GraphMutation::NodeUpsert(NodeRecord::new(
                "derived",
                ["Resolution"],
                json!({"payload":"derived same transaction"}),
            )));
        Ok(batch)
    })
}
fn incoming() -> GraphMutationBatch {
    GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
        "target",
        ["Memory"],
        json!({"value":1}),
    ))])
}

#[test]
fn prepared_batch_runs_once_and_retains_final_checked_mutations() -> GraphStoreResult<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut store = RedCoreGraphStore::memory();
    store.attach_batch_preparer(append_derived(calls.clone()));
    let prepared = store.prepare_batch(incoming())?;
    assert_eq!(prepared.batch().mutations.len(), 2);
    let receipt = store.commit_prepared_batch_with_metadata(
        prepared,
        GraphCommitMetadata {
            author: "verified fixture principal".into(),
            message: "one producer batch".into(),
            timestamp_unix_ms: 1234,
        },
    )?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(receipt.transaction.writes[0].id, "target");
    assert_eq!(receipt.transaction.writes[1].id, "derived");
    let history = store.pin_graph_history()?;
    assert_eq!(history.page(None, 10)?.entries.len(), 1);
    assert_eq!(history.head_revision()?.commit().unwrap(), &receipt.commit);
    assert_eq!(receipt.commit.author, "verified fixture principal");
    let before = store.version();
    let failed = store.compare_and_set_node_property_checked(
        "target",
        "value",
        &json!(1),
        json!(2),
        |_, batch| {
            assert_eq!(batch.mutations.len(), 2);
            Err(GraphStoreError::new(
                "quota_fixture",
                "final augmented quota refused",
            ))
        },
    );
    assert_eq!(failed.unwrap_err().code, "quota_fixture");
    assert_eq!(store.version(), before);
    assert_eq!(store.get_node("target")?.unwrap().properties["value"], 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn prepared_batch_guards_run_before_preparer_and_certificates_cannot_cross_writers(
) -> GraphStoreResult<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut store = RedCoreGraphStore::memory();
    store.commit_batch(incoming())?;
    store.attach_batch_preparer(append_derived(calls.clone()));
    assert!(store
        .insert_node_if_absent(NodeRecord::new("target", ["Memory"], json!({})))?
        .is_none());
    assert!(store
        .compare_and_set_node_property("target", "value", &json!(99), json!(2))?
        .is_none());
    assert!(store.commit_batch_if_version(999, incoming()).is_err());
    assert!(store
        .commit_batch(incoming().with_expected_root(Hash::of(b"wrong root")))
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let prepared = store.prepare_batch(incoming())?;
    let mut another = RedCoreGraphStore::memory();
    assert_eq!(
        another.commit_prepared_batch(prepared).unwrap_err().code,
        "invalid_batch_preparation"
    );
    let stale = store.prepare_batch(incoming())?;
    store.commit_batch(incoming())?;
    let cursor = store.version();
    assert!(store.commit_prepared_batch(stale).is_err());
    assert_eq!(store.version(), cursor);
    let old_policy = store.prepare_batch(incoming())?;
    store.attach_batch_preparer(append_derived(calls));
    assert_eq!(
        store.commit_prepared_batch(old_policy).unwrap_err().code,
        "invalid_batch_preparation"
    );
    Ok(())
}

#[test]
fn prepared_batch_preserves_conditional_guard_and_journal_atomicity() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    };
    let mut store = RedCoreGraphStore::open(dir.path(), options.clone())?;
    store.commit_batch(incoming())?;
    let calls = Arc::new(AtomicUsize::new(0));
    store.attach_batch_preparer(append_derived(calls.clone()));
    let condition = |expected| ConditionalNodePropertyUpdate {
        node_id: "target".into(),
        property: "value".into(),
        expected: json!(expected),
        new_value: json!(2),
    };
    assert!(!store
        .commit_conditional_batch_atomic(vec![condition(99)], GraphMutationBatch::default())?);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let committed = store
        .commit_conditional_batch_atomic_checked(
            vec![condition(1)],
            GraphMutationBatch::default(),
            |_, batch| {
                assert_eq!(batch.mutations.len(), 2);
                Ok(())
            },
        )?
        .unwrap();
    assert_eq!(committed.mutations.len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let cursor = store.version();
    store.inject_journal_append_failure_for_test(0, 0, false)?;
    assert!(store
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new("failed", ["Memory"], json!({}))
        )]))
        .is_err());
    assert_eq!(store.version(), cursor);
    drop(store);
    let reopened = RedCoreGraphStore::open(dir.path(), options)?;
    assert_eq!(reopened.version(), cursor);
    assert!(reopened.get_node("failed")?.is_none());
    assert_eq!(reopened.get_node("target")?.unwrap().properties["value"], 2);
    assert!(reopened.get_node("derived")?.is_some());
    Ok(())
}

#[test]
fn prepared_batch_cannot_reorder_or_replace_original_user_mutations() {
    let mut store = RedCoreGraphStore::memory();
    store.attach_batch_preparer(GraphBatchPreparer::new(|_, _| {
        Ok(GraphMutationBatch::default())
    }));
    assert_eq!(
        store.prepare_batch(incoming()).unwrap_err().code,
        "invalid_batch_preparation"
    );
    store.attach_batch_preparer(GraphBatchPreparer::new(|_, batch| {
        Ok(batch.clone().with_expected_root(Hash::of(b"replacement")))
    }));
    assert_eq!(
        store.prepare_batch(incoming()).unwrap_err().code,
        "invalid_batch_preparation"
    );
    assert!(store.get_node("target").unwrap().is_none());
}
