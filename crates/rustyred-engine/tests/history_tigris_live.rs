use rustyred_thg_core::{
    publish_history_interval, ChunkPublishDisposition, FoldGraphStore, FoldRuntime, GraphWrite,
    HistoryFold, NodeRecord, PublishByChunkHash, S3CompatibleChunkPublisher,
};
use serde_json::json;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use theorem_oplog::PhysicalLog;

#[test]
#[ignore = "requires THEOREM_RUN_TIGRIS_LIVE=1 and disposable Tigris credentials"]
fn history_chunks_publish_reload_and_cleanup_on_tigris() -> Result<(), Box<dyn Error>> {
    if std::env::var("THEOREM_RUN_TIGRIS_LIVE").as_deref() != Ok("1") {
        return Err("THEOREM_RUN_TIGRIS_LIVE must equal 1".into());
    }
    let owned_root = std::env::var("THEOREM_TIGRIS_TEST_PREFIX")
        .map_err(|_| "THEOREM_TIGRIS_TEST_PREFIX must identify a disposable owned prefix")?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let prefix = format!(
        "{}/history-v07-{}-{nonce}",
        owned_root.trim_matches('/'),
        std::process::id()
    );
    let publisher = S3CompatibleChunkPublisher::from_env(prefix)?;

    let directory = tempfile::tempdir()?;
    let mut graph = FoldGraphStore::open(directory.path(), deployment_id())?;
    graph.upsert_node(NodeRecord::new(
        "node:tigris-live",
        ["HistoryFixture"],
        json!({"proof": "live"}),
    ))?;
    drop(graph);

    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut history = FoldRuntime::for_log(HistoryFold::default(), &log)?;
    history.apply_all(&log)?;
    let interval = history.inner().intervals().last().unwrap();
    let runtime = tokio::runtime::Runtime::new()?;
    let receipts = runtime.block_on(publish_history_interval(
        &publisher,
        history.inner(),
        interval,
    ))?;
    assert!(!receipts.is_empty());
    assert!(receipts
        .iter()
        .all(|receipt| receipt.disposition == ChunkPublishDisposition::Published));

    for receipt in &receipts {
        let local = history.inner().chunk_bytes(receipt.address)?.unwrap();
        let remote = runtime
            .block_on(publisher.get_verified(receipt.address))?
            .expect("published Tigris object must reload");
        assert_eq!(remote, local);
        assert!(receipt.e_tag.is_some(), "live S3 receipt records an ETag");
    }

    let replayed = runtime.block_on(publish_history_interval(
        &publisher,
        history.inner(),
        interval,
    ))?;
    assert!(replayed
        .iter()
        .all(|receipt| receipt.disposition == ChunkPublishDisposition::AlreadyPresent));

    for receipt in &receipts {
        assert!(runtime.block_on(publisher.delete_owned(receipt.address))?);
        assert!(runtime
            .block_on(publisher.get_verified(receipt.address))?
            .is_none());
    }
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:history-tigris-live")
        .expect("static deployment id is valid")
}
