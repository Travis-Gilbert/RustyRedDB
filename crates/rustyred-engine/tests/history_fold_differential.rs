use async_trait::async_trait;
use rustyred_thg_core::{
    publish_history_interval, ChunkPublishDisposition, ChunkPublishReceipt, EdgeRecord,
    FoldGraphStore, FoldRuntime, GraphFold, GraphMutation, GraphMutationBatch, GraphSnapshotSource,
    GraphWrite, Hash, HistoryFold, NodeRecord, PublishByChunkHash, RestoreDisposition,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::error::Error;
use std::sync::Mutex;
use theorem_oplog::PhysicalLog;

#[derive(Default)]
struct MemoryPublisher {
    chunks: Mutex<BTreeMap<Hash, Vec<u8>>>,
}

#[async_trait]
impl PublishByChunkHash for MemoryPublisher {
    async fn put_if_absent(
        &self,
        address: Hash,
        bytes: &[u8],
    ) -> rustyred_thg_core::GraphStoreResult<ChunkPublishReceipt> {
        if Hash::of(bytes) != address {
            return Err(rustyred_thg_core::GraphStoreError::new(
                "fixture_hash_mismatch",
                "fixture publisher received bytes under the wrong address",
            ));
        }
        let mut chunks = self.chunks.lock().unwrap();
        let disposition = if let Some(existing) = chunks.get(&address) {
            assert_eq!(existing, bytes);
            ChunkPublishDisposition::AlreadyPresent
        } else {
            chunks.insert(address, bytes.to_vec());
            ChunkPublishDisposition::Published
        };
        Ok(ChunkPublishReceipt {
            address,
            key: address.to_string(),
            disposition,
            e_tag: None,
        })
    }

    async fn get_verified(
        &self,
        address: Hash,
    ) -> rustyred_thg_core::GraphStoreResult<Option<Vec<u8>>> {
        Ok(self.chunks.lock().unwrap().get(&address).cloned())
    }

    async fn delete_owned(&self, address: Hash) -> rustyred_thg_core::GraphStoreResult<bool> {
        Ok(self.chunks.lock().unwrap().remove(&address).is_some())
    }
}

fn node(id: &str, value: i64) -> NodeRecord {
    NodeRecord::new(id, ["HistoryFixture"], json!({"value": value}))
}

fn live_checkout_shape(
    mut snapshot: rustyred_thg_core::GraphSnapshot,
) -> rustyred_thg_core::GraphSnapshot {
    snapshot.nodes.retain(|node| !node.tombstone);
    snapshot.edges.retain(|edge| !edge.tombstone);
    snapshot
}

fn build_log(
    directory: &std::path::Path,
) -> Result<Vec<rustyred_thg_core::GraphSnapshot>, Box<dyn Error>> {
    let mut store = FoldGraphStore::open(directory, deployment_id())?;
    let mut snapshots = Vec::new();

    store.commit_batch(GraphMutationBatch::new([
        GraphMutation::NodeUpsert(node("node:a", 1)),
        GraphMutation::NodeUpsert(node("node:b", 2)),
    ]))?;
    snapshots.push(store.graph_snapshot()?);

    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "node:a",
        "LINKS",
        "node:b",
        json!({"rank": 1}),
    ))?;
    snapshots.push(store.graph_snapshot()?);

    store.upsert_node(node("node:a", 3))?;
    snapshots.push(store.graph_snapshot()?);

    store.delete_node("node:b")?;
    snapshots.push(store.graph_snapshot()?);
    Ok(snapshots)
}

#[test]
fn history_fold_matches_every_graph_prefix_and_restores_checkpoint() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let expected_snapshots = build_log(directory.path())?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut history = FoldRuntime::for_log(HistoryFold::default(), &log)?;
    history.apply_all(&log)?;

    assert_eq!(history.inner().intervals().len(), expected_snapshots.len());
    for (index, expected) in expected_snapshots.iter().enumerate() {
        let position = theorem_oplog::LogPosition((index + 1) as u64);
        let checkout = history
            .inner()
            .checkout_at(position)?
            .expect("accepted graph record has a History checkout");
        assert_eq!(&checkout.snapshot, &live_checkout_shape(expected.clone()));
        assert_eq!(checkout.interval.through, position);

        let mut graph_oracle = FoldRuntime::for_log(GraphFold::default(), &log)?;
        graph_oracle.apply_range(&log, position)?;
        assert_eq!(
            &checkout.snapshot,
            &live_checkout_shape(graph_oracle.inner().store().snapshot())
        );
    }

    let intervals = history.inner().intervals();
    assert!(!intervals[0].delta_chunk_hashes.is_empty());
    assert!(intervals[0].chunk_hashes.len() > intervals[0].delta_chunk_hashes.len());
    assert!(intervals
        .windows(2)
        .all(|pair| pair[1].from_exclusive == pair[0].through));
    assert!(history.inner().repository().commits.len() >= intervals.len());
    assert_eq!(
        history.inner().repository().tree_chunks.len(),
        intervals
            .iter()
            .flat_map(|interval| &interval.delta_chunk_hashes)
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );

    let checkpoint = history.checkpoint(&log)?;
    let mut restored = FoldRuntime::for_log(HistoryFold::default(), &log)?;
    assert!(matches!(
        restored.restore_or_rebuild(&checkpoint, &log)?,
        RestoreDisposition::Restored { .. }
    ));
    assert_eq!(
        restored
            .inner()
            .checkout_at(log.latest_position())?
            .unwrap(),
        history.inner().checkout_at(log.latest_position())?.unwrap()
    );
    for interval in restored.inner().intervals() {
        for address in &interval.chunk_hashes {
            assert!(restored.inner().chunk_bytes(*address)?.is_some());
        }
    }
    Ok(())
}

#[test]
fn history_rebuild_is_deterministic_and_chunk_publish_is_idempotent() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    build_log(directory.path())?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut first = FoldRuntime::for_log(HistoryFold::default(), &log)?;
    let mut second = FoldRuntime::for_log(HistoryFold::default(), &log)?;
    first.apply_all(&log)?;
    second.apply_all(&log)?;

    assert_eq!(first.inner().intervals(), second.inner().intervals());
    assert_eq!(first.inner().repository(), second.inner().repository());

    let publisher = MemoryPublisher::default();
    let interval = first.inner().intervals().last().unwrap();
    let runtime = tokio::runtime::Runtime::new()?;
    let first_receipts = runtime.block_on(publish_history_interval(
        &publisher,
        first.inner(),
        interval,
    ))?;
    let second_receipts = runtime.block_on(publish_history_interval(
        &publisher,
        first.inner(),
        interval,
    ))?;
    assert!(first_receipts
        .iter()
        .all(|receipt| receipt.disposition == ChunkPublishDisposition::Published));
    assert!(second_receipts
        .iter()
        .all(|receipt| receipt.disposition == ChunkPublishDisposition::AlreadyPresent));
    for receipt in first_receipts {
        let local = first.inner().chunk_bytes(receipt.address)?.unwrap();
        let remote = runtime
            .block_on(publisher.get_verified(receipt.address))?
            .unwrap();
        assert_eq!(remote, local);
    }
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:history-fold-differential").unwrap()
}
