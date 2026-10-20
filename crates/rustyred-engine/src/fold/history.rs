//! Prolly-backed History fold over explicit physical-log intervals.

use super::{Fold, FoldCapabilities, FoldError, FoldId, FoldVersion};
use crate::chunk_store::{ChunkStore, MemChunkStore};
use crate::fold_quarantine::{FoldConsumeError, FoldRefusalCode};
use crate::graph::fold_store::{
    GraphFold, GraphIntentBatch, GRAPH_INTENT_NAME, GRAPH_INTENT_NAMESPACE, GRAPH_INTENT_VERSION,
};
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::snapshot::GraphSnapshot;
use crate::hash::Hash;
use crate::versioned_graph::{
    canonical_content_object_preimage, checkout_graph_version, compile_graph_pack,
    compile_graph_pack_incremental_with_store, retarget_graph_pack_version, update_graph_ref,
    CompiledGraphPack, GraphCompileOptions, GraphVersionRepository, ProllyReader, StoredProllyNode,
    DEFAULT_GRAPH_BRANCH,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use theorem_oplog::{LogId, LogPosition, LogRecord};

pub const FOLD_ID: FoldId = FoldId::History;
pub const HISTORY_FOLD_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryInterval {
    pub log_id: LogId,
    pub from_exclusive: LogPosition,
    pub through: LogPosition,
    pub commit_hash: Hash,
    pub tree_hash: Hash,
    pub graph_version: u64,
    pub changed_object_keys: Vec<String>,
    pub delta_chunk_hashes: Vec<Hash>,
    /// Complete tree and logical-object closure required by this pinned root.
    pub chunk_hashes: Vec<Hash>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HistoryCheckout {
    pub interval: HistoryInterval,
    pub snapshot: GraphSnapshot,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryQuery {
    Interval,
    Checkout,
    Intervals,
}

#[derive(Deserialize, Serialize)]
struct HistoryCheckpoint {
    graph: Vec<u8>,
    repository: GraphVersionRepository,
    intervals: Vec<HistoryInterval>,
}

pub struct HistoryFold {
    graph: GraphFold,
    repository: GraphVersionRepository,
    intervals: Vec<HistoryInterval>,
    chunks: MemChunkStore,
}

impl Default for HistoryFold {
    fn default() -> Self {
        Self {
            graph: GraphFold::default(),
            repository: GraphVersionRepository::default(),
            intervals: Vec::new(),
            chunks: MemChunkStore::new(),
        }
    }
}

impl HistoryFold {
    pub fn intervals(&self) -> &[HistoryInterval] {
        &self.intervals
    }

    pub fn repository(&self) -> &GraphVersionRepository {
        &self.repository
    }

    pub fn interval_at(&self, position: LogPosition) -> Option<&HistoryInterval> {
        self.intervals
            .iter()
            .rev()
            .find(|interval| interval.through <= position)
    }

    pub fn checkout_at(&self, position: LogPosition) -> Result<Option<HistoryCheckout>, FoldError> {
        let Some(interval) = self.interval_at(position).cloned() else {
            return Ok(None);
        };
        let checkout =
            checkout_graph_version(&self.repository, &interval.commit_hash.to_prefixed())
                .map_err(history_error)?;
        Ok(Some(HistoryCheckout {
            interval,
            snapshot: checkout.snapshot,
        }))
    }

    pub fn chunk_bytes(&self, address: Hash) -> Result<Option<Vec<u8>>, FoldError> {
        let tree_chunk = self
            .chunks
            .get(address)
            .map(|bytes| bytes.map(|bytes| bytes.to_vec()))
            .map_err(history_error)?;
        if tree_chunk.is_some() {
            return Ok(tree_chunk);
        }
        self.repository
            .objects
            .get(&address)
            .map(canonical_content_object_preimage)
            .transpose()
            .map_err(history_error)
    }

    fn compile_interval(
        &self,
        entry: &LogRecord,
        before: &GraphSnapshot,
        after: &GraphSnapshot,
    ) -> Result<(CompiledGraphPack, Vec<String>), FoldError> {
        let options = history_options(entry.position);
        let Some(prior) = self.current_pack() else {
            return compile_graph_pack(after, options)
                .map(|pack| {
                    let changed = pack
                        .objects
                        .iter()
                        .map(|object| object.key.clone())
                        .collect();
                    (pack, changed)
                })
                .map_err(history_error);
        };
        let batch = snapshot_delta(before, after);
        let incremental =
            compile_graph_pack_incremental_with_store(&self.chunks, prior, &batch, options)
                .map_err(history_error)?;
        let changed = incremental.changed_object_keys;
        let pack =
            retarget_graph_pack_version(incremental.pack, after.version).map_err(history_error)?;
        Ok((pack, changed))
    }

    fn current_pack(&self) -> Option<&CompiledGraphPack> {
        let head = self
            .repository
            .refs
            .iter()
            .find(|reference| reference.name == DEFAULT_GRAPH_BRANCH)?;
        self.repository
            .packs
            .iter()
            .find(|pack| pack.commit.commit_hash == head.commit_hash)
    }

    fn install_chunks(&self, pack: &CompiledGraphPack) -> Result<Vec<Hash>, FoldError> {
        let chunks = pack
            .tree_chunks
            .iter()
            .map(|chunk| (chunk.address, chunk.bytes.as_slice()))
            .collect::<Vec<_>>();
        self.chunks.write(&chunks).map_err(history_error)?;
        Ok(pack.tree_chunks.iter().map(|chunk| chunk.address).collect())
    }

    fn content_closure(&self, pack: &CompiledGraphPack) -> Result<Vec<Hash>, FoldError> {
        let mut reader = ProllyReader::new(&self.chunks);
        let mut pending = vec![pack.tree.root];
        let mut tree_addresses = BTreeMap::<Hash, ()>::new();
        let mut object_addresses = BTreeMap::<Hash, ()>::new();
        while let Some(address) = pending.pop() {
            if tree_addresses.insert(address, ()).is_some() {
                continue;
            }
            let node = reader.read(address).map_err(history_error)?;
            match node.as_ref() {
                StoredProllyNode::Leaf(entries) => {
                    for entry in entries {
                        object_addresses.insert(entry.object_hash, ());
                    }
                }
                StoredProllyNode::Parent(children) => {
                    pending.extend(children.iter().map(|child| child.hash));
                }
            }
        }
        tree_addresses.extend(object_addresses);
        Ok(tree_addresses.into_keys().collect())
    }

    fn restore_chunks(&mut self) -> Result<(), FoldError> {
        let chunks = self
            .repository
            .tree_chunks
            .iter()
            .map(|(address, bytes)| (*address, bytes.as_slice()))
            .collect::<Vec<_>>();
        let store = MemChunkStore::new();
        store.write(&chunks).map_err(history_error)?;
        self.chunks = store;
        Ok(())
    }
}

impl Fold for HistoryFold {
    fn id(&self) -> FoldId {
        FOLD_ID
    }

    fn version(&self) -> FoldVersion {
        FoldVersion::new(HISTORY_FOLD_VERSION).expect("History fold version is non-zero")
    }

    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        decode_history_batch(entry)?;
        let before = self.graph.store().snapshot();
        let mut staged_graph = self.graph.clone();
        staged_graph.apply(entry)?;
        let after = staged_graph.store().snapshot();
        let (pack, changed_object_keys) = self
            .compile_interval(entry, &before, &after)
            .map_err(|error| FoldConsumeError::fatal(error.to_string()))?;
        let delta_chunk_hashes = self
            .install_chunks(&pack)
            .map_err(|error| FoldConsumeError::fatal(error.to_string()))?;
        let chunk_hashes = self
            .content_closure(&pack)
            .map_err(|error| FoldConsumeError::fatal(error.to_string()))?;
        let update = update_graph_ref(
            self.repository.clone(),
            pack.clone(),
            Some(DEFAULT_GRAPH_BRANCH.to_string()),
            Some(u128::from(entry.position.0)),
        )
        .map_err(|error| FoldConsumeError::fatal(error.message))?;
        let from_exclusive = self
            .intervals
            .last()
            .map_or(LogPosition::ZERO, |interval| interval.through);
        self.graph = staged_graph;
        self.repository = update.repository;
        self.intervals.push(HistoryInterval {
            log_id: entry.log_id,
            from_exclusive,
            through: entry.position,
            commit_hash: pack.commit.commit_hash,
            tree_hash: pack.tree.root,
            graph_version: after.version,
            changed_object_keys,
            delta_chunk_hashes,
            chunk_hashes,
        });
        Ok(())
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        let query = serde_json::from_value::<HistoryQuery>(query.clone())
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        match query {
            HistoryQuery::Interval => serde_json::to_value(self.interval_at(position)),
            HistoryQuery::Checkout => serde_json::to_value(self.checkout_at(position)?),
            HistoryQuery::Intervals => serde_json::to_value(
                self.intervals
                    .iter()
                    .filter(|interval| interval.through <= position)
                    .collect::<Vec<_>>(),
            ),
        }
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError> {
        let mut rebuilt = Self::default();
        for record in pinned_records {
            match rebuilt.apply(record) {
                Ok(()) | Err(FoldConsumeError::Refused(_)) => {}
                Err(FoldConsumeError::Fatal(reason)) => {
                    return Err(FoldError::FatalConsume(reason));
                }
            }
        }
        rebuilt.query(key, position)
    }

    fn checkpoint(&self) -> Result<Vec<u8>, FoldError> {
        let graph = self.graph.checkpoint()?;
        serde_json::to_vec(&HistoryCheckpoint {
            graph,
            repository: self.repository.clone(),
            intervals: self.intervals.clone(),
        })
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        let checkpoint = serde_json::from_slice::<HistoryCheckpoint>(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        let mut graph = GraphFold::default();
        graph.restore(&checkpoint.graph)?;
        self.graph = graph;
        self.repository = checkpoint.repository;
        self.intervals = checkpoint.intervals;
        self.restore_chunks()
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        *self = Self::default();
        Ok(())
    }
}

fn history_options(position: LogPosition) -> GraphCompileOptions {
    GraphCompileOptions {
        name: Some("history-fold".to_string()),
        branch: Some(DEFAULT_GRAPH_BRANCH.to_string()),
        author: Some("rustyred-history-fold".to_string()),
        message: Some(format!("materialize log through position {}", position.0)),
        timestamp_unix_ms: Some(u128::from(position.0)),
        include_payloads: true,
        exclude_local_only: false,
        sync_scope_only: false,
        ..GraphCompileOptions::default()
    }
}

fn decode_history_batch(entry: &LogRecord) -> Result<GraphIntentBatch, FoldConsumeError> {
    if entry.intent.kind.namespace() != GRAPH_INTENT_NAMESPACE
        || entry.intent.kind.name() != GRAPH_INTENT_NAME
        || entry.intent.kind.version() != GRAPH_INTENT_VERSION
    {
        return Err(FoldConsumeError::refused(
            FoldRefusalCode::UnsupportedIntent,
            format!("expected {GRAPH_INTENT_NAMESPACE}.{GRAPH_INTENT_NAME}@{GRAPH_INTENT_VERSION}"),
        ));
    }
    serde_json::from_value::<GraphIntentBatch>(entry.intent.payload.clone()).map_err(|error| {
        FoldConsumeError::refused(
            FoldRefusalCode::InvalidPayload,
            format!("history graph batch decode failed: {error}"),
        )
    })
}

fn snapshot_delta(before: &GraphSnapshot, after: &GraphSnapshot) -> GraphMutationBatch {
    let before_nodes = before
        .nodes
        .iter()
        .map(|node| (&node.id, node))
        .collect::<BTreeMap<_, _>>();
    let after_nodes = after
        .nodes
        .iter()
        .map(|node| (&node.id, node))
        .collect::<BTreeMap<_, _>>();
    let before_edges = before
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    let after_edges = after
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    let mut mutations = Vec::new();
    for id in before_edges.keys() {
        if !after_edges.contains_key(id) {
            mutations.push(GraphMutation::EdgeDelete((*id).clone()));
        }
    }
    for id in before_nodes.keys() {
        if !after_nodes.contains_key(id) {
            mutations.push(GraphMutation::NodeDelete((*id).clone()));
        }
    }
    for (id, node) in after_nodes {
        if before_nodes.get(id).copied() != Some(node) {
            mutations.push(GraphMutation::NodeUpsert(node.clone()));
        }
    }
    for (id, edge) in after_edges {
        if before_edges.get(id).copied() != Some(edge) {
            mutations.push(GraphMutation::EdgeUpsert(edge.clone()));
        }
    }
    GraphMutationBatch::new(mutations)
}

fn history_error(error: impl ToString) -> FoldError {
    FoldError::Codec(error.to_string())
}
