//! GraphStore facade backed by Facts, Graph, and Algebra folds over `theorem-oplog`.

use super::attribution::GraphStats;
use super::capability::{
    GraphAtomic, GraphRead, GraphResidency, GraphSnapshotSource, GraphTtl,
    GraphVectorPayloadAccess, GraphVerify, GraphWrite,
};
use super::error::{GraphStoreError, GraphStoreResult};
use super::in_memory::InMemoryGraphStore;
use super::mutation::{
    prepare_conditional_batch, ConditionalNodePropertyUpdate, GraphMutation, GraphMutationBatch,
    GraphWriteResult,
};
use super::query::{NeighborHit, NeighborQuery, NodeQuery};
use super::record::{EdgeRecord, NodeRecord};
use super::snapshot::GraphSnapshot;
use super::verify::{GraphRebuildReport, VerifyReport};
use crate::fold::algebra::AlgebraFold;
use crate::fold::{Fold, FoldCapabilities, FoldError, FoldId, FoldRuntime, FoldVersion};
use crate::fold_quarantine::{FoldConsumeError, FoldRefusalCode, QuarantineReceipt};
use crate::graph::ttl::{node_with_ttl_property, now_ms};
use crate::tensor_derivation::TensorBlockPayloadStore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use theorem_oplog::{
    AccessPath, CommitOutcome, DeploymentId, IntentKind, LogIntent, LogPosition, LogRecord,
    OpDigest, PhysicalLog, Proposal, ProposalCommitter, ReadSet, WriteSet,
};

pub const GRAPH_INTENT_NAMESPACE: &str = "rustyred.graph";
pub const GRAPH_INTENT_NAME: &str = "semantic_batch";
pub const GRAPH_INTENT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "intent", rename_all = "snake_case")]
pub enum GraphSemanticChange {
    NodeObserved { record: NodeRecord },
    RelationshipObserved { record: EdgeRecord },
    NodeRetracted { id: String },
    RelationshipRetracted { id: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphIntentBatch {
    pub changes: Vec<GraphSemanticChange>,
}

impl GraphIntentBatch {
    pub fn from_mutations(batch: GraphMutationBatch) -> GraphStoreResult<Self> {
        if batch.mutations.is_empty() {
            return Err(GraphStoreError::empty_transaction());
        }
        let changes = batch
            .mutations
            .into_iter()
            .map(|mutation| match mutation {
                GraphMutation::NodeUpsert(record) => GraphSemanticChange::NodeObserved { record },
                GraphMutation::EdgeUpsert(record) => {
                    GraphSemanticChange::RelationshipObserved { record }
                }
                GraphMutation::NodeDelete(id) => GraphSemanticChange::NodeRetracted { id },
                GraphMutation::EdgeDelete(id) => GraphSemanticChange::RelationshipRetracted { id },
            })
            .collect();
        Ok(Self {
            changes: normalize_changes(changes),
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FactTuple {
    pub entity: String,
    pub attribute: String,
    pub value: Value,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub asserted_at: LogPosition,
    pub retracted_at: Option<LogPosition>,
    pub epistemic_type: Option<String>,
    pub confidence: Option<f64>,
    pub provenance: Option<Value>,
    pub source_record_id: theorem_oplog::RecordId,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct FactKey {
    entity: String,
    attribute: String,
    value_token: String,
    asserted_at: LogPosition,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FactsFold {
    eavt: BTreeMap<FactKey, FactTuple>,
    current_by_entity: BTreeMap<String, BTreeSet<FactKey>>,
    by_entity_attribute: BTreeMap<(String, String), BTreeSet<FactKey>>,
    by_provenance_source: BTreeMap<String, BTreeSet<FactKey>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FactsFoldCheckpoint {
    tuples: Vec<FactTuple>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FactsQuery {
    pub entity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
}

impl FactsFold {
    pub fn facts_at(&self, query: &FactsQuery, position: LogPosition) -> Vec<FactTuple> {
        let candidates = if let Some(attribute) = &query.attribute {
            self.by_entity_attribute
                .get(&(query.entity.clone(), attribute.clone()))
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        } else {
            self.eavt
                .keys()
                .filter(|key| key.entity == query.entity)
                .collect::<Vec<_>>()
        };
        candidates
            .into_iter()
            .filter_map(|key| self.eavt.get(key))
            .filter(|tuple| {
                tuple.asserted_at <= position
                    && tuple
                        .retracted_at
                        .is_none_or(|retracted_at| retracted_at > position)
            })
            .cloned()
            .collect()
    }

    pub fn eavt_len(&self) -> usize {
        self.eavt.len()
    }

    pub fn provenance_source_count(&self) -> usize {
        self.by_provenance_source.len()
    }

    fn validate_batch(batch: &GraphIntentBatch) -> Result<(), FoldConsumeError> {
        for change in &batch.changes {
            match change {
                GraphSemanticChange::NodeObserved { record } => {
                    require_identifier(&record.id, "node.id")?;
                    if !record.properties.is_object() {
                        return Err(FoldConsumeError::refused(
                            FoldRefusalCode::SchemaViolation,
                            "node properties must be an object",
                        ));
                    }
                }
                GraphSemanticChange::RelationshipObserved { record } => {
                    require_identifier(&record.id, "edge.id")?;
                    require_identifier(&record.from_id, "edge.from_id")?;
                    require_identifier(&record.to_id, "edge.to_id")?;
                    require_identifier(&record.edge_type, "edge.type")?;
                    if !record.properties.is_object() {
                        return Err(FoldConsumeError::refused(
                            FoldRefusalCode::SchemaViolation,
                            "edge properties must be an object",
                        ));
                    }
                }
                GraphSemanticChange::NodeRetracted { id } => {
                    require_identifier(id, "node.id")?;
                }
                GraphSemanticChange::RelationshipRetracted { id } => {
                    require_identifier(id, "edge.id")?;
                }
            }
        }
        Ok(())
    }

    fn apply_batch(
        &mut self,
        batch: &GraphIntentBatch,
        record: &LogRecord,
    ) -> Result<(), FoldConsumeError> {
        for change in normalize_changes(batch.changes.clone()) {
            match change {
                GraphSemanticChange::NodeObserved { record: node } => {
                    require_identifier(&node.id, "node.id")?;
                    self.replace_entity(
                        format!("node:{}", node.id),
                        node_facts(&node, record)?,
                        record.position,
                    );
                }
                GraphSemanticChange::RelationshipObserved { record: edge } => {
                    require_identifier(&edge.id, "edge.id")?;
                    require_identifier(&edge.from_id, "edge.from_id")?;
                    require_identifier(&edge.to_id, "edge.to_id")?;
                    require_identifier(&edge.edge_type, "edge.type")?;
                    self.replace_entity(
                        format!("edge:{}", edge.id),
                        edge_facts(&edge, record)?,
                        record.position,
                    );
                }
                GraphSemanticChange::RelationshipRetracted { id } => {
                    require_identifier(&id, "edge.id")?;
                    self.retract_entity(&format!("edge:{id}"), record.position);
                }
                GraphSemanticChange::NodeRetracted { id } => {
                    require_identifier(&id, "node.id")?;
                    self.retract_entity(&format!("node:{id}"), record.position);
                    let edge_entities = self
                        .current_by_entity
                        .keys()
                        .filter(|entity| entity.starts_with("edge:"))
                        .filter(|entity| {
                            ["from", "to"].into_iter().any(|attribute| {
                                self.facts_at(
                                    &FactsQuery {
                                        entity: (*entity).clone(),
                                        attribute: Some(attribute.to_string()),
                                    },
                                    record.position,
                                )
                                .iter()
                                .any(|tuple| tuple.value == Value::String(id.clone()))
                            })
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    for entity in edge_entities {
                        self.retract_entity(&entity, record.position);
                    }
                }
            }
        }
        Ok(())
    }

    fn replace_entity(&mut self, entity: String, facts: Vec<FactTuple>, position: LogPosition) {
        self.retract_entity(&entity, position);
        for tuple in facts {
            self.insert_tuple(tuple);
        }
    }

    fn insert_tuple(&mut self, tuple: FactTuple) {
        let key = FactKey {
            entity: tuple.entity.clone(),
            attribute: tuple.attribute.clone(),
            value_token: canonical_value_token(&tuple.value),
            asserted_at: tuple.asserted_at,
        };
        if tuple.retracted_at.is_none() {
            self.current_by_entity
                .entry(tuple.entity.clone())
                .or_default()
                .insert(key.clone());
        }
        self.by_entity_attribute
            .entry((tuple.entity.clone(), tuple.attribute.clone()))
            .or_default()
            .insert(key.clone());
        if let Some(source_id) = tuple
            .provenance
            .as_ref()
            .and_then(|value| value.get("source_id"))
            .and_then(Value::as_str)
        {
            self.by_provenance_source
                .entry(source_id.to_string())
                .or_default()
                .insert(key.clone());
        }
        self.eavt.insert(key, tuple);
    }

    fn retract_entity(&mut self, entity: &str, position: LogPosition) {
        if let Some(keys) = self.current_by_entity.remove(entity) {
            for key in keys {
                if let Some(tuple) = self.eavt.get_mut(&key) {
                    tuple.retracted_at = Some(position);
                }
            }
        }
    }
}

impl Fold for FactsFold {
    fn id(&self) -> FoldId {
        FoldId::Facts
    }

    fn version(&self) -> FoldVersion {
        FoldVersion::new(1).expect("Facts fold version is non-zero")
    }

    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        let batch = decode_graph_batch(entry)?;
        let mut staged = self.clone();
        staged.apply_batch(&batch, entry)?;
        *self = staged;
        Ok(())
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        let query = serde_json::from_value::<FactsQuery>(query.clone())
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        serde_json::to_value(self.facts_at(&query, position))
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
        serde_json::to_vec(&FactsFoldCheckpoint {
            tuples: self.eavt.values().cloned().collect(),
        })
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        let checkpoint = serde_json::from_slice::<FactsFoldCheckpoint>(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        *self = Self::default();
        for tuple in checkpoint.tuples {
            self.insert_tuple(tuple);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        *self = Self::default();
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct GraphFold {
    store: InMemoryGraphStore,
    writes_by_position: BTreeMap<LogPosition, Vec<GraphWriteResult>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GraphFoldCheckpoint {
    snapshot: GraphSnapshot,
    writes_by_position: BTreeMap<LogPosition, Vec<GraphWriteResult>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphFoldQuery {
    Node { id: String },
    Edge { id: String },
    Nodes { query: NodeQuery },
    Neighbors { query: NeighborQuery },
}

impl GraphFold {
    pub fn store(&self) -> &InMemoryGraphStore {
        &self.store
    }

    pub fn writes_at(&self, position: LogPosition) -> &[GraphWriteResult] {
        self.writes_by_position
            .get(&position)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn validate_batch(&self, batch: &GraphIntentBatch) -> GraphStoreResult<()> {
        let mut staged = self.store.clone();
        apply_graph_changes(&mut staged, &batch.changes).map(|_| ())
    }
}

impl Fold for GraphFold {
    fn id(&self) -> FoldId {
        FoldId::Graph
    }

    fn version(&self) -> FoldVersion {
        FoldVersion::new(1).expect("Graph fold version is non-zero")
    }

    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        let batch = decode_graph_batch(entry)?;
        let mut staged = self.store.clone();
        let writes = apply_graph_changes(&mut staged, &batch.changes).map_err(|error| {
            FoldConsumeError::refused(
                error_code(&error),
                format!("{}: {}", error.code, error.message),
            )
        })?;
        self.store = staged;
        self.writes_by_position.insert(entry.position, writes);
        Ok(())
    }

    fn query(&self, query: &Value, _position: LogPosition) -> Result<Value, FoldError> {
        let query = serde_json::from_value::<GraphFoldQuery>(query.clone())
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        let value = match query {
            GraphFoldQuery::Node { id } => serde_json::to_value(self.store.get_node_record(&id)),
            GraphFoldQuery::Edge { id } => serde_json::to_value(self.store.get_edge_record(&id)),
            GraphFoldQuery::Nodes { query } => serde_json::to_value(self.store.query_nodes(query)),
            GraphFoldQuery::Neighbors { query } => {
                serde_json::to_value(self.store.neighbors(query))
            }
        };
        value.map_err(|error| FoldError::Codec(error.to_string()))
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
        serde_json::to_vec(&GraphFoldCheckpoint {
            snapshot: self.store.snapshot(),
            writes_by_position: self.writes_by_position.clone(),
        })
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        let checkpoint = serde_json::from_slice::<GraphFoldCheckpoint>(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        self.store = InMemoryGraphStore::from_snapshot(checkpoint.snapshot)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        self.writes_by_position = checkpoint.writes_by_position;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        *self = Self::default();
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldGraphTransaction {
    pub position: LogPosition,
    pub writes: Vec<GraphWriteResult>,
    pub replayed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointReadExecution {
    DirectGraphFoldIndex,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionPinned<T> {
    pub position: LogPosition,
    pub execution: PointReadExecution,
    pub value: T,
}

#[derive(Debug)]
pub struct PinnedGraphRead<'a> {
    store: &'a FoldGraphStore,
    position: LogPosition,
}

impl PinnedGraphRead<'_> {
    pub fn get_node(&self, id: &str) -> PositionPinned<Option<NodeRecord>> {
        PositionPinned {
            position: self.position,
            execution: PointReadExecution::DirectGraphFoldIndex,
            value: self.store.get_node_record(id),
        }
    }

    pub fn get_edge(&self, id: &str) -> PositionPinned<Option<EdgeRecord>> {
        PositionPinned {
            position: self.position,
            execution: PointReadExecution::DirectGraphFoldIndex,
            value: self.store.get_edge_record(id),
        }
    }

    pub fn query_nodes(&self, query: NodeQuery) -> PositionPinned<Vec<NodeRecord>> {
        PositionPinned {
            position: self.position,
            execution: PointReadExecution::DirectGraphFoldIndex,
            value: self.store.query_nodes(query),
        }
    }

    pub fn neighbors(&self, query: NeighborQuery) -> PositionPinned<Vec<NeighborHit>> {
        PositionPinned {
            position: self.position,
            execution: PointReadExecution::DirectGraphFoldIndex,
            value: self.store.neighbors(query),
        }
    }
}

#[derive(Debug)]
pub struct FoldGraphStore {
    committer: ProposalCommitter,
    facts: FoldRuntime<FactsFold>,
    graph: FoldRuntime<GraphFold>,
    algebra: FoldRuntime<AlgebraFold>,
}

impl FoldGraphStore {
    pub fn open(
        directory: impl AsRef<Path>,
        deployment_id: DeploymentId,
    ) -> GraphStoreResult<Self> {
        let log = PhysicalLog::open(directory, deployment_id).map_err(physical_error)?;
        let mut facts = FoldRuntime::for_log(FactsFold::default(), &log).map_err(fold_error)?;
        let mut graph = FoldRuntime::for_log(GraphFold::default(), &log).map_err(fold_error)?;
        let mut algebra = FoldRuntime::for_log(AlgebraFold::default(), &log).map_err(fold_error)?;
        algebra.inner_mut().set_log_id(log.log_id());
        facts.apply_all(&log).map_err(fold_error)?;
        graph.apply_all(&log).map_err(fold_error)?;
        algebra.apply_all(&log).map_err(fold_error)?;
        Ok(Self {
            committer: ProposalCommitter::new(log),
            facts,
            graph,
            algebra,
        })
    }

    pub fn latest_position(&self) -> LogPosition {
        self.committer.log().latest_position()
    }

    pub fn log(&self) -> &PhysicalLog {
        self.committer.log()
    }

    pub fn graph_quarantine(&self) -> &[QuarantineReceipt] {
        self.graph.quarantine_receipts()
    }

    pub fn facts_quarantine(&self) -> &[QuarantineReceipt] {
        self.facts.quarantine_receipts()
    }

    pub fn facts(&self) -> &FactsFold {
        self.facts.inner()
    }

    pub fn algebra(&self) -> &AlgebraFold {
        self.algebra.inner()
    }

    pub fn algebra_runtime_mut(&mut self) -> &mut FoldRuntime<AlgebraFold> {
        &mut self.algebra
    }

    pub fn algebra_projection(
        &mut self,
    ) -> GraphStoreResult<crate::fold::algebra::AlgebraProjection<'_>> {
        let log_id = self.algebra.log_id();
        self.algebra.inner_mut().set_log_id(log_id);
        self.algebra
            .inner_mut()
            .materialize_tokens()
            .map_err(fold_error)?;
        self.algebra.inner().project().map_err(fold_error)
    }

    pub fn algebra_quarantine(&self) -> &[QuarantineReceipt] {
        self.algebra.quarantine_receipts()
    }

    pub fn pin(&self, position: LogPosition) -> GraphStoreResult<PinnedGraphRead<'_>> {
        let current = self.latest_position();
        if position != current {
            return Err(GraphStoreError::new(
                "graph_position_not_materialized",
                format!(
                    "Graph facade is materialized at {current}; requested pinned position {position}"
                ),
            ));
        }
        Ok(PinnedGraphRead {
            store: self,
            position,
        })
    }

    pub fn pin_current(&self) -> PinnedGraphRead<'_> {
        PinnedGraphRead {
            store: self,
            position: self.latest_position(),
        }
    }

    /// Read a bounded live label page from the materialized graph fold.
    pub fn live_nodes_for_label_after_id(
        &self,
        label: &str,
        after_id: Option<&str>,
        limit: usize,
    ) -> Vec<NodeRecord> {
        self.graph_store()
            .live_nodes_for_label_after_id(label, after_id, limit)
    }

    pub fn commit_batch(
        &mut self,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<FoldGraphTransaction> {
        self.commit_intent_batch(GraphIntentBatch::from_mutations(batch)?)
    }

    pub fn delete_node(&mut self, id: &str) -> GraphStoreResult<bool> {
        if self.get_node(id).is_none() {
            return Ok(false);
        }
        self.commit_batch(GraphMutationBatch::new([GraphMutation::NodeDelete(
            id.to_string(),
        )]))?;
        Ok(true)
    }

    pub fn delete_edge(&mut self, id: &str) -> GraphStoreResult<bool> {
        if self.get_edge(id).is_none() {
            return Ok(false);
        }
        self.commit_batch(GraphMutationBatch::new([GraphMutation::EdgeDelete(
            id.to_string(),
        )]))?;
        Ok(true)
    }

    fn commit_intent_batch(
        &mut self,
        batch: GraphIntentBatch,
    ) -> GraphStoreResult<FoldGraphTransaction> {
        FactsFold::validate_batch(&batch).map_err(|error| {
            GraphStoreError::new("facts_fold_validation_failed", error.to_string())
        })?;
        self.graph.inner().validate_batch(&batch)?;
        self.algebra
            .inner()
            .validate_batch(&batch)
            .map_err(|error| {
                GraphStoreError::new("algebra_fold_validation_failed", error.to_string())
            })?;
        let begin = self.latest_position();
        let payload = serde_json::to_value(&batch).map_err(|error| {
            GraphStoreError::new("graph_intent_encode_failed", error.to_string())
        })?;
        let intent = LogIntent::new(
            IntentKind::new(
                GRAPH_INTENT_NAMESPACE,
                GRAPH_INTENT_NAME,
                GRAPH_INTENT_VERSION,
            )
            .map_err(|error| {
                GraphStoreError::new("graph_intent_kind_invalid", error.to_string())
            })?,
            payload.clone(),
        )
        .map_err(|error| GraphStoreError::new("graph_intent_invalid", error.to_string()))?;
        let (read_set, write_set) = self.access_sets(&batch)?;
        let digest = OpDigest::from_bytes(
            serde_json::to_vec(&json!({"begin": begin, "payload": payload})).map_err(|error| {
                GraphStoreError::new("graph_digest_encode_failed", error.to_string())
            })?,
        );
        let proposal = Proposal::new(begin, read_set, write_set, digest, intent);
        let CommitOutcome::Committed(receipt) =
            self.committer.commit(proposal).map_err(commit_error)?
        else {
            return Err(GraphStoreError::new(
                "occ_conflict",
                "graph proposal overlapped a committed write after its begin position",
            ));
        };

        if receipt.append.position > self.graph.current_position() {
            let facts_report = self
                .facts
                .apply_range(self.committer.log(), receipt.append.position)
                .map_err(fold_error)?;
            let graph_report = self
                .graph
                .apply_range(self.committer.log(), receipt.append.position)
                .map_err(fold_error)?;
            let algebra_report = self
                .algebra
                .apply_range(self.committer.log(), receipt.append.position)
                .map_err(fold_error)?;
            if facts_report.quarantined != 0
                || graph_report.quarantined != 0
                || algebra_report.quarantined != 0
            {
                return Err(GraphStoreError::new(
                    "committed_graph_intent_quarantined",
                    "a facade-validated graph intent was quarantined during fold replay",
                ));
            }
        }
        Ok(FoldGraphTransaction {
            position: receipt.append.position,
            writes: self
                .graph
                .inner()
                .writes_at(receipt.append.position)
                .to_vec(),
            replayed: receipt.append.replayed,
        })
    }

    fn access_sets(&self, batch: &GraphIntentBatch) -> GraphStoreResult<(ReadSet, WriteSet)> {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for change in &batch.changes {
            match change {
                GraphSemanticChange::NodeObserved { record } => {
                    let path = entity_path("node", &record.id);
                    reads.push(path.clone());
                    writes.push(path);
                }
                GraphSemanticChange::RelationshipObserved { record } => {
                    reads.push(entity_path("node", &record.from_id));
                    reads.push(entity_path("node", &record.to_id));
                    let path = entity_path("edge", &record.id);
                    reads.push(path.clone());
                    writes.push(path);
                }
                GraphSemanticChange::RelationshipRetracted { id } => {
                    let path = entity_path("edge", id);
                    reads.push(path.clone());
                    writes.push(path);
                }
                GraphSemanticChange::NodeRetracted { id } => {
                    let path = entity_path("node", id);
                    reads.push(path.clone());
                    writes.push(path);
                    for edge in self
                        .live_edge_records()
                        .into_iter()
                        .filter(|edge| edge.from_id == *id || edge.to_id == *id)
                    {
                        let edge_path = entity_path("edge", &edge.id);
                        reads.push(edge_path.clone());
                        writes.push(edge_path);
                    }
                }
            }
        }
        Ok((
            ReadSet::new(reads).map_err(|error| {
                GraphStoreError::new("graph_read_set_invalid", error.to_string())
            })?,
            WriteSet::new(writes).map_err(|error| {
                GraphStoreError::new("graph_write_set_invalid", error.to_string())
            })?,
        ))
    }

    fn graph_store(&self) -> &InMemoryGraphStore {
        self.graph.inner().store()
    }
}

impl GraphWrite for FoldGraphStore {
    fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        self.commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(node)]))?
            .writes
            .into_iter()
            .next()
            .ok_or_else(|| GraphStoreError::new("fold_graph_missing_write", "node write vanished"))
    }

    fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        self.commit_batch(GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge)]))?
            .writes
            .into_iter()
            .next()
            .ok_or_else(|| GraphStoreError::new("fold_graph_missing_write", "edge write vanished"))
    }
}

impl GraphAtomic for FoldGraphStore {
    fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if self.get_node(&node.id).is_some() {
            Ok(None)
        } else {
            self.upsert_node(node).map(Some)
        }
    }

    fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if self.has_historical_node_id(&node.id) {
            Ok(None)
        } else {
            self.upsert_node(node).map(Some)
        }
    }

    fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        let property = property.trim();
        if property.is_empty() {
            return Err(GraphStoreError::empty_field("node.property"));
        }
        let mut node = self.get_node_record(id).ok_or_else(|| {
            GraphStoreError::new("missing_graph_node", format!("node {id} does not exist"))
        })?;
        let current = node
            .properties
            .as_object()
            .and_then(|properties| properties.get(property))
            .cloned()
            .unwrap_or(Value::Null);
        if &current != expected {
            return Ok(None);
        }
        let properties = node.properties.as_object_mut().ok_or_else(|| {
            GraphStoreError::new(
                "invalid_node_properties",
                "node properties must be an object",
            )
        })?;
        if new_value.is_null() {
            properties.remove(property);
        } else {
            properties.insert(property.to_string(), new_value);
        }
        self.upsert_node(node).map(Some)
    }

    fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        let Some(batch) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(false);
        };
        self.commit_batch(batch)?;
        Ok(true)
    }
}

impl GraphRead for FoldGraphStore {
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        self.graph_store().get_node(id)
    }

    fn get_node_record(&self, id: &str) -> Option<NodeRecord> {
        self.graph_store().get_node_record(id)
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        self.graph_store().get_node_record_including_tombstone(id)
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        self.graph_store().has_historical_node_id(id)
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        self.graph_store().get_edge(id)
    }

    fn get_edge_record(&self, id: &str) -> Option<EdgeRecord> {
        self.graph_store().get_edge_record(id)
    }

    fn live_node_records(&self) -> Vec<NodeRecord> {
        self.graph_store().live_node_records()
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        self.graph_store().live_edge_records()
    }

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        self.graph_store().node_ids_for_label(label)
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        self.graph_store().edge_ids_for_type(edge_type)
    }

    fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        self.graph_store().node_ids_for_label_folded(label)
    }

    fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        self.graph_store().edge_ids_for_type_folded(edge_type)
    }

    fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        self.graph_store().node_ids_for_property(key, value)
    }

    fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        self.graph_store()
            .node_ids_for_string_property_exact(key, raw)
    }

    fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        self.graph_store()
            .node_ids_for_string_property_prefix(key, raw_prefix)
    }

    fn graph_version(&self) -> u64 {
        self.graph_store().graph_version()
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        self.graph_store().query_nodes(query)
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        self.graph_store().neighbors(query)
    }

    fn stats(&self) -> GraphStats {
        self.graph_store().stats()
    }
}

impl GraphSnapshotSource for FoldGraphStore {
    fn graph_snapshot(&self) -> GraphStoreResult<GraphSnapshot> {
        Ok(self.graph_store().snapshot())
    }
}

impl GraphVerify for FoldGraphStore {
    fn verify(&self) -> VerifyReport {
        self.graph_store().verify()
    }

    fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        self.graph.inner_mut().store.rebuild_indexes()
    }
}

impl GraphTtl for FoldGraphStore {
    fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult> {
        let node = self.get_node_record(id).ok_or_else(|| {
            GraphStoreError::new("missing_graph_node", format!("node {id} does not exist"))
        })?;
        self.upsert_node(node_with_ttl_property(node, expires_at_ms))
    }

    fn get_node_including_expired(&self, id: &str) -> Option<&NodeRecord> {
        self.graph_store().get_node_including_expired(id)
    }

    fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize> {
        let ids = self
            .graph_store()
            .nodes_expiring_before(now_ms(), usize::MAX)
            .into_iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Ok(0);
        }
        self.commit_batch(GraphMutationBatch::new(
            ids.iter().cloned().map(GraphMutation::NodeDelete),
        ))?;
        Ok(ids.len())
    }

    fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord> {
        self.graph_store().nodes_expiring_before(ts_ms, limit)
    }

    fn ttl_active_count(&self) -> usize {
        self.graph_store().ttl_active_count()
    }
}

impl GraphResidency for FoldGraphStore {
    fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        self.graph.inner_mut().store.evict_node(id)
    }

    fn readmit_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.graph.inner_mut().store.readmit_node(node)
    }

    fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        self.graph.inner_mut().store.evict_edge(id)
    }

    fn readmit_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        self.graph.inner_mut().store.readmit_edge(edge)
    }
}

impl GraphVectorPayloadAccess for FoldGraphStore {
    fn vector_payload_store(&self) -> GraphStoreResult<Arc<dyn TensorBlockPayloadStore>> {
        self.graph_store().vector_payload_store()
    }
}

fn apply_graph_changes(
    store: &mut InMemoryGraphStore,
    changes: &[GraphSemanticChange],
) -> GraphStoreResult<Vec<GraphWriteResult>> {
    let mut writes = Vec::new();
    for change in normalize_changes(changes.to_vec()) {
        match change {
            GraphSemanticChange::RelationshipRetracted { id } => {
                require_graph_identifier(&id, "edge.id")?;
                store.apply_logical_edge_delete(&id)?;
            }
            GraphSemanticChange::NodeRetracted { id } => {
                require_graph_identifier(&id, "node.id")?;
                let incident = store
                    .live_edge_records()
                    .into_iter()
                    .filter(|edge| edge.from_id == id || edge.to_id == id)
                    .map(|edge| edge.id)
                    .collect::<Vec<_>>();
                for edge_id in incident {
                    store.apply_logical_edge_delete(&edge_id)?;
                }
                store.apply_logical_node_delete(&id)?;
            }
            GraphSemanticChange::NodeObserved { record } => {
                writes.push(store.upsert_node(record)?);
            }
            GraphSemanticChange::RelationshipObserved { record } => {
                writes.push(store.upsert_edge(record)?);
            }
        }
    }
    Ok(writes)
}

fn normalize_changes(changes: Vec<GraphSemanticChange>) -> Vec<GraphSemanticChange> {
    let mut edge_retractions = Vec::new();
    let mut node_retractions = Vec::new();
    let mut node_observations = Vec::new();
    let mut edge_observations = Vec::new();
    for change in changes {
        match change {
            GraphSemanticChange::RelationshipRetracted { .. } => edge_retractions.push(change),
            GraphSemanticChange::NodeRetracted { .. } => node_retractions.push(change),
            GraphSemanticChange::NodeObserved { .. } => node_observations.push(change),
            GraphSemanticChange::RelationshipObserved { .. } => edge_observations.push(change),
        }
    }
    edge_retractions
        .into_iter()
        .chain(node_retractions)
        .chain(node_observations)
        .chain(edge_observations)
        .collect()
}

fn decode_graph_batch(record: &LogRecord) -> Result<GraphIntentBatch, FoldConsumeError> {
    if record.intent.kind.namespace() != GRAPH_INTENT_NAMESPACE
        || record.intent.kind.name() != GRAPH_INTENT_NAME
        || record.intent.kind.version() != GRAPH_INTENT_VERSION
    {
        return Err(FoldConsumeError::refused(
            FoldRefusalCode::UnsupportedIntent,
            format!("expected {GRAPH_INTENT_NAMESPACE}.{GRAPH_INTENT_NAME}@{GRAPH_INTENT_VERSION}"),
        ));
    }
    let batch = serde_json::from_value::<GraphIntentBatch>(record.intent.payload.clone()).map_err(
        |error| {
            FoldConsumeError::refused(
                FoldRefusalCode::InvalidPayload,
                format!("graph semantic batch decode failed: {error}"),
            )
        },
    )?;
    if batch.changes.is_empty() {
        return Err(FoldConsumeError::refused(
            FoldRefusalCode::SchemaViolation,
            "graph semantic batch must contain at least one change",
        ));
    }
    Ok(batch)
}

fn node_facts(node: &NodeRecord, source: &LogRecord) -> Result<Vec<FactTuple>, FoldConsumeError> {
    let entity = format!("node:{}", node.id);
    let temporal = temporal_columns(&node.properties);
    let epistemic_type = node
        .properties
        .get("_epistemic_type")
        .and_then(Value::as_str)
        .map(str::to_string);
    let confidence = node.properties.get("_confidence").and_then(Value::as_f64);
    let provenance = node.properties.get("_provenance").cloned();
    let mut facts = node
        .labels
        .iter()
        .cloned()
        .map(Value::String)
        .map(|value| {
            fact_tuple(
                &entity,
                "label",
                value,
                temporal,
                epistemic_type.clone(),
                confidence,
                provenance.clone(),
                source,
            )
        })
        .collect::<Vec<_>>();
    let properties = node.properties.as_object().ok_or_else(|| {
        FoldConsumeError::refused(
            FoldRefusalCode::SchemaViolation,
            "node properties must be an object",
        )
    })?;
    facts.extend(properties.iter().map(|(key, value)| {
        fact_tuple(
            &entity,
            &format!("property:{key}"),
            value.clone(),
            temporal,
            epistemic_type.clone(),
            confidence,
            provenance.clone(),
            source,
        )
    }));
    Ok(facts)
}

fn edge_facts(edge: &EdgeRecord, source: &LogRecord) -> Result<Vec<FactTuple>, FoldConsumeError> {
    let entity = format!("edge:{}", edge.id);
    let temporal = temporal_columns(&edge.properties);
    let epistemic_type = edge.epistemic_type.as_ref().map(ToString::to_string);
    let provenance = edge
        .provenance
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| FoldConsumeError::fatal(error.to_string()))?;
    let mut facts = vec![
        fact_tuple(
            &entity,
            "from",
            Value::String(edge.from_id.clone()),
            temporal,
            epistemic_type.clone(),
            edge.confidence,
            provenance.clone(),
            source,
        ),
        fact_tuple(
            &entity,
            "to",
            Value::String(edge.to_id.clone()),
            temporal,
            epistemic_type.clone(),
            edge.confidence,
            provenance.clone(),
            source,
        ),
        fact_tuple(
            &entity,
            "type",
            Value::String(edge.edge_type.clone()),
            temporal,
            epistemic_type.clone(),
            edge.confidence,
            provenance.clone(),
            source,
        ),
    ];
    let properties = edge.properties.as_object().ok_or_else(|| {
        FoldConsumeError::refused(
            FoldRefusalCode::SchemaViolation,
            "edge properties must be an object",
        )
    })?;
    facts.extend(properties.iter().map(|(key, value)| {
        fact_tuple(
            &entity,
            &format!("property:{key}"),
            value.clone(),
            temporal,
            epistemic_type.clone(),
            edge.confidence,
            provenance.clone(),
            source,
        )
    }));
    Ok(facts)
}

#[allow(clippy::too_many_arguments)]
fn fact_tuple(
    entity: &str,
    attribute: &str,
    value: Value,
    temporal: (Option<i64>, Option<i64>),
    epistemic_type: Option<String>,
    confidence: Option<f64>,
    provenance: Option<Value>,
    source: &LogRecord,
) -> FactTuple {
    FactTuple {
        entity: entity.to_string(),
        attribute: attribute.to_string(),
        value,
        valid_from_ms: temporal.0,
        valid_to_ms: temporal.1,
        asserted_at: source.position,
        retracted_at: None,
        epistemic_type,
        confidence,
        provenance,
        source_record_id: source.record_id,
    }
}

fn temporal_columns(properties: &Value) -> (Option<i64>, Option<i64>) {
    (
        properties.get("_valid_from_ms").and_then(Value::as_i64),
        properties.get("_valid_to_ms").and_then(Value::as_i64),
    )
}

fn canonical_value_token(value: &Value) -> String {
    serde_json::to_string(value).expect("serde_json::Value always serializes")
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), FoldConsumeError> {
    if value.is_empty() || value != value.trim() {
        Err(FoldConsumeError::refused(
            FoldRefusalCode::SchemaViolation,
            format!("{field} must be a canonical non-empty string"),
        ))
    } else {
        Ok(())
    }
}

fn require_graph_identifier(value: &str, field: &'static str) -> GraphStoreResult<()> {
    if value.is_empty() || value != value.trim() {
        Err(GraphStoreError::empty_field(field))
    } else {
        Ok(())
    }
}

fn entity_path(kind: &str, id: &str) -> AccessPath {
    AccessPath::Entity {
        entity: format!("{kind}:{id}"),
    }
}

fn error_code(error: &GraphStoreError) -> FoldRefusalCode {
    if error.code.contains("missing") || error.code.contains("tombstone") {
        FoldRefusalCode::SchemaViolation
    } else {
        FoldRefusalCode::InvalidPayload
    }
}

fn fold_error(error: FoldError) -> GraphStoreError {
    GraphStoreError::new("fold_graph_replay_failed", error.to_string())
}

fn physical_error(error: theorem_oplog::PhysicalLogError) -> GraphStoreError {
    GraphStoreError::new("fold_graph_log_failed", error.to_string())
}

fn commit_error(error: theorem_oplog::CommitError) -> GraphStoreError {
    GraphStoreError::new("fold_graph_commit_failed", error.to_string())
}
