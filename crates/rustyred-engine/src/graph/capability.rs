use std::sync::Arc;

use serde_json::Value;

use crate::graph::attribution::GraphStats;
use crate::graph::error::GraphStoreResult;
use crate::graph::mutation::{ConditionalNodePropertyUpdate, GraphMutationBatch, GraphWriteResult};
use crate::graph::query::{NeighborHit, NeighborQuery, NodeQuery};
use crate::graph::record::{Direction, EdgeRecord, NodeRecord};
use crate::graph::snapshot::GraphSnapshot;
use crate::graph::temporal::{node_time_interval, TimeInterval};
use crate::graph::ttl::{node_is_expired, now_ms};
use crate::graph::verify::{GraphRebuildReport, VerifyReport};
use crate::tensor_derivation::TensorBlockPayloadStore;

pub trait GraphRead {
    fn get_node(&self, id: &str) -> Option<&NodeRecord>;

    fn get_node_record(&self, id: &str) -> Option<NodeRecord> {
        self.get_node(id).cloned()
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        self.get_node_record(id)
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        self.get_node_record_including_tombstone(id).is_some()
    }

    fn get_node_interval(&self, id: &str) -> Option<TimeInterval> {
        self.get_node(id).and_then(node_time_interval)
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord>;

    fn get_edge_record(&self, id: &str) -> Option<EdgeRecord> {
        self.get_edge(id).cloned()
    }

    /// Clone every live node record. Relational lenses use this only to discover
    /// relation definitions; row reads continue to resolve by id.
    fn live_node_records(&self) -> Vec<NodeRecord> {
        self.query_nodes(NodeQuery::default().with_limit(usize::MAX))
    }

    /// Clone every live edge record.
    ///
    /// This is required rather than defaulting to an empty collection: an
    /// adapter that can resolve an edge but cannot enumerate edges must fail to
    /// implement `GraphRead`, not silently expose an incomplete graph.
    fn live_edge_records(&self) -> Vec<EdgeRecord>;

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        self.query_nodes(NodeQuery::label(label).with_limit(usize::MAX))
            .into_iter()
            .map(|node| node.id)
            .collect()
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        self.live_edge_records()
            .into_iter()
            .filter(|edge| edge.edge_type == edge_type)
            .map(|edge| edge.id)
            .collect()
    }

    fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        self.live_node_records()
            .into_iter()
            .filter(|node| {
                node.labels
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(label))
            })
            .map(|node| node.id)
            .collect()
    }

    fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        self.live_edge_records()
            .into_iter()
            .filter(|edge| edge.edge_type.eq_ignore_ascii_case(edge_type))
            .map(|edge| edge.id)
            .collect()
    }

    fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        self.live_node_records()
            .into_iter()
            .filter(|node| node.properties.get(key) == Some(value))
            .map(|node| node.id)
            .collect()
    }

    fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        self.live_node_records()
            .into_iter()
            .filter(|node| node.properties.get(key).and_then(Value::as_str) == Some(raw))
            .map(|node| node.id)
            .collect()
    }

    fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        self.live_node_records()
            .into_iter()
            .filter(|node| {
                node.properties
                    .get(key)
                    .and_then(Value::as_str)
                    .is_some_and(|value| value.starts_with(raw_prefix))
            })
            .map(|node| node.id)
            .collect()
    }

    /// Return the graph freshness watermark without requiring callers to build
    /// a full diagnostic and memory-attribution report.
    fn graph_version(&self) -> u64 {
        self.stats().version
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord>;
    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit>;
    fn stats(&self) -> GraphStats;
}

impl<T> GraphRead for &T
where
    T: GraphRead + ?Sized,
{
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        T::get_node(*self, id)
    }

    fn get_node_record(&self, id: &str) -> Option<NodeRecord> {
        T::get_node_record(*self, id)
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        T::get_node_record_including_tombstone(*self, id)
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        T::has_historical_node_id(*self, id)
    }

    fn get_node_interval(&self, id: &str) -> Option<TimeInterval> {
        T::get_node_interval(*self, id)
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        T::get_edge(*self, id)
    }

    fn get_edge_record(&self, id: &str) -> Option<EdgeRecord> {
        T::get_edge_record(*self, id)
    }

    fn live_node_records(&self) -> Vec<NodeRecord> {
        T::live_node_records(*self)
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        T::live_edge_records(*self)
    }

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        T::node_ids_for_label(*self, label)
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        T::edge_ids_for_type(*self, edge_type)
    }

    fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        T::node_ids_for_label_folded(*self, label)
    }

    fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        T::edge_ids_for_type_folded(*self, edge_type)
    }

    fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        T::node_ids_for_property(*self, key, value)
    }

    fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        T::node_ids_for_string_property_exact(*self, key, raw)
    }

    fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        T::node_ids_for_string_property_prefix(*self, key, raw_prefix)
    }

    fn graph_version(&self) -> u64 {
        T::graph_version(*self)
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        T::query_nodes(*self, query)
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        T::neighbors(*self, query)
    }

    fn stats(&self) -> GraphStats {
        T::stats(*self)
    }
}

impl GraphRead for GraphSnapshot {
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        let now = now_ms();
        self.nodes
            .iter()
            .find(|node| node.id == id && snapshot_node_visible(node, false, now))
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        self.nodes.iter().find(|node| node.id == id).cloned()
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        self.historical_node_ids.contains(id)
    }

    fn graph_version(&self) -> u64 {
        self.version
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        self.edges
            .iter()
            .find(|edge| edge.id == id && !edge.tombstone)
    }

    fn live_node_records(&self) -> Vec<NodeRecord> {
        let now = now_ms();
        self.nodes
            .iter()
            .filter(|node| snapshot_node_visible(node, false, now))
            .cloned()
            .collect()
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        self.edges
            .iter()
            .filter(|edge| !edge.tombstone)
            .cloned()
            .collect()
    }

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        let now = now_ms();
        self.nodes
            .iter()
            .filter(|node| {
                snapshot_node_visible(node, false, now)
                    && node.labels.iter().any(|item| item == label)
            })
            .map(|node| node.id.clone())
            .collect()
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|edge| !edge.tombstone && edge.edge_type == edge_type)
            .map(|edge| edge.id.clone())
            .collect()
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        // `bounded_limit` rather than a second `limit.unwrap_or(100)`: the
        // snapshot used to re-spell the silent hundred-row default, so a read
        // through a snapshot truncated even after the query type stopped doing
        // it. One definition, so the two cannot drift apart again.
        let limit = query.bounded_limit();
        let now = now_ms();
        self.nodes
            .iter()
            .filter(|node| snapshot_node_visible(node, query.include_expired, now))
            .filter(|node| {
                query
                    .label
                    .as_ref()
                    .is_none_or(|label| node.labels.iter().any(|item| item == label))
            })
            .filter(|node| {
                query
                    .properties
                    .iter()
                    .all(|(key, value)| node.properties.get(key) == Some(value))
            })
            .take(limit)
            .cloned()
            .collect()
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        let now = now_ms();
        self.edges
            .iter()
            .filter(|edge| !edge.tombstone)
            .filter(|edge| {
                query
                    .edge_type
                    .as_ref()
                    .is_none_or(|edge_type| edge.edge_type == *edge_type)
            })
            .filter_map(|edge| {
                let node_id = match query.direction {
                    Direction::Out if edge.from_id == query.node_id => &edge.to_id,
                    Direction::In if edge.to_id == query.node_id => &edge.from_id,
                    _ => return None,
                };
                self.nodes
                    .iter()
                    .find(|node| {
                        node.id == *node_id
                            && snapshot_node_visible(node, query.include_expired, now)
                    })
                    .map(|_| NeighborHit {
                        edge_id: edge.id.clone(),
                        node_id: node_id.clone(),
                        edge_type: edge.edge_type.clone(),
                        confidence: edge.confidence,
                        epistemic_type: edge.epistemic_type.clone(),
                    })
            })
            .collect()
    }

    fn stats(&self) -> GraphStats {
        let labels = self
            .nodes
            .iter()
            .filter(|node| !node.tombstone)
            .flat_map(|node| node.labels.iter())
            .collect::<std::collections::BTreeSet<_>>();
        let edge_types = self
            .edges
            .iter()
            .filter(|edge| !edge.tombstone)
            .map(|edge| &edge.edge_type)
            .collect::<std::collections::BTreeSet<_>>();
        GraphStats {
            version: self.version,
            nodes_total: self.nodes.iter().filter(|node| !node.tombstone).count(),
            edges_total: self.edges.iter().filter(|edge| !edge.tombstone).count(),
            labels_total: labels.len(),
            edge_types_total: edge_types.len(),
            ..GraphStats::default()
        }
    }
}

fn snapshot_node_visible(node: &NodeRecord, include_expired: bool, now_ms: i64) -> bool {
    !node.tombstone && (include_expired || !node_is_expired(node, now_ms))
}

pub trait GraphWrite {
    fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult>;
    fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult>;

    /// Publish a mutation batch. The default applies each mutation through
    /// [`upsert_node`]/[`upsert_edge`]. Durable stores override this so one
    /// VFS event or UI edit is one journaled commit.
    fn commit_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        for mutation in batch.mutations {
            match mutation {
                crate::graph::mutation::GraphMutation::NodeUpsert(node) => {
                    self.upsert_node(node)?;
                }
                crate::graph::mutation::GraphMutation::EdgeUpsert(edge) => {
                    self.upsert_edge(edge)?;
                }
                crate::graph::mutation::GraphMutation::NodeDelete(_)
                | crate::graph::mutation::GraphMutation::EdgeDelete(_) => {
                    return Err(crate::graph::error::GraphStoreError::new(
                        "graph_write_delete_unsupported",
                        "default GraphWrite::commit_batch does not apply deletes",
                    ));
                }
            }
        }
        Ok(())
    }
}

pub trait GraphStore: GraphRead + GraphWrite {}

impl<T> GraphStore for T where T: GraphRead + GraphWrite + ?Sized {}

pub trait GraphAtomic {
    fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>>;

    fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>>;

    fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>>;

    fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool>;
}

/// First-class TTL capability.
///
/// A read/write graph without this capability is rejected at compile time:
///
/// ```compile_fail
/// use rustyred_thg_core::{
///     EdgeRecord, GraphRead, GraphStats, GraphStoreResult, GraphTtl, GraphWrite,
///     GraphWriteResult, NeighborHit, NeighborQuery, NodeQuery, NodeRecord,
/// };
///
/// struct ReadWriteOnly;
///
/// impl GraphRead for ReadWriteOnly {
///     fn get_node(&self, _: &str) -> Option<&NodeRecord> { None }
///     fn get_edge(&self, _: &str) -> Option<&EdgeRecord> { None }
///     fn live_edge_records(&self) -> Vec<EdgeRecord> { Vec::new() }
///     fn query_nodes(&self, _: NodeQuery) -> Vec<NodeRecord> { Vec::new() }
///     fn neighbors(&self, _: NeighborQuery) -> Vec<NeighborHit> { Vec::new() }
///     fn stats(&self) -> GraphStats { GraphStats::default() }
/// }
///
/// impl GraphWrite for ReadWriteOnly {
///     fn upsert_node(&mut self, _: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
///         unreachable!()
///     }
///     fn upsert_edge(&mut self, _: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
///         unreachable!()
///     }
/// }
///
/// fn purge(store: &mut impl GraphTtl) {
///     let _ = store.purge_expired_nodes();
/// }
///
/// purge(&mut ReadWriteOnly);
/// ```
pub trait GraphTtl {
    fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult>;

    fn get_node_including_expired(&self, id: &str) -> Option<&NodeRecord>;
    fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize>;
    fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord>;
    fn ttl_active_count(&self) -> usize;
}

pub trait GraphResidency {
    fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>>;
    fn readmit_node(&mut self, node: NodeRecord) -> GraphStoreResult<()>;
    fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>>;
    fn readmit_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()>;
}

pub trait GraphSnapshotSource {
    fn graph_snapshot(&self) -> GraphStoreResult<GraphSnapshot>;
}

pub trait GraphVerify {
    fn verify(&self) -> VerifyReport;
    fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport>;
}

pub trait GraphVectorPayloadAccess {
    fn vector_payload_store(&self) -> GraphStoreResult<Arc<dyn TensorBlockPayloadStore>>;
}
