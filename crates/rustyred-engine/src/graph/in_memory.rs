use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;

use serde_json::Value;

use crate::access_method::AccessMethodCache;
use crate::graph::attribution::{
    GraphMemoryAttribution, GraphStats, JsonAllocationStats, StringAllocationStats,
    StringIndexAllocationStats, TupleIndexAllocationStats, VectorIndexAllocationStats,
};
use crate::graph::capability::{
    GraphAtomic, GraphRead, GraphResidency, GraphSnapshotSource, GraphTtl,
    GraphVectorPayloadAccess, GraphVerify, GraphWrite,
};
use crate::graph::error::{
    is_recoverable_orphan_edge, vector_pin_cleanup_error, GraphStoreError, GraphStoreResult,
};
use crate::graph::footprint::{
    note_graph_memory_recompute, ordered_member_bytes, CommittedFootprint,
};
use crate::graph::index::{
    indexed_properties, merge_candidates, numeric_property, property_index_token, sorted_values,
    string_property_index_token, validate_edge_shape,
};
use crate::graph::mutation::{
    prepare_conditional_batch, ConditionalNodePropertyUpdate, GraphMutation, GraphMutationBatch,
    GraphWriteResult,
};
use crate::graph::query::{NeighborHit, NeighborQuery, NodeQuery};
use crate::graph::record::{normalize_labels, Direction, EdgeRecord, EpistemicType, NodeRecord};
use crate::graph::residency::{COLD_SNAPSHOT_STUB_LABEL, COLD_SNAPSHOT_STUB_PROPERTY};
use crate::graph::snapshot::{
    serialized_json_bytes, BorrowedGraphSnapshot, EdgeValues, GraphSnapshot, MapValues,
};
use crate::graph::ttl::{node_is_expired, now_ms};
use crate::graph::vector_index::{
    serialize_vector_fallback_reasons, HybridScoringConfig, VectorDesignation, VectorIndex,
    VectorIndexShape, VectorPoint,
};
use crate::graph::vector_pin::{
    PreparedVectorProperties, VectorPayloadPinGuard, VectorPayloadPinReceipt,
    VectorPayloadStoreHandle,
};
use crate::graph::verify::{GraphRebuildReport, VerifyReport};
use crate::memory_document::{MemoryDocumentLens, MemoryDocumentQuery};
use crate::ordered::{OrderedDesignation, OrderedIndex};
use crate::tensor_derivation::{InMemoryTensorBlockPayloadStore, TensorBlockPayloadStore};
use crate::vector_eval::{VectorSearchBackend, VectorSearchReceipt};
use crate::vector_payload::{
    decode_legacy_vector_array, pack_vector, read_vector_property_packed,
    vector_pin_publication_key, write_vector_property, VectorPropertyRef,
    BUILTIN_VECTOR_PAYLOAD_PROPERTIES,
};

pub(crate) type EdgeOrdinal = usize;

/// Stable edge slots keep secondary indexes compact.
///
/// The prior representation copied every edge ID into the primary B-tree,
/// edge-type index, and both adjacency indexes. Large graphs therefore owned
/// the same string five times (including `EdgeRecord::id`). This arena keeps
/// the public ID lookup while secondary indexes retain only a machine-word
/// ordinal into `records`.
#[derive(Clone, Debug, Default)]
pub(crate) struct EdgeArena {
    by_id: BTreeMap<String, EdgeOrdinal>,
    records: Vec<Option<EdgeRecord>>,
    free: Vec<EdgeOrdinal>,
}

impl EdgeArena {
    /// Reserve the exact number of edge slots discovered by the first
    /// streaming recovery pass.
    ///
    /// `Vec`'s geometric growth is useful for ordinary writes, but a durable
    /// reopen already knows the retained edge count before it decodes the
    /// first edge payload. Reserving once avoids both the unused power-of-two
    /// tail and the transient high-water allocations from repeated growth.
    pub(crate) fn reserve_recovery_slots(&mut self, additional: usize) {
        self.records.reserve_exact(additional);
    }

    pub(crate) fn len(&self) -> usize {
        self.by_id.len()
    }

    pub(crate) fn get(&self, id: &str) -> Option<&EdgeRecord> {
        self.ordinal(id)
            .and_then(|ordinal| self.get_by_ordinal(ordinal))
    }

    pub(crate) fn ordinal(&self, id: &str) -> Option<EdgeOrdinal> {
        self.by_id.get(id).copied()
    }

    pub(crate) fn get_by_ordinal(&self, ordinal: EdgeOrdinal) -> Option<&EdgeRecord> {
        self.records.get(ordinal).and_then(Option::as_ref)
    }

    pub(crate) fn insert(
        &mut self,
        id: String,
        edge: EdgeRecord,
    ) -> (EdgeOrdinal, Option<EdgeRecord>) {
        if let Some(ordinal) = self.by_id.get(&id).copied() {
            let previous = self.records[ordinal].replace(edge);
            return (ordinal, previous);
        }
        let ordinal = self.free.pop().unwrap_or_else(|| {
            self.records.push(None);
            self.records.len() - 1
        });
        self.records[ordinal] = Some(edge);
        self.by_id.insert(id, ordinal);
        (ordinal, None)
    }

    pub(crate) fn remove(&mut self, id: &str) -> Option<EdgeRecord> {
        let ordinal = self.by_id.remove(id)?;
        let removed = self.records.get_mut(ordinal)?.take();
        if removed.is_some() {
            self.free.push(ordinal);
        }
        removed
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &EdgeRecord> {
        self.by_id
            .values()
            .filter_map(|ordinal| self.get_by_ordinal(*ordinal))
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &EdgeRecord)> {
        self.by_id
            .iter()
            .filter_map(|(id, ordinal)| self.get_by_ordinal(*ordinal).map(|edge| (id, edge)))
    }

    pub(crate) fn iter_with_ordinals(&self) -> impl Iterator<Item = (EdgeOrdinal, &EdgeRecord)> {
        self.records
            .iter()
            .enumerate()
            .filter_map(|(ordinal, edge)| edge.as_ref().map(|edge| (ordinal, edge)))
    }

    fn record_slots_capacity_bytes(&self) -> usize {
        self.records
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<EdgeRecord>>())
    }

    fn free_slots_capacity_bytes(&self) -> usize {
        self.free
            .capacity()
            .saturating_mul(std::mem::size_of::<EdgeOrdinal>())
    }
}

impl std::ops::Index<&str> for EdgeArena {
    type Output = EdgeRecord;

    fn index(&self, id: &str) -> &Self::Output {
        self.get(id).expect("edge id missing from arena")
    }
}

#[cfg(test)]
mod edge_arena_tests {
    use serde_json::json;

    use super::EdgeArena;
    use crate::graph::record::EdgeRecord;

    #[test]
    fn recovery_reservation_avoids_geometric_edge_slot_growth() {
        let edge_count = 137;
        let mut arena = EdgeArena::default();
        arena.reserve_recovery_slots(edge_count);
        let reserved_ptr = arena.records.as_ptr();

        for index in 0..edge_count {
            let id = format!("edge:{index:03}");
            let edge = EdgeRecord::new(id.clone(), "node:from", "RELATES", "node:to", json!({}));
            arena.insert(id, edge);
        }

        assert_eq!(arena.len(), edge_count);
        assert_eq!(arena.records.capacity(), edge_count);
        assert_eq!(
            arena.records.as_ptr(),
            reserved_ptr,
            "recovery must fill one pre-sized allocation without reallocating"
        );
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryGraphStore {
    pub(crate) version: u64,
    pub(crate) historical_node_ids: BTreeSet<String>,
    pub(crate) historical_node_history_complete: bool,
    pub(crate) historical_node_id_limit: usize,
    pub(crate) nodes: BTreeMap<String, NodeRecord>,
    pub(crate) edges: EdgeArena,
    pub(crate) out_adjacency: BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
    pub(crate) in_adjacency: BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
    pub(crate) label_index: BTreeMap<String, BTreeSet<String>>,
    pub(crate) edge_type_index: BTreeMap<String, BTreeSet<EdgeOrdinal>>,
    pub(crate) property_index: BTreeMap<(String, String), BTreeSet<String>>,
    pub(crate) vector_designations: HashMap<(String, String), usize>,
    pub(crate) vector_indexes: HashMap<(String, String), VectorIndex>,
    pub(crate) vector_payload_store: VectorPayloadStoreHandle,
    pub(crate) ordered_indexes: BTreeMap<(String, String), OrderedIndex>,
    /// Generic graph-versioned scalar index cache. It retains derived keys and
    /// node ids only; cloning the graph resets it so equal version numbers on
    /// diverged clones cannot share stale candidates.
    pub(crate) derived_access_methods: AccessMethodCache,
    /// TTL expiration index: maps `_ttl_expires_at_ms` -> set of node ids
    /// that expire at that timestamp. Updated on every node upsert that
    /// has (or had) a TTL property. Sweep iterates `range(..=now_ms)` to
    /// find expired nodes in O(expired_count + log total) instead of
    /// scanning all nodes. Design doc: docs/plans/rustyred-thg-ttl-primitive/.
    pub(crate) ttl_index: BTreeMap<i64, BTreeSet<String>>,
    /// Version-neutral residency markers for nodes whose full records live in
    /// the cold object store. Incident edges remain resident; durable RedCore
    /// stores journal rollback-compatible node+edge records outside snapshots.
    pub(crate) cold_nonresident_node_ids: BTreeSet<String>,
    /// Incrementally maintained decomposition of [`Self::estimated_memory_bytes`].
    ///
    /// Every mutation site states its byte delta here, so the quota path reads
    /// the committed footprint in O(1) instead of serde-serializing the whole
    /// graph. [`Self::recompute_memory_bytes`] is the O(graph) oracle that the
    /// counters are checked against and that constructors/recovery rebuild from.
    /// See `crate::graph::footprint` for the decomposition and delta rules.
    pub(crate) committed_footprint: CommittedFootprint,
}

/// Bound the duplicate resident registry used only by the fail-closed
/// `insert_node_if_never_existed` seam. Once this limit is crossed, the
/// completeness flag becomes false and that seam refuses all future claims;
/// ordinary graph reads and writes remain available.
pub(crate) const MAX_RESIDENT_HISTORICAL_NODE_IDS: usize = 262_144;

impl Default for InMemoryGraphStore {
    fn default() -> Self {
        Self {
            version: 0,
            historical_node_ids: BTreeSet::new(),
            historical_node_history_complete: true,
            historical_node_id_limit: MAX_RESIDENT_HISTORICAL_NODE_IDS,
            nodes: BTreeMap::new(),
            edges: EdgeArena::default(),
            out_adjacency: BTreeMap::new(),
            in_adjacency: BTreeMap::new(),
            label_index: BTreeMap::new(),
            edge_type_index: BTreeMap::new(),
            property_index: BTreeMap::new(),
            vector_designations: HashMap::new(),
            vector_indexes: HashMap::new(),
            vector_payload_store: VectorPayloadStoreHandle::new(Arc::new(
                InMemoryTensorBlockPayloadStore::new(),
            )),
            ordered_indexes: BTreeMap::new(),
            derived_access_methods: AccessMethodCache::ordered(),
            ttl_index: BTreeMap::new(),
            cold_nonresident_node_ids: BTreeSet::new(),
            // An empty store's committed footprint is exactly zero, so no
            // recompute is needed here. Constructors that install records
            // (`from_snapshot`) re-derive the counters from the oracle instead.
            committed_footprint: CommittedFootprint::default(),
        }
    }
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_vector_payload_store(store: Arc<dyn TensorBlockPayloadStore>) -> Self {
        let mut graph = Self::default();
        graph.vector_payload_store = VectorPayloadStoreHandle::new(store);
        graph
    }

    pub(crate) fn set_vector_payload_store(&mut self, store: Arc<dyn TensorBlockPayloadStore>) {
        self.vector_payload_store = VectorPayloadStoreHandle::new(store);
    }

    pub(crate) fn vector_payload_store_handle(&self) -> Arc<dyn TensorBlockPayloadStore> {
        self.vector_payload_store.get()
    }

    pub(crate) fn reserve_recovered_edges(&mut self, edge_records: usize) {
        self.edges.reserve_recovery_slots(edge_records);
    }

    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            version: self.version,
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.values().cloned().collect(),
            historical_node_ids: self.historical_node_ids.clone(),
            historical_node_history_complete: self.historical_node_history_complete,
        }
    }

    /// Clone only live edge records. Operational edge consumers use this
    /// instead of constructing a full `GraphSnapshot`, which would also clone
    /// every node before immediately discarding that allocation.
    pub fn live_edges_snapshot(&self) -> Vec<EdgeRecord> {
        self.edges
            .values()
            .filter(|edge| !edge.tombstone)
            .cloned()
            .collect()
    }

    pub fn from_snapshot(snapshot: GraphSnapshot) -> GraphStoreResult<Self> {
        Self::from_snapshot_with_historical_limit(snapshot, MAX_RESIDENT_HISTORICAL_NODE_IDS)
    }

    pub(crate) fn from_snapshot_with_historical_limit(
        snapshot: GraphSnapshot,
        historical_node_id_limit: usize,
    ) -> GraphStoreResult<Self> {
        let mut store = Self::new();
        store.version = snapshot.version;
        store.historical_node_id_limit = historical_node_id_limit;
        // The persisted registry is authoritative. `apply_recovered_node` records
        // every live node id as a recovery side effect; reinstall the snapshot's
        // set and completeness flag afterward so a truncated incomplete registry
        // cannot expand back into a complete-looking resident set on reopen.
        let historical_node_ids = snapshot.historical_node_ids;
        let historical_node_history_complete = snapshot.historical_node_history_complete;
        let cold_stub_ids = snapshot
            .nodes
            .iter()
            .filter(|node| {
                node.labels
                    .iter()
                    .any(|label| label == COLD_SNAPSHOT_STUB_LABEL)
                    && node
                        .properties
                        .get(COLD_SNAPSHOT_STUB_PROPERTY)
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            })
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        for mut node in snapshot.nodes {
            node.labels = normalize_labels(node.labels);
            store.apply_recovered_node(node)?;
        }
        for edge in snapshot.edges {
            match store.apply_recovered_edge(edge) {
                Ok(()) => {}
                Err(error) if is_recoverable_orphan_edge(&error) => {}
                Err(error) => return Err(error),
            }
        }
        for id in cold_stub_ids {
            store.evict_node(&id)?;
        }
        store.historical_node_ids = BTreeSet::new();
        store.historical_node_history_complete = historical_node_history_complete;
        for id in historical_node_ids {
            if store.historical_node_ids.len() >= store.historical_node_id_limit {
                store.historical_node_history_complete = false;
                break;
            }
            store.historical_node_ids.insert(id);
        }
        store.version = store.version.max(snapshot.version);
        // Reopen is allowed to be O(graph): re-derive the committed footprint
        // from the recovered records and the reinstalled historical registry
        // rather than trusting the per-record deltas applied along the way.
        store.reset_committed_footprint_from_recompute();
        Ok(store)
    }

    pub(crate) fn record_historical_node_id(&mut self, id: &str) {
        if self.historical_node_ids.contains(id) {
            return;
        }
        if self.historical_node_ids.len() >= self.historical_node_id_limit {
            self.historical_node_history_complete = false;
            return;
        }
        self.historical_node_ids.insert(id.to_string());
        self.committed_footprint.add_historical_id(id);
    }

    /// Project the historical registry that would result from applying `mutation`
    /// without mutating this store. Used by the incremental RedCore publish path,
    /// which persists before applying the batch to the resident mirror.
    pub(crate) fn projected_historical_registry_after<'a>(
        &self,
        node_ids: impl IntoIterator<Item = &'a str>,
    ) -> (BTreeSet<String>, bool) {
        let mut historical_node_ids = self.historical_node_ids.clone();
        let mut historical_node_history_complete = self.historical_node_history_complete;
        for id in node_ids {
            if historical_node_ids.contains(id) {
                continue;
            }
            if historical_node_ids.len() >= self.historical_node_id_limit {
                historical_node_history_complete = false;
                continue;
            }
            historical_node_ids.insert(id.to_string());
        }
        (historical_node_ids, historical_node_history_complete)
    }

    pub(crate) fn designated_vector_dimensions(
        &self,
        node: &NodeRecord,
    ) -> GraphStoreResult<BTreeMap<String, usize>> {
        let mut dimensions = BTreeMap::new();
        for ((label, property), dimension) in &self.vector_designations {
            if !node.labels.iter().any(|candidate| candidate == label) {
                continue;
            }
            if let Some(previous) = dimensions.insert(property.clone(), *dimension) {
                if previous != *dimension {
                    return Err(GraphStoreError::new(
                        "vector_designation_conflict",
                        format!(
                            "node {} matches incompatible {previous}- and {dimension}-dimension designations for {property}",
                            node.id
                        ),
                    ));
                }
            }
        }
        for (label, property) in BUILTIN_VECTOR_PAYLOAD_PROPERTIES {
            if !node.labels.iter().any(|candidate| candidate == label) {
                continue;
            }
            let Some(value) = node.properties.get(property) else {
                continue;
            };
            let dimension = match decode_legacy_vector_array(value)? {
                Some(vector) => vector.len(),
                None => VectorPropertyRef::from_value(value)?.dimension,
            };
            if let Some(previous) = dimensions.insert(property.to_string(), dimension) {
                if previous != dimension {
                    return Err(GraphStoreError::new(
                        "vector_designation_conflict",
                        format!(
                            "node {} matches incompatible {previous}- and {dimension}-dimension schemas for {property}",
                            node.id
                        ),
                    ));
                }
            }
        }
        Ok(dimensions)
    }

    pub(crate) fn prepare_designated_vector_properties(
        &self,
        node: &mut NodeRecord,
    ) -> GraphStoreResult<PreparedVectorProperties> {
        let dimensions = self.designated_vector_dimensions(node)?;
        let payload_store = self.vector_payload_store.get();
        let mut prepared = PreparedVectorProperties::default();
        let mut acquired_pins = VectorPayloadPinGuard::new(Arc::clone(&payload_store));
        let prepare_result = (|| {
            for (property, expected_dimension) in dimensions {
                let Some(value) = node.properties.get(&property).cloned() else {
                    continue;
                };
                if let Some(vector) = decode_legacy_vector_array(&value)? {
                    if vector.len() != expected_dimension {
                        return Err(GraphStoreError::new(
                            "dimension_mismatch",
                            format!(
                                "designated property {property} expects {expected_dimension} dimensions, got {}",
                                vector.len()
                            ),
                        ));
                    }
                    // Payload write and verified readback happen before any
                    // graph index, version, or node-map mutation below. Keep
                    // the already packed lanes for the index so it does not
                    // fetch and allocate the same payload again.
                    write_vector_property(node, &property, &vector, payload_store.as_ref())?;
                    let reference = VectorPropertyRef::from_value(
                        node.properties.get(&property).ok_or_else(|| {
                            GraphStoreError::new(
                                "vector_property_missing",
                                format!(
                                    "designated property {property} disappeared after payload write"
                                ),
                            )
                        })?,
                    )?;
                    let receipt = VectorPayloadPinReceipt {
                        publication_key: vector_pin_publication_key(
                            &node.id,
                            &property,
                            &reference.content_id,
                        ),
                        content_id: reference.content_id,
                        pending_released: false,
                        claimed: true,
                    };
                    if let Err(claim_error) =
                        payload_store.claim_tensor_block_pin(&receipt.content_id)
                    {
                        return match payload_store.release_bound_tensor_block_pending(
                            &receipt.content_id,
                            &receipt.publication_key,
                        ) {
                            Ok(_) => Err(claim_error),
                            Err(release_error) => Err(vector_pin_cleanup_error(
                                "failed vector-payload claim",
                                claim_error,
                                release_error,
                            )),
                        };
                    }
                    acquired_pins.extend([receipt]);
                    prepared
                        .packed_by_property
                        .insert(property, pack_vector(&vector));
                } else {
                    let reference = VectorPropertyRef::from_value(&value)?;
                    let receipt = VectorPayloadPinReceipt {
                        publication_key: vector_pin_publication_key(
                            &node.id,
                            &property,
                            &reference.content_id,
                        ),
                        content_id: reference.content_id.clone(),
                        pending_released: false,
                        claimed: true,
                    };
                    if let Err(claim_error) =
                        payload_store.claim_tensor_block_pin(&receipt.content_id)
                    {
                        return match payload_store.release_bound_tensor_block_pending(
                            &receipt.content_id,
                            &receipt.publication_key,
                        ) {
                            Ok(_) => Err(claim_error),
                            Err(release_error) => Err(vector_pin_cleanup_error(
                                "failed vector-payload claim",
                                claim_error,
                                release_error,
                            )),
                        };
                    }
                    acquired_pins.extend([receipt]);
                    let Some((resolved_reference, packed)) =
                        read_vector_property_packed(node, &property, payload_store.as_ref())?
                    else {
                        return Err(GraphStoreError::new(
                            "vector_property_missing",
                            format!("designated property {property} disappeared during preflight"),
                        ));
                    };
                    if resolved_reference.dimension != expected_dimension {
                        return Err(GraphStoreError::new(
                            "dimension_mismatch",
                            format!(
                                "designated property {property} expects {expected_dimension} dimensions, reference declares {}",
                                resolved_reference.dimension
                            ),
                        ));
                    }
                    prepared.packed_by_property.insert(property, packed);
                }
            }
            Ok(())
        })();
        prepare_result?;
        prepared.acquired_pin_receipts = acquired_pins.into_receipts();
        Ok(prepared)
    }

    pub(crate) fn release_vector_payload_pins(
        &self,
        receipts: &[VectorPayloadPinReceipt],
    ) -> GraphStoreResult<()> {
        let mut pins = VectorPayloadPinGuard::new(self.vector_payload_store_handle());
        pins.extend(receipts.iter().cloned());
        pins.cancel()
    }

    pub(crate) fn legacy_vector_node_ids_for_designation(
        &self,
        label: &str,
        property: &str,
    ) -> Vec<String> {
        self.nodes
            .values()
            .filter(|node| {
                !node.tombstone
                    && node.labels.iter().any(|candidate| candidate == label)
                    && node.properties.get(property).is_some_and(Value::is_array)
            })
            .map(|node| node.id.clone())
            .collect()
    }

    pub fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        let (write, acquired_pin_receipts) = self.upsert_node_retaining_vector_pins(node)?;
        self.release_vector_payload_pins(&acquired_pin_receipts)?;
        Ok(write)
    }

    pub(crate) fn upsert_node_retaining_vector_pins(
        &mut self,
        mut node: NodeRecord,
    ) -> GraphStoreResult<(GraphWriteResult, Vec<VectorPayloadPinReceipt>)> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }

        node.labels = normalize_labels(node.labels);
        let mut prepared = self.prepare_designated_vector_properties(&mut node)?;
        let acquired_pin_receipts = std::mem::take(&mut prepared.acquired_pin_receipts);
        let write_result = (|| {
            if self.cold_nonresident_node_ids.remove(&node.id) {
                self.committed_footprint.remove_cold_nonresident_id(&node.id);
            }
            self.record_historical_node_id(&node.id);
            let parent_hash = match self.nodes.get(&node.id) {
                Some(existing) => Some(match &existing.content_hash {
                    Some(hash) => hash.clone(),
                    None => existing.checksum()?,
                }),
                None => None,
            };
            if let Some(existing) = self.nodes.get(&node.id).cloned() {
                self.committed_footprint.remove_node_record(&existing);
                self.remove_node_indexes(&existing);
            }

            self.version += 1;
            node.version = self.version;
            let content_hash = node.checksum()?;
            if node.parent_hashes.is_empty() {
                if let Some(parent_hash) = parent_hash.filter(|parent| parent != &content_hash) {
                    node.parent_hashes.push(parent_hash);
                }
            }
            node.content_hash = Some(content_hash.clone());
            let checksum = node.checksum()?;
            let id = node.id.clone();
            if !node.tombstone {
                self.add_node_indexes(&node);
                self.auto_index_vectors_with_prepared(&node, &mut prepared.packed_by_property)?;
            }
            // Serialize the final record, after the version/hash rewrite, so the
            // counter matches the bytes the snapshot writer will emit.
            self.committed_footprint.add_node_record(&node);
            self.nodes.insert(id.clone(), node);

            Ok(GraphWriteResult {
                id,
                version: self.version,
                checksum,
            })
        })();
        match write_result {
            Ok(write) => Ok((write, acquired_pin_receipts)),
            Err(write_error) => match self.release_vector_payload_pins(&acquired_pin_receipts) {
                Ok(()) => Err(write_error),
                Err(release_error) => Err(vector_pin_cleanup_error(
                    "failed in-memory node publication",
                    write_error,
                    release_error,
                )),
            },
        }
    }

    /// Insert `node` only when no live record with the same id exists.
    /// Returns `Ok(None)` for ordinary dedup contention.
    pub fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if self.get_node(&node.id).is_some() {
            return Ok(None);
        }
        self.upsert_node(node).map(Some)
    }

    pub fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if !self.historical_node_history_complete {
            return Err(GraphStoreError::new(
                "historical_registry_incomplete",
                "cannot reserve a never-before-used node id from incomplete historical state",
            ));
        }
        if self.historical_node_ids.contains(&node.id) {
            return Ok(None);
        }
        self.upsert_node(node).map(Some)
    }

    /// Atomically set a node property when its current value equals `expected`.
    /// Missing properties compare as JSON null. `Value::Null` as `new_value`
    /// removes the property.
    pub fn compare_and_set_node_property(
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
        let mut node = self
            .nodes
            .get(id)
            .filter(|node| !node.tombstone)
            .cloned()
            .ok_or_else(|| {
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
        let mut properties = node
            .properties
            .as_object()
            .cloned()
            .unwrap_or_else(serde_json::Map::new);
        if new_value.is_null() {
            properties.remove(property);
        } else {
            properties.insert(property.to_string(), new_value);
        }
        node.properties = Value::Object(properties);
        self.upsert_node(node).map(Some)
    }

    pub fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        let Some(batch) = prepare_conditional_batch(self, conditions, batch)? else {
            return Ok(false);
        };
        let mut staged = self.clone();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => {
                    staged.upsert_node(node)?;
                }
                GraphMutation::EdgeUpsert(edge) => {
                    staged.upsert_edge(edge)?;
                }
                GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                    return Err(GraphStoreError::new(
                        "conditional_batch_delete_requires_tombstone",
                        "conditional in-memory batches require explicit tombstone records",
                    ));
                }
            }
        }
        *self = staged;
        Ok(true)
    }

    pub(crate) fn apply_recovered_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        self.apply_recovered_node_with_prepared(node, &mut BTreeMap::new())
    }

    pub(crate) fn apply_recovered_node_with_prepared(
        &mut self,
        mut node: NodeRecord,
        prepared: &mut BTreeMap<String, Vec<u8>>,
    ) -> GraphStoreResult<()> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        node.labels = normalize_labels(node.labels);
        if self.cold_nonresident_node_ids.remove(&node.id) {
            self.committed_footprint.remove_cold_nonresident_id(&node.id);
        }
        self.record_historical_node_id(&node.id);
        if node.content_hash.is_none() {
            node.content_hash = Some(node.checksum()?);
        }
        if let Some(existing) = self.nodes.get(&node.id).cloned() {
            self.committed_footprint.remove_node_record(&existing);
            self.remove_node_indexes(&existing);
        }
        if !node.tombstone {
            self.add_node_indexes(&node);
            self.auto_index_vectors_with_prepared(&node, prepared)?;
        }
        self.version = self.version.max(node.version);
        self.committed_footprint.add_node_record(&node);
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub fn upsert_edge(&mut self, mut edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        validate_edge_shape(&edge)?;
        self.require_live_endpoint(&edge, "from", &edge.from_id)?;
        self.require_live_endpoint(&edge, "to", &edge.to_id)?;

        let prior = self.edges.get(&edge.id);
        crate::graph::linked_to::stamp_first_version(&mut edge, prior, self.version + 1)?;

        let parent_hash = match self.edges.get(&edge.id) {
            Some(existing) => Some(match &existing.content_hash {
                Some(hash) => hash.clone(),
                None => existing.checksum()?,
            }),
            None => None,
        };
        if let Some(existing) = self.edges.get(&edge.id).cloned() {
            self.committed_footprint.remove_edge_record(&existing);
            self.remove_edge_indexes(&existing);
        }

        self.version += 1;
        edge.version = self.version;
        let content_hash = edge.checksum()?;
        if edge.parent_hashes.is_empty() {
            if let Some(parent_hash) = parent_hash.filter(|parent| parent != &content_hash) {
                edge.parent_hashes.push(parent_hash);
            }
        }
        edge.content_hash = Some(content_hash.clone());
        let checksum = edge.checksum()?;
        let id = edge.id.clone();
        // Serialize the final record, after the version/hash rewrite, so the
        // counter matches the bytes the snapshot writer will emit.
        self.committed_footprint.add_edge_record(&edge);
        let (ordinal, _) = self.edges.insert(id.clone(), edge.clone());
        if !edge.tombstone {
            self.add_edge_indexes(&edge, ordinal);
        }

        Ok(GraphWriteResult {
            id,
            version: self.version,
            checksum,
        })
    }

    pub(crate) fn apply_recovered_edge(&mut self, mut edge: EdgeRecord) -> GraphStoreResult<()> {
        validate_edge_shape(&edge)?;
        if edge.content_hash.is_none() {
            edge.content_hash = Some(edge.checksum()?);
        }
        if !edge.tombstone {
            self.require_live_endpoint(&edge, "from", &edge.from_id)?;
            self.require_live_endpoint(&edge, "to", &edge.to_id)?;
        }
        if let Some(existing) = self.edges.get(&edge.id).cloned() {
            self.committed_footprint.remove_edge_record(&existing);
            self.remove_edge_indexes(&existing);
        }
        self.version = self.version.max(edge.version);
        self.committed_footprint.add_edge_record(&edge);
        let (ordinal, _) = self.edges.insert(edge.id.clone(), edge.clone());
        if !edge.tombstone {
            self.add_edge_indexes(&edge, ordinal);
        }
        Ok(())
    }

    pub fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        let now = now_ms();
        self.nodes
            .get(id)
            .filter(|node| !node.tombstone && !node_is_expired(node, now))
    }

    /// Like `get_node`, but returns expired nodes too. Used by audit /
    /// forensics callers and by the sweep loop. Tombstoned nodes are
    /// still filtered out (tombstone is a stronger signal than TTL).
    pub fn get_node_including_expired(&self, id: &str) -> Option<&NodeRecord> {
        self.nodes.get(id).filter(|node| !node.tombstone)
    }

    pub fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        self.nodes.get(id).cloned()
    }

    pub fn has_historical_node_id(&self, id: &str) -> bool {
        self.historical_node_ids.contains(id)
    }

    pub fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        self.edges.get(id).filter(|edge| !edge.tombstone)
    }

    pub fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        sorted_values(self.label_index.get(label))
    }

    pub fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        self.edge_type_index
            .get(edge_type)
            .into_iter()
            .flatten()
            .filter_map(|ordinal| self.edges.get_by_ordinal(*ordinal))
            .map(|edge| edge.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        self.label_index
            .iter()
            .filter(|(candidate, _)| candidate.eq_ignore_ascii_case(label))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        self.edge_type_index
            .iter()
            .filter(|(candidate, _)| candidate.eq_ignore_ascii_case(edge_type))
            .flat_map(|(_, ordinals)| {
                ordinals
                    .iter()
                    .filter_map(|ordinal| self.edges.get_by_ordinal(*ordinal))
                    .map(|edge| edge.id.clone())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        let Some(token) = property_index_token(value) else {
            return Vec::new();
        };
        sorted_values(self.property_index.get(&(key.to_string(), token)))
    }

    /// Exact lookup over a raw UTF-8 string property. Unlike the generic JSON
    /// query seam, this never asks callers to JSON-quote or escape the value;
    /// interior NUL bytes remain ordered path-segment separators.
    pub fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        let key = key.trim();
        if key.is_empty() {
            return Vec::new();
        }
        sorted_values(
            self.property_index
                .get(&(key.to_string(), string_property_index_token(raw))),
        )
    }

    /// Ordered raw-prefix scan over one string-property index.
    ///
    /// Results are ordered first by the raw property bytes and then by node id.
    /// A [`crate::doc_tree::PathPrefix`] value therefore scans exactly one graph
    /// filesystem subtree without matching a sibling such as `a/b2` for `a/b`.
    pub fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        let key = key.trim();
        if key.is_empty() {
            return Vec::new();
        }
        let token_prefix = string_property_index_token(raw_prefix);
        self.property_index
            .range((key.to_string(), token_prefix.clone())..)
            .take_while(|((candidate_key, token), _)| {
                candidate_key == key && token.starts_with(&token_prefix)
            })
            .flat_map(|(_, node_ids)| node_ids.iter().cloned())
            .collect()
    }

    /// Ordered raw-prefix scan intersected with one live label index.
    ///
    /// This keeps graph-filesystem scans from materializing every unrelated
    /// path-bearing CodeSymbol/Memory id before filtering. Results retain the
    /// same raw-property then node-id order as the unqualified scan.
    pub fn node_ids_for_label_and_string_property_prefix(
        &self,
        label: &str,
        key: &str,
        raw_prefix: &str,
    ) -> Vec<String> {
        let label = label.trim();
        let key = key.trim();
        let Some(label_ids) = self.label_index.get(label) else {
            return Vec::new();
        };
        if key.is_empty() {
            return Vec::new();
        }
        let token_prefix = string_property_index_token(raw_prefix);
        self.property_index
            .range((key.to_string(), token_prefix.clone())..)
            .take_while(|((candidate_key, token), _)| {
                candidate_key == key && token.starts_with(&token_prefix)
            })
            .flat_map(|(_, node_ids)| {
                node_ids
                    .iter()
                    .filter(|node_id| label_ids.contains(*node_id))
                    .cloned()
            })
            .collect()
    }

    pub fn labels(&self) -> Vec<String> {
        self.label_index.keys().cloned().collect()
    }

    pub fn edge_types(&self) -> Vec<String> {
        self.edge_type_index.keys().cloned().collect()
    }

    pub fn property_keys(&self) -> Vec<String> {
        self.property_index
            .keys()
            .map(|(key, _)| key.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        let mut candidate_ids: Option<BTreeSet<String>> = None;
        if let Some(label) = query.normalized_label() {
            merge_candidates(&mut candidate_ids, self.label_index.get(&label).cloned());
        }
        for (key, value) in &query.properties {
            let key = key.trim();
            if key.is_empty() {
                return Vec::new();
            }
            let Some(token) = property_index_token(value) else {
                return Vec::new();
            };
            merge_candidates(
                &mut candidate_ids,
                self.property_index.get(&(key.to_string(), token)).cloned(),
            );
        }

        let ids = candidate_ids.unwrap_or_else(|| {
            self.nodes
                .values()
                .filter(|node| !node.tombstone)
                .map(|node| node.id.clone())
                .collect()
        });
        // Branch by `include_expired`: the audit path uses
        // `get_node_including_expired` (skips TTL filter); the default
        // path uses `get_node` (applies TTL filter via `node_is_expired`).
        // Branching here keeps the hot path identical to pre-TTL behavior.
        if query.include_expired {
            ids.into_iter()
                .filter_map(|id| self.get_node_including_expired(&id).cloned())
                .take(query.bounded_limit())
                .collect()
        } else {
            ids.into_iter()
                .filter_map(|id| self.get_node(&id).cloned())
                .take(query.bounded_limit())
                .collect()
        }
    }

    /// Read one stable, id-ordered label page without cloning the complete
    /// label index. Used by bounded background maintenance backfills.
    pub fn nodes_for_label_after_id(
        &self,
        label: &str,
        after_id: Option<&str>,
        limit: usize,
    ) -> Vec<NodeRecord> {
        let Some(ids) = self.label_index.get(label.trim()) else {
            return Vec::new();
        };
        let limit = limit.max(1);
        match after_id.map(str::trim).filter(|value| !value.is_empty()) {
            Some(after_id) => ids
                .range((Excluded(after_id.to_string()), Unbounded))
                .filter_map(|id| self.get_node_including_expired(id).cloned())
                .take(limit)
                .collect(),
            None => ids
                .iter()
                .filter_map(|id| self.get_node_including_expired(id).cloned())
                .take(limit)
                .collect(),
        }
    }

    /// Read one stable, id-ordered page from a label index while applying the
    /// normal live-node and TTL visibility rules.
    pub fn live_nodes_for_label_after_id(
        &self,
        label: &str,
        after_id: Option<&str>,
        limit: usize,
    ) -> Vec<NodeRecord> {
        let Some(ids) = self.label_index.get(label.trim()) else {
            return Vec::new();
        };
        let limit = limit.max(1);
        match after_id.map(str::trim).filter(|value| !value.is_empty()) {
            Some(after_id) => ids
                .range((Excluded(after_id.to_string()), Unbounded))
                .filter_map(|id| self.get_node(id).cloned())
                .take(limit)
                .collect(),
            None => ids
                .iter()
                .filter_map(|id| self.get_node(id).cloned())
                .take(limit)
                .collect(),
        }
    }

    pub fn memory_documents_by_updated_at(&self, query: MemoryDocumentQuery) -> Vec<NodeRecord> {
        self.try_memory_documents_by_updated_at(query)
            .expect("in-memory memory-document access-method query failed")
    }

    pub fn try_memory_documents_by_updated_at(
        &self,
        query: MemoryDocumentQuery,
    ) -> GraphStoreResult<Vec<NodeRecord>> {
        MemoryDocumentLens::bind_with_registry(self, self.derived_access_methods.registry())?
            .query(query)
    }

    pub fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        let mut edge_ordinals = BTreeSet::new();
        let include_expired = query.include_expired;
        match &query.edge_type {
            Some(edge_type) => {
                let key = (query.node_id.clone(), edge_type.clone());
                let index = match query.direction {
                    Direction::Out => &self.out_adjacency,
                    Direction::In => &self.in_adjacency,
                };
                if let Some(index_edge_ordinals) = index.get(&key) {
                    edge_ordinals.extend(index_edge_ordinals.iter().copied());
                }
            }
            None => {
                let index = match query.direction {
                    Direction::Out => &self.out_adjacency,
                    Direction::In => &self.in_adjacency,
                };
                for ((_node_id, _edge_type), index_edge_ordinals) in index
                    .range((query.node_id.clone(), String::new())..)
                    .take_while(|((node_id, _), _)| node_id == &query.node_id)
                {
                    edge_ordinals.extend(index_edge_ordinals.iter().copied());
                }
            }
        }

        let mut edges = edge_ordinals
            .into_iter()
            .filter_map(|ordinal| self.edges.get_by_ordinal(ordinal))
            .collect::<Vec<_>>();
        edges.sort_unstable_by(|left, right| left.id.cmp(&right.id));

        let mut hits = Vec::new();
        for edge in edges {
            let node_id = match query.direction {
                Direction::Out => edge.to_id.clone(),
                Direction::In => edge.from_id.clone(),
            };
            // Branch by include_expired so neighbors of TTL-expired targets
            // are filtered out by default (matches query_nodes semantics).
            let target_visible = if include_expired {
                self.get_node_including_expired(&node_id).is_some()
            } else {
                self.get_node(&node_id).is_some()
            };
            if !target_visible {
                continue;
            }
            hits.push(NeighborHit {
                edge_id: edge.id.clone(),
                node_id,
                edge_type: edge.edge_type.clone(),
                confidence: edge.confidence,
                epistemic_type: edge.epistemic_type.clone(),
            });
        }
        hits
    }

    pub fn epistemic_neighbors(
        &self,
        node_id: &str,
        epistemic_types: Option<&[EpistemicType]>,
        min_confidence: Option<f64>,
        max_depth: Option<usize>,
    ) -> Vec<(EdgeRecord, NodeRecord)> {
        let max_depth = max_depth.unwrap_or(1);
        let min_conf = min_confidence.unwrap_or(0.0);
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut queue: std::collections::VecDeque<(String, usize)> =
            std::collections::VecDeque::new();
        let mut results: Vec<(EdgeRecord, NodeRecord)> = Vec::new();

        visited.insert(node_id.to_string());
        queue.push_back((node_id.to_string(), 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for hits in [
                self.neighbors(NeighborQuery::out(&current_id)),
                self.neighbors(NeighborQuery::in_(&current_id)),
            ] {
                for hit in hits {
                    if visited.contains(&hit.node_id) {
                        continue;
                    }
                    let Some(edge) = self.get_edge(&hit.edge_id) else {
                        continue;
                    };
                    if edge.effective_confidence() < min_conf {
                        continue;
                    }
                    if let Some(types) = epistemic_types {
                        match &edge.epistemic_type {
                            Some(et) if types.contains(et) => {}
                            _ => continue,
                        }
                    }
                    let Some(node) = self.get_node(&hit.node_id) else {
                        continue;
                    };
                    visited.insert(hit.node_id.clone());
                    results.push((edge.clone(), node.clone()));
                    queue.push_back((hit.node_id, depth + 1));
                }
            }
        }
        results
    }

    pub fn stats(&self) -> GraphStats {
        let mut stats = self.stats_without_memory_estimate();
        stats.memory_bytes = self.estimated_memory_bytes();
        stats
    }

    pub fn stats_without_memory_estimate(&self) -> GraphStats {
        GraphStats {
            version: self.version,
            nodes_total: self.nodes.values().filter(|node| !node.tombstone).count(),
            edges_total: self.edges.values().filter(|edge| !edge.tombstone).count(),
            labels_total: self.label_index.len(),
            edge_types_total: self.edge_type_index.len(),
            property_keys_total: self.property_key_count(),
            property_indexes_total: self.property_index.len(),
            memory_bytes: 0,
            memory_quota_bytes: 0,
        }
    }

    pub fn designate_vector_property(
        &mut self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> GraphStoreResult<()> {
        if self.vector_designation_is_current_and_migrated(label, property_name, dimension) {
            return Ok(());
        }
        let acquired_pin_receipts =
            self.designate_vector_property_retaining_pins(label, property_name, dimension)?;
        self.release_vector_payload_pins(&acquired_pin_receipts)
    }

    pub(crate) fn vector_designation_is_current_and_migrated(
        &self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> bool {
        self.vector_designations
            .get(&(label.to_string(), property_name.to_string()))
            .is_some_and(|current| *current == dimension)
            && self
                .legacy_vector_node_ids_for_designation(label, property_name)
                .is_empty()
    }

    pub(crate) fn designate_vector_property_retaining_pins(
        &mut self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> GraphStoreResult<Vec<VectorPayloadPinReceipt>> {
        let mut staged = self.clone();
        let acquired_pin_receipts =
            staged.designate_vector_property_in_place(label, property_name, dimension)?;
        *self = staged;
        Ok(acquired_pin_receipts)
    }

    pub(crate) fn designate_vector_property_in_place(
        &mut self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> GraphStoreResult<Vec<VectorPayloadPinReceipt>> {
        let migration_ids = self.legacy_vector_node_ids_for_designation(label, property_name);
        self.install_vector_designation(label, property_name, dimension)?;
        let mut acquired_pins = VectorPayloadPinGuard::new(self.vector_payload_store_handle());
        for id in migration_ids {
            let node = self.nodes.get(&id).cloned().ok_or_else(|| {
                GraphStoreError::new(
                    "vector_migration_node_missing",
                    format!("vector migration node {id} disappeared"),
                )
            })?;
            let (_, pins) = self.upsert_node_retaining_vector_pins(node)?;
            acquired_pins.extend(pins);
        }
        Ok(acquired_pins.into_receipts())
    }

    pub(crate) fn install_vector_designation(
        &mut self,
        label: &str,
        property_name: &str,
        dimension: usize,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_vector_designation",
                "label and property_name must be non-empty".to_string(),
            ));
        }
        if dimension == 0 {
            return Err(GraphStoreError::new(
                "invalid_vector_designation",
                "dimension must be > 0".to_string(),
            ));
        }
        let key = (label.to_string(), property_name.to_string());
        let previous_dimension = self.vector_designations.insert(key.clone(), dimension);
        if previous_dimension != Some(dimension) {
            self.vector_indexes.insert(
                key.clone(),
                VectorIndex::new(dimension, Self::vector_index_seed(&key.0, &key.1)),
            );
        } else {
            self.vector_indexes.entry(key.clone()).or_insert_with(|| {
                VectorIndex::new(dimension, Self::vector_index_seed(&key.0, &key.1))
            });
        }
        let nodes = self.nodes.values().cloned().collect::<Vec<_>>();
        for node in nodes {
            if node.tombstone {
                continue;
            }
            if !node.labels.iter().any(|l| l == label) {
                continue;
            }
            self.auto_index_vectors(&node)?;
        }
        Ok(())
    }

    pub fn vector_designations(&self) -> Vec<VectorDesignation> {
        let mut designations: Vec<_> = self
            .vector_designations
            .iter()
            .map(|((label, property), &dimension)| VectorDesignation {
                label: label.clone(),
                property: property.clone(),
                dimension,
            })
            .collect();
        // Retained revision roots commit to this order, independently of the
        // hash seed or the order in which indexes were installed.
        designations.sort_by(|a, b| {
            (&a.label, &a.property, a.dimension).cmp(&(&b.label, &b.property, b.dimension))
        });
        designations
    }

    pub fn clear_vector_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_vector_designation",
                "label and property_name must be non-empty",
            ));
        }
        let key = (label.to_string(), property_name.to_string());
        self.vector_designations.remove(&key);
        self.vector_indexes.remove(&key);
        Ok(())
    }

    /// Returns immutable exact-search cardinality for one designated index.
    pub fn vector_index_shape(&self, label: &str, property_name: &str) -> Option<VectorIndexShape> {
        self.vector_indexes
            .get(&(label.to_string(), property_name.to_string()))
            .map(|index| VectorIndexShape {
                dimension: index.dimension,
                point_count: index.points.len(),
            })
    }

    pub fn designate_ordered_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_ordered_designation",
                "label and property_name must be non-empty".to_string(),
            ));
        }
        let key = (label.to_string(), property_name.to_string());
        if !self.ordered_indexes.contains_key(&key) {
            self.ordered_indexes
                .entry(key.clone())
                .or_insert_with(OrderedIndex::persistent);
            self.committed_footprint
                .add_ordered_designation(label, property_name);
        }
        let entries = self
            .nodes
            .values()
            .filter(|node| {
                !node.tombstone && node.labels.iter().any(|candidate| candidate == label)
            })
            .filter_map(|node| {
                numeric_property(&node.properties, property_name)
                    .map(|score| (node.id.as_bytes().to_vec(), score))
            })
            .collect::<Vec<_>>();
        let mut added_entry_bytes = 0usize;
        if let Some(index) = self.ordered_indexes.get_mut(&key) {
            for (member, score) in entries {
                let member_len = member.len();
                if index.zadd(member, score)? {
                    added_entry_bytes =
                        added_entry_bytes.saturating_add(ordered_member_bytes(member_len));
                }
            }
        }
        self.committed_footprint
            .add_ordered_entry_bytes(added_entry_bytes);
        Ok(())
    }

    pub fn ordered_designations(&self) -> Vec<OrderedDesignation> {
        let mut designations: Vec<_> = self
            .ordered_indexes
            .keys()
            .map(|(label, property)| OrderedDesignation {
                label: label.clone(),
                property: property.clone(),
            })
            .collect();
        designations.sort_by(|a, b| (&a.label, &a.property).cmp(&(&b.label, &b.property)));
        designations
    }

    pub fn clear_ordered_property(
        &mut self,
        label: &str,
        property_name: &str,
    ) -> GraphStoreResult<()> {
        if label.trim().is_empty() || property_name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_ordered_designation",
                "label and property_name must be non-empty",
            ));
        }
        let removed = self
            .ordered_indexes
            .remove(&(label.to_string(), property_name.to_string()));
        if let Some(index) = removed {
            // One walk of the single index being dropped: its entries are folded
            // into a shared sum, so their share is measured, not guessed. This is
            // an administrative path, never the record-write path.
            self.committed_footprint.remove_ordered_designation(
                label,
                property_name,
                index.estimated_entry_bytes(),
            );
        }
        Ok(())
    }

    pub fn ordered_range_by_score(
        &self,
        label: &str,
        property_name: &str,
        min: f64,
        max: f64,
        limit: Option<usize>,
    ) -> GraphStoreResult<Vec<(String, f64)>> {
        let Some(index) = self
            .ordered_indexes
            .get(&(label.to_string(), property_name.to_string()))
        else {
            return Ok(Vec::new());
        };
        index
            .zrange_by_score(min, max, limit)?
            .into_iter()
            .map(|(member, score)| {
                String::from_utf8(member)
                    .map(|member| (member, score))
                    .map_err(|err| GraphStoreError::new("invalid_ordered_member", err.to_string()))
            })
            .collect()
    }

    pub fn ordered_score(&self, label: &str, property_name: &str, node_id: &str) -> Option<f64> {
        self.ordered_indexes
            .get(&(label.to_string(), property_name.to_string()))
            .and_then(|index| index.zscore(node_id.as_bytes()))
    }

    pub fn index_vector(
        &mut self,
        node_id: &str,
        property_name: &str,
        vector: &[f32],
    ) -> GraphStoreResult<()> {
        let node = self.nodes.get(node_id).ok_or_else(|| {
            GraphStoreError::new("node_not_found", format!("no node with id {node_id}"))
        })?;
        let matching_label = node.labels.iter().find(|label| {
            self.vector_designations
                .contains_key(&(label.to_string(), property_name.to_string()))
        });
        let label = matching_label
            .ok_or_else(|| {
                GraphStoreError::new(
                    "no_vector_designation",
                    format!(
                        "no vector designation covers property {property_name} for node {node_id}"
                    ),
                )
            })?
            .clone();
        let key = (label, property_name.to_string());
        let expected_dim = self.vector_designations[&key];
        if vector.len() != expected_dim {
            return Err(GraphStoreError::new(
                "dimension_mismatch",
                format!("expected {expected_dim} dimensions, got {}", vector.len()),
            ));
        }
        let idx = self.vector_indexes.entry(key.clone()).or_insert_with(|| {
            VectorIndex::new(expected_dim, Self::vector_index_seed(&key.0, &key.1))
        });
        idx.insert(node_id, vector);
        Ok(())
    }

    fn vector_index_seed(label: &str, property: &str) -> u64 {
        let digest = blake3::hash(format!("{label}/{property}").as_bytes());
        u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("blake3 digest"))
    }

    pub fn vector_search(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        Ok(self
            .vector_search_with_receipt(label, property_name, query, k)?
            .hits)
    }

    pub fn vector_search_with_receipt(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        let mut aggregate = VectorSearchReceipt::default();
        let mut matched_indexes = 0usize;
        let mut all_indexes_accelerated = true;
        let mut fallback_reasons = Vec::new();
        for ((index_label, index_property), index) in &self.vector_indexes {
            if index_property != property_name || label.is_some_and(|wanted| wanted != index_label)
            {
                continue;
            }
            if query.len() != index.dimension {
                return Err(GraphStoreError::new(
                    "dimension_mismatch",
                    format!(
                        "query dimension {} does not match index dimension {}",
                        query.len(),
                        index.dimension,
                    ),
                ));
            }
            let receipt = index.search_with_receipt(query, k);
            matched_indexes = matched_indexes.saturating_add(1);
            all_indexes_accelerated &= receipt.work.backend == VectorSearchBackend::Turbovec;
            if let Some(reason) = receipt
                .work
                .fallback_reason
                .as_ref()
                .filter(|reason| !reason.is_empty())
            {
                fallback_reasons.push(reason.clone());
            }
            aggregate.hits.extend(receipt.hits);
            aggregate.work.indexed_points = aggregate
                .work
                .indexed_points
                .saturating_add(receipt.work.indexed_points);
            aggregate.work.distance_evaluations = aggregate
                .work
                .distance_evaluations
                .saturating_add(receipt.work.distance_evaluations);
            aggregate.work.accelerated_candidates = aggregate
                .work
                .accelerated_candidates
                .saturating_add(receipt.work.accelerated_candidates);
            aggregate.work.index_rebuilds = aggregate
                .work
                .index_rebuilds
                .saturating_add(receipt.work.index_rebuilds);
        }
        aggregate.work.backend = if matched_indexes > 0 && all_indexes_accelerated {
            VectorSearchBackend::Turbovec
        } else {
            VectorSearchBackend::ExactCosine
        };
        aggregate.work.fallback_reason = serialize_vector_fallback_reasons(fallback_reasons);
        aggregate.hits.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        aggregate.hits.truncate(k);
        aggregate.work.retained_candidates = aggregate.hits.len();
        Ok(aggregate)
    }

    /// Search only an externally admitted DATAWAVE posting through this
    /// store's existing designated index. Admission remains the caller's job.
    pub fn vector_search_candidates_with_receipt(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        candidates: &BTreeSet<String>,
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        let index = self
            .vector_indexes
            .get(&(label.to_string(), property.to_string()))
            .ok_or_else(|| {
                GraphStoreError::new(
                    "missing_vector_index",
                    "candidate search requires an existing designated index",
                )
            })?;
        index.search_candidates_with_receipt(query, candidates, k)
    }

    pub fn hybrid_search(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
        graph_seeds: &[String],
        max_hops: usize,
        alpha: f32,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        self.hybrid_search_with_config(
            label,
            property_name,
            query,
            k,
            graph_seeds,
            max_hops,
            &HybridScoringConfig::default().with_alpha(alpha),
        )
    }

    pub fn hybrid_search_with_config(
        &self,
        label: Option<&str>,
        property_name: &str,
        query: &[f32],
        k: usize,
        graph_seeds: &[String],
        max_hops: usize,
        config: &HybridScoringConfig,
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        let vector_results = self.vector_search(label, property_name, query, k * 2)?;
        if vector_results.is_empty() {
            return Ok(Vec::new());
        }
        let graph_scores = self.hybrid_graph_scores(graph_seeds, max_hops, config);
        let max_vec_dist = vector_results
            .iter()
            .map(|(_, d)| *d)
            .fold(0.0_f32, f32::max)
            .max(1e-10);
        let alpha = config.alpha.clamp(0.0, 1.0);
        let mut scored: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|(node_id, vec_dist)| {
                let vector_score = 1.0 - (vec_dist / max_vec_dist);
                let graph_score = graph_scores.get(&node_id).copied().unwrap_or(0.0);
                let final_score = (1.0 - alpha) * vector_score + alpha * graph_score;
                (node_id, final_score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    pub(crate) fn hybrid_graph_scores(
        &self,
        graph_seeds: &[String],
        max_hops: usize,
        config: &HybridScoringConfig,
    ) -> HashMap<String, f32> {
        let mut adjacency: HashMap<String, Vec<&EdgeRecord>> = HashMap::new();
        for edge in self.edges.values().filter(|edge| !edge.tombstone) {
            adjacency
                .entry(edge.from_id.clone())
                .or_default()
                .push(edge);
        }
        let mut best: HashMap<String, f32> = HashMap::new();
        let mut queue: VecDeque<(String, usize, f32)> = VecDeque::new();
        for seed in graph_seeds {
            best.insert(seed.clone(), 1.0);
            queue.push_back((seed.clone(), 0, 1.0));
        }
        while let Some((node_id, depth, path_score)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }
            let Some(edges) = adjacency.get(&node_id) else {
                continue;
            };
            for edge in edges {
                let confidence = if config.confidence_weighted_graph_distance {
                    edge.effective_confidence() as f32
                } else {
                    1.0
                };
                let edge_weight = config.edge_type_weight(&edge.edge_type);
                let next_depth = depth + 1;
                let next_score = path_score * edge_weight * confidence / (1.0 + next_depth as f32);
                let should_update = best
                    .get(&edge.to_id)
                    .map(|current| next_score.abs() > current.abs())
                    .unwrap_or(true);
                if should_update {
                    best.insert(edge.to_id.clone(), next_score.clamp(-1.0, 1.0));
                    queue.push_back((edge.to_id.clone(), next_depth, next_score));
                }
            }
        }
        best
    }

    pub(crate) fn auto_index_vectors(&mut self, node: &NodeRecord) -> GraphStoreResult<()> {
        self.auto_index_vectors_with_prepared(node, &mut BTreeMap::new())
    }

    pub(crate) fn auto_index_vectors_with_prepared(
        &mut self,
        node: &NodeRecord,
        prepared: &mut BTreeMap<String, Vec<u8>>,
    ) -> GraphStoreResult<()> {
        let designations: Vec<((String, String), usize)> = self
            .vector_designations
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for ((label, property), dimension) in designations {
            if !node.labels.iter().any(|l| l == &label) {
                continue;
            }
            let Some(value) = node.properties.get(&property) else {
                continue;
            };
            enum CandidateVector {
                Legacy(Vec<f32>),
                Packed(Vec<u8>),
            }
            let candidate = if let Some(vector) = decode_legacy_vector_array(value)? {
                if vector.len() != dimension {
                    continue;
                }
                CandidateVector::Legacy(vector)
            } else {
                let (reference, packed) = match prepared.remove(&property) {
                    Some(packed) => (VectorPropertyRef::from_value(value)?, packed),
                    None => {
                        let Some(resolved) = read_vector_property_packed(
                            node,
                            &property,
                            self.vector_payload_store.inner.as_ref(),
                        )?
                        else {
                            continue;
                        };
                        resolved
                    }
                };
                if reference.dimension != dimension {
                    // Re-designation deliberately leaves old, well-formed
                    // payloads unindexed until the producer re-embeds them at
                    // the new dimension. Ordinary upserts reject this mismatch
                    // during designated-property preflight above.
                    continue;
                }
                CandidateVector::Packed(packed)
            };
            let key = (label, property.clone());
            let idx = self.vector_indexes.entry(key.clone()).or_insert_with(|| {
                VectorIndex::new(dimension, Self::vector_index_seed(&key.0, &key.1))
            });
            match candidate {
                CandidateVector::Legacy(vector) => idx.insert(&node.id, &vector),
                CandidateVector::Packed(packed) => idx.insert_packed(&node.id, packed)?,
            }
        }
        Ok(())
    }

    pub(crate) fn auto_index_ordered(&mut self, node: &NodeRecord) -> GraphStoreResult<()> {
        let designations = self
            .ordered_indexes
            .keys()
            .cloned()
            .collect::<Vec<(String, String)>>();
        let mut added_entry_bytes = 0usize;
        for (label, property) in designations {
            if !node.labels.iter().any(|candidate| candidate == &label) {
                continue;
            }
            let Some(score) = numeric_property(&node.properties, &property) else {
                continue;
            };
            if let Some(index) = self.ordered_indexes.get_mut(&(label, property)) {
                if index.zadd(node.id.as_bytes().to_vec(), score)? {
                    added_entry_bytes =
                        added_entry_bytes.saturating_add(ordered_member_bytes(node.id.len()));
                }
            }
        }
        self.committed_footprint
            .add_ordered_entry_bytes(added_entry_bytes);
        Ok(())
    }

    /// Attribute the live graph without cloning records or constructing a
    /// serialized snapshot. The only report-sized allocations are the vector
    /// index rows and the static exclusions list.
    pub fn memory_attribution(&self) -> GraphMemoryAttribution {
        let mut primary_node_keys = StringAllocationStats::default();
        let mut primary_edge_keys = StringAllocationStats::default();
        let mut node_record_strings = StringAllocationStats::default();
        let mut edge_record_strings = StringAllocationStats::default();
        let mut node_properties = JsonAllocationStats::default();
        let mut edge_properties = JsonAllocationStats::default();
        let mut record_vec_slots_capacity_bytes = 0usize;
        let mut live_nodes = 0usize;
        let mut live_edges = 0usize;

        for (key, node) in &self.nodes {
            if !node.tombstone {
                live_nodes = live_nodes.saturating_add(1);
            }
            primary_node_keys.observe(key);
            node_record_strings.observe(&node.id);
            for label in &node.labels {
                node_record_strings.observe(label);
            }
            if let Some(content_hash) = &node.content_hash {
                node_record_strings.observe(content_hash);
            }
            for parent_hash in &node.parent_hashes {
                node_record_strings.observe(parent_hash);
            }
            record_vec_slots_capacity_bytes = record_vec_slots_capacity_bytes
                .saturating_add(
                    node.labels
                        .capacity()
                        .saturating_mul(std::mem::size_of::<String>()),
                )
                .saturating_add(
                    node.parent_hashes
                        .capacity()
                        .saturating_mul(std::mem::size_of::<String>()),
                );
            observe_json_allocation(&node.properties, &mut node_properties, 0);
        }

        for (key, edge) in self.edges.iter() {
            if !edge.tombstone {
                live_edges = live_edges.saturating_add(1);
            }
            primary_edge_keys.observe(key);
            edge_record_strings.observe(&edge.id);
            edge_record_strings.observe(&edge.from_id);
            edge_record_strings.observe(&edge.to_id);
            edge_record_strings.observe(&edge.edge_type);
            if let Some(content_hash) = &edge.content_hash {
                edge_record_strings.observe(content_hash);
            }
            for parent_hash in &edge.parent_hashes {
                edge_record_strings.observe(parent_hash);
            }
            record_vec_slots_capacity_bytes = record_vec_slots_capacity_bytes.saturating_add(
                edge.parent_hashes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<String>()),
            );
            observe_json_allocation(&edge.properties, &mut edge_properties, 0);
        }

        let mut historical_node_ids = StringAllocationStats::default();
        for node_id in &self.historical_node_ids {
            historical_node_ids.observe(node_id);
        }

        let label_index = string_index_allocation(&self.label_index);
        let edge_type_index = ordinal_string_index_allocation(&self.edge_type_index);
        let out_adjacency = ordinal_tuple_index_allocation(&self.out_adjacency);
        let in_adjacency = ordinal_tuple_index_allocation(&self.in_adjacency);
        let property_index = tuple_index_allocation(&self.property_index);

        let vector_indexes = self
            .vector_indexes
            .iter()
            .map(|((label, property), index)| {
                let mut node_ids = StringAllocationStats::default();
                for node_id in &index.node_ids {
                    node_ids.observe(node_id);
                }
                let mut slot_map_keys = StringAllocationStats::default();
                for node_id in index.slot_by_node_id.keys() {
                    slot_map_keys.observe(node_id);
                }
                let accelerated_index_present = index.resident_len().is_some();
                VectorIndexAllocationStats {
                    label: label.clone(),
                    property: property.clone(),
                    dimension: index.dimension,
                    point_count: index.points.len(),
                    point_slots_capacity_bytes: index
                        .points
                        .capacity()
                        .saturating_mul(std::mem::size_of::<VectorPoint>()),
                    point_values_capacity_bytes: index
                        .points
                        .iter()
                        .map(|point| {
                            point
                                .packed_le
                                .capacity()
                                .saturating_mul(std::mem::size_of::<u8>())
                        })
                        .sum(),
                    node_id_slots_capacity_bytes: index
                        .node_ids
                        .capacity()
                        .saturating_mul(std::mem::size_of::<String>()),
                    node_ids,
                    slot_map_capacity: index.slot_by_node_id.capacity(),
                    slot_map_entry_inline_bytes: index.slot_by_node_id.capacity().saturating_mul(
                        std::mem::size_of::<String>() + std::mem::size_of::<usize>(),
                    ),
                    slot_map_keys,
                    accelerated_index_present,
                    accelerated_index_bytes_accounted: false,
                }
            })
            .collect::<Vec<_>>();

        let record_inline_bytes = self
            .nodes
            .len()
            .saturating_mul(std::mem::size_of::<NodeRecord>())
            .saturating_add(self.edges.record_slots_capacity_bytes());
        record_vec_slots_capacity_bytes =
            record_vec_slots_capacity_bytes.saturating_add(self.edges.free_slots_capacity_bytes());
        let serialized_graph_bytes = serialized_json_bytes(&BorrowedGraphSnapshot {
            version: self.version,
            nodes: MapValues(&self.nodes),
            edges: EdgeValues(&self.edges),
            historical_node_ids: &self.historical_node_ids,
            historical_node_history_complete: self.historical_node_history_complete,
        });
        let accounted_capacity_lower_bound_bytes = record_inline_bytes
            .saturating_add(record_vec_slots_capacity_bytes)
            .saturating_add(primary_node_keys.inline_bytes)
            .saturating_add(primary_node_keys.capacity_bytes)
            .saturating_add(primary_edge_keys.inline_bytes)
            .saturating_add(primary_edge_keys.capacity_bytes)
            .saturating_add(node_record_strings.capacity_bytes)
            .saturating_add(edge_record_strings.capacity_bytes)
            .saturating_add(node_properties.accounted_capacity_bytes())
            .saturating_add(edge_properties.accounted_capacity_bytes())
            .saturating_add(historical_node_ids.inline_bytes)
            .saturating_add(historical_node_ids.capacity_bytes)
            .saturating_add(label_index.accounted_capacity_bytes())
            .saturating_add(edge_type_index.accounted_capacity_bytes())
            .saturating_add(out_adjacency.accounted_capacity_bytes())
            .saturating_add(in_adjacency.accounted_capacity_bytes())
            .saturating_add(property_index.accounted_capacity_bytes())
            .saturating_add(
                vector_indexes
                    .iter()
                    .map(VectorIndexAllocationStats::accounted_capacity_bytes)
                    .sum::<usize>(),
            );

        GraphMemoryAttribution {
            version: self.version,
            node_records: self.nodes.len(),
            live_nodes,
            tombstoned_nodes: self.nodes.len().saturating_sub(live_nodes),
            edge_records: self.edges.len(),
            live_edges,
            tombstoned_edges: self.edges.len().saturating_sub(live_edges),
            record_inline_bytes,
            record_vec_slots_capacity_bytes,
            primary_node_keys,
            primary_edge_keys,
            node_record_strings,
            edge_record_strings,
            node_properties,
            edge_properties,
            historical_node_ids,
            label_index,
            edge_type_index,
            out_adjacency,
            in_adjacency,
            property_index,
            vector_indexes,
            serialized_graph_bytes,
            accounted_capacity_lower_bound_bytes,
            exclusions: vec![
                "BTreeMap/BTreeSet node allocation and allocator metadata",
                "HashMap control bytes beyond inline slot estimates",
                // Frozen attribution-schema wording: G7 removed the bespoke
                // memory-document containers, so that category now contributes
                // zero bytes while the serialized G3 report stays compatible.
                "ordered and memory-document index container overhead",
                "provenance-owned strings outside edge properties",
                "vector payload-store heap (reported by the owning payload store)",
                "accelerated vector backend private allocations",
                "allocator fragmentation and retained free pages",
            ],
        }
    }

    /// Committed resident footprint of the graph, in bytes.
    ///
    /// Served from the incrementally maintained counters in
    /// `crate::graph::footprint`, so it is O(1): the quota path calls it on
    /// every commit batch, and it must not traverse the graph. For every state
    /// reachable through this store's mutation API it returns exactly the value
    /// [`Self::recompute_memory_bytes`] computes, on the same qualifier that
    /// oracle carries: a mutation applied directly to the node or edge maps,
    /// bypassing the mutation API, is not reflected in these counters.
    pub(crate) fn estimated_memory_bytes(&self) -> usize {
        self.committed_snapshot_bytes()
            .saturating_add(self.committed_footprint.derived_index_bytes())
            .saturating_add(self.committed_footprint.cold_nonresident_node_ids_bytes)
            // `Vec::capacity()` reads: genuinely O(1), so they stay computed
            // instead of being mirrored by a counter that would have to track
            // arena growth and reclamation.
            .saturating_add(self.edges.record_slots_capacity_bytes())
            .saturating_add(self.edges.free_slots_capacity_bytes())
    }

    /// Whole-graph recompute of [`Self::estimated_memory_bytes`]: the oracle.
    ///
    /// This is the original O(nodes + edges + index entries) implementation. It
    /// serde-serializes a borrowed snapshot of every node and edge into a
    /// byte-counting writer and walks every secondary index. That is the right
    /// cost for constructors, recovery/reopen, index rebuild, and tests, and the
    /// wrong cost for a record write, so it is never called on the ordinary
    /// write path.
    ///
    /// Every invocation bumps
    /// [`crate::graph::footprint::graph_memory_recompute_invocations`], which is
    /// how tests prove the write path stays off it.
    #[allow(dead_code)] // The oracle of the maintained counters; reached by tests
                        // and by the debug-build parity check at the end of
                        // `reset_committed_footprint_from_recompute`.
    pub(crate) fn recompute_memory_bytes(&self) -> usize {
        note_graph_memory_recompute();
        let snapshot_bytes = serialized_json_bytes(&BorrowedGraphSnapshot {
            version: self.version,
            nodes: MapValues(&self.nodes),
            edges: EdgeValues(&self.edges),
            historical_node_ids: &self.historical_node_ids,
            historical_node_history_complete: self.historical_node_history_complete,
        });
        snapshot_bytes
            + self.edges.record_slots_capacity_bytes()
            + self.edges.free_slots_capacity_bytes()
            + string_index_bytes(&self.label_index)
            + ordinal_string_index_bytes(&self.edge_type_index)
            + ordinal_tuple_index_bytes(&self.out_adjacency)
            + ordinal_tuple_index_bytes(&self.in_adjacency)
            + tuple_index_bytes(&self.property_index)
            + ordered_index_bytes(&self.ordered_indexes)
            + self
                .cold_nonresident_node_ids
                .iter()
                .map(|id| id.len().saturating_add(std::mem::size_of::<String>()))
                .sum::<usize>()
    }

    /// The serialized-snapshot term of [`Self::estimated_memory_bytes`], as
    /// maintained.
    ///
    /// Exposed so the accounting tests can compare it byte for byte against
    /// `serialized_json_bytes` over the live snapshot, which pins the
    /// framing/separator identity independently of the rest of the total.
    pub(crate) fn committed_snapshot_bytes(&self) -> usize {
        self.committed_footprint.snapshot_bytes(
            self.version,
            self.historical_node_history_complete,
            self.nodes.len(),
            self.edges.len(),
            self.historical_node_ids.len(),
        )
    }

    /// Re-derive the maintained counters from the graph itself.
    ///
    /// O(graph), so this belongs to constructors, recovery/reopen, and index
    /// rebuild, not to the write path. In debug builds the rebuilt counters are
    /// checked against the whole-graph oracle, so a mutation site that forgets a
    /// delta fails loudly at the next reopen or rebuild instead of quietly
    /// drifting the quota.
    pub(crate) fn reset_committed_footprint_from_recompute(&mut self) {
        self.committed_footprint = CommittedFootprint::from_store(self);
        debug_assert_eq!(
            self.estimated_memory_bytes(),
            self.recompute_memory_bytes(),
            "maintained committed footprint must equal the whole-graph recompute oracle"
        );
    }

    pub(crate) fn property_key_count(&self) -> usize {
        let mut previous: Option<&str> = None;
        let mut count = 0;
        for (key, _) in self.property_index.keys() {
            if previous != Some(key.as_str()) {
                previous = Some(key.as_str());
                count += 1;
            }
        }
        count
    }

    /// Conservatively estimate the post-commit footprint without cloning the
    /// tenant graph. Existing records remain in the baseline, so replacements
    /// are deliberately treated as additions. Every possible new record and
    /// derived-index contribution is added independently, including keys that
    /// the live indexes already contain. This may reject a write very near
    /// quota, but cannot accumulate across repeated replacements and avoids
    /// request-scale graph copies.
    ///
    /// The baseline is the maintained committed footprint, so this is O(1) plus
    /// one bound per mutation; deletes keep their historical zero contribution.
    pub(crate) fn projected_memory_bytes_upper_bound(&self, batch: &GraphMutationBatch) -> usize {
        batch
            .mutations
            .iter()
            // Reserve enough digits for any graph-version growth as well as
            // array separators for each appended record.
            .fold(
                self.estimated_memory_bytes().saturating_add(20),
                |projected, mutation| {
                    projected.saturating_add(match mutation {
                        GraphMutation::NodeUpsert(node) => self.node_memory_upper_bound(node),
                        GraphMutation::EdgeUpsert(edge) => edge_memory_upper_bound(edge),
                        GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => 0,
                    })
                },
            )
    }

    pub(crate) fn node_memory_upper_bound(&self, node: &NodeRecord) -> usize {
        let mut bytes = serialized_json_bytes(node)
            .saturating_add(1)
            .saturating_add(serialized_json_bytes(&node.id))
            .saturating_add(1);
        for label in &node.labels {
            bytes = bytes
                .saturating_add(label.len())
                .saturating_add(node.id.len());
        }
        for (key, token) in indexed_properties(&node.properties) {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(token.len())
                .saturating_add(node.id.len());
        }
        for (label, property) in self.ordered_indexes.keys() {
            bytes = bytes
                .saturating_add(label.len())
                .saturating_add(property.len())
                .saturating_add(node.id.len())
                .saturating_add(std::mem::size_of::<f64>());
        }
        bytes
    }
}

impl GraphWrite for InMemoryGraphStore {
    fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
        InMemoryGraphStore::upsert_node(self, node)
    }

    fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
        InMemoryGraphStore::upsert_edge(self, edge)
    }
}

impl GraphAtomic for InMemoryGraphStore {
    fn insert_node_if_absent(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        InMemoryGraphStore::insert_node_if_absent(self, node)
    }

    fn insert_node_if_never_existed(
        &mut self,
        node: NodeRecord,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        InMemoryGraphStore::insert_node_if_never_existed(self, node)
    }

    fn compare_and_set_node_property(
        &mut self,
        id: &str,
        property: &str,
        expected: &Value,
        new_value: Value,
    ) -> GraphStoreResult<Option<GraphWriteResult>> {
        InMemoryGraphStore::compare_and_set_node_property(self, id, property, expected, new_value)
    }

    fn commit_conditional_batch_atomic(
        &mut self,
        conditions: Vec<ConditionalNodePropertyUpdate>,
        batch: GraphMutationBatch,
    ) -> GraphStoreResult<bool> {
        InMemoryGraphStore::commit_conditional_batch_atomic(self, conditions, batch)
    }
}

impl GraphRead for InMemoryGraphStore {
    fn get_node(&self, id: &str) -> Option<&NodeRecord> {
        InMemoryGraphStore::get_node(self, id)
    }

    fn get_node_record(&self, id: &str) -> Option<NodeRecord> {
        self.nodes.get(id).cloned()
    }

    fn get_node_record_including_tombstone(&self, id: &str) -> Option<NodeRecord> {
        InMemoryGraphStore::get_node_record_including_tombstone(self, id)
    }

    fn has_historical_node_id(&self, id: &str) -> bool {
        InMemoryGraphStore::has_historical_node_id(self, id)
    }

    fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
        InMemoryGraphStore::get_edge(self, id)
    }

    fn get_edge_record(&self, id: &str) -> Option<EdgeRecord> {
        self.edges.get(id).cloned()
    }

    fn live_node_records(&self) -> Vec<NodeRecord> {
        let now = now_ms();
        self.nodes
            .values()
            .filter(|node| !node.tombstone && !node_is_expired(node, now))
            .cloned()
            .collect()
    }

    fn live_edge_records(&self) -> Vec<EdgeRecord> {
        self.live_edges_snapshot()
    }

    fn node_ids_for_label(&self, label: &str) -> Vec<String> {
        InMemoryGraphStore::node_ids_for_label(self, label)
    }

    fn edge_ids_for_type(&self, edge_type: &str) -> Vec<String> {
        InMemoryGraphStore::edge_ids_for_type(self, edge_type)
    }

    fn node_ids_for_label_folded(&self, label: &str) -> Vec<String> {
        InMemoryGraphStore::node_ids_for_label_folded(self, label)
    }

    fn edge_ids_for_type_folded(&self, edge_type: &str) -> Vec<String> {
        InMemoryGraphStore::edge_ids_for_type_folded(self, edge_type)
    }

    fn node_ids_for_property(&self, key: &str, value: &Value) -> Vec<String> {
        InMemoryGraphStore::node_ids_for_property(self, key, value)
    }

    fn node_ids_for_string_property_exact(&self, key: &str, raw: &str) -> Vec<String> {
        InMemoryGraphStore::node_ids_for_string_property_exact(self, key, raw)
    }

    fn node_ids_for_string_property_prefix(&self, key: &str, raw_prefix: &str) -> Vec<String> {
        InMemoryGraphStore::node_ids_for_string_property_prefix(self, key, raw_prefix)
    }

    fn graph_version(&self) -> u64 {
        self.version
    }

    fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
        InMemoryGraphStore::query_nodes(self, query)
    }

    fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
        InMemoryGraphStore::neighbors(self, query)
    }

    fn stats(&self) -> GraphStats {
        InMemoryGraphStore::stats(self)
    }
}

impl GraphSnapshotSource for InMemoryGraphStore {
    fn graph_snapshot(&self) -> GraphStoreResult<GraphSnapshot> {
        Ok(InMemoryGraphStore::snapshot(self))
    }
}

impl GraphVerify for InMemoryGraphStore {
    fn verify(&self) -> VerifyReport {
        InMemoryGraphStore::verify(self)
    }

    fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        InMemoryGraphStore::rebuild_indexes(self)
    }
}

impl GraphVectorPayloadAccess for InMemoryGraphStore {
    fn vector_payload_store(&self) -> GraphStoreResult<Arc<dyn TensorBlockPayloadStore>> {
        Ok(self.vector_payload_store_handle())
    }
}

impl GraphTtl for InMemoryGraphStore {
    fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult> {
        InMemoryGraphStore::set_node_ttl(self, id, expires_at_ms)
    }

    fn get_node_including_expired(&self, id: &str) -> Option<&NodeRecord> {
        InMemoryGraphStore::get_node_including_expired(self, id)
    }

    fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize> {
        InMemoryGraphStore::purge_expired_nodes(self)
    }

    fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord> {
        InMemoryGraphStore::nodes_expiring_before(self, ts_ms, limit)
    }

    fn ttl_active_count(&self) -> usize {
        InMemoryGraphStore::ttl_active_count(self)
    }
}

impl GraphResidency for InMemoryGraphStore {
    fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        InMemoryGraphStore::evict_node(self, id)
    }

    fn readmit_node(&mut self, node: NodeRecord) -> GraphStoreResult<()> {
        InMemoryGraphStore::readmit_node(self, node)
    }

    fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        InMemoryGraphStore::evict_edge(self, id)
    }

    fn readmit_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<()> {
        InMemoryGraphStore::readmit_edge(self, edge)
    }
}

const JSON_ATTRIBUTION_MAX_DEPTH: usize = 256;

pub(crate) fn observe_json_allocation(
    value: &Value,
    stats: &mut JsonAllocationStats,
    depth: usize,
) {
    stats.values = stats.values.saturating_add(1);
    if depth >= JSON_ATTRIBUTION_MAX_DEPTH {
        stats.depth_truncated_values = stats.depth_truncated_values.saturating_add(1);
        return;
    }
    match value {
        Value::String(value) => stats.strings.observe(value),
        Value::Array(values) => {
            stats.arrays = stats.arrays.saturating_add(1);
            stats.array_slots_capacity_bytes = stats.array_slots_capacity_bytes.saturating_add(
                values
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
            );
            for value in values {
                observe_json_allocation(value, stats, depth.saturating_add(1));
            }
        }
        Value::Object(values) => {
            stats.objects = stats.objects.saturating_add(1);
            stats.object_entries = stats.object_entries.saturating_add(values.len());
            stats.object_entry_inline_bytes = stats.object_entry_inline_bytes.saturating_add(
                values
                    .len()
                    .saturating_mul(std::mem::size_of::<String>() + std::mem::size_of::<Value>()),
            );
            for (key, value) in values {
                stats.object_keys.observe(key);
                observe_json_allocation(value, stats, depth.saturating_add(1));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

pub(crate) fn string_index_allocation(
    index: &BTreeMap<String, BTreeSet<String>>,
) -> StringIndexAllocationStats {
    let mut stats = StringIndexAllocationStats::default();
    for (key, members) in index {
        stats.keys.observe(key);
        stats.memberships = stats.memberships.saturating_add(members.len());
        for member in members {
            stats.members.observe(member);
        }
    }
    stats
}

pub(crate) fn ordinal_string_index_allocation(
    index: &BTreeMap<String, BTreeSet<EdgeOrdinal>>,
) -> StringIndexAllocationStats {
    let mut stats = StringIndexAllocationStats::default();
    for (key, members) in index {
        stats.keys.observe(key);
        stats.memberships = stats.memberships.saturating_add(members.len());
    }
    stats.compact_member_inline_bytes = stats
        .memberships
        .saturating_mul(std::mem::size_of::<EdgeOrdinal>());
    stats
}

pub(crate) fn tuple_index_allocation(
    index: &BTreeMap<(String, String), BTreeSet<String>>,
) -> TupleIndexAllocationStats {
    let mut stats = TupleIndexAllocationStats::default();
    for ((left, right), members) in index {
        stats.left_keys.observe(left);
        stats.right_keys.observe(right);
        stats.memberships = stats.memberships.saturating_add(members.len());
        for member in members {
            stats.members.observe(member);
        }
    }
    stats
}

pub(crate) fn ordinal_tuple_index_allocation(
    index: &BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
) -> TupleIndexAllocationStats {
    let mut stats = TupleIndexAllocationStats::default();
    for ((left, right), members) in index {
        stats.left_keys.observe(left);
        stats.right_keys.observe(right);
        stats.memberships = stats.memberships.saturating_add(members.len());
    }
    stats.compact_member_inline_bytes = stats
        .memberships
        .saturating_mul(std::mem::size_of::<EdgeOrdinal>());
    stats
}

pub(crate) fn string_index_bytes(index: &BTreeMap<String, BTreeSet<String>>) -> usize {
    index
        .iter()
        .map(|(key, values)| key.len() + values.iter().map(String::len).sum::<usize>())
        .sum()
}

pub(crate) fn ordinal_string_index_bytes(index: &BTreeMap<String, BTreeSet<EdgeOrdinal>>) -> usize {
    index
        .iter()
        .map(|(key, values)| {
            key.len().saturating_add(
                values
                    .len()
                    .saturating_mul(std::mem::size_of::<EdgeOrdinal>()),
            )
        })
        .sum()
}

pub(crate) fn tuple_index_bytes(index: &BTreeMap<(String, String), BTreeSet<String>>) -> usize {
    index
        .iter()
        .map(|((left, right), values)| {
            left.len() + right.len() + values.iter().map(String::len).sum::<usize>()
        })
        .sum()
}

pub(crate) fn ordinal_tuple_index_bytes(
    index: &BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
) -> usize {
    index
        .iter()
        .map(|((left, right), values)| {
            left.len().saturating_add(right.len()).saturating_add(
                values
                    .len()
                    .saturating_mul(std::mem::size_of::<EdgeOrdinal>()),
            )
        })
        .sum()
}

pub(crate) fn ordered_index_bytes(indexes: &BTreeMap<(String, String), OrderedIndex>) -> usize {
    indexes
        .iter()
        .map(|((label, property), index)| {
            label.len() + property.len() + index.estimated_entry_bytes()
        })
        .sum()
}

pub(crate) fn edge_memory_upper_bound(edge: &EdgeRecord) -> usize {
    serialized_json_bytes(edge)
        .saturating_add(1)
        .saturating_add(edge.edge_type.len())
        .saturating_add(edge.id.len())
        .saturating_add(edge.from_id.len())
        .saturating_add(edge.edge_type.len())
        .saturating_add(edge.id.len())
        .saturating_add(edge.to_id.len())
        .saturating_add(edge.edge_type.len())
        .saturating_add(edge.id.len())
}
