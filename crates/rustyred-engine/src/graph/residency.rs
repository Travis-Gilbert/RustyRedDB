use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::index::validate_edge_shape;
use crate::graph::record::{normalize_labels, EdgeRecord, NodeRecord};

pub(crate) const COLD_SNAPSHOT_STUB_LABEL: &str = "__RustyRedColdSnapshotStub";
pub(crate) const COLD_SNAPSHOT_STUB_PROPERTY: &str = "_rustyred_cold_snapshot_stub";
pub(crate) const COLD_ROLLBACK_RECORD_PROPERTY: &str = "_rustyred_cold_rollback_record";

impl InMemoryGraphStore {
    /// Apply a `NodeDelete` mutation during AOF recovery. Idempotent: if
    /// the node is already absent (e.g., the snapshot loaded after it
    /// was purged) this is a no-op rather than an error. Removes the
    /// node from `self.nodes` and every secondary index it appears in while its
    /// id remains in the separate history registry.
    ///
    /// Does NOT advance `self.version`. The AOF replay loop advances the
    /// version to `max(current, frame.graph_version)` after this call, which
    /// is sufficient for explicit deletes (whose frame carries `version + 1`)
    /// and correct for TTL purges (whose frame carries the pre-purge version
    /// because `drain_expired_node_ids_before` never bumps the version).
    pub fn apply_recovered_delete(&mut self, id: &str) -> GraphStoreResult<()> {
        self.record_historical_node_id(id);
        if let Some(node) = self.nodes.get(id).cloned() {
            self.committed_footprint.remove_node_record(&node);
            self.remove_node_indexes(&node);
            self.nodes.remove(id);
        }
        if self.cold_nonresident_node_ids.remove(id) {
            self.committed_footprint.remove_cold_nonresident_id(id);
        }
        Ok(())
    }

    /// Remove a live node from the operating store and return it, leaving the
    /// graph version unchanged. A residency change, not a logical mutation (see
    /// the `crate::GraphResidency::evict_node` contract): the node's durable home is the
    /// cold tier. Drops the node and every secondary index entry exactly like
    /// `apply_recovered_delete`, but NEVER touches `self.version`, so the cached
    /// structural priors of the remaining live nodes (PPR cache keyed on the
    /// store version) stay valid. Returns `None` if the node is absent.
    pub fn evict_node(&mut self, id: &str) -> GraphStoreResult<Option<NodeRecord>> {
        let Some(node) = self.nodes.get(id).cloned() else {
            return Ok(None);
        };
        self.committed_footprint.remove_node_record(&node);
        self.remove_node_indexes(&node);
        self.nodes.remove(id);
        if self.cold_nonresident_node_ids.insert(id.to_string()) {
            self.committed_footprint.add_cold_nonresident_id(id);
        }
        Ok(Some(node))
    }

    /// Re-admit a node rehydrated from the cold tier WITHOUT changing the graph
    /// version. Unlike `apply_recovered_node` (which raises the version to the
    /// node's stored version), this leaves `self.version` completely untouched:
    /// rehydration is a pure residency change and must not invalidate the PPR
    /// cache keyed on the store version. The node is inserted verbatim -- it
    /// already carries its `content_hash` and `version` from when it was
    /// committed to the cold tier.
    pub fn readmit_node(&mut self, mut node: NodeRecord) -> GraphStoreResult<()> {
        if node.id.trim().is_empty() {
            return Err(GraphStoreError::empty_field("node.id"));
        }
        if self
            .nodes
            .get(&node.id)
            .is_some_and(|existing| existing.tombstone && existing.version >= node.version)
        {
            return Ok(());
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
            self.auto_index_vectors(&node)?;
        }
        self.committed_footprint.add_node_record(&node);
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub(crate) fn apply_logical_node_delete(&mut self, id: &str) -> GraphStoreResult<bool> {
        let Some(mut node) = self.nodes.get(id).cloned() else {
            return Ok(false);
        };
        if node.tombstone {
            return Ok(false);
        }
        // Subtract the live record's bytes before the tombstone rewrite, then
        // add the rewritten record's bytes at the insert below.
        self.committed_footprint.remove_node_record(&node);
        self.remove_node_indexes(&node);
        self.version = self.version.saturating_add(1);
        node.version = self.version;
        node.tombstone = true;
        node.content_hash = None;
        node.content_hash = Some(node.checksum()?);
        if self.cold_nonresident_node_ids.remove(id) {
            self.committed_footprint.remove_cold_nonresident_id(id);
        }
        self.record_historical_node_id(id);
        self.committed_footprint.add_node_record(&node);
        self.nodes.insert(id.to_string(), node);
        Ok(true)
    }

    /// Edge counterpart of `evict_node`: remove an edge and its indexes from the
    /// operating store, version unchanged. Returns `None` if absent. Used by
    /// warm-tier scope parking to evict a subgraph's incident edges.
    pub fn evict_edge(&mut self, id: &str) -> GraphStoreResult<Option<EdgeRecord>> {
        let Some(edge) = self.edges.get(id).cloned() else {
            return Ok(None);
        };
        self.committed_footprint.remove_edge_record(&edge);
        self.remove_edge_indexes(&edge);
        self.edges.remove(id);
        Ok(Some(edge))
    }

    /// Edge counterpart of `readmit_node`, version-neutral. Mirrors
    /// `apply_recovered_edge` (live-endpoint guard included, since warm-tier
    /// unpark re-admits nodes before edges) but never bumps `self.version`.
    pub fn readmit_edge(&mut self, mut edge: EdgeRecord) -> GraphStoreResult<()> {
        if self
            .edges
            .get(&edge.id)
            .is_some_and(|existing| existing.tombstone && existing.version >= edge.version)
        {
            return Ok(());
        }
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
        self.committed_footprint.add_edge_record(&edge);
        let (ordinal, _) = self.edges.insert(edge.id.clone(), edge.clone());
        if !edge.tombstone {
            self.add_edge_indexes(&edge, ordinal);
        }
        Ok(())
    }

    pub(crate) fn apply_logical_edge_delete(&mut self, id: &str) -> GraphStoreResult<bool> {
        let Some(mut edge) = self.edges.get(id).cloned() else {
            return Ok(false);
        };
        if edge.tombstone {
            return Ok(false);
        }
        // Subtract the live record's bytes before the tombstone rewrite, then
        // add the rewritten record's bytes at the insert below.
        self.committed_footprint.remove_edge_record(&edge);
        self.remove_edge_indexes(&edge);
        self.version += 1;
        edge.version = self.version;
        edge.tombstone = true;
        edge.content_hash = Some(edge.checksum()?);
        self.committed_footprint.add_edge_record(&edge);
        self.edges.insert(id.to_string(), edge);
        Ok(true)
    }
}
