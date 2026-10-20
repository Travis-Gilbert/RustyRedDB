#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
use web_time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::mutation::GraphWriteResult;
use crate::graph::record::NodeRecord;
use crate::graph::temporal::property_i64;

pub const TTL_PROPERTY: &str = "_ttl_expires_at_ms";

/// Read TTL expiration timestamp from a node's properties.
/// Returns `None` if no TTL is set OR if value is `<= 0` (treated as no
/// expiration). The absence-of-property and zero-value cases mean
/// "node never expires," matching Redis-like TTL semantics.
pub fn node_ttl_expires_at_ms(node: &NodeRecord) -> Option<i64> {
    property_i64(&node.properties, TTL_PROPERTY).filter(|expires_at| *expires_at > 0)
}

/// True if the node has a TTL AND `now_ms` is past expiration.
/// False if no TTL is set OR `now_ms` is still within the window.
pub fn node_is_expired(node: &NodeRecord, now_ms: i64) -> bool {
    node_ttl_expires_at_ms(node)
        .map(|expires_at| now_ms > expires_at)
        .unwrap_or(false)
}

/// Wall-clock helper for callers that need "now in Unix milliseconds."
/// Returns 0 on the unreachable case where SystemTime is before UNIX_EPOCH.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Return a copy of `node` with `_ttl_expires_at_ms` set to `expires_at_ms`
/// (or cleared if `None`/non-positive). Used by both `InMemoryGraphStore` and
/// `RedCoreGraphStore` so the TTL-property write semantics live in exactly
/// one place. Callers must still push the returned node through `upsert_node`
/// (or `commit_batch`) to get index + journal-root maintenance.
pub(crate) fn node_with_ttl_property(
    mut node: NodeRecord,
    expires_at_ms: Option<i64>,
) -> NodeRecord {
    if let Value::Object(map) = &mut node.properties {
        match expires_at_ms {
            Some(ts) if ts > 0 => {
                map.insert(
                    TTL_PROPERTY.to_string(),
                    Value::Number(serde_json::Number::from(ts)),
                );
            }
            _ => {
                map.remove(TTL_PROPERTY);
            }
        }
    } else {
        // The `properties` field is not an object. The InMemory upsert
        // path normalizes property shape on write, so in practice we never
        // see anything but Object here, but we construct one defensively
        // in case the caller routed a malformed record.
        let mut map = serde_json::Map::new();
        if let Some(ts) = expires_at_ms.filter(|t| *t > 0) {
            map.insert(
                TTL_PROPERTY.to_string(),
                Value::Number(serde_json::Number::from(ts)),
            );
        }
        node.properties = Value::Object(map);
    }
    node
}

impl InMemoryGraphStore {
    // ---- TTL primitive methods (design v2) ----

    /// Set or clear the TTL on an existing node. `Some(t)` extends or sets
    /// the expiration to `t` milliseconds since epoch; `None` clears the
    /// TTL so the node becomes permanent. Returns the new write result.
    /// Errors if the node does not exist.
    pub fn set_node_ttl(
        &mut self,
        id: &str,
        expires_at_ms: Option<i64>,
    ) -> GraphStoreResult<GraphWriteResult> {
        // Use `get` (live nodes only) here rather than the
        // including-expired path: the contract is "extend or clear TTL
        // on an existing live node," and trying to extend a tombstoned
        // node is a real error worth surfacing. Callers wanting forensic
        // access use `get_node_including_expired` directly.
        let existing = self
            .nodes
            .get(id)
            .filter(|node| !node.tombstone)
            .cloned()
            .ok_or_else(|| {
                GraphStoreError::new("missing_graph_node", format!("node {id} does not exist"))
            })?;
        // Build the updated property bag without mutating the original
        // node in place (upsert_node handles index updates via the
        // remove+add cycle).
        let updated = node_with_ttl_property(existing, expires_at_ms);
        self.upsert_node(updated)
    }

    /// Sweep expired nodes from storage. Returns the count purged.
    /// Iterates `ttl_index.range(..=now_ms)` so cost is bounded by the
    /// number of currently-expired nodes, not by total node count.
    /// Called by the background sweep task; can also be invoked manually.
    pub fn purge_expired_nodes(&mut self) -> GraphStoreResult<usize> {
        Ok(self.drain_expired_node_ids().len())
    }

    /// Like `purge_expired_nodes` but returns the ids of the nodes that
    /// were purged. The caller can use this list to journal each removal
    /// (e.g., RedCore writes one `NodeDelete` AOF op per id so the purge
    /// is durable across restarts). Mirrors the sweep semantics exactly:
    /// uses `now_ms()` for the cutoff, scans `ttl_index.range(..=now)`,
    /// drops the node, its incident edges, and every affected index entry.
    pub fn drain_expired_node_ids(&mut self) -> Vec<String> {
        self.drain_expired_node_ids_before(now_ms())
    }

    pub fn has_expired_nodes_before(&self, expires_at_or_before_ms: i64) -> bool {
        self.ttl_index
            .range(..=expires_at_or_before_ms)
            .any(|(_, ids)| !ids.is_empty())
    }

    pub(crate) fn drain_expired_node_ids_before(
        &mut self,
        expires_at_or_before_ms: i64,
    ) -> Vec<String> {
        // Collect ids first to avoid mutable-borrow conflicts on the
        // ttl_index while we mutate the main `nodes` map.
        let expired_ids: Vec<String> = self
            .ttl_index
            .range(..=expires_at_or_before_ms)
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect();
        let mut purged: Vec<String> = Vec::with_capacity(expired_ids.len());
        for id in expired_ids {
            let incident_edges = self
                .edges
                .values()
                .filter(|edge| !edge.tombstone && (edge.from_id == id || edge.to_id == id))
                .cloned()
                .collect::<Vec<_>>();
            for edge in incident_edges {
                self.committed_footprint.remove_edge_record(&edge);
                self.remove_edge_indexes(&edge);
                self.edges.remove(&edge.id);
            }
            // remove_node_indexes evicts from label/property/ttl indexes;
            // then drop the actual node record. Incident edges have already
            // been removed so no live edge can retain a missing endpoint.
            if let Some(node) = self.nodes.get(&id).cloned() {
                self.committed_footprint.remove_node_record(&node);
                self.remove_node_indexes(&node);
                self.nodes.remove(&id);
                purged.push(id);
            }
        }
        purged
    }

    /// Return nodes whose `_ttl_expires_at_ms <= ts_ms`, sorted by
    /// expiration ascending (soonest first). Used by callers that need
    /// "what's about to time out" visibility -- mentions queue,
    /// presence keys, coordinator agents.
    pub fn nodes_expiring_before(&self, ts_ms: i64, limit: usize) -> Vec<NodeRecord> {
        let cap = if limit == 0 { usize::MAX } else { limit };
        let mut out: Vec<NodeRecord> = Vec::new();
        for (_expires_at, ids) in self.ttl_index.range(..=ts_ms) {
            for id in ids {
                if out.len() >= cap {
                    return out;
                }
                if let Some(node) = self.nodes.get(id).filter(|node| !node.tombstone).cloned() {
                    out.push(node);
                }
            }
        }
        out
    }

    /// Current count of TTL-bearing live nodes across the store.
    /// Used by the diagnostics endpoint to surface an active-nodes gauge.
    pub fn ttl_active_count(&self) -> usize {
        self.ttl_index.values().map(|ids| ids.len()).sum()
    }

    pub(crate) fn add_ttl_index(&mut self, node: &NodeRecord) {
        if let Some(expires_at_ms) = node_ttl_expires_at_ms(node) {
            self.ttl_index
                .entry(expires_at_ms)
                .or_default()
                .insert(node.id.clone());
        }
    }

    pub(crate) fn remove_ttl_index(&mut self, node: &NodeRecord) {
        let Some(expires_at_ms) = node_ttl_expires_at_ms(node) else {
            return;
        };
        if let Some(ids) = self.ttl_index.get_mut(&expires_at_ms) {
            ids.remove(&node.id);
            if ids.is_empty() {
                self.ttl_index.remove(&expires_at_ms);
            }
        }
    }
}
