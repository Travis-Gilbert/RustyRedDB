use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::footprint::{index_insert, index_remove, ordered_member_bytes};
use crate::graph::in_memory::{EdgeOrdinal, InMemoryGraphStore};
use crate::graph::record::{EdgeRecord, NodeRecord};

impl InMemoryGraphStore {
    pub(crate) fn require_live_endpoint(
        &self,
        edge: &EdgeRecord,
        endpoint: &str,
        node_id: &str,
    ) -> GraphStoreResult<()> {
        let Some(node) = self.nodes.get(node_id) else {
            return Err(GraphStoreError::missing_endpoint(
                &edge.id, endpoint, node_id,
            ));
        };
        if node.tombstone {
            return Err(GraphStoreError::tombstoned_endpoint(
                &edge.id, endpoint, node_id,
            ));
        }
        Ok(())
    }

    pub(crate) fn add_node_indexes(&mut self, node: &NodeRecord) {
        for label in &node.labels {
            index_insert(
                &mut self.label_index,
                &mut self.committed_footprint.label_index_bytes,
                label.clone(),
                node.id.clone(),
            );
        }
        for (key, token) in indexed_properties(&node.properties) {
            index_insert(
                &mut self.property_index,
                &mut self.committed_footprint.property_index_bytes,
                (key, token),
                node.id.clone(),
            );
        }
        self.add_ttl_index(node);
        self.auto_index_ordered(node).ok();
    }

    pub(crate) fn remove_node_indexes(&mut self, node: &NodeRecord) {
        for label in &node.labels {
            index_remove(
                &mut self.label_index,
                &mut self.committed_footprint.label_index_bytes,
                label,
                &node.id,
            );
        }
        for (key, token) in indexed_properties(&node.properties) {
            index_remove(
                &mut self.property_index,
                &mut self.committed_footprint.property_index_bytes,
                &(key, token),
                &node.id,
            );
        }
        self.remove_ttl_index(node);
        let mut released_entry_bytes = 0usize;
        for index in self.ordered_indexes.values_mut() {
            if index.zrem(node.id.as_bytes()) {
                released_entry_bytes =
                    released_entry_bytes.saturating_add(ordered_member_bytes(node.id.len()));
            }
        }
        self.committed_footprint
            .remove_ordered_entry_bytes(released_entry_bytes);
        for index in self.vector_indexes.values_mut() {
            index.remove(&node.id);
        }
    }

    pub(crate) fn add_edge_indexes(&mut self, edge: &EdgeRecord, ordinal: EdgeOrdinal) {
        index_insert(
            &mut self.edge_type_index,
            &mut self.committed_footprint.edge_type_index_bytes,
            edge.edge_type.clone(),
            ordinal,
        );
        index_insert(
            &mut self.out_adjacency,
            &mut self.committed_footprint.out_adjacency_bytes,
            (edge.from_id.clone(), edge.edge_type.clone()),
            ordinal,
        );
        index_insert(
            &mut self.in_adjacency,
            &mut self.committed_footprint.in_adjacency_bytes,
            (edge.to_id.clone(), edge.edge_type.clone()),
            ordinal,
        );
    }

    pub(crate) fn remove_edge_indexes(&mut self, edge: &EdgeRecord) {
        let Some(ordinal) = self.edges.ordinal(&edge.id) else {
            return;
        };
        index_remove(
            &mut self.edge_type_index,
            &mut self.committed_footprint.edge_type_index_bytes,
            &edge.edge_type,
            &ordinal,
        );
        index_remove(
            &mut self.out_adjacency,
            &mut self.committed_footprint.out_adjacency_bytes,
            &(edge.from_id.clone(), edge.edge_type.clone()),
            &ordinal,
        );
        index_remove(
            &mut self.in_adjacency,
            &mut self.committed_footprint.in_adjacency_bytes,
            &(edge.to_id.clone(), edge.edge_type.clone()),
            &ordinal,
        );
    }
}

pub(crate) fn validate_edge_shape(edge: &EdgeRecord) -> GraphStoreResult<()> {
    if edge.id.trim().is_empty() {
        return Err(GraphStoreError::empty_field("edge.id"));
    }
    if edge.from_id.trim().is_empty() {
        return Err(GraphStoreError::empty_field("edge.from_id"));
    }
    if edge.to_id.trim().is_empty() {
        return Err(GraphStoreError::empty_field("edge.to_id"));
    }
    if edge.edge_type.trim().is_empty() {
        return Err(GraphStoreError::empty_field("edge.type"));
    }
    if edge
        .confidence
        .is_some_and(|confidence| !confidence.is_finite())
    {
        return Err(GraphStoreError::new(
            "edge_confidence_non_finite",
            format!("edge {} confidence must be finite", edge.id),
        ));
    }
    Ok(())
}

pub(crate) fn numeric_property(properties: &Value, key: &str) -> Option<f64> {
    properties
        .get(key)?
        .as_f64()
        .filter(|score| !score.is_nan())
}

pub(crate) fn sorted_values(values: Option<&BTreeSet<String>>) -> Vec<String> {
    values
        .map(|values| values.iter().cloned().collect())
        .unwrap_or_default()
}

pub(crate) fn merge_candidates(
    candidates: &mut Option<BTreeSet<String>>,
    next: Option<BTreeSet<String>>,
) {
    let next = next.unwrap_or_default();
    match candidates {
        Some(existing) => {
            *existing = existing.intersection(&next).cloned().collect();
        }
        None => *candidates = Some(next),
    }
}

pub(crate) fn indexed_properties(properties: &Value) -> BTreeMap<String, String> {
    let Some(properties) = properties.as_object() else {
        return BTreeMap::new();
    };
    properties
        .iter()
        .filter_map(|(key, value)| {
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            property_index_token(value).map(|token| (key.to_string(), token))
        })
        .collect()
}

pub(crate) fn property_index_token(value: &Value) -> Option<String> {
    match value {
        Value::String(raw) => Some(string_property_index_token(raw)),
        // Retain the historical canonical JSON spelling for non-string
        // scalars. The string type marker prevents `"true"` from colliding
        // with boolean `true` while leaving existing numeric/bool/null index
        // keys deterministic and backward-compatible.
        Value::Null | Value::Bool(_) | Value::Number(_) => serde_json::to_string(value).ok(),
        Value::Array(_) | Value::Object(_) => None,
    }
}

pub(crate) fn string_property_index_token(raw: &str) -> String {
    let mut token = String::with_capacity(2usize.saturating_add(raw.len()));
    token.push_str("s:");
    token.push_str(raw);
    token
}
