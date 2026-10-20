use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphStats {
    pub version: u64,
    pub nodes_total: usize,
    pub edges_total: usize,
    pub labels_total: usize,
    pub edge_types_total: usize,
    pub property_keys_total: usize,
    pub property_indexes_total: usize,
    /// Estimated serialized footprint for live graph records and derived indexes.
    pub memory_bytes: usize,
    pub memory_quota_bytes: usize,
}

/// Allocation-oriented string counters used by the memory attribution
/// diagnostic. `length_bytes` is logical content; `capacity_bytes` is the
/// owned heap reservation. `inline_bytes` accounts for the `String` handles
/// stored in records or index containers, not allocator metadata.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StringAllocationStats {
    pub values: usize,
    pub length_bytes: usize,
    pub capacity_bytes: usize,
    pub inline_bytes: usize,
    pub max_length_bytes: usize,
    pub max_capacity_bytes: usize,
}

impl StringAllocationStats {
    pub(crate) fn observe(&mut self, value: &String) {
        self.values = self.values.saturating_add(1);
        self.length_bytes = self.length_bytes.saturating_add(value.len());
        self.capacity_bytes = self.capacity_bytes.saturating_add(value.capacity());
        self.inline_bytes = self
            .inline_bytes
            .saturating_add(std::mem::size_of::<String>());
        self.max_length_bytes = self.max_length_bytes.max(value.len());
        self.max_capacity_bytes = self.max_capacity_bytes.max(value.capacity());
    }
}

/// Heap-bearing portions of `serde_json::Value` trees. This deliberately
/// avoids materializing JSON text while still exposing array capacity, object
/// entry count, and owned string capacity.
#[derive(Clone, Debug, Default, Serialize)]
pub struct JsonAllocationStats {
    pub values: usize,
    pub arrays: usize,
    pub array_slots_capacity_bytes: usize,
    pub objects: usize,
    pub object_entries: usize,
    pub object_entry_inline_bytes: usize,
    pub strings: StringAllocationStats,
    pub object_keys: StringAllocationStats,
    pub depth_truncated_values: usize,
}

impl JsonAllocationStats {
    pub(crate) fn accounted_capacity_bytes(&self) -> usize {
        self.array_slots_capacity_bytes
            .saturating_add(self.object_entry_inline_bytes)
            .saturating_add(self.strings.capacity_bytes)
            .saturating_add(self.object_keys.capacity_bytes)
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StringIndexAllocationStats {
    pub keys: StringAllocationStats,
    pub members: StringAllocationStats,
    pub memberships: usize,
    /// Inline bytes for compact, non-string members such as edge ordinals.
    pub compact_member_inline_bytes: usize,
}

impl StringIndexAllocationStats {
    pub(crate) fn accounted_capacity_bytes(&self) -> usize {
        self.keys
            .inline_bytes
            .saturating_add(self.keys.capacity_bytes)
            .saturating_add(self.members.inline_bytes)
            .saturating_add(self.members.capacity_bytes)
            .saturating_add(self.compact_member_inline_bytes)
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TupleIndexAllocationStats {
    pub left_keys: StringAllocationStats,
    pub right_keys: StringAllocationStats,
    pub members: StringAllocationStats,
    pub memberships: usize,
    /// Inline bytes for compact, non-string members such as edge ordinals.
    pub compact_member_inline_bytes: usize,
}

impl TupleIndexAllocationStats {
    pub(crate) fn accounted_capacity_bytes(&self) -> usize {
        self.left_keys
            .inline_bytes
            .saturating_add(self.left_keys.capacity_bytes)
            .saturating_add(self.right_keys.inline_bytes)
            .saturating_add(self.right_keys.capacity_bytes)
            .saturating_add(self.members.inline_bytes)
            .saturating_add(self.members.capacity_bytes)
            .saturating_add(self.compact_member_inline_bytes)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorIndexAllocationStats {
    pub label: String,
    pub property: String,
    pub dimension: usize,
    pub point_count: usize,
    pub point_slots_capacity_bytes: usize,
    pub point_values_capacity_bytes: usize,
    pub node_id_slots_capacity_bytes: usize,
    pub node_ids: StringAllocationStats,
    pub slot_map_capacity: usize,
    pub slot_map_entry_inline_bytes: usize,
    pub slot_map_keys: StringAllocationStats,
    pub accelerated_index_present: bool,
    pub accelerated_index_bytes_accounted: bool,
}

impl VectorIndexAllocationStats {
    pub(crate) fn accounted_capacity_bytes(&self) -> usize {
        self.point_slots_capacity_bytes
            .saturating_add(self.point_values_capacity_bytes)
            .saturating_add(self.node_id_slots_capacity_bytes)
            .saturating_add(self.node_ids.capacity_bytes)
            .saturating_add(self.slot_map_entry_inline_bytes)
            .saturating_add(self.slot_map_keys.capacity_bytes)
    }
}

/// Borrowed, allocation-safe memory attribution for one materialized graph.
///
/// `accounted_capacity_lower_bound_bytes` intentionally remains a lower bound:
/// Rust does not expose `BTreeMap` node allocations or the accelerated vector
/// backend's private heap. The exclusions list keeps those blind spots explicit
/// so this report is not mistaken for allocator truth.
#[derive(Clone, Debug, Serialize)]
pub struct GraphMemoryAttribution {
    pub version: u64,
    pub node_records: usize,
    pub live_nodes: usize,
    pub tombstoned_nodes: usize,
    pub edge_records: usize,
    pub live_edges: usize,
    pub tombstoned_edges: usize,
    pub record_inline_bytes: usize,
    pub record_vec_slots_capacity_bytes: usize,
    pub primary_node_keys: StringAllocationStats,
    pub primary_edge_keys: StringAllocationStats,
    pub node_record_strings: StringAllocationStats,
    pub edge_record_strings: StringAllocationStats,
    pub node_properties: JsonAllocationStats,
    pub edge_properties: JsonAllocationStats,
    pub historical_node_ids: StringAllocationStats,
    pub label_index: StringIndexAllocationStats,
    pub edge_type_index: StringIndexAllocationStats,
    pub out_adjacency: TupleIndexAllocationStats,
    pub in_adjacency: TupleIndexAllocationStats,
    pub property_index: TupleIndexAllocationStats,
    pub vector_indexes: Vec<VectorIndexAllocationStats>,
    pub serialized_graph_bytes: usize,
    pub accounted_capacity_lower_bound_bytes: usize,
    pub exclusions: Vec<&'static str>,
}
