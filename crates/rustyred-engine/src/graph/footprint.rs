//! Incrementally maintained committed-footprint accounting.
//!
//! `InMemoryGraphStore::estimated_memory_bytes` is the base term of
//! `projected_memory_bytes_upper_bound`, which the RedCore publish path runs at
//! least once per commit batch (and once more per explicit write entry point),
//! so before this module existed every single record write serde-serialized the
//! whole tenant graph into a byte-counting writer.
//!
//! This module holds the decomposed counters that serve the same value in O(1).
//! Each term of the total has exactly one counter:
//!
//! ```text
//! serialized snapshot  framing (literal JSON framing, version digits, bool)
//!                    + Σ serialized_json_bytes(node) over `nodes`
//!                    + Σ serialized_json_bytes(edge) over the edge arena
//!                    + Σ serialized_json_bytes(id) over `historical_node_ids`
//!                    + array separators (one comma per non-final element)
//! + label_index bytes            (key bytes + member string bytes)
//! + edge_type_index bytes        (key bytes + members * size_of::<EdgeOrdinal>())
//! + out_adjacency bytes          (tuple key bytes + members * size_of::<EdgeOrdinal>())
//! + in_adjacency bytes           (tuple key bytes + members * size_of::<EdgeOrdinal>())
//! + property_index bytes         (tuple key bytes + member string bytes)
//! + ordered_index bytes          (designation key bytes + Σ member + size_of::<f64>())
//! + Σ (id.len() + size_of::<String>()) over `cold_nonresident_node_ids`
//! + edge arena record-slot capacity bytes   (Vec::capacity() read, stays computed)
//! + edge arena free-slot capacity bytes     (Vec::capacity() read, stays computed)
//! ```
//!
//! The two capacity terms are genuinely O(1) reads of `Vec::capacity()`, so they
//! stay computed at lookup time. Every other term is O(container size) in the
//! container it is handed, so it is maintained here with an explicit delta rule
//! at each mutation site. `InMemoryGraphStore::recompute_memory_bytes` keeps the
//! original whole-graph implementation as the oracle: constructors,
//! recovery/reopen, index rebuild, and tests compare against it.
//!
//! Exactness matters: the quota path compares this value against a fixed quota,
//! so a counter that drifts upward would reject writes the oracle admits, and a
//! counter that drifts downward would over-admit. The delta rules below mirror
//! each term's definition byte for byte; the parity tests in
//! `crate::graph::tests` assert `estimated_memory_bytes() ==
//! recompute_memory_bytes()` after every step of a mixed operation sequence.
//!
//! The counters are exact for every state reachable through the store's own
//! mutation API. A caller that mutates `nodes`/`label_index`/... directly (the
//! deliberately-corrupting fixtures in `graph::tests`, which `verify()` is
//! designed to detect) bypasses them; `rebuild_indexes_from_records` and
//! `from_snapshot` re-derive them from the oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::in_memory::{
    ordinal_string_index_bytes, ordinal_tuple_index_bytes, string_index_bytes, tuple_index_bytes,
    EdgeOrdinal, InMemoryGraphStore,
};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::snapshot::{serialized_json_bytes, snapshot_framing_bytes};

/// Process-wide count of whole-graph recomputes.
///
/// Exists so tests can prove the ordinary write path never traverses the graph
/// for memory accounting: take a reading, perform a bounded write against a
/// large store, and assert the counter did not move. Constructors, recovery,
/// index rebuild, and explicit test use of the oracle move it by design.
static GRAPH_MEMORY_RECOMPUTE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);

/// How many times the whole-graph memory recompute oracle
/// (`InMemoryGraphStore::recompute_memory_bytes`) has run in this process.
pub fn graph_memory_recompute_invocations() -> u64 {
    GRAPH_MEMORY_RECOMPUTE_INVOCATIONS.load(Ordering::Relaxed)
}

/// Reset [`graph_memory_recompute_invocations`] to zero.
pub fn reset_graph_memory_recompute_invocations() {
    GRAPH_MEMORY_RECOMPUTE_INVOCATIONS.store(0, Ordering::Relaxed);
}

pub(crate) fn note_graph_memory_recompute() {
    GRAPH_MEMORY_RECOMPUTE_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Byte weight of one index key, matching the index-byte terms above.
pub(crate) trait IndexKeyBytes {
    fn index_key_bytes(&self) -> usize;
}

impl IndexKeyBytes for String {
    fn index_key_bytes(&self) -> usize {
        self.len()
    }
}

impl IndexKeyBytes for (String, String) {
    fn index_key_bytes(&self) -> usize {
        self.0.len().saturating_add(self.1.len())
    }
}

/// Byte weight of one index member, matching the index-byte terms above.
pub(crate) trait IndexMemberBytes {
    fn index_member_bytes(&self) -> usize;
}

impl IndexMemberBytes for String {
    fn index_member_bytes(&self) -> usize {
        self.len()
    }
}

impl IndexMemberBytes for EdgeOrdinal {
    fn index_member_bytes(&self) -> usize {
        std::mem::size_of::<EdgeOrdinal>()
    }
}

/// Insert `value` under `key`, keeping `counter` equal to the index term it
/// mirrors.
///
/// Delta rule: a key that is not yet present adds `key.index_key_bytes()`; a
/// member that is not yet present adds `value.index_member_bytes()`. Updating an
/// existing member is byte-neutral, because the measured terms count key and
/// member bytes, not the score or ordering state.
pub(crate) fn index_insert<K, V>(
    index: &mut BTreeMap<K, BTreeSet<V>>,
    counter: &mut usize,
    key: K,
    value: V,
) where
    K: Ord + IndexKeyBytes,
    V: Ord + IndexMemberBytes,
{
    let key_bytes = key.index_key_bytes();
    let member_bytes = value.index_member_bytes();
    if !index.contains_key(&key) {
        *counter = counter.saturating_add(key_bytes);
    }
    if index.entry(key).or_insert_with(BTreeSet::new).insert(value) {
        *counter = counter.saturating_add(member_bytes);
    }
}

/// Remove `value` from `key`, keeping `counter` equal to the index term it
/// mirrors. Matches the historical `remove_index_value`: a key whose member set
/// becomes empty is dropped from the map, and its key bytes with it.
///
/// Delta rule: an absent member or key removes nothing.
pub(crate) fn index_remove<K, V>(
    index: &mut BTreeMap<K, BTreeSet<V>>,
    counter: &mut usize,
    key: &K,
    value: &V,
) where
    K: Ord + IndexKeyBytes,
    V: Ord + IndexMemberBytes,
{
    let mut key_became_empty = false;
    if let Some(values) = index.get_mut(key) {
        if values.remove(value) {
            *counter = counter.saturating_sub(value.index_member_bytes());
        }
        key_became_empty = values.is_empty();
    }
    if key_became_empty {
        *counter = counter.saturating_sub(key.index_key_bytes());
        index.remove(key);
    }
}

/// Bytes of one ordered-index member: the member itself plus its score.
pub(crate) fn ordered_member_bytes(member_len: usize) -> usize {
    member_len.saturating_add(std::mem::size_of::<f64>())
}

/// One comma separates each pair of adjacent elements of a JSON array, so an
/// array of `count` elements contributes `count - 1` bytes (zero when the array
/// is empty or holds a single element).
fn separator_bytes(count: usize) -> usize {
    count.saturating_sub(1)
}

/// One maintained counter per term of the committed-footprint decomposition.
///
/// `Default` is the footprint of an empty store, which is why `Default` and
/// `InMemoryGraphStore::new` need no recompute; every constructor that installs
/// records (`from_snapshot`) or rebuilds derived state
/// (`rebuild_indexes_from_records`) re-derives the counters through
/// [`CommittedFootprint::from_store`] instead of being left stale.
#[derive(Clone, Debug, Default)]
pub(crate) struct CommittedFootprint {
    /// Σ `serialized_json_bytes(node)` over every entry of `nodes`, tombstones
    /// included, because the measured snapshot serializes tombstones too.
    pub(crate) node_records_json_bytes: usize,
    /// Σ `serialized_json_bytes(edge)` over every arena record holding an
    /// ordinal, tombstones included.
    pub(crate) edge_records_json_bytes: usize,
    /// Σ `serialized_json_bytes(id)` over `historical_node_ids`.
    pub(crate) historical_ids_json_bytes: usize,
    pub(crate) label_index_bytes: usize,
    pub(crate) edge_type_index_bytes: usize,
    pub(crate) out_adjacency_bytes: usize,
    pub(crate) in_adjacency_bytes: usize,
    pub(crate) property_index_bytes: usize,
    /// Σ `(label.len() + property.len())` over the `ordered_indexes` keys.
    pub(crate) ordered_designation_bytes: usize,
    /// Σ `OrderedIndex::estimated_entry_bytes()` over the ordered indexes, i.e.
    /// `(member.len() + size_of::<f64>())` per entry.
    pub(crate) ordered_entry_bytes: usize,
    /// Σ `(id.len() + size_of::<String>())` over `cold_nonresident_node_ids`.
    pub(crate) cold_nonresident_node_ids_bytes: usize,
}

impl CommittedFootprint {
    /// O(graph) rebuild of every counter from the containers themselves.
    ///
    /// Uses the same helper functions as the recompute oracle, so the two can
    /// only agree or disagree on the framing/separator identity, which the
    /// accounting tests assert directly.
    pub(crate) fn from_store(store: &InMemoryGraphStore) -> Self {
        Self {
            node_records_json_bytes: store
                .nodes
                .values()
                .map(serialized_json_bytes)
                .fold(0usize, usize::saturating_add),
            edge_records_json_bytes: store
                .edges
                .values()
                .map(serialized_json_bytes)
                .fold(0usize, usize::saturating_add),
            historical_ids_json_bytes: store
                .historical_node_ids
                .iter()
                .map(|id| serialized_json_bytes(id))
                .fold(0usize, usize::saturating_add),
            label_index_bytes: string_index_bytes(&store.label_index),
            edge_type_index_bytes: ordinal_string_index_bytes(&store.edge_type_index),
            out_adjacency_bytes: ordinal_tuple_index_bytes(&store.out_adjacency),
            in_adjacency_bytes: ordinal_tuple_index_bytes(&store.in_adjacency),
            property_index_bytes: tuple_index_bytes(&store.property_index),
            ordered_designation_bytes: store
                .ordered_indexes
                .keys()
                .map(|(label, property)| label.len().saturating_add(property.len()))
                .fold(0usize, usize::saturating_add),
            ordered_entry_bytes: store
                .ordered_indexes
                .values()
                .map(|index| index.estimated_entry_bytes())
                .fold(0usize, usize::saturating_add),
            cold_nonresident_node_ids_bytes: store
                .cold_nonresident_node_ids
                .iter()
                .map(|id| id.len().saturating_add(std::mem::size_of::<String>()))
                .fold(0usize, usize::saturating_add),
        }
    }

    /// The serialized-snapshot term alone: literal framing, the per-record JSON
    /// sums, and the array separators.
    pub(crate) fn snapshot_bytes(
        &self,
        version: u64,
        historical_node_history_complete: bool,
        node_count: usize,
        edge_count: usize,
        historical_id_count: usize,
    ) -> usize {
        snapshot_framing_bytes(version, historical_node_history_complete)
            .saturating_add(self.node_records_json_bytes)
            .saturating_add(self.edge_records_json_bytes)
            .saturating_add(self.historical_ids_json_bytes)
            .saturating_add(separator_bytes(node_count))
            .saturating_add(separator_bytes(edge_count))
            .saturating_add(separator_bytes(historical_id_count))
    }

    /// The derived-index terms alone: the four string/ordinal secondary indexes,
    /// the property index, and the ordered indexes.
    pub(crate) fn derived_index_bytes(&self) -> usize {
        self.label_index_bytes
            .saturating_add(self.edge_type_index_bytes)
            .saturating_add(self.out_adjacency_bytes)
            .saturating_add(self.in_adjacency_bytes)
            .saturating_add(self.property_index_bytes)
            .saturating_add(self.ordered_designation_bytes)
            .saturating_add(self.ordered_entry_bytes)
    }

    /// Delta rule: an inserted node adds its serialized JSON length to the
    /// retained record bytes. (The ordered-index delta rule lives with the
    /// ordered-index counters; this comment previously described that one.)
    pub(crate) fn add_node_record(&mut self, node: &NodeRecord) {
        self.node_records_json_bytes = self
            .node_records_json_bytes
            .saturating_add(serialized_json_bytes(node));
    }

    pub(crate) fn remove_node_record(&mut self, node: &NodeRecord) {
        self.node_records_json_bytes = self
            .node_records_json_bytes
            .saturating_sub(serialized_json_bytes(node));
    }

    pub(crate) fn add_edge_record(&mut self, edge: &EdgeRecord) {
        self.edge_records_json_bytes = self
            .edge_records_json_bytes
            .saturating_add(serialized_json_bytes(edge));
    }

    pub(crate) fn remove_edge_record(&mut self, edge: &EdgeRecord) {
        self.edge_records_json_bytes = self
            .edge_records_json_bytes
            .saturating_sub(serialized_json_bytes(edge));
    }

    pub(crate) fn add_historical_id(&mut self, id: &str) {
        self.historical_ids_json_bytes = self
            .historical_ids_json_bytes
            .saturating_add(serialized_json_bytes(&id));
    }

    pub(crate) fn add_cold_nonresident_id(&mut self, id: &str) {
        self.cold_nonresident_node_ids_bytes = self
            .cold_nonresident_node_ids_bytes
            .saturating_add(id.len().saturating_add(std::mem::size_of::<String>()));
    }

    pub(crate) fn remove_cold_nonresident_id(&mut self, id: &str) {
        self.cold_nonresident_node_ids_bytes = self
            .cold_nonresident_node_ids_bytes
            .saturating_sub(id.len().saturating_add(std::mem::size_of::<String>()));
    }

    pub(crate) fn add_ordered_designation(&mut self, label: &str, property: &str) {
        self.ordered_designation_bytes = self
            .ordered_designation_bytes
            .saturating_add(label.len().saturating_add(property.len()));
    }

    /// Drop one ordered designation. `entry_bytes` is the removed index's own
    /// `estimated_entry_bytes()`, measured once by the caller.
    ///
    /// Ordered-entry bytes are maintained as a single sum rather than per
    /// designation, so removing a designation measures that one index instead of
    /// guessing its share. Clearing a designation is an administrative
    /// operation that never runs on the record-write path (individual member
    /// `zadd`/`zrem` keep the sum incrementally), and it is the only accounting
    /// site that walks a container.
    pub(crate) fn remove_ordered_designation(
        &mut self,
        label: &str,
        property: &str,
        entry_bytes: usize,
    ) {
        self.ordered_designation_bytes = self
            .ordered_designation_bytes
            .saturating_sub(label.len().saturating_add(property.len()));
        self.ordered_entry_bytes = self.ordered_entry_bytes.saturating_sub(entry_bytes);
    }

    /// Delta rule: an inserted ordered member adds one member plus one score,
    /// measured by the caller with [`ordered_member_bytes`]. Updating an
    /// existing member keeps the same member bytes, so the caller adds zero.
    pub(crate) fn add_ordered_entry_bytes(&mut self, bytes: usize) {
        self.ordered_entry_bytes = self.ordered_entry_bytes.saturating_add(bytes);
    }

    pub(crate) fn remove_ordered_entry_bytes(&mut self, bytes: usize) {
        self.ordered_entry_bytes = self.ordered_entry_bytes.saturating_sub(bytes);
    }

    /// Zero the derived-index and ordered terms where the containers themselves
    /// are cleared wholesale; the re-population that follows re-accumulates.
    pub(crate) fn forget_derived_index_bytes(&mut self) {
        self.label_index_bytes = 0;
        self.edge_type_index_bytes = 0;
        self.out_adjacency_bytes = 0;
        self.in_adjacency_bytes = 0;
        self.property_index_bytes = 0;
        self.ordered_designation_bytes = 0;
        self.ordered_entry_bytes = 0;
    }
}
