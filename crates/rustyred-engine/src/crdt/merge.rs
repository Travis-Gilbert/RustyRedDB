use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::crdt::clock::{ActorId, Hlc};
use crate::graph::capability::{GraphSnapshotSource, GraphStore};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::mutation::GraphMutation;
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::snapshot::GraphSnapshot;
use crate::versioned_graph::resolve_auto_confidence_edge;

const CRDT_HLC_PROPERTY: &str = "_crdt_hlc";

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct VersionVector(pub BTreeMap<ActorId, Hlc>);

impl VersionVector {
    pub fn observe(&mut self, hlc: Hlc) {
        self.0
            .entry(hlc.actor)
            .and_modify(|existing| {
                if hlc > *existing {
                    *existing = hlc;
                }
            })
            .or_insert(hlc);
    }

    pub fn high_water(&self, actor: ActorId) -> Option<Hlc> {
        self.0.get(&actor).copied()
    }

    pub fn is_missing(&self, hlc: Hlc) -> bool {
        self.high_water(hlc.actor)
            .map(|seen| hlc > seen)
            .unwrap_or(true)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StampedMutation {
    pub mutation: GraphMutation,
    pub hlc: Hlc,
}

impl StampedMutation {
    pub fn new(mutation: GraphMutation, hlc: Hlc) -> Self {
        Self { mutation, hlc }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StampedBatch {
    #[serde(default)]
    pub mutations: Vec<StampedMutation>,
    #[serde(default)]
    pub frontier: VersionVector,
}

impl StampedBatch {
    pub fn new(mutations: impl IntoIterator<Item = StampedMutation>) -> Self {
        let mutations = mutations.into_iter().collect::<Vec<_>>();
        let mut frontier = VersionVector::default();
        for stamped in &mutations {
            frontier.observe(stamped.hlc);
        }
        Self {
            mutations,
            frontier,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct JoinReport {
    pub applied: usize,
    pub ignored_stale: usize,
    pub revived: usize,
    pub tombstoned: usize,
}

pub fn join_delta<S: GraphStore>(
    store: &mut S,
    delta: StampedBatch,
) -> GraphStoreResult<JoinReport> {
    try_join_delta(store, delta)
}

pub fn try_join_delta<S: GraphStore>(
    store: &mut S,
    delta: StampedBatch,
) -> GraphStoreResult<JoinReport> {
    let mut report = JoinReport::default();
    for stamped in delta.mutations {
        match stamped.mutation {
            GraphMutation::NodeUpsert(node) => {
                let existing = store.get_node_record(&node.id);
                let merged = merge_node(existing.as_ref(), node, stamped.hlc);
                if existing
                    .as_ref()
                    .map(|record| record.content_address() == merged.content_address())
                    .unwrap_or(false)
                {
                    report.ignored_stale += 1;
                    continue;
                }
                if existing
                    .as_ref()
                    .map(|record| record.tombstone)
                    .unwrap_or(false)
                    && !merged.tombstone
                {
                    report.revived += 1;
                }
                if !existing
                    .as_ref()
                    .map(|record| record.tombstone)
                    .unwrap_or(false)
                    && merged.tombstone
                {
                    report.tombstoned += 1;
                }
                store.upsert_node(merged)?;
                report.applied += 1;
            }
            GraphMutation::EdgeUpsert(edge) => {
                let existing = store.get_edge_record(&edge.id);
                let merged = merge_edge(existing.as_ref(), edge, stamped.hlc);
                if existing
                    .as_ref()
                    .map(|record| record.content_address() == merged.content_address())
                    .unwrap_or(false)
                {
                    report.ignored_stale += 1;
                    continue;
                }
                if existing
                    .as_ref()
                    .map(|record| record.tombstone)
                    .unwrap_or(false)
                    && !merged.tombstone
                {
                    report.revived += 1;
                }
                if !existing
                    .as_ref()
                    .map(|record| record.tombstone)
                    .unwrap_or(false)
                    && merged.tombstone
                {
                    report.tombstoned += 1;
                }
                store.upsert_edge(merged)?;
                report.applied += 1;
            }
            GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                return Err(GraphStoreError::new(
                    "crdt_delete_requires_tombstone_record",
                    "CRDT deltas must encode deletes as stamped tombstone records",
                ));
            }
        }
    }
    Ok(report)
}

pub fn diff_since<S: GraphStore + GraphSnapshotSource>(
    store: &S,
    vector: &VersionVector,
) -> StampedBatch {
    try_diff_since(store, vector).unwrap_or_default()
}

pub fn try_diff_since<S: GraphStore + GraphSnapshotSource>(
    store: &S,
    vector: &VersionVector,
) -> GraphStoreResult<StampedBatch> {
    let snapshot = store.graph_snapshot()?;
    Ok(diff_snapshot_since(snapshot, vector))
}

fn diff_snapshot_since(snapshot: GraphSnapshot, vector: &VersionVector) -> StampedBatch {
    let mut mutations = Vec::new();
    let mut frontier = VersionVector::default();
    for node in snapshot.nodes {
        let hlcs = record_hlcs(&node.properties);
        for hlc in &hlcs {
            frontier.observe(*hlc);
        }
        if let Some(max_hlc) = hlcs.iter().copied().max() {
            if hlcs.iter().any(|hlc| vector.is_missing(*hlc)) {
                mutations.push(StampedMutation::new(
                    GraphMutation::NodeUpsert(node),
                    max_hlc,
                ));
            }
        }
    }
    for edge in snapshot.edges {
        let hlcs = record_hlcs(&edge.properties);
        for hlc in &hlcs {
            frontier.observe(*hlc);
        }
        if let Some(max_hlc) = hlcs.iter().copied().max() {
            if hlcs.iter().any(|hlc| vector.is_missing(*hlc)) {
                mutations.push(StampedMutation::new(
                    GraphMutation::EdgeUpsert(edge),
                    max_hlc,
                ));
            }
        }
    }
    StampedBatch {
        mutations,
        frontier,
    }
}

fn merge_node(existing: Option<&NodeRecord>, incoming: NodeRecord, hlc: Hlc) -> NodeRecord {
    let mut incoming = stamp_node(incoming, hlc);
    let Some(existing) = existing else {
        return incoming;
    };
    let mut merged = existing.clone();
    merged.labels = merge_labels(&existing.labels, &incoming.labels);
    merged.properties = merge_properties(&existing.properties, &incoming.properties, hlc);
    merged.tombstone = merge_tombstone(
        existing.tombstone,
        tombstone_hlc(&existing.properties),
        incoming.tombstone,
        tombstone_hlc(&incoming.properties).unwrap_or(hlc),
    );
    ensure_record_hlc(
        &mut merged.properties,
        record_max_hlc(&incoming.properties).unwrap_or(hlc),
    );
    incoming.properties = merged.properties.clone();
    merged
}

fn merge_edge(existing: Option<&EdgeRecord>, incoming: EdgeRecord, hlc: Hlc) -> EdgeRecord {
    let incoming = stamp_edge(incoming, hlc);
    let Some(existing) = existing else {
        return incoming;
    };
    let incoming_record_hlc = record_hlc(&incoming.properties).unwrap_or(hlc);
    let existing_record_hlc = record_hlc(&existing.properties).unwrap_or_default();
    let mut merged = if incoming_record_hlc >= existing_record_hlc {
        let mut record = incoming.clone();
        record.parent_hashes = existing.parent_hashes.clone();
        record.content_hash = existing.content_hash.clone();
        record.version = existing.version;
        record
    } else {
        existing.clone()
    };
    merged.properties = merge_properties(&existing.properties, &incoming.properties, hlc);
    merged.tombstone = merge_tombstone(
        existing.tombstone,
        tombstone_hlc(&existing.properties),
        incoming.tombstone,
        tombstone_hlc(&incoming.properties).unwrap_or(hlc),
    );
    if existing.confidence != incoming.confidence {
        if let Some((_, winner)) = resolve_auto_confidence_edge(existing, &incoming, 0.0) {
            merged.confidence = winner.confidence;
        }
    }
    ensure_record_hlc(
        &mut merged.properties,
        record_max_hlc(&incoming.properties).unwrap_or(hlc),
    );
    merged
}

fn stamp_node(mut node: NodeRecord, hlc: Hlc) -> NodeRecord {
    stamp_properties(&mut node.properties, hlc);
    node
}

fn stamp_edge(mut edge: EdgeRecord, hlc: Hlc) -> EdgeRecord {
    stamp_properties(&mut edge.properties, hlc);
    edge
}

fn stamp_properties(properties: &mut Value, hlc: Hlc) {
    let map = ensure_object(properties);
    let keys = map
        .keys()
        .filter(|key| key.as_str() != CRDT_HLC_PROPERTY)
        .cloned()
        .collect::<Vec<_>>();
    let meta = ensure_meta(map);
    set_hlc_field(meta, "record", hlc);
    set_hlc_field(meta, "tombstone", hlc);
    let prop_meta = ensure_meta_properties(meta);
    for key in keys {
        prop_meta.entry(key).or_insert_with(|| hlc_to_value(hlc));
    }
}

fn merge_properties(existing: &Value, incoming: &Value, fallback_hlc: Hlc) -> Value {
    let existing_map = existing.as_object().cloned().unwrap_or_default();
    let incoming_map = incoming.as_object().cloned().unwrap_or_default();
    let mut out = existing_map.clone();

    for (key, value) in incoming_map
        .iter()
        .filter(|(key, _)| key.as_str() != CRDT_HLC_PROPERTY)
    {
        let incoming_hlc = property_hlc(incoming, key).unwrap_or(fallback_hlc);
        let existing_hlc = property_hlc(existing, key).unwrap_or_default();
        if incoming_hlc >= existing_hlc {
            out.insert(key.clone(), value.clone());
        }
    }

    let mut merged = Value::Object(out);
    merge_meta(&mut merged, existing, incoming, fallback_hlc);
    merged
}

fn merge_meta(out: &mut Value, existing: &Value, incoming: &Value, fallback_hlc: Hlc) {
    let out_map = ensure_object(out);
    let meta = ensure_meta(out_map);
    for candidate in [
        record_hlc(existing),
        record_hlc(incoming),
        Some(fallback_hlc),
    ]
    .into_iter()
    .flatten()
    {
        let current = get_hlc_field(meta, "record").unwrap_or_default();
        if candidate > current {
            set_hlc_field(meta, "record", candidate);
        }
    }
    for candidate in [tombstone_hlc(existing), tombstone_hlc(incoming)]
        .into_iter()
        .flatten()
    {
        let current = get_hlc_field(meta, "tombstone").unwrap_or_default();
        if candidate > current {
            set_hlc_field(meta, "tombstone", candidate);
        }
    }
    let prop_meta = ensure_meta_properties(meta);
    for source in [existing, incoming] {
        if let Some(source_props) = meta_properties(source) {
            for (key, value) in source_props {
                let Some(hlc) = value_to_hlc(value) else {
                    continue;
                };
                let current = prop_meta
                    .get(key)
                    .and_then(value_to_hlc)
                    .unwrap_or_default();
                if hlc > current {
                    prop_meta.insert(key.clone(), hlc_to_value(hlc));
                }
            }
        }
    }
}

fn merge_tombstone(
    existing_tombstone: bool,
    existing_hlc: Option<Hlc>,
    incoming_tombstone: bool,
    incoming_hlc: Hlc,
) -> bool {
    let existing_hlc = existing_hlc.unwrap_or_default();
    if incoming_tombstone && incoming_hlc > existing_hlc {
        true
    } else if !incoming_tombstone && incoming_hlc >= existing_hlc {
        false
    } else {
        existing_tombstone
    }
}

fn merge_labels(left: &[String], right: &[String]) -> Vec<String> {
    let mut labels = left
        .iter()
        .chain(right.iter())
        .filter(|label| !label.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn ensure_record_hlc(properties: &mut Value, hlc: Hlc) {
    let map = ensure_object(properties);
    let meta = ensure_meta(map);
    let current = get_hlc_field(meta, "record").unwrap_or_default();
    if hlc > current {
        set_hlc_field(meta, "record", hlc);
    }
}

fn record_hlc(properties: &Value) -> Option<Hlc> {
    meta(properties).and_then(|meta| get_hlc_field(meta, "record"))
}

fn tombstone_hlc(properties: &Value) -> Option<Hlc> {
    meta(properties).and_then(|meta| get_hlc_field(meta, "tombstone"))
}

fn property_hlc(properties: &Value, key: &str) -> Option<Hlc> {
    meta_properties(properties)
        .and_then(|props| props.get(key))
        .and_then(value_to_hlc)
}

fn record_hlcs(properties: &Value) -> Vec<Hlc> {
    let mut hlcs = Vec::new();
    if let Some(hlc) = record_hlc(properties) {
        hlcs.push(hlc);
    }
    if let Some(hlc) = tombstone_hlc(properties) {
        hlcs.push(hlc);
    }
    if let Some(props) = meta_properties(properties) {
        for value in props.values() {
            if let Some(hlc) = value_to_hlc(value) {
                hlcs.push(hlc);
            }
        }
    }
    hlcs.sort();
    hlcs.dedup();
    hlcs
}

fn record_max_hlc(properties: &Value) -> Option<Hlc> {
    record_hlcs(properties).into_iter().max()
}

fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    value.as_object_mut().expect("value is object")
}

fn ensure_meta(map: &mut Map<String, Value>) -> &mut Map<String, Value> {
    map.entry(CRDT_HLC_PROPERTY)
        .or_insert_with(|| Value::Object(Map::new()));
    if !map
        .get(CRDT_HLC_PROPERTY)
        .map(Value::is_object)
        .unwrap_or(false)
    {
        map.insert(CRDT_HLC_PROPERTY.to_string(), Value::Object(Map::new()));
    }
    map.get_mut(CRDT_HLC_PROPERTY)
        .and_then(Value::as_object_mut)
        .expect("metadata is object")
}

fn ensure_meta_properties(meta: &mut Map<String, Value>) -> &mut Map<String, Value> {
    meta.entry("properties".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !meta
        .get("properties")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        meta.insert("properties".to_string(), Value::Object(Map::new()));
    }
    meta.get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("properties metadata is object")
}

fn meta(properties: &Value) -> Option<&Map<String, Value>> {
    properties
        .as_object()
        .and_then(|map| map.get(CRDT_HLC_PROPERTY))
        .and_then(Value::as_object)
}

fn meta_properties(properties: &Value) -> Option<&Map<String, Value>> {
    meta(properties)
        .and_then(|meta| meta.get("properties"))
        .and_then(Value::as_object)
}

fn get_hlc_field(meta: &Map<String, Value>, key: &str) -> Option<Hlc> {
    meta.get(key).and_then(value_to_hlc)
}

fn set_hlc_field(meta: &mut Map<String, Value>, key: &str, hlc: Hlc) {
    meta.insert(key.to_string(), hlc_to_value(hlc));
}

fn value_to_hlc(value: &Value) -> Option<Hlc> {
    serde_json::from_value(value.clone()).ok()
}

fn hlc_to_value(hlc: Hlc) -> Value {
    serde_json::to_value(hlc).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::graph::attribution::GraphStats;
    use crate::graph::capability::{GraphRead, GraphSnapshotSource, GraphVerify, GraphWrite};
    use crate::graph::error::GraphStoreError;
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::mutation::GraphWriteResult;
    use crate::graph::query::{NeighborHit, NeighborQuery, NodeQuery};
    use crate::graph::record::{EdgeRecord, NodeRecord};
    use crate::graph::snapshot::GraphSnapshot;
    use crate::graph::verify::{GraphRebuildReport, VerifyReport};

    struct FailOnSecondUpsert {
        inner: InMemoryGraphStore,
        writes: usize,
    }

    impl GraphWrite for FailOnSecondUpsert {
        fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
            self.writes += 1;
            if self.writes == 2 {
                return Err(GraphStoreError::new(
                    "injected_write_failure",
                    "second write",
                ));
            }
            self.inner.upsert_node(node)
        }

        fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
            self.inner.upsert_edge(edge)
        }
    }

    impl GraphRead for FailOnSecondUpsert {
        fn get_node(&self, id: &str) -> Option<&NodeRecord> {
            self.inner.get_node(id)
        }

        fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
            self.inner.get_edge(id)
        }

        fn live_edge_records(&self) -> Vec<EdgeRecord> {
            GraphRead::live_edge_records(&self.inner)
        }

        fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
            self.inner.query_nodes(query)
        }

        fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
            self.inner.neighbors(query)
        }

        fn stats(&self) -> GraphStats {
            self.inner.stats()
        }
    }

    impl GraphSnapshotSource for FailOnSecondUpsert {
        fn graph_snapshot(&self) -> GraphStoreResult<GraphSnapshot> {
            self.inner.graph_snapshot()
        }
    }

    impl GraphVerify for FailOnSecondUpsert {
        fn verify(&self) -> VerifyReport {
            self.inner.verify()
        }

        fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
            self.inner.rebuild_indexes()
        }
    }

    fn hlc(actor: &str, physical_ms: i64, logical: u32) -> Hlc {
        Hlc::new(physical_ms, logical, ActorId::from_label(actor))
    }

    fn node_delta(id: &str, key: &str, value: Value, stamp: Hlc) -> StampedBatch {
        StampedBatch::new([StampedMutation::new(
            GraphMutation::NodeUpsert(NodeRecord::new(id, ["Thing"], json!({ key: value }))),
            stamp,
        )])
    }

    fn edge_delta(edge: EdgeRecord, stamp: Hlc) -> StampedBatch {
        StampedBatch::new([StampedMutation::new(GraphMutation::EdgeUpsert(edge), stamp)])
    }

    fn canonical_snapshot(mut snapshot: GraphSnapshot) -> GraphSnapshot {
        snapshot.version = 0;
        for node in &mut snapshot.nodes {
            node.version = 0;
            node.content_hash = None;
            node.parent_hashes.clear();
        }
        for edge in &mut snapshot.edges {
            edge.version = 0;
            edge.content_hash = None;
            edge.parent_hashes.clear();
        }
        snapshot
    }

    #[test]
    fn disjoint_property_deltas_commute() {
        let a = node_delta("n:1", "left", json!(1), hlc("codex", 10, 0));
        let b = node_delta("n:1", "right", json!(2), hlc("claude", 11, 0));

        let mut left = InMemoryGraphStore::new();
        let mut right = InMemoryGraphStore::new();
        join_delta(&mut left, a.clone()).unwrap();
        join_delta(&mut left, b.clone()).unwrap();
        join_delta(&mut right, b).unwrap();
        join_delta(&mut right, a).unwrap();

        assert_eq!(
            canonical_snapshot(left.snapshot()),
            canonical_snapshot(right.snapshot())
        );
        let node = left.get_node("n:1").unwrap();
        assert_eq!(node.properties["left"], json!(1));
        assert_eq!(node.properties["right"], json!(2));
    }

    #[test]
    fn join_is_idempotent() {
        let delta = node_delta("n:1", "name", json!("first"), hlc("codex", 10, 0));
        let mut store = InMemoryGraphStore::new();

        join_delta(&mut store, delta.clone()).unwrap();
        let once = store.snapshot();
        let report = join_delta(&mut store, delta).unwrap();

        assert_eq!(report.ignored_stale, 1);
        assert_eq!(store.snapshot(), once);
    }

    #[test]
    fn join_delta_surfaces_failure_after_partial_application() {
        let mut store = FailOnSecondUpsert {
            inner: InMemoryGraphStore::new(),
            writes: 0,
        };
        let delta = StampedBatch::new([
            StampedMutation::new(
                GraphMutation::NodeUpsert(NodeRecord::new("n:1", ["Thing"], json!({}))),
                hlc("codex", 10, 0),
            ),
            StampedMutation::new(
                GraphMutation::NodeUpsert(NodeRecord::new("n:2", ["Thing"], json!({}))),
                hlc("codex", 11, 0),
            ),
        ]);

        let error = join_delta(&mut store, delta).unwrap_err();

        assert_eq!(error.code, "injected_write_failure");
        assert!(store.get_node("n:1").is_some());
        assert!(store.get_node("n:2").is_none());
    }

    #[test]
    fn concurrent_same_key_uses_hlc_max() {
        let older = node_delta("n:1", "name", json!("old"), hlc("codex", 10, 0));
        let newer = node_delta("n:1", "name", json!("new"), hlc("claude", 20, 0));

        let mut left = InMemoryGraphStore::new();
        let mut right = InMemoryGraphStore::new();
        join_delta(&mut left, older.clone()).unwrap();
        join_delta(&mut left, newer.clone()).unwrap();
        join_delta(&mut right, newer).unwrap();
        join_delta(&mut right, older).unwrap();

        assert_eq!(
            canonical_snapshot(left.snapshot()),
            canonical_snapshot(right.snapshot())
        );
        assert_eq!(
            left.get_node("n:1").unwrap().properties["name"],
            json!("new")
        );
    }

    #[test]
    fn confidence_conflicts_use_auto_confidence_resolution() {
        let mut base = InMemoryGraphStore::new();
        for id in ["a", "b"] {
            base.upsert_node(NodeRecord::new(id, ["Thing"], json!({})))
                .unwrap();
        }
        let low = EdgeRecord::new("e:1", "a", "SUPPORTS", "b", json!({})).with_confidence(0.4);
        let high = EdgeRecord::new("e:1", "a", "SUPPORTS", "b", json!({})).with_confidence(0.9);

        let mut left = base.clone();
        let mut right = base;
        join_delta(&mut left, edge_delta(low.clone(), hlc("codex", 10, 0))).unwrap();
        join_delta(&mut left, edge_delta(high.clone(), hlc("claude", 11, 0))).unwrap();
        join_delta(&mut right, edge_delta(high, hlc("claude", 11, 0))).unwrap();
        join_delta(&mut right, edge_delta(low, hlc("codex", 10, 0))).unwrap();

        assert_eq!(
            canonical_snapshot(left.snapshot()),
            canonical_snapshot(right.snapshot())
        );
        assert_eq!(left.get_edge("e:1").unwrap().confidence, Some(0.9));
    }

    #[test]
    fn later_readd_revives_tombstoned_node() {
        let mut delete = NodeRecord::new("n:1", ["Thing"], json!({ "name": "gone" }));
        delete.tombstone = true;
        let readd = NodeRecord::new("n:1", ["Thing"], json!({ "name": "back" }));
        let mut store = InMemoryGraphStore::new();

        join_delta(
            &mut store,
            StampedBatch::new([StampedMutation::new(
                GraphMutation::NodeUpsert(delete),
                hlc("codex", 10, 0),
            )]),
        )
        .unwrap();
        let report = join_delta(
            &mut store,
            StampedBatch::new([StampedMutation::new(
                GraphMutation::NodeUpsert(readd),
                hlc("claude", 11, 0),
            )]),
        )
        .unwrap();

        assert_eq!(report.revived, 1);
        assert!(!store.get_node("n:1").unwrap().tombstone);
        assert_eq!(
            store.get_node("n:1").unwrap().properties["name"],
            json!("back")
        );
    }

    #[test]
    fn diff_since_returns_records_newer_than_vector() {
        let first = node_delta("n:1", "name", json!("one"), hlc("codex", 10, 0));
        let second = node_delta("n:2", "name", json!("two"), hlc("claude", 20, 0));
        let mut store = InMemoryGraphStore::new();
        join_delta(&mut store, first).unwrap();
        join_delta(&mut store, second).unwrap();
        let mut vector = VersionVector::default();
        vector.observe(hlc("codex", 10, 0));

        let diff = diff_since(&store, &vector);

        assert_eq!(diff.mutations.len(), 1);
        match &diff.mutations[0].mutation {
            GraphMutation::NodeUpsert(node) => assert_eq!(node.id, "n:2"),
            _ => panic!("expected node"),
        }
    }

    #[test]
    fn diff_since_returns_record_when_any_actor_component_is_missing() {
        let mut store = InMemoryGraphStore::new();
        join_delta(
            &mut store,
            node_delta("n:1", "left", json!(1), hlc("codex", 10, 0)),
        )
        .unwrap();
        join_delta(
            &mut store,
            node_delta("n:1", "right", json!(2), hlc("claude", 20, 0)),
        )
        .unwrap();
        let mut vector = VersionVector::default();
        vector.observe(hlc("claude", 20, 0));

        let diff = diff_since(&store, &vector);

        assert_eq!(diff.mutations.len(), 1);
        assert!(diff
            .frontier
            .high_water(ActorId::from_label("codex"))
            .is_some());
        match &diff.mutations[0].mutation {
            GraphMutation::NodeUpsert(node) => {
                assert_eq!(node.id, "n:1");
                assert_eq!(node.properties["left"], json!(1));
                assert_eq!(node.properties["right"], json!(2));
            }
            _ => panic!("expected node"),
        }
    }

    // ---- Property tests (SPEC Part 2 A2.1 / A2.2) -------------------------
    // The spec asks for a property test over random delta pairs. This crate has
    // no proptest dependency, so a seeded LCG drives deterministic pseudo-random
    // coverage (reproducible, no new dep). Each generated delta gets a unique
    // physical_ms, so HLCs are globally unique - the real clock's invariant and
    // the precondition that makes `join` a well-defined join-semilattice.
    // Comparisons go through canonical_snapshot so per-replica version counters
    // and record order do not produce false negatives.

    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items[(self.next_u64() >> 33) as usize % items.len()]
        }
    }

    fn random_delta(rng: &mut Lcg, seq: i64) -> StampedBatch {
        let id = rng.pick(&["n:1", "n:2", "n:3"]);
        let key = rng.pick(&["a", "b", "c"]);
        let actor = rng.pick(&["codex", "claude", "ai"]);
        let value = json!((rng.next_u64() >> 40) % 100);
        let mut node = NodeRecord::new(id, ["Thing"], json!({ key: value }));
        // Occasional delete exercises the add-wins tombstone path.
        node.tombstone = (rng.next_u64() >> 40) % 6 == 0;
        // 1000 + seq is unique per delta -> globally unique HLCs.
        StampedBatch::new([StampedMutation::new(
            GraphMutation::NodeUpsert(node),
            hlc(actor, 1000 + seq, 0),
        )])
    }

    #[test]
    fn property_join_commutes_over_random_pairs() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let mut seq = 0;
        for _ in 0..500 {
            let a = random_delta(&mut rng, seq);
            seq += 1;
            let b = random_delta(&mut rng, seq);
            seq += 1;

            let mut left = InMemoryGraphStore::new();
            let mut right = InMemoryGraphStore::new();
            join_delta(&mut left, a.clone()).unwrap();
            join_delta(&mut left, b.clone()).unwrap();
            join_delta(&mut right, b).unwrap();
            join_delta(&mut right, a).unwrap();

            assert_eq!(
                canonical_snapshot(left.snapshot()),
                canonical_snapshot(right.snapshot()),
                "join must commute: A then B == B then A"
            );
        }
    }

    #[test]
    fn property_join_is_idempotent_over_random_deltas() {
        let mut rng = Lcg(0x0fed_cba9_8765_4321);
        for seq in 0..500 {
            let delta = random_delta(&mut rng, seq);
            let mut store = InMemoryGraphStore::new();
            join_delta(&mut store, delta.clone()).unwrap();
            let once = canonical_snapshot(store.snapshot());
            join_delta(&mut store, delta).unwrap();
            assert_eq!(
                canonical_snapshot(store.snapshot()),
                once,
                "re-applying a delta is a no-op"
            );
        }
    }
}
