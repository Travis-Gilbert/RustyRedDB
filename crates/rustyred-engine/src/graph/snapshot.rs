use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

use crate::graph::in_memory::EdgeArena;
use crate::graph::record::{EdgeRecord, NodeRecord};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GraphSnapshot {
    pub version: u64,
    pub nodes: Vec<NodeRecord>,
    pub edges: Vec<EdgeRecord>,
    #[serde(default)]
    pub historical_node_ids: BTreeSet<String>,
    #[serde(default)]
    pub historical_node_history_complete: bool,
}

/// Serialize the live graph with the same shape as [`GraphSnapshot`] without
/// first cloning every record into snapshot-owned vectors.
#[derive(Serialize)]
pub(crate) struct BorrowedGraphSnapshot<'a> {
    pub(crate) version: u64,
    pub(crate) nodes: MapValues<'a, NodeRecord>,
    pub(crate) edges: EdgeValues<'a>,
    pub(crate) historical_node_ids: &'a BTreeSet<String>,
    pub(crate) historical_node_history_complete: bool,
}

#[cfg(test)]
#[derive(Serialize)]
pub(crate) struct BorrowedRedCoreSnapshotEnvelope<'a> {
    pub(crate) version: u32,
    pub(crate) txn_id: u64,
    pub(crate) graph: BorrowedGraphSnapshot<'a>,
    pub(crate) historical_node_ids: &'a BTreeSet<String>,
    pub(crate) historical_node_history_complete: Option<bool>,
}

pub(crate) struct MapValues<'a, T>(pub(crate) &'a BTreeMap<String, T>);

pub(crate) struct EdgeValues<'a>(pub(crate) &'a EdgeArena);

impl<T> Serialize for MapValues<'_, T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0.values() {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

impl Serialize for EdgeValues<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0.values() {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

#[derive(Default)]
struct JsonByteCounter {
    bytes: usize,
}

impl Write for JsonByteCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes = self.bytes.saturating_add(buffer.len());
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn serialized_json_bytes(value: &impl Serialize) -> usize {
    let mut counter = JsonByteCounter::default();
    serde_json::to_writer(&mut counter, value)
        .map(|()| counter.bytes)
        .unwrap_or_default()
}

/// Every byte of a serialized graph snapshot that is not part of an individual
/// node, edge, or historical node id.
///
/// This is the bracketing, the four key names, the version digits, the
/// history-completeness boolean, and the array brackets, measured by running the
/// same serializer over the same type with empty collections instead of the
/// live ones. Empty `BTreeMap`/`BTreeSet`/`Vec` do not allocate, so this is an
/// O(1) call, and deriving the framing from serde itself keeps it exact as the
/// snapshot's shape evolves instead of hard-coding a character count.
///
/// Array separators are not part of the framing: they depend on the element
/// counts. See `CommittedFootprint::total`.
pub(crate) fn snapshot_framing_bytes(
    version: u64,
    historical_node_history_complete: bool,
) -> usize {
    let nodes: BTreeMap<String, NodeRecord> = BTreeMap::new();
    let edges = EdgeArena::default();
    let historical_node_ids: BTreeSet<String> = BTreeSet::new();
    serialized_json_bytes(&BorrowedGraphSnapshot {
        version,
        nodes: MapValues(&nodes),
        edges: EdgeValues(&edges),
        historical_node_ids: &historical_node_ids,
        historical_node_history_complete,
    })
}
