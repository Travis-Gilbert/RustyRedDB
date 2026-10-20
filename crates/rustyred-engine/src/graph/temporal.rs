use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::record::{EdgeRecord, NodeRecord};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TimeInterval {
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
}

impl TimeInterval {
    pub fn contains_ms(&self, at_ms: i64) -> bool {
        self.start_ms.map(|start| at_ms >= start).unwrap_or(true)
            && self.end_ms.map(|end| at_ms <= end).unwrap_or(true)
    }

    pub fn overlaps(&self, other: TimeInterval) -> bool {
        let self_start = self.start_ms.unwrap_or(i64::MIN);
        let self_end = self.end_ms.unwrap_or(i64::MAX);
        let other_start = other.start_ms.unwrap_or(i64::MIN);
        let other_end = other.end_ms.unwrap_or(i64::MAX);
        self_start <= other_end && other_start <= self_end
    }
}

pub fn node_time_interval(node: &NodeRecord) -> Option<TimeInterval> {
    let start_ms = property_i64(&node.properties, "t_start_ms");
    let end_ms = property_i64(&node.properties, "t_end_ms");
    if start_ms.is_none() && end_ms.is_none() {
        return None;
    }
    Some(TimeInterval { start_ms, end_ms })
}

pub fn edge_time_interval(edge: &EdgeRecord) -> Option<TimeInterval> {
    let start_ms = property_i64(&edge.properties, "t_start_ms");
    let end_ms = property_i64(&edge.properties, "t_end_ms");
    if start_ms.is_none() && end_ms.is_none() {
        return None;
    }
    Some(TimeInterval { start_ms, end_ms })
}

// ---- TTL helpers (design v2: first-class node-level TTL) ----

/// System property name for node-level TTL. Naming mirrors `t_start_ms` /
/// `t_end_ms` (the existing time-interval convention). Underscore prefix
/// marks it as system-managed metadata.

pub(crate) fn property_i64(properties: &Value, key: &str) -> Option<i64> {
    properties.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str().and_then(|raw| raw.parse::<i64>().ok()))
    })
}
