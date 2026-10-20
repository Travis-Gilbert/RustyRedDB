//! Sync scope by record class (SPEC-THEOREM-NODE-SYNC D1 / D5).
//!
//! One table, two consumers (fleet freshness and node sync). Fresh-class
//! domains are connected-only: live surfaces you visit, never data you sync.
//! Session-monotonic and relaxed classes are the accumulated self and sync.

use serde::{Deserialize, Serialize};

use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::snapshot::GraphSnapshot;

/// Property marker: this record must never leave the device via a sync pack.
/// Immutable after creation; promoting requires a copy with new provenance.
pub const LOCAL_ONLY_PROPERTY: &str = "local_only";

/// Named refusal when a fresh-class tool is invoked offline in synced mode.
pub const OFFLINE_FRESH_SURFACE: &str = "offline_fresh_surface";

/// How a domain participates in node sync.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncTransport {
    /// Accumulated self: packs, diffs, merges, CAS.
    Synced,
    /// Live surface: resolve through the hosted door; never pack.
    Connected,
}

/// One row of the shared class table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyncDomainClass {
    pub label: &'static str,
    pub transport: SyncTransport,
}

/// Shared class table for fleet freshness and node sync.
pub const SYNC_DOMAIN_CLASSES: &[SyncDomainClass] = &[
    // Synced: accumulated self
    SyncDomainClass {
        label: "MemoryDocument",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "MemoryNode",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "MemoryItem",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "MemoryAtom",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "HarnessMemory",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "DataRecord",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "DataRegistryEntry",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "Library",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "SkillPack",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "IndexSpineRecord",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "SyncConflictShadow",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "SyncReceipt",
        transport: SyncTransport::Synced,
    },
    SyncDomainClass {
        label: "SyncDevice",
        transport: SyncTransport::Synced,
    },
    // Connected only: fresh-class surfaces
    SyncDomainClass {
        label: "CoordinationRoom",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationMessage",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationIntent",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationPresence",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationRecord",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationStream",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationStreamEvent",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "CoordinationMember",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "HarnessCoordination",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "Plan",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "PlanProofReceipt",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "PlanVerifyReceipt",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "HarnessRun",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "TypedCommitment",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "WorkGraphNode",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "TaskNode",
        transport: SyncTransport::Connected,
    },
    SyncDomainClass {
        label: "IdentityBinding",
        transport: SyncTransport::Connected,
    },
];

/// Labels treated as hosted-wins on contested merge (publications / lineage).
pub const HOSTED_WINS_LABELS: &[&str] = &["DataRegistryEntry", "Library", "SkillPack"];

pub fn transport_for_label(label: &str) -> Option<SyncTransport> {
    SYNC_DOMAIN_CLASSES
        .iter()
        .find(|row| row.label == label)
        .map(|row| row.transport)
}

pub fn node_transport(node: &NodeRecord) -> Option<SyncTransport> {
    let mut saw_connected = false;
    let mut saw_synced = false;
    for label in &node.labels {
        match transport_for_label(label) {
            Some(SyncTransport::Connected) => saw_connected = true,
            Some(SyncTransport::Synced) => saw_synced = true,
            None => {}
        }
    }
    // Any connected label wins: a coordination message must never sync even if
    // it also carries a generic MemoryAtom-style label by mistake.
    if saw_connected {
        Some(SyncTransport::Connected)
    } else if saw_synced {
        Some(SyncTransport::Synced)
    } else {
        None
    }
}

pub fn is_local_only_node(node: &NodeRecord) -> bool {
    node.properties
        .get(LOCAL_ONLY_PROPERTY)
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

pub fn mark_local_only(node: &mut NodeRecord) {
    if let Some(map) = node.properties.as_object_mut() {
        map.insert(
            LOCAL_ONLY_PROPERTY.to_string(),
            serde_json::Value::Bool(true),
        );
    }
}

/// Refuse flipping `local_only` off after creation. Returns true when the
/// existing marker was preserved onto `next`.
pub fn preserve_local_only_marker(existing: &NodeRecord, next: &mut NodeRecord) -> bool {
    if !is_local_only_node(existing) {
        return false;
    }
    mark_local_only(next);
    true
}

pub fn is_synced_node(node: &NodeRecord) -> bool {
    !is_local_only_node(node) && matches!(node_transport(node), Some(SyncTransport::Synced))
}

pub fn is_connected_node(node: &NodeRecord) -> bool {
    matches!(node_transport(node), Some(SyncTransport::Connected))
}

pub fn is_hosted_wins_node(node: &NodeRecord) -> bool {
    node.labels
        .iter()
        .any(|label| HOSTED_WINS_LABELS.contains(&label.as_str()))
}

fn edge_endpoints_synced(snapshot: &GraphSnapshot, edge: &EdgeRecord) -> bool {
    let from_ok = snapshot
        .nodes
        .iter()
        .find(|node| node.id == edge.from_id)
        .is_some_and(is_synced_node);
    let to_ok = snapshot
        .nodes
        .iter()
        .find(|node| node.id == edge.to_id)
        .is_some_and(is_synced_node);
    from_ok && to_ok && !edge_is_local_only(edge)
}

fn edge_is_local_only(edge: &EdgeRecord) -> bool {
    edge.properties
        .get(LOCAL_ONLY_PROPERTY)
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

/// Filter a snapshot to the synced accumulated self, excluding local_only.
pub fn filter_snapshot_for_sync(snapshot: &GraphSnapshot) -> GraphSnapshot {
    let nodes: Vec<NodeRecord> = snapshot
        .nodes
        .iter()
        .filter(|node| is_synced_node(node))
        .cloned()
        .collect();
    let node_ids: std::collections::BTreeSet<String> =
        nodes.iter().map(|node| node.id.clone()).collect();
    let edges: Vec<EdgeRecord> = snapshot
        .edges
        .iter()
        .filter(|edge| {
            !edge_is_local_only(edge)
                && node_ids.contains(&edge.from_id)
                && node_ids.contains(&edge.to_id)
                && edge_endpoints_synced(snapshot, edge)
        })
        .cloned()
        .collect();
    GraphSnapshot {
        version: snapshot.version,
        historical_node_ids: nodes.iter().map(|node| node.id.clone()).collect(),
        historical_node_history_complete: snapshot.historical_node_history_complete,
        nodes,
        edges,
    }
}

/// Strip local_only records from any pack compilation path.
pub fn exclude_local_only_from_snapshot(snapshot: &GraphSnapshot) -> GraphSnapshot {
    let nodes: Vec<NodeRecord> = snapshot
        .nodes
        .iter()
        .filter(|node| !is_local_only_node(node))
        .cloned()
        .collect();
    let node_ids: std::collections::BTreeSet<String> =
        nodes.iter().map(|node| node.id.clone()).collect();
    let edges: Vec<EdgeRecord> = snapshot
        .edges
        .iter()
        .filter(|edge| {
            !edge_is_local_only(edge)
                && node_ids.contains(&edge.from_id)
                && node_ids.contains(&edge.to_id)
        })
        .cloned()
        .collect();
    GraphSnapshot {
        version: snapshot.version,
        historical_node_ids: node_ids,
        historical_node_history_complete: snapshot.historical_node_history_complete,
        nodes,
        edges,
    }
}

/// Tool-name prefixes / exact names that are fresh-class connected surfaces.
pub fn tool_is_fresh_surface(tool_name: &str) -> bool {
    let name = tool_name.trim();
    name.starts_with("coordination_")
        || name.starts_with("stream_")
        || name.starts_with("presence")
        || name == "coordinate"
        || name == "mentions"
        || name == "mentions_wait"
        || name.starts_with("plan")
        || name.starts_with("harness_begin")
        || name.starts_with("harness_step")
        || name.starts_with("harness_run")
        || name.contains("commitment")
        || name.contains("work_graph")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::graph::record::NodeRecord;

    fn mem_doc(id: &str, local_only: bool) -> NodeRecord {
        let mut node = NodeRecord::new(
            id,
            ["MemoryDocument", "HarnessMemory"],
            json!({ "updated_at": "2026-07-31T00:00:00Z" }),
        );
        if local_only {
            mark_local_only(&mut node);
        }
        node
    }

    #[test]
    fn fresh_class_never_enters_sync_pack() {
        let snapshot = GraphSnapshot {
            version: 1,
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            nodes: vec![
                mem_doc("mem:1", false),
                NodeRecord::new(
                    "coord:1",
                    ["CoordinationMessage", "HarnessCoordination"],
                    json!({}),
                ),
            ],
            edges: vec![],
        };
        let filtered = filter_snapshot_for_sync(&snapshot);
        assert_eq!(filtered.nodes.len(), 1);
        assert_eq!(filtered.nodes[0].id, "mem:1");
    }

    #[test]
    fn local_only_excluded_from_every_pack_filter() {
        let snapshot = GraphSnapshot {
            version: 1,
            historical_node_ids: Default::default(),
            historical_node_history_complete: false,
            nodes: vec![mem_doc("mem:local", true), mem_doc("mem:sync", false)],
            edges: vec![],
        };
        let sync = filter_snapshot_for_sync(&snapshot);
        assert_eq!(sync.nodes.len(), 1);
        assert_eq!(sync.nodes[0].id, "mem:sync");
        let all = exclude_local_only_from_snapshot(&snapshot);
        assert_eq!(all.nodes.len(), 1);
        assert_eq!(all.nodes[0].id, "mem:sync");
    }

    #[test]
    fn preserve_local_only_is_immutable() {
        let existing = mem_doc("mem:1", true);
        let mut next = mem_doc("mem:1", false);
        assert!(preserve_local_only_marker(&existing, &mut next));
        assert!(is_local_only_node(&next));
    }
}
