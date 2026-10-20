//! Per-class sync merge policy (SPEC-THEOREM-NODE-SYNC D4).
//!
//! No interactive conflict prompts. Contested records resolve by policy and
//! leave a SyncConflictShadow plus a receipt entry.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::snapshot::GraphSnapshot;
use crate::hash::Hash;
use crate::merge_registry::MergeRegistry;
use crate::sync_scope::{is_hosted_wins_node, is_synced_node};
use crate::versioned_graph::{
    compile_graph_pack, merge_graph_snapshots_with_registry, CompiledGraphPack,
    GraphCompileOptions, GraphMergeOptions, GraphMergeResult, GraphMergeStrategy, GraphObjectKind,
};
use crate::GraphStoreResult;

pub const SYNC_CONFLICT_SHADOW_LABEL: &str = "SyncConflictShadow";
pub const SYNC_SHADOW_OF_PROPERTY: &str = "shadow_of";
pub const SYNC_SHADOW_LOSER_SIDE_PROPERTY: &str = "loser_side";
pub const SYNC_SHADOW_REASON_PROPERTY: &str = "reason";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMergeSide {
    Local,
    Hosted,
}

impl SyncMergeSide {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Hosted => "hosted",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SyncConflictEntry {
    pub key: String,
    pub kind: GraphObjectKind,
    pub winner_side: SyncMergeSide,
    pub reason: String,
    pub shadow_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SyncMergeOutcome {
    pub status: String,
    pub merged_snapshot: GraphSnapshot,
    pub merged_pack: CompiledGraphPack,
    pub conflicts: Vec<SyncConflictEntry>,
    pub core_merge: GraphMergeResult,
}

fn record_updated_at(node: &NodeRecord) -> &str {
    node.properties
        .get("updated_at")
        .and_then(Value::as_str)
        .unwrap_or("")
}

#[cfg(test)]
fn node_by_id<'a>(snapshot: &'a GraphSnapshot, id: &str) -> Option<&'a NodeRecord> {
    snapshot.nodes.iter().find(|node| node.id == id)
}

fn shadow_id_for(winner_id: &str, loser_hash: &str) -> String {
    format!(
        "sync-shadow:{}:{}",
        winner_id,
        Hash::of(loser_hash.as_bytes()).to_prefixed()
    )
}

fn make_shadow(
    winner_id: &str,
    loser: &NodeRecord,
    loser_side: SyncMergeSide,
    reason: &str,
) -> NodeRecord {
    let loser_hash = loser
        .content_hash
        .clone()
        .unwrap_or_else(|| Hash::of(loser.id.as_bytes()).to_prefixed());
    let id = shadow_id_for(winner_id, &loser_hash);
    NodeRecord::new(
        id,
        [SYNC_CONFLICT_SHADOW_LABEL, "HarnessMemory"],
        json!({
            SYNC_SHADOW_OF_PROPERTY: winner_id,
            SYNC_SHADOW_LOSER_SIDE_PROPERTY: loser_side.as_str(),
            SYNC_SHADOW_REASON_PROPERTY: reason,
            "loser_id": loser.id,
            "loser_labels": loser.labels,
            "loser_properties": loser.properties,
            "loser_tombstone": loser.tombstone,
            "loser_version": loser.version,
            "updated_at": record_updated_at(loser),
        }),
    )
}

fn choose_newest(
    local: &NodeRecord,
    hosted: &NodeRecord,
) -> (SyncMergeSide, NodeRecord, NodeRecord) {
    let local_ts = record_updated_at(local);
    let hosted_ts = record_updated_at(hosted);
    if hosted_ts > local_ts {
        (SyncMergeSide::Hosted, hosted.clone(), local.clone())
    } else if local_ts > hosted_ts {
        (SyncMergeSide::Local, local.clone(), hosted.clone())
    } else if hosted.version >= local.version {
        (SyncMergeSide::Hosted, hosted.clone(), local.clone())
    } else {
        (SyncMergeSide::Local, local.clone(), hosted.clone())
    }
}

/// Resolve a contested synced node by D4 policy. Returns winner, optional
/// shadow for the loser, and a conflict entry when a shadow was created.
fn resolve_contested_node(
    local: Option<&NodeRecord>,
    hosted: Option<&NodeRecord>,
) -> Option<(NodeRecord, Option<NodeRecord>, Option<SyncConflictEntry>)> {
    match (local, hosted) {
        (None, None) => None,
        (Some(local), None) => Some((local.clone(), None, None)),
        (None, Some(hosted)) => Some((hosted.clone(), None, None)),
        (Some(local), Some(hosted)) => {
            if local.content_address().ok() == hosted.content_address().ok()
                && local.tombstone == hosted.tombstone
            {
                return Some((local.clone(), None, None));
            }

            // Tombstone wins over an edit on the other side.
            if local.tombstone != hosted.tombstone {
                let (winner_side, winner, loser) = if local.tombstone {
                    (SyncMergeSide::Local, local.clone(), hosted.clone())
                } else {
                    (SyncMergeSide::Hosted, hosted.clone(), local.clone())
                };
                let shadow =
                    make_shadow(&winner.id, &loser, opposite(winner_side), "tombstone_wins");
                let entry = SyncConflictEntry {
                    key: format!("node:{}", winner.id),
                    kind: GraphObjectKind::Node,
                    winner_side,
                    reason: "tombstone_wins".to_string(),
                    shadow_id: shadow.id.clone(),
                };
                return Some((winner, Some(shadow), Some(entry)));
            }

            if is_hosted_wins_node(local) || is_hosted_wins_node(hosted) {
                let shadow = make_shadow(&hosted.id, local, SyncMergeSide::Local, "hosted_wins");
                let entry = SyncConflictEntry {
                    key: format!("node:{}", hosted.id),
                    kind: GraphObjectKind::Node,
                    winner_side: SyncMergeSide::Hosted,
                    reason: "hosted_wins".to_string(),
                    shadow_id: shadow.id.clone(),
                };
                return Some((hosted.clone(), Some(shadow), Some(entry)));
            }

            let (winner_side, winner, loser) = choose_newest(local, hosted);
            let shadow = make_shadow(&winner.id, &loser, opposite(winner_side), "newest_write");
            let entry = SyncConflictEntry {
                key: format!("node:{}", winner.id),
                kind: GraphObjectKind::Node,
                winner_side,
                reason: "newest_write".to_string(),
                shadow_id: shadow.id.clone(),
            };
            Some((winner, Some(shadow), Some(entry)))
        }
    }
}

fn opposite(side: SyncMergeSide) -> SyncMergeSide {
    match side {
        SyncMergeSide::Local => SyncMergeSide::Hosted,
        SyncMergeSide::Hosted => SyncMergeSide::Local,
    }
}

fn index_nodes(snapshot: &GraphSnapshot) -> BTreeMap<String, &NodeRecord> {
    snapshot
        .nodes
        .iter()
        .filter(|node| {
            is_synced_node(node) || node.labels.iter().any(|l| l == SYNC_CONFLICT_SHADOW_LABEL)
        })
        .map(|node| (node.id.clone(), node))
        .collect()
}

fn index_edges(snapshot: &GraphSnapshot) -> BTreeMap<String, &EdgeRecord> {
    snapshot
        .edges
        .iter()
        .map(|edge| (edge.id.clone(), edge))
        .collect()
}

/// Three-way sync merge with D4 policy. Always produces a merged snapshot:
/// core AutoConfidence first, then policy resolution for remaining conflicts.
pub fn merge_sync_snapshots(
    base: &GraphSnapshot,
    local: &GraphSnapshot,
    hosted: &GraphSnapshot,
    options: GraphMergeOptions,
) -> GraphStoreResult<SyncMergeOutcome> {
    let mut options = options;
    if matches!(options.strategy, GraphMergeStrategy::Manual) {
        // v1 never prompts; Manual collapses to AutoConfidence + policy.
        options.strategy = GraphMergeStrategy::AutoConfidence;
    }

    let core = merge_graph_snapshots_with_registry(
        base,
        local,
        hosted,
        options.clone(),
        &MergeRegistry::substrate_sync_defaults(),
    )?;

    if core.status == "clean" {
        if let (Some(merged_snapshot), Some(merged_pack)) =
            (core.merged_snapshot.clone(), core.merged_pack.clone())
        {
            return Ok(SyncMergeOutcome {
                status: "clean".to_string(),
                merged_snapshot,
                merged_pack,
                conflicts: Vec::new(),
                core_merge: core,
            });
        }
    }

    // Policy pass over the union of synced keys from local and hosted.
    let local_nodes = index_nodes(local);
    let hosted_nodes = index_nodes(hosted);
    let base_nodes = index_nodes(base);
    let mut ids: BTreeSet<String> = BTreeSet::new();
    ids.extend(local_nodes.keys().cloned());
    ids.extend(hosted_nodes.keys().cloned());
    ids.extend(base_nodes.keys().cloned());

    let mut merged_nodes: BTreeMap<String, NodeRecord> = BTreeMap::new();
    let mut conflicts = Vec::new();

    for id in ids {
        let base_node = base_nodes.get(&id).copied();
        let local_node = local_nodes.get(&id).copied();
        let hosted_node = hosted_nodes.get(&id).copied();

        // Unchanged on both sides: keep base/local.
        if local_node.map(|n| n.content_address().ok())
            == hosted_node.map(|n| n.content_address().ok())
        {
            if let Some(node) = local_node.or(hosted_node).or(base_node) {
                merged_nodes.insert(id, node.clone());
            }
            continue;
        }
        if base_node.map(|n| n.content_address().ok())
            == local_node.map(|n| n.content_address().ok())
        {
            if let Some(node) = hosted_node.or(local_node) {
                merged_nodes.insert(id, node.clone());
            }
            continue;
        }
        if base_node.map(|n| n.content_address().ok())
            == hosted_node.map(|n| n.content_address().ok())
        {
            if let Some(node) = local_node.or(hosted_node) {
                merged_nodes.insert(id, node.clone());
            }
            continue;
        }

        if let Some((winner, shadow, entry)) = resolve_contested_node(local_node, hosted_node) {
            if let Some(shadow) = shadow {
                merged_nodes.insert(shadow.id.clone(), shadow);
            }
            if let Some(entry) = entry {
                conflicts.push(entry);
            }
            merged_nodes.insert(winner.id.clone(), winner);
        }
    }

    // Additive union for edges: prefer hosted on content conflict when both exist.
    let local_edges = index_edges(local);
    let hosted_edges = index_edges(hosted);
    let mut edge_ids: BTreeSet<String> = BTreeSet::new();
    edge_ids.extend(local_edges.keys().cloned());
    edge_ids.extend(hosted_edges.keys().cloned());
    let mut merged_edges: BTreeMap<String, EdgeRecord> = BTreeMap::new();
    for id in edge_ids {
        match (local_edges.get(&id), hosted_edges.get(&id)) {
            (Some(local), Some(hosted)) if local.tombstone != hosted.tombstone => {
                let winner = if local.tombstone {
                    (*local).clone()
                } else {
                    (*hosted).clone()
                };
                merged_edges.insert(id, winner);
            }
            (Some(local), Some(hosted)) => {
                if local.content_address().ok() == hosted.content_address().ok() {
                    merged_edges.insert(id, (*local).clone());
                } else {
                    merged_edges.insert(id, (*hosted).clone());
                }
            }
            (Some(local), None) => {
                merged_edges.insert(id, (*local).clone());
            }
            (None, Some(hosted)) => {
                merged_edges.insert(id, (*hosted).clone());
            }
            (None, None) => {}
        }
    }

    let historical_node_ids = merged_nodes.keys().cloned().collect();
    let merged_snapshot = GraphSnapshot {
        version: base
            .version
            .max(local.version)
            .max(hosted.version)
            .saturating_add(1),
        historical_node_ids,
        historical_node_history_complete: false,
        nodes: merged_nodes.into_values().collect(),
        edges: merged_edges.into_values().collect(),
    };
    let merged_pack = compile_graph_pack(
        &merged_snapshot,
        GraphCompileOptions {
            name: options
                .name
                .clone()
                .or_else(|| Some("sync-merged-pack".to_string())),
            branch: options.branch.clone(),
            parent_commits: options.parent_commits.clone(),
            author: options.author.clone(),
            message: options
                .message
                .clone()
                .or_else(|| Some("node sync merge".to_string())),
            timestamp_unix_ms: options.timestamp_unix_ms,
            include_payloads: options.include_payloads,
            exclude_local_only: true,
            sync_scope_only: true,
        },
    )?;

    Ok(SyncMergeOutcome {
        status: if conflicts.is_empty() {
            "clean".to_string()
        } else {
            "resolved_with_shadows".to_string()
        },
        merged_snapshot,
        merged_pack,
        conflicts,
        core_merge: core,
    })
}

/// List SyncConflictShadow nodes from a snapshot for `theorem sync --conflicts`.
pub fn list_conflict_shadows(snapshot: &GraphSnapshot) -> Vec<&NodeRecord> {
    snapshot
        .nodes
        .iter()
        .filter(|node| {
            node.labels
                .iter()
                .any(|label| label == SYNC_CONFLICT_SHADOW_LABEL)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync_scope::mark_local_only;

    fn snap(nodes: Vec<NodeRecord>) -> GraphSnapshot {
        let historical_node_ids = nodes.iter().map(|n| n.id.clone()).collect();
        GraphSnapshot {
            version: 1,
            historical_node_ids,
            historical_node_history_complete: false,
            nodes,
            edges: vec![],
        }
    }

    fn mem(id: &str, updated_at: &str, content: &str) -> NodeRecord {
        NodeRecord::new(
            id,
            ["MemoryDocument", "HarnessMemory"],
            json!({
                "updated_at": updated_at,
                "content": content,
                "doc_id": id,
            }),
        )
    }

    #[test]
    fn newest_write_keeps_loser_as_shadow() {
        let base = snap(vec![mem("mem:1", "2026-07-01T00:00:00Z", "base")]);
        let local = snap(vec![mem("mem:1", "2026-07-02T00:00:00Z", "local")]);
        let hosted = snap(vec![mem("mem:1", "2026-07-03T00:00:00Z", "hosted")]);
        let outcome = merge_sync_snapshots(&base, &local, &hosted, GraphMergeOptions::default())
            .expect("merge");
        assert_eq!(outcome.conflicts.len(), 1);
        assert_eq!(outcome.conflicts[0].reason, "newest_write");
        assert_eq!(outcome.conflicts[0].winner_side, SyncMergeSide::Hosted);
        let winner = node_by_id(&outcome.merged_snapshot, "mem:1").expect("winner");
        assert_eq!(winner.properties["content"], "hosted");
        assert!(!list_conflict_shadows(&outcome.merged_snapshot).is_empty());
    }

    #[test]
    fn tombstone_wins_and_shadows_edit() {
        let base = snap(vec![mem("mem:1", "2026-07-01T00:00:00Z", "base")]);
        let mut local = mem("mem:1", "2026-07-02T00:00:00Z", "edited locally");
        local.tombstone = false;
        let mut hosted = mem("mem:1", "2026-07-01T12:00:00Z", "base");
        hosted.tombstone = true;
        let outcome = merge_sync_snapshots(
            &base,
            &snap(vec![local]),
            &snap(vec![hosted]),
            GraphMergeOptions::default(),
        )
        .expect("merge");
        let winner = node_by_id(&outcome.merged_snapshot, "mem:1").expect("winner");
        assert!(winner.tombstone);
        assert_eq!(outcome.conflicts[0].reason, "tombstone_wins");
    }

    #[test]
    fn hosted_wins_for_registry() {
        let base = snap(vec![NodeRecord::new(
            "reg:1",
            ["DataRegistryEntry"],
            json!({ "updated_at": "2026-07-01T00:00:00Z", "name": "base" }),
        )]);
        let local = snap(vec![NodeRecord::new(
            "reg:1",
            ["DataRegistryEntry"],
            json!({ "updated_at": "2026-07-04T00:00:00Z", "name": "local" }),
        )]);
        let hosted = snap(vec![NodeRecord::new(
            "reg:1",
            ["DataRegistryEntry"],
            json!({ "updated_at": "2026-07-02T00:00:00Z", "name": "hosted" }),
        )]);
        let outcome = merge_sync_snapshots(&base, &local, &hosted, GraphMergeOptions::default())
            .expect("merge");
        assert_eq!(outcome.conflicts[0].reason, "hosted_wins");
        assert_eq!(
            node_by_id(&outcome.merged_snapshot, "reg:1")
                .unwrap()
                .properties["name"],
            "hosted"
        );
    }

    #[test]
    fn local_only_never_enters_merged_pack_objects() {
        let mut local_only = mem("mem:secret", "2026-07-02T00:00:00Z", "secret");
        mark_local_only(&mut local_only);
        let base = snap(vec![]);
        let local = snap(vec![
            local_only,
            mem("mem:ok", "2026-07-02T00:00:00Z", "ok"),
        ]);
        let hosted = snap(vec![]);
        let outcome = merge_sync_snapshots(&base, &local, &hosted, GraphMergeOptions::default())
            .expect("merge");
        let ids: Vec<_> = outcome
            .merged_pack
            .objects
            .iter()
            .filter_map(|object| object.payload.as_ref()?.get("id")?.as_str())
            .collect();
        assert!(ids.contains(&"mem:ok"));
        assert!(!ids.contains(&"mem:secret"));
    }
}
