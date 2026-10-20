//! Incremental topology mark planning for epistemic shadow writeback.
//!
//! The full topology oracle is still the CSR view. This module compares the
//! before/after mark snapshots and returns exactly the nodes whose topology
//! marks changed, so callers can rewrite those shadows instead of every node.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph_csr::CsrGraph;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TopologyMark {
    pub node_id: String,
    pub weak_component: usize,
    pub scc_component: usize,
    pub articulation_point: bool,
    pub bridge_incident: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TopologySnapshot {
    pub marks: BTreeMap<String, TopologyMark>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct IncrementalTopologyPlan {
    pub changed_node_ids: Vec<String>,
    pub writeback_count: usize,
    pub full_node_count: usize,
}

pub fn topology_snapshot(csr: &CsrGraph) -> TopologySnapshot {
    let mut marks = BTreeMap::new();
    let weak_components = component_lookup(csr, csr.connected_components());
    let scc_components = component_lookup(csr, csr.strongly_connected_components());
    let (articulation_points, bridges) = csr.articulation_points_and_bridges();
    let articulation_points = articulation_points.into_iter().collect::<BTreeSet<_>>();
    let bridge_incident = bridges
        .into_iter()
        .flat_map(|(left, right)| [left, right])
        .collect::<BTreeSet<_>>();
    for dense in 0..csr.node_count() {
        let Some(node_id) = csr.id_of(dense) else {
            continue;
        };
        marks.insert(
            node_id.to_string(),
            TopologyMark {
                node_id: node_id.to_string(),
                weak_component: weak_components.get(&dense).copied().unwrap_or(dense),
                scc_component: scc_components.get(&dense).copied().unwrap_or(dense),
                articulation_point: articulation_points.contains(&dense),
                bridge_incident: bridge_incident.contains(&dense),
            },
        );
    }
    TopologySnapshot { marks }
}

pub fn plan_incremental_topology_update(
    before: &TopologySnapshot,
    after: &TopologySnapshot,
) -> IncrementalTopologyPlan {
    let changed_node_ids = topology_changed_nodes(before, after);
    IncrementalTopologyPlan {
        writeback_count: changed_node_ids.len(),
        full_node_count: after.marks.len(),
        changed_node_ids,
    }
}

pub fn topology_changed_nodes(before: &TopologySnapshot, after: &TopologySnapshot) -> Vec<String> {
    let ids = before
        .marks
        .keys()
        .chain(after.marks.keys())
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ids.into_iter()
        .filter(|node_id| before.marks.get(*node_id) != after.marks.get(*node_id))
        .map(ToString::to_string)
        .collect()
}

fn component_lookup(csr: &CsrGraph, components: Vec<Vec<usize>>) -> BTreeMap<usize, usize> {
    let mut lookup = BTreeMap::new();
    for mut component in components {
        component.sort_unstable();
        let component_index = component.first().copied().unwrap_or_default();
        for dense in component {
            if dense < csr.node_count() {
                lookup.insert(dense, component_index);
            }
        }
    }
    lookup
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_topology_plan_matches_full_snapshot_diff() {
        let before = CsrGraph::from_edges(vec![
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("c", "a", 1.0),
            ("a", "c", 1.0),
        ]);
        let after = CsrGraph::from_edges(vec![
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("c", "a", 1.0),
            ("a", "c", 1.0),
            ("c", "d", 1.0),
        ]);
        let before_snapshot = topology_snapshot(&before);
        let after_snapshot = topology_snapshot(&after);
        let full_changed = topology_changed_nodes(&before_snapshot, &after_snapshot);
        let plan = plan_incremental_topology_update(&before_snapshot, &after_snapshot);

        assert_eq!(plan.changed_node_ids, full_changed);
        assert_eq!(plan.writeback_count, plan.changed_node_ids.len());
        assert!(
            plan.writeback_count < plan.full_node_count,
            "incremental writeback should not rewrite unchanged topology marks"
        );
        assert!(plan.changed_node_ids.contains(&"c".to_string()));
        assert!(plan.changed_node_ids.contains(&"d".to_string()));
        assert!(!plan.changed_node_ids.contains(&"b".to_string()));
    }
}
