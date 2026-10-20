use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph::attribution::GraphStats;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerifyProblem {
    pub kind: String,
    pub id: String,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub stats: GraphStats,
    pub problems: Vec<VerifyProblem>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphRebuildReport {
    pub repaired: bool,
    pub before: VerifyReport,
    pub after: VerifyReport,
}

#[cfg(feature = "redis-store")]
#[derive(Default)]
pub(crate) struct ExpectedIndexes {
    pub(crate) out_adjacency: BTreeMap<(String, String), BTreeSet<String>>,
    pub(crate) in_adjacency: BTreeMap<(String, String), BTreeSet<String>>,
    pub(crate) label_index: BTreeMap<String, BTreeSet<String>>,
    pub(crate) edge_type_index: BTreeMap<String, BTreeSet<String>>,
    pub(crate) property_index: BTreeMap<(String, String), BTreeSet<String>>,
}

#[derive(Default)]
struct ExpectedOrdinalIndexes {
    pub(crate) out_adjacency: BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
    pub(crate) in_adjacency: BTreeMap<(String, String), BTreeSet<EdgeOrdinal>>,
    pub(crate) label_index: BTreeMap<String, BTreeSet<String>>,
    pub(crate) edge_type_index: BTreeMap<String, BTreeSet<EdgeOrdinal>>,
    pub(crate) property_index: BTreeMap<(String, String), BTreeSet<String>>,
}

use crate::graph::error::GraphStoreResult;
use crate::graph::in_memory::{EdgeOrdinal, InMemoryGraphStore};
use crate::graph::index::indexed_properties;
use crate::ordered::OrderedIndex;

impl InMemoryGraphStore {
    pub fn verify(&self) -> VerifyReport {
        let mut expected = ExpectedOrdinalIndexes::default();
        let mut problems = Vec::new();

        for node in self.nodes.values().filter(|node| !node.tombstone) {
            for label in &node.labels {
                expected
                    .label_index
                    .entry(label.clone())
                    .or_default()
                    .insert(node.id.clone());
            }
            for (key, token) in indexed_properties(&node.properties) {
                expected
                    .property_index
                    .entry((key, token))
                    .or_default()
                    .insert(node.id.clone());
            }
        }

        for (ordinal, edge) in self
            .edges
            .iter_with_ordinals()
            .filter(|(_, edge)| !edge.tombstone)
        {
            if self.get_node(&edge.from_id).is_none()
                && !self.cold_nonresident_node_ids.contains(&edge.from_id)
            {
                problems.push(VerifyProblem {
                    kind: "missing_from_endpoint".to_string(),
                    id: edge.id.clone(),
                    detail: format!("from endpoint {} is not a live node", edge.from_id),
                });
            }
            if self.get_node(&edge.to_id).is_none()
                && !self.cold_nonresident_node_ids.contains(&edge.to_id)
            {
                problems.push(VerifyProblem {
                    kind: "missing_to_endpoint".to_string(),
                    id: edge.id.clone(),
                    detail: format!("to endpoint {} is not a live node", edge.to_id),
                });
            }
            expected
                .edge_type_index
                .entry(edge.edge_type.clone())
                .or_default()
                .insert(ordinal);
            expected
                .out_adjacency
                .entry((edge.from_id.clone(), edge.edge_type.clone()))
                .or_default()
                .insert(ordinal);
            expected
                .in_adjacency
                .entry((edge.to_id.clone(), edge.edge_type.clone()))
                .or_default()
                .insert(ordinal);
        }

        if self.label_index != expected.label_index {
            problems.push(VerifyProblem {
                kind: "label_index_drift".to_string(),
                id: "label_index".to_string(),
                detail: "label index does not match live node labels".to_string(),
            });
        }
        if self.edge_type_index != expected.edge_type_index {
            problems.push(VerifyProblem {
                kind: "edge_type_index_drift".to_string(),
                id: "edge_type_index".to_string(),
                detail: "edge type index does not match live edge types".to_string(),
            });
        }
        if self.property_index != expected.property_index {
            problems.push(VerifyProblem {
                kind: "property_index_drift".to_string(),
                id: "property_index".to_string(),
                detail: "property index does not match live scalar node properties".to_string(),
            });
        }
        if self.out_adjacency != expected.out_adjacency {
            problems.push(VerifyProblem {
                kind: "out_adjacency_drift".to_string(),
                id: "out_adjacency".to_string(),
                detail: "out adjacency index does not match live edges".to_string(),
            });
        }
        if self.in_adjacency != expected.in_adjacency {
            problems.push(VerifyProblem {
                kind: "in_adjacency_drift".to_string(),
                id: "in_adjacency".to_string(),
                detail: "in adjacency index does not match live edges".to_string(),
            });
        }

        VerifyReport {
            ok: problems.is_empty(),
            stats: self.stats(),
            problems,
        }
    }

    pub fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
        let before = self.verify();
        self.rebuild_indexes_from_records();
        let after = self.verify();
        Ok(GraphRebuildReport {
            repaired: !before.ok && after.ok,
            before,
            after,
        })
    }

    pub(crate) fn rebuild_indexes_from_records(&mut self) {
        self.out_adjacency.clear();
        self.in_adjacency.clear();
        self.label_index.clear();
        self.edge_type_index.clear();
        self.property_index.clear();
        self.committed_footprint.forget_derived_index_bytes();
        let ordered_designations = self.ordered_designations();
        self.ordered_indexes.clear();
        for designation in ordered_designations {
            self.ordered_indexes.insert(
                (designation.label, designation.property),
                OrderedIndex::persistent(),
            );
        }
        // TTL index is symmetric with the other secondary indexes:
        // cleared here and repopulated via add_node_indexes below.
        // This gives us startup-scan TTL index rebuild for free
        // whenever rebuild_indexes is called (e.g., after snapshot
        // recovery).
        self.ttl_index.clear();
        let nodes = self.nodes.values().cloned().collect::<Vec<_>>();
        let edges = self
            .edges
            .iter_with_ordinals()
            .map(|(ordinal, edge)| (ordinal, edge.clone()))
            .collect::<Vec<_>>();
        for node in nodes {
            if !node.tombstone {
                self.add_node_indexes(&node);
            }
        }
        for (ordinal, edge) in edges {
            if !edge.tombstone {
                self.add_edge_indexes(&edge, ordinal);
            }
        }
        // A rebuild is an O(graph) path by construction, so re-derive the
        // committed footprint from the oracle rather than trusting the deltas
        // accumulated above (the ordered re-designations above deliberately
        // restart empty, and a caller may have tampered with the containers).
        self.reset_committed_footprint_from_recompute();
    }
}
