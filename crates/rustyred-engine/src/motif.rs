//! Motif extraction over the tenant graph (IDEA-SHAPE-SURFACE A1).
//!
//! The analogy engine in [`crate::analogy`] scores and maps relational motifs
//! but has nothing to feed it from a live tenant graph. This module closes that
//! gap: it extracts candidate [`RelationalSubgraph`]s as bounded, typed ego
//! neighborhoods around concept-bearing nodes, each stamped with its domain and
//! a stable content-derived `subgraph_id`. Extraction is pure structure over the
//! graph: no score or position is ever read back from a renderer.
//!
//! Parameters (radius, size caps, admitted relations) are the named `MOTIF_*`
//! constants on [`MotifConfig`], not magic numbers, so a deployment tunes them
//! without a code change and the defaults stay legible.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::analogy::{RelationalSignatureEdge, RelationalSignatureNode, RelationalSubgraph};
use crate::graph::capability::GraphStore;
use crate::graph::error::GraphStoreResult;
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{Direction, NodeRecord};
use crate::state::stable_hash;

/// Default ego-neighborhood radius in hops. The spec bounds a motif to 1..=2
/// hops; a radius-2 neighborhood already contains the radius-1 one, so one bound
/// covers both.
pub const MOTIF_RADIUS: usize = 2;
/// Cap on nodes admitted into a single extracted motif.
pub const MOTIF_MAX_NODES: usize = 12;
/// Cap on edges admitted into a single extracted motif.
pub const MOTIF_MAX_EDGES: usize = 24;
/// Cap on seed centers scanned per extraction pass.
// ponytail: whole-graph reseed each pass; make it commit-incremental if the
// tenant graph outgrows a single bounded scan.
pub const MOTIF_SEED_LIMIT: usize = 1024;
/// Default semantic relation edge types admitted into motifs. Bookkeeping edges
/// (analogical shadows, epistemic shadows, provenance) are never semantic
/// relations and are excluded by omission. Tunable per deployment.
pub const MOTIF_ADMITTED_RELATIONS: &[&str] = &[
    "SIMILAR_TO",
    "RELATES_TO",
    "REFERENCES",
    "DERIVES_FROM",
    "PART_OF",
    "CONTAINS",
    "DEPENDS_ON",
    "CAUSES",
];
/// Node property consulted first for a node's domain.
pub const MOTIF_DOMAIN_PROPERTY: &str = "domain";
/// Node property used as the human label on a signature node.
pub const MOTIF_LABEL_PROPERTY: &str = "label";
/// Fallback domain when no facet carries one.
pub const MOTIF_UNKNOWN_DOMAIN: &str = "unknown";

/// Which facet a motif's domain was derived from, recorded per the A1 rule that
/// the extractor names its domain source when no explicit domain property exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainSource {
    Property,
    Collection,
    Source,
    Unknown,
}

/// Extraction parameters. Defaults are the named `MOTIF_*` constants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifConfig {
    pub radius: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub admitted_relations: Vec<String>,
    /// Node labels treated as concept-bearing seed centers. Empty means every
    /// live node is a candidate center (bounded by `seed_limit`).
    pub concept_labels: Vec<String>,
    pub domain_property: String,
    pub label_property: String,
    pub seed_limit: Option<usize>,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            radius: MOTIF_RADIUS,
            max_nodes: MOTIF_MAX_NODES,
            max_edges: MOTIF_MAX_EDGES,
            admitted_relations: MOTIF_ADMITTED_RELATIONS
                .iter()
                .map(|relation| relation.to_string())
                .collect(),
            concept_labels: Vec::new(),
            domain_property: MOTIF_DOMAIN_PROPERTY.to_string(),
            label_property: MOTIF_LABEL_PROPERTY.to_string(),
            seed_limit: Some(MOTIF_SEED_LIMIT),
        }
    }
}

impl MotifConfig {
    fn admits(&self, edge_type: &str) -> bool {
        self.admitted_relations
            .iter()
            .any(|relation| relation == edge_type)
    }
}

/// One extracted motif plus the facet its domain was read from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedMotif {
    pub subgraph: RelationalSubgraph,
    pub domain_source: DomainSource,
}

/// Extract candidate motifs from the live tenant graph (A1). Bounded typed ego
/// neighborhoods around concept-bearing centers, deduplicated by their stable
/// content-derived `subgraph_id`, none exceeding the configured caps.
pub fn extract_motifs<S: GraphStore>(
    store: &S,
    config: &MotifConfig,
) -> GraphStoreResult<Vec<ExtractedMotif>> {
    let mut motifs: BTreeMap<String, ExtractedMotif> = BTreeMap::new();
    for center in seed_centers(store, config) {
        if let Some(motif) = extract_motif_around(store, &center, config)? {
            motifs
                .entry(motif.subgraph.subgraph_id.clone())
                .or_insert(motif);
        }
    }
    Ok(motifs.into_values().collect())
}

fn seed_centers<S: GraphStore>(store: &S, config: &MotifConfig) -> Vec<NodeRecord> {
    if config.concept_labels.is_empty() {
        return store.query_nodes(NodeQuery {
            label: None,
            properties: BTreeMap::new(),
            limit: config.seed_limit,
            include_expired: false,
        });
    }
    let mut seeds = Vec::new();
    let mut seen = BTreeSet::new();
    for label in &config.concept_labels {
        for node in store.query_nodes(NodeQuery {
            label: Some(label.clone()),
            properties: BTreeMap::new(),
            limit: config.seed_limit,
            include_expired: false,
        }) {
            if seen.insert(node.id.clone()) {
                seeds.push(node);
            }
        }
    }
    seeds
}

fn extract_motif_around<S: GraphStore>(
    store: &S,
    center: &NodeRecord,
    config: &MotifConfig,
) -> GraphStoreResult<Option<ExtractedMotif>> {
    let members = ego_members(store, &center.id, config);
    if members.len() < 2 {
        return Ok(None);
    }

    // Induced edges: each admitted directed edge whose endpoints are both in the
    // member set, collected once from its source via the Out direction and
    // capped at `max_edges`.
    let mut edge_keys: BTreeSet<(String, String, String)> = BTreeSet::new();
    'collect: for node in &members {
        for hit in store.neighbors(NeighborQuery {
            node_id: node.clone(),
            direction: Direction::Out,
            edge_type: None,
            include_expired: false,
        }) {
            if config.admits(&hit.edge_type) && members.contains(&hit.node_id) {
                edge_keys.insert((node.clone(), hit.edge_type, hit.node_id));
                if edge_keys.len() >= config.max_edges {
                    break 'collect;
                }
            }
        }
    }
    if edge_keys.is_empty() {
        return Ok(None);
    }

    let (domain, domain_source) = derive_domain(center, config);
    let nodes = members
        .iter()
        .filter_map(|id| {
            store.get_node(id).map(|record| RelationalSignatureNode {
                node_id: id.clone(),
                label: node_label(record, config),
                domain: derive_domain(record, config).0,
            })
        })
        .collect::<Vec<_>>();
    let edges = edge_keys
        .into_iter()
        .map(|(from_id, relation, to_id)| RelationalSignatureEdge {
            from_id,
            relation,
            to_id,
        })
        .collect::<Vec<_>>();
    let subgraph_id = motif_id(&domain, &nodes, &edges)?;
    Ok(Some(ExtractedMotif {
        subgraph: RelationalSubgraph {
            subgraph_id,
            domain,
            nodes,
            edges,
        },
        domain_source,
    }))
}

/// Breadth-first ego membership up to `config.radius` hops over admitted
/// relations in either direction, capped at `config.max_nodes`.
fn ego_members<S: GraphStore>(store: &S, center: &str, config: &MotifConfig) -> BTreeSet<String> {
    let mut members = BTreeSet::from([center.to_string()]);
    let mut frontier = vec![center.to_string()];
    for _hop in 0..config.radius {
        if members.len() >= config.max_nodes {
            break;
        }
        let mut next = Vec::new();
        for node in &frontier {
            for direction in [Direction::Out, Direction::In] {
                for hit in store.neighbors(NeighborQuery {
                    node_id: node.clone(),
                    direction,
                    edge_type: None,
                    include_expired: false,
                }) {
                    if !config.admits(&hit.edge_type) {
                        continue;
                    }
                    if members.len() >= config.max_nodes {
                        break;
                    }
                    if members.insert(hit.node_id.clone()) {
                        next.push(hit.node_id);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    members
}

fn derive_domain(node: &NodeRecord, config: &MotifConfig) -> (String, DomainSource) {
    let property = |key: &str| {
        node.properties
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|value| !value.is_empty())
    };
    if let Some(domain) = property(&config.domain_property) {
        return (domain, DomainSource::Property);
    }
    if let Some(domain) = property("collection") {
        return (domain, DomainSource::Collection);
    }
    if let Some(domain) = property("source") {
        return (domain, DomainSource::Source);
    }
    (MOTIF_UNKNOWN_DOMAIN.to_string(), DomainSource::Unknown)
}

fn node_label(node: &NodeRecord, config: &MotifConfig) -> String {
    node.properties
        .get(&config.label_property)
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|value| !value.is_empty())
        .or_else(|| node.labels.first().cloned())
        .unwrap_or_else(|| node.id.clone())
}

/// Content-derived, stable id over the sorted node ids, sorted edges, and domain,
/// so re-extracting an unchanged motif yields the same id (A2's no-duplicates
/// invariant rests on this).
fn motif_id(
    domain: &str,
    nodes: &[RelationalSignatureNode],
    edges: &[RelationalSignatureEdge],
) -> GraphStoreResult<String> {
    #[derive(Serialize)]
    struct MotifContent<'a> {
        kind: &'static str,
        domain: &'a str,
        node_ids: Vec<&'a str>,
        edges: Vec<(&'a str, &'a str, &'a str)>,
    }
    let mut node_ids = nodes
        .iter()
        .map(|node| node.node_id.as_str())
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    let mut edge_tuples = edges
        .iter()
        .map(|edge| {
            (
                edge.from_id.as_str(),
                edge.relation.as_str(),
                edge.to_id.as_str(),
            )
        })
        .collect::<Vec<_>>();
    edge_tuples.sort_unstable();
    Ok(format!(
        "motif:{}",
        stable_hash(MotifContent {
            kind: "relational_motif_v1",
            domain,
            node_ids,
            edges: edge_tuples,
        })?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analogy::relational_signature;
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::record::EdgeRecord;
    use serde_json::json;

    fn concept(store: &mut InMemoryGraphStore, id: &str, label: &str, domain: &str) {
        store
            .upsert_node(NodeRecord::new(
                id,
                ["Concept"],
                json!({ "label": label, "domain": domain }),
            ))
            .unwrap();
    }

    fn relation(store: &mut InMemoryGraphStore, id: &str, from: &str, rel: &str, to: &str) {
        store
            .upsert_edge(EdgeRecord::new(id, from, rel, to, json!({})))
            .unwrap();
    }

    fn hand_built(
        domain: &str,
        nodes: &[(&str, &str)],
        edges: &[(&str, &str, &str)],
    ) -> RelationalSubgraph {
        RelationalSubgraph {
            subgraph_id: format!("hand:{domain}"),
            domain: domain.to_string(),
            nodes: nodes
                .iter()
                .map(|(id, label)| RelationalSignatureNode {
                    node_id: id.to_string(),
                    label: label.to_string(),
                    domain: domain.to_string(),
                })
                .collect(),
            edges: edges
                .iter()
                .map(|(from, rel, to)| RelationalSignatureEdge {
                    from_id: from.to_string(),
                    relation: rel.to_string(),
                    to_id: to.to_string(),
                })
                .collect(),
        }
    }

    fn seeded_store() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        // Solar motif (astronomy).
        concept(&mut store, "sun", "sun", "astronomy");
        concept(&mut store, "earth", "earth", "astronomy");
        concept(&mut store, "mars", "mars", "astronomy");
        relation(&mut store, "e-earth-sun", "earth", "orbits", "sun");
        relation(&mut store, "e-mars-sun", "mars", "orbits", "sun");
        // Atom motif (chemistry), same structure, different surface.
        concept(&mut store, "nucleus", "nucleus", "chemistry");
        concept(&mut store, "electron-a", "electron", "chemistry");
        concept(&mut store, "electron-b", "electron", "chemistry");
        relation(&mut store, "e-ea-nuc", "electron-a", "orbits", "nucleus");
        relation(&mut store, "e-eb-nuc", "electron-b", "orbits", "nucleus");
        // Distractors joined only by a non-admitted relation.
        concept(&mut store, "card-1", "catalog entry", "catalog");
        concept(&mut store, "card-2", "catalog entry", "catalog");
        relation(&mut store, "e-card", "card-1", "listed_with", "card-2");
        store
    }

    fn orbits_config() -> MotifConfig {
        MotifConfig {
            admitted_relations: vec!["orbits".to_string()],
            ..MotifConfig::default()
        }
    }

    #[test]
    fn extractor_recovers_solar_and_atom_signatures_ignoring_distractors() {
        let store = seeded_store();
        let motifs = extract_motifs(&store, &orbits_config()).unwrap();

        let solar = hand_built(
            "astronomy",
            &[("sun", "sun"), ("earth", "earth"), ("mars", "mars")],
            &[("earth", "orbits", "sun"), ("mars", "orbits", "sun")],
        );
        let atom = hand_built(
            "chemistry",
            &[
                ("nucleus", "nucleus"),
                ("electron-a", "electron"),
                ("electron-b", "electron"),
            ],
            &[
                ("electron-a", "orbits", "nucleus"),
                ("electron-b", "orbits", "nucleus"),
            ],
        );
        let solar_sig = relational_signature(&solar);
        let atom_sig = relational_signature(&atom);

        let has_astronomy = motifs.iter().any(|motif| {
            motif.subgraph.domain == "astronomy"
                && relational_signature(&motif.subgraph) == solar_sig
        });
        let has_chemistry = motifs.iter().any(|motif| {
            motif.subgraph.domain == "chemistry"
                && relational_signature(&motif.subgraph) == atom_sig
        });
        assert!(has_astronomy, "solar motif recovered by structure");
        assert!(has_chemistry, "atom motif recovered by structure");

        // The catalog distractor carries no admitted relation, so it is not a motif.
        assert!(
            motifs
                .iter()
                .all(|motif| motif.subgraph.domain != "catalog"),
            "distractors with no admitted relation produce no motif"
        );
    }

    #[test]
    fn no_extracted_motif_exceeds_the_caps() {
        let store = seeded_store();
        let config = orbits_config();
        for motif in extract_motifs(&store, &config).unwrap() {
            assert!(motif.subgraph.nodes.len() <= config.max_nodes);
            assert!(motif.subgraph.edges.len() <= config.max_edges);
            assert_eq!(motif.domain_source, DomainSource::Property);
        }
    }

    #[test]
    fn subgraph_ids_are_stable_across_runs() {
        let store = seeded_store();
        let config = orbits_config();
        let first = extract_motifs(&store, &config).unwrap();
        let second = extract_motifs(&store, &config).unwrap();
        let ids = |motifs: &[ExtractedMotif]| {
            motifs
                .iter()
                .map(|motif| motif.subgraph.subgraph_id.clone())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(ids(&first), ids(&second));
        // Solar and atom are distinct motifs; distractors add none.
        assert_eq!(first.len(), 2);
    }

    #[test]
    fn domain_falls_back_to_collection_then_source() {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new(
                "a",
                ["Concept"],
                json!({ "label": "a", "collection": "inbox" }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "b",
                ["Concept"],
                json!({ "label": "b", "collection": "inbox" }),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new("e", "a", "RELATES_TO", "b", json!({})))
            .unwrap();

        let motifs = extract_motifs(&store, &MotifConfig::default()).unwrap();
        assert_eq!(motifs.len(), 1);
        assert_eq!(motifs[0].subgraph.domain, "inbox");
        assert_eq!(motifs[0].domain_source, DomainSource::Collection);
    }
}
