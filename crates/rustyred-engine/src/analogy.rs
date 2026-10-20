//! Relational-signature retrieval for analogical candidates.
//!
//! This is the pure structural oracle for the analogy engine. It scores motifs
//! by relation multiset and degree pattern, not labels or embeddings, so a
//! solar-system motif can retrieve an atom motif while surface-similar but
//! structurally different subgraphs are rejected. The GraphBLAS implementation
//! can lower this signature/matching contract later without changing the tests.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::analogy_thompson::ThompsonAnalogyOracle;
use crate::epistemic::{
    epistemic_shadow_node_id, has_epistemic_shadow_edge_id, DEFAULT_EPISTEMIC_ENGINE_VERSION,
    EPISTEMIC_SHADOW_LABEL, HAS_EPISTEMIC_SHADOW,
};
use crate::graph::capability::GraphStore;
use crate::graph::error::GraphStoreResult;
use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord, Provenance};
use crate::graph::ttl::now_ms;
use crate::ranking::RankCandidate;
use crate::state::stable_hash;

pub const ANALOGICAL_HYPOTHESIS_LABEL: &str = "AnalogicalTransferHypothesis";
pub const ANALOGICAL_SCHEMA_LABEL: &str = "AnalogicalSchema";
pub const ANALOGICAL_SCHEMA_INSTANCE_LABEL: &str = "AnalogicalSchemaInstance";
pub const ANALOGICAL_SCHEMA_INSTANCE_EDGE: &str = "ANALOGICAL_SCHEMA_INSTANCE";
pub const ANALOGICAL_USES_SCHEMA_EDGE: &str = "ANALOGICAL_USES_SCHEMA";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RelationalSignatureNode {
    pub node_id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub domain: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RelationalSignatureEdge {
    pub from_id: String,
    pub relation: String,
    pub to_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RelationalSubgraph {
    pub subgraph_id: String,
    pub domain: String,
    #[serde(default)]
    pub nodes: Vec<RelationalSignatureNode>,
    #[serde(default)]
    pub edges: Vec<RelationalSignatureEdge>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RelationalSignature {
    pub relation_counts: BTreeMap<String, usize>,
    pub degree_pattern: Vec<(usize, usize)>,
    pub connected: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalSignatureMatch {
    pub subgraph_id: String,
    pub domain: String,
    pub structural_score: f64,
    pub surface_overlap: f64,
    pub relation_overlap: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StructureMapping {
    pub source_subgraph_id: String,
    pub target_subgraph_id: String,
    pub node_mappings: Vec<(String, String)>,
    pub relation_mappings: Vec<(String, String)>,
    pub systematicity_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchemaRecord {
    pub schema_id: String,
    pub relation_counts: BTreeMap<String, usize>,
    pub degree_pattern: Vec<(usize, usize)>,
    pub instance_subgraph_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SchemaRegistry {
    pub schemas: BTreeMap<String, SchemaRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalogicalTransferHypothesis {
    pub hypothesis_id: String,
    pub shadow_node_id: String,
    pub schema_id: String,
    pub source_subgraph_id: String,
    pub target_subgraph_id: String,
    pub source_edge: RelationalSignatureEdge,
    pub projected_edge: RelationalSignatureEdge,
    pub node_mappings: Vec<(String, String)>,
    pub relation_mappings: Vec<(String, String)>,
    pub confidence: f64,
    pub epistemic_status: String,
    pub provenance: Vec<String>,
    #[serde(default)]
    pub invalidated_by: Option<String>,
}

impl AnalogicalTransferHypothesis {
    pub fn rank_candidate(&self) -> RankCandidate {
        let mut candidate = RankCandidate::new(self.shadow_node_id.clone());
        candidate.acceptance_status = self.epistemic_status.clone();
        candidate.epistemic_weight = self.confidence as f32;
        candidate
    }
}

pub fn relational_signature(subgraph: &RelationalSubgraph) -> RelationalSignature {
    let mut relation_counts = BTreeMap::new();
    let mut degrees = BTreeMap::<String, (usize, usize)>::new();
    for node in &subgraph.nodes {
        degrees.entry(node.node_id.clone()).or_default();
    }
    for edge in &subgraph.edges {
        *relation_counts.entry(edge.relation.clone()).or_insert(0) += 1;
        degrees.entry(edge.from_id.clone()).or_default().0 += 1;
        degrees.entry(edge.to_id.clone()).or_default().1 += 1;
    }
    let mut degree_pattern = degrees.into_values().collect::<Vec<_>>();
    degree_pattern.sort();
    RelationalSignature {
        relation_counts,
        connected: is_connected(subgraph),
        degree_pattern,
    }
}

pub fn retrieve_relational_analogies(
    target: &RelationalSubgraph,
    candidates: &[RelationalSubgraph],
    min_structural_score: f64,
) -> Vec<RelationalSignatureMatch> {
    let target_signature = relational_signature(target);
    let mut matches = Vec::new();
    for candidate in candidates {
        if candidate.subgraph_id == target.subgraph_id || candidate.domain == target.domain {
            continue;
        }
        let candidate_signature = relational_signature(candidate);
        let relation_overlap = relation_overlap(
            &target_signature.relation_counts,
            &candidate_signature.relation_counts,
        );
        let degree_overlap = degree_overlap(
            &target_signature.degree_pattern,
            &candidate_signature.degree_pattern,
        );
        let connectivity_bonus = if target_signature.connected && candidate_signature.connected {
            0.1
        } else {
            0.0
        };
        let structural_score =
            (0.65 * relation_overlap + 0.35 * degree_overlap + connectivity_bonus).min(1.0);
        if structural_score < min_structural_score {
            continue;
        }
        matches.push(RelationalSignatureMatch {
            subgraph_id: candidate.subgraph_id.clone(),
            domain: candidate.domain.clone(),
            structural_score,
            surface_overlap: surface_overlap(target, candidate),
            relation_overlap,
        });
    }
    matches.sort_by(|left, right| {
        right
            .structural_score
            .partial_cmp(&left.structural_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.surface_overlap
                    .partial_cmp(&right.surface_overlap)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.subgraph_id.cmp(&right.subgraph_id))
    });
    matches
}

pub fn align_structure_mapping(
    source: &RelationalSubgraph,
    target: &RelationalSubgraph,
) -> StructureMapping {
    let source_roles = node_roles(source);
    let target_roles = node_roles(target);
    let mut node_mappings = Vec::new();
    for (source_id, source_role) in &source_roles {
        if let Some((target_id, _)) = target_roles
            .iter()
            .find(|(_, target_role)| *target_role == source_role)
        {
            if !node_mappings
                .iter()
                .any(|(_, mapped_target): &(String, String)| mapped_target == target_id)
            {
                node_mappings.push((source_id.clone(), target_id.clone()));
            }
        }
    }
    let mut relation_mappings = Vec::new();
    let target_relations = target
        .edges
        .iter()
        .map(|edge| edge.relation.clone())
        .collect::<BTreeSet<_>>();
    for relation in source
        .edges
        .iter()
        .map(|edge| edge.relation.clone())
        .collect::<BTreeSet<_>>()
    {
        if target_relations.contains(&relation) {
            relation_mappings.push((relation.clone(), relation));
        }
    }
    let mapped_edge_count = mapped_relation_edges(source, target, &node_mappings);
    let possible_edges = source.edges.len().max(target.edges.len()).max(1);
    let connected_bonus = if mapped_edge_count > 1 && is_connected(source) && is_connected(target) {
        0.2
    } else {
        0.0
    };
    StructureMapping {
        source_subgraph_id: source.subgraph_id.clone(),
        target_subgraph_id: target.subgraph_id.clone(),
        node_mappings,
        relation_mappings,
        systematicity_score: ((mapped_edge_count as f64 / possible_edges as f64) + connected_bonus)
            .min(1.0),
    }
}

pub fn extract_schema(mapping: &StructureMapping, source: &RelationalSubgraph) -> SchemaRecord {
    let signature = relational_signature(source);
    let schema_id = schema_id_for_signature(&signature);
    let mut instance_subgraph_ids = vec![
        mapping.source_subgraph_id.clone(),
        mapping.target_subgraph_id.clone(),
    ];
    instance_subgraph_ids.sort();
    instance_subgraph_ids.dedup();
    SchemaRecord {
        schema_id,
        relation_counts: signature.relation_counts,
        degree_pattern: signature.degree_pattern,
        instance_subgraph_ids,
    }
}

pub fn accrete_schema(registry: &mut SchemaRegistry, schema: SchemaRecord) -> SchemaRecord {
    if let Some(existing) = registry.schemas.get_mut(&schema.schema_id) {
        for instance in schema.instance_subgraph_ids {
            if !existing.instance_subgraph_ids.contains(&instance) {
                existing.instance_subgraph_ids.push(instance);
            }
        }
        existing.instance_subgraph_ids.sort();
        return existing.clone();
    }
    registry
        .schemas
        .insert(schema.schema_id.clone(), schema.clone());
    schema
}

pub fn write_schema_record<S: GraphStore>(
    store: &mut S,
    schema: &SchemaRecord,
) -> GraphStoreResult<()> {
    store.upsert_node(NodeRecord::new(
        schema.schema_id.clone(),
        [ANALOGICAL_SCHEMA_LABEL],
        json!({
            "schema_id": schema.schema_id,
            "relation_counts": schema.relation_counts,
            "degree_pattern": schema.degree_pattern,
            "instance_subgraph_ids": schema.instance_subgraph_ids,
        }),
    ))?;
    for instance in &schema.instance_subgraph_ids {
        store.upsert_node(NodeRecord::new(
            instance.clone(),
            [ANALOGICAL_SCHEMA_INSTANCE_LABEL],
            json!({
                "subgraph_id": instance,
                "schema_id": schema.schema_id,
            }),
        ))?;
        store.upsert_edge(EdgeRecord::new(
            schema_instance_edge_id(&schema.schema_id, instance)?,
            schema.schema_id.clone(),
            ANALOGICAL_SCHEMA_INSTANCE_EDGE,
            instance.clone(),
            json!({
                "schema_id": schema.schema_id,
                "subgraph_id": instance,
            }),
        ))?;
    }
    Ok(())
}

pub fn project_transfer_hypotheses(
    source: &RelationalSubgraph,
    target: &RelationalSubgraph,
    mapping: &StructureMapping,
    schema: &SchemaRecord,
    base_confidence: f64,
    oracle: Option<&ThompsonAnalogyOracle>,
) -> GraphStoreResult<Vec<AnalogicalTransferHypothesis>> {
    let node_mapping = mapping
        .node_mappings
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    let relation_mapping = mapping
        .relation_mappings
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    let mut hypotheses = Vec::new();
    for source_edge in &source.edges {
        let Some(projected_from) = node_mapping.get(&source_edge.from_id) else {
            continue;
        };
        let Some(projected_to) = node_mapping.get(&source_edge.to_id) else {
            continue;
        };
        let projected_relation = relation_mapping
            .get(&source_edge.relation)
            .cloned()
            .unwrap_or_else(|| source_edge.relation.clone());
        if target.edges.iter().any(|target_edge| {
            target_edge.from_id == *projected_from
                && target_edge.to_id == *projected_to
                && target_edge.relation == projected_relation
        }) {
            continue;
        }
        let projected_edge = RelationalSignatureEdge {
            from_id: projected_from.clone(),
            relation: projected_relation,
            to_id: projected_to.clone(),
        };
        hypotheses.push(transfer_hypothesis(
            source,
            target,
            mapping,
            schema,
            source_edge,
            projected_edge,
            base_confidence,
            oracle,
        )?);
    }
    Ok(hypotheses)
}

pub fn downgrade_invalid_transfer(
    hypothesis: &AnalogicalTransferHypothesis,
    reason: impl Into<String>,
) -> AnalogicalTransferHypothesis {
    let mut downgraded = hypothesis.clone();
    downgraded.epistemic_status = "downgraded".to_string();
    downgraded.confidence = (downgraded.confidence * 0.4).min(0.19);
    downgraded.invalidated_by = Some(reason.into());
    downgraded
}

pub fn write_analogical_hypothesis_shadow<S: GraphStore>(
    store: &mut S,
    hypothesis: &AnalogicalTransferHypothesis,
) -> GraphStoreResult<()> {
    let computed_at = now_ms();
    store.upsert_node(NodeRecord::new(
        hypothesis.hypothesis_id.clone(),
        [ANALOGICAL_HYPOTHESIS_LABEL],
        json!({
            "hypothesis_id": hypothesis.hypothesis_id,
            "shadow_node_id": hypothesis.shadow_node_id,
            "schema_id": hypothesis.schema_id,
            "source_subgraph_id": hypothesis.source_subgraph_id,
            "target_subgraph_id": hypothesis.target_subgraph_id,
            "source_edge": hypothesis.source_edge,
            "projected_edge": hypothesis.projected_edge,
            "node_mappings": hypothesis.node_mappings,
            "relation_mappings": hypothesis.relation_mappings,
            "confidence": hypothesis.confidence,
            "epistemic_status": hypothesis.epistemic_status,
            "provenance": hypothesis.provenance,
            "invalidated_by": hypothesis.invalidated_by,
            "quarantine": true,
        }),
    ))?;
    store.upsert_node(NodeRecord::new(
        hypothesis.schema_id.clone(),
        [ANALOGICAL_SCHEMA_LABEL],
        json!({
            "schema_id": hypothesis.schema_id,
        }),
    ))?;
    store.upsert_node(NodeRecord::new(
        hypothesis.shadow_node_id.clone(),
        [EPISTEMIC_SHADOW_LABEL, ANALOGICAL_HYPOTHESIS_LABEL],
        json!({
            "content_node_id": hypothesis.hypothesis_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "schema_id": hypothesis.schema_id,
            "source_subgraph_id": hypothesis.source_subgraph_id,
            "target_subgraph_id": hypothesis.target_subgraph_id,
            "source_edge": hypothesis.source_edge,
            "projected_edge": hypothesis.projected_edge,
            "node_mappings": hypothesis.node_mappings,
            "relation_mappings": hypothesis.relation_mappings,
            "confidence": hypothesis.confidence,
            "epistemic_status": hypothesis.epistemic_status,
            "provenance": hypothesis.provenance,
            "invalidated_by": hypothesis.invalidated_by,
            "grounded_extension_status": "undecided",
            "support_in_degree": 0,
            "attack_in_degree": 0,
            "unsupported_leaf": true,
            "orphan": false,
            "bridge_score": 0.0,
            "contradiction_cycle_id": null,
            "predicted_edges": [{
                "target_content_id": hypothesis.projected_edge.to_id,
                "relation": hypothesis.projected_edge.relation,
                "confidence": hypothesis.confidence,
                "quarantine": true,
            }],
            "structural_role_vector": [],
            "source_kind": "structural",
            "engine": "rustyred-thg-core.analogical_transfer",
            "engine_version": DEFAULT_EPISTEMIC_ENGINE_VERSION,
            "computed_at": computed_at,
            "quarantine": true,
        }),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        has_epistemic_shadow_edge_id(&hypothesis.hypothesis_id, &hypothesis.shadow_node_id)?,
        hypothesis.hypothesis_id.clone(),
        HAS_EPISTEMIC_SHADOW,
        hypothesis.shadow_node_id.clone(),
        json!({
            "engine_version": DEFAULT_EPISTEMIC_ENGINE_VERSION,
            "computed_at": computed_at,
            "source": "analogical_transfer_v1",
        }),
    ))?;
    store.upsert_edge(
        EdgeRecord::new(
            analogical_schema_edge_id(&hypothesis.shadow_node_id, &hypothesis.schema_id)?,
            hypothesis.shadow_node_id.clone(),
            ANALOGICAL_USES_SCHEMA_EDGE,
            hypothesis.schema_id.clone(),
            json!({
                "schema_id": hypothesis.schema_id,
                "epistemic_status": hypothesis.epistemic_status,
                "quarantine": true,
            }),
        )
        .with_confidence(hypothesis.confidence)
        .with_epistemic_type(EpistemicType::Derives)
        .with_provenance(Provenance {
            source_id: Some(hypothesis.hypothesis_id.clone()),
            timestamp: None,
            method: Some("analogical_transfer_v1".to_string()),
        }),
    )?;
    Ok(())
}

fn transfer_hypothesis(
    source: &RelationalSubgraph,
    target: &RelationalSubgraph,
    mapping: &StructureMapping,
    schema: &SchemaRecord,
    source_edge: &RelationalSignatureEdge,
    projected_edge: RelationalSignatureEdge,
    base_confidence: f64,
    oracle: Option<&ThompsonAnalogyOracle>,
) -> GraphStoreResult<AnalogicalTransferHypothesis> {
    let hypothesis_id = analogical_hypothesis_id(
        &source.subgraph_id,
        &target.subgraph_id,
        &schema.schema_id,
        source_edge,
        &projected_edge,
    )?;
    let confidence = if let Some(o) = oracle {
        // Thompson sample from schema Beta posterior, clamped to (0.01, 0.99) so
        // no single sample hard-commits a hypothesis as certainly true or false
        // before the posterior has enough evidence.
        o.thompson_sample(&schema.schema_id, &hypothesis_id)
            .clamp(0.01, 0.49)
    } else {
        // Deterministic fallback: structural prior scaled by systematicity.
        // Clamped to a tighter quarantine ceiling (0.49) because the deterministic
        // path carries no posterior evidence and must never cross the 0.5 barrier.
        (base_confidence.clamp(0.0, 1.0) * mapping.systematicity_score.clamp(0.0, 1.0) * 0.5)
            .clamp(0.01, 0.49)
    };
    let shadow_node_id =
        epistemic_shadow_node_id(&hypothesis_id, DEFAULT_EPISTEMIC_ENGINE_VERSION)?;
    Ok(AnalogicalTransferHypothesis {
        hypothesis_id,
        shadow_node_id,
        schema_id: schema.schema_id.clone(),
        source_subgraph_id: source.subgraph_id.clone(),
        target_subgraph_id: target.subgraph_id.clone(),
        source_edge: source_edge.clone(),
        projected_edge,
        node_mappings: mapping.node_mappings.clone(),
        relation_mappings: mapping.relation_mappings.clone(),
        confidence,
        epistemic_status: "hypothesis".to_string(),
        provenance: vec![
            format!("source_subgraph:{}", source.subgraph_id),
            format!("target_subgraph:{}", target.subgraph_id),
            format!("schema:{}", schema.schema_id),
        ],
        invalidated_by: None,
    })
}

fn relation_overlap(left: &BTreeMap<String, usize>, right: &BTreeMap<String, usize>) -> f64 {
    let mut shared = 0usize;
    let mut total = 0usize;
    for relation in left.keys().chain(right.keys()).collect::<BTreeSet<_>>() {
        let l = left.get(relation).copied().unwrap_or(0);
        let r = right.get(relation).copied().unwrap_or(0);
        shared += l.min(r);
        total += l.max(r);
    }
    if total == 0 {
        0.0
    } else {
        shared as f64 / total as f64
    }
}

fn degree_overlap(left: &[(usize, usize)], right: &[(usize, usize)]) -> f64 {
    let left = left.iter().collect::<BTreeSet<_>>();
    let right = right.iter().collect::<BTreeSet<_>>();
    let shared = left.intersection(&right).count();
    let total = left.union(&right).count();
    if total == 0 {
        0.0
    } else {
        shared as f64 / total as f64
    }
}

fn surface_overlap(target: &RelationalSubgraph, candidate: &RelationalSubgraph) -> f64 {
    let target_terms = target
        .nodes
        .iter()
        .flat_map(|node| node.label.split_whitespace())
        .map(|term| term.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let candidate_terms = candidate
        .nodes
        .iter()
        .flat_map(|node| node.label.split_whitespace())
        .map(|term| term.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    if target_terms.is_empty() && candidate_terms.is_empty() {
        return 0.0;
    }
    let shared = target_terms.intersection(&candidate_terms).count();
    let total = target_terms.union(&candidate_terms).count();
    shared as f64 / total as f64
}

fn node_roles(subgraph: &RelationalSubgraph) -> BTreeMap<String, (usize, usize)> {
    let mut roles = BTreeMap::<String, (usize, usize)>::new();
    for node in &subgraph.nodes {
        roles.entry(node.node_id.clone()).or_default();
    }
    for edge in &subgraph.edges {
        roles.entry(edge.from_id.clone()).or_default().0 += 1;
        roles.entry(edge.to_id.clone()).or_default().1 += 1;
    }
    roles
}

fn mapped_relation_edges(
    source: &RelationalSubgraph,
    target: &RelationalSubgraph,
    node_mappings: &[(String, String)],
) -> usize {
    let mapping = node_mappings
        .iter()
        .cloned()
        .collect::<BTreeMap<String, String>>();
    source
        .edges
        .iter()
        .filter(|source_edge| {
            let Some(mapped_from) = mapping.get(&source_edge.from_id) else {
                return false;
            };
            let Some(mapped_to) = mapping.get(&source_edge.to_id) else {
                return false;
            };
            target.edges.iter().any(|target_edge| {
                &target_edge.from_id == mapped_from
                    && &target_edge.to_id == mapped_to
                    && target_edge.relation == source_edge.relation
            })
        })
        .count()
}

fn schema_id_for_signature(signature: &RelationalSignature) -> String {
    let hash = crate::symbolic::stable_hash_value(&serde_json::json!({
        "relation_counts": signature.relation_counts,
        "degree_pattern": signature.degree_pattern,
    }))
    .unwrap_or_else(|_| "unhashable".to_string());
    format!("schema:{hash}")
}

fn analogical_hypothesis_id(
    source_subgraph_id: &str,
    target_subgraph_id: &str,
    schema_id: &str,
    source_edge: &RelationalSignatureEdge,
    projected_edge: &RelationalSignatureEdge,
) -> GraphStoreResult<String> {
    Ok(format!(
        "analogy:hypothesis:{}",
        stable_hash(json!({
            "source_subgraph_id": source_subgraph_id,
            "target_subgraph_id": target_subgraph_id,
            "schema_id": schema_id,
            "source_edge": source_edge,
            "projected_edge": projected_edge,
        }))?
    ))
}

fn analogical_schema_edge_id(shadow_node_id: &str, schema_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "analogy:uses_schema:{}",
        stable_hash(json!([shadow_node_id, schema_id]))?
    ))
}

fn schema_instance_edge_id(schema_id: &str, instance_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "analogy:schema_instance:{}",
        stable_hash(json!([schema_id, instance_id]))?
    ))
}

fn is_connected(subgraph: &RelationalSubgraph) -> bool {
    if subgraph.nodes.is_empty() {
        return false;
    }
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for node in &subgraph.nodes {
        adjacency.entry(node.node_id.clone()).or_default();
    }
    for edge in &subgraph.edges {
        adjacency
            .entry(edge.from_id.clone())
            .or_default()
            .insert(edge.to_id.clone());
        adjacency
            .entry(edge.to_id.clone())
            .or_default()
            .insert(edge.from_id.clone());
    }
    let start = subgraph.nodes[0].node_id.clone();
    let mut seen = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            stack.extend(neighbors.iter().cloned());
        }
    }
    seen.len() == adjacency.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{read_epistemic_shadow, GroundedExtensionStatus};
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::query::NeighborQuery;
    use crate::ranking::{apply_cascade, QueryContext, RankingRule};

    fn node(node_id: &str, label: &str, domain: &str) -> RelationalSignatureNode {
        RelationalSignatureNode {
            node_id: node_id.to_string(),
            label: label.to_string(),
            domain: domain.to_string(),
        }
    }

    fn edge(from_id: &str, relation: &str, to_id: &str) -> RelationalSignatureEdge {
        RelationalSignatureEdge {
            from_id: from_id.to_string(),
            relation: relation.to_string(),
            to_id: to_id.to_string(),
        }
    }

    #[test]
    fn solar_system_retrieves_atom_by_structure_not_surface() {
        let solar = RelationalSubgraph {
            subgraph_id: "solar".to_string(),
            domain: "astronomy".to_string(),
            nodes: vec![
                node("sun", "sun", "astronomy"),
                node("earth", "earth", "astronomy"),
                node("mars", "mars", "astronomy"),
            ],
            edges: vec![
                edge("earth", "orbits", "sun"),
                edge("mars", "orbits", "sun"),
            ],
        };
        let atom = RelationalSubgraph {
            subgraph_id: "atom".to_string(),
            domain: "chemistry".to_string(),
            nodes: vec![
                node("nucleus", "nucleus", "chemistry"),
                node("electron-a", "electron", "chemistry"),
                node("electron-b", "electron", "chemistry"),
            ],
            edges: vec![
                edge("electron-a", "orbits", "nucleus"),
                edge("electron-b", "orbits", "nucleus"),
            ],
        };
        let surface_similar_wrong = RelationalSubgraph {
            subgraph_id: "planet_catalog".to_string(),
            domain: "catalog".to_string(),
            nodes: vec![
                node("sun-card", "sun", "catalog"),
                node("earth-card", "earth", "catalog"),
                node("mars-card", "mars", "catalog"),
            ],
            edges: vec![
                edge("sun-card", "listed_with", "earth-card"),
                edge("earth-card", "listed_with", "mars-card"),
            ],
        };

        let matches = retrieve_relational_analogies(&solar, &[atom, surface_similar_wrong], 0.75);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].subgraph_id, "atom");
        assert_eq!(matches[0].domain, "chemistry");
        assert!(
            matches[0].surface_overlap < 0.1,
            "far analogy should be low-surface but high-structure"
        );
    }

    #[test]
    fn alignment_prefers_connected_relational_mapping_and_schema_accretes() {
        let solar = RelationalSubgraph {
            subgraph_id: "solar".to_string(),
            domain: "astronomy".to_string(),
            nodes: vec![
                node("sun", "bright central body", "astronomy"),
                node("earth", "small orbiting body", "astronomy"),
                node("mars", "small orbiting body", "astronomy"),
            ],
            edges: vec![
                edge("earth", "orbits", "sun"),
                edge("mars", "orbits", "sun"),
            ],
        };
        let atom = RelationalSubgraph {
            subgraph_id: "atom".to_string(),
            domain: "chemistry".to_string(),
            nodes: vec![
                node("nucleus", "bright central body", "chemistry"),
                node("electron-a", "small orbiting body", "chemistry"),
                node("electron-b", "small orbiting body", "chemistry"),
            ],
            edges: vec![
                edge("electron-a", "orbits", "nucleus"),
                edge("electron-b", "orbits", "nucleus"),
            ],
        };
        let attribute_overlap = RelationalSubgraph {
            subgraph_id: "attribute-overlap".to_string(),
            domain: "catalog".to_string(),
            nodes: vec![
                node("bright", "bright central body", "catalog"),
                node("small", "small orbiting body", "catalog"),
            ],
            edges: vec![edge("bright", "described_with", "small")],
        };

        let structural = align_structure_mapping(&atom, &solar);
        let attributes = align_structure_mapping(&attribute_overlap, &solar);

        assert!(
            structural.systematicity_score > attributes.systematicity_score,
            "connected relation-consistent mapping should beat surface attributes"
        );
        assert_eq!(
            structural.relation_mappings,
            vec![("orbits".to_string(), "orbits".to_string())]
        );

        let schema = extract_schema(&structural, &atom);
        let mut registry = SchemaRegistry::default();
        let first = accrete_schema(&mut registry, schema);
        assert_eq!(first.instance_subgraph_ids.len(), 2);

        let lithium = RelationalSubgraph {
            subgraph_id: "lithium".to_string(),
            domain: "chemistry".to_string(),
            nodes: vec![
                node("li-nucleus", "nucleus", "chemistry"),
                node("li-electron-a", "electron", "chemistry"),
                node("li-electron-b", "electron", "chemistry"),
            ],
            edges: vec![
                edge("li-electron-a", "orbits", "li-nucleus"),
                edge("li-electron-b", "orbits", "li-nucleus"),
            ],
        };
        let lithium_mapping = align_structure_mapping(&lithium, &solar);
        let accreted = accrete_schema(&mut registry, extract_schema(&lithium_mapping, &lithium));
        assert_eq!(registry.schemas.len(), 1);
        assert!(accreted
            .instance_subgraph_ids
            .contains(&"lithium".to_string()));

        let mut store = InMemoryGraphStore::default();
        write_schema_record(&mut store, &accreted).expect("schema graph write");
        let schema_node = store.get_node(&accreted.schema_id).expect("schema node");
        assert!(schema_node
            .labels
            .contains(&ANALOGICAL_SCHEMA_LABEL.to_string()));
        let linked_instances = store
            .neighbors(
                NeighborQuery::out(&accreted.schema_id)
                    .with_edge_type(ANALOGICAL_SCHEMA_INSTANCE_EDGE),
            )
            .into_iter()
            .map(|hit| hit.node_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            linked_instances,
            accreted
                .instance_subgraph_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn transfer_creates_quarantined_hypothesis_shadow_and_downgrades_invalid_transfer() {
        let solar = RelationalSubgraph {
            subgraph_id: "solar".to_string(),
            domain: "astronomy".to_string(),
            nodes: vec![
                node("sun", "sun", "astronomy"),
                node("earth", "earth", "astronomy"),
                node("mars", "mars", "astronomy"),
            ],
            edges: vec![
                edge("earth", "orbits", "sun"),
                edge("mars", "orbits", "sun"),
            ],
        };
        let atom = RelationalSubgraph {
            subgraph_id: "atom".to_string(),
            domain: "chemistry".to_string(),
            nodes: vec![
                node("nucleus", "nucleus", "chemistry"),
                node("electron-a", "electron", "chemistry"),
                node("electron-b", "electron", "chemistry"),
            ],
            edges: vec![
                edge("electron-a", "orbits", "nucleus"),
                edge("electron-b", "orbits", "nucleus"),
            ],
        };
        let mapping = align_structure_mapping(&solar, &atom);
        let schema = extract_schema(&mapping, &solar);
        let transfer_source = RelationalSubgraph {
            edges: vec![
                edge("earth", "orbits", "sun"),
                edge("mars", "orbits", "sun"),
                edge("sun", "emits_to", "earth"),
            ],
            ..solar
        };

        let hypotheses =
            project_transfer_hypotheses(&transfer_source, &atom, &mapping, &schema, 0.8, None)
                .unwrap();

        assert_eq!(hypotheses.len(), 1);
        let hypothesis = &hypotheses[0];
        assert_eq!(hypothesis.epistemic_status, "hypothesis");
        assert!(hypothesis.confidence < 0.5);
        assert_eq!(hypothesis.projected_edge.from_id, "nucleus");
        assert_eq!(hypothesis.projected_edge.relation, "emits_to");
        assert_eq!(hypothesis.projected_edge.to_id, "electron-a");
        assert!(hypothesis
            .provenance
            .iter()
            .any(|item| item == "source_subgraph:solar"));
        assert_eq!(hypothesis.schema_id, schema.schema_id);

        let mut store = InMemoryGraphStore::default();
        write_analogical_hypothesis_shadow(&mut store, hypothesis).expect("write hypothesis");
        let shadow = store
            .get_node(&hypothesis.shadow_node_id)
            .expect("hypothesis shadow node");
        assert!(shadow.labels.contains(&EPISTEMIC_SHADOW_LABEL.to_string()));
        assert!(shadow
            .labels
            .contains(&ANALOGICAL_HYPOTHESIS_LABEL.to_string()));
        assert_eq!(shadow.properties["epistemic_status"], "hypothesis");
        assert_eq!(shadow.properties["quarantine"], true);
        assert_eq!(shadow.properties["schema_id"], hypothesis.schema_id);
        let readout =
            read_epistemic_shadow(&store, &hypothesis.hypothesis_id).expect("shadow readout");
        assert_eq!(
            readout.grounded_extension_status,
            GroundedExtensionStatus::Undecided
        );
        assert_eq!(readout.predicted_edges.len(), 1);
        assert!(readout.predicted_edges[0].quarantine);
        assert_eq!(readout.predicted_edges[0].relation, "emits_to");

        let mut query = QueryContext::from_query("emits");
        query.epistemic.allow_provisional = true;
        let accepted = apply_cascade(
            vec![hypothesis.rank_candidate()],
            &[RankingRule::EpistemicStatus],
            &query,
        );
        assert_eq!(
            accepted.ranked.len(),
            1,
            "ranking admits hypothesis under the provisional gate"
        );

        let downgraded =
            downgrade_invalid_transfer(hypothesis, "target evidence contradicted relation");
        let original_bucket = RankingRule::EpistemicStatus
            .bucket(&hypothesis.rank_candidate(), &query)
            .expect("original bucket");
        let downgraded_bucket = RankingRule::EpistemicStatus
            .bucket(&downgraded.rank_candidate(), &query)
            .expect("downgraded bucket");
        assert!(
            downgraded_bucket > original_bucket,
            "invalid transfer should remain visible but lower-ranked"
        );
        assert_eq!(
            downgraded.invalidated_by.as_deref(),
            Some("target evidence contradicted relation")
        );
    }
}
