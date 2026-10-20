//! The standing analogy pass (IDEA-SHAPE-SURFACE A2).
//!
//! Composes motif extraction ([`crate::motif`]) with the analogy engine
//! ([`crate::analogy`]) into one idempotent pass over a tenant graph: extract
//! candidate motifs, retrieve cross-domain structural analogies, align the
//! matched pairs, accrete a schema, and project quarantined transfer
//! hypotheses, persisting through the engine's existing writers. Every persisted
//! id is content-derived, so re-running on unchanged input creates no
//! duplicates. The pass returns a receipt of what it saw; it never mutates a
//! confidence and never reads structure back from a renderer.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::analogy::{
    accrete_schema, align_structure_mapping, extract_schema, project_transfer_hypotheses,
    retrieve_relational_analogies, write_analogical_hypothesis_shadow, write_schema_record,
    RelationalSubgraph, SchemaRegistry, StructureMapping, ANALOGICAL_SCHEMA_INSTANCE_LABEL,
};
use crate::analogy_thompson::ThompsonAnalogyOracle;
use crate::graph::capability::GraphStore;
use crate::graph::error::GraphStoreResult;
use crate::graph::record::NodeRecord;
use crate::motif::{extract_motifs, MotifConfig};
use crate::state::stable_hash;

/// Minimum structural score for a retrieved cross-domain pair to be aligned.
pub const ANALOGY_PASS_MIN_STRUCTURAL_SCORE: f64 = 0.75;
/// Base confidence handed to hypothesis projection. The engine scales this below
/// the 0.5 quarantine ceiling; it is a floor on the input, not the output.
pub const ANALOGY_PASS_BASE_CONFIDENCE: f64 = 0.5;
/// Graph label for a persisted structure mapping between two motif instances, so
/// the analogy surface can render node and relation correspondences and
/// systematicity without recomputing them.
pub const ANALOGICAL_MAPPING_LABEL: &str = "AnalogicalStructureMapping";

/// Configuration for one analogy pass. Defaults are the named `ANALOGY_PASS_*`
/// constants over a default [`MotifConfig`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalogyPassConfig {
    pub motif: MotifConfig,
    pub min_structural_score: f64,
    pub base_confidence: f64,
}

impl Default for AnalogyPassConfig {
    fn default() -> Self {
        Self {
            motif: MotifConfig::default(),
            min_structural_score: ANALOGY_PASS_MIN_STRUCTURAL_SCORE,
            base_confidence: ANALOGY_PASS_BASE_CONFIDENCE,
        }
    }
}

/// What one pass observed and persisted (A2's per-run receipt).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalogyPassReceipt {
    pub subgraphs: usize,
    pub matches: usize,
    pub schemas: usize,
    pub hypotheses: usize,
}

/// Run the standing analogy pass over the live graph (A2). Extracts motifs,
/// retrieves cross-domain analogies above the configured threshold, aligns each
/// matched pair once (canonical source = lexicographically smaller motif id),
/// accretes a schema, and projects quarantined transfer hypotheses, persisting
/// through the engine's writers. Returns a receipt of the counts.
pub fn run_analogy_pass<S: GraphStore>(
    store: &mut S,
    config: &AnalogyPassConfig,
    oracle: Option<&ThompsonAnalogyOracle>,
) -> GraphStoreResult<AnalogyPassReceipt> {
    let subgraphs = extract_motifs(store, &config.motif)?
        .into_iter()
        .map(|motif| motif.subgraph)
        .collect::<Vec<_>>();

    let mut registry = SchemaRegistry::default();
    let mut matches = 0usize;
    let mut hypotheses = 0usize;

    for source in &subgraphs {
        for hit in retrieve_relational_analogies(source, &subgraphs, config.min_structural_score) {
            // Each cross-domain matched pair is processed once, from the
            // lexicographically smaller motif id, so the schema and hypotheses
            // are stable regardless of iteration order.
            if source.subgraph_id >= hit.subgraph_id {
                continue;
            }
            let Some(target) = subgraphs
                .iter()
                .find(|candidate| candidate.subgraph_id == hit.subgraph_id)
            else {
                continue;
            };
            matches += 1;
            let mapping = align_structure_mapping(source, target);
            let schema = accrete_schema(&mut registry, extract_schema(&mapping, source));
            // Project hypotheses first: write_analogical_hypothesis_shadow re-upserts
            // a minimal schema node, so writing the full schema record last keeps its
            // relation counts, degree pattern, and instance ids intact.
            for hypothesis in project_transfer_hypotheses(
                source,
                target,
                &mapping,
                &schema,
                config.base_confidence,
                oracle,
            )? {
                write_analogical_hypothesis_shadow(store, &hypothesis)?;
                hypotheses += 1;
            }
            write_schema_record(store, &schema)?;
            // Persist the motif structure and the mapping so the analogy surface can
            // render instances (nodes, edges, domain) and mappings (systematicity),
            // not just references. Enrichment runs after write_schema_record.
            persist_motif_instance(store, source)?;
            persist_motif_instance(store, target)?;
            write_structure_mapping(store, &mapping)?;
        }
    }

    Ok(AnalogyPassReceipt {
        subgraphs: subgraphs.len(),
        matches,
        schemas: registry.schemas.len(),
        hypotheses,
    })
}

/// Enrich a motif's `AnalogicalSchemaInstance` node with its full structure
/// (domain, nodes, edges) so the analogy surface can render it. Idempotent: the
/// node id is the motif's content-derived `subgraph_id`. Schema membership stays
/// on the `ANALOGICAL_SCHEMA_INSTANCE` edge, not this node.
fn persist_motif_instance<S: GraphStore>(
    store: &mut S,
    subgraph: &RelationalSubgraph,
) -> GraphStoreResult<()> {
    store.upsert_node(NodeRecord::new(
        subgraph.subgraph_id.clone(),
        [ANALOGICAL_SCHEMA_INSTANCE_LABEL],
        json!({
            "subgraph_id": subgraph.subgraph_id,
            "domain": subgraph.domain,
            "nodes": subgraph.nodes,
            "edges": subgraph.edges,
        }),
    ))?;
    Ok(())
}

/// Persist a structure mapping as a node so the analogy surface can render node
/// and relation correspondences and systematicity. Idempotent: the id is derived
/// from the source and target subgraph ids.
fn write_structure_mapping<S: GraphStore>(
    store: &mut S,
    mapping: &StructureMapping,
) -> GraphStoreResult<()> {
    let mapping_id = format!(
        "analogy-mapping:{}",
        stable_hash((&mapping.source_subgraph_id, &mapping.target_subgraph_id))?
    );
    store.upsert_node(NodeRecord::new(
        mapping_id,
        [ANALOGICAL_MAPPING_LABEL],
        json!({
            "source_subgraph_id": mapping.source_subgraph_id,
            "target_subgraph_id": mapping.target_subgraph_id,
            "node_mappings": mapping.node_mappings,
            "relation_mappings": mapping.relation_mappings,
            "systematicity_score": mapping.systematicity_score,
        }),
    ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analogy::{
        ANALOGICAL_HYPOTHESIS_LABEL, ANALOGICAL_SCHEMA_INSTANCE_LABEL, ANALOGICAL_SCHEMA_LABEL,
    };
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::query::NodeQuery;
    use crate::graph::record::{EdgeRecord, NodeRecord};
    use serde_json::json;

    fn concept(store: &mut InMemoryGraphStore, id: &str, domain: &str) {
        store
            .upsert_node(NodeRecord::new(
                id,
                ["Concept"],
                json!({ "label": id, "domain": domain }),
            ))
            .unwrap();
    }

    fn relation(store: &mut InMemoryGraphStore, id: &str, from: &str, rel: &str, to: &str) {
        store
            .upsert_edge(EdgeRecord::new(id, from, rel, to, json!({})))
            .unwrap();
    }

    /// Two cross-domain motifs sharing a skeleton but differing on one relation
    /// label, so a projected edge is genuinely novel in the target (a transfer).
    fn seeded_store() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        // astronomy: earth->sun, mars->sun (both "orbits").
        concept(&mut store, "sun", "astronomy");
        concept(&mut store, "earth", "astronomy");
        concept(&mut store, "mars", "astronomy");
        relation(&mut store, "a1", "earth", "orbits", "sun");
        relation(&mut store, "a2", "mars", "orbits", "sun");
        // chemistry: ea->nucleus ("binds"), eb->nucleus ("orbits"). Same skeleton,
        // one different relation, so projecting "orbits" onto the ea position is
        // novel here.
        concept(&mut store, "nucleus", "chemistry");
        concept(&mut store, "electron-a", "chemistry");
        concept(&mut store, "electron-b", "chemistry");
        relation(&mut store, "c1", "electron-a", "binds", "nucleus");
        relation(&mut store, "c2", "electron-b", "orbits", "nucleus");
        store
    }

    fn pass_config() -> AnalogyPassConfig {
        AnalogyPassConfig {
            motif: MotifConfig {
                admitted_relations: vec!["orbits".to_string(), "binds".to_string()],
                ..MotifConfig::default()
            },
            // The transfer pair is structurally close but not identical, so relax
            // the threshold below the symmetric default for the fixture.
            min_structural_score: 0.5,
            ..AnalogyPassConfig::default()
        }
    }

    fn count_label(store: &InMemoryGraphStore, label: &str) -> usize {
        store
            .query_nodes(NodeQuery {
                label: Some(label.to_string()),
                properties: Default::default(),
                limit: None,
                include_expired: false,
            })
            .len()
    }

    #[test]
    fn pass_persists_schema_instances_and_a_quarantined_hypothesis() {
        let mut store = seeded_store();
        let receipt = run_analogy_pass(&mut store, &pass_config(), None).unwrap();

        assert!(receipt.subgraphs >= 2, "both motifs extracted");
        assert!(receipt.matches >= 1, "cross-domain match found");
        assert!(receipt.schemas >= 1, "schema accreted");
        assert!(receipt.hypotheses >= 1, "at least one transfer hypothesis");

        assert!(count_label(&store, ANALOGICAL_SCHEMA_LABEL) >= 1);
        assert!(count_label(&store, ANALOGICAL_SCHEMA_INSTANCE_LABEL) >= 2);

        let hypotheses = store.query_nodes(NodeQuery {
            label: Some(ANALOGICAL_HYPOTHESIS_LABEL.to_string()),
            properties: Default::default(),
            limit: None,
            include_expired: false,
        });
        assert!(
            hypotheses
                .iter()
                .any(|node| node.properties.get("quarantine") == Some(&json!(true))),
            "a quarantined hypothesis shadow with a projected edge is persisted"
        );

        // Instances carry their motif structure (A3 renders nodes/edges/domain).
        let instances = store.query_nodes(NodeQuery {
            label: Some(ANALOGICAL_SCHEMA_INSTANCE_LABEL.to_string()),
            properties: Default::default(),
            limit: None,
            include_expired: false,
        });
        assert!(
            instances.iter().any(|node| {
                node.properties
                    .get("edges")
                    .and_then(|edges| edges.as_array())
                    .is_some_and(|edges| !edges.is_empty())
                    && node
                        .properties
                        .get("domain")
                        .and_then(|d| d.as_str())
                        .is_some()
            }),
            "instance nodes carry their motif structure (domain, nodes, edges)"
        );

        // The structure mapping is persisted (A3 renders systematicity).
        assert!(
            count_label(&store, ANALOGICAL_MAPPING_LABEL) >= 1,
            "the structure mapping is persisted"
        );

        // The full schema record survives the hypothesis shadow write.
        let schemas = store.query_nodes(NodeQuery {
            label: Some(ANALOGICAL_SCHEMA_LABEL.to_string()),
            properties: Default::default(),
            limit: None,
            include_expired: false,
        });
        assert!(
            schemas.iter().any(|node| node
                .properties
                .get("relation_counts")
                .and_then(|counts| counts.as_object())
                .is_some_and(|counts| !counts.is_empty())),
            "the full schema record survives (relation counts not clobbered)"
        );
    }

    #[test]
    fn re_running_creates_no_duplicates() {
        let mut store = seeded_store();
        let config = pass_config();
        let first = run_analogy_pass(&mut store, &config, None).unwrap();

        let schema_nodes = count_label(&store, ANALOGICAL_SCHEMA_LABEL);
        let instance_nodes = count_label(&store, ANALOGICAL_SCHEMA_INSTANCE_LABEL);
        let hypothesis_nodes = count_label(&store, ANALOGICAL_HYPOTHESIS_LABEL);
        let mapping_nodes = count_label(&store, ANALOGICAL_MAPPING_LABEL);

        let second = run_analogy_pass(&mut store, &config, None).unwrap();

        // Stable ids: the second pass upserts the same nodes, never new ones.
        assert_eq!(schema_nodes, count_label(&store, ANALOGICAL_SCHEMA_LABEL));
        assert_eq!(
            instance_nodes,
            count_label(&store, ANALOGICAL_SCHEMA_INSTANCE_LABEL)
        );
        assert_eq!(
            hypothesis_nodes,
            count_label(&store, ANALOGICAL_HYPOTHESIS_LABEL)
        );
        assert_eq!(mapping_nodes, count_label(&store, ANALOGICAL_MAPPING_LABEL));
        assert_eq!(first, second, "the receipt is stable across identical runs");
    }
}
