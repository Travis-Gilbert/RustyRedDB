//! Derivation trace (why explanation) over a derivation graph.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::store::{
    DerivationStore, ASSUMPTION_LABEL, DERIVATION_ANCESTOR_EDGE, DERIVATION_ASSUMES_EDGE,
    DERIVATION_LABEL,
};
use crate::{AlgebraicClass, DualFoldError, EdgeDirection, EdgeOntology};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhyNode {
    pub node_id: String,
    pub is_assumption: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supports: Vec<WhyNode>,
}

impl WhyNode {
    pub fn assumption(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            is_assumption: true,
            supports: Vec::new(),
        }
    }

    pub fn derivation(node_id: impl Into<String>, supports: Vec<WhyNode>) -> Self {
        Self {
            node_id: node_id.into(),
            is_assumption: false,
            supports,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhyTrace {
    pub root: WhyNode,
    pub total_supporters: usize,
    pub assumption_count: usize,
}

/// One examined edge in a code-fact explanation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CodeWhyEdge {
    pub edge_id: String,
    pub canonical_type: String,
    pub class: AlgebraicClass,
    pub direction: EdgeDirection,
}

/// Direct provenance for a materialized `CodeGraphFact`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CodeWhyTrace {
    pub fact_id: String,
    pub examined: Vec<CodeWhyEdge>,
}

/// Explain the exact edge set stored by a code fact.
///
/// The fact owns the examined ids; the store remains authoritative for their
/// types. Every type is resolved through the same ontology used to fold the
/// fact, so an unknown edge refuses rather than acquiring an invented class.
pub fn why_code_fact<'a>(
    store: &impl DerivationStore,
    fact_id: impl Into<String>,
    examined: impl IntoIterator<Item = &'a str>,
    ontology: &EdgeOntology,
) -> Result<CodeWhyTrace, DualFoldError> {
    let mut edge_ids: Vec<String> = examined.into_iter().map(str::to_string).collect();
    edge_ids.sort();
    edge_ids.dedup();
    let mut edges = Vec::with_capacity(edge_ids.len());
    for edge_id in edge_ids {
        let edge_type = store
            .edge_type(&edge_id)
            .ok_or_else(|| DualFoldError::UntypedEdge {
                edge_id: edge_id.clone(),
            })?;
        let spec = ontology
            .resolve(&edge_type)
            .ok_or_else(|| DualFoldError::UnclassifiedEdge {
                edge_id: edge_id.clone(),
                edge_type: edge_type.clone(),
            })?;
        edges.push(CodeWhyEdge {
            edge_id,
            canonical_type: spec.canonical.clone(),
            class: spec.class,
            direction: spec.direction,
        });
    }
    Ok(CodeWhyTrace {
        fact_id: fact_id.into(),
        examined: edges,
    })
}

pub fn why(store: &impl DerivationStore, node_id: &str) -> Option<WhyTrace> {
    let labels = store.node_labels(node_id)?;
    let is_derivation = labels.iter().any(|l| l == DERIVATION_LABEL);
    let is_assumption = labels.iter().any(|l| l == ASSUMPTION_LABEL);
    if !is_derivation && !is_assumption {
        return None;
    }
    let mut visited = HashSet::new();
    let root = trace_node(store, node_id, &mut visited);
    let (total, assumptions) = count_nodes(&root);
    Some(WhyTrace {
        root,
        total_supporters: total.saturating_sub(1),
        assumption_count: assumptions,
    })
}

fn trace_node(
    store: &impl DerivationStore,
    node_id: &str,
    visited: &mut HashSet<String>,
) -> WhyNode {
    if !visited.insert(node_id.to_string()) {
        return WhyNode::assumption(format!("{node_id} <cycle>"));
    }
    let is_assumption = store
        .node_labels(node_id)
        .is_some_and(|labels| labels.iter().any(|l| l == ASSUMPTION_LABEL));
    if is_assumption {
        return WhyNode::assumption(node_id);
    }
    let mut supporters = Vec::new();
    for hit in store.neighbors_in(node_id, DERIVATION_ANCESTOR_EDGE) {
        supporters.push(trace_node(store, &hit.node_id, visited));
    }
    for hit in store.neighbors_in(node_id, DERIVATION_ASSUMES_EDGE) {
        supporters.push(trace_node(store, &hit.node_id, visited));
    }
    if supporters.is_empty() {
        WhyNode::assumption(node_id)
    } else {
        WhyNode::derivation(node_id, supporters)
    }
}

fn count_nodes(node: &WhyNode) -> (usize, usize) {
    let mut total = 1usize;
    let mut assumptions = if node.is_assumption { 1 } else { 0 };
    for child in &node.supports {
        let (ct, ca) = count_nodes(child);
        total += ct;
        assumptions += ca;
    }
    (total, assumptions)
}
