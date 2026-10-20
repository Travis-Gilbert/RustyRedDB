//! D6: graph-analytic and reachability operators as first-class plan nodes.
//!
//! A hybrid query plan composes three first-class nodes -- a boolean prefilter
//! (a candidate node set, e.g. the result of a relational predicate), a graph
//! step evaluated as matrix multiply (reachability expansion, or a LAGraph
//! analytic like PageRank), and a bounded rerank -- into one pipeline that hands
//! a small, ranked candidate set to downstream vector/graph reranking. This is
//! the planner seam the handoff describes: intersect a boolean prefilter, run a
//! reachability step as matrix multiply, hand a bounded candidate set onward.
//!
//! Feature-gated behind `graphblas`.

use std::collections::BTreeMap;

use rustyred_thg_graphblas::{GrbError, LaGraph};

use crate::graphblas_adjacency::TypedAdjacency;

/// The graph step of a hybrid plan: how the candidate set is expanded/scored.
#[derive(Clone, Debug)]
pub enum GraphStep {
    /// Reachability expansion over an edge type, scored by how many prefilter
    /// seeds reach each node (a repeated masked matrix-vector multiply).
    Reachability { edge_type: String },
    /// A LAGraph analytic (PageRank) over an edge type, scoring the prefiltered
    /// candidates by their global metric.
    PageRank {
        edge_type: String,
        damping: f64,
        tol: f64,
        itermax: i32,
    },
}

/// A first-class hybrid plan node: prefilter -> graph step -> bounded rerank.
#[derive(Clone, Debug)]
pub struct HybridGraphPlan {
    /// Boolean prefilter result: the candidate node ids to start from / rank.
    pub prefilter: Vec<String>,
    pub step: GraphStep,
    /// Keep the top-k by score (the bound handed to downstream reranking).
    pub top_k: usize,
}

/// Trace of the executed plan, mirroring the relational planner's `PlanTrace`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HybridGraphTrace {
    pub prefilter_size: usize,
    pub candidate_set_size: usize,
    pub returned: usize,
    pub step: String,
}

/// Result of a hybrid plan: a bounded, score-ranked candidate set plus a trace.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridGraphResult {
    pub ranked: Vec<(String, f64)>,
    pub trace: HybridGraphTrace,
}

fn rank_and_bound(mut scored: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> {
    // Descending score; node id as a deterministic tiebreak.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored
}

/// Execute a hybrid graph plan over the typed adjacency.
pub fn execute(
    adj: &TypedAdjacency,
    plan: &HybridGraphPlan,
) -> Result<HybridGraphResult, GrbError> {
    let prefilter_size = plan.prefilter.len();

    let (scored, step_label): (Vec<(String, f64)>, String) = match &plan.step {
        GraphStep::Reachability { edge_type } => {
            // score(node) = number of prefilter seeds that reach it (inclusive),
            // via the reachability-semiring traversal (repeated masked mxv).
            let mut counts: BTreeMap<String, f64> = BTreeMap::new();
            for seed in &plan.prefilter {
                for reached in adj.reachable_ids(edge_type, &[seed.as_str()])? {
                    *counts.entry(reached).or_insert(0.0) += 1.0;
                }
            }
            (
                counts.into_iter().collect(),
                format!("reachability({edge_type})"),
            )
        }
        GraphStep::PageRank {
            edge_type,
            damping,
            tol,
            itermax,
        } => {
            let scored = match adj.matrix(edge_type) {
                Some(m) => {
                    let mut g = LaGraph::directed(m.dup()?)?;
                    let pr = g.pagerank(*damping, *tol, *itermax)?;
                    plan.prefilter
                        .iter()
                        .filter_map(|id| {
                            adj.index_of(id)
                                .and_then(|i| pr.get(i as usize).copied())
                                .map(|score| (id.clone(), score))
                        })
                        .collect()
                }
                None => Vec::new(),
            };
            (scored, format!("pagerank({edge_type})"))
        }
    };

    let candidate_set_size = scored.len();
    let ranked = rank_and_bound(scored, plan.top_k);
    let returned = ranked.len();
    Ok(HybridGraphResult {
        ranked,
        trace: HybridGraphTrace {
            prefilter_size,
            candidate_set_size,
            returned,
            step: step_label,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::record::EdgeRecord;
    use serde_json::json;

    fn edge(id: &str, from: &str, to: &str) -> EdgeRecord {
        EdgeRecord::new(id, from, "LINK", to, json!({}))
    }

    // D6: prefilter -> reachability (matrix multiply) -> bounded rerank.
    // a->b->c->d and x->c. Seeds {a, x}; node c is reached by both, so it
    // ranks first; top_k=1 bounds the candidate set to {c}.
    #[test]
    fn hybrid_reachability_plan_bounds_and_ranks() {
        let records = vec![
            edge("e1", "a", "b"),
            edge("e2", "b", "c"),
            edge("e3", "c", "d"),
            edge("e4", "x", "c"),
        ];
        let adj = TypedAdjacency::from_records(&records).unwrap();
        let plan = HybridGraphPlan {
            prefilter: vec!["a".into(), "x".into()],
            step: GraphStep::Reachability {
                edge_type: "LINK".into(),
            },
            top_k: 1,
        };
        let result = execute(&adj, &plan).unwrap();
        assert_eq!(result.ranked, vec![("c".to_string(), 2.0)]);
        assert_eq!(result.trace.prefilter_size, 2);
        assert_eq!(result.trace.candidate_set_size, 5); // a,b,c,d,x
        assert_eq!(result.trace.returned, 1);
    }

    // D6 + D4: a LAGraph analytic (PageRank) as a plan node. a->c, b->c: c is
    // the hub and ranks highest among the prefiltered candidates.
    #[test]
    fn hybrid_pagerank_plan_ranks_hub() {
        let records = vec![edge("e1", "a", "c"), edge("e2", "b", "c")];
        let adj = TypedAdjacency::from_records(&records).unwrap();
        let plan = HybridGraphPlan {
            prefilter: vec!["a".into(), "b".into(), "c".into()],
            step: GraphStep::PageRank {
                edge_type: "LINK".into(),
                damping: 0.85,
                tol: 1e-4,
                itermax: 100,
            },
            top_k: 1,
        };
        let result = execute(&adj, &plan).unwrap();
        assert_eq!(result.ranked.len(), 1);
        assert_eq!(result.ranked[0].0, "c");
    }
}
