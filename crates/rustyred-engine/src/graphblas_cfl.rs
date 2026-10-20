//! D5: CFL-reachability dataflow (points-to / taint) as matrix multiplication.
//!
//! Interprocedural dataflow facts are computed as context-free-language
//! reachability: edge labels are terminals, derived relations are nonterminals,
//! and grammar productions become boolean matrix operations over the typed
//! adjacency matrices (D2), iterated to a fixpoint. The classic encoding (a
//! normalized grammar of epsilon / unary / binary productions):
//!
//! * `A -> e`    : `M_A |= I`            (identity)
//! * `A -> B`    : `M_A |= M_B`
//! * `A -> B C`  : `M_A |= M_B (lor.land) M_C`   (boolean matrix multiply)
//!
//! Results (e.g. a points-to relation) are written back as derived edges with
//! provenance, feeding the same authority layers as the rest of the engine.
//!
//! Feature-gated behind `graphblas`.

use std::collections::{BTreeMap, BTreeSet};

use rustyred_thg_graphblas::{
    matrix_or_assign, mxm_into, Descriptor, ElementType, GrbError, Matrix, Semiring,
};
use serde_json::json;

use crate::graph::capability::GraphStore;
use crate::graph::error::GraphStoreResult;
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graphblas_adjacency::TypedAdjacency;

/// A normalized context-free production over edge-label terminals and relation
/// nonterminals. Symbol names are strings; terminals are the names of edge
/// types present in the [`TypedAdjacency`].
#[derive(Clone, Debug)]
pub enum Production {
    /// `A -> e` (epsilon: the identity / reflexive relation).
    Epsilon(String),
    /// `A -> B`.
    Unary(String, String),
    /// `A -> B C`.
    Binary(String, String, String),
}

/// A context-free grammar driving CFL-reachability.
#[derive(Clone, Debug, Default)]
pub struct CflGrammar {
    pub productions: Vec<Production>,
}

impl CflGrammar {
    pub fn new(productions: Vec<Production>) -> Self {
        Self { productions }
    }
}

fn total_nvals(mats: &BTreeMap<String, Matrix>) -> Result<u64, GrbError> {
    let mut t = 0;
    for m in mats.values() {
        t += m.nvals()?;
    }
    Ok(t)
}

/// Solve CFL-reachability over `adj` for `grammar`, returning the relation
/// matrix for every symbol. Terminal matrices are seeded from the typed
/// adjacency (by edge label); nonterminal matrices start empty and grow to a
/// fixpoint via boolean matrix multiply + union. Monotone, so the total entry
/// count strictly increases until convergence.
pub fn solve(
    adj: &TypedAdjacency,
    grammar: &CflGrammar,
) -> Result<BTreeMap<String, Matrix>, GrbError> {
    let n = adj.node_count().max(1);

    // Every symbol referenced anywhere in the grammar gets a matrix.
    let mut symbols: BTreeSet<&str> = BTreeSet::new();
    for p in &grammar.productions {
        match p {
            Production::Epsilon(a) => {
                symbols.insert(a);
            }
            Production::Unary(a, b) => {
                symbols.insert(a);
                symbols.insert(b);
            }
            Production::Binary(a, b, c) => {
                symbols.insert(a);
                symbols.insert(b);
                symbols.insert(c);
            }
        }
    }

    let mut mats: BTreeMap<String, Matrix> = BTreeMap::new();
    for s in &symbols {
        // Terminal -> seed from the adjacency; nonterminal -> empty.
        let m = match adj.matrix(s) {
            Some(t) => t.dup()?,
            None => Matrix::new(ElementType::Bool, n, n)?,
        };
        mats.insert((*s).to_string(), m);
    }

    let semi = Semiring::reachability_bool();
    let none = Descriptor::none();

    loop {
        let before = total_nvals(&mats)?;
        for p in &grammar.productions {
            match p {
                Production::Epsilon(a) => {
                    let id = Matrix::identity_bool(n)?;
                    matrix_or_assign(mats.get_mut(a).expect("symbol seeded"), &id)?;
                }
                Production::Unary(a, b) => {
                    let src = mats[b].dup()?;
                    matrix_or_assign(mats.get_mut(a).expect("symbol seeded"), &src)?;
                }
                Production::Binary(a, b, c) => {
                    // t = M_B (lor.land) M_C, into a fresh matrix so the borrows
                    // of B and C end before A is taken mutably (handles A == B/C).
                    let mut t = Matrix::new(ElementType::Bool, n, n)?;
                    mxm_into(&mut t, None, &semi, &mats[b], &mats[c], &none)?;
                    matrix_or_assign(mats.get_mut(a).expect("symbol seeded"), &t)?;
                }
            }
        }
        if total_nvals(&mats)? == before {
            break;
        }
    }
    Ok(mats)
}

/// Write a derived relation back to a graph store as edges carrying provenance
/// (`derived_by = "cfl-reachability"`, plus the relation name). Returns the
/// number of edges written. Idempotent on `(relation, from, to)`.
pub fn write_back<S: GraphStore>(
    store: &mut S,
    pairs: &[(String, String)],
    label: &str,
    relation: &str,
) -> GraphStoreResult<usize> {
    let mut written = 0usize;
    for (from, to) in pairs {
        // Ensure endpoints exist (a store may reject dangling edges). Existing
        // nodes are left untouched so we never clobber real labels/properties.
        for ep in [from, to] {
            if store.get_node(ep).is_none() {
                store.upsert_node(NodeRecord::new(ep.clone(), ["CflNode"], json!({})))?;
            }
        }
        let id = format!("cfl:{relation}:{from}->{to}");
        let rec = EdgeRecord::new(
            id,
            from.clone(),
            label,
            to.clone(),
            json!({ "derived_by": "cfl-reachability", "relation": relation }),
        );
        store.upsert_edge(rec)?;
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::in_memory::InMemoryGraphStore;

    fn edge(id: &str, from: &str, ty: &str, to: &str) -> EdgeRecord {
        EdgeRecord::new(id, from, ty, to, json!({}))
    }

    /// Field-insensitive points-to: PointsTo -> new | assign PointsTo.
    fn points_to_grammar() -> CflGrammar {
        CflGrammar::new(vec![
            Production::Unary("PointsTo".into(), "new".into()),
            Production::Binary("PointsTo".into(), "assign".into(), "PointsTo".into()),
        ])
    }

    // ACCEPTANCE (D5): correct points-to on a fixture program with a known
    // answer. Program: p = new o1; q = p; r = q. So p, q, r all point to o1.
    #[test]
    fn points_to_known_answer() {
        let records = vec![
            edge("e1", "p", "new", "o1"),
            edge("e2", "q", "assign", "p"),
            edge("e3", "r", "assign", "q"),
        ];
        let adj = TypedAdjacency::from_records(&records).unwrap();
        let mats = solve(&adj, &points_to_grammar()).unwrap();
        let pt = mats.get("PointsTo").unwrap();
        let pairs = adj.pairs_for(pt).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("p".to_string(), "o1".to_string()),
                ("q".to_string(), "o1".to_string()),
                ("r".to_string(), "o1".to_string()),
            ]
        );
    }

    // Derived facts are written back to the store with provenance.
    #[test]
    fn derived_points_to_written_back() {
        let records = vec![edge("e1", "p", "new", "o1"), edge("e2", "q", "assign", "p")];
        let adj = TypedAdjacency::from_records(&records).unwrap();
        let mats = solve(&adj, &points_to_grammar()).unwrap();
        let pairs = adj.pairs_for(mats.get("PointsTo").unwrap()).unwrap();

        let mut store = InMemoryGraphStore::default();
        let wrote = write_back(&mut store, &pairs, "POINTS_TO", "PointsTo").unwrap();
        assert_eq!(wrote, 2); // p->o1, q->o1
        assert!(crate::GraphRead::get_edge(&store, "cfl:PointsTo:q->o1").is_some());
    }
}
