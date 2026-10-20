//! D2: typed adjacency-matrix layer over [`crate::graph_csr`].
//!
//! Each relationship type maps to its own sparse boolean GraphBLAS matrix,
//! sharing one growable node-id <-> dense-index map. Matrices update
//! incrementally as the graph mutates (driven by [`crate::hooks`]), and the
//! union of the per-type matrices is required to agree structurally with the
//! [`CsrGraph`] analytics lens built from the same edges.
//!
//! Feature-gated behind `graphblas`: pulls in the native `rustyred-thg-graphblas`
//! crate (SuiteSparse:GraphBLAS + LAGraph).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use rustyred_thg_graphblas::{ElementType, GrbError, Matrix};

use crate::graph::record::EdgeRecord;
use crate::hooks::{
    coalesce_per_id, HookError, HookHandler, HookOutcome, HookRegistration, MutationKind,
    MutationMatcher,
};

type AdjResult<T> = Result<T, GrbError>;

/// A growable bidirectional node-id <-> dense matrix-index map.
#[derive(Default)]
pub struct NodeIndex {
    ids: Vec<String>,
    index: HashMap<String, u64>,
}

impl NodeIndex {
    pub fn len(&self) -> u64 {
        self.ids.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn index_of(&self, id: &str) -> Option<u64> {
        self.index.get(id).copied()
    }

    pub fn id_of(&self, i: u64) -> Option<&str> {
        self.ids.get(i as usize).map(String::as_str)
    }

    /// Intern an id, returning `(index, newly_created)`.
    fn intern(&mut self, id: &str) -> (u64, bool) {
        if let Some(&i) = self.index.get(id) {
            return (i, false);
        }
        let i = self.ids.len() as u64;
        self.ids.push(id.to_string());
        self.index.insert(id.to_string(), i);
        (i, true)
    }
}

/// Typed adjacency: one boolean matrix per edge type over a shared node index.
pub struct TypedAdjacency {
    nodes: NodeIndex,
    bool_mats: BTreeMap<String, Matrix>,
    /// edge_id -> (from_idx, to_idx, edge_type), so a delete (which cannot
    /// re-read the store for endpoints) still resolves its matrix cell.
    edges: HashMap<String, (u64, u64, String)>,
}

impl Default for TypedAdjacency {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedAdjacency {
    pub fn new() -> Self {
        Self {
            nodes: NodeIndex::default(),
            bool_mats: BTreeMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Number of distinct nodes seen (the matrix dimension).
    pub fn node_count(&self) -> u64 {
        self.nodes.len()
    }

    /// The boolean adjacency matrix for a relationship type, if any edges of
    /// that type have been seen.
    pub fn matrix(&self, edge_type: &str) -> Option<&Matrix> {
        self.bool_mats.get(edge_type)
    }

    /// The relationship types with their own matrix.
    pub fn edge_types(&self) -> impl Iterator<Item = &str> {
        self.bool_mats.keys().map(String::as_str)
    }

    /// Grow every existing matrix to the current node count.
    fn ensure_dim(&mut self) -> AdjResult<()> {
        let n = self.nodes.len();
        for m in self.bool_mats.values_mut() {
            if m.nrows()? < n {
                m.resize(n, n)?;
            }
        }
        Ok(())
    }

    /// Insert (or refresh) a directed edge of `edge_type`, interning endpoints
    /// and growing matrices as needed. `edge_id` keys the edge for later removal.
    pub fn upsert_edge(
        &mut self,
        edge_id: &str,
        from: &str,
        to: &str,
        edge_type: &str,
    ) -> AdjResult<()> {
        let (fi, _) = self.nodes.intern(from);
        let (ti, _) = self.nodes.intern(to);
        self.ensure_dim()?;
        let n = self.nodes.len();
        if !self.bool_mats.contains_key(edge_type) {
            self.bool_mats
                .insert(edge_type.to_string(), Matrix::new(ElementType::Bool, n, n)?);
        }
        let m = self
            .bool_mats
            .get_mut(edge_type)
            .expect("matrix present after insert");
        m.set_bool(fi, ti, true)?;
        self.edges
            .insert(edge_id.to_string(), (fi, ti, edge_type.to_string()));
        Ok(())
    }

    /// Remove an edge by its id (no-op if unknown).
    pub fn remove_edge_by_id(&mut self, edge_id: &str) -> AdjResult<()> {
        if let Some((fi, ti, et)) = self.edges.remove(edge_id) {
            if let Some(m) = self.bool_mats.get_mut(&et) {
                m.remove(fi, ti)?;
            }
        }
        Ok(())
    }

    /// Build from a batch of edge records (skipping tombstones), keyed by the
    /// record id.
    pub fn from_records(records: &[EdgeRecord]) -> AdjResult<Self> {
        let mut adj = Self::new();
        for e in records.iter().filter(|e| !e.tombstone) {
            adj.upsert_edge(&e.id, &e.from_id, &e.to_id, &e.edge_type)?;
        }
        Ok(adj)
    }

    /// The union of all per-type matrices as deduplicated, sorted
    /// `(from_id, to_id)` pairs. Compared against [`CsrGraph::directed_pairs`].
    pub fn directed_pairs(&self) -> AdjResult<Vec<(String, String)>> {
        let mut set = BTreeSet::new();
        for m in self.bool_mats.values() {
            for (i, j) in m.bool_tuples()? {
                let from = self.nodes.id_of(i).unwrap_or_default().to_string();
                let to = self.nodes.id_of(j).unwrap_or_default().to_string();
                set.insert((from, to));
            }
        }
        Ok(set.into_iter().collect())
    }

    /// Node id for a dense index (shared across all per-type matrices).
    pub fn node_id(&self, i: u64) -> Option<&str> {
        self.nodes.id_of(i)
    }

    /// Dense index for a node id.
    pub fn index_of(&self, id: &str) -> Option<u64> {
        self.nodes.index_of(id)
    }

    /// Map a result matrix's `true` entries (over this adjacency's node index)
    /// to deduplicated, sorted `(from_id, to_id)` pairs -- e.g. a derived
    /// points-to or reachability relation.
    pub fn pairs_for(&self, m: &Matrix) -> AdjResult<Vec<(String, String)>> {
        let mut out = BTreeSet::new();
        for (i, j) in m.bool_tuples()? {
            let from = self.nodes.id_of(i).unwrap_or_default().to_string();
            let to = self.nodes.id_of(j).unwrap_or_default().to_string();
            out.insert((from, to));
        }
        Ok(out.into_iter().collect())
    }

    /// Forward reachability over `edge_type` from string `sources`, as node ids.
    /// Runs the GraphBLAS semiring traversal; compared against
    /// [`CsrGraph::reachable_from`].
    pub fn reachable_ids(&self, edge_type: &str, sources: &[&str]) -> AdjResult<BTreeSet<String>> {
        let m = match self.bool_mats.get(edge_type) {
            Some(m) => m,
            None => return Ok(BTreeSet::new()),
        };
        let src_idx: Vec<u64> = sources
            .iter()
            .filter_map(|s| self.nodes.index_of(s))
            .collect();
        let reached = rustyred_thg_graphblas::reachable_from(m, self.nodes.len(), &src_idx)?;
        let ids = reached
            .indices_bool()?
            .into_iter()
            .filter_map(|i| self.nodes.id_of(i).map(String::from))
            .collect();
        Ok(ids)
    }
}

/// A [`HookRegistration`] that keeps a shared [`TypedAdjacency`] consistent with
/// the graph: on every edge upsert/delete it resolves the edge's endpoints from
/// the store and applies the incremental matrix update. Attach its emitter to a
/// store so the matrices and the graph stay in lockstep.
pub fn adjacency_maintenance_hook(shared: Arc<Mutex<TypedAdjacency>>) -> HookRegistration {
    let handler: HookHandler = Arc::new(move |ctx, events| {
        let mut adj = shared
            .lock()
            .map_err(|_| HookError::new("typed-adjacency mutex poisoned"))?;
        let mut wrote = 0usize;
        for ev in events {
            match ev.kind {
                MutationKind::EdgeUpserted => {
                    // Resolve endpoints from the post-commit edge record. The
                    // concrete store's inherent get_edge returns Result<Option<_>>
                    // (it shadows the trait's Option<_>).
                    if let Ok(Some(rec)) = ctx.store.get_edge(&ev.id) {
                        let (from, to, et) = (
                            rec.from_id.clone(),
                            rec.to_id.clone(),
                            rec.edge_type.clone(),
                        );
                        adj.upsert_edge(&ev.id, &from, &to, &et)
                            .map_err(|e| HookError::new(e.to_string()))?;
                        wrote += 1;
                    }
                }
                MutationKind::EdgeDeleted => {
                    adj.remove_edge_by_id(&ev.id)
                        .map_err(|e| HookError::new(e.to_string()))?;
                    wrote += 1;
                }
                MutationKind::NodeUpserted | MutationKind::NodeDeleted => {}
            }
        }
        Ok(HookOutcome::Wrote { mutations: wrote })
    });

    HookRegistration::new(
        "graphblas-typed-adjacency",
        MutationMatcher::any().with_kinds([MutationKind::EdgeUpserted, MutationKind::EdgeDeleted]),
        coalesce_per_id,
        handler,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::record::{EdgeRecord, NodeRecord};
    use crate::graph_csr::CsrGraph;
    use crate::graph_store::{RedCoreDurability, RedCoreGraphStore, RedCoreOptions};
    use crate::hooks::{HookContext, MutationEvent};
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("gb-adj-{tag}-{}-{n}", std::process::id()))
    }

    fn edge(id: &str, from: &str, ty: &str, to: &str) -> EdgeRecord {
        EdgeRecord::new(id, from, ty, to, json!({}))
    }

    // ACCEPTANCE (D2): after a sequence of node/edge mutations, the typed
    // matrices and the CSR agree. Built and mutated through the same incremental
    // API the hook drives.
    #[test]
    fn typed_matrices_agree_with_csr() {
        let mut records = vec![
            edge("e1", "a", "KNOWS", "b"),
            edge("e2", "b", "KNOWS", "c"),
            edge("e3", "a", "CITES", "c"),
        ];

        let adj = TypedAdjacency::from_records(&records).unwrap();
        let csr = CsrGraph::from_records(&records);

        assert_eq!(adj.directed_pairs().unwrap(), csr.directed_pairs());
        assert_eq!(adj.node_count() as usize, csr.node_count());
        // per-type matrices: KNOWS has 2 edges, CITES has 1
        assert_eq!(adj.matrix("KNOWS").unwrap().nvals().unwrap(), 2);
        assert_eq!(adj.matrix("CITES").unwrap().nvals().unwrap(), 1);

        // Mutation: drop the b->c KNOWS edge incrementally, then re-derive CSR.
        let mut adj = adj;
        adj.remove_edge_by_id("e2").unwrap();
        records.retain(|e| e.id != "e2");
        let csr2 = CsrGraph::from_records(&records);
        assert_eq!(adj.directed_pairs().unwrap(), csr2.directed_pairs());
        assert_eq!(
            adj.directed_pairs().unwrap(),
            vec![
                ("a".to_string(), "b".to_string()),
                ("a".to_string(), "c".to_string())
            ]
        );
    }

    // The incremental updates actually flow through a hooks.rs handler reading a
    // real store: upsert three edges, then delete one, via the registration's
    // handler, and confirm the shared adjacency tracks the graph.
    #[test]
    fn hook_handler_tracks_store_edges() {
        let dir = unique_dir("hook");
        let mut store = RedCoreGraphStore::open(
            &dir,
            RedCoreOptions {
                durability: RedCoreDurability::AofAlways,
                snapshot_interval_writes: 100,
                strict_acid: false,
            },
        )
        .unwrap();
        for id in ["a", "b", "c"] {
            store
                .upsert_node(NodeRecord::new(id, ["Node"], json!({})))
                .unwrap();
        }
        for e in [
            edge("e1", "a", "KNOWS", "b"),
            edge("e2", "b", "KNOWS", "c"),
            edge("e3", "a", "CITES", "c"),
        ] {
            store.upsert_edge(e).unwrap();
        }

        let shared = Arc::new(Mutex::new(TypedAdjacency::new()));
        let reg = adjacency_maintenance_hook(shared.clone());

        let upserts: Vec<MutationEvent> = ["e1", "e2", "e3"]
            .iter()
            .map(|id| {
                MutationEvent::new(MutationKind::EdgeUpserted, "t", *id, vec![], vec![], 0, 0)
            })
            .collect();
        {
            let mut ctx = HookContext {
                store: &mut store,
                tenant: "t",
                depth: 0,
            };
            (reg.handler)(&mut ctx, &upserts).unwrap();
        }
        assert_eq!(
            shared.lock().unwrap().directed_pairs().unwrap(),
            vec![
                ("a".to_string(), "b".to_string()),
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "c".to_string()),
            ]
        );

        let delete = vec![MutationEvent::new(
            MutationKind::EdgeDeleted,
            "t",
            "e2",
            vec![],
            vec![],
            0,
            0,
        )];
        {
            let mut ctx = HookContext {
                store: &mut store,
                tenant: "t",
                depth: 0,
            };
            (reg.handler)(&mut ctx, &delete).unwrap();
        }
        assert_eq!(
            shared.lock().unwrap().directed_pairs().unwrap(),
            vec![
                ("a".to_string(), "b".to_string()),
                ("a".to_string(), "c".to_string()),
            ]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ACCEPTANCE (D3): GraphBLAS semiring traversal reproduces the walks that
    // graph_csr.rs produces (forward reachability from a source).
    #[test]
    fn semiring_traversal_matches_csr_walks() {
        let records = vec![
            edge("e1", "a", "LINK", "b"),
            edge("e2", "b", "LINK", "c"),
            edge("e3", "c", "LINK", "d"),
            edge("e4", "a", "LINK", "c"),
            edge("e5", "x", "LINK", "y"), // disconnected component
        ];
        let adj = TypedAdjacency::from_records(&records).unwrap();
        let csr = CsrGraph::from_records(&records);

        let gb_reached = adj.reachable_ids("LINK", &["a"]).unwrap();
        let csr_reached: BTreeSet<String> = csr
            .reachable_from(&[csr.index_of("a").unwrap()])
            .into_iter()
            .map(|i| csr.id_of(i).unwrap().to_string())
            .collect();

        assert_eq!(gb_reached, csr_reached);
        assert_eq!(
            gb_reached,
            ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect()
        );
    }
}
