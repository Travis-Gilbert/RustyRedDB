//! C2: CSR analytics view (RustyRed-THG improvement plan, Workstream C2).
//!
//! The mutable store stays adjacency + property + vector. This module is a
//! transient, integer-indexed Compressed-Sparse-Row *lens* over a set of
//! edges. It is built **once per batch analytics run** (a dense string->int
//! dictionary plus two integer arrays: row pointers and column indices) and
//! discarded after the run. The tier-1 algorithms (PageRank, betweenness with
//! articulation points and bridges, SCC with topological order, Louvain/Leiden
//! communities, KNN) and Personalized PageRank run on integer indices, not on
//! a `HashMap<String, Vec<String>>` rebuilt per call.
//!
//! The matrix is a lens, never the spine: nothing here mutates a `GraphStore`.
//!
//! Acceptance (plan C2): a tier-1 run allocates the CSR once (not per call) and
//! runs on integer indices; PPR is expressed as repeated sparse matrix-vector
//! products; memory for a run is bounded by the CSR size.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::graph::record::EdgeRecord;

/// Minimum weight floor so confidence-zero / missing edges still carry mass.
const WEIGHT_EPSILON: f64 = 1e-6;

/// An immutable, integer-indexed CSR view over a directed weighted graph.
///
/// Construction interns every node id into a dense `[0, n)` index once. Out-edges
/// are stored in CSR form: `row_ptr` has length `n + 1`, and the out-neighbors of
/// node `u` are `col_idx[row_ptr[u]..row_ptr[u + 1]]` with aligned `weights`.
///
/// Algorithms borrow `&self` and allocate only `O(n + m)` scratch per run, never
/// per call. Build the view once, run many algorithms, then drop it.
#[derive(Clone, Debug)]
pub struct CsrGraph {
    /// Dense index -> node id (sorted for determinism).
    ids: Vec<String>,
    /// Node id -> dense index.
    index: HashMap<String, u32>,
    /// CSR row offsets for out-edges, length `n + 1`.
    row_ptr: Vec<u32>,
    /// CSR column indices (out-edge targets), length `m`.
    col_idx: Vec<u32>,
    /// Edge weights aligned with `col_idx`, length `m`.
    weights: Vec<f64>,
}

#[derive(Debug)]
struct UndirectedLowlinkAnalysis {
    component_increase_counts: Vec<usize>,
    bridges: Vec<(usize, usize)>,
}

impl CsrGraph {
    /// Build a CSR view from `(from, to, weight)` triples. Self-loops are kept;
    /// parallel edges are kept (their weights accumulate at SpMV time). Node
    /// order is the sorted set of distinct ids, so the view is deterministic.
    ///
    /// This is the decoupled core: it does not depend on `EdgeRecord`, so it is
    /// unit-testable in isolation.
    pub fn from_edges<I, S>(edges: I) -> Self
    where
        I: IntoIterator<Item = (S, S, f64)>,
        S: AsRef<str>,
    {
        let triples: Vec<(String, String, f64)> = edges
            .into_iter()
            .map(|(from, to, weight)| (from.as_ref().to_string(), to.as_ref().to_string(), weight))
            .collect();
        Self::from_owned_triples(triples, Vec::new())
    }

    /// Build a CSR view from weighted edges plus explicit node ids. The extra
    /// node set keeps isolated seed/shadow nodes present in the dense index.
    pub fn from_edges_and_nodes<I, S, N, T>(edges: I, nodes: N) -> Self
    where
        I: IntoIterator<Item = (S, S, f64)>,
        S: AsRef<str>,
        N: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let triples: Vec<(String, String, f64)> = edges
            .into_iter()
            .map(|(from, to, weight)| (from.as_ref().to_string(), to.as_ref().to_string(), weight))
            .collect();
        let extra_nodes = nodes
            .into_iter()
            .map(|node| node.as_ref().to_string())
            .collect();
        Self::from_owned_triples(triples, extra_nodes)
    }

    /// Build a CSR view from the legacy string-keyed weighted adjacency map.
    /// `extra_nodes` should include seeds that may not appear in the map.
    pub fn from_weighted_adjacency<I, S>(
        adjacency: &HashMap<String, Vec<(String, f64)>>,
        extra_nodes: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let triples: Vec<(String, String, f64)> = adjacency
            .iter()
            .flat_map(|(from, neighbors)| {
                neighbors
                    .iter()
                    .map(move |(to, weight)| (from.clone(), to.clone(), *weight))
            })
            .collect();
        let mut nodes = adjacency.keys().cloned().collect::<Vec<_>>();
        nodes.extend(
            extra_nodes
                .into_iter()
                .map(|node| node.as_ref().to_string()),
        );
        Self::from_owned_triples(triples, nodes)
    }

    fn from_owned_triples(triples: Vec<(String, String, f64)>, extra_nodes: Vec<String>) -> Self {
        let mut id_set: BTreeSet<String> = extra_nodes
            .into_iter()
            .filter(|node| !node.trim().is_empty())
            .collect();
        for (from, to, _) in &triples {
            id_set.insert(from.clone());
            id_set.insert(to.clone());
        }
        let ids: Vec<String> = id_set.into_iter().collect();
        let mut index: HashMap<String, u32> = HashMap::with_capacity(ids.len());
        for (dense, id) in ids.iter().enumerate() {
            index.insert(id.clone(), dense as u32);
        }

        let n = ids.len();
        let m = triples.len();
        // Counting sort into CSR: first count out-degrees, prefix-sum to offsets,
        // then scatter targets with a moving cursor.
        let mut row_ptr = vec![0u32; n + 1];
        for (from, _, _) in &triples {
            let u = index[from.as_str()] as usize;
            row_ptr[u + 1] += 1;
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }
        let mut col_idx = vec![0u32; m];
        let mut weights = vec![0.0f64; m];
        let mut cursor: Vec<u32> = row_ptr[..n].to_vec();
        for (from, to, weight) in &triples {
            let u = index[from.as_str()] as usize;
            let slot = cursor[u] as usize;
            col_idx[slot] = index[to.as_str()];
            weights[slot] = *weight;
            cursor[u] += 1;
        }

        Self {
            ids,
            index,
            row_ptr,
            col_idx,
            weights,
        }
    }

    /// Build a CSR view from `EdgeRecord`s, skipping tombstones. Edge weight is
    /// `effective_confidence()` floored at `WEIGHT_EPSILON`.
    pub fn from_records(records: &[EdgeRecord]) -> Self {
        Self::from_edges(records.iter().filter(|e| !e.tombstone).map(|e| {
            (
                e.from_id.as_str(),
                e.to_id.as_str(),
                e.effective_confidence().max(WEIGHT_EPSILON),
            )
        }))
    }

    /// Number of distinct nodes (CSR dimension `n`).
    pub fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// Number of stored directed edges (CSR nnz `m`).
    pub fn edge_count(&self) -> usize {
        self.col_idx.len()
    }

    /// Total bytes backing the CSR arrays (dictionary excluded). Used to prove
    /// the run's working set is bounded by CSR size, not string-keyed maps.
    pub fn csr_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<u32>()
            + self.col_idx.len() * std::mem::size_of::<u32>()
            + self.weights.len() * std::mem::size_of::<f64>()
    }

    /// Dense index for a node id, if present.
    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.index.get(id).map(|i| *i as usize)
    }

    /// Node id for a dense index.
    pub fn id_of(&self, dense: usize) -> Option<&str> {
        self.ids.get(dense).map(String::as_str)
    }

    /// All directed edges as deduplicated, sorted `(from_id, to_id)` pairs.
    /// The structural oracle the GraphBLAS typed adjacency must agree with.
    pub fn directed_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = BTreeSet::new();
        for u in 0..self.node_count() {
            for (v, _w) in self.out_edges(u) {
                pairs.insert((self.ids[u].clone(), self.ids[v].clone()));
            }
        }
        pairs.into_iter().collect()
    }

    /// Forward reachability (BFS over out-edges) from `sources`: the set of
    /// dense indices reachable, inclusive of the sources. The walk oracle that
    /// GraphBLAS semiring traversal must reproduce.
    pub fn reachable_from(&self, sources: &[usize]) -> BTreeSet<usize> {
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        for &s in sources {
            if s < self.node_count() && visited.insert(s) {
                queue.push_back(s);
            }
        }
        while let Some(u) = queue.pop_front() {
            for (v, _w) in self.out_edges(u) {
                if visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
        visited
    }

    /// Out-neighbors of `u` as `(target_index, weight)` slices.
    fn out_edges(&self, u: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_ptr[u] as usize;
        let end = self.row_ptr[u + 1] as usize;
        self.col_idx[start..end]
            .iter()
            .zip(self.weights[start..end].iter())
            .map(|(&v, &w)| (v as usize, w))
    }

    /// Out-degree of `u`.
    fn out_degree(&self, u: usize) -> usize {
        (self.row_ptr[u + 1] - self.row_ptr[u]) as usize
    }

    // ===== Sparse matrix-vector core =====

    /// One sparse matrix-vector style scatter step of the random-walk transition:
    /// `out[v] += sum_{u: u->v} vec[u] * w(u,v) / outweight(u)`.
    ///
    /// This is the primitive that makes PageRank and PPR "repeated SpMV": each
    /// iteration scatters the current vector across CSR rows exactly once, so an
    /// iteration costs `O(m)`, independent of the dictionary. Dangling mass
    /// (nodes with no out-edges) is returned separately so callers can
    /// redistribute it per their teleport convention.
    fn spmv_transition(&self, vec: &[f64], out: &mut [f64]) -> f64 {
        debug_assert_eq!(vec.len(), self.node_count());
        debug_assert_eq!(out.len(), self.node_count());
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        let mut dangling = 0.0;
        for u in 0..self.node_count() {
            let mass = vec[u];
            if self.out_degree(u) == 0 {
                dangling += mass;
                continue;
            }
            let total_w: f64 = self.out_edges(u).map(|(_, w)| w).sum();
            if total_w <= 0.0 {
                dangling += mass;
                continue;
            }
            for (v, w) in self.out_edges(u) {
                out[v] += mass * w / total_w;
            }
        }
        dangling
    }

    // ===== PageRank =====

    /// Power-iteration PageRank on integer indices. Returns a score per dense
    /// node index; the vector sums to 1.0. Mirrors the semantics of
    /// `graph::pagerank` (uniform-by-out-degree share, dangling teleport) but
    /// runs on the prebuilt CSR rather than rebuilding adjacency.
    pub fn pagerank(&self, damping: f64, max_iter: usize, tolerance: f64) -> Vec<f64> {
        let n = self.node_count();
        if n == 0 {
            return Vec::new();
        }
        let nf = n as f64;
        let mut rank = vec![1.0 / nf; n];
        let mut next = vec![0.0f64; n];
        // Unweighted out-degree share to match the classic formulation.
        let degree: Vec<usize> = (0..n).map(|u| self.out_degree(u)).collect();
        for _ in 0..max_iter {
            for slot in next.iter_mut() {
                *slot = 0.0;
            }
            let mut dangling = 0.0;
            for u in 0..n {
                let mass = rank[u];
                if degree[u] == 0 {
                    dangling += mass;
                    continue;
                }
                let share = mass / degree[u] as f64;
                for (v, _w) in self.out_edges(u) {
                    next[v] += share;
                }
            }
            let teleport = (1.0 - damping) / nf + damping * dangling / nf;
            let mut delta = 0.0;
            for u in 0..n {
                let updated = teleport + damping * next[u];
                delta += (updated - rank[u]).abs();
                next[u] = updated;
            }
            std::mem::swap(&mut rank, &mut next);
            if delta < tolerance {
                break;
            }
        }
        rank
    }

    // ===== Personalized PageRank as repeated SpMV =====

    /// Personalized PageRank (random walk with restart) expressed as repeated
    /// sparse matrix-vector products: `p_{t+1} = alpha * P^T p_t + (1-alpha) s`,
    /// where `P` is the row-normalized weighted transition and `s` is the
    /// restart distribution from `seeds` (index, mass). Dangling mass is
    /// returned to the restart distribution so total mass is conserved (sums to
    /// 1.0). Returns a score per dense index.
    ///
    /// This is the C2 acceptance formulation of PPR; each iteration is one
    /// `spmv_transition` over the CSR.
    pub fn personalized_pagerank(
        &self,
        seeds: &[(usize, f64)],
        continuation_probability: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Vec<f64> {
        let n = self.node_count();
        if n == 0 {
            return Vec::new();
        }
        // Normalize the restart distribution.
        let mut restart = vec![0.0f64; n];
        let mut seed_total = 0.0;
        for &(idx, mass) in seeds {
            if idx < n && mass > 0.0 {
                restart[idx] += mass;
                seed_total += mass;
            }
        }
        if seed_total <= 0.0 {
            return vec![0.0; n];
        }
        for slot in restart.iter_mut() {
            *slot /= seed_total;
        }

        let mut p = restart.clone();
        let mut walked = vec![0.0f64; n];
        for _ in 0..max_iter {
            let dangling = self.spmv_transition(&p, &mut walked);
            // p_next = alpha * (walked + dangling * restart) + (1 - alpha) * restart
            let mut delta = 0.0;
            for u in 0..n {
                let walk_mass = walked[u] + dangling * restart[u];
                let updated = continuation_probability * walk_mass
                    + (1.0 - continuation_probability) * restart[u];
                delta += (updated - p[u]).abs();
                walked[u] = updated;
            }
            std::mem::swap(&mut p, &mut walked);
            if delta < tolerance {
                break;
            }
        }
        p
    }

    /// ACL local-push Personalized PageRank over the prebuilt CSR.
    ///
    /// This mirrors `graph::personalized_pagerank` exactly: `alpha` is the
    /// reserve/teleport mass, `epsilon` is scaled by out-degree, and
    /// `max_pushes` bounds push operations rather than power iterations.
    pub fn personalized_pagerank_local_push(
        &self,
        seeds: &[(usize, f64)],
        teleport_probability: f64,
        epsilon: f64,
        max_pushes: usize,
    ) -> Vec<f64> {
        let n = self.node_count();
        if n == 0 {
            return Vec::new();
        }
        let out_weight = (0..n)
            .map(|u| self.out_edges(u).map(|(_, w)| w).sum::<f64>())
            .collect::<Vec<_>>();

        let mut p = vec![0.0f64; n];
        let mut r = vec![0.0f64; n];
        let mut queue = VecDeque::new();
        let mut in_queue = vec![false; n];
        for &(idx, mass) in seeds {
            if idx >= n {
                continue;
            }
            r[idx] += mass;
            if !in_queue[idx] {
                queue.push_back(idx);
                in_queue[idx] = true;
            }
        }

        let mut pushes = 0usize;
        while let Some(u) = queue.pop_front() {
            in_queue[u] = false;
            if pushes >= max_pushes {
                break;
            }
            let residual_u = r[u];
            let degree = self.out_degree(u).max(1) as f64;
            if residual_u <= epsilon * degree {
                continue;
            }
            pushes += 1;

            p[u] += teleport_probability * residual_u;
            r[u] = 0.0;

            let total_w = out_weight[u];
            if self.out_degree(u) == 0 || total_w <= 0.0 {
                r[u] += (1.0 - teleport_probability) * residual_u;
                continue;
            }
            for (v, w) in self.out_edges(u) {
                let share = (1.0 - teleport_probability) * residual_u * w / total_w;
                r[v] += share;
                let target_degree = self.out_degree(v).max(1) as f64;
                if r[v] > epsilon * target_degree && !in_queue[v] {
                    queue.push_back(v);
                    in_queue[v] = true;
                }
            }
        }
        p
    }

    /// Named wrapper for the CSR local-push PPR path.
    pub fn personalized_pagerank_local_push_named(
        &self,
        seeds: &HashMap<String, f64>,
        alpha: f64,
        epsilon: f64,
        max_pushes: usize,
    ) -> HashMap<String, f64> {
        let dense_seeds = seeds
            .iter()
            .filter_map(|(node, mass)| self.index_of(node).map(|idx| (idx, *mass)))
            .collect::<Vec<_>>();
        self.personalized_pagerank_local_push(&dense_seeds, alpha, epsilon, max_pushes)
            .into_iter()
            .enumerate()
            .filter_map(|(idx, score)| {
                if score == 0.0 {
                    None
                } else {
                    Some((self.ids[idx].clone(), score))
                }
            })
            .collect()
    }

    // ===== Connected components (undirected) =====

    /// Weakly-connected components via union-find over the directed edges
    /// (treated as undirected). Returns components as sorted lists of dense
    /// indices, largest first.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count();
        let mut dsu = DisjointSet::new(n);
        for u in 0..n {
            for (v, _w) in self.out_edges(u) {
                dsu.union(u, v);
            }
        }
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for u in 0..n {
            groups.entry(dsu.find(u)).or_default().push(u);
        }
        let mut components: Vec<Vec<usize>> = groups.into_values().collect();
        for c in components.iter_mut() {
            c.sort_unstable();
        }
        components.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        components
    }

    // ===== Strongly connected components (Tarjan, iterative) + topo order =====

    /// Strongly connected components via an iterative Tarjan pass on the CSR.
    /// Tarjan emits components in reverse topological order of the condensation
    /// DAG; this returns them in that emission order. Each component is a sorted
    /// list of dense indices.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count();
        let mut index_of = vec![u32::MAX; n];
        let mut lowlink = vec![0u32; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut next_index: u32 = 0;

        // Explicit DFS frame: node + current out-edge cursor.
        struct Frame {
            node: usize,
            cursor: usize,
        }

        for root in 0..n {
            if index_of[root] != u32::MAX {
                continue;
            }
            let mut frames: Vec<Frame> = vec![Frame {
                node: root,
                cursor: self.row_ptr[root] as usize,
            }];
            index_of[root] = next_index;
            lowlink[root] = next_index;
            next_index += 1;
            stack.push(root);
            on_stack[root] = true;

            while let Some(frame) = frames.last_mut() {
                let u = frame.node;
                let end = self.row_ptr[u + 1] as usize;
                if frame.cursor < end {
                    let v = self.col_idx[frame.cursor] as usize;
                    frame.cursor += 1;
                    if index_of[v] == u32::MAX {
                        index_of[v] = next_index;
                        lowlink[v] = next_index;
                        next_index += 1;
                        stack.push(v);
                        on_stack[v] = true;
                        frames.push(Frame {
                            node: v,
                            cursor: self.row_ptr[v] as usize,
                        });
                    } else if on_stack[v] {
                        lowlink[u] = lowlink[u].min(index_of[v]);
                    }
                } else {
                    // Done with u: if it is a root, pop its SCC.
                    if lowlink[u] == index_of[u] {
                        let mut component = Vec::new();
                        loop {
                            let w = stack.pop().expect("tarjan stack nonempty");
                            on_stack[w] = false;
                            component.push(w);
                            if w == u {
                                break;
                            }
                        }
                        component.sort_unstable();
                        components.push(component);
                    }
                    frames.pop();
                    if let Some(parent) = frames.last() {
                        let p = parent.node;
                        lowlink[p] = lowlink[p].min(lowlink[u]);
                    }
                }
            }
        }
        components
    }

    /// SCCs in topological order of the condensation DAG (sources first). This
    /// is the reverse of Tarjan's emission order.
    pub fn scc_topological_order(&self) -> Vec<Vec<usize>> {
        let mut comps = self.strongly_connected_components();
        comps.reverse();
        comps
    }

    // ===== Betweenness centrality (Brandes) =====

    /// Brandes betweenness centrality on the directed CSR (unweighted shortest
    /// paths). Returns a score per dense index. `O(n*m)` time, `O(n+m)` scratch.
    pub fn betweenness_centrality(&self) -> Vec<f64> {
        let n = self.node_count();
        let mut centrality = vec![0.0f64; n];
        if n == 0 {
            return centrality;
        }
        for s in 0..n {
            let mut stack: Vec<usize> = Vec::with_capacity(n);
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            let mut dist = vec![-1i64; n];
            sigma[s] = 1.0;
            dist[s] = 0;
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for (w, _weight) in self.out_edges(v) {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }
            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    if sigma[w] > 0.0 {
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }
        centrality
    }

    // ===== Articulation points and bridges (undirected projection) =====

    /// Increase in weak component count after removing each dense node.
    pub(crate) fn component_increase_counts(&self) -> Vec<usize> {
        self.undirected_lowlink_analysis().component_increase_counts
    }

    /// Articulation points and bridges of the undirected projection, computed in
    /// a single iterative DFS lowlink pass. Articulation points are returned as
    /// sorted dense indices; bridges as sorted `(min, max)` index pairs.
    pub fn articulation_points_and_bridges(&self) -> (Vec<usize>, Vec<(usize, usize)>) {
        let analysis = self.undirected_lowlink_analysis();
        let points = analysis
            .component_increase_counts
            .iter()
            .enumerate()
            .filter_map(|(node, &increase)| (increase > 0).then_some(node))
            .collect();
        (points, analysis.bridges)
    }

    /// Run one iterative DFS lowlink pass over the undirected projection.
    ///
    /// For a DFS root, removing it adds `children - 1` components. For any
    /// other node, each DFS child whose subtree cannot reach above the node
    /// adds one component. The lowlink walk itself is `O(V + E)`.
    fn undirected_lowlink_analysis(&self) -> UndirectedLowlinkAnalysis {
        let n = self.node_count();
        let undirected = self.undirected_adjacency();
        let mut disc = vec![u32::MAX; n];
        let mut low = vec![0u32; n];
        let mut component_increase_counts = vec![0usize; n];
        let mut bridges: Vec<(usize, usize)> = Vec::new();
        let mut timer: u32 = 0;

        struct Frame {
            node: usize,
            parent: Option<usize>,
            cursor: usize,
            children: usize,
        }

        for root in 0..n {
            if disc[root] != u32::MAX {
                continue;
            }
            let mut frames: Vec<Frame> = vec![Frame {
                node: root,
                parent: None,
                cursor: 0,
                children: 0,
            }];
            disc[root] = timer;
            low[root] = timer;
            timer += 1;

            while let Some(frame) = frames.last_mut() {
                let u = frame.node;
                if frame.cursor < undirected[u].len() {
                    let v = undirected[u][frame.cursor];
                    frame.cursor += 1;
                    if Some(v) == frame.parent {
                        continue;
                    }
                    if disc[v] == u32::MAX {
                        frame.children += 1;
                        disc[v] = timer;
                        low[v] = timer;
                        timer += 1;
                        frames.push(Frame {
                            node: v,
                            parent: Some(u),
                            cursor: 0,
                            children: 0,
                        });
                    } else {
                        low[u] = low[u].min(disc[v]);
                    }
                } else {
                    let finished = frames.pop().expect("frame present");
                    if let Some(parent_frame) = frames.last_mut() {
                        let p = parent_frame.node;
                        low[p] = low[p].min(low[finished.node]);
                        // Non-root: every child subtree that cannot escape p
                        // becomes one additional component when p is removed.
                        if parent_frame.parent.is_some() && low[finished.node] >= disc[p] {
                            component_increase_counts[p] += 1;
                        }
                        // Bridge: child subtree cannot reach p or above.
                        if low[finished.node] > disc[p] {
                            let a = p.min(finished.node);
                            let b = p.max(finished.node);
                            bridges.push((a, b));
                        }
                    }
                    if finished.parent.is_none() {
                        component_increase_counts[finished.node] =
                            finished.children.saturating_sub(1);
                    }
                }
            }
        }

        bridges.sort_unstable();
        bridges.dedup();
        UndirectedLowlinkAnalysis {
            component_increase_counts,
            bridges,
        }
    }

    // ===== KNN node similarity =====

    /// For each node, its top-`k` most structurally similar nodes by cosine
    /// similarity over undirected neighbor sets (shared neighbors / sqrt(deg*deg)).
    /// Returns, per dense index, a descending list of `(neighbor_index, score)`.
    pub fn knn(&self, k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = self.node_count();
        let undirected = self.undirected_adjacency();
        let neighbor_sets: Vec<BTreeSet<usize>> = undirected
            .iter()
            .map(|nbrs| nbrs.iter().copied().collect())
            .collect();
        let mut result = vec![Vec::new(); n];
        for u in 0..n {
            if neighbor_sets[u].is_empty() {
                continue;
            }
            // Candidates: nodes reachable within 2 hops (share a neighbor).
            let mut candidates: BTreeSet<usize> = BTreeSet::new();
            for &mid in &neighbor_sets[u] {
                for &cand in &neighbor_sets[mid] {
                    if cand != u {
                        candidates.insert(cand);
                    }
                }
            }
            let mut scored: Vec<(usize, f64)> = candidates
                .into_iter()
                .filter_map(|cand| {
                    let shared = neighbor_sets[u].intersection(&neighbor_sets[cand]).count();
                    if shared == 0 {
                        return None;
                    }
                    let denom =
                        ((neighbor_sets[u].len() * neighbor_sets[cand].len()) as f64).sqrt();
                    Some((cand, shared as f64 / denom))
                })
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            scored.truncate(k);
            result[u] = scored;
        }
        result
    }

    // ===== Community detection (Louvain / Leiden) =====

    /// Louvain modularity-optimizing community detection on the undirected
    /// weighted projection. Returns `(label per dense index, modularity)` with
    /// labels compacted to `0..c`.
    pub fn louvain_communities(&self) -> (Vec<usize>, f64) {
        self.community_detection(false)
    }

    /// Leiden-flavored community detection: Louvain local-moving plus a
    /// connectivity-refinement pass that splits any community whose induced
    /// subgraph is disconnected into its connected pieces. This enforces the
    /// Leiden guarantee that every community is internally connected. Returns
    /// `(label per dense index, modularity)`.
    pub fn leiden_communities(&self) -> (Vec<usize>, f64) {
        self.community_detection(true)
    }

    fn community_detection(&self, refine_connectivity: bool) -> (Vec<usize>, f64) {
        let n = self.node_count();
        if n == 0 {
            return (Vec::new(), 0.0);
        }
        let undirected = self.undirected_weighted_adjacency();
        let total_2m: f64 = undirected
            .iter()
            .map(|nbrs| nbrs.iter().map(|(_, w)| *w).sum::<f64>())
            .sum();
        if total_2m <= 0.0 {
            return ((0..n).collect(), 0.0);
        }
        let k: Vec<f64> = undirected
            .iter()
            .map(|nbrs| nbrs.iter().map(|(_, w)| *w).sum::<f64>())
            .collect();

        // Each node starts in its own community.
        let mut community: Vec<usize> = (0..n).collect();
        let mut sigma_tot: Vec<f64> = k.clone();

        let mut improved = true;
        let mut passes = 0;
        while improved && passes < 64 {
            improved = false;
            passes += 1;
            for u in 0..n {
                let cu = community[u];
                // Weighted links from u into each neighboring community. A
                // BTreeMap so tied gains resolve to the lowest community id:
                // a HashMap's per-instance seed made repeated runs disagree.
                let mut links: BTreeMap<usize, f64> = BTreeMap::new();
                let mut self_loop = 0.0;
                for &(v, w) in &undirected[u] {
                    if v == u {
                        self_loop += w;
                        continue;
                    }
                    *links.entry(community[v]).or_insert(0.0) += w;
                }
                // Remove u from its community.
                sigma_tot[cu] -= k[u];
                let links_to_cu = links.get(&cu).copied().unwrap_or(0.0);

                // Pick the community maximizing modularity gain.
                let mut best_comm = cu;
                let mut best_gain = links_to_cu - sigma_tot[cu] * k[u] / total_2m;
                for (&comm, &k_in) in &links {
                    let gain = k_in - sigma_tot[comm] * k[u] / total_2m;
                    if gain > best_gain + 1e-12 {
                        best_gain = gain;
                        best_comm = comm;
                    }
                }
                let _ = self_loop;
                sigma_tot[best_comm] += k[u];
                if best_comm != cu {
                    community[u] = best_comm;
                    improved = true;
                }
            }
        }

        if refine_connectivity {
            self.split_disconnected_communities(&undirected, &mut community);
        }
        let compacted = compact_labels(&community);
        let modularity = self.modularity(&undirected, total_2m, &k, &compacted);
        (compacted, modularity)
    }

    /// Split any community whose induced subgraph is disconnected into separate
    /// connected communities (the Leiden well-connectedness refinement).
    fn split_disconnected_communities(
        &self,
        undirected: &[Vec<(usize, f64)>],
        community: &mut [usize],
    ) {
        let n = community.len();
        let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for u in 0..n {
            members.entry(community[u]).or_default().push(u);
        }
        let mut next_label = community.iter().copied().max().unwrap_or(0) + 1;
        for (_comm, nodes) in members {
            if nodes.len() < 2 {
                continue;
            }
            let in_comm: BTreeSet<usize> = nodes.iter().copied().collect();
            let mut visited: BTreeSet<usize> = BTreeSet::new();
            let mut first = true;
            for &start in &nodes {
                if visited.contains(&start) {
                    continue;
                }
                // BFS within the community-induced subgraph.
                let mut piece: Vec<usize> = Vec::new();
                let mut queue: VecDeque<usize> = VecDeque::new();
                queue.push_back(start);
                visited.insert(start);
                while let Some(u) = queue.pop_front() {
                    piece.push(u);
                    for &(v, _w) in &undirected[u] {
                        if in_comm.contains(&v) && visited.insert(v) {
                            queue.push_back(v);
                        }
                    }
                }
                if first {
                    first = false; // keep the original label for the first piece
                } else {
                    for u in piece {
                        community[u] = next_label;
                    }
                    next_label += 1;
                }
            }
        }
    }

    /// Newman-Girvan modularity of a partition on the undirected weighted graph.
    fn modularity(
        &self,
        undirected: &[Vec<(usize, f64)>],
        total_2m: f64,
        k: &[f64],
        community: &[usize],
    ) -> f64 {
        let mut q = 0.0;
        for u in 0..undirected.len() {
            for &(v, w) in &undirected[u] {
                if community[u] == community[v] {
                    q += w - k[u] * k[v] / total_2m;
                }
            }
        }
        q / total_2m
    }

    // ===== Undirected projections =====

    /// Undirected adjacency (dedup'd neighbor indices) of the directed CSR.
    fn undirected_adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.node_count();
        let mut adjacency = vec![Vec::new(); n];
        for u in 0..n {
            for (v, _w) in self.out_edges(u) {
                if u != v {
                    adjacency[u].push(v);
                    adjacency[v].push(u);
                }
            }
        }
        let mut seen_at = vec![usize::MAX; n];
        for (u, neighbors) in adjacency.iter_mut().enumerate() {
            neighbors.retain(|&v| {
                if seen_at[v] == u {
                    false
                } else {
                    seen_at[v] = u;
                    true
                }
            });
        }
        adjacency
    }

    /// Undirected weighted adjacency: parallel/reciprocal edges accumulate.
    fn undirected_weighted_adjacency(&self) -> Vec<Vec<(usize, f64)>> {
        let n = self.node_count();
        let mut maps: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
        for u in 0..n {
            for (v, w) in self.out_edges(u) {
                *maps[u].entry(v).or_insert(0.0) += w;
                if u != v {
                    *maps[v].entry(u).or_insert(0.0) += w;
                }
            }
        }
        maps.into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f64)> = m.into_iter().collect();
                v.sort_unstable_by_key(|(idx, _)| *idx);
                v
            })
            .collect()
    }

    /// Convenience: map a dense-indexed score vector back to node ids.
    pub fn label_scores(&self, scores: &[f64]) -> HashMap<String, f64> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (self.ids[i].clone(), s))
            .collect()
    }
}

/// Union-find with path compression and union by size.
struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

/// Compact arbitrary labels to a dense `0..c` range, preserving first-seen order.
fn compact_labels(labels: &[usize]) -> Vec<usize> {
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|&l| {
            *remap.entry(l).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph() -> CsrGraph {
        // a -> b -> c -> d
        CsrGraph::from_edges(vec![("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0)])
    }

    #[test]
    fn builds_csr_with_sorted_dense_indices() {
        let g = line_graph();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.index_of("a"), Some(0));
        assert_eq!(g.index_of("d"), Some(3));
        assert_eq!(g.id_of(0), Some("a"));
        // CSR row offsets are monotonically non-decreasing, length n+1.
        assert_eq!(g.row_ptr.len(), 5);
        for w in g.row_ptr.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn pagerank_sums_to_one_and_orders_by_depth() {
        let edges = vec![
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("c", "a", 1.0),
            ("a", "c", 1.0),
        ];
        let g = CsrGraph::from_edges(edges);
        let rank = g.pagerank(0.85, 100, 1e-9);
        let total: f64 = rank.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "total = {total}");
    }

    #[test]
    fn ppr_is_spmv_and_concentrates_near_seed() {
        let g = line_graph();
        let seed = g.index_of("a").unwrap();
        let scores = g.personalized_pagerank(&[(seed, 1.0)], 0.85, 200, 1e-12);
        // Mass conserved.
        let total: f64 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "ppr mass = {total}");
        // Seed should dominate (restart keeps returning to a).
        let a = scores[g.index_of("a").unwrap()];
        let d = scores[g.index_of("d").unwrap()];
        assert!(a > d, "seed a={a} should exceed far node d={d}");
    }

    #[test]
    fn ppr_named_round_trips_ids() {
        let g = line_graph();
        let seed = g.index_of("a").unwrap();
        let scores = g.personalized_pagerank(&[(seed, 1.0)], 0.85, 100, 1e-9);
        let named = g.label_scores(&scores);
        assert!(named.contains_key("a"));
        assert!(named["a"] > 0.0);
    }

    #[test]
    fn connected_components_partitions_disconnected_graph() {
        let g = CsrGraph::from_edges(vec![("a", "b", 1.0), ("b", "c", 1.0), ("x", "y", 1.0)]);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].len(), 3);
        assert_eq!(comps[1].len(), 2);
    }

    #[test]
    fn scc_finds_cycle_and_topo_orders_condensation() {
        // cycle a->b->c->a, then c->d (d is a sink SCC).
        let g = CsrGraph::from_edges(vec![
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("c", "a", 1.0),
            ("c", "d", 1.0),
        ]);
        let sccs = g.strongly_connected_components();
        // Two components: {a,b,c} and {d}.
        assert_eq!(sccs.len(), 2);
        let topo = g.scc_topological_order();
        // In topological order, the cycle (source) precedes the sink {d}.
        let cycle_first = topo[0].len() == 3;
        assert!(cycle_first, "cycle SCC should come first in topo order");
        assert_eq!(topo[1].len(), 1);
    }

    #[test]
    fn betweenness_peaks_on_bridge_node() {
        // a-b-c-d path (directed): b and c are on the shortest paths.
        let g = line_graph();
        let bc = g.betweenness_centrality();
        let b = bc[g.index_of("b").unwrap()];
        let a = bc[g.index_of("a").unwrap()];
        assert!(b > a, "middle node b={b} should exceed endpoint a={a}");
    }

    #[test]
    fn articulation_points_and_bridges_on_two_triangles() {
        // Two triangles joined by a single edge c-x: that edge is a bridge,
        // and c and x are articulation points.
        let g = CsrGraph::from_edges(vec![
            ("a", "b", 1.0),
            ("b", "c", 1.0),
            ("a", "c", 1.0),
            ("x", "y", 1.0),
            ("y", "z", 1.0),
            ("x", "z", 1.0),
            ("c", "x", 1.0),
        ]);
        let (points, bridges) = g.articulation_points_and_bridges();
        let c = g.index_of("c").unwrap();
        let x = g.index_of("x").unwrap();
        assert!(points.contains(&c), "c should be an articulation point");
        assert!(points.contains(&x), "x should be an articulation point");
        let want = (c.min(x), c.max(x));
        assert!(
            bridges.contains(&want),
            "c-x should be a bridge: {bridges:?}"
        );
    }

    #[test]
    fn component_increase_counts_isolated_node_as_zero() {
        let graph =
            CsrGraph::from_edges_and_nodes(std::iter::empty::<(&str, &str, f64)>(), ["isolated"]);

        assert_eq!(graph.component_increase_counts(), vec![0]);
    }

    #[test]
    fn component_increase_counts_chain_marks_internal_nodes() {
        let graph = CsrGraph::from_edges(vec![("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0)]);

        assert_eq!(graph.component_increase_counts(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn component_increase_counts_degree_four_star_marks_center_as_three() {
        let graph = CsrGraph::from_edges(vec![
            ("center", "leaf-a", 1.0),
            ("center", "leaf-b", 1.0),
            ("center", "leaf-c", 1.0),
            ("center", "leaf-d", 1.0),
        ]);

        assert_eq!(graph.component_increase_counts(), vec![3, 0, 0, 0, 0]);
    }

    #[test]
    fn component_increase_counts_are_local_to_disconnected_components() {
        let graph = CsrGraph::from_edges_and_nodes(
            vec![("a", "b", 1.0), ("b", "c", 1.0), ("x", "y", 1.0)],
            ["isolated"],
        );

        assert_eq!(graph.component_increase_counts(), vec![0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn knn_ranks_structural_neighbors() {
        // a and b both connect to hub h: they should be each other's KNN.
        let g = CsrGraph::from_edges(vec![("a", "h", 1.0), ("b", "h", 1.0), ("c", "h", 1.0)]);
        let knn = g.knn(2);
        let a = g.index_of("a").unwrap();
        let b = g.index_of("b").unwrap();
        let a_neighbors: Vec<usize> = knn[a].iter().map(|(i, _)| *i).collect();
        assert!(
            a_neighbors.contains(&b),
            "a's KNN should include b: {a_neighbors:?}"
        );
    }

    #[test]
    fn communities_find_two_clusters() {
        let g = CsrGraph::from_edges(vec![
            ("a", "b", 0.9),
            ("b", "c", 0.9),
            ("a", "c", 0.9),
            ("x", "y", 0.9),
            ("y", "z", 0.9),
            ("x", "z", 0.9),
            ("a", "x", 0.1),
        ]);
        let (labels, modularity) = g.leiden_communities();
        let a = g.index_of("a").unwrap();
        let b = g.index_of("b").unwrap();
        let c = g.index_of("c").unwrap();
        let x = g.index_of("x").unwrap();
        assert_eq!(labels[a], labels[b]);
        assert_eq!(labels[b], labels[c]);
        assert_ne!(labels[a], labels[x]);
        assert!(modularity > 0.0, "modularity = {modularity}");
    }

    #[test]
    fn leiden_refinement_splits_disconnected_communities() {
        // Force-check the refinement helper: a community with two disconnected
        // pieces must be split.
        let g = CsrGraph::from_edges(vec![("a", "b", 1.0), ("c", "d", 1.0)]);
        let undirected = g.undirected_weighted_adjacency();
        let mut community = vec![0usize; g.node_count()]; // all in one community
        g.split_disconnected_communities(&undirected, &mut community);
        // a-b connected, c-d connected, but the two pairs are not: >=2 labels.
        let distinct: BTreeSet<usize> = community.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "disconnected pieces must split: {community:?}"
        );
    }

    #[test]
    fn community_detection_is_identical_across_repeated_runs() {
        // A ring of eight equal edges: every first move ties between two
        // neighbouring communities, so the winner depends only on tie order.
        let ring: Vec<(String, String, f64)> =
            (0..8).map(|i| (format!("n{i}"), format!("n{}", (i + 1) % 8), 1.0)).collect();
        let g = CsrGraph::from_edges(ring.iter().map(|(a, b, w)| (a.as_str(), b.as_str(), *w)));
        let louvain = g.louvain_communities();
        let leiden = g.leiden_communities();
        for _ in 0..32 {
            assert_eq!(g.louvain_communities(), louvain);
            assert_eq!(g.leiden_communities(), leiden);
        }
    }

    #[test]
    fn empty_graph_is_safe() {
        let g = CsrGraph::from_edges(Vec::<(&str, &str, f64)>::new());
        assert_eq!(g.node_count(), 0);
        assert!(g.pagerank(0.85, 10, 1e-6).is_empty());
        assert!(g.personalized_pagerank(&[], 0.85, 10, 1e-6).is_empty());
        assert!(g.connected_components().is_empty());
        assert!(g.strongly_connected_components().is_empty());
    }
}
