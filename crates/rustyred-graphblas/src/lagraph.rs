//! D4: LAGraph algorithm bindings over the safe handle layer.
//!
//! [`LaGraph`] wraps a `LAGraph_Graph`, which takes ownership of its adjacency
//! matrix. Each algorithm computes the cached properties LAGraph requires
//! (transpose `AT`, out-degree, symmetric-structure flag) before calling in.
//! Directed vs undirected is fixed at construction: the undirected algorithms
//! (connected components, triangle count, k-truss) need a structurally
//! symmetric matrix with no self-edges, which this layer prepares.
//!
//! BFS, PageRank, betweenness, connected components, and SSSP are stable
//! LAGraph; k-truss is LAGraphX.

use crate::context::ensure_init;
use crate::error::{check, Result};
use crate::matrix::Matrix;
use crate::sys;
use crate::vector::Vector;
use std::os::raw::c_char;
use std::ptr;

const MSG_LEN: usize = 256;

/// An owned LAGraph graph; freed (with its matrices) on drop.
pub struct LaGraph {
    raw: sys::LAGraph_Graph,
    n: u64,
}

impl LaGraph {
    fn new(matrix: Matrix, kind: sys::LAGraph_Kind) -> Result<Self> {
        ensure_init();
        let n = matrix.nrows()?;
        let mut g: sys::LAGraph_Graph = ptr::null_mut();
        let mut a = matrix.into_raw();
        let mut msg = [0 as c_char; MSG_LEN];
        let rc = unsafe { sys::LAGraph_New(&mut g, &mut a, kind, msg.as_mut_ptr()) };
        if let Err(e) = check("LAGraph_New", rc) {
            // On failure ownership of the matrix was not taken; free it.
            if !a.is_null() {
                unsafe { sys::GrB_Matrix_free(&mut a) };
            }
            return Err(e);
        }
        Ok(Self { raw: g, n })
    }

    /// Directed graph: BFS, PageRank, betweenness, SSSP.
    pub fn directed(matrix: Matrix) -> Result<Self> {
        Self::new(matrix, sys::LAGraph_Kind_LAGraph_ADJACENCY_DIRECTED)
    }

    /// Undirected graph: connected components, triangle count, k-truss. The
    /// matrix must be structurally symmetric.
    pub fn undirected(matrix: Matrix) -> Result<Self> {
        Self::new(matrix, sys::LAGraph_Kind_LAGraph_ADJACENCY_UNDIRECTED)
    }

    pub fn node_count(&self) -> u64 {
        self.n
    }

    fn msg() -> [c_char; MSG_LEN] {
        [0 as c_char; MSG_LEN]
    }

    fn cache_at(&mut self) -> Result<()> {
        let mut m = Self::msg();
        check("LAGraph_Cached_AT", unsafe {
            sys::LAGraph_Cached_AT(self.raw, m.as_mut_ptr())
        })
    }

    fn cache_out_degree(&mut self) -> Result<()> {
        let mut m = Self::msg();
        check("LAGraph_Cached_OutDegree", unsafe {
            sys::LAGraph_Cached_OutDegree(self.raw, m.as_mut_ptr())
        })
    }

    fn cache_symmetric(&mut self) -> Result<()> {
        let mut m = Self::msg();
        check("LAGraph_Cached_IsSymmetricStructure", unsafe {
            sys::LAGraph_Cached_IsSymmetricStructure(self.raw, m.as_mut_ptr())
        })
    }

    fn delete_self_edges(&mut self) -> Result<()> {
        let mut m = Self::msg();
        check("LAGraph_DeleteSelfEdges", unsafe {
            sys::LAGraph_DeleteSelfEdges(self.raw, m.as_mut_ptr())
        })
    }

    fn cache_nself_edges(&mut self) -> Result<()> {
        let mut m = Self::msg();
        check("LAGraph_Cached_NSelfEdges", unsafe {
            sys::LAGraph_Cached_NSelfEdges(self.raw, m.as_mut_ptr())
        })
    }

    /// Breadth-first search from `src`: `(levels, parents)` per node, `-1` where
    /// unreached. `parents[i]` is the predecessor node id (`src` is its own).
    pub fn bfs(&mut self, src: u64) -> Result<(Vec<i64>, Vec<i64>)> {
        self.cache_at()?;
        self.cache_out_degree()?;
        let mut level: sys::GrB_Vector = ptr::null_mut();
        let mut parent: sys::GrB_Vector = ptr::null_mut();
        let mut m = Self::msg();
        check("LAGr_BreadthFirstSearch", unsafe {
            sys::LAGr_BreadthFirstSearch(&mut level, &mut parent, self.raw, src, m.as_mut_ptr())
        })?;
        let levels = Vector::from_raw(level).to_dense_i64(self.n, -1)?;
        let parents = Vector::from_raw(parent).to_dense_i64(self.n, -1)?;
        Ok((levels, parents))
    }

    /// PageRank centrality per node (sinks handled). Requires `AT` + out-degree.
    pub fn pagerank(&mut self, damping: f64, tol: f64, itermax: i32) -> Result<Vec<f64>> {
        self.cache_at()?;
        self.cache_out_degree()?;
        let mut centrality: sys::GrB_Vector = ptr::null_mut();
        let mut iters: i32 = 0;
        let mut m = Self::msg();
        check("LAGr_PageRank", unsafe {
            sys::LAGr_PageRank(
                &mut centrality,
                &mut iters,
                self.raw,
                // LAGraph PageRank is single-precision; the FP32 centrality
                // output reads back fine via GraphBLAS extractElement typecast.
                damping as f32,
                tol as f32,
                itermax,
                m.as_mut_ptr(),
            )
        })?;
        Vector::from_raw(centrality).to_dense_f64(self.n, 0.0)
    }

    /// Approximate betweenness centrality using the given source nodes.
    pub fn betweenness(&mut self, sources: &[u64]) -> Result<Vec<f64>> {
        self.cache_at()?;
        let mut centrality: sys::GrB_Vector = ptr::null_mut();
        let mut m = Self::msg();
        check("LAGr_Betweenness", unsafe {
            sys::LAGr_Betweenness(
                &mut centrality,
                self.raw,
                sources.as_ptr(),
                sources.len() as i32,
                m.as_mut_ptr(),
            )
        })?;
        Vector::from_raw(centrality).to_dense_f64(self.n, 0.0)
    }

    /// Connected components: `component[i]` is the representative node id of i's
    /// component. Requires an undirected (symmetric) graph.
    pub fn connected_components(&mut self) -> Result<Vec<i64>> {
        self.cache_symmetric()?;
        let mut component: sys::GrB_Vector = ptr::null_mut();
        let mut m = Self::msg();
        check("LAGr_ConnectedComponents", unsafe {
            sys::LAGr_ConnectedComponents(&mut component, self.raw, m.as_mut_ptr())
        })?;
        Vector::from_raw(component).to_dense_i64(self.n, -1)
    }

    /// Single-source shortest path lengths from `src` over an f64-weighted
    /// adjacency; `+inf` for unreachable. `delta` tunes delta-stepping.
    pub fn sssp(&mut self, src: u64, delta: f64) -> Result<Vec<f64>> {
        let mut ds: sys::GrB_Scalar = ptr::null_mut();
        check("GrB_Scalar_new", unsafe {
            sys::GrB_Scalar_new(&mut ds, sys::GrB_FP64)
        })?;
        if let Err(e) = check("GrB_Scalar_setElement_FP64", unsafe {
            sys::GrB_Scalar_setElement_FP64(ds, delta)
        }) {
            unsafe { sys::GrB_Scalar_free(&mut ds) };
            return Err(e);
        }
        let mut path: sys::GrB_Vector = ptr::null_mut();
        let mut m = Self::msg();
        let rc = unsafe {
            sys::LAGr_SingleSourceShortestPath(&mut path, self.raw, src, ds, m.as_mut_ptr())
        };
        unsafe { sys::GrB_Scalar_free(&mut ds) };
        check("LAGr_SingleSourceShortestPath", rc)?;
        Vector::from_raw(path).to_dense_f64(self.n, f64::INFINITY)
    }

    /// Total number of triangles. Requires an undirected graph; self-edges are
    /// removed first.
    pub fn triangle_count(&mut self) -> Result<u64> {
        // TriangleCount requires out_degree, nself_edges, and symmetric-structure
        // to all be cached (LAGRAPH_NOT_CACHED otherwise).
        self.delete_self_edges()?;
        self.cache_out_degree()?;
        self.cache_symmetric()?;
        self.cache_nself_edges()?;
        let mut nt: u64 = 0;
        let mut m = Self::msg();
        check("LAGr_TriangleCount", unsafe {
            sys::LAGr_TriangleCount(
                &mut nt,
                self.raw,
                ptr::null_mut(),
                ptr::null_mut(),
                m.as_mut_ptr(),
            )
        })?;
        Ok(nt)
    }

    /// The k-truss subgraph (every edge in at least `k - 2` triangles) as a
    /// boolean adjacency matrix. `k >= 3`. Undirected; self-edges removed first.
    /// (LAGraphX.)
    pub fn ktruss(&mut self, k: u32) -> Result<Matrix> {
        self.delete_self_edges()?;
        let mut c: sys::GrB_Matrix = ptr::null_mut();
        let mut m = Self::msg();
        check("LAGraph_KTruss", unsafe {
            sys::LAGraph_KTruss(&mut c, self.raw, k, m.as_mut_ptr())
        })?;
        Ok(Matrix::from_raw(c))
    }
}

impl Drop for LaGraph {
    fn drop(&mut self) {
        let mut m = Self::msg();
        unsafe { sys::LAGraph_Delete(&mut self.raw, m.as_mut_ptr()) };
    }
}

// SAFETY: like Matrix/Vector, a LAGraph_Graph handle has no thread affinity and
// may move between threads under external synchronization; not Sync.
unsafe impl Send for LaGraph {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric (undirected) boolean adjacency from undirected edges.
    fn undirected(n: u64, edges: &[(u64, u64)]) -> Matrix {
        let mut sym = Vec::new();
        for &(i, j) in edges {
            sym.push((i, j));
            sym.push((j, i));
        }
        Matrix::from_bool_edges(n, &sym).unwrap()
    }

    fn argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn bfs_levels_and_parents() {
        // directed path 0->1->2->3, node 4 isolated
        let a = Matrix::from_bool_edges(5, &[(0, 1), (1, 2), (2, 3)]).unwrap();
        let mut g = LaGraph::directed(a).unwrap();
        let (levels, parents) = g.bfs(0).unwrap();
        assert_eq!(levels, vec![0, 1, 2, 3, -1]);
        assert_eq!(parents[1], 0);
        assert_eq!(parents[2], 1);
        assert_eq!(parents[3], 2);
        assert_eq!(parents[4], -1);
    }

    #[test]
    fn pagerank_ranks_hub_highest() {
        // 0->2, 1->2: node 2 is the sink hub and should rank highest.
        let a = Matrix::from_bool_edges(3, &[(0, 2), (1, 2)]).unwrap();
        let mut g = LaGraph::directed(a).unwrap();
        let pr = g.pagerank(0.85, 1e-4, 100).unwrap();
        assert_eq!(argmax(&pr), 2);
    }

    #[test]
    fn betweenness_bridge_highest() {
        // undirected path 0-1-2: node 1 is the bridge.
        let a = undirected(3, &[(0, 1), (1, 2)]);
        let mut g = LaGraph::directed(a).unwrap(); // betweenness uses AT; symmetric input
        let bc = g.betweenness(&[0, 1, 2]).unwrap();
        assert!(bc[1] > bc[0] && bc[1] > bc[2]);
    }

    #[test]
    fn connected_components_partition() {
        // two components: {0,1} and {2,3,4}
        let a = undirected(5, &[(0, 1), (2, 3), (3, 4)]);
        let mut g = LaGraph::undirected(a).unwrap();
        let comp = g.connected_components().unwrap();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[2], comp[3]);
        assert_eq!(comp[3], comp[4]);
        assert_ne!(comp[0], comp[2]);
    }

    #[test]
    fn sssp_shortest_via_relaxation() {
        // 0->1 (1), 1->2 (2), 0->2 (10): best 0->2 is 3.
        let a = Matrix::from_weighted_edges(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)]).unwrap();
        let mut g = LaGraph::directed(a).unwrap();
        let d = g.sssp(0, 3.0).unwrap();
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 3.0);
    }

    #[test]
    fn triangle_count_one() {
        // undirected triangle 0-1-2 plus a pendant edge 2-3
        let a = undirected(4, &[(0, 1), (1, 2), (0, 2), (2, 3)]);
        let mut g = LaGraph::undirected(a).unwrap();
        assert_eq!(g.triangle_count().unwrap(), 1);
    }

    #[test]
    fn ktruss_keeps_triangle_drops_pendant() {
        // triangle 0-1-2 (each edge in 1 triangle = k-2 for k=3) plus pendant 2-3
        let a = undirected(4, &[(0, 1), (1, 2), (0, 2), (2, 3)]);
        let mut g = LaGraph::undirected(a).unwrap();
        let truss = g.ktruss(3).unwrap();
        // 3 undirected triangle edges, stored symmetrically = 6 entries; pendant gone
        assert_eq!(truss.nvals().unwrap(), 6);
    }
}
