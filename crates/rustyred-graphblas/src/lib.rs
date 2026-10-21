//! GraphBLAS + LAGraph FFI and safe sparse-linear-algebra layer for RustyRed.
//!
//! This is a leaf crate (no workspace dependencies). The raw FFI lives in
//! [`sys`]; a safe handle layer ([`Matrix`], [`Vector`], and -- as the layer
//! widens -- descriptors, semirings, monoids, and operators) is built on top of
//! it with RAII drop semantics. Graph integration (mapping the engine's
//! `CsrGraph`/`GraphStore` onto typed adjacency matrices, hook-driven
//! incremental updates, semiring traversal, LAGraph access methods, CFL-
//! reachability, and planner plan-nodes) lives in `rustyred-thg-core` behind its
//! optional `graphblas` feature, so the dependency edge stays acyclic
//! (core -> graphblas, never the reverse).
//!
//! Links SuiteSparse:GraphBLAS (Apache-2.0) and LAGraph (BSD); copies no
//! GPL / CC-BY-NC / SSPL source.
#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]

/// Raw bindgen FFI over `GraphBLAS.h` + `LAGraph.h`.
pub mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

mod context;
mod error;
mod lagraph;
mod matrix;
mod ops;
mod traversal;
mod types;
mod vector;

pub use context::{ensure_init, graphblas_version};
pub use error::{GrbError, Result};
pub use lagraph::LaGraph;
pub use matrix::Matrix;
pub use ops::{BinaryOp, Descriptor, Monoid, Semiring};
pub use traversal::{
    bfs_levels_from, matrix_or_assign, mxm_into, mxv_into, reachable_from, set_reachability,
    shortest_paths_min_plus, walk_counts,
};
pub use types::ElementType;
pub use vector::Vector;

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn version_is_pinned_9_4_5() {
        assert_eq!(graphblas_version(), (9, 4, 5));
    }

    #[test]
    fn matrix_vector_lifecycle() {
        let mut a = Matrix::new_bool(3, 3).unwrap();
        a.set_bool(0, 1, true).unwrap();
        a.set_bool(1, 2, true).unwrap();
        assert_eq!(a.nrows().unwrap(), 3);
        assert_eq!(a.nvals().unwrap(), 2);
        assert!(a.memory_usage().unwrap() > 0);
    }

    /// Runtime proof of the core compute path: one masked-free matrix-vector
    /// multiply over the boolean LOR_LAND semiring. With A(i,j)=edge i->j and
    /// edges 0->1, 1->2, `w = A * u` for u={2} yields the predecessors of 2,
    /// i.e. {1}. This exercises matrix build, vector build, semiring mxv, lazy
    /// (non-blocking) materialization on read, and RAII free -- end to end.
    #[test]
    fn mxv_predecessor_reachability() {
        let mut a = Matrix::new_bool(3, 3).unwrap();
        a.set_bool(0, 1, true).unwrap();
        a.set_bool(1, 2, true).unwrap();

        let mut u = Vector::new_bool(3).unwrap();
        u.set_bool(2, true).unwrap();
        let w = Vector::new_bool(3).unwrap();

        let info = unsafe {
            sys::GrB_mxv(
                w.as_raw(),                      // output
                ptr::null_mut(),                 // mask
                ptr::null_mut(),                 // accum
                sys::GrB_LOR_LAND_SEMIRING_BOOL, // semiring
                a.as_raw(),
                u.as_raw(),
                ptr::null_mut(), // descriptor
            )
        };
        assert_eq!(info, sys::GrB_Info_GrB_SUCCESS, "mxv failed: {info}");

        assert_eq!(w.nvals().unwrap(), 1);
        assert_eq!(w.get_bool(1).unwrap(), Some(true));
        assert_eq!(w.get_bool(0).unwrap(), None);
        assert_eq!(w.get_bool(2).unwrap(), None);
    }
}
