//! Safe element-type selector over GraphBLAS built-in `GrB_Type` globals.

use crate::context::ensure_init;
use crate::sys;

/// The element types this layer constructs matrices and vectors over. Maps to
/// GraphBLAS built-in type globals, so the public API never exposes a raw
/// `GrB_Type` pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    /// `GrB_BOOL` -- the default adjacency / reachability element type.
    Bool,
    /// `GrB_INT64` -- BFS levels/parents, counts.
    Int64,
    /// `GrB_UINT64` -- node/token index coordinates.
    Uint64,
    /// `GrB_FP32` -- MemTensor activation and compose domain.
    Fp32,
    /// `GrB_FP64` -- edge weights, min-plus shortest path, PageRank.
    Fp64,
}

impl ElementType {
    /// The built-in `GrB_Type` global. Valid only after initialization, which
    /// this performs.
    pub(crate) fn grb_type(self) -> sys::GrB_Type {
        ensure_init();
        // Built-in type globals are non-null and immutable after init.
        unsafe {
            match self {
                ElementType::Bool => sys::GrB_BOOL,
                ElementType::Int64 => sys::GrB_INT64,
                ElementType::Uint64 => sys::GrB_UINT64,
                ElementType::Fp32 => sys::GrB_FP32,
                ElementType::Fp64 => sys::GrB_FP64,
            }
        }
    }
}
