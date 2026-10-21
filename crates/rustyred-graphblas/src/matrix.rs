//! Owned `GrB_Matrix` handle with RAII drop semantics.

use crate::context::ensure_init;
use crate::error::{check, Result};
use crate::sys;
use crate::types::ElementType;
use std::ptr;

/// An owned GraphBLAS matrix. Freed with `GrB_Matrix_free` on drop.
///
/// Not `Clone` (a clone would alias one `GrB_Matrix` and double-free); use
/// GraphBLAS `dup` semantics when an independent copy is needed.
pub struct Matrix {
    pub(crate) raw: sys::GrB_Matrix,
}

impl Matrix {
    /// Create an empty `nrows x ncols` matrix of the given element type.
    pub fn new(ty: ElementType, nrows: u64, ncols: u64) -> Result<Self> {
        ensure_init();
        let mut raw: sys::GrB_Matrix = ptr::null_mut();
        check("GrB_Matrix_new", unsafe {
            sys::GrB_Matrix_new(&mut raw, ty.grb_type(), nrows, ncols)
        })?;
        Ok(Self { raw })
    }

    /// Create an empty boolean matrix (the default adjacency element type).
    pub fn new_bool(nrows: u64, ncols: u64) -> Result<Self> {
        Self::new(ElementType::Bool, nrows, ncols)
    }

    /// An `n x n` boolean identity (diagonal true): the epsilon relation in
    /// CFL-reachability grammars.
    pub fn identity_bool(n: u64) -> Result<Self> {
        let mut m = Self::new(ElementType::Bool, n, n)?;
        for i in 0..n {
            m.set_bool(i, i, true)?;
        }
        Ok(m)
    }

    /// Set `A(i, j) = x` (boolean).
    pub fn set_bool(&mut self, i: u64, j: u64, x: bool) -> Result<()> {
        check("GrB_Matrix_setElement_BOOL", unsafe {
            sys::GrB_Matrix_setElement_BOOL(self.raw, x, i, j)
        })
    }

    /// Set `A(i, j) = x` (f64), for weighted adjacency.
    pub fn set_fp64(&mut self, i: u64, j: u64, x: f64) -> Result<()> {
        check("GrB_Matrix_setElement_FP64", unsafe {
            sys::GrB_Matrix_setElement_FP64(self.raw, x, i, j)
        })
    }

    /// Read `A(i, j)` (f64); `Ok(None)` if the entry is not stored.
    pub fn get_fp64(&self, i: u64, j: u64) -> Result<Option<f64>> {
        let mut x = 0.0f64;
        let info = unsafe { sys::GrB_Matrix_extractElement_FP64(&mut x, self.raw, i, j) };
        if info == sys::GrB_Info_GrB_NO_VALUE {
            return Ok(None);
        }
        check("GrB_Matrix_extractElement_FP64", info)?;
        Ok(Some(x))
    }

    /// Number of stored (explicit) entries.
    pub fn nvals(&self) -> Result<u64> {
        let mut n: sys::GrB_Index = 0;
        check("GrB_Matrix_nvals", unsafe {
            sys::GrB_Matrix_nvals(&mut n, self.raw)
        })?;
        Ok(n)
    }

    /// Native bytes currently owned by this matrix, as reported by
    /// SuiteSparse:GraphBLAS (including format metadata and allocations).
    pub fn memory_usage(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        check("GxB_Matrix_memoryUsage", unsafe {
            sys::GxB_Matrix_memoryUsage(&mut bytes, self.raw)
        })?;
        Ok(bytes)
    }

    /// Row dimension.
    pub fn nrows(&self) -> Result<u64> {
        let mut n: sys::GrB_Index = 0;
        check("GrB_Matrix_nrows", unsafe {
            sys::GrB_Matrix_nrows(&mut n, self.raw)
        })?;
        Ok(n)
    }

    /// Column dimension.
    pub fn ncols(&self) -> Result<u64> {
        let mut n: sys::GrB_Index = 0;
        check("GrB_Matrix_ncols", unsafe {
            sys::GrB_Matrix_ncols(&mut n, self.raw)
        })?;
        Ok(n)
    }

    /// Deep copy (GraphBLAS `GrB_Matrix_dup`); the copy is independently owned.
    pub fn dup(&self) -> Result<Self> {
        let mut raw: sys::GrB_Matrix = ptr::null_mut();
        check("GrB_Matrix_dup", unsafe {
            sys::GrB_Matrix_dup(&mut raw, self.raw)
        })?;
        Ok(Self { raw })
    }

    /// Grow or shrink to `nrows x ncols`, preserving entries within bounds.
    /// Growing is how the typed adjacency layer admits newly-seen nodes.
    pub fn resize(&mut self, nrows: u64, ncols: u64) -> Result<()> {
        check("GxB_Matrix_resize", unsafe {
            sys::GxB_Matrix_resize(self.raw, nrows, ncols)
        })
    }

    /// Remove the entry at `(i, j)` if present (no-op if absent).
    pub fn remove(&mut self, i: u64, j: u64) -> Result<()> {
        check("GrB_Matrix_removeElement", unsafe {
            sys::GrB_Matrix_removeElement(self.raw, i, j)
        })
    }

    /// The `(row, col)` positions of all stored `true` entries, for comparing
    /// matrix structure against another adjacency representation.
    pub fn bool_tuples(&self) -> Result<Vec<(u64, u64)>> {
        let n = self.nvals()? as usize;
        let mut rows = vec![0u64; n];
        let mut cols = vec![0u64; n];
        let mut vals = vec![false; n];
        let mut nout: sys::GrB_Index = n as u64;
        check("GrB_Matrix_extractTuples_BOOL", unsafe {
            sys::GrB_Matrix_extractTuples_BOOL(
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                vals.as_mut_ptr(),
                &mut nout,
                self.raw,
            )
        })?;
        rows.truncate(nout as usize);
        cols.truncate(nout as usize);
        Ok(rows.into_iter().zip(cols).collect())
    }

    /// Wrap an externally-created `GrB_Matrix` (e.g. a LAGraph output). Takes
    /// ownership: the handle is freed on drop.
    pub(crate) fn from_raw(raw: sys::GrB_Matrix) -> Self {
        Self { raw }
    }

    /// Consume the wrapper and return the raw handle WITHOUT freeing it. The
    /// caller takes ownership (e.g. handing it to `LAGraph_New`, which moves the
    /// matrix into the graph).
    pub(crate) fn into_raw(self) -> sys::GrB_Matrix {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }

    pub(crate) fn as_raw(&self) -> sys::GrB_Matrix {
        self.raw
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        // GrB_Matrix_free is null-safe and idempotent; nulls the handle.
        unsafe {
            sys::GrB_Matrix_free(&mut self.raw);
        }
    }
}

// SAFETY: a GrB_Matrix is a heap handle with no thread affinity. GraphBLAS
// supports use from any thread (just not concurrently on a single object), so
// the handle may move between threads under external synchronization (e.g. a
// Mutex). We deliberately do NOT impl Sync: shared cross-thread &-access is not
// guaranteed safe.
unsafe impl Send for Matrix {}
