//! GraphBLAS operator handles: binary operators, monoids, semirings, and
//! descriptors.
//!
//! Each wraps a `GrB_*` handle and tracks ownership. Built-in globals (e.g.
//! `GrB_LOR_LAND_SEMIRING_BOOL`) are *borrowed* and must never be freed; user-
//! created handles are *owned* and freed on drop. Freeing a built-in is
//! undefined behavior in GraphBLAS, so the `owned` flag is the safety boundary.

use crate::context::ensure_init;
use crate::error::{check, Result};
use crate::sys;
use std::ptr;

/// Define a constructor that borrows a built-in GraphBLAS global (never freed).
macro_rules! builtin {
    ($(#[$m:meta])* $name:ident, $global:ident) => {
        $(#[$m])*
        pub fn $name() -> Self {
            ensure_init();
            // Built-in globals are non-null and immutable after init.
            Self { raw: unsafe { sys::$global }, owned: false }
        }
    };
}

/// A binary operator (built-in or user-created).
pub struct BinaryOp {
    pub(crate) raw: sys::GrB_BinaryOp,
    owned: bool,
}

impl BinaryOp {
    builtin!(/// `plus` over f64.
        plus_fp64, GrB_PLUS_FP64);
    builtin!(/// `times` over f64.
        times_fp64, GrB_TIMES_FP64);
    builtin!(/// `min` over f64.
        min_fp64, GrB_MIN_FP64);
    builtin!(/// `plus` over f32.
        plus_fp32, GrB_PLUS_FP32);
    builtin!(/// `times` over f32.
        times_fp32, GrB_TIMES_FP32);
    builtin!(/// `max` over f32.
        max_fp32, GrB_MAX_FP32);
    builtin!(/// `plus` over i64.
        plus_int64, GrB_PLUS_INT64);
    builtin!(/// `times` over i64.
        times_int64, GrB_TIMES_INT64);

    pub(crate) fn as_raw(&self) -> sys::GrB_BinaryOp {
        self.raw
    }
}

impl Drop for BinaryOp {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                sys::GrB_BinaryOp_free(&mut self.raw);
            }
        }
    }
}

/// A commutative monoid (associative binary op + identity).
pub struct Monoid {
    pub(crate) raw: sys::GrB_Monoid,
    owned: bool,
}

impl Monoid {
    builtin!(/// boolean OR monoid (identity false).
        lor_bool, GrB_LOR_MONOID_BOOL);
    builtin!(/// f64 min monoid (identity +inf).
        min_fp64, GrB_MIN_MONOID_FP64);
    builtin!(/// f64 plus monoid (identity 0).
        plus_fp64, GrB_PLUS_MONOID_FP64);
    builtin!(/// f32 plus monoid (identity 0).
        plus_fp32, GrB_PLUS_MONOID_FP32);
    builtin!(/// f32 max monoid (identity -inf).
        max_fp32, GrB_MAX_MONOID_FP32);
    builtin!(/// i64 plus monoid (identity 0).
        plus_int64, GrB_PLUS_MONOID_INT64);

    /// User-created f64 monoid from a binary op and identity element.
    pub fn new_fp64(op: &BinaryOp, identity: f64) -> Result<Self> {
        ensure_init();
        let mut raw: sys::GrB_Monoid = ptr::null_mut();
        check("GrB_Monoid_new_FP64", unsafe {
            sys::GrB_Monoid_new_FP64(&mut raw, op.as_raw(), identity)
        })?;
        Ok(Self { raw, owned: true })
    }

    pub(crate) fn as_raw(&self) -> sys::GrB_Monoid {
        self.raw
    }
}

impl Drop for Monoid {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                sys::GrB_Monoid_free(&mut self.raw);
            }
        }
    }
}

/// A semiring (add monoid + multiply op). The three named in the handoff are
/// built-ins; arbitrary semirings combine a [`Monoid`] and a [`BinaryOp`].
pub struct Semiring {
    pub(crate) raw: sys::GrB_Semiring,
    owned: bool,
}

impl Semiring {
    builtin!(/// Reachability / connectivity: boolean LOR.LAND.
        reachability_bool, GrB_LOR_LAND_SEMIRING_BOOL);
    builtin!(/// Shortest path: min-plus over f64.
        min_plus_fp64, GrB_MIN_PLUS_SEMIRING_FP64);
    builtin!(/// Path / walk counting: plus-times over f64.
        plus_times_fp64, GrB_PLUS_TIMES_SEMIRING_FP64);
    builtin!(/// Activation spread: plus-times over f32.
        plus_times_fp32, GrB_PLUS_TIMES_SEMIRING_FP32);
    builtin!(/// Competitive activation: max-plus over f32.
        max_plus_fp32, GrB_MAX_PLUS_SEMIRING_FP32);
    builtin!(/// Path / walk counting (exact integer): plus-times over i64.
        plus_times_int64, GrB_PLUS_TIMES_SEMIRING_INT64);

    /// User-created semiring from an additive monoid and a multiplicative op.
    pub fn new(add: &Monoid, mul: &BinaryOp) -> Result<Self> {
        ensure_init();
        let mut raw: sys::GrB_Semiring = ptr::null_mut();
        check("GrB_Semiring_new", unsafe {
            sys::GrB_Semiring_new(&mut raw, add.as_raw(), mul.as_raw())
        })?;
        Ok(Self { raw, owned: true })
    }

    pub(crate) fn as_raw(&self) -> sys::GrB_Semiring {
        self.raw
    }
}

impl Drop for Semiring {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                sys::GrB_Semiring_free(&mut self.raw);
            }
        }
    }
}

/// A descriptor controlling transpose, mask complement, and output replace.
/// The common traversal cases use built-in global descriptors (no allocation).
pub struct Descriptor {
    pub(crate) raw: sys::GrB_Descriptor,
    owned: bool,
}

impl Descriptor {
    /// The default (null) descriptor: no transpose, no mask complement.
    pub fn none() -> Self {
        Self {
            raw: ptr::null_mut(),
            owned: false,
        }
    }

    builtin!(/// Transpose the first input (A -> A^T).
        transpose_a, GrB_DESC_T0);
    builtin!(/// Masked forward BFS step: replace + structural-complement mask +
        /// transpose input 0, i.e. `w<!mask,replace> = A^T (+).(*) u`.
        masked_forward_step, GrB_DESC_RSCT0);
    builtin!(/// Replace + structural-complement mask, no transpose.
        masked_replace, GrB_DESC_RSC);

    pub(crate) fn as_raw(&self) -> sys::GrB_Descriptor {
        self.raw
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                sys::GrB_Descriptor_free(&mut self.raw);
            }
        }
    }
}
