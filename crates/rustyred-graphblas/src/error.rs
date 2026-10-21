//! Error type wrapping a `GrB_Info` return code.

use crate::sys;
use core::fmt;

pub type Result<T> = core::result::Result<T, GrbError>;

/// A failed GraphBLAS / LAGraph call: the operation name plus its `GrB_Info`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GrbError {
    pub op: &'static str,
    pub info: sys::GrB_Info,
}

impl GrbError {
    pub fn info(&self) -> sys::GrB_Info {
        self.info
    }
    pub fn code_name(&self) -> &'static str {
        info_name(self.info)
    }
}

impl fmt::Display for GrbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed: {} ({})",
            self.op,
            info_name(self.info),
            self.info
        )
    }
}

impl fmt::Debug for GrbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GrbError {{ op: {:?}, info: {} ({}) }}",
            self.op,
            info_name(self.info),
            self.info
        )
    }
}

impl std::error::Error for GrbError {}

/// Convert a `GrB_Info` into a `Result`, attaching the operation name.
#[inline]
pub(crate) fn check(op: &'static str, info: sys::GrB_Info) -> Result<()> {
    if info == sys::GrB_Info_GrB_SUCCESS {
        Ok(())
    } else {
        Err(GrbError { op, info })
    }
}

fn info_name(info: sys::GrB_Info) -> &'static str {
    match info {
        sys::GrB_Info_GrB_SUCCESS => "GrB_SUCCESS",
        sys::GrB_Info_GrB_NO_VALUE => "GrB_NO_VALUE",
        sys::GrB_Info_GrB_UNINITIALIZED_OBJECT => "GrB_UNINITIALIZED_OBJECT",
        sys::GrB_Info_GrB_NULL_POINTER => "GrB_NULL_POINTER",
        sys::GrB_Info_GrB_INVALID_VALUE => "GrB_INVALID_VALUE",
        sys::GrB_Info_GrB_INVALID_INDEX => "GrB_INVALID_INDEX",
        sys::GrB_Info_GrB_DOMAIN_MISMATCH => "GrB_DOMAIN_MISMATCH",
        sys::GrB_Info_GrB_DIMENSION_MISMATCH => "GrB_DIMENSION_MISMATCH",
        sys::GrB_Info_GrB_OUTPUT_NOT_EMPTY => "GrB_OUTPUT_NOT_EMPTY",
        sys::GrB_Info_GrB_NOT_IMPLEMENTED => "GrB_NOT_IMPLEMENTED",
        sys::GrB_Info_GrB_PANIC => "GrB_PANIC",
        sys::GrB_Info_GrB_OUT_OF_MEMORY => "GrB_OUT_OF_MEMORY",
        sys::GrB_Info_GrB_INSUFFICIENT_SPACE => "GrB_INSUFFICIENT_SPACE",
        sys::GrB_Info_GrB_INVALID_OBJECT => "GrB_INVALID_OBJECT",
        sys::GrB_Info_GrB_INDEX_OUT_OF_BOUNDS => "GrB_INDEX_OUT_OF_BOUNDS",
        sys::GrB_Info_GrB_EMPTY_OBJECT => "GrB_EMPTY_OBJECT",
        sys::GrB_Info_GxB_JIT_ERROR => "GxB_JIT_ERROR",
        _ => "GrB_<other>",
    }
}
