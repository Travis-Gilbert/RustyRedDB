//! Process-wide GraphBLAS + LAGraph initialization.

use crate::sys;
use std::os::raw::c_char;
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize GraphBLAS + LAGraph exactly once for the process (thread-safe,
/// idempotent). `LAGraph_Init` initializes the underlying GraphBLAS context in
/// non-blocking mode, so this covers both libraries. Every safe constructor
/// calls this first, so callers never have to.
pub fn ensure_init() {
    INIT.call_once(|| {
        let mut msg = [0 as c_char; 256];
        let rc = unsafe { sys::LAGraph_Init(msg.as_mut_ptr()) };
        assert_eq!(rc, 0, "LAGraph_Init failed (rc={rc})");
        // Production fold replay must not depend on a host C compiler or a
        // writable JIT cache. SuiteSparse falls back to generic/prebuilt
        // kernels when the runtime C-JIT is disabled.
        let rc = unsafe {
            sys::GxB_Global_Option_set_INT32(
                sys::GxB_Option_Field_GxB_JIT_C_CONTROL,
                sys::GxB_JIT_Control_GxB_JIT_OFF as i32,
            )
        };
        assert_eq!(
            rc,
            sys::GrB_Info_GrB_SUCCESS,
            "disabling the GraphBLAS runtime C-JIT failed (rc={rc})"
        );
    });
}

/// `(major, minor, sub)` of the linked SuiteSparse:GraphBLAS, read from the
/// header constants baked in at build time.
pub fn graphblas_version() -> (u32, u32, u32) {
    (
        sys::GxB_IMPLEMENTATION_MAJOR,
        sys::GxB_IMPLEMENTATION_MINOR,
        sys::GxB_IMPLEMENTATION_SUB,
    )
}
