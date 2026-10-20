//! The one place the file-locking backend is named.
//!
//! Four files in this crate take an advisory lock on a file —
//! [`crate::journal`], [`crate::graph_store`], [`crate::zerocopy`] and
//! [`crate::rr_file::lock`] — and they all did it through `fs2::FileExt`.
//! `fs2` has no wasm32 support at all (its ten compile errors are
//! `cannot find module or crate `sys``), so the dependency is now
//! `cfg(not(target_arch = "wasm32"))` in this crate's manifest and the call
//! sites import the trait from here instead. That is the whole port: four
//! import lines change, and the locking call sites themselves do not.
//!
//! A wasm32 build has no filesystem to lock. `std::fs::File::open` compiles
//! there and fails at run time, and a browser build's only storage is its own
//! origin's OPFS, written by one runtime — so there is no second writer for a
//! lock to exclude and nothing for it to exclude it with.
//!
//! The wasm32 arm is deliberately **not** a no-op. Absent locking is not the
//! same as a satisfied lock: `rr_file::lock`'s contract is "one writable owner
//! per file, and a *named* refusal on a conflicting writable open", and a
//! `try_lock_exclusive` that silently returned `Ok(())` would turn that
//! boundary into a promise nothing keeps. So every method refuses by name, and
//! a caller learns which case it is in rather than believing it holds a lock.
//! This is the same choice [`theorem-oplog`](https://docs.rs/theorem-oplog)'s
//! `open_log_file` makes for the same reason: refuse by name rather than fail
//! inside a filesystem call that has no filesystem behind it.
//!
//! The trait is not narrowed to the three methods these call sites use —
//! `try_lock_exclusive`, `lock_shared` and `unlock` — because it exists to be
//! indistinguishable from `fs2::FileExt` at the call site, and a missing
//! method would reintroduce the `cfg` this module removes.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use fs2::FileExt;

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::FileExt;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::fs::File;
    use std::io;

    /// Every method's refusal. It names the build, the reason, and what the
    /// caller should do instead, because a caller that reaches this is either
    /// on a target that cannot honour the lock or has taken a path it should
    /// not have reached.
    const NO_FILESYSTEM: &str = "file locking is a native-only boundary in this build: wasm32 has \
                                 no filesystem to lock, and a browser build owns its store \
                                 through one runtime, so there is no second writer to exclude";

    fn unsupported() -> io::Error {
        io::Error::new(io::ErrorKind::Unsupported, NO_FILESYSTEM)
    }

    /// Mirrors [`fs2::FileExt`]'s method set and signatures, so importing this
    /// in place of `fs2::FileExt` is the only change a call site sees.
    ///
    /// Two of the five have no caller in this crate today
    /// (`lock_exclusive`, `try_lock_shared`), which dead-code analysis reports
    /// on a private trait. They stay: the point of this trait is to have
    /// fs2's shape, and dropping the methods nothing calls would put a `cfg`
    /// back into whichever call site reaches for the next one.
    ///
    /// The `io::Result` shape is kept even though no input can make these
    /// succeed: callers write `try_lock_exclusive(...)?` and match on the
    /// error, and returning anything else would force every one of them to
    /// grow a `cfg`.
    #[allow(dead_code)]
    pub(crate) trait FileExt {
        fn try_lock_exclusive(&self) -> io::Result<()>;
        fn lock_exclusive(&self) -> io::Result<()>;
        fn try_lock_shared(&self) -> io::Result<()>;
        fn lock_shared(&self) -> io::Result<()>;
        fn unlock(&self) -> io::Result<()>;
    }

    impl FileExt for File {
        fn try_lock_exclusive(&self) -> io::Result<()> {
            Err(unsupported())
        }

        fn lock_exclusive(&self) -> io::Result<()> {
            Err(unsupported())
        }

        fn try_lock_shared(&self) -> io::Result<()> {
            Err(unsupported())
        }

        fn lock_shared(&self) -> io::Result<()> {
            Err(unsupported())
        }

        fn unlock(&self) -> io::Result<()> {
            Err(unsupported())
        }
    }
}
