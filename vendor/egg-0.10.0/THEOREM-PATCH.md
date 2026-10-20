# Theorem patch: formatting closure inference

Source: crates.io `egg` 0.10.0, MIT, copied byte-for-byte from Cargo's verified
registry cache before applying the change below.

`src/dot.rs` now returns `Ok::<(), fmt::Error>(())` from the `try_for_each`
closure used by its `Display` implementation. The pinned GPUI dependency graph
enables logging integrations that introduce multiple `From<fmt::Error>`
implementations; on the current Rust compiler, the unannotated `Ok(())` is
therefore ambiguous. The annotation selects the error type already required by
the enclosing `fmt::Result` and changes no runtime behavior.
