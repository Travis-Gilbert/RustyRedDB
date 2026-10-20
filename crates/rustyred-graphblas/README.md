# rustyred-thg-graphblas

GraphBLAS + LAGraph FFI and safe sparse-linear-algebra layer for RustyRed: matrix/vector/semiring handles, masked mxv/mxm semiring traversal, and LAGraph algorithm bindings. Links SuiteSparse:GraphBLAS (Apache-2.0) and LAGraph (BSD); copies no GPL/CC-BY-NC/SSPL code.

## What it is

GraphBLAS + LAGraph FFI and safe sparse-linear-algebra layer for RustyRed.

This is a leaf crate (no workspace dependencies). The raw FFI lives in
[`sys`]; a safe handle layer ([`Matrix`], [`Vector`], and -- as the layer
widens -- descriptors, semirings, monoids, and operators) is built on top of
it with RAII drop semantics. Graph integration (mapping the engine's
`CsrGraph`/`GraphStore` onto typed adjacency matrices, hook-driven
incremental updates, semiring traversal, LAGraph access methods, CFL-
reachability, and planner plan-nodes) lives in `rustyred-thg-core` behind its
optional `graphblas` feature, so the dependency edge stays acyclic
(core -> graphblas, never the reverse).

Links SuiteSparse:GraphBLAS (Apache-2.0) and LAGraph (BSD); copies no
GPL / CC-BY-NC / SSPL source.

## Build and test

```bash
cd rustyredcore_THG && cargo test -p rustyred-thg-graphblas
```

Part of the `rustyredcore_THG` Cargo workspace. See the crate table in [CLAUDE.md](../../../CLAUDE.md) for how this fits the substrate. This README is generated from the crate's `Cargo.toml` description and `//!` module docs; edit those and regenerate with `scripts/gen-crate-readmes.sh`.
