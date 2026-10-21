# rustyred-graphblas

Experimental SuiteSparse:GraphBLAS and LAGraph FFI plus safe matrix, vector,
semiring and algorithm handles. The Rust import remains
`rustyred_thg_graphblas`. This leaf has no engine dependency; optional engine
integration is the `graphblas` feature of `rustyred-engine`.

Native prerequisites include a C/CMake build toolchain, libclang for bindgen,
SuiteSparse:GraphBLAS and LAGraph headers, and the `graphblas`, `lagraph` and
`lagraphx` shared libraries. SuiteSparse:GraphBLAS is Apache-2.0 and LAGraph is
BSD; their source is not copied into this package.

For an existing native installation, run from the public RustyRedDB repository
root (use a prefix without spaces):

```sh
RUSTYRED_GRAPHBLAS_PREFIX=/absolute/native-prefix \
  cargo +1.96.1 test --locked -p rustyred-graphblas
```

The prefix must contain `lib/` plus `include/` or `include/suitesparse/`.
When the prefix is unset, the explicit GraphBLAS build uses the included
`graphblas.sh` to build pinned SuiteSparse:GraphBLAS v9.4.5 and LAGraph v1.2.1
in the existing per-user cache. That path needs public-source access and the
native build tools; `RUSTYRED_GRAPHBLAS_VENDOR` can identify an already captured
source pair. The default database build does not enable this feature.

The general native beta CI matrix does not qualify GraphBLAS. Retain separate
native link/runtime receipts before claiming support for this extension on a
target. See [release preparation](../../docs/releasing.md).
