# rustyred-derivation

Semiring trait, fold/why derivation machinery, and lift adapters for the Theorem Harness Experience Compiler.

## What it is

Derivation substrate for the Theorem Harness Experience Compiler.

Provides:
- runtime algebra descriptors plus the compatibility [`Semiring`] ABI
- law-certified Boolean, Counting, Tropical, Viterbi, and Corroboration semirings
- legacy independent noisy-OR probability aggregation without a false semiring claim
- [`ontology`] — the edge-type signature every valuation interprets
- [`dual`] — free dual-indeterminate provenance polynomials `N[X, Xbar]`
- [`interpret`](mod@crate::interpret) — the six semirings as homomorphic readouts of one polynomial
- [`fold`](mod@crate::fold) — accumulate derivation scores over a derivation graph
- [`why`](mod@crate::why) — trace derivation ancestors from a target node

## Build and test

```bash
cargo +1.96.1 test --locked -p rustyred-derivation
```

Part of the public RustyRedDB workspace. Run the command above from the repository root. The Rust library import remains `rustyred_thg_derivation`.

This beta.2 family is a release candidate; see [release preparation](../../docs/releasing.md) for the Git/package route and supported matrix.
