# rustyred-thg-derivation

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
cd rustyredcore_THG && cargo test -p rustyred-thg-derivation
```

Part of the `rustyredcore_THG` Cargo workspace. See the crate table in [CLAUDE.md](../../../CLAUDE.md) for how this fits the substrate. This README is generated from the crate's `Cargo.toml` description and `//!` module docs; edit those and regenerate with `scripts/gen-crate-readmes.sh`.
