# theorem-oplog

Typed append-only operation ledger and deterministic folds for Theorem.

## What it is

Typed, append-only operation ledger for Theorem.

Operations are the durable source. Receipts, timelines, actor diffs, and
HunkSets are deterministic projections over an [`OpRange`]. The mutable
lane-tip node is only a repairable acceleration structure, following the
existing Harness event-journal pattern.

This journal extends the graph substrate; it does not replace the Harness
run-transition ledger. Harness transitions remain the lifecycle authority
for runs, while `TheoremOp` records typed user, model, and tool effects that
can be folded across run boundaries into review surfaces.

## Build and test

```bash
cd rustyredcore_THG && cargo test -p theorem-oplog
```

Part of the `rustyredcore_THG` Cargo workspace. See the crate table in [CLAUDE.md](../../../CLAUDE.md) for how this fits the substrate. This README is generated from the crate's `Cargo.toml` description and `//!` module docs; edit those and regenerate with `scripts/gen-crate-readmes.sh`.
