# rustyred-log

The retained physical append-only operation log, imported as `theorem_oplog`.
Its crate root is `src/physical_lib.rs`. It owns typed records, digest-indexed
append/replay, positions, pinned prefixes, verification, provenance, proposal
commit decisions, overlap and scheduling primitives. The source includes the
v2 compatibility fixture and preserves the physical protocol version identities.

This package does not contain the old graph-backed application `TheoremOp`
projection or a Harness runtime. The engine's explicit `FoldGraphStore` composes
this physical owner with its derived folds. The `Database` facade's graph/chunk
catalog and native `.RR` file are a separate documented storage boundary; see
[storage ownership](../../docs/extraction/storage-owners.md).

An append error is precommit only after confirmed durable rollback. An
unconfirmed result poisons the handle and requires reopen/reconciliation;
replayed committed digests retain their original positions. See the public
[API failure contract](../../docs/api-contract.md) for the typed outcomes.

From the public RustyRedDB repository root:

```sh
cargo +1.96.1 test --locked -p rustyred-log
```

Native storage is the default. The wasm-only `browser-opfs` feature retains the
OPFS storage seam; it is an explicit integration and has a separate browser
acceptance requirement, not a native `.RR` implementation.
