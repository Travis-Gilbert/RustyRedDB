# Downstream API and Redis repair candidate

The `984453329f99ae41a5d40e0297218b242ff5075a` qualification remains historical evidence. The implementation in this checkpoint changes production source and requires new native, distribution and actual Theorem consumer qualification. This document does not mark that qualification complete.

The actual Theorem default-feature consumer build found 17 missing API references before executing its tests. Those definitions were absent from both the committed plan pin and the committed extraction source. Existing preserved Theorem snapshots contained the real implementations: the generic logical-exchange module and tests were untracked, with companion files modified. Snapshot HEAD `9bf8f93dc74180b6d3cac1c6ba157f77ead98c42` is context only; it is not the identity of committed source blobs. Recovery uses content hashes and selective source comparison.

Recovered generic source includes bounded logical exchange, bounded chunk reads, atomic catalog compare-and-swap under the existing owner locks, and its actual projection codec/preflight dependency closure. All 12 logical-exchange tests, 10 codec tests, three journal grouped-CAS tests and one bounded-frame test are retained. The snapshot logical-exchange module SHA256 is `7bb5f95d7444662a52b7c70c00179267dd6badfe7c4e171be3b81e7fd496cc24`; normalized Rust formatting is the only whole-module adaptation. Existing journal checkpoint/rollback functions and native `.RR` source remain unchanged by this recovery.

Projection V1 remains the default. Compressed V2 is experimental, explicitly selected for a fresh empty projection; it neither changes the `.RR` format nor migrates an existing store. Model/checkpoint selection and Theorem's semantic-vector field policy stay downstream. Product request contracts, tenant-edge preparation, retained unique-claim admission and old product-ledger carrier/range validation also remain downstream, using the recovered real implementations rather than compile-only stubs.

The Redis writer had a separate reachable `LINKED_TO` bypass. Its WATCH transaction now calls the existing canonical first-version validator after reading the prior edge and allocating the candidate version, before hashing or queued writes. The production repair is seven lines. Four isolated live Valkey tests reproduced four failures before the repair; after-runtime qualification is pending at this checkpoint. Each test verifies that the endpoint's process ID belongs to its own spawned temporary server before any mutation. Ordinary nonreserved edges are covered too.

An old malformed Redis `LINKED_TO` edge without a valid first-publication stamp now refuses updates. Inferring its original stamp from its latest version would fabricate history; this repair does not backfill or migrate live data. Existing reads are unchanged.

Normal default-feature integration CI does not execute the four ignored Redis-wire tests. Their explicit live command is:

```sh
cargo +1.96.1 test --locked -p rustyred-engine --profile dev \
  --features redis-store --test redis_linked_to -- --ignored --test-threads=1
```

`valkey-server` must be on PATH. This command spawns temporary loopback servers and never accepts a configured external Redis endpoint. A native compile-only Redis feature check is not a substitute for this runtime oracle.

The 11 changed/new production and test files were scanned with Gitleaks 8.30.1 before this checkpoint: zero findings. Full new-candidate source/history/package/artifact review and final immutable Git consumer execution remain required. No tag, package upload, main-branch merge or live migration is authorized or performed by this checkpoint.
