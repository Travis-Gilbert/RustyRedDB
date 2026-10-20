# rustyred-engine

The Rusty Red graph/storage engine, imported as `rustyred_thg_core` for source compatibility. Embedded operations require no account, Django/Python runtime, model provider, Redis server, or network listener. Most applications use the `rustyred::Database` facade; advanced callers can use these retained engine APIs directly.

It is a property-graph core plus one durable content-addressed chunk store, a graph-native File/Directory namespace, a native relational/SQL planner with pluggable access methods and rank fusion, exact vector search, BM25 full-text and H3 spatial primitives, temporal/TTL facilities, git-style versioned-graph commits, post-commit reactive hooks, a CRDT layer, stream logs, and a family of symbolic engines.

Current storage identities are SHA-256 only. A legacy Blake3 chunk lookup may
install a SHA-256-addressed compatibility alias to the canonical SHA-256
payload; that digest is migration input, not a second content identity or a
current write format.

## Graph store implementations and capabilities

`GraphStore` is the `GraphRead + GraphWrite` base bound. Optional operations are compiler-visible traits: `GraphAtomic`, `GraphTtl`, `GraphResidency`, `GraphSnapshotSource`, `GraphVerify`, and `GraphVectorPayloadAccess`. A store that lacks a capability does not implement its trait; the base bound has no `*_not_supported` defaults.

- `InMemoryGraphStore::new()` is ephemeral and in-process; it carries the inherent vector/ordered designation API.
- `RedCoreGraphStore::open(data_dir, RedCoreOptions)` is durable through the unified content-addressed journal catalog; graph roots, payload roots, and tombstones share its commit barrier. `RedCoreGraphStore::memory()` uses one shared non-durable `MemChunkStore`. `RedCoreDurability` retains `AofEverysec`, `AofAlways`, and `SnapshotOnly` as compatibility policy names, all mapped to journal publication modes; `None` remains non-durable. Open locks the directory, replays the committed journal, and ignores unreferenced crash-orphan chunks.
- Every durable open retains a bounded `RecoveryProfile`: journal open and replay, node plus label/property index assembly, edge plus adjacency/type index assembly, cold-residency filtering, each ordered/vector designation, and vector acceleration finalization. Linux samples include current and high-water RSS. `GraphMemoryAttribution` remains the post-open allocation lower-bound oracle.
- `RedisGraphStore::new(url, prefix)` / `::tenant(...)` is out-of-process. Feature-gated behind `redis-store`.

UFCS gotcha: `RedCoreGraphStore` and `RedisGraphStore` define inherent `get_node`/`upsert_node`/etc. that shadow the trait methods with owned, fallible signatures. To call the borrowed trait surface, write `<RedCoreGraphStore as GraphRead>::get_node(&store, id)` or the corresponding `GraphWrite` method.

## Modalities and key API

- Graph: `NodeRecord`, `EdgeRecord` (`confidence`, `epistemic_type`, `provenance`, `content_hash`, `parent_hashes`), `NeighborQuery`, `NodeQuery`, `GraphStats`. Algorithms (`graph.rs`): `pagerank`, `personalized_pagerank`, `connected_components`, `label_propagation_communities`, `expand_bounded[_weighted]`, `paths_shortest[_weighted]`. Cached PPR in `ppr_cache.rs` (`cached_personalized_pagerank`, keyed on `stats().version`). CSR adjacency in `graph_csr.rs`.
- Vector: `designate_vector_property(label, property, dimension)`, `vector_search(label, property, query, k)`, `hybrid_search` / `hybrid_search_with_config` (`HybridScoringConfig`). Default is exact normalized cosine. The unconditional `rustyred-turbovec` support fork retains TurboVec 1.0.0 encoding semantics: a resident `IdMapIndex` is built once a designation holds 64 points, then updated incrementally and re-ranked by exact cosine. This is not HNSW.
- Relational / SQL: `relational.rs` (`RelationalLens<G>`, `RelationDefinition`, `RelationSource`, `ResolvedRow`, graph-backed `NativeCatalog`), `access_method.rs` (`AccessMethod`, `AccessMethodRegistry::with_native_defaults`, `Predicate`, `ModalityResolver`), `planner.rs` (`QueryIr`, `execute_query`, `execute_query_with_resolver`, `compile_graphql_selection`, `FusionPolicy::{Rrf, Weighted, Cascade}`), `ranking.rs` (`RankingRule`, `apply_cascade`, `EpistemicGate`). The lens stores no rows or property values; it resolves them from its graph handle.
- Full-text: `fulltext.rs` (`FullTextIndex` wrapping the upstream `bm25` crate; `make_fulltext_backend`). A standalone primitive reached via `ModalityResolver`, not a store method. Optional `tantivy` backend.
- Spatial: `spatial.rs` (`SpatialIndex` over h3o, `radius_search`/`bbox_search`). Standalone primitive via `ModalityResolver`. Optional `s2` backend.
- Temporal: `working_log.rs` (`TemporalFact` bitemporal, `WorkingLog` cursor-resumable). Node TTL (`TTL_PROPERTY`, `purge_expired_nodes`). `TimeSeriesAccessMethod` for range predicates.
- Filesystem: File/Directory graph nodes carry raw NUL-separated `PathKey` index values and `CONTAINS` hierarchy edges; `file_payload.rs` stores bodies behind strict content-defined-chunk manifests (4 KiB minimum, 16 KiB target, 64 KiB maximum) in RedCore's unified SHA-256 chunk store. `doc_tree.rs` is a bounded, deprecated compatibility decoder used only for one-time legacy namespace import.
- Cold storage: `cold_index.rs` (`ColdIndex`, `InMemory`/`Disk`/`Ordered`), `ordered.rs` (`OrderedIndex` zset-shaped, `EvictionFrontier` coldest-first O(k log n)), `cold_fragments.rs` (columnar fragments with zone maps, `Zstandard`/`DoubleDelta`/`RunLength`). Residency changes do not bump `stats().version`.
- Images / binary: arbitrary byte bodies resolve through verified manifest and terminal-chunk addresses in the unified store (`sha256:<hex>`); no body bytes live in the graph namespace.
- Versioned graph: `versioned_graph.rs` (`build_prolly_tree`, `compile_graph_pack`, `checkout_graph_version`, `diff_graph_snapshots`, `merge_graph_snapshots`, `GraphCommit`, refs/branches default `main`).
- CRDT: `crdt/` (`join_delta`, `diff_since`, `Hlc`, `VersionVector`, `StampedBatch`, `JoinReport`).
- Streams: `stream.rs` (`StreamLog`, `StreamRegistry`, live-tail events).
- Plugins: `plugin.rs` (`RustyRedPlugin`, `PluginRegistry`, on-demand compute, the twin of hooks).
- Instant KG: `instant_kg.rs` (`HarnessInstantKg`, code KG ingest plus impact).

## Symbolic engines and hooks

- `symbolic.rs`: Datalog (`derive_datalog_receipt`, 14 built-in rules), probabilistic (`probabilistic_source_reliability`, `probabilistic_expected_value`), evolution archive (`evolution_archive`), determinism helpers (`stable_hash_json`, `canonical_json`). `epistemic.rs`: e-graph dedup (egg) plus the epistemic enrichment suite (`structural_epistemic_pass`, `run_epistemic_cron_pass`).
- `hooks.rs`: post-commit reactive compute. `HookDispatcher`, `HookHandler`, `HookRegistration`, `MutationEvent`/`MutationKind`, `coalesce_per_id`. Off the writer's critical path (bounded queue), idempotent and coalesced, fail-open (handler errors caught and skipped), depth-bounded. Plain `std::thread` worker (tokio-free). `RedCoreGraphStore` wires it via `attach_hook_emitter`.

## Command/executor layer

`executor.rs` (`ThgExecutor`, `InMemoryThgExecutor`, `execute_request_json`), `commands.rs` (`ThgCommand`/`ThgRequest`/`ThgResponse`), `state.rs` (`ThgState`, `stable_hash`), `errors.rs` (`ThgError`/`ThgResult`).

## Features

`default = ["redb-floor"]`. `redis-store` (Redis backend; `redis 0.27`), `tantivy` (alternative full-text), `s2` (alternative spatial). `rustyred-turbovec` is an unconditional support-package dependency (library `turbovec`); its package version follows the beta family while its encoding version stays 1.0.0. Always-on deps include `bm25`, `egg`, `h3o`, `imbl`, `roaring`, `sha2`, `zstd`, `fs2`.

## Build and test

```bash
cargo +1.96.1 test --locked -p rustyred-engine
cargo +1.96.1 test --locked --no-run -p rustyred-engine   # compile-only coherence
```

Large offline test suite across `src/` unit modules and `tests/` (`document_tier_acceptance`, `epistemic_acceptance`, `hook_dispatch`, `planner_multimodal_acceptance`, `prolly_incremental_commit`, `redcore_snapshot_replay_parity`, `relational_core_acceptance`, and others). Explicitly ignored or environment-gated live tests retain their separate prerequisites; source compilation does not qualify those services.

Run these commands from the public RustyRedDB repository root. See [release preparation](../../docs/releasing.md) for the supported native/optional feature matrix and [storage ownership](../../docs/extraction/storage-owners.md) for the graph and physical-log boundaries.
