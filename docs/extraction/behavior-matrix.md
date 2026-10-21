# Behavior matrix

This records what the beta keeps, what it refuses, and what was not brought across. It is the inventory for the extract at `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`.

## Storage owners

| Owner | Role | Beta status |
| --- | --- | --- |
| `RedCoreGraphStore` plus `redcore.journal` | Directory database. `Database::create_directory` and `open_directory`. | Kept. Opening does not create a missing journal. |
| `RustyRed` `.RR` file | Single-file database. Magic `THEORRDB`, format 1, header 4096 bytes. | Kept. `Database::create`, `open`, and `open_read_only` compose `RustyRed` with `RedCoreGraphStore::open_with_chunk_store`. |
| `theorem_oplog` `PhysicalLog` and fold store | The physical operation ledger for explicit fold-backed consumers. | Preserved independently, matching source topology. RedCore never routed commits through it. See [storage owners](storage-owners.md). |

## Public operations

| Operation | Path | Failure |
| --- | --- | --- |
| Create `.RR` | `Database::create` | Existing path: `AlreadyExists` / `rr_already_exists`. |
| Open `.RR` | `Database::open` | Missing path: `NotFound`. Second writer: `rr_writer_held`. |
| Read-only open | `Database::open_read_only` | No writer claim. Mutations return `ReadOnly`. Verify does not rebuild indexes. |
| In memory | `Database::in_memory` | No files. |
| Read Cypher | `Database::query` | Parser, read gate, `execute_read`. Parameters are bound values. |
| Write Cypher | refused | `cypher_write_unsupported`. Writes go through `mutate`. |
| Typed write | `mutate`, `upsert_node` | Confirmed precommit refusal is `Err`; indeterminate durability returns `CommitOutcomeUnknown` / `chunk_store_journal_poisoned` and requires close/reopen/reconciliation. Known committed hook/pin cleanup does not turn publication into failure. The facade never retries. Advanced fold commits return a durable transaction with `maintenance_failure` after postappend materialization failure; trait adapters retain `committed_fold_maintenance_failed`, distinct from `fold_commit_outcome_unknown`. See [API contract](../api-contract.md). |
| Vector | `designate_vector` | Wrong length: `dimension_mismatch`. Create does not designate `File.embedding`. A designated value is stored as a content-addressed payload (`content_id` and `dimension`). |
| Control events | engine write path | Each created record also stores a `ControlEvent` node. A fresh database has none. |
| File payload | `put_file_node`, `read_file_node` | Caller bytes, content-addressed manifest. |
| Directory migration | `migrate_journal` | Source bytes stay put. Target gets a new database id. |
| Inspect / verify | `inspect`, `verify` | Verify does not write. |

## Removed or not claimed

| Item | Disposition |
| --- | --- |
| GraphBLAS / SuiteSparse | Preserved experimental optional `graphblas` extension; native prerequisites required. |
| Query saturation | Preserved optional generic `rustyred-query-saturate` library; not required by the default facade. |
| Harness loop gate in `feature_dsl` | Original assertion restored downstream against the canonical `DynamicFeature` and real harness validator. `memory_admission` independently retains captured SHA-256 goldens. |
| Automatic embeddings, project bootstrap, MCP, login, serve | Not part of create or open. |
| `object_store` Amazon S3 builder | Still compiled because the history object store names it. Local create, open, and query do not construct it. |
| crates.io publication | Not done. Names were unclaimed on 2026-10-20. That is not a reservation. |
| Compiler floor | Candidate manifests require the verified baseline Rust 1.96.1. No untested lower MSRV is claimed. |
| Theorem cutover | Prepared in a separate integration branch through canonical crate/type reexports. G10 remains dependent on actual downstream tests and the immutable candidate pin. |

## Consumer paths that stay in Theorem

`apps/rustyred-embedded` still designates embeddings and boots product configuration. That behavior was not moved. Theorem authorization, browser-origin grants, and product validators stay downstream of this library.
