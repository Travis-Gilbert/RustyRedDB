# Behavior matrix

This records what the beta keeps, what it refuses, and what was not brought across. It is the inventory for the extract at `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`.

## Storage owners

| Owner | Role | Beta status |
| --- | --- | --- |
| `RedCoreGraphStore` plus `redcore.journal` | Directory database. `Database::create_directory` and `open_directory`. | Kept. Opening does not create a missing journal. |
| `RustyRed` `.RR` file | Single-file database. Magic `THEORRDB`, format 1, header 4096 bytes. | Kept. `Database::create`, `open`, and `open_read_only` compose `RustyRed` with `RedCoreGraphStore::open_with_chunk_store`. |
| `theorem_oplog` `PhysicalLog` and fold store | A second durable log the engine crate links. | Moved with the crate. RedCore commits do not pass through it. The two owners are not unified in this beta. |

## Public operations

| Operation | Path | Failure |
| --- | --- | --- |
| Create `.RR` | `Database::create` | Existing path: `AlreadyExists` / `rr_already_exists`. |
| Open `.RR` | `Database::open` | Missing path: `NotFound`. Second writer: `rr_writer_held`. |
| Read-only open | `Database::open_read_only` | No writer claim. Mutations return `ReadOnly`. Verify does not rebuild indexes. |
| In memory | `Database::in_memory` | No files. |
| Read Cypher | `Database::query` | Parser, read gate, `execute_read`. Parameters are bound values. |
| Write Cypher | refused | `cypher_write_unsupported`. Writes go through `mutate`. |
| Typed write | `mutate`, `upsert_node` | Pre-commit refusal is `Err`. A hook or pin cleanup after publication stays `Ok`. The facade does not retry. |
| Vector | `designate_vector` | Wrong length: `dimension_mismatch`. Create does not designate `File.embedding`. A designated value is stored as a content-addressed payload (`content_id` and `dimension`). |
| Control events | engine write path | Each created record also stores a `ControlEvent` node. A fresh database has none. |
| File payload | `put_file_node`, `read_file_node` | Caller bytes, content-addressed manifest. |
| Directory migration | `migrate_journal` | Source bytes stay put. Target gets a new database id. |
| Inspect / verify | `inspect`, `verify` | Verify does not write. |

## Removed or not claimed

| Item | Disposition |
| --- | --- |
| GraphBLAS / SuiteSparse | Not in the beta build. Code behind the old feature flag is not compiled. |
| Query saturation | Downstream of the query crate. Not a beta dependency. |
| Harness loop gate in `feature_dsl` | Not copied. The database assertion in `memory_admission` uses captured SHA-256 goldens. |
| Automatic embeddings, project bootstrap, MCP, login, serve | Not part of create or open. |
| `object_store` Amazon S3 builder | Still compiled because the history object store names it. Local create, open, and query do not construct it. |
| crates.io publication | Not done. Names were unclaimed on 2026-10-20. That is not a reservation. |
| Rust 1.89 as a verified MSRV | Declared on engine manifests. The extract was compiled with 1.96.1 only. |
| Theorem cutover | The in-tree engine in the Theorem checkout is still the copy Theorem builds. This repository is the intended single source after that checkout pins this git revision. |

## Consumer paths that stay in Theorem

`apps/rustyred-embedded` still designates embeddings and boots product configuration. That behavior was not moved. Theorem authorization, browser-origin grants, and product validators stay downstream of this library.
