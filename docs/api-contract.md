# API contract

Frozen against the engine types in this extract.

| Method | Behavior |
| --- | --- |
| `Database::create` / `create_with` | Creates a `.RR` file and a writer claim. Existing path: `ErrorKind::AlreadyExists` (`rr_already_exists`). |
| `Database::open` / `open_with` | Opens an existing file for write. Missing path: `ErrorKind::NotFound` (`not_found`), and no file is created. Second writer: `rr_writer_held`. |
| `Database::open_read_only` | No claim. Mutations return `read_only`. Verify and query do not publish. |
| `Database::in_memory` | Empty graph, no file. |
| `Database::create_directory` / `open_directory` | Directory journal. Same already-exists and not-found rules, applied to `redcore.journal`. |
| `Database::query` | `parse_cypher_pest`, read classification, `execute_read`. Parameters are a `BTreeMap`, not interpolated text. Write clauses return `cypher_write_unsupported`. |
| `Database::mutate` / `upsert_node` | `RedCoreGraphStore::commit_batch` / `upsert_node`. A created record also stores a `ControlEvent` node. A fresh database has none. |
| `Database::put_file_node` / `read_file_node` | Existing file-manifest payload path. |
| `Database::designate_vector` | Existing designation. A later write of the wrong width is `dimension_mismatch`. Nothing is designated at create. A designated value is stored as a content-addressed payload reference (`content_id`, `dimension`), not left as the original JSON array. |
| `Database::verify` | `RedCoreGraphStore::verify`. It does not call `rebuild_indexes`. |
| `Database::migrate_journal` | `migrate_journal_to_rr`. Source bytes stay put. |
| `Database::engine` | The `RedCoreGraphStore`. This is the advanced API, not a second database. |

`Ok` from `mutate` means the engine published the commit. Post-commit pin cleanup does not flip that into an error. The engine test `redcore_postcommit_vector_pin_cleanup_failure_does_not_fail_durable_transaction` is the witness. This crate does not retry a failed commit.

Cypher here is the existing read dialect: `MATCH`, `WHERE`, `RETURN`, path patterns, and the unsupported-construct list from `rustyred-thg-query`. It is not full openCypher, SQL, or GraphQL. `RETURN n.id` reads a property named `id`. The node identity is on the node value (`RETURN n`, field `id`).

Full-text and spatial indexes are explicit engine APIs (`tantivy` and `s2` features, H3 by default). They are not built just because a string or a coordinate was stored.

Package version `0.1.0-beta.1` is not the file format version. The file format is version 1.
