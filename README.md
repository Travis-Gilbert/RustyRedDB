# Rusty Red

Rusty Red is an embedded property-graph database. One `.RR` file is a database: create it, write records, close it, copy the file, and open it somewhere else.

```rust
use rustyred::{Database, NodeRecord};
use serde_json::json;

let mut db = Database::create("notes.rr")?;
db.upsert_node(NodeRecord::new("part:bolt", ["Part"], json!({"sku": "B-1"})))?;
```

The command line is smaller than the library:

```text
rustyred create notes.rr
rustyred query notes.rr --cypher 'MATCH (p:Part) RETURN p' --params-json '{}'
rustyred inspect notes.rr --json
rustyred verify notes.rr --json
```

Queries are read-only. Writes go through `Database::mutate` and `Database::upsert_node`. There is no account, no server, and no model. After you designate a vector width, a numeric array is stored as a content-addressed payload. The library does not turn that array into a semantic embedding.

Build with Rust 1.96.1 (`rust-toolchain.toml`). The crate manifests say 1.89; that older compiler has not been re-run as a release gate for this extract.

```text
cargo test -p rustyred --locked
cargo run -p parts-catalog -- parts.rr
```

See `docs/getting-started.md`, `docs/api-contract.md`, and `docs/storage-format.md`.

The engine crates (`rustyred-engine` and its siblings) are the database implementation. `rustyred` is the supported entry point. GraphBLAS, query saturation, HTTP, and MCP are not in this beta.
