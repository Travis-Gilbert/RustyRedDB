# rustyred-query

The shared read-Cypher parser, read gate, executor, dialect classification and
query receipts. The Rust import remains `rustyred_thg_query`. Writes use the
typed engine mutation API; this crate does not provide a Cypher write language,
SQL, GraphQL, a server or an MCP dispatcher.

Most applications call `rustyred::Database::query`. Retained lower-level callers
can parse, classify and execute reads with the same engine types.

From the public RustyRedDB repository root:

```sh
cargo +1.96.1 test --locked -p rustyred-query
```

See the public [API contract](../../docs/api-contract.md) for supported dialect
limits and parameter binding.
