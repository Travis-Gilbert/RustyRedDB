# rustyred-cli

The local `rustyred` command delegates to `rustyred::Database` and the same
engine used by embedded callers. It provides create, read-only query, inspect
and non-mutating verify, without a listener, account or model connection.

From the public RustyRedDB repository root:

```sh
cargo +1.96.1 run --locked -p rustyred-cli -- create notes.rr
cargo +1.96.1 run --locked -p rustyred-cli -- query notes.rr --cypher 'MATCH (n) RETURN n'
cargo +1.96.1 run --locked -p rustyred-cli -- inspect notes.rr --json
cargo +1.96.1 run --locked -p rustyred-cli -- verify notes.rr --json
```

Results use JSON on stdout and diagnostics on stderr. Writes to application
records use the typed Rust SDK. An indeterminate commit uses exit status 6 and
requires reopen/reconciliation; the CLI does not retry it automatically.

The beta.2 package family is not uploaded to crates.io. See the public
[release procedure](../../docs/releasing.md) for Git/package verification and
separate publication approval.
