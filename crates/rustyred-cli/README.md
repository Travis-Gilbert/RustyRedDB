# rustyred-cli

The local `rustyred` command delegates to `rustyred::Database` and the same
engine used by embedded callers. It provides create, read-only query, inspect
and non-mutating verify, without a listener, account or model connection.

From the public RustyRedDB repository root:

```sh
cargo +1.96.1 run --locked -p rustyred-cli -- --help
cargo +1.96.1 run --locked -p rustyred-cli -- --version
cargo +1.96.1 run --locked -p rustyred-cli -- create notes.rr
cargo +1.96.1 run --locked -p rustyred-cli -- query notes.rr --cypher 'MATCH (n) RETURN n' --params-json '{}'
cargo +1.96.1 run --locked -p rustyred-cli -- inspect notes.rr --json
cargo +1.96.1 run --locked -p rustyred-cli -- verify notes.rr --json
```

Clap validates subcommands, paths and flags before database access. Unknown
flags, extra positional arguments, missing values, empty Cypher and non-object
`--params-json` values are usage errors (exit 2). `query --help` describes query
arguments; `--params-json` defaults to `{}`.

Command results use JSON on stdout; diagnostics and invalid-usage output go to
stderr. Help and version print text to stdout and succeed without opening a
database. Writes to application records use the typed Rust SDK. An indeterminate commit uses exit status 6 and
requires reopen/reconciliation; the CLI does not retry it automatically.

The beta.2 package family is not uploaded to crates.io. See the public
[release procedure](../../docs/releasing.md) for Git/package verification and
separate publication approval.

The earlier b4 native CI and package receipts are historical library/storage
evidence. The strict CLI candidate requires its own final native and delivered binary qualification; this README does not claim that pending validation passed.
