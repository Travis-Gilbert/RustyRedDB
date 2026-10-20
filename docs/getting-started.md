# Getting started

Native storage was qualified on `aarch64-apple-darwin` and
`x86_64-unknown-linux-gnu` at historical revision
`b4ebfc32fe5e543ed44ff52eb86eea6b122de3f3` in [run 39493345294](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39493345294).
The subsequent strict CLI/artifact candidate requires final validation against
its own immutable SHA; an added CI job alone is not a passing receipt.

## Library

This is the latest immutable SDK baseline tested on both native targets. It predates the strict
CLI parser repair. The final candidate install pin and receipts will replace
this historical baseline after its validation; do not float on a branch.

```toml
rustyred = { git = "https://github.com/Travis-Gilbert/RustyRedDB", rev = "b4ebfc32fe5e543ed44ff52eb86eea6b122de3f3" }
```

The historical `v0.1.0-beta.1` tag remains unchanged and does not contain these candidate repairs. Registry publication is not claimed.

`Database::create` makes a `.RR` file. `Database::open` requires the file to exist. `Database::open_read_only` takes no writer lock. `Database::in_memory` is process-local. `Database::create_directory` is the older directory layout (`redcore.journal` inside a folder) and is not a single file.

## CLI

From a checkout:

```text
cargo run -p rustyred-cli -- --help
cargo run -p rustyred-cli -- create notes.rr
cargo run -p rustyred-cli -- inspect notes.rr --json
```

Command results use JSON on stdout; diagnostics and usage errors go to stderr.
Help/version print text to stdout and succeed without opening a database.
Unknown flags, extra arguments, missing values and non-object `--params-json`
are usage errors. Query parameters default to `{}`.

Exit `0` is success, `1` is an unexpected inspection/verification failure,
`2` is usage, `3` is open/create failure, `4` is a query error, `5` is a verify
report that is not ok, and `6` is an indeterminate commit outcome requiring
reopen and reconciliation before retry.

## Example

`examples/parts-catalog` writes two parts, one supplier, caller-supplied 3-vectors, and a small file payload.

## What opening does not do

It does not create a project, a user, a filesystem namespace, or an embedding field. It does not call a model or open a socket as part of create/open/query.
