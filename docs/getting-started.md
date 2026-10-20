# Getting started

The immutable implementation candidate is
`984453329f99ae41a5d40e0297218b242ff5075a`. A fresh outside-workspace public Git
consumer passed both graph/vector/payload/history and explicit full-text/spatial
runtime tests under OS network denial; see the [receipt](extraction/distribution-evidence/public-git-consumer-9844533.json).
Native macOS arm64 and GNU Linux x86-64 qualification, complete release bundles
and clean measurement passed in
[run 39509130713](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39509130713).
Downloaded bundles were verified against that exact clean source. The delivered
optimized Mac CLI also passed its real lifecycle and strict-argument checks
under OS network denial; Linux optimized binary inspection and native packaged
CLI runtime are separate receipts. See the [distribution record](extraction/distribution-review.md).

## Library

Use the immutable tested implementation pin rather than a moving branch:

```toml
rustyred = { git = "https://github.com/Travis-Gilbert/RustyRedDB", rev = "984453329f99ae41a5d40e0297218b242ff5075a" }
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
