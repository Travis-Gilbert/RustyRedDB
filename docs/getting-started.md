# Getting started

Supported native targets for this beta, once their tests have been run on that machine: `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`. A target that has not executed the storage tests is compile-only.

## Library

Add this git dependency. Do not float on a branch.

```toml
rustyred = { git = "https://github.com/Travis-Gilbert/RustyRedDB", tag = "v0.1.0-beta.1" }
```

`Database::create` makes a `.RR` file. `Database::open` requires the file to exist. `Database::open_read_only` takes no writer lock. `Database::in_memory` is process-local. `Database::create_directory` is the older directory layout (`redcore.journal` inside a folder) and is not a single file.

## CLI

From a checkout:

```text
cargo run -p rustyred-cli -- create notes.rr
cargo run -p rustyred-cli -- inspect notes.rr --json
```

JSON results go to stdout. A one-line diagnostic goes to stderr. Exit `0` is success, `2` is usage, `3` is open/create failure, `4` is a query error, `5` is a verify report that is not ok, and `6` is an indeterminate commit outcome requiring reopen and reconciliation before retry.

## Example

`examples/parts-catalog` writes two parts, one supplier, caller-supplied 3-vectors, and a small file payload.

## What opening does not do

It does not create a project, a user, a filesystem namespace, or an embedding field. It does not call a model or open a socket as part of create/open/query.
