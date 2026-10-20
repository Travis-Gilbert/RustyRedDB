# Release preparation and publication

The candidate family is `0.1.0-beta.2`. The published Git tag
`v0.1.0-beta.1` remains unchanged. Package versions do not change the `.RR`
format (`THEORRDB`, version 1, 4096-byte header).

Rust 1.96.1 is the supported compiler floor for this candidate. The older
1.85/1.89 manifest numbers were inherited and had not been qualified for the
complete dependency closure. CI uses 1.96.1 on native Apple Silicon macOS and
x86-64 GNU Linux, asserts its actual host, and runs storage tests on each.
Adding a CI job is not evidence that its latest candidate run passed: retain
the run URL, commit and result before advertising that target.

## Native build prerequisites and storage operation

Native compilation needs the platform C compiler/linker for dependencies such
as zstd and ring: Xcode Command Line Tools on macOS, or the ordinary native
build toolchain on GNU Linux. The default engine does not require GraphBLAS;
its explicit experimental feature additionally needs SuiteSparse/LAGraph.

The facade maintains in-process graph/index state; it does not expose a memory
budget. A single `.RR` durability file is not a claim of constant memory use or
fully memory-mapped execution. Retained low-level residency and fold primitives
are explicit APIs. Tiny example measurements do not qualify a large-graph
working-set bound.

For a backup, close every writable handle and writable chunk-store clone,
copy the `.RR` file to a new path, then run `rustyred verify COPY.rr --json` and
read representative records from that copy. Read-only query/verification do not
repair or advance it. Readers are pinned to the catalog recovered at open;
reopen to observe later commits. Do not copy an active writer's file.

Keep an untouched copy before attempting writable recovery. Writable opening
may truncate an incomplete tail; corrupt committed data or an unknown format
can still be refused. Preserve the original bytes and the structured error
code when reporting a refusal; verification is not a repair command. For a
commit-outcome-unknown error, close/reopen and reconcile the operation identity
before retrying. See [storage format](storage-format.md) and the
[API failure contract](api-contract.md) for the separate physical-log and
known-committed maintenance outcomes.

## Package route

`rustyred-egg` and `rustyred-turbovec` are explicitly named support forks in
`vendor/`, with library names `egg` and `turbovec`. They preserve the extracted
Rust 1.96 annotation and wasm pointer-width patch. They are not the upstream
registry packages with the same API. Each support fork must be published and
reviewed with its notices; consumers must never silently resolve upstream
implementations that omit those patches.

For a development rehearsal, choose a new output directory outside this checkout:

```sh
CARGO_TARGET_DIR=/absolute/disposable-target CARGO_INCREMENTAL=0 \
  python3 scripts/package_rehearsal.py --allow-dirty \
  --output /absolute/new-package-rehearsal
```

The script uses stable Cargo 1.96.1 `package --workspace --exclude-lockfile
--no-verify` to create actual normalized `.crate` archives. The lockfile is
omitted because the derivation crate's test dependency on its engine consumer
forms a development-only cycle: packaging a per-crate lockfile otherwise asks
the public registry for an as-yet unpublished engine. No test or behavior is
removed. The complete committed workspace lock remains the release-suite pin;
package consumers generate a lock from the normalized public dependency graph.

`cargo vendor --locked --versioned-dirs` captures the external closure. The
normalized archives are unpacked into a documented Cargo directory source,
with their Cargo checksum metadata. In a new outside-workspace crate, ordinary
version dependencies resolve through that source. The script removes credential
environment variables, gives Cargo a fresh home, disables Cargo downloads, runs
the external graph/vector/payload/history/index consumer, compile-checks every
archived engine/query test target, runs the parts-catalog example twice,
installs the packaged CLI in the debug profile, and
runs create, inspect, verify and query. There are no consumer `[patch]` entries,
path dependencies, sibling source dependencies, or registry uploads. Finished
test/example/CLI executables run with OS-level network denial: `sandbox-exec`
on macOS and a fresh `sudo unshare --net` network namespace on Linux. A
sandbox failure fails the rehearsal; Cargo offline alone is not runtime proof.

This is a package-distribution rehearsal, not G12's actual crates.io install.
The evidence JSON records package checksums, archive file lists, the compiler,
source revision and whether the source was dirty. Preserve the consumer tree.
Review normalized manifests and all archive contents, including tests, fixtures,
bindings and comments; a feature flag does not exclude its source.

## Checks and supported feature matrix

Default native builds use `redb-floor`. CI checks no default backend and the
independent `fjall-floor`, `redis-store`, `tantivy`, and `s2` additions. These
compile checks do not establish optional service/runtime behavior. The facade's
`test-fault-injection` feature is diagnostic only and enabled for failure-oracle
tests. GraphBLAS is experimental, requires the named native libraries, and is
not part of the default supported matrix. Browser OPFS source is retained; it
is not a supported native `.RR` target and needs its own browser oracle.

Test execution uses the source oracle's `opt-level = 1` test profile, including
outside-workspace package tests via `CARGO_PROFILE_TEST_OPT_LEVEL=1`; package
normalization does not inherit the source workspace's profile table. Fixed
latency budgets are not loosened to compensate for an unoptimized test build.

CI also checks formatting of changed first-party Rust files with child-module
traversal disabled (the extraction has inherited whole-tree formatting debt),
and runs Clippy, doctests, the example, support
crates and the core/query suites. Clippy warnings inherited in the extraction
are visible; a successful Clippy command without `-D warnings` does not assert
that this large imported codebase is warning-free.

## Review artifacts

Install `cargo-about` 0.9.2 (`cargo install --locked --version 0.9.2
cargo-about`) for dependency notice generation. On a clean committed candidate:

```sh
CARGO_TARGET_DIR=/absolute/disposable-target CARGO_INCREMENTAL=0 \
  scripts/release-check --output /absolute/new-release-review
```

This refuses dirty source and creates normalized package archives, an executable,
a source archive from the exact Git commit, SHA256SUMS and compiler/source
provenance, plus a dependency license-text HTML bundle generated by
`cargo-about --locked --offline --workspace --all-features --fail`. Ship that
bundle beside the CLI binary. It includes selected license texts and exact
crate source-archive links, including the retained MPL-2.0 dependency source.
The SPDX branch list in `about.toml` is a notice-generation selection; it does
not establish ownership of extracted code. The manually dispatched `prepare-reviewed-release` workflow accepts
only a full SHA and runs on both native targets. Configure required reviewers
for the `release-review` environment. Its token has read-only repository access;
it scans source history and prepares artifacts, with no publication credentials,
registry uploads, release creation, or tag mutation.

Review licenses and attribution, secret-scan results, source history intended
for publication, normalized manifests, API/storage compatibility, downstream
Theorem evidence and the full G01–G12 record before requesting publication.
A secret scanner does not establish ownership or review third-party licensing.

## Authorized publication, later

Publication is a separate explicit action. Verify ownership/availability of
every crate name, including the support forks, immediately before publication.
Publish in Cargo's dependency order: leaf support/contracts/scope/Bayes/log/
egraph and GraphBLAS crates, then Datalog/derivation, engine, query/saturation,
facade, CLI. Even an optional GraphBLAS dependency must be registry-resolvable
before publishing the engine.
The derivation-to-engine edge is test-only; it does not reverse the production
order. Optional extension order must follow the recorded dependency inventory.

Record each successful upload, version and package checksum. Wait for actual
registry availability between dependents. A partial upload is partial publication.
Install the real version from crates.io in a clean credential-free environment,
run the external example and CLI, then bind the immutable release tag to that
verified commit. Never move beta.1 or any published tag, overwrite a version,
or migrate live databases as part of this process.
