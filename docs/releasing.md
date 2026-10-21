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

`rustyred-egraph` explicitly activates `syn/full` as a host build dependency.
Its no-op build script exists because pinned `egglog-add-primitive` 2.0.0 parses
full expressions without declaring that feature itself. [Cargo resolver2](https://doc.rust-lang.org/cargo/reference/resolver.html) keeps
host and target features separate; unrelated workspace procedural macros must
not supply the missing requirement. The pinned evaluator is unchanged.

For a development rehearsal, choose a new output directory outside this checkout:

```sh
CARGO_TARGET_DIR=/absolute/disposable-target CARGO_INCREMENTAL=0 \
  python3 scripts/package_rehearsal.py --allow-dirty \
  --output /absolute/new-package-rehearsal
```

The script uses stable Cargo 1.96.1 `package --workspace --locked --no-verify`
with Cargo's ordinary generated package lockfiles. The original derivation
unit-test dependency on engine made per-package lock generation cyclic. All
15 real-engine fold/why behavioral tests now live in engine's integration suite,
with their actual CoreStore adapters and assertions retained; derivation no
longer depends upward on engine for tests. No production algorithm changed.
This permits ordinary package preparation and Cargo workspace publish dry-run
without excluding lockfiles. Build/runtime qualification still runs separately;
`--no-verify` skips Cargo's duplicate implicit package build only.

`cargo vendor --locked --versioned-dirs` captures the external closure. The
normalized archives are unpacked into a documented Cargo directory source,
with their Cargo checksum metadata. In a new outside-workspace crate, ordinary
version dependencies resolve through that source. The script removes credential
environment variables, gives Cargo a fresh home, disables Cargo downloads, runs
the external graph/vector/payload/history/index consumer, compile-checks all targets in every
archive except the separately qualified native GraphBLAS extension, runs the parts-catalog example twice,
installs the packaged CLI in the debug profile with the directory-source
configuration supplied explicitly (`cargo install` ignores project-local
configuration discovery), and runs create, inspect, verify and query. There are no consumer `[patch]` entries,
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

Engine unit latency fixtures run in the declared test profile (root optimization
level 1). The complete engine integration target set runs with `--profile dev
--test '*'`: its scaling oracle was calibrated with root level 0 and external
dependencies level 2. CI and `release-check` preserve both declared profiles and
run every integration target with `--no-fail-fast`, without changing numerical
ceilings. Tests that inspect process-global counters run serially. Query and
other suites keep the normal test profile. Outside-workspace consumers keep
Cargo defaults: test and dev optimization level 0, with no forwarded workspace
optimization override. They exercise the public API and do not execute the
engine's private timing fixtures. Package normalization does not inherit workspace
profile tables. See [baseline evidence](extraction/baseline.md).

CI also checks formatting of changed first-party Rust files with child-module
traversal disabled (the extraction has inherited whole-tree formatting debt),
and runs Clippy, doctests, the example, support
crates and the core/query suites. Clippy warnings inherited in the extraction
are visible; a successful Clippy command without `-D warnings` does not assert
that this large imported codebase is warning-free.

## Review artifacts

Install `cargo-about` 0.9.2 (`cargo install --locked --version 0.9.2 --features cli
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
only a full SHA and runs on both native targets. Before using that workflow, register it on the default branch and configure
required reviewers for the `release-review` environment. Read-only inspection
on 2026-10-20 found the workflow unregistered and the environment absent (404);
no protection rule is claimed. The existing registered CI workflow supports
manual dispatch from the corrected branch with a required full reviewed SHA,
asserts exact checkout in both native and measurement jobs, and leaves ordinary
push/PR refs unchanged. Its token has read-only repository access and it
prepares artifacts, with no publication credentials, registry uploads, release
creation or tag mutation. This registered CI workflow does not run Gitleaks.
Local exact-candidate source, history and actual archive scans are retained as
separate receipts. The prepared, currently unregistered `release.yml` includes
a history-scan step; its presence is not an executed CI scan.

Native CI invokes this actual release script after its feature gates, reusing
its workspace target directory. It uploads each target's optimized CLI,
exact-SHA source archive, ordinary normalized crates with generated Cargo.lock,
SHA256SUMS, provenance, notices and package evidence. Partial package evidence
can survive a failed job; it is not a complete bundle or release approval.
Support-fork tests retain all assertions; egg DOT-writing fixtures require an
ignored `vendor/egg-0.10.0/target` scratch directory. Final native artifact
qualification for exact `984453329f99ae41a5d40e0297218b242ff5075a` passed
[run 39509130713](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39509130713).
Both workflow and checked-out source are that candidate. Downloaded bundles
passed checksum, exact-source, normalized-closure, notices and secret review;
all 15 archives equal the clean local package and actual publish-prepared
archives. The delivered optimized Mac CLI passed nine lifecycle/strict-argument
operations under OS network denial. The optimized Linux binary received header
and checksum inspection; its execution is not claimed. Linux native packaged
CLI runtime is separate evidence. See the [distribution record](extraction/distribution-review.md). Historical b4/06
receipts remain separately identified and cannot qualify a future commit.

Review licenses and attribution, secret-scan results, source history intended
for publication, normalized manifests, API/storage compatibility, downstream
Theorem evidence and the full G01–G12 record before requesting publication.
A secret scanner does not establish ownership or review third-party licensing. The exact `9844533` source/history review retained seven raw heuristic matches, all public source SHA256 values in an earlier evidence document. Each value was verified against its named exact `06acfc4` Git blob. The reviewed history rerun suppresses only those exact historical commit/file/rule/line fingerprints; no broad path or hash rule is used. Actual normalized crate contents had zero findings. See the [raw counts and disposition](extraction/distribution-evidence/source-secret-scan-9844533.json) and [per-value proof](extraction/distribution-evidence/verified-source-checksum-false-positives-9844533.json). A later documentation attestation receives its own scan and cannot change the source identity of these artifacts.

## Authorized publication, later

Publication is a separate explicit action. Verify ownership/availability of
every crate name, including the support forks, immediately before publication.
Cargo 1.96.1 supports workspace publication in dependency order. Before any
separately authorized upload, run its actual nonuploading preparation:

```sh
cargo +1.96.1 publish --workspace --exclude parts-catalog \
  --dry-run --no-verify --locked
```

The clean exact `9844533` candidate passed this actual command for all 15 packages;
every proposed upload was explicitly aborted by `--dry-run`. See the
[clean receipt](extraction/distribution-evidence/publish-dry-run-9844533.json).
The historical exact06 failure and acyclic development-preparation receipts
remain separately identified; the test dependency cycle was removed while
retaining all 15 actual CoreStore behavioral oracles in engine's integration
suite. No production algorithm was replaced or test assertion weakened.
`publish` has no `--exclude-lockfile` option.
The successful dry run checks archive preparation and dependency ordering;
`--no-verify` does not establish compilation, runtime behavior, upload rights
or registry availability. Those have separate gates. Optional GraphBLAS must
also become registry-resolvable before engine; Cargo orchestrates dependent
uploads and waits for availability. Keep actual source/package checksums and
stop if any stage fails.

Record each successful upload, version and package checksum. Wait for actual
registry availability between dependents. A partial upload is partial publication.
Install the real version from crates.io in a clean credential-free environment,
run the external example and CLI, then bind the immutable release tag to that
verified commit. Never move beta.1 or any published tag, overwrite a version,
or migrate live databases as part of this process.


## Actual post-publication registry smoke (G12)

This live gate is **NOT RUN** until an authorized release is actually available
on crates.io. The helper never publishes, uploads a package, creates a tag or
changes ownership:

```sh
RUSTYRED_RUN_REGISTRY_SMOKE=1 python3 scripts/registry_smoke.py \
  --version 0.1.0-beta.2 --output /absolute/new-registry-smoke-evidence
```

Use source fixtures matching that published version. The new output directory
must be outside the checkout and have no ancestor Cargo configuration. The
helper uses fresh credential-free `HOME` and `CARGO_HOME`, an allowlisted
environment, the installed Rust 1.96.1 toolchain and an exact crates.io version
dependency for the unrelated consumer. It rejects nonregistry dependencies,
runs the full consumer test binary, and executes the actual parts example twice
to test reopen. It then uses `cargo install --locked --registry crates-io --version
=0.1.0-beta.2 rustyred-cli` into its isolated installation root, runs help/version (asserting the exact requested version),
create/inspect/verify/query on that delivered executable and checks read-only
operations preserve database bytes. All finished executables run under macOS
`sandbox-exec` network denial or Linux `sudo -n unshare --net`; Linux therefore
requires the same noninteractive namespace privilege as native CI.

Downloads/builds use the public registry; runtime network access is denied.
Actual package sources, command outputs/statuses, lock/binary/fixture hashes
and an `evidence.json` are retained. An unset opt-in variable refuses before
creating an output directory or contacting the registry. That refusal and
Python syntax were checked locally; they do not satisfy G12. Archive and Git
proofs cannot substitute for this actual registry gate. The actual registry dependency resolution and installed binary hashes are retained;
this gate uses the real install rather than substituting the directory-source rehearsal.
