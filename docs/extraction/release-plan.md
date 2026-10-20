# Rusty Red OSS Release Implementation Plan

**Status:** Proposed release plan; no implementation or publication has been performed.
**Date:** October 14, 2026.
**Goal:** Extract the current Rusty Red database into a new, independently installable OSS repository that unrelated applications and Theorem can both consume.
**Architecture:** One canonical database implementation, a small public Rust entry point, and optional adapters. Theorem owns its product behavior and depends on Rusty Red; Rusty Red never depends on Theorem's application runtime.
**Tech stack:** Existing Rust 2021 crates, Cargo, current storage/query implementations, and existing tests. Begin with the source repository's Rust 1.96.1 toolchain, then qualify the standalone package's actual supported compiler range.
**Spec:** The release contract in sections 1–5 of this document governs the task sequence in section 6.
**Source baseline:** `Travis-Gilbert/Theorem` at `d6a7062c14c11c6838362bda307e267b2297c684`.

> For implementation agents: execute one testable task at a time. Use the installed execution-plan workflow where available. Preserve the user's scope and the source's behavior; a failing build is not permission to delete capabilities or tests. This document does not authorize creating a public repository, publishing packages, merging changes, or migrating live data.

## 1. Release definition

The first supported release is an **embedded-first Rusty Red beta**, not a new database implementation and not all of Theorem in a smaller repository.

A developer must be able to install the library into an unrelated Rust application, create a database, write and query its own records, retain data across processes, and inspect it without a Theorem account, checkout, configuration file, model provider, Redis instance, or running server.

The implementation has two useful checkpoints:

**Standalone development checkpoint:** a dependency-clean extraction works from an external consumer using the existing supported directory-backed persistence path. This is useful implementation progress, not proof of the complete beta or of single-file persistence.

**Supported public beta:** the native `.RR` graph path, package distribution, independent consumer tests, documentation, and Theorem compatibility gates all pass. The CLI and public Rust library use this same engine.

No calendar estimate is an acceptance criterion. The earlier week ranges were rough judgments, not measured delivery forecasts. Re-estimate remaining work from the actual dependency closure, baseline results, and completed first external-consumer test.

### Proposed distribution

The product name is **Rusty Red**. `Travis-Gilbert/RustyRed` is a working repository-name proposal, not an existing or reserved destination. Use a fresh repository with a reviewed, selective source import. Preserve origin and attribution in an extraction record; do not copy private repository history by default.

The preferred public crate is `rustyred`; the command is `rustyred`. Workspace paths and package names below are proposed identifiers. Before freezing them, check registry availability and the user's ownership of existing names. A GitHub repository name does not establish crates.io ownership.

Choose the first prerelease version after that check. A genuinely new package family can begin with `0.1.0-beta.1`; an existing published package requires an explicit successor/versioning decision. Do not imply compatibility with either historical public Rusty Red line solely through version numbers.

MIT is the proposed license for the user's extractable code, consistent with the inspected core manifest. Retain existing third-party and dual-license notices. Ownership and redistribution review remain release gates, not assumptions established by a Cargo license field.

## 2. Release contract

### R01 — Independent installation

The library, its build scripts, its public features, and the release test suite have no dependency on private Theorem source or credentials. Every first-party dependency is included in the standalone source distribution or publicly available through the documented distribution route.

An ordinary consumer must not reproduce Rusty Red's workspace-level Cargo patches. Build dependencies and advertised target-specific/optional dependencies count too. Source features do not hide private code from a public package archive.

### R02 — Preserve the database, remove the product coupling

Preserve the existing general-purpose property graph, graph algorithms, atomic mutation support, vector search, index/query primitives, temporal facilities, content-addressed payloads, graph history/versioning, and generic extension machinery. Existing standalone primitives may remain explicit APIs rather than becoming newly automatic services.

Do not promise complete SQL, GraphQL, Cypher, PostgreSQL, or Redis compatibility. Document the actual query dialect and tested operations. In particular, the inspected shared Cypher crate exposes read execution; writes can use the existing typed mutation API without inventing a new Cypher write language.

Do not delete a generic capability just because its code has a Theorem-related name. Conversely, do not move a whole crate merely because its name begins with `rustyred`.

### R03 — Generic startup

Opening a new database creates storage metadata, not a preconfigured Theorem workspace. It does not create a user project, initialize a filesystem namespace, designate a `File.embedding` field, select an embedding provider, call an LLM, crawl anything, or start a background agent.

Vector storage accepts caller-supplied embeddings. Automatic embedding generation is an explicit integration, not a condition of opening the database. Deterministic hash vectors must not be described as semantic model embeddings.

The caller selects its data path and configuration. The OSS library does not discover `theorem.toml`, Theorem environment credentials, or an existing personal database implicitly.

### R04 — One owner per storage and operation boundary

The Rust API, CLI, and later transport adapters delegate to the same database operations. Direct embedded calls do not route through Theorem's MCP dispatcher.

Retain the existing log, fold, graph, journal, and root-publication owners. Extraction does not introduce a replacement database underneath them. The physical operation log and its derived folds must not be collapsed into an older graph-only snapshot if that loses current semantics.

### R05 — Native persistence

The supported beta provides an end-to-end, queryable native `.RR` database: create, write, commit, close, reopen, and read in another process. A `.RR` chunk-store handle alone is insufficient.

After all writable handles are cleanly closed, copying the `.RR` file must preserve the standalone database's authoritative graph and payload state. Temporary lock information and rebuildable caches may be separate and must be documented. Any additional authoritative file required by an advertised feature means that feature has not met the single-file claim.

Theorem's existing directory/log deployments need not be migrated to `.RR` as part of this extraction. Preserve their existing storage topology while sharing the extracted implementation.

### R06 — Durability, isolation, and honest failure

An acknowledged durable commit survives reopen. Interrupted writes do not publish half a graph/payload transaction. No incompatible second writer obtains the same database through another handle or path alias. Writer ownership lasts as long as any writable clone remains alive.

Read-only access must not repair files, advance roots, initialize missing indexes by writing, or permit mutations through a reachable handle. Do not promise live cross-process reader refresh until explicitly tested; document whether reads are pinned or refreshed.

Unsupported formats and corrupt committed data produce bounded, structured failures. Verification is non-mutating by default. Post-commit maintenance failure must not be reported as though an already committed write never happened.

### R07 — Preserve identities and formats

Keep existing hash preimages, domain separators, serialized type representations, root identities, and format markers during the extraction. In particular, retain the inspected `.RR` magic `THEORRDB`, format version `1`, and header length `4096` unless a separately approved format change and migration requires otherwise.

Rename Rust packages without casually renaming persisted labels, string tags, record IDs, or protocol fields. Retain one generic owner for shared identities; Theorem re-exports those types instead of maintaining look-alike copies.

A standalone tool must not drop an unfamiliar named root during compaction or migration. Preserve it only when its complete reachability can be verified; otherwise refuse the destructive operation.

### R08 — One canonical source after cutover

The standalone repository becomes the editable owner of extracted generic code. Theorem consumes a single pinned source revision/release of that code. Temporary extraction copies are not two permanent development lines.

Compatibility re-exports and a thin downstream adapter are allowed. A second storage implementation or a facade over the still-private Theorem engine is not.

### R09 — Publish only verified claims

The release describes observed support, not source comments as proof. All advertised operations have an external-consumer test. Optional functionality is explicitly stable, experimental, or not part of the distribution.

The proposed supported beta targets are `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`. Other targets may be retained and compile-checked, but native execution and storage tests are required before claiming support. Preserve any browser/other-target compatibility needed for the Theorem cutover; do not silently regress existing consumers to simplify a native release.

### R10 — Controlled publication

The sanitized source snapshot, licenses, manifests, examples, fixtures, generated files, and package contents are reviewed before public release. Disabling a feature does not remove its source from an archive.

Repository publication, crates.io uploads, release tags, and live data migrations are separate authorized actions. Published versions are never overwritten. A failed release is corrected with a new version, not concealed by moving its tag.

## 3. Scope and ownership

### In the supported beta

The public Rust entry point; the dependency-clean database engine; existing read-query execution and typed mutations; graph/vector/index/payload/history APIs; native durability; a minimal local CLI; examples; CI; source attribution and licensing; package/release verification; and a verified Theorem downstream integration.

The CLI's first surface is deliberately small: `create`, `query`, `inspect`, and `verify`. Query is read-only for the first beta; writes use the typed SDK. CLI output supports machine-readable JSON and ordinary diagnostics. No listener, daemon, model connection, browser, or login is introduced.

Existing generic symbolic, derivation, filesystem, browser-storage, and alternate-backend modules are inventoried individually. Keep the generic implementation where it can be made independent. Do not make new, polished end-user products around those modules a beta requirement. Features retained in the release source must still have a fully public dependency closure and an honest support classification.

### Follow-on distribution work, not beta blockers

HTTP/gRPC service packaging; database-only MCP tools; PostgreSQL/RESP servers; Python/JavaScript/C bindings; Docker/Homebrew installers; hosted deployment templates; graphical administration; distributed write support; and a public browser-storage product.

No existing algorithm or database capability needs to be removed merely because its network endpoint is deferred. Generic graph retrieval can remain a library feature without bundling an agent service.

### Initial source map

All source paths below are relative to the pinned Theorem checkout. Destinations are proposals within the new repository.

| Source | Proposed ownership/action |
|---|---|
| `rustyredcore_THG/crates/rustyred-thg-core` | `crates/rustyred-engine`; retain internal module structure except where an actual extraction boundary requires change |
| `rustyredcore_THG/crates/rustyred-thg-query` | `crates/rustyred-query`; direct read-query API and existing dialect/receipt behavior |
| `rustyredcore_THG/crates/theorem-oplog` | `crates/rustyred-log`; preserve the actual `src/physical_lib.rs` crate root and its reachable modules |
| `rustyredcore_THG/crates/theorem-bayes` | `crates/rustyred-bayes`; move generic numerical/provenance helpers needed by the engine |
| `rustyredcore_THG/crates/datalog`, `egraph` | `crates/rustyred-datalog`, `crates/rustyred-egraph`; preserve current library identities through manifest aliases where useful |
| `rustyredcore_THG/crates/rustyred-thg-derivation` | `crates/rustyred-derivation`; retain generic semiring/derivation operations |
| Database-used portions of `theorem-surface-contracts` | `crates/rustyred-contracts`; extract only the shared database/sync/storage/link contracts, not UI, Shell, or application registries |
| Required generic parts of `theorem-capability-scope` | `crates/rustyred-scope`, or the contracts crate if separation is demonstrably unnecessary; one owner, not copied definitions |
| `apps/rustyred-embedded` | Reference and selective reuse for `crates/rustyred`; do not move its MCP-driven configuration, automatic embedding, or product bootstrapping wholesale |
| `rustyred-thg-query-saturate`, `rustyred-thg-graphblas` | Assess as independently optional extensions; retain required generic behavior and test advertised combinations |
| `rustyred-thg-server`, `rustyred-thg-mcp`, agent/harness/UI crates | Stay downstream in Theorem; narrow transport packages may be extracted later |
| `rustyredcore_THG/vendor/*`, root patches/config | Carry only patches reachable from the selected public dependency closure, with a working consumer-distribution strategy |

This table is not a completed dependency inventory. Task 1 establishes the exact crate, module, build-script, fixture, and target closure, including dependencies outside `rustyredcore_THG`.

## 4. Public API and compatibility decisions

Prefer a small `Database` facade over existing owners rather than a new universal operation language. Proposed entry-point responsibilities are: `create`, `open`, `open_read_only`, and `in_memory`; read-query execution; typed atomic mutation; inspection; and verification. Exact Rust signatures are frozen in Task 3 against the real existing types, before the CLI is implemented.

Advanced consumers can use the extracted low-level traits and modules. A concise introduction must not imply that all existing functionality was replaced by four methods.

Separate crate/API versioning from on-disk format versioning. A package rename alone never requires a data migration. Document supported old/new reader-writer combinations and unknown-root behavior.

Old public Rusty Red releases are historical products, not automatically compatible formats. Their import path is a separate compatibility decision. The first supported migration is the inspected current journal lineage to `.RR`, verified without modifying the source. Do not claim that this migrates an unrelated old AOF/snapshot distribution.

For Theorem, moving shared types requires changing every participating first-party consumer to the same source identity. A crate with the same name/version from two sources can still create incompatible Rust types. Manifest aliases reduce import churn; they do not repair an API mismatch or prevent duplicated dependency instances by themselves.

## 5. Global constraints and review focus

No rewrite of storage, query language, model runtime, or UI. No blanket split of large files. No program-wide rename. No fabricated replacement implementation to make a test pass. No default runtime network access. No mandatory Redis, model weights, account, daemon, or async runtime setup for ordinary local operations.

Keep existing feature boundaries initially. Introduce additional feature separation only where it removes product coupling or a concrete mandatory dependency. The default build must not require GraphBLAS/native accelerators or initialize external services. Normal native build dependencies, such as compression libraries, must be documented rather than mislabeled as absent.

Use a private staging checkout for the extraction. Do not manipulate another agent's shared worktree. Baseline and candidate runs must record revision, toolchain, flags, target, and meaningful failures. An unchanged file is not evidence that a failing test predates the extraction.

Five especially important review conditions, each assigned to a task below:

1. A fresh consumer has none of the maintainer's Cargo patches, lockfile assumptions, or private credentials — Tasks 2 and 7.
2. A read-only query or verification path tries to initialize or repair a store — Tasks 3 and 4.
3. Compaction/reopen encounters payload roots, log/fold state, or unknown domains that a graph-only test missed — Task 4.
4. Moving a product-specific validator creates a bypass for existing Theorem writes — Tasks 2 and 6.
5. A write commits, but a derived index/hook fails afterward and the caller retries it as an uncommitted operation — Tasks 3 and 4.

## 6. Implementation sequence

### Task 1 — Freeze the extraction and behavior baseline

**Requirements:** R01, R02, R07, R09, R10.
**Inputs:** Pinned source checkout and this release contract.
**Outputs:** `docs/extraction/baseline.md`, `docs/extraction/source-map.json`, `docs/extraction/dependencies.json`, `docs/extraction/behavior-matrix.md`, and `docs/extraction/release-identity.md` in the private staging project.

- [ ] Record the exact source SHA and compiler. Use the pinned snapshot, not a moving default branch. Record active consumer paths through `RedCoreGraphStore`, `.RR`, and log/fold storage separately.
- [ ] Generate Cargo metadata and dependency trees distinguishing normal, build, dev, optional, and target-specific dependencies. Include manifests, `include_*` resources, build scripts, external fixtures, workspace inheritance, and patch sources.
- [ ] Mark each selected item keep/move, downstream-only, replace-with-existing-generic-owner, or excluded-with-reason. Record every removed behavior, not just every moved file.
- [ ] Run the source core and query suites, plus relevant embedded/storage integration tests. Initial commands include `cargo +1.96.1 test --manifest-path rustyredcore_THG/Cargo.toml -p rustyred-thg-core --locked` and the corresponding `rustyred-thg-query` command. Run embedded tests through their separate manifest when the selected behavior depends on that app.
- [ ] Create synthetic baseline fixtures for graph mutation, vector payloads, full-text/spatial primitives, versioned graph state, current directory reopen, `.RR` roots, and hash/serialization parity. Never use a personal database as a public fixture.
- [ ] Verify repository/crate naming and ownership; record release package names, library import names, version policy, and publication order before editing package identities.

**Acceptance:** The inventory explains the executable dependency closure and what must remain compatible. Baseline failures have concrete logs and owning boundaries. No crate-count estimate is substituted for this work.

### Task 2 — Produce an independent engine and supporting crates

**Requirements:** R01, R02, R04, R07, R08.
**Files:** Mapped `crates/*`; root `Cargo.toml`, lockfile, toolchain/config; `tests/dependency_boundary`; and `docs/extraction/source-map.json`.
**Inputs:** Task 1's source map and frozen package identities.
**Outputs:** The moved generic engine, log, contracts, and query crates with no production dependency into Theorem.

- [ ] Add a dependency-boundary check that rejects private Theorem paths/repositories and product runtime dependencies. Prove it fails against the unmodified extraction.
- [ ] Move generic leaf crates with their actual manifests, crate roots, licenses, tests, and resources. Preserve the physical log implementation rather than selecting files by familiar names.
- [ ] Extract database-owned identity, synchronization, browser-storage, and link contracts from the surface crate into neutral owners. Theorem-side re-exports are prepared for Task 6. Preserve serialized representations and validation.
- [ ] Identify product policy in write paths. Keep genuinely generic reserved-record semantics in Rusty Red; move application-only rules to an enforced downstream preparation boundary. Tests must prove existing Theorem writers cannot bypass required checks. Do not simply remove `LINKED_TO` validation to make imports disappear.
- [ ] Replace harness-dependent test helpers with shared generic helpers or independently captured golden values. Keep the test's behavioral assertion; do not delete it because its old helper imports the harness.
- [ ] Resolve reachable patches for real consumers. Prefer a tested upstream registry version; otherwise a legally redistributable, explicitly named public fork or incorporated code with preserved notices. Do not rely on a dependency's root `[patch]` or merely add a registry version that lacks the necessary fix.
- [ ] Run core/query tests from an unrelated temporary Cargo root, without Theorem mounted and without a consumer `[patch]`. Record the actual dependency graph.

**Acceptance:** Source checkout and external consumer compile and exercise the preserved baseline. Default startup requires no server or network. A direct read/query/write/reopen works using the existing directory-backed path. This is the standalone development checkpoint, not the completed `.RR` beta.

### Task 3 — Expose a generic Rust entry point

**Requirements:** R02, R03, R04, R06.
**Files:** `crates/rustyred/src/{lib,database,options,error}.rs`, crate manifest, `tests/embedded_api.rs`, and `docs/api-contract.md`.
**Inputs:** Task 2's engine, query crate, and neutral contracts.
**Outputs:** One documented public entry point consumed by the CLI and independent example.

- [ ] Add `new_database_has_no_application_bootstrap`: only storage metadata exists after creation; no File embedding designation, project, root directory, model call, or external network activity occurs.
- [ ] Freeze the exact facade signatures and ownership model against the real core types in `docs/api-contract.md`. Prefer re-exports/delegation over duplicate node, edge, query, or mutation types.
- [ ] Provide explicit create/open/read-only/in-memory entry points. Creation refuses existing data; opening a missing existing database returns a distinct error rather than silently creating a new one.
- [ ] Route read queries through the existing parser/read gate/executor and writes through existing typed atomic mutations. Direct operations must not dispatch MCP JSON-RPC. Query parameters remain bound data, not interpolated source.
- [ ] Keep full-text, spatial, temporal, history, and payload primitives accessible. Document any required index designation or resolver installation; do not claim automatic indexing simply because the primitive exists.
- [ ] Preserve structured storage/query errors and distinguish pre-commit failure from committed-with-maintenance-failure. Do not add unbounded retries.
- [ ] Add tests for wrong vector dimensions, invalid queries, unsupported dialect constructs, malformed persisted data, read-only mutation attempts, and an empty fresh database. Run source-versus-candidate parity on the baseline fixtures.

**Acceptance:** An independent application can use the database without understanding Theorem's tenants, capabilities, provider configuration, or transport protocol. The lower-level APIs remain available where deliberately retained.

### Task 4 — Prove native `.RR` integration and safe maintenance

**Requirements:** R04–R07.
**Files:** Existing engine `rr_file`, `rr_migrate`, `journal`, `graph_store`, `chunk_store`, log/fold integration as actually needed; `tests/rr_roundtrip.rs`, `tests/rr_failures.rs`, and `docs/storage-format.md`.
**Inputs:** Task 3's facade and Task 1's actual persistence-path map.
**Outputs:** A queryable `.RR` path preserving one authoritative storage owner, and tested maintenance behavior.

- [ ] Add `rr_graph_and_payload_roundtrip_across_processes` before changing integration. Create graph records, relationships, caller-supplied vector payloads, binary/file payloads, and versioned graph state. Close every handle, copy only the `.RR` file, and verify in another process.
- [ ] Locate and reuse any existing graph-over-`.RR` connection. Where absent, add the narrow constructor/storage-injection connection to the current owner; do not implement a second writer, journal, or serializer. Map log/fold participation explicitly rather than dropping it.
- [ ] Prove second-writer refusal in the same process and another process; writable-clone lifetime; create-on-existing refusal; supported path-alias behavior; and read-only no-write behavior including initialization/repair side effects.
- [ ] Inject interrupted append, failure before root publication, root-publication/fsync failure, and corrupt committed content. Reopen through the public API and assert the named durable outcome. A failed commit with uncertain publication must not invite unsafe automatic retry.
- [ ] Test current-journal migration to a new `.RR` target with source-byte preservation, independent reopen, root/payload identity comparison, and cleanup of incomplete targets. Never migrate the user's live data in this task.
- [ ] Test compaction against graph, file/vector payloads, history, pinned readers, and named roots. Unknown reachability must cause a refusal rather than omission. Verify that no live acknowledged data becomes unreachable.
- [ ] Run the storage suite on both supported native targets. Keep API version and file-format version documented independently.

**Acceptance:** `.RR` is a functioning graph database through the actual facade, not only a file header and chunk store. Existing directory-backed deployments continue to work. A remaining authoritative sidecar or unverified log relationship is a named beta blocker, not a documentation workaround.

### Task 5 — Ship the small CLI and unrelated example

**Requirements:** R01–R06, R09.
**Files:** `crates/rustyred-cli/src/{main,commands,output}.rs`, CLI manifest; `examples/parts-catalog`; `tests/cli_contract.rs`; `tests/external-consumer`; and `docs/getting-started.md`.
**Inputs:** Frozen API from Task 3 and verified `.RR` from Task 4.
**Outputs:** A local CLI and an independently buildable example application.

- [ ] Implement `rustyred create PATH`, `rustyred query PATH --cypher QUERY --params-json JSON`, `rustyred inspect PATH --json`, and `rustyred verify PATH --json` as thin facade clients. Do not implement `serve`, an agent tool catalog, or a second query executor.
- [ ] Read/inspect/verify default to read-only. Define stable exit statuses and JSON error codes; diagnostics go to stderr, machine data to stdout. Verification does not repair unless a separate future operation explicitly requests repair.
- [ ] Build a parts-and-suppliers example using its own labels, caller-provided vectors, graph queries, and persistence. It must not depend on a Theorem schema or model-generated fixture.
- [ ] Copy the external-consumer project outside every workspace and parent Cargo configuration. Test a pinned Git dependency and the packaged distribution separately; a local path-only success is not the release proof.
- [ ] Execute the same example after closing/reopening in another process and with runtime networking disabled. Dependencies can be fetched before that offline-runtime test; offline execution is not a promise of offline first installation.

**Acceptance:** A developer follows the README without a Theorem checkout or account. CLI output and library results agree for the same database. The documented install commands identify the actual package names frozen in Task 1.

### Task 6 — Cut Theorem over without an internal fork

**Requirements:** R02, R04, R06–R09.
**Files:** A separate Theorem integration branch: affected manifests, contract re-exports, narrow product adapters, embedded wrapper, and consumer tests.
**Inputs:** The exact standalone candidate revision, plus Task 1's consumer/behavior map.
**Outputs:** Theorem resolves extracted code from one canonical source and preserves required product behavior.

- [ ] Use a local override during integration only. Final dependency changes pin the reviewed standalone Git revision or release; no dependency floats on `main`.
- [ ] Use Cargo package aliases/re-export shims where they preserve the existing import surface. Keep shared database/log/contract types sourced once throughout the graph.
- [ ] Preserve Theorem's application bootstrapping, authorization, browser-origin restrictions, and product-specific validators downstream. Genericizing database configuration is not permission to remove product protections.
- [ ] Run native query/mutation/reopen tests through actual Theorem consumers, including the embedded adapter and affected harness/server paths. Run relevant browser/other-target checks where those consumers use extracted code.
- [ ] Check dependency duplication and prove equivalent behavior against identical isolated fixtures. Do not compare implementations by performing a real external mutation twice.
- [ ] Remove generic implementation bodies from Theorem only after the replacement resolves and consumers pass. Keep documented compatibility shims where needed; they delegate and cannot become a second editable engine.

**Acceptance:** Theorem uses the same extracted database code as the unrelated consumer. Existing directory/log deployments are preserved. Integration is verified on the pinned candidate; unrelated historical failures are reported without being falsely attributed or silently suppressed.

### Task 7 — Make distribution verifiable

**Requirements:** R01, R07, R09, R10.
**Files:** `.github/workflows/{ci,release}.yml`, `scripts/release-check`, package metadata/include lists, README, LICENSE/NOTICES, SECURITY, CONTRIBUTING, CHANGELOG, `docs/{compatibility,benchmarks,releasing}.md`, and release evidence.
**Inputs:** Validated candidate, source/license inventory, and supported feature/target matrix.
**Outputs:** Release-candidate artifacts with package, consumer, and documentation evidence.

- [ ] Run normal tests, supported feature combinations, rustdoc/examples, formatting, and appropriate Clippy checks. Add targeted storage/failure tests as release gates. Do not use an unsupported all-features combination as a substitute for a documented matrix.
- [ ] Verify the claimed minimum Rust version instead of inheriting an untested manifest number. Record cold build time, peak build memory, binary size, and representative database/query measurements; do not invent performance ratios or universal thresholds.
- [ ] Inspect `cargo package --list`, normalized manifests, source archives, and release binaries. Scan source, fixtures, logs, generated bindings, comments, and repository history intended for release for secrets and unrelated private data.
- [ ] Rehearse packaging and verification of interdependent unpublished crates using the pinned Cargo version's supported workspace packaging, or an isolated temporary test registry if required. Do not upload throwaway packages to the public registry merely to unblock a dry run. Record which route actually ran.
- [ ] Prove that normalized packages and a fresh consumer work without root patches, sibling source directories, or private credentials. A package's path-plus-version dependency must resolve to the correct public implementation after publication, not silently lose a required patch.
- [ ] Prepare source/CLI release artifacts from a clean pinned revision, with checksums and provenance. CI for public contributions has read-only permissions; publishing credentials exist only in an explicitly authorized release job, not in tests or PR artifacts.
- [ ] Write documentation from executed examples. Include dialect limits, supported targets, memory/residency behavior, writer/reader semantics, recovery/backup procedure, file-format compatibility, old-public-release differences, and where users report problems.

**Acceptance:** A reviewer can associate every public claim with a specific test or measurement and the exact candidate. Packaging success is not inferred from workspace compilation. Public publication remains disabled until authorized.

### Task 8 — Publish the reviewed beta and verify the consumer path

**Requirements:** R01, R08–R10.
**Inputs:** Approved destination/visibility, release identity, and Task 7's reviewed artifacts.
**Outputs:** Public source and packages, a verified installation path, and recorded downstream integration.

- [ ] Obtain explicit approval for the destination repository's public visibility and the exact release contents. Publish only the reviewed source/attribution history; do not make the whole Theorem repository public.
- [ ] Publish supporting crates in dependency order, wait for registry resolution, and then publish the facade/CLI. Record package versions and source digests. Partial publication is recorded honestly; it is not the completed release.
- [ ] From a clean environment, install the actual public package and run the independent example and CLI verification. Repeat without maintainer credentials. Do not mark the public beta complete before this post-publication check.
- [ ] Finalize the release notes and immutable tag only against the verified candidate. If a published package is defective, stop the announcement, correct the issue, and release a new version; never overwrite or silently move the failed version.
- [ ] Land the separately reviewed Theorem cutover with the verified pin when authorized. Document where future generic fixes belong. Existing historical public repositories receive a successor notice only through separately approved edits.

**Acceptance:** A stranger can install the released software and reproduce the example, and Theorem's cutover uses the same source. No credential, unpublished dependency, or private checkout is needed.

## 7. Release evidence checklist

Each item records command or procedure, exact revision, target, toolchain, outcome, and artifact location. Empty or skipped evidence is not passing evidence.

| Gate | Required result | Owner task |
|---|---|---|
| G01 Dependency independence | Public runtime/build/advertised feature closure; standalone tests do not require private code | 1, 2, 7 |
| G02 Generic startup | No application objects, embedding provider, or network initialization | 3 |
| G03 Query/mutation parity | Supported dialect and typed writes match baseline semantics | 2, 3 |
| G04 Multimodal preservation | Graph, supplied vectors, text/spatial/time primitives, and payloads exercised through retained public APIs | 3, 5 |
| G05 Durable `.RR` roundtrip | Cross-process close/copy/reopen retains graph and referenced payloads | 4 |
| G06 Writer/read-only boundaries | Conflicting writer refused; cloned writer retains lock; read-only does not write | 4 |
| G07 Failure/recovery | Torn writes, fsync failures, corrupt committed state, and committed maintenance failure handled honestly | 4 |
| G08 Maintenance compatibility | Source-preserving migration and safe compaction including named roots/history | 4 |
| G09 External consumer | Outside-workspace Git and package consumption both run | 5, 7, 8 |
| G10 Theorem cutover | Same code/types, required product behavior retained, no internal storage fork | 6, 8 |
| G11 Publication safety | Correct licenses, attribution, sanitized archive/history, normalized manifests inspected | 1, 7, 8 |
| G12 Public installation | Actual registry install and CLI/example pass after publication | 8 |

## 8. Execution and scope control

The dependency-critical sequence is Tasks 1 → 2 → 3 → 4 → 5. Theorem integration (Task 6) can begin after Task 2 and is reverified after storage/API changes. Packaging and documentation preparation (Task 7) can also begin after Task 2; its final gate waits for Tasks 4–6. Task 8 waits for all release gates and explicit publication approval. This is not eight entirely serial workstreams.

Keep dependency/contract changes under one integration owner. Read-only dependency analysis, fixture work, documentation, and release packaging can proceed in parallel once interfaces are fixed. Do not have multiple agents independently redesign storage or shared IDs.

Commit mechanical moves separately from semantic changes. Every task receives a focused review; persistence and permission-boundary changes receive an independent review of the raw code and failure evidence, not only the implementer's summary.

Reassess scope immediately when the extraction would require a new storage format, a new query language, a new distributed runtime, or major product behavior removal. Preserve the useful standalone development checkpoint; record the concrete beta blocker and the smallest repair. Do not turn uncertainty into an unbounded cleanup campaign or quietly redefine beta completion.

The final outcome is not “a repository with Rusty Red code.” It is **one maintained database that both an unrelated application and Theorem can install and rely on, with a verifiable boundary between the database and the product.**

## 9. Source notes and limits

This plan uses read-only source inspection, not executed builds. No Cargo tests, extraction rehearsal, registry-ownership check, benchmark, live-data migration, or public publication was run in preparing it.

Pinned source observations:

- **S01:** Core manifest: `rustyredcore_THG/crates/rustyred-thg-core/Cargo.toml`. Direct production dependencies include the log, surface contracts, scope, Bayes, Datalog/egraph, and derivation crates; `theorem-harness-core` is a dev-dependency, not a production requirement.
- **S02:** Surface manifest: `rustyredcore_THG/crates/theorem-surface-contracts/Cargo.toml`. Includes AGPUI, Shell protocol, and block/body registries; moving it wholesale would preserve application coupling.
- **S03:** Log manifest: `rustyredcore_THG/crates/theorem-oplog/Cargo.toml`. Its actual library root is `src/physical_lib.rs`; the extracted code must follow that root.
- **S04:** Embedded manifest and implementation: `apps/rustyred-embedded/Cargo.toml` and `src/lib.rs`. Currently `publish = false`, with direct MCP and code-embedding dependencies; startup designates File embeddings and initializes filesystem state.
- **S05:** Native file implementation: `rustyredcore_THG/crates/rustyred-thg-core/src/rr_file/{mod,header}.rs`. `RustyRed` exposes its journal handle and catalog; header uses `THEORRDB`, format `1`, and 4096 bytes. This is not by itself evidence of the public graph-over-single-file integration.
- **S06:** Query entry point: `rustyredcore_THG/crates/rustyred-thg-query/src/lib.rs`. Exports parsing, dialect, read gating/execution, explanation, and receipts.
- **S07:** Core README: `rustyredcore_THG/crates/rustyred-thg-core/README.md`. Useful API map, not a substitute for test execution. It documents exact cosine/TurboVec rather than HNSW; full-text/spatial primitives are accessed through explicit APIs/resolvers.
- **S08:** Root workspace patches: `rustyredcore_THG/Cargo.toml`. Contains `egg`, `turbovec`, and additional product-related patches; select by actual closure, not by copying them all.
- **S09:** `rust-toolchain.toml` pins `1.96.1`. This is the extraction baseline, not a claim that the standalone dependency closure has already passed on it or on the manifest's lower minimum.

External rules checked on October 14, 2026: Cargo Book “Overriding Dependencies” (`[patch]` is taken from the consuming workspace root); “Specifying Dependencies” (Git revisions, package aliases, and registry/path dependency behavior); “cargo package” (normalized packaging and verification); and “Publishing on crates.io” (name ownership and immutable published versions). These explain the independent-consumer and publication gates, not whether this particular code already passes them.
