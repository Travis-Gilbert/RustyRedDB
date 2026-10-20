# OSS implementation completion record

Contract: [release-plan.md](release-plan.md), requirements R01-R10 and gates G01-G12. The original `v0.1.0-beta.1` tag is unchanged. Repairs prepare unpublished `0.1.0-beta.2`; publication and landing the separate Theorem branch require authorization.

## Source and review boundary

The immutable implementation candidate is `984453329f99ae41a5d40e0297218b242ff5075a`. [PR #1](https://github.com/Travis-Gilbert/RustyRedDB/pull/1) carries the OSS repairs; [Theorem PR #865](https://github.com/Travis-Gilbert/Theorem/pull/865) carries the canonical-source cutover. The candidate includes the corrected notice-generator bootstrap and removes the development-only derivation-to-engine publication cycle. All 15 real-store derivation tests move to an engine integration target with their assertions and adapters preserved. Earlier `06acfc4` runtime results and the `0bc679d` workflow bootstrap are historical evidence; they do not substitute for final candidate execution.

The plan pin is Theorem `d6a7062c14c11c6838362bda307e267b2297c684`. Cursor extracted `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de` into `bdc5c749ac29639e4b0427e161db8afd6af9beb1`; beta.1 resolves to `f3003dcc81dfac3eddae40e124301f35402a0384`. The repair retains subsequent generic storage work and compares it with the actual plan pin. It does not silently substitute the later lineage as the baseline.

## Response to the supplied review

| Finding | Implementation and evidence | Result |
| --- | --- | --- |
| SR-001, public dependency closure | Explicit public support forks, transitive dependency boundary checks and normalized archive compilation. No consumer root patch is required. | Verified G01 |
| SR-002, preserved generic modules | Graph, query, vectors, binary payloads, history, physical log, optional GraphBLAS and query saturation retained. Source-map and module parity identify ownership changes. | Verified G03 |
| SR-003, query dialect | Existing parser/read gate/executor and typed writes retained; dialect and structured refusal documented. | Verified G03 |
| SR-004, generic startup | Public creation oracle checks storage metadata only; independent runtime executes with networking denied. | Verified G02 |
| SR-005, storage ownership | Facade injects the existing `.RR` chunk owner into RedCore. Physical log/fold topology is preserved explicitly; it was never the RedCore journal. | Verified G05 |
| SR-006, cross-process `.RR` | Only the closed file is copied; another process verifies graph, relationships, supplied vectors, exact binary payloads and versioned history. | Verified G05 |
| SR-007, directory persistence | Existing create/open directory path retained and exercised. | Verified G05 |
| SR-008, isolation | Same/other-process writer refusal, writable-clone lifetime, path aliases and byte-unchanged read-only opens exercised. | Verified G06 |
| SR-009, honest recovery | Rejection, unknown publication and known committed maintenance failure have distinct outcomes. Public reopen, native injected failures, physical log rollback and subprocess crash tests exercise them. | Verified G07 |
| SR-010, file identity | `THEORRDB`, format 1 and 4096-byte header unchanged. | Verified G08 |
| SR-011, maintenance | Migration preserves source bytes; compaction refuses unknown named-root reachability before writing and preserves acknowledged content/history. | Verified G08 |
| SR-012, one Theorem source | Sixteen direct manifest edges pin the same implementation revision; dynamic inventory checks 27 affected Cargo lock roots and shared Rust type identity. Generic bodies are removed only after consumer execution passes. | Pending final Theorem consumer execution |
| SR-013, target claims | Storage/runtime gates run on aarch64 macOS and x86_64 Linux. Browser worker compilation and experimental native GraphBLAS qualification are separately scoped. | Verified both native targets |
| SR-014, publication safety | Original tag preserved, source/history/archives scanned, package licenses and attribution prepared, no live migration. Exact clean source/binary/crate artifacts carry checksums and provenance. | Verified release preparation G11 |
| SR-015, actual source baseline | Exact plan-pin suites executed. Failures and the pre-compilation embedded lock portability refusal remain recorded; successful baseline evidence is not fabricated. | Verified baseline capture |
| SR-016, dependency inventory | Cargo inventory covers 519 packages, normal/build/dev/optional/target dependencies, build scripts and resources; 433 candidate comparisons plus reverse source-coverage and separate contract-selection audits include the explicit test-ownership repair. | Verified inventory |
| SR-017, retained harness assertion | Original `feature_config_delta_can_be_rejected_by_substrate_gate` moved downstream, using canonical DynamicFeature and the real harness validator. Generic memory admission retains captured goldens. | Pending final Theorem consumer execution |
| SR-018, `LINKED_TO` enforcement | Generic implementation retained; downstream validator and authenticated graph-edit admission checks exercise product protection. | Pending final Theorem consumer execution |
| SR-019, unrelated consumers | Exact pinned Git and actual normalized packaged consumers execute separately, without private code or root patches. | Verified G09 |
| SR-020, distribution gates | Rust 1.96.1 is the verified floor. CI, Clippy, feature matrix, measured clean build/example behavior, artifact preparation and an env-gated actual-registry smoke are implemented. Standard workspace publication dry run prepares all 15 packages without uploads; public support-fork suites remain enforced. | Verified release preparation G11 |
| SR-021, registry installation | Requires actual publication and then a fresh credential-free public-registry installation; Git/package rehearsal cannot substitute. | Awaiting publication authorization |

## Test interpretation

The source baseline is not all green. The source core unit run passed 832 with four failures and one inherited ignored test; query passed 81. Integration failures exposed pre-existing ACK framing, hook dispatch scheduling and an outdated aggregate-vector assertion. The candidate repairs those test boundaries while preserving the assertions and production storage semantics. See [baseline](baseline.md) and [storage receipts](storage-evidence.json).

Final candidate CI passed on both supported native targets, including actual package consumers and artifact preparation. Linux's full integration set passed 322 tests across 27 targets, with eight declared ignored tests; all 15 relocated derivation tests passed. Exact per-target counts, command scopes and ignored-test dispositions live in the storage receipt. Delivered source/CLI/crate bundles were independently verified; the optimized Mac CLI ran under network denial. Linux optimized-binary evidence is checksum/header inspection, separate from its native packaged-CLI runtime proof.

The source-versus-candidate fixture captures mutation serialization, persisted node versions and content hashes, relationships, vectors, binary CAS payloads, history, query AST/parameters/results, and explicit BM25/H3 primitives. It was executed on the actual plan-pin source, then executed unchanged against the candidate. This is distinct from source-file comparison. Historical development runs and cancelled duplicate local compilations are retained as historical evidence, never counted as final execution.

Timing tests retain their ceilings. Their actual source profile was recovered before calibration: latency unit checks use test root opt-level 1, and the complete integration set uses dev root opt-level 0 with dependency opt-level 2. Serial execution prevents tests from stealing process-global counters or injected faults. Native failure injection wraps real file operations; explicitly named synthetic helper stores cover only unit error paths.

## Remaining authority

No production database migration, main-branch merge, new release tag or registry upload was performed. G12 stays unsatisfied until separately authorized publication and post-publication installation. The public beta must not be announced as complete before that gate.

Harness GraphQL/recall tools were unavailable in this session. These durable source/evidence records are the degraded coordination path; no Harness persistence is claimed. External PR comments did not provide a substantive review; the PRs remain the review boundary.
