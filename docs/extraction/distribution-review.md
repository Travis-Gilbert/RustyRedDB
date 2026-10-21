# Distribution qualification

Implementation candidate: `984453329f99ae41a5d40e0297218b242ff5075a`.
The later documentation attestation does not change the source identity of
these archives or binaries. No crate publication, tag change, main merge or
live database migration is part of this qualification.

## Independent completed preparation

- [Actual Cargo publication dry run](distribution-evidence/publish-dry-run-9844533.json): clean exact candidate, all 15 packages prepared with ordinary generated Cargo.lock files; every proposed upload aborted by `--dry-run`.
- [Ordinary local package archives](distribution-evidence/local-archives-9844533.json): all 15 actual `.crate` contents, clean Git identity and SHA256. These are byte-identical to the [actual publication-prepared archives](distribution-evidence/publish-prepared-archives-9844533.json).
- [Outside-workspace public Git consumer](distribution-evidence/public-git-consumer-9844533.json): both real graph/vector/binary/history and full-text/spatial tests passed with the exact Git pin, default consumer profiles, fresh HOME, public-only credential-free Cargo cache and OS network denial.
- [Native GraphBLAS package](distribution-evidence/native-graphblas-9844533.json): all 15 real library tests passed from the actual normalized archive under macOS network denial, using the recorded SuiteSparse/LAGraph installation. [Engine feature check](distribution-evidence/engine-graphblas-9844533.json) passed separately. This is experimental aarch64 macOS qualification only.
- [Clean GNU Linux measurement](distribution-evidence/clean-measurement-9844533.json): empty target, unchanged clean source before/after; see [measurement limits](../benchmarks.md).
- [Source/history secret review](distribution-evidence/source-secret-scan-9844533.json): seven raw heuristic matches, all individually verified public source checksums; unresolved zero. [Actual local crate scans](distribution-evidence/archive-secret-scan-9844533.json) had zero findings. Raw reports and exact per-value proof are retained.

## Delivered bundles

[Run 39509130713](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39509130713)
completed successfully on both native targets and its independent clean
measurement job. [Artifact IDs, GitHub digests and package-evidence hashes](distribution-evidence/downloaded-artifact-binding-9844533.json)
identify the downloaded bundles.

- [Mac bundle verification](distribution-evidence/macos-artifacts-9844533.json) and [Linux bundle verification](distribution-evidence/linux-artifacts-9844533.json): each 19 checksums, all 602 exact Git source files, 15 ordinary-lock registry-only crate archives, 14 independent archive target checks, and completed actual normalized consumer/example/installed CLI runtime evidence under OS network denial. Consumer profiles use Cargo defaults, without workspace patches or private sibling sources.
- [Mac archive binding](distribution-evidence/macos-archive-proof-binding-9844533.json) and [Linux archive binding](distribution-evidence/linux-archive-proof-binding-9844533.json): all 15 archive hashes match across both targets, local clean packaging and actual publication dry-run preparation. The separately executed experimental native GraphBLAS archive is the same delivered byte sequence. Notices also match the retained all-feature generation.
- [Actual delivered optimized Mac CLI](distribution-evidence/macos-delivered-optimized-cli-9844533.json): nine operations passed with fresh HOME and network denial: help/version, create/inspect/verify/query, and unknown-flag refusal on create/inspect/query. Rejected create produced no file; reads/refusals preserved `.RR` bytes.
- [Delivered optimized Linux binary inspection](distribution-evidence/linux-optimized-binary-inspection-9844533.json): ELF x86-64 headers, provenance and checksum passed. **Execution of the delivered optimized Linux binary is not claimed.** Actual native Linux CLI tests and installed normalized-package debug CLI lifecycle passed separately in CI.
- [Mac delivered-byte secret review](distribution-evidence/macos-delivered-secret-scan-9844533.json) and [Linux delivered-byte secret review](distribution-evidence/linux-delivered-secret-scan-9844533.json): each actual 15-crate contents scan had zero findings; each actual source archive had the same seven raw public-checksum heuristic matches, independently verified against named immutable Git blobs, with zero unresolved secrets. Raw redacted reports are retained; no suppression was applied to these archive scans.

These receipts identify the immutable implementation candidate. The subsequent
documentation/consumer-template attestation is scanned separately and does not
relabel its changed tree as the source used to produce these artifacts.

## Publication boundary

All 15 names still require a fresh registry availability and authenticated
ownership check at separately authorized publication. The prepared
[registry smoke helper](../releasing.md#actual-post-publication-registry-smoke-g12)
uses actual versioned crates.io sources and installation, with runtime network
denial. That live post-publication G12 gate has not run; local archives and Git
consumers cannot satisfy it.
