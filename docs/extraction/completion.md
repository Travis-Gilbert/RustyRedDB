# OSS implementation completion record

Governing contract: [release-plan.md](release-plan.md), requirements R01-R10 and gates G01-G12. The existing immutable `v0.1.0-beta.1` tag is preserved. Repairs target `0.1.0-beta.2`; this record does not authorize a new public release or registry upload.

## Provenance

Plan source: Theorem `d6a7062c14c11c6838362bda307e267b2297c684`. Cursor extracted `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`, a different lineage, into library commit `bdc5c749ac29639e4b0427e161db8afd6af9beb1`; the existing release points to `f3003dcc81dfac3eddae40e124301f35402a0384`. The repair preserves subsequent generic storage work rather than reverting it. Source baseline runs and module comparisons must distinguish those revisions.

## Implemented repairs and active verification

1. The exact plan baseline is isolated and running. Dependency/resource inventory covers 519 packages and all dependency kinds. Both source lineages are explicit in `source-parity.json`.
2. Neutral shared contracts have one owner; optional GraphBLAS/query saturation sources are restored. The original harness gate assertion lives downstream, and the prior-pack product command remains in Theorem.
3. Public `.RR` graph/vector/file/history copy/reopen and failure tests pass on both native targets at 3e42. Journal and physical-log uncertainty is typed separately from rejection; known committed fold-maintenance failures retain their committed position. The on-disk formats are unchanged.
4. Theorem integration is isolated on `Travis-Gilbert/rustyred-oss-cutover`. Five Cargo roots resolve one canonical Git source. Product contract identity and the retained Bayes CLI pass; other consumer checks are active. Generic source bodies are removed only after acceptance.
5. Normalized archives and their CLI/consumer are exercised outside a workspace with fresh Cargo configuration and OS-denied runtime networking. Clean Linux build/example measurements passed at d14 (see `distribution-evidence/clean-measurement-d14.json`). Final clean artifacts and native suites remain explicit gates.

## Authority and external gates

No live database migration is needed or authorized. Existing directory/log deployments retain their storage topology. Public registry uploads and new release tags remain separate actions after review. G12 requires an actual registry installation after publication and cannot be satisfied by local packaging.

The Harness GraphQL/recall tools are not exposed in this session. Local source, this durable record, and exact validation logs serve as the degraded coordination path; no Harness persistence success is claimed.

## Evidence and review

[Gate ledger](gates.json) records G01–G12 with oracle/evidence classes and no allowed substitution. [Storage receipts](storage-evidence.json) distinguish development snapshots, exact native CI commits and pending source-baseline parity. [Distribution evidence](distribution-evidence/) contains scans, licenses and clean measurement data. [PR #1](https://github.com/Travis-Gilbert/RustyRedDB/pull/1) is the review boundary.

The first native run's two failures are retained: an obsolete upstream vendor manifest assertion, and an unoptimized timing run. The assertion now separately freezes encoding 1.0.0 and checks the maintained fork's release pin; the timing profile restores the original opt-level 1 without increasing the threshold. Actual normalized CLI installation also exposed Cargo's install-config discovery rule, repaired with an explicit isolated config path.

An optional Linux Sprite lane was stopped before tests: its Python pyenv shim blocked before the dependency-boundary command. No Linux pass is inferred from that runner. Native GitHub Actions is the Linux execution oracle; its logs and job outcomes remain authoritative.
