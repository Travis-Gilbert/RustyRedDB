# OSS implementation completion record

Governing contract: [release-plan.md](release-plan.md), requirements R01-R10 and gates G01-G12. The existing immutable `v0.1.0-beta.1` tag is preserved. Repairs target `0.1.0-beta.2`; this record does not authorize a new public release or registry upload.

## Provenance

Plan source: Theorem `d6a7062c14c11c6838362bda307e267b2297c684`. Cursor extracted `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`, a different lineage, into library commit `bdc5c749ac29639e4b0427e161db8afd6af9beb1`; the existing release points to `f3003dcc81dfac3eddae40e124301f35402a0384`. The repair preserves subsequent generic storage work rather than reverting it. Source baseline runs and module comparisons must distinguish those revisions.

## Implemented repairs and qualification checkpoint

The exact plan baseline completed with its original failures recorded in [baseline.md](baseline.md). Source-versus-candidate synthetic fixture parity ran on the actual plan pin. The dependency/resource inventory covers 519 packages; source comparisons cover 432 files and resources.

Historical implementation `b4ebfc32fe5e543ed44ff52eb86eea6b122de3f3` passed [native CI run 39493345294](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39493345294) on macOS aarch64 and Linux x86_64, including complete engine integrations, query/source fixture, native storage/failure/crash, Clippy, features and normalized package consumers. Full public support-fork suites and the actual normalized native GraphBLAS crate also passed separately. These receipts qualify that revision; they do not qualify later code automatically.

The subsequent CLI review found that the manual parser could silently accept unknown flags. The current repair uses the already-locked Clap library and the spec-named `commands.rs`, preserving thin facade calls and stable result/error exits. New contract tests require help/version success and argument/JSON rejection before storage. Their final runtime execution is pending. CI now executes the actual clean artifact script and support-fork suites; outside consumers use Cargo defaults without imposing the engine's private timing profile.

Theorem integration remains isolated on `Travis-Gilbert/rustyred-oss-cutover`. Sixteen direct declarations and 27 affected Cargo lock roots are inventoried. Exact-pin native consumers, embedded adapter, browser worker, product command, product ledger and removal of the old generic bodies are still acceptance gates. Earlier 3e42/b44 checks and superseded b4 compilations are historical, never final runtime passes.

The new CLI/artifact revision requires one clean qualification and a corresponding Theorem repin. Exact source identities, final measurements and per-target artifacts are recorded only after execution. The gate ledger at this checkpoint is explicitly historical b4 storage/library evidence; publication remains disabled.

## Authority and external gates

No live database migration is needed or authorized. Existing directory/log deployments retain their storage topology. Public registry uploads and new release tags remain separate actions after review. G12 requires an actual registry installation after publication and cannot be satisfied by local packaging.

The Harness GraphQL/recall tools are not exposed in this session. Local source, this durable record, and exact validation logs serve as the degraded coordination path; no Harness persistence success is claimed.

## Evidence and review

[Gate ledger](gates.json) records G01–G12 with oracle/evidence classes and no allowed substitution. [Storage receipts](storage-evidence.json) distinguish development snapshots, exact native CI commits and completed source-baseline and fixture parity. [Distribution evidence](distribution-evidence/) contains scans, licenses and clean measurement data. [PR #1](https://github.com/Travis-Gilbert/RustyRedDB/pull/1) is the review boundary.

Earlier native run failures are retained: an obsolete upstream vendor manifest assertion, and an unoptimized timing run. The assertion now separately freezes encoding 1.0.0 and checks the maintained fork's release pin; the timing profile restores the original opt-level 1 without increasing the threshold. Actual normalized CLI installation also exposed Cargo's install-config discovery rule, repaired with an explicit isolated config path.

An optional Linux Sprite lane was stopped before tests: its Python pyenv shim blocked before the dependency-boundary command. No Linux pass is inferred from that runner. Native GitHub Actions is the Linux execution oracle; its logs and job outcomes remain authoritative.
