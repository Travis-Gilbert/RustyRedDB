# Measured build and example behavior

Measurements are tied to source state, compiler, target, flags and artifact
checksums. They do not establish a universal throughput or memory bound.

## Reproduce a clean candidate measurement

On a clean committed checkout, use new directories outside the source tree:

```sh
CARGO_BUILD_JOBS=3 CARGO_INCREMENTAL=0 \
  uv run --no-project scripts/measure_release.py \
  --target-dir /absolute/new-empty-measurement-target \
  --output /absolute/new-measurement-evidence
```

The script declares its pinned `psutil==7.2.2` tooling dependency for `uv`.
An isolated Python venv with that dependency works as well. The target directory
must be empty. Registry, compiler-wrapper and toolchain caches can still be
warm; the JSON records the wrapper and calls this a cold **target-directory**
build.

The `clean-candidate-measurement` job in [CI](https://github.com/Travis-Gilbert/RustyRedDB/actions/workflows/ci.yml)
runs this procedure on native x86-64 GNU Linux with Rust 1.96.1 and uploads
`clean-candidate-measurement-x86_64-unknown-linux-gnu`. Retain the run URL and
exact commit with its `measurements.json` and `build.log`.

[Run 39509130713](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39509130713)
executed the measurement successfully on clean revision
`984453329f99ae41a5d40e0297218b242ff5075a`. The
[retained receipt](extraction/distribution-evidence/clean-measurement-9844533.json)
records identical clean source hashes before and after execution:
`cee36fc497cd6276721538428626315de0025d9de5d34fc9ebb09354f2ee500d`.

| Observation | Measured result |
| --- | --- |
| Cold target build, three jobs | 193.256 seconds |
| Sampled peak sum of process-tree RSS | 3,227,852,800 bytes |
| Largest individual child high-water RSS | 2,560,946,176 bytes |
| Release CLI binary | 8,722,232 bytes |
| Create, seed and query, five runs | 6.06–7.32 ms |
| Reopen and query, five runs | 2.13–2.29 ms |

Earlier clean `06acfc4` observations remain in the separately named original and manual
repeat receipts. They describe that historical source and do not replace the
current `9844533` measurements or establish a comparative performance ratio.

Each query returned two rows. These observations describe this tiny example
on the recorded Linux host, compiler and flags. The memory-sampling limits
below apply; this is not a capacity or maximum-memory guarantee.

The script hashes all tracked and nonignored source files and records the Git
revision and dirty state before and after the build/example executions. Clean
candidate attribution requires both states to be clean and identical. A source
change causes a nonzero result after saving explicitly developmental evidence.
`--allow-dirty` is reserved for development measurements and cannot grant clean
candidate attribution.

## What the artifact measures

- Wall-clock build duration for `cargo +1.96.1 build --locked --release -p
  rustyred-cli -p parts-catalog`, with an initially empty target directory.
- Sampled peak sum of resident memory for Cargo and its recursive live process
  tree, using psutil at 100 ms intervals. Shared pages can be counted more than
  once, and brief peaks can be missed. The largest individual child high-water
  RSS from `getrusage` is recorded separately.
- Optimized CLI binary size in bytes.
- Five real create/seed/query runs and five reopen/query runs of parts-catalog,
  each returning both supplier-to-part relationships. The example also stores
  caller-supplied vectors and exact binary drawing bytes. Measurements include
  process startup, database open/close, and real durability barriers.

The tiny example is a reproducible storage/query observation. It does not
characterize a large graph, full-text workload, optional remote service,
GraphBLAS, concurrent writes or maximum capacity. The facade has no configured
memory budget; see [storage operation](releasing.md).

## Current development observations

The local Apple Silicon development build began from dirty, mutable extraction
source. Source and revision changed while it compiled. Its original instrument
records the maximum individual child RSS, without a complete process-tree
sample. Any resulting numbers must remain `development_measurement` evidence;
the revision observed at completion cannot retroactively bind that build to a
clean candidate. No clean-candidate build/memory numbers are claimed from it.

The latency regression fixture in the core suite has a separate source oracle:
its optimized source-unit test profile is `opt-level = 1`; its budget is
unchanged. Normalized-package consumers use Cargo's default test/dev optimization
level 0 and do not run that private engine timing fixture.
The full engine integration target set has a separate calibrated dev profile:
root optimization level 0 and external dependencies level 2, as recorded in the
[baseline evidence](extraction/baseline.md). It runs with the original scaling
ceilings and includes every integration target.
