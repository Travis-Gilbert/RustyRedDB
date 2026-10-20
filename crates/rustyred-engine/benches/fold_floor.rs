use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use rustyred_thg_core::{FoldStorage, FoldStorageMutation};

const BATCHES: usize = 100;
const MUTATIONS_PER_BATCH: usize = 10;

fn benchmark_floors(criterion: &mut Criterion) {
    let mut writes = criterion.benchmark_group("fold_floor_immediate_writes");
    writes.sample_size(10);
    writes.throughput(Throughput::Elements((BATCHES * MUTATIONS_PER_BATCH) as u64));
    for kind in enabled_floors() {
        let directory = tempfile::tempdir().expect("create metric directory");
        let logical_bytes = run_workload(kind, directory.path()).expect("run metric workload");
        let recovery_started = Instant::now();
        let reopened = open_floor(kind, directory.path()).expect("reopen metric floor");
        let recovery_us = recovery_started.elapsed().as_micros();
        let disk_bytes = reopened.disk_bytes().expect("measure metric footprint");
        let amplification = disk_bytes as f64 / logical_bytes.max(1) as f64;
        println!(
            "W11_METRIC floor={} logical_bytes={} disk_bytes={} amplification={:.3} recovery_us={}",
            kind.name(),
            logical_bytes,
            disk_bytes,
            amplification,
            recovery_us
        );
        assert_eq!(
            reopened
                .get("bench", b"key:0999")
                .expect("read metric sentinel"),
            Some(value_for(999))
        );

        writes.bench_with_input(
            BenchmarkId::from_parameter(kind.name()),
            &kind,
            |bench, kind| {
                bench.iter_batched(
                    || tempfile::tempdir().expect("create benchmark directory"),
                    |directory| {
                        black_box(
                            run_workload(*kind, directory.path())
                                .expect("run immediately durable workload"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    writes.finish();

    let mut recovery = criterion.benchmark_group("fold_floor_recovery");
    recovery.sample_size(10);
    recovery.throughput(Throughput::Elements((BATCHES * MUTATIONS_PER_BATCH) as u64));
    for kind in enabled_floors() {
        recovery.bench_with_input(
            BenchmarkId::from_parameter(kind.name()),
            &kind,
            |bench, kind| {
                bench.iter_batched(
                    || {
                        let directory = tempfile::tempdir().expect("create recovery directory");
                        run_workload(*kind, directory.path()).expect("seed recovery floor");
                        directory
                    },
                    |directory| {
                        black_box(
                            open_floor(*kind, directory.path()).expect("recover benchmark floor"),
                        )
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    recovery.finish();
}

fn run_workload(kind: FloorKind, path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let storage = open_floor(kind, path)?;
    let mut logical_bytes = 0_u64;
    for batch in 0..BATCHES {
        let mutations = (0..MUTATIONS_PER_BATCH)
            .map(|offset| {
                let index = batch * MUTATIONS_PER_BATCH + offset;
                let key = format!("key:{index:04}").into_bytes();
                let value = value_for(index);
                logical_bytes += (key.len() + value.len()) as u64;
                FoldStorageMutation::put("bench", key, value)
            })
            .collect::<Vec<_>>();
        storage.write_batch(&mutations)?;
    }
    Ok(logical_bytes)
}

fn value_for(index: usize) -> Vec<u8> {
    format!("value:{index:04}:{}", "x".repeat(240)).into_bytes()
}

#[derive(Clone, Copy)]
enum FloorKind {
    #[cfg(feature = "redb-floor")]
    Redb,
    #[cfg(feature = "fjall-floor")]
    Fjall,
}

impl FloorKind {
    fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "redb-floor")]
            Self::Redb => "redb",
            #[cfg(feature = "fjall-floor")]
            Self::Fjall => "fjall",
        }
    }
}

fn enabled_floors() -> Vec<FloorKind> {
    vec![
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb,
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall,
    ]
}

fn open_floor(
    kind: FloorKind,
    path: &Path,
) -> Result<Box<dyn FoldStorage>, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb => Ok(Box::new(rustyred_thg_core::RedbFoldStorage::open(path)?)),
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall => Ok(Box::new(rustyred_thg_core::FjallFoldStorage::open(path)?)),
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_floors
}
criterion_main!(benches);
