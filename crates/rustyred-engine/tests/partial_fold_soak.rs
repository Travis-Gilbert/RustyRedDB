use std::error::Error;
use std::process::Command;
use std::sync::Arc;

use rustyred_thg_core::{
    ColdIndex, FoldStorage, InMemoryColdIndex, MutationOrder, PartialFoldRuntime, PartialOperator,
    PositionedFoldIndex, RedbFoldStorage,
};
use theorem_oplog::LogPosition;

#[test]
#[ignore = "isolated Zipf and RSS soak; run with THEOREM_RUN_SOAK=1"]
fn zipf_reads_match_full_oracle_without_unbounded_rss_or_accounting() -> Result<(), Box<dyn Error>>
{
    assert_eq!(
        std::env::var("THEOREM_RUN_SOAK").as_deref(),
        Ok("1"),
        "set THEOREM_RUN_SOAK=1 for the isolated soak"
    );
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(InMemoryColdIndex::new());
    let mut runtime = PartialFoldRuntime::new(PositionedFoldIndex::new(storage, cold), 128 * 1024)?;
    let mut oracle = Vec::new();
    for index in 0..1_000_u64 {
        let key = format!("entity:{index:04}");
        let value = format!("value:{index:04}:{}", "x".repeat(96)).into_bytes();
        runtime.apply_update(
            PartialOperator::FactsCurrentEntity,
            key.as_bytes(),
            MutationOrder::new(LogPosition(index + 1), 0)?,
            Some(value.clone()),
        )?;
        oracle.push((key, value, LogPosition(index + 1)));
    }

    let mut random = DeterministicRandom::new(0x50A4_217F_05D0_0001);
    let mut rss_samples = Vec::new();
    for epoch in 0..12 {
        for _ in 0..5_000 {
            let index = random.zipfish(oracle.len());
            let (key, expected, position) = &oracle[index];
            let read = runtime.read_at(
                PartialOperator::FactsCurrentEntity,
                key.as_bytes(),
                *position,
            )?;
            assert_eq!(read.payload.as_deref(), Some(expected.as_slice()));
            assert!(runtime.resident_bytes() <= runtime.budget_bytes());
            if random.usize(251) == 0 {
                runtime.evict(PartialOperator::FactsCurrentEntity, key.as_bytes())?;
            }
        }
        rss_samples.push(current_rss_kib()?);
        println!(
            "epoch={epoch} resident_bytes={} resident_keys={} rss_kib={}",
            runtime.resident_bytes(),
            runtime.resident_len(),
            rss_samples.last().expect("just sampled")
        );
    }

    let steady = &rss_samples[4..];
    let min = *steady.iter().min().expect("steady samples");
    let max = *steady.iter().max().expect("steady samples");
    assert!(
        max.saturating_sub(min) <= 16 * 1024,
        "steady-state RSS range {} KiB exceeds 16 MiB: {steady:?}",
        max.saturating_sub(min)
    );
    assert!(runtime.resident_len() < oracle.len());
    Ok(())
}

fn current_rss_kib() -> Result<u64, Box<dyn Error>> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    if !output.status.success() {
        return Err("ps failed while sampling isolated-process RSS".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().parse()?)
}

struct DeterministicRandom(u64);

impl DeterministicRandom {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }

    fn zipfish(&mut self, upper: usize) -> usize {
        self.usize(upper)
            .min(self.usize(upper))
            .min(self.usize(upper))
    }
}
