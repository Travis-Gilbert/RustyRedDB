#![cfg(feature = "graphblas")]

use rustyred_thg_core::fold::algebra::AlgebraFold;
use rustyred_thg_core::{EdgeRecord, Fold, FoldGraphStore, GraphWrite, NodeRecord};
use serde_json::json;
use std::error::Error;
use std::process::Command;
use theorem_oplog::PhysicalLog;

const LINK_TYPE: &str = "LINKS";

#[test]
fn isolated_process_reports_lazy_residency_and_bounded_rss() -> Result<(), Box<dyn Error>> {
    let output = Command::new(std::env::current_exe()?)
        .args(["--exact", "algebra_rss_child", "--nocapture"])
        .env("RUSTYRED_ALGEBRA_RSS_CHILD", "1")
        .output()?;
    assert!(
        output.status.success(),
        "isolated Algebra RSS probe failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    eprint!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

#[test]
fn algebra_rss_child() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("RUSTYRED_ALGEBRA_RSS_CHILD").is_none() {
        return Ok(());
    }
    let directory = tempfile::tempdir()?;
    build_fixture(directory.path())?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let before_fold = max_rss_bytes()?;

    let mut fold = AlgebraFold::default();
    for record in log.records() {
        fold.apply(record)?;
    }
    let before_query = max_rss_bytes()?;
    let logical_growth = before_query.saturating_sub(before_fold);
    let empty = fold.residency()?;
    assert_eq!(empty.regions, 0);
    assert_eq!(empty.matrix_entries, 0);
    assert_eq!(empty.matrix_bytes, 0);
    assert!(
        logical_growth <= 16 * 1024 * 1024,
        "log replay allocated {logical_growth} unaccounted RSS bytes before a native query"
    );

    let latest = log.latest_position();
    let query = json!({
        "kind": "reachable",
        "edge_type": LINK_TYPE,
        "sources": ["node:0"]
    });
    fold.upquery(&query, latest, log.records())?;
    assert!(fold.residency()?.matrix_bytes > 0);
    for _ in 0..32 {
        fold.evict_region(LINK_TYPE);
        fold.upquery(&query, latest, log.records())?;
    }
    let baseline = max_rss_bytes()?;
    for _ in 0..512 {
        fold.evict_region(LINK_TYPE);
        fold.upquery(&query, latest, log.records())?;
    }
    let end = max_rss_bytes()?;
    let repeated_growth = end.saturating_sub(baseline);
    println!(
        "Algebra isolated RSS: before_fold={before_fold} before_query={before_query} logical_growth={logical_growth} baseline={baseline} end={end} repeated_growth={repeated_growth} matrix_bytes={}",
        fold.residency()?.matrix_bytes
    );
    assert!(
        repeated_growth <= 32 * 1024 * 1024,
        "repeated native materialize/evict grew max RSS by {repeated_growth} bytes"
    );
    Ok(())
}

fn build_fixture(directory: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let mut store = FoldGraphStore::open(directory, deployment_id())?;
    store.commit_batch(rustyred_thg_core::GraphMutationBatch::new((0..16).map(
        |index| {
            rustyred_thg_core::GraphMutation::NodeUpsert(NodeRecord::new(
                format!("node:{index}"),
                ["AlgebraRssFixture"],
                json!({"fixture": true}),
            ))
        },
    )))?;
    for index in 0..128 {
        let from = index % 16;
        let to = (index * 7 + 1) % 16;
        store.upsert_edge(EdgeRecord::new(
            format!("edge:{index}"),
            format!("node:{from}"),
            LINK_TYPE,
            format!("node:{to}"),
            json!({"fixture": true}),
        ))?;
    }
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:algebra-fold-rss").unwrap()
}

fn max_rss_bytes() -> Result<u64, Box<dyn Error>> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let rss = unsafe { usage.assume_init().ru_maxrss as u64 };
    #[cfg(target_os = "macos")]
    return Ok(rss);
    #[cfg(not(target_os = "macos"))]
    return Ok(rss.saturating_mul(1024));
}
