//! Spec 2 (native ordered index) acceptance tests, reliability-critical half.
//!
//! Handoff: rustyred-ordered-index-handoff. Sorted-set semantics (NaN reject,
//! equal-score lexicographic order) are covered by `ordered::tests`, and the
//! scoped RESP ZSET surface by `rustyred-thg-resp-server::protocol::tests`.
//! This file proves the two store-level acceptance criteria those cannot:
//!   #2 transient mode is commit-free (the reliability-critical property), and
//!   #3 a persistent ordered index round-trips through commit + reopen.

use rustyred_thg_core::{NodeRecord, RedCoreDurability, RedCoreGraphStore, RedCoreOptions};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("rustyred-{tag}-{nanos}"))
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(meta) = p.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn always_durable() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1_000_000,
        strict_acid: false,
    }
}

/// Acceptance #2 (the reliability-critical test): a named transient ordered set
/// absorbs a large volume of zadd/zpop while the durable store's committed size
/// stays exactly flat -- the frontier's millions of priority writes never reach
/// the commit path, so they cannot churn the graph or its on-disk footprint.
#[test]
fn transient_ordered_mode_is_commit_free() {
    let dir = unique_dir("ordered-transient");
    let mut store = RedCoreGraphStore::open(&dir, always_durable()).unwrap();

    // A few real commits so the durable store is non-empty (and so we'd notice
    // if transient churn started appending to the AOF).
    for i in 0..10 {
        store
            .upsert_node(NodeRecord::new(
                format!("seed:{i}"),
                ["Doc"],
                json!({ "i": i }),
            ))
            .unwrap();
    }
    let committed_before = dir_size(&dir);
    assert!(committed_before > 0, "real commits produced durable bytes");

    // Heavy transient churn on the frontier-style ordered set.
    let n = 50_000usize;
    for i in 0..n {
        store
            .transient_ordered_zadd(
                "frontier",
                format!("url:{i}").into_bytes(),
                (i % 997) as f64,
            )
            .unwrap();
    }
    assert_eq!(store.transient_ordered_zcard("frontier"), n);

    let mut popped = 0usize;
    while store.transient_ordered_zpop_max("frontier").is_some() {
        popped += 1;
    }
    assert_eq!(popped, n, "every transient member pops back out");

    let committed_after = dir_size(&dir);
    assert_eq!(
        committed_before, committed_after,
        "transient ordered churn must write zero durable bytes (commit-free)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Acceptance #2 corollary: transient zpop_max returns members in descending
/// PPR-priority order (what the crawl frontier relies on for next-URL).
#[test]
fn transient_zpop_max_returns_descending_priority() {
    let dir = unique_dir("ordered-priority");
    let mut store = RedCoreGraphStore::open(&dir, always_durable()).unwrap();

    store
        .transient_ordered_zadd("frontier", b"low".to_vec(), 1.0)
        .unwrap();
    store
        .transient_ordered_zadd("frontier", b"high".to_vec(), 9.0)
        .unwrap();
    store
        .transient_ordered_zadd("frontier", b"mid".to_vec(), 5.0)
        .unwrap();

    assert_eq!(
        store.transient_ordered_zpop_max("frontier"),
        Some((b"high".to_vec(), 9.0))
    );
    assert_eq!(
        store.transient_ordered_zpop_max("frontier"),
        Some((b"mid".to_vec(), 5.0))
    );
    assert_eq!(
        store.transient_ordered_zpop_max("frontier"),
        Some((b"low".to_vec(), 1.0))
    );
    assert_eq!(store.transient_ordered_zpop_max("frontier"), None);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Acceptance #3: a persistent ordered index over a (label, numeric-property)
/// survives commit and reload -- the durable designation plus the node upserts
/// replay on reopen and the index serves the same score order.
#[test]
fn persistent_ordered_index_round_trips_through_reopen() {
    let dir = unique_dir("ordered-persistent");

    {
        let mut store = RedCoreGraphStore::open(&dir, always_durable()).unwrap();
        store.designate_ordered_property("Score", "rank").unwrap();
        for (id, rank) in [("a", 3.0_f64), ("b", 1.0), ("c", 2.0)] {
            store
                .upsert_node(NodeRecord::new(id, ["Score"], json!({ "rank": rank })))
                .unwrap();
        }
        let ranked: Vec<String> = store
            .ordered_range_by_score("Score", "rank", 0.0, 10.0, None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ranked, vec!["b", "c", "a"], "pre-reopen score order");
    }

    let reopened = RedCoreGraphStore::open(&dir, always_durable()).unwrap();
    let ranked: Vec<String> = reopened
        .ordered_range_by_score("Score", "rank", 0.0, 10.0, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        ranked,
        vec!["b", "c", "a"],
        "persistent ordered index survives reopen in score order"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
