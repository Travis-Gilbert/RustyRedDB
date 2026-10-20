//! Scale-invariance guard for memory accounting on the graph write path.
//!
//! # What defect this guards
//!
//! `InMemoryGraphStore::estimated_memory_bytes` used to be an O(entire graph)
//! cost: it serde-serialized every node and every edge on every call. It is the
//! base term of `projected_memory_bytes_upper_bound`, which the publish path runs
//! on every commit batch (`RedCoreGraphStore::commit_batch`, and in the server
//! `enforce_prepared_memory_quota`), so every single record write paid for the
//! whole graph. That is the measured reason a streamed chat turn made no
//! progress. The repair serves that value from counters maintained per mutation
//! site (`crate::graph::footprint`) and keeps the whole-graph implementation as
//! the oracle, `recompute_memory_bytes`, off the write path.
//!
//! The same small mutation is applied to graphs of 1k, 10k and 100k similarly
//! shaped records, and the cost of *memory accounting* must not grow as stored
//! history grows.
//!
//! # Oracles, in order of authority
//!
//! 1. **Work count - machine independent, and the actual verdict.**
//!    `graph_memory_recompute_invocations` counts invocations of the whole-graph
//!    recompute. It is read immediately before and after each window and the
//!    delta must be exactly zero at every scale. The counter is process-wide, so
//!    a window is the claim: a write that recomputed would move it in *every*
//!    window, while another thread of this binary recomputing can spoil only an
//!    individual window. Each window is retried a bounded number of times and one
//!    clean window is the pass; the observed deltas are reported either way.
//! 2. **Rebuild parity - a speedup must not be bought with drift.**
//!    `from_snapshot` re-derives the counters from the records themselves and,
//!    with debug assertions on (the default test profile), asserts them against
//!    the whole-graph serde oracle inside
//!    `reset_committed_footprint_from_recompute`. So `maintained == rebuilt` and
//!    `rebuilt == oracle` compose to `maintained == oracle`. The test also checks
//!    that the rebuild advanced the counter, which proves the oracle assertion it
//!    leans on actually ran instead of being compiled out. The serde-computed
//!    snapshot term is measured live as well, and the maintained total must cover
//!    it.
//! 3. **Wall clock on the accounting call - secondary, generous.**
//!    `projected_memory_bytes_upper_bound` is timed on its own, per scale, and
//!    the largest scale must stay within a small constant multiple of the
//!    smallest over a floor. That call is exactly where the defect lived (its
//!    base term), and the same tier also measures one whole-graph serde pass live
//!    (`memory_attribution`, reported as `whole_graph_serde`), which is the
//!    magnitude that ceiling stands in for. The accounting call is asserted to
//!    stay at least 25x below that pass; hundreds to thousands of times was
//!    observed. That makes the bound a real second net for the regression the
//!    counter cannot see: a re-serialization that bypasses the counted recompute
//!    path.
//!
//! # What is measured but deliberately NOT asserted
//!
//! The end-to-end mutation window (quota projection plus the publish) is timed
//! and reported, but not asserted flat, because it is not flat: the RedCore
//! publish path still carries its own per-commit whole-graph cost that has
//! nothing to do with memory accounting (measured at roughly 5 to 12 microseconds
//! per stored record per commit, so a single record write against a 100k record
//! graph costs about 0.9 to 1.7 seconds, with the work counter standing still).
//! Asserting a flat bound there would be asserting a property the accounting
//! repair does not control, so the numbers are logged and the finding is reported
//! instead. The accounting term itself is asserted flat by oracle 3.
//!
//! # Failure meanings
//!
//! * A nonzero work-count delta is the defect returning: something on the write
//!   path asked for a whole-graph traversal per commit.
//! * A failed rebuild comparison means the fast path is also the wrong path: the
//!   maintained counters no longer describe the graph the oracle sees, so the
//!   quota would be priced off a number that is not the graph.
//! * A failed accounting-time ceiling means the whole-graph serialization is back
//!   inside the quota projection.
//!
//! # Tiers and how to run them
//!
//! `[1k, 10k]` runs by default. The 100k tier is `#[ignore]`d because the publish
//! path must compile one full graph pack before any incremental commit is
//! possible, and that bootstrap costs about 1 millisecond per stored record, so
//! the tier needs roughly 3.5 minutes mostly for setup. It is not dropped: the
//! ignored test carries the same assertions at 100x, with the exact command in
//! its reason string.
//!
//! # One measurement path, on purpose
//!
//! The counter is process-wide and tests of one binary share a process under
//! `cargo test`, so every tier, window and rebuild of a run happens inside one
//! test function. The retry windows above cover a parallel harness; running the
//! ignored tier alone (`--test-threads 1`) is the fully clean way.

use std::time::{Duration, Instant};

use rustyred_thg_core::graph::mutation::{GraphMutation, GraphMutationBatch};
use rustyred_thg_core::graph::record::{EdgeRecord, NodeRecord};
use rustyred_thg_core::{
    graph_memory_recompute_invocations, InMemoryGraphStore, RedCoreGraphStore,
};
use serde_json::json;

/// Default tiers: the same shape at 10x apart.
const DEFAULT_SCALES: [usize; 2] = [1_000, 10_000];

/// The ignored tier re-runs the small baseline so the same ratio applies at 100x.
const HUNDRED_K_SCALES: [usize; 2] = [1_000, 100_000];

/// Bounded retries per window; see the file doc.
const WINDOW_ATTEMPTS: usize = 6;

/// Generous ceiling for the accounting-call observation, over the floor below.
const MAX_ACCOUNTING_RATIO: f64 = 12.0;

/// How far below one whole-graph serde pass the accounting call must stay at the
/// same tier. Deliberately modest: the observed separation is thousands of times.
const ACCOUNTING_ORDERS_OF_MAGNITUDE: u32 = 25;

/// Floor under the small-scale denominator. The accounting call legitimately
/// takes tens of microseconds, and scheduler noise must not turn that into a
/// ratio failure. A whole-graph serialization at the large scale (roughly a
/// hundred milliseconds against a 100k record graph) is still orders of
/// magnitude above the resulting ceiling.
const ACCOUNTING_TIMING_FLOOR: Duration = Duration::from_micros(200);

/// One fixture record: a few labels and properties, not a giant payload, so the
/// 100k tier stays affordable.
fn fixture_node(index: usize) -> NodeRecord {
    NodeRecord::new(
        format!("scaling:node:{index:06}"),
        ["ScaleFixture", "Record"],
        json!({
            "name": format!("record-{index:06}"),
            "rank": index % 97,
            "shard": index % 8,
        }),
    )
}

fn fixture_edge(index: usize, records: usize) -> EdgeRecord {
    EdgeRecord::new(
        format!("scaling:edge:{index:06}"),
        format!("scaling:node:{index:06}"),
        "NEXT",
        format!("scaling:node:{:06}", (index + 1) % records),
        json!({ "weight": 1.0, "slot": index % 16 }),
    )
}

/// The mutation under test, byte-identical at every scale. The node the small
/// tiers do not already hold is written first, then an edge that prices a second
/// record shape through the same quota projection.
fn mutation_node() -> NodeRecord {
    NodeRecord::new(
        "scaling:write:node",
        ["ScaleFixture", "Record"],
        json!({ "name": "bounded write", "rank": 1, "shard": 3 }),
    )
}

fn mutation_edge(records: usize) -> EdgeRecord {
    EdgeRecord::new(
        "scaling:write:edge",
        "scaling:write:node",
        "WROTE",
        format!("scaling:node:{:06}", records - 1),
        json!({ "weight": 0.5, "slot": 2 }),
    )
}

/// One tier's graph, built through one bulk commit: the documented bulk-load
/// shape, because a per-record commit stream would itself be quadratic in the
/// publish path and this test is not about that. Construction is O(graph) by
/// design and sits outside every window.
fn build_scaled_store(records: usize) -> (RedCoreGraphStore, Duration) {
    let started = Instant::now();
    let mut store = RedCoreGraphStore::memory();
    let mutations = (0..records)
        .map(|index| GraphMutation::NodeUpsert(fixture_node(index)))
        .chain((0..records).map(|index| GraphMutation::EdgeUpsert(fixture_edge(index, records))));
    store
        .commit_batch(GraphMutationBatch::new(mutations))
        .expect("the bulk fixture commit must publish");
    (store, started.elapsed())
}

/// The committed footprint the quota path reads, through the public surface.
fn committed_bytes(store: &RedCoreGraphStore) -> usize {
    store
        .stats()
        .expect("stats must succeed on a memory-backed store")
        .memory_bytes
}

struct MutationWindow {
    deltas: Vec<u64>,
    elapsed: Vec<Duration>,
    clean_attempt: Option<usize>,
}

impl MutationWindow {
    /// The clean window's elapsed time. Only meaningful once the caller has
    /// asserted `clean_attempt.is_some()`.
    fn clean_elapsed(&self) -> Duration {
        self.elapsed[self
            .clean_attempt
            .expect("a clean window is the claim, and the caller asserts it")]
    }
}

/// Read the whole-graph recompute counter around `apply`, exactly as the claim is
/// stated: reading before, mutation region, reading after.
fn mutation_window(
    store: &mut RedCoreGraphStore,
    mut apply: impl FnMut(&mut RedCoreGraphStore),
    attempts: usize,
) -> MutationWindow {
    let mut deltas = Vec::with_capacity(attempts);
    let mut elapsed = Vec::with_capacity(attempts);
    let mut clean_attempt = None;
    for attempt in 0..attempts {
        let before = graph_memory_recompute_invocations();
        let started = Instant::now();
        apply(store);
        let taken = started.elapsed();
        let after = graph_memory_recompute_invocations();
        deltas.push(after.saturating_sub(before));
        elapsed.push(taken);
        if after == before {
            clean_attempt = Some(attempt);
            break;
        }
    }
    MutationWindow {
        deltas,
        elapsed,
        clean_attempt,
    }
}

struct ScaleReading {
    records: usize,
    build: Duration,
    /// The quota projection alone: the repaired base term.
    node_projection: MutationWindow,
    edge_projection: MutationWindow,
    /// The quota projection plus the real publish, the server's per-commit shape.
    node_mutation: MutationWindow,
    edge_mutation: MutationWindow,
    /// One whole-graph pass (`memory_attribution`, which serde-serializes every
    /// record): the cost class the repair moved off the write path, measured live
    /// so the accounting call can be compared against it at the same tier.
    whole_graph_serde: Duration,
    maintained_before: usize,
    maintained_after: usize,
    measured_snapshot_term: usize,
    rebuilt_bytes: usize,
    rebuild_work_delta: u64,
}

impl ScaleReading {
    fn accounting_cost(&self) -> Duration {
        self.node_projection.clean_elapsed() + self.edge_projection.clean_elapsed()
    }

    fn end_to_end_cost(&self) -> Duration {
        self.node_mutation.clean_elapsed() + self.edge_mutation.clean_elapsed()
    }

    fn summarize(&self) -> String {
        format!(
            "scale {} records: build {:?}; accounting call (node/edge) deltas {:?}/{:?} clean elapsed {:?}/{:?}; \
             projection+publish (node/edge) deltas {:?}/{:?} clean elapsed {:?}/{:?}; \
             whole-graph serde pass {:?}; maintained committed bytes {} -> {}; serde-measured snapshot term {}; \
             rebuilt-from-records total {}; rebuild work delta {}",
            self.records,
            self.build,
            self.node_projection.deltas,
            self.edge_projection.deltas,
            self.node_projection.clean_elapsed(),
            self.edge_projection.clean_elapsed(),
            self.node_mutation.deltas,
            self.edge_mutation.deltas,
            self.node_mutation.clean_elapsed(),
            self.edge_mutation.clean_elapsed(),
            self.whole_graph_serde,
            self.maintained_before,
            self.maintained_after,
            self.measured_snapshot_term,
            self.rebuilt_bytes,
            self.rebuild_work_delta,
        )
    }
}

fn measure_scale(records: usize) -> ScaleReading {
    let (mut store, build) = build_scaled_store(records);
    let maintained_before = committed_bytes(&store);

    let node = mutation_node();
    let node_batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(node.clone())]);
    let edge = mutation_edge(records);
    let edge_batch = GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge.clone())]);

    // The accounting call on its own. This is the base-term read the defect was
    // in: it must be O(1) in the graph and must not reach the recompute path.
    let node_projection = mutation_window(
        &mut store,
        |store| {
            std::hint::black_box(store.projected_memory_bytes_upper_bound(&node_batch));
        },
        WINDOW_ATTEMPTS,
    );
    let edge_projection = mutation_window(
        &mut store,
        |store| {
            std::hint::black_box(store.projected_memory_bytes_upper_bound(&edge_batch));
        },
        WINDOW_ATTEMPTS,
    );

    // The server's per-commit shape: price the batch against the quota, publish
    // it. The node goes first because the edge mutation needs its endpoint live.
    let node_mutation = mutation_window(
        &mut store,
        |store| {
            let projection = store.projected_memory_bytes_upper_bound(&node_batch);
            store
                .upsert_node(node.clone())
                .expect("the node upsert must publish");
            std::hint::black_box(projection);
        },
        WINDOW_ATTEMPTS,
    );
    let edge_mutation = mutation_window(
        &mut store,
        |store| {
            let projection = store.projected_memory_bytes_upper_bound(&edge_batch);
            store
                .upsert_edge(edge.clone())
                .expect("the edge upsert must publish");
            std::hint::black_box(projection);
        },
        WINDOW_ATTEMPTS,
    );

    let maintained_after = committed_bytes(&store);
    let serde_started = Instant::now();
    let measured_snapshot_term = store.memory_attribution().serialized_graph_bytes;
    let whole_graph_serde = serde_started.elapsed();

    // Rebuild parity. O(graph) on purpose and deliberately outside the windows:
    // this is the oracle side of the comparison, not the write path.
    let before_rebuild = graph_memory_recompute_invocations();
    let rebuilt = InMemoryGraphStore::from_snapshot(store.graph_snapshot())
        .expect("the mutated graph must reopen from its own snapshot");
    let rebuild_work_delta = graph_memory_recompute_invocations().saturating_sub(before_rebuild);
    let rebuilt_bytes = rebuilt.stats().memory_bytes;

    ScaleReading {
        records,
        build,
        node_projection,
        edge_projection,
        node_mutation,
        edge_mutation,
        maintained_before,
        maintained_after,
        measured_snapshot_term,
        whole_graph_serde,
        rebuilt_bytes,
        rebuild_work_delta,
    }
}

fn run_scales(scales: &[usize]) {
    let mut readings = Vec::with_capacity(scales.len());

    for records in scales {
        let reading = measure_scale(*records);
        let summary = reading.summarize();

        // 1. Work count: none of the four windows ran a whole-graph recompute.
        for (kind, window) in [
            ("node accounting call", &reading.node_projection),
            ("edge accounting call", &reading.edge_projection),
            ("node projection+publish", &reading.node_mutation),
            ("edge projection+publish", &reading.edge_mutation),
        ] {
            let clean = window.clean_attempt.unwrap_or_else(|| {
                panic!(
                    "a bounded {kind} over {} records recomputed the whole graph in all {} windows, \
                     work deltas {:?}: {summary}",
                    reading.records,
                    window.deltas.len(),
                    window.deltas,
                )
            });
            assert_eq!(
                window.deltas[clean], 0,
                "the clean {kind} window must show a zero work-count delta: {summary}",
            );
        }

        // The accounting call must be orders of magnitude below one whole-graph
        // pass at the same tier. Both numbers come from this machine, this tier
        // and this moment, so the comparison does not depend on machine speed.
        assert!(
            reading.accounting_cost() * ACCOUNTING_ORDERS_OF_MAGNITUDE <= reading.whole_graph_serde,
            "the quota projection's base term must stay {}x below one whole-graph pass at the same \
             tier, so this is the whole-graph serialization back on the write path: {summary}",
            ACCOUNTING_ORDERS_OF_MAGNITUDE,
        );

        // 2. Correctness: the accounting saw the write and still describes the
        //    same graph the rebuild-from-records oracle describes.
        assert!(
            reading.maintained_after > reading.maintained_before,
            "the two writes must grow the committed footprint: {summary}",
        );
        assert!(
            reading.maintained_after >= reading.measured_snapshot_term,
            "the maintained total must cover the serde-measured snapshot term: {summary}",
        );
        assert_eq!(
            reading.rebuilt_bytes, reading.maintained_after,
            "a store rebuilt from the mutated store's snapshot must account the same total bytes, \
             which is what makes the maintained counters comparable across a restart: {summary}",
        );
        if cfg!(debug_assertions) {
            // The rebuild re-derives the counters and then asserts them against
            // the whole-graph oracle. Proving the counter moved proves that
            // assertion ran, so `maintained == rebuilt` above transitively means
            // `maintained == oracle`.
            assert!(
                reading.rebuild_work_delta > 0,
                "the rebuild-from-records oracle must actually run in this profile, otherwise the \
                 parity comparison above proves nothing: {summary}",
            );
        }

        readings.push(reading);
    }

    for reading in &readings {
        println!(
            "accounting_scaling {} | accounting cost {:?} | projection+publish cost {:?}",
            reading.summarize(),
            reading.accounting_cost(),
            reading.end_to_end_cost(),
        );
    }

    // 3. The accounting call itself must not get more expensive with stored
    //    history. Generous on purpose: see the file doc.
    let smallest = &readings[0];
    let largest = &readings[readings.len() - 1];
    let baseline = smallest.accounting_cost().max(ACCOUNTING_TIMING_FLOOR);
    let ratio = largest.accounting_cost().as_secs_f64() / baseline.as_secs_f64();
    assert!(
        ratio <= MAX_ACCOUNTING_RATIO,
        "the quota projection's base term cost {ratio:.2}x more at {} records than at {} records \
         (ceiling {MAX_ACCOUNTING_RATIO}x over a {ACCOUNTING_TIMING_FLOOR:?} floor), which is the \
         whole-graph serialization back inside `estimated_memory_bytes`. {}",
        largest.records,
        smallest.records,
        readings
            .iter()
            .map(ScaleReading::summarize)
            .collect::<Vec<_>>()
            .join(" | "),
    );
}

/// 1k and 10k: the default gate. The work-count oracle at the top of the file is
/// the pass/fail verdict; the wall-clock bound on the accounting call is loose by
/// construction.
#[test]
fn a_bounded_write_stays_flat_as_stored_history_grows() {
    run_scales(&DEFAULT_SCALES);
}

/// The 100k tier. Ignored by default only because the publish path must compile
/// one full graph pack before any commit is incremental, and that bootstrap costs
/// about a millisecond per stored record, so this tier needs roughly three and a
/// half minutes mostly for setup. Every assertion of the default tier runs here
/// at 100x.
#[test]
#[ignore = "100k tier: one O(graph) publish-path bootstrap pack compile costs ~3 minutes of setup, and the tier runs ~4 minutes total; run alone with `cargo nextest run -p rustyred-thg-core --run-ignored ignored-only -E 'test(accounting_scaling)' --test-threads 1`"]
fn a_bounded_write_stays_flat_at_one_hundred_thousand_records() {
    run_scales(&HUNDRED_K_SCALES);
}
