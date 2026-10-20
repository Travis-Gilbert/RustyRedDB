//! Boundary oracle for the whole-graph Prolly parity audit.
//!
//! # What defect this guards
//!
//! `validate_incremental_prolly_parity` used to run as a publish precondition on
//! every incremental commit: it collected every entry of the prior tree and
//! rebuilt the whole tree from scratch in a private `MemChunkStore` purely to
//! assert that the incremental builder agreed with it. The default mode was
//! `Always` whenever `debug_assertions` were on -- which is the profile a local
//! acceptance run of the authority uses -- so in a debug build *every* commit
//! rebuilt the entire graph's Prolly tree. Measured by `commit_scaling.rs` at
//! the 10k tier (30,038 stored records, one node upsert plus one edge upsert per
//! commit): median steady commit 117.4 ms with the validation selected, 38.6 ms
//! with `RUSTYRED_PROLLY_VALIDATE=off`, so the oracle was 78.8 ms, 67% of the
//! commit.
//!
//! # What replaced it
//!
//! The oracle is not deleted and not sampled down. It moved to the boundaries
//! that already walk the whole graph:
//!
//! 1. **On demand**, `RedCoreGraphStore::verify_tree_parity` (and the
//!    free-standing `audit_published_prolly_parity` it calls): rebuild the
//!    published logical tree from its own entries, compare the rebuild against
//!    the published root and entry count, then cross-check those counts against
//!    the manifest that names the tree.
//! 2. **At restart**, the recovery checkpoint in `RedCoreGraphStore::recover`,
//!    which audits the recovered head once, before the store serves a read or a
//!    write, and fails the open closed on divergence. It records a
//!    `prolly_parity_audit` stage sample in the recovery profile.
//!
//! `RUSTYRED_PROLLY_VALIDATE=off` still switches the checkpoint off;
//! `always` / `sampled` / `RUSTYRED_PROLLY_VALIDATE_PPM` still filter it.
//!
//! # Oracles, in order of authority
//!
//! 1. **Work count -- machine independent, and the actual verdict.**
//!    `graph_prolly_parity_validations` counts whole-graph parity audits. It is
//!    read immediately before and after each window and the delta must be zero
//!    for every bounded write at every tier. The counter is process-wide and
//!    tests of one binary share a process, so a window is the claim and each is
//!    retried a bounded number of times with the observed deltas reported: a
//!    write that audited would move the counter in *every* window, while another
//!    test in this binary auditing concurrently can spoil only individual
//!    windows.
//! 2. **Liveness -- a counter that never moves proves nothing.** The same
//!    counter must move by exactly one for one explicit on-demand audit, and the
//!    audit must accept a real published tree at every tier.
//! 3. **The oracle still catches planted divergence.** Three plants, all through
//!    public API or the crate's own test seam, asserting the error code
//!    `graph_prolly_validation_divergence` rather than "something failed".
//! 4. **The boundary is live at restart.** A durable store reopened in the
//!    default profile carries a `prolly_parity_audit` sample in its recovery
//!    profile, which is per-store evidence and does not depend on a process-wide
//!    counter read.
//!
//! # What is measured but deliberately not asserted
//!
//! Wall clock is reported, not asserted, for the same reason
//! `commit_scaling.rs` reports rather than asserts it: the publish path still
//! carries a whole-registry commitment and chunk verification on every commit,
//! so per-commit cost is still not flat in stored history. The verdict here is
//! the counted work.
//!
//! # Tiers
//!
//! `[1k, 10k]` runs by default, matching `accounting_scaling.rs`. The 10k tier
//! costs roughly 11 s in one-time bootstrap, because the first publish into an
//! empty store compiles one full graph pack by design.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use rustyred_thg_core::graph::mutation::{GraphMutation, GraphMutationBatch};
use rustyred_thg_core::graph::record::{EdgeRecord, NodeRecord};
use rustyred_thg_core::versioned_graph::{
    audit_published_prolly_parity, build_prolly_tree_from_entries, build_prolly_tree_incremental,
    canonical_leaf_preimage, compile_graph_pack, validate_incremental_prolly_parity,
};
use rustyred_thg_core::{
    graph_prolly_parity_validations, reset_graph_prolly_parity_validations, ChunkStore,
    GraphCompileOptions, GraphObjectKind, GraphProllyTree, GraphSnapshot, GraphTreeEntry, Hash,
    MemChunkStore, RedCoreDurability, RedCoreGraphStore, RedCoreOptions,
};
use serde_json::json;

/// The same shape at 10x apart, as in `accounting_scaling.rs`.
const TIERS: [usize; 2] = [1_000, 10_000];

/// Bounded writes measured per tier.
const STEPS: usize = 3;

/// Bounded retries per window; see the file doc.
const WINDOW_ATTEMPTS: usize = 6;

fn fixture_node(index: usize) -> NodeRecord {
    NodeRecord::new(
        format!("parity:node:{index:06}"),
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
        format!("parity:edge:{index:06}"),
        format!("parity:node:{index:06}"),
        "NEXT",
        format!("parity:node:{:06}", (index + 1) % records),
        json!({ "weight": 1.0, "slot": index % 16 }),
    )
}

/// The documented bulk-load shape: one commit, O(records) by design.
fn bulk_mutations(records: usize) -> Vec<GraphMutation> {
    (0..records)
        .map(|index| GraphMutation::NodeUpsert(fixture_node(index)))
        .chain((0..records).map(|index| GraphMutation::EdgeUpsert(fixture_edge(index, records))))
        .collect()
}

/// The bounded write under test: one node upsert plus one edge upsert.
fn bounded_batch(records: usize, ordinal: usize) -> GraphMutationBatch {
    GraphMutationBatch::new([
        GraphMutation::NodeUpsert(NodeRecord::new(
            format!("parity:write:node:{ordinal:03}"),
            ["ScaleFixture", "Record"],
            json!({ "name": "bounded write", "rank": ordinal % 97 }),
        )),
        GraphMutation::EdgeUpsert(EdgeRecord::new(
            format!("parity:write:edge:{ordinal:03}"),
            format!("parity:write:node:{ordinal:03}"),
            "WROTE",
            format!("parity:node:{:06}", records - 1),
            json!({ "weight": 0.5 }),
        )),
    ])
}

struct Window {
    deltas: Vec<u64>,
    elapsed: Vec<Duration>,
    clean: Option<usize>,
}

impl Window {
    fn clean_elapsed(&self) -> Duration {
        self.elapsed[self
            .clean
            .expect("the caller asserts a clean window before reading its time")]
    }

    fn deltas_line(&self) -> String {
        self.deltas
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Read the parity-audit counter around `apply`, exactly as the claim is stated:
/// reading before, mutation region, reading after.
fn window(
    store: &mut RedCoreGraphStore,
    mut apply: impl FnMut(&mut RedCoreGraphStore),
    attempts: usize,
) -> Window {
    let mut deltas = Vec::with_capacity(attempts);
    let mut elapsed = Vec::with_capacity(attempts);
    let mut clean = None;
    for attempt in 0..attempts {
        let before = graph_prolly_parity_validations();
        let started = Instant::now();
        apply(store);
        let taken = started.elapsed();
        let after = graph_prolly_parity_validations();
        deltas.push(after.saturating_sub(before));
        elapsed.push(taken);
        if after == before {
            clean = Some(attempt);
            break;
        }
    }
    Window {
        deltas,
        elapsed,
        clean,
    }
}

/// The regression oracle for this finding.
///
/// Asserted, as counted work rather than wall clock: no bounded write at either
/// tier runs a whole-graph parity audit, on the incremental branch or on the
/// staged branch; and the counter is live, because one explicit on-demand audit
/// moves it and accepts the tree it audited. Wall clock is logged next to it so
/// the counted verdict can be read against the cost that produced it.
#[test]
fn a_bounded_write_runs_no_whole_graph_parity_audit() {
    for records in TIERS {
        reset_graph_prolly_parity_validations();

        let mut store = RedCoreGraphStore::memory();
        let started = Instant::now();
        store
            .commit_batch(GraphMutationBatch::new(bulk_mutations(records)))
            .expect("the bulk fixture commit must publish");
        let bootstrap = started.elapsed();
        let bootstrap_delta = graph_prolly_parity_validations();
        assert_eq!(
            bootstrap_delta, 0,
            "the one-time bulk publish at {records} records is O(records) by design, but it must \
             not be an audit: audits are counted work, not a side effect of building a pack"
        );

        let mut steps = Vec::with_capacity(STEPS);
        for ordinal in 1..=STEPS {
            let window = window(
                &mut store,
                |store| {
                    store
                        .commit_batch(bounded_batch(records, ordinal))
                        .expect("a bounded upsert commit must publish");
                },
                WINDOW_ATTEMPTS,
            );
            assert!(
                window.clean.is_some(),
                "bounded upsert commit #{ordinal} at {records} stored records ran a whole-graph \
                 parity audit; observed counter deltas across {} attempts: [{}]",
                window.deltas.len(),
                window.deltas_line(),
            );
            steps.push(window.clean_elapsed());
        }

        // The staged branch as well: a batch containing a delete cannot be
        // expressed as an incremental delta, so it clones the whole in-memory
        // graph. It must also not audit.
        let victim = format!("parity:victim:{records}");
        store
            .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                NodeRecord::new(victim.clone(), ["ScaleFixture"], json!({ "probe": true })),
            )]))
            .expect("the probe victim must publish");
        let staged = window(
            &mut store,
            |store| {
                store
                    .commit_batch(GraphMutationBatch::new([GraphMutation::NodeDelete(
                        victim.clone(),
                    )]))
                    .expect("the delete batch must publish");
            },
            WINDOW_ATTEMPTS,
        );
        assert!(
            staged.clean.is_some(),
            "the staged (delete) branch at {records} stored records ran a whole-graph parity \
             audit; observed counter deltas: [{}]",
            staged.deltas_line(),
        );

        // Liveness: the counter must move for the thing it counts, and the audit
        // it counted must accept a real published tree.
        let audit_before = graph_prolly_parity_validations();
        let audit = store
            .verify_tree_parity()
            .expect("the on-demand audit must run")
            .expect("a published store has a tree to audit");
        assert_eq!(
            graph_prolly_parity_validations() - audit_before,
            1,
            "one on-demand audit must move the counter by exactly one, otherwise zero deltas \
             above are not evidence of anything"
        );
        assert!(
            audit.parity_holds(),
            "a store that published normally must be canonical for its own entries: {audit:?}"
        );

        let stats = store.stats().expect("stats must succeed");
        assert_eq!(
            audit.live_nodes, stats.nodes_total,
            "the audited live node count must be the store's live node count"
        );
        assert_eq!(
            audit.live_edges, stats.edges_total,
            "the audited live edge count must be the store's live edge count"
        );

        let steady = steps.iter().sum::<Duration>() / u32::try_from(steps.len()).unwrap();
        println!(
            "prolly_parity_boundary | tier {records:>6} stored records | bootstrap {bootstrap:?} | \
             steady commit mean {steady:?} over {STEPS} steps | audits on the write path: 0 | \
             audits on demand: 1 | tree entries {} | live nodes {} | live edges {}",
            audit.entries_total, audit.live_nodes, audit.live_edges,
        );
    }
}

/// The oracle is exercised, not just reachable.
///
/// Four plants, each asserting a specific error code, because "something failed"
/// is not evidence that the parity comparison ran:
///
/// 1. **The comparison arm, which is what only a from-scratch rebuild can see.**
///    One leaf holding 64 entries, written with the crate's own canonical leaf
///    encoder: its declared count matches its contents, no parent descriptor
///    disagrees with it, so the traversal accepts it -- and the canonical
///    rebuild splits those entries into several leaves, so the roots differ.
///    This is the shape a leaf that grew past the split threshold without being
///    re-chunked takes.
/// 2. **The count arm.** The right root with one entry too many declared.
/// 3. **Reversion.** Reverting a delta through the same incremental builder must
///    return the canonical shape for the reverted entry set; if it reuses stale
///    spines instead of re-chunking them, the audit catches it.
/// 4. **The blind spot, asserted rather than glossed.** A delta dropped before it
///    reached the tree, and never counted, leaves a tree that is canonical for
///    the entries it holds. The boundary form accepts it; the replay form, which
///    is handed the batch, refuses it.
#[test]
fn the_parity_audit_catches_a_planted_divergence() {
    let store = MemChunkStore::new();
    let entries = (0..64usize)
        .map(|index| GraphTreeEntry {
            key: format!("node/{index:06}"),
            kind: GraphObjectKind::Node,
            object_hash: Hash::of(format!("planted object {index}").as_bytes()),
            tombstone: false,
        })
        .collect::<Vec<_>>();
    let prior =
        build_prolly_tree_from_entries(&store, entries.clone()).expect("the prior tree must build");
    assert!(
        prior.root != Hash::of(&canonical_leaf_preimage(&entries).unwrap()),
        "the canonical build must split 64 entries across more than one leaf, otherwise plant 1 \
         below tests nothing"
    );

    let upserts = BTreeMap::from([(
        "node/999999".to_string(),
        GraphTreeEntry {
            key: "node/999999".to_string(),
            kind: GraphObjectKind::Node,
            object_hash: Hash::of(b"planted delta"),
            tombstone: false,
        },
    )]);
    let incremental = build_prolly_tree_incremental(&store, &prior, &upserts)
        .expect("the incremental build must succeed")
        .tree;

    // The control: a real incremental tree is canonical for its own entries, so
    // the refusals below are not an oracle that refuses everything.
    let audit = audit_published_prolly_parity(&store, &incremental)
        .expect("a real incremental tree must audit clean");
    assert!(audit.parity_holds(), "{audit:?}");
    assert_eq!(audit.entries_total, entries.len() + 1);
    assert_eq!(audit.live_nodes, entries.len() + 1);

    // Plant 1: one unsplit leaf holding the whole entry set. Consistent with
    // itself, not the canonical chunking of what it holds, and the root cannot
    // be repaired by any local consistency check -- reaching this shape is what
    // the oracle exists to refuse.
    let leaf_bytes = canonical_leaf_preimage(&entries).expect("the leaf encoder must run");
    let leaf_address = Hash::of(&leaf_bytes);
    store
        .write(&[(leaf_address, leaf_bytes.as_slice())])
        .expect("the planted leaf must be writable");
    let unsplit = GraphProllyTree {
        root: leaf_address,
        entries_total: entries.len(),
    };
    assert_eq!(
        audit_published_prolly_parity(&store, &unsplit)
            .expect_err("an unsplit leaf must be refused")
            .code,
        "graph_prolly_validation_divergence",
        "the canonical comparison must be what refuses an internally consistent but \
         non-canonically chunked tree"
    );

    // Plant 2: the right root, one entry too many declared. The traversal itself
    // refuses this one, before any rebuild: the root's own descriptor and the
    // declared count disagree.
    let miscounted = GraphProllyTree {
        root: incremental.root,
        entries_total: incremental.entries_total + 1,
    };
    assert_eq!(
        audit_published_prolly_parity(&store, &miscounted)
            .expect_err("a miscounted tree must be refused")
            .code,
        "graph_tree_entry_count_mismatch",
    );

    // Plant 3: revert the delta through the incremental builder. A tombstone
    // keeps the entry count, so this is a shape assertion: the reverted tree must
    // be canonical for the entries it now holds.
    let reverted = build_prolly_tree_incremental(
        &store,
        &incremental,
        &BTreeMap::from([(
            "node/999999".to_string(),
            GraphTreeEntry {
                key: "node/999999".to_string(),
                kind: GraphObjectKind::Node,
                object_hash: Hash::of(b"planted delta"),
                tombstone: true,
            },
        )]),
    )
    .expect("the reverting build must succeed")
    .tree;
    let reverted_audit = audit_published_prolly_parity(&store, &reverted)
        .expect("a reverted tree must still be canonical for its own entries");
    assert!(
        reverted_audit.parity_holds(),
        "reverting a delta must return the canonical shape for the reverted entry set: \
         {reverted_audit:?}"
    );
    assert_eq!(
        reverted_audit.canonical_entries_total, incremental.entries_total,
        "the reverted tree keeps its tombstone entry, so its count is the incremental count"
    );

    // Plant 4, the blind spot, asserted so it is a documented property rather
    // than a surprise: a tree that never received a delta and never counted it is
    // canonical, and the boundary form accepts it.
    assert!(audit_published_prolly_parity(&store, &prior)
        .expect("the untouched prior tree is canonical")
        .parity_holds());
    assert_eq!(
        validate_incremental_prolly_parity(&store, &prior, &upserts, &prior)
            .expect_err("the replay form, which has the batch, must see the dropped delta")
            .code,
        "graph_prolly_validation_divergence",
    );
}

/// The restart checkpoint, on a store that reopened normally.
///
/// Two claims, both measured here rather than asserted from the code:
///
/// 1. The checkpoint runs. The evidence is per-store rather than the process-wide
///    counter: the recovery profile of the reopened store must carry a
///    `prolly_parity_audit` sample, which only the checkpoint writes.
/// 2. The checkpoint is a constant factor on a path that is already O(records).
///    The audit is timed on its own through the public on-demand entry point at
///    the same size as the store that was reopened, and printed against the
///    reopen that it rode on.
#[test]
fn a_durable_restart_audits_the_recovered_head_at_the_boundary() {
    let dir = tempfile::tempdir().expect("a temporary RedCore directory");
    let options = || RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    };
    // Enough records that the audit is measurable next to the reopen it rides on,
    // and small enough that the one-time bootstrap stays around a second.
    const SEED_RECORDS: usize = 2_000;

    let expected = {
        let mut store =
            RedCoreGraphStore::open(dir.path(), options()).expect("the store must open");
        let mut mutations = (0..SEED_RECORDS)
            .map(|index| GraphMutation::NodeUpsert(fixture_node(index)))
            .collect::<Vec<_>>();
        mutations.push(GraphMutation::EdgeUpsert(fixture_edge(0, SEED_RECORDS)));
        store
            .commit_batch(GraphMutationBatch::new(mutations))
            .expect("the bootstrap commit must publish");
        store
            .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                NodeRecord::new("parity:later", ["ScaleFixture"], json!({ "rank": 7 })),
            )]))
            .expect("the bounded commit must publish");
        store.stats().expect("stats must succeed")
    };

    let reopen_started = Instant::now();
    let reopened = RedCoreGraphStore::open(dir.path(), options()).expect("the store must reopen");
    let reopen = reopen_started.elapsed();
    let stages = reopened
        .recovery_profile()
        .samples
        .iter()
        .map(|sample| sample.stage.as_str())
        .collect::<Vec<_>>();
    assert!(
        stages.contains(&"prolly_parity_audit"),
        "the restart checkpoint must audit the recovered head and record the stage; recovery \
         profile stages were {stages:?}"
    );

    let audit_started = Instant::now();
    let audit = reopened
        .verify_tree_parity()
        .expect("the on-demand audit must run")
        .expect("a recovered store has a published tree");
    let audit_elapsed = audit_started.elapsed();
    assert!(audit.parity_holds(), "{audit:?}");
    assert_eq!(audit.live_nodes, expected.nodes_total);
    assert_eq!(audit.live_edges, expected.edges_total);
    println!(
        "prolly_parity_boundary | durable restart at {} stored records | recovery stages {stages:?} \
         | reopen {reopen:?} | one whole-graph audit at this size {audit_elapsed:?} | audited entries {} \
         | live nodes {} | live edges {}",
        expected.nodes_total + expected.edges_total,
        audit.entries_total,
        audit.live_nodes,
        audit.live_edges,
    );
}

/// The bulk compile carries no audit, because it has no prior tree to compare
/// against: a full pack build is the reference the oracle compares *to*, so
/// counting it as an audit would make every zero-delta assertion above
/// meaningless.
#[test]
fn the_bulk_compile_is_not_an_audit() {
    let snapshot = GraphSnapshot {
        version: 1,
        nodes: (0..32).map(fixture_node).collect(),
        edges: Vec::new(),
        historical_node_ids: BTreeSet::new(),
        historical_node_history_complete: false,
    };
    reset_graph_prolly_parity_validations();
    let pack = compile_graph_pack(
        &snapshot,
        GraphCompileOptions {
            include_payloads: false,
            sync_scope_only: false,
            exclude_local_only: false,
            ..GraphCompileOptions::default()
        },
    )
    .expect("the bulk compile must succeed");
    assert_eq!(
        graph_prolly_parity_validations(),
        0,
        "building a pack from a snapshot has no prior tree to validate against, and must not be \
         counted as a parity audit"
    );
    assert_eq!(pack.tree.entries_total, 32);
    assert!(
        pack.tree_transport_complete,
        "a full compile transports every chunk it references"
    );
}
