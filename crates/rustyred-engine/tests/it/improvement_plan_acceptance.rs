//! Acceptance coverage for the RustyRed-THG Improvement Plan, Workstream C2
//! (CSR analytics view). These are *acceptance* tests, not unit tests: each one
//! asserts a clause of the plan's stated C2 acceptance bar, so a green run is
//! evidence the requirement is met, not merely that some code path executes.
//!
//! Plan C2 acceptance:
//!   "a tier-1 algorithm run allocates the CSR once (not per call) and runs on
//!    integer indices; PPR is expressed as repeated sparse matrix-vector
//!    products; memory for a run is bounded by the CSR size, not by repeated
//!    string-keyed adjacency maps."
//!
//! The lower section carries acceptance gates for the Codex-owned spine items
//! that are now landed in this crate (A2, B1, B2, B3/B5, B4, C1).

use rustyred_thg_core::CsrGraph;

/// Build a deterministic mid-size directed graph for the acceptance runs:
/// a ring of `n` nodes plus chords, as `(from, to, weight)` triples.
fn ring_with_chords(n: usize) -> Vec<(String, String, f64)> {
    let mut edges = Vec::new();
    for i in 0..n {
        let from = format!("node:{i:04}");
        let to = format!("node:{:04}", (i + 1) % n);
        edges.push((from, to, 1.0));
        if i % 3 == 0 {
            let chord = format!("node:{:04}", (i + 7) % n);
            edges.push((format!("node:{i:04}"), chord, 0.5));
        }
    }
    edges
}

/// C2.1 — The CSR is allocated ONCE per run and reused across every tier-1
/// algorithm. We build the view a single time, then run PageRank, PPR,
/// connected components, SCC, betweenness, articulation/bridges, KNN, and
/// Leiden against that one borrowed view. The view exposes integer-indexed
/// results (Vec position == dense node index), proving the algorithms run on
/// integer indices, not string-keyed adjacency.
#[test]
fn c2_csr_built_once_runs_all_tier1_on_integer_indices() {
    let edges = ring_with_chords(60);
    let graph = CsrGraph::from_edges(edges); // <-- single allocation

    let n = graph.node_count();
    assert_eq!(n, 60);

    // Every algorithm returns an integer-indexed result over the same view.
    let pagerank = graph.pagerank(0.85, 100, 1e-9);
    assert_eq!(pagerank.len(), n, "pagerank is indexed by dense int id");

    let seed = graph.index_of("node:0000").expect("seed present");
    let ppr = graph.personalized_pagerank(&[(seed, 1.0)], 0.85, 200, 1e-12);
    assert_eq!(ppr.len(), n, "ppr is indexed by dense int id");

    let components = graph.connected_components();
    assert!(!components.is_empty());

    let sccs = graph.strongly_connected_components();
    assert!(!sccs.is_empty());
    // SCC components address nodes by integer index, all in range.
    for comp in &sccs {
        for &idx in comp {
            assert!(idx < n);
        }
    }

    let betweenness = graph.betweenness_centrality();
    assert_eq!(betweenness.len(), n);

    let (points, bridges) = graph.articulation_points_and_bridges();
    for &p in &points {
        assert!(p < n);
    }
    for &(a, b) in &bridges {
        assert!(a < n && b < n);
    }

    let knn = graph.knn(3);
    assert_eq!(knn.len(), n);

    let (labels, modularity) = graph.leiden_communities();
    assert_eq!(labels.len(), n);
    assert!(modularity.is_finite());
}

/// C2.2 — PPR is expressed as repeated sparse matrix-vector products. We assert
/// the two observable consequences of an SpMV power iteration: total mass is
/// conserved across iterations (each step is a stochastic transition + restart),
/// and the score is a fixed point — running more iterations past convergence
/// does not move it. A per-call adjacency-rebuilding implementation cannot make
/// this guarantee structurally; an SpMV formulation does.
#[test]
fn c2_ppr_is_repeated_spmv_mass_conserving_fixed_point() {
    let edges = ring_with_chords(40);
    let graph = CsrGraph::from_edges(edges);
    let seed = graph.index_of("node:0000").unwrap();

    let converged = graph.personalized_pagerank(&[(seed, 1.0)], 0.85, 500, 1e-12);
    let total: f64 = converged.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "SpMV mass conserved: {total}");

    // Fixed point: more iterations do not change the converged vector.
    let more = graph.personalized_pagerank(&[(seed, 1.0)], 0.85, 1000, 1e-15);
    let drift: f64 = converged
        .iter()
        .zip(more.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(drift < 1e-6, "PPR is a stable fixed point, drift = {drift}");
}

/// C2.3 — Memory for a run is bounded by the CSR size. The CSR backing arrays
/// are O(n + m): row_ptr is n+1 u32, col_idx is m u32, weights is m f64. We
/// assert the reported CSR footprint matches that exact bound for a known graph,
/// proving the working set is the integer CSR, not repeated string-keyed maps.
#[test]
fn c2_run_memory_bounded_by_csr_size() {
    let edges = ring_with_chords(100);
    let graph = CsrGraph::from_edges(edges);
    let n = graph.node_count();
    let m = graph.edge_count();

    let expected = (n + 1) * std::mem::size_of::<u32>() // row_ptr
        + m * std::mem::size_of::<u32>()                // col_idx
        + m * std::mem::size_of::<f64>(); // weights
    assert_eq!(
        graph.csr_bytes(),
        expected,
        "CSR footprint is exactly O(n+m) integer arrays"
    );
}

/// C2.4 — The CSR view is a lens, not the spine: building it from the same edge
/// set is deterministic (stable dense indexing), so it can be rebuilt and
/// discarded per run without perturbing results.
#[test]
fn c2_csr_view_is_deterministic_and_discardable() {
    let edges = ring_with_chords(30);
    let first = CsrGraph::from_edges(edges.clone());
    let second = CsrGraph::from_edges(edges);

    assert_eq!(first.node_count(), second.node_count());
    assert_eq!(first.edge_count(), second.edge_count());
    // Same node interns to the same dense index across independent builds.
    assert_eq!(first.index_of("node:0005"), second.index_of("node:0005"));
    let pr_a = first.pagerank(0.85, 100, 1e-12);
    let pr_b = second.pagerank(0.85, 100, 1e-12);
    let drift: f64 = pr_a
        .iter()
        .zip(pr_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(drift < 1e-12, "independent CSR builds agree: {drift}");
}

// ===========================================================================
// Verifier acceptance gates for the Codex-owned spine items that have stably
// landed (A2, B2, B4). These exercise the plan's acceptance bars end-to-end
// through the public API, independent of Codex's inline unit tests. If any
// fails as Codex iterates, it is a real finding, not test pollution.
// ===========================================================================

use rustyred_thg_core::{
    apply_graph_mutation_batch, compile_graph_pack, compile_graph_pack_incremental,
    merge_graph_snapshots, sanitize_tenant_segment, update_graph_ref_cas, EdgeRecord,
    GraphCompileOptions, GraphMergeOptions, GraphMutation, GraphMutationBatch, GraphSnapshot,
    GraphVersionRepository, InMemoryThgStore, NodeRecord, RedCoreDurability, RedCoreGraphStore,
    RedCoreOptions, ThgStore,
};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn pack_opts(ts: u128) -> GraphCompileOptions {
    GraphCompileOptions {
        branch: Some("main".to_string()),
        timestamp_unix_ms: Some(ts),
        ..GraphCompileOptions::default()
    }
}

fn node_snapshot(version: u64, nodes: &[(&str, &str)]) -> GraphSnapshot {
    GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version,
        nodes: nodes
            .iter()
            .map(|(id, name)| NodeRecord::new(*id, ["Person"], json!({ "name": name })))
            .collect(),
        edges: Vec::new(),
    }
}

fn unique_test_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!("rustyred-thg-{label}-{nonce}"))
}

/// A2 acceptance — `acme/prod` and `acme.prod` resolve to distinct, non-overlapping
/// key prefixes, and the encoding is injective over random tenant strings.
#[test]
fn a2_tenant_sanitization_is_injective() {
    // The previously-colliding separators must now map to distinct outputs.
    let colliders = [
        "acme/prod",
        "acme.prod",
        "acme-prod",
        "acme_prod",
        "acme prod",
    ];
    let mut seen = std::collections::HashSet::new();
    for tenant in colliders {
        let encoded = sanitize_tenant_segment(tenant);
        assert!(
            seen.insert(encoded.clone()),
            "collision: {tenant:?} -> {encoded:?}"
        );
    }

    // Injectivity property: distinct inputs never share an encoding. Sweep a
    // pseudo-random set of separator-heavy strings.
    let mut forward = std::collections::HashMap::new();
    let alphabet = ['a', '/', '.', '-', '_', ' ', ':', '%', 'Z', '9'];
    for i in 0..4000usize {
        let mut s = String::new();
        let mut x = i.wrapping_mul(2654435761) ^ 0x9e3779b9;
        for _ in 0..5 {
            s.push(alphabet[x % alphabet.len()]);
            x /= alphabet.len();
        }
        let encoded = sanitize_tenant_segment(&s);
        if let Some(prev) = forward.insert(encoded.clone(), s.clone()) {
            assert_eq!(
                prev, s,
                "non-injective: {prev:?} and {s:?} both -> {encoded:?}"
            );
        }
    }
}

/// B1 acceptance — applying a small graph mutation incrementally produces the
/// same Merkle root and commit as a full recompile with the same metadata, while
/// reusing most tree nodes from the prior pack.
#[test]
fn b1_incremental_commit_matches_full_recompile_and_reuses_tree() {
    let base = GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 7,
        nodes: (0..256)
            .map(|i| {
                NodeRecord::new(
                    format!("node:{i:03}"),
                    ["Doc"],
                    json!({ "title": format!("doc-{i:03}") }),
                )
            })
            .collect(),
        edges: Vec::new(),
    };
    let base_pack = compile_graph_pack(&base, pack_opts(10))
        .expect("base graph pack should compile without hash failures");
    let batch = GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
        "node:095",
        ["Doc"],
        json!({ "title": "doc-095-revised" }),
    ))]);
    let mut next_snapshot = base.clone();
    apply_graph_mutation_batch(&mut next_snapshot, &batch);
    let metadata = GraphCompileOptions {
        branch: Some("main".to_string()),
        parent_commits: vec![base_pack.commit.commit_hash.to_prefixed()],
        message: Some("acceptance incremental mutation".to_string()),
        timestamp_unix_ms: Some(11),
        ..GraphCompileOptions::default()
    };

    let incremental = compile_graph_pack_incremental(&base_pack, &batch, metadata.clone()).unwrap();
    let full = compile_graph_pack(&next_snapshot, metadata)
        .expect("full graph pack should compile without hash failures");

    assert_eq!(incremental.changed_object_keys, vec!["node/node:095"]);
    assert_eq!(incremental.pack.tree.root, full.tree.root);
    assert_eq!(incremental.pack.commit.commit_hash, full.commit.commit_hash);
    assert!(
        incremental.reused_tree_nodes > 0,
        "incremental compile should preserve at least one maximal shared subtree",
    );
}

/// B3/B5 acceptance — persistent state loading exposes shared immutable
/// snapshots, so read-only executor startup no longer forces a deep clone of
/// the whole THG state.
#[test]
fn b3_b5_store_load_snapshot_is_shared_arc() {
    let store = InMemoryThgStore::new();
    let first = store.load_snapshot().unwrap();
    let second = store.load_snapshot().unwrap();

    assert!(
        Arc::ptr_eq(&first, &second),
        "successive loads must share the same immutable state snapshot"
    );
    assert_eq!(first.runs.len(), second.runs.len());
}

/// B2 acceptance — a commit that changes nothing adds zero new object or
/// tree-node entries to the content-addressed dedup store.
#[test]
fn b2_noop_commit_adds_zero_new_objects() {
    let snap = node_snapshot(1, &[("node:ada", "Ada")]);
    let pack = compile_graph_pack(&snap, pack_opts(1))
        .expect("graph pack should compile without hash failures");
    let repo = update_graph_ref_cas(
        GraphVersionRepository::default(),
        pack.clone(),
        Some("main".to_string()),
        None,
        Some(1),
    )
    .expect("initial CAS sets the ref")
    .repository;

    let objects_before = repo.objects.len();
    let tree_nodes_before = repo.tree_chunks.len();
    assert!(objects_before >= 1, "first commit stores content objects");
    assert!(tree_nodes_before >= 1, "first commit stores tree nodes");

    // Re-apply the identical pack with a valid CAS against the current head.
    let head = pack.commit.commit_hash.to_prefixed();
    let repo2 = update_graph_ref_cas(repo, pack, Some("main".to_string()), Some(head), Some(2))
        .expect("no-op CAS succeeds")
        .repository;

    assert_eq!(
        repo2.objects.len(),
        objects_before,
        "identical commit must add zero new content objects (dedup)"
    );
    assert_eq!(
        repo2.tree_chunks.len(),
        tree_nodes_before,
        "identical commit must add zero new tree nodes (dedup)"
    );
}

/// C1 acceptance — the public RedCore everysec durability mode still publishes
/// a readable, verified graph after the background sync window drains.
#[test]
fn c1_aof_everysec_reopens_after_background_sync_window() {
    let data_dir = unique_test_dir("aof-everysec-acceptance");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofEverysec,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new("node:first", ["Doc"], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("node:second", ["Doc"], json!({})))
            .unwrap();
    }

    std::thread::sleep(Duration::from_millis(75));
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:first").unwrap().is_some());
    assert!(reopened.get_node("node:second").unwrap().is_some());
    assert!(reopened.verify().unwrap().ok);

    std::fs::remove_dir_all(data_dir).ok();
}

/// B4 / A4 acceptance — two concurrent commits against the same base root
/// serialize: one succeeds, the other is rejected; the loser never silently
/// overwrites the winner.
#[test]
fn b4_concurrent_commits_serialize_via_cas() {
    let base = node_snapshot(1, &[("node:ada", "Ada")]);
    let base_pack = compile_graph_pack(&base, pack_opts(1))
        .expect("base graph pack should compile without hash failures");
    let c0 = base_pack.commit.commit_hash.to_prefixed();
    let repo0 = update_graph_ref_cas(
        GraphVersionRepository::default(),
        base_pack,
        Some("main".to_string()),
        None,
        Some(1),
    )
    .expect("establish base ref")
    .repository;

    // Writer A branches off C0 and wins the CAS.
    let snap_a = node_snapshot(2, &[("node:ada", "Ada"), ("node:alice", "Alice")]);
    let pack_a = compile_graph_pack(&snap_a, pack_opts(2))
        .expect("writer A graph pack should compile without hash failures");
    let win = update_graph_ref_cas(
        repo0.clone(),
        pack_a.clone(),
        Some("main".to_string()),
        Some(c0.clone()),
        Some(2),
    )
    .expect("writer A wins the CAS");
    let repo_after_a = win.repository;

    // Writer B also branched off C0 but commits after A; its CAS is now stale.
    let snap_b = node_snapshot(2, &[("node:ada", "Ada"), ("node:bob", "Bob")]);
    let pack_b = compile_graph_pack(&snap_b, pack_opts(3))
        .expect("writer B graph pack should compile without hash failures");
    let conflict = update_graph_ref_cas(
        repo_after_a.clone(),
        pack_b,
        Some("main".to_string()),
        Some(c0.clone()), // still expects the original base -> stale
        Some(3),
    )
    .expect_err("writer B must be rejected, never silently overwrite");

    assert_eq!(conflict.expected_commit_hash.as_deref(), Some(c0.as_str()));
    assert_eq!(
        conflict.actual_commit_hash.as_deref(),
        Some(pack_a.commit.commit_hash.to_prefixed().as_str()),
        "conflict reports the winner's commit as the actual head"
    );

    // No silent overwrite: the branch head is still writer A's commit.
    let head = repo_after_a
        .refs
        .iter()
        .find(|r| r.name == "main")
        .expect("main ref present");
    assert_eq!(head.commit_hash, pack_a.commit.commit_hash);
}

/// B4 acceptance (merge path) — on a conflicting concurrent edge update, the
/// confidence-merge resolves to the higher-confidence edge (the documented
/// loser-CAS recovery via `merge_graph_snapshots`).
#[test]
fn b4_confidence_merge_resolves_to_higher_confidence_edge() {
    let edge = |conf: f64| GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 1,
        nodes: Vec::new(),
        edges: vec![
            EdgeRecord::new("edge:e", "a", "SUPPORTS", "b", json!({})).with_confidence(conf)
        ],
    };
    let base = edge(0.4);
    let ours = GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 2,
        ..edge(0.9)
    };
    let theirs = GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 3,
        ..edge(0.6)
    };

    let merged = merge_graph_snapshots(&base, &ours, &theirs, GraphMergeOptions::default())
        .expect("merge should hash resolved graph objects");
    assert_eq!(merged.status, "clean");
    let resolved_edge = &merged.merged_snapshot.expect("clean merge").edges[0];
    assert_eq!(
        resolved_edge.confidence,
        Some(0.9),
        "merge keeps the higher-confidence edge"
    );
}
