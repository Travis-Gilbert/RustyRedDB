//! Acceptance suite for the shadow epistemic graph.
//!
//! Proves the observable acceptance criteria of SPEC-EPISTEMICRAG and
//! SPEC-EPISTEMICRAG-INSTANT end-to-end against the shipped core API
//! (`rustyred_thg_core::epistemic`). Criteria that live in sibling crates are
//! covered there and noted inline:
//!   - INSTANT #4 (code drift) -> `rustyred-thg-code::code_epistemic_hook` tests.
//!   - EPISTEMICRAG #2 (recall include_epistemic) -> `rustyred-thg-memory` tests.
//!
//! These are integration tests over public APIs only; they touch no private
//! state and add zero source surface, so they are safe to co-author beside the
//! crate they verify.

use std::collections::HashMap;
use std::sync::Mutex;

use rustyred_thg_core::epistemic::{
    effective_confidence, epistemic_egraph_dedup, epistemic_shadow_node_id, epistemic_shadow_ppr,
    read_epistemic_shadow, read_same_eclass, run_epistemic_cron_pass,
    source_reliability_from_verification_outcomes, structural_epistemic_pass, EpistemicAnnotation,
    EpistemicAnnotations, EpistemicCandidatePair, EpistemicCongruence, EpistemicCronInput,
    EpistemicDedupConfig, EpistemicEnricher, EpistemicEnrichmentError, EpistemicEnrichmentMode,
    EpistemicSourceKind, GroundedExtensionStatus, PredictedEdgePointer, SourceReliability,
    StructuralEpistemicConfig, StructuralEpistemicInput, VerificationReliabilityObservation,
    DEFAULT_EPISTEMIC_ENGINE_VERSION, EPISTEMIC_DETERMINISTIC_FALLBACK_ENV, EPISTEMIC_SHADOW_LABEL,
    SAME_ECLASS,
};
use rustyred_thg_core::{EdgeRecord, GraphRead, InMemoryGraphStore, NodeQuery, NodeRecord};
use serde_json::json;

static ENV_LOCK: Mutex<()> = Mutex::new(());

// --------------------------------------------------------------------------- //
// Test fixtures
// --------------------------------------------------------------------------- //

fn claim(id: &str, text: &str) -> NodeRecord {
    NodeRecord::new(id, ["Claim"], json!({ "content": text }))
}

fn shadow_count(store: &InMemoryGraphStore) -> usize {
    store
        .query_nodes(NodeQuery::label(EPISTEMIC_SHADOW_LABEL).with_limit(100_000))
        .len()
}

/// An enricher that returns canned annotations (the success path / a stand-in
/// for Theseus over gRPC).
struct StaticEnricher(EpistemicAnnotations);
impl EpistemicEnricher for StaticEnricher {
    fn enrich(
        &self,
        _subgraph: rustyred_thg_core::epistemic::UserSubgraph,
        _mode: EpistemicEnrichmentMode,
    ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError> {
        Ok(self.0.clone())
    }
}

/// An enricher that always fails, simulating a dropped gRPC connection.
struct DroppingEnricher;
impl EpistemicEnricher for DroppingEnricher {
    fn enrich(
        &self,
        _subgraph: rustyred_thg_core::epistemic::UserSubgraph,
        _mode: EpistemicEnrichmentMode,
    ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError> {
        Err(EpistemicEnrichmentError::new(
            "grpc_drop",
            "deadline exceeded",
        ))
    }
}

fn run_structural(store: &mut InMemoryGraphStore, batch: &[&str], pairs: Vec<(&str, &str)>) {
    let candidate_pairs = pairs
        .into_iter()
        .map(|(l, r)| EpistemicCandidatePair {
            left_content_id: l.to_string(),
            right_content_id: r.to_string(),
        })
        .collect();
    structural_epistemic_pass(
        store,
        StructuralEpistemicInput {
            batch_node_ids: batch.iter().map(|s| s.to_string()).collect(),
            candidate_pairs,
            explicit_relations: Vec::new(),
            config: StructuralEpistemicConfig::default(),
        },
    )
    .expect("structural pass");
}

// --------------------------------------------------------------------------- //
// SPEC-EPISTEMICRAG-INSTANT
// --------------------------------------------------------------------------- //

#[test]
fn instant_tier1_populates_with_no_model() {
    // INSTANT #2: Tier 1 fields populate on any ingested subgraph, no model.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("a", "claim a")).unwrap();
    store.upsert_node(claim("b", "claim b")).unwrap();
    store.upsert_node(claim("orphan", "lonely")).unwrap();
    store
        .upsert_edge(EdgeRecord::new("e1", "a", "CITES", "b", json!({})))
        .unwrap();

    run_structural(&mut store, &["a", "b", "orphan"], vec![]);

    let sb = read_epistemic_shadow(&store, "b").expect("shadow b");
    assert_eq!(sb.support_in_degree, 1, "b is supported by a (CITES)");
    assert!(!sb.unsupported_leaf);
    let sa = read_epistemic_shadow(&store, "a").expect("shadow a");
    assert!(sa.unsupported_leaf, "a has no supporting in-edge");
    let so = read_epistemic_shadow(&store, "orphan").expect("shadow orphan");
    assert!(so.orphan, "disconnected node is an orphan");
}

#[test]
fn instant_planted_contradiction_undercuts_and_grounded_marks_loser_out() {
    // INSTANT #3 / EPISTEMICRAG (grounded extension): a contradiction becomes an
    // UNDERCUTS edge and the grounded extension marks the loser OUT. Lexical
    // pair inference is an explicit deterministic fallback, not the product
    // default.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV, "1");
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(claim("A", "the cache is safe under concurrency"))
        .unwrap();
    store
        .upsert_node(claim("B", "the cache is not safe under concurrency"))
        .unwrap();

    run_structural(&mut store, &["A", "B"], vec![("A", "B")]);
    std::env::remove_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV);

    let sa = read_epistemic_shadow(&store, "A").expect("shadow A");
    let sb = read_epistemic_shadow(&store, "B").expect("shadow B");
    let statuses = [sa.grounded_extension_status, sb.grounded_extension_status];
    assert!(
        statuses.contains(&GroundedExtensionStatus::Out),
        "the loser is marked out (got {statuses:?})"
    );
    assert!(
        statuses.contains(&GroundedExtensionStatus::In),
        "the winner stays in"
    );
}

#[test]
fn instant_checked_pairs_are_hnsw_bounded_not_quadratic() {
    // INSTANT #5: the checked-pair count scales with k*claims, not claims^2.
    let mut store = InMemoryGraphStore::new();
    let n = 20usize;
    let ids: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
    for id in &ids {
        store.upsert_node(claim(id, "neutral statement")).unwrap();
    }
    let k = 4usize;
    // A k-bounded candidate set: k neighbors per claim.
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in 1..=k {
            pairs.push(EpistemicCandidatePair {
                left_content_id: format!("c{i}"),
                right_content_id: format!("c{}", (i + j) % n),
            });
        }
    }
    let readout = structural_epistemic_pass(
        &mut store,
        StructuralEpistemicInput {
            batch_node_ids: ids.clone(),
            candidate_pairs: pairs,
            explicit_relations: Vec::new(),
            config: StructuralEpistemicConfig {
                candidate_top_k: k,
                ..StructuralEpistemicConfig::default()
            },
        },
    )
    .unwrap();
    assert!(
        readout.checked_pair_count <= k * n,
        "checked {} <= k*n {}",
        readout.checked_pair_count,
        k * n
    );
    assert!(
        readout.checked_pair_count < n * (n - 1) / 2,
        "bounded well below all-pairs {}",
        n * (n - 1) / 2
    );
}

#[test]
fn instant_shadow_carries_source_kind_structural() {
    // INSTANT #6 / EPISTEMICRAG #7: structural fields are tagged source_kind.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("x", "a claim")).unwrap();
    run_structural(&mut store, &["x"], vec![]);
    let sx = read_epistemic_shadow(&store, "x").expect("shadow x");
    assert_eq!(sx.source_kind, EpistemicSourceKind::Structural);
    assert!(
        sx.predicted_edges.is_empty(),
        "no learned predicted edges yet"
    );
}

#[test]
fn instant_structural_pass_is_idempotent() {
    // INSTANT (idempotency) / EPISTEMICRAG #6: a re-run creates no duplicates.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("a", "alpha")).unwrap();
    store.upsert_node(claim("b", "beta")).unwrap();
    store
        .upsert_edge(EdgeRecord::new("e", "a", "CITES", "b", json!({})))
        .unwrap();
    run_structural(&mut store, &["a", "b"], vec![]);
    let first = shadow_count(&store);
    run_structural(&mut store, &["a", "b"], vec![]);
    let second = shadow_count(&store);
    assert_eq!(first, second, "no duplicate shadow nodes on re-run");
}

// --------------------------------------------------------------------------- //
// SPEC-EPISTEMICRAG (learned layer + cron)
// --------------------------------------------------------------------------- //

fn learned_annotation(content_id: &str, predicted_target: &str) -> EpistemicAnnotation {
    EpistemicAnnotation {
        content_node_id: content_id.to_string(),
        predicted_edges: vec![PredictedEdgePointer {
            target_content_id: predicted_target.to_string(),
            relation: "RELATES".to_string(),
            confidence: 0.9,
            quarantine: true,
        }],
        completion_confidence: Some(0.9),
        structural_role_vector: vec![0.1, 0.2, 0.3],
        source_reliability: Some(SourceReliability {
            alpha: 4.0,
            beta: 1.0,
            mean: 0.8,
        }),
        community_id: Some("c1".to_string()),
        grounded_extension_status: None,
    }
}

#[test]
fn epistemicrag_cron_adds_learned_fields_preserving_structural() {
    // EPISTEMICRAG #1 + #7 + INSTANT #7: after the structural pass then a cron
    // pass, the shadow has non-null structural fields AND learned fields, with
    // both source kinds distinguishable; structural is untouched.
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(claim("doc", "a documented claim"))
        .unwrap();
    store.upsert_node(claim("dep", "a dependency")).unwrap();
    store
        .upsert_edge(EdgeRecord::new("e", "dep", "SUPPORTS", "doc", json!({})))
        .unwrap();
    run_structural(&mut store, &["doc", "dep"], vec![]);

    let before = read_epistemic_shadow(&store, "doc").unwrap();
    assert_eq!(before.source_kind, EpistemicSourceKind::Structural);
    let structural_support = before.support_in_degree;

    let enricher = StaticEnricher(EpistemicAnnotations {
        annotations: vec![learned_annotation("doc", "ghost")],
        support_relations: Vec::new(),
        attack_relations: Vec::new(),
    });
    let report = run_epistemic_cron_pass(
        &mut store,
        EpistemicCronInput {
            content_node_ids: vec!["doc".into(), "dep".into()],
            ..EpistemicCronInput::default()
        },
        &enricher,
    )
    .unwrap();
    assert!(report.grpc_ok);
    assert_eq!(report.annotations_received, 1);

    let after = read_epistemic_shadow(&store, "doc").unwrap();
    // Structural preserved.
    assert_eq!(
        after.support_in_degree, structural_support,
        "structural in-degree preserved"
    );
    assert!(after.grounded_extension_status == before.grounded_extension_status);
    // Learned added + distinguishable.
    assert_eq!(after.source_kind, EpistemicSourceKind::Mixed, "now mixed");
    assert_eq!(after.completion_confidence, Some(0.9));
    assert!(after.source_reliability.is_some());
    assert!(
        after
            .field_provenance
            .values()
            .any(|p| p.source_kind == EpistemicSourceKind::Learned),
        "learned field provenance present"
    );
    assert!(
        after
            .field_provenance
            .values()
            .any(|p| p.source_kind == EpistemicSourceKind::Structural),
        "structural field provenance preserved"
    );
}

#[test]
fn source_reliability_from_verification_outcomes_updates_effective_confidence() {
    let corroborated = source_reliability_from_verification_outcomes(
        1.0,
        1.0,
        vec![
            VerificationReliabilityObservation {
                source_id: "source:a".to_string(),
                corroborated: true,
                weight: 2.0,
            },
            VerificationReliabilityObservation {
                source_id: "source:a".to_string(),
                corroborated: true,
                weight: 1.0,
            },
        ],
    );
    let contradicted = source_reliability_from_verification_outcomes(
        1.0,
        1.0,
        vec![
            VerificationReliabilityObservation {
                source_id: "source:b".to_string(),
                corroborated: false,
                weight: 2.0,
            },
            VerificationReliabilityObservation {
                source_id: "source:b".to_string(),
                corroborated: false,
                weight: 1.0,
            },
        ],
    );

    assert_eq!(corroborated.alpha, 4.0);
    assert_eq!(corroborated.beta, 1.0);
    assert_eq!(contradicted.alpha, 1.0);
    assert_eq!(contradicted.beta, 4.0);
    assert!(effective_confidence(0.9, Some(&corroborated)) > 0.7);
    assert!(effective_confidence(0.9, Some(&contradicted)) < 0.3);
    assert_eq!(effective_confidence(0.9, None), 0.9);
}

#[test]
fn epistemicrag_predicted_edges_are_quarantined_pointers_not_injected_nodes() {
    // EPISTEMICRAG #5: predicted-edge targets appear as pointers, never injected
    // as full content nodes.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("doc", "a claim")).unwrap();
    run_structural(&mut store, &["doc"], vec![]);

    let enricher = StaticEnricher(EpistemicAnnotations {
        annotations: vec![learned_annotation("doc", "ghost_target")],
        support_relations: Vec::new(),
        attack_relations: Vec::new(),
    });
    run_epistemic_cron_pass(
        &mut store,
        EpistemicCronInput {
            content_node_ids: vec!["doc".into()],
            ..EpistemicCronInput::default()
        },
        &enricher,
    )
    .unwrap();

    let shadow = read_epistemic_shadow(&store, "doc").unwrap();
    assert_eq!(shadow.predicted_edges.len(), 1);
    assert!(
        shadow.predicted_edges[0].quarantine,
        "predicted edge quarantined"
    );
    assert_eq!(shadow.predicted_edges[0].target_content_id, "ghost_target");
    // The predicted target was never created as a node.
    assert!(
        GraphRead::get_node(&store, "ghost_target").is_none(),
        "predicted target must not be injected as a content node"
    );
}

#[test]
fn epistemicrag_grpc_drop_completes_cleanly_without_partial_writes() {
    // EPISTEMICRAG #4: a simulated gRPC drop leaves the cron completing cleanly,
    // no partial learned-field writes, and no deleted shadow nodes.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("doc", "a claim")).unwrap();
    run_structural(&mut store, &["doc"], vec![]);
    let before = shadow_count(&store);
    let before_shadow = read_epistemic_shadow(&store, "doc").unwrap();
    assert_eq!(before_shadow.source_kind, EpistemicSourceKind::Structural);

    let report = run_epistemic_cron_pass(
        &mut store,
        EpistemicCronInput {
            content_node_ids: vec!["doc".into()],
            ..EpistemicCronInput::default()
        },
        &DroppingEnricher,
    )
    .unwrap();
    assert!(report.attempted);
    assert!(report.no_op, "drop is a clean no-op");
    assert!(!report.grpc_ok);

    // Shadow node still present, still purely structural (no partial learned write).
    assert_eq!(shadow_count(&store), before, "no shadow nodes deleted");
    let after = read_epistemic_shadow(&store, "doc").unwrap();
    assert_eq!(
        after.source_kind,
        EpistemicSourceKind::Structural,
        "no learned fields written"
    );
    assert!(after.completion_confidence.is_none());
}

#[test]
fn epistemicrag_cron_is_idempotent_by_engine_version() {
    // EPISTEMICRAG #6: re-running the cron with the same engine_version creates
    // no duplicate shadow nodes or edges.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("doc", "a claim")).unwrap();
    run_structural(&mut store, &["doc"], vec![]);
    let enricher = StaticEnricher(EpistemicAnnotations {
        annotations: vec![learned_annotation("doc", "ghost")],
        support_relations: Vec::new(),
        attack_relations: Vec::new(),
    });
    let input = || EpistemicCronInput {
        content_node_ids: vec!["doc".into()],
        ..EpistemicCronInput::default()
    };
    run_epistemic_cron_pass(&mut store, input(), &enricher).unwrap();
    let first = shadow_count(&store);
    run_epistemic_cron_pass(&mut store, input(), &enricher).unwrap();
    let second = shadow_count(&store);
    assert_eq!(first, second, "no duplicate shadow nodes on cron re-run");
}

#[test]
fn epistemicrag_density_floor_gates_the_learned_pass() {
    // Gate before first run: completion runs only above an edge-density floor.
    let mut store = InMemoryGraphStore::new();
    // Three nodes, no edges -> density 0, below any positive floor.
    for id in ["a", "b", "c"] {
        store.upsert_node(claim(id, "x")).unwrap();
    }
    let enricher = StaticEnricher(EpistemicAnnotations::default());
    let report = run_epistemic_cron_pass(
        &mut store,
        EpistemicCronInput {
            content_node_ids: vec!["a".into(), "b".into(), "c".into()],
            density_floor: 5.0,
            ..EpistemicCronInput::default()
        },
        &enricher,
    )
    .unwrap();
    assert!(
        report.no_op,
        "sparse subgraph gated out of the learned pass"
    );
    assert!(
        report.skipped_reason.contains("edge_density_below_floor"),
        "skipped for density, got {}",
        report.skipped_reason
    );
}

#[test]
fn epistemicrag_ppr_over_shadow_layer_returns_ranking() {
    // EPISTEMICRAG #3 / recall deliverable 6: PPR runs over shadow-to-shadow
    // edges and returns a ranking.
    let mut store = InMemoryGraphStore::new();
    // A chain of equivalent claims: infer_pair_relation emits SUPPORTS for
    // normalized-equal text, so the shadow layer gains SUPPORTS edges to walk.
    store.upsert_node(claim("a", "the cache is safe")).unwrap();
    store.upsert_node(claim("b", "the cache is safe")).unwrap();
    store.upsert_node(claim("c", "the cache is safe")).unwrap();
    run_structural(&mut store, &["a", "b", "c"], vec![("a", "b"), ("b", "c")]);

    let mut seeds = HashMap::new();
    seeds.insert("a".to_string(), 1.0);
    let ranking = epistemic_shadow_ppr(&store, &seeds, 10, 0.85, 1e-6, 1000)
        .expect("shadow PPR should hash its cache key");
    assert!(!ranking.is_empty(), "shadow-layer PPR returns a ranking");
}

// --------------------------------------------------------------------------- //
// Strand A phase 3 / cut 9: e-graph SameEClass dedup
// --------------------------------------------------------------------------- //

/// Count edges of a given type in the store (no public edge-count accessor).
fn edge_type_count(store: &InMemoryGraphStore, edge_type: &str) -> usize {
    store
        .query_nodes(NodeQuery::default().with_limit(100_000))
        .into_iter()
        .flat_map(|node| {
            store
                .neighbors(
                    rustyred_thg_core::NeighborQuery::out(&node.id).with_edge_type(edge_type),
                )
                .into_iter()
                .map(|hit| hit.edge_id)
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

#[test]
fn dedup_collapses_provably_equivalent_states_onto_one_eclass() {
    // CUT 9: provably-equivalent shadow states collapse many-to-one. "enabled"
    // and "not not enabled" are the same proposition by double-negation; the
    // e-graph proves it where a string dedup cannot.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("a", "telemetry is on")).unwrap();
    store.upsert_node(claim("b", "telemetry is on")).unwrap();
    store
        .upsert_node(claim("c", "telemetry is not not on"))
        .unwrap();
    store.upsert_node(claim("solo", "logging is off")).unwrap();
    run_structural(&mut store, &["a", "b", "c", "solo"], vec![]);

    let report =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");

    assert_eq!(report.shadows_examined, 4);
    assert_eq!(report.classes.len(), 1, "a/b/c form a single class");
    let class = &report.classes[0];
    assert_eq!(class.member_content_ids, vec!["a", "b", "c"]);
    // Two non-representatives collapse onto one representative (many-to-one).
    assert_eq!(report.members_collapsed, 2);
    assert_eq!(report.same_eclass_edges_written, 2);
    assert_eq!(report.singleton_count, 1, "the 'solo' claim stands alone");
    // The representative carries no outgoing SameEClass edge; the other two do.
    let rep_shadow = &class.representative_shadow_id;
    let mut pointed_at_rep = 0;
    for content in ["a", "b", "c"] {
        let shadow = epistemic_shadow_node_id(content, DEFAULT_EPISTEMIC_ENGINE_VERSION)
            .expect("shadow node ID should be content-addressable");
        match read_same_eclass(&store, &shadow) {
            Some(same) => {
                assert_eq!(&same.representative_shadow_id, rep_shadow);
                pointed_at_rep += 1;
            }
            None => assert_eq!(&shadow, rep_shadow, "only the representative has no edge"),
        }
    }
    assert_eq!(pointed_at_rep, 2);
}

#[test]
fn dedup_never_mutates_the_content_graph() {
    // CUT 9 invariant: the dedup writes ONLY SameEClass edges. Content nodes,
    // content edges, and the shadow node set are all untouched.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("a", "the disk is full")).unwrap();
    store.upsert_node(claim("b", "the disk is full")).unwrap();
    store
        .upsert_edge(EdgeRecord::new("rel", "a", "CITES", "b", json!({})))
        .unwrap();
    run_structural(&mut store, &["a", "b"], vec![]);

    let nodes_before = store.stats().nodes_total;
    let shadows_before = shadow_count(&store);
    let cites_before = edge_type_count(&store, "CITES");
    // Content node "a" snapshot.
    let a_before = store.get_node("a").cloned().expect("a exists");

    let report =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");
    assert_eq!(report.same_eclass_edges_written, 1);

    assert_eq!(store.stats().nodes_total, nodes_before, "no nodes added");
    assert_eq!(
        shadow_count(&store),
        shadows_before,
        "no shadows added/removed"
    );
    assert_eq!(
        edge_type_count(&store, "CITES"),
        cites_before,
        "content edges intact"
    );
    assert_eq!(
        store.get_node("a").cloned().expect("a still exists"),
        a_before,
        "content node payload byte-identical"
    );
    // The only new edges are SameEClass.
    assert_eq!(edge_type_count(&store, SAME_ECLASS), 1);
}

#[test]
fn dedup_same_eclass_edges_feed_shadow_ppr() {
    // CUT 9 payoff: SameEClass edges are part of the shadow-PPR adjacency, so a
    // seed at one class member reaches the representative through the new edge.
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(claim("a", "the model converged"))
        .unwrap();
    store
        .upsert_node(claim("b", "the model converged"))
        .unwrap();
    run_structural(&mut store, &["a", "b"], vec![]);
    // No SUPPORTS/UNDERCUTS were inferred (no candidate pairs), so the only
    // shadow-to-shadow edge after dedup is SameEClass.
    let report =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");
    let rep = report.classes[0].representative_shadow_id.clone();
    let member = report.classes[0]
        .member_shadow_ids
        .iter()
        .find(|id| **id != rep)
        .cloned()
        .expect("a non-representative member");

    // Seed the member's content node; PPR should rank the representative shadow.
    let member_content = if epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION)
        .expect("shadow node ID should be content-addressable")
        == member
    {
        "a"
    } else {
        "b"
    };
    let mut seeds = HashMap::new();
    seeds.insert(member_content.to_string(), 1.0);
    let ranking = epistemic_shadow_ppr(&store, &seeds, 10, 0.85, 1e-6, 5000)
        .expect("shadow PPR should hash its cache key");
    assert!(
        ranking.iter().any(|(id, _)| id == &rep),
        "SameEClass edge carries PPR mass to the representative"
    );
}

#[test]
fn dedup_surfaces_through_recall_readout_and_is_idempotent() {
    // CUT 9 read path + idempotency: read_epistemic_shadow surfaces same_eclass
    // for a collapsed member and None for the representative; a second run adds
    // no edges.
    let mut store = InMemoryGraphStore::new();
    store.upsert_node(claim("a", "the queue is empty")).unwrap();
    store.upsert_node(claim("b", "the queue is empty")).unwrap();
    run_structural(&mut store, &["a", "b"], vec![]);

    let first =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");
    let rep = first.classes[0].representative_content_id.clone();
    let member = if rep == "a" { "b" } else { "a" };

    let member_readout = read_epistemic_shadow(&store, member).expect("member shadow");
    let same = member_readout
        .same_eclass
        .expect("member carries same_eclass");
    assert_eq!(
        same.representative_shadow_id,
        epistemic_shadow_node_id(&rep, DEFAULT_EPISTEMIC_ENGINE_VERSION)
            .expect("representative shadow ID should be content-addressable")
    );
    assert_eq!(same.canonical_form, "the queue is empty");

    let rep_readout = read_epistemic_shadow(&store, &rep).expect("rep shadow");
    assert!(
        rep_readout.same_eclass.is_none(),
        "the representative is a class head, not a member"
    );

    // Idempotent: re-running yields the same class and writes no new edges.
    let edges_before = edge_type_count(&store, SAME_ECLASS);
    let second =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");
    assert_eq!(second.classes[0].class_id, first.classes[0].class_id);
    assert_eq!(edge_type_count(&store, SAME_ECLASS), edges_before);
}

#[test]
fn dedup_claim_and_standing_default_keeps_in_and_out_apart() {
    // CUT 9 congruence predicate: the default (ClaimAndStanding) treats standing
    // as part of the state, so an attacked (`out`) claim is not merged with an
    // unattacked (`in`) copy of the same proposition.
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(claim("win", "rollout is healthy"))
        .unwrap();
    store
        .upsert_node(claim("lose", "rollout is healthy"))
        .unwrap();
    store
        .upsert_node(claim("atk", "rollout is broken"))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "a",
            "atk",
            "CONTRADICTS",
            "lose",
            json!({}),
        ))
        .unwrap();
    run_structural(&mut store, &["win", "lose", "atk"], vec![]);

    // Sanity: lose is grounded out, win is in.
    assert_eq!(
        read_epistemic_shadow(&store, "lose")
            .unwrap()
            .grounded_extension_status,
        GroundedExtensionStatus::Out
    );
    assert_eq!(
        read_epistemic_shadow(&store, "win")
            .unwrap()
            .grounded_extension_status,
        GroundedExtensionStatus::In
    );

    let default_run =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");
    assert!(
        !default_run.classes.iter().any(|c| {
            c.member_content_ids.contains(&"win".to_string())
                && c.member_content_ids.contains(&"lose".to_string())
        }),
        "ClaimAndStanding must not merge an `in` claim with an `out` claim"
    );

    // ClaimOnly ignores standing and merges them.
    let mut store2 = InMemoryGraphStore::new();
    store2
        .upsert_node(claim("win", "rollout is healthy"))
        .unwrap();
    store2
        .upsert_node(claim("lose", "rollout is healthy"))
        .unwrap();
    store2
        .upsert_node(claim("atk", "rollout is broken"))
        .unwrap();
    store2
        .upsert_edge(EdgeRecord::new(
            "a",
            "atk",
            "CONTRADICTS",
            "lose",
            json!({}),
        ))
        .unwrap();
    run_structural(&mut store2, &["win", "lose", "atk"], vec![]);
    let claim_only = epistemic_egraph_dedup(
        &mut store2,
        &[],
        EpistemicDedupConfig {
            congruence: EpistemicCongruence::ClaimOnly,
            ..EpistemicDedupConfig::default()
        },
    )
    .expect("dedup");
    assert!(
        claim_only.classes.iter().any(|c| {
            c.member_content_ids.contains(&"win".to_string())
                && c.member_content_ids.contains(&"lose".to_string())
        }),
        "ClaimOnly merges identical claims across standing"
    );
}

/// Plant a content node plus its EpistemicShadow directly (bypassing the
/// O(n^2) structural pass) so scale tests stay fast.
fn plant_shadow(store: &mut InMemoryGraphStore, content_id: &str, text: &str, status: &str) {
    store.upsert_node(claim(content_id, text)).unwrap();
    let shadow_id = epistemic_shadow_node_id(content_id, DEFAULT_EPISTEMIC_ENGINE_VERSION)
        .expect("shadow node ID should be content-addressable");
    store
        .upsert_node(NodeRecord::new(
            &shadow_id,
            [EPISTEMIC_SHADOW_LABEL],
            json!({ "content_node_id": content_id, "grounded_extension_status": status }),
        ))
        .unwrap();
}

#[test]
fn dedup_congruence_runs_at_scale_beyond_default_node_limit() {
    // Regression for the egg node_limit cliff: a batch whose seed e-graph
    // exceeds egg's default 10_000-node limit must STILL apply the
    // double-negation rule and collapse every planted pair. On the unfixed code
    // (Runner::default()) the rule fires on zero terms and this asserts to ~0
    // collapses; with explicit limits it saturates and collapses all pairs.
    let mut store = InMemoryGraphStore::new();
    let pairs = 4_000usize; // ~5 e-graph nodes/pair => ~20k nodes, well past 10k.
    for i in 0..pairs {
        let prop = format!("proposition number {i} holds");
        plant_shadow(&mut store, &format!("p{i}"), &prop, "in");
        plant_shadow(
            &mut store,
            &format!("d{i}"),
            &format!("not not {prop}"),
            "in",
        );
    }

    let report =
        epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).expect("dedup");

    assert_eq!(report.shadows_examined, pairs * 2);
    assert_eq!(
        report.classes.len(),
        pairs,
        "each plain/double-negation pair must collapse despite the large batch"
    );
    assert_eq!(report.members_collapsed, pairs);
    assert_eq!(report.same_eclass_edges_written, pairs);
}
