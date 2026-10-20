//! End-to-end proof of the graph-level hook dispatch path against a real
//! `RedCoreGraphStore`. Covers the spec's observable acceptance criteria for
//! the primitive itself (coalescing, failure isolation, loop convergence, and
//! the off-critical-path non-blocking emit). Handler-specific criteria
//! (centrality population, crawl reprioritization) live with their plugins.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rustyred_thg_core::{
    coalesce_by_subgraph, facts_from_subgraph, materialize_closure, run_saturation,
    saturation_handler, saturation_hook_registration, EdgeRecord, GraphMutation,
    GraphMutationBatch, HookContext, HookDispatcher, HookDispatcherConfig, HookError, HookHandler,
    HookOutcome, HookRegistration, MutationEvent, MutationKind, MutationMatcher, NodeQuery,
    NodeRecord, RedCoreDurability, RedCoreGraphStore, RedCoreOptions, SaturationConfig,
    SaturationProgram, SaturationTemplateId, SATURATION_DERIVED_STATEMENT_LABEL,
};
use serde_json::json;

// coalesce_key must be a plain fn pointer (no captures).
fn coalesce_all(_event: &MutationEvent) -> Option<String> {
    Some("all".to_string())
}

#[test]
fn entity_resolution_real_store_events_coalesce_upsert_and_retraction_per_component() {
    let calls = Arc::new(Mutex::new(Vec::<Vec<(MutationKind, String)>>::new()));
    let calls_h = Arc::clone(&calls);
    let handler: HookHandler = Arc::new(move |ctx, events| {
        calls_h.lock().unwrap().push(
            events
                .iter()
                .map(|event| (event.kind, event.id.clone()))
                .collect(),
        );
        saturation_handler(ctx, events)
    });
    let registration = HookRegistration::new(
        "entity-saturation-real-path",
        MutationMatcher::any()
            .with_kinds([MutationKind::EdgeUpserted, MutationKind::EdgeDeleted])
            .with_labels(["match_edge"]),
        coalesce_by_subgraph,
        handler,
    );
    let store = shared_store();
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![registration], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());
    for id in ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "x", "y"] {
        store
            .lock()
            .unwrap()
            .upsert_node(NodeRecord::new(id, ["Record"], json!({})))
            .unwrap();
    }
    let mut edges = (0..7)
        .map(|index| {
            (
                format!("e{index}"),
                format!("n{index}"),
                format!("n{}", index + 1),
                "n0".to_string(),
            )
        })
        .collect::<Vec<_>>();
    edges.push(("xy".into(), "x".into(), "y".into(), "x".into()));
    for (id, left, right, component) in edges {
        store.lock().unwrap().upsert_edge(EdgeRecord::new(&id, &left, "match_edge", &right, json!({
            "left_record_id": left, "right_record_id": right, "weight": 1.0,
            "comparison_vector": [2, 1], (format!("component_key:entity-component:{component}|{left}|{right}")): true
        }))).unwrap();
    }
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert!(store.lock().unwrap().delete_edge("e3").unwrap());
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    let call_log = calls.lock().unwrap();
    assert_eq!(call_log.len(), 3);
    assert_eq!(
        call_log[2],
        vec![(MutationKind::EdgeDeleted, "e3".to_string())]
    );
    drop(call_log);
    let statements = store
        .lock()
        .unwrap()
        .query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(10_000))
        .unwrap();
    let has = |subject: &str, target: &str| {
        statements.iter().any(|node| {
            node.properties["relation"] == json!("same_entity")
                && node.properties["subject_id"] == json!(subject)
                && node.properties["attributes"]["target_id"] == json!(target)
        })
    };
    assert!(has("n0", "n3"));
    assert!(has("n4", "n7"));
    assert!(!has("n0", "n7"));
    assert!(has("x", "y"));
    let right_version = statements
        .iter()
        .find(|node| {
            node.properties["subject_id"] == "n4"
                && node.properties["attributes"]["target_id"] == "n7"
        })
        .unwrap()
        .version;
    let mut update = store.lock().unwrap().get_edge("e0").unwrap().unwrap();
    update.properties["weight"] = json!(2.0);
    store.lock().unwrap().upsert_edge(update).unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    assert_eq!(calls.lock().unwrap().len(), 4);
    let right_after = store
        .lock()
        .unwrap()
        .query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(10_000))
        .unwrap()
        .into_iter()
        .find(|node| {
            node.properties["subject_id"] == "n4"
                && node.properties["attributes"]["target_id"] == "n7"
        })
        .unwrap();
    assert_eq!(
        right_after.version, right_version,
        "other fragment was not recomputed"
    );
    store.lock().unwrap().upsert_edge(EdgeRecord::new("e3", "n3", "match_edge", "n4", json!({
        "left_record_id":"n3", "right_record_id":"n4", "weight":1.0, "comparison_vector":[2,1],
        "component_key:entity-component:n0|n3|n4": true
    }))).unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    assert_eq!(
        calls.lock().unwrap().len(),
        5,
        "bridge insertion dispatched"
    );
    let joined = store
        .lock()
        .unwrap()
        .query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(10_000))
        .unwrap();
    assert!(joined
        .iter()
        .any(|node| node.properties["subject_id"] == "n0"
            && node.properties["attributes"]["target_id"] == "n7"));
}

fn coalesce_per_id(_event: &MutationEvent) -> Option<String> {
    None
}

fn fast_config() -> HookDispatcherConfig {
    HookDispatcherConfig {
        debounce: Duration::from_millis(80),
        idle_poll: Duration::from_millis(10),
        ..Default::default()
    }
}

fn shared_store() -> Arc<Mutex<RedCoreGraphStore>> {
    Arc::new(Mutex::new(RedCoreGraphStore::memory()))
}

#[test]
fn mixed_node_delete_and_edge_replacement_emit_immutable_prior_and_final_current_contexts() {
    let store = shared_store();
    {
        let mut store = store.lock().unwrap();
        store.set_hook_tenant("Travis-Gilbert");
        for node in [
            NodeRecord::new(
                "old:a",
                ["CodeSymbol", "OldFrom"],
                json!({ "repo_id": "repo:old", "generation": 1 }),
            ),
            NodeRecord::new(
                "old:b",
                ["CodeSymbol", "OldTo"],
                json!({ "repo_id": "repo:old", "generation": 1 }),
            ),
            NodeRecord::new(
                "new:a",
                ["CodeSymbol", "NewFrom"],
                json!({ "repo_id": "repo:new", "generation": 2 }),
            ),
            NodeRecord::new(
                "new:b",
                ["CodeSymbol", "NewTo"],
                json!({ "repo_id": "repo:new", "generation": 2 }),
            ),
        ] {
            store.upsert_node(node).unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new(
                "edge:move",
                "old:a",
                "SUPPORTS",
                "old:b",
                json!({ "weight": 1 }),
            ))
            .unwrap();
    }

    let seen = Arc::new(Mutex::new(Vec::<MutationEvent>::new()));
    let seen_h = Arc::clone(&seen);
    let handler: HookHandler = Arc::new(move |_ctx, events| {
        seen_h.lock().unwrap().extend(events.iter().cloned());
        Ok(HookOutcome::Done)
    });
    let registration = HookRegistration::new(
        "capture-edge-replacement-context",
        MutationMatcher::any().with_kinds([MutationKind::EdgeUpserted]),
        coalesce_per_id,
        handler,
    );
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![registration], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    let transaction = store
        .lock()
        .unwrap()
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeDelete("old:a".to_string()),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:move",
                "new:a",
                "SUPPORTS",
                "new:b",
                json!({ "weight": 2 }),
            )),
        ]))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let commit = seen[0]
        .commit_context
        .as_ref()
        .expect("mixed transaction carries a commit envelope");
    assert_eq!(commit.txn_id, transaction.txn_id);
    assert_eq!(commit.graph_version, transaction.graph_version);
    assert_eq!(commit.sequence, 1);
    let transition = seen[0]
        .edge_transition
        .as_ref()
        .expect("edge replacement carries both sides");
    let prior = transition.prior.as_ref().expect("prior edge context");
    assert_eq!(
        (prior.from_id.as_str(), prior.to_id.as_str()),
        ("old:a", "old:b")
    );
    assert_eq!(prior.from_labels, ["CodeSymbol", "OldFrom"]);
    assert_eq!(prior.to_labels, ["CodeSymbol", "OldTo"]);
    assert_eq!(prior.edge_type, "SUPPORTS");
    assert_eq!(
        prior.from_scope_context.repo_id.as_deref(),
        Some("repo:old")
    );
    assert_eq!(prior.from_scope_context.generation, Some(1));
    assert_eq!(prior.to_scope_context.repo_id.as_deref(), Some("repo:old"));
    assert_eq!(prior.to_scope_context.generation, Some(1));

    let current = transition.current.as_ref().expect("current edge context");
    assert_eq!(
        (current.from_id.as_str(), current.to_id.as_str()),
        ("new:a", "new:b")
    );
    assert_eq!(current.from_labels, ["CodeSymbol", "NewFrom"]);
    assert_eq!(current.to_labels, ["CodeSymbol", "NewTo"]);
    assert_eq!(current.edge_type, "SUPPORTS");
    assert_ne!(prior.property_hash, current.property_hash);
    assert_eq!(
        current.from_scope_context.repo_id.as_deref(),
        Some("repo:new")
    );
    assert_eq!(current.from_scope_context.generation, Some(2));
    assert_eq!(
        current.to_scope_context.repo_id.as_deref(),
        Some("repo:new")
    );
    assert_eq!(current.to_scope_context.generation, Some(2));
}

#[test]
fn incremental_endpoint_move_and_edge_upsert_emit_precommit_and_final_contexts() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let data_dir = std::env::temp_dir().join(format!(
        "rustyred-incremental-edge-context-{}-{suffix}",
        std::process::id()
    ));
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let store = Arc::new(Mutex::new(
        RedCoreGraphStore::open(&data_dir, options).unwrap(),
    ));
    {
        let mut store = store.lock().unwrap();
        store.set_hook_tenant("Travis-Gilbert");
        store
            .upsert_node(NodeRecord::new(
                "symbol:from",
                ["CodeSymbol", "OldFrom"],
                json!({ "repo_id": "repo:a", "generation": 1 }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "symbol:to",
                ["CodeSymbol", "StableTo"],
                json!({ "repo_id": "repo:a", "generation": 1 }),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "edge:scope-move",
                "symbol:from",
                "SUPPORTS",
                "symbol:to",
                json!({ "weight": 1 }),
            ))
            .unwrap();
    }

    let seen = Arc::new(Mutex::new(Vec::<MutationEvent>::new()));
    let seen_h = Arc::clone(&seen);
    let handler: HookHandler = Arc::new(move |_ctx, events| {
        seen_h.lock().unwrap().extend(events.iter().cloned());
        Ok(HookOutcome::Done)
    });
    let registration = HookRegistration::new(
        "capture-incremental-edge-context",
        MutationMatcher::any().with_kinds([MutationKind::EdgeUpserted]),
        coalesce_per_id,
        handler,
    );
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![registration], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    let transaction = store
        .lock()
        .unwrap()
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "symbol:from",
                ["CodeSymbol", "NewFrom"],
                json!({ "repo_id": "repo:b", "generation": 2 }),
            )),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:scope-move",
                "symbol:from",
                "SUPPORTS",
                "symbol:to",
                json!({ "weight": 2 }),
            )),
        ]))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let commit = seen[0]
        .commit_context
        .as_ref()
        .expect("incremental transaction carries a commit envelope");
    assert_eq!(commit.txn_id, transaction.txn_id);
    assert_eq!(commit.graph_version, transaction.graph_version);
    assert_eq!(commit.sequence, 1);
    let transition = seen[0]
        .edge_transition
        .as_ref()
        .expect("edge upsert carries both sides");
    let prior = transition.prior.as_ref().expect("prior edge context");
    assert_eq!(
        (prior.from_id.as_str(), prior.to_id.as_str()),
        ("symbol:from", "symbol:to")
    );
    assert_eq!(prior.from_labels, ["CodeSymbol", "OldFrom"]);
    assert_eq!(prior.to_labels, ["CodeSymbol", "StableTo"]);
    assert_eq!(prior.edge_type, "SUPPORTS");
    assert_eq!(prior.from_scope_context.repo_id.as_deref(), Some("repo:a"));
    assert_eq!(prior.from_scope_context.generation, Some(1));

    let current = transition.current.as_ref().expect("current edge context");
    assert_eq!(
        (current.from_id.as_str(), current.to_id.as_str()),
        ("symbol:from", "symbol:to")
    );
    assert_eq!(current.from_labels, ["CodeSymbol", "NewFrom"]);
    assert_eq!(current.to_labels, ["CodeSymbol", "StableTo"]);
    assert_eq!(current.edge_type, "SUPPORTS");
    assert_ne!(prior.property_hash, current.property_hash);
    assert_eq!(
        current.from_scope_context.repo_id.as_deref(),
        Some("repo:b")
    );
    assert_eq!(current.from_scope_context.generation, Some(2));
    assert_eq!(current.to_scope_context.repo_id.as_deref(), Some("repo:a"));
    assert_eq!(current.to_scope_context.generation, Some(1));
    drop(seen);
    drop(dispatcher);
    drop(store);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn repeated_edge_upserts_emit_each_incremental_transition_in_order() {
    assert_repeated_edge_transitions(false);
}

#[test]
fn repeated_edge_upserts_emit_each_staged_transition_in_order() {
    assert_repeated_edge_transitions(true);
}

fn assert_repeated_edge_transitions(force_staged_transaction: bool) {
    let store = shared_store();
    {
        let mut graph = store.lock().unwrap();
        graph.set_hook_tenant("Travis-Gilbert");
        graph
            .upsert_node(NodeRecord::new(
                "symbol:from",
                ["CodeSymbol"],
                json!({ "repo_id": "repo:a", "generation": 1 }),
            ))
            .unwrap();
        graph
            .upsert_node(NodeRecord::new(
                "symbol:to",
                ["CodeSymbol"],
                json!({ "repo_id": "repo:a", "generation": 1 }),
            ))
            .unwrap();
    }

    let seen = Arc::new(Mutex::new(Vec::<MutationEvent>::new()));
    let seen_h = Arc::clone(&seen);
    let handler: HookHandler = Arc::new(move |_ctx, events| {
        seen_h.lock().unwrap().extend(events.iter().cloned());
        Ok(HookOutcome::Done)
    });
    let registration = HookRegistration::new(
        "capture-repeated-edge-upserts",
        MutationMatcher::any().with_kinds([MutationKind::EdgeUpserted]),
        coalesce_per_id,
        handler,
    );
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![registration], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    let mut mutations = Vec::new();
    if force_staged_transaction {
        mutations.push(GraphMutation::NodeDelete("node:missing".to_string()));
    }
    mutations.extend([
        GraphMutation::EdgeUpsert(EdgeRecord::new(
            "edge:repeated",
            "symbol:from",
            "SUPPORTS",
            "symbol:to",
            json!({ "weight": 1 }),
        )),
        GraphMutation::EdgeUpsert(EdgeRecord::new(
            "edge:repeated",
            "symbol:from",
            "SUPPORTS",
            "symbol:to",
            json!({ "weight": 2 }),
        )),
    ]);
    store
        .lock()
        .unwrap()
        .commit_batch(GraphMutationBatch::new(mutations))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));

    let mut events = seen.lock().unwrap().clone();
    events.sort_by_key(|event| {
        event
            .commit_context
            .as_ref()
            .expect("authoritative event")
            .sequence
    });
    assert_eq!(events.len(), 2);
    let first = events[0]
        .edge_transition
        .as_ref()
        .expect("first transition");
    let second = events[1]
        .edge_transition
        .as_ref()
        .expect("second transition");
    assert!(first.prior.is_none());
    assert_eq!(events[1].changed_props, ["weight"]);
    assert_eq!(
        second.prior.as_ref().expect("staged prior").property_hash,
        first.current.as_ref().expect("first current").property_hash
    );
    assert_ne!(
        second.prior.as_ref().expect("staged prior").property_hash,
        second
            .current
            .as_ref()
            .expect("second current")
            .property_hash
    );
}

#[test]
fn entity_retraction_survives_redcore_reopen_without_process_scope_cache() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let data_dir = std::env::temp_dir().join(format!(
        "rustyred-retraction-reopen-{}-{suffix}",
        std::process::id()
    ));
    let options = RedCoreOptions {
        durability: RedCoreDurability::SnapshotOnly,
        snapshot_interval_writes: 1,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for id in ["a", "b", "c", "x", "y"] {
            store
                .upsert_node(NodeRecord::new(id, ["Record"], json!({})))
                .unwrap();
        }
        for (id, from, to) in [("ab", "a", "b"), ("bc", "b", "c"), ("xy", "x", "y")] {
            store
                .upsert_edge(EdgeRecord::new(
                    id,
                    from,
                    "match_edge",
                    to,
                    json!({ "weight": 1.0 }),
                ))
                .unwrap();
        }

        // Seed durable derived state directly. The process-local hook cache is
        // deliberately never populated before the store is closed.
        for ids in [
            vec!["a".into(), "b".into(), "c".into()],
            vec!["x".into(), "y".into()],
        ] {
            let closure = run_saturation(
                facts_from_subgraph(&store, &ids)
                    .expect("fixture subgraph should produce saturation facts"),
                &SaturationProgram::from_template(SaturationTemplateId::EntityResolution),
            )
            .expect("fixture saturation should produce a closure");
            materialize_closure(&mut store, closure, SaturationConfig::default()).unwrap();
        }
    }

    {
        let store = Arc::new(Mutex::new(
            RedCoreGraphStore::open(&data_dir, options.clone()).unwrap(),
        ));
        let unrelated_version = derived_same_entity(&store.lock().unwrap(), "x", "y")
            .expect("unrelated component was materialized")
            .version;
        let dispatcher = HookDispatcher::start(
            Arc::clone(&store),
            vec![saturation_hook_registration()],
            fast_config(),
        );
        store
            .lock()
            .unwrap()
            .attach_hook_emitter(dispatcher.emitter());

        assert!(store.lock().unwrap().delete_edge("bc").unwrap());
        assert!(dispatcher.quiesce(Duration::from_secs(10)), "queue drained");

        let store = store.lock().unwrap();
        assert!(derived_same_entity(&store, "a", "b").is_some());
        assert!(derived_same_entity(&store, "a", "c").is_none());
        assert!(derived_same_entity(&store, "b", "c").is_none());
        assert_eq!(
            derived_same_entity(&store, "x", "y").unwrap().version,
            unrelated_version,
            "unrelated component must not be recomputed"
        );
    }

    // A second open proves the hook's tombstones and fragment rekeying were
    // themselves durable, not merely visible in the worker's memory.
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(derived_same_entity(&reopened, "a", "b").is_some());
    assert!(derived_same_entity(&reopened, "a", "c").is_none());
    assert!(derived_same_entity(&reopened, "x", "y").is_some());
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

fn derived_same_entity(
    store: &RedCoreGraphStore,
    subject: &str,
    target: &str,
) -> Option<NodeRecord> {
    store
        .query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(10_000))
        .unwrap()
        .into_iter()
        .find(|node| {
            node.properties["relation"] == json!("same_entity")
                && node.properties["subject_id"] == json!(subject)
                && node.properties["attributes"]["target_id"] == json!(target)
        })
}

/// Acceptance 1: an ingest of N CodeSymbol upserts fires the hook exactly once
/// per coalesced batch, sees all N events, and its derived write lands.
#[test]
fn coalesces_ingest_storm_into_one_call_and_writes_back() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(AtomicUsize::new(0));
    let calls_h = Arc::clone(&calls);
    let seen_h = Arc::clone(&seen);

    let handler: HookHandler = Arc::new(move |ctx: &mut HookContext, events: &[MutationEvent]| {
        calls_h.fetch_add(1, Ordering::SeqCst);
        seen_h.fetch_add(events.len(), Ordering::SeqCst);
        // Derived write-back: a summary node carrying the coalesced count.
        ctx.store.upsert_node(NodeRecord::new(
            "repo-summary",
            ["RepoSummary"],
            json!({ "symbol_count": events.len() }),
        ))?;
        Ok(HookOutcome::Wrote { mutations: 1 })
    });

    let reg = HookRegistration::new(
        "counter",
        MutationMatcher::any()
            .with_kinds([MutationKind::NodeUpserted])
            .with_labels(["CodeSymbol"]),
        coalesce_all,
        handler,
    );

    let store = shared_store();
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![reg], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    for i in 0..50 {
        store
            .lock()
            .unwrap()
            .upsert_node(NodeRecord::new(
                format!("code:symbol:{i}"),
                ["CodeSymbol"],
                json!({ "repo_id": "repo:x", "name": format!("f{i}") }),
            ))
            .unwrap();
    }

    assert!(dispatcher.quiesce(Duration::from_secs(5)), "queue drained");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "one coalesced handler call"
    );
    assert_eq!(seen.load(Ordering::SeqCst), 50, "handler saw all 50 events");

    let summary = store.lock().unwrap().get_node("repo-summary").unwrap();
    assert_eq!(
        summary.expect("summary written back").properties["symbol_count"],
        json!(50)
    );
}

/// Acceptance 3: a handler that returns Err is logged + skipped; the triggering
/// write still commits, the store stays usable, and the queue keeps draining
/// (a subsequent good hook still fires).
#[test]
fn handler_error_is_isolated_and_queue_keeps_draining() {
    let good_calls = Arc::new(AtomicUsize::new(0));
    let gc = Arc::clone(&good_calls);

    let boom: HookHandler = Arc::new(|_ctx: &mut HookContext, _events: &[MutationEvent]| {
        Err(HookError::new("intentional handler failure"))
    });
    let boom_reg = HookRegistration::new(
        "boom",
        MutationMatcher::any().with_labels(["Boom"]),
        coalesce_per_id,
        boom,
    );

    let good: HookHandler = Arc::new(move |ctx: &mut HookContext, events: &[MutationEvent]| {
        gc.fetch_add(1, Ordering::SeqCst);
        ctx.store.upsert_node(NodeRecord::new(
            "good-marker",
            ["Marker"],
            json!({ "n": events.len() }),
        ))?;
        Ok(HookOutcome::Done)
    });
    let good_reg = HookRegistration::new(
        "good",
        MutationMatcher::any().with_labels(["Good"]),
        coalesce_per_id,
        good,
    );

    let store = shared_store();
    let dispatcher =
        HookDispatcher::start(Arc::clone(&store), vec![boom_reg, good_reg], fast_config());
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("b1", ["Boom"], json!({})))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));

    // Triggering write committed despite the handler erroring.
    assert!(store.lock().unwrap().get_node("b1").unwrap().is_some());
    // Store remains usable.
    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("after", ["X"], json!({})))
        .unwrap();
    assert!(store.lock().unwrap().get_node("after").unwrap().is_some());

    // Queue still draining: the good hook fires on a later write.
    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("g1", ["Good"], json!({})))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    assert_eq!(good_calls.load(Ordering::SeqCst), 1);
    assert!(store
        .lock()
        .unwrap()
        .get_node("good-marker")
        .unwrap()
        .is_some());
}

/// Extends acceptance 3: a panicking handler must not poison the store mutex.
/// After the panic the store still locks and serves reads/writes.
#[test]
fn handler_panic_does_not_poison_the_store() {
    let after_calls = Arc::new(AtomicUsize::new(0));
    let ac = Arc::clone(&after_calls);

    let panicky: HookHandler = Arc::new(|_ctx: &mut HookContext, _events: &[MutationEvent]| {
        panic!("intentional handler panic");
    });
    let panic_reg = HookRegistration::new(
        "panic",
        MutationMatcher::any().with_labels(["Panic"]),
        coalesce_per_id,
        panicky,
    );
    let after: HookHandler = Arc::new(move |_ctx: &mut HookContext, _events: &[MutationEvent]| {
        ac.fetch_add(1, Ordering::SeqCst);
        Ok(HookOutcome::Done)
    });
    let after_reg = HookRegistration::new(
        "after",
        MutationMatcher::any().with_labels(["After"]),
        coalesce_per_id,
        after,
    );

    let store = shared_store();
    let dispatcher = HookDispatcher::start(
        Arc::clone(&store),
        vec![panic_reg, after_reg],
        fast_config(),
    );
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("p1", ["Panic"], json!({})))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));

    // Mutex not poisoned: lock still succeeds and the store is consistent.
    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("p2", ["X"], json!({})))
        .unwrap();
    assert!(store.lock().unwrap().get_node("p2").unwrap().is_some());

    // The worker survived the panic and keeps dispatching.
    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("a1", ["After"], json!({})))
        .unwrap();
    assert!(dispatcher.quiesce(Duration::from_secs(5)));
    assert_eq!(after_calls.load(Ordering::SeqCst), 1);
}

/// Acceptance 5: a handler that writes a property feeding its own matcher
/// converges (bounded by max_depth) instead of looping forever.
#[test]
fn self_feeding_hook_converges_at_max_depth() {
    let runs = Arc::new(AtomicUsize::new(0));
    let r = Arc::clone(&runs);

    // Always writes a fresh value, so every generation would re-trigger if not
    // for the depth guard.
    let loopy: HookHandler = Arc::new(move |ctx: &mut HookContext, events: &[MutationEvent]| {
        r.fetch_add(1, Ordering::SeqCst);
        for event in events {
            let tick = ctx
                .store
                .get_node(&event.id)
                .ok()
                .flatten()
                .and_then(|node| node.properties.get("tick").and_then(|v| v.as_u64()))
                .unwrap_or(0);
            ctx.store.upsert_node(NodeRecord::new(
                &event.id,
                ["Loopy"],
                json!({ "tick": tick + 1 }),
            ))?;
        }
        Ok(HookOutcome::Done)
    });
    let reg = HookRegistration::new(
        "loopy",
        MutationMatcher::any()
            .with_kinds([MutationKind::NodeUpserted])
            .with_labels(["Loopy"]),
        coalesce_per_id,
        loopy,
    );

    let store = shared_store();
    let config = HookDispatcherConfig {
        debounce: Duration::from_millis(20),
        idle_poll: Duration::from_millis(10),
        max_depth: 3,
        ..Default::default()
    };
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![reg], config);
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    store
        .lock()
        .unwrap()
        .upsert_node(NodeRecord::new("loop1", ["Loopy"], json!({ "tick": 0 })))
        .unwrap();

    assert!(dispatcher.quiesce(Duration::from_secs(5)), "converged");
    let n = runs.load(Ordering::SeqCst);
    // Foreground write is depth 0 -> handler gen1 (writes depth1) -> gen2
    // (depth2) -> gen3 (depth3) -> depth3 event dropped at the guard. Bounded.
    assert!(n >= 1 && n <= 3, "bounded by max_depth, got {n}");

    let tick = store
        .lock()
        .unwrap()
        .get_node("loop1")
        .unwrap()
        .unwrap()
        .properties["tick"]
        .as_u64()
        .unwrap();
    assert!(tick >= 1 && tick <= 3, "tick bounded, got {tick}");
}

/// Acceptance 4 (mechanic): the writer's emit is non-blocking and off the
/// critical path. A write returns immediately even when the matched handler is
/// slow, because emit is a try-send and the worker runs the handler later.
#[test]
fn write_does_not_block_on_a_slow_handler() {
    let done = Arc::new(AtomicBool::new(false));
    let d = Arc::clone(&done);

    let slow: HookHandler = Arc::new(move |_ctx: &mut HookContext, _events: &[MutationEvent]| {
        thread::sleep(Duration::from_millis(300));
        d.store(true, Ordering::SeqCst);
        Ok(HookOutcome::Done)
    });
    let reg = HookRegistration::new(
        "slow",
        MutationMatcher::any().with_labels(["Slow"]),
        coalesce_per_id,
        slow,
    );

    let store = shared_store();
    let config = HookDispatcherConfig {
        debounce: Duration::from_millis(5),
        idle_poll: Duration::from_millis(5),
        ..Default::default()
    };
    let dispatcher = HookDispatcher::start(Arc::clone(&store), vec![reg], config);
    store
        .lock()
        .unwrap()
        .attach_hook_emitter(dispatcher.emitter());

    {
        let mut guard = store.lock().unwrap();
        let t0 = Instant::now();
        guard
            .upsert_node(NodeRecord::new("s1", ["Slow"], json!({})))
            .unwrap();
        let elapsed = t0.elapsed();
        // The write returned without waiting on the 300ms handler. (The worker
        // also cannot have started: it needs this very lock to run a handler.)
        assert!(
            elapsed < Duration::from_millis(100),
            "write blocked on hook? took {elapsed:?}"
        );
        assert!(
            !done.load(Ordering::SeqCst),
            "handler ran on the writer path"
        );
    }

    assert!(dispatcher.quiesce(Duration::from_secs(2)));
    assert!(
        done.load(Ordering::SeqCst),
        "handler eventually ran off-path"
    );
}

/// A store with no emitter attached behaves exactly as before: writes succeed,
/// nothing is dispatched. Proves the seam is strictly opt-in.
#[test]
fn no_emitter_attached_is_a_noop() {
    let mut store = RedCoreGraphStore::memory();
    assert!(!store.has_hook_emitter());
    store
        .upsert_node(NodeRecord::new("n1", ["X"], json!({ "a": 1 })))
        .unwrap();
    assert!(store.get_node("n1").unwrap().is_some());
}
