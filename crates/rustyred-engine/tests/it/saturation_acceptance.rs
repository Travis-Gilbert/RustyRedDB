use rustyred_thg_core::coalesce_by_subgraph;
use rustyred_thg_core::{
    differential_check, epistemic_egraph_dedup, epistemic_shadow_node_id, facts_from_payload,
    facts_from_subgraph, materialize_closure, read_same_eclass, run_saturation, saturation_handler,
    validate_egglog_program, write_statement, EdgeRecord, EpistemicDedupConfig, FlatObject,
    HookContext, HookOutcome, InMemoryGraphStore, MutationEvent, MutationKind, NeighborQuery,
    NodeQuery, NodeRecord, RedCoreGraphStore, SaturationConfig, SaturationProgram,
    SaturationTemplateId, StatementQuery, StatementRecord, HAS_OBJECT, HAS_SUBJECT,
    SATURATION_DERIVED_STATEMENT_LABEL, SATURATION_SHARED_RULE_IDS, STATEMENT_LABEL,
};
use serde_json::{json, Value};

fn claim(id: &str, text: &str) -> NodeRecord {
    NodeRecord::new(
        id,
        ["Claim"],
        json!({ "claim_text": text, "status": "active" }),
    )
}

fn object(id: &str) -> NodeRecord {
    NodeRecord::new(id, ["Object"], json!({}))
}

fn dep_edge(id: &str, from: &str, to: &str, confidence: f64) -> EdgeRecord {
    EdgeRecord::new(
        id,
        from,
        "CLAIM_DEPENDENCY",
        to,
        json!({ "strength": confidence }),
    )
    .with_confidence(confidence)
}

fn dep_fact(id: &str, from: &str, to: &str, confidence: f64) -> Value {
    json!({
        "fact_id": id,
        "relation": "claim_dependency",
        "entity_id": id,
        "attributes": {
            "claim_id": from,
            "depends_on_object_id": to,
            "justification_type": "dependency",
            "strength": confidence
        },
        "source_ref": "test"
    })
}

fn support_stmt<'a>(
    closure: &'a rustyred_thg_core::SaturationClosure,
    subject: &str,
    target: &str,
) -> &'a rustyred_thg_core::SaturationDerivedStatement {
    closure
        .derived_statements
        .iter()
        .find(|statement| {
            statement.rule_id == "support_reachability"
                && statement.subject_id == subject
                && statement.attributes["target_id"] == json!(target)
        })
        .expect("support_reachability statement")
}

fn derived_node_by_subject<'a>(
    store: &'a InMemoryGraphStore,
    subject: &str,
    target: &str,
) -> Option<NodeRecord> {
    store
        .query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(100_000))
        .into_iter()
        .find(|node| {
            node.properties["subject_id"] == json!(subject)
                && node.properties["attributes"]["target_id"] == json!(target)
        })
}

#[test]
fn real_egglog_program_parses_and_runs() {
    let program = SaturationProgram::default();
    let (tuple_count, output_count) =
        validate_egglog_program(&program).expect("egglog program parses and runs");

    assert_eq!(tuple_count, 0);
    assert_eq!(output_count, 2);
}

#[test]
fn named_saturation_template_runs_bounded_read_only_support_reachability() {
    let program = SaturationProgram::from_template(SaturationTemplateId::SupportReachability);
    assert_eq!(program.template_id.as_deref(), Some("support_reachability"));
    assert_eq!(program.writeback_policy, "read-only");
    assert_eq!(program.iteration_bound, 32);

    let facts = facts_from_payload(&json!({
        "facts": [
            dep_fact("dep:c1:c2", "c1", "c2", 0.9),
            dep_fact("dep:c2:o1", "c2", "o1", 0.8)
        ]
    }))
    .unwrap();
    let closure = run_saturation(facts, &program)
        .expect("support saturation should content-address derived statements");

    assert_eq!(closure.template_id.as_deref(), Some("support_reachability"));
    assert_eq!(closure.writeback_policy, "read-only");
    let support = support_stmt(&closure, "c1", "o1");
    assert_eq!(support.writeback_policy, "read-only");
    assert!(closure
        .derived_statements
        .iter()
        .all(|statement| statement.rule_id == "support_reachability"));
}

#[test]
fn equivalence_template_does_not_leak_other_rule_families() {
    let program = SaturationProgram::from_template(SaturationTemplateId::EquivalenceDedup);
    assert_eq!(program.template_id.as_deref(), Some("equivalence_dedup"));
    assert!(program.shared_rule_ids.is_empty());

    let facts = facts_from_payload(&json!({
        "facts": [
            dep_fact("dep:c1:c2", "c1", "c2", 0.9),
            dep_fact("dep:c2:o1", "c2", "o1", 0.8),
            {
                "fact_id": "path:long",
                "relation": "evidence_path",
                "entity_id": "c1",
                "attributes": { "path_length": 4 },
                "source_ref": "test"
            },
            {
                "fact_id": "edge:contradicts",
                "relation": "edge",
                "entity_id": "edge:contradicts",
                "attributes": {
                    "edge_type": "contradicts",
                    "acceptance_status": "contested",
                    "from_object_id": "c1",
                    "to_object_id": "o1"
                },
                "source_ref": "test"
            }
        ]
    }))
    .unwrap();
    let closure = run_saturation(facts, &program)
        .expect("equivalence saturation should content-address its empty closure");

    assert_eq!(closure.template_id.as_deref(), Some("equivalence_dedup"));
    assert!(
        closure.derived_statements.is_empty(),
        "equivalence template must not emit shared, support, or contradiction statements"
    );
}

#[test]
fn entity_resolution_template_round_trips_and_replays_rule_selection() {
    let serialized = serde_json::to_string(&SaturationTemplateId::EntityResolution).unwrap();
    assert_eq!(serialized, "\"entity_resolution\"");

    let replayed: SaturationTemplateId = serde_json::from_str(&serialized).unwrap();
    assert_eq!(replayed, SaturationTemplateId::EntityResolution);

    let program = SaturationProgram::from_template(replayed);
    assert_eq!(program.template_id.as_deref(), Some("entity_resolution"));
    assert_eq!(program.shared_rule_ids, vec!["same_entity"]);
}

#[test]
fn differential_check_preserves_byte_parity_subset() {
    let payload = json!({
        "facts": [
            dep_fact("dep:c1:o1", "c1", "o1", 0.7),
            {
                "fact_id": "path:long",
                "relation": "evidence_path",
                "entity_id": "c1",
                "attributes": { "path_length": 4 },
                "source_ref": "test"
            },
            {
                "fact_id": "edge:contradicts",
                "relation": "edge",
                "entity_id": "edge:contradicts",
                "attributes": {
                    "edge_type": "contradicts",
                    "from_object_id": "c1",
                    "to_object_id": "c2",
                    "acceptance_status": "accepted"
                },
                "source_ref": "test"
            }
        ],
        "rule_ids": SATURATION_SHARED_RULE_IDS
    });

    let report = differential_check(&payload).expect("differential check");

    assert_eq!(report.reference_engine, "python-reference-datalog");
    assert_eq!(report.engine, "egglog-saturation");
    assert_eq!(report.expected_count, 4);
    assert_eq!(report.actual_count, 4);
    assert!(report.missing_fact_ids.is_empty());
}

#[test]
fn saturation_reproduces_double_negation_same_eclass_edges() {
    let mut old_store = InMemoryGraphStore::new();
    let mut new_store = InMemoryGraphStore::new();
    for store in [&mut old_store, &mut new_store] {
        store.upsert_node(claim("a", "feature enabled")).unwrap();
        store
            .upsert_node(claim("b", "feature not not enabled"))
            .unwrap();
        for id in ["a", "b"] {
            let shadow_id = epistemic_shadow_node_id(id, "epistemic-v1")
                .expect("shadow node ID should be content-addressable");
            store
                .upsert_node(NodeRecord::new(
                    &shadow_id,
                    ["EpistemicShadow"],
                    json!({
                        "content_node_id": id,
                        "grounded_extension_status": "in",
                        "quarantine": true
                    }),
                ))
                .unwrap();
            store
                .upsert_edge(EdgeRecord::new(
                    format!("has:{id}"),
                    id,
                    "HasEpistemicShadow",
                    &shadow_id,
                    json!({}),
                ))
                .unwrap();
        }
    }

    let old = epistemic_egraph_dedup(&mut old_store, &[], EpistemicDedupConfig::default()).unwrap();
    let facts = facts_from_subgraph(&new_store, &[])
        .expect("fixture subgraph should produce saturation facts");
    let closure = run_saturation(facts, &SaturationProgram::default())
        .expect("double-negation saturation should content-address its closure");
    assert_eq!(closure.equivalence_classes.len(), 1);
    assert_eq!(
        closure.equivalence_classes[0].class_id,
        old.classes[0].class_id
    );

    materialize_closure(
        &mut new_store,
        closure,
        SaturationConfig {
            computed_at: 1,
            ..SaturationConfig::default()
        },
    )
    .unwrap();
    let member_shadow = epistemic_shadow_node_id("b", "epistemic-v1")
        .expect("member shadow ID should be content-addressable");
    let same = read_same_eclass(&new_store, &member_shadow).expect("SameEClass edge");
    assert_eq!(same.class_id, old.classes[0].class_id);
}

#[test]
fn confidence_fuses_by_max_and_keeps_all_dependency_contributors() {
    let payload = json!({
        "facts": [
            dep_fact("dep:ab", "a", "b", 0.5),
            dep_fact("dep:bd", "b", "d", 0.8),
            dep_fact("dep:ac", "a", "c", 0.9),
            dep_fact("dep:cd", "c", "d", 0.9)
        ]
    });
    let facts = facts_from_payload(&payload).unwrap();
    let closure = run_saturation(facts, &SaturationProgram::default())
        .expect("dependency saturation should content-address derived statements");

    assert!(closure.egglog_error.is_none());
    assert!(closure.egglog_tuple_count > 0);
    assert_eq!(closure.egglog_checked_pair_count, 0);
    let stmt = support_stmt(&closure, "a", "d");

    // Without path attenuation: max of (0.5*0.8=0.4) and (0.9*0.9=0.81).
    assert!((stmt.confidence - 0.81).abs() < 1e-9);
    // DepSet merge unions all dependency fact ids from competing paths.
    assert!(!stmt.contributors.is_empty());
    assert_eq!(
        stmt.dependency_fact_ids,
        vec!["dep:ab", "dep:ac", "dep:bd", "dep:cd"]
    );
}

#[test]
fn saturation_produces_transitive_support_chain_single_pass_receipt_misses() {
    let payload = json!({
        "facts": [
            dep_fact("dep:ab", "a", "b", 1.0),
            dep_fact("dep:bc", "b", "c", 1.0),
            dep_fact("dep:cd", "c", "d", 1.0)
        ]
    });
    let receipt = rustyred_thg_core::derive_datalog_receipt(&json!({
        "facts": payload["facts"].clone(),
        "rule_ids": ["dependent_claim"]
    }))
    .unwrap();
    assert!(
        receipt["derived_facts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|fact| fact["relation"] != json!("support_reachable")),
        "single-pass receipt has no transitive support relation"
    );

    let facts = facts_from_payload(&payload).unwrap();
    let closure = run_saturation(facts, &SaturationProgram::default())
        .expect("transitive saturation should content-address derived statements");
    let stmt = support_stmt(&closure, "a", "d");

    assert_eq!(stmt.attributes["path_length"], json!(3));
}

#[test]
fn entity_resolution_template_matches_reference_transitive_confidence() {
    let payload = json!({
        "facts": [
            {
                "fact_id": "pair:ab",
                "relation": "match_edge",
                "entity_id": "pair:ab",
                "attributes": {
                    "left_record_id": "a",
                    "right_record_id": "b",
                    "weight": 0.8,
                    "comparison_vector": [2, 1]
                },
                "source_ref": "resolve"
            },
            {
                "fact_id": "pair:bc",
                "relation": "match_edge",
                "entity_id": "pair:bc",
                "attributes": {
                    "left_record_id": "c",
                    "right_record_id": "b",
                    "weight": 0.5,
                    "comparison_vector": [1, 2]
                },
                "source_ref": "resolve"
            }
        ],
        "rule_ids": ["same_entity"]
    });

    let facts = facts_from_payload(&payload).unwrap();
    let closure = run_saturation(
        facts,
        &SaturationProgram::from_template(SaturationTemplateId::EntityResolution),
    )
    .expect("entity-resolution saturation should content-address derived statements");
    let transitive = closure
        .derived_statements
        .iter()
        .find(|statement| {
            statement.relation == "same_entity"
                && statement.subject_id == "a"
                && statement.attributes["target_id"] == json!("c")
        })
        .expect("transitive entity membership");

    let expected_confidence =
        (1.0 / (1.0 + (-0.8_f64).exp2())) * (1.0 / (1.0 + (-0.5_f64).exp2()));
    assert!((transitive.confidence - expected_confidence).abs() < 1e-12);
    assert_eq!(transitive.dependency_fact_ids, vec!["pair:ab", "pair:bc"]);
    let actual: std::collections::BTreeSet<_> = closure
        .derived_statements
        .iter()
        .filter(|s| s.relation == "same_entity")
        .map(|s| {
            (
                s.subject_id.as_str(),
                s.attributes["target_id"].as_str().unwrap_or(""),
            )
        })
        .collect();
    assert!(actual.contains(&("a", "b")));
    assert!(actual.contains(&("b", "c")));
    assert!(actual.contains(&("a", "c")));
}

#[test]
fn entity_resolution_preserves_log2_weights_and_closes_long_symmetric_paths() {
    let mut facts = Vec::new();
    for index in 0..35 {
        facts.push(json!({
            "fact_id": format!("pair:{index}:{}", index + 1),
            "relation": "match_edge",
            "entity_id": format!("pair:{index}:{}", index + 1),
            "attributes": {
                "left_record_id": format!("r:{index:02}"),
                "right_record_id": format!("r:{:02}", index + 1),
                "weight": if index == 0 { -1.0 } else if index == 1 { 2.0 } else { 0.0 },
                "comparison_vector": [2, 1]
            },
            "source_ref": "resolve"
        }));
    }
    let closure = run_saturation(
        facts_from_payload(&json!({"facts": facts})).unwrap(),
        &SaturationProgram::from_template(SaturationTemplateId::EntityResolution),
    )
    .expect("long-path saturation should content-address derived statements");
    let direct = closure
        .derived_statements
        .iter()
        .find(|statement| {
            statement.subject_id == "r:00" && statement.attributes["target_id"] == "r:01"
        })
        .unwrap();
    assert!((direct.confidence - (1.0 / 3.0)).abs() < 1e-12);
    let greater_than_one = closure
        .derived_statements
        .iter()
        .find(|statement| {
            statement.subject_id == "r:01" && statement.attributes["target_id"] == "r:02"
        })
        .unwrap();
    assert!((greater_than_one.confidence - 0.8).abs() < 1e-12);
    assert!(closure.derived_statements.iter().any(|statement| {
        statement.subject_id == "r:00" && statement.attributes["target_id"] == "r:35"
    }));
    assert!(closure.derived_statements.iter().any(|statement| {
        statement.subject_id == "r:34" && statement.attributes["target_id"] == "r:35"
    }));
    assert!(closure.egglog_error.is_none());
    assert!(closure.egglog_iteration_budget >= 36);
    assert_eq!(closure.egglog_checked_pair_count, 1);
    assert!(closure.egglog_tuple_count > 0);
}

#[test]
fn entity_resolution_scope_cache_is_tenant_qualified() {
    for (tenant, component) in [("tenant-a", "a"), ("tenant-b", "b")] {
        let event = MutationEvent::new(
            MutationKind::EdgeUpserted,
            tenant,
            "same-fact",
            vec!["match_edge".into()],
            vec![format!("component_key:entity-component:{component}|x|y")],
            1,
            0,
        );
        assert_eq!(
            coalesce_by_subgraph(&event),
            Some(format!("saturation-subgraph:entity-component:{component}"))
        );
    }
    for (tenant, component) in [("tenant-a", "a"), ("tenant-b", "b")] {
        let event = MutationEvent::new(
            MutationKind::EdgeDeleted,
            tenant,
            "same-fact",
            vec!["match_edge".into()],
            vec![],
            2,
            0,
        );
        assert_eq!(
            coalesce_by_subgraph(&event),
            Some(format!("saturation-subgraph:entity-component:{component}"))
        );
    }
}

#[test]
fn materialization_is_idempotent_and_marks_derived_artifacts_quarantined() {
    let mut store = InMemoryGraphStore::new();
    for id in ["a", "b", "c", "d"] {
        store.upsert_node(object(id)).unwrap();
    }
    for edge in [
        dep_edge("dep:ab", "a", "b", 0.5),
        dep_edge("dep:bd", "b", "d", 0.8),
        dep_edge("dep:ac", "a", "c", 0.9),
        dep_edge("dep:cd", "c", "d", 0.9),
    ] {
        store.upsert_edge(edge).unwrap();
    }

    let config = SaturationConfig {
        computed_at: 42,
        ..SaturationConfig::default()
    };
    let closure = run_saturation(
        facts_from_subgraph(&store, &[]).expect("fixture subgraph should produce saturation facts"),
        &SaturationProgram::default(),
    )
    .expect("fixture saturation should content-address derived statements");
    materialize_closure(&mut store, closure.clone(), config.clone()).unwrap();
    let stats_after_first = store.stats();

    let second = materialize_closure(&mut store, closure, config).unwrap();
    let stats_after_second = store.stats();

    assert_eq!(stats_after_second, stats_after_first);
    assert_eq!(second.nodes_written, 0);
    assert_eq!(second.edges_written, 0);
    let derived_nodes =
        store.query_nodes(NodeQuery::label(SATURATION_DERIVED_STATEMENT_LABEL).with_limit(100_000));
    assert!(!derived_nodes.is_empty());
    assert!(derived_nodes
        .iter()
        .all(|node| node.properties["quarantine"] == json!(true)));
    assert!(derived_nodes
        .iter()
        .all(|node| node.properties.get("field_provenance").is_some()));
}

#[test]
fn saturation_reads_asserted_statement_hyperedges_as_facts() {
    let mut store = InMemoryGraphStore::new();
    for id in ["a", "b", "c"] {
        store.upsert_node(object(id)).unwrap();
    }
    let ab = StatementRecord::assert(
        "a",
        "claim_dependency",
        "b",
        json!({ "confidence": 1.0, "justification_type": "dependency" }),
    )
    .expect("asserted statement should be content-addressable");
    let bc = StatementRecord::assert(
        "b",
        "claim_dependency",
        "c",
        json!({ "confidence": 1.0, "justification_type": "dependency" }),
    )
    .expect("asserted statement should be content-addressable");
    write_statement(&mut store, ab.clone(), false).unwrap();
    write_statement(&mut store, bc, false).unwrap();

    let facts = facts_from_subgraph(&store, &[])
        .expect("asserted statements should produce saturation facts");
    assert!(facts
        .facts
        .iter()
        .any(|fact| fact["relation"] == json!("claim_dependency") && fact["fact_id"] == ab.id));
    let closure = run_saturation(facts, &SaturationProgram::default())
        .expect("asserted-statement saturation should content-address derived statements");
    let stmt = support_stmt(&closure, "a", "c");

    assert_eq!(stmt.attributes["path_length"], json!(2));
}

#[test]
fn saturation_materializes_statement_hypernodes_and_flatten_view() {
    let mut store = InMemoryGraphStore::new();
    for id in ["a", "b", "d"] {
        store.upsert_node(object(id)).unwrap();
    }
    for edge in [
        dep_edge("dep:ab", "a", "b", 0.5),
        dep_edge("dep:bd", "b", "d", 0.8),
    ] {
        store.upsert_edge(edge).unwrap();
    }
    let closure = run_saturation(
        facts_from_subgraph(&store, &[]).expect("fixture subgraph should produce saturation facts"),
        &SaturationProgram::default(),
    )
    .expect("fixture saturation should content-address derived statements");
    materialize_closure(
        &mut store,
        closure,
        SaturationConfig {
            computed_at: 11,
            ..SaturationConfig::default()
        },
    )
    .unwrap();

    let derived = derived_node_by_subject(&store, "a", "d").expect("a->d statement");
    assert!(derived.labels.iter().any(|label| label == STATEMENT_LABEL));
    assert!(store
        .neighbors(NeighborQuery::out(&derived.id).with_edge_type(HAS_SUBJECT))
        .into_iter()
        .any(|hit| hit.node_id == "a"));
    assert!(store
        .neighbors(NeighborQuery::out(&derived.id).with_edge_type(HAS_OBJECT))
        .into_iter()
        .any(|hit| hit.node_id == "d"));

    let triples = rustyred_thg_core::flatten_statements(
        &store,
        StatementQuery::subject("a").with_relation("support_reachable"),
    );
    let triple = triples
        .iter()
        .find(|triple| triple.object == FlatObject::Entity("d".to_string()))
        .expect("flattened a support_reachable d triple");
    assert!((triple.confidence.value() - 0.4).abs() < 1e-9);
    assert_eq!(
        triple
            .provenance
            .as_ref()
            .expect("statement provenance")
            .dependency_fact_ids,
        vec!["dep:ab", "dep:bd"]
    );
}

#[test]
fn retraction_recomputes_confidence_and_tombstones_only_unsupported_view() {
    let mut store = InMemoryGraphStore::new();
    for id in ["a", "b", "c", "d", "x"] {
        store.upsert_node(object(id)).unwrap();
    }
    for edge in [
        dep_edge("dep:ab", "a", "b", 0.5),
        dep_edge("dep:bd", "b", "d", 0.8),
        dep_edge("dep:ac", "a", "c", 0.9),
        dep_edge("dep:cd", "c", "d", 0.9),
        dep_edge("dep:xc", "x", "c", 1.0),
    ] {
        store.upsert_edge(edge).unwrap();
    }
    let config = SaturationConfig {
        computed_at: 7,
        ..SaturationConfig::default()
    };
    let closure = run_saturation(
        facts_from_subgraph(&store, &[]).expect("fixture subgraph should produce saturation facts"),
        &SaturationProgram::default(),
    )
    .expect("pre-retraction saturation should content-address derived statements");
    materialize_closure(&mut store, closure, config.clone()).unwrap();
    assert!(derived_node_by_subject(&store, "x", "d").is_some());

    let mut removed = store.get_edge("dep:cd").unwrap().clone();
    removed.tombstone = true;
    store.upsert_edge(removed).unwrap();
    let closure = run_saturation(
        facts_from_subgraph(&store, &[])
            .expect("retracted subgraph should produce saturation facts"),
        &SaturationProgram::default(),
    )
    .expect("post-retraction saturation should content-address derived statements");
    let report = materialize_closure(
        &mut store,
        closure,
        SaturationConfig {
            retracted_fact_ids: vec!["dep:cd".to_string()],
            ..config
        },
    )
    .unwrap();

    let ad = derived_node_by_subject(&store, "a", "d").expect("a->d survives");
    assert!((ad.properties["confidence"].as_f64().unwrap() - 0.4).abs() < 1e-9);
    let deps = ad.properties["dependency_fact_ids"].as_array().unwrap();
    assert!(!deps.iter().any(|dep| dep == "dep:ac" || dep == "dep:cd"));
    assert!(derived_node_by_subject(&store, "x", "d").is_none());
    assert_eq!(report.revision.stale_nodes_tombstoned, 2);
}

#[test]
fn saturation_handler_reaches_fixpoint_within_one_call() {
    let mut store = RedCoreGraphStore::memory();
    for id in ["a", "b", "c", "d"] {
        store.upsert_node(object(id)).unwrap();
    }
    for edge in [
        dep_edge("dep:ab", "a", "b", 1.0),
        dep_edge("dep:bc", "b", "c", 1.0),
        dep_edge("dep:cd", "c", "d", 1.0),
    ] {
        store.upsert_edge(edge).unwrap();
    }
    let events = ["dep:ab", "dep:bc", "dep:cd"]
        .into_iter()
        .map(|id| {
            MutationEvent::new(
                MutationKind::EdgeUpserted,
                "tenant",
                id,
                vec!["CLAIM_DEPENDENCY".to_string()],
                Vec::new(),
                1,
                0,
            )
        })
        .collect::<Vec<_>>();
    let mut ctx = HookContext {
        store: &mut store,
        tenant: "tenant",
        depth: 1,
    };

    let outcome = saturation_handler(&mut ctx, &events).expect("handler");

    assert!(matches!(outcome, HookOutcome::Wrote { .. }));
    let facts = facts_from_subgraph(ctx.store, &[])
        .expect("handler result should produce saturation facts");
    let closure = run_saturation(facts, &SaturationProgram::default())
        .expect("handler verification saturation should content-address derived statements");
    let stmt = support_stmt(&closure, "a", "d");
    assert_eq!(stmt.attributes["path_length"], json!(3));
}
