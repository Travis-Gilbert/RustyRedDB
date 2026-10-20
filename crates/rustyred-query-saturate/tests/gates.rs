use std::collections::{BTreeMap, BTreeSet};

use egg::{Extractor, RecExpr, Runner, StopReason};
use proptest::prelude::*;
use rustyred_thg_core::{
    execute_query, AccessMethodRegistry, LayerId, LayerRegistry, Predicate, PredicateMode, QueryIr,
    QueryRelation, RankingRegistry, RegionRef, RelationalLens, ResolvedRow, ScalarBound,
    ScalarValue,
};
use rustyred_thg_query_saturate::cost::{
    ensure_snapshot_current, predicate_cost_key, rank_node_layer_cardinality, CostSnapshot,
    QueryCostFunction, SaturationAnalysis,
};
use rustyred_thg_query_saturate::proof::semantic_fingerprint;
use rustyred_thg_query_saturate::rules::{
    known_rule_ids, LAYER_MERGE, LAYER_SPLIT, PREDICATE_EQUALS_RANGE, PREDICATE_PREFIX_RANGE,
    RANK_CANDIDATE_ASSOCIATE, RANK_CANDIDATE_ASSOCIATE_REVERSE, RANK_CANDIDATE_PUSHDOWN,
    SET_ASSOCIATE, SET_COMMUTE, TOPK_LIMIT_COLLAPSE, TOPK_LIMIT_UNION,
};
use rustyred_thg_query_saturate::{
    expr_to_query_input, expr_to_query_ir, query_input_to_expr, replay_query_proof,
    rewrite_rules_v1, saturate_query, EncodedAttr, ExtractionStrategy,
    QueryCostFunction as PublicQueryCostFunction, QueryLang, QueryPlanProof, RegistryHandles,
    SaturateConfig,
};
use serde::Serialize;

fn query() -> QueryIr {
    QueryIr {
        relations: vec![QueryRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
            predicates: vec![
                Predicate::Equals {
                    column: "age".to_string(),
                    value: ScalarValue::I64(7),
                },
                Predicate::Prefix {
                    column: "name".to_string(),
                    prefix: "al".to_string(),
                },
            ],
        }],
        limit: Some(10),
        ..QueryIr::default()
    }
}

fn fixture() -> RelationalLens {
    let mut store = RelationalLens::new();
    for (id, age, name) in [
        ("row:a", 7, "alpha"),
        ("row:b", 7, "alpine"),
        ("row:c", 8, "beta"),
    ] {
        store
            .upsert_row(ResolvedRow::new(
                "items",
                id,
                BTreeMap::from([
                    ("age".to_string(), ScalarValue::I64(age)),
                    ("name".to_string(), ScalarValue::String(name.to_string())),
                ]),
            ))
            .unwrap();
    }
    execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "i".to_string(),
                relation: "items".to_string(),
                predicates: Vec::new(),
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();
    store
}

fn handles<'a>(store: &'a RelationalLens, layers: &'a LayerRegistry) -> RegistryHandles<'a> {
    RegistryHandles {
        access_methods: store.access_methods(),
        ranking_methods: store.ranking_methods(),
        layers,
        current_graph_version: store.access_methods().indexed_graph_version().unwrap(),
    }
}

#[test]
fn golden_multi_predicate_term_is_stable_and_round_trips() {
    let input = query();
    let term = query_input_to_expr(&input, 7, "snapshot:g1").unwrap();
    assert_eq!(
        term.to_string(),
        include_str!("../golden/multi_predicate.term").trim()
    );
    assert_eq!(expr_to_query_input(&term).unwrap(), input);
}

#[test]
fn golden_fixture_saturates_to_fixed_point_and_round_trips() {
    let term: RecExpr<QueryLang> = include_str!("../golden/multi_predicate.term")
        .trim()
        .parse()
        .unwrap();
    let runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_expr(&term)
        .with_iter_limit(20)
        .with_node_limit(100_000)
        .run(&rewrite_rules_v1());
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    assert_eq!(expr_to_query_ir(&term).unwrap().body, query());
}

#[test]
fn relation_order_round_trips_first_seen() {
    let input = QueryIr {
        relations: vec![
            QueryRelation {
                alias: "z".to_string(),
                relation: "last_named_first".to_string(),
                predicates: Vec::new(),
            },
            QueryRelation {
                alias: "a".to_string(),
                relation: "first_named_second".to_string(),
                predicates: Vec::new(),
            },
        ],
        ..QueryIr::default()
    };
    let expr = query_input_to_expr(&input, 1, "snapshot").unwrap();
    assert_eq!(expr_to_query_ir(&expr).unwrap().body, input);
}

#[test]
fn rank_construction_refuses_multiple_layers() {
    let input = QueryIr {
        relations: vec![QueryRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
            predicates: vec![Predicate::Expand {
                from: "root".to_string(),
                layers: vec![LayerId::from("composition"), LayerId::from("authority")],
                dir: rustyred_thg_core::Direction::Out,
                mode: PredicateMode::Rank,
            }],
        }],
        ..QueryIr::default()
    };
    let error = query_input_to_expr(&input, 1, "snapshot").unwrap_err();
    assert_eq!(error.code, "multi_layer_rank_undefined");
}

#[test]
fn rewrite_library_has_every_stable_family_id() {
    let names = rewrite_rules_v1()
        .into_iter()
        .map(|rule| rule.name.to_string())
        .collect::<BTreeSet<_>>();
    for required in [
        SET_COMMUTE,
        SET_ASSOCIATE,
        LAYER_SPLIT,
        LAYER_MERGE,
        PREDICATE_EQUALS_RANGE,
        PREDICATE_PREFIX_RANGE,
        RANK_CANDIDATE_PUSHDOWN,
        RANK_CANDIDATE_ASSOCIATE,
        RANK_CANDIDATE_ASSOCIATE_REVERSE,
        TOPK_LIMIT_UNION,
        TOPK_LIMIT_COLLAPSE,
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    assert_eq!(
        known_rule_ids()
            .iter()
            .map(|name| name.to_string())
            .collect::<BTreeSet<_>>(),
        names
    );
}

#[test]
fn set_and_layer_families_join_equivalent_terms() {
    let leaf = EncodedAttr::from_json(&"x").unwrap().to_string();
    let cases = [
        (
            format!("(intersect {leaf} (union {leaf} {leaf}))"),
            format!("(intersect (union {leaf} {leaf}) {leaf})"),
        ),
        (
            format!(
                "(expand {leaf} (layers (layer {leaf}) (layers (layer {leaf}) layers-empty)) {leaf})"
            ),
            format!(
                "(union (expand {leaf} (layers (layer {leaf}) layers-empty) {leaf}) (expand {leaf} (layers (layer {leaf}) layers-empty) {leaf}))"
            ),
        ),
    ];
    let rules = rewrite_rules_v1();
    for (before, after) in cases {
        let before = before.parse().unwrap();
        let after = after.parse().unwrap();
        let mut runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
            .with_expr(&before)
            .with_iter_limit(4)
            .run(&rules);
        let left = runner.egraph.add_expr(&before);
        let right = runner.egraph.add_expr(&after);
        runner.egraph.rebuild();
        assert_eq!(runner.egraph.find(left), runner.egraph.find(right));
    }
}

#[test]
fn predicate_algebra_families_preserve_semantic_fingerprint() {
    let input = query_input_to_expr(&query(), 7, "snapshot").unwrap();
    let rules = rewrite_rules_v1();
    let runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_explanations_enabled()
        .with_expr(&input)
        .with_iter_limit(4)
        .run(&rules);
    let root = runner.roots[0];
    assert!(
        runner.egraph[root].nodes.len() > 1 || runner.egraph.total_size() > input.as_ref().len()
    );

    let snapshot =
        CostSnapshot::from_cost_maps(7, BTreeMap::new(), BTreeMap::new(), BTreeMap::new()).unwrap();
    let (_, extracted) =
        Extractor::new(&runner.egraph, QueryCostFunction { snapshot }).find_best(root);
    assert_eq!(
        semantic_fingerprint(&input.to_string()).unwrap(),
        semantic_fingerprint(&extracted.to_string()).unwrap()
    );
}

#[test]
fn topk_collapse_uses_the_smaller_limit() {
    let relation = EncodedAttr::from_json(&"relation").unwrap().to_string();
    let five = EncodedAttr::from_json(&5usize).unwrap().to_string();
    let ten = EncodedAttr::from_json(&10usize).unwrap().to_string();
    let before = format!("(limit (limit {relation} {ten}) {five})")
        .parse()
        .unwrap();
    let after = format!("(limit {relation} {five})").parse().unwrap();
    let rules = rewrite_rules_v1();
    let mut runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_expr(&before)
        .with_iter_limit(4)
        .run(&rules);
    let left = runner.egraph.add_expr(&before);
    let right = runner.egraph.add_expr(&after);
    runner.egraph.rebuild();
    assert_eq!(runner.egraph.find(left), runner.egraph.find(right));
}

#[test]
fn rank_pushdown_and_topk_union_join_their_normal_forms() {
    let a = EncodedAttr::from_json(&"a").unwrap().to_string();
    let b = EncodedAttr::from_json(&"b").unwrap().to_string();
    let c = EncodedAttr::from_json(&"c").unwrap().to_string();
    let predicate = EncodedAttr::from_json(&"predicate").unwrap().to_string();
    let k = EncodedAttr::from_json(&5usize).unwrap().to_string();
    let layer = format!("(layer {a})");
    let cases = [
        (
            format!(
                "(rank (intersect {a} {b}) {predicate} {layer} {k})"
            ),
            format!(
                "(rank (intersect {b} {a}) {predicate} {layer} {k})"
            ),
        ),
        (
            format!(
                "(rank (intersect (intersect {a} {b}) {c}) {predicate} {layer} {k})"
            ),
            format!(
                "(rank (intersect {a} (intersect {b} {c})) {predicate} {layer} {k})"
            ),
        ),
        (
            format!(
                "(limit (union (rank {a} {predicate} {layer} {k}) (rank {b} {predicate} {layer} {k})) {k})"
            ),
            format!(
                "(limit (union (limit (rank {a} {predicate} {layer} {k}) {k}) (limit (rank {b} {predicate} {layer} {k}) {k})) {k})"
            ),
        ),
    ];
    let rules = rewrite_rules_v1();
    for (before, after) in cases {
        let before = before.parse().unwrap();
        let after = after.parse().unwrap();
        let mut runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
            .with_expr(&before)
            .with_iter_limit(4)
            .run(&rules);
        let left = runner.egraph.add_expr(&before);
        let right = runner.egraph.add_expr(&after);
        runner.egraph.rebuild();
        assert_eq!(runner.egraph.find(left), runner.egraph.find(right));
    }
}

#[test]
fn multi_layer_cost_penalty_flips_split_vs_merged_extraction() {
    let input = QueryIr {
        relations: vec![QueryRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
            predicates: vec![Predicate::Expand {
                from: "root".to_string(),
                layers: vec![LayerId::from("authority"), LayerId::from("composition")],
                dir: rustyred_thg_core::Direction::Out,
                mode: PredicateMode::Filter,
            }],
        }],
        ..QueryIr::default()
    };
    let expr = query_input_to_expr(&input, 1, "snapshot").unwrap();
    let runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_expr(&expr)
        .run(&rewrite_rules_v1());
    let root = runner.roots[0];
    let snapshot = || {
        CostSnapshot::from_cost_maps(
            1,
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::from([
                ("composition".to_string(), 1.0),
                ("authority".to_string(), 1.0),
            ]),
        )
        .unwrap()
    };
    let merged = Extractor::new(
        &runner.egraph,
        QueryCostFunction {
            snapshot: snapshot().with_combined_layer_penalty(0.0).unwrap(),
        },
    )
    .find_best(root)
    .1;
    let split = Extractor::new(
        &runner.egraph,
        QueryCostFunction {
            snapshot: snapshot().with_combined_layer_penalty(5.0).unwrap(),
        },
    )
    .find_best(root)
    .1;

    assert!(!merged.to_string().contains("(union "));
    assert!(split.to_string().contains("(union "));
    assert_eq!(expr_to_query_input(&merged).unwrap(), input);
    assert_eq!(expr_to_query_input(&split).unwrap(), input);
}

#[test]
fn cost_snapshot_keeps_full_layer_descriptors() {
    let store = fixture();
    let layers = LayerRegistry::with_native_defaults();
    let snapshot = CostSnapshot::capture(
        &query(),
        store.access_methods(),
        store.ranking_methods(),
        &layers,
    )
    .unwrap();
    let composition = snapshot.layer_descriptors.get("composition").unwrap();
    assert_eq!(
        composition.durability,
        rustyred_thg_core::LayerDurability::Durable
    );
    assert_eq!(
        composition.cardinality,
        rustyred_thg_core::LayerCardinality::Single
    );
    assert!(composition.pinnable);
}

#[test]
fn cost_flip_changes_extracted_predicate_form() {
    let input = QueryIr {
        relations: vec![QueryRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
            predicates: vec![Predicate::Equals {
                column: "age".to_string(),
                value: ScalarValue::I64(7),
            }],
        }],
        ..QueryIr::default()
    };
    let equals = input.relations[0].predicates[0].clone();
    let range = Predicate::Range {
        column: "age".to_string(),
        lo: ScalarBound::Included(ScalarValue::I64(7)),
        hi: ScalarBound::Included(ScalarValue::I64(7)),
    };
    let equals_key = predicate_cost_key("items", &equals).unwrap();
    let range_key = predicate_cost_key("items", &range).unwrap();
    let expr = query_input_to_expr(&input, 1, "snapshot").unwrap();
    let runner: Runner<QueryLang, SaturationAnalysis, ()> = Runner::new(SaturationAnalysis)
        .with_expr(&expr)
        .run(&rewrite_rules_v1());
    let root = runner.roots[0];
    let extract = |equals_cost, range_cost| {
        let snapshot = CostSnapshot::from_cost_maps(
            1,
            BTreeMap::from([
                (equals_key.clone(), equals_cost),
                (range_key.clone(), range_cost),
            ]),
            BTreeMap::new(),
            BTreeMap::new(),
        )
        .unwrap();
        Extractor::new(&runner.egraph, PublicQueryCostFunction { snapshot })
            .find_best(root)
            .1
    };
    let equals_plan = expr_to_query_input(&extract(1.0, 10.0)).unwrap();
    let range_plan = expr_to_query_input(&extract(10.0, 1.0)).unwrap();
    assert!(matches!(
        equals_plan.relations[0].predicates[0],
        Predicate::Equals { .. }
    ));
    assert!(matches!(
        range_plan.relations[0].predicates[0],
        Predicate::Range { .. }
    ));
}

#[test]
fn stale_snapshot_refuses() {
    let registry = AccessMethodRegistry::with_native_defaults();
    registry.mark_current(3);
    let snapshot =
        CostSnapshot::from_cost_maps(2, BTreeMap::new(), BTreeMap::new(), BTreeMap::new()).unwrap();
    let error = ensure_snapshot_current(&snapshot, &registry).unwrap_err();
    assert_eq!(error.code, "stale_cost_snapshot");
}

#[test]
fn proof_completeness_tamper_rejection_and_determinism() {
    let store = fixture();
    let layers = LayerRegistry::with_native_defaults();
    let config = SaturateConfig::enabled();
    let first = saturate_query(&query(), &handles(&store, &layers), &config).unwrap();
    let second = saturate_query(&query(), &handles(&store, &layers), &config).unwrap();
    assert_eq!(first.extracted_term, second.extracted_term);
    assert!(first.proof.verified);
    assert!(replay_query_proof(&first.proof).verified);

    let mut unknown = first.proof.clone();
    unknown.steps[0].rule_id = "unknown-rule".to_string();
    let report = replay_query_proof(&unknown);
    assert!(!report.verified);
    assert!(report.message.contains("unknown-rule"));

    let mut broken = first.proof.clone();
    broken.steps[0].before_term = "layers-empty".to_string();
    assert!(!replay_query_proof(&broken).verified);

    let mut changed = first.proof.clone();
    changed.output_term = query_input_to_expr(
        &QueryIr {
            relations: vec![QueryRelation {
                alias: "i".to_string(),
                relation: "items".to_string(),
                predicates: vec![],
            }],
            ..QueryIr::default()
        },
        7,
        "snapshot",
    )
    .unwrap()
    .to_string();
    assert!(!replay_query_proof(&changed).verified);
}

#[derive(Serialize)]
struct ForgedRoot {
    graph_version: u64,
    joins: Vec<rustyred_thg_core::JoinPredicate>,
    projection: Vec<rustyred_thg_core::Projection>,
    fusion: rustyred_thg_core::FusionPolicy,
    semiring: Option<rustyred_thg_core::ProvenanceSemiringKind>,
    pinned_graph_version: Option<u64>,
}

#[derive(Serialize)]
struct ForgedRelation {
    alias: String,
    relation: String,
}

#[derive(Serialize)]
struct ForgedPredicate {
    relation: ForgedRelation,
    ordinal: usize,
    predicate: Predicate,
}

#[derive(Serialize)]
struct ForgedAllRows {
    all_rows: bool,
}

fn generated_predicate() -> impl Strategy<Value = Predicate> {
    prop_oneof![
        ("[a-z]{1,8}", -100i64..100).prop_map(|(column, value)| Predicate::Equals {
            column,
            value: ScalarValue::I64(value),
        }),
        ("[a-z]{1,8}", -100i64..100, 0i64..100).prop_map(|(column, lo, width)| Predicate::Range {
            column,
            lo: ScalarBound::Included(ScalarValue::I64(lo)),
            hi: ScalarBound::Included(ScalarValue::I64(lo + width)),
        }),
        ("[a-z]{1,8}", "[a-z]{0,6}")
            .prop_map(|(column, prefix)| Predicate::Prefix { column, prefix }),
        ("[a-z]{1,8}", -10_000i64..10_000, 0i64..10_000).prop_map(|(column, lo_ms, width)| {
            Predicate::TimeRange {
                column,
                lo_ms,
                hi_ms: lo_ms + width,
            }
        }),
        ("[a-z]{1,8}", "[a-z]{1,10}").prop_map(|(column, query)| Predicate::TextMatch {
            column,
            query,
            mode: PredicateMode::Filter,
        }),
        ("[a-z]{1,10}", "[a-z]{1,10}").prop_map(|(from, layer)| Predicate::Expand {
            from,
            layers: vec![LayerId::from(layer)],
            dir: rustyred_thg_core::Direction::Out,
            mode: PredicateMode::Filter,
        }),
        (
            "[a-z]{1,8}",
            prop::collection::vec(-100i16..100, 1..8),
            1usize..20
        )
            .prop_map(|(column, values, k)| Predicate::Knn {
                column,
                query: values
                    .into_iter()
                    .map(|value| value as f32 / 100.0)
                    .collect(),
                k,
            }),
        ("[a-z]{1,8}", -80i16..80, -170i16..170, 1i16..10).prop_map(|(column, lat, lon, span)| {
            Predicate::GeoWithin {
                column,
                region: RegionRef::Bbox {
                    min_lat: lat as f64,
                    min_lon: lon as f64,
                    max_lat: (lat + span).min(90) as f64,
                    max_lon: (lon + span).min(180) as f64,
                },
                lat_property: Some("lat".to_string()),
                lon_property: Some("lon".to_string()),
                label: Some("Place".to_string()),
            }
        }),
    ]
}

#[test]
fn proof_replay_rechecks_multi_layer_rank_invariant() {
    let mut expr = RecExpr::default();
    let root = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&ForgedRoot {
            graph_version: 1,
            joins: Vec::new(),
            projection: Vec::new(),
            fusion: rustyred_thg_core::FusionPolicy::default(),
            semiring: None,
            pinned_graph_version: None,
        })
        .unwrap(),
    ));
    let snapshot = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&"snapshot".to_string()).unwrap(),
    ));
    let relation = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&ForgedRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
        })
        .unwrap(),
    ));
    let all = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&ForgedAllRows { all_rows: true }).unwrap(),
    ));
    let source = expr.add(QueryLang::Scan([relation, all]));
    let predicate = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&ForgedPredicate {
            relation: ForgedRelation {
                alias: "i".to_string(),
                relation: "items".to_string(),
            },
            ordinal: 0,
            predicate: Predicate::Expand {
                from: "root".to_string(),
                layers: vec![LayerId::from("composition"), LayerId::from("authority")],
                dir: rustyred_thg_core::Direction::Out,
                mode: PredicateMode::Rank,
            },
        })
        .unwrap(),
    ));
    let layer_attr = expr.add(QueryLang::Attr(
        EncodedAttr::from_json(&Some(LayerId::from("composition"))).unwrap(),
    ));
    let layer = expr.add(QueryLang::Layer([layer_attr]));
    let k = expr.add(QueryLang::Attr(EncodedAttr::from_json(&5usize).unwrap()));
    let rank = expr.add(QueryLang::Rank([source, predicate, layer, k]));
    expr.add(QueryLang::QueryRoot([root, snapshot, rank]));
    let term = expr.to_string();
    let proof = QueryPlanProof {
        library: "query-rewrite-library-v1".to_string(),
        input_term: term.clone(),
        output_term: term,
        extraction_strategy: ExtractionStrategy::DagAware,
        runner_stop_reason: "forged".to_string(),
        steps: Vec::new(),
        verified: false,
        analysis: Default::default(),
    };
    let report = replay_query_proof(&proof);
    assert!(!report.verified);
    assert!(report.message.contains("exactly one layer"));
}

proptest! {
    #[test]
    fn generated_queries_round_trip_byte_identically(
        predicates in prop::collection::vec(generated_predicate(), 0..8),
        limit in prop::option::of(1usize..100),
    ) {
        let input = QueryIr {
            relations: vec![QueryRelation {
                alias: "i".to_string(),
                relation: "items".to_string(),
                predicates,
            }],
            limit,
            ..QueryIr::default()
        };
        let expr = query_input_to_expr(&input, 42, "snapshot").unwrap();
        prop_assert_eq!(expr_to_query_ir(&expr).unwrap().body, input);
    }

    #[test]
    fn no_rule_widens_rank_layer_cardinality(layer in "[a-z]{1,12}") {
        let input = QueryIr {
            relations: vec![QueryRelation {
                alias: "i".to_string(),
                relation: "items".to_string(),
                predicates: vec![Predicate::Expand {
                    from: "root".to_string(),
                    layers: vec![LayerId::from(layer)],
                    dir: rustyred_thg_core::Direction::Out,
                    mode: PredicateMode::Rank,
                }],
            }],
            limit: Some(5),
            ..QueryIr::default()
        };
        let expr = query_input_to_expr(&input, 1, "snapshot").unwrap();
        let runner: Runner<QueryLang, SaturationAnalysis, ()> =
            Runner::new(SaturationAnalysis)
                .with_expr(&expr)
                .with_iter_limit(6)
                .run(&rewrite_rules_v1());
        for class in runner.egraph.classes() {
            for node in &class.nodes {
                if matches!(node, QueryLang::Rank(_)) {
                    prop_assert!(
                        rank_node_layer_cardinality(&runner.egraph, node)
                            .is_some_and(|cardinality| cardinality <= 1)
                    );
                }
            }
            prop_assert!(class.data.max_rank_layer_cardinality <= 1);
        }
        let (_, extracted) = Extractor::new(
            &runner.egraph,
            QueryCostFunction {
                snapshot: CostSnapshot::from_cost_maps(
                    1,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    BTreeMap::new(),
                ).unwrap(),
            },
        ).find_best(runner.roots[0]);
        let output = expr_to_query_input(&extracted).unwrap();
        for predicate in &output.relations[0].predicates {
            if let Predicate::Expand { layers, mode: PredicateMode::Rank, .. } = predicate {
                prop_assert_eq!(layers.len(), 1);
            }
        }
    }
}

#[test]
fn core_does_not_depend_on_query_saturate() {
    let manifest = include_str!("../../rustyred-engine/Cargo.toml");
    assert!(!manifest.contains("rustyred-thg-query-saturate"));
    assert!(!manifest.contains("rustyred-query-saturate"));
    let _ = RankingRegistry::with_native_defaults();
}
