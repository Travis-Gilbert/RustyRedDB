use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustyred_thg_core::{
    execute_query, execute_query_with_resolver, AmResult, Direction, LayerId, LayerRegistry,
    ModalityResolver, Predicate, PredicateMode, Projection, QueryIr, QueryRelation, RankOutcome,
    RankedRow, RegionRef, RelationalLens, ResolvedRow, ScalarBound, ScalarValue,
};
use rustyred_thg_query_saturate::{saturate_query, RegistryHandles, SaturateConfig};

#[derive(Default)]
struct FixtureResolver;

impl ModalityResolver for FixtureResolver {
    fn vector_knn(
        &self,
        _relation: &str,
        _column: &str,
        _query: &[f32],
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<RankOutcome> {
        let mut rows = [
            ("row:c".to_string(), 0.9),
            ("row:b".to_string(), 0.8),
            ("row:a".to_string(), 0.7),
        ]
        .into_iter()
        .filter(|(id, _)| candidates.is_none_or(|candidates| candidates.contains(id)))
        .take(k)
        .map(|(row_id, score)| RankedRow::new(row_id, score))
        .collect::<Vec<_>>();
        rows.truncate(k);
        Ok(RankOutcome {
            rows,
            strategy: Some("fixture_exact".to_string()),
            overfetch_rounds: 0,
        })
    }

    fn text_rank(
        &self,
        _relation: &str,
        _column: &str,
        _query: &str,
        _candidates: Option<&BTreeSet<String>>,
        _k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        Ok(Vec::new())
    }

    fn expand_proximity(
        &self,
        _from: &str,
        _layers: &[LayerId],
        _dir: Direction,
        _candidates: Option<&BTreeSet<String>>,
        _k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        Ok(Vec::new())
    }

    fn text_contains(
        &self,
        _relation: &str,
        _column: &str,
        _query: &str,
    ) -> AmResult<Option<Vec<String>>> {
        Ok(None)
    }

    fn geo_overapprox(
        &self,
        _relation: &str,
        _column: &str,
        _lat_property: Option<&str>,
        _lon_property: Option<&str>,
        _label: Option<&str>,
        _region: &RegionRef,
    ) -> AmResult<Option<Vec<String>>> {
        Ok(None)
    }

    fn expand_reachable(
        &self,
        _from: &str,
        _layers: &[LayerId],
        _dir: Direction,
    ) -> AmResult<Option<Vec<(String, LayerId)>>> {
        Ok(None)
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
            .expect("seed row");
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
    .expect("prime access methods");
    store
}

fn handles<'a>(store: &'a RelationalLens, layers: &'a LayerRegistry) -> RegistryHandles<'a> {
    RegistryHandles {
        access_methods: store.access_methods(),
        ranking_methods: store.ranking_methods(),
        layers,
        current_graph_version: store
            .access_methods()
            .indexed_graph_version()
            .expect("indexed fixture"),
    }
}

fn ids_and_scores(result: &rustyred_thg_core::QueryResult) -> Vec<(String, Option<f32>)> {
    result
        .rows
        .iter()
        .map(|row| {
            (
                row.values
                    .get("i.id")
                    .and_then(ScalarValue::as_str)
                    .expect("projected id")
                    .to_string(),
                row.score,
            )
        })
        .collect()
}

#[test]
fn saturated_filters_execute_identically_and_improve_snapshot_cost() {
    let store = fixture();
    let layers = LayerRegistry::with_native_defaults();
    let query = QueryIr {
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
        ..QueryIr::default()
    };
    let greedy =
        execute_query_with_resolver(&store, query.clone(), &FixtureResolver).expect("greedy query");
    let plan = saturate_query(
        &query,
        &handles(&store, &layers),
        &SaturateConfig::enabled(),
    )
    .expect("saturated plan");
    let saturated = execute_query_with_resolver(&store, plan.query.clone(), &FixtureResolver)
        .expect("saturated query");

    assert_eq!(ids_and_scores(&greedy), ids_and_scores(&saturated));
    assert!(plan.extracted_cost < plan.greedy_cost);
    assert!(!plan.proof.steps.is_empty());
    assert!(plan.proof.verified);
}

#[test]
fn saturated_rank_preserves_ordered_ids_and_scores() {
    let store = fixture();
    let layers = LayerRegistry::with_native_defaults();
    let query = QueryIr {
        relations: vec![QueryRelation {
            alias: "i".to_string(),
            relation: "items".to_string(),
            predicates: vec![
                Predicate::Prefix {
                    column: "name".to_string(),
                    prefix: "a".to_string(),
                },
                Predicate::Knn {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0],
                    k: 2,
                },
            ],
        }],
        limit: Some(2),
        ..QueryIr::default()
    };
    let greedy =
        execute_query_with_resolver(&store, query.clone(), &FixtureResolver).expect("greedy rank");
    let plan = saturate_query(
        &query,
        &handles(&store, &layers),
        &SaturateConfig::enabled(),
    )
    .expect("saturated rank plan");
    let saturated =
        execute_query_with_resolver(&store, plan.query, &FixtureResolver).expect("saturated rank");

    assert_eq!(ids_and_scores(&greedy), ids_and_scores(&saturated));
}

#[derive(Default)]
struct CorpusResolver {
    vectors: BTreeMap<(String, String), BTreeMap<String, Vec<f32>>>,
    texts: BTreeMap<(String, String), BTreeMap<String, String>>,
    coords: BTreeMap<(String, String), BTreeMap<String, (f64, f64)>>,
    edges: BTreeMap<(String, String), Vec<String>>,
}

impl CorpusResolver {
    fn reachable(&self, from: &str, layer: &str, dir: Direction) -> BTreeMap<String, usize> {
        let mut reached = BTreeMap::new();
        let mut queue = VecDeque::from([(from.to_string(), 0usize)]);
        while let Some((node, distance)) = queue.pop_front() {
            let neighbors = match dir {
                Direction::Out => self
                    .edges
                    .get(&(node.clone(), layer.to_string()))
                    .cloned()
                    .unwrap_or_default(),
                Direction::In => self
                    .edges
                    .iter()
                    .filter(|((_, candidate_layer), targets)| {
                        candidate_layer == layer && targets.contains(&node)
                    })
                    .map(|((source, _), _)| source.clone())
                    .collect(),
            };
            for next in neighbors {
                if next != from && !reached.contains_key(&next) {
                    reached.insert(next.clone(), distance + 1);
                    queue.push_back((next, distance + 1));
                }
            }
        }
        reached
    }
}

impl ModalityResolver for CorpusResolver {
    fn vector_knn(
        &self,
        relation: &str,
        column: &str,
        query: &[f32],
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<RankOutcome> {
        let mut rows = self
            .vectors
            .get(&(relation.to_string(), column.to_string()))
            .into_iter()
            .flat_map(|vectors| vectors.iter())
            .filter(|(id, vector)| {
                vector.len() == query.len()
                    && candidates.is_none_or(|candidates| candidates.contains(*id))
            })
            .map(|(id, vector)| {
                let score = vector
                    .iter()
                    .zip(query)
                    .map(|(left, right)| left * right)
                    .sum();
                RankedRow::new(id.clone(), score)
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.row_id.cmp(&right.row_id))
        });
        rows.truncate(k);
        Ok(RankOutcome {
            rows,
            strategy: Some("corpus_exact".to_string()),
            overfetch_rounds: 0,
        })
    }

    fn text_rank(
        &self,
        relation: &str,
        column: &str,
        query: &str,
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        let query = query.to_lowercase();
        let mut rows = self
            .texts
            .get(&(relation.to_string(), column.to_string()))
            .into_iter()
            .flat_map(|texts| texts.iter())
            .filter(|(id, text)| {
                candidates.is_none_or(|candidates| candidates.contains(*id))
                    && text.to_lowercase().contains(&query)
            })
            .map(|(id, text)| {
                RankedRow::new(
                    id.clone(),
                    text.to_lowercase().matches(&query).count() as f32,
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.row_id.cmp(&right.row_id))
        });
        rows.truncate(k);
        Ok(rows)
    }

    fn expand_proximity(
        &self,
        from: &str,
        layers: &[LayerId],
        dir: Direction,
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        if layers.len() != 1 {
            return Ok(Vec::new());
        }
        let layer = &layers[0];
        let mut rows = self
            .reachable(from, layer.as_str(), dir)
            .into_iter()
            .filter(|(id, _)| candidates.is_none_or(|candidates| candidates.contains(id)))
            .map(|(row_id, distance)| RankedRow {
                row_id,
                score: 1.0 / (distance as f32 + 1.0),
                layer: Some(layer.clone()),
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.row_id.cmp(&right.row_id))
        });
        rows.truncate(k);
        Ok(rows)
    }

    fn text_contains(
        &self,
        relation: &str,
        column: &str,
        query: &str,
    ) -> AmResult<Option<Vec<String>>> {
        let Some(texts) = self.texts.get(&(relation.to_string(), column.to_string())) else {
            return Ok(None);
        };
        let query = query.to_lowercase();
        Ok(Some(
            texts
                .iter()
                .filter(|(_, text)| text.to_lowercase().contains(&query))
                .map(|(id, _)| id.clone())
                .collect(),
        ))
    }

    fn geo_overapprox(
        &self,
        relation: &str,
        column: &str,
        _lat_property: Option<&str>,
        _lon_property: Option<&str>,
        _label: Option<&str>,
        region: &RegionRef,
    ) -> AmResult<Option<Vec<String>>> {
        let RegionRef::Bbox {
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        } = region;
        Ok(self
            .coords
            .get(&(relation.to_string(), column.to_string()))
            .map(|coords| {
                coords
                    .iter()
                    .filter(|(_, (lat, lon))| {
                        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
                    })
                    .map(|(id, _)| id.clone())
                    .collect()
            }))
    }

    fn expand_reachable(
        &self,
        from: &str,
        layers: &[LayerId],
        dir: Direction,
    ) -> AmResult<Option<Vec<(String, LayerId)>>> {
        let mut tagged = Vec::new();
        for layer in layers {
            tagged.extend(
                self.reachable(from, layer.as_str(), dir.clone())
                    .into_keys()
                    .map(|id| (id, layer.clone())),
            );
        }
        Ok(Some(tagged))
    }
}

fn corpus_fixture() -> (RelationalLens, CorpusResolver) {
    let mut store = RelationalLens::new();
    let mut resolver = CorpusResolver::default();
    for (id, kind, time, body, vector) in [
        ("doc:a", "episode", 12, "alpha beta", vec![0.7, 0.3]),
        ("doc:b", "episode", 14, "alpha alpha", vec![1.0, 0.0]),
        ("doc:c", "episode", 15, "beta gamma", vec![0.8, 0.2]),
        ("doc:d", "note", 30, "gamma", vec![0.2, 0.8]),
    ] {
        store
            .upsert_row(ResolvedRow::new(
                "docs",
                id,
                BTreeMap::from([
                    ("kind".to_string(), ScalarValue::String(kind.to_string())),
                    ("t_ms".to_string(), ScalarValue::I64(time)),
                    ("body".to_string(), ScalarValue::String(body.to_string())),
                ]),
            ))
            .unwrap();
        resolver
            .vectors
            .entry(("docs".to_string(), "embedding".to_string()))
            .or_default()
            .insert(id.to_string(), vector);
        resolver
            .texts
            .entry(("docs".to_string(), "body".to_string()))
            .or_default()
            .insert(id.to_string(), body.to_string());
    }
    for (id, lat, lon) in [("place:inside", 0.5, 0.5), ("place:outside", 2.0, 2.0)] {
        store
            .upsert_row(ResolvedRow::new(
                "places",
                id,
                BTreeMap::from([
                    ("lat".to_string(), ScalarValue::F64(lat)),
                    ("lon".to_string(), ScalarValue::F64(lon)),
                ]),
            ))
            .unwrap();
        resolver
            .coords
            .entry(("places".to_string(), "geo".to_string()))
            .or_default()
            .insert(id.to_string(), (lat, lon));
    }
    for id in ["node:root", "node:a", "node:b", "node:both"] {
        store
            .upsert_row(ResolvedRow::new(
                "nodes",
                id,
                BTreeMap::from([("name".to_string(), ScalarValue::String(id.to_string()))]),
            ))
            .unwrap();
    }
    for (layer, target) in [
        ("composition", "node:a"),
        ("composition", "node:both"),
        ("authority", "node:b"),
        ("authority", "node:both"),
    ] {
        resolver
            .edges
            .entry(("node:root".to_string(), layer.to_string()))
            .or_default()
            .push(target.to_string());
    }
    execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "d".to_string(),
                relation: "docs".to_string(),
                predicates: Vec::new(),
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();
    (store, resolver)
}

fn relation(alias: &str, name: &str, predicates: Vec<Predicate>) -> QueryIr {
    QueryIr {
        relations: vec![QueryRelation {
            alias: alias.to_string(),
            relation: name.to_string(),
            predicates,
        }],
        projection: vec![Projection {
            alias: alias.to_string(),
            column: "id".to_string(),
        }],
        ..QueryIr::default()
    }
}

fn result_rows(result: &rustyred_thg_core::QueryResult, alias: &str) -> Vec<(String, Option<f32>)> {
    result
        .rows
        .iter()
        .map(|row| {
            (
                row.values[&format!("{alias}.id")]
                    .as_str()
                    .unwrap()
                    .to_string(),
                row.score,
            )
        })
        .collect()
}

#[test]
fn existing_multimodal_and_multiplex_shapes_are_differentially_equivalent() {
    let (store, resolver) = corpus_fixture();
    let layers = LayerRegistry::with_native_defaults();
    let mut knn = relation(
        "d",
        "docs",
        vec![
            Predicate::Knn {
                column: "embedding".to_string(),
                query: vec![1.0, 0.0],
                k: 2,
            },
            Predicate::Equals {
                column: "kind".to_string(),
                value: ScalarValue::String("episode".to_string()),
            },
            Predicate::TimeRange {
                column: "t_ms".to_string(),
                lo_ms: 10,
                hi_ms: 20,
            },
        ],
    );
    knn.limit = Some(2);
    let cases = vec![
        (
            "equals",
            relation(
                "d",
                "docs",
                vec![Predicate::Equals {
                    column: "kind".to_string(),
                    value: ScalarValue::String("episode".to_string()),
                }],
            ),
            false,
        ),
        (
            "time_range",
            relation(
                "d",
                "docs",
                vec![Predicate::TimeRange {
                    column: "t_ms".to_string(),
                    lo_ms: 13,
                    hi_ms: 20,
                }],
            ),
            false,
        ),
        (
            "numeric_range",
            relation(
                "d",
                "docs",
                vec![Predicate::Range {
                    column: "t_ms".to_string(),
                    lo: ScalarBound::Included(ScalarValue::I64(13)),
                    hi: ScalarBound::Included(ScalarValue::I64(20)),
                }],
            ),
            false,
        ),
        (
            "text_filter",
            relation(
                "d",
                "docs",
                vec![Predicate::TextMatch {
                    column: "body".to_string(),
                    query: "alpha".to_string(),
                    mode: PredicateMode::Filter,
                }],
            ),
            false,
        ),
        (
            "geo_bbox",
            relation(
                "p",
                "places",
                vec![Predicate::GeoWithin {
                    column: "geo".to_string(),
                    region: RegionRef::Bbox {
                        min_lat: 0.0,
                        min_lon: 0.0,
                        max_lat: 1.0,
                        max_lon: 1.0,
                    },
                    lat_property: Some("lat".to_string()),
                    lon_property: Some("lon".to_string()),
                    label: Some("Place".to_string()),
                }],
            ),
            false,
        ),
        (
            "expand_filter_one_layer",
            relation(
                "n",
                "nodes",
                vec![Predicate::Expand {
                    from: "node:root".to_string(),
                    layers: vec![LayerId::from("composition")],
                    dir: Direction::Out,
                    mode: PredicateMode::Filter,
                }],
            ),
            false,
        ),
        (
            "expand_filter_multi_layer",
            relation(
                "n",
                "nodes",
                vec![Predicate::Expand {
                    from: "node:root".to_string(),
                    layers: vec![LayerId::from("composition"), LayerId::from("authority")],
                    dir: Direction::Out,
                    mode: PredicateMode::Filter,
                }],
            ),
            false,
        ),
        ("knn_equals_time", knn, true),
        (
            "expand_rank",
            relation(
                "n",
                "nodes",
                vec![Predicate::Expand {
                    from: "node:root".to_string(),
                    layers: vec![LayerId::from("composition")],
                    dir: Direction::Out,
                    mode: PredicateMode::Rank,
                }],
            ),
            true,
        ),
    ];

    for (name, query, ranked) in cases {
        let alias = query.relations[0].alias.clone();
        let greedy = execute_query_with_resolver(&store, query.clone(), &resolver)
            .unwrap_or_else(|error| panic!("{name} greedy failed: {error:?}"));
        let plan = saturate_query(
            &query,
            &handles(&store, &layers),
            &SaturateConfig::enabled(),
        )
        .unwrap_or_else(|error| panic!("{name} saturation failed: {error:?}"));
        let saturated = execute_query_with_resolver(&store, plan.query, &resolver)
            .unwrap_or_else(|error| panic!("{name} saturated execution failed: {error:?}"));
        let mut greedy_rows = result_rows(&greedy, &alias);
        let mut saturated_rows = result_rows(&saturated, &alias);
        if !ranked {
            greedy_rows.sort_by(|left, right| left.0.cmp(&right.0));
            saturated_rows.sort_by(|left, right| left.0.cmp(&right.0));
        }
        assert_eq!(greedy_rows, saturated_rows, "{name}");
    }
}

#[test]
fn multi_layer_rank_refuses_in_saturation_and_execution() {
    let (store, resolver) = corpus_fixture();
    let layers = LayerRegistry::with_native_defaults();
    let query = relation(
        "n",
        "nodes",
        vec![Predicate::Expand {
            from: "node:root".to_string(),
            layers: vec![LayerId::from("composition"), LayerId::from("authority")],
            dir: Direction::Out,
            mode: PredicateMode::Rank,
        }],
    );
    let saturation_error = saturate_query(
        &query,
        &handles(&store, &layers),
        &SaturateConfig::enabled(),
    )
    .unwrap_err();
    assert_eq!(saturation_error.code, "multi_layer_rank_undefined");
    let execution_error = execute_query_with_resolver(&store, query, &resolver).unwrap_err();
    assert_eq!(execution_error.code, "multi_layer_rank_undefined");
}
