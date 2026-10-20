//! SPEC-MULTIPLEX-LAYERS gates (ML5) and acceptance for ML1–ML3.

use std::collections::{BTreeMap, BTreeSet};

use rustyred_thg_core::{
    execute_query_with_resolver, AmResult, Direction, LayerCardinality, LayerDurability, LayerId,
    LayerRegistry, ModalityResolver, Predicate, PredicateMode, QueryIr, QueryRelation, RankOutcome,
    RankedRow, RegionRef, RelationalLens, ResolvedRow, RowIdStream, ScalarValue,
};

#[derive(Default)]
struct TwoLayerResolver {
    /// Adjacency keyed by (from, layer_id_str).
    out: BTreeMap<(String, String), Vec<String>>,
}

impl TwoLayerResolver {
    fn edge(&mut self, from: &str, layer: &str, to: &str) {
        self.out
            .entry((from.into(), layer.into()))
            .or_default()
            .push(to.into());
    }
}

impl ModalityResolver for TwoLayerResolver {
    fn vector_knn(
        &self,
        _relation: &str,
        _column: &str,
        _query: &[f32],
        _candidates: Option<&BTreeSet<String>>,
        _k: usize,
    ) -> AmResult<RankOutcome> {
        Ok(RankOutcome::default())
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
        from: &str,
        layers: &[LayerId],
        _dir: Direction,
        _candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        if layers.len() != 1 {
            return Ok(Vec::new());
        }
        let layer = &layers[0];
        let mut rows = self
            .out
            .get(&(from.into(), layer.as_str().into()))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, row_id)| RankedRow {
                row_id,
                score: 1.0 / (i as f32 + 1.0),
                layer: Some(layer.clone()),
            })
            .collect::<Vec<_>>();
        rows.truncate(k);
        Ok(rows)
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
        from: &str,
        layers: &[LayerId],
        _dir: Direction,
    ) -> AmResult<Option<Vec<(String, LayerId)>>> {
        let mut tagged = Vec::new();
        let mut seen = BTreeSet::new();
        for layer in layers {
            for to in self
                .out
                .get(&(from.into(), layer.as_str().into()))
                .cloned()
                .unwrap_or_default()
            {
                if seen.insert((to.clone(), layer.clone())) {
                    tagged.push((to, layer.clone()));
                }
            }
        }
        Ok(Some(tagged))
    }
}

fn seed_nodes(store: &mut RelationalLens<rustyred_thg_core::InMemoryGraphStore>) {
    for id in ["node:root", "node:a", "node:b", "node:both"] {
        store
            .upsert_row(ResolvedRow::new(
                "nodes",
                id,
                BTreeMap::from([("name".into(), ScalarValue::String(id.into()))]),
            ))
            .unwrap();
    }
}

/// Gate 1: multi-layer Rank never returns rows; it refuses.
#[test]
fn gate_no_silent_concatenation_multi_layer_rank_refuses() {
    let mut store = RelationalLens::new();
    seed_nodes(&mut store);
    let mut resolver = TwoLayerResolver::default();
    resolver.edge("node:root", "composition", "node:a");
    resolver.edge("node:root", "authority", "node:b");

    let err = execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".into(),
                relation: "nodes".into(),
                predicates: vec![Predicate::Expand {
                    from: "node:root".into(),
                    layers: vec!["composition".into(), "authority".into()],
                    dir: Direction::Out,
                    mode: PredicateMode::Rank,
                }],
            }],
            ..QueryIr::default()
        },
        &resolver,
    )
    .expect_err("multi-layer rank must refuse");
    assert_eq!(err.code, "multi_layer_rank_undefined");
    assert!(err.message.contains("2"));
}

/// Gate 2 / ML2: two-layer Filter unions with per-row provenance sets.
#[test]
fn gate_provenance_completeness_and_multiplex_union() {
    let mut resolver = TwoLayerResolver::default();
    resolver.edge("node:root", "composition", "node:a");
    resolver.edge("node:root", "authority", "node:b");
    resolver.edge("node:root", "composition", "node:both");
    resolver.edge("node:root", "authority", "node:both");

    let tagged = resolver
        .expand_reachable(
            "node:root",
            &["composition".into(), "authority".into()],
            Direction::Out,
        )
        .unwrap()
        .unwrap();

    let mut provenance: BTreeMap<String, BTreeSet<LayerId>> = BTreeMap::new();
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (row_id, layer) in tagged {
        provenance.entry(row_id.clone()).or_default().insert(layer);
        if seen.insert(row_id.clone()) {
            rows.push(row_id);
        }
    }
    let stream = RowIdStream::with_layers(rows, seen.len(), provenance);
    assert!(stream.rows.contains(&"node:a".into()));
    assert!(stream.rows.contains(&"node:b".into()));
    assert!(stream.rows.contains(&"node:both".into()));
    assert_eq!(stream.rows.iter().filter(|r| *r == "node:both").count(), 1);
    let both = stream.layer_provenance.get("node:both").expect("both");
    assert!(both.contains(&LayerId::from("composition")));
    assert!(both.contains(&LayerId::from("authority")));
    for row in &stream.rows {
        let set = stream
            .layer_provenance
            .get(row)
            .expect("every row has provenance");
        assert!(!set.is_empty());
    }
}

/// Gate 3: version-pinned query over presence refuses.
#[test]
fn gate_pin_safety_refuses_presence() {
    let mut store = RelationalLens::new();
    seed_nodes(&mut store);
    let resolver = TwoLayerResolver::default();

    let err = execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".into(),
                relation: "nodes".into(),
                predicates: vec![Predicate::Expand {
                    from: "node:root".into(),
                    layers: vec!["presence".into()],
                    dir: Direction::Out,
                    mode: PredicateMode::Filter,
                }],
            }],
            pinned_graph_version: Some(42),
            ..QueryIr::default()
        },
        &resolver,
    )
    .expect_err("presence is not pinnable");
    assert_eq!(err.code, "layer_not_pinnable");
    assert!(err.message.contains("presence"));

    // Same query without a pin succeeds (resolver returns empty tagged set).
    execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".into(),
                relation: "nodes".into(),
                predicates: vec![Predicate::Expand {
                    from: "node:root".into(),
                    layers: vec!["presence".into()],
                    dir: Direction::Out,
                    mode: PredicateMode::Filter,
                }],
            }],
            pinned_graph_version: None,
            ..QueryIr::default()
        },
        &resolver,
    )
    .expect("unpinned presence query ok");

    // Pinned durable-only succeeds.
    execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".into(),
                relation: "nodes".into(),
                predicates: vec![Predicate::Expand {
                    from: "node:root".into(),
                    layers: vec!["composition".into()],
                    dir: Direction::Out,
                    mode: PredicateMode::Filter,
                }],
            }],
            pinned_graph_version: Some(42),
            ..QueryIr::default()
        },
        &resolver,
    )
    .expect("pinned durable layer ok");
}

/// Gate 4: one-element layer vector matches single-layer expand behavior.
#[test]
fn gate_identity_migration_one_element_vector() {
    let mut resolver = TwoLayerResolver::default();
    resolver.edge("node:root", "LINKS", "node:a");
    resolver.edge("node:root", "LINKS", "node:b");

    let one = resolver
        .expand_reachable("node:root", &["LINKS".into()], Direction::Out)
        .unwrap()
        .unwrap();
    let ids: Vec<_> = one.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec!["node:a".to_string(), "node:b".to_string()]);

    let ranked = resolver
        .expand_proximity("node:root", &["LINKS".into()], Direction::Out, None, 10)
        .unwrap();
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].layer.as_ref().unwrap().as_str(), "LINKS");
}

#[test]
fn native_defaults_annotation_multi_presence_ephemeral() {
    let registry = LayerRegistry::with_native_defaults();
    assert_eq!(registry.layers().count(), 7);
    let annotation = registry.get(&"annotation".into()).unwrap();
    assert_eq!(annotation.cardinality, LayerCardinality::Multi);
    let presence = registry.get(&"presence".into()).unwrap();
    assert_eq!(presence.durability, LayerDurability::Ephemeral);
    assert!(!presence.pinnable);
}
