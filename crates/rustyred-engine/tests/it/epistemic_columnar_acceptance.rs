use std::collections::{BTreeMap, BTreeSet};

use rustyred_thg_core::{
    execute_query_with_resolver, AmResult, Direction, FusionPolicy, HotColumnEncoding,
    ModalityResolver, Predicate, PredicateMode, Projection, QueryIr, QueryRelation, RankOutcome,
    RankedRow, RankingRule, RegionRef, RelationalLens, ResolvedRow, RowId, ScalarBound,
    ScalarValue,
};

fn text(value: &str) -> ScalarValue {
    ScalarValue::String(value.to_string())
}

fn shadow_row(
    id: &str,
    epistemic_weight: f64,
    source_reliability: f64,
    justification_prior: f64,
    valid_from_ms: i64,
    acceptance_status: &str,
    superseded: bool,
) -> ResolvedRow {
    ResolvedRow::new(
        "epistemic",
        id,
        BTreeMap::from([
            (
                "epistemic_weight".to_string(),
                ScalarValue::F64(epistemic_weight),
            ),
            (
                "source_reliability".to_string(),
                ScalarValue::F64(source_reliability),
            ),
            (
                "justification_prior".to_string(),
                ScalarValue::F64(justification_prior),
            ),
            ("valid_from_ms".to_string(), ScalarValue::I64(valid_from_ms)),
            ("acceptance_status".to_string(), text(acceptance_status)),
            ("superseded".to_string(), ScalarValue::Bool(superseded)),
            ("body".to_string(), text("alpha evidence")),
        ]),
    )
}

#[derive(Default)]
struct EpistemicResolver {
    reachable: Vec<RowId>,
    text_scores: Vec<RankedRow>,
}

impl ModalityResolver for EpistemicResolver {
    fn vector_knn(
        &self,
        _relation: &str,
        _column: &str,
        _query: &[f32],
        _candidates: Option<&BTreeSet<RowId>>,
        _k: usize,
    ) -> AmResult<RankOutcome> {
        Ok(RankOutcome::default())
    }

    fn text_rank(
        &self,
        _relation: &str,
        _column: &str,
        _query: &str,
        candidates: Option<&BTreeSet<RowId>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        Ok(self
            .text_scores
            .iter()
            .filter(|row| candidates.is_none_or(|candidates| candidates.contains(&row.row_id)))
            .take(k)
            .cloned()
            .collect())
    }

    fn expand_proximity(
        &self,
        _from: &str,
        _layers: &[rustyred_thg_core::LayerId],
        _dir: Direction,
        _candidates: Option<&BTreeSet<RowId>>,
        _k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        Ok(Vec::new())
    }

    fn text_contains(
        &self,
        _relation: &str,
        _column: &str,
        _query: &str,
    ) -> AmResult<Option<Vec<RowId>>> {
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
    ) -> AmResult<Option<Vec<RowId>>> {
        Ok(None)
    }

    fn expand_reachable(
        &self,
        _from: &str,
        layers: &[rustyred_thg_core::LayerId],
        _dir: Direction,
    ) -> AmResult<Option<Vec<(RowId, rustyred_thg_core::LayerId)>>> {
        let layer = layers.first().cloned().unwrap_or_else(|| "NEAR".into());
        Ok(Some(
            self.reachable
                .iter()
                .cloned()
                .map(|row_id| (row_id, layer.clone()))
                .collect(),
        ))
    }
}

#[test]
fn indexed_epistemic_filters_intersect_and_trace_hot_columnar_group() {
    let mut store = RelationalLens::new();
    for idx in 0..64 {
        let high_weight = idx >= 48;
        let accepted = idx % 4 == 0;
        let superseded = idx % 2 == 1;
        let row = shadow_row(
            &format!("shadow:{idx:02}"),
            if high_weight { 1.6 } else { 0.8 },
            if accepted { 0.9 } else { 0.4 },
            0.5,
            idx * 10,
            if accepted { "accepted" } else { "contested" },
            superseded,
        );
        store.upsert_row(row).unwrap();
    }
    let group = store.hot_column_group("epistemic").unwrap();
    assert_eq!(
        group.encodings.get("acceptance_status"),
        Some(&HotColumnEncoding::Dictionary)
    );
    assert_eq!(
        group.encodings.get("superseded"),
        Some(&HotColumnEncoding::BitPacked)
    );
    assert_eq!(
        group.encodings.get("epistemic_weight"),
        Some(&HotColumnEncoding::QuantizedU16)
    );
    assert!(group.encoded_columnar_bytes < group.unencoded_columnar_bytes);
    assert!(group.encoded_columnar_bytes < group.rowwise_property_bytes);

    let resolver = EpistemicResolver {
        reachable: (32..64).map(|idx| format!("shadow:{idx:02}")).collect(),
        text_scores: Vec::new(),
    };
    let result = execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "e".to_string(),
                relation: "epistemic".to_string(),
                predicates: vec![
                    Predicate::Range {
                        column: "epistemic_weight".to_string(),
                        lo: ScalarBound::Included(ScalarValue::F64(1.5)),
                        hi: ScalarBound::Unbounded,
                    },
                    Predicate::Equals {
                        column: "acceptance_status".to_string(),
                        value: text("accepted"),
                    },
                    Predicate::Equals {
                        column: "superseded".to_string(),
                        value: ScalarValue::Bool(false),
                    },
                    Predicate::Range {
                        column: "valid_from_ms".to_string(),
                        lo: ScalarBound::Unbounded,
                        hi: ScalarBound::Included(ScalarValue::I64(560)),
                    },
                    Predicate::Expand {
                        from: "shadow:origin".to_string(),
                        layers: vec!["NEAR".into()],
                        dir: Direction::Out,
                        mode: PredicateMode::Filter,
                    },
                ],
            }],
            projection: vec![Projection {
                alias: "e".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
        &resolver,
    )
    .unwrap();

    let ids = result
        .rows
        .iter()
        .filter_map(|row| row.get("e.id").and_then(ScalarValue::as_str))
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["shadow:48", "shadow:52", "shadow:56"]);
    assert_eq!(result.trace.full_relation_scans, 0);
    assert!(result.trace.used_roaring_bitmaps);
    assert!(result.trace.bitmap_intersections >= 4);
    assert!(result
        .trace
        .access_paths
        .iter()
        .any(|path| path.method == "expand"));
    assert!(result
        .trace
        .access_paths
        .iter()
        .filter(|path| path.method == "ordered")
        .all(|path| path.visited_rows < 64));
    let weight_trace = result
        .trace
        .columnar_groups
        .iter()
        .find(|trace| trace.column == "epistemic_weight")
        .unwrap();
    assert!(weight_trace.blocks_skipped > 0);
    assert!(weight_trace.encoded_columnar_bytes < weight_trace.rowwise_property_bytes);
}

#[test]
fn cascade_ranking_reads_epistemic_standing_from_hot_columnar_group() {
    let mut store = RelationalLens::new();
    store
        .upsert_row(shadow_row(
            "shadow:a", 1.0, 0.5, 0.5, 100, "accepted", false,
        ))
        .unwrap();
    store
        .upsert_row(shadow_row(
            "shadow:b", 1.5, 0.9, 0.5, 100, "accepted", false,
        ))
        .unwrap();
    store
        .upsert_row(shadow_row(
            "shadow:c",
            1.5,
            0.9,
            0.5,
            100,
            "contested",
            false,
        ))
        .unwrap();

    let resolver = EpistemicResolver {
        text_scores: vec![
            RankedRow {
                row_id: "shadow:a".to_string(),
                score: 1.0,
                layer: None,
            },
            RankedRow {
                row_id: "shadow:b".to_string(),
                score: 1.0,
                layer: None,
            },
            RankedRow {
                row_id: "shadow:c".to_string(),
                score: 1.0,
                layer: None,
            },
        ],
        ..EpistemicResolver::default()
    };
    let result = execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "e".to_string(),
                relation: "epistemic".to_string(),
                predicates: vec![Predicate::TextMatch {
                    column: "body".to_string(),
                    query: "alpha".to_string(),
                    mode: PredicateMode::Rank,
                }],
            }],
            projection: vec![Projection {
                alias: "e".to_string(),
                column: "id".to_string(),
            }],
            fusion: FusionPolicy::Cascade {
                rules: vec![RankingRule::EpistemicStatus, RankingRule::Text],
            },
            limit: Some(3),
            ..QueryIr::default()
        },
        &resolver,
    )
    .unwrap();

    let ids = result
        .rows
        .iter()
        .filter_map(|row| row.get("e.id").and_then(ScalarValue::as_str))
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["shadow:b", "shadow:a"]);
    assert!(result
        .trace
        .columnar_groups
        .iter()
        .any(|trace| trace.predicate == "cascade_epistemic_read"
            && trace.source == "hot_epistemic_column_group"
            && trace.rows_returned == 3));
}
