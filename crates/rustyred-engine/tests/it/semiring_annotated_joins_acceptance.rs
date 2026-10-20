use std::collections::BTreeMap;

use rustyred_thg_core::{
    compile_graphql_selection, execute_query, GraphqlJoinSelection, GraphqlSelection,
    JoinPredicate, Projection, ProvenanceSemiringKind, QueryIr, QueryRelation, RelationalLens,
    ResolvedRow, ScalarValue,
};

fn text(value: &str) -> ScalarValue {
    ScalarValue::String(value.to_string())
}

fn row(relation: &str, id: &str, values: &[(&str, ScalarValue)]) -> ResolvedRow {
    ResolvedRow::new(
        relation,
        id,
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    )
}

fn path_store() -> RelationalLens {
    let mut store = RelationalLens::new();
    for (id, join_key, target, confidence, cost) in [
        ("left:a", "k:a", "target:a", 0.9, 2.0),
        ("left:b", "k:b", "target:b", 0.4, 5.0),
    ] {
        store
            .upsert_row(row(
                "left_paths",
                id,
                &[
                    ("join_key", text(join_key)),
                    ("target", text(target)),
                    ("epistemic_weight", ScalarValue::F64(confidence)),
                    ("cost", ScalarValue::F64(cost)),
                ],
            ))
            .unwrap();
    }
    for (id, join_key, target, confidence, cost) in [
        ("right:a", "k:a", "target:a", 0.5, 7.0),
        ("right:b", "k:b", "target:b", 0.9, 1.0),
    ] {
        store
            .upsert_row(row(
                "right_paths",
                id,
                &[
                    ("join_key", text(join_key)),
                    ("target", text(target)),
                    ("epistemic_weight", ScalarValue::F64(confidence)),
                    ("cost", ScalarValue::F64(cost)),
                ],
            ))
            .unwrap();
    }
    store
}

fn joined_query(semiring: Option<ProvenanceSemiringKind>) -> QueryIr {
    QueryIr {
        relations: vec![
            QueryRelation {
                alias: "l".to_string(),
                relation: "left_paths".to_string(),
                predicates: Vec::new(),
            },
            QueryRelation {
                alias: "r".to_string(),
                relation: "right_paths".to_string(),
                predicates: Vec::new(),
            },
        ],
        joins: vec![JoinPredicate {
            left_alias: "l".to_string(),
            left_column: "join_key".to_string(),
            right_alias: "r".to_string(),
            right_column: "join_key".to_string(),
            kind: Default::default(),
        }],
        projection: vec![Projection {
            alias: "r".to_string(),
            column: "target".to_string(),
        }],
        semiring,
        ..QueryIr::default()
    }
}

#[test]
fn viterbi_scan_seeds_from_epistemic_weight_without_changing_unannotated_rows() {
    let mut store = RelationalLens::new();
    store
        .upsert_row(row(
            "shadow",
            "shadow:a",
            &[
                ("id", text("shadow:a")),
                ("epistemic_weight", ScalarValue::F64(0.75)),
            ],
        ))
        .unwrap();

    let unannotated = execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "s".to_string(),
                relation: "shadow".to_string(),
                predicates: Vec::new(),
            }],
            projection: vec![Projection {
                alias: "s".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();
    let annotated = execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "s".to_string(),
                relation: "shadow".to_string(),
                predicates: Vec::new(),
            }],
            projection: vec![Projection {
                alias: "s".to_string(),
                column: "id".to_string(),
            }],
            semiring: Some(ProvenanceSemiringKind::Viterbi),
            ..QueryIr::default()
        },
    )
    .unwrap();

    assert_eq!(unannotated.rows[0].values, annotated.rows[0].values);
    assert!(unannotated.rows[0].provenance.is_none());
    assert_eq!(
        annotated.rows[0].provenance.as_ref().map(|p| p.value),
        Some(0.75)
    );
}

#[test]
fn viterbi_join_products_annotations_and_orders_by_composed_confidence() {
    let store = path_store();
    let result = execute_query(
        &store,
        QueryIr {
            limit: Some(2),
            ..joined_query(Some(ProvenanceSemiringKind::Viterbi))
        },
    )
    .unwrap();

    let ids = result
        .rows
        .iter()
        .filter_map(|row| row.get("r.target").and_then(ScalarValue::as_str))
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["target:a", "target:b"]);
    assert_eq!(
        result.rows[0].provenance.as_ref().map(|p| p.value),
        Some(0.45)
    );
    assert_eq!(
        result.rows[1].provenance.as_ref().map(|p| p.value),
        Some(0.36000000000000004)
    );
    assert_eq!(result.trace.semiring.as_deref(), Some("viterbi"));
    assert_eq!(result.trace.derived_annotations.get("row:0"), Some(&0.45));
}

#[test]
fn duplicate_projection_merges_with_viterbi_max_and_tropical_min() {
    let mut store = RelationalLens::new();
    for (id, join_key, confidence, cost) in
        [("left:a", "k:a", 0.9, 2.0), ("left:b", "k:b", 0.5, 5.0)]
    {
        store
            .upsert_row(row(
                "left_paths",
                id,
                &[
                    ("join_key", text(join_key)),
                    ("epistemic_weight", ScalarValue::F64(confidence)),
                    ("cost", ScalarValue::F64(cost)),
                ],
            ))
            .unwrap();
    }
    for (id, join_key, confidence, cost) in
        [("right:a", "k:a", 0.8, 7.0), ("right:b", "k:b", 0.7, 1.0)]
    {
        store
            .upsert_row(row(
                "right_paths",
                id,
                &[
                    ("join_key", text(join_key)),
                    ("target", text("shared")),
                    ("epistemic_weight", ScalarValue::F64(confidence)),
                    ("cost", ScalarValue::F64(cost)),
                ],
            ))
            .unwrap();
    }

    let viterbi =
        execute_query(&store, joined_query(Some(ProvenanceSemiringKind::Viterbi))).unwrap();
    assert_eq!(viterbi.rows.len(), 1);
    assert_eq!(
        viterbi.rows[0].provenance.as_ref().map(|p| p.value),
        Some(0.7200000000000001)
    );

    let tropical =
        execute_query(&store, joined_query(Some(ProvenanceSemiringKind::Tropical))).unwrap();
    assert_eq!(tropical.rows.len(), 1);
    assert_eq!(
        tropical.rows[0].provenance.as_ref().map(|p| p.value),
        Some(6.0)
    );
    assert_eq!(
        tropical.trace.semiring_graphblas.as_deref(),
        Some("min_plus_fp64")
    );
}

#[test]
fn unannotated_join_shape_remains_unchanged() {
    let store = path_store();
    let result = execute_query(&store, joined_query(None)).unwrap();

    assert_eq!(result.rows.len(), 2);
    assert!(result.rows.iter().all(|row| row.provenance.is_none()));
    assert_eq!(result.trace.semiring, None);
    assert!(result.trace.derived_annotations.is_empty());
}

#[test]
fn graphql_compiled_selection_exposes_derived_provenance() {
    let store = path_store();
    let query = compile_graphql_selection(GraphqlSelection {
        relation: "left_paths".to_string(),
        alias: Some("l".to_string()),
        fields: Vec::new(),
        joins: vec![GraphqlJoinSelection {
            relation: "right_paths".to_string(),
            alias: Some("r".to_string()),
            left_column: "join_key".to_string(),
            right_column: "join_key".to_string(),
            fields: vec!["target".to_string()],
        }],
        limit: Some(1),
        semiring: Some(ProvenanceSemiringKind::Viterbi),
    });
    let result = execute_query(&store, query).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].get("r.target").and_then(ScalarValue::as_str),
        Some("target:a")
    );
    assert_eq!(
        result.rows[0].provenance.as_ref().map(|p| p.value),
        Some(0.45)
    );
    assert_eq!(result.trace.derived_annotations.get("row:0"), Some(&0.45));
}
