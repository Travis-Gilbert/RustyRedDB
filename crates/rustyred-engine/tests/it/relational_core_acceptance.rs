//! Spec-anchored acceptance suite for SPEC-RUSTYRED-RELATIONAL-CORE.
//!
//! The implementation (the access-method seam, the cost-based planner, the
//! native relational core + catalog, the TileDB-format cold fragments, the
//! working-layer log) is exercised here through the PUBLIC crate boundary, one
//! test per named acceptance criterion from section 12 of the spec. This is the
//! definition-of-done artifact: each `acceptance_NN_*` test quotes the criterion
//! it proves, so the spec and the code stay reconciled.
//!
//! Two of the ten criteria live in the `rustyred-thg-memory` crate (it depends
//! on this one, so its tests cannot live here without a dependency cycle) and
//! are proven there:
//!   - #1 (eviction/rehydration make no sqlx call or network hop; the cold tier
//!     is a native `InMemoryColdIndex`/`DiskColdIndex`, never `PostgresColdIndex`)
//!     -> `rustyred-thg-memory/tests/storage_spine_acceptance.rs`
//!     (`decayed_node_is_evicted_yet_survives_rehydration`).
//!   - #6 (an aged row is dropped from the hot tier; a contradicted fact is
//!     marked invalid with its validity interval preserved and still queryable
//!     as history, not overwritten) -> `evict_decayed` in the same suite plus
//!     `invalidate_on_contradiction` /
//!     `contradiction_invalidates_old_functional_edge_without_deleting_it` in
//!     `rustyred-thg-memory/src/lib.rs`.

use std::collections::BTreeMap;

use rustyred_thg_core::cold_fragments::{
    ColdFragment, ColdFragmentStore, CompressionFilter, PromotionPolicy,
};
use rustyred_thg_core::working_log::WorkingLog;
use rustyred_thg_core::{
    compile_graphql_selection, execute_query, EdgeRecord, GraphSnapshot, GraphqlJoinSelection,
    GraphqlSelection, JoinPredicate, NativeBillingAccountRecord, NativeCatalog,
    NativeProjectRecord, NativeTenantRecord, NodeRecord, Predicate, Projection, QueryIr,
    QueryRelation, RelationalLens, ResolvedRow, ScalarBound, ScalarValue,
};
use serde_json::json;

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

fn text(value: &str) -> ScalarValue {
    ScalarValue::String(value.to_string())
}

// --- #2 -------------------------------------------------------------------
// "A query filtering a relation by a scalar predicate and a time-range
// predicate returns only matching rows, consulting the ordered method and the
// time method, without a full-relation or full-graph scan."
#[test]
fn acceptance_02_scalar_and_time_range_use_ordered_and_time_methods_without_full_scan() {
    let mut store = RelationalLens::new();
    store
        .upsert_row(row(
            "memory",
            "m1",
            &[("kind", text("episode")), ("t_ms", ScalarValue::I64(5))],
        ))
        .unwrap();
    store
        .upsert_row(row(
            "memory",
            "m2",
            &[("kind", text("episode")), ("t_ms", ScalarValue::I64(15))],
        ))
        .unwrap();
    store
        .upsert_row(row(
            "memory",
            "m3",
            &[("kind", text("note")), ("t_ms", ScalarValue::I64(15))],
        ))
        .unwrap();

    let result = execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "m".to_string(),
                relation: "memory".to_string(),
                predicates: vec![
                    Predicate::Equals {
                        column: "kind".to_string(),
                        value: text("episode"),
                    },
                    Predicate::TimeRange {
                        column: "t_ms".to_string(),
                        lo_ms: 10,
                        hi_ms: 20,
                    },
                ],
            }],
            projection: vec![Projection {
                alias: "m".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();

    // Only m2 satisfies kind=episode AND t in [10,20].
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].get("m.id"), Some(&text("m2")));
    // No full-relation scan: both predicates were served by an access method.
    assert_eq!(result.trace.full_relation_scans, 0);
    let methods: Vec<&str> = result
        .trace
        .access_paths
        .iter()
        .map(|path| path.method.as_str())
        .collect();
    assert!(methods.contains(&"ordered"), "scalar -> ordered method");
    assert!(methods.contains(&"time_series"), "time -> time method");
}

// --- #3 -------------------------------------------------------------------
// "A join between a content relation and an epistemic relation on a key returns
// the many-to-many association through the planner's join, not through
// one-to-one shadow edges."
#[test]
fn acceptance_03_content_epistemic_many_to_many_join_through_planner() {
    let mut store = RelationalLens::new();
    store
        .upsert_row(row("content", "c1", &[("content_key", text("doc:1"))]))
        .unwrap();
    store
        .upsert_row(row(
            "epistemic",
            "e1",
            &[("content_key", text("doc:1")), ("claim", text("supports"))],
        ))
        .unwrap();
    store
        .upsert_row(row(
            "epistemic",
            "e2",
            &[("content_key", text("doc:1")), ("claim", text("undercuts"))],
        ))
        .unwrap();

    let result = execute_query(
        &store,
        QueryIr {
            relations: vec![
                QueryRelation {
                    alias: "c".to_string(),
                    relation: "content".to_string(),
                    predicates: Vec::new(),
                },
                QueryRelation {
                    alias: "e".to_string(),
                    relation: "epistemic".to_string(),
                    predicates: Vec::new(),
                },
            ],
            joins: vec![JoinPredicate {
                left_alias: "c".to_string(),
                left_column: "content_key".to_string(),
                right_alias: "e".to_string(),
                right_column: "content_key".to_string(),
                kind: Default::default(),
            }],
            projection: vec![Projection {
                alias: "e".to_string(),
                column: "claim".to_string(),
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();

    // One content row associates with two epistemic rows: a many-to-many result
    // resolved by the planner's join, not a single shadow edge.
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
    let claims: Vec<_> = result
        .rows
        .iter()
        .filter_map(|r| r.get("e.claim"))
        .cloned()
        .collect();
    assert!(claims.contains(&text("supports")));
    assert!(claims.contains(&text("undercuts")));
}

#[test]
fn acceptance_03a_nodes_edges_nodes_join_from_graph_lens() {
    let snapshot = GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 1,
        nodes: vec![
            NodeRecord::new("node:a", vec!["Person"], json!({ "name": "Alice" })),
            NodeRecord::new("node:b", vec!["Person"], json!({ "name": "Bob" })),
            NodeRecord::new("node:c", vec!["Person"], json!({ "name": "Cara" })),
            NodeRecord::new("node:d", vec!["Person"], json!({ "name": "Drew" })),
        ],
        edges: vec![
            EdgeRecord::new(
                "edge:1",
                "node:a",
                "KNOWS",
                "node:b",
                json!({ "weight": "10" }),
            ),
            EdgeRecord::new(
                "edge:2",
                "node:a",
                "KNOWS",
                "node:c",
                json!({ "weight": "7" }),
            ),
        ],
    };

    let store = RelationalLens::bind(&snapshot).unwrap();

    let result = execute_query(
        &store,
        QueryIr {
            relations: vec![
                QueryRelation {
                    alias: "source".to_string(),
                    relation: "nodes".to_string(),
                    predicates: Vec::new(),
                },
                QueryRelation {
                    alias: "edge".to_string(),
                    relation: "edges".to_string(),
                    predicates: Vec::new(),
                },
                QueryRelation {
                    alias: "target".to_string(),
                    relation: "nodes".to_string(),
                    predicates: Vec::new(),
                },
            ],
            joins: vec![
                JoinPredicate {
                    left_alias: "source".to_string(),
                    left_column: "id".to_string(),
                    right_alias: "edge".to_string(),
                    right_column: "from_id".to_string(),
                    kind: Default::default(),
                },
                JoinPredicate {
                    left_alias: "edge".to_string(),
                    left_column: "to_id".to_string(),
                    right_alias: "target".to_string(),
                    right_column: "id".to_string(),
                    kind: Default::default(),
                },
            ],
            projection: vec![
                Projection {
                    alias: "source".to_string(),
                    column: "id".to_string(),
                },
                Projection {
                    alias: "edge".to_string(),
                    column: "id".to_string(),
                },
                Projection {
                    alias: "target".to_string(),
                    column: "id".to_string(),
                },
            ],
            ..QueryIr::default()
        },
    )
    .unwrap();

    // Node a has two outgoing edges and should produce two joined rows through the planner join.
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
    let mut targets = result
        .rows
        .iter()
        .filter_map(|row| row.get("target.id").and_then(ScalarValue::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    targets.sort();
    assert_eq!(targets, vec!["node:b".to_string(), "node:c".to_string()]);
}

// --- #4 -------------------------------------------------------------------
// "The planner ranks access paths by cost and intersects multiple methods for a
// conjunctive predicate with roaring bitmaps."
#[test]
fn acceptance_04_cost_ranked_paths_intersect_with_roaring_bitmaps() {
    let mut store = RelationalLens::new();
    for (id, kind, t) in [
        ("m1", "episode", 5),
        ("m2", "episode", 15),
        ("m3", "note", 15),
    ] {
        store
            .upsert_row(row(
                "memory",
                id,
                &[("kind", text(kind)), ("t_ms", ScalarValue::I64(t))],
            ))
            .unwrap();
    }

    let result = execute_query(
        &store,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "m".to_string(),
                relation: "memory".to_string(),
                predicates: vec![
                    Predicate::Equals {
                        column: "kind".to_string(),
                        value: text("episode"),
                    },
                    Predicate::TimeRange {
                        column: "t_ms".to_string(),
                        lo_ms: 10,
                        hi_ms: 20,
                    },
                ],
            }],
            ..QueryIr::default()
        },
    )
    .unwrap();

    // Two single-method bitmaps intersected into one conjunctive result.
    assert!(result.trace.used_roaring_bitmaps);
    assert_eq!(result.trace.bitmap_intersections, 1);
    // Cost ranking happened: every chosen path carries a cost estimate.
    assert!(!result.trace.access_paths.is_empty());
    assert!(result
        .trace
        .access_paths
        .iter()
        .all(|path| path.est_work >= 0.0 && path.est_rows >= 0.0));
}

// --- #5 -------------------------------------------------------------------
// "A working row recalled past the promotion threshold appears as a cold
// fragment and remains findable by every access method that covered it."
#[test]
fn acceptance_05_recalled_row_promotes_to_cold_fragment_and_stays_findable() {
    let working = row(
        "memory",
        "m1",
        &[("t_ms", ScalarValue::I64(10)), ("kind", text("episode"))],
    );
    let mut fragments = ColdFragmentStore::new();
    let policy = PromotionPolicy::new(2, vec!["t_ms".to_string(), "kind".to_string()]);

    // Below threshold: stays hot.
    let not_yet = policy.promote_if_recalled(&working, 1, &mut fragments);
    assert!(!not_yet.promoted);

    // Past threshold: a cold fragment is written, and the hot side keeps a
    // lightweight handle carrying the indexed columns + the residency pointer.
    let promoted = policy.promote_if_recalled(&working, 2, &mut fragments);
    assert!(promoted.promoted);
    assert!(promoted.fragment_id.is_some());
    assert_eq!(
        promoted.hot_handle.values.get("kind"),
        Some(&text("episode"))
    );
    assert!(promoted.hot_handle.values.contains_key("cold_fragment_id"));

    // Still findable: a range query over the cold fragment returns the row.
    let found = fragments
        .range_query(
            "memory",
            "t_ms",
            ScalarBound::Included(ScalarValue::I64(5)),
            ScalarBound::Included(ScalarValue::I64(15)),
        )
        .unwrap();
    assert_eq!(found.rows, vec!["m1".to_string()]);
}

// --- #7 -------------------------------------------------------------------
// "Cold fragments are stored columnar with per-attribute compression, and a
// range query skips fragments whose zone-map bounds exclude the range."
#[test]
fn acceptance_07_cold_fragments_are_columnar_compressed_and_zone_map_pruned() {
    let rows = vec![
        row(
            "memory",
            "m1",
            &[("t_ms", ScalarValue::I64(10)), ("kind", text("episode"))],
        ),
        row(
            "memory",
            "m2",
            &[("t_ms", ScalarValue::I64(20)), ("kind", text("episode"))],
        ),
    ];
    let fragment = ColdFragment::from_rows("frag:1", "memory", &rows);

    // Per-attribute columnar layout with attribute-appropriate compression:
    // timestamps -> double-delta, low-cardinality text -> run-length.
    let t_column = fragment.columns.get("t_ms").unwrap();
    assert_eq!(t_column.values.len(), 2);
    assert!(t_column.filters.contains(&CompressionFilter::DoubleDelta));
    assert!(fragment
        .columns
        .get("kind")
        .unwrap()
        .filters
        .contains(&CompressionFilter::RunLength));

    // Zone-map prune: a range entirely above the fragment's max is skipped.
    let skipped = fragment
        .range_query(
            "t_ms",
            ScalarBound::Included(ScalarValue::I64(100)),
            ScalarBound::Included(ScalarValue::I64(200)),
        )
        .unwrap();
    assert_eq!(skipped.stats.fragments_skipped, 1);
    assert!(skipped.rows.is_empty());

    // A range that overlaps the zone map visits the fragment and returns hits.
    let hit = fragment
        .range_query(
            "t_ms",
            ScalarBound::Included(ScalarValue::I64(15)),
            ScalarBound::Included(ScalarValue::I64(25)),
        )
        .unwrap();
    assert_eq!(hit.stats.fragments_visited, 1);
    assert_eq!(hit.rows, vec!["m2".to_string()]);
}

// --- #8 -------------------------------------------------------------------
// "A subscriber to the working-layer log receives episode and access events in
// order and can resume from a cursor."
#[test]
fn acceptance_08_working_log_subscriber_resumes_in_order_from_cursor() {
    let mut log = WorkingLog::new();
    let first = log.append_episode("episode:1", json!({ "text": "hello" }));
    log.append_access("episode:1");
    log.append_mutation("episode:1", json!({ "field": "status" }));

    // Resume after the first cursor: the next two events, in cursor order.
    let after_first = log.subscribe_after(first.cursor, 0);
    assert_eq!(after_first.len(), 2);
    assert_eq!(after_first[0].cursor, first.cursor + 1);
    assert_eq!(after_first[1].cursor, first.cursor + 2);
    assert!(after_first.windows(2).all(|w| w[0].cursor < w[1].cursor));

    // A bounded subscribe from the start respects the limit (cursor paging).
    let limited = log.subscribe_after(0, 2);
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0].cursor, first.cursor);
}

// --- #9 -------------------------------------------------------------------
// "The GraphQL endpoint resolves a selection spanning more than one relation in
// a single planner pass."
#[test]
fn acceptance_09_graphql_selection_spanning_relations_resolves_in_one_planner_pass() {
    let mut store = RelationalLens::new();
    store
        .upsert_row(row("content", "c1", &[("key", text("k1"))]))
        .unwrap();
    store
        .upsert_row(row(
            "epistemic",
            "e1",
            &[("content_key", text("k1")), ("claim", text("grounded"))],
        ))
        .unwrap();

    // A GraphQL selection that reaches across content -> epistemic compiles to
    // one QueryIr and executes in a single planner pass.
    let query = compile_graphql_selection(GraphqlSelection {
        relation: "content".to_string(),
        alias: Some("content".to_string()),
        fields: vec!["key".to_string()],
        joins: vec![GraphqlJoinSelection {
            relation: "epistemic".to_string(),
            alias: Some("epistemic".to_string()),
            left_column: "key".to_string(),
            right_column: "content_key".to_string(),
            fields: vec!["claim".to_string()],
        }],
        limit: None,
        semiring: None,
    });
    let result = execute_query(&store, query).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].get("epistemic.claim"),
        Some(&text("grounded"))
    );
    assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
}

// --- #10 ------------------------------------------------------------------
// "The administrative tables enforce the foreign-key and uniqueness constraints
// from the current catalog DDL."
#[test]
fn acceptance_10_native_catalog_enforces_foreign_key_and_uniqueness() {
    let mut catalog = NativeCatalog::new();

    // Foreign key: a project for a non-existent tenant is rejected.
    let orphan = catalog.upsert_project(NativeProjectRecord {
        tenant_id: "tenant:missing".to_string(),
        project_slug: "theorem".to_string(),
        display_name: None,
    });
    assert_eq!(orphan.unwrap_err().code, "catalog_foreign_key_violation");

    catalog
        .upsert_tenant(NativeTenantRecord {
            tenant_id: "tenant:a".to_string(),
            slug: "a".to_string(),
            display_name: Some("Tenant A".to_string()),
        })
        .unwrap();
    catalog
        .upsert_project(NativeProjectRecord {
            tenant_id: "tenant:a".to_string(),
            project_slug: "theorem".to_string(),
            display_name: None,
        })
        .unwrap();
    catalog
        .upsert_billing_account(NativeBillingAccountRecord {
            tenant_id: "tenant:a".to_string(),
            plan: "pro".to_string(),
            status: "active".to_string(),
        })
        .unwrap();

    // Uniqueness: a second tenant cannot reuse an existing slug.
    let dup = catalog.upsert_tenant(NativeTenantRecord {
        tenant_id: "tenant:b".to_string(),
        slug: "a".to_string(),
        display_name: None,
    });
    assert_eq!(dup.unwrap_err().code, "catalog_unique_violation");
}
