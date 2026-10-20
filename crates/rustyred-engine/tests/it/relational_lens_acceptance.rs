use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;

use rustyred_thg_core::index::TrigramAccessMethod;
use rustyred_thg_core::{
    AccessMethod, AccessMethodRegistry, EdgeRecord, GraphRead, InMemoryGraphStore,
    MemoryDocumentLens, MemoryDocumentQuery, NativeAuthPrincipalRecord, NativeBillingAccountRecord,
    NativeCatalog, NativeProjectRecord, NativeTenantRecord, NodeRecord, OrderedAccessMethod,
    Predicate, PredicateMode, RelationSchema, RelationalLens, ResolvedRow, RowChange,
    RowChangeKind, ScalarBound, ScalarValue, TimeSeriesAccessMethod,
};

fn ordered_name_rows(lens: &RelationalLens, relation: &str, name: &str) -> Vec<String> {
    let predicate = Predicate::Equals {
        column: "name".to_string(),
        value: ScalarValue::String(name.to_string()),
    };
    lens.access_methods()
        .methods()
        .find(|method| method.name() == "ordered")
        .unwrap()
        .scan(relation, &predicate)
        .unwrap()
        .rows
}

struct LegacyMaterializedRowsFixture {
    row_heap_surrogate: InMemoryGraphStore,
    row_copies: usize,
}

impl LegacyMaterializedRowsFixture {
    fn from_graph(graph: &InMemoryGraphStore) -> Self {
        let mut row_heap_surrogate = InMemoryGraphStore::new();
        let mut row_copies = 0usize;

        for node in graph.live_node_records() {
            let relations = std::iter::once("nodes".to_string()).chain(node.labels.iter().cloned());
            for relation in relations {
                row_heap_surrogate
                    .upsert_node(NodeRecord::new(
                        format!("legacy-row:{relation}:{}", node.id),
                        ["LegacyMaterializedRelationalRow"],
                        json!({
                            "values": node.properties.clone(),
                            "properties": node.properties.clone(),
                        }),
                    ))
                    .unwrap();
                row_copies += 1;
            }
        }

        Self {
            row_heap_surrogate,
            row_copies,
        }
    }

    fn attributed_resident_bytes(&self) -> usize {
        self.row_heap_surrogate
            .memory_attribution()
            .accounted_capacity_lower_bound_bytes
    }
}

#[test]
fn label_lens_matches_materialized_order_and_tracks_graph_mutation() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "node:z",
            ["Person", "Researcher", "Writer"],
            json!({"name": "Ada", "rank": 2}),
        ))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new(
            "node:a",
            ["Person"],
            json!({"name": "Grace", "rank": 1}),
        ))
        .unwrap();

    let lens = RelationalLens::bind(&graph).unwrap();
    let person = lens.relation("person").unwrap();
    assert_eq!(
        person.row_ids(),
        vec!["node:a".to_string(), "node:z".to_string()]
    );
    assert_eq!(
        lens.relation("researcher").unwrap().row_ids(),
        vec!["node:z".to_string()]
    );
    assert_eq!(lens.graph().stats().nodes_total, 2);
    assert_eq!(
        person.get("node:z").unwrap().values.get("name"),
        Some(&ScalarValue::String("Ada".to_string()))
    );
    drop(person);
    drop(lens);

    let mut node = graph.get_node_record("node:z").unwrap();
    node.properties["name"] = json!("Augusta");
    graph.upsert_node(node).unwrap();

    let lens = RelationalLens::bind(&graph).unwrap();
    assert_eq!(
        lens.relation("person")
            .unwrap()
            .get("node:z")
            .unwrap()
            .values
            .get("name"),
        Some(&ScalarValue::String("Augusta".to_string()))
    );
}

#[test]
fn multi_label_fixture_reports_reclaimed_resident_bytes() {
    const NODE_COUNT: usize = 128;
    const LABELS_PER_NODE: usize = 3;
    const LABEL_CARDINALITY: usize = 3;

    let properties = json!({
        "name": "relational-lens-fixture",
        "rank": 7,
        "summary": "A fixed property payload makes the mean serialized size reproducible.",
    });
    let mean_property_json_bytes = serde_json::to_vec(&properties).unwrap().len();
    let mut graph = InMemoryGraphStore::new();
    for index in 0..NODE_COUNT {
        graph
            .upsert_node(NodeRecord::new(
                format!("fixture:{index:04}"),
                ["Person", "Researcher", "Writer"],
                properties.clone(),
            ))
            .unwrap();
    }

    let after = graph.memory_attribution();
    let legacy = LegacyMaterializedRowsFixture::from_graph(&graph);
    let legacy_row_bytes = legacy.attributed_resident_bytes();
    let before_resident_bytes = after
        .accounted_capacity_lower_bound_bytes
        .saturating_add(legacy_row_bytes);
    let after_resident_bytes = after.accounted_capacity_lower_bound_bytes;
    let reclaimed_resident_bytes = before_resident_bytes.saturating_sub(after_resident_bytes);

    assert_eq!(after.node_records, NODE_COUNT);
    assert_eq!(after.live_nodes, NODE_COUNT);
    assert_eq!(
        graph.get_node_record("fixture:0000").unwrap().labels.len(),
        3
    );
    assert_eq!(legacy.row_copies, NODE_COUNT * (LABELS_PER_NODE + 1));
    assert_eq!(reclaimed_resident_bytes, legacy_row_bytes);
    assert!(before_resident_bytes > after_resident_bytes);

    println!(
        "relational_lens_resident_fixture \
         node_count={NODE_COUNT} labels_per_node={LABELS_PER_NODE} \
         label_cardinality={LABEL_CARDINALITY} \
         mean_property_json_bytes={mean_property_json_bytes} \
         legacy_row_copies={} before_resident_bytes={before_resident_bytes} \
         after_resident_bytes={after_resident_bytes} \
         reclaimed_resident_bytes={reclaimed_resident_bytes}",
        legacy.row_copies,
    );
}

#[test]
fn relation_local_aliases_never_replace_physical_graph_identity() {
    let mut lens = RelationalLens::new();
    lens.upsert_row(ResolvedRow::new(
        "alpha",
        "shared",
        BTreeMap::from([(
            "name".to_string(),
            ScalarValue::String("alpha row".to_string()),
        )]),
    ))
    .unwrap();
    lens.upsert_row(ResolvedRow::new(
        "beta",
        "shared",
        BTreeMap::from([(
            "name".to_string(),
            ScalarValue::String("beta row".to_string()),
        )]),
    ))
    .unwrap();

    let nodes = lens.relation("nodes").unwrap().rows().collect::<Vec<_>>();
    assert_eq!(nodes.len(), 2);
    assert_eq!(
        nodes
            .iter()
            .map(|row| row.id.clone())
            .collect::<BTreeSet<_>>()
            .len(),
        2
    );
    assert_eq!(
        nodes
            .iter()
            .filter_map(|row| {
                row.values
                    .get("name")
                    .and_then(ScalarValue::as_str)
                    .map(str::to_string)
            })
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["alpha row".to_string(), "beta row".to_string()])
    );

    let beta = lens.relation("beta").unwrap();
    let beta_ids = beta.row_ids();
    assert_eq!(beta_ids.len(), 1);
    assert_ne!(beta_ids[0], "shared");
    assert_eq!(
        beta.rows().next().unwrap().values.get("name"),
        Some(&ScalarValue::String("beta row".to_string()))
    );
    assert_eq!(
        beta.get("shared").unwrap().values.get("name"),
        Some(&ScalarValue::String("beta row".to_string()))
    );
    drop(beta);
    assert_eq!(ordered_name_rows(&lens, "beta", "beta row"), beta_ids);

    lens.upsert_row(ResolvedRow::new(
        "beta",
        "shared",
        BTreeMap::from([(
            "name".to_string(),
            ScalarValue::String("beta updated".to_string()),
        )]),
    ))
    .unwrap();
    assert!(ordered_name_rows(&lens, "beta", "beta row").is_empty());
    assert_eq!(ordered_name_rows(&lens, "beta", "beta updated"), beta_ids);

    assert!(lens.delete_row("beta", "shared").unwrap());
    assert!(ordered_name_rows(&lens, "beta", "beta updated").is_empty());
}

#[test]
fn normalized_relations_union_case_colliding_graph_indexes() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new("lower", ["person"], json!({})))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new("upper", ["Person"], json!({})))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "edge:lower",
            "lower",
            "knows",
            "upper",
            json!({}),
        ))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "edge:upper",
            "upper",
            "KNOWS",
            "lower",
            json!({}),
        ))
        .unwrap();

    let lens = RelationalLens::bind(&graph).unwrap();
    assert_eq!(
        lens.relation("person").unwrap().row_ids(),
        vec!["lower".to_string(), "upper".to_string()]
    );
    assert_eq!(
        lens.relation("knows").unwrap().row_ids(),
        vec!["edge:lower".to_string(), "edge:upper".to_string()]
    );
}

#[test]
fn colliding_label_and_edge_type_relations_are_namespaced_by_source() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new("person", ["Person"], json!({})))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new("other", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "edge:person",
            "person",
            "PERSON",
            "other",
            json!({}),
        ))
        .unwrap();

    let mut lens = RelationalLens::from_graph(graph).unwrap();
    assert!(lens.relation("person").is_none());
    assert_eq!(
        lens.relation("label:person").unwrap().row_ids(),
        vec!["person".to_string()]
    );
    assert_eq!(
        lens.relation("edge:person").unwrap().row_ids(),
        vec!["edge:person".to_string()]
    );

    assert!(lens.delete_row("edge:person", "edge:person").unwrap());
    assert!(lens.graph().get_edge("edge:person").is_none());
    assert!(lens.graph().get_node("person").is_some());
    assert!(lens.graph().verify().ok);
}

#[test]
fn normalized_relation_alias_resolves_original_case_encoded_id() {
    let mut lens = RelationalLens::new();
    lens.graph_mut()
        .upsert_node(NodeRecord::new("shared", ["Other"], json!({})))
        .unwrap();
    lens.upsert_row(ResolvedRow::new(
        "Person",
        "shared",
        BTreeMap::from([(
            "name".to_string(),
            ScalarValue::String("encoded row".to_string()),
        )]),
    ))
    .unwrap();
    assert_eq!(lens.relation("person").unwrap().row_ids().len(), 1);
    assert_eq!(lens.relation("PERSON").unwrap().row_ids().len(), 1);

    let mut rebound = RelationalLens::from_graph(lens.graph().clone()).unwrap();
    let person = rebound.relation("person").unwrap();
    let physical_ids = person.row_ids();
    assert_eq!(physical_ids.len(), 1);
    assert_eq!(person.get("shared").unwrap().id, physical_ids[0]);
    drop(person);

    rebound
        .upsert_row(ResolvedRow::new(
            "person",
            "shared",
            BTreeMap::from([(
                "name".to_string(),
                ScalarValue::String("updated row".to_string()),
            )]),
        ))
        .unwrap();
    assert_eq!(rebound.graph().stats().nodes_total, 2);
    assert_eq!(rebound.relation("person").unwrap().row_ids(), physical_ids);
    assert_eq!(
        rebound
            .relation("person")
            .unwrap()
            .get("shared")
            .unwrap()
            .values
            .get("name"),
        Some(&ScalarValue::String("updated row".to_string()))
    );

    assert!(rebound.rename_relation("person", "human").unwrap());
    assert_eq!(rebound.graph().stats().nodes_total, 2);
    let human = rebound.relation("human").unwrap();
    assert_eq!(human.row_ids(), physical_ids);
    assert_eq!(
        human.get("shared").unwrap().values.get("name"),
        Some(&ScalarValue::String("updated row".to_string()))
    );
}

#[test]
fn reserved_relation_names_do_not_cross_contaminate_builtin_projections() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "reserved:a",
            ["Edges"],
            json!({ "label_only": true }),
        ))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new("reserved:b", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "reserved:edge",
            "reserved:a",
            "Nodes",
            "reserved:b",
            json!({ "edge_only": true }),
        ))
        .unwrap();

    let lens = RelationalLens::from_graph(graph).unwrap();
    let nodes = lens.relation("NODES").unwrap();
    assert_eq!(
        nodes.row_ids(),
        vec!["reserved:a".to_string(), "reserved:b".to_string()]
    );
    assert!(!nodes
        .schema
        .columns
        .iter()
        .any(|column| column.name == "edge_only"));
    let edges = lens.relation("edges").unwrap();
    assert_eq!(edges.row_ids(), vec!["reserved:edge".to_string()]);
    assert!(!edges
        .schema
        .columns
        .iter()
        .any(|column| column.name == "label_only"));
}

#[test]
fn direct_graph_mutation_refreshes_new_relation_definitions_on_drop() {
    let mut lens = RelationalLens::new();
    lens.create_relation(RelationSchema {
        relation: "declared_empty".to_string(),
        columns: vec![rustyred_thg_core::ColumnSchema {
            name: "declared_value".to_string(),
            nullable: false,
            indexed: false,
        }],
    });

    {
        let mut graph = lens.graph_mut();
        graph
            .upsert_node(NodeRecord::new(
                "new:a",
                ["NewLabel"],
                json!({ "new_value": "a" }),
            ))
            .unwrap();
        graph
            .upsert_node(NodeRecord::new(
                "new:b",
                ["NewLabel"],
                json!({ "new_value": "b" }),
            ))
            .unwrap();
        graph
            .upsert_edge(EdgeRecord::new(
                "edge:new",
                "new:a",
                "NEW_EDGE_TYPE",
                "new:b",
                json!({ "weight": 1 }),
            ))
            .unwrap();
    }

    assert_eq!(
        lens.relation("newlabel").unwrap().row_ids(),
        vec!["new:a".to_string(), "new:b".to_string()]
    );
    assert_eq!(
        lens.relation("new_edge_type").unwrap().row_ids(),
        vec!["edge:new".to_string()]
    );
    assert!(lens.relation("declared_empty").is_some());
}

#[test]
fn relation_rename_preserves_canonical_multi_label_identity() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "shared",
            ["Person", "Other"],
            json!({"name": "shared row"}),
        ))
        .unwrap();
    let mut lens = RelationalLens::from_graph(graph).unwrap();

    assert!(lens.rename_relation("person", "human").unwrap());
    assert_eq!(lens.graph().stats().nodes_total, 1);
    assert_eq!(
        lens.relation("human").unwrap().row_ids(),
        vec!["shared".to_string()]
    );
    assert_eq!(
        lens.relation("other").unwrap().row_ids(),
        vec!["shared".to_string()]
    );
    assert!(lens.relation("person").is_none());
}

#[test]
fn relation_rename_merges_existing_destination_schema() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "person-row",
            ["Person"],
            json!({"source_only": "source"}),
        ))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new(
            "human-row",
            ["human"],
            json!({"destination_only": "destination"}),
        ))
        .unwrap();
    let mut lens = RelationalLens::from_graph(graph).unwrap();

    assert!(lens.rename_relation("person", "human").unwrap());
    let human = lens.relation("human").unwrap();
    assert_eq!(
        human.row_ids(),
        vec!["human-row".to_string(), "person-row".to_string()]
    );
    assert_eq!(
        human
            .schema
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "content_hash",
            "destination_only",
            "id",
            "label",
            "source_only",
        ])
    );
    assert_eq!(lens.graph().stats().nodes_total, 2);
}

#[test]
fn relation_rename_rejects_case_variant_non_label_destination_before_mutation() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "person-row",
            ["Person"],
            json!({"name": "person"}),
        ))
        .unwrap();
    let mut lens = RelationalLens::from_graph(graph).unwrap();
    let graph_version = lens.graph().stats().version;

    let error = lens.rename_relation("person", "Edges").unwrap_err();
    assert_eq!(error.code, "relational_rename_source_conflict");
    assert_eq!(lens.graph().stats().version, graph_version);
    assert_eq!(
        lens.relation("person").unwrap().row_ids(),
        vec!["person-row".to_string()]
    );
    assert!(lens
        .graph()
        .get_node("person-row")
        .unwrap()
        .labels
        .iter()
        .any(|label| label == "Person"));
}

#[test]
fn relation_rename_rejects_graph_backed_edges_without_mutation() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new("node:a", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new("node:b", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "RELATED_TO",
            "node:b",
            json!({}),
        ))
        .unwrap();
    let mut lens = RelationalLens::from_graph(graph).unwrap();
    let graph_version = lens.graph().stats().version;

    let error = lens.rename_relation("edges", "archived_edges").unwrap_err();
    assert_eq!(error.code, "relational_rename_unsupported_source");
    assert_eq!(lens.graph().stats().version, graph_version);
    assert!(lens.graph().get_edge("edge:ab").is_some());
    assert!(lens.relation("edges").unwrap().get("edge:ab").is_some());
    assert!(lens.relation("archived_edges").is_none());
}

#[test]
fn dropping_builtin_nodes_is_rejected_before_mutation() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new("node:a", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new("node:b", ["Entity"], json!({})))
        .unwrap();
    graph
        .upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "RELATED_TO",
            "node:b",
            json!({}),
        ))
        .unwrap();
    let mut lens = RelationalLens::from_graph(graph).unwrap();
    let graph_version = lens.graph().stats().version;

    let error = lens.drop_relation("NODES").unwrap_err();
    assert_eq!(error.code, "relational_drop_unsupported_source");
    assert_eq!(lens.graph().stats().version, graph_version);
    assert!(lens.graph().get_node("node:a").is_some());
    assert!(lens.graph().get_node("node:b").is_some());
    assert!(lens.graph().get_edge("edge:ab").is_some());
    assert!(lens.graph().verify().ok);
}

#[test]
fn relational_node_removal_cascades_incident_edges() {
    let graph_with_edge = || {
        let mut graph = InMemoryGraphStore::new();
        graph
            .upsert_node(NodeRecord::new("person", ["Person"], json!({})))
            .unwrap();
        graph
            .upsert_node(NodeRecord::new("other", ["Entity"], json!({})))
            .unwrap();
        graph
            .upsert_edge(EdgeRecord::new(
                "edge:person-other",
                "person",
                "KNOWS",
                "other",
                json!({}),
            ))
            .unwrap();
        graph
    };

    let mut direct = RelationalLens::from_graph(graph_with_edge()).unwrap();
    assert!(direct.delete_row("nodes", "person").unwrap());
    assert!(direct.graph().get_node("person").is_none());
    assert!(direct.graph().get_edge("edge:person-other").is_none());
    assert!(direct.graph().verify().ok);

    let mut dropped = RelationalLens::from_graph(graph_with_edge()).unwrap();
    assert!(dropped.drop_relation("person").unwrap());
    assert!(dropped.graph().get_node("person").is_none());
    assert!(dropped.graph().get_edge("edge:person-other").is_none());
    assert!(dropped.graph().verify().ok);
}

#[test]
fn native_catalog_is_graph_backed() {
    let mut catalog = NativeCatalog::new();
    catalog
        .upsert_tenant(NativeTenantRecord {
            tenant_id: "tenant:a".to_string(),
            slug: "alpha".to_string(),
            display_name: Some("Alpha".to_string()),
        })
        .unwrap();
    catalog
        .upsert_project(NativeProjectRecord {
            tenant_id: "tenant:a".to_string(),
            project_slug: "theorem".to_string(),
            display_name: Some("Theorem".to_string()),
        })
        .unwrap();
    catalog
        .upsert_billing_account(NativeBillingAccountRecord {
            tenant_id: "tenant:a".to_string(),
            plan: "pro".to_string(),
            status: "active".to_string(),
        })
        .unwrap();
    catalog
        .upsert_auth_principal(NativeAuthPrincipalRecord {
            principal_id: "principal:1".to_string(),
            tenant_id: "tenant:a".to_string(),
            kind: "api_key".to_string(),
            token_hash: Some("sha256:preserved".to_string()),
            scopes: vec!["memory:read".to_string(), "memory:write".to_string()],
        })
        .unwrap();

    assert_eq!(catalog.graph().stats().nodes_total, 4);
    assert_eq!(
        catalog
            .graph()
            .live_node_records()
            .into_iter()
            .flat_map(|node| node.labels)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "_rustyred_native_auth_principal".to_string(),
            "_rustyred_native_billing_account".to_string(),
            "_rustyred_native_project".to_string(),
            "_rustyred_native_tenant".to_string(),
        ])
    );

    let mut rebound = NativeCatalog::from_graph(catalog.graph().clone());
    let tenant_node_id = rebound
        .graph()
        .node_ids_for_label("_rustyred_native_tenant")
        .into_iter()
        .next()
        .unwrap();
    let mut tenant_node = rebound.graph().get_node_record(&tenant_node_id).unwrap();
    tenant_node.properties["display_name"] = json!("Graph Alpha");
    rebound.graph_mut().upsert_node(tenant_node).unwrap();

    assert_eq!(
        rebound.tenants(),
        vec![NativeTenantRecord {
            tenant_id: "tenant:a".to_string(),
            slug: "alpha".to_string(),
            display_name: Some("Graph Alpha".to_string()),
        }]
    );
    assert_eq!(rebound.projects_for_tenant("tenant:a").len(), 1);
    assert_eq!(
        rebound.billing_account_for_tenant("tenant:a"),
        Some(NativeBillingAccountRecord {
            tenant_id: "tenant:a".to_string(),
            plan: "pro".to_string(),
            status: "active".to_string(),
        })
    );
    assert_eq!(
        rebound.auth_principals_for_tenant("tenant:a"),
        vec![NativeAuthPrincipalRecord {
            principal_id: "principal:1".to_string(),
            tenant_id: "tenant:a".to_string(),
            kind: "api_key".to_string(),
            token_hash: Some("sha256:preserved".to_string()),
            scopes: vec!["memory:read".to_string(), "memory:write".to_string()],
        }]
    );

    rebound
        .upsert_tenant(NativeTenantRecord {
            tenant_id: "tenant:b".to_string(),
            slug: "beta".to_string(),
            display_name: None,
        })
        .unwrap();
    rebound
        .upsert_auth_principal(NativeAuthPrincipalRecord {
            principal_id: "principal:1".to_string(),
            tenant_id: "tenant:b".to_string(),
            kind: "api_key".to_string(),
            token_hash: Some("sha256:moved".to_string()),
            scopes: vec!["memory:read".to_string()],
        })
        .unwrap();
    assert!(rebound.auth_principals_for_tenant("tenant:a").is_empty());
    assert_eq!(rebound.auth_principals_for_tenant("tenant:b").len(), 1);
    assert_eq!(rebound.graph().stats().nodes_total, 5);
}

#[test]
fn native_catalog_rejects_malformed_and_noncanonical_reserved_nodes() {
    let canonical_tenant_id = "_rustyred_native_catalog:tenant:8:tenant:a";
    let mut catalog = NativeCatalog::new();
    catalog
        .graph_mut()
        .upsert_node(NodeRecord::new(
            canonical_tenant_id,
            ["_rustyred_native_tenant"],
            json!({"slug": "alpha", "display_name": "Malformed Alpha"}),
        ))
        .unwrap();
    catalog
        .graph_mut()
        .upsert_node(NodeRecord::new(
            "noncanonical-tenant",
            ["_rustyred_native_tenant"],
            json!({
                "tenant_id": "tenant:ghost",
                "slug": "ghost",
                "display_name": "Noncanonical Ghost"
            }),
        ))
        .unwrap();

    assert!(catalog.tenants().is_empty());
    for tenant_id in ["tenant:a", "tenant:ghost"] {
        let error = catalog
            .upsert_project(NativeProjectRecord {
                tenant_id: tenant_id.to_string(),
                project_slug: "theorem".to_string(),
                display_name: None,
            })
            .unwrap_err();
        assert_eq!(error.code, "catalog_foreign_key_violation");
    }

    let graph_version = catalog.graph().stats().version;
    let malformed = catalog
        .graph()
        .get_node_record(canonical_tenant_id)
        .unwrap();
    let error = catalog
        .upsert_tenant(NativeTenantRecord {
            tenant_id: "tenant:a".to_string(),
            slug: "alpha".to_string(),
            display_name: Some("Alpha".to_string()),
        })
        .unwrap_err();
    assert_eq!(error.code, "catalog_reserved_id_collision");
    assert_eq!(catalog.graph().stats().version, graph_version);
    assert_eq!(
        catalog.graph().get_node_record(canonical_tenant_id),
        Some(malformed)
    );
}

#[test]
fn memory_documents_and_scalar_indexes_rebuild_from_graph() {
    let mut graph = InMemoryGraphStore::new();
    for (id, tenant, status, updated_at) in [
        ("older", "Tenant-A", "active", "2026-06-27T00:00:01Z"),
        ("archived", "Tenant-A", "archived", "2026-06-27T00:00:02Z"),
        ("newer", "Tenant-A", "active", "2026-06-27T00:00:03Z"),
        ("deleted", "Tenant-A", "deleted", "2026-06-27T00:00:04Z"),
        ("other", "Tenant-B", "active", "2026-06-27T00:00:05Z"),
    ] {
        graph
            .upsert_node(NodeRecord::new(
                format!("memory:document:{tenant}:{id}"),
                ["HarnessMemory", "MemoryAtom", "MemoryDocument"],
                json!({
                    "tenant_slug": tenant,
                    "doc_id": id,
                    "status": status,
                    "updated_at": updated_at,
                    "created_at": updated_at,
                }),
            ))
            .unwrap();
    }

    let memory_lens = MemoryDocumentLens::bind(&graph).unwrap();
    let memory_rows = memory_lens
        .query(
            MemoryDocumentQuery::tenant("Tenant-A")
                .with_before("2026-06-27T00:00:04Z")
                .with_limit(10),
        )
        .unwrap();
    assert_eq!(
        memory_rows
            .iter()
            .map(|node| node.properties["doc_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["newer", "archived", "older"]
    );

    graph
        .upsert_node(NodeRecord::new(
            "event:1",
            ["Event"],
            json!({
                "name": "Ada",
                "at_ms": 1_700_000_000_000_i64,
                "body": "graph rebuild candidate",
            }),
        ))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new(
            "event:2",
            ["Event"],
            json!({
                "name": "Grace",
                "at_ms": 1_700_000_000_100_i64,
                "body": "second candidate",
            }),
        ))
        .unwrap();

    let ordered = OrderedAccessMethod::new();
    let time_series = TimeSeriesAccessMethod::new();
    let trigram = TrigramAccessMethod::new();
    let mut registry = AccessMethodRegistry::new();
    registry.register(ordered.clone());
    registry.register(time_series.clone());
    registry.register(trigram.clone());
    let mut lens = RelationalLens::from_graph_with_access_methods(graph, registry).unwrap();

    assert_eq!(ordered.stats().rebuilds, 1);
    assert_eq!(time_series.stats().rebuilds, 1);
    assert_eq!(trigram.stats().rebuilds, 1);
    assert_eq!(ordered.stats().write_updates, 0);
    assert_eq!(time_series.stats().write_updates, 0);
    assert_eq!(trigram.stats().write_updates, 0);
    assert_eq!(
        lens.access_methods().indexed_graph_version(),
        Some(lens.graph().stats().version)
    );

    let equals_ada = Predicate::Equals {
        column: "name".to_string(),
        value: ScalarValue::String("Ada".to_string()),
    };
    let time_window = Predicate::TimeRange {
        column: "at_ms".to_string(),
        lo_ms: 1_700_000_000_000,
        hi_ms: 1_700_000_000_050,
    };
    let text_match = Predicate::TextMatch {
        column: "body".to_string(),
        query: "rebuild".to_string(),
        mode: PredicateMode::Filter,
    };
    assert_eq!(
        ordered.scan("event", &equals_ada).unwrap().rows,
        vec!["event:1"]
    );
    assert_eq!(
        ordered
            .scan(
                "event",
                &Predicate::Range {
                    column: "name".to_string(),
                    lo: ScalarBound::Unbounded,
                    hi: ScalarBound::Unbounded,
                },
            )
            .unwrap()
            .rows,
        vec!["event:1", "event:2"]
    );
    assert_eq!(
        time_series.scan("event", &time_window).unwrap().rows,
        vec!["event:1"]
    );
    assert_eq!(
        trigram.scan("event", &text_match).unwrap().rows,
        vec!["event:1"]
    );

    let write_fed_ordered = OrderedAccessMethod::new();
    let write_fed_time_series = TimeSeriesAccessMethod::new();
    let write_fed_trigram = TrigramAccessMethod::new();
    let event_one_values = BTreeMap::from([
        ("name".to_string(), ScalarValue::String("Ada".to_string())),
        ("at_ms".to_string(), ScalarValue::I64(1_700_000_000_000)),
        (
            "body".to_string(),
            ScalarValue::String("graph rebuild candidate".to_string()),
        ),
    ]);
    let write = RowChange {
        relation: "event".to_string(),
        row_id: "event:1".to_string(),
        values: event_one_values,
        properties: json!({}),
        kind: RowChangeKind::Upsert,
    };
    write_fed_ordered.on_write(&write).unwrap();
    write_fed_time_series.on_write(&write).unwrap();
    write_fed_trigram.on_write(&write).unwrap();
    assert_eq!(
        ordered.scan("event", &equals_ada).unwrap().rows,
        write_fed_ordered.scan("event", &equals_ada).unwrap().rows
    );
    assert_eq!(
        time_series.scan("event", &time_window).unwrap().rows,
        write_fed_time_series
            .scan("event", &time_window)
            .unwrap()
            .rows
    );
    assert_eq!(
        trigram.scan("event", &text_match).unwrap().rows,
        write_fed_trigram.scan("event", &text_match).unwrap().rows
    );

    let mut event = lens.graph().get_node_record("event:1").unwrap();
    event.properties["name"] = json!("Augusta");
    lens.graph_mut().upsert_node(event).unwrap();
    assert_eq!(lens.access_methods().indexed_graph_version(), None);
    lens.ensure_access_methods_current().unwrap();
    assert_eq!(
        lens.access_methods().indexed_graph_version(),
        Some(lens.graph().stats().version)
    );
    assert!(ordered.scan("event", &equals_ada).unwrap().rows.is_empty());
    assert_eq!(
        ordered
            .scan(
                "event",
                &Predicate::Equals {
                    column: "name".to_string(),
                    value: ScalarValue::String("Augusta".to_string()),
                },
            )
            .unwrap()
            .rows,
        vec!["event:1"]
    );
    assert_eq!(ordered.stats().rebuilds, 2);
    assert_eq!(ordered.stats().write_updates, 0);
    assert_eq!(
        lens.relation("event")
            .unwrap()
            .get("event:1")
            .unwrap()
            .values
            .get("name"),
        Some(&ScalarValue::String("Augusta".to_string()))
    );
}
