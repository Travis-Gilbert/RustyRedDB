//! Real graph-store derivation oracles live with the engine to keep publication acyclic.

mod fold_tests {
    use rustyred_thg_core::graph::in_memory::InMemoryGraphStore;
    use rustyred_thg_core::graph::query::NeighborQuery;
    use rustyred_thg_core::graph::record::{EdgeRecord, NodeRecord};
    use rustyred_thg_derivation::fold::*;
    use rustyred_thg_derivation::semiring::{
        BooleanSemiring, CountingSemiring, ProbabilitySemiring, TropicalSemiring,
    };
    use rustyred_thg_derivation::store::{
        DerivationStore, ANNOTATION_PROPERTY, DERIVATION_ANCESTOR_EDGE,
    };
    use rustyred_thg_derivation::store::{
        ASSUMPTION_LABEL, DERIVATION_ASSUMES_EDGE, DERIVATION_LABEL,
    };
    use serde_json::json;

    struct CoreStore<'a>(&'a InMemoryGraphStore);

    impl DerivationStore for CoreStore<'_> {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            self.0.get_node(node_id).map(|n| n.labels.clone())
        }

        fn edge_annotation(&self, edge_id: &str, property: &str) -> Option<f64> {
            self.0
                .get_edge(edge_id)
                .and_then(|edge| edge.properties.get(property)?.as_f64())
        }

        fn neighbors_in(
            &self,
            node_id: &str,
            edge_type: &str,
        ) -> Vec<rustyred_thg_derivation::store::DerivationNeighbor> {
            self.0
                .neighbors(NeighborQuery::in_(node_id).with_edge_type(edge_type))
                .into_iter()
                .map(|hit| rustyred_thg_derivation::store::DerivationNeighbor {
                    node_id: hit.node_id,
                    edge_id: hit.edge_id,
                })
                .collect()
        }

        fn edge_type(&self, edge_id: &str) -> Option<String> {
            self.0.get_edge(edge_id).map(|edge| edge.edge_type.clone())
        }

        fn incoming_edge_types(&self, node_id: &str) -> Option<Vec<String>> {
            Some(
                self.0
                    .neighbors(NeighborQuery::in_(node_id))
                    .into_iter()
                    .filter_map(|hit| self.0.get_edge(&hit.edge_id))
                    .map(|edge| edge.edge_type.clone())
                    .collect(),
            )
        }
    }

    fn build_three_layer_chain() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new(
                "A",
                [ASSUMPTION_LABEL],
                json!({"value": "ground"}),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "B",
                [DERIVATION_LABEL],
                json!({"value": "step1"}),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "C",
                [DERIVATION_LABEL],
                json!({"value": "step2"}),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "e_ab",
                "A",
                DERIVATION_ASSUMES_EDGE,
                "B",
                json!({ ANNOTATION_PROPERTY: 0.9 }),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "e_bc",
                "B",
                DERIVATION_ANCESTOR_EDGE,
                "C",
                json!({ ANNOTATION_PROPERTY: 0.8 }),
            ))
            .unwrap();
        store
    }

    #[test]
    fn fold_boolean_three_layer() {
        let store = build_three_layer_chain();
        assert!(fold::<BooleanSemiring>(&CoreStore(&store), "C"));
    }

    #[test]
    fn fold_counting_three_layer() {
        let store = build_three_layer_chain();
        assert_eq!(fold::<CountingSemiring>(&CoreStore(&store), "C"), 1);
    }

    #[test]
    fn fold_probability_three_layer() {
        let store = build_three_layer_chain();
        assert!((fold::<ProbabilitySemiring>(&CoreStore(&store), "C") - 0.72).abs() < 1e-9);
    }

    #[test]
    fn fold_tropical_three_layer() {
        let store = build_three_layer_chain();
        assert!((fold::<TropicalSemiring>(&CoreStore(&store), "C") - 1.7).abs() < 1e-9);
    }

    #[test]
    fn fold_assumption_returns_one() {
        let store = build_three_layer_chain();
        assert!(fold::<BooleanSemiring>(&CoreStore(&store), "A"));
        assert_eq!(fold::<CountingSemiring>(&CoreStore(&store), "A"), 1);
    }

    #[test]
    fn fold_missing_node_returns_zero() {
        let store = InMemoryGraphStore::default();
        assert!(!fold::<BooleanSemiring>(&CoreStore(&store), "missing"));
        assert_eq!(fold::<CountingSemiring>(&CoreStore(&store), "missing"), 0);
    }

    fn build_diamond() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("A", [ASSUMPTION_LABEL], json!({})))
            .unwrap();
        for id in ["B1", "B2", "C"] {
            store
                .upsert_node(NodeRecord::new(id, [DERIVATION_LABEL], json!({})))
                .unwrap();
        }
        for (id, frm, to, ann) in [
            ("e_ab1", "A", "B1", 0.9),
            ("e_ab2", "A", "B2", 0.7),
            ("e_b1c", "B1", "C", 0.8),
            ("e_b2c", "B2", "C", 0.6),
        ] {
            store
                .upsert_edge(EdgeRecord::new(
                    id,
                    frm,
                    DERIVATION_ASSUMES_EDGE,
                    to,
                    json!({"annotation": ann}),
                ))
                .unwrap();
        }
        store
    }

    #[test]
    fn fold_probability_diamond() {
        let store = build_diamond();
        let expected = 0.72_f64 + 0.42 - 0.72 * 0.42;
        assert!((fold::<ProbabilitySemiring>(&CoreStore(&store), "C") - expected).abs() < 1e-9);
    }

    #[test]
    fn fold_counting_diamond() {
        let store = build_diamond();
        assert_eq!(fold::<CountingSemiring>(&CoreStore(&store), "C"), 2);
    }

    #[test]
    fn fold_tropical_diamond() {
        let store = build_diamond();
        assert!((fold::<TropicalSemiring>(&CoreStore(&store), "C") - 1.3).abs() < 1e-9);
    }

    #[test]
    fn code_symbol_fold_mints_inbound_call_edge_variables() {
        let mut store = InMemoryGraphStore::default();
        for id in ["caller_a", "caller_b", "target"] {
            store
                .upsert_node(NodeRecord::new(id, ["CodeSymbol"], json!({})))
                .unwrap();
        }
        for (id, from) in [("call:a", "caller_a"), ("call:b", "caller_b")] {
            store
                .upsert_edge(EdgeRecord::new(
                    id,
                    from,
                    rustyred_thg_derivation::CALLS_SYMBOL_EDGE,
                    "target",
                    json!({}),
                ))
                .unwrap();
        }

        let polynomial = fold_code_incoming(
            &CoreStore(&store),
            "target",
            rustyred_thg_derivation::code_graph_ontology(),
        )
        .expect("code fold");
        let variables: Vec<_> = polynomial
            .generators()
            .into_iter()
            .map(|generator| generator.token.as_str())
            .collect();
        assert_eq!(variables, vec!["call:a", "call:b"]);
    }

    #[test]
    fn code_symbol_fold_refuses_an_unclassified_inbound_edge() {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("caller", ["CodeSymbol"], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("target", ["CodeSymbol"], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "mystery:1",
                "caller",
                "MYSTERY_CODE_EDGE",
                "target",
                json!({}),
            ))
            .unwrap();

        let error = fold_code_incoming(
            &CoreStore(&store),
            "target",
            rustyred_thg_derivation::code_graph_ontology(),
        )
        .expect_err("unknown edges must refuse");
        assert_eq!(
            error,
            DualFoldError::UnclassifiedEdge {
                edge_id: "mystery:1".to_string(),
                edge_type: "MYSTERY_CODE_EDGE".to_string(),
            }
        );
    }
}

mod why_tests {
    use rustyred_thg_core::graph::in_memory::InMemoryGraphStore;
    use rustyred_thg_core::graph::query::NeighborQuery;
    use rustyred_thg_core::graph::record::{EdgeRecord, NodeRecord};
    use rustyred_thg_derivation::store::{
        ASSUMPTION_LABEL, DERIVATION_ASSUMES_EDGE, DERIVATION_LABEL,
    };
    use rustyred_thg_derivation::why::*;
    use rustyred_thg_derivation::{AlgebraicClass, DerivationStore};
    use serde_json::json;

    struct CoreStore<'a>(&'a InMemoryGraphStore);

    impl DerivationStore for CoreStore<'_> {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            self.0.get_node(node_id).map(|n| n.labels.clone())
        }

        fn edge_annotation(&self, edge_id: &str, property: &str) -> Option<f64> {
            self.0
                .get_edge(edge_id)
                .and_then(|edge| edge.properties.get(property)?.as_f64())
        }

        fn neighbors_in(
            &self,
            node_id: &str,
            edge_type: &str,
        ) -> Vec<rustyred_thg_derivation::store::DerivationNeighbor> {
            self.0
                .neighbors(NeighborQuery::in_(node_id).with_edge_type(edge_type))
                .into_iter()
                .map(|hit| rustyred_thg_derivation::store::DerivationNeighbor {
                    node_id: hit.node_id,
                    edge_id: hit.edge_id,
                })
                .collect()
        }

        fn edge_type(&self, edge_id: &str) -> Option<String> {
            self.0.get_edge(edge_id).map(|edge| edge.edge_type.clone())
        }
    }

    fn build_chain() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("A", [ASSUMPTION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("B", [DERIVATION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("C", [DERIVATION_LABEL], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "e_ab",
                "A",
                DERIVATION_ASSUMES_EDGE,
                "B",
                json!({"annotation": 1.0}),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "e_bc",
                "B",
                DERIVATION_ASSUMES_EDGE,
                "C",
                json!({"annotation": 1.0}),
            ))
            .unwrap();
        store
    }

    #[test]
    fn why_three_layer_chain() {
        let store = build_chain();
        let trace = why(&CoreStore(&store), "C").unwrap();
        assert_eq!(trace.root.node_id, "C");
        assert_eq!(trace.total_supporters, 2);
        assert_eq!(trace.assumption_count, 1);
    }

    #[test]
    fn why_assumption_is_leaf() {
        let store = build_chain();
        let trace = why(&CoreStore(&store), "A").unwrap();
        assert!(trace.root.is_assumption);
    }

    #[test]
    fn why_missing_node_is_none() {
        let store = InMemoryGraphStore::default();
        assert!(why(&CoreStore(&store), "missing").is_none());
    }

    #[test]
    fn why_code_fact_returns_examined_ids_and_ontology_classes() {
        let mut store = InMemoryGraphStore::default();
        store
            .upsert_node(NodeRecord::new("caller", ["CodeSymbol"], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("target", ["CodeSymbol"], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "call:1",
                "caller",
                rustyred_thg_derivation::CALLS_SYMBOL_EDGE,
                "target",
                json!({}),
            ))
            .unwrap();

        let trace = why_code_fact(
            &CoreStore(&store),
            "fact:inbound:target",
            ["call:1"],
            rustyred_thg_derivation::code_graph_ontology(),
        )
        .expect("code why");
        assert_eq!(trace.fact_id, "fact:inbound:target");
        assert_eq!(trace.examined.len(), 1);
        assert_eq!(trace.examined[0].edge_id, "call:1");
        assert_eq!(
            trace.examined[0].canonical_type,
            rustyred_thg_derivation::CALLS_SYMBOL_EDGE
        );
        assert_eq!(trace.examined[0].class, AlgebraicClass::MonotonePositive);
    }
}
