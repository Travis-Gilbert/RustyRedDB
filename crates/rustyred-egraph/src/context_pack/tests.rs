use super::*;

#[test]
fn actual_python_context_fixtures_keep_items_costs_and_trace_identities() {
    let fixtures: Vec<Value> = serde_json::from_str(include_str!("python-fixtures.json")).unwrap();
    for fixture in fixtures {
        let input: ContextPackInput = serde_json::from_value(fixture["input"].clone()).unwrap();
        let result = extract(&input).unwrap();
        let expected = &fixture["python_receipt"];
        for key in [
            "engine",
            "input_hash",
            "domain",
            "equivalent",
            "original_cost",
            "extracted_cost",
            "rewrite_trace",
        ] {
            assert_eq!(result.receipt[key], expected[key], "{key}");
        }
        assert_eq!(
            result.receipt["extraction"]["items"],
            expected["extraction"]["items"]
        );
        assert_eq!(result.receipt["native_backend"], "rust-egg-context-pack");
        assert_ne!(
            result.receipt["output_hash"], expected["output_hash"],
            "native metadata has actual engine identities"
        );
        assert!(result.receipt["extraction"]["metadata"]
            .get("egg_probe")
            .is_none());
        assert!(result
            .saturations
            .iter()
            .all(|s| s.stop_reason == StopReason::FixedPoint));
        assert_eq!(result.extractions.len(), 2);
    }
}

#[test]
fn actual_union_rules_remove_optional_duplicates_and_preserve_every_required_item() {
    let input = ContextPackInput {
        expression_id: "ordered-required".into(),
        cost_config: json!({}),
        items: vec![
            json!({"id":"empty","obligation_id":"x","semantic_hash":"k","text":""}),
            json!({"id":"first","obligation_id":"x","semantic_hash":"k","text":"first"}),
            json!({"id":"required","obligation_id":"x","semantic_hash":"k","text":"required","hard_required":true}),
            json!({"id":"duplicate","obligation_id":"x","semantic_hash":"k","text":"later"}),
            json!({"id":"required-empty","text":"","hard_required":true}),
        ],
    };
    let result = extract(&input).unwrap();
    let ids: Vec<_> = result.receipt["extraction"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["first", "required", "required-empty"]);
    assert!(result.saturations[0].matches_per_rule["drop_empty_optional"] > 0);
    assert!(result.saturations[1].matches_per_rule["dedupe_same_obligation"] > 0);
    assert!(result.extractions[1].cost < result.extractions[0].cost);
}

#[test]
fn semantic_receipt_is_replayable_while_observations_remain_actual() {
    let input = ContextPackInput {
        expression_id: "replay".into(),
        items: vec![json!({"id":"x","text":"keep"})],
        cost_config: json!({}),
    };
    let first = extract(&input).unwrap();
    let second = extract(&input).unwrap();
    assert_eq!(first.receipt, second.receipt);
    assert!(first
        .saturations
        .iter()
        .all(|s| s.iterations > 0 && s.wall_time_ns > 0));
    let wire = extract_json(&serde_json::to_string(&input).unwrap()).unwrap();
    let parsed: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(
        parsed["native_evidence"]["extractions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn canonical_pack_resolves_directly_and_legacy_zero_cost_does_not_block_removal() {
    let pack = rule_pack();
    pack.validate().unwrap();
    let mut engine = Engine::new(pack).unwrap();
    let result = engine
        .saturate(Schedule::FixedPoint {
            rulesets: vec!["context_pack".into()],
            max_iterations: 8,
        })
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::FixedPoint);
    let input = ContextPackInput {
        expression_id: "zero".into(),
        items: vec![json!({"text":"","tokens":0})],
        cost_config: json!({"item_weight":0}),
    };
    let result = extract(&input).unwrap();
    assert_eq!(result.receipt["extraction"]["items"], json!([]));
    assert_eq!(result.receipt["original_cost"], json!(0.0));
    assert_eq!(result.extractions.last().unwrap().cost, 0);
}

#[test]
fn consuming_resolver_engine_refuses_custom_or_previously_touched_state() {
    let input = ContextPackInput {
        expression_id: "resolver".into(),
        items: vec![],
        cost_config: json!({}),
    };
    let fresh = Engine::new(rule_pack()).unwrap();
    assert!(fresh.is_pristine());
    assert!(extract_with_engine(fresh, &input).is_ok());
    let mut touched = Engine::new(rule_pack()).unwrap();
    let before = touched.num_tuples();
    touched.execute("(relation Extra (i64))").unwrap();
    assert_eq!(
        touched.num_tuples(),
        before,
        "schema-only mutation evades tuple counting"
    );
    assert!(!touched.is_pristine());
    assert!(extract_with_engine(touched, &input).is_err());
    let custom = Engine::new(RulePack::new(
        "(relation Custom (i64))",
        CostModel::default(),
    ))
    .unwrap();
    assert!(extract_with_engine(custom, &input).is_err());
}

#[test]
fn executed_native_legacy_fixtures_preserve_contract_and_record_metadata_hash_change() {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("legacy-native-fixtures.json")).unwrap();
    for fixture in fixtures {
        let input: ContextPackInput = serde_json::from_value(fixture["input"].clone()).unwrap();
        let result = extract(&input).unwrap();
        let legacy = &fixture["legacy_native_receipt"];
        for key in [
            "engine",
            "native_backend",
            "input_hash",
            "domain",
            "equivalent",
            "original_cost",
            "extracted_cost",
            "rewrite_trace",
        ] {
            assert_eq!(result.receipt[key], legacy[key], "native {key}");
        }
        for key in ["expression_id", "domain", "items"] {
            assert_eq!(result.receipt["extraction"][key], legacy["extraction"][key]);
        }
        assert_eq!(
            legacy["extraction"]["metadata"]["egg_probe"]["engine"],
            "egg"
        );
        assert!(result.receipt["extraction"]["metadata"]
            .get("egg_probe")
            .is_none());
        assert_ne!(result.receipt["output_hash"], legacy["output_hash"]);
        println!(
            "metadata_hash_transition {}",
            json!({"expression_id":input.expression_id,"legacy_output_hash":legacy["output_hash"],"actual_output_hash":result.receipt["output_hash"],"legacy_metadata":legacy["extraction"]["metadata"],"actual_metadata":result.receipt["extraction"]["metadata"]})
        );
    }
}
