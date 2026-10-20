use crate::{Compiled, ContentHash, Engine, FactPack, Runtime};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

const REACH: &str = r#"
(relation edge (String String))
(relation path (String String))
(rule ((edge a b)) ((path a b)) :name "base")
(rule ((path a b) (edge b c)) ((path a c)) :name "transitive")
"#;

fn chain() -> FactPack {
    FactPack::new(BTreeMap::from([(
        "edge".into(),
        vec![
            vec![json!("A"), json!("B")],
            vec![json!("B"), json!("C")],
            vec![json!("C"), json!("D")],
        ],
    )]))
}

#[test]
fn runtime_transitive_proofs_resolve_to_actual_inputs_or_conclusions() {
    let facts = chain();
    let result = Runtime::new(REACH).unwrap().derive(&facts);
    result.ensure_complete().unwrap();
    assert_eq!(result.engine, "egraph-theorem");
    assert_eq!(result.derived.len(), 6);
    assert!(result
        .derived
        .iter()
        .any(|row| row.tuple == vec![json!("A"), json!("D")]));
    let mut available: BTreeSet<String> = facts
        .relations
        .iter()
        .flat_map(|(name, rows)| {
            rows.iter().map(move |row| {
                ContentHash::digest(serde_json::to_vec(&(name, row)).unwrap()).to_string()
            })
        })
        .collect();
    available.extend(result.derived.iter().map(|row| row.fact_id.clone()));
    assert!(result
        .provenance
        .iter()
        .all(|proof| !proof.premises.is_empty()
            && proof.premises.iter().all(|id| available.contains(id))));
    assert!(result.stats.matches > 0 && result.stats.iterations > 0);
    let serialized = serde_json::to_vec(&result).unwrap();
    assert_eq!(
        serde_json::from_slice::<crate::DerivationReceipt>(&serialized).unwrap(),
        result
    );
}

#[test]
fn runtime_retains_literal_program_fact_authority() {
    let rules = format!("{REACH}\n(edge \"A\" \"B\")");
    let result = Runtime::new(rules)
        .unwrap()
        .derive(&FactPack::new(BTreeMap::new()));
    result.ensure_complete().unwrap();
    assert_eq!(result.derived.len(), 2);
    let seed = result
        .provenance
        .iter()
        .find(|proof| proof.premises.is_empty())
        .unwrap();
    assert_eq!(seed.rule, format!("program-fact:{}", result.rule_pack));
    assert!(result
        .provenance
        .iter()
        .any(|proof| proof.premises == vec![seed.conclusion.clone()]));
}

#[test]
fn runtime_repeat_does_not_reuse_previous_run_facts() {
    let engine = Runtime::new(REACH).unwrap();
    assert_eq!(engine.derive(&chain()).derived.len(), 6);
    let result = engine.derive(&FactPack::new(BTreeMap::new()));
    result.ensure_complete().unwrap();
    assert!(result.derived.is_empty());
}

#[test]
fn incomplete_and_malformed_runs_are_not_successful_empty_results() {
    let limited = Runtime::new(REACH)
        .unwrap()
        .with_iteration_limit(1)
        .unwrap()
        .derive(&chain());
    assert!(!limited.stats.complete);
    assert!(limited.ensure_complete().is_err());
    let bad = FactPack::new(BTreeMap::from([(
        "edge".into(),
        vec![vec![json!({"A":1}), json!("B")]],
    )]));
    let result = Runtime::new(REACH).unwrap().derive(&bad);
    assert!(result.stats.refusal.is_some());
    assert!(result.ensure_complete().is_err());
    assert!(Runtime::new("(datatype N (N i64)) (union (N 1) (N 2))").is_err());
}

#[test]
fn both_engines_refuse_tampered_fact_identity() {
    let mut facts = chain();
    facts.relations.clear();
    for engine in [
        &Compiled::default() as &dyn Engine,
        &Runtime::new(REACH).unwrap(),
    ] {
        let receipt = engine.derive(&facts);
        assert!(receipt
            .stats
            .refusal
            .as_deref()
            .unwrap()
            .contains("identity"));
        assert!(receipt.ensure_complete().is_err());
    }
}

#[test]
fn compiled_retains_absence_proof_and_legacy_receipt_identity() {
    let rows = vec![
        json!({"relation":"claim", "entity_id":"C1", "attributes":{"status":"proposed"}, "source_ref":"test", "fact_id":"claim:1"}),
    ];
    let facts = crate::fact_pack_from_legacy_rows(&rows).unwrap();
    let receipt = Compiled::new(vec!["unsupported_claim".into()]).derive(&facts);
    receipt.ensure_complete().unwrap();
    assert_eq!(receipt.engine, "rust-datafrog-core");
    assert_eq!(receipt.derived.len(), 1);
    assert_eq!(receipt.provenance[0].premises, vec!["claim:1"]);
    let legacy =
        crate::derive_legacy_receipt(&json!({"facts":rows,"rule_ids":["unsupported_claim"]}))
            .unwrap();
    assert_eq!(legacy["engine"], "python-reference-datalog");
    assert_eq!(
        legacy["derived_facts"][0]["fact_id"],
        receipt.derived[0].fact_id
    );
}

#[test]
fn retained_native_identity_rule_carries_the_two_actual_match_edges() {
    let facts = crate::fact_pack_from_legacy_rows(&[
        json!({"relation":"match_edge","entity_id":"AB","fact_id":"edge:AB",
            "attributes":{"left_record_id":"A","right_record_id":"B","weight":100.0}}),
        json!({"relation":"match_edge","entity_id":"BC","fact_id":"edge:BC",
            "attributes":{"left_record_id":"B","right_record_id":"C","weight":100.0}}),
    ])
    .unwrap();
    let receipt = Compiled::new(vec!["same_entity".into()]).derive(&facts);
    receipt.ensure_complete().unwrap();
    let inferred = receipt
        .derived
        .iter()
        .find(|row| row.subject_id == "A" && row.attributes["target_id"] == "C")
        .unwrap();
    assert_eq!(inferred.attributes["path_length"], 2);
    let proof = receipt
        .provenance
        .iter()
        .find(|proof| proof.conclusion == inferred.fact_id)
        .unwrap();
    assert_eq!(proof.premises, vec!["edge:AB", "edge:BC"]);
    assert_eq!(inferred.confidence, 0.9);
}

#[test]
fn serialized_scalar_fact_packs_preserve_their_exact_content_identity() {
    // This actual offload observation round-tripped as 10.848667 without
    // serde_json's exact float parser, changing the evidence's content hash.
    let observation = 10.848666999999999_f64;
    let facts = FactPack::new(BTreeMap::from([(
        "elapsed".into(),
        vec![vec![json!(observation)]],
    )]));
    let bytes = serde_json::to_vec(&facts).unwrap();
    let reopened: FactPack = serde_json::from_slice(&bytes).unwrap();
    reopened.validate().unwrap();
    assert_eq!(reopened, facts);
    let receipt = Runtime::new("(relation elapsed (f64)) (relation observed (f64)) (rule ((elapsed n)) ((observed n)) :name \"record\")")
        .unwrap().derive(&reopened);
    receipt.ensure_complete().unwrap();
    let wire = serde_json::to_vec(&receipt).unwrap();
    let decoded: crate::DerivationReceipt = serde_json::from_slice(&wire).unwrap();
    assert_eq!(decoded, receipt);
}

#[test]
fn runtime_abc_negative_evidence_binds_closed_input_and_rule_pack() {
    let program = r#"
      (relation edge (String String String))
      (relation shared_intermediate (String String String))
      (rule ((edge a b k1) (edge c b k2) (!= a c) (not (edge a c _)))
        ((shared_intermediate a c b)) :name "abc")
    "#;
    let runtime = Runtime::new(program).unwrap();
    let rows = vec![
        vec![json!("a"), json!("b"), json!("k1")],
        vec![json!("c"), json!("b"), json!("k2")],
    ];
    let facts = FactPack::new(BTreeMap::from([("edge".into(), rows.clone())]));
    let absent = runtime.derive(&facts);
    absent.ensure_complete().unwrap();
    let output = absent
        .derived
        .iter()
        .find(|row| row.tuple == vec![json!("a"), json!("c"), json!("b")])
        .unwrap();
    let proof = absent
        .provenance
        .iter()
        .find(|proof| proof.conclusion == output.fact_id)
        .unwrap();
    assert_eq!(proof.premises.len(), 2);
    assert_eq!(proof.negative_premises.len(), 1);
    let evidence = &proof.negative_premises[0];
    assert_eq!(
        evidence.pattern,
        vec![Some(json!("a")), Some(json!("c")), None]
    );
    assert_eq!(evidence.relation, "edge");
    assert_eq!(evidence.fact_pack, facts.hash);
    assert_eq!(evidence.rule_pack, absent.rule_pack);
    assert_eq!(evidence.stratum, 1);
    let reopened: crate::DerivationReceipt =
        serde_json::from_slice(&serde_json::to_vec(&absent).unwrap()).unwrap();
    assert_eq!(absent, reopened);
    let mut direct = rows;
    direct.push(vec![json!("a"), json!("c"), json!("arbitrary-kind")]);
    let present = runtime.derive(&FactPack::new(BTreeMap::from([("edge".into(), direct)])));
    present.ensure_complete().unwrap();
    assert!(!present.derived.iter().any(|row| row.tuple == output.tuple));
    let reverse = &present.provenance[0].negative_premises[0];
    assert_ne!(evidence.fact_pack, reverse.fact_pack);
    assert_ne!(evidence.input_facts, reverse.input_facts);
    assert_ne!(evidence.closed_relation, reverse.closed_relation);
    assert_eq!(runtime.derive(&facts).provenance, absent.provenance);
    let changed_rules = Runtime::new(format!(
        "{program}\n(relation marker (bool))\n(marker true)"
    ))
    .unwrap()
    .derive(&facts);
    changed_rules.ensure_complete().unwrap();
    assert_ne!(changed_rules.rule_pack, absent.rule_pack);
    assert!(changed_rules
        .provenance
        .iter()
        .flat_map(|p| &p.negative_premises)
        .all(|n| n.rule_pack == changed_rules.rule_pack));
}

#[test]
fn runtime_negative_recursion_and_unsafe_variables_refuse_and_budget_stays_incomplete() {
    for text in [
        "(relation P (i64)) (relation Q (i64)) (rule ((not (P x))) ((Q x)))",
        "(relation P (i64)) (relation Q (i64)) (rule ((P x) (not (Q x))) ((Q x)))",
    ] {
        assert!(Runtime::new(text).is_err());
    }
    let runtime = Runtime::new(format!("{REACH}\n(relation candidate (String String))\n(relation missing (String String))\n(rule ((candidate x y) (not (path x y))) ((missing x y)) :name \"missing\")")).unwrap().with_iteration_limit(1).unwrap();
    let mut relations = chain().relations;
    relations.insert("candidate".into(), vec![vec![json!("A"), json!("D")]]);
    let receipt = runtime.derive(&FactPack::new(relations));
    assert!(receipt.ensure_complete().is_err());
    assert!(
        receipt.stats.refusal.is_none(),
        "bounded execution is distinct from invalid input"
    );
    assert!(!receipt.stats.complete);
    assert!(!receipt.derived.iter().any(|row| row.relation == "missing"));
    assert!(receipt
        .provenance
        .iter()
        .all(|proof| proof.negative_premises.is_empty()));
}

#[test]
fn positive_derivation_wire_remains_compatible_without_empty_negative_field() {
    let old = json!({"rule":"r", "conclusion":"c", "premises":["p"]});
    let proof: crate::Derivation = serde_json::from_value(old.clone()).unwrap();
    assert!(proof.negative_premises.is_empty());
    assert_eq!(serde_json::to_value(proof).unwrap(), old);
    let result = Runtime::new(REACH).unwrap().derive(&chain());
    result.ensure_complete().unwrap();
    assert!(serde_json::to_value(&result).unwrap()["provenance"]
        .as_array()
        .unwrap()
        .iter()
        .all(|p| p.get("negative_premises").is_none()));
}

#[test]
fn runtime_preserves_legacy_positive_rule_order_with_escaped_names() {
    let runtime = Runtime::new(
        r#"
      (relation seed (i64)) (relation output (i64)) (seed 1)
      (rule ((seed x)) ((output x)) :name " first")
      (rule ((seed x)) ((output x)) :name "\nfirst")
    "#,
    )
    .unwrap();
    let receipt = runtime.derive(&FactPack::new(BTreeMap::new()));
    receipt.ensure_complete().unwrap();
    assert_eq!(
        receipt
            .derived
            .iter()
            .find(|row| row.relation == "output")
            .unwrap()
            .rule_id,
        "\nfirst"
    );
}
