use crate::*;
use std::collections::BTreeMap;

const REWRITES: &str = r#"
(datatype Math (Num i64) (Add Math Math :cost 10))
(ruleset simplify)
(rewrite (Add x (Num 0)) x :ruleset simplify)
"#;

#[test]
fn saturation_and_extraction_match_pinned_engine_observations() {
    let pack = RulePack::new(REWRITES, CostModel::default());
    let mut engine = Engine::new(pack.clone()).unwrap();
    engine.execute("(let root (Add (Num 7) (Num 0)))").unwrap();
    let root = engine.root("root").unwrap();
    let report = engine
        .saturate(Schedule::Run {
            ruleset: "simplify".into(),
            iterations: 1,
        })
        .unwrap();
    let (term, extraction) = engine.extract(root, &CostModel::default()).unwrap();
    let mut reference = egglog::EGraph::default();
    let outputs = reference
        .parse_and_run_program(
            None,
            &format!(
                "{REWRITES}\n(let root (Add (Num 7) (Num 0)))\n(run simplify 1)\n(extract root)"
            ),
        )
        .unwrap();
    for output in outputs {
        match output {
            egglog::CommandOutput::RunSchedule(actual) => {
                assert_eq!(report.iterations, actual.iterations.len());
                assert_eq!(
                    report.matches,
                    actual.num_matches_per_rule.values().sum::<usize>()
                );
                assert!(report.matches > 0);
            }
            egglog::CommandOutput::ExtractBest(dag, cost, id) => {
                assert_eq!(term.expression, dag.to_string(id));
                assert_eq!(extraction.cost, cost);
            }
            _ => {}
        }
    }
    let serialized = reference.serialize(egglog::SerializeConfig::default());
    assert_eq!(report.enodes, serialized.egraph.nodes.len());
    assert_eq!(
        report.eclasses,
        serialized
            .egraph
            .nodes
            .values()
            .map(|n| &n.eclass)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    assert_eq!(report.tuples, reference.num_tuples());
    assert_eq!(term.expression, "(Num 7)");
    assert_eq!(extraction.rule_pack, pack.id);
    assert_eq!(extraction.term_hash, ContentHash::digest(b"(Num 7)"));
    assert_eq!(report.stop_reason, StopReason::Scheduled);
}

#[test]
fn custom_costs_choose_actual_alternative_and_roots_are_engine_bound() {
    let program = "(datatype T (A :cost 1) (B :cost 10)) (union (A) (B))";
    let mut engine = Engine::new(RulePack::new(program, CostModel::default())).unwrap();
    let root = engine.root("(A)").unwrap();
    assert_eq!(
        engine
            .extract(root, &CostModel::default())
            .unwrap()
            .0
            .expression,
        "(A)"
    );
    let costs = CostModel {
        constructor_costs: BTreeMap::from([("A".into(), 100), ("B".into(), 1)]),
    };
    let (term, receipt) = engine.extract(root, &costs).unwrap();
    assert_eq!(term.expression, "(B)");
    assert_eq!(receipt.cost, 1);
    assert_eq!(receipt.cost_model, costs.content_hash());
    let other = Engine::new(RulePack::new(program, CostModel::default())).unwrap();
    assert!(other.extract(root, &costs).is_err());
}

const RELATIONS: &str = r#"
(relation edge (String String))
(relation path (String String))
(rule ((edge a b)) ((path a b)) :name "base")
(rule ((path a b) (edge b c)) ((path a c)) :name "transitive")
"#;
fn row(a: &str, b: &str) -> Vec<Atom> {
    vec![Atom::String(a.into()), Atom::String(b.into())]
}

#[test]
fn relational_closure_and_provenance_are_produced_by_actual_rule_matches() {
    let engine = RelationalEngine::new(RulePack::new(RELATIONS, CostModel::default())).unwrap();
    let facts = BTreeMap::from([(
        "edge".into(),
        vec![row("a", "b"), row("b", "c"), row("a", "d"), row("d", "c")],
    )]);
    let receipt = engine.derive(&facts, 20).unwrap();
    assert_eq!(receipt.saturation.stop_reason, StopReason::FixedPoint);
    assert!(receipt.relations["path"].contains(&row("a", "c")));
    let proofs = receipt
        .provenance
        .iter()
        .filter(|p| p.rule == "transitive" && p.output.tuple == row("a", "c"))
        .collect::<Vec<_>>();
    assert_eq!(proofs.len(), 2);
    assert!(proofs.iter().any(|p| p.premises
        == vec![
            RelationalFact {
                relation: "path".into(),
                tuple: row("a", "b")
            },
            RelationalFact {
                relation: "edge".into(),
                tuple: row("b", "c")
            }
        ]));
    assert!(proofs.iter().any(|p| p.premises[1].tuple == row("d", "c")));
    let empty = engine.derive(&BTreeMap::new(), 20).unwrap();
    assert!(
        empty.relations["path"].is_empty(),
        "a reusable evaluator must not leak prior fact packs"
    );
    let again = engine.derive(&facts, 20).unwrap();
    assert_eq!(receipt.relations, again.relations);
    assert_eq!(receipt.provenance, again.provenance);
    assert_eq!(
        serde_json::from_slice::<RelationalReceipt>(&serde_json::to_vec(&receipt).unwrap())
            .unwrap(),
        receipt
    );
}

#[test]
fn relational_iteration_limit_does_not_claim_a_fixpoint() {
    let engine = RelationalEngine::new(RulePack::new(RELATIONS, CostModel::default())).unwrap();
    let facts = BTreeMap::from([("edge".into(), vec![row("a", "b"), row("b", "c")])]);
    let receipt = engine.derive(&facts, 1).unwrap();
    assert_eq!(receipt.saturation.stop_reason, StopReason::IterationLimit);
    assert_eq!(receipt.saturation.iterations, 1);
    let zero = engine.derive(&facts, 0).unwrap();
    assert_eq!(zero.saturation.stop_reason, StopReason::IterationLimit);
    assert_eq!(zero.saturation.iterations, 0);
}

#[test]
fn relational_ast_refuses_equality_mutation_and_hidden_io() {
    for forbidden in [
        "(datatype T (A))",
        "(sort T)",
        "(constructor A () T)",
        "(function f () i64 :merge new)",
        "(rewrite 1 2)",
        "(birewrite 1 2)",
        "(union 1 2)",
        "(fail (union 1 2))",
        "(include \"missing.egg\")",
        "(input edge \"private.tsv\")",
        "(output \"out.tsv\" 1)",
        "(let hidden 1)",
        "(run 1)",
        "(rule ((edge a b)) ((union a b)))",
        "(rule ((edge a b)) ((delete (edge a b))))",
        "(rule ((edge a b)) ((set (edge a b) true)))",
        "(relation fake (T))",
    ] {
        let program = format!("{RELATIONS}\n{forbidden}");
        assert!(
            RelationalEngine::new(RulePack::new(program, CostModel::default())).is_err(),
            "admitted {forbidden}"
        );
    }
    let engine = RelationalEngine::new(RulePack::new(RELATIONS, CostModel::default())).unwrap();
    let literal = "(union (A) (B))\n\"quoted\"";
    let receipt = engine
        .derive(
            &BTreeMap::from([("edge".into(), vec![row(literal, "safe")])]),
            10,
        )
        .unwrap();
    assert!(
        receipt.relations["path"].contains(&row(literal, "safe")),
        "literal data is not parsed as executable text"
    );
}

#[test]
fn relational_scalar_equality_is_a_filter_without_eclasses() {
    let source = r#"(relation amount (String i64)) (relation chosen (String i64))
        (rule ((amount who n) (> n 2) (= next (+ n 1))) ((chosen who next)) :name "increment")"#;
    let engine = RelationalEngine::new(RulePack::new(source, CostModel::default())).unwrap();
    let input = BTreeMap::from([(
        "amount".into(),
        vec![
            vec![Atom::String("yes".into()), Atom::I64(3)],
            vec![Atom::String("no".into()), Atom::I64(1)],
        ],
    )]);
    let output = engine.derive(&input, 10).unwrap();
    assert_eq!(
        output.relations["chosen"],
        vec![vec![Atom::String("yes".into()), Atom::I64(4)]]
    );
    assert_eq!(output.provenance.len(), 1);
    assert_eq!(output.provenance[0].premises[0].tuple, input["amount"][0]);
}

#[test]
fn nonfinite_and_mismatched_facts_are_refused_before_insertion() {
    let engine = RelationalEngine::new(RulePack::new(
        "(relation number (f64))",
        CostModel::default(),
    ))
    .unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(engine
            .derive(
                &BTreeMap::from([(
                    "number".into(),
                    vec![vec![Atom::F64(1.0)], vec![Atom::F64(value)]]
                )]),
                2
            )
            .is_err());
        assert!(engine.derive(&BTreeMap::new(), 2).unwrap().relations["number"].is_empty());
    }
    assert!(engine
        .derive(
            &BTreeMap::from([("number".into(), vec![vec![Atom::I64(1)]])]),
            2
        )
        .is_err());
    assert!(engine
        .derive(
            &BTreeMap::from([("unknown".into(), vec![vec![Atom::F64(1.0)]])]),
            2
        )
        .is_err());
}

#[test]
fn content_identity_and_pure_engine_boundary_are_validated() {
    let mut pack = RulePack::new(REWRITES, CostModel::default());
    assert_eq!(pack.id.to_string().parse::<ContentHash>().unwrap(), pack.id);
    assert!(serde_json::from_str::<ContentHash>("\"sha256:wrong\"").is_err());
    pack.program.push(' ');
    assert!(Engine::new(pack).is_err());
    for program in [
        "(include \"private.egg\")",
        "(fail (output \"private.tsv\" 1))",
    ] {
        assert!(Engine::new(RulePack::new(program, CostModel::default())).is_err());
    }
}

#[test]
fn invalid_relational_rule_names_and_schedule_names_are_fallible() {
    let duplicate = format!("{RELATIONS}\n(rule ((edge a b)) ((path b a)) :name \"base\")");
    assert!(RelationalEngine::new(RulePack::new(duplicate, CostModel::default())).is_err());
    let mut engine = Engine::new(RulePack::new(REWRITES, CostModel::default())).unwrap();
    assert!(engine
        .saturate(Schedule::Run {
            ruleset: "absent".into(),
            iterations: 1
        })
        .is_err());
}

#[test]
fn program_literal_seeds_retain_rule_pack_authority_beside_actual_derivations() {
    let pack = RulePack::new(
        format!("{RELATIONS}\n(edge \"seed\" \"target\")"),
        CostModel::default(),
    );
    let expected_hash = pack.id.clone();
    let engine = RelationalEngine::new(pack).unwrap();
    let receipt = engine.derive(&BTreeMap::new(), 10).unwrap();
    assert_eq!(receipt.rule_pack, expected_hash);
    assert_eq!(
        receipt.program_facts,
        vec![RelationalFact {
            relation: "edge".into(),
            tuple: row("seed", "target")
        }]
    );
    assert_eq!(receipt.relations["path"], vec![row("seed", "target")]);
    let proof = receipt
        .provenance
        .iter()
        .find(|proof| proof.rule == "base")
        .unwrap();
    assert_eq!(proof.premises, receipt.program_facts);
    assert_eq!(
        proof.output,
        RelationalFact {
            relation: "path".into(),
            tuple: row("seed", "target")
        }
    );
}

#[test]
fn repeated_rule_names_across_rulesets_are_refused_without_ambiguous_provenance() {
    let source = r#"(relation P (i64)) (relation R (i64))
        (ruleset a) (ruleset b)
        (rule ((P x)) ((R x)) :ruleset a :name "edge")
        (rule ((P x)) ((R x)) :ruleset b :name "edge")"#;
    assert!(RelationalEngine::new(RulePack::new(source, CostModel::default())).is_err());
}


#[test]
fn actual_f64_term_roundtrip_preserves_scalar_bits() {
    let atom = crate::Atom::F64(10.848666999999999);
    let restored: crate::Atom = serde_json::from_slice(&serde_json::to_vec(&atom).unwrap()).unwrap();
    assert_eq!(restored, atom);
}
