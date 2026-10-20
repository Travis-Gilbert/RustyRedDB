use super::*;
const ABC: &str = r#"
(relation edge (String String String))
(relation shared_intermediate (String String String))
(rule ((edge a b k1) (edge c b k2) (!= a c) (not (edge a c _)))
      ((shared_intermediate a c b)) :name "swanson-abc")
"#;
fn s(value: &str) -> Atom {
    Atom::String(value.into())
}
fn engine(program: &str) -> RelationalEngine {
    RelationalEngine::new(RulePack::new(program, Default::default())).unwrap()
}
fn edges() -> BTreeMap<String, Vec<Vec<Atom>>> {
    BTreeMap::from([(
        "edge".into(),
        vec![vec![s("a"), s("b"), s("k1")], vec![s("c"), s("b"), s("k2")]],
    )])
}

#[test]
fn actual_abc_antijoin_suppresses_any_direct_kind_and_records_absence() {
    let runtime = engine(ABC);
    let input = edges();
    let absent = runtime.derive(&input, 100).unwrap();
    assert!(absent.relations["shared_intermediate"].contains(&vec![s("a"), s("c"), s("b")]));
    let proof = absent
        .provenance
        .iter()
        .find(|p| p.output.tuple == vec![s("a"), s("c"), s("b")])
        .unwrap();
    assert_eq!(proof.premises.len(), 2);
    assert_eq!(proof.negative_premises.len(), 1);
    let negative = &proof.negative_premises[0];
    assert_eq!(negative.relation, "edge");
    assert_eq!(negative.pattern, vec![Some(s("a")), Some(s("c")), None]);
    assert_eq!(negative.rule_pack, absent.rule_pack);
    assert_eq!(negative.stratum, 1);
    let mut direct = input.clone();
    direct
        .get_mut("edge")
        .unwrap()
        .push(vec![s("a"), s("c"), s("other-kind")]);
    let present = runtime.derive(&direct, 100).unwrap();
    assert!(!present.relations["shared_intermediate"].contains(&vec![s("a"), s("c"), s("b")]));
    let reverse = present
        .provenance
        .iter()
        .find(|p| p.output.tuple == vec![s("c"), s("a"), s("b")])
        .unwrap();
    assert_ne!(
        negative.input_facts,
        reverse.negative_premises[0].input_facts
    );
    assert_ne!(
        negative.closed_relation,
        reverse.negative_premises[0].closed_relation
    );
    assert_eq!(
        runtime.derive(&input, 100).unwrap().provenance,
        absent.provenance,
        "invocations are isolated"
    );
    let mut reordered = input;
    reordered.get_mut("edge").unwrap().reverse();
    reordered
        .get_mut("edge")
        .unwrap()
        .push(vec![s("a"), s("b"), s("k1")]);
    assert_eq!(
        runtime.derive(&reordered, 100).unwrap().provenance,
        absent.provenance
    );
}

#[test]
fn lower_positive_recursion_reaches_fixed_point_before_negative_read() {
    let runtime = engine(
        r#"
      (relation edge (String String)) (relation reach (String String))
      (relation candidate (String String)) (relation missing (String String))
      (rule ((edge x y)) ((reach x y)) :name "base")
      (rule ((reach x y) (edge y z)) ((reach x z)) :name "transitive")
      (rule ((candidate x y) (not (reach x y))) ((missing x y)) :name "missing")
    "#,
    );
    let facts = BTreeMap::from([
        (
            "edge".into(),
            vec![vec![s("a"), s("b")], vec![s("b"), s("c")]],
        ),
        (
            "candidate".into(),
            vec![vec![s("a"), s("c")], vec![s("a"), s("d")]],
        ),
    ]);
    let receipt = runtime.derive(&facts, 100).unwrap();
    assert_eq!(receipt.relations["missing"], vec![vec![s("a"), s("d")]]);
    assert_eq!(
        receipt.saturation.stop_reason,
        crate::StopReason::FixedPoint
    );
    assert!(
        receipt
            .saturation
            .matches_per_rule
            .keys()
            .any(|name| name.contains("seal_")),
        "projection ran in actual evaluator"
    );
    let bounded = runtime.derive(&facts, 1).unwrap();
    assert_eq!(
        bounded.saturation.stop_reason,
        crate::StopReason::IterationLimit
    );
    assert!(bounded.relations["missing"].is_empty());
    assert!(bounded
        .provenance
        .iter()
        .all(|p| p.negative_premises.is_empty()));
}

#[test]
fn wildcard_safety_and_negative_cycles_are_refused_before_execution() {
    for program in [
      "(relation P (i64)) (relation Q (i64)) (rule ((not (P x))) ((Q x)))",
      "(relation P (i64)) (relation Q (i64)) (rule ((P x) (not (Q y))) ((Q x)))",
      "(relation P (i64)) (relation Q (i64)) (rule ((P x)) ((Q _)))",
      "(relation P (i64)) (rule ((P x) (not (P x))) ((P x)))",
      "(relation S (i64)) (relation P (i64)) (relation Q (i64)) (rule ((S x) (not (Q x))) ((P x))) (rule ((S x) (not (P x))) ((Q x)))",
    ]{assert!(RelationalEngine::new(RulePack::new(program,Default::default())).is_err(),"{program}");}
    let runtime=engine("(relation P (i64 i64)) (relation Q (i64)) (relation R (i64)) (rule ((P x _) (not (Q _))) ((R x)) :name \"wild\")");
    let facts = BTreeMap::from([("P".into(), vec![vec![Atom::I64(1), Atom::I64(2)]])]);
    assert_eq!(
        runtime.derive(&facts, 100).unwrap().relations["R"],
        vec![vec![Atom::I64(1)]]
    );
}

#[test]
fn literal_seeds_stratified_negation_and_scalar_not_remain_distinct() {
    let runtime = engine(
        r#"
      (relation seed (i64)) (relation blocked (i64)) (relation clear (i64)) (relation flagged (i64))
      (seed 1) (seed 2) (blocked 1)
      (rule ((seed x) (not (blocked x))) ((clear x)) :name "clear")
      (rule ((seed x) (not (clear x)) (= (not false) true)) ((flagged x)) :name "flagged")
    "#,
    );
    let receipt = runtime.derive(&BTreeMap::new(), 100).unwrap();
    assert_eq!(receipt.relations["clear"], vec![vec![Atom::I64(2)]]);
    assert_eq!(receipt.relations["flagged"], vec![vec![Atom::I64(1)]]);
    assert_eq!(
        receipt
            .provenance
            .iter()
            .find(|p| p.rule == "flagged")
            .unwrap()
            .negative_premises[0]
            .stratum,
        2
    );
    assert_eq!(receipt.program_facts.len(), 3);
}

#[test]
fn negative_projection_preserves_scalar_types_repeated_variables_and_anonymous_columns() {
    let runtime = engine(
        r#"
      (relation candidate (String String f64 bool))
      (relation blocked (String String f64 bool i64))
      (relation missing (String String f64 bool))
      (rule ((candidate a b n flag) (not (blocked a b n flag _)))
        ((missing a b n flag)) :name "typed")
    "#,
    );
    let special = "quote\"\\\n\u{1f}";
    let candidate = vec![s(special), s("a,b"), Atom::F64(-0.0), Atom::Bool(false)];
    let distinct = vec![
        s(special),
        s("a,b"),
        Atom::F64(10.848666999999999),
        Atom::Bool(true),
    ];
    let mut block = candidate.clone();
    block[2] = Atom::F64(0.0);
    block.push(Atom::I64(42));
    let facts = BTreeMap::from([
        ("candidate".into(), vec![candidate, distinct.clone()]),
        ("blocked".into(), vec![block]),
    ]);
    let receipt = runtime.derive(&facts, 100).unwrap();
    assert_eq!(receipt.relations["missing"], vec![distinct]);
    assert_eq!(
        serde_json::from_slice::<RelationalReceipt>(&serde_json::to_vec(&receipt).unwrap())
            .unwrap(),
        receipt
    );

    let wildcard = engine(
        r#"
      (relation P (i64 i64)) (relation Q (i64 i64)) (relation R (i64))
      (rule ((P x _) (P x _) (not (Q x x))) ((R x)) :name "repeated")
    "#,
    );
    let facts = BTreeMap::from([
        (
            "P".into(),
            vec![
                vec![Atom::I64(1), Atom::I64(2)],
                vec![Atom::I64(1), Atom::I64(3)],
            ],
        ),
        ("Q".into(), vec![vec![Atom::I64(1), Atom::I64(9)]]),
    ]);
    let receipt = wildcard.derive(&facts, 100).unwrap();
    assert_eq!(receipt.relations["R"], vec![vec![Atom::I64(1)]]);
    assert_eq!(
        receipt.provenance.len(),
        4,
        "each positive underscore is independent"
    );
    assert!(receipt
        .provenance
        .iter()
        .all(|proof| proof.negative_premises[0].pattern
            == vec![Some(Atom::I64(1)), Some(Atom::I64(1))]));
}

#[test]
fn grounded_empty_and_fully_anonymous_absence_are_actual_closed_queries() {
    let program = r#"
      (relation edge (String String)) (relation empty (bool)) (relation missing (bool))
      (rule ((not (edge _ _))) ((empty true)) :name "empty")
      (rule ((not (edge "a" "b"))) ((missing true)) :name "literal")
    "#;
    let runtime = engine(program);
    let absent = runtime.derive(&BTreeMap::new(), 100).unwrap();
    assert_eq!(absent.relations["empty"], vec![vec![Atom::Bool(true)]]);
    assert!(absent
        .provenance
        .iter()
        .all(|p| p.premises.is_empty() && p.negative_premises.len() == 1));
    let present = runtime
        .derive(
            &BTreeMap::from([("edge".into(), vec![vec![s("a"), s("b")]])]),
            100,
        )
        .unwrap();
    assert!(present.relations["empty"].is_empty());
    assert!(present.relations["missing"].is_empty());
}

#[test]
fn abc_matches_reference_relation_for_every_three_node_directed_graph() {
    let runtime = engine(ABC);
    let vertices = ["a", "b", "c"];
    let possible: Vec<_> = vertices
        .iter()
        .flat_map(|a| {
            vertices
                .iter()
                .filter(move |c| a != *c)
                .map(move |c| (*a, *c))
        })
        .collect();
    for mask in 0_u64..(1 << possible.len()) {
        let selected: BTreeSet<_> = possible
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, edge)| *edge)
            .collect();
        let facts = BTreeMap::from([(
            "edge".into(),
            selected
                .iter()
                .enumerate()
                .map(|(i, (a, b))| {
                    vec![s(a), s(b), s(if i % 2 == 0 { "kind-a" } else { "kind-b" })]
                })
                .collect(),
        )]);
        let actual = runtime.derive(&facts, 100).unwrap();
        assert_eq!(actual.saturation.stop_reason, crate::StopReason::FixedPoint);
        let mut expected = Vec::new();
        for a in vertices {
            for c in vertices {
                for b in vertices {
                    if a != c
                        && selected.contains(&(a, b))
                        && selected.contains(&(c, b))
                        && !selected.contains(&(a, c))
                    {
                        expected.push(vec![s(a), s(c), s(b)]);
                    }
                }
            }
        }
        expected.sort_by_cached_key(canonical);
        assert_eq!(
            actual.relations["shared_intermediate"], expected,
            "graph mask {mask}"
        );
        assert!(actual
            .provenance
            .iter()
            .all(|proof| proof.negative_premises.len() == 1));
    }
    let interrupted_freeze = runtime.derive(&edges(), 2).unwrap();
    assert_eq!(
        interrupted_freeze.saturation.stop_reason,
        crate::StopReason::IterationLimit
    );
    assert!(interrupted_freeze.relations["shared_intermediate"].is_empty());
    assert!(interrupted_freeze.provenance.is_empty());
}
