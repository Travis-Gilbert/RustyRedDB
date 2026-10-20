//! Acceptance criteria for SPEC-INCREMENTAL-DERIVATION-AND-RETRACTION-1.0.

use std::collections::{BTreeMap, BTreeSet};

use rustyred_thg_core::abduction::{
    get_hypothesis_space, put_explanation, renormalize_over_survivors, retract_assumption,
    Assumable, Explanation,
};
use rustyred_thg_core::incremental::{
    apply_delta, derivation_delta_consumer_shape_note, distinct, distinct_delta, edge_fact,
    install_rules, path_fact, path_program, present_facts, supported_fact, Atom, FactDelta, Rule,
};
use rustyred_thg_core::provenance_semiring::{
    bounded_recursive_semiring_derivation, ProvenanceSemiring, ProvenanceSemiringKind,
    RecursiveSemiringEdge,
};
use rustyred_thg_core::InMemoryGraphStore;

fn nonrecursive_program() -> Vec<Rule> {
    vec![
        Rule::new("derive_q", Atom::new("q", ["X"]), [Atom::new("p", ["X"])]),
        Rule::new(
            "derive_q_from_r",
            Atom::new("q", ["X"]),
            [Atom::new("r", ["X"])],
        ),
    ]
}

#[test]
fn ac1_add_base_fact_touches_dependency_cone_not_store_size() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();

    // 10000 unrelated base facts that justify nothing under the installed rules.
    let mut unrelated = Vec::new();
    for i in 0..10_000 {
        let fact = format!("unrelated({i})");
        unrelated.push(FactDelta::add(fact));
    }
    apply_delta(&mut store, &unrelated).unwrap();

    let delta = apply_delta(&mut store, &[FactDelta::add("p(a)")]).unwrap();
    assert_eq!(delta.added, vec!["q(a)".to_string()]);
    assert!(
        delta.facts_touched < 100,
        "facts_touched {} should be bounded by the dependency cone, not store size",
        delta.facts_touched
    );
    assert!(present_facts(&store).unwrap().contains("q(a)"));
}

#[test]
fn ac2_retract_single_justification_removes_consequence() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(&mut store, &[FactDelta::add("p(a)")]).unwrap();
    assert!(present_facts(&store).unwrap().contains("q(a)"));

    let delta = apply_delta(&mut store, &[FactDelta::remove("p(a)")]).unwrap();
    assert!(delta.removed.contains(&"q(a)".to_string()));
    assert!(!present_facts(&store).unwrap().contains("q(a)"));
}

#[test]
fn ac3_retract_one_of_two_justifications_keeps_multiplicity_one() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(
        &mut store,
        &[FactDelta::add("p(a)"), FactDelta::add("r(a)")],
    )
    .unwrap();
    let before = supported_fact(&store, "q(a)").unwrap().expect("q(a)");
    assert!(before.multiplicity >= 2);

    apply_delta(&mut store, &[FactDelta::remove("p(a)")]).unwrap();
    let after = supported_fact(&store, "q(a)")
        .unwrap()
        .expect("q(a) remains");
    assert_eq!(after.multiplicity, 1);
    assert!(present_facts(&store).unwrap().contains("q(a)"));
}

#[test]
fn ac4_distinct_clamps_zset() {
    let mut facts = BTreeMap::from([
        ("a".to_string(), 3),
        ("b".to_string(), 0),
        ("c".to_string(), -2),
    ]);
    distinct(&mut facts);
    assert_eq!(facts, BTreeMap::from([("a".to_string(), 1)]));
}

#[test]
fn ac5_distinct_delta_nonlinear_zero_boundary() {
    let previous = BTreeMap::from([("a".to_string(), 1)]);
    let next_same_side = BTreeMap::from([("a".to_string(), 4)]);
    assert!(distinct_delta(&previous, &next_same_side).is_empty());

    let next_zero = BTreeMap::from([("a".to_string(), 0)]);
    assert_eq!(
        distinct_delta(&previous, &next_zero),
        BTreeMap::from([("a".to_string(), -1)])
    );
}

#[test]
fn ac6_cycle_retract_uses_dred_not_counting() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &path_program()).unwrap();
    apply_delta(
        &mut store,
        &[
            FactDelta::add(edge_fact("a", "b")),
            FactDelta::add(edge_fact("b", "c")),
            FactDelta::add(edge_fact("c", "a")),
        ],
    )
    .unwrap();

    let before = present_facts(&store).unwrap();
    assert!(before.contains(&path_fact("a", "b")));
    assert!(before.contains(&path_fact("a", "a")));

    let delta = apply_delta(&mut store, &[FactDelta::remove(edge_fact("a", "b"))]).unwrap();
    let after = present_facts(&store).unwrap();

    assert!(!after.contains(&path_fact("a", "b")));
    assert!(after.contains(&path_fact("b", "c")));
    assert!(after.contains(&path_fact("c", "a")));
    assert!(after.contains(&path_fact("b", "a")));
    assert!(
        delta.over_deleted > delta.removed.len(),
        "over_deleted {} should exceed removed {} (DRed re-derived some facts)",
        delta.over_deleted,
        delta.removed.len()
    );
}

#[test]
fn ac7_delta_then_inverse_round_trips_byte_identical() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();

    let mut rng = 0xC0FFEE_u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng
    };

    for _ in 0..100 {
        let original = present_facts(&store).unwrap();
        let label = (next() % 50) as u32;
        let use_p = next() % 2 == 0;
        let fact = if use_p {
            format!("p(x{label})")
        } else {
            format!("r(x{label})")
        };
        let add = FactDelta::add(fact.clone());
        apply_delta(&mut store, &[add]).unwrap();
        apply_delta(&mut store, &[FactDelta::remove(fact)]).unwrap();
        assert_eq!(present_facts(&store).unwrap(), original);
    }
}

#[test]
fn ac8_no_nonpositive_multiplicity_at_rest() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(
        &mut store,
        &[FactDelta::add("p(a)"), FactDelta::add("r(a)")],
    )
    .unwrap();
    apply_delta(&mut store, &[FactDelta::remove("p(a)")]).unwrap();
    for fact_id in present_facts(&store).unwrap() {
        let fact = supported_fact(&store, &fact_id).unwrap().unwrap();
        assert!(
            fact.multiplicity > 0,
            "{fact_id} multiplicity {}",
            fact.multiplicity
        );
    }
}

#[test]
fn ac9_abduction_retract_invalidates_and_weakens_without_renormalizing() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(
        &mut store,
        &[
            FactDelta::add("p(shared)"),
            FactDelta::add("r(b)"),
            FactDelta::add("r(c)"),
        ],
    )
    .unwrap();

    put_explanation(
        &mut store,
        Explanation::new(
            "expl:a",
            vec![Assumable::new("p(shared)")],
            vec!["q(shared)".to_string()],
            1.0,
            vec!["q(shared)".to_string()],
        ),
    )
    .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:b",
            vec![Assumable::new("p(shared)"), Assumable::new("r(b)")],
            vec!["q(b)".to_string()],
            3.0,
            vec!["q(shared)".to_string(), "q(b)".to_string()],
        ),
    )
    .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:c",
            vec![Assumable::new("r(c)")],
            vec!["q(c)".to_string()],
            1.0,
            vec!["q(c)".to_string()],
        ),
    )
    .unwrap();

    let prior = get_hypothesis_space(&store).unwrap();
    let prior_posteriors = prior.active_posteriors();

    let report = retract_assumption(&mut store, "expl:a", &Assumable::new("p(shared)")).unwrap();
    assert!(report
        .explanations_invalidated
        .contains(&"expl:a".to_string()));
    assert!(report.explanations_weakened.contains(&"expl:b".to_string()));
    assert!(!report
        .explanations_invalidated
        .contains(&"expl:c".to_string()));
    assert!(!report.explanations_weakened.contains(&"expl:c".to_string()));

    let after = get_hypothesis_space(&store).unwrap();
    let survivors: BTreeSet<String> = after
        .explanations
        .iter()
        .filter(|(_, e)| e.status == rustyred_thg_core::abduction::ExplanationStatus::Active)
        .map(|(id, _)| id.clone())
        .collect();
    let renormalized = renormalize_over_survivors(&prior_posteriors, &survivors);
    let b_recomputed = after.explanations["expl:b"].posterior;
    let b_renorm = renormalized["expl:b"];
    assert!(
        (b_recomputed - b_renorm).abs() > 1e-9,
        "recomputed posterior {b_recomputed} must differ from renormalized {b_renorm}"
    );
}

#[test]
fn ac10_surviving_entailed_observations_neither_invalidated_nor_weakened() {
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(
        &mut store,
        &[FactDelta::add("p(a)"), FactDelta::add("r(b)")],
    )
    .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:keep",
            vec![Assumable::new("r(b)")],
            vec!["q(b)".to_string()],
            1.0,
            vec!["q(b)".to_string()],
        ),
    )
    .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:drop",
            vec![Assumable::new("p(a)")],
            vec!["q(a)".to_string()],
            1.0,
            vec!["q(a)".to_string()],
        ),
    )
    .unwrap();

    let report = retract_assumption(&mut store, "expl:drop", &Assumable::new("p(a)")).unwrap();
    assert!(report
        .explanations_invalidated
        .contains(&"expl:drop".to_string()));
    assert!(!report
        .explanations_invalidated
        .contains(&"expl:keep".to_string()));
    assert!(!report
        .explanations_weakened
        .contains(&"expl:keep".to_string()));
}

#[test]
fn ac11_provenance_gates_unchanged_and_distinct_safe() {
    assert!(ProvenanceSemiring::new(ProvenanceSemiringKind::Viterbi).is_absorptive());
    assert!(!ProvenanceSemiring::new(ProvenanceSemiringKind::Counting).is_absorptive());
    assert!(ProvenanceSemiringKind::Counting.recursion_safe_with_distinct());

    let err = bounded_recursive_semiring_derivation(
        &[RecursiveSemiringEdge {
            from_id: "a".to_string(),
            to_id: "b".to_string(),
            value: 1.0,
        }],
        &BTreeMap::from([("a".to_string(), 1.0)]),
        ProvenanceSemiringKind::Counting,
        2,
    )
    .expect_err("counting still rejected on bounded_recursive path");
    assert_eq!(err.code, "recursive_semiring_not_absorptive");
}

#[test]
fn ac12_apply_delta_deterministic_across_runs() {
    let changes = vec![
        FactDelta::add("p(a)"),
        FactDelta::add("r(a)"),
        FactDelta::add("p(b)"),
        FactDelta::remove("p(a)"),
    ];
    let mut fingerprints = BTreeSet::new();
    for _ in 0..100 {
        let mut store = InMemoryGraphStore::new();
        install_rules(&mut store, &nonrecursive_program()).unwrap();
        for change in &changes {
            apply_delta(&mut store, std::slice::from_ref(change)).unwrap();
        }
        let present = present_facts(&store).unwrap();
        fingerprints.insert(present.into_iter().collect::<Vec<_>>().join("|"));
    }
    assert_eq!(fingerprints.len(), 1);
}

#[test]
fn d5_delta_shape_fits_standing_refinery_read_model() {
    let note = derivation_delta_consumer_shape_note();
    assert_eq!(note["wired"], false);
    assert!(note["standing"].as_str().unwrap().contains("added/removed"));
}


#[test]
fn unchanged_index_observations_preserve_graph_revision_and_latest_receipt() {
    use rustyred_thg_core::incremental::{
        ensure_base_fact, last_derivation_receipt, record_justified_fact, Justification,
    };
    let mut store = InMemoryGraphStore::new();
    install_rules(&mut store, &nonrecursive_program()).unwrap();
    apply_delta(&mut store, &[FactDelta::add("p(a)")]).unwrap();
    let receipt = serde_json::to_value(last_derivation_receipt(&store).unwrap()).unwrap();
    assert!(!receipt.is_null());
    let before = store.stats();
    ensure_base_fact(&mut store, "p(a)").unwrap();
    assert_eq!(store.stats(), before);

    let proof = Justification { rule_id: "explicit-proof".into(), premises: vec!["p(a)".into()] };
    record_justified_fact(&mut store, "s(a)", proof.clone(), false).unwrap();
    let after_new = store.stats();
    assert!(after_new.version > before.version);
    assert!(present_facts(&store).unwrap().contains("s(a)"));
    record_justified_fact(&mut store, "s(a)", proof, false).unwrap();
    assert_eq!(store.stats(), after_new);
    assert_eq!(serde_json::to_value(last_derivation_receipt(&store).unwrap()).unwrap(), receipt);
}
