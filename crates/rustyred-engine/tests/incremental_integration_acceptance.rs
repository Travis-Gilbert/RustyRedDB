//! Integration ACs for saturation / derivation / epistemic bridges (P4).
//!
//! These exercise the native GraphStore paths, not only `install_rules` fixtures.

use rustyred_thg_core::abduction::{
    get_hypothesis_space, put_explanation, retract_assumption as abduction_retract, Assumable,
    Explanation, ExplanationStatus,
};
use rustyred_thg_core::epistemic::EPISTEMIC_SUPPORTS;
use rustyred_thg_core::incremental::{
    apply_delta, ensure_base_fact, present_facts, supported_fact, FactDelta,
};
use rustyred_thg_core::statement::StatementSemiring;
use rustyred_thg_core::{
    materialize_closure, put_assumption, put_derivation, retract_assumption, AssumptionRecord,
    DerivationAnnotation, DerivationKind, DerivationSpec, EdgeRecord, InMemoryGraphStore,
    NodeRecord, SaturationClosure, SaturationConfig, SaturationContributor,
    SaturationDerivedStatement,
};
use serde_json::json;

fn statement(
    fact_id: &str,
    deps: Vec<&str>,
    contributors: Vec<Vec<&str>>,
) -> SaturationDerivedStatement {
    SaturationDerivedStatement {
        fact_id: fact_id.to_string(),
        rule_id: "support_reachability".to_string(),
        relation: "support_reachable".to_string(),
        subject_id: "subj:a".to_string(),
        reason: "integration".to_string(),
        dependency_fact_ids: deps.into_iter().map(str::to_string).collect(),
        contributors: contributors
            .into_iter()
            .map(|deps| SaturationContributor {
                dependency_fact_ids: deps.into_iter().map(str::to_string).collect(),
                confidence: 1.0,
            })
            .collect(),
        attributes: json!({}),
        confidence: 1.0,
        writeback_policy: "always".to_string(),
        engine: "test".to_string(),
    }
}

fn closure(statements: Vec<SaturationDerivedStatement>) -> SaturationClosure {
    SaturationClosure {
        component_key: "cmp:test".to_string(),
        derived_statements: statements,
        equivalence_classes: Vec::new(),
        iterations: 1,
        egglog_tuple_count: 0,
        egglog_output_count: 1,
        egglog_iteration_budget: 8,
        egglog_checked_pair_count: 0,
        egglog_error: None,
        semiring: StatementSemiring::Counting,
        ..SaturationClosure::default()
    }
}

#[test]
fn p1_saturation_materialize_projects_and_retracts_incremental_index() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("subj:a", ["Object"], json!({})))
        .unwrap();

    let derived = statement("derived:support:a", vec!["dep:base"], Vec::new());
    materialize_closure(
        &mut store,
        closure(vec![derived.clone()]),
        SaturationConfig::default(),
    )
    .unwrap();
    assert!(
        present_facts(&store).unwrap().contains("derived:support:a"),
        "saturation writeback must project derived facts into the incremental index"
    );
    let supported = supported_fact(&store, "derived:support:a")
        .unwrap()
        .expect("supported");
    assert_eq!(supported.relation.as_deref(), Some("support_reachable"));
    assert!(supported
        .justifications
        .iter()
        .any(|j| j.premises.contains(&"dep:base".to_string())));

    materialize_closure(
        &mut store,
        closure(Vec::new()),
        SaturationConfig {
            retracted_fact_ids: vec!["dep:base".to_string()],
            ..SaturationConfig::default()
        },
    )
    .unwrap();

    assert!(
        !present_facts(&store).unwrap().contains("derived:support:a"),
        "retracting a saturation dependency must un-derive via apply_delta"
    );
}

#[test]
fn p1_alternative_contributors_survive_single_path_retract() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("subj:a", ["Object"], json!({})))
        .unwrap();

    let derived = statement(
        "derived:a-to-d",
        vec!["dep:ab", "dep:ac", "dep:bd", "dep:cd"],
        vec![vec!["dep:ab", "dep:bd"], vec!["dep:ac", "dep:cd"]],
    );
    materialize_closure(
        &mut store,
        closure(vec![derived]),
        SaturationConfig::default(),
    )
    .unwrap();
    let supported = supported_fact(&store, "derived:a-to-d")
        .unwrap()
        .expect("supported");
    assert_eq!(supported.justifications.len(), 2);
    assert_eq!(supported.multiplicity, 2);

    apply_delta(&mut store, &[FactDelta::remove("dep:ab")]).unwrap();

    let supported = supported_fact(&store, "derived:a-to-d")
        .unwrap()
        .expect("independent path must keep the fact present");
    assert!(supported.is_present());
    assert_eq!(supported.multiplicity, 1);
    assert_eq!(supported.justifications.len(), 1);
    assert!(supported.justifications[0]
        .premises
        .iter()
        .all(|p| p != "dep:ab"));
}

#[test]
fn p1_prune_without_retracted_ids_drops_stale_incremental_facts() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("subj:a", ["Object"], json!({})))
        .unwrap();
    materialize_closure(
        &mut store,
        closure(vec![statement(
            "derived:support:a",
            vec!["dep:base"],
            Vec::new(),
        )]),
        SaturationConfig::default(),
    )
    .unwrap();
    assert!(present_facts(&store).unwrap().contains("derived:support:a"));

    materialize_closure(&mut store, closure(Vec::new()), SaturationConfig::default()).unwrap();

    assert!(
        !present_facts(&store).unwrap().contains("derived:support:a"),
        "rematerialize with prune and empty retracted_fact_ids must still drop stale incremental facts"
    );
}

#[test]
fn p2_derivation_retract_drives_incremental_maintenance() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new(
            "value:in",
            ["Value"],
            json!({ "value": "in" }),
        ))
        .unwrap();
    store
        .upsert_node(NodeRecord::new(
            "value:out",
            ["Value"],
            json!({ "value": "out" }),
        ))
        .unwrap();
    put_assumption(
        &mut store,
        AssumptionRecord::active("assumption:v1", "contract v1"),
    )
    .unwrap();
    put_derivation(
        &mut store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: "operator:derive".to_string(),
            input_refs: vec!["value:in".to_string()],
            output_refs: vec!["value:out".to_string()],
            assumption_refs: vec!["assumption:v1".to_string()],
            params: json!({}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation::default(),
        },
    )
    .unwrap();

    assert!(present_facts(&store).unwrap().contains("value:out"));

    let impacted = retract_assumption(&mut store, "assumption:v1").unwrap();
    assert!(impacted.iter().any(|id| id == "value:out"));
    assert!(
        !present_facts(&store).unwrap().contains("value:out"),
        "derivation::retract_assumption must remove outputs from the incremental present set"
    );

    put_derivation(
        &mut store,
        DerivationSpec {
            kind: DerivationKind::Functional,
            operator_id: "operator:derive".to_string(),
            input_refs: vec!["value:in".to_string()],
            output_refs: vec!["value:out".to_string()],
            assumption_refs: vec!["assumption:v1".to_string()],
            params: json!({}),
            environment_version: "test:v1".to_string(),
            annotation: DerivationAnnotation::default(),
        },
    )
    .unwrap();
    assert!(
        !present_facts(&store).unwrap().contains("value:out"),
        "put_derivation must not resurrect an inactive assumption as a base premise"
    );
}

#[test]
fn p3_epistemic_supports_cone_retracts_without_datalog_rules() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("fact:assumed", ["Fact"], json!({})))
        .unwrap();
    store
        .upsert_node(NodeRecord::new("fact:dependent", ["Fact"], json!({})))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "edge:supports",
            "fact:assumed",
            EPISTEMIC_SUPPORTS,
            "fact:dependent",
            json!({}),
        ))
        .unwrap();

    put_explanation(
        &mut store,
        Explanation::new(
            "expl:1",
            vec![Assumable::new("fact:assumed")],
            vec!["obs:1".to_string()],
            1.0,
            vec!["fact:assumed".to_string(), "fact:dependent".to_string()],
        ),
    )
    .unwrap();

    ensure_base_fact(&mut store, "fact:assumed").unwrap();
    apply_delta(&mut store, &[FactDelta::add("obs:1")]).unwrap();

    let report = abduction_retract(&mut store, "expl:1", &Assumable::new("fact:assumed")).unwrap();

    assert!(
        report.facts_removed.iter().any(|id| id == "fact:dependent"),
        "epistemic Supports walk must include dependents even without Datalog rules: {:?}",
        report.facts_removed
    );
    assert!(!present_facts(&store).unwrap().contains("fact:dependent"));
}

#[test]
fn p3_independent_support_survives_and_chain_retracts() {
    let mut store = InMemoryGraphStore::new();
    for id in ["fact:a", "fact:b", "fact:c", "fact:d", "fact:s"] {
        store
            .upsert_node(NodeRecord::new(id, ["Fact"], json!({})))
            .unwrap();
    }
    store
        .upsert_edge(EdgeRecord::new(
            "e:a-c",
            "fact:a",
            EPISTEMIC_SUPPORTS,
            "fact:c",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e:b-c",
            "fact:b",
            EPISTEMIC_SUPPORTS,
            "fact:c",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e:c-d",
            "fact:c",
            EPISTEMIC_SUPPORTS,
            "fact:d",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e:s-a",
            "fact:s",
            EPISTEMIC_SUPPORTS,
            "fact:a",
            json!({}),
        ))
        .unwrap();

    put_explanation(
        &mut store,
        Explanation::new(
            "expl:a",
            vec![Assumable::new("fact:a")],
            vec!["obs:keep".to_string()],
            1.0,
            vec!["fact:a".to_string()],
        ),
    )
    .unwrap();
    apply_delta(&mut store, &[FactDelta::add("obs:keep")]).unwrap();

    let report = abduction_retract(&mut store, "expl:a", &Assumable::new("fact:a")).unwrap();
    let present = present_facts(&store).unwrap();
    assert!(
        present.contains("fact:c"),
        "C must remain via independent supporter B: removed {:?}",
        report.facts_removed
    );
    assert!(
        present.contains("fact:b"),
        "independent source B must not be retracted"
    );
    assert!(
        !report.facts_removed.iter().any(|id| id == "fact:s"),
        "incoming supporter S of A must not be retracted: {:?}",
        report.facts_removed
    );
    assert!(
        present.contains("fact:d"),
        "D depends on C; C survived so D stays"
    );
}

#[test]
fn p3_multi_hop_cone_retracts_when_no_independent_support() {
    let mut store = InMemoryGraphStore::new();
    for id in ["fact:a", "fact:c", "fact:d"] {
        store
            .upsert_node(NodeRecord::new(id, ["Fact"], json!({})))
            .unwrap();
    }
    store
        .upsert_edge(EdgeRecord::new(
            "e:a-c",
            "fact:a",
            EPISTEMIC_SUPPORTS,
            "fact:c",
            json!({}),
        ))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e:c-d",
            "fact:c",
            EPISTEMIC_SUPPORTS,
            "fact:d",
            json!({}),
        ))
        .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:a",
            vec![Assumable::new("fact:a")],
            vec!["obs:keep".to_string()],
            1.0,
            vec![
                "fact:a".to_string(),
                "fact:c".to_string(),
                "fact:d".to_string(),
            ],
        ),
    )
    .unwrap();
    apply_delta(&mut store, &[FactDelta::add("obs:keep")]).unwrap();

    let report = abduction_retract(&mut store, "expl:a", &Assumable::new("fact:a")).unwrap();
    assert!(report.facts_removed.iter().any(|id| id == "fact:c"));
    assert!(
        report.facts_removed.iter().any(|id| id == "fact:d"),
        "multi-hop dependent D must retract: {:?}",
        report.facts_removed
    );
}

#[test]
fn p3_explanation_not_assuming_seed_invalidates_when_entailed_claim_drops() {
    let mut store = InMemoryGraphStore::new();
    store
        .upsert_node(NodeRecord::new("fact:a", ["Fact"], json!({})))
        .unwrap();
    store
        .upsert_node(NodeRecord::new("fact:c", ["Fact"], json!({})))
        .unwrap();
    store
        .upsert_edge(EdgeRecord::new(
            "e:a-c",
            "fact:a",
            EPISTEMIC_SUPPORTS,
            "fact:c",
            json!({}),
        ))
        .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:a",
            vec![Assumable::new("fact:a")],
            vec!["obs:keep".to_string()],
            1.0,
            vec!["fact:a".to_string()],
        ),
    )
    .unwrap();
    put_explanation(
        &mut store,
        Explanation::new(
            "expl:b",
            vec![Assumable::new("fact:b")],
            vec!["fact:c".to_string()],
            2.0,
            vec!["fact:b".to_string(), "fact:c".to_string()],
        ),
    )
    .unwrap();
    apply_delta(&mut store, &[FactDelta::add("obs:keep")]).unwrap();

    let report = abduction_retract(&mut store, "expl:a", &Assumable::new("fact:a")).unwrap();
    assert!(
        report
            .explanations_invalidated
            .iter()
            .any(|id| id == "expl:b"),
        "explanation that entails removed claim C must falsify even if it did not assume A: {:?}",
        report
    );
    let space = get_hypothesis_space(&store).unwrap();
    assert_eq!(
        space.explanations.get("expl:b").unwrap().status,
        ExplanationStatus::Falsified
    );
}
