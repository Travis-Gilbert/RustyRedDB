//! Bridges from saturation, derivation, and epistemic graphs into incremental maintenance.
//!
//! Closes the architectural gap where the fixture Datalog engine existed beside
//! RustyRed's saturation and derivation paths without sharing multiplicity state.

use std::collections::{BTreeSet, VecDeque};

use crate::epistemic::EPISTEMIC_SUPPORTS;
use crate::graph::capability::GraphStore;
use crate::graph::query::NeighborQuery;
use crate::incremental::delta::{
    apply_delta, apply_distinct_for_predicates, drop_derived_facts, ensure_base_fact,
    ensure_observational_or_active_premise, present_facts, projected_saturation_facts,
    record_justified_fact, record_justified_fact_with_relation, store_saturation_projection,
    DerivationDelta, FactDelta,
};
use crate::incremental::support::Justification;
use crate::saturation::{SaturationClosure, SaturationConfig, SaturationDerivedStatement};
use crate::GraphStoreResult;

const CLAIM_DEPENDENCY_EDGE_TYPES: [&str; 4] = [
    "ClaimDependency",
    "DEPENDS_ON",
    "DependsOn",
    "claim_dependency",
];

/// After saturation materializes statements, record each as a justified derived
/// fact, drop statements the new closure omitted, and retract any configured
/// retracted base facts through `apply_delta`.
pub fn sync_saturation_materialize<S: GraphStore + ?Sized>(
    store: &mut S,
    closure: &SaturationClosure,
    config: &SaturationConfig,
) -> GraphStoreResult<Option<DerivationDelta>> {
    let live_ids: BTreeSet<String> = closure
        .derived_statements
        .iter()
        .map(|statement| statement.fact_id.clone())
        .collect();
    let prior = projected_saturation_facts(store, &closure.component_key)?;
    let stale: Vec<String> = prior.difference(&live_ids).cloned().collect();

    for statement in &closure.derived_statements {
        record_saturation_statement(store, statement)?;
    }

    apply_distinct_for_predicates(
        store,
        &closure
            .derived_statements
            .iter()
            .map(|statement| statement.relation.clone())
            .collect::<BTreeSet<_>>(),
    )?;

    let mut combined = DerivationDelta::empty();
    if !stale.is_empty() {
        merge_delta(&mut combined, drop_derived_facts(store, &stale)?);
    }
    store_saturation_projection(store, &closure.component_key, live_ids)?;

    if !config.retracted_fact_ids.is_empty() {
        let deltas: Vec<FactDelta> = config
            .retracted_fact_ids
            .iter()
            .map(|id| FactDelta::remove(id.clone()))
            .collect();
        merge_delta(&mut combined, apply_delta(store, &deltas)?);
    }

    if combined.added.is_empty() && combined.removed.is_empty() && combined.facts_touched == 0 {
        return Ok(None);
    }
    Ok(Some(combined))
}

fn merge_delta(into: &mut DerivationDelta, from: DerivationDelta) {
    into.added.extend(from.added);
    into.removed.extend(from.removed);
    into.added.sort();
    into.added.dedup();
    into.removed.sort();
    into.removed.dedup();
    into.iterations = into.iterations.max(from.iterations);
    into.facts_touched += from.facts_touched;
    into.over_deleted += from.over_deleted;
}

pub fn record_saturation_statement<S: GraphStore + ?Sized>(
    store: &mut S,
    statement: &SaturationDerivedStatement,
) -> GraphStoreResult<()> {
    let justifications = saturation_justifications(statement);
    let mut premises = BTreeSet::new();
    for justification in &justifications {
        premises.extend(justification.premises.iter().cloned());
    }
    for dep in premises {
        ensure_base_fact(store, &dep)?;
    }
    for justification in justifications {
        record_justified_fact_with_relation(
            store,
            &statement.fact_id,
            justification,
            false,
            Some(&statement.relation),
        )?;
    }
    Ok(())
}

fn saturation_justifications(statement: &SaturationDerivedStatement) -> Vec<Justification> {
    if statement.contributors.is_empty() {
        return vec![Justification {
            rule_id: statement.rule_id.clone(),
            premises: statement.dependency_fact_ids.clone(),
        }];
    }
    statement
        .contributors
        .iter()
        .map(|contributor| Justification {
            rule_id: statement.rule_id.clone(),
            premises: contributor.dependency_fact_ids.clone(),
        })
        .collect()
}

/// Record a derivation record's outputs as justified facts over input+assumption premises.
pub fn record_derivation_outputs<S: GraphStore + ?Sized>(
    store: &mut S,
    derivation_id: &str,
    input_refs: &[String],
    assumption_refs: &[String],
    output_refs: &[String],
) -> GraphStoreResult<()> {
    let mut premises = Vec::new();
    premises.extend(input_refs.iter().cloned());
    premises.extend(assumption_refs.iter().cloned());
    for premise in &premises {
        ensure_observational_or_active_premise(store, premise)?;
    }
    let present = present_facts(store)?;
    if !premises.iter().all(|premise| present.contains(premise)) {
        return Ok(());
    }
    for output in output_refs {
        record_justified_fact(
            store,
            output,
            Justification {
                rule_id: derivation_id.to_string(),
                premises: premises.clone(),
            },
            false,
        )?;
    }
    Ok(())
}

/// Retract an assumption through incremental maintenance and return removed fact ids.
pub fn retract_assumption_facts<S: GraphStore + ?Sized>(
    store: &mut S,
    assumption_id: &str,
) -> GraphStoreResult<DerivationDelta> {
    ensure_base_fact(store, assumption_id)?;
    apply_delta(store, &[FactDelta::remove(assumption_id.to_string())])
}

/// Project epistemic Supports / claim-dependency edges as derived justifications.
///
/// BFS follows only the dependent direction. Each discovered claim records
/// **every** incoming supporter as its own justification so independent
/// sources keep the claim present after one supporter is retracted.
pub fn project_epistemic_justifications<S: GraphStore + ?Sized>(
    store: &mut S,
    seed_id: &str,
) -> GraphStoreResult<Vec<String>> {
    let ordered = {
        let edges = epistemic_support_edges(store, seed_id)?;
        let mut seen = BTreeSet::new();
        let mut order = Vec::new();
        for (dependent, _) in edges {
            if seen.insert(dependent.clone()) {
                order.push(dependent);
            }
        }
        order
    };
    for dependent in &ordered {
        let mut supporters = BTreeSet::new();
        for hit in store.neighbors(NeighborQuery::in_(dependent).with_edge_type(EPISTEMIC_SUPPORTS))
        {
            supporters.insert(hit.node_id);
        }
        for edge_type in CLAIM_DEPENDENCY_EDGE_TYPES {
            for hit in store.neighbors(NeighborQuery::out(dependent).with_edge_type(edge_type)) {
                supporters.insert(hit.node_id);
            }
        }
        for supporter in supporters {
            ensure_observational_or_active_premise(store, &supporter)?;
            if !present_facts(store)?.contains(&supporter) {
                continue;
            }
            record_justified_fact(
                store,
                dependent,
                Justification {
                    rule_id: "epistemic:Supports".to_string(),
                    premises: vec![supporter],
                },
                false,
            )?;
        }
    }
    Ok(ordered)
}

/// Walk the dependent cone: outgoing `Supports` (supporter → claim) and
/// incoming claim-dependency edges (dependent → this node).
pub fn epistemic_support_dependents<S: GraphStore + ?Sized>(
    store: &S,
    seed_id: &str,
) -> GraphStoreResult<Vec<String>> {
    let mut dependents: Vec<String> = epistemic_support_edges(store, seed_id)?
        .into_iter()
        .map(|(dependent, _)| dependent)
        .collect();
    dependents.sort();
    dependents.dedup();
    Ok(dependents)
}

/// `(dependent, premise)` pairs in the dependent-only cone.
pub fn epistemic_support_edges<S: GraphStore + ?Sized>(
    store: &S,
    seed_id: &str,
) -> GraphStoreResult<Vec<(String, String)>> {
    let mut seen = BTreeSet::from([seed_id.to_string()]);
    let mut queue = VecDeque::from([seed_id.to_string()]);
    let mut edges = Vec::new();
    while let Some(node_id) = queue.pop_front() {
        for hit in store.neighbors(NeighborQuery::out(&node_id).with_edge_type(EPISTEMIC_SUPPORTS))
        {
            edges.push((hit.node_id.clone(), node_id.clone()));
            if seen.insert(hit.node_id.clone()) {
                queue.push_back(hit.node_id);
            }
        }
        for edge_type in CLAIM_DEPENDENCY_EDGE_TYPES {
            for hit in store.neighbors(NeighborQuery::in_(&node_id).with_edge_type(edge_type)) {
                edges.push((hit.node_id.clone(), node_id.clone()));
                if seen.insert(hit.node_id.clone()) {
                    queue.push_back(hit.node_id);
                }
            }
        }
    }
    Ok(edges)
}

/// Rebuild explanation support ids from facts still present in the incremental index.
pub fn surviving_support(
    store: &(impl GraphStore + ?Sized),
    prior_support: &[String],
) -> GraphStoreResult<Vec<String>> {
    let present = present_facts(store)?;
    Ok(prior_support
        .iter()
        .filter(|id| present.contains(*id))
        .cloned()
        .collect())
}
