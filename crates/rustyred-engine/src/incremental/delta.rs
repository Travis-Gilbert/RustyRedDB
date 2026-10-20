//! Incremental derivation delta application (counting + DRed).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::incremental::rules::{
    fact_predicate, fire_rule_full, fire_rule_with_delta, format_fact, parse_fact, Atom, Rule,
    RuleProgram,
};
use crate::incremental::support::{distinct_delta, Justification, Multiplicity, SupportedFact};
use crate::provenance_semiring::ProvenanceSemiringKind;
use crate::query_receipt::{
    QueryKind, QueryOutcomeLabel, QueryReceipt, ReceiptScope, TheoremReceiptFields,
};
use crate::state::stable_hash;
use crate::{now_ms, GraphStore, GraphStoreError, GraphStoreResult, NodeRecord};

pub const INCREMENTAL_INDEX_NODE_ID: &str = "incremental:derivation_index";
pub const INCREMENTAL_INDEX_LABEL: &str = "IncrementalDerivationIndex";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FactDelta {
    pub fact_id: String,
    pub change: Multiplicity,
}

impl FactDelta {
    pub fn add(fact_id: impl Into<String>) -> Self {
        Self {
            fact_id: fact_id.into(),
            change: 1,
        }
    }

    pub fn remove(fact_id: impl Into<String>) -> Self {
        Self {
            fact_id: fact_id.into(),
            change: -1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DerivationDelta {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub iterations: usize,
    pub facts_touched: usize,
    pub over_deleted: usize,
}

impl DerivationDelta {
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
            iterations: 0,
            facts_touched: 0,
            over_deleted: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredicateStrategy {
    Counting,
    DeleteRederive,
}

pub trait DerivationSubscriber: Send + Sync {
    fn on_derivation_delta(&self, delta: &DerivationDelta);
}

fn subscriber_registry() -> &'static Mutex<Vec<&'static dyn DerivationSubscriber>> {
    static REGISTRY: OnceLock<Mutex<Vec<&'static dyn DerivationSubscriber>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a process-global subscriber. Notification failure is logged and
/// does not fail the maintenance pass. No subscribers are registered by this
/// module; standing/refinery/read_model can attach later.
pub fn register_derivation_subscriber(subscriber: &'static dyn DerivationSubscriber) {
    subscriber_registry()
        .lock()
        .expect("subscriber registry")
        .push(subscriber);
}

fn notify_subscribers(delta: &DerivationDelta) {
    let Ok(guard) = subscriber_registry().lock() else {
        eprintln!("derivation subscriber registry poisoned; skipping notify");
        return;
    };
    for subscriber in guard.iter() {
        // Trait method is infallible; keep the logged-failure contract for
        // future fallible adapters by catching panics at the boundary.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            subscriber.on_derivation_delta(delta);
        }));
        if let Err(payload) = result {
            eprintln!("derivation subscriber panicked: {payload:?}");
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DerivationIndex {
    facts: BTreeMap<String, SupportedFact>,
    program: RuleProgram,
    semiring: ProvenanceSemiringKind,
    /// Live saturation-derived fact ids keyed by component, used to drop
    /// statements that a later `prune_stale` materialize no longer emits.
    #[serde(default)]
    saturation_projections: BTreeMap<String, BTreeSet<String>>,
}

impl Default for DerivationIndex {
    fn default() -> Self {
        Self {
            facts: BTreeMap::new(),
            program: RuleProgram::default(),
            semiring: ProvenanceSemiringKind::Counting,
            saturation_projections: BTreeMap::new(),
        }
    }
}

fn blank_derived(fact_id: impl Into<String>) -> SupportedFact {
    SupportedFact {
        fact_id: fact_id.into(),
        multiplicity: 0,
        base: false,
        justifications: Vec::new(),
        relation: None,
    }
}

impl DerivationIndex {
    fn present_set(&self) -> BTreeSet<String> {
        self.facts
            .iter()
            .filter(|(_, fact)| fact.is_present())
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn assert_at_rest(&self) -> GraphStoreResult<()> {
        for (id, fact) in &self.facts {
            if fact.multiplicity <= 0 {
                return Err(GraphStoreError::new(
                    "invalid_at_rest_multiplicity",
                    format!("fact {id} at rest with multiplicity {}", fact.multiplicity),
                ));
            }
        }
        Ok(())
    }
}

pub fn install_rules<S: GraphStore + ?Sized>(
    store: &mut S,
    rules: &[Rule],
) -> GraphStoreResult<()> {
    let mut index = load_index(store).unwrap_or_default();
    index.program = RuleProgram::from_rules(rules.to_vec());
    index.semiring = ProvenanceSemiringKind::Counting;
    save_index(store, &index)
}

pub fn install_rules_with_semiring<S: GraphStore + ?Sized>(
    store: &mut S,
    rules: &[Rule],
    semiring: ProvenanceSemiringKind,
) -> GraphStoreResult<()> {
    let mut index = load_index(store).unwrap_or_default();
    index.program = RuleProgram::from_rules(rules.to_vec());
    index.semiring = semiring;
    save_index(store, &index)
}

pub fn supported_fact<S: GraphStore + ?Sized>(
    store: &S,
    fact_id: &str,
) -> GraphStoreResult<Option<SupportedFact>> {
    Ok(load_index(store)?.facts.get(fact_id).cloned())
}

pub fn present_facts<S: GraphStore + ?Sized>(store: &S) -> GraphStoreResult<BTreeSet<String>> {
    Ok(load_index(store)?.present_set())
}

/// Ensure a base fact exists in the incremental index (idempotent).
///
/// Does not flip an existing derived fact to base.
pub fn ensure_base_fact<S: GraphStore + ?Sized>(
    store: &mut S,
    fact_id: &str,
) -> GraphStoreResult<()> {
    let mut index = load_index(store)?;
    index
        .facts
        .entry(fact_id.to_string())
        .or_insert_with(|| SupportedFact::base(fact_id.to_string()));
    save_index(store, &index)
}

/// Seed a premise only when it is already present or is a live observational
/// / active-assumption input. Retracted assumptions and absent ids are left
/// out of the index so `put_derivation` cannot resurrect them as base facts.
pub fn ensure_observational_or_active_premise<S: GraphStore + ?Sized>(
    store: &mut S,
    fact_id: &str,
) -> GraphStoreResult<()> {
    let index = load_index(store)?;
    if index.facts.contains_key(fact_id) {
        return Ok(());
    }
    drop(index);
    let Some(node) = store.get_node_record(fact_id) else {
        return Ok(());
    };
    if node.labels.iter().any(|label| label == "Assumption") {
        let active = node
            .properties
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !active {
            return Ok(());
        }
    }
    ensure_base_fact(store, fact_id)
}

/// Record or strengthen a justified derived (or base) fact without firing rules.
pub fn record_justified_fact<S: GraphStore + ?Sized>(
    store: &mut S,
    fact_id: &str,
    justification: Justification,
    base: bool,
) -> GraphStoreResult<()> {
    record_justified_fact_with_relation(store, fact_id, justification, base, None)
}

/// Like [`record_justified_fact`], tagging the fact with a saturation relation
/// so later distinct/DRed selection does not have to parse opaque ids.
pub fn record_justified_fact_with_relation<S: GraphStore + ?Sized>(
    store: &mut S,
    fact_id: &str,
    justification: Justification,
    base: bool,
    relation: Option<&str>,
) -> GraphStoreResult<()> {
    let mut index = load_index(store)?;
    let premises_present = justification.premises.iter().all(|premise| {
        index
            .facts
            .get(premise)
            .map(SupportedFact::is_present)
            .unwrap_or(false)
    });
    if !premises_present && !base {
        return Ok(());
    }
    let entry = index.facts.entry(fact_id.to_string()).or_insert_with(|| {
        let mut fact = blank_derived(fact_id);
        fact.base = base;
        fact
    });
    if base {
        entry.base = true;
    }
    if let Some(relation) = relation {
        if entry.relation.is_none() {
            entry.relation = Some(relation.to_string());
        }
    }
    if !entry
        .justifications
        .iter()
        .any(|existing| existing == &justification)
    {
        entry.justifications.push(justification);
    }
    if entry.base {
        entry.multiplicity = 1 + entry.justifications.len() as i64;
    } else {
        entry.multiplicity = entry.justifications.len() as i64;
    }
    save_index(store, &index)
}

fn fact_relation(fact_id: &str, fact: &SupportedFact) -> Option<String> {
    fact.relation.clone().or_else(|| fact_predicate(fact_id))
}

/// Apply set-semantics `distinct` to facts whose stored relation or parsed
/// predicate is in `predicates`. Presence is clamped; justification count is
/// kept as the multiplicity witness so AC3 counting still holds.
pub fn apply_distinct_for_predicates<S: GraphStore + ?Sized>(
    store: &mut S,
    predicates: &BTreeSet<String>,
) -> GraphStoreResult<()> {
    if predicates.is_empty() {
        return Ok(());
    }
    let mut index = load_index(store)?;
    let mut multiplicities: BTreeMap<String, Multiplicity> = BTreeMap::new();
    for (id, fact) in &index.facts {
        if let Some(pred) = fact_relation(id, fact) {
            if predicates.contains(&pred) {
                multiplicities.insert(id.clone(), fact.multiplicity);
            }
        }
    }
    let previous = multiplicities.clone();
    let mut next = multiplicities;
    crate::incremental::support::distinct(&mut next);
    let _ = distinct_delta(&previous, &next);
    for (id, fact) in index.facts.iter_mut() {
        if fact.base {
            continue;
        }
        let Some(pred) = fact_relation(id, fact) else {
            continue;
        };
        if !predicates.contains(&pred) {
            continue;
        }
        if next.get(id).copied().unwrap_or(0) > 0 {
            fact.multiplicity = fact.justifications.len() as i64;
        } else {
            fact.multiplicity = 0;
        }
    }
    index
        .facts
        .retain(|_, fact| fact.multiplicity > 0 || fact.base);
    save_index(store, &index)
}

/// Drop previously projected derived facts (and propagate) without treating
/// them as base retractions. Used when saturation rematerialize omits a
/// statement and `retracted_fact_ids` is empty.
pub fn drop_derived_facts<S: GraphStore + ?Sized>(
    store: &mut S,
    fact_ids: &[String],
) -> GraphStoreResult<DerivationDelta> {
    if fact_ids.is_empty() {
        return Ok(DerivationDelta::empty());
    }
    let mut index = load_index(store)?;
    let before_present = index.present_set();
    let mut lost: BTreeSet<String> = BTreeSet::new();
    for id in fact_ids {
        if index
            .facts
            .get(id)
            .is_some_and(|fact| !fact.base && fact.is_present())
        {
            lost.insert(id.clone());
        }
    }
    for id in &lost {
        index.facts.remove(id);
    }
    let mut facts_touched = lost.clone();
    let mut iterations = 0usize;
    for id in &lost {
        iterations = iterations.max(propagate_negative_counting(
            &mut index,
            id,
            &mut facts_touched,
        )?);
    }
    let stale: Vec<String> = index
        .facts
        .iter()
        .filter(|(_, fact)| fact.multiplicity <= 0)
        .map(|(id, _)| id.clone())
        .collect();
    for id in stale {
        index.facts.remove(&id);
    }
    index.assert_at_rest()?;
    let after_present = index.present_set();
    let mut added: Vec<String> = after_present.difference(&before_present).cloned().collect();
    let mut removed: Vec<String> = before_present.difference(&after_present).cloned().collect();
    added.sort();
    removed.sort();
    let delta = DerivationDelta {
        added,
        removed,
        iterations,
        facts_touched: facts_touched.len(),
        over_deleted: 0,
    };
    save_index(store, &index)?;
    notify_subscribers(&delta);
    Ok(delta)
}

pub(crate) fn projected_saturation_facts<S: GraphStore + ?Sized>(
    store: &S,
    component_key: &str,
) -> GraphStoreResult<BTreeSet<String>> {
    Ok(load_index(store)?
        .saturation_projections
        .get(component_key)
        .cloned()
        .unwrap_or_default())
}

pub(crate) fn store_saturation_projection<S: GraphStore + ?Sized>(
    store: &mut S,
    component_key: &str,
    fact_ids: BTreeSet<String>,
) -> GraphStoreResult<()> {
    let mut index = load_index(store)?;
    if fact_ids.is_empty() {
        index.saturation_projections.remove(component_key);
    } else {
        index
            .saturation_projections
            .insert(component_key.to_string(), fact_ids);
    }
    save_index(store, &index)
}

pub fn last_derivation_receipt<S: GraphStore + ?Sized>(
    store: &S,
) -> GraphStoreResult<Option<QueryReceipt>> {
    let node = match store.get_node_record(INCREMENTAL_INDEX_NODE_ID) {
        Some(node) => node,
        None => return Ok(None),
    };
    let Some(receipt) = node.properties.get("last_receipt") else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_value(receipt.clone()).map_err(
        |error| GraphStoreError::new("invalid_derivation_receipt", error.to_string()),
    )?))
}

fn load_index<S: GraphStore + ?Sized>(store: &S) -> GraphStoreResult<DerivationIndex> {
    let Some(node) = store.get_node_record(INCREMENTAL_INDEX_NODE_ID) else {
        return Ok(DerivationIndex::default());
    };
    let mut filtered = node.properties.clone();
    if let Some(obj) = filtered.as_object_mut() {
        obj.remove("last_receipt");
    }
    serde_json::from_value(filtered)
        .map_err(|error| GraphStoreError::new("invalid_derivation_index", error.to_string()))
}

fn save_index<S: GraphStore + ?Sized>(
    store: &mut S,
    index: &DerivationIndex,
) -> GraphStoreResult<()> {
    let mut properties = serde_json::to_value(index)
        .map_err(|error| GraphStoreError::new("serialize_derivation_index", error.to_string()))?;
    if let Some(existing) = store.get_node_record(INCREMENTAL_INDEX_NODE_ID) {
        if let Some(receipt) = existing.properties.get("last_receipt") {
            if let Some(obj) = properties.as_object_mut() {
                obj.insert("last_receipt".to_string(), receipt.clone());
            }
        }
        // Re-observing an unchanged premise or justification must not create a
        // graph revision. Keep the latest receipt while comparing the exact
        // index payload that would be published.
        if !existing.tombstone
            && existing.labels == [INCREMENTAL_INDEX_LABEL]
            && existing.properties == properties
        {
            return Ok(());
        }
    }
    store.upsert_node(NodeRecord::new(
        INCREMENTAL_INDEX_NODE_ID,
        [INCREMENTAL_INDEX_LABEL],
        properties,
    ))?;
    Ok(())
}

fn save_index_with_receipt<S: GraphStore + ?Sized>(
    store: &mut S,
    index: &DerivationIndex,
    receipt: &QueryReceipt,
) -> GraphStoreResult<()> {
    let mut properties = serde_json::to_value(index)
        .map_err(|error| GraphStoreError::new("serialize_derivation_index", error.to_string()))?;
    if let Some(obj) = properties.as_object_mut() {
        obj.insert(
            "last_receipt".to_string(),
            serde_json::to_value(receipt).map_err(|error| {
                GraphStoreError::new("serialize_derivation_receipt", error.to_string())
            })?,
        );
    }
    store.upsert_node(NodeRecord::new(
        INCREMENTAL_INDEX_NODE_ID,
        [INCREMENTAL_INDEX_LABEL],
        properties,
    ))?;
    Ok(())
}

pub fn apply_delta<S: GraphStore + ?Sized>(
    store: &mut S,
    base_changes: &[FactDelta],
) -> GraphStoreResult<DerivationDelta> {
    let mut index = load_index(store)?;
    if !index.semiring.recursion_safe_with_distinct()
        && index
            .program
            .rules
            .iter()
            .any(|rule| index.program.is_recursive(rule.head_predicate()))
    {
        return Err(GraphStoreError::new(
            "recursive_semiring_not_safe",
            format!(
                "semiring {} is not recursion_safe_with_distinct",
                index.semiring.name()
            ),
        ));
    }

    let before_present = index.present_set();
    let mut iterations = 0usize;
    let mut facts_touched: BTreeSet<String> = BTreeSet::new();
    let mut over_deleted = 0usize;
    let mut strategy_by_predicate: BTreeMap<String, PredicateStrategy> = BTreeMap::new();

    let mut lost_presence: BTreeSet<String> = BTreeSet::new();
    let mut gained_presence: BTreeSet<String> = BTreeSet::new();

    for change in base_changes {
        facts_touched.insert(change.fact_id.clone());
        let predicate = fact_predicate(&change.fact_id).unwrap_or_default();
        let recursive = index.program.is_recursive(&predicate);
        strategy_by_predicate.insert(
            predicate,
            if recursive {
                PredicateStrategy::DeleteRederive
            } else {
                PredicateStrategy::Counting
            },
        );

        let presence_before = index
            .facts
            .get(&change.fact_id)
            .map(SupportedFact::is_present)
            .unwrap_or(false);

        let treat_as_base = change.change > 0
            || index
                .facts
                .get(&change.fact_id)
                .map(|fact| fact.base)
                .unwrap_or(true);
        apply_base_or_count_change(&mut index, change, treat_as_base)?;

        let presence_after = index
            .facts
            .get(&change.fact_id)
            .map(SupportedFact::is_present)
            .unwrap_or(false);

        if !presence_before && presence_after {
            gained_presence.insert(change.fact_id.clone());
        } else if presence_before && !presence_after {
            lost_presence.insert(change.fact_id.clone());
        } else if presence_after && change.change > 0 {
            gained_presence.insert(change.fact_id.clone());
        }
    }

    for fact_id in &gained_presence {
        let derived = propagate_positive(&mut index, fact_id, &mut facts_touched)?;
        iterations = iterations.max(derived);
    }

    if !lost_presence.is_empty() {
        let cone_hits_recursive = justification_cone_hits_recursive(&index, &lost_presence);
        if cone_hits_recursive {
            for pred in index.program.recursive_predicates.clone() {
                strategy_by_predicate.insert(pred, PredicateStrategy::DeleteRederive);
            }
            let (iters, over) = dred_retract(&mut index, &lost_presence, &mut facts_touched)?;
            iterations = iterations.max(iters);
            over_deleted += over;
        } else {
            for fact_id in &lost_presence {
                let derived = propagate_negative_counting(&mut index, fact_id, &mut facts_touched)?;
                iterations = iterations.max(derived);
            }
        }
    }

    // Drop facts with non-positive multiplicity.
    let stale: Vec<String> = index
        .facts
        .iter()
        .filter(|(_, fact)| fact.multiplicity <= 0)
        .map(|(id, _)| id.clone())
        .collect();
    for id in stale {
        index.facts.remove(&id);
    }

    index.assert_at_rest()?;

    let after_present = index.present_set();
    let base_input: BTreeSet<String> = base_changes
        .iter()
        .map(|change| change.fact_id.clone())
        .collect();
    // DerivationDelta reports derived-set churn. Base facts in the input
    // change feed are excluded (acceptance: one added derived fact, not the
    // base that justified it).
    let mut added: Vec<String> = after_present
        .difference(&before_present)
        .filter(|id| !base_input.contains(*id))
        .cloned()
        .collect();
    let mut removed: Vec<String> = before_present
        .difference(&after_present)
        .filter(|id| !base_input.contains(*id))
        .cloned()
        .collect();
    added.sort();
    removed.sort();

    let delta = DerivationDelta {
        added,
        removed,
        iterations,
        facts_touched: facts_touched.len(),
        over_deleted,
    };

    let receipt = build_receipt(&delta, &strategy_by_predicate, index.semiring, base_changes)?;
    save_index_with_receipt(store, &index, &receipt)?;
    notify_subscribers(&delta);
    Ok(delta)
}

fn justification_cone_hits_recursive(index: &DerivationIndex, seeds: &BTreeSet<String>) -> bool {
    if index.program.recursive_predicates.is_empty() {
        return false;
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = seeds.iter().cloned().collect();
    while let Some(fact_id) = queue.pop_front() {
        if !seen.insert(fact_id.clone()) {
            continue;
        }
        if fact_predicate(&fact_id)
            .map(|pred| index.program.is_recursive(&pred))
            .unwrap_or(false)
        {
            return true;
        }
        for (id, fact) in &index.facts {
            if fact
                .justifications
                .iter()
                .any(|just| just.premises.iter().any(|p| p == &fact_id))
            {
                if fact_predicate(id)
                    .map(|pred| index.program.is_recursive(&pred))
                    .unwrap_or(false)
                {
                    return true;
                }
                if seen.insert(id.clone()) {
                    queue.push_back(id.clone());
                }
            }
        }
    }
    false
}

fn apply_base_or_count_change(
    index: &mut DerivationIndex,
    change: &FactDelta,
    as_base: bool,
) -> GraphStoreResult<()> {
    let entry = index
        .facts
        .entry(change.fact_id.clone())
        .or_insert_with(|| {
            let mut fact = blank_derived(change.fact_id.clone());
            fact.base = as_base;
            fact
        });
    if as_base {
        entry.base = true;
    }
    entry.multiplicity += change.change;
    Ok(())
}

fn propagate_positive(
    index: &mut DerivationIndex,
    seed_fact: &str,
    facts_touched: &mut BTreeSet<String>,
) -> GraphStoreResult<usize> {
    let Some(seed) = parse_fact(seed_fact) else {
        return Ok(0);
    };
    let mut work: VecDeque<Atom> = VecDeque::from([seed]);
    let mut iterations = 0usize;
    let mut seen_delta: BTreeSet<String> = BTreeSet::new();
    seen_delta.insert(seed_fact.to_string());

    while let Some(delta_atom) = work.pop_front() {
        iterations += 1;
        let present = index.present_set();
        for rule in &index.program.rules.clone() {
            if index.program.is_recursive(rule.head_predicate()) {
                // Recursive positive: still fire, then distinct.
            }
            for (head_id, premises) in fire_rule_with_delta(rule, &delta_atom, &present) {
                facts_touched.insert(head_id.clone());
                let justification = Justification {
                    rule_id: rule.id.clone(),
                    premises,
                };
                let crossed = add_justification(index, &head_id, justification);
                if crossed && seen_delta.insert(head_id.clone()) {
                    if let Some(atom) = parse_fact(&head_id) {
                        work.push_back(atom);
                    }
                }
            }
        }
        // Apply distinct to recursive heads touched this iteration.
        apply_distinct_to_recursive(index);
    }
    Ok(iterations)
}

fn propagate_negative_counting(
    index: &mut DerivationIndex,
    seed_fact: &str,
    facts_touched: &mut BTreeSet<String>,
) -> GraphStoreResult<usize> {
    let mut work: VecDeque<String> = VecDeque::from([seed_fact.to_string()]);
    let mut iterations = 0usize;
    let mut seen = BTreeSet::new();
    seen.insert(seed_fact.to_string());

    while let Some(lost) = work.pop_front() {
        iterations += 1;
        let dependents: Vec<(String, Justification)> = index
            .facts
            .iter()
            .flat_map(|(id, fact)| {
                fact.justifications
                    .iter()
                    .filter(|just| just.premises.iter().any(|p| p == &lost))
                    .map(|just| (id.clone(), just.clone()))
            })
            .collect();

        for (fact_id, justification) in dependents {
            facts_touched.insert(fact_id.clone());
            let presence_before = index
                .facts
                .get(&fact_id)
                .map(|f| f.is_present())
                .unwrap_or(false);
            remove_justification(index, &fact_id, &justification);
            let presence_after = index
                .facts
                .get(&fact_id)
                .map(|f| f.is_present())
                .unwrap_or(false);
            if presence_before && !presence_after && seen.insert(fact_id.clone()) {
                work.push_back(fact_id);
            }
        }
    }
    Ok(iterations)
}

fn dred_retract(
    index: &mut DerivationIndex,
    seeds: &BTreeSet<String>,
    facts_touched: &mut BTreeSet<String>,
) -> GraphStoreResult<(usize, usize)> {
    // Phase 1: over-delete transitive closure through justifications.
    let mut over_delete: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = seeds.iter().cloned().collect();
    while let Some(fact_id) = queue.pop_front() {
        if !over_delete.insert(fact_id.clone()) {
            continue;
        }
        facts_touched.insert(fact_id.clone());
        let dependents: Vec<String> = index
            .facts
            .iter()
            .filter(|(_, fact)| {
                !fact.base
                    && fact
                        .justifications
                        .iter()
                        .any(|just| just.premises.iter().any(|p| p == &fact_id))
            })
            .map(|(id, _)| id.clone())
            .collect();
        for dep in dependents {
            queue.push_back(dep);
        }
    }

    // Phase 1 over-delete set size (includes facts that may come back in
    // phase 2). Count every fact removed from the working set here.
    let over_deleted = over_delete.len();

    for id in &over_delete {
        if let Some(fact) = index.facts.remove(id) {
            // Keep unretracted base facts.
            if fact.base && !seeds.contains(id) {
                index.facts.insert(id.clone(), fact);
            }
            // Derived and retracted bases stay removed for phase 2 re-derive.
        }
    }

    // Phase 2: re-derive from surviving facts outside the over-deleted set.
    let mut iterations = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        iterations += 1;
        let present = index.present_set();
        for rule in &index.program.rules.clone() {
            for (head_id, premises) in fire_rule_full(rule, &present) {
                let all_premises_present = premises.iter().all(|p| present.contains(p));
                if !all_premises_present {
                    continue;
                }
                facts_touched.insert(head_id.clone());
                let justification = Justification {
                    rule_id: rule.id.clone(),
                    premises,
                };
                if add_justification(index, &head_id, justification) {
                    changed = true;
                }
            }
        }
        apply_distinct_to_recursive(index);
        if iterations > 10_000 {
            return Err(GraphStoreError::new(
                "dred_iteration_bound",
                "DRed re-derive exceeded iteration bound",
            ));
        }
    }

    Ok((iterations, over_deleted))
}

fn add_justification(
    index: &mut DerivationIndex,
    fact_id: &str,
    justification: Justification,
) -> bool {
    let entry = index
        .facts
        .entry(fact_id.to_string())
        .or_insert_with(|| blank_derived(fact_id));
    if entry
        .justifications
        .iter()
        .any(|existing| existing == &justification)
    {
        return false;
    }
    let before = entry.multiplicity;
    entry.justifications.push(justification);
    entry.multiplicity = entry.justifications.len() as i64;
    if entry.base {
        // Base facts keep at least their base multiplicity.
        entry.multiplicity = entry.multiplicity.max(1);
    }
    before <= 0 && entry.multiplicity > 0
}

fn remove_justification(index: &mut DerivationIndex, fact_id: &str, justification: &Justification) {
    let Some(entry) = index.facts.get_mut(fact_id) else {
        return;
    };
    entry
        .justifications
        .retain(|existing| existing != justification);
    if entry.base {
        entry.multiplicity = 1 + entry.justifications.len() as i64;
    } else {
        entry.multiplicity = entry.justifications.len() as i64;
    }
}

fn apply_distinct_to_recursive(index: &mut DerivationIndex) {
    let mut multiplicities: BTreeMap<String, Multiplicity> = BTreeMap::new();
    for pred in &index.program.recursive_predicates {
        for (id, fact) in &index.facts {
            if fact_relation(id, fact).as_deref() == Some(pred.as_str()) {
                multiplicities.insert(id.clone(), fact.multiplicity);
            }
        }
    }
    let previous = multiplicities.clone();
    let mut next = multiplicities;
    crate::incremental::support::distinct(&mut next);
    let _ = distinct_delta(&previous, &next);
    for (id, mult) in next {
        if let Some(fact) = index.facts.get_mut(&id) {
            if !fact.base {
                // Distinct clamps presence; keep justification count as support
                // witness but clamp reported multiplicity for recursive set semantics.
                fact.multiplicity = if mult > 0 {
                    fact.justifications.len().max(1) as i64
                } else {
                    0
                };
            }
        }
    }
}

fn build_receipt(
    delta: &DerivationDelta,
    strategies: &BTreeMap<String, PredicateStrategy>,
    semiring: ProvenanceSemiringKind,
    base_changes: &[FactDelta],
) -> GraphStoreResult<QueryReceipt> {
    let mut scope = ReceiptScope::new();
    scope.insert("surface".to_string(), "incremental_derivation".to_string());
    scope.insert("iterations".to_string(), delta.iterations.to_string());
    scope.insert("facts_touched".to_string(), delta.facts_touched.to_string());
    scope.insert("over_deleted".to_string(), delta.over_deleted.to_string());
    scope.insert("semiring".to_string(), semiring.name().to_string());
    for (pred, strategy) in strategies {
        scope.insert(
            format!("strategy:{pred}"),
            match strategy {
                PredicateStrategy::Counting => "counting".to_string(),
                PredicateStrategy::DeleteRederive => "delete_rederive".to_string(),
            },
        );
    }

    let mut theorem = TheoremReceiptFields {
        provenance_kind: semiring.name().to_string(),
        ..Default::default()
    };
    theorem
        .receipt_fields
        .insert("iterations".to_string(), delta.iterations.to_string());
    theorem
        .receipt_fields
        .insert("facts_touched".to_string(), delta.facts_touched.to_string());
    theorem
        .receipt_fields
        .insert("over_deleted".to_string(), delta.over_deleted.to_string());
    theorem.receipt_fields.insert(
        "strategies".to_string(),
        serde_json::to_string(strategies).unwrap_or_default(),
    );

    let signature = stable_hash(json!({
        "base_changes": base_changes,
        "delta": delta,
        "strategies": strategies,
        "semiring": semiring.name(),
    }))?;

    Ok(QueryReceipt {
        schema_version: 2,
        id: format!("receipt:incremental:{signature}"),
        query_signature: signature,
        query_kind: QueryKind::GraphExpand,
        scope,
        otel_genai: Default::default(),
        theorem,
        access_paths_considered: Vec::new(),
        access_paths_used: Vec::new(),
        indexes_used: Vec::new(),
        candidate_counts_by_stage: BTreeMap::new(),
        full_scan_count: 0,
        hydrated_object_count: delta.added.len() + delta.removed.len(),
        graph_expansion_count: delta.facts_touched,
        latency_by_stage_ms: BTreeMap::new(),
        token_cost: None,
        result_count: delta.added.len() + delta.removed.len(),
        accepted_result_ids: delta.added.clone(),
        cited_result_ids: Vec::new(),
        dismissed_result_ids: delta.removed.clone(),
        outcome_label: QueryOutcomeLabel::Success,
        created_at: now_ms(),
    })
}

/// Confirm deliverable 5 shape fit: standing firings, refinery, and read-model
/// indexes all consume added/removed identity lists. `DerivationDelta` matches.
pub fn derivation_delta_consumer_shape_note() -> Value {
    json!({
        "standing": "StandingFiring carries previous/current StandingResult; added/removed fact ids can drive a StandingMatch refresh without full rescan",
        "refinery": "refinery corpus refresh keys on content ids; DerivationDelta.added/removed are the same grain",
        "read_model_index": "CompositeIndex/PartialIndex upsert_node/remove_node take node ids; DerivationDelta lists are directly applicable",
        "wired": false
    })
}

pub fn edge_fact(from: &str, to: &str) -> String {
    format_fact("edge", &[from.to_string(), to.to_string()])
}

pub fn path_fact(from: &str, to: &str) -> String {
    format_fact("path", &[from.to_string(), to.to_string()])
}

pub fn path_program() -> Vec<Rule> {
    vec![
        Rule::new(
            "path_base",
            Atom::new("path", ["X", "Y"]),
            [Atom::new("edge", ["X", "Y"])],
        ),
        Rule::new(
            "path_trans",
            Atom::new("path", ["X", "Y"]),
            [Atom::new("path", ["X", "Z"]), Atom::new("path", ["Z", "Y"])],
        ),
    ]
}
