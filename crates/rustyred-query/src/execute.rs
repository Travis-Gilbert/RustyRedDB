use std::collections::{BTreeMap, BTreeSet};

use rustyred_thg_core::{
    GraphStoreError, InMemoryGraphStore, NeighborHit, NeighborQuery, NodeQuery, NodeRecord,
};
use serde_json::{json, Map, Value};

use crate::cypher::ast::{
    AggOp, CmpOp, CypherPattern, EdgeChain, EdgeDirection, EdgePattern, EdgeVarLength, NodePattern,
    ParsedCypher, PropertyFilter, ReturnItem, WhereExpr, WithClause, WithItem,
};
use crate::cypher::planner::{
    self, AggregateOutput, AggregateSpec, PlanSteeringState, PlannerSteeringDecision,
    PlannerSteeringPolicy,
};
use crate::error::CypherError;
use crate::read_gate::{neighbor_direction, node_matches_pattern};

pub trait CypherGraph {
    fn get_node(&self, id: &str) -> Result<Option<NodeRecord>, GraphStoreError>;
    fn query_nodes(&self, query: NodeQuery) -> Result<Vec<NodeRecord>, GraphStoreError>;
    fn neighbors(&self, query: NeighborQuery) -> Result<Vec<NeighborHit>, GraphStoreError>;
    fn graph_version(&self) -> Result<u64, GraphStoreError>;
}

#[derive(Clone, Debug)]
pub struct ReadOutcome {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub named_rows: Vec<Map<String, Value>>,
    pub truncated: bool,
    pub total_available: usize,
    pub shape_key: String,
    pub decision: Option<PlannerSteeringDecision>,
    pub selected_candidate_id: String,
    pub native_candidate_id: String,
    pub cost_units: f64,
    pub seed_count: usize,
    pub edges_touched: usize,
    pub plan_operation: String,
}

#[derive(Clone, Debug)]
pub struct PlanPreview {
    pub shape_key: String,
    pub selected_candidate_id: String,
    pub native_candidate_id: String,
    pub decision: Option<PlannerSteeringDecision>,
}

pub fn explain_read(parsed: &ParsedCypher, steering: Option<&PlanSteeringState>) -> PlanPreview {
    match &parsed.pattern {
        CypherPattern::Node(node) => PlanPreview {
            shape_key: format!("node|{}", node.label.as_deref().unwrap_or("*")),
            selected_candidate_id: "node_scan".into(),
            native_candidate_id: "node_scan".into(),
            decision: Some(PlannerSteeringDecision {
                selected_candidate_id: "node_scan".into(),
                native_candidate_id: "node_scan".into(),
                abstained: true,
                reason: "single_candidate".into(),
                score: 0.0,
                observed_candidate_count: 0,
            }),
        },
        CypherPattern::Edge(edge) => {
            let left_filter = parsed
                .eq_filter()
                .filter(|filter| filter.binding == edge.left.binding);
            let right_filter = parsed
                .eq_filter()
                .filter(|filter| filter.binding == edge.right.binding);
            let right_anchorable = edge.right.label.is_some() || !edge.right.properties.is_empty();
            let candidates = planner::enumerate_edge_pattern_candidates(right_anchorable);
            let shape_key = planner::edge_pattern_shape_key(
                edge.left.label.as_deref(),
                &property_keys(&edge.left, left_filter.as_ref()),
                &edge.edge_type,
                edge.right.label.as_deref(),
                &property_keys(&edge.right, right_filter.as_ref()),
            );
            let decision = steering.and_then(|state| {
                planner::steer_plan_candidates(
                    &candidates,
                    &state.metrics_for(&shape_key),
                    PlannerSteeringPolicy {
                        trial_nonce: Some(state.next_trial_nonce()),
                        ..PlannerSteeringPolicy::default()
                    },
                )
            });
            let native = candidates
                .iter()
                .min_by_key(|candidate| candidate.native_rank)
                .map(|candidate| candidate.candidate_id.clone())
                .unwrap_or_else(|| planner::PLAN_CANDIDATE_EXPAND_LEFT_OUT.to_string());
            let selected = decision
                .as_ref()
                .map(|item| item.selected_candidate_id.clone())
                .unwrap_or_else(|| native.clone());
            PlanPreview {
                shape_key,
                selected_candidate_id: selected,
                native_candidate_id: native,
                decision,
            }
        }
        CypherPattern::EdgeChain(chain) => PlanPreview {
            shape_key: format!(
                "chain|{}",
                chain
                    .steps
                    .iter()
                    .map(|step| step.edge_type.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            selected_candidate_id: "chain_expand".into(),
            native_candidate_id: "chain_expand".into(),
            decision: Some(PlannerSteeringDecision {
                selected_candidate_id: "chain_expand".into(),
                native_candidate_id: "chain_expand".into(),
                abstained: true,
                reason: "single_candidate".into(),
                score: 0.0,
                observed_candidate_count: 0,
            }),
        },
        CypherPattern::EdgeVarLength(var) => PlanPreview {
            shape_key: format!("var|{}", var.edge_type),
            selected_candidate_id: "var_expand".into(),
            native_candidate_id: "var_expand".into(),
            decision: Some(PlannerSteeringDecision {
                selected_candidate_id: "var_expand".into(),
                native_candidate_id: "var_expand".into(),
                abstained: true,
                reason: "single_candidate".into(),
                score: 0.0,
                observed_candidate_count: 0,
            }),
        },
    }
}

#[derive(Clone)]
struct Binding {
    nodes: BTreeMap<String, NodeRecord>,
    path_nodes: BTreeMap<String, Vec<String>>,
    path_rels: BTreeMap<String, Vec<String>>,
}

impl Binding {
    fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            path_nodes: BTreeMap::new(),
            path_rels: BTreeMap::new(),
        }
    }
}

pub fn execute_read<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    steering: Option<&PlanSteeringState>,
    row_cap: usize,
) -> Result<ReadOutcome, CypherError> {
    let cap = row_cap.max(1);
    let needs_full_set = needs_wide_scan(parsed);
    let budget = match_budget(parsed, cap);
    let (bindings, meta) = match &parsed.pattern {
        CypherPattern::Node(node) => execute_node(store, parsed, node, budget)?,
        CypherPattern::Edge(edge) => execute_edge(store, parsed, edge, steering, budget)?,
        CypherPattern::EdgeChain(chain) => execute_chain(store, parsed, chain, budget)?,
        CypherPattern::EdgeVarLength(var) => execute_var_length(store, parsed, var, budget)?,
    };
    let scan_truncated = bindings.len() >= budget;
    if needs_full_set && scan_truncated {
        return Err(CypherError::invalid(
            "exactness_budget_exceeded",
            format!(
                "query requires more than {} matched rows for an exact result",
                budget.saturating_sub(1)
            ),
        ));
    }
    let mut named = shape_bindings(parsed, &bindings)?;
    if parsed.distinct {
        let mut seen = BTreeSet::new();
        named.retain(|row| seen.insert(serde_json::to_string(row).unwrap_or_default()));
    }
    planner::sort_rows(&mut named, &parsed.order_by);
    if let Some(skip) = parsed.skip {
        named = planner::skip_rows(named, skip);
    }
    if parsed.limit_specified {
        named.truncate(parsed.limit);
    }
    let total_available = named.len();
    let truncated = named.len() > cap || scan_truncated;
    if named.len() > cap {
        named.truncate(cap);
    }
    let columns: Vec<String> = parsed
        .returns
        .iter()
        .map(|item| item.key().to_string())
        .collect();
    let rows = named
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| row.get(col).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Ok(ReadOutcome {
        columns,
        rows,
        named_rows: named,
        truncated,
        total_available,
        shape_key: meta.shape_key,
        decision: meta.decision,
        selected_candidate_id: meta.selected,
        native_candidate_id: meta.native,
        cost_units: meta.cost,
        seed_count: meta.seed_count,
        edges_touched: meta.edges_touched,
        plan_operation: meta.plan_operation,
    })
}

fn needs_wide_scan(parsed: &ParsedCypher) -> bool {
    parsed.distinct
        || parsed.skip.is_some()
        || !parsed.order_by.is_empty()
        || parsed.with_clause.is_some()
        || returns_have_aggregate(&parsed.returns)
        || (parsed.where_expr.is_some() && parsed.eq_filter().is_none())
}

fn match_budget(parsed: &ParsedCypher, cap: usize) -> usize {
    if needs_wide_scan(parsed) {
        10_000usize.max(cap).saturating_add(1)
    } else {
        let ast_need = parsed
            .skip
            .unwrap_or(0)
            .saturating_add(if parsed.limit_specified {
                parsed.limit
            } else {
                cap
            });
        ast_need.max(cap).saturating_add(1)
    }
}

fn returns_have_aggregate(items: &[ReturnItem]) -> bool {
    items.iter().any(is_aggregate_item)
}

fn is_aggregate_item(item: &ReturnItem) -> bool {
    match item {
        ReturnItem::Count { .. } | ReturnItem::Aggregate { .. } => true,
        ReturnItem::Aliased { item, .. } => is_aggregate_item(item),
        _ => false,
    }
}

fn shape_bindings(
    parsed: &ParsedCypher,
    bindings: &[Binding],
) -> Result<Vec<Map<String, Value>>, CypherError> {
    if let Some(with) = &parsed.with_clause {
        let rows = apply_with(with, bindings);
        return Ok(project_return_from_rows(parsed, rows));
    }
    if returns_have_aggregate(&parsed.returns) {
        return Ok(apply_return_aggregate(parsed, bindings));
    }
    bindings
        .iter()
        .map(|binding| project_binding(parsed, binding))
        .collect()
}

fn apply_with(with: &WithClause, bindings: &[Binding]) -> Vec<Map<String, Value>> {
    let group_keys: Vec<String> = with
        .items
        .iter()
        .filter_map(|item| match item {
            WithItem::Field { alias, .. } => Some(alias.clone()),
            WithItem::Aggregate { .. } => None,
        })
        .collect();
    let aggs: Vec<AggregateOutput> = with
        .items
        .iter()
        .filter_map(|item| match item {
            WithItem::Aggregate {
                op,
                alias,
                binding,
                key,
            } => Some(AggregateOutput {
                alias: alias.clone(),
                op: *op,
                source_key: agg_source_key(binding, key),
            }),
            WithItem::Field { .. } => None,
        })
        .collect();
    let mut source_rows = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let mut row = Map::new();
        for item in &with.items {
            match item {
                WithItem::Field {
                    binding: name,
                    key,
                    alias,
                } => {
                    row.insert(alias.clone(), field_value(binding, name, key.as_deref()));
                }
                WithItem::Aggregate {
                    binding: name, key, ..
                } => {
                    if let Some(source_key) = agg_source_key(name, key) {
                        row.insert(
                            source_key.clone(),
                            field_value(binding, name.as_deref().unwrap_or(""), key.as_deref()),
                        );
                    }
                }
            }
        }
        source_rows.push(row);
    }
    if aggs.is_empty() {
        source_rows
    } else {
        planner::aggregate(&source_rows, &AggregateSpec { group_keys, aggs })
    }
}

fn apply_return_aggregate(parsed: &ParsedCypher, bindings: &[Binding]) -> Vec<Map<String, Value>> {
    let mut group_keys = Vec::new();
    let mut aggs = Vec::new();
    for item in &parsed.returns {
        collect_return_agg_spec(item, &mut group_keys, &mut aggs);
    }
    let mut source_rows = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let mut row = Map::new();
        for item in &parsed.returns {
            fill_return_source(item, binding, &mut row);
        }
        source_rows.push(row);
    }
    planner::aggregate(&source_rows, &AggregateSpec { group_keys, aggs })
}

fn collect_return_agg_spec(
    item: &ReturnItem,
    group_keys: &mut Vec<String>,
    aggs: &mut Vec<AggregateOutput>,
) {
    match item {
        ReturnItem::Aliased { item, alias } => match item.as_ref() {
            ReturnItem::Count { .. } => aggs.push(AggregateOutput {
                alias: alias.clone(),
                op: AggOp::Count,
                source_key: None,
            }),
            ReturnItem::Aggregate {
                op, binding, key, ..
            } => aggs.push(AggregateOutput {
                alias: alias.clone(),
                op: *op,
                source_key: agg_source_key(binding, key),
            }),
            _ => group_keys.push(alias.clone()),
        },
        ReturnItem::Count { expression, .. } => aggs.push(AggregateOutput {
            alias: expression.clone(),
            op: AggOp::Count,
            source_key: None,
        }),
        ReturnItem::Aggregate {
            op,
            binding,
            key,
            expression,
            ..
        } => aggs.push(AggregateOutput {
            alias: expression.clone(),
            op: *op,
            source_key: agg_source_key(binding, key),
        }),
        _ => group_keys.push(item.key().to_string()),
    }
}

fn fill_return_source(item: &ReturnItem, binding: &Binding, row: &mut Map<String, Value>) {
    match item {
        ReturnItem::Aliased { item, alias } => match item.as_ref() {
            ReturnItem::Aggregate {
                binding: name, key, ..
            } => {
                if let Some(source_key) = agg_source_key(name, key) {
                    row.insert(
                        source_key.clone(),
                        field_value(binding, name.as_deref().unwrap_or(""), key.as_deref()),
                    );
                }
                row.insert(alias.clone(), Value::Null);
            }
            ReturnItem::Count { .. } => {
                row.insert(alias.clone(), json!(1));
            }
            other => {
                if let Ok((_, value)) = project_item(other, binding) {
                    row.insert(alias.clone(), value);
                }
            }
        },
        ReturnItem::Aggregate {
            binding: name,
            key,
            expression,
            ..
        } => {
            if let Some(source_key) = agg_source_key(name, key) {
                row.insert(
                    source_key.clone(),
                    field_value(binding, name.as_deref().unwrap_or(""), key.as_deref()),
                );
            }
            row.insert(expression.clone(), Value::Null);
        }
        ReturnItem::Count { expression, .. } => {
            row.insert(expression.clone(), json!(1));
        }
        other => {
            if let Ok((key, value)) = project_item(other, binding) {
                row.insert(key, value);
            }
        }
    }
}

fn project_return_from_rows(
    parsed: &ParsedCypher,
    rows: Vec<Map<String, Value>>,
) -> Vec<Map<String, Value>> {
    rows.into_iter()
        .map(|row| {
            let mut out = Map::new();
            for item in &parsed.returns {
                let key = item.key().to_string();
                let value = row.get(&key).cloned().unwrap_or(Value::Null);
                out.insert(key, value);
            }
            out
        })
        .collect()
}

fn agg_source_key(binding: &Option<String>, key: &Option<String>) -> Option<String> {
    match (binding, key) {
        (Some(binding), Some(key)) => Some(format!("{binding}.{key}")),
        (None, Some(key)) => Some(key.clone()),
        _ => None,
    }
}

fn field_value(binding: &Binding, name: &str, key: Option<&str>) -> Value {
    match key {
        Some(key) => binding
            .nodes
            .get(name)
            .and_then(|node| node.properties.get(key).cloned())
            .unwrap_or(Value::Null),
        None => binding
            .nodes
            .get(name)
            .map(node_value)
            .unwrap_or(Value::Null),
    }
}

struct ExecMeta {
    shape_key: String,
    decision: Option<PlannerSteeringDecision>,
    selected: String,
    native: String,
    cost: f64,
    seed_count: usize,
    edges_touched: usize,
    plan_operation: String,
}

fn execute_node<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    node: &NodePattern,
    budget: usize,
) -> Result<(Vec<Binding>, ExecMeta), CypherError> {
    let query = node_query_for(node, parsed.eq_filter().as_ref(), budget);
    let mut nodes = store.query_nodes(query)?;
    nodes.retain(|record| node_matches_pattern(record, node.label.as_deref(), &node.properties));
    if let Some(filter) = parsed.eq_filter() {
        nodes.retain(|record| node_matches_filter(record, &filter));
    }
    let mut bindings = Vec::new();
    for record in &nodes {
        let mut binding = Binding::new();
        binding.nodes.insert(node.binding.clone(), record.clone());
        if !binding_allowed(parsed, &binding) {
            continue;
        }
        bindings.push(binding);
        if bindings.len() >= budget {
            break;
        }
    }
    Ok((
        bindings,
        ExecMeta {
            shape_key: format!("node|{}", node.label.as_deref().unwrap_or("*")),
            decision: None,
            selected: "node_scan".into(),
            native: "node_scan".into(),
            cost: nodes.len() as f64,
            seed_count: nodes.len(),
            edges_touched: 0,
            plan_operation: if node.label.is_some() || !node.properties.is_empty() {
                "node_index_seek".into()
            } else {
                "node_scan".into()
            },
        },
    ))
}

fn execute_edge<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    edge: &EdgePattern,
    steering: Option<&PlanSteeringState>,
    cap: usize,
) -> Result<(Vec<Binding>, ExecMeta), CypherError> {
    let left_filter = parsed
        .eq_filter()
        .filter(|filter| filter.binding == edge.left.binding);
    let right_filter = parsed
        .eq_filter()
        .filter(|filter| filter.binding == edge.right.binding);
    let right_anchorable = edge.right.label.is_some() || !edge.right.properties.is_empty();
    let candidates = planner::enumerate_edge_pattern_candidates(right_anchorable);
    let shape_key = planner::edge_pattern_shape_key(
        edge.left.label.as_deref(),
        &property_keys(&edge.left, left_filter.as_ref()),
        &edge.edge_type,
        edge.right.label.as_deref(),
        &property_keys(&edge.right, right_filter.as_ref()),
    );
    let decision = steering.and_then(|state| {
        planner::steer_plan_candidates(
            &candidates,
            &state.metrics_for(&shape_key),
            PlannerSteeringPolicy {
                trial_nonce: Some(state.next_trial_nonce()),
                ..PlannerSteeringPolicy::default()
            },
        )
    });
    let native = candidates
        .iter()
        .min_by_key(|candidate| candidate.native_rank)
        .map(|candidate| candidate.candidate_id.clone())
        .unwrap_or_else(|| planner::PLAN_CANDIDATE_EXPAND_LEFT_OUT.to_string());
    let selected = decision
        .as_ref()
        .map(|item| item.selected_candidate_id.clone())
        .unwrap_or_else(|| native.clone());
    let expand_right = selected == planner::PLAN_CANDIDATE_EXPAND_RIGHT_IN;
    let (named, seed_count, edges_touched) = if expand_right {
        expand_edge(store, parsed, edge, true, cap)?
    } else {
        expand_edge(store, parsed, edge, false, cap)?
    };
    let cost = (seed_count + edges_touched) as f64;
    if let Some(state) = steering {
        state.record(&shape_key, &selected, cost, true);
    }
    Ok((
        named,
        ExecMeta {
            shape_key,
            decision,
            selected,
            native,
            cost,
            seed_count,
            edges_touched,
            plan_operation: if expand_right {
                "node_index_seek_expand_in".into()
            } else {
                "node_index_seek_expand_out".into()
            },
        },
    ))
}

fn expand_edge<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    edge: &EdgePattern,
    from_right: bool,
    cap: usize,
) -> Result<(Vec<Binding>, usize, usize), CypherError> {
    let (seed_pat, target_pat, dirs) = if from_right {
        (
            &edge.right,
            &edge.left,
            neighbor_direction(reverse_dir(edge.direction)),
        )
    } else {
        (&edge.left, &edge.right, neighbor_direction(edge.direction))
    };
    let seed_filter = parsed
        .eq_filter()
        .filter(|filter| filter.binding == seed_pat.binding);
    let seeds = store.query_nodes(node_query_for(seed_pat, seed_filter.as_ref(), cap))?;
    let mut named = Vec::new();
    let mut edges_touched = 0usize;
    for seed in &seeds {
        if !node_matches_pattern(seed, seed_pat.label.as_deref(), &seed_pat.properties) {
            continue;
        }
        for direction in &dirs {
            let hits = store.neighbors(NeighborQuery {
                node_id: seed.id.clone(),
                direction: direction.clone(),
                edge_type: Some(edge.edge_type.clone()),
                include_expired: false,
            })?;
            edges_touched += hits.len();
            for hit in hits {
                let Some(target) = store.get_node(&hit.node_id)? else {
                    continue;
                };
                if !node_matches_pattern(
                    &target,
                    target_pat.label.as_deref(),
                    &target_pat.properties,
                ) {
                    continue;
                }
                let mut binding = Binding::new();
                if from_right {
                    binding
                        .nodes
                        .insert(edge.right.binding.clone(), seed.clone());
                    binding
                        .nodes
                        .insert(edge.left.binding.clone(), target.clone());
                } else {
                    binding
                        .nodes
                        .insert(edge.left.binding.clone(), seed.clone());
                    binding
                        .nodes
                        .insert(edge.right.binding.clone(), target.clone());
                }
                if let Some(path) = &edge.path_binding {
                    let (left, right) = if from_right {
                        (&target, seed)
                    } else {
                        (seed, &target)
                    };
                    binding
                        .path_nodes
                        .insert(path.clone(), vec![left.id.clone(), right.id.clone()]);
                    binding
                        .path_rels
                        .insert(path.clone(), vec![hit.edge_id.clone()]);
                }
                if !binding_allowed(parsed, &binding) {
                    continue;
                }
                named.push(binding);
                if named.len() >= cap {
                    return Ok((named, seeds.len(), edges_touched));
                }
            }
        }
    }
    Ok((named, seeds.len(), edges_touched))
}

fn execute_chain<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    chain: &EdgeChain,
    cap: usize,
) -> Result<(Vec<Binding>, ExecMeta), CypherError> {
    let seed_filter = parsed
        .eq_filter()
        .filter(|filter| filter.binding == chain.start.binding);
    let seeds = store.query_nodes(node_query_for(&chain.start, seed_filter.as_ref(), cap))?;
    let mut named = Vec::new();
    let mut edges_touched = 0usize;
    for seed in &seeds {
        if !node_matches_pattern(seed, chain.start.label.as_deref(), &chain.start.properties) {
            continue;
        }
        let mut binding = Binding::new();
        binding
            .nodes
            .insert(chain.start.binding.clone(), seed.clone());
        if let Some(path) = &chain.path_binding {
            binding
                .path_nodes
                .insert(path.clone(), vec![seed.id.clone()]);
            binding.path_rels.insert(path.clone(), Vec::new());
        }
        walk_chain(
            store,
            parsed,
            chain,
            0,
            &mut binding,
            &mut named,
            &mut edges_touched,
            cap,
        )?;
        if named.len() >= cap {
            break;
        }
    }
    Ok((
        named,
        ExecMeta {
            shape_key: "edge_chain".into(),
            decision: None,
            selected: "chain_expand".into(),
            native: "chain_expand".into(),
            cost: edges_touched as f64,
            seed_count: seeds.len(),
            edges_touched,
            plan_operation: "chain_expand".into(),
        },
    ))
}

fn walk_chain<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    chain: &EdgeChain,
    step_index: usize,
    binding: &mut Binding,
    named: &mut Vec<Binding>,
    edges_touched: &mut usize,
    cap: usize,
) -> Result<(), CypherError> {
    if named.len() >= cap {
        return Ok(());
    }
    if step_index >= chain.steps.len() {
        if binding_allowed(parsed, binding) {
            named.push(binding.clone());
        }
        return Ok(());
    }
    let step = &chain.steps[step_index];
    let current = binding
        .nodes
        .get(&current_binding(chain, step_index))
        .cloned()
        .ok_or_else(|| CypherError::invalid("missing_chain_binding", "chain walk lost a node"))?;
    let endpoints = expand_hop(
        store,
        &current,
        &step.edge_type,
        step.direction,
        step.min,
        step.max,
        step.is_var_length,
        &step.target,
        edges_touched,
    )?;
    for (target, path_nodes, path_rels) in endpoints {
        let mut next = binding.clone();
        next.nodes
            .insert(step.target.binding.clone(), target.clone());
        if let Some(path) = &chain.path_binding {
            next.path_nodes
                .entry(path.clone())
                .or_default()
                .extend(path_nodes);
            next.path_rels
                .entry(path.clone())
                .or_default()
                .extend(path_rels);
        }
        walk_chain(
            store,
            parsed,
            chain,
            step_index + 1,
            &mut next,
            named,
            edges_touched,
            cap,
        )?;
        if named.len() >= cap {
            return Ok(());
        }
    }
    Ok(())
}

fn current_binding(chain: &EdgeChain, step_index: usize) -> String {
    if step_index == 0 {
        chain.start.binding.clone()
    } else {
        chain.steps[step_index - 1].target.binding.clone()
    }
}

fn execute_var_length<G: CypherGraph>(
    store: &G,
    parsed: &ParsedCypher,
    var: &EdgeVarLength,
    cap: usize,
) -> Result<(Vec<Binding>, ExecMeta), CypherError> {
    let seed_filter = parsed
        .eq_filter()
        .filter(|filter| filter.binding == var.from.binding);
    let seeds = store.query_nodes(node_query_for(&var.from, seed_filter.as_ref(), cap))?;
    let mut named = Vec::new();
    let mut edges_touched = 0usize;
    for seed in &seeds {
        if !node_matches_pattern(seed, var.from.label.as_deref(), &var.from.properties) {
            continue;
        }
        let endpoints = expand_hop(
            store,
            seed,
            &var.edge_type,
            var.direction,
            var.min,
            var.max,
            true,
            &var.to,
            &mut edges_touched,
        )?;
        for (target, path_nodes, path_rels) in endpoints {
            let mut binding = Binding::new();
            binding.nodes.insert(var.from.binding.clone(), seed.clone());
            binding.nodes.insert(var.to.binding.clone(), target);
            if let Some(path) = &var.path_binding {
                let mut nodes = vec![seed.id.clone()];
                nodes.extend(path_nodes);
                binding.path_nodes.insert(path.clone(), nodes);
                binding.path_rels.insert(path.clone(), path_rels);
            }
            if !binding_allowed(parsed, &binding) {
                continue;
            }
            named.push(binding);
            if named.len() >= cap {
                break;
            }
        }
        if named.len() >= cap {
            break;
        }
    }
    Ok((
        named,
        ExecMeta {
            shape_key: format!("var_length|{}", var.edge_type),
            decision: None,
            selected: "variable_length_expand".into(),
            native: "variable_length_expand".into(),
            cost: edges_touched as f64,
            seed_count: seeds.len(),
            edges_touched,
            plan_operation: "variable_length_expand".into(),
        },
    ))
}

fn expand_hop<G: CypherGraph>(
    store: &G,
    start: &NodeRecord,
    edge_type: &str,
    direction: EdgeDirection,
    min: usize,
    max: Option<usize>,
    var_length: bool,
    target_pat: &NodePattern,
    edges_touched: &mut usize,
) -> Result<Vec<(NodeRecord, Vec<String>, Vec<String>)>, CypherError> {
    if !var_length {
        let mut out = Vec::new();
        for dir in neighbor_direction(direction) {
            let hits = store.neighbors(NeighborQuery {
                node_id: start.id.clone(),
                direction: dir,
                edge_type: Some(edge_type.to_string()),
                include_expired: false,
            })?;
            *edges_touched += hits.len();
            for hit in hits {
                let Some(node) = store.get_node(&hit.node_id)? else {
                    continue;
                };
                if node_matches_pattern(&node, target_pat.label.as_deref(), &target_pat.properties)
                {
                    out.push((node, vec![hit.node_id.clone()], vec![hit.edge_id.clone()]));
                }
            }
        }
        return Ok(out);
    }
    let max_hops = max.unwrap_or(8).max(min).min(32);
    let mut found = Vec::new();
    if min == 0 && node_matches_pattern(start, target_pat.label.as_deref(), &target_pat.properties)
    {
        found.push((start.clone(), Vec::new(), Vec::new()));
    }
    let mut frontier: Vec<(String, Vec<String>, Vec<String>)> =
        vec![(start.id.clone(), vec![start.id.clone()], Vec::new())];
    let mut visited: BTreeSet<String> = BTreeSet::from([start.id.clone()]);
    for depth in 1..=max_hops {
        let mut next = Vec::new();
        for (id, path_nodes, path_rels) in frontier {
            for dir in neighbor_direction(direction) {
                let hits = store.neighbors(NeighborQuery {
                    node_id: id.clone(),
                    direction: dir,
                    edge_type: Some(edge_type.to_string()),
                    include_expired: false,
                })?;
                *edges_touched += hits.len();
                for hit in hits {
                    if visited.contains(&hit.node_id) {
                        continue;
                    }
                    let mut next_nodes = path_nodes.clone();
                    next_nodes.push(hit.node_id.clone());
                    let mut next_rels = path_rels.clone();
                    next_rels.push(hit.edge_id.clone());
                    if depth >= min {
                        if let Some(node) = store.get_node(&hit.node_id)? {
                            if node_matches_pattern(
                                &node,
                                target_pat.label.as_deref(),
                                &target_pat.properties,
                            ) {
                                // Exclude the hop start so callers can prepend their own seed.
                                let hop_nodes = next_nodes[1..].to_vec();
                                found.push((node, hop_nodes, next_rels.clone()));
                            }
                        }
                    }
                    visited.insert(hit.node_id.clone());
                    next.push((hit.node_id, next_nodes, next_rels));
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    Ok(found)
}

fn project_binding(
    parsed: &ParsedCypher,
    binding: &Binding,
) -> Result<Map<String, Value>, CypherError> {
    let mut row = Map::new();
    for item in &parsed.returns {
        let (key, value) = project_item(item, binding)?;
        row.insert(key, value);
    }
    for order in &parsed.order_by {
        if !row.contains_key(&order.expression) {
            row.insert(
                order.expression.clone(),
                resolve_order_value(binding, &order.expression),
            );
        }
    }
    Ok(row)
}

fn resolve_order_value(binding: &Binding, expression: &str) -> Value {
    if let Some((name, key)) = expression.split_once('.') {
        field_value(binding, name, Some(key))
    } else {
        binding
            .nodes
            .get(expression)
            .map(node_value)
            .unwrap_or(Value::Null)
    }
}

fn project_item(item: &ReturnItem, binding: &Binding) -> Result<(String, Value), CypherError> {
    match item {
        ReturnItem::Variable(name) => Ok((
            name.clone(),
            binding
                .nodes
                .get(name)
                .map(node_value)
                .unwrap_or(Value::Null),
        )),
        ReturnItem::Property {
            binding: name,
            key,
            expression,
        } => Ok((
            expression.clone(),
            binding
                .nodes
                .get(name)
                .and_then(|node| node.properties.get(key).cloned())
                .unwrap_or(Value::Null),
        )),
        ReturnItem::Count { expression, .. } => Ok((expression.clone(), json!(1))),
        ReturnItem::Path {
            binding: name,
            expression,
        } => Ok((
            expression.clone(),
            json!({
                "nodes": binding.path_nodes.get(name).cloned().unwrap_or_default(),
                "relationships": binding.path_rels.get(name).cloned().unwrap_or_default(),
            }),
        )),
        ReturnItem::PathNodes {
            binding: name,
            expression,
        } => Ok((
            expression.clone(),
            json!(binding.path_nodes.get(name).cloned().unwrap_or_default()),
        )),
        ReturnItem::PathRels {
            binding: name,
            expression,
        } => Ok((
            expression.clone(),
            json!(binding.path_rels.get(name).cloned().unwrap_or_default()),
        )),
        ReturnItem::PathLength {
            binding: name,
            expression,
        } => Ok((
            expression.clone(),
            json!(binding.path_rels.get(name).map(Vec::len).unwrap_or(0)),
        )),
        ReturnItem::Aggregate { expression, .. } => Ok((expression.clone(), Value::Null)),
        ReturnItem::Aliased { item, alias } => {
            let (_, value) = project_item(item, binding)?;
            Ok((alias.clone(), value))
        }
    }
}

fn node_value(node: &NodeRecord) -> Value {
    json!({
        "id": node.id,
        "labels": node.labels,
        "properties": node.properties,
    })
}

fn node_query_for(node: &NodePattern, filter: Option<&PropertyFilter>, limit: usize) -> NodeQuery {
    let mut query = NodeQuery {
        label: node.label.clone(),
        limit: Some(limit.max(1)),
        ..NodeQuery::default()
    };
    for (key, value) in &node.properties {
        query.properties.insert(key.clone(), value.clone());
    }
    if let Some(filter) = filter {
        if filter.binding == node.binding {
            query
                .properties
                .insert(filter.key.clone(), filter.value.clone());
        }
    }
    query
}

fn property_keys(node: &NodePattern, filter: Option<&PropertyFilter>) -> Vec<String> {
    let mut keys: Vec<String> = node.properties.keys().cloned().collect();
    if let Some(filter) = filter {
        if !keys.contains(&filter.key) {
            keys.push(filter.key.clone());
        }
    }
    keys.sort();
    keys
}

fn node_matches_filter(node: &NodeRecord, filter: &PropertyFilter) -> bool {
    node.properties.get(&filter.key) == Some(&filter.value)
}

fn binding_allowed(parsed: &ParsedCypher, binding: &Binding) -> bool {
    parsed
        .where_expr
        .as_ref()
        .is_none_or(|expr| eval_where(binding, expr) == Some(true))
}

fn eval_where(binding: &Binding, expr: &WhereExpr) -> Option<bool> {
    match expr {
        WhereExpr::Cmp {
            binding: name,
            key,
            op,
            value,
        } => {
            let node = binding.nodes.get(name)?;
            match node.properties.get(key) {
                None => None,
                Some(got) => Some(compare_values(got, value, *op)),
            }
        }
        WhereExpr::In {
            binding: name,
            key,
            values,
        } => {
            let node = binding.nodes.get(name)?;
            match node.properties.get(key) {
                None => None,
                Some(got) => Some(values.iter().any(|value| got == value)),
            }
        }
        WhereExpr::And(left, right) => {
            match (eval_where(binding, left), eval_where(binding, right)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        }
        WhereExpr::Or(left, right) => match (eval_where(binding, left), eval_where(binding, right))
        {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        WhereExpr::Not(inner) => eval_where(binding, inner).map(|value| !value),
    }
}

fn compare_values(left: &Value, right: &Value, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => left == right,
        CmpOp::Ne => left != right,
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
            let Some(ord) = value_ord(left, right) else {
                return false;
            };
            match op {
                CmpOp::Lt => ord.is_lt(),
                CmpOp::Le => ord.is_le(),
                CmpOp::Gt => ord.is_gt(),
                CmpOp::Ge => ord.is_ge(),
                _ => false,
            }
        }
    }
}

fn value_ord(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => a.as_f64()?.partial_cmp(&b.as_f64()?),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn reverse_dir(direction: EdgeDirection) -> EdgeDirection {
    match direction {
        EdgeDirection::Out => EdgeDirection::In,
        EdgeDirection::In => EdgeDirection::Out,
        EdgeDirection::Both => EdgeDirection::Both,
    }
}

impl CypherGraph for InMemoryGraphStore {
    fn get_node(&self, id: &str) -> Result<Option<NodeRecord>, GraphStoreError> {
        Ok(InMemoryGraphStore::get_node(self, id).cloned())
    }

    fn query_nodes(&self, query: NodeQuery) -> Result<Vec<NodeRecord>, GraphStoreError> {
        Ok(InMemoryGraphStore::query_nodes(self, query))
    }

    fn neighbors(&self, query: NeighborQuery) -> Result<Vec<NeighborHit>, GraphStoreError> {
        Ok(InMemoryGraphStore::neighbors(self, query))
    }

    fn graph_version(&self) -> Result<u64, GraphStoreError> {
        Ok(self.stats().version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_cypher_pest;
    use rustyred_thg_core::{EdgeRecord, NodeRecord};
    use std::collections::BTreeMap;

    fn fixture() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::new();
        for (id, name, n) in [("a", "Ada", 1), ("b", "Grace", 2), ("c", "Kay", 3)] {
            store
                .upsert_node(NodeRecord::new(
                    id,
                    ["Person"],
                    json!({"name": name, "n": n}),
                ))
                .unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new("e1", "a", "KNOWS", "b", json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new("e2", "b", "KNOWS", "c", json!({})))
            .unwrap();
        store
    }

    fn run(store: &InMemoryGraphStore, query: &str) -> ReadOutcome {
        let parsed = parse_cypher_pest(query, &BTreeMap::new()).unwrap();
        execute_read(store, &parsed, None, 200).unwrap()
    }

    #[test]
    fn executes_boolean_where() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (n:Person) WHERE n.n > 1 AND n.n <= 3 RETURN n.name",
        );
        assert_eq!(out.rows.len(), 2);
    }

    #[test]
    fn executes_inbound_edge() {
        let store = fixture();
        let out = run(&store, "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b");
        assert!(!out.rows.is_empty());
    }

    #[test]
    fn executes_in_predicate() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (n:Person) WHERE n.name IN ['Ada', 'Kay'] RETURN n",
        );
        assert_eq!(out.rows.len(), 2);
    }

    #[test]
    fn executes_var_length_chain() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (a:Person {name: 'Ada'})-[:KNOWS*1..2]->(b:Person) RETURN b",
        );
        assert_eq!(out.rows.len(), 2);
    }

    #[test]
    fn executes_path_projection_and_alias() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH p = (a:Person {name: 'Ada'})-[:KNOWS]->(b:Person) RETURN length(p) AS hops, nodes(p) AS ns",
        );
        assert_eq!(out.columns, vec!["hops".to_string(), "ns".to_string()]);
        assert_eq!(out.rows[0][0], json!(1));
        assert_eq!(out.rows[0][1], json!(["a", "b"]));
    }

    #[test]
    fn executes_inbound_var_length() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (a:Person {name: 'Kay'})<-[:KNOWS*1..2]-(b:Person) RETURN b.name",
        );
        assert_eq!(out.rows.len(), 2);
    }

    #[test]
    fn executes_distinct() {
        let store = fixture();
        let out = run(&store, "MATCH (n:Person) RETURN DISTINCT n");
        assert_eq!(out.rows.len(), 3);
    }

    #[test]
    fn executes_count_as_one_row() {
        let store = fixture();
        let out = run(&store, "MATCH (n:Person) RETURN count(n)");
        assert_eq!(out.columns, vec!["count(n)".to_string()]);
        assert_eq!(out.rows, vec![vec![json!(3)]]);
    }

    #[test]
    fn executes_with_group_and_count() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (n:Person) WITH n.name AS name, count(n) AS c RETURN name, c",
        );
        assert_eq!(out.columns, vec!["name".to_string(), "c".to_string()]);
        assert_eq!(out.rows.len(), 3);
        assert!(out.rows.iter().all(|row| row[1] == json!(1)));
    }

    #[test]
    fn executes_order_by_and_limit() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (n:Person) RETURN n.name ORDER BY n.n DESC LIMIT 1",
        );
        assert_eq!(out.rows, vec![vec![json!("Kay")]]);
        assert!(!out.truncated);
    }

    #[test]
    fn residual_where_is_not_prefix_truncated() {
        let mut store = InMemoryGraphStore::new();
        for i in 0..40 {
            store
                .upsert_node(NodeRecord::new(
                    format!("p{i}"),
                    ["Person"],
                    json!({ "name": format!("n{i}"), "n": i }),
                ))
                .unwrap();
        }
        let parsed = parse_cypher_pest(
            "MATCH (n:Person) WHERE n.n > 35 RETURN n.name",
            &BTreeMap::new(),
        )
        .unwrap();
        let out = execute_read(&store, &parsed, None, 5).unwrap();
        assert_eq!(out.rows.len(), 4);
    }

    #[test]
    fn zero_hop_range_emits_seed() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH (a:Person {name: 'Ada'})-[:KNOWS*0..0]->(b:Person) RETURN b.name",
        );
        assert_eq!(out.rows, vec![vec![json!("Ada")]]);
    }

    #[test]
    fn not_missing_property_does_not_keep_the_row() {
        let store = fixture();
        let out = run(&store, "MATCH (n:Person) WHERE NOT n.missing = 1 RETURN n");
        assert!(out.rows.is_empty());
    }

    #[test]
    fn var_length_relationships_use_edge_ids() {
        let store = fixture();
        let out = run(
            &store,
            "MATCH p = (a:Person {name: 'Ada'})-[:KNOWS*1..1]->(b:Person) RETURN relationships(p)",
        );
        assert_eq!(out.rows[0][0], json!(["e1"]));
    }
}
