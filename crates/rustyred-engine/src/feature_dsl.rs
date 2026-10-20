//! Sandboxed Feature DSL for ranking/relevance self-modification.
//!
//! The DSL is data, not source. Evaluation is a bounded tree walk over a closed
//! serde enum, with read-only access to a preloaded graph neighborhood.

use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph::{expand_bounded, expand_bounded_weighted, paths_shortest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const DEFAULT_MAX_AST_DEPTH: usize = 8;
pub const DEFAULT_MAX_TRAVERSAL_DEPTH: usize = 4;
pub const DEFAULT_MAX_STEPS: u32 = 256;
pub const DEFAULT_MAX_TRAVERSAL_NODES: usize = 100;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FeatureExpr {
    Const {
        #[serde(deserialize_with = "deserialize_feature_number")]
        value: f64,
    },
    String {
        value: String,
    },
    NodeField {
        key: String,
    },
    EdgeField {
        key: String,
    },
    Compare {
        operator: CompareOp,
        lhs: Box<FeatureExpr>,
        rhs: Box<FeatureExpr>,
    },
    And {
        lhs: Box<FeatureExpr>,
        rhs: Box<FeatureExpr>,
    },
    Or {
        lhs: Box<FeatureExpr>,
        rhs: Box<FeatureExpr>,
    },
    Not {
        inner: Box<FeatureExpr>,
    },
    Arith {
        operator: ArithOp,
        lhs: Box<FeatureExpr>,
        rhs: Box<FeatureExpr>,
    },
    Traverse {
        kind: TraverseKind,
        max_depth: usize,
    },
    Count {
        inner: Box<FeatureExpr>,
    },
    Exists {
        traversal: Box<FeatureExpr>,
        predicate: Box<FeatureExpr>,
    },
}

fn deserialize_feature_number<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let number = serde_json::Number::deserialize(deserializer)?;
    number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| serde::de::Error::custom("feature number is outside the finite f64 range"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraverseKind {
    Expand,
    ExpandWeighted {
        min_confidence: f64,
    },
    ShortestPath {
        #[serde(default)]
        target: TraversalTarget,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraversalTarget {
    #[default]
    Other,
    NodeId(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalValue {
    Number(f64),
    Bool(bool),
    Text(String),
    Nodes(Vec<String>),
    Sentinel(EvalSentinel),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalSentinel {
    BudgetExhausted,
    DepthExceeded,
    MissingBinding,
    InvalidTraversal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalResult {
    pub value: EvalValue,
    pub steps_used: u32,
    pub exhausted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureEvalBudget {
    pub max_steps: u32,
    pub max_ast_depth: usize,
    pub max_traversal_depth: usize,
    pub max_traversal_nodes: usize,
}

impl Default for FeatureEvalBudget {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_ast_depth: DEFAULT_MAX_AST_DEPTH,
            max_traversal_depth: DEFAULT_MAX_TRAVERSAL_DEPTH,
            max_traversal_nodes: DEFAULT_MAX_TRAVERSAL_NODES,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureEvalContext {
    pub focus: NodeRecord,
    pub other: Option<NodeRecord>,
    pub edge: Option<EdgeRecord>,
    pub nodes: BTreeMap<String, NodeRecord>,
    pub edges: Vec<EdgeRecord>,
}

impl FeatureEvalContext {
    pub fn new(focus: NodeRecord, nodes: Vec<NodeRecord>, edges: Vec<EdgeRecord>) -> Self {
        let mut by_id = nodes
            .into_iter()
            .map(|node| (node.id.clone(), node))
            .collect::<BTreeMap<_, _>>();
        by_id
            .entry(focus.id.clone())
            .or_insert_with(|| focus.clone());
        Self {
            focus,
            other: None,
            edge: None,
            nodes: by_id,
            edges,
        }
    }

    pub fn with_other(mut self, other: NodeRecord) -> Self {
        self.nodes
            .entry(other.id.clone())
            .or_insert_with(|| other.clone());
        self.other = Some(other);
        self
    }

    pub fn with_edge(mut self, edge: EdgeRecord) -> Self {
        self.edge = Some(edge);
        self
    }

    fn with_focus(&self, focus_id: &str) -> Option<Self> {
        let focus = self.nodes.get(focus_id)?.clone();
        Some(Self {
            focus,
            other: self.other.clone(),
            edge: self.edge.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicFeatureStatus {
    Proposed,
    Active,
    RolledBack,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicFeature {
    pub name: String,
    pub signature: Vec<String>,
    pub ast: FeatureExpr,
    pub status: DynamicFeatureStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_rule_id: Option<String>,
}

impl DynamicFeature {
    pub fn config_key(&self) -> String {
        format!("feature_dsl.{}", self.name)
    }

    pub fn config_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

pub fn eval_feature(
    expr: &FeatureExpr,
    context: &FeatureEvalContext,
    budget: FeatureEvalBudget,
) -> EvalResult {
    let mut state = EvalState {
        budget,
        steps_used: 0,
        exhausted: false,
    };
    let value = eval(expr, context, &mut state, 0);
    EvalResult {
        value,
        steps_used: state.steps_used,
        exhausted: state.exhausted,
    }
}

pub fn feature_score(
    expr: &FeatureExpr,
    context: &FeatureEvalContext,
    budget: FeatureEvalBudget,
) -> f32 {
    match eval_feature(expr, context, budget).value {
        EvalValue::Number(value) => finite(value).clamp(0.0, 1.0) as f32,
        EvalValue::Bool(value) => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        EvalValue::Nodes(nodes) => {
            (nodes.len() as f32 / DEFAULT_MAX_TRAVERSAL_NODES as f32).clamp(0.0, 1.0)
        }
        EvalValue::Text(text) => {
            if text.is_empty() {
                0.0
            } else {
                1.0
            }
        }
        EvalValue::Sentinel(_) => 0.0,
    }
}

struct EvalState {
    budget: FeatureEvalBudget,
    steps_used: u32,
    exhausted: bool,
}

impl EvalState {
    fn tick(&mut self) -> bool {
        if self.steps_used >= self.budget.max_steps {
            self.exhausted = true;
            return false;
        }
        self.steps_used += 1;
        true
    }
}

fn eval(
    expr: &FeatureExpr,
    context: &FeatureEvalContext,
    state: &mut EvalState,
    depth: usize,
) -> EvalValue {
    if !state.tick() {
        return EvalValue::Sentinel(EvalSentinel::BudgetExhausted);
    }
    if depth > state.budget.max_ast_depth {
        state.exhausted = true;
        return EvalValue::Sentinel(EvalSentinel::DepthExceeded);
    }

    match expr {
        FeatureExpr::Const { value } => EvalValue::Number(finite(*value)),
        FeatureExpr::String { value } => EvalValue::Text(value.clone()),
        FeatureExpr::NodeField { key } => node_field(&context.focus, key),
        FeatureExpr::EdgeField { key } => context
            .edge
            .as_ref()
            .map(|edge| edge_field(edge, key))
            .unwrap_or(EvalValue::Number(0.0)),
        FeatureExpr::Compare { operator, lhs, rhs } => {
            let left = eval(lhs, context, state, depth + 1);
            if matches!(left, EvalValue::Sentinel(_)) {
                return left;
            }
            let right = eval(rhs, context, state, depth + 1);
            if matches!(right, EvalValue::Sentinel(_)) {
                return right;
            }
            EvalValue::Bool(compare_values(*operator, &left, &right))
        }
        FeatureExpr::And { lhs, rhs } => {
            let left = eval(lhs, context, state, depth + 1);
            if matches!(left, EvalValue::Sentinel(_)) {
                return left;
            }
            if !left.as_bool() {
                return EvalValue::Bool(false);
            }
            let right = eval(rhs, context, state, depth + 1);
            if matches!(right, EvalValue::Sentinel(_)) {
                return right;
            }
            EvalValue::Bool(right.as_bool())
        }
        FeatureExpr::Or { lhs, rhs } => {
            let left = eval(lhs, context, state, depth + 1);
            if matches!(left, EvalValue::Sentinel(_)) {
                return left;
            }
            if left.as_bool() {
                return EvalValue::Bool(true);
            }
            let right = eval(rhs, context, state, depth + 1);
            if matches!(right, EvalValue::Sentinel(_)) {
                return right;
            }
            EvalValue::Bool(right.as_bool())
        }
        FeatureExpr::Not { inner } => {
            let value = eval(inner, context, state, depth + 1);
            if matches!(value, EvalValue::Sentinel(_)) {
                return value;
            }
            EvalValue::Bool(!value.as_bool())
        }
        FeatureExpr::Arith { operator, lhs, rhs } => {
            let left = eval(lhs, context, state, depth + 1);
            if matches!(left, EvalValue::Sentinel(_)) {
                return left;
            }
            let right = eval(rhs, context, state, depth + 1);
            if matches!(right, EvalValue::Sentinel(_)) {
                return right;
            }
            let left = left.as_number();
            let right = right.as_number();
            EvalValue::Number(match operator {
                ArithOp::Add => finite(left + right),
                ArithOp::Sub => finite(left - right),
                ArithOp::Mul => finite(left * right),
                ArithOp::Min => left.min(right),
                ArithOp::Max => left.max(right),
            })
        }
        FeatureExpr::Traverse { kind, max_depth } => traverse(kind, *max_depth, context, state),
        FeatureExpr::Count { inner } => match eval(inner, context, state, depth + 1) {
            EvalValue::Nodes(nodes) => EvalValue::Number(nodes.len() as f64),
            EvalValue::Sentinel(sentinel) => EvalValue::Sentinel(sentinel),
            _ => EvalValue::Number(0.0),
        },
        FeatureExpr::Exists {
            traversal,
            predicate,
        } => match eval(traversal, context, state, depth + 1) {
            EvalValue::Nodes(nodes) => {
                for node_id in nodes {
                    let Some(next_context) = context.with_focus(&node_id) else {
                        continue;
                    };
                    if eval(predicate, &next_context, state, depth + 1).as_bool() {
                        return EvalValue::Bool(true);
                    }
                }
                EvalValue::Bool(false)
            }
            EvalValue::Sentinel(sentinel) => EvalValue::Sentinel(sentinel),
            _ => EvalValue::Bool(false),
        },
    }
}

fn traverse(
    kind: &TraverseKind,
    requested_depth: usize,
    context: &FeatureEvalContext,
    state: &mut EvalState,
) -> EvalValue {
    let max_depth = requested_depth.min(state.budget.max_traversal_depth);
    match kind {
        TraverseKind::Expand => {
            let tuples = edge_tuples(&context.edges);
            EvalValue::Nodes(limit_nodes(
                expand_bounded(tuples, vec![context.focus.id.clone()], max_depth)
                    .into_iter()
                    .map(|(node_id, _)| node_id),
                state.budget.max_traversal_nodes,
            ))
        }
        TraverseKind::ExpandWeighted { min_confidence } => EvalValue::Nodes(limit_nodes(
            expand_bounded_weighted(
                &context.edges,
                std::slice::from_ref(&context.focus.id),
                max_depth,
                finite(*min_confidence).clamp(0.0, 1.0),
            ),
            state.budget.max_traversal_nodes,
        )),
        TraverseKind::ShortestPath { target } => {
            let target_id = match target {
                TraversalTarget::Other => context.other.as_ref().map(|node| node.id.clone()),
                TraversalTarget::NodeId(id) => Some(id.clone()),
            };
            let Some(target_id) = target_id else {
                return EvalValue::Sentinel(EvalSentinel::MissingBinding);
            };
            let path = paths_shortest(
                edge_tuples(&context.edges),
                context.focus.id.clone(),
                target_id,
                max_depth,
            );
            if path.is_empty() {
                EvalValue::Nodes(Vec::new())
            } else {
                EvalValue::Nodes(limit_nodes(path, state.budget.max_traversal_nodes))
            }
        }
    }
}

fn edge_tuples(edges: &[EdgeRecord]) -> Vec<(String, String, String)> {
    edges
        .iter()
        .filter(|edge| !edge.tombstone)
        .map(|edge| {
            (
                edge.from_id.clone(),
                edge.edge_type.clone(),
                edge.to_id.clone(),
            )
        })
        .collect()
}

fn limit_nodes<I>(nodes: I, limit: usize) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for node in nodes {
        if seen.insert(node.clone()) {
            out.push(node);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn node_field(node: &NodeRecord, key: &str) -> EvalValue {
    match key {
        "id" => EvalValue::Text(node.id.clone()),
        "label" => EvalValue::Text(node.labels.first().cloned().unwrap_or_default()),
        "labels_count" => EvalValue::Number(node.labels.len() as f64),
        "version" => EvalValue::Number(node.version as f64),
        "tombstone" => EvalValue::Bool(node.tombstone),
        _ => value_field(node.properties.get(key)),
    }
}

fn edge_field(edge: &EdgeRecord, key: &str) -> EvalValue {
    match key {
        "id" => EvalValue::Text(edge.id.clone()),
        "from_id" => EvalValue::Text(edge.from_id.clone()),
        "to_id" => EvalValue::Text(edge.to_id.clone()),
        "edge_type" | "type" => EvalValue::Text(edge.edge_type.clone()),
        "version" => EvalValue::Number(edge.version as f64),
        "tombstone" => EvalValue::Bool(edge.tombstone),
        "confidence" | "effective_confidence" => EvalValue::Number(edge.effective_confidence()),
        _ => value_field(edge.properties.get(key)),
    }
}

fn value_field(value: Option<&Value>) -> EvalValue {
    match value {
        Some(Value::Bool(value)) => EvalValue::Bool(*value),
        Some(Value::Number(value)) => EvalValue::Number(value.as_f64().map(finite).unwrap_or(0.0)),
        Some(Value::String(value)) => EvalValue::Text(value.clone()),
        Some(Value::Array(value)) => EvalValue::Number(value.len() as f64),
        _ => EvalValue::Number(0.0),
    }
}

fn compare_values(operator: CompareOp, left: &EvalValue, right: &EvalValue) -> bool {
    match operator {
        CompareOp::Eq => left.as_text() == right.as_text(),
        CompareOp::Ne => left.as_text() != right.as_text(),
        CompareOp::Lt => left.as_number() < right.as_number(),
        CompareOp::Le => left.as_number() <= right.as_number(),
        CompareOp::Gt => left.as_number() > right.as_number(),
        CompareOp::Ge => left.as_number() >= right.as_number(),
    }
}

impl EvalValue {
    fn as_number(&self) -> f64 {
        match self {
            Self::Number(value) => finite(*value),
            Self::Bool(value) => {
                if *value {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Text(value) => value.trim().parse::<f64>().unwrap_or(0.0),
            Self::Nodes(value) => value.len() as f64,
            Self::Sentinel(_) => 0.0,
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Number(value) => finite(*value) != 0.0,
            Self::Text(value) => !value.is_empty(),
            Self::Nodes(value) => !value.is_empty(),
            Self::Sentinel(_) => false,
        }
    }

    fn as_text(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Number(value) => finite(*value).to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Nodes(value) => value.join(","),
            Self::Sentinel(value) => format!("{value:?}"),
        }
    }
}

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

pub fn malicious_probe_expressions() -> Vec<FeatureExpr> {
    vec![
        FeatureExpr::NodeField {
            key: "__import__('os').system('id')".to_string(),
        },
        FeatureExpr::String {
            value: "lambda: open('/etc/passwd')".to_string(),
        },
        deep_not_chain(32),
    ]
}

fn deep_not_chain(depth: usize) -> FeatureExpr {
    let mut expr = FeatureExpr::Const { value: 1.0 };
    for _ in 0..depth {
        expr = FeatureExpr::Not {
            inner: Box::new(expr),
        };
    }
    expr
}

#[cfg(test)]
mod feature_dsl_tests {
    use super::*;
    use crate::graph::expand_bounded;
    use serde_json::json;

    #[test]
    fn ast_round_trips_through_json() {
        let expr = FeatureExpr::Compare {
            operator: CompareOp::Ge,
            lhs: Box::new(FeatureExpr::EdgeField {
                key: "confidence".to_string(),
            }),
            rhs: Box::new(FeatureExpr::Const { value: 0.5 }),
        };

        let encoded = serde_json::to_string(&expr).unwrap();
        let decoded: FeatureExpr = serde_json::from_str(&encoded).unwrap_or_else(|error| {
            panic!("FeatureExpr JSON failed to round-trip: {error}; encoded={encoded}")
        });

        assert_eq!(decoded, expr);
        assert!(!encoded.contains("eval"));
    }

    #[test]
    fn edge_confidence_compare_uses_defined_defaults() {
        let edge = EdgeRecord::new("e1", "a", "supports", "b", json!({})).with_confidence(0.6);
        let ctx = fixture_context().with_edge(edge);
        let expr = FeatureExpr::Compare {
            operator: CompareOp::Ge,
            lhs: Box::new(FeatureExpr::EdgeField {
                key: "confidence".to_string(),
            }),
            rhs: Box::new(FeatureExpr::Const { value: 0.5 }),
        };

        assert_eq!(
            eval_feature(&expr, &ctx, FeatureEvalBudget::default()).value,
            EvalValue::Bool(true)
        );

        let missing = FeatureExpr::Compare {
            operator: CompareOp::Eq,
            lhs: Box::new(FeatureExpr::NodeField {
                key: "missing_numeric".to_string(),
            }),
            rhs: Box::new(FeatureExpr::Const { value: 0.0 }),
        };
        assert_eq!(
            eval_feature(&missing, &ctx, FeatureEvalBudget::default()).value,
            EvalValue::Bool(true)
        );
    }

    #[test]
    fn traversal_delegates_to_graph_expand() {
        let ctx = fixture_context();
        let expr = FeatureExpr::Traverse {
            kind: TraverseKind::Expand,
            max_depth: 2,
        };
        let direct = expand_bounded(edge_tuples(&ctx.edges), vec!["a".to_string()], 2)
            .into_iter()
            .map(|(node, _)| node)
            .collect::<Vec<_>>();

        assert_eq!(
            eval_feature(&expr, &ctx, FeatureEvalBudget::default()).value,
            EvalValue::Nodes(direct)
        );
    }

    #[test]
    fn pathological_depth_returns_sentinel_within_budget() {
        let expr = deep_not_chain(32);
        let result = eval_feature(
            &expr,
            &fixture_context(),
            FeatureEvalBudget {
                max_steps: 64,
                max_ast_depth: 8,
                max_traversal_depth: 4,
                max_traversal_nodes: 100,
            },
        );

        assert_eq!(
            result.value,
            EvalValue::Sentinel(EvalSentinel::DepthExceeded)
        );
        assert!(result.exhausted);
        assert!(result.steps_used <= 64);
    }

    #[test]
    fn deterministic_fuzz_corpus_never_panics_or_overruns_budget() {
        let ctx = fixture_context();
        for seed in 0..10_000u64 {
            let expr = random_expr(seed, 0);
            let result = std::panic::catch_unwind(|| {
                eval_feature(
                    &expr,
                    &ctx,
                    FeatureEvalBudget {
                        max_steps: 128,
                        max_ast_depth: 8,
                        max_traversal_depth: 4,
                        max_traversal_nodes: 32,
                    },
                )
            })
            .expect("feature DSL evaluation must not panic");
            assert!(result.steps_used <= 128);
        }

        for expr in malicious_probe_expressions() {
            let result = eval_feature(
                &expr,
                &ctx,
                FeatureEvalBudget {
                    max_steps: 64,
                    max_ast_depth: 8,
                    max_traversal_depth: 4,
                    max_traversal_nodes: 32,
                },
            );
            assert!(result.steps_used <= 64);
        }
    }

    #[test]
    fn dynamic_feature_serializes_as_config_value() {
        let feature = DynamicFeature {
            name: "has_supporting_neighbor".to_string(),
            signature: vec!["self".to_string()],
            ast: FeatureExpr::Count {
                inner: Box::new(FeatureExpr::Traverse {
                    kind: TraverseKind::Expand,
                    max_depth: 1,
                }),
            },
            status: DynamicFeatureStatus::Proposed,
            source_rule_id: Some("rule:1".to_string()),
        };

        assert_eq!(feature.config_key(), "feature_dsl.has_supporting_neighbor");
        assert_eq!(feature.config_value()["name"], "has_supporting_neighbor");
        assert_eq!(feature.config_value()["status"], "proposed");
    }

    fn fixture_context() -> FeatureEvalContext {
        let a = NodeRecord::new(
            "a",
            ["Claim"],
            json!({"score": 0.7, "title": "replicates method"}),
        );
        let b = NodeRecord::new("b", ["Claim"], json!({"score": 0.3}));
        let c = NodeRecord::new("c", ["Claim"], json!({"score": 0.9}));
        let edges = vec![
            EdgeRecord::new("e1", "a", "supports", "b", json!({})).with_confidence(0.9),
            EdgeRecord::new("e2", "b", "supports", "c", json!({})).with_confidence(0.8),
        ];
        FeatureEvalContext::new(a, vec![b.clone(), c], edges).with_other(b)
    }

    fn random_expr(seed: u64, depth: usize) -> FeatureExpr {
        let next = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        if depth >= 6 {
            return match seed % 4 {
                0 => FeatureExpr::Const {
                    value: (seed % 100) as f64 / 100.0,
                },
                1 => FeatureExpr::NodeField {
                    key: "score".to_string(),
                },
                2 => FeatureExpr::EdgeField {
                    key: "confidence".to_string(),
                },
                _ => FeatureExpr::String {
                    value: "x".to_string(),
                },
            };
        }
        match seed % 9 {
            0 => FeatureExpr::And {
                lhs: Box::new(random_expr(next, depth + 1)),
                rhs: Box::new(random_expr(next.rotate_left(7), depth + 1)),
            },
            1 => FeatureExpr::Or {
                lhs: Box::new(random_expr(next, depth + 1)),
                rhs: Box::new(random_expr(next.rotate_left(11), depth + 1)),
            },
            2 => FeatureExpr::Compare {
                operator: CompareOp::Ge,
                lhs: Box::new(random_expr(next, depth + 1)),
                rhs: Box::new(random_expr(next.rotate_left(13), depth + 1)),
            },
            3 => FeatureExpr::Arith {
                operator: ArithOp::Max,
                lhs: Box::new(random_expr(next, depth + 1)),
                rhs: Box::new(random_expr(next.rotate_left(17), depth + 1)),
            },
            4 => FeatureExpr::Not {
                inner: Box::new(random_expr(next, depth + 1)),
            },
            5 => FeatureExpr::Count {
                inner: Box::new(FeatureExpr::Traverse {
                    kind: TraverseKind::Expand,
                    max_depth: (seed as usize % 8) + 1,
                }),
            },
            6 => FeatureExpr::Exists {
                traversal: Box::new(FeatureExpr::Traverse {
                    kind: TraverseKind::ExpandWeighted {
                        min_confidence: 0.5,
                    },
                    max_depth: 2,
                }),
                predicate: Box::new(FeatureExpr::Compare {
                    operator: CompareOp::Ge,
                    lhs: Box::new(FeatureExpr::NodeField {
                        key: "score".to_string(),
                    }),
                    rhs: Box::new(FeatureExpr::Const { value: 0.5 }),
                }),
            },
            7 => FeatureExpr::Traverse {
                kind: TraverseKind::ShortestPath {
                    target: TraversalTarget::Other,
                },
                max_depth: 4,
            },
            _ => FeatureExpr::NodeField {
                key: "__proto__".to_string(),
            },
        }
    }
}
