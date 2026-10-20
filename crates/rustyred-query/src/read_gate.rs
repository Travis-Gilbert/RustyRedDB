use rustyred_thg_core::{Direction, NeighborQuery, NodeQuery, NodeRecord};

use crate::cypher::ast::{
    CypherPattern, EdgeDirection, ParsedCypher, ReturnItem, WhereExpr, WriteClause,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClauseClass {
    Read,
    Write,
}

/// Walk the AST after pest parse. Unknown clause kinds fail closed.
pub fn classify_ast(parsed: &ParsedCypher) -> Result<ClauseClass, crate::error::CypherError> {
    if !parsed.writes.is_empty() {
        for write in &parsed.writes {
            match write {
                WriteClause::CreateNode { .. }
                | WriteClause::CreateEdge { .. }
                | WriteClause::Merge { .. }
                | WriteClause::Set { .. }
                | WriteClause::Delete { .. } => {}
            }
        }
        return Ok(ClauseClass::Write);
    }
    classify_pattern(&parsed.pattern)?;
    classify_returns(&parsed.returns)?;
    if let Some(expr) = &parsed.where_expr {
        classify_where(expr)?;
    }
    Ok(ClauseClass::Read)
}

fn classify_pattern(pattern: &CypherPattern) -> Result<(), crate::error::CypherError> {
    match pattern {
        CypherPattern::Node(_)
        | CypherPattern::Edge(_)
        | CypherPattern::EdgeChain(_)
        | CypherPattern::EdgeVarLength(_) => Ok(()),
    }
}

fn classify_returns(returns: &[ReturnItem]) -> Result<(), crate::error::CypherError> {
    for item in returns {
        match item {
            ReturnItem::Variable(_)
            | ReturnItem::Property { .. }
            | ReturnItem::Count { .. }
            | ReturnItem::Path { .. }
            | ReturnItem::PathNodes { .. }
            | ReturnItem::PathRels { .. }
            | ReturnItem::PathLength { .. }
            | ReturnItem::Aggregate { .. } => {}
            ReturnItem::Aliased { item, .. } => classify_returns(std::slice::from_ref(item))?,
        }
    }
    Ok(())
}

fn classify_where(expr: &WhereExpr) -> Result<(), crate::error::CypherError> {
    match expr {
        WhereExpr::Cmp { .. } | WhereExpr::In { .. } => Ok(()),
        WhereExpr::And(left, right) | WhereExpr::Or(left, right) => {
            classify_where(left)?;
            classify_where(right)
        }
        WhereExpr::Not(inner) => classify_where(inner),
    }
}

pub fn referenced_labels(parsed: &ParsedCypher) -> Vec<String> {
    let mut labels = Vec::new();
    push_pattern_labels(&parsed.pattern, &mut labels);
    labels
}

pub fn referenced_edge_types(parsed: &ParsedCypher) -> Vec<String> {
    let mut types = Vec::new();
    push_pattern_edges(&parsed.pattern, &mut types);
    types
}

pub fn referenced_properties(parsed: &ParsedCypher) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(expr) = &parsed.where_expr {
        push_where_keys(expr, &mut keys);
    }
    keys
}

fn push_where_keys(expr: &WhereExpr, keys: &mut Vec<String>) {
    match expr {
        WhereExpr::Cmp { key, .. } | WhereExpr::In { key, .. } => keys.push(key.clone()),
        WhereExpr::And(left, right) | WhereExpr::Or(left, right) => {
            push_where_keys(left, keys);
            push_where_keys(right, keys);
        }
        WhereExpr::Not(inner) => push_where_keys(inner, keys),
    }
}

fn push_pattern_labels(pattern: &CypherPattern, labels: &mut Vec<String>) {
    match pattern {
        CypherPattern::Node(node) => push_node_label(node, labels),
        CypherPattern::Edge(edge) => {
            push_node_label(&edge.left, labels);
            push_node_label(&edge.right, labels);
        }
        CypherPattern::EdgeChain(chain) => {
            push_node_label(&chain.start, labels);
            for step in &chain.steps {
                push_node_label(&step.target, labels);
            }
        }
        CypherPattern::EdgeVarLength(var) => {
            push_node_label(&var.from, labels);
            push_node_label(&var.to, labels);
        }
    }
}

fn push_node_label(node: &crate::cypher::ast::NodePattern, labels: &mut Vec<String>) {
    if let Some(label) = &node.label {
        labels.push(label.clone());
    }
}

fn push_pattern_edges(pattern: &CypherPattern, types: &mut Vec<String>) {
    match pattern {
        CypherPattern::Node(_) => {}
        CypherPattern::Edge(edge) => types.push(edge.edge_type.clone()),
        CypherPattern::EdgeChain(chain) => {
            for step in &chain.steps {
                types.push(step.edge_type.clone());
            }
        }
        CypherPattern::EdgeVarLength(var) => types.push(var.edge_type.clone()),
    }
}

pub fn write_clause_name(parsed: &ParsedCypher) -> Option<&'static str> {
    parsed.writes.first().map(|write| match write {
        WriteClause::CreateNode { .. } | WriteClause::CreateEdge { .. } => "CREATE",
        WriteClause::Merge { .. } => "MERGE",
        WriteClause::Set { .. } => "SET",
        WriteClause::Delete { detach: true, .. } => "DETACH DELETE",
        WriteClause::Delete { .. } => "DELETE",
    })
}

pub fn neighbor_direction(direction: EdgeDirection) -> Vec<Direction> {
    match direction {
        EdgeDirection::Out => vec![Direction::Out],
        EdgeDirection::In => vec![Direction::In],
        EdgeDirection::Both => vec![Direction::Out, Direction::In],
    }
}

pub fn node_query(
    label: Option<&str>,
    properties: &[(String, serde_json::Value)],
    limit: usize,
) -> NodeQuery {
    let mut query = NodeQuery {
        label: label.map(str::to_string),
        limit: Some(limit.max(1)),
        ..NodeQuery::default()
    };
    for (key, value) in properties {
        query.properties.insert(key.clone(), value.clone());
    }
    query
}

pub fn neighbor_query(node_id: &str, edge_type: &str, direction: Direction) -> NeighborQuery {
    NeighborQuery {
        node_id: node_id.to_string(),
        direction,
        edge_type: Some(edge_type.to_string()),
        include_expired: false,
    }
}

pub fn node_matches_pattern(
    node: &NodeRecord,
    label: Option<&str>,
    properties: &std::collections::BTreeMap<String, serde_json::Value>,
) -> bool {
    if let Some(label) = label {
        if !node.labels.iter().any(|candidate| candidate == label) {
            return false;
        }
    }
    let Some(object) = node.properties.as_object() else {
        return properties.is_empty();
    };
    properties
        .iter()
        .all(|(key, value)| object.get(key) == Some(value))
}
