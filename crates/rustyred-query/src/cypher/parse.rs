use std::collections::BTreeMap;

// ponytail: avoids intermediate Vec allocation from split_whitespace().collect::<Vec<_>>().join()
fn collapse_ws(s: &str, sep: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push_str(sep);
        }
        out.push_str(w);
    }
    out
}

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use serde_json::Value;

use super::ast::{
    AggOp, CmpOp, CypherPattern, EdgeChain, EdgeDirection, EdgePattern, EdgeStep, EdgeVarLength,
    MergeBranch, NodePattern, OrderBy, ParsedCypher, ReturnItem, SetExpr,
    WhereExpr, WithClause, WithItem, WriteClause,
};
use crate::error::{pest_span, CypherError as QuerySurfaceError};

#[derive(Parser)]
#[grammar = "cypher/grammar.pest"]
pub struct CypherPestParser;

const DEFAULT_LIMIT: usize = 100;

pub fn parse_cypher_pest(
    query: &str,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let normalized = normalize_query(query);
    if normalized.is_empty() {
        return Err(QuerySurfaceError::invalid(
            "empty_cypher_query",
            "query is required",
        ));
    }

    let mut pairs = CypherPestParser::parse(Rule::query, &normalized).map_err(|err| {
        QuerySurfaceError::invalid("parse_error", format!("pest parse error: {err}"))
            .with_span(pest_span(&normalized, &err))
    })?;
    let query_pair = pairs
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "empty pest output"))?;

    let body_pair = query_pair
        .into_inner()
        .find(|p| matches!(p.as_rule(), Rule::read_query | Rule::write_query))
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "missing query body"))?;
    // Clone the normalized string before passing the body pair onward. The
    // pest Pair holds `&str` slices into `normalized`, so we cannot move the
    // String while the Pair is still in scope.
    let normalized_owned = normalized.clone();
    match body_pair.as_rule() {
        Rule::read_query => parse_read_query(normalized_owned, body_pair, params),
        Rule::write_query => parse_write_query(normalized_owned, body_pair, params),
        _ => unreachable!(),
    }
}

fn parse_read_query(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let mut pattern: Option<CypherPattern> = None;
    let mut where_expr: Option<WhereExpr> = None;
    let mut returns: Vec<ReturnItem> = Vec::new();
    let mut limit: usize = DEFAULT_LIMIT;
    let mut limit_specified = false;
    let mut distinct = false;
    let mut with_clause: Option<WithClause> = None;
    let mut order_by: Vec<OrderBy> = Vec::new();
    let mut skip: Option<usize> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_clause => {
                pattern = Some(parse_match(inner, params)?);
            }
            Rule::where_clause => {
                where_expr = Some(parse_where_expr(inner, params)?);
            }
            Rule::with_clause => {
                with_clause = Some(parse_with_clause(inner, params)?);
            }
            Rule::return_clause => {
                let parsed_return = parse_return_clause(inner)?;
                distinct = parsed_return.0;
                returns = parsed_return.1;
            }
            Rule::order_clause => {
                order_by = parse_order_clause(inner)?;
            }
            Rule::skip_clause => {
                skip = Some(parse_skip_clause(inner)?);
            }
            Rule::limit_clause => {
                limit = parse_limit_literal(inner)?;
                limit_specified = true;
            }
            Rule::EOI => {}
            other => {
                return Err(QuerySurfaceError::invalid(
                    "invalid_cypher_query",
                    format!("unexpected clause rule: {other:?}"),
                ));
            }
        }
    }

    let pattern = pattern.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "missing MATCH clause")
    })?;

    if returns.is_empty() {
        return Err(QuerySurfaceError::invalid(
            "empty_return_clause",
            "RETURN clause is required",
        ));
    }

    // If the MATCH bound a path alias (e.g. `MATCH p = ...`), any RETURN p
    // becomes a ReturnItem::Path rather than ReturnItem::Variable.
    let path_binding = match &pattern {
        CypherPattern::Edge(edge) => edge.path_binding.clone(),
        CypherPattern::EdgeChain(c) => c.path_binding.clone(),
        CypherPattern::EdgeVarLength(v) => v.path_binding.clone(),
        CypherPattern::Node(_) => None,
    };
    if let Some(name) = &path_binding {
        for item in returns.iter_mut() {
            rewrite_path_return(item, name);
        }
    }

    Ok(ParsedCypher {
        normalized,
        pattern,
        where_filter: where_expr.as_ref().and_then(WhereExpr::as_eq_filter),
        where_expr,
        returns,
        limit,
        limit_specified,
        distinct,
        writes: Vec::new(),
        with_clause,
        order_by,
        skip,
        path_binding,
    })
}

fn parse_write_query(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "empty write query"))?;
    match inner.as_rule() {
        Rule::create_node_only => parse_create_node_only(normalized, inner, params),
        Rule::merge_clause => parse_merge_clause(normalized, inner, params),
        Rule::match_with_set => parse_match_with_set(normalized, inner, params),
        Rule::match_with_delete => parse_match_with_delete(normalized, inner, params),
        other => Err(QuerySurfaceError::invalid(
            "invalid_cypher_query",
            format!("unsupported write rule: {other:?}"),
        )),
    }
}

fn parse_create_node_only(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let create_target = pair.into_inner().next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "CREATE missing target")
    })?;
    let target_inner = create_target.into_inner().next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "CREATE missing target inner")
    })?;
    match target_inner.as_rule() {
        Rule::node_pattern => {
            let node = parse_node_pattern(target_inner, params)?;
            Ok(ParsedCypher {
                normalized,
                pattern: CypherPattern::Node(node.clone()),
                where_filter: None,
                where_expr: None,
                returns: Vec::new(),
                limit: 0,
                limit_specified: false,
                distinct: false,
                writes: vec![WriteClause::CreateNode { node }],
                with_clause: None,
                order_by: Vec::new(),
                skip: None,
                path_binding: None,
            })
        }
        Rule::create_edge_form => {
            let mut iter = target_inner.into_inner();
            let left_pair = iter.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "CREATE edge missing left node")
            })?;
            let left = parse_node_pattern(left_pair, params)?;
            let continuation = iter.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "CREATE edge missing relation")
            })?;
            let step = parse_edge_continuation(continuation, params)?;
            let edge = EdgePattern {
                left: left.clone(),
                edge_type: step.edge_type,
                right: step.target,
                direction: step.direction,
                path_binding: None,
            };
            Ok(ParsedCypher {
                normalized,
                pattern: CypherPattern::Edge(edge.clone()),
                where_filter: None,
                where_expr: None,
                returns: Vec::new(),
                limit: 0,
                limit_specified: false,
                distinct: false,
                writes: vec![WriteClause::CreateEdge { edge }],
                with_clause: None,
                order_by: Vec::new(),
                skip: None,
                path_binding: None,
            })
        }
        other => Err(QuerySurfaceError::invalid(
            "invalid_cypher_query",
            format!("unexpected CREATE target: {other:?}"),
        )),
    }
}

fn parse_merge_clause(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let mut node: Option<NodePattern> = None;
    let mut on_create: Option<MergeBranch> = None;
    let mut on_match: Option<MergeBranch> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::node_pattern => {
                node = Some(parse_node_pattern(child, params)?);
            }
            Rule::on_create_branch => {
                let set_list = child.into_inner().next().ok_or_else(|| {
                    QuerySurfaceError::invalid("invalid_cypher_query", "MERGE branch missing SET")
                })?;
                on_create = Some(parse_set_list(set_list, params)?);
            }
            Rule::on_match_branch => {
                let set_list = child.into_inner().next().ok_or_else(|| {
                    QuerySurfaceError::invalid("invalid_cypher_query", "MERGE branch missing SET")
                })?;
                on_match = Some(parse_set_list(set_list, params)?);
            }
            _ => {}
        }
    }
    let node = node.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "MERGE missing node pattern")
    })?;
    Ok(ParsedCypher {
        normalized,
        pattern: CypherPattern::Node(node.clone()),
        where_filter: None,
        returns: Vec::new(),
        limit: 0,
        writes: vec![WriteClause::Merge {
            node,
            on_create,
            on_match,
        }],
        with_clause: None,
        order_by: Vec::new(),
        skip: None,
        where_expr: None,
        limit_specified: false,
        distinct: false,
        path_binding: None,
    })
}

fn parse_match_with_set(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let mut pattern: Option<CypherPattern> = None;
    let mut where_expr: Option<WhereExpr> = None;
    let mut set_list_pair: Option<Pair<Rule>> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::match_clause => pattern = Some(parse_match(child, params)?),
            Rule::where_clause => where_expr = Some(parse_where_expr(child, params)?),
            Rule::set_list => set_list_pair = Some(child),
            _ => {}
        }
    }
    let pattern = pattern.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "SET requires MATCH clause")
    })?;
    let set_list = set_list_pair
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "missing SET list"))?;
    let branch = parse_set_list(set_list, params)?;
    let writes: Vec<WriteClause> = branch
        .sets
        .into_iter()
        .map(|(binding, key, value)| WriteClause::Set {
            binding,
            key,
            value,
        })
        .collect();
    Ok(ParsedCypher {
        normalized,
        pattern,
        where_filter: where_expr.as_ref().and_then(WhereExpr::as_eq_filter),
        where_expr,
        returns: Vec::new(),
        limit: 0,
        writes,
        with_clause: None,
        order_by: Vec::new(),
        skip: None,
        limit_specified: false,
        distinct: false,
        path_binding: None,
    })
}

fn parse_match_with_delete(
    normalized: String,
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<ParsedCypher, QuerySurfaceError> {
    let mut pattern: Option<CypherPattern> = None;
    let mut where_expr: Option<WhereExpr> = None;
    let mut delete: Option<(String, bool)> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::match_clause => pattern = Some(parse_match(child, params)?),
            Rule::where_clause => where_expr = Some(parse_where_expr(child, params)?),
            Rule::delete_form => {
                let form = child.into_inner().next().ok_or_else(|| {
                    QuerySurfaceError::invalid(
                        "invalid_cypher_query",
                        "delete form missing inner rule",
                    )
                })?;
                let detach = matches!(form.as_rule(), Rule::detach_delete);
                let binding = form
                    .into_inner()
                    .next()
                    .ok_or_else(|| {
                        QuerySurfaceError::invalid(
                            "invalid_cypher_query",
                            "DELETE missing target binding",
                        )
                    })?
                    .as_str()
                    .to_string();
                delete = Some((binding, detach));
            }
            _ => {}
        }
    }
    let pattern = pattern.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "DELETE requires MATCH clause")
    })?;
    let (binding, detach) = delete.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "DELETE clause missing")
    })?;
    Ok(ParsedCypher {
        normalized,
        pattern,
        where_filter: where_expr.as_ref().and_then(WhereExpr::as_eq_filter),
        where_expr,
        returns: Vec::new(),
        limit: 0,
        writes: vec![WriteClause::Delete { binding, detach }],
        with_clause: None,
        order_by: Vec::new(),
        skip: None,
        limit_specified: false,
        distinct: false,
        path_binding: None,
    })
}

fn parse_set_list(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<MergeBranch, QuerySurfaceError> {
    let mut sets: Vec<(String, String, SetExpr)> = Vec::new();
    for child in pair.into_inner() {
        if !matches!(child.as_rule(), Rule::set_assignment) {
            continue;
        }
        let mut iter = child.into_inner();
        let path_pair = iter.next().ok_or_else(|| {
            QuerySurfaceError::invalid("invalid_cypher_query", "SET missing property path")
        })?;
        let mut idents = path_pair.into_inner();
        let binding = idents
            .next()
            .ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "SET path missing binding")
            })?
            .as_str()
            .to_string();
        let key = idents
            .next()
            .ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "SET path missing key")
            })?
            .as_str()
            .to_string();
        let value_pair = iter.next().ok_or_else(|| {
            QuerySurfaceError::invalid("invalid_cypher_query", "SET missing right-hand side")
        })?;
        let value = parse_set_value_expr(value_pair, params)?;
        sets.push((binding, key, value));
    }
    Ok(MergeBranch { sets })
}

fn parse_set_value_expr(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<SetExpr, QuerySurfaceError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "empty SET value"))?;
    match inner.as_rule() {
        Rule::increment_expr => {
            let mut iter = inner.into_inner();
            let path_pair = iter.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "increment missing base path")
            })?;
            let mut idents = path_pair.into_inner();
            let base_binding = idents
                .next()
                .ok_or_else(|| {
                    QuerySurfaceError::invalid("invalid_cypher_query", "increment missing binding")
                })?
                .as_str()
                .to_string();
            let base_key = idents
                .next()
                .ok_or_else(|| {
                    QuerySurfaceError::invalid("invalid_cypher_query", "increment missing key")
                })?
                .as_str()
                .to_string();
            let delta_pair = iter.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_cypher_query", "increment missing delta")
            })?;
            let delta = parse_value(delta_pair, params)?;
            Ok(SetExpr::Increment {
                base_binding,
                base_key,
                delta,
            })
        }
        Rule::value => Ok(SetExpr::Literal(parse_value(inner, params)?)),
        other => Err(QuerySurfaceError::invalid(
            "invalid_cypher_query",
            format!("unsupported SET value rule: {other:?}"),
        )),
    }
}

fn normalize_query(query: &str) -> String {
    collapse_ws(&query, " ")
}

fn parse_match(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<CypherPattern, QuerySurfaceError> {
    let mut path_binding: Option<String> = None;
    let mut pattern_pair: Option<Pair<Rule>> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::path_binding => {
                if let Some(ident) = child.into_inner().next() {
                    path_binding = Some(ident.as_str().to_string());
                }
            }
            Rule::pattern => {
                pattern_pair = Some(child);
            }
            _ => {}
        }
    }
    let pattern_pair = pattern_pair.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "MATCH pattern missing")
    })?;

    let inner_pair = pattern_pair
        .into_inner()
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "empty MATCH pattern"))?;
    match inner_pair.as_rule() {
        Rule::node_pattern => {
            let node = parse_node_pattern(inner_pair, params)?;
            Ok(CypherPattern::Node(node))
        }
        Rule::edge_chain_pattern => parse_edge_chain_pattern(inner_pair, params, path_binding),
        other => Err(QuerySurfaceError::invalid(
            "invalid_cypher_query",
            format!("unexpected pattern rule: {other:?}"),
        )),
    }
}

fn parse_edge_continuation(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<EdgeStep, QuerySurfaceError> {
    let inner = pair.into_inner().next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "empty edge continuation")
    })?;
    let direction = match inner.as_rule() {
        Rule::outbound_cont => EdgeDirection::Out,
        Rule::inbound_cont => EdgeDirection::In,
        Rule::undirected_cont => EdgeDirection::Both,
        other => {
            return Err(QuerySurfaceError::invalid(
                "invalid_cypher_query",
                format!("unexpected edge continuation: {other:?}"),
            ))
        }
    };
    let mut sub = inner.into_inner();
    let rel_body = sub.next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "missing rel body in chain")
    })?;
    let target_pair = sub.next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "missing target in chain")
    })?;
    let (edge_type, min, max, is_var_length) = parse_rel_body(rel_body)?;
    let target = parse_node_pattern(target_pair, params)?;
    Ok(EdgeStep {
        edge_type,
        target,
        direction,
        min,
        max,
        is_var_length,
    })
}

fn parse_rel_body(
    pair: Pair<Rule>,
) -> Result<(String, usize, Option<usize>, bool), QuerySurfaceError> {
    let mut edge_type = None;
    let mut min = 1usize;
    let mut max = Some(1usize);
    let mut is_var_length = false;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::rel_type => edge_type = Some(parse_rel_type(child)?),
            Rule::var_length_star => {
                is_var_length = true;
                let (parsed_min, parsed_max) = parse_var_length_star(child)?;
                min = parsed_min;
                max = parsed_max;
            }
            _ => {}
        }
    }
    let edge_type = edge_type.ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_cypher_query", "relationship type missing")
    })?;
    Ok((edge_type, min, max, is_var_length))
}

fn parse_var_length_star(pair: Pair<Rule>) -> Result<(usize, Option<usize>), QuerySurfaceError> {
    for child in pair.into_inner() {
        if matches!(child.as_rule(), Rule::var_length_range) {
            return parse_var_length_range(child);
        }
    }
    Ok((1, None))
}

fn parse_var_length_range(pair: Pair<Rule>) -> Result<(usize, Option<usize>), QuerySurfaceError> {
    let mut numbers = pair.into_inner();
    let first = numbers
        .next()
        .ok_or_else(|| {
            QuerySurfaceError::invalid("invalid_cypher_query", "var-length range missing minimum")
        })?
        .as_str()
        .parse::<usize>()
        .map_err(|err| {
            QuerySurfaceError::invalid(
                "invalid_cypher_query",
                format!("invalid var-length min: {err}"),
            )
        })?;
    match numbers.next() {
        Some(second) => {
            let max = second.as_str().parse::<usize>().map_err(|err| {
                QuerySurfaceError::invalid(
                    "invalid_cypher_query",
                    format!("invalid var-length max: {err}"),
                )
            })?;
            Ok((first, Some(max)))
        }
        None => Ok((first, Some(first))),
    }
}

fn parse_edge_chain_pattern(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
    path_binding: Option<String>,
) -> Result<CypherPattern, QuerySurfaceError> {
    let mut iter = pair.into_inner();
    let start_pair = iter
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_cypher_query", "missing chain start"))?;
    let start = parse_node_pattern(start_pair, params)?;
    let mut steps: Vec<EdgeStep> = Vec::new();
    for cont in iter {
        if !matches!(cont.as_rule(), Rule::edge_continuation) {
            continue;
        }
        steps.push(parse_edge_continuation(cont, params)?);
    }
    if steps.len() == 1 {
        let step = steps.into_iter().next().expect("one step");
        if step.is_var_length {
            return Ok(CypherPattern::EdgeVarLength(EdgeVarLength {
                from: start,
                edge_type: step.edge_type,
                min: step.min,
                max: step.max,
                to: step.target,
                path_binding,
                direction: step.direction,
            }));
        }
        return Ok(CypherPattern::Edge(EdgePattern {
            left: start,
            edge_type: step.edge_type,
            right: step.target,
            direction: step.direction,
            path_binding,
        }));
    }
    Ok(CypherPattern::EdgeChain(EdgeChain {
        start,
        steps,
        path_binding,
    }))
}

fn parse_rel_type(pair: Pair<Rule>) -> Result<String, QuerySurfaceError> {
    for inner in pair.into_inner() {
        if matches!(inner.as_rule(), Rule::ident) {
            return Ok(inner.as_str().to_string());
        }
    }
    Err(QuerySurfaceError::invalid(
        "invalid_cypher_query",
        "relationship type missing identifier",
    ))
}

fn parse_node_pattern(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<NodePattern, QuerySurfaceError> {
    let mut binding: Option<String> = None;
    let mut label: Option<String> = None;
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binding => {
                binding = Some(inner.as_str().to_string());
            }
            Rule::label => {
                for child in inner.into_inner() {
                    if matches!(child.as_rule(), Rule::ident) {
                        label = Some(child.as_str().to_string());
                    }
                }
            }
            Rule::property_block => {
                properties = parse_property_block(inner, params)?;
            }
            _ => {}
        }
    }
    let binding = binding.unwrap_or_else(|| "_anon".to_string());
    Ok(NodePattern {
        binding,
        label,
        properties,
    })
}

fn parse_property_block(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, Value>, QuerySurfaceError> {
    let mut out = BTreeMap::new();
    for entry in pair.into_inner() {
        if !matches!(entry.as_rule(), Rule::property_pair) {
            continue;
        }
        let mut name: Option<String> = None;
        let mut value: Option<Value> = None;
        for child in entry.into_inner() {
            match child.as_rule() {
                Rule::ident if name.is_none() => name = Some(child.as_str().to_string()),
                Rule::value => value = Some(parse_value(child, params)?),
                _ => {}
            }
        }
        if let (Some(name), Some(value)) = (name, value) {
            out.insert(name, value);
        }
    }
    Ok(out)
}

fn parse_value(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<Value, QuerySurfaceError> {
    if let Some(inner) = pair.into_inner().next() {
        return match inner.as_rule() {
            Rule::param => {
                let name = inner.as_str().trim_start_matches('$').to_string();
                params
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| QuerySurfaceError::missing_param(&name))
            }
            Rule::string => {
                let raw = inner.as_str();
                let stripped = &raw[1..raw.len() - 1];
                Ok(Value::String(stripped.to_string()))
            }
            Rule::number => {
                let text = inner.as_str();
                if let Ok(int) = text.parse::<i64>() {
                    Ok(Value::Number(int.into()))
                } else if let Ok(float) = text.parse::<f64>() {
                    serde_json::Number::from_f64(float)
                        .map(Value::Number)
                        .ok_or_else(|| {
                            QuerySurfaceError::invalid(
                                "invalid_cypher_value",
                                format!("non-finite number literal: {text}"),
                            )
                        })
                } else {
                    Err(QuerySurfaceError::invalid(
                        "invalid_cypher_value",
                        format!("unparseable number: {text}"),
                    ))
                }
            }
            Rule::boolean => Ok(Value::Bool(inner.as_str().eq_ignore_ascii_case("true"))),
            Rule::null => Ok(Value::Null),
            other => Err(QuerySurfaceError::invalid(
                "invalid_cypher_value",
                format!("unsupported value rule: {other:?}"),
            )),
        };
    }
    Err(QuerySurfaceError::invalid(
        "invalid_cypher_value",
        "empty value",
    ))
}

fn parse_where_expr(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    for inner in pair.into_inner() {
        if matches!(inner.as_rule(), Rule::where_expr) {
            return parse_or_expr(inner, params);
        }
        if matches!(inner.as_rule(), Rule::or_expr) {
            return parse_or_expr_inner(inner, params);
        }
    }
    Err(QuerySurfaceError::invalid(
        "invalid_where_filter",
        "WHERE expression missing",
    ))
}

fn parse_or_expr(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    let inner = pair.into_inner().next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_where_filter", "empty WHERE expression")
    })?;
    parse_or_expr_inner(inner, params)
}

fn parse_or_expr_inner(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    let mut terms: Vec<WhereExpr> = Vec::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::and_expr => terms.push(parse_and_expr(child, params)?),
            Rule::or_kw => {}
            other => {
                return Err(QuerySurfaceError::invalid(
                    "invalid_where_filter",
                    format!("unexpected OR child: {other:?}"),
                ))
            }
        }
    }
    fold_bin(terms, |a, b| WhereExpr::Or(Box::new(a), Box::new(b)))
}

fn parse_and_expr(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    let mut terms: Vec<WhereExpr> = Vec::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::not_expr => terms.push(parse_not_expr(child, params)?),
            Rule::and_kw => {}
            other => {
                return Err(QuerySurfaceError::invalid(
                    "invalid_where_filter",
                    format!("unexpected AND child: {other:?}"),
                ))
            }
        }
    }
    fold_bin(terms, |a, b| WhereExpr::And(Box::new(a), Box::new(b)))
}

fn parse_not_expr(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    let mut negated = false;
    let mut inner_expr: Option<WhereExpr> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::not_kw => negated = true,
            Rule::not_expr => inner_expr = Some(parse_not_expr(child, params)?),
            Rule::grouped => {
                let grouped = child.into_inner().next().ok_or_else(|| {
                    QuerySurfaceError::invalid("invalid_where_filter", "empty grouped WHERE")
                })?;
                inner_expr = Some(parse_or_expr(grouped, params)?);
            }
            Rule::predicate => inner_expr = Some(parse_predicate(child, params)?),
            Rule::where_expr => inner_expr = Some(parse_or_expr(child, params)?),
            other => {
                return Err(QuerySurfaceError::invalid(
                    "invalid_where_filter",
                    format!("unexpected NOT child: {other:?}"),
                ))
            }
        }
    }
    let expr = inner_expr
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_where_filter", "missing NOT operand"))?;
    if negated {
        Ok(WhereExpr::Not(Box::new(expr)))
    } else {
        Ok(expr)
    }
}

fn parse_predicate(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<WhereExpr, QuerySurfaceError> {
    let mut path: Option<(String, String)> = None;
    let mut op: Option<CmpOp> = None;
    let mut is_in = false;
    let mut value: Option<Value> = None;
    let mut values: Option<Vec<Value>> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::property_path => path = Some(parse_property_path(child)?),
            Rule::pred_op => {
                let text = child.as_str();
                if text.eq_ignore_ascii_case("IN") {
                    is_in = true;
                } else {
                    op = Some(parse_cmp_op(text)?);
                }
            }
            Rule::in_kw => is_in = true,
            Rule::pred_rhs => {
                for rhs in child.into_inner() {
                    match rhs.as_rule() {
                        Rule::value_list => values = Some(parse_value_list(rhs, params)?),
                        Rule::value => value = Some(parse_value(rhs, params)?),
                        _ => {}
                    }
                }
            }
            Rule::value => value = Some(parse_value(child, params)?),
            Rule::value_list => values = Some(parse_value_list(child, params)?),
            _ => {}
        }
    }
    let (binding, key) = path
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_where_filter", "WHERE path missing"))?;
    if is_in {
        let values = values
            .ok_or_else(|| QuerySurfaceError::invalid("invalid_where_filter", "IN list missing"))?;
        return Ok(WhereExpr::In {
            binding,
            key,
            values,
        });
    }
    let value = value
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_where_filter", "WHERE value missing"))?;
    Ok(WhereExpr::Cmp {
        binding,
        key,
        op: op.unwrap_or(CmpOp::Eq),
        value,
    })
}

fn parse_cmp_op(text: &str) -> Result<CmpOp, QuerySurfaceError> {
    match text {
        "=" => Ok(CmpOp::Eq),
        "<>" => Ok(CmpOp::Ne),
        "<" => Ok(CmpOp::Lt),
        "<=" => Ok(CmpOp::Le),
        ">" => Ok(CmpOp::Gt),
        ">=" => Ok(CmpOp::Ge),
        other => Err(QuerySurfaceError::invalid(
            "invalid_where_filter",
            format!("unknown comparison operator {other}"),
        )),
    }
}

fn parse_property_path(pair: Pair<Rule>) -> Result<(String, String), QuerySurfaceError> {
    let mut idents = pair.into_inner();
    let binding = idents
        .next()
        .ok_or_else(|| {
            QuerySurfaceError::invalid("invalid_where_filter", "missing property binding")
        })?
        .as_str()
        .to_string();
    let key = idents
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_where_filter", "missing property key"))?
        .as_str()
        .to_string();
    Ok((binding, key))
}

fn parse_value_list(
    pair: Pair<Rule>,
    params: &BTreeMap<String, Value>,
) -> Result<Vec<Value>, QuerySurfaceError> {
    let mut out = Vec::new();
    for child in pair.into_inner() {
        if matches!(child.as_rule(), Rule::value) {
            out.push(parse_value(child, params)?);
        }
    }
    Ok(out)
}

fn fold_bin(
    terms: Vec<WhereExpr>,
    wrap: fn(WhereExpr, WhereExpr) -> WhereExpr,
) -> Result<WhereExpr, QuerySurfaceError> {
    let mut iter = terms.into_iter();
    let first = iter.next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_where_filter", "empty boolean expression")
    })?;
    Ok(iter.fold(first, wrap))
}

fn parse_return_clause(pair: Pair<Rule>) -> Result<(bool, Vec<ReturnItem>), QuerySurfaceError> {
    let mut distinct = false;
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::distinct_kw => distinct = true,
            Rule::return_items => items = parse_return_item_list(inner)?,
            _ => {}
        }
    }
    Ok((distinct, items))
}

fn parse_return_item_list(pair: Pair<Rule>) -> Result<Vec<ReturnItem>, QuerySurfaceError> {
    let mut items = Vec::new();
    for item_pair in pair.into_inner() {
        if !matches!(item_pair.as_rule(), Rule::return_item) {
            continue;
        }
        items.push(parse_one_return_item(item_pair)?);
    }
    Ok(items)
}

fn parse_one_return_item(item_pair: Pair<Rule>) -> Result<ReturnItem, QuerySurfaceError> {
    let raw = item_pair.as_str().trim().to_string();
    let mut sub_iter = item_pair.into_inner();
    let value_pair = sub_iter
        .next()
        .ok_or_else(|| QuerySurfaceError::invalid("invalid_return_clause", "empty return item"))?;
    let alias = sub_iter.find(|child| matches!(child.as_rule(), Rule::ident));
    let inner_pair = if matches!(value_pair.as_rule(), Rule::return_value) {
        value_pair.into_inner().next().ok_or_else(|| {
            QuerySurfaceError::invalid("invalid_return_clause", "empty return value")
        })?
    } else {
        value_pair
    };
    let mut item = match inner_pair.as_rule() {
        Rule::aggregate_call => parse_aggregate_return(inner_pair, &raw)?,
        Rule::path_fn => parse_path_fn_return(inner_pair)?,
        Rule::property_path => {
            let (binding, key) = parse_property_path(inner_pair)?;
            ReturnItem::Property {
                binding,
                key: key.clone(),
                expression: raw.clone(),
            }
        }
        Rule::ident => ReturnItem::Variable(inner_pair.as_str().to_string()),
        other => {
            return Err(QuerySurfaceError::invalid(
                "invalid_return_clause",
                format!("unsupported return item: {other:?}"),
            ))
        }
    };
    if let Some(alias) = alias {
        item = ReturnItem::Aliased {
            item: Box::new(item),
            alias: alias.as_str().to_string(),
        };
    }
    Ok(item)
}

fn parse_path_fn_return(pair: Pair<Rule>) -> Result<ReturnItem, QuerySurfaceError> {
    let mut name = String::new();
    let mut binding = String::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::path_fn_name => name = child.as_str().to_ascii_lowercase(),
            Rule::ident => binding = child.as_str().to_string(),
            _ => {}
        }
    }
    let expression = format!("{name}({binding})");
    match name.as_str() {
        "nodes" => Ok(ReturnItem::PathNodes {
            binding,
            expression,
        }),
        "relationships" => Ok(ReturnItem::PathRels {
            binding,
            expression,
        }),
        "length" => Ok(ReturnItem::PathLength {
            binding,
            expression,
        }),
        other => Err(QuerySurfaceError::invalid(
            "invalid_return_clause",
            format!("unsupported path function {other}"),
        )),
    }
}

fn parse_aggregate_return(
    inner_pair: Pair<Rule>,
    raw: &str,
) -> Result<ReturnItem, QuerySurfaceError> {
    let mut agg_iter = inner_pair.into_inner();
    let op_pair = agg_iter.next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_return_clause", "aggregate missing operator")
    })?;
    let op = parse_agg_op(op_pair.as_str())?;
    let arg_pair = agg_iter.next().ok_or_else(|| {
        QuerySurfaceError::invalid("invalid_return_clause", "aggregate missing argument")
    })?;
    let arg_inner = arg_pair.clone().into_inner().next();
    let (binding, key) = match arg_inner {
        None => (None, None),
        Some(p) => match p.as_rule() {
            Rule::property_path => {
                let (b, k) = parse_property_path(p)?;
                (Some(b), Some(k))
            }
            Rule::ident => (Some(p.as_str().to_string()), None),
            other => {
                return Err(QuerySurfaceError::invalid(
                    "invalid_return_clause",
                    format!("unsupported agg argument rule: {other:?}"),
                ))
            }
        },
    };
    if matches!(op, AggOp::Count) {
        Ok(ReturnItem::Count {
            binding,
            expression: raw.to_string(),
        })
    } else {
        Ok(ReturnItem::Aggregate {
            op,
            binding,
            key,
            expression: raw.to_string(),
        })
    }
}

fn rewrite_path_return(item: &mut ReturnItem, path_name: &str) {
    match item {
        ReturnItem::Variable(binding) if binding == path_name => {
            *item = ReturnItem::Path {
                binding: binding.clone(),
                expression: binding.clone(),
            };
        }
        ReturnItem::Aliased { item: inner, .. } => rewrite_path_return(inner, path_name),
        _ => {}
    }
}

fn parse_return_items(pair: Pair<Rule>) -> Result<Vec<ReturnItem>, QuerySurfaceError> {
    parse_return_clause(pair).map(|(_, items)| items)
}

fn parse_limit_literal(pair: Pair<Rule>) -> Result<usize, QuerySurfaceError> {
    for inner in pair.into_inner() {
        if matches!(inner.as_rule(), Rule::integer) {
            return inner.as_str().parse::<usize>().map_err(|err| {
                QuerySurfaceError::invalid(
                    "invalid_limit_literal",
                    format!("limit must be a non-negative integer: {err}"),
                )
            });
        }
    }
    Err(QuerySurfaceError::invalid(
        "invalid_limit_literal",
        "LIMIT requires an integer literal",
    ))
}

fn parse_agg_op(text: &str) -> Result<AggOp, QuerySurfaceError> {
    match text.to_ascii_uppercase().as_str() {
        "COUNT" => Ok(AggOp::Count),
        "SUM" => Ok(AggOp::Sum),
        "AVG" => Ok(AggOp::Avg),
        "MIN" => Ok(AggOp::Min),
        "MAX" => Ok(AggOp::Max),
        other => Err(QuerySurfaceError::invalid(
            "invalid_aggregate_op",
            format!("unsupported aggregate operator: {other}"),
        )),
    }
}

fn parse_with_clause(
    pair: Pair<Rule>,
    _params: &BTreeMap<String, Value>,
) -> Result<WithClause, QuerySurfaceError> {
    let mut items: Vec<WithItem> = Vec::new();
    for inner in pair.into_inner() {
        if !matches!(inner.as_rule(), Rule::with_items) {
            continue;
        }
        for entry in inner.into_inner() {
            if !matches!(entry.as_rule(), Rule::with_item) {
                continue;
            }
            let mut sub = entry.into_inner();
            let expr_pair = sub.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_with_clause", "WITH item missing expression")
            })?;
            let alias_pair = sub.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_with_clause", "WITH item missing alias")
            })?;
            let alias = alias_pair.as_str().to_string();
            match expr_pair.as_rule() {
                Rule::aggregate_call => {
                    let mut agg_iter = expr_pair.into_inner();
                    let op_pair = agg_iter.next().ok_or_else(|| {
                        QuerySurfaceError::invalid(
                            "invalid_with_clause",
                            "aggregate missing operator",
                        )
                    })?;
                    let op = parse_agg_op(op_pair.as_str())?;
                    let arg_pair = agg_iter.next().ok_or_else(|| {
                        QuerySurfaceError::invalid(
                            "invalid_with_clause",
                            "aggregate missing argument",
                        )
                    })?;
                    let arg_inner = arg_pair.into_inner().next();
                    let (binding, key) = match arg_inner {
                        None => (None, None),
                        Some(p) => match p.as_rule() {
                            Rule::property_path => {
                                let mut idents = p.into_inner();
                                let b = idents
                                    .next()
                                    .ok_or_else(|| {
                                        QuerySurfaceError::invalid(
                                            "invalid_with_clause",
                                            "agg arg missing binding",
                                        )
                                    })?
                                    .as_str()
                                    .to_string();
                                let k = idents
                                    .next()
                                    .ok_or_else(|| {
                                        QuerySurfaceError::invalid(
                                            "invalid_with_clause",
                                            "agg arg missing key",
                                        )
                                    })?
                                    .as_str()
                                    .to_string();
                                (Some(b), Some(k))
                            }
                            Rule::ident => (Some(p.as_str().to_string()), None),
                            other => {
                                return Err(QuerySurfaceError::invalid(
                                    "invalid_with_clause",
                                    format!("unsupported agg arg rule: {other:?}"),
                                ));
                            }
                        },
                    };
                    items.push(WithItem::Aggregate {
                        op,
                        binding,
                        key,
                        alias,
                    });
                }
                Rule::property_path => {
                    let mut idents = expr_pair.into_inner();
                    let binding = idents
                        .next()
                        .ok_or_else(|| {
                            QuerySurfaceError::invalid(
                                "invalid_with_clause",
                                "property path missing binding",
                            )
                        })?
                        .as_str()
                        .to_string();
                    let key = idents
                        .next()
                        .ok_or_else(|| {
                            QuerySurfaceError::invalid(
                                "invalid_with_clause",
                                "property path missing key",
                            )
                        })?
                        .as_str()
                        .to_string();
                    items.push(WithItem::Field {
                        binding,
                        key: Some(key),
                        alias,
                    });
                }
                Rule::ident => {
                    items.push(WithItem::Field {
                        binding: expr_pair.as_str().to_string(),
                        key: None,
                        alias,
                    });
                }
                other => {
                    return Err(QuerySurfaceError::invalid(
                        "invalid_with_clause",
                        format!("unsupported WITH expression: {other:?}"),
                    ));
                }
            }
        }
    }
    Ok(WithClause { items })
}

fn parse_order_clause(pair: Pair<Rule>) -> Result<Vec<OrderBy>, QuerySurfaceError> {
    let mut out: Vec<OrderBy> = Vec::new();
    for inner in pair.into_inner() {
        if !matches!(inner.as_rule(), Rule::order_items) {
            continue;
        }
        for entry in inner.into_inner() {
            if !matches!(entry.as_rule(), Rule::order_item) {
                continue;
            }
            let mut iter = entry.into_inner();
            let expr_pair = iter.next().ok_or_else(|| {
                QuerySurfaceError::invalid("invalid_order_clause", "missing ORDER BY expression")
            })?;
            let expression = expr_pair.as_str().to_string();
            let descending = match iter.next() {
                Some(dir) => dir.as_str().eq_ignore_ascii_case("DESC"),
                None => false,
            };
            out.push(OrderBy {
                expression,
                descending,
            });
        }
    }
    Ok(out)
}

fn parse_skip_clause(pair: Pair<Rule>) -> Result<usize, QuerySurfaceError> {
    for inner in pair.into_inner() {
        if matches!(inner.as_rule(), Rule::integer) {
            return inner.as_str().parse::<usize>().map_err(|err| {
                QuerySurfaceError::invalid(
                    "invalid_skip_literal",
                    format!("SKIP requires a non-negative integer: {err}"),
                )
            });
        }
    }
    Err(QuerySurfaceError::invalid(
        "invalid_skip_literal",
        "SKIP missing integer",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_parses_simple_node_match() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc) RETURN n LIMIT 10");
        assert!(
            pairs.is_ok(),
            "expected MATCH (n:Doc) RETURN n LIMIT 10 to parse: {:?}",
            pairs
        );
    }

    #[test]
    fn grammar_parses_where_filter() {
        let pairs =
            CypherPestParser::parse(Rule::query, "MATCH (n:Doc) WHERE n.path = $value RETURN n");
        assert!(pairs.is_ok(), "expected WHERE filter to parse: {:?}", pairs);
    }

    #[test]
    fn grammar_parses_count_star() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc) RETURN count(n)");
        assert!(pairs.is_ok(), "expected COUNT to parse: {:?}", pairs);
    }

    #[test]
    fn grammar_parses_single_hop_edge() {
        let pairs = CypherPestParser::parse(
            Rule::query,
            "MATCH (a:Doc)-[:CITES]->(b:Doc) RETURN a, b LIMIT 5",
        );
        assert!(
            pairs.is_ok(),
            "expected single-hop edge to parse: {:?}",
            pairs
        );
    }
}

#[cfg(test)]
mod parse_to_ast_tests {
    use super::*;
    use crate::cypher::ast::{CypherPattern, ReturnItem, WhereExpr};
    use std::collections::BTreeMap;

    #[test]
    fn parse_simple_node_match() {
        let parsed =
            parse_cypher_pest("MATCH (n:Doc) RETURN n LIMIT 10", &BTreeMap::new()).unwrap();
        assert_eq!(parsed.limit, 10);
        let CypherPattern::Node(node) = &parsed.pattern else {
            panic!("expected node pattern");
        };
        assert_eq!(node.binding, "n");
        assert_eq!(node.label.as_deref(), Some("Doc"));
        assert_eq!(parsed.returns.len(), 1);
        assert!(matches!(parsed.returns[0], ReturnItem::Variable(ref b) if b == "n"));
    }

    #[test]
    fn parse_default_limit_when_omitted() {
        let parsed = parse_cypher_pest("MATCH (n:Doc) RETURN n", &BTreeMap::new()).unwrap();
        assert_eq!(parsed.limit, 100);
    }

    #[test]
    fn parse_where_property_eq_param() {
        let mut params = BTreeMap::new();
        params.insert("value".to_string(), serde_json::json!("src/lib.rs"));
        let parsed =
            parse_cypher_pest("MATCH (n:File) WHERE n.path = $value RETURN n", &params).unwrap();
        let filter = parsed.eq_filter().expect("expected WHERE filter");
        assert_eq!(filter.binding, "n");
        assert_eq!(filter.key, "path");
        assert_eq!(filter.value, serde_json::json!("src/lib.rs"));
        assert_eq!(
            parsed.where_filter,
            parsed.where_expr.as_ref().and_then(WhereExpr::as_eq_filter)
        );
    }

    #[test]
    fn where_filter_is_eq_projection_never_a_second_source() {
        let simple =
            parse_cypher_pest("MATCH (n:Doc) WHERE n.k = 1 RETURN n", &BTreeMap::new()).unwrap();
        assert_eq!(
            simple.where_filter,
            simple.where_expr.as_ref().and_then(WhereExpr::as_eq_filter)
        );
        assert!(simple.where_filter.is_some());

        let compound = parse_cypher_pest(
            "MATCH (n:Doc) WHERE n.k = 1 AND n.n = 2 RETURN n",
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(compound.where_expr.is_some());
        assert!(compound.where_filter.is_none());
        assert_eq!(
            compound.where_filter,
            compound
                .where_expr
                .as_ref()
                .and_then(WhereExpr::as_eq_filter)
        );

        let write =
            parse_cypher_pest("MATCH (n:Doc) WHERE n.k = 1 SET n.k = 2", &BTreeMap::new()).unwrap();
        assert!(write.where_expr.is_some());
        assert_eq!(
            write.where_filter,
            write.where_expr.as_ref().and_then(WhereExpr::as_eq_filter)
        );
    }

    #[test]
    fn parse_count_star_into_count_item() {
        let parsed = parse_cypher_pest("MATCH (n:Doc) RETURN count(n)", &BTreeMap::new()).unwrap();
        assert_eq!(parsed.returns.len(), 1);
        let ReturnItem::Count {
            binding,
            expression,
        } = &parsed.returns[0]
        else {
            panic!("expected count return item, got {:?}", parsed.returns[0]);
        };
        assert_eq!(binding.as_deref(), Some("n"));
        assert!(expression.contains("count"));
    }

    #[test]
    fn parse_single_hop_edge_pattern() {
        let parsed = parse_cypher_pest(
            "MATCH (a:Doc)-[:CITES]->(b:Doc) RETURN a, b LIMIT 5",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::Edge(edge) = &parsed.pattern else {
            panic!("expected edge pattern");
        };
        assert_eq!(edge.left.binding, "a");
        assert_eq!(edge.edge_type, "CITES");
        assert_eq!(edge.right.binding, "b");
    }

    #[test]
    fn parse_property_returns_trim_whitespace_between_items() {
        let parsed = parse_cypher_pest(
            "MATCH (a:File {path: $path})-[:IMPORTS]->(b:File) RETURN a.path, b.path LIMIT 10",
            &BTreeMap::from([(String::from("path"), serde_json::json!("src/lib.rs"))]),
        )
        .unwrap();
        let keys: Vec<&str> = parsed.returns.iter().map(ReturnItem::key).collect();
        assert_eq!(keys, vec!["a.path", "b.path"]);
    }

    #[test]
    fn parse_missing_param_errors() {
        let err = parse_cypher_pest(
            "MATCH (n:Doc) WHERE n.path = $value RETURN n",
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(format!("{:?}", err).contains("missing_cypher_param"));
    }

    #[test]
    fn parse_empty_query_errors() {
        let err = parse_cypher_pest("   ", &BTreeMap::new()).unwrap_err();
        assert!(format!("{:?}", err).contains("empty_cypher_query"));
    }

    #[test]
    fn parse_multi_hop_into_edge_chain() {
        let parsed = parse_cypher_pest(
            "MATCH (a:Doc)-[:T1]->(b:Doc)-[:T2]->(c:Doc) RETURN c",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::EdgeChain(chain) = &parsed.pattern else {
            panic!("expected EdgeChain pattern, got {:?}", parsed.pattern);
        };
        assert_eq!(chain.start.binding, "a");
        assert_eq!(chain.steps.len(), 2);
        assert_eq!(chain.steps[0].edge_type, "T1");
        assert_eq!(chain.steps[1].target.binding, "c");
    }

    #[test]
    fn parse_bounded_var_length_into_edge_var_length() {
        let parsed = parse_cypher_pest(
            "MATCH (a:Doc)-[:T*1..3]->(b:Doc) RETURN b LIMIT 5",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::EdgeVarLength(var) = &parsed.pattern else {
            panic!("expected EdgeVarLength pattern");
        };
        assert_eq!(var.min, 1);
        assert_eq!(var.max, Some(3));
        assert_eq!(var.edge_type, "T");
        assert_eq!(parsed.limit, 5);
    }

    #[test]
    fn parse_unbounded_var_length_returns_max_none() {
        let parsed =
            parse_cypher_pest("MATCH (a:Doc)-[:T*]->(b:Doc) RETURN b", &BTreeMap::new()).unwrap();
        let CypherPattern::EdgeVarLength(var) = &parsed.pattern else {
            panic!("expected EdgeVarLength pattern");
        };
        assert_eq!(var.min, 1);
        assert_eq!(var.max, None);
    }

    #[test]
    fn parse_path_binding_stores_alias() {
        let parsed = parse_cypher_pest(
            "MATCH p = (a:Doc)-[:T*]->(b:Doc) RETURN p",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::EdgeVarLength(var) = &parsed.pattern else {
            panic!("expected EdgeVarLength pattern");
        };
        assert_eq!(var.path_binding.as_deref(), Some("p"));
        assert_eq!(var.direction, EdgeDirection::Out);
        assert!(matches!(
            parsed.returns[0],
            ReturnItem::Path { ref binding, .. } if binding == "p"
        ));
    }

    #[test]
    fn parse_single_hop_path_binding_attaches_to_edge() {
        let parsed = parse_cypher_pest(
            "MATCH p = (a:Doc)-[:T]->(b:Doc) RETURN length(p) AS hops, nodes(p)",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::Edge(edge) = &parsed.pattern else {
            panic!("expected Edge pattern");
        };
        assert_eq!(edge.path_binding.as_deref(), Some("p"));
        assert_eq!(parsed.path_binding.as_deref(), Some("p"));
    }

    #[test]
    fn parse_inbound_var_length_preserves_direction() {
        let parsed = parse_cypher_pest(
            "MATCH (a:Doc)<-[:T*1..2]-(b:Doc) RETURN a, b",
            &BTreeMap::new(),
        )
        .unwrap();
        let CypherPattern::EdgeVarLength(var) = &parsed.pattern else {
            panic!("expected EdgeVarLength pattern");
        };
        assert_eq!(var.direction, EdgeDirection::In);
    }

    #[test]
    fn parse_create_node_emits_write_clause() {
        let parsed = parse_cypher_pest(
            "CREATE (n:Doc {id: 'a', path: 'src/lib.rs'})",
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(parsed.writes.len(), 1);
        let WriteClause::CreateNode { node } = &parsed.writes[0] else {
            panic!("expected CreateNode write clause");
        };
        assert_eq!(node.label.as_deref(), Some("Doc"));
        assert_eq!(node.properties.get("id").unwrap(), &serde_json::json!("a"));
    }

    #[test]
    fn parse_merge_with_on_create_and_on_match_emits_merge_clause() {
        let parsed = parse_cypher_pest(
            "MERGE (n:Doc {id: 'a'}) ON CREATE SET n.seen = 1 ON MATCH SET n.seen = n.seen + 1",
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(parsed.writes.len(), 1);
        let WriteClause::Merge {
            node,
            on_create,
            on_match,
        } = &parsed.writes[0]
        else {
            panic!("expected Merge write clause");
        };
        assert_eq!(node.label.as_deref(), Some("Doc"));
        let on_create = on_create.as_ref().expect("on_create branch present");
        let on_match = on_match.as_ref().expect("on_match branch present");
        assert_eq!(on_create.sets.len(), 1);
        let (_, _, on_match_expr) = &on_match.sets[0];
        assert!(matches!(on_match_expr, SetExpr::Increment { .. }));
    }

    #[test]
    fn parse_match_set_emits_set_clause() {
        let parsed = parse_cypher_pest(
            "MATCH (n:Doc {id: 'a'}) SET n.flag = true",
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(parsed.writes.len(), 1);
        let WriteClause::Set {
            binding,
            key,
            value,
        } = &parsed.writes[0]
        else {
            panic!("expected Set write clause");
        };
        assert_eq!(binding, "n");
        assert_eq!(key, "flag");
        assert!(matches!(value, SetExpr::Literal(_)));
    }

    #[test]
    fn parse_match_delete_emits_delete_clause() {
        let parsed =
            parse_cypher_pest("MATCH (n:Doc {id: 'a'}) DELETE n", &BTreeMap::new()).unwrap();
        assert_eq!(parsed.writes.len(), 1);
        let WriteClause::Delete { binding, detach } = &parsed.writes[0] else {
            panic!("expected Delete write clause");
        };
        assert_eq!(binding, "n");
        assert!(!*detach);
    }

    #[test]
    fn parse_match_detach_delete_emits_detach_flag() {
        let parsed =
            parse_cypher_pest("MATCH (n:Doc {id: 'a'}) DETACH DELETE n", &BTreeMap::new()).unwrap();
        let WriteClause::Delete { detach, .. } = &parsed.writes[0] else {
            panic!("expected Delete write clause");
        };
        assert!(*detach);
    }

    #[test]
    fn parse_sum_aggregate_into_return_item() {
        let parsed =
            parse_cypher_pest("MATCH (n:Doc) RETURN sum(n.score)", &BTreeMap::new()).unwrap();
        let ReturnItem::Aggregate {
            op, binding, key, ..
        } = &parsed.returns[0]
        else {
            panic!("expected Aggregate ReturnItem, got {:?}", parsed.returns[0]);
        };
        assert_eq!(*op, AggOp::Sum);
        assert_eq!(binding.as_deref(), Some("n"));
        assert_eq!(key.as_deref(), Some("score"));
    }

    #[test]
    fn parse_with_clause_carries_field_and_aggregate_items() {
        let parsed = parse_cypher_pest(
            "MATCH (n:Doc) WITH n.category AS cat, count(n) AS c RETURN cat, c",
            &BTreeMap::new(),
        )
        .unwrap();
        let with = parsed.with_clause.as_ref().expect("expected WITH clause");
        assert_eq!(with.items.len(), 2);
        let WithItem::Field { alias, key, .. } = &with.items[0] else {
            panic!("expected Field WithItem");
        };
        assert_eq!(alias, "cat");
        assert_eq!(key.as_deref(), Some("category"));
        let WithItem::Aggregate { op, alias, .. } = &with.items[1] else {
            panic!("expected Aggregate WithItem");
        };
        assert_eq!(*op, AggOp::Count);
        assert_eq!(alias, "c");
    }

    #[test]
    fn parse_order_by_desc_and_skip() {
        let parsed = parse_cypher_pest(
            "MATCH (n:Doc) RETURN n ORDER BY n.created DESC SKIP 2 LIMIT 5",
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(parsed.order_by.len(), 1);
        assert!(parsed.order_by[0].descending);
        assert_eq!(parsed.skip, Some(2));
        assert_eq!(parsed.limit, 5);
    }
}

#[cfg(test)]
mod write_grammar_tests {
    use super::*;

    #[test]
    fn grammar_parses_create_node() {
        let pairs =
            CypherPestParser::parse(Rule::query, "CREATE (n:Doc {id: 'a', path: 'src/lib.rs'})");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_merge_with_on_create_and_on_match() {
        let pairs = CypherPestParser::parse(
            Rule::query,
            "MERGE (n:Doc {id: 'a'}) ON CREATE SET n.seen = 1 ON MATCH SET n.seen = n.seen + 1",
        );
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_match_set() {
        let pairs =
            CypherPestParser::parse(Rule::query, "MATCH (n:Doc {id: 'a'}) SET n.flag = true");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_match_delete() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc {id: 'a'}) DELETE n");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_match_detach_delete() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc {id: 'a'}) DETACH DELETE n");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }
}

#[cfg(test)]
mod aggregation_grammar_tests {
    use super::*;

    #[test]
    fn grammar_parses_sum_aggregation() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc) RETURN sum(n.score)");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_with_clause() {
        let pairs = CypherPestParser::parse(
            Rule::query,
            "MATCH (n:Doc) WITH n.category AS cat, count(n) AS c RETURN cat, c",
        );
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_order_by_desc() {
        let pairs = CypherPestParser::parse(
            Rule::query,
            "MATCH (n:Doc) RETURN n ORDER BY n.created DESC LIMIT 5",
        );
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_skip() {
        let pairs = CypherPestParser::parse(Rule::query, "MATCH (n:Doc) RETURN n SKIP 10 LIMIT 5");
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_full_with_pipeline() {
        let pairs = CypherPestParser::parse(
            Rule::query,
            "MATCH (n:Doc) WITH n.category AS cat, count(n) AS c RETURN cat, c ORDER BY c DESC LIMIT 10",
        );
        assert!(pairs.is_ok(), "{:?}", pairs);
    }

    #[test]
    fn grammar_parses_boolean_where_and_comparisons() {
        let parsed = parse_cypher_pest(
            "MATCH (n:Doc) WHERE n.k = 1 AND n.n > 2 OR n.k IN [1, 2] RETURN n",
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(parsed.where_expr.is_some());
    }

    #[test]
    fn grammar_parses_inbound_and_undirected() {
        assert!(parse_cypher_pest(
            "MATCH (a:Doc)<-[:CITES]-(b:Doc) RETURN a, b",
            &BTreeMap::new(),
        )
        .is_ok());
        assert!(parse_cypher_pest(
            "MATCH (a:Doc)-[:CITES]-(b:Doc) RETURN a, b",
            &BTreeMap::new(),
        )
        .is_ok());
    }

    #[test]
    fn grammar_parses_var_length_in_chain() {
        let parsed = parse_cypher_pest(
            "MATCH (a:Doc)-[:CITES*1..3]->(b:Doc)-[:MENTIONS]->(c:Term) RETURN a, b, c",
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(matches!(parsed.pattern, CypherPattern::EdgeChain(_)));
    }

    #[test]
    fn grammar_parses_path_projection_distinct_and_alias() {
        let parsed = parse_cypher_pest(
            "MATCH p = (a:Doc)-[:CITES]->(b:Doc) RETURN DISTINCT nodes(p) AS ns, length(p) AS hops",
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(parsed.distinct);
        assert_eq!(parsed.returns.len(), 2);
    }
}
