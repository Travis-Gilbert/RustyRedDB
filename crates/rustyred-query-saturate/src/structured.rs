//! Lossless lowering of supported read-only syntax into the native query algebra.
//! Unsupported operators refuse; none are stripped before exact execution.
use rustyred_thg_core::{
    JoinPredicate, Predicate, Projection, QueryIr, QueryRelation, ScalarBound, ScalarValue,
};
use serde_json::Value;
use sqlparser::{ast::*, dialect::GenericDialect, parser::Parser};
use std::collections::BTreeMap;

pub fn lower_sql(sql: &str) -> Result<QueryIr, String> {
    let statements = Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| e.to_string())?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Err("structured SQL requires exactly one read-only SELECT".into());
    };
    if query.with.is_some()
        || query.order_by.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err("SELECT query modifier has no exact QueryIR lowering".into());
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err("set operations require an exact QueryIR lowering".into());
    };
    if select.distinct.is_some()
        || select.select_modifiers.is_some()
        || select.top.is_some()
        || select.exclude.is_some()
        || select.into.is_some()
        || !select.lateral_views.is_empty()
        || select.prewhere.is_some()
        || !select.connect_by.is_empty()
        || !matches!(&select.group_by, GroupByExpr::Expressions(v,m) if v.is_empty() && m.is_empty())
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
        || select.having.is_some()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.value_table_mode.is_some()
        || !select.optimizer_hints.is_empty()
    {
        return Err("SELECT clause has no exact QueryIR lowering".into());
    }
    let [from] = select.from.as_slice() else {
        return Err("SELECT requires one FROM root with explicit inner joins".into());
    };
    let mut result = QueryIr {
        relations: vec![relation(&from.relation)?],
        ..QueryIr::default()
    };
    for join in &from.joins {
        if join.global {
            return Err("GLOBAL JOIN unsupported".into());
        }
        let (JoinOperator::Join(JoinConstraint::On(on))
        | JoinOperator::Inner(JoinConstraint::On(on))) = &join.join_operator
        else {
            return Err("only explicit inner equijoins have exact lowering".into());
        };
        let next = relation(&join.relation)?;
        if result.relations.iter().any(|r| r.alias == next.alias) {
            return Err("duplicate relation alias".into());
        }
        result.relations.push(next);
        lower_join(on, &mut result)?;
    }
    if let Some(selection) = &select.selection {
        lower_filter(selection, &mut result)?;
    }
    if select.projection.len() == 1
        && matches!(&select.projection[0], SelectItem::Wildcard(options) if options == &WildcardAdditionalOptions::default())
    {
        // An empty native projection retains every scalar column.
    } else {
        for item in &select.projection {
            let SelectItem::UnnamedExpr(expr) = item else {
                return Err("projection expression/alias unsupported by QueryIR".into());
            };
            let (alias, column) = column(expr, &result)?;
            result.projection.push(Projection { alias, column });
        }
    }
    result.limit = match &query.limit_clause {
        None => None,
        Some(LimitClause::LimitOffset {
            limit: Some(expr),
            offset: None,
            limit_by,
        }) if limit_by.is_empty() => {
            let Expr::Value(value) = expr else {
                return Err("LIMIT requires a nonnegative integer".into());
            };
            let sqlparser::ast::Value::Number(number, _) = &value.value else {
                return Err("LIMIT requires a nonnegative integer".into());
            };
            Some(
                number
                    .parse()
                    .map_err(|_| "LIMIT requires a nonnegative integer")?,
            )
        }
        _ => return Err("LIMIT/OFFSET clause has no exact QueryIR lowering".into()),
    };
    Ok(result)
}

fn relation(factor: &TableFactor) -> Result<QueryRelation, String> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
    } = factor
    else {
        return Err("table function/subquery has no exact QueryIR lowering".into());
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || *with_ordinality
        || !partitions.is_empty()
        || json_path.is_some()
        || sample.is_some()
        || !index_hints.is_empty()
        || alias.as_ref().is_some_and(|a| !a.columns.is_empty())
    {
        return Err("table modifier has no exact QueryIR lowering".into());
    }
    let [ObjectNamePart::Identifier(name)] = name.0.as_slice() else {
        return Err("relation must be an unqualified native relation name".into());
    };
    Ok(QueryRelation {
        alias: alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or_else(|| name.value.clone()),
        relation: name.value.clone(),
        predicates: vec![],
    })
}
fn column(expr: &Expr, query: &QueryIr) -> Result<(String, String), String> {
    let (alias, col) = match expr {
        Expr::Identifier(col) if query.relations.len() == 1 => {
            (query.relations[0].alias.clone(), col.value.clone())
        }
        Expr::CompoundIdentifier(parts) if parts.len() == 2 => {
            (parts[0].value.clone(), parts[1].value.clone())
        }
        _ => return Err("column must be qualified when multiple relations are present".into()),
    };
    if !query.relations.iter().any(|r| r.alias == alias) {
        return Err("unknown relation alias".into());
    }
    Ok((alias, col))
}
fn lower_join(expr: &Expr, query: &mut QueryIr) -> Result<(), String> {
    match expr {
        Expr::Nested(inner) => lower_join(inner, query),
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            lower_join(left, query)?;
            lower_join(right, query)
        }
        Expr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            let (left_alias, left_column) = column(left, query)?;
            let (right_alias, right_column) = column(right, query)?;
            if left_alias == right_alias {
                return Err("join must connect distinct relations".into());
            }
            query.joins.push(JoinPredicate {
                left_alias,
                left_column,
                right_alias,
                right_column,
                kind: Default::default(),
            });
            Ok(())
        }
        _ => Err("JOIN requires conjunctions of column equality".into()),
    }
}
fn scalar(expr: &Expr) -> Result<ScalarValue, String> {
    match expr {
        Expr::Value(value) => match &value.value {
            sqlparser::ast::Value::SingleQuotedString(s) => Ok(ScalarValue::String(s.clone())),
            sqlparser::ast::Value::Boolean(b) => Ok(ScalarValue::Bool(*b)),
            sqlparser::ast::Value::Number(n, _) => n
                .parse::<i64>()
                .map(ScalarValue::I64)
                .or_else(|_| n.parse::<f64>().map(ScalarValue::F64))
                .map_err(|_| "invalid number".into()),
            _ => Err("literal has no exact native scalar representation".into()),
        },
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match scalar(expr)? {
            ScalarValue::I64(n) => n
                .checked_neg()
                .map(ScalarValue::I64)
                .ok_or_else(|| "integer overflow".into()),
            ScalarValue::F64(n) => Ok(ScalarValue::F64(-n)),
            _ => Err("negation requires number".into()),
        },
        _ => Err("predicate requires a scalar literal".into()),
    }
}
fn lower_filter(expr: &Expr, query: &mut QueryIr) -> Result<(), String> {
    match expr {
        Expr::Nested(inner) => lower_filter(inner, query),
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            lower_filter(left, query)?;
            lower_filter(right, query)
        }
        Expr::BinaryOp { left, op, right } => {
            let (alias, column) = column(left, query)?;
            let value = scalar(right)?;
            if matches!(&value,ScalarValue::F64(f) if !f.is_finite()) {
                return Err("nonfinite scalar".into());
            }
            let predicate = match op {
                BinaryOperator::Eq => Predicate::Equals { column, value },
                BinaryOperator::Gt => Predicate::Range {
                    column,
                    lo: ScalarBound::Excluded(value),
                    hi: ScalarBound::Unbounded,
                },
                BinaryOperator::GtEq => Predicate::Range {
                    column,
                    lo: ScalarBound::Included(value),
                    hi: ScalarBound::Unbounded,
                },
                BinaryOperator::Lt => Predicate::Range {
                    column,
                    lo: ScalarBound::Unbounded,
                    hi: ScalarBound::Excluded(value),
                },
                BinaryOperator::LtEq => Predicate::Range {
                    column,
                    lo: ScalarBound::Unbounded,
                    hi: ScalarBound::Included(value),
                },
                _ => return Err("predicate operator has no exact native lowering".into()),
            };
            query
                .relations
                .iter_mut()
                .find(|r| r.alias == alias)
                .unwrap()
                .predicates
                .push(predicate);
            Ok(())
        }
        _ => Err("predicate requires conjunctions of scalar comparisons".into()),
    }
}

pub fn lower_cypher(text: &str, params: &BTreeMap<String, Value>) -> Result<QueryIr, String> {
    use crate::cypher::ast::{CypherPattern, ReturnItem};
    let parsed =
        crate::cypher::parse::parse_cypher_pest(text, params).map_err(|e| e.to_string())?;
    if !parsed.writes.is_empty()
        || parsed.with_clause.is_some()
        || !parsed.order_by.is_empty()
        || parsed.skip.is_some()
    {
        return Err("Cypher writes/pipelines/order/skip have no exact QueryIR lowering".into());
    }
    let CypherPattern::Node(node) = parsed.pattern else {
        return Err("Cypher graph traversal requires a live expansion resolver".into());
    };
    if node.label.as_deref().is_some_and(|label| {
        label.eq_ignore_ascii_case("nodes") || label.eq_ignore_ascii_case("edges")
    }) {
        return Err("reserved relation name cannot stand in for a Cypher label".into());
    }
    let relation = node
        .label
        .as_deref()
        .unwrap_or("nodes")
        .trim()
        .to_ascii_lowercase();
    let mut predicates = Vec::new();
    for (column, value) in node.properties {
        predicates.push(Predicate::Equals {
            column,
            value: ScalarValue::from_json(&value).ok_or("Cypher property is not scalar")?,
        });
    }
    if let Some(filter) = parsed.where_filter {
        if filter.binding != node.binding {
            return Err("unbound Cypher WHERE variable".into());
        }
        predicates.push(Predicate::Equals {
            column: filter.key,
            value: ScalarValue::from_json(&filter.value)
                .ok_or("Cypher WHERE value is not scalar")?,
        });
    }
    let mut projection = Vec::new();
    // Whole-node values are materialized by the host from the same snapshot,
    // after this scalar identity projection. Nested properties never traverse
    // the scalar algebra and are not approximated as a scalar row.
    let full_node = matches!(parsed.returns.as_slice(), [ReturnItem::Variable(binding)] if binding == &node.binding);
    if full_node {
        projection.push(Projection {
            alias: node.binding.clone(),
            column: "id".into(),
        });
    } else {
        for item in parsed.returns {
            let ReturnItem::Property { binding, key, .. } = item else {
                return Err("Cypher aggregate/path projection requires an exact lowering".into());
            };
            if binding != node.binding {
                return Err("unbound Cypher RETURN variable".into());
            }
            projection.push(Projection {
                alias: binding,
                column: key,
            });
        }
    }
    Ok(QueryIr {
        relations: vec![QueryRelation {
            alias: node.binding,
            relation,
            predicates,
        }],
        projection,
        limit: Some(parsed.limit),
        ..QueryIr::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sql_ast_preserves_predicates_and_join() {
        let q=lower_sql("SELECT a.id, b.name FROM nodes a JOIN nodes b ON a.owner = b.id WHERE a.name = 'two  spaces' AND b.age >= 3 LIMIT 4").unwrap();
        assert_eq!(q.joins.len(), 1);
        assert_eq!(q.relations[0].predicates.len(), 1);
        assert_eq!(q.relations[1].predicates.len(), 1);
        assert_eq!(q.limit, Some(4));
        assert!(
            matches!(&q.relations[0].predicates[0],Predicate::Equals{value:ScalarValue::String(v),..} if v=="two  spaces")
        );
    }
    #[test]
    fn sql_unsupported_semantics_never_disappear() {
        for sql in [
            "DELETE FROM nodes",
            "SELECT * FROM nodes; SELECT * FROM nodes",
            "SELECT count(*) FROM nodes",
            "SELECT * FROM nodes WHERE x=1 OR x=2",
            "SELECT * FROM nodes ORDER BY x",
            "SELECT DISTINCT x FROM nodes",
            "SELECT * FROM nodes LIMIT 1 OFFSET 2",
            "SELECT * FROM nodes WHERE x IS NULL",
            "SELECT x AS y FROM nodes",
        ] {
            assert!(lower_sql(sql).is_err(), "{sql}");
        }
    }
    #[test]
    fn cypher_shared_parser_preserves_literals_and_parameters() {
        let q = lower_cypher(
            "MATCH (n:Doc {name: 'two  spaces'}) WHERE n.id=$id RETURN n.name LIMIT 4",
            &BTreeMap::from([("id".into(), Value::String("a".into()))]),
        )
        .unwrap();
        assert_eq!(q.relations[0].relation, "doc");
        assert_eq!(q.relations[0].predicates.len(), 2);
        assert_eq!(q.limit, Some(4));
        assert!(
            matches!(&q.relations[0].predicates[0],Predicate::Equals{value:ScalarValue::String(v),..} if v=="two  spaces")
        );
    }
    #[test]
    fn cypher_unsupported_semantics_refuse() {
        for text in [
            "MATCH (n) RETURN count(n)",
            "MATCH (n:nodes) RETURN n.id",
            "CREATE (n:Doc {id:'a'})",
            "MATCH (n) RETURN n ORDER BY n.id",
            "MATCH (a)-[:LINK]->(b) RETURN b",
        ] {
            assert!(lower_cypher(text, &BTreeMap::new()).is_err(), "{text}");
        }
    }
}
