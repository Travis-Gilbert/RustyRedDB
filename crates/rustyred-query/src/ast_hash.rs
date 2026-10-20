use sha2::{Digest, Sha256};

use crate::cypher::ast::{
    CmpOp, CypherPattern, EdgeStep, NodePattern, ParsedCypher, ReturnItem, WhereExpr, WriteClause,
};

pub fn ast_hash(parsed: &ParsedCypher) -> String {
    let canonical = canonicalize(parsed);
    digest(&canonical)
}

pub fn params_hash(params: &serde_json::Value) -> String {
    digest(&serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string()))
}

pub fn sql_ast_hash(sql: &str) -> String {
    let collapsed: String = sql.split_whitespace().collect::<Vec<_>>().join(" ");
    digest(&collapsed.to_ascii_lowercase())
}

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("sha256:{hex}", hex = hex::encode(hasher.finalize()))
}

fn canonicalize(parsed: &ParsedCypher) -> String {
    let mut aliases: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut next = 0u32;
    let mut intern = |name: &str| {
        aliases
            .entry(name.to_ascii_lowercase())
            .or_insert_with(|| {
                let mapped = format!("a{next}");
                next += 1;
                mapped
            })
            .clone()
    };
    let mut out = String::new();
    out.push_str(&pattern_canon(&parsed.pattern, &mut intern));
    if let Some(expr) = &parsed.where_expr {
        out.push_str("|where:");
        out.push_str(&where_canon(expr, &mut intern));
    } else if let Some(filter) = parsed.eq_filter() {
        out.push_str(&format!(
            "|where:{}={}={}",
            intern(&filter.binding),
            filter.key.to_ascii_lowercase(),
            filter.value
        ));
    }
    if parsed.distinct {
        out.push_str("|distinct");
    }
    out.push_str("|return:");
    for item in &parsed.returns {
        out.push_str(&return_canon(item, &mut intern));
        out.push(',');
    }
    if let Some(skip) = parsed.skip {
        out.push_str(&format!("|skip:{skip}"));
    }
    if parsed.limit_specified {
        out.push_str(&format!("|limit:{}", parsed.limit));
    }
    if let Some(with) = &parsed.with_clause {
        out.push_str("|with:");
        for item in &with.items {
            match item {
                crate::cypher::ast::WithItem::Field { alias, key, .. } => {
                    out.push_str(&format!(
                        "field:{}:{},",
                        alias.to_ascii_lowercase(),
                        key.as_deref().unwrap_or("*").to_ascii_lowercase()
                    ));
                }
                crate::cypher::ast::WithItem::Aggregate { op, alias, .. } => {
                    out.push_str(&format!("agg:{op:?}:{},", alias.to_ascii_lowercase()));
                }
            }
        }
    }
    if !parsed.order_by.is_empty() {
        out.push_str("|order:");
        for clause in &parsed.order_by {
            out.push_str(&format!(
                "{}{},",
                clause.expression.to_ascii_lowercase(),
                if clause.descending { ":desc" } else { ":asc" }
            ));
        }
    }
    for write in &parsed.writes {
        out.push_str(&format!("|write:{}", write_name(write)));
    }
    out.to_ascii_lowercase()
}

fn pattern_canon(pattern: &CypherPattern, intern: &mut impl FnMut(&str) -> String) -> String {
    match pattern {
        CypherPattern::Node(node) => format!("node:{}", node_canon(node, intern)),
        CypherPattern::Edge(edge) => format!(
            "edge:{}-[{:?}:{}]->{}",
            node_canon(&edge.left, intern),
            edge.direction,
            edge.edge_type.to_ascii_lowercase(),
            node_canon(&edge.right, intern)
        ),
        CypherPattern::EdgeChain(chain) => {
            let mut s = format!("chain:{}", node_canon(&chain.start, intern));
            for step in &chain.steps {
                s.push_str(&step_canon(step, intern));
            }
            s
        }
        CypherPattern::EdgeVarLength(var) => format!(
            "var:{}-[:{}*{:?}..{:?}]->{}",
            node_canon(&var.from, intern),
            var.edge_type.to_ascii_lowercase(),
            var.min,
            var.max,
            node_canon(&var.to, intern)
        ),
    }
}

fn node_canon(node: &NodePattern, intern: &mut impl FnMut(&str) -> String) -> String {
    format!(
        "({}:{}{:?})",
        intern(&node.binding),
        node.label.as_deref().unwrap_or("*").to_ascii_lowercase(),
        node.properties
    )
}

fn step_canon(step: &EdgeStep, intern: &mut impl FnMut(&str) -> String) -> String {
    format!(
        "-[{:?}:{}*{}..{:?}]->{}",
        step.direction,
        step.edge_type.to_ascii_lowercase(),
        step.min,
        step.max,
        node_canon(&step.target, intern)
    )
}

fn where_canon(expr: &WhereExpr, intern: &mut impl FnMut(&str) -> String) -> String {
    match expr {
        WhereExpr::Cmp {
            binding,
            key,
            op,
            value,
        } => format!(
            "{}{}.{}={}",
            intern(binding),
            cmp_token(*op),
            key.to_ascii_lowercase(),
            value
        ),
        WhereExpr::In {
            binding,
            key,
            values,
        } => format!(
            "{}{}.{} in {:?}",
            intern(binding),
            key.to_ascii_lowercase(),
            key,
            values
        ),
        WhereExpr::And(left, right) => format!(
            "({} and {})",
            where_canon(left, intern),
            where_canon(right, intern)
        ),
        WhereExpr::Or(left, right) => format!(
            "({} or {})",
            where_canon(left, intern),
            where_canon(right, intern)
        ),
        WhereExpr::Not(inner) => format!("(not {})", where_canon(inner, intern)),
    }
}

fn cmp_token(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "=",
        CmpOp::Ne => "<>",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

fn return_canon(item: &ReturnItem, intern: &mut impl FnMut(&str) -> String) -> String {
    match item {
        ReturnItem::Variable(binding) => intern(binding),
        ReturnItem::Property { binding, key, .. } => {
            format!("{}.{}", intern(binding), key.to_ascii_lowercase())
        }
        ReturnItem::Count { binding, .. } => {
            format!(
                "count({})",
                binding.as_deref().map(intern).unwrap_or("*".into())
            )
        }
        ReturnItem::Path { binding, .. } => format!("path:{}", intern(binding)),
        ReturnItem::PathNodes { binding, .. } => format!("nodes({})", intern(binding)),
        ReturnItem::PathRels { binding, .. } => format!("rels({})", intern(binding)),
        ReturnItem::PathLength { binding, .. } => format!("length({})", intern(binding)),
        ReturnItem::Aggregate {
            op, binding, key, ..
        } => format!(
            "{op:?}({}.{:?})",
            binding.as_deref().map(intern).unwrap_or_default(),
            key
        ),
        ReturnItem::Aliased { item, alias } => {
            format!(
                "{} as {}",
                return_canon(item, intern),
                alias.to_ascii_lowercase()
            )
        }
    }
}

fn write_name(write: &WriteClause) -> &'static str {
    match write {
        WriteClause::CreateNode { .. } | WriteClause::CreateEdge { .. } => "create",
        WriteClause::Merge { .. } => "merge",
        WriteClause::Set { .. } => "set",
        WriteClause::Delete { .. } => "delete",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_cypher_pest;
    use std::collections::BTreeMap;

    #[test]
    fn alias_and_whitespace_share_ast_hash() {
        let a = parse_cypher_pest("MATCH (n:Doc) RETURN n", &BTreeMap::new()).unwrap();
        let b = parse_cypher_pest("match  (x:Doc)  return x", &BTreeMap::new()).unwrap();
        assert_eq!(ast_hash(&a), ast_hash(&b));
    }

    #[test]
    fn different_literals_hash_differently() {
        let a =
            parse_cypher_pest("MATCH (n:Doc) WHERE n.k = 1 RETURN n", &BTreeMap::new()).unwrap();
        let b =
            parse_cypher_pest("MATCH (n:Doc) WHERE n.k = 2 RETURN n", &BTreeMap::new()).unwrap();
        assert_ne!(ast_hash(&a), ast_hash(&b));
    }

    #[test]
    fn return_aliases_are_part_of_the_hash() {
        let a = parse_cypher_pest("MATCH (n:Doc) RETURN n AS foo", &BTreeMap::new()).unwrap();
        let b = parse_cypher_pest("MATCH (n:Doc) RETURN n AS bar", &BTreeMap::new()).unwrap();
        assert_ne!(ast_hash(&a), ast_hash(&b));
    }
}
