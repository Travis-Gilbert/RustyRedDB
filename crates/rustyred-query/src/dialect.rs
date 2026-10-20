use serde::Serialize;

use crate::cypher::parse::{parse_cypher_pest, CypherPestParser};
use crate::cypher::Rule;
use pest::Parser;

#[derive(Clone, Debug, Serialize)]
pub struct DialectBlock {
    pub cypher_clauses_supported: Vec<String>,
    pub cypher_constructs_unsupported: Vec<String>,
    pub sql_clauses_supported: Vec<String>,
    pub sql_constructs_unsupported: Vec<String>,
    pub schema_version: String,
    pub notes: Vec<String>,
}

pub struct UnsupportedConstruct {
    pub name: &'static str,
    pub probe: &'static str,
}

pub const SUPPORTED_PROBES: &[(&str, &str)] = &[
    (
        "MATCH (single pattern, optional path binding)",
        "MATCH p = (n:Doc) RETURN n",
    ),
    (
        "WHERE (AND / OR / NOT, comparisons, IN)",
        "MATCH (n:Doc) WHERE n.k = 1 AND NOT n.n = 2 OR n.k IN [1, 2] RETURN n",
    ),
    ("WITH ... AS", "MATCH (n:Doc) WITH n AS x RETURN x"),
    (
        "ORDER BY ... ASC|DESC",
        "MATCH (n:Doc) RETURN n ORDER BY n.k DESC",
    ),
    ("SKIP", "MATCH (n:Doc) RETURN n SKIP 1"),
    ("LIMIT", "MATCH (n:Doc) RETURN n LIMIT 2"),
    ("RETURN", "MATCH (n:Doc) RETURN n"),
    (
        "aggregates: COUNT SUM AVG MIN MAX",
        "MATCH (n:Doc) RETURN count(n), sum(n.k), avg(n.k), min(n.k), max(n.k)",
    ),
    (
        "patterns: node, outbound/inbound/undirected chain, variable length -[:TYPE*n..m]->",
        "MATCH (a:Doc)-[:CITES*1..3]->(b:Doc)<-[:CITES]-(c:Doc) RETURN a, b, c",
    ),
    ("DISTINCT", "MATCH (n:Doc) RETURN DISTINCT n"),
    ("RETURN ... AS", "MATCH (n:Doc) RETURN n AS doc"),
    (
        "path functions nodes() relationships() length()",
        "MATCH p = (a:Doc)-[:CITES]->(b:Doc) RETURN nodes(p), relationships(p), length(p)",
    ),
];

pub const UNSUPPORTED_CONSTRUCTS: &[UnsupportedConstruct] = &[
    UnsupportedConstruct {
        name: "OPTIONAL MATCH",
        probe: "OPTIONAL MATCH (n:Doc) RETURN n",
    },
    UnsupportedConstruct {
        name: "multiple comma-separated patterns in one MATCH",
        probe: "MATCH (a:Doc), (b:Doc) RETURN a, b",
    },
    UnsupportedConstruct {
        name: "subqueries",
        probe: "MATCH (n:Doc) WHERE EXISTS { MATCH (n)-[:CITES]->(m) } RETURN n",
    },
    UnsupportedConstruct {
        name: "UNION",
        probe: "MATCH (n:Doc) RETURN n UNION MATCH (m:File) RETURN m",
    },
    UnsupportedConstruct {
        name: "list comprehensions",
        probe: "MATCH (n:Doc) RETURN [x IN n.tags | x]",
    },
    UnsupportedConstruct {
        name: "user-defined functions",
        probe: "MATCH (n:Doc) RETURN myUdf(n)",
    },
    UnsupportedConstruct {
        name: "CONTAINS",
        probe: "MATCH (n:Doc) WHERE n.title CONTAINS 'a' RETURN n",
    },
    UnsupportedConstruct {
        name: "STARTS WITH",
        probe: "MATCH (n:Doc) WHERE n.title STARTS WITH 'a' RETURN n",
    },
    UnsupportedConstruct {
        name: "COUNT(DISTINCT)",
        probe: "MATCH (n:Doc) RETURN COUNT(DISTINCT n)",
    },
    UnsupportedConstruct {
        name: "WHERE after WITH",
        probe: "MATCH (n:Doc) WITH count(n) AS c WHERE c > 1 RETURN c",
    },
];

pub const VARIABLE_LENGTH_PROBES: &[&str] = &[
    "MATCH (a:Doc)-[:CITES*1..3]->(b:Doc) RETURN a, b",
    "MATCH (a:Doc)-[:CITES*2]->(b:Doc) RETURN a, b",
    "MATCH (a:Doc)-[:CITES*]->(b:Doc) RETURN a, b",
];

pub const SQL_SUPPORTED: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP BY",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "aggregates: COUNT SUM AVG MIN MAX",
    "INNER JOIN / JOIN",
];

pub const SQL_UNSUPPORTED: &[UnsupportedConstruct] = &[
    UnsupportedConstruct {
        name: "INSERT",
        probe: "INSERT INTO t VALUES (1)",
    },
    UnsupportedConstruct {
        name: "UPDATE",
        probe: "UPDATE t SET x = 1",
    },
    UnsupportedConstruct {
        name: "DELETE",
        probe: "DELETE FROM t",
    },
    UnsupportedConstruct {
        name: "CREATE TABLE",
        probe: "CREATE TABLE t (id INT)",
    },
    UnsupportedConstruct {
        name: "WITH (CTE)",
        probe: "WITH s AS (SELECT 1) SELECT * FROM s",
    },
];

pub fn dialect_block(schema_version: &str) -> DialectBlock {
    DialectBlock {
        cypher_clauses_supported: SUPPORTED_PROBES
            .iter()
            .map(|(label, _)| (*label).to_string())
            .collect(),
        cypher_constructs_unsupported: UNSUPPORTED_CONSTRUCTS
            .iter()
            .map(|item| item.name.to_string())
            .collect(),
        sql_clauses_supported: SQL_SUPPORTED
            .iter()
            .map(|item| (*item).to_string())
            .collect(),
        sql_constructs_unsupported: SQL_UNSUPPORTED
            .iter()
            .map(|item| item.name.to_string())
            .collect(),
        schema_version: schema_version.to_string(),
        notes: vec!["unsupported constructs are refused by name with a pest span".to_string()],
    }
}

pub fn detect_unsupported(query: &str) -> Option<&'static UnsupportedConstruct> {
    detect_in(query, UNSUPPORTED_CONSTRUCTS)
}

pub fn detect_sql_unsupported(query: &str) -> Option<&'static UnsupportedConstruct> {
    detect_in(query, SQL_UNSUPPORTED)
}

fn detect_in(
    query: &str,
    items: &'static [UnsupportedConstruct],
) -> Option<&'static UnsupportedConstruct> {
    let lower = query.to_ascii_lowercase();
    let stripped = strip_quoted_literals(&lower);
    for item in items {
        let needle = item.name.to_ascii_lowercase();
        if needle == "multiple comma-separated patterns in one match" {
            if stripped.contains("match") && (stripped.contains("), (") || stripped.contains("),("))
            {
                return Some(item);
            }
            continue;
        }
        if needle == "subqueries" {
            if stripped.contains("exists {") || stripped.contains("exists{") {
                return Some(item);
            }
            continue;
        }
        if needle == "list comprehensions" {
            if stripped.contains(" in ") && stripped.contains('[') && stripped.contains('|') {
                return Some(item);
            }
            continue;
        }
        if needle == "count(distinct)" {
            let collapsed: String = stripped.chars().filter(|ch| !ch.is_whitespace()).collect();
            if collapsed.contains("count(distinct") {
                return Some(item);
            }
            continue;
        }
        if needle == "where after with" {
            if let Some(with_idx) = stripped.find(" with ") {
                if stripped[with_idx + 6..].contains(" where ") {
                    return Some(item);
                }
            }
            continue;
        }
        if needle == "with (cte)" {
            if stripped.trim_start().starts_with("with ") {
                return Some(item);
            }
            continue;
        }
        if needle == "user-defined functions" {
            let known = [
                "count",
                "sum",
                "avg",
                "min",
                "max",
                "nodes",
                "relationships",
                "length",
            ];
            if let Some(idx) = stripped.find("return ") {
                let rest = stripped[idx + 7..].trim();
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim();
                    if !name.is_empty()
                        && name
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                        && !known.contains(&name)
                    {
                        return Some(item);
                    }
                }
            }
            continue;
        }
        if stripped.contains(&needle) {
            return Some(item);
        }
    }
    None
}

fn strip_quoted_literals(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' || ch == '"' {
            let quote = ch;
            while let Some(next) = chars.next() {
                if next == '\\' {
                    chars.next();
                    continue;
                }
                if next == quote {
                    break;
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

pub fn supported_alternative(name: &str) -> &'static str {
    match name {
        "multiple comma-separated patterns in one MATCH" => {
            "rewrite as successive MATCH ... WITH stages, one pattern each"
        }
        "AND / OR / NOT in WHERE" => {
            "rewrite as a second MATCH-and-WITH stage or a params pre-filter"
        }
        "inbound or undirected arrows <-[...]- and -[...]-" => {
            "reverse the pattern and bind the target in WHERE"
        }
        "COUNT(DISTINCT)" => {
            "no equivalent in the supported dialect; COUNT the binding, or RETURN DISTINCT in a separate query"
        }
        "WHERE after WITH" => {
            "no equivalent in the supported dialect; filter in MATCH WHERE before WITH"
        }
        _ => "no equivalent in the supported dialect",
    }
}

pub fn parse_probe(query: &str) -> bool {
    CypherPestParser::parse(Rule::query, query).is_ok()
        && parse_cypher_pest(query, &Default::default()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_probes_parse() {
        for (label, probe) in SUPPORTED_PROBES {
            assert!(
                parse_probe(probe),
                "supported clause {label} failed to parse: {probe}"
            );
        }
    }

    #[test]
    fn unsupported_probes_fail_to_parse() {
        for item in UNSUPPORTED_CONSTRUCTS {
            assert!(
                !parse_probe(item.probe),
                "unsupported construct {} started parsing: {}",
                item.name,
                item.probe
            );
        }
    }

    #[test]
    fn variable_length_probes_parse() {
        for probe in VARIABLE_LENGTH_PROBES {
            assert!(parse_probe(probe), "variable-length probe failed: {probe}");
        }
    }

    #[test]
    fn detect_unsupported_names_every_probe() {
        for item in UNSUPPORTED_CONSTRUCTS {
            assert_eq!(
                detect_unsupported(item.probe).map(|found| found.name),
                Some(item.name),
                "detector missed {}",
                item.name
            );
        }
    }
}
