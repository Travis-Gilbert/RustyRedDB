use std::str::FromStr;

use egg::{rewrite, Applier, EGraph, Id, Pattern, PatternAst, Rewrite, Subst, Symbol, Var};
use rustyred_thg_core::{Predicate, ScalarBound, ScalarValue};

use crate::cost::SaturationAnalysis;
use crate::lang::{EncodedAttr, PredicateAttrs, QueryLang};

pub type QueryRewrite = Rewrite<QueryLang, SaturationAnalysis>;

pub const SET_COMMUTE: &str = "set-commute";
pub const SET_COMMUTE_UNION: &str = "set-commute-union";
pub const SET_ASSOCIATE: &str = "set-associate";
pub const SET_ASSOCIATE_REVERSE: &str = "set-associate-reverse";
pub const SET_ASSOCIATE_UNION: &str = "set-associate-union";
pub const SET_ASSOCIATE_UNION_REVERSE: &str = "set-associate-union-reverse";
pub const LAYER_SPLIT: &str = "layer-split";
pub const LAYER_MERGE: &str = "layer-merge";
pub const PREDICATE_EQUALS_RANGE: &str = "predicate-equals-range";
pub const PREDICATE_PREFIX_RANGE: &str = "predicate-prefix-range";
pub const RANK_CANDIDATE_PUSHDOWN: &str = "rank-candidate-pushdown";
pub const RANK_CANDIDATE_ASSOCIATE: &str = "rank-candidate-associate";
pub const RANK_CANDIDATE_ASSOCIATE_REVERSE: &str = "rank-candidate-associate-reverse";
pub const TOPK_LIMIT_UNION: &str = "topk-limit-union";
pub const TOPK_LIMIT_COLLAPSE: &str = "topk-limit-collapse";
pub const EXISTS_TO_SEMI: &str = "exists-to-semi";
pub const NOT_EXISTS_TO_ANTI: &str = "not-exists-to-anti";

pub fn rewrite_rules_v1() -> Vec<QueryRewrite> {
    vec![
        rewrite!(SET_COMMUTE; "(intersect ?a ?b)" => "(intersect ?b ?a)"),
        rewrite!(SET_COMMUTE_UNION; "(union ?a ?b)" => "(union ?b ?a)"),
        rewrite!(
            SET_ASSOCIATE_REVERSE;
            "(intersect (intersect ?a ?b) ?c)" => "(intersect ?a (intersect ?b ?c))"
        ),
        rewrite!(
            SET_ASSOCIATE_UNION;
            "(intersect ?a (intersect ?b ?c))" => "(intersect (intersect ?a ?b) ?c)"
        ),
        rewrite!(
            SET_ASSOCIATE_UNION_REVERSE;
            "(union (union ?a ?b) ?c)" => "(union ?a (union ?b ?c))"
        ),
        rewrite!(
            SET_ASSOCIATE;
            "(union ?a (union ?b ?c))" => "(union (union ?a ?b) ?c)"
        ),
        rewrite!(
            LAYER_SPLIT;
            "(expand ?from (layers ?a (layers ?b layers-empty)) ?dir)" =>
            "(union
                (expand ?from (layers ?a layers-empty) ?dir)
                (expand ?from (layers ?b layers-empty) ?dir))"
        ),
        rewrite!(
            LAYER_MERGE;
            "(union
                (expand ?from (layers ?a layers-empty) ?dir)
                (expand ?from (layers ?b layers-empty) ?dir))" =>
            "(expand ?from (layers ?a (layers ?b layers-empty)) ?dir)"
        ),
        predicate_algebra_rule(PREDICATE_EQUALS_RANGE, AlgebraFamily::EqualsRange),
        predicate_algebra_rule(PREDICATE_PREFIX_RANGE, AlgebraFamily::PrefixRange),
        rewrite!(
            RANK_CANDIDATE_PUSHDOWN;
            "(rank (intersect ?a ?b) ?p ?layer ?k)" =>
            "(rank (intersect ?b ?a) ?p ?layer ?k)"
        ),
        rewrite!(
            RANK_CANDIDATE_ASSOCIATE;
            "(rank (intersect (intersect ?a ?b) ?c) ?p ?layer ?k)" =>
            "(rank (intersect ?a (intersect ?b ?c)) ?p ?layer ?k)"
        ),
        rewrite!(
            RANK_CANDIDATE_ASSOCIATE_REVERSE;
            "(rank (intersect ?a (intersect ?b ?c)) ?p ?layer ?k)" =>
            "(rank (intersect (intersect ?a ?b) ?c) ?p ?layer ?k)"
        ),
        rewrite!(
            TOPK_LIMIT_UNION;
            "(limit
                (union
                    (rank ?a ?pa ?la ?ka)
                    (rank ?b ?pb ?lb ?kb))
                ?k)" =>
            "(limit
                (union
                    (limit (rank ?a ?pa ?la ?ka) ?k)
                    (limit (rank ?b ?pb ?lb ?kb) ?k))
                ?k)"
        ),
        topk_collapse_rule(),
        rewrite!(EXISTS_TO_SEMI; "(exists ?q)" => "(semi-join ?q)"),
        rewrite!(NOT_EXISTS_TO_ANTI; "(not-exists ?q)" => "(anti-join ?q)"),
    ]
}

pub fn known_rule_ids() -> &'static [&'static str] {
    &[
        SET_COMMUTE,
        SET_COMMUTE_UNION,
        SET_ASSOCIATE,
        SET_ASSOCIATE_REVERSE,
        SET_ASSOCIATE_UNION,
        SET_ASSOCIATE_UNION_REVERSE,
        LAYER_SPLIT,
        LAYER_MERGE,
        PREDICATE_EQUALS_RANGE,
        PREDICATE_PREFIX_RANGE,
        RANK_CANDIDATE_PUSHDOWN,
        RANK_CANDIDATE_ASSOCIATE,
        RANK_CANDIDATE_ASSOCIATE_REVERSE,
        TOPK_LIMIT_UNION,
        TOPK_LIMIT_COLLAPSE,
        EXISTS_TO_SEMI,
        NOT_EXISTS_TO_ANTI,
    ]
}

#[derive(Clone, Copy)]
enum AlgebraFamily {
    EqualsRange,
    PrefixRange,
}

fn predicate_algebra_rule(name: &str, family: AlgebraFamily) -> QueryRewrite {
    Rewrite::new(
        name,
        Pattern::<QueryLang>::from_str("(scan ?relation ?predicate)")
            .expect("valid predicate algebra search pattern"),
        PredicateAlgebraApplier {
            relation: "?relation".parse().expect("valid variable"),
            predicate: "?predicate".parse().expect("valid variable"),
            family,
        },
    )
    .expect("predicate algebra variables are bound")
}

struct PredicateAlgebraApplier {
    relation: Var,
    predicate: Var,
    family: AlgebraFamily,
}

impl Applier<QueryLang, SaturationAnalysis> for PredicateAlgebraApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<QueryLang, SaturationAnalysis>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<QueryLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(mut attrs) =
            decode_attr_from_eclass::<PredicateAttrs>(egraph, subst[self.predicate])
        else {
            return Vec::new();
        };
        let replacement = match self.family {
            AlgebraFamily::EqualsRange => equals_range_counterpart(&attrs.predicate),
            AlgebraFamily::PrefixRange => prefix_range_counterpart(&attrs.predicate),
        };
        let Some(replacement) = replacement else {
            return Vec::new();
        };
        attrs.predicate = replacement;
        let Ok(encoded) = EncodedAttr::from_json(&attrs) else {
            return Vec::new();
        };
        if let Some(searcher_ast) = _searcher_ast {
            let Ok(rhs) = PatternAst::<QueryLang>::from_str(&format!("(scan ?relation {encoded})"))
            else {
                return Vec::new();
            };
            let (id, changed) = egraph.union_instantiations(searcher_ast, &rhs, subst, rule_name);
            if changed {
                vec![id]
            } else {
                Vec::new()
            }
        } else {
            let predicate = egraph.add(QueryLang::Attr(encoded));
            let scan = egraph.add(QueryLang::Scan([subst[self.relation], predicate]));
            if egraph.union(eclass, scan) {
                vec![scan]
            } else {
                Vec::new()
            }
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.relation, self.predicate]
    }
}

fn topk_collapse_rule() -> QueryRewrite {
    Rewrite::new(
        TOPK_LIMIT_COLLAPSE,
        Pattern::<QueryLang>::from_str("(limit (limit ?source ?inner) ?outer)")
            .expect("valid topk search pattern"),
        TopkCollapseApplier {
            source: "?source".parse().expect("valid variable"),
            inner: "?inner".parse().expect("valid variable"),
            outer: "?outer".parse().expect("valid variable"),
        },
    )
    .expect("topk variables are bound")
}

struct TopkCollapseApplier {
    source: Var,
    inner: Var,
    outer: Var,
}

impl Applier<QueryLang, SaturationAnalysis> for TopkCollapseApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<QueryLang, SaturationAnalysis>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<QueryLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(inner) = decode_attr_from_eclass::<usize>(egraph, subst[self.inner]) else {
            return Vec::new();
        };
        let Some(outer) = decode_attr_from_eclass::<usize>(egraph, subst[self.outer]) else {
            return Vec::new();
        };
        let Ok(encoded) = EncodedAttr::from_json(&inner.min(outer)) else {
            return Vec::new();
        };
        if let Some(searcher_ast) = _searcher_ast {
            let Ok(rhs) = PatternAst::<QueryLang>::from_str(&format!("(limit ?source {encoded})"))
            else {
                return Vec::new();
            };
            let (id, changed) = egraph.union_instantiations(searcher_ast, &rhs, subst, rule_name);
            if changed {
                vec![id]
            } else {
                Vec::new()
            }
        } else {
            let k = egraph.add(QueryLang::Attr(encoded));
            let limit = egraph.add(QueryLang::Limit([subst[self.source], k]));
            if egraph.union(eclass, limit) {
                vec![limit]
            } else {
                Vec::new()
            }
        }
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.source, self.inner, self.outer]
    }
}

fn decode_attr_from_eclass<T: serde::de::DeserializeOwned>(
    egraph: &EGraph<QueryLang, SaturationAnalysis>,
    id: Id,
) -> Option<T> {
    egraph[id].nodes.iter().find_map(|node| match node {
        QueryLang::Attr(attr) => attr.decode().ok(),
        _ => None,
    })
}

pub(crate) fn algebra_equivalents(predicate: &Predicate) -> Vec<Predicate> {
    let mut predicates = vec![predicate.clone()];
    if let Some(counterpart) = equals_range_counterpart(predicate) {
        predicates.push(counterpart);
    }
    if let Some(counterpart) = prefix_range_counterpart(predicate) {
        predicates.push(counterpart);
    }
    predicates
}

pub(crate) fn canonical_algebra_predicate(predicate: &Predicate) -> Predicate {
    match predicate {
        Predicate::Range { lo, hi, .. } if lo == hi => {
            equals_range_counterpart(predicate).unwrap_or_else(|| predicate.clone())
        }
        Predicate::Range {
            lo: ScalarBound::Included(ScalarValue::String(_)),
            ..
        } => prefix_range_counterpart(predicate).unwrap_or_else(|| predicate.clone()),
        _ => predicate.clone(),
    }
}

fn equals_range_counterpart(predicate: &Predicate) -> Option<Predicate> {
    match predicate {
        Predicate::Equals { column, value } if value.as_f64().is_some() => Some(Predicate::Range {
            column: column.clone(),
            lo: ScalarBound::Included(value.clone()),
            hi: ScalarBound::Included(value.clone()),
        }),
        Predicate::Range { column, lo, hi }
            if lo == hi
                && matches!(
                    lo,
                    ScalarBound::Included(ScalarValue::I64(_))
                        | ScalarBound::Included(ScalarValue::F64(_))
                ) =>
        {
            let ScalarBound::Included(value) = lo else {
                return None;
            };
            Some(Predicate::Equals {
                column: column.clone(),
                value: value.clone(),
            })
        }
        _ => None,
    }
}

fn prefix_range_counterpart(predicate: &Predicate) -> Option<Predicate> {
    match predicate {
        Predicate::Prefix { column, prefix } => Some(Predicate::Range {
            column: column.clone(),
            lo: ScalarBound::Included(ScalarValue::String(prefix.clone())),
            hi: prefix_upper_bound(prefix)
                .map(ScalarValue::String)
                .map(ScalarBound::Excluded)
                .unwrap_or(ScalarBound::Unbounded),
        }),
        Predicate::Range { column, lo, hi } => {
            let ScalarBound::Included(ScalarValue::String(prefix)) = lo else {
                return None;
            };
            let expected = prefix_upper_bound(prefix)
                .map(ScalarValue::String)
                .map(ScalarBound::Excluded)
                .unwrap_or(ScalarBound::Unbounded);
            (hi == &expected).then(|| Predicate::Prefix {
                column: column.clone(),
                prefix: prefix.clone(),
            })
        }
        _ => None,
    }
}

fn prefix_upper_bound(prefix: &str) -> Option<String> {
    let mut chars = prefix.chars().collect::<Vec<_>>();
    for index in (0..chars.len()).rev() {
        let codepoint = chars[index] as u32;
        let next = match codepoint {
            0..=0xd7fe => char::from_u32(codepoint + 1),
            0xd7ff => char::from_u32(0xe000),
            0xe000..=0x10fffe => char::from_u32(codepoint + 1),
            _ => None,
        };
        if let Some(next) = next {
            chars.truncate(index + 1);
            chars[index] = next;
            return Some(chars.into_iter().collect());
        }
    }
    None
}
