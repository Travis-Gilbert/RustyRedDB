use std::fmt;
use std::str::FromStr;

use egg::{Id, RecExpr};
use rustyred_thg_core::{
    Direction, FusionPolicy, GraphStoreError, GraphStoreResult, JoinPredicate, LayerId, Predicate,
    PredicateMode, Projection, ProvenanceSemiringKind, QueryIr as QueryInput, QueryRelation,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

egg::define_language! {
    pub enum QueryLang {
        "scan" = Scan([Id; 2]),
        "expand" = Expand([Id; 3]),
        "layer" = Layer([Id; 1]),
        "layers" = Layers([Id; 2]),
        "layers-empty" = LayersEmpty,
        "intersect" = Intersect([Id; 2]),
        "union" = Union([Id; 2]),
        "rank" = Rank([Id; 4]),
        "limit" = Limit([Id; 2]),
        "query" = QueryRoot([Id; 3]),
        "exists" = Exists([Id; 1]),
        "not-exists" = NotExists([Id; 1]),
        "semi-join" = SemiJoin([Id; 1]),
        "anti-join" = AntiJoin([Id; 1]),
        Attr(EncodedAttr),
    }
}

pub type QueryInputAlias = QueryInput;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QueryIr {
    pub graph_version: u64,
    pub cost_snapshot_ref: String,
    pub body: QueryInput,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct RootAttrs {
    pub graph_version: u64,
    pub joins: Vec<JoinPredicate>,
    pub projection: Vec<Projection>,
    pub fusion: FusionPolicy,
    pub semiring: Option<ProvenanceSemiringKind>,
    pub pinned_graph_version: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct RelationAttrs {
    pub alias: String,
    pub relation: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct PredicateAttrs {
    pub relation: RelationAttrs,
    pub ordinal: usize,
    pub predicate: Predicate,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct AllRowsAttrs {
    pub all_rows: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ExpandAttrs {
    pub relation: RelationAttrs,
    pub ordinal: usize,
    pub from: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct EncodedAttr(String);

impl EncodedAttr {
    pub fn from_json<T: Serialize>(value: &T) -> GraphStoreResult<Self> {
        let bytes = serde_json::to_vec(value).map_err(|source| {
            GraphStoreError::new(
                "query_saturate_attribute_encode",
                format!("failed to encode query attribute: {source}"),
            )
        })?;
        Ok(Self(hex_encode(&bytes)))
    }

    pub fn decode<T: DeserializeOwned>(&self) -> GraphStoreResult<T> {
        let bytes = hex_decode(&self.0).map_err(|source| {
            GraphStoreError::new(
                "query_saturate_attribute_decode",
                format!("failed to decode query attribute: {source}"),
            )
        })?;
        serde_json::from_slice(&bytes).map_err(|source| {
            GraphStoreError::new(
                "query_saturate_attribute_decode",
                format!("failed to decode query attribute: {source}"),
            )
        })
    }
}

impl fmt::Display for EncodedAttr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "attr:{}", self.0)
    }
}

impl FromStr for EncodedAttr {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let payload = value
            .strip_prefix("attr:")
            .ok_or_else(|| "encoded attrs must start with attr:".to_string())?;
        hex_decode(payload)?;
        Ok(Self(payload.to_string()))
    }
}

pub fn query_input_to_expr(
    input: &QueryInput,
    graph_version: u64,
    snapshot_ref: impl Into<String>,
) -> GraphStoreResult<RecExpr<QueryLang>> {
    query_ir_to_expr(&QueryIr {
        graph_version,
        cost_snapshot_ref: snapshot_ref.into(),
        body: input.clone(),
    })
}

pub fn query_ir_to_expr(ir: &QueryIr) -> GraphStoreResult<RecExpr<QueryLang>> {
    rustyred_thg_core::validate_query_ir(&ir.body)?;
    let mut expr = RecExpr::default();
    let root_attrs = RootAttrs {
        graph_version: ir.graph_version,
        joins: ir.body.joins.clone(),
        projection: ir.body.projection.clone(),
        fusion: ir.body.fusion.clone(),
        semiring: ir.body.semiring,
        pinned_graph_version: ir.body.pinned_graph_version,
    };
    let graph_version = push_attr(&mut expr, &root_attrs)?;
    let snapshot_ref = push_attr(&mut expr, &ir.cost_snapshot_ref)?;

    let mut relation_bodies = Vec::with_capacity(ir.body.relations.len());
    for relation in &ir.body.relations {
        relation_bodies.push(encode_relation(&mut expr, relation, ir.body.limit)?);
    }
    let mut body = relation_bodies
        .into_iter()
        .next()
        .ok_or_else(|| GraphStoreError::new("empty_relational_query", "query has no relations"))?;
    for relation in ir.body.relations.iter().skip(1) {
        let next = encode_relation(&mut expr, relation, ir.body.limit)?;
        body = expr.add(QueryLang::Intersect([body, next]));
    }
    if let Some(limit) = ir.body.limit {
        let limit_attr = push_attr(&mut expr, &limit)?;
        body = expr.add(QueryLang::Limit([body, limit_attr]));
    }
    expr.add(QueryLang::QueryRoot([graph_version, snapshot_ref, body]));
    Ok(expr)
}

fn encode_relation(
    expr: &mut RecExpr<QueryLang>,
    relation: &QueryRelation,
    query_limit: Option<usize>,
) -> GraphStoreResult<Id> {
    let relation_attrs = RelationAttrs {
        alias: relation.alias.clone(),
        relation: relation.relation.clone(),
    };
    let relation_attr = push_attr(expr, &relation_attrs)?;
    let all_attr = push_attr(expr, &AllRowsAttrs { all_rows: true })?;
    let mut source = expr.add(QueryLang::Scan([relation_attr, all_attr]));
    let mut has_filter = false;

    for (ordinal, predicate) in relation.predicates.iter().enumerate() {
        if predicate_mode(predicate) == PredicateMode::Rank {
            continue;
        }
        let term = encode_filter(expr, &relation_attrs, ordinal, predicate)?;
        source = if has_filter {
            expr.add(QueryLang::Intersect([source, term]))
        } else {
            has_filter = true;
            term
        };
    }

    for (ordinal, predicate) in relation.predicates.iter().enumerate() {
        if predicate_mode(predicate) != PredicateMode::Rank {
            continue;
        }
        let attrs = PredicateAttrs {
            relation: relation_attrs.clone(),
            ordinal,
            predicate: predicate.clone(),
        };
        let predicate_attr = push_attr(expr, &attrs)?;
        let layers = match predicate {
            Predicate::Expand { layers, .. } => layers.clone(),
            _ => Vec::new(),
        };
        let k = predicate_k(predicate).or(query_limit).unwrap_or(usize::MAX);
        source = rank_term(expr, source, predicate_attr, &layers, k)?;
    }
    Ok(source)
}

fn encode_filter(
    expr: &mut RecExpr<QueryLang>,
    relation: &RelationAttrs,
    ordinal: usize,
    predicate: &Predicate,
) -> GraphStoreResult<Id> {
    match predicate {
        Predicate::Expand {
            from, layers, dir, ..
        } => {
            let from = push_attr(
                expr,
                &ExpandAttrs {
                    relation: relation.clone(),
                    ordinal,
                    from: from.clone(),
                },
            )?;
            let layer_set = encode_layer_set(expr, layers)?;
            let dir = push_attr(expr, dir)?;
            Ok(expr.add(QueryLang::Expand([from, layer_set, dir])))
        }
        _ => {
            let relation_attr = push_attr(expr, relation)?;
            let predicate_attr = push_attr(
                expr,
                &PredicateAttrs {
                    relation: relation.clone(),
                    ordinal,
                    predicate: predicate.clone(),
                },
            )?;
            Ok(expr.add(QueryLang::Scan([relation_attr, predicate_attr])))
        }
    }
}

pub fn rank_term(
    expr: &mut RecExpr<QueryLang>,
    source: Id,
    predicate_attr: Id,
    layers: &[LayerId],
    k: usize,
) -> GraphStoreResult<Id> {
    if layers.len() > 1 {
        return Err(GraphStoreError::new(
            "multi_layer_rank_undefined",
            format!(
                "rank mode requires at most one layer; got {} layers",
                layers.len()
            ),
        ));
    }
    let layer_attr = push_attr(expr, &layers.first().cloned())?;
    let layer = expr.add(QueryLang::Layer([layer_attr]));
    let k = push_attr(expr, &k)?;
    Ok(expr.add(QueryLang::Rank([source, predicate_attr, layer, k])))
}

pub fn expr_to_query_ir(expr: &RecExpr<QueryLang>) -> GraphStoreResult<QueryIr> {
    decode_query_ir(expr, true)
}

fn decode_query_ir(
    expr: &RecExpr<QueryLang>,
    restore_input_order: bool,
) -> GraphStoreResult<QueryIr> {
    let root = expr
        .as_ref()
        .last()
        .ok_or_else(|| GraphStoreError::new("empty_query_expression", "query term is empty"))?;
    let QueryLang::QueryRoot([root_id, snapshot_id, body_id]) = root else {
        return Err(unexpected("query root", root));
    };
    let root: RootAttrs = decode_attr(expr, *root_id)?;
    let cost_snapshot_ref: String = decode_attr(expr, *snapshot_id)?;
    let mut decoded = DecodedBody::default();
    decode_body(expr, *body_id, &mut decoded)?;

    let mut relations = Vec::with_capacity(decoded.relations.len());
    for mut relation in decoded.relations {
        if restore_input_order {
            relation.predicates.sort_by_key(|(ordinal, _)| *ordinal);
        }
        relation
            .predicates
            .dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
        relations.push(QueryRelation {
            alias: relation.attrs.alias,
            relation: relation.attrs.relation,
            predicates: relation
                .predicates
                .into_iter()
                .map(|(_, predicate)| predicate)
                .collect(),
        });
    }
    let body = QueryInput {
        relations,
        joins: root.joins,
        projection: root.projection,
        limit: decoded.limit,
        fusion: root.fusion,
        semiring: root.semiring,
        pinned_graph_version: root.pinned_graph_version,
        ..QueryInput::default()
    };
    rustyred_thg_core::validate_query_ir(&body)?;
    Ok(QueryIr {
        graph_version: root.graph_version,
        cost_snapshot_ref,
        body,
    })
}

pub fn expr_to_query_input(expr: &RecExpr<QueryLang>) -> GraphStoreResult<QueryInput> {
    Ok(decode_query_ir(expr, false)?.body)
}

#[derive(Default)]
struct DecodedBody {
    relations: Vec<DecodedRelation>,
    limit: Option<usize>,
}

struct DecodedRelation {
    attrs: RelationAttrs,
    predicates: Vec<(usize, Predicate)>,
}

fn decode_body(
    expr: &RecExpr<QueryLang>,
    id: Id,
    decoded: &mut DecodedBody,
) -> GraphStoreResult<()> {
    match &expr[id] {
        QueryLang::Intersect([left, right]) => {
            decode_body(expr, *left, decoded)?;
            decode_body(expr, *right, decoded)
        }
        QueryLang::Union([left, right]) => {
            let left_expand = decode_single_expand(expr, *left)?;
            let right_expand = decode_single_expand(expr, *right)?;
            if left_expand.0 != right_expand.0
                || left_expand.1 != right_expand.1
                || left_expand.2 != right_expand.2
                || left_expand.4 != right_expand.4
            {
                return Err(GraphStoreError::new(
                    "unsupported_query_union",
                    "only exact unions of matching expand predicates lower to core QueryIr",
                ));
            }
            let mut layers = left_expand.3;
            layers.extend(right_expand.3);
            layers.sort();
            layers.dedup();
            insert_predicate(
                decoded,
                left_expand.0,
                left_expand.1,
                Predicate::Expand {
                    from: left_expand.2,
                    layers,
                    dir: left_expand.4,
                    mode: PredicateMode::Filter,
                },
            );
            Ok(())
        }
        QueryLang::Scan([relation_id, predicate_id]) => {
            let relation: RelationAttrs = decode_attr(expr, *relation_id)?;
            if decode_attr::<AllRowsAttrs>(expr, *predicate_id).is_ok() {
                ensure_relation(decoded, relation);
                return Ok(());
            }
            let predicate: PredicateAttrs = decode_attr(expr, *predicate_id)?;
            insert_predicate(
                decoded,
                predicate.relation,
                predicate.ordinal,
                predicate.predicate,
            );
            Ok(())
        }
        QueryLang::Expand(_) => {
            let (relation, ordinal, from, layers, dir) = decode_single_expand(expr, id)?;
            insert_predicate(
                decoded,
                relation,
                ordinal,
                Predicate::Expand {
                    from,
                    layers,
                    dir,
                    mode: PredicateMode::Filter,
                },
            );
            Ok(())
        }
        QueryLang::Rank([source, predicate_id, layer_id, _k]) => {
            decode_body(expr, *source, decoded)?;
            let predicate: PredicateAttrs = decode_attr(expr, *predicate_id)?;
            let rank_layer = decode_rank_layer(expr, *layer_id)?;
            if let Predicate::Expand { layers, .. } = &predicate.predicate {
                if layers.len() != 1 || rank_layer.as_ref() != layers.first() {
                    return Err(GraphStoreError::new(
                        "multi_layer_rank_undefined",
                        "rank term must carry exactly the predicate's one expand layer",
                    ));
                }
            }
            insert_predicate(
                decoded,
                predicate.relation,
                predicate.ordinal,
                predicate.predicate,
            );
            Ok(())
        }
        QueryLang::Limit([source, k_id]) => {
            decode_body(expr, *source, decoded)?;
            decoded.limit = Some(decode_attr(expr, *k_id)?);
            Ok(())
        }
        QueryLang::QueryRoot([_, _, body]) => decode_body(expr, *body, decoded),
        found => Err(unexpected("query body", found)),
    }
}

fn decode_single_expand(
    expr: &RecExpr<QueryLang>,
    id: Id,
) -> GraphStoreResult<(RelationAttrs, usize, String, Vec<LayerId>, Direction)> {
    let QueryLang::Expand([from_id, layers_id, dir_id]) = &expr[id] else {
        return Err(unexpected("single expand", &expr[id]));
    };
    let from: ExpandAttrs = decode_attr(expr, *from_id)?;
    let layers = decode_layer_set(expr, *layers_id)?;
    let dir = decode_attr(expr, *dir_id)?;
    Ok((from.relation, from.ordinal, from.from, layers, dir))
}

fn insert_predicate(
    decoded: &mut DecodedBody,
    relation: RelationAttrs,
    ordinal: usize,
    predicate: Predicate,
) {
    let index = decoded
        .relations
        .iter()
        .position(|existing| existing.attrs.alias == relation.alias)
        .unwrap_or_else(|| {
            decoded.relations.push(DecodedRelation {
                attrs: relation.clone(),
                predicates: Vec::new(),
            });
            decoded.relations.len() - 1
        });
    let entry = &mut decoded.relations[index];
    if let Some((_, existing)) = entry
        .predicates
        .iter_mut()
        .find(|(existing_ordinal, _)| *existing_ordinal == ordinal)
    {
        if let (
            Predicate::Expand {
                from: existing_from,
                layers: existing_layers,
                dir: existing_dir,
                mode: PredicateMode::Filter,
            },
            Predicate::Expand {
                from,
                layers,
                dir,
                mode: PredicateMode::Filter,
            },
        ) = (existing, &predicate)
        {
            if existing_from == from && existing_dir == dir {
                existing_layers.extend(layers.iter().cloned());
                existing_layers.sort();
                existing_layers.dedup();
                return;
            }
        }
    }
    entry.predicates.push((ordinal, predicate));
}

fn ensure_relation(decoded: &mut DecodedBody, relation: RelationAttrs) {
    if !decoded
        .relations
        .iter()
        .any(|existing| existing.attrs.alias == relation.alias)
    {
        decoded.relations.push(DecodedRelation {
            attrs: relation,
            predicates: Vec::new(),
        });
    }
}

pub(crate) fn encode_layer_set(
    expr: &mut RecExpr<QueryLang>,
    layers: &[LayerId],
) -> GraphStoreResult<Id> {
    let mut rest = expr.add(QueryLang::LayersEmpty);
    for layer in layers.iter().rev() {
        let attr = push_attr(expr, layer)?;
        let layer = expr.add(QueryLang::Layer([attr]));
        rest = expr.add(QueryLang::Layers([layer, rest]));
    }
    Ok(rest)
}

pub(crate) fn decode_layer_set(
    expr: &RecExpr<QueryLang>,
    id: Id,
) -> GraphStoreResult<Vec<LayerId>> {
    let mut layers = Vec::new();
    let mut current = id;
    loop {
        match &expr[current] {
            QueryLang::LayersEmpty => break,
            QueryLang::Layers([layer_id, rest]) => {
                let QueryLang::Layer([attr_id]) = &expr[*layer_id] else {
                    return Err(unexpected("layer", &expr[*layer_id]));
                };
                layers.push(decode_attr(expr, *attr_id)?);
                current = *rest;
            }
            found => return Err(unexpected("layer set", found)),
        }
    }
    Ok(layers)
}

fn decode_rank_layer(expr: &RecExpr<QueryLang>, id: Id) -> GraphStoreResult<Option<LayerId>> {
    let QueryLang::Layer([attr]) = &expr[id] else {
        return Err(unexpected("rank layer", &expr[id]));
    };
    decode_attr(expr, *attr)
}

pub(crate) fn push_attr<T: Serialize>(
    expr: &mut RecExpr<QueryLang>,
    value: &T,
) -> GraphStoreResult<Id> {
    Ok(expr.add(QueryLang::Attr(EncodedAttr::from_json(value)?)))
}

pub(crate) fn decode_attr<T: DeserializeOwned>(
    expr: &RecExpr<QueryLang>,
    id: Id,
) -> GraphStoreResult<T> {
    match &expr[id] {
        QueryLang::Attr(value) => value.decode(),
        found => Err(unexpected("attribute leaf", found)),
    }
}

fn predicate_mode(predicate: &Predicate) -> PredicateMode {
    match predicate {
        Predicate::Knn { .. } => PredicateMode::Rank,
        Predicate::TextMatch { mode, .. } | Predicate::Expand { mode, .. } => *mode,
        _ => PredicateMode::Filter,
    }
}

fn predicate_k(predicate: &Predicate) -> Option<usize> {
    match predicate {
        Predicate::Knn { k, .. } => Some(*k),
        _ => None,
    }
}

fn unexpected(context: &'static str, found: &QueryLang) -> GraphStoreError {
    GraphStoreError::new(
        "invalid_query_saturate_term",
        format!("expected {context}, found {found}"),
    )
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode(input: &str) -> Result<Vec<u8>, String> {
    if input.len() % 2 != 0 {
        return Err("hex payload has odd length".to_string());
    }
    input
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?))
        .collect()
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hex byte {byte}")),
    }
}
