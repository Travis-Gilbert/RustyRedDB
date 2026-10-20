use std::collections::{BTreeMap, BTreeSet};

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use serde_json::Value;

use crate::access_method::{
    AccessMethod, ColumnId, Cost, ModalityResolver, NoModalityResolver, Predicate, PredicateMode,
    RankedRow, RegionRef, RelationId, RowId, ScalarValue,
};
use crate::cold_fragments::{is_epistemic_column, FragmentRangeStats, HotColumnGroup};
use crate::colt::{
    asof_nearest_preceding, candidate_algorithm, execute_binary_join, execute_mask_join,
    select_join_candidate, Emission, JOIN_CANDIDATE_COLT_BINARY, NATIVE_JOIN_ALGORITHM,
};
use crate::graph::capability::GraphRead;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::provenance_semiring::{ProvenanceSemiring, ProvenanceSemiringKind, SemiringAnnotation};
use crate::ranking::{apply_cascade, compute_term_match, QueryContext, RankCandidate, RankingRule};
use crate::relational::{RelationalLens, ResolvedRow};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QueryRelation {
    pub alias: String,
    pub relation: RelationId,
    #[serde(default)]
    pub predicates: Vec<Predicate>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinKind {
    #[default]
    Inner,
    Left,
    Semi,
    Anti,
    Asof,
}

impl JoinKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inner => "inner",
            Self::Left => "left",
            Self::Semi => "semi",
            Self::Anti => "anti",
            Self::Asof => "asof",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinPlanPoint {
    #[default]
    Binary,
    Generic,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinBacking {
    #[default]
    Colt,
    Mask,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JoinPredicate {
    pub left_alias: String,
    pub left_column: ColumnId,
    pub right_alias: String,
    pub right_column: ColumnId,
    #[serde(default)]
    pub kind: JoinKind,
}

impl Default for JoinPredicate {
    fn default() -> Self {
        Self {
            left_alias: String::new(),
            left_column: String::new(),
            right_alias: String::new(),
            right_column: String::new(),
            kind: JoinKind::Inner,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Projection {
    pub alias: String,
    pub column: ColumnId,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct QueryIr {
    pub relations: Vec<QueryRelation>,
    #[serde(default)]
    pub joins: Vec<JoinPredicate>,
    #[serde(default)]
    pub projection: Vec<Projection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default)]
    pub fusion: FusionPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semiring: Option<ProvenanceSemiringKind>,
    /// When set, the query is version-pinned (`?v=`). Non-pinnable layers in
    /// any Expand predicate refuse with `layer_not_pinnable`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_graph_version: Option<u64>,
    /// Free-join plan point. Binary is the native floor (current hash join).
    #[serde(default)]
    pub join_plan_point: JoinPlanPoint,
    /// Physical backing. Colt is the native floor; Mask is the GraphBLAS candidate.
    #[serde(default)]
    pub join_backing: JoinBacking,
    /// Selected steering candidate id. None means cold-start native floor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_candidate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_steering_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_observation_count: Option<u32>,
}

/// A foreign relation in a production query plan. The bound is checked before
/// the provider is invoked, so refusal cannot spend even one remote read.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FederationRelation {
    pub alias: String,
    pub block_urls: Vec<String>,
    pub join_column: ColumnId,
    pub max_foreign_reads: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FederationJoinQuery {
    pub local: QueryRelation,
    pub local_join_column: ColumnId,
    pub foreign: FederationRelation,
    #[serde(default)]
    pub projection: Vec<Projection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Adapter boundary implemented by a GraphStore-backed federation service.
pub trait FederatedRowProvider {
    fn resolve_foreign_row(&mut self, block_url: &str) -> GraphStoreResult<ResolvedRow>;
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FusionPolicy {
    Rrf {
        k: usize,
    },
    Weighted {
        weights: BTreeMap<String, f32>,
    },
    /// Ordered ranking-rule cascade (SPEC-RUSTYRED-RANKING-CASCADE): bucket-sort by each rule in
    /// turn, later rules only breaking ties left by earlier ones. The epistemic rules are first
    /// class and reorderable, so trust can rank ahead of relevance.
    Cascade {
        rules: Vec<RankingRule>,
    },
}

impl Default for FusionPolicy {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AccessPathTrace {
    pub relation: RelationId,
    pub alias: String,
    pub predicate: String,
    pub method: String,
    pub est_rows: f64,
    pub est_work: f64,
    pub returned_rows: usize,
    pub visited_rows: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct PlanTrace {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plan_stages: Vec<PlanStageTrace>,
    pub access_paths: Vec<AccessPathTrace>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub access_paths_considered: Vec<AccessPathTrace>,
    pub full_relation_scans: usize,
    pub bitmap_intersections: usize,
    pub used_roaring_bitmaps: bool,
    pub join_algorithm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_steering_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_observation_count: Option<u32>,
    #[serde(default)]
    pub intermediate_tuples: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub colt_build_counts: Vec<usize>,
    pub joined_rows: usize,
    pub candidate_set_size: usize,
    pub rankers: Vec<RankerTrace>,
    pub fusion: String,
    pub knn_strategy: Option<String>,
    pub knn_overfetch_rounds: usize,
    /// Cascade fusion only: the rule order actually applied, and per-ranked-row the bucket vector
    /// (one ordinal per rule), so the trace answers "ranked here because rule R put it in bucket N".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cascade_rule_order: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cascade_buckets: BTreeMap<String, Vec<u32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columnar_groups: Vec<ColumnarGroupTrace>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semiring: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semiring_graphblas: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub derived_annotations: BTreeMap<String, f64>,
    /// Bounded federation relation accounting. These fields live on the same
    /// trace as local access methods so a caller receives one budget receipt.
    #[serde(default)]
    pub federation_foreign_reads: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub federation_foreign_read_budget: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub federation_receipt_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ColumnarGroupTrace {
    pub relation: RelationId,
    pub alias: String,
    pub group: String,
    pub predicate: String,
    pub column: ColumnId,
    pub source: String,
    pub row_count: usize,
    pub block_size: usize,
    pub blocks_visited: usize,
    pub blocks_skipped: usize,
    pub rows_returned: usize,
    pub rowwise_property_bytes: usize,
    pub unencoded_columnar_bytes: usize,
    pub encoded_columnar_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlanStageTrace {
    pub stage: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_count: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RankerTrace {
    pub relation: RelationId,
    pub alias: String,
    pub predicate: String,
    pub method: String,
    pub contributed_rows: usize,
    pub score_source: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QueryResult {
    pub rows: Vec<QueryOutputRow>,
    pub trace: PlanTrace,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QueryOutputRow {
    #[serde(flatten)]
    pub values: BTreeMap<String, ScalarValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<SemiringAnnotation>,
}

impl std::ops::Deref for QueryOutputRow {
    type Target = BTreeMap<String, ScalarValue>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Execute a query with no live modality subsystems bound. `Knn`, `TextMatch { Rank }`, and
/// `Expand { Rank }` predicates therefore return no rows, and modality filters impose no
/// restriction beyond their residual checks. Scalar, time, and join behavior is unchanged. Use
/// [`execute_query_with_resolver`] to bind the live vector/text/spatial/graph indexes.
pub fn execute_query<G: GraphRead>(
    store: &RelationalLens<G>,
    query: QueryIr,
) -> GraphStoreResult<QueryResult> {
    execute_query_with_resolver(store, query, &NoModalityResolver)
}

/// Execute one local relation and one bounded foreign relation as a single
/// planner operation. The foreign side is materialized only after the complete
/// requested read set passes its bound.
pub fn execute_bounded_federation_join<G: GraphRead>(
    store: &RelationalLens<G>,
    query: FederationJoinQuery,
    provider: &mut dyn FederatedRowProvider,
) -> GraphStoreResult<QueryResult> {
    if query.local.alias.trim().is_empty() || query.foreign.alias.trim().is_empty() {
        return Err(GraphStoreError::new(
            "invalid_federation_relation_alias",
            "local and federation relation aliases are required",
        ));
    }
    if query.foreign.block_urls.len() > query.foreign.max_foreign_reads {
        return Err(GraphStoreError::new(
            "federation_join_budget_exceeded",
            format!(
                "{} foreign reads exceed budget {}",
                query.foreign.block_urls.len(),
                query.foreign.max_foreign_reads
            ),
        ));
    }

    let mut local = execute_query(
        store,
        QueryIr {
            relations: vec![query.local.clone()],
            limit: None,
            ..QueryIr::default()
        },
    )?;
    let mut foreign_rows = Vec::with_capacity(query.foreign.block_urls.len());
    for block_url in &query.foreign.block_urls {
        foreign_rows.push(provider.resolve_foreign_row(block_url)?);
    }

    let local_key = format!("{}.{}", query.local.alias, query.local_join_column);
    let mut rows = Vec::new();
    for local_row in &local.rows {
        let Some(local_value) = local_row.values.get(&local_key) else {
            continue;
        };
        for foreign_row in &foreign_rows {
            if foreign_row.value(&query.foreign.join_column) != Some(local_value) {
                continue;
            }
            let mut values = local_row.values.clone();
            values.extend(foreign_row.values.iter().map(|(column, value)| {
                (format!("{}.{}", query.foreign.alias, column), value.clone())
            }));
            if !query.projection.is_empty() {
                let projected = query
                    .projection
                    .iter()
                    .filter_map(|projection| {
                        let key = format!("{}.{}", projection.alias, projection.column);
                        values.get(&key).cloned().map(|value| (key, value))
                    })
                    .collect();
                values = projected;
            }
            rows.push(QueryOutputRow {
                values,
                score: local_row.score,
                provenance: local_row.provenance.clone(),
            });
            if query.limit.is_some_and(|limit| rows.len() >= limit) {
                break;
            }
        }
        if query.limit.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }

    let reads = foreign_rows.len();
    let receipt_hash = crate::stable_hash(serde_json::json!({
        "local_relation": query.local.relation,
        "local_alias": query.local.alias,
        "foreign_alias": query.foreign.alias,
        "block_urls": query.foreign.block_urls,
        "foreign_reads": reads,
        "foreign_read_budget": query.foreign.max_foreign_reads,
        "joined_rows": rows.len(),
    }))?;
    local.trace.join_algorithm = Some("bounded_federation_hash_join".to_string());
    local.trace.joined_rows = rows.len();
    local.trace.federation_foreign_reads = reads;
    local.trace.federation_foreign_read_budget = Some(query.foreign.max_foreign_reads);
    local.trace.federation_receipt_hash = Some(receipt_hash);
    push_plan_stage(
        &mut local.trace,
        "federation_relation",
        format!(
            "{} foreign read(s) within budget {}",
            reads, query.foreign.max_foreign_reads
        ),
        Some(rows.len()),
    );
    Ok(QueryResult {
        rows,
        trace: local.trace,
    })
}

/// Execute a query, resolving every modality predicate (vector kNN, BM25 relevance, graph
/// proximity, spatial/full-text/expand filters) by node id against the live subsystems behind
/// `resolver`. No modality data is read from the relational store; only scalar and structured
/// columns live there, used for residual exact checks.
pub fn execute_query_with_resolver<G: GraphRead>(
    store: &RelationalLens<G>,
    query: QueryIr,
    resolver: &dyn ModalityResolver,
) -> GraphStoreResult<QueryResult> {
    validate_query_ir(&query)?;
    if query.pinned_graph_version.is_some() {
        let layers = expand_layers_in_query(&query);
        if !layers.is_empty() {
            crate::layer::LayerRegistry::with_native_defaults().check_pinnable(&layers)?;
        }
    }
    store.ensure_access_methods_current()?;

    let semiring = query.semiring.map(ProvenanceSemiring::new);
    let mut trace = PlanTrace {
        fusion: fusion_name(&query.fusion).to_string(),
        semiring: query.semiring.map(|kind| kind.name().to_string()),
        semiring_graphblas: query
            .semiring
            .and_then(ProvenanceSemiringKind::graphblas_semiring)
            .map(str::to_string),
        ..PlanTrace::default()
    };
    push_plan_stage(
        &mut trace,
        "logical_plan",
        format!(
            "{} relation(s), {} join(s), {} projection(s)",
            query.relations.len(),
            query.joins.len(),
            query.projection.len()
        ),
        None,
    );
    let mut per_alias: BTreeMap<String, Vec<ScoredResolvedRow>> = BTreeMap::new();
    for relation_query in &query.relations {
        push_plan_stage(
            &mut trace,
            "table_provider_scan",
            format!("{} as {}", relation_query.relation, relation_query.alias),
            None,
        );
        let relation = store.relation(&relation_query.relation).ok_or_else(|| {
            GraphStoreError::new(
                "unknown_relation",
                format!("unknown relation {}", relation_query.relation),
            )
        })?;
        let (filter_predicates, rank_predicates) = partition_predicates(&relation_query.predicates);
        let candidate_row_ids = candidate_rows(
            store,
            relation_query,
            &filter_predicates,
            rank_predicates.is_empty(),
            resolver,
            &mut trace,
        )?;
        let residual_row_ids = candidate_row_ids
            .into_iter()
            .filter_map(|id| relation.get(&id))
            .filter(|row| {
                filter_predicates
                    .iter()
                    .all(|predicate| row_matches(row, predicate))
            })
            .map(|row| row.id)
            .collect::<Vec<_>>();
        trace.candidate_set_size += residual_row_ids.len();
        // The rank phase scores candidate node ids (not ordinals): when filters ran, restrict to
        // the residual set; with no filters, rank over the whole subsystem.
        let candidate_ids: Option<BTreeSet<RowId>> = if filter_predicates.is_empty() {
            None
        } else {
            Some(residual_row_ids.iter().cloned().collect())
        };
        let scores = score_rankers(
            store,
            relation_query,
            &rank_predicates,
            candidate_ids.as_ref(),
            residual_row_ids.len(),
            effective_rank_limit(&query, &rank_predicates, residual_row_ids.len()),
            &query.fusion,
            resolver,
            &mut trace,
        )?;
        let mut rows = residual_row_ids
            .into_iter()
            .filter_map(|id| {
                let row = relation.get(&id)?;
                let score = scores.as_ref().and_then(|scores| scores.get(&id).copied());
                if scores.is_some() && score.is_none() {
                    return None;
                }
                let annotation = semiring.map(|semiring| seed_annotation(&row, semiring));
                Some(ScoredResolvedRow {
                    row,
                    score,
                    annotation,
                })
            })
            .collect::<Vec<_>>();
        if scores.is_some() {
            rows.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.row.id.cmp(&right.row.id))
            });
        }
        push_plan_stage(
            &mut trace,
            "relation_candidates",
            relation_query.alias.clone(),
            Some(rows.len()),
        );
        per_alias.insert(relation_query.alias.clone(), rows);
    }

    let mut joined = execute_joins(&per_alias, &query.joins, &query, &mut trace)?;
    let join_detail = trace
        .join_algorithm
        .clone()
        .unwrap_or_else(|| "single_relation".to_string());
    push_plan_stage(&mut trace, "join", join_detail, Some(joined.len()));
    let mut out = Vec::new();
    if let Some(semiring) = semiring {
        let mut merged = BTreeMap::<String, QueryOutputRow>::new();
        for rowset in joined {
            let values = project_rowset(&rowset, &query.projection);
            let annotation_value = rowset_annotation(&rowset, semiring);
            let score =
                rowset_score(&rowset).or_else(|| semiring_score(semiring.kind, annotation_value));
            let key = projected_values_key(&values);
            match merged.get_mut(&key) {
                Some(existing) => {
                    let prior = existing
                        .provenance
                        .as_ref()
                        .map(|annotation| annotation.value)
                        .unwrap_or_else(|| semiring.zero());
                    let combined = semiring.sum(prior, annotation_value);
                    existing.provenance = Some(SemiringAnnotation {
                        semiring: semiring.kind,
                        value: combined,
                    });
                    existing.score = semiring_score(semiring.kind, combined);
                }
                None => {
                    merged.insert(
                        key,
                        QueryOutputRow {
                            score,
                            values,
                            provenance: Some(SemiringAnnotation {
                                semiring: semiring.kind,
                                value: annotation_value,
                            }),
                        },
                    );
                }
            }
        }
        out = merged.into_values().collect();
        sort_annotated_rows(&mut out, semiring.kind);
        if let Some(limit) = query.limit {
            out.truncate(limit);
        }
        for (index, row) in out.iter().enumerate() {
            if let Some(annotation) = &row.provenance {
                trace
                    .derived_annotations
                    .insert(format!("row:{index}"), annotation.value);
            }
        }
    } else {
        let ranked = joined.iter().any(|rowset| rowset_score(rowset).is_some());
        if ranked {
            joined.sort_by(|left, right| {
                rowset_score(right)
                    .partial_cmp(&rowset_score(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| rowset_ids(left).cmp(&rowset_ids(right)))
            });
        }
        for rowset in joined {
            out.push(QueryOutputRow {
                score: rowset_score(&rowset),
                values: project_rowset(&rowset, &query.projection),
                provenance: None,
            });
            if query.limit.is_some_and(|limit| out.len() >= limit) {
                break;
            }
        }
    }
    trace.joined_rows = out.len();
    push_plan_stage(
        &mut trace,
        "projection",
        "project_output_rows",
        Some(out.len()),
    );
    Ok(QueryResult { rows: out, trace })
}

fn expand_layers_in_query(query: &QueryIr) -> Vec<crate::layer::LayerId> {
    let mut layers = Vec::new();
    let mut seen = BTreeSet::new();
    for relation in &query.relations {
        for predicate in &relation.predicates {
            if let Predicate::Expand {
                layers: expand_layers,
                ..
            } = predicate
            {
                for layer in expand_layers {
                    if seen.insert(layer.clone()) {
                        layers.push(layer.clone());
                    }
                }
            }
        }
    }
    layers
}

pub fn validate_query_ir(query: &QueryIr) -> GraphStoreResult<()> {
    if query.relations.is_empty() {
        return Err(GraphStoreError::new(
            "empty_relational_query",
            "query requires at least one relation",
        ));
    }
    let mut aliases = BTreeSet::new();
    for relation in &query.relations {
        if relation.alias.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_query_relation_alias",
                "relation alias is required",
            ));
        }
        if relation.relation.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_query_relation",
                format!("relation {} has an empty relation id", relation.alias),
            ));
        }
        if !aliases.insert(relation.alias.clone()) {
            return Err(GraphStoreError::new(
                "duplicate_query_relation_alias",
                format!(
                    "relation alias {} is defined more than once",
                    relation.alias
                ),
            ));
        }
    }
    for join in &query.joins {
        if !aliases.contains(&join.left_alias) {
            return Err(GraphStoreError::new(
                "unknown_query_join_alias",
                format!("join references unknown left alias {}", join.left_alias),
            ));
        }
        if !aliases.contains(&join.right_alias) {
            return Err(GraphStoreError::new(
                "unknown_query_join_alias",
                format!("join references unknown right alias {}", join.right_alias),
            ));
        }
    }
    for projection in &query.projection {
        if !aliases.contains(&projection.alias) {
            return Err(GraphStoreError::new(
                "unknown_query_projection_alias",
                format!("projection references unknown alias {}", projection.alias),
            ));
        }
    }
    if let FusionPolicy::Weighted { weights } = &query.fusion {
        let ranker_methods = query
            .relations
            .iter()
            .flat_map(|relation| relation.predicates.iter())
            .filter_map(ranker_method_name)
            .collect::<BTreeSet<_>>();
        for method in weights.keys() {
            if !ranker_methods.contains(method.as_str()) {
                let available = ranker_methods
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(GraphStoreError::new(
                    "unknown_weighted_fusion_method",
                    format!(
                        "weighted fusion references unknown method {method}; available methods: {available}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn push_plan_stage(
    trace: &mut PlanTrace,
    stage: impl Into<String>,
    detail: impl Into<String>,
    row_count: Option<usize>,
) {
    trace.plan_stages.push(PlanStageTrace {
        stage: stage.into(),
        detail: detail.into(),
        row_count,
    });
}

fn candidate_rows<G: GraphRead>(
    store: &RelationalLens<G>,
    relation_query: &QueryRelation,
    filter_predicates: &[&Predicate],
    count_unfiltered_scan: bool,
    resolver: &dyn ModalityResolver,
    trace: &mut PlanTrace,
) -> GraphStoreResult<Vec<RowId>> {
    let relation = store.relation(&relation_query.relation).ok_or_else(|| {
        GraphStoreError::new(
            "unknown_relation",
            format!("unknown relation {}", relation_query.relation),
        )
    })?;
    if filter_predicates.is_empty() {
        if count_unfiltered_scan {
            trace.full_relation_scans += 1;
        }
        return Ok(relation.row_ids());
    }
    let row_ord = relation
        .row_ids()
        .into_iter()
        .enumerate()
        .map(|(index, id)| (id, index as u32))
        .collect::<BTreeMap<_, _>>();
    let ord_row = row_ord
        .iter()
        .map(|(row, ord)| (*ord, row.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut bitmaps = Vec::new();
    let mut residual_only = false;
    for predicate in filter_predicates {
        // Modality filters (geo / full-text / expand) resolve over-approximate candidate ids from
        // the live subsystem by id; the residual exact check runs later in `row_matches`.
        if let Some(result) =
            modality_filter_candidates(relation_query, predicate, resolver, trace)?
        {
            match result {
                ModalityFilterResult::Ids(ids) => {
                    bitmaps.push(row_ids_to_bitmap(ids.iter(), &row_ord));
                }
                ModalityFilterResult::ResidualOnly => residual_only = true,
            }
            continue;
        }
        // Scalar / time predicate: choose the cheapest equivalent access method.
        let mut best: Option<(&dyn AccessMethod, Cost)> = None;
        for method in store.access_methods().methods() {
            if !method.supports(&relation_query.relation, predicate) {
                continue;
            }
            let Some(cost) = method.cost(&relation_query.relation, predicate) else {
                continue;
            };
            trace.access_paths_considered.push(AccessPathTrace {
                relation: relation_query.relation.clone(),
                alias: relation_query.alias.clone(),
                predicate: predicate.kind_name().to_string(),
                method: method.name().to_string(),
                est_rows: cost.est_rows,
                est_work: cost.est_work,
                returned_rows: 0,
                visited_rows: 0,
            });
            if best
                .map(|(_, best_cost)| cost.rank_key() < best_cost.rank_key())
                .unwrap_or(true)
            {
                best = Some((method, cost));
            }
        }
        let Some((method, cost)) = best else {
            trace.full_relation_scans += 1;
            let scan = AccessPathTrace {
                relation: relation_query.relation.clone(),
                alias: relation_query.alias.clone(),
                predicate: predicate.kind_name().to_string(),
                method: "full_scan".to_string(),
                est_rows: relation.row_ids().len() as f64,
                est_work: relation.row_ids().len() as f64,
                returned_rows: relation.row_ids().len(),
                visited_rows: relation.row_ids().len(),
            };
            trace.access_paths_considered.push(scan.clone());
            trace.access_paths.push(scan);
            let relation_row_ids = relation.row_ids();
            bitmaps.push(row_ids_to_bitmap(relation_row_ids.iter(), &row_ord));
            continue;
        };
        let stream = method.scan(&relation_query.relation, predicate)?;
        trace.access_paths.push(AccessPathTrace {
            relation: relation_query.relation.clone(),
            alias: relation_query.alias.clone(),
            predicate: predicate.kind_name().to_string(),
            method: method.name().to_string(),
            est_rows: cost.est_rows,
            est_work: cost.est_work,
            returned_rows: stream.rows.len(),
            visited_rows: stream.visited,
        });
        push_epistemic_columnar_filter_trace(store, relation_query, predicate, trace)?;
        bitmaps.push(row_ids_to_bitmap(stream.rows.iter(), &row_ord));
    }

    if bitmaps.is_empty() {
        // Only residual-only filters ran (e.g. a geo predicate with no addressable spatial index):
        // scan the relation and let the residual exact checks in `row_matches` decide membership.
        if residual_only {
            trace.full_relation_scans += 1;
        }
        return Ok(relation.row_ids());
    }

    let mut iter = bitmaps.into_iter();
    let mut bitmap = iter.next().unwrap_or_default();
    for next in iter {
        bitmap &= next;
        trace.bitmap_intersections += 1;
        trace.used_roaring_bitmaps = true;
    }
    Ok(bitmap
        .iter()
        .filter_map(|ord| ord_row.get(&ord).cloned())
        .collect())
}

/// Outcome of resolving a modality filter predicate against the live subsystem.
enum ModalityFilterResult {
    /// Over-approximate candidate ids to intersect into the candidate set.
    Ids(Vec<RowId>),
    /// No addressable index for this predicate; rely on the residual exact check over a full scan.
    ResidualOnly,
}

/// Resolve a geo / full-text / expand *filter* predicate via the live subsystem. Returns `None`
/// for non-modality (scalar/time) predicates so the caller falls through to the access methods.
fn modality_filter_candidates(
    relation_query: &QueryRelation,
    predicate: &Predicate,
    resolver: &dyn ModalityResolver,
    trace: &mut PlanTrace,
) -> GraphStoreResult<Option<ModalityFilterResult>> {
    let (ids, method) = match predicate {
        Predicate::TextMatch {
            column,
            query,
            mode: PredicateMode::Filter,
        } => match resolver.text_contains(&relation_query.relation, column, query)? {
            Some(ids) => (ids, "text"),
            None => {
                push_modality_access_path(
                    relation_query,
                    predicate,
                    "text_residual_scan",
                    0,
                    trace,
                );
                return Ok(Some(ModalityFilterResult::ResidualOnly));
            }
        },
        Predicate::Expand {
            from,
            layers,
            dir,
            mode: PredicateMode::Filter,
        } => match resolver.expand_reachable(from, layers, dir.clone())? {
            Some(tagged) => {
                let mut ids = Vec::new();
                let mut seen = BTreeSet::new();
                for (row_id, _layer) in &tagged {
                    if seen.insert(row_id.clone()) {
                        ids.push(row_id.clone());
                    }
                }
                // Provenance is available on tagged pairs; filter intersection
                // still uses the deduped id set. Callers that need provenance
                // use expand_reachable / RowIdStream::with_layers directly.
                let _ = tagged;
                (ids, "expand")
            }
            None => {
                return Err(GraphStoreError::new(
                    "unsupported_expand_filter",
                    "expand filter requires a bound graph modality resolver",
                ));
            }
        },
        Predicate::GeoWithin {
            column,
            region,
            lat_property,
            lon_property,
            label,
        } => match resolver.geo_overapprox(
            &relation_query.relation,
            column,
            lat_property.as_deref(),
            lon_property.as_deref(),
            label.as_deref(),
            region,
        )? {
            Some(ids) => (ids, "geo_index"),
            None => {
                push_modality_access_path(relation_query, predicate, "geo_residual_scan", 0, trace);
                return Ok(Some(ModalityFilterResult::ResidualOnly));
            }
        },
        _ => return Ok(None),
    };
    let returned = ids.len();
    push_modality_access_path(relation_query, predicate, method, returned, trace);
    Ok(Some(ModalityFilterResult::Ids(ids)))
}

fn push_modality_access_path(
    relation_query: &QueryRelation,
    predicate: &Predicate,
    method: &str,
    returned: usize,
    trace: &mut PlanTrace,
) {
    trace.access_paths.push(AccessPathTrace {
        relation: relation_query.relation.clone(),
        alias: relation_query.alias.clone(),
        predicate: predicate.kind_name().to_string(),
        method: method.to_string(),
        est_rows: -1.0,
        est_work: -1.0,
        returned_rows: returned,
        visited_rows: returned,
    });
}

fn row_ids_to_bitmap<'a>(
    rows: impl Iterator<Item = &'a RowId>,
    row_ord: &BTreeMap<RowId, u32>,
) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for row in rows {
        if let Some(ord) = row_ord.get(row) {
            bitmap.insert(*ord);
        }
    }
    bitmap
}

fn push_epistemic_columnar_filter_trace<G: GraphRead>(
    store: &RelationalLens<G>,
    relation_query: &QueryRelation,
    predicate: &Predicate,
    trace: &mut PlanTrace,
) -> GraphStoreResult<()> {
    let Some(column) = predicate.column() else {
        return Ok(());
    };
    if !is_epistemic_column(column) {
        return Ok(());
    }
    let Some(group) = store.hot_column_group(&relation_query.relation) else {
        return Ok(());
    };
    let stats = match predicate {
        Predicate::Range { column, lo, hi } => {
            group.range_query(column, lo.clone(), hi.clone())?.stats
        }
        Predicate::Equals { column, value } => columnar_equals_stats(&group, column, value),
        _ => FragmentRangeStats {
            fragments_visited: group.fragments.len(),
            fragments_skipped: 0,
            rows_returned: 0,
        },
    };
    trace.columnar_groups.push(columnar_trace(
        relation_query,
        &group,
        predicate.kind_name(),
        column,
        "hot_epistemic_column_group",
        stats,
    ));
    Ok(())
}

fn columnar_equals_stats(
    group: &HotColumnGroup,
    column: &str,
    value: &ScalarValue,
) -> FragmentRangeStats {
    let rows_returned = group
        .fragments
        .iter()
        .flat_map(|fragment| fragment.row_ids.iter())
        .filter(|row_id| group.value(row_id, column) == Some(value))
        .count();
    FragmentRangeStats {
        fragments_visited: group.fragments.len(),
        fragments_skipped: 0,
        rows_returned,
    }
}

fn columnar_trace(
    relation_query: &QueryRelation,
    group: &HotColumnGroup,
    predicate: &str,
    column: &str,
    source: &str,
    stats: FragmentRangeStats,
) -> ColumnarGroupTrace {
    ColumnarGroupTrace {
        relation: relation_query.relation.clone(),
        alias: relation_query.alias.clone(),
        group: group.group_id.clone(),
        predicate: predicate.to_string(),
        column: column.to_string(),
        source: source.to_string(),
        row_count: group.row_count,
        block_size: group.block_size,
        blocks_visited: stats.fragments_visited,
        blocks_skipped: stats.fragments_skipped,
        rows_returned: stats.rows_returned,
        rowwise_property_bytes: group.rowwise_property_bytes,
        unencoded_columnar_bytes: group.unencoded_columnar_bytes,
        encoded_columnar_bytes: group.encoded_columnar_bytes,
    }
}

#[derive(Clone, Debug)]
struct ScoredResolvedRow {
    row: ResolvedRow,
    score: Option<f32>,
    annotation: Option<f64>,
}

fn partition_predicates(predicates: &[Predicate]) -> (Vec<&Predicate>, Vec<&Predicate>) {
    predicates
        .iter()
        .partition(|predicate| !predicate_is_ranker(predicate))
}

fn predicate_is_ranker(predicate: &Predicate) -> bool {
    ranker_method_name(predicate).is_some()
}

fn ranker_method_name(predicate: &Predicate) -> Option<&'static str> {
    match predicate {
        Predicate::Knn { .. } => Some("vector"),
        Predicate::TextMatch {
            mode: PredicateMode::Rank,
            ..
        } => Some("text_rank"),
        Predicate::Expand {
            mode: PredicateMode::Rank,
            ..
        } => Some("expand_ppr"),
        _ => None,
    }
}

fn effective_rank_limit(
    query: &QueryIr,
    rank_predicates: &[&Predicate],
    candidate_count: usize,
) -> usize {
    query
        .limit
        .or_else(|| {
            rank_predicates
                .iter()
                .filter_map(|predicate| match predicate {
                    Predicate::Knn { k, .. } => Some(*k),
                    _ => None,
                })
                .min()
        })
        .unwrap_or(candidate_count)
        .min(candidate_count)
}

#[allow(clippy::too_many_arguments)]
fn score_rankers<G: GraphRead>(
    store: &RelationalLens<G>,
    relation_query: &QueryRelation,
    rank_predicates: &[&Predicate],
    candidates: Option<&BTreeSet<RowId>>,
    candidate_count: usize,
    k: usize,
    fusion: &FusionPolicy,
    resolver: &dyn ModalityResolver,
    trace: &mut PlanTrace,
) -> GraphStoreResult<Option<BTreeMap<RowId, f32>>> {
    if rank_predicates.is_empty() {
        return Ok(None);
    }
    let mut lists: Vec<(String, Vec<RankedRow>)> = Vec::new();
    for predicate in rank_predicates {
        let mut best: Option<(&dyn crate::access_method::RankingAccessMethod, Cost)> = None;
        for method in store.ranking_methods().methods() {
            if !method.supports(&relation_query.relation, predicate) {
                continue;
            }
            let Some(cost) =
                method.cost(&relation_query.relation, predicate, Some(candidate_count))
            else {
                continue;
            };
            trace.access_paths_considered.push(AccessPathTrace {
                relation: relation_query.relation.clone(),
                alias: relation_query.alias.clone(),
                predicate: predicate.kind_name().to_string(),
                method: ranker_access_method(method.name()),
                est_rows: cost.est_rows,
                est_work: cost.est_work,
                returned_rows: 0,
                visited_rows: 0,
            });
            if best
                .map(|(_, best_cost)| cost.rank_key() < best_cost.rank_key())
                .unwrap_or(true)
            {
                best = Some((method, cost));
            }
        }
        let Some((method, cost)) = best else {
            return Err(GraphStoreError::new(
                "unsupported_ranker",
                format!(
                    "no ranking access method supports {} on relation {}",
                    predicate.kind_name(),
                    relation_query.relation
                ),
            ));
        };
        let rank_k = rank_k_for_predicate(predicate, k, candidate_count);
        let outcome = method.rank(
            &relation_query.relation,
            predicate,
            candidates,
            rank_k,
            resolver,
        )?;
        // The execution strategy and overfetch round count are reported by the resolver: the trace
        // records what actually ran rather than a fixed label.
        if let Some(strategy) = outcome.strategy {
            trace.knn_strategy = Some(strategy);
            trace.knn_overfetch_rounds = trace.knn_overfetch_rounds.max(outcome.overfetch_rounds);
        }
        let method_name = method.name().to_string();
        let access_method = ranker_access_method(&method_name);
        trace.access_paths.push(AccessPathTrace {
            relation: relation_query.relation.clone(),
            alias: relation_query.alias.clone(),
            predicate: predicate.kind_name().to_string(),
            method: access_method,
            est_rows: cost.est_rows,
            est_work: cost.est_work,
            returned_rows: outcome.rows.len(),
            visited_rows: outcome.rows.len(),
        });
        trace.rankers.push(RankerTrace {
            relation: relation_query.relation.clone(),
            alias: relation_query.alias.clone(),
            predicate: predicate.kind_name().to_string(),
            method: method_name.clone(),
            contributed_rows: outcome.rows.len(),
            score_source: score_source(method_name.as_str()),
        });
        lists.push((method_name, outcome.rows));
    }
    let scores = match fusion {
        FusionPolicy::Cascade { rules } => {
            cascade_scores(store, relation_query, rules, rank_predicates, &lists, trace)?
        }
        _ => fuse_rankings(lists, fusion),
    };
    Ok(Some(scores))
}

/// Assemble a [`RankCandidate`] per ranked row from the per-ranker score lists plus the row's own
/// epistemic columns, run [`apply_cascade`], and emit a synthetic descending score so the execute
/// loop's existing sort-by-score reproduces the cascade order (and excluded rows drop out). The
/// rule order and per-row bucket vectors are recorded into the trace.
///
/// Epistemic standing is read as scalar/structured columns on the ranked relation
/// (`acceptance_status`, `epistemic_weight`, `source_reliability` or `justification_prior`,
/// `valid_from_ms`, `superseded`); a multi-relation join that projects those columns onto the
/// ranked rows is the caller's job, exactly as the spec's "supplies the signal through the existing
/// join" intends. `term_match` is computed only when the ranked relation denormalizes the text
/// column (the lexical rules are primarily the web path's; for the structured path the text lives
/// in the resolver, not the row).
fn cascade_scores<G: GraphRead>(
    store: &RelationalLens<G>,
    relation_query: &QueryRelation,
    rules: &[RankingRule],
    rank_predicates: &[&Predicate],
    lists: &[(String, Vec<RankedRow>)],
    trace: &mut PlanTrace,
) -> GraphStoreResult<BTreeMap<RowId, f32>> {
    let by_method = |name: &str| -> BTreeMap<&str, f32> {
        lists
            .iter()
            .filter(|(method, _)| method == name)
            .flat_map(|(_, rows)| rows.iter().map(|row| (row.row_id.as_str(), row.score)))
            .collect()
    };
    let vector = by_method("vector");
    let text = by_method("text_rank");
    let expand = by_method("expand_ppr");

    let mut candidate_ids = BTreeSet::new();
    for (_, rows) in lists {
        candidate_ids.extend(rows.iter().map(|row| row.row_id.clone()));
    }
    let column_group = store.hot_column_group(&relation_query.relation);
    if let Some(group) = &column_group {
        trace.columnar_groups.push(columnar_trace(
            relation_query,
            group,
            "cascade_epistemic_read",
            "epistemic_columns",
            "hot_epistemic_column_group",
            FragmentRangeStats {
                fragments_visited: group.fragments.len(),
                fragments_skipped: 0,
                rows_returned: candidate_ids.len(),
            },
        ));
    }

    let query = cascade_query_context(rank_predicates);
    let text_column = rank_predicates
        .iter()
        .find_map(|predicate| match predicate {
            Predicate::TextMatch {
                column,
                mode: PredicateMode::Rank,
                ..
            } => Some(column.as_str()),
            _ => None,
        });

    let relation = store.relation(&relation_query.relation);
    let candidates = candidate_ids
        .into_iter()
        .map(|id| {
            let mut candidate = RankCandidate::new(id.clone());
            candidate.vector_score = vector.get(id.as_str()).copied();
            candidate.bm25_score = text.get(id.as_str()).copied();
            candidate.graph_hops = expand.get(id.as_str()).map(|score| score_to_hops(*score));
            if let Some(row) = relation.as_ref().and_then(|relation| relation.get(&id)) {
                candidate.acceptance_status =
                    group_str_col(column_group.as_ref(), &id, "acceptance_status")
                        .or_else(|| str_col(&row, "acceptance_status"))
                        .unwrap_or_default();
                candidate.epistemic_weight =
                    group_f64_col(column_group.as_ref(), &id, "epistemic_weight")
                        .or_else(|| f64_col(&row, "epistemic_weight"))
                        .map(|v| v as f32)
                        .unwrap_or(1.0);
                candidate.source_reliability =
                    group_f64_col(column_group.as_ref(), &id, "source_reliability")
                        .or_else(|| {
                            group_f64_col(column_group.as_ref(), &id, "justification_prior")
                        })
                        .or_else(|| f64_col(&row, "source_reliability"))
                        .or_else(|| f64_col(&row, "justification_prior"))
                        .map(|v| v as f32);
                candidate.valid_from_ms =
                    group_i64_col(column_group.as_ref(), &id, "valid_from_ms")
                        .or_else(|| i64_col(&row, "valid_from_ms"));
                candidate.superseded = group_bool_col(column_group.as_ref(), &id, "superseded")
                    .or_else(|| bool_col(&row, "superseded"))
                    .unwrap_or(false);
                if !query.terms.is_empty() {
                    if let Some(body) = text_column.and_then(|column| str_col(&row, column)) {
                        candidate.term_match = Some(compute_term_match(&body, &query));
                    }
                }
            }
            candidate
        })
        .collect::<Vec<_>>();

    let outcome = apply_cascade(candidates, rules, &query);
    trace.cascade_rule_order = outcome.rule_order;
    let n = outcome.ranked.len();
    let mut scores = BTreeMap::new();
    for (index, ranked) in outcome.ranked.into_iter().enumerate() {
        scores.insert(ranked.row_id.clone(), (n - index) as f32);
        trace.cascade_buckets.insert(ranked.row_id, ranked.buckets);
    }
    Ok(scores)
}

/// Build the cascade query context from the rank predicates: terms from the `TextMatch { Rank }`
/// query, default typo tolerance and admission gate. The structured path carries no recency clock
/// in `QueryIr`, so recency degrades to superseded-last; an absolute recency reference / `as_of`
/// cutoff on this path is a named follow-up.
fn cascade_query_context(rank_predicates: &[&Predicate]) -> QueryContext {
    let terms = rank_predicates
        .iter()
        .find_map(|predicate| match predicate {
            Predicate::TextMatch {
                query,
                mode: PredicateMode::Rank,
                ..
            } => Some(query.as_str()),
            _ => None,
        })
        .map(QueryContext::from_query)
        .unwrap_or_default();
    terms
}

/// Convert an `expand_ppr` proximity score (`1 / (hops + 1)`) back to a hop count for the
/// `GraphProximity` rule. Saturates for non-positive scores; the rule caps the band.
fn score_to_hops(score: f32) -> u32 {
    if score <= 0.0 {
        return u32::MAX;
    }
    ((1.0 / score) - 1.0).round().max(0.0) as u32
}

fn group_str_col(group: Option<&HotColumnGroup>, row_id: &str, key: &str) -> Option<String> {
    group
        .and_then(|group| group.value(row_id, key))
        .and_then(ScalarValue::as_str)
        .map(str::to_string)
}

fn group_f64_col(group: Option<&HotColumnGroup>, row_id: &str, key: &str) -> Option<f64> {
    group
        .and_then(|group| group.value(row_id, key))
        .and_then(ScalarValue::as_f64)
}

fn group_i64_col(group: Option<&HotColumnGroup>, row_id: &str, key: &str) -> Option<i64> {
    group
        .and_then(|group| group.value(row_id, key))
        .and_then(ScalarValue::as_i64)
}

fn group_bool_col(group: Option<&HotColumnGroup>, row_id: &str, key: &str) -> Option<bool> {
    match group.and_then(|group| group.value(row_id, key)) {
        Some(ScalarValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn str_col(row: &ResolvedRow, key: &str) -> Option<String> {
    row.values
        .get(key)
        .and_then(ScalarValue::as_str)
        .map(str::to_string)
        .or_else(|| {
            row.properties
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn f64_col(row: &ResolvedRow, key: &str) -> Option<f64> {
    row.values
        .get(key)
        .and_then(ScalarValue::as_f64)
        .or_else(|| row.properties.get(key).and_then(Value::as_f64))
}

fn i64_col(row: &ResolvedRow, key: &str) -> Option<i64> {
    row.values
        .get(key)
        .and_then(ScalarValue::as_i64)
        .or_else(|| row.properties.get(key).and_then(Value::as_i64))
}

fn bool_col(row: &ResolvedRow, key: &str) -> Option<bool> {
    match row.values.get(key) {
        Some(ScalarValue::Bool(value)) => Some(*value),
        _ => row.properties.get(key).and_then(Value::as_bool),
    }
}

fn rank_k_for_predicate(predicate: &Predicate, query_k: usize, candidate_count: usize) -> usize {
    match predicate {
        Predicate::Knn { k, .. } => (*k).min(candidate_count),
        _ => query_k.min(candidate_count),
    }
}

fn fuse_rankings(
    lists: Vec<(String, Vec<RankedRow>)>,
    fusion: &FusionPolicy,
) -> BTreeMap<RowId, f32> {
    if lists.len() == 1 {
        return lists
            .into_iter()
            .next()
            .unwrap()
            .1
            .into_iter()
            .map(|row| (row.row_id, row.score))
            .collect();
    }
    let mut scores = BTreeMap::new();
    match fusion {
        FusionPolicy::Rrf { k } => {
            let k = *k as f32;
            for (_, rows) in lists {
                for (index, row) in rows.into_iter().enumerate() {
                    *scores.entry(row.row_id).or_insert(0.0) += 1.0 / (k + index as f32 + 1.0);
                }
            }
        }
        FusionPolicy::Weighted { weights } => {
            for (method, rows) in lists {
                let weight = weights.get(&method).copied().unwrap_or(1.0);
                for row in rows {
                    *scores.entry(row.row_id).or_insert(0.0) += row.score * weight;
                }
            }
        }
        // Cascade is diverted in `score_rankers` before fusion (it needs the rows + query context),
        // so this arm is unreachable in practice; fall back to RRF rather than panic.
        FusionPolicy::Cascade { .. } => {
            for (_, rows) in lists {
                for (index, row) in rows.into_iter().enumerate() {
                    *scores.entry(row.row_id).or_insert(0.0) += 1.0 / (60.0 + index as f32 + 1.0);
                }
            }
        }
    }
    scores
}

fn fusion_name(fusion: &FusionPolicy) -> &'static str {
    match fusion {
        FusionPolicy::Rrf { .. } => "rrf",
        FusionPolicy::Weighted { .. } => "weighted",
        FusionPolicy::Cascade { .. } => "cascade",
    }
}

fn score_source(method: &str) -> String {
    match method {
        "vector" => "cosine_similarity",
        "text_rank" => "bm25",
        "expand_ppr" => "hop_distance",
        other => other,
    }
    .to_string()
}

fn ranker_access_method(name: &str) -> String {
    match name {
        "text_rank" => "text".to_string(),
        other => other.to_string(),
    }
}

fn execute_joins(
    per_alias: &BTreeMap<String, Vec<ScoredResolvedRow>>,
    joins: &[JoinPredicate],
    query: &QueryIr,
    trace: &mut PlanTrace,
) -> GraphStoreResult<Vec<BTreeMap<String, ScoredResolvedRow>>> {
    if joins.is_empty() {
        let Some((alias, rows)) = per_alias.iter().next() else {
            return Ok(Vec::new());
        };
        return Ok(rows
            .iter()
            .cloned()
            .map(|row| BTreeMap::from([(alias.clone(), row)]))
            .collect());
    }

    let candidate = select_join_candidate(
        joins,
        query.join_candidate.as_deref(),
        query.join_plan_point,
        query.join_backing,
    );
    let cold_start = query.join_candidate.is_none() && query.join_backing == JoinBacking::Colt;
    trace.join_candidate_id = Some(candidate.to_string());
    trace.join_steering_reason = query.join_steering_reason.clone().or_else(|| {
        Some(if cold_start {
            "cold_start_native_floor".to_string()
        } else {
            "enumerated_candidate".to_string()
        })
    });
    trace.join_observation_count = query.join_observation_count.or(Some(0));
    trace.join_algorithm = Some(if cold_start && candidate == JOIN_CANDIDATE_COLT_BINARY {
        NATIVE_JOIN_ALGORITHM.to_string()
    } else {
        candidate_algorithm(candidate).to_string()
    });

    let first_join = &joins[0];
    let (mut result, mut stats) = apply_one_join(per_alias, first_join, candidate, None)?;
    for join in joins.iter().skip(1) {
        let (next, next_stats) = apply_one_join(per_alias, join, candidate, Some(result))?;
        result = next;
        stats.intermediate_tuples = stats
            .intermediate_tuples
            .saturating_add(next_stats.intermediate_tuples);
        stats.build_counts.extend(next_stats.build_counts);
    }
    trace.joined_rows = result.len();
    trace.intermediate_tuples = stats.intermediate_tuples;
    trace.colt_build_counts = stats.build_counts;
    Ok(result)
}

fn apply_one_join(
    per_alias: &BTreeMap<String, Vec<ScoredResolvedRow>>,
    join: &JoinPredicate,
    candidate: &str,
    prior: Option<Vec<BTreeMap<String, ScoredResolvedRow>>>,
) -> GraphStoreResult<(
    Vec<BTreeMap<String, ScoredResolvedRow>>,
    crate::colt::ColtStats,
)> {
    let right_rows = per_alias
        .get(&join.right_alias)
        .ok_or_else(|| GraphStoreError::new("unknown_join_alias", join.right_alias.clone()))?;
    let right_keys: Vec<Option<String>> = right_rows
        .iter()
        .map(|row| join_key(&row.row, &join.right_column))
        .collect();

    if let Some(prior) = prior {
        let left_keys: Vec<Option<String>> = prior
            .iter()
            .map(|rowset| {
                rowset
                    .get(&join.left_alias)
                    .and_then(|row| join_key(&row.row, &join.left_column))
            })
            .collect();
        let (pairs, stats) = join_pairs(&left_keys, &right_keys, join, candidate)?;
        return Ok((
            assemble_pairs(
                &pairs,
                |left_index| prior[left_index].clone(),
                &join.right_alias,
                right_rows,
                join.kind,
            ),
            stats,
        ));
    }

    let left_rows = per_alias
        .get(&join.left_alias)
        .ok_or_else(|| GraphStoreError::new("unknown_join_alias", join.left_alias.clone()))?;
    let left_keys: Vec<Option<String>> = left_rows
        .iter()
        .map(|row| join_key(&row.row, &join.left_column))
        .collect();
    let (pairs, stats) = join_pairs(&left_keys, &right_keys, join, candidate)?;
    Ok((
        assemble_pairs(
            &pairs,
            |left_index| BTreeMap::from([(join.left_alias.clone(), left_rows[left_index].clone())]),
            &join.right_alias,
            right_rows,
            join.kind,
        ),
        stats,
    ))
}

fn join_pairs(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    join: &JoinPredicate,
    candidate: &str,
) -> GraphStoreResult<(Vec<(usize, Option<usize>)>, crate::colt::ColtStats)> {
    if join.kind == JoinKind::Asof {
        return asof_pairs(left_keys, right_keys);
    }
    let emission = Emission::from_kind(join.kind).unwrap_or(Emission::Inner);
    Ok(if candidate == crate::colt::JOIN_CANDIDATE_MASK {
        execute_mask_join(left_keys, right_keys, emission)
    } else if candidate == crate::colt::JOIN_CANDIDATE_COLT_GENERIC {
        crate::colt::execute_generic_join(left_keys, right_keys, emission)
    } else {
        execute_binary_join(left_keys, right_keys, emission)
    })
}

fn asof_pairs(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
) -> GraphStoreResult<(Vec<(usize, Option<usize>)>, crate::colt::ColtStats)> {
    let mut last: Option<&str> = None;
    for key in right_keys.iter().flatten() {
        if let Some(prev) = last {
            if key.as_str() < prev {
                return Err(GraphStoreError::new(
                    "asof_requires_sorted_inputs",
                    "asof build lane must already be sorted; no sort operator is built",
                ));
            }
        }
        last = Some(key.as_str());
    }
    let build: Vec<(String, usize)> = right_keys
        .iter()
        .enumerate()
        .filter_map(|(index, key)| key.clone().map(|key| (key, index)))
        .collect();
    let mut pairs = Vec::new();
    for (left_index, left_key) in left_keys.iter().enumerate() {
        let Some(key) = left_key else {
            continue;
        };
        if let Some(right_index) = asof_nearest_preceding(key, &build)? {
            pairs.push((left_index, Some(*right_index)));
        }
    }
    Ok((
        pairs,
        crate::colt::ColtStats {
            algorithm: "asof_merge".to_string(),
            candidate_id: JOIN_CANDIDATE_COLT_BINARY.to_string(),
            ..crate::colt::ColtStats::default()
        },
    ))
}

fn assemble_pairs(
    pairs: &[(usize, Option<usize>)],
    left_rowset: impl Fn(usize) -> BTreeMap<String, ScoredResolvedRow>,
    right_alias: &str,
    right_rows: &[ScoredResolvedRow],
    kind: JoinKind,
) -> Vec<BTreeMap<String, ScoredResolvedRow>> {
    let mut result = Vec::with_capacity(pairs.len());
    for (left_index, right_index) in pairs {
        let mut rowset = left_rowset(*left_index);
        match (kind, right_index) {
            (JoinKind::Semi | JoinKind::Anti, _) => {}
            (_, Some(right_index)) => {
                rowset.insert(right_alias.to_string(), right_rows[*right_index].clone());
            }
            (JoinKind::Left | JoinKind::Asof, None) => {}
            _ => {}
        }
        result.push(rowset);
    }
    result
}

fn project_rowset(
    rowset: &BTreeMap<String, ScoredResolvedRow>,
    projection: &[Projection],
) -> BTreeMap<String, ScalarValue> {
    if projection.is_empty() {
        return rowset
            .iter()
            .flat_map(|(alias, row)| {
                let mut fields = Vec::new();
                fields.push((
                    format!("{alias}.id"),
                    ScalarValue::String(row.row.id.clone()),
                ));
                fields.extend(
                    row.row
                        .values
                        .iter()
                        .map(|(column, value)| (format!("{alias}.{column}"), value.clone())),
                );
                fields
            })
            .collect();
    }
    projection
        .iter()
        .filter_map(|field| {
            let row = rowset.get(&field.alias)?;
            if field.column == "id" {
                return Some((
                    format!("{}.{}", field.alias, field.column),
                    ScalarValue::String(row.row.id.clone()),
                ));
            }
            row.row
                .values
                .get(&field.column)
                .cloned()
                .map(|value| (format!("{}.{}", field.alias, field.column), value))
        })
        .collect()
}

fn rowset_score(rowset: &BTreeMap<String, ScoredResolvedRow>) -> Option<f32> {
    let mut scores = rowset.values().filter_map(|row| row.score);
    let first = scores.next()?;
    Some(scores.fold(first, |acc, score| acc + score))
}

fn rowset_ids(rowset: &BTreeMap<String, ScoredResolvedRow>) -> Vec<(&str, &str)> {
    rowset
        .iter()
        .map(|(alias, row)| (alias.as_str(), row.row.id.as_str()))
        .collect()
}

fn seed_annotation(row: &ResolvedRow, semiring: ProvenanceSemiring) -> f64 {
    match semiring.kind {
        ProvenanceSemiringKind::Viterbi => f64_col(row, "epistemic_weight")
            .or_else(|| f64_col(row, "justification_prior"))
            .unwrap_or_else(|| semiring.one())
            .clamp(0.0, 1.0),
        ProvenanceSemiringKind::Tropical => f64_col(row, "cost")
            .or_else(|| f64_col(row, "path_cost"))
            .or_else(|| {
                f64_col(row, "source_reliability")
                    .map(|reliability| 1.0 - reliability.clamp(0.0, 1.0))
            })
            .unwrap_or_else(|| semiring.one()),
        ProvenanceSemiringKind::Boolean | ProvenanceSemiringKind::Counting => semiring.one(),
        ProvenanceSemiringKind::Corroboration | ProvenanceSemiringKind::Probability => {
            f64_col(row, "epistemic_weight")
                .or_else(|| f64_col(row, "justification_prior"))
                .or_else(|| f64_col(row, "acceptability"))
                .unwrap_or_else(|| semiring.one())
                .clamp(0.0, 1.0)
        }
    }
}

fn rowset_annotation(
    rowset: &BTreeMap<String, ScoredResolvedRow>,
    semiring: ProvenanceSemiring,
) -> f64 {
    rowset
        .values()
        .filter_map(|row| row.annotation)
        .fold(semiring.one(), |acc, annotation| {
            semiring.product(acc, annotation)
        })
}

fn projected_values_key(values: &BTreeMap<String, ScalarValue>) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| format!("{values:?}"))
}

fn semiring_score(kind: ProvenanceSemiringKind, value: f64) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    Some(match kind {
        ProvenanceSemiringKind::Tropical => -value as f32,
        _ => value as f32,
    })
}

fn sort_annotated_rows(rows: &mut [QueryOutputRow], kind: ProvenanceSemiringKind) {
    rows.sort_by(|left, right| {
        let left_value = left
            .provenance
            .as_ref()
            .map(|annotation| annotation.value)
            .unwrap_or(0.0);
        let right_value = right
            .provenance
            .as_ref()
            .map(|annotation| annotation.value)
            .unwrap_or(0.0);
        let ordering = match kind {
            ProvenanceSemiringKind::Tropical => left_value
                .partial_cmp(&right_value)
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => right_value
                .partial_cmp(&left_value)
                .unwrap_or(std::cmp::Ordering::Equal),
        };
        ordering.then_with(|| {
            projected_values_key(&left.values).cmp(&projected_values_key(&right.values))
        })
    });
}

fn join_key(row: &ResolvedRow, column: &str) -> Option<String> {
    if column == "id" {
        return Some(row.id.clone());
    }
    row.values.get(column).map(scalar_key)
}

fn scalar_key(value: &ScalarValue) -> String {
    match value {
        ScalarValue::String(value) => value.clone(),
        ScalarValue::I64(value) => value.to_string(),
        ScalarValue::F64(value) => value.to_string(),
        ScalarValue::Bool(value) => value.to_string(),
    }
}

fn row_matches(row: &ResolvedRow, predicate: &Predicate) -> bool {
    match predicate {
        Predicate::Equals { column, value } => row.values.get(column) == Some(value),
        Predicate::Range { column, lo, hi } => row.values.get(column).is_some_and(|value| {
            value
                .as_f64()
                .map(|score| lo.contains_numeric(score, true) && hi.contains_numeric(score, false))
                .or_else(|| {
                    value
                        .as_str()
                        .map(|text| lo.contains_text(text, true) && hi.contains_text(text, false))
                })
                .unwrap_or(false)
        }),
        Predicate::Prefix { column, prefix } => row
            .values
            .get(column)
            .and_then(ScalarValue::as_str)
            .map(|value| value.starts_with(prefix))
            .unwrap_or(false),
        Predicate::TimeRange {
            column,
            lo_ms,
            hi_ms,
        } => row
            .values
            .get(column)
            .and_then(ScalarValue::as_i64)
            .map(|value| value >= *lo_ms && value <= *hi_ms)
            .unwrap_or(false),
        Predicate::GeoWithin {
            column,
            lat_property,
            lon_property,
            region:
                RegionRef::Bbox {
                    min_lat,
                    min_lon,
                    max_lat,
                    max_lon,
                },
            ..
        } => crate::ranking::row_coords(
            row,
            column,
            lat_property.as_deref(),
            lon_property.as_deref(),
        )
        .map(|(lat, lon)| lat >= *min_lat && lat <= *max_lat && lon >= *min_lon && lon <= *max_lon)
        .unwrap_or(false),
        Predicate::TextMatch {
            column,
            query,
            mode: PredicateMode::Filter,
        } => crate::ranking::row_text_matches(row, column, query),
        Predicate::Expand {
            mode: PredicateMode::Filter,
            ..
        } => true,
        Predicate::Knn { .. }
        | Predicate::TextMatch {
            mode: PredicateMode::Rank,
            ..
        }
        | Predicate::Expand {
            mode: PredicateMode::Rank,
            ..
        } => false,
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphqlSelection {
    pub relation: RelationId,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub fields: Vec<ColumnId>,
    #[serde(default)]
    pub joins: Vec<GraphqlJoinSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semiring: Option<ProvenanceSemiringKind>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphqlJoinSelection {
    pub relation: RelationId,
    #[serde(default)]
    pub alias: Option<String>,
    pub left_column: ColumnId,
    pub right_column: ColumnId,
    #[serde(default)]
    pub fields: Vec<ColumnId>,
}

pub fn compile_graphql_selection(selection: GraphqlSelection) -> QueryIr {
    let root_alias = selection
        .alias
        .clone()
        .unwrap_or_else(|| selection.relation.clone());
    let mut relations = vec![QueryRelation {
        alias: root_alias.clone(),
        relation: selection.relation.clone(),
        predicates: Vec::new(),
    }];
    let mut joins = Vec::new();
    let mut projection = selection
        .fields
        .into_iter()
        .map(|column| Projection {
            alias: root_alias.clone(),
            column,
        })
        .collect::<Vec<_>>();
    let mut seen_aliases = BTreeSet::from([root_alias.clone()]);
    for nested in selection.joins {
        let alias = nested.alias.unwrap_or_else(|| nested.relation.clone());
        if seen_aliases.insert(alias.clone()) {
            relations.push(QueryRelation {
                alias: alias.clone(),
                relation: nested.relation,
                predicates: Vec::new(),
            });
        }
        joins.push(JoinPredicate {
            left_alias: root_alias.clone(),
            left_column: nested.left_column,
            right_alias: alias.clone(),
            right_column: nested.right_column,
            kind: JoinKind::Inner,
        });
        projection.extend(nested.fields.into_iter().map(|column| Projection {
            alias: alias.clone(),
            column,
        }));
    }
    QueryIr {
        relations,
        joins,
        projection,
        limit: selection.limit,
        fusion: FusionPolicy::default(),
        semiring: selection.semiring,
        pinned_graph_version: None,
        ..QueryIr::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::access_method::{ScalarBound, ScalarValue};
    use crate::relational::{RelationalLens, ResolvedRow};

    fn row(relation: &str, id: &str, values: &[(&str, ScalarValue)]) -> ResolvedRow {
        ResolvedRow::new(
            relation,
            id,
            values
                .iter()
                .map(|(key, value)| ((*key).to_string(), value.clone()))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    #[test]
    fn scalar_and_time_predicates_use_access_methods_and_roaring_intersection() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "memory",
                "m1",
                &[
                    ("kind", ScalarValue::String("episode".to_string())),
                    ("t_ms", ScalarValue::I64(5)),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "memory",
                "m2",
                &[
                    ("kind", ScalarValue::String("episode".to_string())),
                    ("t_ms", ScalarValue::I64(15)),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "memory",
                "m3",
                &[
                    ("kind", ScalarValue::String("note".to_string())),
                    ("t_ms", ScalarValue::I64(15)),
                ],
            ))
            .unwrap();

        let result = execute_query(
            &store,
            QueryIr {
                relations: vec![QueryRelation {
                    alias: "m".to_string(),
                    relation: "memory".to_string(),
                    predicates: vec![
                        Predicate::Equals {
                            column: "kind".to_string(),
                            value: ScalarValue::String("episode".to_string()),
                        },
                        Predicate::TimeRange {
                            column: "t_ms".to_string(),
                            lo_ms: 10,
                            hi_ms: 20,
                        },
                    ],
                }],
                projection: vec![Projection {
                    alias: "m".to_string(),
                    column: "id".to_string(),
                }],
                ..QueryIr::default()
            },
        )
        .unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0].get("m.id"),
            Some(&ScalarValue::String("m2".to_string()))
        );
        assert_eq!(result.trace.full_relation_scans, 0);
        assert!(result.trace.used_roaring_bitmaps);
        assert_eq!(result.trace.bitmap_intersections, 1);
        assert!(result
            .trace
            .access_paths
            .iter()
            .any(|path| path.method == "ordered" && path.predicate == "equals"));
        assert!(result
            .trace
            .access_paths
            .iter()
            .any(|path| path.method == "time_series" && path.predicate == "time_range"));
        assert!(result
            .trace
            .plan_stages
            .iter()
            .any(|stage| stage.stage == "logical_plan"));
        assert!(result
            .trace
            .plan_stages
            .iter()
            .any(|stage| stage.stage == "relation_candidates"
                && stage.detail == "m"
                && stage.row_count == Some(1)));
    }

    #[test]
    fn planner_rejects_invalid_alias_references_before_execution() {
        let err = validate_query_ir(&QueryIr {
            relations: vec![QueryRelation {
                alias: "memory".to_string(),
                relation: "memory".to_string(),
                predicates: Vec::new(),
            }],
            projection: vec![Projection {
                alias: "missing".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        })
        .unwrap_err();

        assert_eq!(err.code, "unknown_query_projection_alias");

        let err = validate_query_ir(&QueryIr {
            relations: vec![
                QueryRelation {
                    alias: "memory".to_string(),
                    relation: "memory".to_string(),
                    predicates: Vec::new(),
                },
                QueryRelation {
                    alias: "memory".to_string(),
                    relation: "other_memory".to_string(),
                    predicates: Vec::new(),
                },
            ],
            ..QueryIr::default()
        })
        .unwrap_err();

        assert_eq!(err.code, "duplicate_query_relation_alias");
    }

    #[test]
    fn planner_hash_join_returns_many_to_many_content_epistemic_association() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "content",
                "c1",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("title", ScalarValue::String("Root".to_string())),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "epistemic",
                "e1",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("claim", ScalarValue::String("supports".to_string())),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "epistemic",
                "e2",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("claim", ScalarValue::String("undercuts".to_string())),
                ],
            ))
            .unwrap();

        let result = execute_query(
            &store,
            QueryIr {
                relations: vec![
                    QueryRelation {
                        alias: "c".to_string(),
                        relation: "content".to_string(),
                        predicates: Vec::new(),
                    },
                    QueryRelation {
                        alias: "e".to_string(),
                        relation: "epistemic".to_string(),
                        predicates: Vec::new(),
                    },
                ],
                joins: vec![JoinPredicate {
                    left_alias: "c".to_string(),
                    left_column: "content_key".to_string(),
                    right_alias: "e".to_string(),
                    right_column: "content_key".to_string(),
                    kind: JoinKind::Inner,
                }],
                projection: vec![
                    Projection {
                        alias: "c".to_string(),
                        column: "title".to_string(),
                    },
                    Projection {
                        alias: "e".to_string(),
                        column: "claim".to_string(),
                    },
                ],
                ..QueryIr::default()
            },
        )
        .unwrap();

        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
        let claims = result
            .rows
            .iter()
            .filter_map(|row| row.get("e.claim"))
            .collect::<Vec<_>>();
        assert!(claims.contains(&&ScalarValue::String("supports".to_string())));
        assert!(claims.contains(&&ScalarValue::String("undercuts".to_string())));
    }

    #[test]
    fn generic_plan_point_explain_names_generic_join() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "content",
                "c1",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("title", ScalarValue::String("Root".to_string())),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "epistemic",
                "e1",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("claim", ScalarValue::String("supports".to_string())),
                ],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "memory",
                "m1",
                &[
                    ("content_key", ScalarValue::String("doc:1".to_string())),
                    ("note", ScalarValue::String("kept".to_string())),
                ],
            ))
            .unwrap();
        let result = execute_query(
            &store,
            QueryIr {
                relations: vec![
                    QueryRelation {
                        alias: "c".to_string(),
                        relation: "content".to_string(),
                        predicates: Vec::new(),
                    },
                    QueryRelation {
                        alias: "e".to_string(),
                        relation: "epistemic".to_string(),
                        predicates: Vec::new(),
                    },
                    QueryRelation {
                        alias: "m".to_string(),
                        relation: "memory".to_string(),
                        predicates: Vec::new(),
                    },
                ],
                joins: vec![
                    JoinPredicate {
                        left_alias: "c".to_string(),
                        left_column: "content_key".to_string(),
                        right_alias: "e".to_string(),
                        right_column: "content_key".to_string(),
                        kind: JoinKind::Inner,
                    },
                    JoinPredicate {
                        left_alias: "c".to_string(),
                        left_column: "content_key".to_string(),
                        right_alias: "m".to_string(),
                        right_column: "content_key".to_string(),
                        kind: JoinKind::Inner,
                    },
                ],
                projection: vec![
                    Projection {
                        alias: "c".to_string(),
                        column: "title".to_string(),
                    },
                    Projection {
                        alias: "e".to_string(),
                        column: "claim".to_string(),
                    },
                    Projection {
                        alias: "m".to_string(),
                        column: "note".to_string(),
                    },
                ],
                join_plan_point: JoinPlanPoint::Generic,
                ..QueryIr::default()
            },
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.trace.join_algorithm.as_deref(), Some("generic_join"));
        assert_eq!(
            result.trace.join_candidate_id.as_deref(),
            Some("join_colt_generic")
        );
    }

    #[test]
    fn graphql_selection_compiles_to_single_planner_pass_join() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "content",
                "c1",
                &[("key", ScalarValue::String("k1".to_string()))],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "epistemic",
                "e1",
                &[
                    ("content_key", ScalarValue::String("k1".to_string())),
                    ("claim", ScalarValue::String("grounded".to_string())),
                ],
            ))
            .unwrap();

        let query = compile_graphql_selection(GraphqlSelection {
            relation: "content".to_string(),
            alias: Some("content".to_string()),
            fields: vec!["key".to_string()],
            joins: vec![GraphqlJoinSelection {
                relation: "epistemic".to_string(),
                alias: Some("epistemic".to_string()),
                left_column: "key".to_string(),
                right_column: "content_key".to_string(),
                fields: vec!["claim".to_string()],
            }],
            limit: None,
            semiring: None,
        });
        let result = execute_query(&store, query).unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0].get("epistemic.claim"),
            Some(&ScalarValue::String("grounded".to_string()))
        );
        assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
    }

    #[test]
    fn range_predicate_filters_exclusive_bounds_after_index_scan() {
        let mut store = RelationalLens::new();
        for (id, score) in [("a", 1), ("b", 2), ("c", 3)] {
            store
                .upsert_row(row("scores", id, &[("score", ScalarValue::I64(score))]))
                .unwrap();
        }
        let result = execute_query(
            &store,
            QueryIr {
                relations: vec![QueryRelation {
                    alias: "s".to_string(),
                    relation: "scores".to_string(),
                    predicates: vec![Predicate::Range {
                        column: "score".to_string(),
                        lo: ScalarBound::Excluded(ScalarValue::I64(1)),
                        hi: ScalarBound::Included(ScalarValue::I64(3)),
                    }],
                }],
                projection: vec![Projection {
                    alias: "s".to_string(),
                    column: "id".to_string(),
                }],
                ..QueryIr::default()
            },
        )
        .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.trace.access_paths[0].method, "ordered");
    }

    #[test]
    fn left_join_pads_brief_assembly_miss() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "runs",
                "run-1",
                &[("id", ScalarValue::String("run-1".into()))],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "recommendations",
                "rec-other",
                &[("run_id", ScalarValue::String("run-other".into()))],
            ))
            .unwrap();
        let result = execute_query(
            &store,
            QueryIr {
                relations: vec![
                    QueryRelation {
                        alias: "run".to_string(),
                        relation: "runs".to_string(),
                        predicates: Vec::new(),
                    },
                    QueryRelation {
                        alias: "rec".to_string(),
                        relation: "recommendations".to_string(),
                        predicates: Vec::new(),
                    },
                ],
                joins: vec![JoinPredicate {
                    left_alias: "run".to_string(),
                    left_column: "id".to_string(),
                    right_alias: "rec".to_string(),
                    right_column: "run_id".to_string(),
                    kind: JoinKind::Left,
                }],
                projection: vec![
                    Projection {
                        alias: "run".to_string(),
                        column: "id".to_string(),
                    },
                    Projection {
                        alias: "rec".to_string(),
                        column: "id".to_string(),
                    },
                ],
                ..QueryIr::default()
            },
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0].get("run.id"),
            Some(&ScalarValue::String("run-1".into()))
        );
        assert!(result.rows[0].get("rec.id").is_none());
        assert_eq!(result.trace.join_algorithm.as_deref(), Some("hash_join"));
        assert_eq!(
            result.trace.join_steering_reason.as_deref(),
            Some("cold_start_native_floor")
        );
    }

    #[test]
    fn asof_refuses_unsorted_build_lane() {
        let mut store = RelationalLens::new();
        store
            .upsert_row(row(
                "events",
                "e1",
                &[("key", ScalarValue::String("v2".into()))],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "watermarks",
                "wm-a",
                &[("key", ScalarValue::String("v9".into()))],
            ))
            .unwrap();
        store
            .upsert_row(row(
                "watermarks",
                "wm-b",
                &[("key", ScalarValue::String("v1".into()))],
            ))
            .unwrap();
        let err = execute_query(
            &store,
            QueryIr {
                relations: vec![
                    QueryRelation {
                        alias: "e".to_string(),
                        relation: "events".to_string(),
                        predicates: Vec::new(),
                    },
                    QueryRelation {
                        alias: "w".to_string(),
                        relation: "watermarks".to_string(),
                        predicates: Vec::new(),
                    },
                ],
                joins: vec![JoinPredicate {
                    left_alias: "e".to_string(),
                    left_column: "key".to_string(),
                    right_alias: "w".to_string(),
                    right_column: "key".to_string(),
                    kind: JoinKind::Asof,
                }],
                projection: vec![Projection {
                    alias: "e".to_string(),
                    column: "id".to_string(),
                }],
                ..QueryIr::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.code, "asof_requires_sorted_inputs");
    }
}
