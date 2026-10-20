//! Acceptance coverage for SPEC-MULTIMODAL-PLANNER-UNIFY.
//!
//! The invariant under test: boolean filters intersect through access methods, while
//! relevance/similarity rankers survive into a separate score-fusion phase. Crucially, modality
//! data is resolved *by node id* from a live [`ModalityResolver`] (here an in-memory stand-in for
//! the TurboVec / full-text / spatial / graph subsystems) and is NOT copied into the relational
//! store: the relational rows carry only scalar/structured columns used for residual exact checks.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustyred_thg_core::{
    execute_query_with_resolver, AmResult, Direction, FusionPolicy, ModalityResolver, Predicate,
    PredicateMode, Projection, QueryIr, QueryRelation, RankOutcome, RankedRow, RankingRule,
    RegionRef, RelationalLens, ResolvedRow, ScalarValue,
};

// --- in-memory live-subsystem stand-in --------------------------------------------------------

#[derive(Default)]
struct InMemoryModality {
    vectors: BTreeMap<(String, String), BTreeMap<String, Vec<f32>>>,
    texts: BTreeMap<(String, String), BTreeMap<String, String>>,
    coords: BTreeMap<(String, String), BTreeMap<String, (f64, f64)>>,
    out_edges: BTreeMap<(String, String), Vec<String>>,
}

impl InMemoryModality {
    fn vector(&mut self, relation: &str, column: &str, id: &str, v: Vec<f32>) {
        self.vectors
            .entry((relation.into(), column.into()))
            .or_default()
            .insert(id.into(), v);
    }

    fn text(&mut self, relation: &str, column: &str, id: &str, s: &str) {
        self.texts
            .entry((relation.into(), column.into()))
            .or_default()
            .insert(id.into(), s.into());
    }

    fn coord(&mut self, relation: &str, column: &str, id: &str, lat: f64, lon: f64) {
        self.coords
            .entry((relation.into(), column.into()))
            .or_default()
            .insert(id.into(), (lat, lon));
    }

    fn edge(&mut self, from: &str, edge_type: &str, to: &str) {
        self.out_edges
            .entry((from.into(), edge_type.into()))
            .or_default()
            .push(to.into());
    }

    /// Sorted (similarity desc, id asc) list over the whole index for a column.
    fn ranked_vectors(&self, relation: &str, column: &str, query: &[f32]) -> Vec<(String, f32)> {
        let Some(index) = self.vectors.get(&(relation.into(), column.into())) else {
            return Vec::new();
        };
        let qn = normalize(query);
        let mut all = index
            .iter()
            .filter(|(_, v)| v.len() == query.len())
            .map(|(id, v)| (id.clone(), cosine(&qn, &normalize(v))))
            .collect::<Vec<_>>();
        all.sort_by(|a, b| score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
        all
    }

    fn reachable(&self, from: &str, edge_type: &str, dir: Direction) -> BTreeMap<String, usize> {
        let mut visited = BTreeMap::new();
        let mut queue = VecDeque::from([(from.to_string(), 0usize)]);
        while let Some((node, dist)) = queue.pop_front() {
            let neighbors: Vec<String> = match dir {
                Direction::Out => self
                    .out_edges
                    .get(&(node.clone(), edge_type.to_string()))
                    .cloned()
                    .unwrap_or_default(),
                Direction::In => self
                    .out_edges
                    .iter()
                    .filter(|((_, et), tos)| et == edge_type && tos.contains(&node))
                    .map(|((f, _), _)| f.clone())
                    .collect(),
            };
            for next in neighbors {
                if next == from || visited.contains_key(&next) {
                    continue;
                }
                visited.insert(next.clone(), dist + 1);
                queue.push_back((next, dist + 1));
            }
        }
        visited
    }
}

impl ModalityResolver for InMemoryModality {
    fn vector_knn(
        &self,
        relation: &str,
        column: &str,
        query: &[f32],
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<RankOutcome> {
        let all = self.ranked_vectors(relation, column, query);
        if all.is_empty() {
            return Ok(RankOutcome::default());
        }
        match candidates {
            // No filters: direct top-k from the index.
            None => Ok(RankOutcome {
                rows: all
                    .into_iter()
                    .take(k)
                    .map(|(row_id, score)| RankedRow {
                        row_id,
                        score,
                        layer: None,
                    })
                    .collect(),
                strategy: Some("index_topk".into()),
                overfetch_rounds: 0,
            }),
            // Small candidate set: score it exactly (lossless), identical to brute force over C.
            Some(cands) if cands.len() <= 4 * k.max(1) => {
                let mut rows = all
                    .into_iter()
                    .filter(|(id, _)| cands.contains(id))
                    .map(|(row_id, score)| RankedRow {
                        row_id,
                        score,
                        layer: None,
                    })
                    .collect::<Vec<_>>();
                rows.truncate(k);
                Ok(RankOutcome {
                    rows,
                    strategy: Some("exact_over_candidates".into()),
                    overfetch_rounds: 0,
                })
            }
            // Large candidate set: approximate index top-k, intersect with C, overfetch until k survive.
            Some(cands) => {
                let mut factor = 8usize;
                let mut rounds = 0usize;
                loop {
                    rounds += 1;
                    let fetch = (k.max(1) * factor).min(all.len());
                    let kept = all
                        .iter()
                        .take(fetch)
                        .filter(|(id, _)| cands.contains(id))
                        .take(k)
                        .map(|(row_id, score)| RankedRow {
                            row_id: row_id.clone(),
                            score: *score,
                            layer: None,
                        })
                        .collect::<Vec<_>>();
                    if kept.len() >= k || fetch >= all.len() || rounds >= 3 {
                        return Ok(RankOutcome {
                            rows: kept,
                            strategy: Some("filtered_overfetch".into()),
                            overfetch_rounds: rounds,
                        });
                    }
                    factor *= 2;
                }
            }
        }
    }

    fn text_rank(
        &self,
        relation: &str,
        column: &str,
        query: &str,
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        let Some(raw) = self.texts.get(&(relation.into(), column.into())) else {
            return Ok(Vec::new());
        };
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let docs = raw
            .iter()
            .map(|(id, s)| (id.clone(), tokenize(s)))
            .collect::<BTreeMap<_, _>>();
        let avg =
            (docs.values().map(Vec::len).sum::<usize>() as f32 / docs.len().max(1) as f32).max(1.0);
        let n = docs.len() as f32;
        let mut rows = Vec::new();
        for (id, doc_tokens) in &docs {
            if candidates.map(|c| !c.contains(id)).unwrap_or(false) {
                continue;
            }
            let mut score = 0.0f32;
            for token in &query_tokens {
                let tf = doc_tokens.iter().filter(|t| *t == token).count() as f32;
                if tf == 0.0 {
                    continue;
                }
                let df = docs
                    .values()
                    .filter(|tokens| tokens.iter().any(|t| t == token))
                    .count() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                let norm = 1.0 - 0.75 + 0.75 * doc_tokens.len() as f32 / avg;
                score += idf * tf * (1.2 + 1.0) / (tf + 1.2 * norm);
            }
            if score > 0.0 {
                rows.push(RankedRow {
                    row_id: id.clone(),
                    score,
                    layer: None,
                });
            }
        }
        rows.sort_by(|a, b| score_desc(a.score, b.score).then_with(|| a.row_id.cmp(&b.row_id)));
        rows.truncate(k);
        Ok(rows)
    }

    fn expand_proximity(
        &self,
        from: &str,
        layers: &[rustyred_thg_core::LayerId],
        dir: Direction,
        candidates: Option<&BTreeSet<String>>,
        k: usize,
    ) -> AmResult<Vec<RankedRow>> {
        if layers.len() != 1 {
            return Ok(Vec::new());
        }
        let layer = &layers[0];
        let mut rows = self
            .reachable(from, layer.as_str(), dir)
            .into_iter()
            .filter(|(id, _)| candidates.map(|c| c.contains(id)).unwrap_or(true))
            .map(|(row_id, distance)| RankedRow {
                row_id,
                score: 1.0 / (distance as f32 + 1.0),
                layer: Some(layer.clone()),
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| score_desc(a.score, b.score).then_with(|| a.row_id.cmp(&b.row_id)));
        rows.truncate(k);
        Ok(rows)
    }

    fn text_contains(
        &self,
        relation: &str,
        column: &str,
        query: &str,
    ) -> AmResult<Option<Vec<String>>> {
        let Some(docs) = self.texts.get(&(relation.into(), column.into())) else {
            return Ok(None);
        };
        let query_tokens = tokenize(query);
        Ok(Some(
            docs.iter()
                .filter(|(_, s)| {
                    let doc_tokens = tokenize(s);
                    query_tokens.iter().all(|t| doc_tokens.contains(t))
                })
                .map(|(id, _)| id.clone())
                .collect(),
        ))
    }

    fn geo_overapprox(
        &self,
        relation: &str,
        column: &str,
        _lat_property: Option<&str>,
        _lon_property: Option<&str>,
        _label: Option<&str>,
        region: &RegionRef,
    ) -> AmResult<Option<Vec<String>>> {
        let RegionRef::Bbox {
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        } = region;
        let Some(coords) = self.coords.get(&(relation.into(), column.into())) else {
            return Ok(None);
        };
        // Simulate H3/S2 cell over-approximation with a pad, so a point inside a boundary cell but
        // outside the exact bbox survives the index scan and is removed by the residual check.
        let lat_pad = ((*max_lat - *min_lat).abs() * 0.25).max(0.001);
        let lon_pad = ((*max_lon - *min_lon).abs() * 0.25).max(0.001);
        let ids = coords
            .iter()
            .filter(|(_, (lat, lon))| {
                *lat >= *min_lat - lat_pad
                    && *lat <= *max_lat + lat_pad
                    && *lon >= *min_lon - lon_pad
                    && *lon <= *max_lon + lon_pad
            })
            .map(|(id, _)| id.clone())
            .collect();
        Ok(Some(ids))
    }

    fn expand_reachable(
        &self,
        from: &str,
        layers: &[rustyred_thg_core::LayerId],
        dir: Direction,
    ) -> AmResult<Option<Vec<(String, rustyred_thg_core::LayerId)>>> {
        let mut tagged = Vec::new();
        let mut seen = BTreeSet::new();
        for layer in layers {
            for row_id in self
                .reachable(from, layer.as_str(), dir.clone())
                .into_keys()
            {
                if seen.insert((row_id.clone(), layer.clone())) {
                    tagged.push((row_id, layer.clone()));
                }
            }
        }
        Ok(Some(tagged))
    }
}

// --- helpers ----------------------------------------------------------------------------------

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn score_desc(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

fn text(value: &str) -> ScalarValue {
    ScalarValue::String(value.to_string())
}

fn scalar_row(relation: &str, id: &str, values: &[(&str, ScalarValue)]) -> ResolvedRow {
    ResolvedRow::new(
        relation,
        id,
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    )
}

fn id_order(result: &rustyred_thg_core::QueryResult, field: &str) -> Vec<String> {
    result
        .rows
        .iter()
        .filter_map(|row| match row.get(field) {
            Some(ScalarValue::String(value)) => Some(value.clone()),
            _ => None,
        })
        .collect()
}

fn run(
    store: &RelationalLens,
    resolver: &InMemoryModality,
    query: QueryIr,
) -> rustyred_thg_core::QueryResult {
    execute_query_with_resolver(store, query, resolver).unwrap()
}

fn text_filter_matches(row_text: &str, query: &str) -> bool {
    let row_tokens = tokenize(row_text);
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return false;
    }
    query_tokens
        .iter()
        .all(|query_token| row_tokens.contains(query_token))
}

fn knn_relation(predicates: Vec<Predicate>) -> QueryRelation {
    QueryRelation {
        alias: "d".to_string(),
        relation: "docs".to_string(),
        predicates,
    }
}

fn doc_projection() -> Vec<Projection> {
    vec![Projection {
        alias: "d".to_string(),
        column: "id".to_string(),
    }]
}

// --- acceptance criteria ----------------------------------------------------------------------

/// AC1: Knn + Equals + TimeRange returns at most k rows, each satisfying the scalar/time filters,
/// ordered by similarity, scored, with the vector method scoring the filtered candidate set and
/// zero full-relation scans. Modality vectors live only in the resolver, not the relational rows.
#[test]
fn compound_filters_preserve_knn_scores_and_order() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, kind, t_ms, embedding) in [
        ("doc:a", "episode", 12, vec![0.7, 0.3]),
        ("doc:b", "episode", 14, vec![1.0, 0.0]),
        ("doc:c", "episode", 15, vec![0.8, 0.2]),
        ("doc:d", "note", 14, vec![0.99, 0.01]),
        ("doc:e", "episode", 40, vec![0.98, 0.02]),
    ] {
        store
            .upsert_row(scalar_row(
                "docs",
                id,
                &[("kind", text(kind)), ("t_ms", ScalarValue::I64(t_ms))],
            ))
            .unwrap();
        modality.vector("docs", "embedding", id, embedding);
    }

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![knn_relation(vec![
                Predicate::Knn {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0],
                    k: 2,
                },
                Predicate::Equals {
                    column: "kind".to_string(),
                    value: text("episode"),
                },
                Predicate::TimeRange {
                    column: "t_ms".to_string(),
                    lo_ms: 10,
                    hi_ms: 20,
                },
            ])],
            projection: doc_projection(),
            ..QueryIr::default()
        },
    );

    assert_eq!(id_order(&result, "d.id"), vec!["doc:b", "doc:c"]);
    assert!(result.rows.iter().all(|row| row.score.is_some()));
    assert_eq!(result.trace.full_relation_scans, 0);
    assert_eq!(result.trace.candidate_set_size, 3);
    assert_eq!(
        result.trace.knn_strategy.as_deref(),
        Some("exact_over_candidates")
    );
    assert!(result
        .trace
        .rankers
        .iter()
        .any(|ranker| ranker.method == "vector" && ranker.contributed_rows == 2));
}

#[test]
fn knn_query_records_vector_access_path_and_matches_reference_rank() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    let rows = [
        ("doc:a", "keep", vec![1.0, 0.0]),
        ("doc:b", "keep", vec![0.8, 0.1]),
        ("doc:c", "drop", vec![0.7, 0.2]),
    ];
    for (id, bucket, embedding) in rows {
        store
            .upsert_row(scalar_row(
                "docs",
                id,
                &[("bucket", text(bucket)), ("title", text("title"))],
            ))
            .unwrap();
        modality.vector("docs", "embedding", id, embedding);
    }

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "d".to_string(),
                relation: "docs".to_string(),
                predicates: vec![
                    Predicate::Equals {
                        column: "bucket".to_string(),
                        value: text("keep"),
                    },
                    Predicate::Knn {
                        column: "embedding".to_string(),
                        query: vec![1.0, 0.0],
                        k: 2,
                    },
                ],
            }],
            projection: doc_projection(),
            ..QueryIr::default()
        },
    );

    let expected = modality
        .ranked_vectors("docs", "embedding", &[1.0, 0.0])
        .into_iter()
        .filter(|(id, _)| {
            store
                .relation("docs")
                .and_then(|relation| relation.get(id))
                .is_some_and(|row| {
                    row.values.get("bucket").and_then(ScalarValue::as_str) == Some("keep")
                })
        })
        .take(2)
        .map(|(id, _)| id)
        .collect::<Vec<_>>();

    assert_eq!(id_order(&result, "d.id"), expected);
    assert!(result
        .trace
        .access_paths
        .iter()
        .any(|path| path.method == "vector"));
    assert_eq!(result.trace.full_relation_scans, 0);
    assert!(result
        .trace
        .rankers
        .iter()
        .any(|ranker| ranker.method == "vector"));
}

/// AC2: for |C| <= 4k, filtered kNN returns exactly the same top-k and order as an independent
/// brute-force similarity computation over C, and the trace reports `exact_over_candidates`.
#[test]
fn small_candidate_knn_matches_brute_force_over_candidates() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    let rows = [
        ("doc:a", "keep", vec![0.1, 0.9]),
        ("doc:b", "keep", vec![0.9, 0.1]),
        ("doc:c", "keep", vec![0.8, 0.2]),
        ("doc:d", "drop", vec![1.0, 0.0]),
    ];
    for (id, bucket, embedding) in rows.clone() {
        store
            .upsert_row(scalar_row("docs", id, &[("bucket", text(bucket))]))
            .unwrap();
        modality.vector("docs", "embedding", id, embedding);
    }

    let query = vec![1.0, 0.0];
    let k = 2;
    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![knn_relation(vec![
                Predicate::Equals {
                    column: "bucket".to_string(),
                    value: text("keep"),
                },
                Predicate::Knn {
                    column: "embedding".to_string(),
                    query: query.clone(),
                    k,
                },
            ])],
            projection: doc_projection(),
            ..QueryIr::default()
        },
    );

    // Independent brute-force reference over the candidate set C = {a, b, c} (bucket == keep).
    let qn = normalize(&query);
    let mut reference = rows
        .iter()
        .filter(|(_, bucket, _)| *bucket == "keep")
        .map(|(id, _, v)| (id.to_string(), cosine(&qn, &normalize(v))))
        .collect::<Vec<_>>();
    reference.sort_by(|a, b| score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
    let reference_top_k = reference
        .into_iter()
        .take(k)
        .map(|(id, _)| id)
        .collect::<Vec<_>>();

    assert_eq!(id_order(&result, "d.id"), reference_top_k);
    assert_eq!(
        result.trace.knn_strategy.as_deref(),
        Some("exact_over_candidates")
    );
}

#[test]
fn text_filter_records_plan_trace_and_matches_exact_term_filter() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, body) in [
        ("doc:a", "alpha beta"),
        ("doc:b", "alpha gamma"),
        ("doc:c", "beta gamma"),
    ] {
        store
            .upsert_row(scalar_row("docs", id, &[("body", text(body))]))
            .unwrap();
        modality.text("docs", "body", id, body);
    }

    let query = "alpha gamma";
    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "d".to_string(),
                relation: "docs".to_string(),
                predicates: vec![Predicate::TextMatch {
                    column: "body".to_string(),
                    query: query.to_string(),
                    mode: PredicateMode::Filter,
                }],
            }],
            projection: doc_projection(),
            ..QueryIr::default()
        },
    );

    let expected = [("doc:a", "alpha beta"), ("doc:b", "alpha gamma")]
        .into_iter()
        .map(|(id, body)| {
            let include = text_filter_matches(body, query);
            (id, include)
        })
        .filter_map(|(id, include)| include.then_some(id.to_string()))
        .collect::<Vec<_>>();

    assert_eq!(id_order(&result, "d.id"), expected);
    assert_eq!(result.trace.full_relation_scans, 0);
    assert!(result
        .trace
        .access_paths
        .iter()
        .any(|path| path.method == "text"));
}

/// AC3: with a candidate set larger than the threshold AND a filter that removes the most-similar
/// rows from the index, the query still returns k results, the trace reports the approximate
/// (overfetch) strategy, and the overfetch loop genuinely iterates more than once.
#[test]
fn large_candidate_knn_overfetches_and_preserves_k() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    // 50 vectors, descending similarity to [1, 0] as the index grows.
    for index in 0..50 {
        let id = format!("doc:{index:02}");
        let x = 1.0 - (index as f32 * 0.02);
        let y = index as f32 * 0.02;
        // Keep only the 10 LEAST-similar rows; the most-similar (doc:00..) are filtered out, so the
        // approximate top-k repeatedly misses the candidate set and must overfetch.
        let bucket = if index >= 40 { "keep" } else { "drop" };
        store
            .upsert_row(scalar_row("docs", &id, &[("bucket", text(bucket))]))
            .unwrap();
        modality.vector("docs", "embedding", &id, vec![x, y]);
    }

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![knn_relation(vec![
                Predicate::Equals {
                    column: "bucket".to_string(),
                    value: text("keep"),
                },
                Predicate::Knn {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0],
                    k: 2,
                },
            ])],
            projection: doc_projection(),
            ..QueryIr::default()
        },
    );

    assert_eq!(result.rows.len(), 2);
    // The two most-similar of the kept (least-similar) rows are doc:40 and doc:41.
    assert_eq!(id_order(&result, "d.id"), vec!["doc:40", "doc:41"]);
    assert_eq!(
        result.trace.knn_strategy.as_deref(),
        Some("filtered_overfetch")
    );
    assert!(
        result.trace.knn_overfetch_rounds >= 2,
        "overfetch must genuinely loop, got {} round(s)",
        result.trace.knn_overfetch_rounds
    );
}

/// AC4: Knn + TextMatch { Rank } orders rows by RRF over the two score lists, matching an
/// independent RRF computation, and the trace lists both ranker contributions with fusion == rrf.
#[test]
fn two_ranker_rrf_fusion_matches_reference() {
    let (store, modality, rows) = two_ranker_fixture();
    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![knn_relation(two_ranker_predicates())],
            projection: doc_projection(),
            limit: Some(3),
            ..QueryIr::default()
        },
    );

    // Independent RRF reference (k = 60) from the two ranked lists the resolver produces.
    let vector_list = modality
        .vector_knn("docs", "embedding", &[1.0, 0.0], None, 3)
        .unwrap()
        .rows;
    let text_list = modality
        .text_rank("docs", "body", "alpha", None, 3)
        .unwrap();
    let reference = reciprocal_rank_fusion(&[vector_list, text_list], 60.0);

    assert_eq!(id_order(&result, "d.id"), reference);
    assert_eq!(result.rows.len(), rows.len());
    assert_eq!(result.trace.fusion, "rrf");
    assert_eq!(result.trace.rankers.len(), 2);
}

/// AC7 (weighted variant): the same two-ranker query under weighted fusion produces the
/// weighted-sum order, here dominated by the heavily weighted text ranker.
#[test]
fn weighted_fusion_uses_method_weights() {
    let (store, modality, _) = two_ranker_fixture();
    let weights = BTreeMap::from([("text_rank".to_string(), 5.0), ("vector".to_string(), 0.1)]);
    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![knn_relation(two_ranker_predicates())],
            projection: doc_projection(),
            limit: Some(3),
            fusion: FusionPolicy::Weighted { weights },
            ..QueryIr::default()
        },
    );

    // text order is b > c > a, weighted 50x over the vector order, so the full order follows text.
    assert_eq!(id_order(&result, "d.id"), vec!["doc:b", "doc:c", "doc:a"]);
    assert_eq!(result.trace.fusion, "weighted");
}

#[test]
fn weighted_fusion_rejects_unknown_method_names() {
    let (store, modality, _) = two_ranker_fixture();
    let error = execute_query_with_resolver(
        &store,
        QueryIr {
            relations: vec![knn_relation(two_ranker_predicates())],
            projection: doc_projection(),
            limit: Some(3),
            fusion: FusionPolicy::Weighted {
                weights: BTreeMap::from([("typo_ranker".to_string(), 1.0)]),
            },
            ..QueryIr::default()
        },
        &modality,
    )
    .unwrap_err();

    assert_eq!(error.code, "unknown_weighted_fusion_method");
    assert!(error.message.contains("typo_ranker"));
    assert!(error.message.contains("vector"));
}

/// Bonus (Expand { Rank }): graph proximity ranks reachable rows by hop distance, restricted to
/// the scalar-filtered candidate set, with an honest `hop_distance` score source.
#[test]
fn expand_rank_orders_by_hop_distance() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, label) in [
        ("node:root", "root"),
        ("node:a", "doc"),
        ("node:b", "doc"),
        ("node:c", "doc"),
    ] {
        store
            .upsert_row(scalar_row("nodes", id, &[("label", text(label))]))
            .unwrap();
    }
    modality.edge("node:root", "LINKS", "node:a");
    modality.edge("node:a", "LINKS", "node:b");
    modality.edge("node:b", "LINKS", "node:c");

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".to_string(),
                relation: "nodes".to_string(),
                predicates: vec![
                    Predicate::Equals {
                        column: "label".to_string(),
                        value: text("doc"),
                    },
                    Predicate::Expand {
                        from: "node:root".to_string(),
                        layers: vec!["LINKS".into()],
                        dir: Direction::Out,
                        mode: PredicateMode::Rank,
                    },
                ],
            }],
            projection: vec![Projection {
                alias: "n".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    );

    // a (1 hop) closer than b (2) closer than c (3); all are label == doc.
    assert_eq!(
        id_order(&result, "n.id"),
        vec!["node:a", "node:b", "node:c"]
    );
    assert!(result
        .trace
        .rankers
        .iter()
        .any(|ranker| ranker.method == "expand_ppr" && ranker.score_source == "hop_distance"));
}

/// AC5: Expand { Filter } + Equals returns only rows reachable from X that satisfy the scalar
/// filter. Reachability comes from the live graph resolver; the scalar filter intersects it.
#[test]
fn expand_filter_intersects_with_scalar_filter() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, label) in [
        ("node:root", "root"),
        ("node:a", "doc"),
        ("node:b", "doc"),
        ("node:c", "note"),
    ] {
        store
            .upsert_row(scalar_row("nodes", id, &[("label", text(label))]))
            .unwrap();
    }
    modality.edge("node:root", "LINKS", "node:a");
    modality.edge("node:a", "LINKS", "node:b");
    modality.edge("node:root", "LINKS", "node:c");

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "n".to_string(),
                relation: "nodes".to_string(),
                predicates: vec![
                    Predicate::Expand {
                        from: "node:root".to_string(),
                        layers: vec!["LINKS".into()],
                        dir: Direction::Out,
                        mode: PredicateMode::Filter,
                    },
                    Predicate::Equals {
                        column: "label".to_string(),
                        value: text("doc"),
                    },
                ],
            }],
            projection: vec![Projection {
                alias: "n".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    );

    assert_eq!(id_order(&result, "n.id"), vec!["node:a", "node:b"]);
}

/// AC6: a point inside a boundary cell (so it survives the over-approximate spatial index) but
/// outside the exact bbox is excluded by the residual check over the row's own coordinates.
#[test]
fn geo_overapprox_is_residually_exact() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, lat, lon) in [("place:inside", 0.5, 0.5), ("place:outside", 1.0005, 0.5)] {
        store
            .upsert_row(scalar_row(
                "places",
                id,
                &[
                    ("lat", ScalarValue::F64(lat)),
                    ("lon", ScalarValue::F64(lon)),
                ],
            ))
            .unwrap();
        modality.coord("places", "geo", id, lat, lon);
    }

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "p".to_string(),
                relation: "places".to_string(),
                predicates: vec![Predicate::GeoWithin {
                    column: "geo".to_string(),
                    region: RegionRef::Bbox {
                        min_lat: 0.0,
                        min_lon: 0.0,
                        max_lat: 1.0,
                        max_lon: 1.0,
                    },
                    lat_property: None,
                    lon_property: None,
                    label: None,
                }],
            }],
            projection: vec![Projection {
                alias: "p".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    );

    assert_eq!(id_order(&result, "p.id"), vec!["place:inside"]);
    assert_eq!(result.trace.full_relation_scans, 0);
    assert_eq!(result.trace.access_paths[0].method, "geo_index");
}

/// (b) Live spatial address: GeoWithin carrying `lat_property`/`lon_property`/`label` plumbs the
/// designation through to the resolver, and the residual check reads coordinates from the explicit
/// property names (here `latitude`/`longitude`, distinct from the `geo` over-approx column).
#[test]
fn geo_explicit_lat_lon_properties_used_for_residual() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    for (id, lat, lon) in [("place:inside", 0.5, 0.5), ("place:outside", 1.0005, 0.5)] {
        store
            .upsert_row(scalar_row(
                "places",
                id,
                &[
                    ("latitude", ScalarValue::F64(lat)),
                    ("longitude", ScalarValue::F64(lon)),
                ],
            ))
            .unwrap();
        // The over-approx index is keyed by the logical `geo` column; the residual reads the
        // explicit property names, so both must point at the same point to agree.
        modality.coord("places", "geo", id, lat, lon);
    }

    let result = run(
        &store,
        &modality,
        QueryIr {
            relations: vec![QueryRelation {
                alias: "p".to_string(),
                relation: "places".to_string(),
                predicates: vec![Predicate::GeoWithin {
                    column: "geo".to_string(),
                    region: RegionRef::Bbox {
                        min_lat: 0.0,
                        min_lon: 0.0,
                        max_lat: 1.0,
                        max_lon: 1.0,
                    },
                    lat_property: Some("latitude".to_string()),
                    lon_property: Some("longitude".to_string()),
                    label: Some("Place".to_string()),
                }],
            }],
            projection: vec![Projection {
                alias: "p".to_string(),
                column: "id".to_string(),
            }],
            ..QueryIr::default()
        },
    );

    assert_eq!(id_order(&result, "p.id"), vec!["place:inside"]);
    assert_eq!(result.trace.full_relation_scans, 0);
    assert_eq!(result.trace.access_paths[0].method, "geo_index");
}

/// D2 + the demonstration (SPEC-RUSTYRED-RANKING-CASCADE): the SAME query ranks a strong-but-
/// undercut document first under RRF, but the supported document first under a `Cascade`. Proven
/// both with the epistemic rules ahead of relevance (trust dominates) and behind them (banding ties
/// Vector+Text, EpistemicStatus breaks the tie so the undercut doc still ranks below the grounded
/// one that has a lower vector score). The trace names the rule that moved it.
#[test]
fn cascade_demotes_undercut_below_supported_against_rrf() {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    // doc:a: strong relevance (top vector + equal text) but undercut. doc:b: supported (grounded),
    // a lower vector score in the same band, same body. acceptance_status lives as a scalar column
    // on the ranked relation (the join's epistemic standing projected onto the content row).
    for (id, status, embedding) in [
        ("doc:a", "undercuts", vec![1.0, 0.0]),
        ("doc:b", "grounded", vec![0.9, 0.1]),
    ] {
        store
            .upsert_row(scalar_row(
                "docs",
                id,
                &[("acceptance_status", text(status))],
            ))
            .unwrap();
        modality.vector("docs", "embedding", id, embedding);
        modality.text("docs", "body", id, "alpha alpha");
    }
    let predicates = || {
        vec![
            Predicate::Knn {
                column: "embedding".to_string(),
                query: vec![1.0, 0.0],
                k: 2,
            },
            Predicate::TextMatch {
                column: "body".to_string(),
                query: "alpha".to_string(),
                mode: PredicateMode::Rank,
            },
        ]
    };
    let query = |fusion: FusionPolicy| QueryIr {
        relations: vec![knn_relation(predicates())],
        projection: doc_projection(),
        fusion,
        ..QueryIr::default()
    };

    // RRF: strong relevance wins; the undercut doc ranks first.
    let rrf = run(&store, &modality, query(FusionPolicy::Rrf { k: 60 }));
    assert_eq!(id_order(&rrf, "d.id"), vec!["doc:a", "doc:b"]);
    assert_eq!(rrf.trace.fusion, "rrf");

    // Cascade, epistemic ahead of relevance: trust dominates; the supported doc ranks first.
    let trust_first = run(
        &store,
        &modality,
        query(FusionPolicy::Cascade {
            rules: vec![
                RankingRule::EpistemicStatus,
                RankingRule::Vector,
                RankingRule::Text,
            ],
        }),
    );
    assert_eq!(id_order(&trust_first, "d.id"), vec!["doc:b", "doc:a"]);
    assert_eq!(trust_first.trace.fusion, "cascade");
    assert_eq!(
        trust_first
            .trace
            .cascade_rule_order
            .first()
            .map(String::as_str),
        Some("epistemic_status")
    );
    // The trace explains it: doc:b's EpistemicStatus bucket beats doc:a's.
    assert!(
        trust_first.trace.cascade_buckets["doc:b"][0]
            < trust_first.trace.cascade_buckets["doc:a"][0]
    );

    // Cascade, relevance ahead (the literal D2 sentence): Vector+Text bands tie, EpistemicStatus
    // breaks the tie, so the undercut doc ranks below the grounded one despite its lower vector.
    let relevance_first = run(
        &store,
        &modality,
        query(FusionPolicy::Cascade {
            rules: vec![
                RankingRule::Vector,
                RankingRule::Text,
                RankingRule::EpistemicStatus,
            ],
        }),
    );
    assert_eq!(id_order(&relevance_first, "d.id"), vec!["doc:b", "doc:a"]);
}

// --- shared fixtures --------------------------------------------------------------------------

fn two_ranker_fixture() -> (RelationalLens, InMemoryModality, Vec<&'static str>) {
    let mut store = RelationalLens::new();
    let mut modality = InMemoryModality::default();
    let rows = vec!["doc:a", "doc:b", "doc:c"];
    for (id, body, embedding) in [
        ("doc:a", "alpha", vec![1.0, 0.0]),
        ("doc:b", "alpha alpha alpha", vec![0.9, 0.1]),
        ("doc:c", "alpha alpha", vec![0.8, 0.2]),
    ] {
        store
            .upsert_row(scalar_row("docs", id, &[("body", text(body))]))
            .unwrap();
        modality.vector("docs", "embedding", id, embedding);
        modality.text("docs", "body", id, body);
    }
    (store, modality, rows)
}

fn two_ranker_predicates() -> Vec<Predicate> {
    vec![
        Predicate::Knn {
            column: "embedding".to_string(),
            query: vec![1.0, 0.0],
            k: 3,
        },
        Predicate::TextMatch {
            column: "body".to_string(),
            query: "alpha".to_string(),
            mode: PredicateMode::Rank,
        },
    ]
}

fn reciprocal_rank_fusion(lists: &[Vec<RankedRow>], k: f32) -> Vec<String> {
    let mut scores: BTreeMap<String, f32> = BTreeMap::new();
    for list in lists {
        for (index, row) in list.iter().enumerate() {
            *scores.entry(row.row_id.clone()).or_insert(0.0) += 1.0 / (k + index as f32 + 1.0);
        }
    }
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(id, _)| id).collect()
}
