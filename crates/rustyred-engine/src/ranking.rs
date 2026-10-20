//! Ranking access methods for the multimodal planner rank phase.
//!
//! Each method is a thin, stateless adapter: it declares which predicate it ranks
//! (`supports`), gives the planner a selection cost, and delegates the actual scoring to the
//! live [`ModalityResolver`] supplied at query time. No modality data (vectors, term postings,
//! coordinates, edges) is held here or copied into the relational store; the resolver resolves
//! everything by node id from the live index subsystems (TurboVec, full-text, spatial, graph).
//!
//! The residual exact-check helpers (`row_text_matches`, `row_geo_within`) stay here because the
//! planner runs them over the relation's own scalar/structured columns after an over-approximate
//! filter, exactly as the spec's "residual exact check in `row_matches`" requires.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::access_method::{
    AmResult, Cost, ModalityResolver, Predicate, PredicateMode, RankOutcome, RankingAccessMethod,
    RankingRegistry, RowId, ScalarValue,
};
use crate::graph::error::GraphStoreError;

impl RankingRegistry {
    pub fn with_native_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(VectorRankingMethod::new());
        registry.register(TextRankingMethod::new());
        registry.register(ExpandRankingMethod::new());
        registry
    }
}

/// kNN ranker (`vector`). Delegates to the live vector index (TurboVec) by node id.
#[derive(Clone, Copy, Debug, Default)]
pub struct VectorRankingMethod;

impl VectorRankingMethod {
    pub fn new() -> Self {
        Self
    }
}

impl RankingAccessMethod for VectorRankingMethod {
    fn name(&self) -> &'static str {
        "vector"
    }

    fn supports(&self, _relation: &str, predicate: &Predicate) -> bool {
        matches!(predicate, Predicate::Knn { .. })
    }

    fn cost(
        &self,
        _relation: &str,
        predicate: &Predicate,
        candidates: Option<usize>,
    ) -> Option<Cost> {
        let Predicate::Knn { k, .. } = predicate else {
            return None;
        };
        let rows = candidates.unwrap_or((*k).max(1)).max(1) as f64;
        Some(Cost::new((*k).max(1) as f64, rows))
    }

    fn rank(
        &self,
        relation: &str,
        predicate: &Predicate,
        candidates: Option<&BTreeSet<RowId>>,
        k: usize,
        resolver: &dyn ModalityResolver,
    ) -> AmResult<RankOutcome> {
        let Predicate::Knn { column, query, .. } = predicate else {
            return Ok(RankOutcome::default());
        };
        if k == 0 || query.iter().any(|value| !value.is_finite()) {
            return Ok(RankOutcome::default());
        }
        resolver.vector_knn(relation, column, query, candidates, k)
    }
}

/// Relevance ranker (`text_rank`). Delegates BM25 to the live full-text index by node id.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextRankingMethod;

impl TextRankingMethod {
    pub fn new() -> Self {
        Self
    }
}

impl RankingAccessMethod for TextRankingMethod {
    fn name(&self) -> &'static str {
        "text_rank"
    }

    fn supports(&self, _relation: &str, predicate: &Predicate) -> bool {
        matches!(
            predicate,
            Predicate::TextMatch {
                mode: PredicateMode::Rank,
                ..
            }
        )
    }

    fn cost(
        &self,
        _relation: &str,
        predicate: &Predicate,
        candidates: Option<usize>,
    ) -> Option<Cost> {
        if !self.supports("", predicate) {
            return None;
        }
        let rows = candidates.unwrap_or(1).max(1) as f64;
        Some(Cost::new(rows, rows.log2() + 1.0))
    }

    fn rank(
        &self,
        relation: &str,
        predicate: &Predicate,
        candidates: Option<&BTreeSet<RowId>>,
        k: usize,
        resolver: &dyn ModalityResolver,
    ) -> AmResult<RankOutcome> {
        let Predicate::TextMatch { column, query, .. } = predicate else {
            return Ok(RankOutcome::default());
        };
        if k == 0 {
            return Ok(RankOutcome::default());
        }
        Ok(RankOutcome::scored(
            resolver.text_rank(relation, column, query, candidates, k)?,
        ))
    }
}

/// Proximity ranker (`expand_ppr`). Delegates hop-distance / PPR to the live graph by node id.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandRankingMethod;

impl ExpandRankingMethod {
    pub fn new() -> Self {
        Self
    }
}

impl RankingAccessMethod for ExpandRankingMethod {
    fn name(&self) -> &'static str {
        "expand_ppr"
    }

    fn supports(&self, _relation: &str, predicate: &Predicate) -> bool {
        matches!(
            predicate,
            Predicate::Expand {
                mode: PredicateMode::Rank,
                ..
            }
        )
    }

    fn cost(
        &self,
        _relation: &str,
        predicate: &Predicate,
        candidates: Option<usize>,
    ) -> Option<Cost> {
        if !self.supports("", predicate) {
            return None;
        }
        let rows = candidates.unwrap_or(64).max(1) as f64;
        Some(Cost::new(rows, rows * 0.5))
    }

    fn rank(
        &self,
        _relation: &str,
        predicate: &Predicate,
        candidates: Option<&BTreeSet<RowId>>,
        k: usize,
        resolver: &dyn ModalityResolver,
    ) -> AmResult<RankOutcome> {
        let Predicate::Expand {
            from, layers, dir, ..
        } = predicate
        else {
            return Ok(RankOutcome::default());
        };
        if layers.len() != 1 {
            return Err(GraphStoreError::new(
                "multi_layer_rank_undefined",
                format!(
                    "rank mode requires exactly one layer; scores from different layers are not comparable (got {} layers)",
                    layers.len()
                ),
            ));
        }
        if k == 0 {
            return Ok(RankOutcome::default());
        }
        Ok(RankOutcome::scored(resolver.expand_proximity(
            from,
            layers,
            dir.clone(),
            candidates,
            k,
        )?))
    }
}

/// Residual term-presence check over the row's own scalar/structured text column. Run by the
/// planner after the over-approximate full-text filter narrows the candidate set.
pub(crate) fn row_text_matches(
    row: &crate::relational::ResolvedRow,
    column: &str,
    query: &str,
) -> bool {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return false;
    }
    let text = row
        .values
        .get(column)
        .and_then(ScalarValue::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            row.properties
                .get(column)
                .and_then(Value::as_str)
                .map(ToString::to_string)
        });
    let Some(text) = text else {
        return false;
    };
    let row_tokens = tokenize(&text);
    query_tokens
        .iter()
        .all(|query_token| row_tokens.contains(query_token))
}

/// Resolve a row's coordinates for the residual geo check, run by the planner after the
/// over-approximate spatial filter (so a point inside a boundary cell but outside the exact bbox is
/// excluded). Explicit `lat_property`/`lon_property` (the spatial designation) take precedence;
/// otherwise fall back to a `{column}` `{lat,lon}` object, `{column}_lat`/`{column}_lon`, or bare
/// `lat`/`lon` for the conventional `geo` column.
pub(crate) fn row_coords(
    row: &crate::relational::ResolvedRow,
    column: &str,
    lat_property: Option<&str>,
    lon_property: Option<&str>,
) -> Option<(f64, f64)> {
    if let (Some(lat_key), Some(lon_key)) = (lat_property, lon_property) {
        return Some((coord_field(row, lat_key)?, coord_field(row, lon_key)?));
    }
    if let Some(obj) = row.properties.get(column).and_then(Value::as_object) {
        let lat = obj.get("lat").and_then(Value::as_f64)?;
        let lon = obj.get("lon").and_then(Value::as_f64)?;
        return Some((lat, lon));
    }
    let lat = coord_field(row, &format!("{column}_lat"))
        .or_else(|| (column == "geo").then(|| coord_field(row, "lat")).flatten())?;
    let lon = coord_field(row, &format!("{column}_lon"))
        .or_else(|| (column == "geo").then(|| coord_field(row, "lon")).flatten())?;
    Some((lat, lon))
}

fn coord_field(row: &crate::relational::ResolvedRow, key: &str) -> Option<f64> {
    row.values
        .get(key)
        .and_then(ScalarValue::as_f64)
        .or_else(|| row.properties.get(key).and_then(Value::as_f64))
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

// === Ranking-rule cascade (SPEC-RUSTYRED-RANKING-CASCADE) =====================================
//
// A Meilisearch-style bucket-sort: ordered rules, each only breaking ties left by the prior rule.
// Each rule maps a candidate to a bucket ordinal (lower is better) whose maximum is a query/config
// constant, not a dataset rank, so a candidate's position is explainable independent of the index.
// `None` from a rule excludes the candidate. The epistemic rules are the differentiator and are
// reorderable: trust ahead of relevance demotes a contradicted-but-strong result below a supported
// one. Both search paths fill `RankCandidate` from their own row types and share `apply_cascade`.

// Banding constants. These are deliberate fixed grains, not dataset quantiles.
const BANDS: u32 = 10; // similarity/relevance/reliability score bands (0 = best)
const MAX_HOPS: u32 = 6; // graph-proximity hop cap
const PROX_CAP: u32 = 8; // per-adjacent-pair proximity penalty cap (Meilisearch default)
const ATTR_CAP: u32 = 64; // first-match position cap for the single-attribute model
const RECENCY_WORST: u32 = BANDS + 1; // superseded lands strictly below every aged/unknown band
                                      // ponytail: weekly recency grain; tune AGE_BAND_MS, or pass a per-query value, if recency matters.
const AGE_BAND_MS: i64 = 7 * 24 * 3_600 * 1_000;

/// One ranking rule. Lexical UX rules (the forgiving instant-search feel), relevance rules
/// (bucketized ranker scores), and epistemic rules (the graph/trust differentiator). Reorderable:
/// the cascade takes the order as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankingRule {
    Words,
    Typo,
    Proximity,
    Attribute,
    Exactness,
    Vector,
    Text,
    GraphProximity,
    SourceReliability,
    Recency,
    EpistemicStatus,
    Feature,
}

impl RankingRule {
    /// Stable name for the trace ("ranked here because rule R put it in bucket N").
    pub fn name(self) -> &'static str {
        match self {
            Self::Words => "words",
            Self::Typo => "typo",
            Self::Proximity => "proximity",
            Self::Attribute => "attribute",
            Self::Exactness => "exactness",
            Self::Vector => "vector",
            Self::Text => "text",
            Self::GraphProximity => "graph_proximity",
            Self::SourceReliability => "source_reliability",
            Self::Recency => "recency",
            Self::EpistemicStatus => "epistemic_status",
            Self::Feature => "feature",
        }
    }

    /// Bucket this candidate under this rule. `None` excludes it (retracted/acquaintance under
    /// `EpistemicStatus`; a hard temporal cutoff under `Recency`). A rule whose signal is absent
    /// places the candidate in its worst bucket rather than panicking, so a partial candidate
    /// degrades instead of failing.
    pub fn bucket(self, c: &RankCandidate, q: &QueryContext) -> Option<u32> {
        let terms = q.terms.len() as u32;
        Some(match self {
            Self::Words => match &c.term_match {
                Some(tm) => terms.saturating_sub(tm.matched_terms.min(terms)),
                None => terms,
            },
            Self::Typo => match &c.term_match {
                Some(tm) => tm.typos_per_term.iter().sum::<u32>().min(terms * 2),
                None => terms * 2,
            },
            Self::Proximity => {
                let worst = terms.saturating_sub(1) * PROX_CAP;
                match &c.term_match {
                    Some(tm) => proximity(&tm.positions).min(worst),
                    None => worst,
                }
            }
            Self::Attribute => match &c.term_match {
                Some(tm) => tm
                    .positions
                    .iter()
                    .min()
                    .copied()
                    .unwrap_or(ATTR_CAP)
                    .min(ATTR_CAP),
                None => ATTR_CAP,
            },
            Self::Exactness => match &c.term_match {
                Some(tm) => terms.saturating_sub(tm.exact_terms.min(terms)),
                None => terms,
            },
            Self::Vector => c.vector_score.map(|s| band(s, BANDS)).unwrap_or(BANDS),
            // bm25 is unbounded >= 0; squash to [0,1) so the band is index-independent.
            Self::Text => c
                .bm25_score
                .map(|s| band(s / (s + 1.0), BANDS))
                .unwrap_or(BANDS),
            Self::GraphProximity => c.graph_hops.map(|h| h.min(MAX_HOPS)).unwrap_or(MAX_HOPS),
            Self::SourceReliability => c
                .source_reliability
                .map(|s| band(s, BANDS))
                .unwrap_or(BANDS),
            Self::Recency => return recency_bucket(c, q),
            Self::EpistemicStatus => return epistemic_bucket(c, q),
            Self::Feature => c.feature_score.map(|s| band(s, BANDS)).unwrap_or(BANDS),
        })
    }
}

/// Bi-temporal recency band: superseded to the worst band, then newer-is-better when a reference
/// time is supplied. A hard `as_of` cutoff excludes a not-yet-valid candidate (`None`).
fn recency_bucket(c: &RankCandidate, q: &QueryContext) -> Option<u32> {
    if let (Some(as_of), Some(valid_from)) = (q.as_of_ms, c.valid_from_ms) {
        if valid_from > as_of {
            return None; // not yet valid at the cutoff
        }
    }
    if c.superseded {
        return Some(RECENCY_WORST);
    }
    match q.recency_reference_ms {
        Some(reference) => Some(match c.valid_from_ms {
            Some(valid_from) => {
                let age = (reference - valid_from).max(0);
                ((age / AGE_BAND_MS).min(BANDS as i64)) as u32
            }
            None => BANDS, // unknown valid-time: below any aged band, above superseded
        }),
        // No reference clock: order superseded-last, present-time over unknown-time.
        None => Some(match c.valid_from_ms {
            Some(_) => 0,
            None => 1,
        }),
    }
}

/// Epistemic standing band: supported above undercut/contested, retracted and acquaintance
/// excluded, provisional/contested gated by config (mirrors the web path's Stage 10). The
/// acceptance status sets the primary tier; the epistemic weight (axiomatic 1.5x, explanatory 1.2x)
/// is a sub-band within the tier, so a stronger-warranted result ranks ahead of a peer.
fn epistemic_bucket(c: &RankCandidate, q: &QueryContext) -> Option<u32> {
    if c.epistemic_weight <= 0.0 {
        return None; // acquaintance: dropped, same as the web path
    }
    let status = c.acceptance_status.trim().to_lowercase();
    match status.as_str() {
        "retracted" => return None,
        "contested" | "disputed" if !q.epistemic.allow_contested => return None,
        "provisional" | "hypothesis" if !q.epistemic.allow_provisional => return None,
        _ => {}
    }
    let status_tier = match status.as_str() {
        "axiomatic" => 0,
        "accepted" | "supported" | "grounded" | "explanatory" => 1,
        "hypothesis" | "provisional" => 2,
        "contested" | "disputed" | "undercut" | "undercuts" | "attacked" => 3,
        "downgraded" | "invalidated" => 3,
        // empty / neutral / provisional / unknown
        _ => 2,
    };
    let weight_tier = if c.epistemic_weight >= 1.5 {
        0
    } else if c.epistemic_weight >= 1.2 {
        1
    } else if c.epistemic_weight >= 1.0 {
        2
    } else {
        3
    };
    Some(status_tier * 4 + weight_tier)
}

/// Map a [0,1] score to a band (0 = best). Higher score, lower bucket. Out-of-range clamps.
fn band(value01: f32, bands: u32) -> u32 {
    let v = value01.clamp(0.0, 1.0);
    (((1.0 - v) * bands as f32).floor() as u32).min(bands)
}

/// Sum of per-adjacent-pair gaps between matched term positions, each capped at `PROX_CAP`.
/// Adjacent terms (gap of one) cost nothing; far-apart terms cost more. Fewer than two positions
/// has no proximity penalty.
fn proximity(positions: &[u32]) -> u32 {
    if positions.len() < 2 {
        return 0;
    }
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    sorted
        .windows(2)
        .map(|w| (w[1] - w[0]).saturating_sub(1).min(PROX_CAP))
        .sum()
}

/// Per-query knobs the rules read: the tokenized query terms (`Words`/`Typo`/`Proximity`/
/// `Exactness`), typo tolerance, the recency reference clock and hard temporal cutoff, and the
/// provisional/contested admission gate.
#[derive(Clone, Debug, Default)]
pub struct QueryContext {
    pub terms: Vec<String>,
    pub typo: TypoConfig,
    pub recency_reference_ms: Option<i64>,
    pub as_of_ms: Option<i64>,
    pub epistemic: EpistemicGate,
}

impl QueryContext {
    /// Build a context from a raw query string (tokenized + lowercased), default typo + gate.
    pub fn from_query(query: &str) -> Self {
        Self {
            terms: tokenize(query),
            ..Self::default()
        }
    }
}

/// Typo tolerance, Meilisearch defaults: one typo allowed at five or more characters, two at nine
/// or more. Prefix matching and typo tolerance both apply to the last query term.
#[derive(Clone, Copy, Debug)]
pub struct TypoConfig {
    pub enabled: bool,
    pub one_typo_min_len: usize,
    pub two_typo_min_len: usize,
}

impl Default for TypoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            one_typo_min_len: 5,
            two_typo_min_len: 9,
        }
    }
}

/// Whether provisional / contested results are admitted (demoted) or dropped. Default drops both,
/// matching the web path's Stage 10 unless the caller opts in.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpistemicGate {
    pub allow_provisional: bool,
    pub allow_contested: bool,
}

/// Term-level match detail for the lexical UX rules, derived from the document text (D3). The two
/// paths compute this from their own text column via [`compute_term_match`].
#[derive(Clone, Debug, Default)]
pub struct TermMatch {
    pub matched_terms: u32,
    pub typos_per_term: Vec<u32>,
    pub positions: Vec<u32>,
    pub exact_terms: u32,
}

/// A candidate carrying every signal the rules read. Each path fills this from its own row type;
/// rules never touch a path-specific struct. `epistemic_weight` defaults to non-acquaintance and
/// `acceptance_status` to neutral, so a row that carries no epistemic columns is ranked, not
/// dropped.
#[derive(Clone, Debug)]
pub struct RankCandidate {
    pub row_id: RowId,
    pub term_match: Option<TermMatch>,
    pub vector_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub graph_hops: Option<u32>,
    pub source_reliability: Option<f32>,
    pub valid_from_ms: Option<i64>,
    pub superseded: bool,
    pub epistemic_weight: f32,
    pub acceptance_status: String,
    pub feature_score: Option<f32>,
}

impl RankCandidate {
    /// A bare candidate keyed by id with neutral epistemic standing and no relevance signals.
    pub fn new(row_id: impl Into<RowId>) -> Self {
        Self {
            row_id: row_id.into(),
            term_match: None,
            vector_score: None,
            bm25_score: None,
            graph_hops: None,
            source_reliability: None,
            valid_from_ms: None,
            superseded: false,
            epistemic_weight: 1.0,
            acceptance_status: String::new(),
            feature_score: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureRankingConfig {
    pub enabled: bool,
    pub weight: f32,
}

impl Default for FeatureRankingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 1.0,
        }
    }
}

pub fn apply_feature_scores<I>(
    mut candidates: Vec<RankCandidate>,
    scores: I,
    config: FeatureRankingConfig,
) -> Vec<RankCandidate>
where
    I: IntoIterator<Item = (RowId, f32)>,
{
    if !config.enabled {
        return candidates;
    }
    let scores = scores
        .into_iter()
        .map(|(row_id, score)| (row_id, (score * config.weight.max(0.0)).clamp(0.0, 1.0)))
        .collect::<BTreeMap<_, _>>();
    for candidate in &mut candidates {
        if let Some(score) = scores.get(&candidate.row_id) {
            candidate.feature_score = Some(*score);
        }
    }
    candidates
}

/// A candidate in final order, carrying the bucket it landed in under each rule (rule order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankedCandidate {
    pub row_id: RowId,
    pub buckets: Vec<u32>,
}

/// The cascade result: candidates in final order plus the rule order actually applied (for the
/// trace).
#[derive(Clone, Debug, Default)]
pub struct CascadeOutcome {
    pub ranked: Vec<RankedCandidate>,
    pub rule_order: Vec<String>,
}

/// Apply the cascade: compute each candidate's bucket vector, drop any candidate a rule excluded,
/// and sort lexicographically by the bucket vector with `row_id` as the final, stable tiebreak.
/// Reordering `rules` reorders the result; that is the property the spec proves.
pub fn apply_cascade(
    candidates: Vec<RankCandidate>,
    rules: &[RankingRule],
    query: &QueryContext,
) -> CascadeOutcome {
    let mut ranked = candidates
        .into_iter()
        .filter_map(|candidate| {
            let mut buckets = Vec::with_capacity(rules.len());
            for rule in rules {
                buckets.push(rule.bucket(&candidate, query)?); // None excludes the candidate
            }
            Some(RankedCandidate {
                row_id: candidate.row_id,
                buckets,
            })
        })
        .collect::<Vec<_>>();
    // Vec<u32> compares lexicographically: earlier rules dominate, later rules break ties.
    ranked.sort_by(|a, b| {
        a.buckets
            .cmp(&b.buckets)
            .then_with(|| a.row_id.cmp(&b.row_id))
    });
    CascadeOutcome {
        ranked,
        rule_order: rules.iter().map(|rule| rule.name().to_string()).collect(),
    }
}

/// Derive [`TermMatch`] for a document body against the query terms (D3). Exact, prefix (last term
/// only), and bounded-Levenshtein fuzzy matching with Meilisearch-default typo tolerance; positions
/// feed `Proximity`/`Attribute`, exact count feeds `Exactness`. Query terms are assumed normalized
/// (tokenized + lowercased), e.g. from [`QueryContext::from_query`].
//
// ponytail: in-crate bounded Levenshtein is the FTS arm today (matching the existing token-overlap
// residual); swap to tantivy's FuzzyTermQuery/PrefixQuery when the full-text index goes native.
pub fn compute_term_match(text: &str, query: &QueryContext) -> TermMatch {
    let doc = tokenize(text);
    let last = query.terms.len().saturating_sub(1);
    let mut out = TermMatch::default();
    for (i, term) in query.terms.iter().enumerate() {
        let max_typos = allowed_typos(term, &query.typo);
        if let Some((position, typos, exact)) = best_match(&doc, term, max_typos, i == last) {
            out.matched_terms += 1;
            out.typos_per_term.push(typos);
            out.positions.push(position as u32);
            if exact {
                out.exact_terms += 1;
            }
        }
    }
    out
}

fn allowed_typos(term: &str, cfg: &TypoConfig) -> u32 {
    if !cfg.enabled {
        return 0;
    }
    let len = term.chars().count();
    if len >= cfg.two_typo_min_len {
        2
    } else if len >= cfg.one_typo_min_len {
        1
    } else {
        0
    }
}

/// Best match of `term` over the doc tokens: returns (position, typos, is_exact). Exact equality
/// wins; on the last term a prefix expansion matches with zero typos but is not "exact"; otherwise
/// a bounded-Levenshtein match within `max_typos`. Lowest typo count wins, exact breaks ties.
fn best_match(
    doc: &[String],
    term: &str,
    max_typos: u32,
    is_last: bool,
) -> Option<(usize, u32, bool)> {
    let mut best: Option<(usize, u32, bool)> = None;
    for (position, token) in doc.iter().enumerate() {
        let found = if token == term {
            Some((0, true))
        } else if is_last && token.starts_with(term) {
            Some((0, false)) // prefix expansion: zero edits, but not an exact term
        } else {
            levenshtein(term, token, max_typos).and_then(|d| (d > 0).then_some((d, false)))
        };
        if let Some((typos, exact)) = found {
            let better = match &best {
                None => true,
                Some((_, best_typos, best_exact)) => {
                    typos < *best_typos || (typos == *best_typos && exact && !*best_exact)
                }
            };
            if better {
                best = Some((position, typos, exact));
            }
        }
    }
    best
}

/// Levenshtein edit distance, bounded: returns `None` early once the distance must exceed `max`.
fn levenshtein(a: &str, b: &str, max: u32) -> Option<u32> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if (a.len() as i64 - b.len() as i64).unsigned_abs() > max as u64 {
        return None;
    }
    let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i as u32 + 1];
        let mut row_min = cur[0];
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            let val = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            cur.push(val);
            row_min = row_min.min(val);
        }
        if row_min > max {
            return None; // every cell in this row already exceeds the budget
        }
        prev = cur;
    }
    let distance = prev[b.len()];
    (distance <= max).then_some(distance)
}

#[cfg(test)]
mod cascade_tests {
    use super::*;

    fn order(outcome: &CascadeOutcome) -> Vec<String> {
        outcome.ranked.iter().map(|c| c.row_id.clone()).collect()
    }

    fn cand(id: &str, bm25: f32, status: &str) -> RankCandidate {
        RankCandidate {
            bm25_score: Some(bm25),
            acceptance_status: status.to_string(),
            ..RankCandidate::new(id)
        }
    }

    #[test]
    fn feature_rule_is_default_off_and_scores_when_enabled() {
        let q = QueryContext::default();
        let candidates = vec![RankCandidate::new("a"), RankCandidate::new("b")];
        let base = apply_cascade(candidates.clone(), &[RankingRule::Feature], &q);
        let disabled = apply_feature_scores(
            candidates.clone(),
            [("b".to_string(), 1.0), ("a".to_string(), 0.2)],
            FeatureRankingConfig::default(),
        );
        let disabled_out = apply_cascade(disabled, &[RankingRule::Feature], &q);
        assert_eq!(base.ranked, disabled_out.ranked);

        let enabled = apply_feature_scores(
            candidates,
            [("b".to_string(), 1.0), ("a".to_string(), 0.2)],
            FeatureRankingConfig {
                enabled: true,
                weight: 1.0,
            },
        );
        let enabled_out = apply_cascade(enabled, &[RankingRule::Feature], &q);
        assert_eq!(order(&enabled_out), vec!["b", "a"]);
    }

    #[test]
    fn missing_source_reliability_keeps_ranking_fallback_reachable() {
        let q = QueryContext::default();
        let out = apply_cascade(
            vec![RankCandidate::new("a"), RankCandidate::new("b")],
            &[RankingRule::SourceReliability],
            &q,
        );

        assert_eq!(order(&out), vec!["a", "b"]);
    }

    // D1 acceptance 1: [Text, EpistemicStatus] excludes a high-bm25 retracted candidate and ranks
    // a lower-bm25 supported candidate first.
    #[test]
    fn retracted_excluded_and_supported_ranks_first() {
        let q = QueryContext::default();
        let strong_retracted = cand("a", 9.0, "retracted");
        let weak_supported = cand("b", 1.0, "supported");
        let out = apply_cascade(
            vec![strong_retracted, weak_supported],
            &[RankingRule::Text, RankingRule::EpistemicStatus],
            &q,
        );
        assert_eq!(order(&out), vec!["b"]);
    }

    // D1 acceptance 2: reordering the SAME candidates changes the order from one field. Trust-first
    // ranks the supported-but-weak doc above the strong-but-undercut one; text-first flips it.
    #[test]
    fn reordering_rules_changes_outcome() {
        let q = QueryContext::default();
        let supported_weak = cand("x", 1.0, "supported"); // trustworthy, weak text
        let undercut_strong = cand("y", 9.0, "undercuts"); // strong text, attacked

        let trust_first = apply_cascade(
            vec![supported_weak.clone(), undercut_strong.clone()],
            &[RankingRule::EpistemicStatus, RankingRule::Text],
            &q,
        );
        assert_eq!(order(&trust_first), vec!["x", "y"]);

        let text_first = apply_cascade(
            vec![supported_weak, undercut_strong],
            &[RankingRule::Text, RankingRule::EpistemicStatus],
            &q,
        );
        assert_eq!(order(&text_first), vec!["y", "x"]);
    }

    // D3 acceptance: a one-typo query matches and ranks below an exact match under Typo; disabling
    // typo tolerance excludes the fuzzy match; a prefix on the last term matches.
    #[test]
    fn typo_tolerance_and_prefix_term_match() {
        let doc = "the database guide";

        let typo = QueryContext {
            terms: vec!["databse".to_string()],
            ..QueryContext::default()
        };
        let tm = compute_term_match(doc, &typo);
        assert_eq!(tm.matched_terms, 1);
        assert_eq!(tm.typos_per_term, vec![1]);
        assert_eq!(tm.exact_terms, 0);

        let exact = QueryContext {
            terms: vec!["database".to_string()],
            ..QueryContext::default()
        };
        let tme = compute_term_match(doc, &exact);
        assert_eq!(tme.typos_per_term, vec![0]);
        assert_eq!(tme.exact_terms, 1);
        // Typo rule ranks the exact match (bucket 0) above the one-typo match (bucket 1).
        assert!(
            RankingRule::Typo.bucket(
                &RankCandidate {
                    term_match: Some(tme),
                    ..RankCandidate::new("e")
                },
                &exact
            ) < RankingRule::Typo.bucket(
                &RankCandidate {
                    term_match: Some(tm),
                    ..RankCandidate::new("t")
                },
                &typo
            )
        );

        let disabled = QueryContext {
            terms: vec!["databse".to_string()],
            typo: TypoConfig {
                enabled: false,
                ..TypoConfig::default()
            },
            ..QueryContext::default()
        };
        assert_eq!(compute_term_match(doc, &disabled).matched_terms, 0);

        let prefix = QueryContext {
            terms: vec!["data".to_string()],
            ..QueryContext::default()
        };
        let tmp = compute_term_match(doc, &prefix);
        assert_eq!(tmp.matched_terms, 1);
        assert_eq!(tmp.exact_terms, 0); // prefix expansion is not exact
    }
}
