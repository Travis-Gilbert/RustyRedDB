//! Trigram candidate index (Russ Cox candidate-then-verify design).
//!
//! The index answers one question and refuses to answer any other: which
//! documents *could* contain this query? It never claims a document does
//! contain it. Verification is the caller's job, and the exact lane in
//! `rustyred-thg-find` does it with `memchr`/`aho-corasick` over the candidate
//! documents only.
//!
//! Two properties are load-bearing:
//!
//! - **Superset guarantee.** `candidates(q)` always contains every document
//!   whose text contains `q`. A query with fewer than three bytes has no
//!   trigrams to constrain on, so the honest answer is `CandidateSet::All`
//!   rather than a silently empty set.
//! - **No document text is read.** Answering from posting lists alone is the
//!   whole point; a query whose trigrams are absent from the index resolves to
//!   the empty set without a single byte of document text being touched.
//!
//! Trigrams are indexed ASCII-case-folded. Folding only ever widens the
//! candidate set, so the superset guarantee survives, and it lets a
//! case-insensitive verifier reuse the same index.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::access_method::{
    AccessMethod, AccessMethodRow, AccessMethodStats, AmResult, ColumnId, Cost, Predicate,
    PredicateMode, RelationId, RowChange, RowChangeKind, RowId, RowIdStream, ScalarValue,
};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::index_manifest::{
    IndexBackend, IndexBuildStatus, IndexCreatedBy, IndexKind, IndexManifest, IndexScope,
};

/// Access-method name as it appears in planner traces and index manifests.
pub const TRIGRAM_ACCESS_METHOD: &str = "trigram";

/// Document identity in the trigram index. Node ids are strings everywhere in
/// this crate; the alias exists so the spec signature reads as written.
pub type NodeId = String;

/// A packed case-folded byte trigram. `[u8; 3]` in a `u32` keeps posting-list
/// keys `Ord` and cheap to compare.
type Trigram = u32;

fn pack(window: &[u8]) -> Trigram {
    ((window[0] as u32) << 16) | ((window[1] as u32) << 8) | (window[2] as u32)
}

/// Case-folded byte trigrams of `text`, deduplicated and sorted.
///
/// ASCII case folding is safe on UTF-8 bytes: the folded range `A-Z` never
/// overlaps a multi-byte continuation byte, so folding cannot corrupt a
/// codepoint boundary.
fn trigrams(text: &str) -> Vec<Trigram> {
    let folded = text.as_bytes().to_ascii_lowercase();
    if folded.len() < 3 {
        return Vec::new();
    }
    let mut out: Vec<Trigram> = folded.windows(3).map(pack).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The candidate answer. `All` is the honest response when the query carries
/// no trigram constraint; it is not a failure and it is not "everything
/// matched".
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidateSet {
    /// The query constrains nothing. Every document is a candidate.
    All,
    /// The query constrains to exactly these documents.
    Docs(BTreeSet<NodeId>),
}

impl CandidateSet {
    /// True when no document can match. Only `Docs` can be empty; `All` never is.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::All => false,
            Self::Docs(docs) => docs.is_empty(),
        }
    }

    /// Number of named candidates, or `None` when unconstrained.
    pub fn len(&self) -> Option<usize> {
        match self {
            Self::All => None,
            Self::Docs(docs) => Some(docs.len()),
        }
    }

    pub fn contains(&self, doc: &str) -> bool {
        match self {
            Self::All => true,
            Self::Docs(docs) => docs.contains(doc),
        }
    }

    /// Resolve against the full document population. `All` widens to every id.
    pub fn resolve<'a, I>(&self, population: I) -> BTreeSet<NodeId>
    where
        I: IntoIterator<Item = &'a str>,
    {
        match self {
            Self::All => population.into_iter().map(str::to_string).collect(),
            Self::Docs(docs) => docs.clone(),
        }
    }

    pub fn iter(&self) -> Option<impl Iterator<Item = &NodeId>> {
        match self {
            Self::All => None,
            Self::Docs(docs) => Some(docs.iter()),
        }
    }
}

/// Trigram to posting-list index over a document corpus.
#[derive(Clone, Debug, Default)]
pub struct TrigramIndex {
    postings: BTreeMap<Trigram, BTreeSet<NodeId>>,
    /// Per-document trigram set, so `remove` is proportional to the document's
    /// distinct trigrams rather than to the whole index.
    documents: BTreeMap<NodeId, Vec<Trigram>>,
}

impl TrigramIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index `text` under `doc`, replacing any previous text for that document.
    pub fn insert(&mut self, doc: NodeId, text: &str) {
        self.remove(doc.clone());
        let grams = trigrams(text);
        for gram in &grams {
            self.postings.entry(*gram).or_default().insert(doc.clone());
        }
        self.documents.insert(doc, grams);
    }

    /// Drop `doc` and every posting that referenced it.
    pub fn remove(&mut self, doc: NodeId) {
        let Some(grams) = self.documents.remove(&doc) else {
            return;
        };
        for gram in grams {
            let drained = match self.postings.get_mut(&gram) {
                Some(posting) => {
                    posting.remove(&doc);
                    posting.is_empty()
                }
                None => false,
            };
            if drained {
                self.postings.remove(&gram);
            }
        }
    }

    /// Documents that could contain `query`. Superset guarantee holds; no
    /// document text is read.
    pub fn candidates(&self, query: &str) -> CandidateSet {
        let grams = trigrams(query);
        if grams.is_empty() {
            return CandidateSet::All;
        }
        // Intersect the smallest posting list first: the accumulator can only
        // shrink, so starting small keeps the work proportional to the answer.
        let mut ordered: Vec<&BTreeSet<NodeId>> = Vec::with_capacity(grams.len());
        for gram in &grams {
            match self.postings.get(gram) {
                Some(posting) => ordered.push(posting),
                // One absent trigram is enough: nothing can match.
                None => return CandidateSet::Docs(BTreeSet::new()),
            }
        }
        ordered.sort_by_key(|posting| posting.len());

        let mut accumulator = ordered[0].clone();
        for posting in &ordered[1..] {
            accumulator.retain(|doc| posting.contains(doc));
            if accumulator.is_empty() {
                break;
            }
        }
        CandidateSet::Docs(accumulator)
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn posting_count(&self) -> usize {
        self.postings.len()
    }

    pub fn contains_document(&self, doc: &str) -> bool {
        self.documents.contains_key(doc)
    }
}

/// Adaptive-index-spine definition for a trigram index, mirroring the
/// `TemporalIndexDefinition` / `SpatialIndexDefinition` shape in
/// `adaptive_index`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TrigramIndexDefinition {
    pub manifest_id: String,
    pub target_label: String,
    pub text_property: String,
}

impl TrigramIndexDefinition {
    pub fn new(
        manifest_id: impl Into<String>,
        target_label: impl Into<String>,
        text_property: impl Into<String>,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            target_label: target_label.into(),
            text_property: text_property.into(),
        }
    }

    /// Fallible because the chunk-store refactor made manifest construction,
    /// target-property attachment, and hash refresh all validate. Mirrors
    /// `FullTextDesignation::to_manifest`, the sibling designation in this
    /// crate — a trigram manifest that silently skipped that validation would
    /// be the one index in the spine nobody could trust.
    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} trigram candidate index", self.target_label),
            IndexKind::Trigram,
            IndexBackend::RustyredCore,
            scope,
            self.target_label.clone(),
            created_by,
        )?
        .with_target_properties(vec![self.text_property.clone()])?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Default)]
struct TrigramState {
    indexes: BTreeMap<(RelationId, ColumnId), TrigramIndex>,
}

/// Planner-facing access method over per-(relation, column) trigram indexes.
///
/// It answers `Predicate::TextMatch` in filter mode only. Rank-mode text goes
/// to `TextRankingMethod` in `ranking`; substring candidacy and BM25 relevance
/// are different questions and must not be swapped by the cost model.
#[derive(Clone, Debug, Default)]
pub struct TrigramAccessMethod {
    state: Arc<Mutex<TrigramState>>,
    stats: Arc<Mutex<AccessMethodStats>>,
}

impl std::fmt::Debug for TrigramState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TrigramState")
            .field("indexes", &self.indexes.len())
            .finish()
    }
}

impl TrigramAccessMethod {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> AccessMethodStats {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }

    /// Candidate set for one (relation, column) without going through the
    /// planner. The exact lane uses this directly.
    pub fn candidates(&self, relation: &str, column: &str, query: &str) -> CandidateSet {
        self.state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .indexes
                    .get(&(relation.to_string(), column.to_string()))
                    .map(|index| index.candidates(query))
            })
            // No index for this column yet: unconstrained, never a false empty.
            .unwrap_or(CandidateSet::All)
    }

    pub fn document_count(&self, relation: &str, column: &str) -> usize {
        self.state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .indexes
                    .get(&(relation.to_string(), column.to_string()))
                    .map(TrigramIndex::document_count)
            })
            .unwrap_or(0)
    }
}

fn filter_text_match(predicate: &Predicate) -> Option<(&ColumnId, &String)> {
    match predicate {
        Predicate::TextMatch {
            column,
            query,
            mode: PredicateMode::Filter,
        } => Some((column, query)),
        _ => None,
    }
}

impl AccessMethod for TrigramAccessMethod {
    fn name(&self) -> &'static str {
        TRIGRAM_ACCESS_METHOD
    }

    fn supports(&self, _relation: &str, predicate: &Predicate) -> bool {
        filter_text_match(predicate).is_some()
    }

    fn cost(&self, relation: &str, predicate: &Predicate) -> Option<Cost> {
        let (column, query) = filter_text_match(predicate)?;
        let state = self.state.lock().ok()?;
        let index = state.indexes.get(&(relation.to_string(), column.clone()))?;
        let total = index.document_count();
        let grams = trigrams(query);
        if grams.is_empty() {
            // No constraint: the caller pays for the whole corpus.
            return Some(Cost::new(total.max(1) as f64, total.max(1) as f64));
        }
        // Each additional trigram narrows the answer; posting-list walks are
        // the only work.
        let selectivity = 1.0 / (grams.len() as f64 + 1.0);
        Some(Cost::new(
            (total as f64 * selectivity).max(1.0),
            grams.len() as f64 * (total.max(1) as f64).log2().max(1.0),
        ))
    }

    fn scan(&self, relation: &str, predicate: &Predicate) -> AmResult<RowIdStream> {
        let Some((column, query)) = filter_text_match(predicate) else {
            return Ok(RowIdStream::new(Vec::new(), 0));
        };
        let state = self.state.lock().map_err(poisoned)?;
        let Some(index) = state.indexes.get(&(relation.to_string(), column.clone())) else {
            return Ok(RowIdStream::new(Vec::new(), 0));
        };
        let rows: Vec<RowId> = match index.candidates(query) {
            // Unconstrained queries hand back every indexed document rather
            // than pretending the answer is empty.
            CandidateSet::All => index.documents.keys().cloned().collect(),
            CandidateSet::Docs(docs) => docs.into_iter().collect(),
        };
        let visited = rows.len();
        if let Ok(mut stats) = self.stats.lock() {
            stats.scans += 1;
            stats.rows_visited += visited;
        }
        Ok(RowIdStream::new(rows, visited))
    }

    fn rebuild_from_graph_rows(&self, rows: &[AccessMethodRow]) -> AmResult<()> {
        let mut rebuilt = TrigramState::default();
        for row in rows {
            for (column, value) in &row.values {
                let ScalarValue::String(text) = value else {
                    continue;
                };
                rebuilt
                    .indexes
                    .entry((row.relation.clone(), column.clone()))
                    .or_default()
                    .insert(row.row_id.clone(), text);
            }
        }
        *self.state.lock().map_err(poisoned)? = rebuilt;
        if let Ok(mut stats) = self.stats.lock() {
            stats.rebuilds += 1;
        }
        Ok(())
    }

    fn on_write(&self, change: &RowChange) -> AmResult<()> {
        if let Ok(mut stats) = self.stats.lock() {
            stats.write_updates += 1;
        }
        let mut state = self.state.lock().map_err(poisoned)?;
        if change.kind == RowChangeKind::Delete {
            for column in change.values.keys() {
                if let Some(index) = state
                    .indexes
                    .get_mut(&(change.relation.clone(), column.clone()))
                {
                    index.remove(change.row_id.clone());
                }
            }
            return Ok(());
        }
        for (column, value) in &change.values {
            let ScalarValue::String(text) = value else {
                continue;
            };
            state
                .indexes
                .entry((change.relation.clone(), column.clone()))
                .or_default()
                .insert(change.row_id.clone(), text);
        }
        Ok(())
    }
}

fn poisoned<T>(_: T) -> GraphStoreError {
    GraphStoreError::new(
        "trigram_index_poisoned",
        "trigram index mutex was poisoned".to_string(),
    )
}

/// Build a standalone index over `(doc, text)` pairs. Convenience for callers
/// that hold a corpus rather than a relational change stream.
pub fn build_index<I, D, T>(documents: I) -> TrigramIndex
where
    I: IntoIterator<Item = (D, T)>,
    D: Into<NodeId>,
    T: AsRef<str>,
{
    let mut index = TrigramIndex::new();
    for (doc, text) in documents {
        index.insert(doc.into(), text.as_ref());
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TrigramIndex {
        let mut index = TrigramIndex::new();
        index.insert("doc:a".to_string(), "the membrane admits candidates");
        index.insert("doc:b".to_string(), "candidate admission is gated");
        index.insert("doc:c".to_string(), "trigram indexes narrow the scan");
        index.insert("doc:d".to_string(), "");
        index
    }

    fn naive_scan<'a>(corpus: &[(&'a str, &'a str)], query: &str) -> BTreeSet<String> {
        let needle = query.to_ascii_lowercase();
        corpus
            .iter()
            .filter(|(_, text)| text.to_ascii_lowercase().contains(&needle))
            .map(|(doc, _)| (*doc).to_string())
            .collect()
    }

    const CORPUS: &[(&str, &str)] = &[
        ("doc:a", "the membrane admits candidates"),
        ("doc:b", "candidate admission is gated"),
        ("doc:c", "trigram indexes narrow the scan"),
        ("doc:d", ""),
    ];

    #[test]
    fn trigram_candidates_are_a_superset_of_naive_matches() {
        let index = fixture();
        for query in [
            "candidate",
            "admis",
            "the",
            "scan",
            "membrane admits",
            "gated",
            "trigram",
            "CANDIDATE",
            "absent phrase",
        ] {
            let truth = naive_scan(CORPUS, query);
            let candidates = index.candidates(query);
            for doc in &truth {
                assert!(
                    candidates.contains(doc),
                    "query {query:?} lost true match {doc} from {candidates:?}"
                );
            }
        }
    }

    #[test]
    fn trigram_candidates_absent_from_index_return_empty_without_reading_text() {
        let index = fixture();
        let candidates = index.candidates("zzqx-not-present");
        assert_eq!(candidates, CandidateSet::Docs(BTreeSet::new()));
        assert!(candidates.is_empty());
        assert_eq!(candidates.len(), Some(0));
    }

    #[test]
    fn trigram_short_query_is_unconstrained_not_empty() {
        let index = fixture();
        assert_eq!(index.candidates("ab"), CandidateSet::All);
        assert!(!index.candidates("ab").is_empty());
        assert_eq!(index.candidates("ab").len(), None);
    }

    #[test]
    fn trigram_insert_and_remove_are_reflected_in_candidates() {
        let mut index = fixture();
        assert!(index.candidates("candidate").contains("doc:b"));

        index.remove("doc:b".to_string());
        assert!(!index.candidates("candidate").contains("doc:b"));
        assert!(!index.contains_document("doc:b"));

        index.insert("doc:e".to_string(), "a fresh candidate arrives");
        assert!(index.candidates("candidate").contains("doc:e"));

        // Re-inserting replaces rather than accumulating.
        index.insert("doc:e".to_string(), "nothing to see");
        assert!(!index.candidates("candidate").contains("doc:e"));
        assert!(index.candidates("nothing").contains("doc:e"));
    }

    #[test]
    fn trigram_remove_drains_empty_postings() {
        let mut index = TrigramIndex::new();
        index.insert("only".to_string(), "solitary");
        assert!(index.posting_count() > 0);
        index.remove("only".to_string());
        assert_eq!(index.posting_count(), 0);
        assert_eq!(index.document_count(), 0);
    }

    #[test]
    fn trigram_index_is_case_folded() {
        let mut index = TrigramIndex::new();
        index.insert("doc:a".to_string(), "Membrane");
        assert!(index.candidates("membrane").contains("doc:a"));
        assert!(index.candidates("MEMBRANE").contains("doc:a"));
    }

    #[test]
    fn trigram_access_method_scans_from_postings() {
        let method = TrigramAccessMethod::new();
        for (doc, text) in CORPUS {
            method
                .on_write(&RowChange {
                    relation: "Item".to_string(),
                    row_id: (*doc).to_string(),
                    values: BTreeMap::from([(
                        "body".to_string(),
                        ScalarValue::String((*text).to_string()),
                    )]),
                    properties: serde_json::Value::Null,
                    kind: RowChangeKind::Upsert,
                })
                .expect("write");
        }
        let predicate = Predicate::TextMatch {
            column: "body".to_string(),
            query: "candidate".to_string(),
            mode: PredicateMode::Filter,
        };
        assert!(method.supports("Item", &predicate));
        assert!(method.cost("Item", &predicate).is_some());
        let stream = method.scan("Item", &predicate).expect("scan");
        assert!(stream.rows.contains(&"doc:a".to_string()));
        assert!(stream.rows.contains(&"doc:b".to_string()));
        assert!(!stream.rows.contains(&"doc:c".to_string()));
        assert_eq!(method.stats().scans, 1);
    }

    #[test]
    fn trigram_access_method_declines_rank_mode_text() {
        let method = TrigramAccessMethod::new();
        let predicate = Predicate::TextMatch {
            column: "body".to_string(),
            query: "candidate".to_string(),
            mode: PredicateMode::Rank,
        };
        assert!(!method.supports("Item", &predicate));
        assert!(method.cost("Item", &predicate).is_none());
    }

    #[test]
    fn trigram_access_method_handles_delete() {
        let method = TrigramAccessMethod::new();
        let values = BTreeMap::from([(
            "body".to_string(),
            ScalarValue::String("candidate text".to_string()),
        )]);
        method
            .on_write(&RowChange {
                relation: "Item".to_string(),
                row_id: "doc:a".to_string(),
                values: values.clone(),
                properties: serde_json::Value::Null,
                kind: RowChangeKind::Upsert,
            })
            .expect("insert");
        assert_eq!(method.document_count("Item", "body"), 1);
        method
            .on_write(&RowChange {
                relation: "Item".to_string(),
                row_id: "doc:a".to_string(),
                values,
                properties: serde_json::Value::Null,
                kind: RowChangeKind::Delete,
            })
            .expect("delete");
        assert_eq!(method.document_count("Item", "body"), 0);
    }

    #[test]
    fn trigram_definition_mints_an_active_manifest() {
        let manifest = TrigramIndexDefinition::new("idx:trigram:item", "Item", "body")
            .to_manifest(IndexScope::Tenant, IndexCreatedBy::System)
            .expect("a trigram manifest over a valid definition passes validation");
        assert_eq!(manifest.kind, IndexKind::Trigram);
        assert_eq!(manifest.backend, IndexBackend::RustyredCore);
        assert_eq!(manifest.build_status, IndexBuildStatus::Active);
        assert_eq!(manifest.target_properties, vec!["body".to_string()]);
    }

    #[test]
    fn trigram_build_index_helper_matches_manual_inserts() {
        let built = build_index(CORPUS.iter().copied());
        let manual = fixture();
        assert_eq!(built.document_count(), manual.document_count());
        assert_eq!(
            built.candidates("candidate"),
            manual.candidates("candidate")
        );
    }
}
