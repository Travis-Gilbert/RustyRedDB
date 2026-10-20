//! Phase 5: BM25 full-text search.
//!
//! The default backend wraps the upstream `bm25` crate's in-memory search
//! engine. Tantivy remains an explicit optional backend for native, Lucene-style
//! indexing when the crate is built with the `tantivy` feature and selected via
//! `RUSTY_RED_FULLTEXT_BACKEND=tantivy`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use bm25::{Document as Bm25Document, Language, SearchEngine, SearchEngineBuilder};
use serde::{Deserialize, Serialize};

use crate::index_manifest::{
    IndexBackend, IndexBuildStatus, IndexCreatedBy, IndexKind, IndexManifest, IndexScope,
};
use crate::GraphStoreResult;

/// §P5-A pa5.3: env var that selects the fulltext backend at construction time.
pub const RUSTY_RED_FULLTEXT_BACKEND_ENV: &str = "RUSTY_RED_FULLTEXT_BACKEND";

// Canonical backend-name strings; the dispatcher accepts a few aliases per kind.
pub(crate) const FULLTEXT_BACKEND_BM25: &str = "bm25";
// Legacy aliases accepted for existing deployments. They now select the
// upstream `bm25` crate backend rather than the retired local scorer.
pub(crate) const FULLTEXT_BACKEND_HAND_ROLLED: &str = "hand_rolled";
pub(crate) const FULLTEXT_BACKEND_TANTIVY: &str = "tantivy";

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FullTextDesignation {
    pub label: String,
    pub property: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FullTextSearchBackend {
    Bm25,
    Tantivy,
}

impl FullTextSearchBackend {
    fn index_backend(self) -> IndexBackend {
        match self {
            Self::Bm25 => IndexBackend::Bm25,
            Self::Tantivy => IndexBackend::Tantivy,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FieldedFullTextDefinition {
    pub manifest_id: String,
    pub target_label: String,
    pub fields: Vec<String>,
    pub backend: FullTextSearchBackend,
}

impl FieldedFullTextDefinition {
    pub fn bm25(
        manifest_id: impl Into<String>,
        target_label: impl Into<String>,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            target_label: target_label.into(),
            fields: fields.into_iter().map(Into::into).collect(),
            backend: FullTextSearchBackend::Bm25,
        }
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} fielded full-text", self.target_label),
            IndexKind::FullText,
            self.backend.index_backend(),
            scope,
            self.target_label.clone(),
            created_by,
        )?
        .with_target_properties(self.fields.clone())?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

/// §P5-A pa5.1 + cross-cutting cc.3: the full-text designation is now a
/// required, owned field. The per-tenant map already keys by (label, property),
/// so wrapping the designation in `Option` was redundant and prone to silent
/// fall-throughs ("missing designation" only surfaced via `.unwrap_or_default`).
pub struct FullTextIndex {
    pub designation: FullTextDesignation,
    documents: BTreeMap<String, String>,
    engine_state: Mutex<Bm25EngineState>,
}

struct Bm25EngineState {
    engine: SearchEngine<String>,
    dirty: bool,
}

impl std::fmt::Debug for FullTextIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullTextIndex")
            .field("designation", &self.designation)
            .field("doc_count", &self.documents.len())
            .finish()
    }
}

/// §P5-A pa5.1: trait abstraction over the full-text storage layer. The
/// default `FullTextIndex` wraps the upstream `bm25` crate. A tantivy-backed
/// implementation lives behind the optional `tantivy` feature flag.
pub trait FullTextBackend: Send + Sync + std::fmt::Debug {
    /// Index (or re-index) a document under `doc_id` with the given text.
    fn upsert(&mut self, doc_id: &str, text: &str);
    /// Remove a document from the index.
    fn remove(&mut self, doc_id: &str);
    /// Return the top-k scored hits for the query.
    fn search(&self, query: &str, k: usize) -> Vec<(String, f32)>;
    /// Return the designation this backend was created for.
    fn designation(&self) -> &FullTextDesignation;
    /// Number of documents currently indexed.
    fn doc_count(&self) -> usize;
    /// Make all queued mutations durable and visible to searchers.
    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Constant-time operational accounting for diagnostics.
    fn diagnostics(&self) -> FullTextBackendDiagnostics;
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FullTextBackendDiagnostics {
    pub backend: String,
    pub document_count: usize,
    pub persistent: bool,
    pub path: Option<String>,
    pub pending_writes: bool,
    pub commits_since_open: u64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FieldedFullTextDocument {
    pub doc_id: String,
    pub fields: BTreeMap<String, String>,
}

impl FieldedFullTextDocument {
    pub fn new(doc_id: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            fields: BTreeMap::new(),
        }
    }

    pub fn with_field(mut self, field: impl Into<String>, text: impl Into<String>) -> Self {
        self.fields.insert(field.into(), text.into());
        self
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FullTextSnippet {
    pub field: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FieldedFullTextHit {
    pub doc_id: String,
    pub score: f32,
    pub field_scores: BTreeMap<String, f32>,
    pub snippets: Vec<FullTextSnippet>,
}

#[derive(Debug)]
pub struct FieldedFullTextIndex {
    designation_label: String,
    field_indexes: BTreeMap<String, FullTextIndex>,
    field_boosts: BTreeMap<String, f32>,
    raw_fields: BTreeMap<String, BTreeMap<String, String>>,
}

impl FieldedFullTextIndex {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            designation_label: label.into(),
            field_indexes: BTreeMap::new(),
            field_boosts: BTreeMap::new(),
            raw_fields: BTreeMap::new(),
        }
    }

    pub fn set_field_boost(&mut self, field: impl Into<String>, boost: f32) {
        self.field_boosts.insert(field.into(), boost.max(0.0));
    }

    pub fn upsert(&mut self, document: FieldedFullTextDocument) {
        let doc_id = document.doc_id.clone();
        let designation_label = self.designation_label.clone();
        self.remove(&doc_id);
        for (field, text) in &document.fields {
            let index_text = if is_code_field(field) {
                code_aware_terms(text).join(" ")
            } else {
                text.clone()
            };
            self.field_indexes
                .entry(field.clone())
                .or_insert_with(|| {
                    FullTextIndex::for_designation(FullTextDesignation {
                        label: designation_label.clone(),
                        property: field.clone(),
                    })
                })
                .upsert(&doc_id, &index_text);
        }
        self.raw_fields.insert(doc_id, document.fields);
    }

    pub fn remove(&mut self, doc_id: &str) {
        for index in self.field_indexes.values_mut() {
            index.remove(doc_id);
        }
        self.raw_fields.remove(doc_id);
    }

    /// Export fielded documents for durable snapshot reopen (no Tantivy required).
    pub fn export_documents(&self) -> Vec<FieldedFullTextDocument> {
        self.raw_fields
            .iter()
            .map(|(doc_id, fields)| FieldedFullTextDocument {
                doc_id: doc_id.clone(),
                fields: fields.clone(),
            })
            .collect()
    }

    /// Iterate fielded documents in stable `doc_id` order.
    pub fn iter_documents(&self) -> impl Iterator<Item = FieldedFullTextDocument> + '_ {
        self.raw_fields
            .iter()
            .map(|(doc_id, fields)| FieldedFullTextDocument {
                doc_id: doc_id.clone(),
                fields: fields.clone(),
            })
    }

    /// Field boosts used at query time (for snapshot round-trip).
    pub fn field_boosts(&self) -> &BTreeMap<String, f32> {
        &self.field_boosts
    }

    pub fn designation_label(&self) -> &str {
        &self.designation_label
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<FieldedFullTextHit> {
        let mut scores: BTreeMap<String, FieldedFullTextHit> = BTreeMap::new();
        for (field, index) in &self.field_indexes {
            let boost = self.field_boosts.get(field).copied().unwrap_or(1.0);
            for (doc_id, score) in index.search(query, k.saturating_mul(4).max(k)) {
                let weighted = score * boost;
                let hit = scores
                    .entry(doc_id.clone())
                    .or_insert_with(|| FieldedFullTextHit {
                        doc_id: doc_id.clone(),
                        score: 0.0,
                        field_scores: BTreeMap::new(),
                        snippets: Vec::new(),
                    });
                hit.score += weighted;
                hit.field_scores.insert(field.clone(), weighted);
            }
        }
        let mut hits = scores.into_values().collect::<Vec<_>>();
        for hit in &mut hits {
            hit.snippets = self.snippets(&hit.doc_id, query, 2);
        }
        sort_fulltext_hits(&mut hits, k);
        hits
    }

    pub fn phrase_search(&self, phrase: &str, k: usize) -> Vec<FieldedFullTextHit> {
        self.literal_scan(phrase, k, LiteralMode::Phrase)
    }

    pub fn prefix_search(&self, prefix: &str, k: usize) -> Vec<FieldedFullTextHit> {
        self.literal_scan(prefix, k, LiteralMode::Prefix)
    }

    pub fn fuzzy_search(
        &self,
        term: &str,
        max_distance: usize,
        k: usize,
    ) -> Vec<FieldedFullTextHit> {
        let query = term.to_ascii_lowercase();
        let mut hits = Vec::new();
        for (doc_id, fields) in &self.raw_fields {
            let mut score = 0.0;
            let mut field_scores = BTreeMap::new();
            for (field, text) in fields {
                let matched = tokenize(text)
                    .into_iter()
                    .chain(code_aware_terms(text))
                    .any(|token| edit_distance(&token, &query) <= max_distance);
                if matched {
                    let boosted = self.field_boosts.get(field).copied().unwrap_or(1.0);
                    score += boosted;
                    field_scores.insert(field.clone(), boosted);
                }
            }
            if score > 0.0 {
                hits.push(FieldedFullTextHit {
                    doc_id: doc_id.clone(),
                    score,
                    field_scores,
                    snippets: self.snippets(doc_id, term, 2),
                });
            }
        }
        sort_fulltext_hits(&mut hits, k);
        hits
    }

    fn literal_scan(&self, needle: &str, k: usize, mode: LiteralMode) -> Vec<FieldedFullTextHit> {
        let needle = needle.to_ascii_lowercase();
        let mut hits = Vec::new();
        for (doc_id, fields) in &self.raw_fields {
            let mut score = 0.0;
            let mut field_scores = BTreeMap::new();
            for (field, text) in fields {
                let matched = match mode {
                    LiteralMode::Phrase => text.to_ascii_lowercase().contains(&needle),
                    LiteralMode::Prefix => tokenize(text)
                        .into_iter()
                        .chain(code_aware_terms(text))
                        .any(|token| token.starts_with(&needle)),
                };
                if matched {
                    let boosted = self.field_boosts.get(field).copied().unwrap_or(1.0);
                    score += boosted;
                    field_scores.insert(field.clone(), boosted);
                }
            }
            if score > 0.0 {
                hits.push(FieldedFullTextHit {
                    doc_id: doc_id.clone(),
                    score,
                    field_scores,
                    snippets: self.snippets(doc_id, &needle, 2),
                });
            }
        }
        sort_fulltext_hits(&mut hits, k);
        hits
    }

    fn snippets(&self, doc_id: &str, query: &str, limit: usize) -> Vec<FullTextSnippet> {
        let mut snippets = Vec::new();
        let Some(fields) = self.raw_fields.get(doc_id) else {
            return snippets;
        };
        let terms = tokenize(query);
        for (field, text) in fields {
            let lower = text.to_ascii_lowercase();
            let hit = terms
                .iter()
                .find_map(|term| lower.find(term).map(|start| (start, term.len())))
                .or_else(|| {
                    let needle = query.to_ascii_lowercase();
                    lower.find(&needle).map(|start| (start, needle.len()))
                });
            if let Some((start, len)) = hit {
                let snippet_start = clamp_to_char_boundary(text, start.saturating_sub(40));
                let snippet_end = clamp_to_char_boundary(text, (start + len + 40).min(text.len()));
                snippets.push(FullTextSnippet {
                    field: field.clone(),
                    start,
                    end: start + len,
                    text: text[snippet_start..snippet_end].to_string(),
                });
                if snippets.len() >= limit {
                    break;
                }
            }
        }
        snippets
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LiteralMode {
    Phrase,
    Prefix,
}

impl FullTextIndex {
    pub fn for_designation(d: FullTextDesignation) -> Self {
        Self {
            designation: d,
            documents: BTreeMap::new(),
            engine_state: Mutex::new(Bm25EngineState {
                engine: empty_bm25_engine(),
                dirty: false,
            }),
        }
    }

    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    pub fn upsert(&mut self, doc_id: &str, text: &str) {
        self.documents.insert(doc_id.to_string(), text.to_string());
        self.mark_engine_dirty();
    }

    pub fn remove(&mut self, doc_id: &str) {
        if self.documents.remove(doc_id).is_some() {
            self.mark_engine_dirty();
        }
    }

    /// Return the top-k doc_ids ranked by BM25 against the query string.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if self.documents.is_empty() {
            return Vec::new();
        }
        let mut state = match self.engine_state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.engine = build_bm25_engine(&self.documents);
                state.dirty = false;
                self.engine_state.clear_poison();
                state
            }
        };
        if state.dirty {
            state.engine = build_bm25_engine(&self.documents);
            state.dirty = false;
        }
        let mut entries: Vec<(String, f32)> = state
            .engine
            .search(query, Option::<usize>::None)
            .into_iter()
            .map(|result| (result.document.id, result.score))
            .collect();
        entries.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        entries.truncate(k);
        entries
    }

    fn mark_engine_dirty(&mut self) {
        if self.engine_state.is_poisoned() {
            self.engine_state.clear_poison();
        }
        self.engine_state
            .get_mut()
            .expect("fulltext bm25 engine mutex poisoned")
            .dirty = true;
    }
}

fn build_bm25_engine(documents: &BTreeMap<String, String>) -> SearchEngine<String> {
    if documents.is_empty() {
        empty_bm25_engine()
    } else {
        SearchEngineBuilder::<String>::with_documents(
            Language::English,
            documents
                .iter()
                .map(|(id, contents)| Bm25Document::new(id.clone(), contents.clone())),
        )
        .build()
    }
}

fn empty_bm25_engine() -> SearchEngine<String> {
    SearchEngineBuilder::<String>::with_avgdl(1.0).build()
}

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "these",
    "those",
];

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .filter(|s| !STOP_WORDS.contains(&s.as_str()))
        .collect()
}

fn is_code_field(field: &str) -> bool {
    matches!(
        field,
        "name" | "symbol" | "symbol_path" | "path" | "file_path" | "body" | "docstring"
    )
}

fn code_aware_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for raw in text.split_whitespace() {
        let trimmed = raw.trim_matches(|c: char| {
            !(c.is_alphanumeric() || matches!(c, '_' | ':' | '.' | '/' | '-'))
        });
        if trimmed.is_empty() {
            continue;
        }
        terms.push(trimmed.to_ascii_lowercase());
        terms.extend(
            trimmed
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|part| !part.is_empty())
                .map(|part| part.to_ascii_lowercase()),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

fn sort_fulltext_hits(hits: &mut Vec<FieldedFullTextHit>, k: usize) {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    hits.truncate(k);
}

fn clamp_to_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right_chars = right.chars().collect::<Vec<_>>();
    let mut prev = (0..=right_chars.len()).collect::<Vec<_>>();
    let mut cur = vec![0; right_chars.len() + 1];
    for (i, left_ch) in left.chars().enumerate() {
        cur[0] = i + 1;
        for (j, right_ch) in right_chars.iter().enumerate() {
            let cost = usize::from(left_ch != *right_ch);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[right_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_ranks_relevant_doc_higher() {
        let mut idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        idx.upsert(
            "d1",
            "Rust is a systems programming language focused on safety.",
        );
        idx.upsert(
            "d2",
            "Python is a popular dynamic programming language for data science.",
        );
        idx.upsert("d3", "The graph database holds tenant snapshots.");

        let results = idx.search("rust programming", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "d1");
        // d2 should appear since it shares "programming"
        let ids: Vec<&str> = results.iter().map(|(i, _)| i.as_str()).collect();
        assert!(ids.contains(&"d2"));
    }

    #[test]
    fn search_recovers_rebuildable_engine_after_mutex_poison() {
        let mut idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        idx.upsert("d1", "rust graph engine");
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = idx.engine_state.lock().unwrap();
            panic!("poison rebuildable engine state");
        }));

        let results = idx.search("rust", 1);

        assert_eq!(results[0].0, "d1");
    }

    #[test]
    fn upsert_recovers_rebuildable_engine_after_mutex_poison() {
        let mut idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        idx.upsert("d1", "rust graph engine");
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = idx.engine_state.lock().unwrap();
            panic!("poison rebuildable engine state");
        }));

        idx.upsert("d2", "durable memory index");
        let results = idx.search("durable", 1);

        assert_eq!(results[0].0, "d2");
    }

    #[test]
    fn remove_excludes_doc_from_future_searches() {
        let mut idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        idx.upsert("d1", "alpha beta");
        idx.upsert("d2", "alpha");
        idx.remove("d1");
        let results = idx.search("alpha", 5);
        let ids: Vec<&str> = results.iter().map(|(i, _)| i.as_str()).collect();
        assert!(!ids.contains(&"d1"));
        assert!(ids.contains(&"d2"));
    }

    #[test]
    fn upsert_replaces_existing_text() {
        let mut idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        idx.upsert("d1", "knowledge graph database");
        // overwrite
        idx.upsert("d1", "weather forecast for tomorrow");
        let r = idx.search("knowledge", 5);
        assert!(r.is_empty());
        let r = idx.search("weather", 5);
        assert_eq!(r[0].0, "d1");
    }

    // §P5-A pa5.1 + cc.3: backend trait + designation-required guarantees.

    #[test]
    fn fulltext_index_implements_fulltext_backend_trait() {
        let designation = FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        };
        let mut backend: Box<dyn FullTextBackend> =
            Box::new(FullTextIndex::for_designation(designation));
        backend.upsert("d1", "the quick brown fox");
        backend.upsert("d2", "the lazy dog");
        let hits = backend.search("fox", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, "d1");
        assert_eq!(backend.doc_count(), 2);
    }

    #[test]
    fn designation_is_owned_value_not_optional() {
        let idx = FullTextIndex::for_designation(FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        });
        assert_eq!(idx.designation.label, "Doc");
        assert_eq!(idx.designation.property, "text");
    }

    // §P5-A pa5.3: env-switch factory. Tests go through the pure dispatcher so
    // they can run in parallel without mutating the global env.

    fn fixture_designation() -> FullTextDesignation {
        FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        }
    }

    #[test]
    fn make_fulltext_backend_defaults_to_bm25_crate() {
        let backend = make_fulltext_backend_from_value(fixture_designation(), "").unwrap();
        assert_eq!(backend.designation().label, "Doc");
        assert_eq!(backend.doc_count(), 0);
    }

    #[test]
    fn make_fulltext_backend_rejects_unknown_backend() {
        let err = make_fulltext_backend_from_value(fixture_designation(), "elastic")
            .expect_err("unknown backend should error");
        assert_eq!(err.code(), "unknown_fulltext_backend");
    }

    #[cfg(not(feature = "tantivy"))]
    #[test]
    fn make_fulltext_backend_errors_when_tantivy_requested_without_feature() {
        let err = make_fulltext_backend_from_value(fixture_designation(), "tantivy")
            .expect_err("tantivy without feature should error");
        assert_eq!(err.code(), "unknown_fulltext_backend");
        assert!(err.message().to_ascii_lowercase().contains("tantivy"));
    }
}

// §P5-A pa5.1: the upstream bm25 crate backend wired through the trait. The
// tantivy alternative lives in `crate::fulltext_tantivy` behind the feature.
impl FullTextBackend for FullTextIndex {
    fn upsert(&mut self, doc_id: &str, text: &str) {
        FullTextIndex::upsert(self, doc_id, text);
    }

    fn remove(&mut self, doc_id: &str) {
        FullTextIndex::remove(self, doc_id);
    }

    fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        FullTextIndex::search(self, query, k)
    }

    fn designation(&self) -> &FullTextDesignation {
        &self.designation
    }

    fn doc_count(&self) -> usize {
        FullTextIndex::doc_count(self)
    }

    fn diagnostics(&self) -> FullTextBackendDiagnostics {
        FullTextBackendDiagnostics {
            backend: FULLTEXT_BACKEND_BM25.to_string(),
            document_count: self.doc_count(),
            persistent: false,
            path: None,
            pending_writes: self
                .engine_state
                .lock()
                .map(|state| state.dirty)
                .unwrap_or(true),
            commits_since_open: 0,
        }
    }
}

/// §P5-A pa5.3: env-switch factory. Reads `RUSTY_RED_FULLTEXT_BACKEND` and
/// forwards to the pure dispatcher below. The pure dispatcher exists so unit
/// tests can exercise every code path without racing on a process-global env
/// var; this thin wrapper is the production entry point.
pub fn make_fulltext_backend(
    designation: FullTextDesignation,
) -> Result<Box<dyn FullTextBackend>, FullTextBackendError> {
    let raw = std::env::var(RUSTY_RED_FULLTEXT_BACKEND_ENV).unwrap_or_default();
    make_fulltext_backend_from_value(designation, &raw)
}

/// Construct the persistent Tantivy backend used for embedded memory
/// designations. It deliberately bypasses the process-global backend switch:
/// memory recall must not silently fall back to a heap-duplicating index.
pub fn make_persistent_fulltext_backend(
    designation: FullTextDesignation,
    path: &Path,
) -> Result<Box<dyn FullTextBackend>, FullTextBackendError> {
    #[cfg(feature = "tantivy")]
    {
        return Ok(Box::new(
            crate::fulltext_tantivy::TantivyFullTextBackend::open_or_create(designation, path)
                .map_err(FullTextBackendError::TantivyInit)?,
        ));
    }
    #[cfg(not(feature = "tantivy"))]
    {
        let _ = (designation, path);
        Err(FullTextBackendError::UnknownBackend(
            "persistent fulltext requires building with --features tantivy".to_string(),
        ))
    }
}

/// Pure dispatcher; takes the env-var value as an explicit parameter so unit
/// tests can run in parallel without mutating global state. Default ("" /
/// "bm25") returns the upstream bm25 crate backend. Legacy "hand_rolled" /
/// "hand-rolled" aliases are still accepted for existing deployments.
/// `"tantivy"` returns the tantivy-backed impl when compiled with
/// `--features tantivy`; otherwise an explicit error so the caller knows to
/// rebuild.
pub fn make_fulltext_backend_from_value(
    designation: FullTextDesignation,
    raw: &str,
) -> Result<Box<dyn FullTextBackend>, FullTextBackendError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | FULLTEXT_BACKEND_BM25 | FULLTEXT_BACKEND_HAND_ROLLED | "hand-rolled" => {
            Ok(Box::new(FullTextIndex::for_designation(designation)))
        }
        FULLTEXT_BACKEND_TANTIVY => {
            #[cfg(feature = "tantivy")]
            {
                Ok(Box::new(
                    crate::fulltext_tantivy::TantivyFullTextBackend::new(designation)
                        .map_err(FullTextBackendError::TantivyInit)?,
                ))
            }
            #[cfg(not(feature = "tantivy"))]
            {
                let _ = designation;
                Err(FullTextBackendError::UnknownBackend(
                    "tantivy backend requires building with --features tantivy".to_string(),
                ))
            }
        }
        other => Err(FullTextBackendError::UnknownBackend(format!(
            "unknown {RUSTY_RED_FULLTEXT_BACKEND_ENV} value: {other}"
        ))),
    }
}

#[derive(Debug, Clone)]
pub enum FullTextBackendError {
    UnknownBackend(String),
    TantivyInit(String),
}

impl FullTextBackendError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownBackend(_) => "unknown_fulltext_backend",
            Self::TantivyInit(_) => "tantivy_init_failed",
        }
    }
    pub fn message(&self) -> String {
        match self {
            Self::UnknownBackend(s) => s.clone(),
            Self::TantivyInit(s) => format!("tantivy initialization failed: {s}"),
        }
    }
}

impl std::fmt::Display for FullTextBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for FullTextBackendError {}
