//! §P5-A pa5.2: tantivy-backed full-text index. Behind the `tantivy` feature
//! flag. Selected at runtime via `RUSTY_RED_FULLTEXT_BACKEND=tantivy` through
//! the `crate::fulltext::make_fulltext_backend` factory.
//!
//! The upstream `bm25` crate wrapper in `fulltext.rs` remains the default;
//! tantivy is the perf-oriented alternative the original SPEC named. Both
//! implement `FullTextBackend` so the rest of the system reads them through a
//! uniform interface.
//!
//! Embedded memory designations use `open_or_create` over `MmapDirectory`.
//! Mutations queue in the writer until `FullTextBackend::flush`. Persistent
//! recovery indexes disable Tantivy's automatic background merge policy so a
//! completed commit cannot trigger an unbounded post-warm merge allocation.

use std::path::{Path, PathBuf};

use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    doc,
    merge_policy::NoMergePolicy,
    query::QueryParser,
    schema::{Field, Schema, Value, STORED, STRING, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term,
};

use crate::fulltext::{FullTextBackend, FullTextBackendDiagnostics, FullTextDesignation};

/// Tantivy 0.22's documented minimum is 15 MB; we sit right at that floor so
/// many concurrent designations don't multiply by a much larger per-writer
/// arena. A future bulk-upsert path can share writers across designations.
const WRITER_MEMORY_BUDGET: usize = 15_000_000;

pub struct TantivyFullTextBackend {
    designation: FullTextDesignation,
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    doc_id_field: Field,
    text_field: Field,
    storage_path: Option<PathBuf>,
    dirty: bool,
    pending_error: Option<String>,
    commits_since_open: u64,
}

impl std::fmt::Debug for TantivyFullTextBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyFullTextBackend")
            .field("designation", &self.designation)
            .field("storage_path", &self.storage_path)
            .field("dirty", &self.dirty)
            .finish_non_exhaustive()
    }
}

impl TantivyFullTextBackend {
    /// Returns either an initialized backend or a human-readable init error
    /// string (the factory wraps this in `FullTextBackendError::TantivyInit`).
    pub fn new(designation: FullTextDesignation) -> Result<Self, String> {
        let (schema, doc_id_field, text_field) = fulltext_schema();
        let index = Index::create_in_ram(schema);
        Self::from_index(designation, index, doc_id_field, text_field, None)
    }

    pub fn open_or_create(designation: FullTextDesignation, path: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(path)
            .map_err(|err| format!("could not create tantivy directory: {err}"))?;
        let (schema, doc_id_field, text_field) = fulltext_schema();
        let directory = MmapDirectory::open(path)
            .map_err(|err| format!("could not open tantivy mmap directory: {err}"))?;
        let index = Index::open_or_create(directory, schema)
            .map_err(|err| format!("could not open or create tantivy index: {err}"))?;
        Self::from_index(
            designation,
            index,
            doc_id_field,
            text_field,
            Some(path.to_path_buf()),
        )
    }

    fn from_index(
        designation: FullTextDesignation,
        index: Index,
        doc_id_field: Field,
        text_field: Field,
        storage_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let writer = index
            .writer(WRITER_MEMORY_BUDGET)
            .map_err(|err| format!("could not create tantivy IndexWriter: {err}"))?;
        if storage_path.is_some() {
            writer.set_merge_policy(Box::new(NoMergePolicy));
        }
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|err| format!("could not build tantivy reader: {err}"))?;
        Ok(Self {
            designation,
            index,
            writer,
            reader,
            doc_id_field,
            text_field,
            storage_path,
            dirty: false,
            pending_error: None,
            commits_since_open: 0,
        })
    }

    /// Delete any existing doc with the given id; STRING tokenization on
    /// `doc_id_field` keeps the term match exact.
    fn delete_doc_id(&mut self, doc_id: &str) {
        let term = Term::from_field_text(self.doc_id_field, doc_id);
        let _ = self.writer.delete_term(term);
    }

    fn commit_pending(&mut self) -> Result<(), String> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        if !self.dirty {
            return Ok(());
        }
        self.writer
            .commit()
            .map_err(|err| format!("could not commit tantivy index: {err}"))?;
        self.reader
            .reload()
            .map_err(|err| format!("could not reload tantivy reader: {err}"))?;
        self.dirty = false;
        self.commits_since_open = self.commits_since_open.saturating_add(1);
        Ok(())
    }
}

fn fulltext_schema() -> (Schema, Field, Field) {
    let mut schema_builder = Schema::builder();
    let doc_id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
    let text_field = schema_builder.add_text_field("text", TEXT);
    (schema_builder.build(), doc_id_field, text_field)
}

impl FullTextBackend for TantivyFullTextBackend {
    fn upsert(&mut self, doc_id: &str, text: &str) {
        self.delete_doc_id(doc_id);
        if let Err(error) = self.writer.add_document(doc!(
            self.doc_id_field => doc_id,
            self.text_field => text,
        )) {
            self.pending_error = Some(format!("could not add tantivy document: {error}"));
        }
        self.dirty = true;
    }

    fn remove(&mut self, doc_id: &str) {
        self.delete_doc_id(doc_id);
        self.dirty = true;
    }

    fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if query.trim().is_empty() || k == 0 {
            return Vec::new();
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let parsed = match parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        let top = match searcher.search(&parsed, &TopDocs::with_limit(k)) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let Ok(doc) = searcher.doc::<TantivyDocument>(addr) else {
                continue;
            };
            // `get_first` returns an `OwnedValue` ref; tantivy 0.22 exposes
            // `as_str()` on text values.
            if let Some(value) = doc.get_first(self.doc_id_field) {
                if let Some(text) = value.as_str() {
                    out.push((text.to_string(), score));
                }
            }
        }
        out
    }

    fn designation(&self) -> &FullTextDesignation {
        &self.designation
    }

    fn doc_count(&self) -> usize {
        let searcher = self.reader.searcher();
        searcher.num_docs() as usize
    }

    fn flush(&mut self) -> Result<(), String> {
        self.commit_pending()
    }

    fn diagnostics(&self) -> FullTextBackendDiagnostics {
        FullTextBackendDiagnostics {
            backend: "tantivy".to_string(),
            document_count: self.doc_count(),
            persistent: self.storage_path.is_some(),
            path: self
                .storage_path
                .as_ref()
                .map(|path| path.display().to_string()),
            pending_writes: self.dirty || self.pending_error.is_some(),
            commits_since_open: self.commits_since_open,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn designation() -> FullTextDesignation {
        FullTextDesignation {
            label: "Doc".into(),
            property: "text".into(),
        }
    }

    #[test]
    fn tantivy_backend_basic_upsert_and_search() {
        let mut backend = TantivyFullTextBackend::new(designation()).unwrap();
        backend.upsert("d1", "the quick brown fox jumps");
        backend.upsert("d2", "the lazy dog sleeps");
        backend.flush().unwrap();
        let hits = backend.search("fox", 5);
        assert!(!hits.is_empty(), "expected at least one hit for 'fox'");
        assert_eq!(hits[0].0, "d1");
        assert_eq!(backend.doc_count(), 2);
    }

    #[test]
    fn tantivy_backend_remove_invalidates_doc() {
        let mut backend = TantivyFullTextBackend::new(designation()).unwrap();
        backend.upsert("d1", "rust is great");
        backend.upsert("d2", "tantivy is great");
        backend.remove("d1");
        backend.flush().unwrap();
        let hits = backend.search("rust", 5);
        let hit_ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
        assert!(!hit_ids.contains(&"d1"), "d1 should be removed");
        // doc_count should reflect the remaining live doc.
        assert_eq!(backend.doc_count(), 1);
    }

    #[test]
    fn tantivy_backend_upsert_replaces_existing_doc() {
        let mut backend = TantivyFullTextBackend::new(designation()).unwrap();
        backend.upsert("d1", "knowledge graph database");
        backend.upsert("d1", "weather forecast for tomorrow");
        backend.flush().unwrap();
        // The original "knowledge" content is gone; only "weather" should hit.
        let hits = backend.search("knowledge", 5);
        assert!(hits.iter().all(|(id, _)| id != "d1"));
        let hits = backend.search("weather", 5);
        assert!(hits.iter().any(|(id, _)| id == "d1"));
    }

    #[test]
    fn tantivy_backend_designation_round_trips() {
        let backend = TantivyFullTextBackend::new(designation()).unwrap();
        assert_eq!(backend.designation().label, "Doc");
        assert_eq!(backend.designation().property, "text");
    }

    #[test]
    fn persistent_backend_batches_and_reopens() {
        let path = std::env::temp_dir().join(format!(
            "rustyred-tantivy-{}-{}",
            std::process::id(),
            crate::now_ms()
        ));
        {
            let mut backend = TantivyFullTextBackend::open_or_create(designation(), &path).unwrap();
            assert_eq!(
                format!("{:?}", backend.writer.get_merge_policy()),
                "NoMergePolicy"
            );
            backend.upsert("d1", "durable memory index");
            backend.upsert("d2", "batched graph recall");
            assert_eq!(backend.diagnostics().commits_since_open, 0);
            backend.flush().unwrap();
            let diagnostics = backend.diagnostics();
            assert_eq!(diagnostics.document_count, 2);
            assert_eq!(diagnostics.commits_since_open, 1);
            assert!(diagnostics.persistent);
            assert!(!diagnostics.pending_writes);
        }
        {
            let backend = TantivyFullTextBackend::open_or_create(designation(), &path).unwrap();
            assert_eq!(backend.doc_count(), 2);
            assert!(backend
                .search("durable", 5)
                .iter()
                .any(|(id, _)| id == "d1"));
        }
        std::fs::remove_dir_all(path).ok();
    }

    /// The reason batch writers must flush.
    ///
    /// `rustyred-thg-server` stages a whole mutation batch before committing so
    /// an N-node batch leaves one segment instead of N. That is only safe
    /// because a staged tantivy write is invisible to search until `flush`; if
    /// this contract ever changed, staging without flushing would silently drop
    /// writes from recall rather than fail loudly.
    #[test]
    fn staged_upserts_are_invisible_until_flush() {
        let mut backend = TantivyFullTextBackend::new(designation()).unwrap();
        backend.upsert("d1", "batched memory write");
        assert!(
            backend.search("batched", 5).is_empty(),
            "a staged upsert must not be searchable before flush"
        );
        backend.flush().unwrap();
        assert!(
            backend
                .search("batched", 5)
                .iter()
                .any(|(id, _)| id == "d1"),
            "flush must publish every staged upsert"
        );
    }

    fn bench_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rustyred-tantivy-bench-{tag}-{}-{}",
            std::process::id(),
            crate::now_ms()
        ))
    }

    fn bench_text(index: usize) -> String {
        // Shaped like a MemoryAtom search_text: title, then summary, then body.
        format!(
            "memory atom {index} recall budget receipt\n\
             tenant scoped note about graph store index {} designation\n\
             the harness recorded a decision covering fulltext vector ppr \
             candidate enumeration and the {} lane at revision {index}",
            index % 97,
            if index % 2 == 0 { "seed" } else { "scan" }
        )
    }

    /// The decision this bench exists to settle: does warming
    /// `MemoryAtom.search_text` belong on boot or on a background job?
    ///
    /// `designate_fulltext_property` walks the label in `FULLTEXT_WARM_PAGE_SIZE`
    /// (64) pages and commits after every page, so the production corpus of
    /// 86,369 nodes costs ~1,350 tantivy commits. Persistent backends also run
    /// `NoMergePolicy`, so each commit leaves a segment behind permanently.
    /// Variant B commits once to separate the tokenization cost from the
    /// commit cost. Measured 2026-08-03 at the production corpus size:
    ///
    /// ```text
    /// docs                  86369
    /// commit-per-64-page      168.87s  commits=1350
    /// single-commit             1.62s  commits=1
    /// reopen (boot warm)       0.399s
    /// query after warm         0.009s  hits=15
    /// ```
    ///
    /// Indexing the corpus costs 1.6s; a commit costs ~124ms and 1,350 of them
    /// cost the other 167 seconds. Note the reopen and query rows are measured
    /// against the 1,350-segment paged index: many segments did not make reads
    /// slow, so commit count is a write-side cost, not a read-side one.
    ///
    /// Ignored by default: it writes a real index and takes real time.
    /// `cargo test -p rustyred-thg-core --features tantivy \
    ///   fulltext_warm_cost_at_corpus_scale -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement, not a gate: writes a full-size index"]
    fn fulltext_warm_cost_at_corpus_scale() {
        const PRODUCTION_PAGE_SIZE: usize = 64;
        let docs: usize = std::env::var("FULLTEXT_BENCH_DOCS")
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
            .unwrap_or(86_369);

        let paged_path = bench_path("paged");
        let mut backend =
            TantivyFullTextBackend::open_or_create(designation(), &paged_path).unwrap();
        let started = std::time::Instant::now();
        for index in 0..docs {
            backend.upsert(&format!("atom-{index}"), &bench_text(index));
            if (index + 1) % PRODUCTION_PAGE_SIZE == 0 {
                backend.flush().unwrap();
            }
        }
        backend.flush().unwrap();
        let paged_elapsed = started.elapsed();
        let paged_commits = backend.diagnostics().commits_since_open;
        let paged_docs = backend.doc_count();
        drop(backend);

        let single_path = bench_path("single");
        let mut backend =
            TantivyFullTextBackend::open_or_create(designation(), &single_path).unwrap();
        let started = std::time::Instant::now();
        for index in 0..docs {
            backend.upsert(&format!("atom-{index}"), &bench_text(index));
        }
        backend.flush().unwrap();
        let single_elapsed = started.elapsed();
        let single_commits = backend.diagnostics().commits_since_open;
        let single_docs = backend.doc_count();
        drop(backend);

        // The reopen path: `designate_fulltext_property` returns early when the
        // persisted index already has documents, so this is what a boot warm
        // actually costs on a tenant that has been indexed before.
        let started = std::time::Instant::now();
        let backend = TantivyFullTextBackend::open_or_create(designation(), &paged_path).unwrap();
        let reopened_docs = backend.doc_count();
        let reopen_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let hits = backend.search("designation", 15);
        let query_elapsed = started.elapsed();

        eprintln!("docs                  {docs}");
        eprintln!(
            "commit-per-64-page    {:>8.2}s  commits={paged_commits}  indexed={paged_docs}",
            paged_elapsed.as_secs_f64()
        );
        eprintln!(
            "single-commit         {:>8.2}s  commits={single_commits}  indexed={single_docs}",
            single_elapsed.as_secs_f64()
        );
        eprintln!(
            "reopen (boot warm)    {:>8.3}s  indexed={reopened_docs}",
            reopen_elapsed.as_secs_f64()
        );
        eprintln!(
            "query after warm      {:>8.3}s  hits={}",
            query_elapsed.as_secs_f64(),
            hits.len()
        );

        assert_eq!(paged_docs, docs, "paged warm must index every document");
        assert_eq!(
            single_docs, docs,
            "single-commit warm must index every document"
        );
        assert_eq!(
            reopened_docs, docs,
            "reopen must recover the persisted index"
        );

        drop(backend);
        std::fs::remove_dir_all(paged_path).ok();
        std::fs::remove_dir_all(single_path).ok();
    }
}
