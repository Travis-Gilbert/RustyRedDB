//! Full-text indexes kept inside the database.
//!
//! SOURCING: Tantivy (the `tantivy` feature) tokenizes, indexes, ranks with
//! BM25 and cuts snippets. This module only keeps each designated label's
//! Tantivy files in the chunk store, as file payloads named by one `fulltext`
//! catalog root that RedCore publishes in the same catalog commit as the
//! graph, so opening a database reads its index back instead of rebuilding it.
//!
//! The root records the graph transaction it was published with. When that is
//! not the recovered graph's (a build without Tantivy committed, or a commit
//! failed after Tantivy took its writes), the index is rebuilt from the nodes
//! and published again.
//!
//! The `fulltext` and `fulltext_tantivy` backends index one text field in
//! memory or in a filesystem directory. These indexes need boosted fields,
//! positions for phrases and snippets, and files kept as chunk payloads, so
//! they build their own Tantivy schema.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::file_payload::FilePayloadStagingGuard;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;

#[cfg(feature = "tantivy")]
pub(crate) use indexed::FullTextStore;
#[cfg(not(feature = "tantivy"))]
pub(crate) use unavailable::FullTextStore;

const FULLTEXT_ROOT_DOMAIN: u8 = 19;
const FULLTEXT_ROOT_VERSION: u8 = 1;
const MAX_FULLTEXT_ROOT_BYTES: usize = 16 * 1024 * 1024;

/// One node matching a full-text search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FullTextHit {
    pub id: String,
    /// BM25 over the designated fields, each scaled by its boost.
    pub score: f32,
    /// A passage of the best-matching field, HTML-escaped, with every match
    /// wrapped in `<mark>` and `…` where the passage is cut from the text.
    pub snippet: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FullTextRoot {
    graph_transaction: u64,
    indexes: BTreeMap<String, StoredIndex>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredIndex {
    fields: Vec<(String, f32)>,
    /// Tantivy file name to the address of its file-payload manifest.
    files: BTreeMap<String, Hash>,
}

impl FullTextRoot {
    fn to_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        let body = serde_json::to_vec(self).map_err(|error| invalid(error.to_string()))?;
        if body.len() > MAX_FULLTEXT_ROOT_BYTES {
            return Err(invalid("full-text root exceeds its byte limit"));
        }
        let mut bytes = vec![FULLTEXT_ROOT_DOMAIN, FULLTEXT_ROOT_VERSION];
        bytes.extend(body);
        Ok(bytes)
    }

    fn from_bytes(address: Hash, bytes: &[u8]) -> GraphStoreResult<Self> {
        if bytes.get(..2) != Some(&[FULLTEXT_ROOT_DOMAIN, FULLTEXT_ROOT_VERSION]) {
            return Err(invalid("full-text root domain or version is invalid"));
        }
        if Hash::of(bytes) != address {
            return Err(invalid("full-text root content address mismatch"));
        }
        let root: Self =
            serde_json::from_slice(&bytes[2..]).map_err(|error| invalid(error.to_string()))?;
        if root.to_bytes()? != bytes {
            return Err(invalid("full-text root is not canonical"));
        }
        Ok(root)
    }
}

/// The file manifests a committed full-text root keeps reachable.
pub(crate) fn decode_fulltext_root(address: Hash, bytes: &[u8]) -> GraphStoreResult<Vec<Hash>> {
    Ok(FullTextRoot::from_bytes(address, bytes)?
        .indexes
        .into_values()
        .flat_map(|index| index.files.into_values())
        .collect())
}

/// A root and its files, written and pinned, waiting for the catalog commit
/// that names it.
pub(crate) struct StagedFullText<'a> {
    pub(crate) root: Hash,
    #[cfg_attr(not(feature = "tantivy"), allow(dead_code))]
    indexes: BTreeMap<String, StoredIndex>,
    #[cfg_attr(not(feature = "tantivy"), allow(dead_code))]
    guards: Vec<FilePayloadStagingGuard<'a>>,
}

fn invalid(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("fulltext_root_invalid", message)
}

#[cfg(feature = "tantivy")]
mod indexed {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt;
    use std::path::Path;

    use serde_json::Value;
    use tantivy::collector::TopDocs;
    use tantivy::directory::RamDirectory;
    use tantivy::merge_policy::{LogMergePolicy, MergePolicy, NoMergePolicy};
    use tantivy::query::{
        BooleanQuery, BoostQuery, ConstScoreQuery, Occur, PhrasePrefixQuery, PhraseQuery, Query,
        TermQuery, TermSetQuery,
    };
    use tantivy::schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as _, STORED,
        STRING,
    };
    use tantivy::tokenizer::{
        AsciiFoldingFilter, LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer,
        TokenStream,
    };
    use tantivy::{
        Directory, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, SnippetGenerator,
        TantivyDocument, Term,
    };

    use super::{invalid, FullTextHit, FullTextRoot, StagedFullText, StoredIndex};
    use crate::chunk_store::ChunkStore;
    use crate::file_payload::{read_file_payload, stage_file_payload, FileManifestRef};
    use crate::graph::error::{GraphStoreError, GraphStoreResult};
    use crate::graph::mutation::GraphMutation;
    use crate::graph::record::NodeRecord;
    use crate::hash::Hash;
    use crate::object_store::UnifiedChunkStore;

    const ANALYZER: &str = "rustyred_text";
    /// One indexing thread keeps one new segment per commit.
    const WRITER_THREADS: usize = 1;
    /// Tantivy's per-thread minimum.
    const WRITER_MEMORY_BUDGET: usize = 15_000_000;
    /// Terms the last, still-being-typed word expands to, most frequent first.
    const MAX_PREFIX_TERMS: usize = 50;
    /// Files Tantivy rewrites on every commit, so they are never reused.
    const META_FILES: [&str; 2] = ["meta.json", ".managed.json"];

    /// Every designated label's index, held in memory and mirrored to the
    /// chunk store by [`Self::stage`].
    #[derive(Default)]
    pub(crate) struct FullTextStore {
        indexes: BTreeMap<String, LabelIndex>,
        /// A rebuild after a failed commit itself failed: searches refuse and
        /// the next commit rebuilds first.
        stale: bool,
        /// Rebuilt since the last publication.
        unpublished: bool,
    }

    impl fmt::Debug for FullTextStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("FullTextStore")
                .field("labels", &self.indexes.keys().collect::<Vec<_>>())
                .field("stale", &self.stale)
                .field("unpublished", &self.unpublished)
                .finish()
        }
    }

    impl FullTextStore {
        pub(crate) fn has_indexes(&self) -> bool {
            !self.indexes.is_empty()
        }

        pub(crate) fn unpublished(&self) -> bool {
            self.unpublished
        }

        /// Read the committed root back. An index published at another graph
        /// transaction, or one whose files cannot be opened, is rebuilt.
        pub(crate) fn recover(
            &mut self,
            store: &dyn ChunkStore,
            root: Hash,
            graph_transaction: u64,
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            let bytes = store
                .get(root)?
                .ok_or_else(|| invalid(format!("committed full-text root {root} is missing")))?;
            let committed = FullTextRoot::from_bytes(root, &bytes)?;
            let current = committed.graph_transaction == graph_transaction;
            for (label, stored) in committed.indexes {
                let opened = if current {
                    LabelIndex::open_stored(store, &stored).ok()
                } else {
                    None
                };
                let index = match opened {
                    Some(index) => index,
                    None => {
                        self.unpublished = true;
                        LabelIndex::filled(&label, &stored.fields, nodes)?
                    }
                };
                self.indexes.insert(label, index);
            }
            Ok(())
        }

        /// Index `fields` of every `label` node. Returns whether anything
        /// changed: the same fields again leave the index as it is.
        pub(crate) fn designate(
            &mut self,
            label: &str,
            fields: &[(&str, f32)],
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<bool> {
            let fields = validate_designation(label, fields)?;
            if self
                .indexes
                .get(label)
                .is_some_and(|index| index.fields == fields)
            {
                return Ok(false);
            }
            let index = LabelIndex::filled(label, &fields, nodes)?;
            self.indexes.insert(label.to_string(), index);
            Ok(true)
        }

        /// Take one commit's node writes. `before` is the graph they apply to.
        pub(crate) fn apply(
            &mut self,
            mutations: &[GraphMutation],
            before: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            if self.stale {
                self.rebuild(before)?;
            }
            for (label, index) in &mut self.indexes {
                let indexed_before =
                    |id: &str| before.get(id).is_some_and(|node| carries(node, label));
                let mut indexed_now = BTreeSet::new();
                let mut changed = false;
                for mutation in mutations {
                    let (id, current) = match mutation {
                        GraphMutation::NodeUpsert(node) => (node.id.as_str(), Some(node)),
                        GraphMutation::NodeDelete(id) => (id.as_str(), None),
                        _ => continue,
                    };
                    match current.filter(|node| carries(node, label)) {
                        Some(node) => {
                            index.put(node)?;
                            indexed_now.insert(id);
                            changed = true;
                        }
                        None if indexed_now.remove(id) || indexed_before(id) => {
                            index.remove(id);
                            changed = true;
                        }
                        None => {}
                    }
                }
                if changed {
                    index.commit()?;
                }
            }
            Ok(())
        }

        /// Write every index's new files and a root naming them all, without
        /// publishing it.
        pub(crate) fn stage<'a>(
            &self,
            publication: &dyn ChunkStore,
            store: &'a dyn UnifiedChunkStore,
            graph_transaction: u64,
        ) -> GraphStoreResult<StagedFullText<'a>> {
            let mut guards = Vec::new();
            let mut indexes = BTreeMap::new();
            for (label, index) in &self.indexes {
                indexes.insert(label.clone(), index.stage(store, label, &mut guards)?);
            }
            let root = FullTextRoot {
                graph_transaction,
                indexes,
            };
            let bytes = root.to_bytes()?;
            let address = Hash::of(&bytes);
            publication.write(&[(address, bytes.as_slice())])?;
            Ok(StagedFullText {
                root: address,
                indexes: root.indexes,
                guards,
            })
        }

        /// The catalog now names `staged`: its files are reused from here on.
        pub(crate) fn published(&mut self, staged: StagedFullText<'_>) {
            for (label, stored) in staged.indexes {
                if let Some(index) = self.indexes.get_mut(&label) {
                    index.durable = stored
                        .files
                        .into_iter()
                        .filter(|(name, _)| !META_FILES.contains(&name.as_str()))
                        .collect();
                }
            }
            for guard in staged.guards {
                guard.finish_after_publication();
            }
            self.unpublished = false;
        }

        /// Rebuild every index from `nodes`, keeping the designations.
        pub(crate) fn rebuild(
            &mut self,
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            self.stale = true;
            for (label, index) in &mut self.indexes {
                *index = LabelIndex::filled(label, &index.fields, nodes)?;
            }
            self.stale = false;
            self.unpublished = true;
            Ok(())
        }

        pub(crate) fn search(
            &self,
            label: &str,
            query: &str,
            limit: usize,
            within: Option<&[String]>,
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<Vec<FullTextHit>> {
            if self.stale {
                return Err(GraphStoreError::new(
                    "fulltext_stale",
                    "the full-text index could not be rebuilt after a failed commit; \
                     the next commit or open rebuilds it",
                ));
            }
            let index = self.indexes.get(label).ok_or_else(|| {
                GraphStoreError::new(
                    "fulltext_not_designated",
                    format!("label {label:?} has no full-text designation"),
                )
            })?;
            index.search(query, limit, within, nodes)
        }
    }

    struct LabelIndex {
        fields: Vec<(String, f32)>,
        index: Index,
        writer: IndexWriter,
        reader: IndexReader,
        id: Field,
        text: Vec<Field>,
        /// Segment files already in the chunk store. Tantivy never rewrites a
        /// segment file under the same name.
        durable: BTreeMap<String, Hash>,
    }

    impl LabelIndex {
        fn filled(
            label: &str,
            fields: &[(String, f32)],
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<Self> {
            let mut index = Self::new(fields, Index::create_in_ram(schema(fields.len())))?;
            for node in nodes.values().filter(|node| carries(node, label)) {
                index.put(node)?;
            }
            index.commit()?;
            Ok(index)
        }

        fn open_stored(store: &dyn ChunkStore, stored: &StoredIndex) -> GraphStoreResult<Self> {
            let directory = RamDirectory::create();
            for (name, address) in &stored.files {
                let bytes = read_file_payload(store, &FileManifestRef::new(*address))?;
                directory
                    .atomic_write(Path::new(name), &bytes)
                    .map_err(tantivy_error)?;
            }
            let mut index = Self::new(
                &stored.fields,
                Index::open(directory).map_err(tantivy_error)?,
            )?;
            index.durable = stored
                .files
                .iter()
                .filter(|(name, _)| !META_FILES.contains(&name.as_str()))
                .map(|(name, address)| (name.clone(), *address))
                .collect();
            Ok(index)
        }

        fn new(fields: &[(String, f32)], index: Index) -> GraphStoreResult<Self> {
            index.tokenizers().register(ANALYZER, analyzer());
            let schema = index.schema();
            let id = schema.get_field("id").map_err(tantivy_error)?;
            let text = (0..fields.len())
                .map(|position| schema.get_field(&format!("text{position}")))
                .collect::<Result<Vec<_>, _>>()
                .map_err(tantivy_error)?;
            let writer = index
                .writer_with_num_threads(WRITER_THREADS, WRITER_MEMORY_BUDGET)
                .map_err(tantivy_error)?;
            // Merges run in `commit`, so every file a snapshot lists stays put
            // until the snapshot is read.
            writer.set_merge_policy(Box::new(NoMergePolicy));
            let reader = index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()
                .map_err(tantivy_error)?;
            Ok(Self {
                fields: fields.to_vec(),
                index,
                writer,
                reader,
                id,
                text,
                durable: BTreeMap::new(),
            })
        }

        fn put(&mut self, node: &NodeRecord) -> GraphStoreResult<()> {
            self.remove(&node.id);
            let mut document = TantivyDocument::default();
            document.add_text(self.id, &node.id);
            for ((property, _), field) in self.fields.iter().zip(&self.text) {
                if let Some(text) = field_text(node, property) {
                    document.add_text(*field, text);
                }
            }
            self.writer.add_document(document).map_err(tantivy_error)?;
            Ok(())
        }

        fn remove(&mut self, id: &str) {
            self.writer.delete_term(Term::from_field_text(self.id, id));
        }

        /// Commit, then merge whatever Tantivy's log merge policy would merge
        /// in the background, here and now.
        fn commit(&mut self) -> GraphStoreResult<()> {
            self.writer.commit().map_err(tantivy_error)?;
            let segments = self
                .index
                .searchable_segment_metas()
                .map_err(tantivy_error)?;
            for candidate in LogMergePolicy::default().compute_merge_candidates(&segments) {
                self.writer
                    .merge(&candidate.0)
                    .wait()
                    .map_err(tantivy_error)?;
            }
            // The directory is in memory: drop the files merges replaced.
            self.writer
                .garbage_collect_files()
                .wait()
                .map_err(tantivy_error)?;
            self.reader.reload().map_err(tantivy_error)
        }

        fn stage<'a>(
            &self,
            store: &'a dyn UnifiedChunkStore,
            label: &str,
            guards: &mut Vec<crate::file_payload::FilePayloadStagingGuard<'a>>,
        ) -> GraphStoreResult<StoredIndex> {
            let directory = self.index.directory();
            let mut names = META_FILES
                .map(String::from)
                .into_iter()
                .collect::<BTreeSet<_>>();
            for segment in self
                .index
                .searchable_segment_metas()
                .map_err(tantivy_error)?
            {
                for path in segment.list_files() {
                    names.insert(path.to_string_lossy().into_owned());
                }
            }
            let mut files = BTreeMap::new();
            for name in names {
                let path = Path::new(&name);
                if !directory.exists(path).map_err(tantivy_error)? {
                    continue;
                }
                let address = match self.durable.get(&name) {
                    Some(address) => *address,
                    None => {
                        let bytes = directory.atomic_read(path).map_err(tantivy_error)?;
                        let staged =
                            stage_file_payload(store, &format!("fulltext:{label}:{name}"), &bytes)?;
                        let address = staged.reference().address()?;
                        guards.push(staged);
                        address
                    }
                };
                files.insert(name, address);
            }
            Ok(StoredIndex {
                fields: self.fields.clone(),
                files,
            })
        }

        /// Every whitespace-separated word must match one field, the last one
        /// also as the prefix of a longer word. A word that tokenizes into
        /// several tokens matches as a phrase.
        fn search(
            &self,
            query: &str,
            limit: usize,
            within: Option<&[String]>,
            nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<Vec<FullTextHit>> {
            let words = query.split_whitespace().collect::<Vec<_>>();
            if words.is_empty() || limit == 0 {
                return Ok(Vec::new());
            }
            let searcher = self.reader.searcher();
            let mut analyzer = analyzer();
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            for (position, word) in words.iter().enumerate() {
                let mut tokens = Vec::new();
                analyzer
                    .token_stream(word)
                    .process(&mut |token| tokens.push(token.text.clone()));
                if tokens.is_empty() {
                    continue;
                }
                let last = position + 1 == words.len();
                let mut alternatives: Vec<(Occur, Box<dyn Query>)> = Vec::new();
                for ((_, boost), field) in self.fields.iter().zip(&self.text) {
                    let terms = tokens
                        .iter()
                        .map(|token| Term::from_field_text(*field, token))
                        .collect::<Vec<_>>();
                    let matchers: Vec<Box<dyn Query>> = match (terms.len(), last) {
                        (1, false) => vec![Box::new(TermQuery::new(
                            terms[0].clone(),
                            IndexRecordOption::WithFreqs,
                        ))],
                        (1, true) => prefix_terms(&searcher, *field, &tokens[0])?
                            .into_iter()
                            .map(|term| {
                                Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
                                    as Box<dyn Query>
                            })
                            .collect(),
                        (_, false) => vec![Box::new(PhraseQuery::new(terms))],
                        (_, true) => vec![Box::new(PhrasePrefixQuery::new(terms))],
                    };
                    alternatives.extend(matchers.into_iter().map(|matcher| {
                        (
                            Occur::Should,
                            Box::new(BoostQuery::new(matcher, *boost)) as Box<dyn Query>,
                        )
                    }));
                }
                if alternatives.is_empty() {
                    return Ok(Vec::new());
                }
                clauses.push((Occur::Must, Box::new(BooleanQuery::new(alternatives))));
            }
            if clauses.is_empty() {
                return Ok(Vec::new());
            }
            if let Some(ids) = within {
                let ids =
                    TermSetQuery::new(ids.iter().map(|id| Term::from_field_text(self.id, id)));
                clauses.push((
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(Box::new(ids), 0.0)),
                ));
            }
            let query = BooleanQuery::new(clauses);
            let top = searcher
                .search(&query, &TopDocs::with_limit(limit))
                .map_err(tantivy_error)?;
            let generators = self
                .text
                .iter()
                .map(|field| SnippetGenerator::create(&searcher, &query, *field))
                .collect::<Result<Vec<_>, _>>()
                .map_err(tantivy_error)?;
            let mut hits = Vec::with_capacity(top.len());
            for (score, address) in top {
                let document: TantivyDocument = searcher.doc(address).map_err(tantivy_error)?;
                let Some(id) = document.get_first(self.id).and_then(|value| value.as_str()) else {
                    continue;
                };
                hits.push(FullTextHit {
                    id: id.to_string(),
                    score,
                    snippet: nodes
                        .get(id)
                        .map(|node| self.snippet(node, &generators))
                        .unwrap_or_default(),
                });
            }
            Ok(hits)
        }

        /// The passage of the field with the most matches, the earlier field
        /// on a tie.
        fn snippet(&self, node: &NodeRecord, generators: &[SnippetGenerator]) -> String {
            let best = self
                .fields
                .iter()
                .zip(generators)
                .filter_map(|((property, _), generator)| {
                    let text = field_text(node, property)?;
                    let snippet = generator.snippet(&text);
                    (!snippet.highlighted().is_empty()).then_some((text, snippet))
                })
                .rev()
                .max_by_key(|(_, snippet)| snippet.highlighted().len());
            let Some((text, mut snippet)) = best else {
                return String::new();
            };
            snippet.set_snippet_prefix_postfix("<mark>", "</mark>");
            let start = text.find(snippet.fragment()).unwrap_or(0);
            let mut passage = String::new();
            if start > 0 {
                passage.push('…');
            }
            passage.push_str(&snippet.to_html());
            if start + snippet.fragment().len() < text.len() {
                passage.push('…');
            }
            passage
        }
    }

    /// Up to [`MAX_PREFIX_TERMS`] indexed terms of `field` starting with
    /// `prefix`, by how many documents use them. Tantivy's own prefix queries
    /// score every expansion the same and highlight none of them.
    fn prefix_terms(
        searcher: &Searcher,
        field: Field,
        prefix: &str,
    ) -> GraphStoreResult<Vec<Term>> {
        let mut frequencies = BTreeMap::<Vec<u8>, u64>::new();
        for segment in searcher.segment_readers() {
            let inverted = segment.inverted_index(field).map_err(tantivy_error)?;
            let mut terms = inverted
                .terms()
                .range()
                .ge(prefix.as_bytes())
                .into_stream()
                .map_err(tantivy_error)?;
            while terms.advance() && terms.key().starts_with(prefix.as_bytes()) {
                *frequencies.entry(terms.key().to_vec()).or_default() +=
                    u64::from(terms.value().doc_freq);
            }
        }
        let mut ranked = frequencies.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        Ok(ranked
            .into_iter()
            .take(MAX_PREFIX_TERMS)
            .filter_map(|(bytes, _)| String::from_utf8(bytes).ok())
            .map(|text| Term::from_field_text(field, &text))
            .collect())
    }

    fn schema(fields: usize) -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("id", STRING | STORED);
        let text = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(ANALYZER)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );
        for position in 0..fields {
            builder.add_text_field(&format!("text{position}"), text.clone());
        }
        builder.build()
    }

    /// Unicode words, lowercased, with accents folded to ASCII.
    fn analyzer() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(80))
            .filter(LowerCaser)
            .filter(AsciiFoldingFilter)
            .build()
    }

    fn carries(node: &NodeRecord, label: &str) -> bool {
        !node.tombstone && node.labels.iter().any(|candidate| candidate == label)
    }

    /// A string property, or an array's strings one per line.
    fn field_text(node: &NodeRecord, property: &str) -> Option<String> {
        match node.properties.get(property)? {
            Value::String(text) => Some(text.clone()),
            Value::Array(items) => {
                let lines = items.iter().filter_map(Value::as_str).collect::<Vec<_>>();
                (!lines.is_empty()).then(|| lines.join("\n"))
            }
            _ => None,
        }
    }

    fn validate_designation(
        label: &str,
        fields: &[(&str, f32)],
    ) -> GraphStoreResult<Vec<(String, f32)>> {
        let refuse =
            |message: String| GraphStoreError::new("fulltext_designation_invalid", message);
        if label.trim().is_empty() {
            return Err(refuse("a full-text designation needs a label".into()));
        }
        if fields.is_empty() {
            return Err(refuse(format!("{label:?} names no fields to index")));
        }
        let mut seen = BTreeSet::new();
        for (field, boost) in fields {
            if field.trim().is_empty() || !seen.insert(*field) {
                return Err(refuse(format!(
                    "{label:?} names an empty or repeated field"
                )));
            }
            if !boost.is_finite() || *boost <= 0.0 {
                return Err(refuse(format!("field {field:?} needs a positive boost")));
            }
        }
        Ok(fields
            .iter()
            .map(|(field, boost)| ((*field).to_string(), *boost))
            .collect())
    }

    fn tantivy_error(error: impl fmt::Display) -> GraphStoreError {
        GraphStoreError::new("fulltext_index_failed", error.to_string())
    }
}

#[cfg(not(feature = "tantivy"))]
mod unavailable {
    use std::collections::BTreeMap;

    use super::StagedFullText;
    use crate::chunk_store::ChunkStore;
    use crate::graph::error::{GraphStoreError, GraphStoreResult};
    use crate::graph::mutation::GraphMutation;
    use crate::graph::record::NodeRecord;
    use crate::hash::Hash;
    use crate::object_store::UnifiedChunkStore;

    /// Without Tantivy nothing is indexed and a committed root is left as it
    /// is. Commits do not advance it, so a build with Tantivy rebuilds it.
    #[derive(Debug, Default)]
    pub(crate) struct FullTextStore;

    impl FullTextStore {
        pub(crate) fn has_indexes(&self) -> bool {
            false
        }

        pub(crate) fn unpublished(&self) -> bool {
            false
        }

        pub(crate) fn recover(
            &mut self,
            _store: &dyn ChunkStore,
            _root: Hash,
            _graph_transaction: u64,
            _nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            Ok(())
        }

        pub(crate) fn apply(
            &mut self,
            _mutations: &[GraphMutation],
            _before: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            Ok(())
        }

        pub(crate) fn stage<'a>(
            &self,
            _publication: &dyn ChunkStore,
            _store: &'a dyn UnifiedChunkStore,
            _graph_transaction: u64,
        ) -> GraphStoreResult<StagedFullText<'a>> {
            Err(GraphStoreError::new(
                "fulltext_unavailable",
                "this build has no full-text index (the tantivy feature is off)",
            ))
        }

        pub(crate) fn published(&mut self, _staged: StagedFullText<'_>) {}

        pub(crate) fn rebuild(
            &mut self,
            _nodes: &BTreeMap<String, NodeRecord>,
        ) -> GraphStoreResult<()> {
            Ok(())
        }
    }
}
