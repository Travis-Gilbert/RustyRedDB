//! Read-only compatibility decoder for the retired document tree.
//!
//! The live filesystem namespace is the graph: `File` and `Directory` nodes
//! carry raw [`PathKey`] values and `CONTAINS` edges express hierarchy. This
//! module remains only so installations can import bounded legacy
//! `doc-tree.json` snapshots and the former `documents` named root. New
//! production code must not use [`LegacyDocTree`] as a namespace or body owner.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::chunk_store::{
    validate_catalog_root, CatalogCommitResult, CommitMeta, ROOT_NAME_DOCUMENTS,
};
use crate::cold_index::ColdTierKind;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::object_store::{content_hash_bytes, decompress_cold_bytes, DiskObjectStore};

const PATH_SEPARATOR: u8 = 0;
const MAX_PREVIOUS_HASHES: usize = 64;
const DOC_TREE_FORMAT_VERSION: u8 = 1;
const DOMAIN_DOC_ROOT: u8 = 8;
const DOMAIN_DOC_ENTRY: u8 = 9;
const DOC_BODY_NONE: u8 = 0;
const DOC_BODY_INLINE: u8 = 1;
const DOC_BODY_OVERFLOW: u8 = 2;
const MAX_DOC_ENTRIES: usize = 1_000_000;
const MAX_DOC_PATH_BYTES: usize = 1024 * 1024;
const MAX_DOC_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_LEGACY_DOC_TREE_JSON_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_LEGACY_INLINE_THRESHOLD: usize = 4096;

/// Hierarchical path whose byte order is the tree order. Segments are separated
/// by NUL, which sorts below ordinary path bytes and prevents `a/b2` from
/// matching the `a/b` prefix.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PathKey(Vec<u8>);

impl PathKey {
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> GraphStoreResult<Self> {
        let bytes = bytes.into();
        validate_path_key_bytes(&bytes)?;
        Ok(Self(bytes))
    }

    pub fn from_slash_path(path: &str) -> GraphStoreResult<Self> {
        if path.is_empty() || path.starts_with('/') || path.ends_with('/') || path.contains("//") {
            return Err(invalid_path(
                "slash path must be non-empty, relative, and contain no empty segments",
            ));
        }
        Self::from_segments(path.split('/'))
    }

    pub fn from_segments<'a>(
        segments: impl IntoIterator<Item = &'a str>,
    ) -> GraphStoreResult<Self> {
        Self::from_bytes(encode_segments(segments)?)
    }

    #[deprecated(note = "use PathPrefix::from_segments; prefixes are not exact PathKey values")]
    pub fn prefix_from_segments<'a>(
        segments: impl IntoIterator<Item = &'a str>,
    ) -> GraphStoreResult<PathPrefix> {
        PathPrefix::from_segments(segments)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Raw UTF-8 property value used by the graph's ordered path index.
    /// Interior NUL bytes are the validated segment separators.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("PathKey construction validates UTF-8")
    }

    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        self.0.starts_with(prefix)
    }

    pub fn to_slash_path(&self) -> String {
        self.0
            .split(|byte| *byte == PATH_SEPARATOR)
            .filter(|segment| !segment.is_empty())
            .map(|segment| {
                std::str::from_utf8(segment)
                    .expect("PathKey construction validates UTF-8")
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("/")
    }
}

impl<'de> Deserialize<'de> for PathKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Self::from_bytes(bytes)
            .map_err(|error| serde::de::Error::custom(format!("{}: {}", error.code, error.message)))
    }
}

/// Validated ordered-range prefix for [`PathKey`].
///
/// A non-root prefix is an exact PathKey encoding followed by one NUL byte,
/// so `a/b` descendants cannot collide with `a/b2`. The empty prefix denotes
/// the virtual root and scans the complete namespace.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PathPrefix(Vec<u8>);

impl PathPrefix {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> GraphStoreResult<Self> {
        let bytes = bytes.into();
        validate_path_prefix_bytes(&bytes)?;
        Ok(Self(bytes))
    }

    pub fn from_slash_path(path: &str) -> GraphStoreResult<Self> {
        if path.is_empty() || path == "/" {
            return Ok(Self::root());
        }
        if path.starts_with('/') || path.ends_with('/') || path.contains("//") {
            return Err(invalid_path(
                "slash prefix must be relative and contain no empty segments",
            ));
        }
        Self::from_segments(path.split('/'))
    }

    pub fn from_segments<'a>(
        segments: impl IntoIterator<Item = &'a str>,
    ) -> GraphStoreResult<Self> {
        let segments = segments.into_iter().collect::<Vec<_>>();
        if segments.is_empty() {
            return Ok(Self::root());
        }
        let mut bytes = encode_segments(segments)?;
        bytes.push(PATH_SEPARATOR);
        Self::from_bytes(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("PathPrefix construction validates UTF-8")
    }

    pub fn matches(&self, path: &PathKey) -> bool {
        path.as_bytes().starts_with(&self.0)
    }
}

impl<'de> Deserialize<'de> for PathPrefix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Self::from_bytes(bytes)
            .map_err(|error| serde::de::Error::custom(format!("{}: {}", error.code, error.message)))
    }
}

/// Leaf payload for one document path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DocEntry {
    /// Raw-body sha256. Inline entries keep this too so updates retain an
    /// addressable identity even when no overflow fetch is needed.
    pub content_hash: Option<String>,
    /// Small body bytes cached in the canonical entry leaf. New writes are raw;
    /// `inline_compressed` remains only to decode legacy JSON snapshots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<Vec<u8>>,
    /// Validated path hint relative to the runtime-only workspace root. The
    /// field name is retained for source compatibility with pre-journal users.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_path: Option<String>,
    #[serde(default)]
    pub inline_compressed: bool,
    pub tier: ColdTierKind,
    pub size: u64,
    pub created_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gist: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_hashes: Vec<String>,
    /// Number of unique older versions intentionally omitted from the bounded
    /// retained history. History loss is therefore explicit rather than silent.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub history_elided_count: u64,
}

impl DocEntry {
    pub fn is_inline(&self) -> bool {
        self.inline.is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyDocTree {
    entries: BTreeMap<PathKey, DocEntry>,
    inline_threshold: usize,
    /// Last canonical document root written or reopened. This is decoded cache
    /// state; publication still happens only through the committed catalog.
    root_address: Option<Hash>,
    /// Runtime-only base for `DocEntry::real_path` relative hints.
    workspace_root: Option<PathBuf>,
}

impl LegacyDocTree {
    fn empty(inline_threshold: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            inline_threshold,
            root_address: None,
            workspace_root: None,
        }
    }

    /// Strictly decode the historical `{data_dir}/doc-tree.json` representation.
    ///
    /// The old file is a JSON array of `(slash_path, DocEntry)` pairs. Import is
    /// bounded before deserialization and every path, body reference, inline
    /// body, history list, and duplicate is validated before any entry becomes
    /// visible to the caller.
    pub fn decode_legacy_json(bytes: &[u8]) -> GraphStoreResult<Self> {
        if bytes.len() > MAX_LEGACY_DOC_TREE_JSON_BYTES {
            return Err(GraphStoreError::new(
                "legacy_doc_tree_json_too_large",
                format!(
                    "legacy doc-tree JSON is {} bytes; limit is {MAX_LEGACY_DOC_TREE_JSON_BYTES}",
                    bytes.len()
                ),
            ));
        }
        let entries: Vec<(String, DocEntry)> = serde_json::from_slice(bytes).map_err(|error| {
            GraphStoreError::new("legacy_doc_tree_json_invalid", error.to_string())
        })?;
        if entries.len() > MAX_DOC_ENTRIES {
            return Err(GraphStoreError::new(
                "legacy_doc_tree_entry_count_too_large",
                format!(
                    "legacy doc tree has {} entries; limit is {MAX_DOC_ENTRIES}",
                    entries.len()
                ),
            ));
        }

        let mut tree = Self::empty(DEFAULT_LEGACY_INLINE_THRESHOLD);
        for (slash_path, entry) in entries {
            if slash_path.len() > MAX_DOC_PATH_BYTES {
                return Err(invalid_path(
                    "legacy slash path exceeds the canonical byte limit",
                ));
            }
            let path = PathKey::from_slash_path(&slash_path)?;
            validate_legacy_entry(&entry)?;
            if tree.entries.insert(path, entry).is_some() {
                return Err(GraphStoreError::new(
                    "legacy_doc_tree_duplicate_path",
                    format!("legacy doc tree repeats path {slash_path:?}"),
                ));
            }
        }
        Ok(tree)
    }

    pub fn with_workspace_root(mut self, workspace_root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(workspace_root.into());
        self
    }

    pub fn set_workspace_root(&mut self, workspace_root: impl Into<PathBuf>) {
        self.workspace_root = Some(workspace_root.into());
    }

    pub fn root_address(&self) -> Option<Hash> {
        self.root_address
    }

    pub fn inline_threshold(&self) -> usize {
        self.inline_threshold
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, path: &PathKey) -> Option<&DocEntry> {
        self.entries.get(path)
    }

    fn put(&mut self, path: PathKey, mut entry: DocEntry) {
        if let Some(previous) = self.entries.get(&path) {
            if previous.content_hash != entry.content_hash {
                if let Some(hash) = previous.content_hash.as_ref() {
                    entry.previous_hashes.push(hash.clone());
                }
            }
            entry
                .previous_hashes
                .extend(previous.previous_hashes.clone());
            entry.history_elided_count = entry
                .history_elided_count
                .saturating_add(previous.history_elided_count);
            let mut seen = BTreeSet::new();
            if let Some(current) = entry.content_hash.as_ref() {
                seen.insert(current.clone());
            }
            entry
                .previous_hashes
                .retain(|hash| seen.insert(hash.clone()));
            let dropped = entry
                .previous_hashes
                .len()
                .saturating_sub(MAX_PREVIOUS_HASHES);
            entry.previous_hashes.truncate(MAX_PREVIOUS_HASHES);
            entry.history_elided_count = entry
                .history_elided_count
                .saturating_add(u64::try_from(dropped).unwrap_or(u64::MAX));
        }
        self.entries.insert(path, entry);
        self.root_address = None;
    }

    /// Type-safe legacy range used only while importing the retired namespace.
    pub fn legacy_range<'a>(
        &'a self,
        prefix: &'a PathPrefix,
    ) -> impl Iterator<Item = (&'a PathKey, &'a DocEntry)> + 'a {
        self.entries
            .range(PathKey(prefix.as_bytes().to_vec())..)
            .take_while(move |(path, _)| prefix.matches(path))
    }

    /// Iterate every legacy entry in canonical path order.
    pub fn legacy_entries(&self) -> impl Iterator<Item = (&PathKey, &DocEntry)> {
        self.entries.iter()
    }

    fn put_body(
        &mut self,
        path: PathKey,
        body: &[u8],
        tier: ColdTierKind,
        created_ms: i64,
        gist: Option<String>,
        object_store: &DiskObjectStore,
    ) -> GraphStoreResult<DocEntry> {
        let hash = content_hash_bytes(body);
        // The raw body is always durable before the leaf can point at it. The
        // inline copy is an index-locality optimization, never the only copy.
        let stored_hash = object_store.put_document_bytes(body)?;
        if stored_hash != hash {
            return Err(GraphStoreError::new(
                "doc_tree_body_address_mismatch",
                format!("document bytes stored as {stored_hash}, expected {hash}"),
            ));
        }
        let size = u64::try_from(body.len()).map_err(|_| {
            GraphStoreError::new("doc_tree_size_overflow", "body length exceeds u64")
        })?;
        let entry = if body.len() <= self.inline_threshold {
            DocEntry {
                content_hash: Some(hash),
                inline: Some(body.to_vec()),
                real_path: None,
                inline_compressed: false,
                tier,
                size,
                created_ms,
                gist,
                previous_hashes: Vec::new(),
                history_elided_count: 0,
            }
        } else {
            DocEntry {
                content_hash: Some(stored_hash),
                inline: None,
                real_path: None,
                inline_compressed: false,
                tier,
                size,
                created_ms,
                gist,
                previous_hashes: Vec::new(),
                history_elided_count: 0,
            }
        };
        self.put(path.clone(), entry.clone());
        Ok(self.entries.get(&path).cloned().unwrap_or(entry))
    }

    pub fn resolve_body(
        &self,
        path: &PathKey,
        object_store: &DiskObjectStore,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        let Some(entry) = self.entries.get(path) else {
            return Ok(None);
        };
        resolve_entry_body(entry, self.workspace_root.as_deref(), object_store).map(Some)
    }

    /// Write verified entry leaves and the sorted document root. This stages
    /// chunks only; [`Self::publish`] is the root visibility barrier.
    fn write_root(&mut self, object_store: &DiskObjectStore) -> GraphStoreResult<Hash> {
        let paths = self.entries.keys().cloned().collect::<Vec<_>>();
        let mut encoded_entries = Vec::with_capacity(paths.len());
        for path in paths {
            let mut entry = self
                .entries
                .get(&path)
                .cloned()
                .expect("path came from the LegacyDocTree cache");
            prepare_entry_for_persist(&mut entry, self.workspace_root.as_deref(), object_store)?;
            let bytes = encode_doc_entry(&path, &entry)?;
            let address = Hash::of(&bytes);
            self.entries.insert(path.clone(), entry);
            encoded_entries.push((path, address, bytes));
        }

        let root_bytes = encode_doc_root(
            self.inline_threshold,
            encoded_entries
                .iter()
                .map(|(path, address, _)| (path, *address)),
        )?;
        let root_address = Hash::of(&root_bytes);
        let mut owned = encoded_entries
            .iter()
            .map(|(_, address, bytes)| (*address, bytes.clone()))
            .collect::<Vec<_>>();
        owned.push((root_address, root_bytes));
        let borrowed = owned
            .iter()
            .map(|(address, bytes)| (*address, bytes.as_slice()))
            .collect::<Vec<_>>();
        let chunks = object_store.chunk_store();
        chunks.write(&borrowed)?;
        for (address, expected) in &owned {
            let actual = chunks.get(*address)?.ok_or_else(|| {
                GraphStoreError::new(
                    "doc_tree_chunk_missing_after_write",
                    format!("document chunk {address} is absent after write"),
                )
            })?;
            if actual.as_ref() != expected.as_slice() {
                return Err(GraphStoreError::new(
                    "doc_tree_chunk_readback_mismatch",
                    format!("document chunk {address} did not round-trip byte-identically"),
                ));
            }
        }
        self.root_address = Some(root_address);
        Ok(root_address)
    }

    /// Publish the document root through the journal's atomic named-root
    /// catalog update. Unrelated graph and RedCore roots are preserved.
    fn publish(
        &mut self,
        object_store: &DiskObjectStore,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        let root = self.write_root(object_store)?;
        object_store.chunk_store().commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_DOCUMENTS.to_string(), Some(root))]),
            meta,
        )
    }

    /// Decode the retired `documents` named root for one-time graph import.
    /// `Ok(None)` means the store has no committed catalog or no legacy root.
    pub fn open_legacy_committed(object_store: &DiskObjectStore) -> GraphStoreResult<Option<Self>> {
        Self::open_committed_impl(object_store)
    }

    fn open_committed_impl(object_store: &DiskObjectStore) -> GraphStoreResult<Option<Self>> {
        let chunks = object_store.chunk_store();
        let Some(catalog_address) = chunks.read_root()? else {
            return Ok(None);
        };
        let catalog = validate_catalog_root(chunks.as_chunk_store(), catalog_address)?;
        let Some(root_address) = catalog.get(ROOT_NAME_DOCUMENTS) else {
            return Ok(None);
        };
        let root_bytes = chunks.get(root_address)?.ok_or_else(|| {
            GraphStoreError::new(
                "doc_tree_root_missing",
                format!("committed document root {root_address} is missing"),
            )
        })?;
        let decoded = decode_doc_root(root_address, &root_bytes)?;
        let mut entries = BTreeMap::new();
        for (path, entry_address) in decoded.entries {
            let bytes = chunks.get(entry_address)?.ok_or_else(|| {
                GraphStoreError::new(
                    "doc_tree_entry_missing",
                    format!(
                        "document root {root_address} points {:?} at missing leaf {entry_address}",
                        path.to_slash_path()
                    ),
                )
            })?;
            let (_, entry) = decode_doc_entry(entry_address, Some(&path), &bytes)?;
            verify_entry_bodies_present(&entry, object_store)?;
            if entries.insert(path.clone(), entry).is_some() {
                return Err(GraphStoreError::new(
                    "doc_tree_root_duplicate_path",
                    format!(
                        "document root contains duplicate path {:?}",
                        path.to_slash_path()
                    ),
                ));
            }
        }
        Ok(Some(Self {
            entries,
            inline_threshold: decoded.inline_threshold,
            root_address: Some(root_address),
            workspace_root: None,
        }))
    }

    /// Remove the retired `documents` root only when the caller proves the
    /// latest committed catalog still names the graph root produced by its
    /// completed import transaction. This makes cleanup a post-durability step
    /// and prevents a stale importer from clearing the only remaining source.
    /// This helper is only for a documents root already sharing the graph's
    /// physical store. For the former standalone `objects/chunk-store.journal`,
    /// use [`DiskObjectStore::clear_legacy_documents_root_after_graph_import`].
    ///
    /// Returns `Ok(None)` when the legacy root is already absent.
    pub fn clear_legacy_committed_root_after_graph_import(
        object_store: &DiskObjectStore,
        expected_graph_root: Hash,
        meta: CommitMeta,
    ) -> GraphStoreResult<Option<CatalogCommitResult>> {
        let chunks = object_store.chunk_store();
        let catalog_address = chunks.read_root()?.ok_or_else(|| {
            GraphStoreError::new(
                "legacy_doc_tree_catalog_missing",
                "cannot clear the legacy document root without a committed catalog",
            )
        })?;
        let catalog = validate_catalog_root(chunks.as_chunk_store(), catalog_address)?;
        let actual_graph_root = catalog.graph_root()?;
        if actual_graph_root != expected_graph_root {
            return Err(GraphStoreError::new(
                "legacy_doc_tree_graph_import_not_durable",
                format!(
                    "catalog graph root is {actual_graph_root}, expected imported root {expected_graph_root}"
                ),
            ));
        }
        if catalog.get(ROOT_NAME_DOCUMENTS).is_none() {
            return Ok(None);
        }
        chunks
            .commit_catalog_updates(
                &BTreeMap::from([(ROOT_NAME_DOCUMENTS.to_string(), None)]),
                meta,
            )
            .map(Some)
    }

    pub fn open_legacy_committed_with_workspace_root(
        object_store: &DiskObjectStore,
        workspace_root: impl Into<PathBuf>,
    ) -> GraphStoreResult<Option<Self>> {
        let workspace_root = workspace_root.into();
        Self::open_legacy_committed(object_store).map(|tree| {
            tree.map(|mut tree| {
                tree.workspace_root = Some(workspace_root);
                tree
            })
        })
    }
}

/// Explicitly test-only builder for historical `documents` roots.
///
/// Production code must never construct or publish this namespace. This type
/// exists solely so migration and reachability tests can seed the exact bytes
/// emitted by pre-graph RustyRed releases.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct LegacyDocTreeFixture {
    tree: LegacyDocTree,
}

impl LegacyDocTreeFixture {
    pub fn new(inline_threshold: usize) -> Self {
        Self {
            tree: LegacyDocTree::empty(inline_threshold),
        }
    }

    pub fn put_body(
        &mut self,
        path: PathKey,
        body: &[u8],
        tier: ColdTierKind,
        created_ms: i64,
        gist: Option<String>,
        object_store: &DiskObjectStore,
    ) -> GraphStoreResult<DocEntry> {
        self.tree
            .put_body(path, body, tier, created_ms, gist, object_store)
    }

    pub fn publish(
        &mut self,
        object_store: &DiskObjectStore,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        self.tree.publish(object_store, meta)
    }

    pub fn get(&self, path: &PathKey) -> Option<&DocEntry> {
        self.tree.get(path)
    }

    pub fn resolve_body(
        &self,
        path: &PathKey,
        object_store: &DiskObjectStore,
    ) -> GraphStoreResult<Option<Vec<u8>>> {
        self.tree.resolve_body(path, object_store)
    }
}

fn resolve_entry_body(
    entry: &DocEntry,
    workspace_root: Option<&Path>,
    object_store: &DiskObjectStore,
) -> GraphStoreResult<Vec<u8>> {
    if let (Some(root), Some(hint)) = (workspace_root, entry.real_path.as_deref()) {
        if let Some(body) = read_verified_hint(root, Path::new(hint), entry) {
            return Ok(body);
        }
    }
    let Some(hash) = entry.content_hash.as_ref() else {
        return Err(GraphStoreError::new(
            "invalid_doc_entry",
            "document entry has neither inline body nor content hash",
        ));
    };
    if let Some(body) = object_store.get_document_bytes_read_only(hash)? {
        verify_body(entry, &body, "stored document body")?;
        return Ok(body);
    }
    // Legacy/emergency fallback only. Canonical reopen rejects a tree whose raw
    // body chunk is missing, so normal reads always reach the one store above.
    if let Some(inline) = entry.inline.as_ref() {
        let body = if entry.inline_compressed {
            decompress_cold_bytes(inline)?
        } else {
            inline.clone()
        };
        verify_body(entry, &body, "inline fallback body")?;
        return Ok(body);
    }
    Err(GraphStoreError::new(
        "missing_doc_body",
        format!("document body {hash} is absent after the path hint failed verification"),
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DocReachability {
    Root { entries: Vec<Hash> },
    Entry { bodies: Vec<Hash> },
}

/// Strict format seam for the global typed reachability walker.
pub(crate) fn decode_doc_reachability(
    expected: Hash,
    bytes: &[u8],
) -> GraphStoreResult<DocReachability> {
    verify_physical_address(expected, bytes)?;
    match bytes.first().copied() {
        Some(DOMAIN_DOC_ROOT) => Ok(DocReachability::Root {
            entries: decode_doc_root(expected, bytes)?
                .entries
                .into_iter()
                .map(|(_, address)| address)
                .collect(),
        }),
        Some(DOMAIN_DOC_ENTRY) => {
            let (_, entry) = decode_doc_entry(expected, None, bytes)?;
            let mut bodies = Vec::with_capacity(1 + entry.previous_hashes.len());
            if let Some(current) = entry.content_hash.as_deref() {
                bodies.push(Hash::parse_prefixed(current)?);
            }
            for previous in &entry.previous_hashes {
                bodies.push(Hash::parse_prefixed(previous)?);
            }
            Ok(DocReachability::Entry { bodies })
        }
        Some(domain) => Err(GraphStoreError::new(
            "doc_tree_domain_invalid",
            format!("document chunk {expected} has unsupported domain {domain}"),
        )),
        None => Err(GraphStoreError::new(
            "doc_tree_chunk_truncated",
            format!("document chunk {expected} is empty"),
        )),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodedDocRoot {
    inline_threshold: usize,
    entries: Vec<(PathKey, Hash)>,
}

fn encode_doc_root<'a>(
    inline_threshold: usize,
    entries: impl IntoIterator<Item = (&'a PathKey, Hash)>,
) -> GraphStoreResult<Vec<u8>> {
    let entries = entries.into_iter().collect::<Vec<_>>();
    if entries.len() > MAX_DOC_ENTRIES {
        return Err(GraphStoreError::new(
            "doc_tree_entry_count_too_large",
            format!("document root has {} entries", entries.len()),
        ));
    }
    let mut output = Vec::new();
    output.push(DOMAIN_DOC_ROOT);
    output.push(DOC_TREE_FORMAT_VERSION);
    output.extend_from_slice(&checked_u64(inline_threshold, "inline threshold")?.to_le_bytes());
    output.extend_from_slice(&checked_u32(entries.len(), "entry count")?.to_le_bytes());
    let mut previous: Option<&PathKey> = None;
    for (path, address) in entries {
        validate_path_key_bytes(path.as_bytes())?;
        if previous.is_some_and(|previous| previous >= path) {
            return Err(GraphStoreError::new(
                "doc_tree_root_not_canonical",
                "document root paths must be strictly increasing",
            ));
        }
        push_bytes(&mut output, path.as_bytes(), "document path")?;
        output.extend_from_slice(address.as_bytes());
        previous = Some(path);
    }
    Ok(output)
}

fn decode_doc_root(expected: Hash, bytes: &[u8]) -> GraphStoreResult<DecodedDocRoot> {
    verify_physical_address(expected, bytes)?;
    let mut decoder = DocDecoder::new(bytes, "document root");
    decoder.expect(DOMAIN_DOC_ROOT, "domain")?;
    decoder.expect(DOC_TREE_FORMAT_VERSION, "format version")?;
    let inline_threshold = decoder.usize("inline threshold")?;
    let count = decoder.count("entry count", MAX_DOC_ENTRIES)?;
    let mut entries = Vec::with_capacity(count);
    let mut previous: Option<PathKey> = None;
    for _ in 0..count {
        let path =
            PathKey::from_bytes(decoder.bytes("document path", MAX_DOC_PATH_BYTES)?.to_vec())?;
        if previous.as_ref().is_some_and(|previous| previous >= &path) {
            return Err(GraphStoreError::new(
                "doc_tree_root_not_canonical",
                "document root paths must be strictly increasing",
            ));
        }
        let address = decoder.hash("document entry address")?;
        previous = Some(path.clone());
        entries.push((path, address));
    }
    decoder.finish()?;
    let decoded = DecodedDocRoot {
        inline_threshold,
        entries,
    };
    let canonical = encode_doc_root(
        decoded.inline_threshold,
        decoded
            .entries
            .iter()
            .map(|(path, address)| (path, *address)),
    )?;
    if canonical != bytes {
        return Err(GraphStoreError::new(
            "doc_tree_root_not_canonical",
            format!("document root {expected} is not canonically encoded"),
        ));
    }
    Ok(decoded)
}

fn encode_doc_entry(path: &PathKey, entry: &DocEntry) -> GraphStoreResult<Vec<u8>> {
    validate_path_key_bytes(path.as_bytes())?;
    let mut output = Vec::new();
    output.push(DOMAIN_DOC_ENTRY);
    output.push(DOC_TREE_FORMAT_VERSION);
    push_bytes(&mut output, path.as_bytes(), "document path")?;

    let current = entry
        .content_hash
        .as_deref()
        .map(Hash::parse_prefixed)
        .transpose()?;
    match (current, entry.inline.as_deref()) {
        (None, None) => {
            if entry.real_path.is_some()
                || entry.size != 0
                || !entry.previous_hashes.is_empty()
                || entry.history_elided_count != 0
            {
                return Err(invalid_entry(
                    "bodyless document entry carries body metadata or history",
                ));
            }
            output.push(DOC_BODY_NONE);
        }
        (None, Some(_)) => return Err(invalid_entry("inline body has no content hash")),
        (Some(address), Some(inline)) => {
            let raw = if entry.inline_compressed {
                decompress_cold_bytes(inline)?
            } else {
                inline.to_vec()
            };
            if raw.len() > MAX_DOC_STRING_BYTES {
                return Err(invalid_entry(
                    "inline body exceeds the canonical byte limit",
                ));
            }
            verify_body_bytes(address, entry.size, &raw, "inline document body")?;
            output.push(DOC_BODY_INLINE);
            output.extend_from_slice(address.as_bytes());
            push_bytes(&mut output, &raw, "inline body")?;
        }
        (Some(address), None) => {
            output.push(DOC_BODY_OVERFLOW);
            output.extend_from_slice(address.as_bytes());
        }
    }

    match entry.real_path.as_deref() {
        None => output.push(0),
        Some(hint) => {
            if hint.len() > MAX_DOC_PATH_BYTES {
                return Err(invalid_entry(
                    "relative path hint exceeds the canonical byte limit",
                ));
            }
            let normalized = validate_relative_hint(Path::new(hint))?;
            let normalized = normalized
                .to_str()
                .ok_or_else(|| invalid_entry("relative path hint is not valid UTF-8"))?;
            if normalized != hint {
                return Err(invalid_entry("relative path hint is not normalized"));
            }
            output.push(1);
            push_string(&mut output, hint, "relative path hint")?;
        }
    }
    output.push(match entry.tier {
        ColdTierKind::Cold => 0,
        ColdTierKind::Warm => 1,
    });
    output.extend_from_slice(&entry.size.to_le_bytes());
    output.extend_from_slice(&entry.created_ms.to_le_bytes());
    match entry.gist.as_deref() {
        None => output.push(0),
        Some(gist) => {
            if gist.len() > MAX_DOC_STRING_BYTES {
                return Err(invalid_entry("gist exceeds the canonical byte limit"));
            }
            output.push(1);
            push_string(&mut output, gist, "gist")?;
        }
    }
    output.extend_from_slice(&entry.history_elided_count.to_le_bytes());
    if entry.previous_hashes.len() > MAX_PREVIOUS_HASHES {
        return Err(invalid_entry("retained history exceeds the maximum"));
    }
    output.extend_from_slice(
        &checked_u32(entry.previous_hashes.len(), "history count")?.to_le_bytes(),
    );
    let mut seen = BTreeSet::new();
    if let Some(current) = current {
        seen.insert(current);
    }
    for previous in &entry.previous_hashes {
        let previous = Hash::parse_prefixed(previous)?;
        if !seen.insert(previous) {
            return Err(invalid_entry(
                "retained history contains the current or a duplicate hash",
            ));
        }
        output.extend_from_slice(previous.as_bytes());
    }
    Ok(output)
}

fn decode_doc_entry(
    expected: Hash,
    expected_path: Option<&PathKey>,
    bytes: &[u8],
) -> GraphStoreResult<(PathKey, DocEntry)> {
    verify_physical_address(expected, bytes)?;
    let mut decoder = DocDecoder::new(bytes, "document entry");
    decoder.expect(DOMAIN_DOC_ENTRY, "domain")?;
    decoder.expect(DOC_TREE_FORMAT_VERSION, "format version")?;
    let path = PathKey::from_bytes(decoder.bytes("document path", MAX_DOC_PATH_BYTES)?.to_vec())?;
    if expected_path.is_some_and(|expected_path| expected_path != &path) {
        return Err(GraphStoreError::new(
            "doc_tree_entry_path_mismatch",
            format!("entry {expected} is bound to {:?}", path.to_slash_path()),
        ));
    }
    let body_kind = decoder.byte("body kind")?;
    let (content_hash, inline) = match body_kind {
        DOC_BODY_NONE => (None, None),
        DOC_BODY_INLINE => {
            let address = decoder.hash("content hash")?;
            let body = decoder.bytes("inline body", MAX_DOC_STRING_BYTES)?.to_vec();
            (Some(address.to_prefixed()), Some(body))
        }
        DOC_BODY_OVERFLOW => (Some(decoder.hash("content hash")?.to_prefixed()), None),
        other => {
            return Err(invalid_entry(format!(
                "document body kind {other} is unsupported"
            )))
        }
    };
    let real_path = match decoder.byte("relative hint tag")? {
        0 => None,
        1 => Some(decoder.string("relative path hint", MAX_DOC_PATH_BYTES)?),
        other => {
            return Err(invalid_entry(format!(
                "relative hint tag {other} is invalid"
            )))
        }
    };
    let tier = match decoder.byte("tier")? {
        0 => ColdTierKind::Cold,
        1 => ColdTierKind::Warm,
        other => return Err(invalid_entry(format!("document tier {other} is invalid"))),
    };
    let size = decoder.u64("body size")?;
    let created_ms = decoder.i64("creation time")?;
    let gist = match decoder.byte("gist tag")? {
        0 => None,
        1 => Some(decoder.string("gist", MAX_DOC_STRING_BYTES)?),
        other => return Err(invalid_entry(format!("gist tag {other} is invalid"))),
    };
    let history_elided_count = decoder.u64("elided history count")?;
    let history_count = decoder.count("history count", MAX_PREVIOUS_HASHES)?;
    let mut previous_hashes = Vec::with_capacity(history_count);
    for _ in 0..history_count {
        previous_hashes.push(decoder.hash("previous body hash")?.to_prefixed());
    }
    decoder.finish()?;
    let entry = DocEntry {
        content_hash,
        inline,
        real_path,
        inline_compressed: false,
        tier,
        size,
        created_ms,
        gist,
        previous_hashes,
        history_elided_count,
    };
    let canonical = encode_doc_entry(&path, &entry)?;
    if canonical != bytes {
        return Err(GraphStoreError::new(
            "doc_tree_entry_not_canonical",
            format!("document entry {expected} is not canonically encoded"),
        ));
    }
    Ok((path, entry))
}

fn prepare_entry_for_persist(
    entry: &mut DocEntry,
    workspace_root: Option<&Path>,
    object_store: &DiskObjectStore,
) -> GraphStoreResult<()> {
    if entry.inline_compressed {
        let inline = entry
            .inline
            .as_deref()
            .ok_or_else(|| invalid_entry("inline_compressed is set without inline bytes"))?;
        entry.inline = Some(decompress_cold_bytes(inline)?);
        entry.inline_compressed = false;
    }
    let Some(current) = entry.content_hash.as_deref() else {
        if entry.inline.is_some() || entry.real_path.is_some() || entry.size != 0 {
            return Err(invalid_entry("bodyless entry carries body metadata"));
        }
        if !entry.previous_hashes.is_empty() {
            entry.history_elided_count = entry
                .history_elided_count
                .saturating_add(u64::try_from(entry.previous_hashes.len()).unwrap_or(u64::MAX));
            entry.previous_hashes.clear();
        }
        return Ok(());
    };
    let expected = Hash::parse_prefixed(current)?;
    let stored = object_store.get_document_bytes(current)?;
    let body = match stored {
        Some(body) => body,
        None => {
            let fallback = if let Some(inline) = entry.inline.as_ref() {
                Some(inline.clone())
            } else if let (Some(root), Some(hint)) = (workspace_root, entry.real_path.as_deref()) {
                read_verified_hint(root, Path::new(hint), entry)
            } else {
                None
            }
            .ok_or_else(|| {
                GraphStoreError::new(
                    "missing_doc_body",
                    format!("document body {current} has no durable or verified fallback bytes"),
                )
            })?;
            let stored = object_store.put_document_bytes(&fallback)?;
            if stored != current {
                return Err(GraphStoreError::new(
                    "doc_tree_body_address_mismatch",
                    format!("fallback body stored as {stored}, expected {current}"),
                ));
            }
            fallback
        }
    };
    verify_body_bytes(expected, entry.size, &body, "document body")?;

    let mut retained = Vec::with_capacity(entry.previous_hashes.len());
    let mut seen = BTreeSet::from([expected]);
    let mut elided = 0_u64;
    for previous in &entry.previous_hashes {
        let address = Hash::parse_prefixed(previous)?;
        if !seen.insert(address) {
            continue;
        }
        match object_store.get_document_bytes(previous)? {
            Some(body) if Hash::of(&body) == address => retained.push(previous.clone()),
            _ => elided = elided.saturating_add(1),
        }
    }
    if retained.len() > MAX_PREVIOUS_HASHES {
        elided = elided.saturating_add(
            u64::try_from(retained.len() - MAX_PREVIOUS_HASHES).unwrap_or(u64::MAX),
        );
        retained.truncate(MAX_PREVIOUS_HASHES);
    }
    entry.previous_hashes = retained;
    entry.history_elided_count = entry.history_elided_count.saturating_add(elided);
    Ok(())
}

fn verify_entry_bodies_present(
    entry: &DocEntry,
    object_store: &DiskObjectStore,
) -> GraphStoreResult<()> {
    let chunks = object_store.chunk_store();
    if let Some(current) = entry.content_hash.as_deref() {
        let expected = Hash::parse_prefixed(current)?;
        let body = chunks.get(expected)?.ok_or_else(|| {
            GraphStoreError::new(
                "missing_doc_body",
                format!("document entry points at missing current body {current}"),
            )
        })?;
        verify_body_bytes(expected, entry.size, body.as_ref(), "current document body")?;
    }
    for previous in &entry.previous_hashes {
        let expected = Hash::parse_prefixed(previous)?;
        let body = chunks.get(expected)?.ok_or_else(|| {
            GraphStoreError::new(
                "missing_doc_body",
                format!("document entry points at missing history body {previous}"),
            )
        })?;
        if Hash::of(body.as_ref()) != expected {
            return Err(GraphStoreError::new(
                "doc_tree_body_address_mismatch",
                format!("history body {previous} failed content verification"),
            ));
        }
    }
    Ok(())
}

fn verify_body(entry: &DocEntry, body: &[u8], context: &str) -> GraphStoreResult<()> {
    let expected = entry
        .content_hash
        .as_deref()
        .ok_or_else(|| invalid_entry(format!("{context} has no content hash")))?;
    verify_body_bytes(Hash::parse_prefixed(expected)?, entry.size, body, context)
}

fn verify_body_bytes(
    expected: Hash,
    expected_size: u64,
    body: &[u8],
    context: &str,
) -> GraphStoreResult<()> {
    let actual_size = u64::try_from(body.len())
        .map_err(|_| GraphStoreError::new("doc_tree_size_overflow", "body length exceeds u64"))?;
    let actual = Hash::of(body);
    if actual != expected || actual_size != expected_size {
        return Err(GraphStoreError::new(
            "doc_tree_body_address_mismatch",
            format!(
                "{context} hashes to {actual} at {actual_size} bytes, expected {expected} at {expected_size} bytes"
            ),
        ));
    }
    Ok(())
}

fn read_verified_hint(root: &Path, hint: &Path, entry: &DocEntry) -> Option<Vec<u8>> {
    let hint = validate_relative_hint(hint).ok()?;
    let canonical_root = root.canonicalize().ok()?;
    let canonical_candidate = canonical_root.join(hint).canonicalize().ok()?;
    if !canonical_candidate.starts_with(&canonical_root) {
        return None;
    }
    let file = std::fs::File::open(canonical_candidate).ok()?;
    let mut body = Vec::new();
    file.take(entry.size.saturating_add(1))
        .read_to_end(&mut body)
        .ok()?;
    (verify_body(entry, &body, "workspace path hint").is_ok()).then_some(body)
}

fn validate_relative_hint(path: &Path) -> GraphStoreResult<PathBuf> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(invalid_hint("path hint must be non-empty and relative"));
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(segment) if !segment.is_empty() && segment.to_str().is_some() => {
                normalized.push(segment)
            }
            _ => {
                return Err(invalid_hint(
                    "path hint must contain only normal UTF-8 components",
                ))
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(invalid_hint("path hint must not normalize to empty"));
    }
    Ok(normalized)
}

fn validate_path_key_bytes(bytes: &[u8]) -> GraphStoreResult<()> {
    if bytes.is_empty() {
        return Err(invalid_path("path key must not be empty"));
    }
    if bytes.len() > MAX_DOC_PATH_BYTES {
        return Err(invalid_path("path key exceeds the canonical byte limit"));
    }
    if bytes.first() == Some(&PATH_SEPARATOR)
        || bytes.last() == Some(&PATH_SEPARATOR)
        || bytes
            .windows(2)
            .any(|pair| pair == [PATH_SEPARATOR, PATH_SEPARATOR])
    {
        return Err(invalid_path("path key contains an empty segment"));
    }
    for segment in bytes.split(|byte| *byte == PATH_SEPARATOR) {
        let segment = std::str::from_utf8(segment)
            .map_err(|_| invalid_path("path key segments must be valid UTF-8"))?;
        if segment.contains('/') {
            return Err(invalid_path("path key segments must not contain slash"));
        }
        if matches!(segment, "." | "..") {
            return Err(invalid_path("path traversal segments are not allowed"));
        }
    }
    Ok(())
}

fn validate_path_prefix_bytes(bytes: &[u8]) -> GraphStoreResult<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    if bytes.len() > MAX_DOC_PATH_BYTES.saturating_add(1) {
        return Err(invalid_path("path prefix exceeds the canonical byte limit"));
    }
    if bytes.last() != Some(&PATH_SEPARATOR) {
        return Err(invalid_path(
            "non-root path prefix must end at a NUL segment boundary",
        ));
    }
    validate_path_key_bytes(&bytes[..bytes.len() - 1])
}

fn verify_physical_address(expected: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual != expected {
        return Err(GraphStoreError::new(
            "doc_tree_chunk_address_mismatch",
            format!("document chunk bytes hash to {actual}, expected {expected}"),
        ));
    }
    Ok(())
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8], field: &str) -> GraphStoreResult<()> {
    output.extend_from_slice(&checked_u32(bytes.len(), field)?.to_le_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

fn push_string(output: &mut Vec<u8>, value: &str, field: &str) -> GraphStoreResult<()> {
    push_bytes(output, value.as_bytes(), field)
}

fn checked_u32(value: usize, field: &str) -> GraphStoreResult<u32> {
    u32::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "doc_tree_length_overflow",
            format!("{field} exceeds the canonical u32 limit"),
        )
    })
}

fn checked_u64(value: usize, field: &str) -> GraphStoreResult<u64> {
    u64::try_from(value).map_err(|_| {
        GraphStoreError::new(
            "doc_tree_length_overflow",
            format!("{field} exceeds the canonical u64 limit"),
        )
    })
}

struct DocDecoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    context: &'static str,
}

impl<'a> DocDecoder<'a> {
    fn new(bytes: &'a [u8], context: &'static str) -> Self {
        Self {
            bytes,
            offset: 0,
            context,
        }
    }

    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| self.truncated(field))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| self.truncated(field))?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn expect(&mut self, expected: u8, field: &str) -> GraphStoreResult<()> {
        let actual = self.byte(field)?;
        if actual != expected {
            return Err(GraphStoreError::new(
                "doc_tree_format_invalid",
                format!("{} {field} is {actual}, expected {expected}", self.context),
            ));
        }
        Ok(())
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, field)?.try_into().expect("four-byte slice"),
        ))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, field)?.try_into().expect("eight-byte slice"),
        ))
    }

    fn i64(&mut self, field: &str) -> GraphStoreResult<i64> {
        Ok(i64::from_le_bytes(
            self.take(8, field)?.try_into().expect("eight-byte slice"),
        ))
    }

    fn usize(&mut self, field: &str) -> GraphStoreResult<usize> {
        usize::try_from(self.u64(field)?).map_err(|_| {
            GraphStoreError::new(
                "doc_tree_length_overflow",
                format!("{} {field} exceeds usize", self.context),
            )
        })
    }

    fn count(&mut self, field: &str, maximum: usize) -> GraphStoreResult<usize> {
        let count = usize::try_from(self.u32(field)?)
            .map_err(|_| GraphStoreError::new("doc_tree_length_overflow", "count exceeds usize"))?;
        if count > maximum {
            return Err(GraphStoreError::new(
                "doc_tree_count_too_large",
                format!("{} {field} {count} exceeds {maximum}", self.context),
            ));
        }
        Ok(count)
    }

    fn bytes(&mut self, field: &str, maximum: usize) -> GraphStoreResult<&'a [u8]> {
        let length = self.count(field, maximum)?;
        self.take(length, field)
    }

    fn string(&mut self, field: &str, maximum: usize) -> GraphStoreResult<String> {
        let bytes = self.bytes(field, maximum)?;
        std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|_| invalid_entry(format!("{field} is not valid UTF-8")))
    }

    fn hash(&mut self, field: &str) -> GraphStoreResult<Hash> {
        Ok(Hash::from_bytes(
            self.take(32, field)?
                .try_into()
                .expect("thirty-two-byte slice"),
        ))
    }

    fn finish(self) -> GraphStoreResult<()> {
        if self.offset != self.bytes.len() {
            return Err(GraphStoreError::new(
                "doc_tree_trailing_bytes",
                format!(
                    "{} has {} trailing bytes",
                    self.context,
                    self.bytes.len() - self.offset
                ),
            ));
        }
        Ok(())
    }

    fn truncated(&self, field: &str) -> GraphStoreError {
        GraphStoreError::new(
            "doc_tree_chunk_truncated",
            format!("{} is truncated while reading {field}", self.context),
        )
    }
}

fn invalid_entry(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("invalid_doc_entry", message.into())
}

fn validate_legacy_entry(entry: &DocEntry) -> GraphStoreResult<()> {
    let current = entry
        .content_hash
        .as_deref()
        .map(Hash::parse_prefixed)
        .transpose()?;
    if let Some(inline) = entry.inline.as_deref() {
        if inline.len() > MAX_DOC_STRING_BYTES {
            return Err(invalid_entry(
                "legacy inline body exceeds the bounded import limit",
            ));
        }
        if current.is_none() {
            return Err(invalid_entry("legacy inline body has no content hash"));
        }
        if !entry.inline_compressed && u64::try_from(inline.len()).ok() != Some(entry.size) {
            return Err(invalid_entry(
                "legacy inline body length does not match declared size",
            ));
        }
    } else if entry.inline_compressed {
        return Err(invalid_entry(
            "legacy inline_compressed flag is set without inline bytes",
        ));
    }
    if current.is_none() && (entry.inline.is_some() || entry.real_path.is_some() || entry.size != 0)
    {
        return Err(invalid_entry(
            "legacy bodyless entry carries body bytes, a path hint, or nonzero size",
        ));
    }
    if entry
        .real_path
        .as_deref()
        .is_some_and(|path| path.len() > MAX_DOC_PATH_BYTES)
    {
        return Err(invalid_entry(
            "legacy real-path hint exceeds the bounded import limit",
        ));
    }
    if entry
        .gist
        .as_deref()
        .is_some_and(|gist| gist.len() > MAX_DOC_STRING_BYTES)
    {
        return Err(invalid_entry(
            "legacy gist exceeds the bounded import limit",
        ));
    }
    if entry.previous_hashes.len() > MAX_PREVIOUS_HASHES {
        return Err(invalid_entry(
            "legacy retained history exceeds the bounded import limit",
        ));
    }
    let mut seen = BTreeSet::new();
    if let Some(current) = current {
        seen.insert(current);
    }
    for previous in &entry.previous_hashes {
        let previous = Hash::parse_prefixed(previous)?;
        if !seen.insert(previous) {
            return Err(invalid_entry(
                "legacy retained history contains the current or a duplicate hash",
            ));
        }
    }
    Ok(())
}

fn invalid_hint(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("invalid_doc_tree_real_path_hint", message.into())
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

fn encode_segments<'a>(segments: impl IntoIterator<Item = &'a str>) -> GraphStoreResult<Vec<u8>> {
    let mut out = Vec::new();
    let mut count = 0usize;
    for segment in segments {
        if segment.is_empty() {
            return Err(invalid_path("path segment must not be empty"));
        }
        if segment.as_bytes().contains(&PATH_SEPARATOR) || segment.contains('/') {
            return Err(invalid_path("path segment must not contain NUL or slash"));
        }
        if matches!(segment, "." | "..") {
            return Err(invalid_path("path traversal segments are not allowed"));
        }
        if count > 0 {
            out.push(PATH_SEPARATOR);
        }
        out.extend_from_slice(segment.as_bytes());
        count += 1;
    }
    if count == 0 {
        return Err(invalid_path("path requires at least one segment"));
    }
    if out.len() > MAX_DOC_PATH_BYTES {
        return Err(invalid_path("path key exceeds the canonical byte limit"));
    }
    Ok(out)
}

fn invalid_path(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("invalid_doc_tree_path", message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{ChunkStore, RootCatalog};
    use crate::journal::JournalChunkStore;
    use crate::object_store::UnifiedChunkStore;
    use std::sync::Arc;

    fn temp_store(name: &str) -> (std::path::PathBuf, DiskObjectStore) {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = DiskObjectStore::open(&dir).unwrap();
        (dir, store)
    }

    fn test_meta(version: u64) -> CommitMeta {
        CommitMeta {
            parents: Vec::new(),
            author: "doc-tree-test".to_string(),
            message: format!("document root {version}"),
            timestamp_unix_ms: u128::from(version),
            graph_version: version,
        }
    }

    fn journal_store(name: &str) -> (std::path::PathBuf, std::path::PathBuf, DiskObjectStore) {
        let dir = std::env::temp_dir().join(format!(
            "{name}-journal-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chunks.journal");
        let journal = Arc::new(JournalChunkStore::open(&path).unwrap());
        let graph_bytes = b"document-test graph root";
        let graph = Hash::of(graph_bytes);
        journal.write(&[(graph, graph_bytes.as_slice())]).unwrap();
        let catalog = RootCatalog::graph(graph);
        let catalog_address = catalog.write_to(journal.as_ref()).unwrap();
        journal.commit(catalog_address, test_meta(0)).unwrap();
        let unified: Arc<dyn UnifiedChunkStore> = journal;
        let store = DiskObjectStore::from_chunk_store(unified).unwrap();
        (dir, path, store)
    }

    #[test]
    fn path_key_and_prefix_are_distinct_validated_raw_encodings() {
        let segments = (0..128)
            .map(|index| format!("segment-{index:03}"))
            .collect::<Vec<_>>();
        let borrowed = segments.iter().map(String::as_str).collect::<Vec<_>>();
        let key = PathKey::from_segments(borrowed.iter().copied()).unwrap();
        let prefix = PathPrefix::from_segments(borrowed.iter().copied()).unwrap();
        assert_eq!(prefix.as_bytes().last(), Some(&PATH_SEPARATOR));
        assert_eq!(
            &prefix.as_bytes()[..prefix.as_bytes().len() - 1],
            key.as_bytes()
        );
        assert!(prefix.matches(
            &PathKey::from_segments(
                borrowed
                    .iter()
                    .copied()
                    .chain(std::iter::once("descendant"))
            )
            .unwrap()
        ));
        assert_eq!(PathPrefix::root().as_bytes(), b"");
        assert!(PathPrefix::root().matches(&key));

        let control = PathKey::from_segments(["unicodé", "line\n\tbreak"]).unwrap();
        assert_eq!(control.to_slash_path(), "unicodé/line\n\tbreak");
        assert!(control.as_str().contains('\0'));

        for invalid in [
            "",
            "/absolute",
            "trailing/",
            "double//slash",
            ".",
            "../escape",
        ] {
            assert!(
                PathKey::from_slash_path(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        for invalid in ["", ".", "..", "has/slash", "has\0nul"] {
            assert!(
                PathKey::from_segments([invalid]).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(PathKey::from_bytes(b"a\0".to_vec()).is_err());
        assert!(PathPrefix::from_bytes(b"a".to_vec()).is_err());
        assert!(PathPrefix::from_bytes(b"a\0\0".to_vec()).is_err());
    }

    #[test]
    fn path_prefix_collision_pair_stops_at_segment_boundary() {
        let prefix = PathPrefix::from_segments(["a", "b"]).unwrap();
        assert!(prefix.matches(&PathKey::from_slash_path("a/b/child").unwrap()));
        assert!(!prefix.matches(&PathKey::from_slash_path("a/b2/child").unwrap()));
        assert!(!prefix.matches(&PathKey::from_slash_path("a/b").unwrap()));
    }

    #[test]
    fn prefix_scan_returns_namespace_in_path_order() {
        let (_dir, store) = temp_store("doc-tree-prefix");
        let mut tree = LegacyDocTree::empty(64);
        for path in [
            ["tenant", "project", "episode", "2"],
            ["tenant", "project", "episode", "1"],
            ["tenant", "project2", "episode", "3"],
        ] {
            let key = PathKey::from_segments(path).unwrap();
            tree.put_body(
                key,
                path.join("/").as_bytes(),
                ColdTierKind::Cold,
                1,
                None,
                &store,
            )
            .unwrap();
        }
        let prefix = PathPrefix::from_segments(["tenant", "project"]).unwrap();
        let paths = tree
            .legacy_range(&prefix)
            .map(|(path, _)| path.to_slash_path())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "tenant/project/episode/1".to_string(),
                "tenant/project/episode/2".to_string()
            ]
        );
    }

    #[test]
    fn inline_and_overflow_documents_round_trip_identical_bytes() {
        let (dir, store) = temp_store("doc-tree-roundtrip");
        let mut tree = LegacyDocTree::empty(8);
        let small = b"tiny";
        let large = b"this body is larger than the inline threshold";
        let small_path = PathKey::from_slash_path("tenant/project/doc/small").unwrap();
        let large_path = PathKey::from_slash_path("tenant/project/doc/large").unwrap();

        let small_entry = tree
            .put_body(
                small_path.clone(),
                small,
                ColdTierKind::Cold,
                10,
                Some("small".to_string()),
                &store,
            )
            .unwrap();
        let large_entry = tree
            .put_body(
                large_path.clone(),
                large,
                ColdTierKind::Cold,
                10,
                None,
                &store,
            )
            .unwrap();

        assert!(small_entry.is_inline());
        assert!(!small_entry.inline_compressed);
        assert!(!large_entry.is_inline());
        assert_eq!(
            store
                .get_document_bytes(small_entry.content_hash.as_deref().unwrap())
                .unwrap()
                .as_deref(),
            Some(small.as_slice())
        );
        assert_eq!(
            store
                .get_document_bytes(large_entry.content_hash.as_deref().unwrap())
                .unwrap()
                .as_deref(),
            Some(large.as_slice())
        );
        assert_eq!(
            tree.resolve_body(&small_path, &store).unwrap().unwrap(),
            small
        );
        assert_eq!(
            tree.resolve_body(&large_path, &store).unwrap().unwrap(),
            large
        );
        let expected_large_hash = content_hash_bytes(large);
        assert_eq!(
            large_entry.content_hash.as_deref(),
            Some(expected_large_hash.as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bounded_legacy_json_decoder_rejects_traversal_and_duplicate_paths() {
        let entry = DocEntry {
            content_hash: None,
            inline: None,
            real_path: None,
            inline_compressed: false,
            tier: ColdTierKind::Cold,
            size: 0,
            created_ms: 1,
            gist: None,
            previous_hashes: Vec::new(),
            history_elided_count: 0,
        };
        let bytes = serde_json::to_vec(&vec![("docs/readme".to_string(), entry.clone())]).unwrap();
        let decoded = LegacyDocTree::decode_legacy_json(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded.legacy_entries().next().unwrap().0.as_str(),
            "docs\0readme"
        );

        let traversal =
            serde_json::to_vec(&vec![("docs/../escape".to_string(), entry.clone())]).unwrap();
        assert_eq!(
            LegacyDocTree::decode_legacy_json(&traversal)
                .unwrap_err()
                .code,
            "invalid_doc_tree_path"
        );
        let duplicate = serde_json::to_vec(&vec![
            ("docs/readme".to_string(), entry.clone()),
            ("docs/readme".to_string(), entry),
        ])
        .unwrap();
        assert_eq!(
            LegacyDocTree::decode_legacy_json(&duplicate)
                .unwrap_err()
                .code,
            "legacy_doc_tree_duplicate_path"
        );
    }

    #[test]
    fn history_retains_sixty_four_hashes_and_counts_explicit_elision() {
        let (dir, store) = temp_store("doc-tree-history-elision");
        let mut tree = LegacyDocTree::empty(64);
        let path = PathKey::from_slash_path("tenant/project/doc/versioned").unwrap();
        let mut final_entry = None;
        for version in 0..66 {
            let body = format!("version-{version:03}");
            final_entry = Some(
                tree.put_body(
                    path.clone(),
                    body.as_bytes(),
                    ColdTierKind::Cold,
                    version,
                    None,
                    &store,
                )
                .unwrap(),
            );
        }

        let final_entry = final_entry.unwrap();
        let newest_retained = content_hash_bytes(b"version-064");
        let oldest_retained = content_hash_bytes(b"version-001");
        assert_eq!(final_entry.previous_hashes.len(), MAX_PREVIOUS_HASHES);
        assert_eq!(final_entry.history_elided_count, 1);
        assert_eq!(
            final_entry.previous_hashes.first().map(String::as_str),
            Some(newest_retained.as_str())
        );
        assert_eq!(
            final_entry.previous_hashes.last().map(String::as_str),
            Some(oldest_retained.as_str())
        );
        for retained in &final_entry.previous_hashes {
            assert!(store.get_document_bytes(retained).unwrap().is_some());
        }

        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn publication_reopens_latest_catalog_and_strict_reachability() {
        let (dir, journal_path, store) = journal_store("doc-tree-publish");
        let small_path = PathKey::from_slash_path("tenant/project/doc/small").unwrap();
        let large_path = PathKey::from_slash_path("tenant/project/doc/large").unwrap();
        let mut tree = LegacyDocTree::empty(8);
        let initial_small = tree
            .put_body(
                small_path.clone(),
                b"tiny-v1",
                ColdTierKind::Cold,
                1,
                None,
                &store,
            )
            .unwrap();
        let large = tree
            .put_body(
                large_path.clone(),
                b"this body is larger than the inline threshold",
                ColdTierKind::Cold,
                1,
                None,
                &store,
            )
            .unwrap();

        let first = tree.publish(&store, test_meta(1)).unwrap();
        let first_doc_root = tree.root_address().unwrap();
        let latest_small = tree
            .put_body(
                small_path.clone(),
                b"tiny-v2",
                ColdTierKind::Warm,
                2,
                Some("latest".to_string()),
                &store,
            )
            .unwrap();
        let latest = tree.publish(&store, test_meta(2)).unwrap();
        let latest_doc_root = tree.root_address().unwrap();
        assert_ne!(first_doc_root, latest_doc_root);
        assert_ne!(first.catalog_root, latest.catalog_root);

        let chunks = store.chunk_store();
        assert_eq!(chunks.read_root().unwrap(), Some(latest.catalog_root));
        let catalog = validate_catalog_root(chunks.as_chunk_store(), latest.catalog_root).unwrap();
        assert_eq!(catalog.get(ROOT_NAME_DOCUMENTS), Some(latest_doc_root));
        assert_eq!(
            catalog.get(crate::chunk_store::ROOT_NAME_GRAPH),
            Some(Hash::of(b"document-test graph root")),
            "document publication must preserve the graph named root"
        );

        let root_bytes = chunks.get(latest_doc_root).unwrap().unwrap();
        let entry_addresses = match decode_doc_reachability(latest_doc_root, &root_bytes).unwrap() {
            DocReachability::Root { entries } => entries,
            DocReachability::Entry { .. } => panic!("document root decoded as an entry"),
        };
        assert_eq!(entry_addresses.len(), 2);
        let mut reachable_bodies = BTreeSet::new();
        for entry_address in entry_addresses {
            let entry_bytes = chunks.get(entry_address).unwrap().unwrap();
            match decode_doc_reachability(entry_address, &entry_bytes).unwrap() {
                DocReachability::Entry { bodies } => reachable_bodies.extend(bodies),
                DocReachability::Root { .. } => panic!("document entry decoded as a root"),
            }
        }
        let expected_bodies = BTreeSet::from([
            Hash::parse_prefixed(initial_small.content_hash.as_deref().unwrap()).unwrap(),
            Hash::parse_prefixed(latest_small.content_hash.as_deref().unwrap()).unwrap(),
            Hash::parse_prefixed(large.content_hash.as_deref().unwrap()).unwrap(),
        ]);
        assert_eq!(reachable_bodies, expected_bodies);

        drop(chunks);
        drop(store);
        let reopened_backend = Arc::new(JournalChunkStore::open(&journal_path).unwrap());
        let reopened_unified: Arc<dyn UnifiedChunkStore> = reopened_backend;
        let reopened_store = DiskObjectStore::from_chunk_store(reopened_unified).unwrap();
        let mut reopened = LegacyDocTree::open_legacy_committed(&reopened_store)
            .unwrap()
            .expect("latest catalog must name a document root");
        assert_eq!(reopened.root_address(), Some(latest_doc_root));
        let mut poisoned_inline = reopened.get(&small_path).unwrap().clone();
        assert!(poisoned_inline.inline.is_some());
        poisoned_inline.inline = Some(b"poisoned decoded-cache copy".to_vec());
        reopened.entries.insert(small_path.clone(), poisoned_inline);
        assert_eq!(
            reopened
                .resolve_body(&small_path, &reopened_store)
                .unwrap()
                .unwrap(),
            b"tiny-v2"
        );
        assert_eq!(
            reopened
                .resolve_body(&large_path, &reopened_store)
                .unwrap()
                .unwrap(),
            b"this body is larger than the inline threshold"
        );

        drop(reopened);
        drop(reopened_store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_document_root_clears_only_after_exact_durable_graph_import() {
        let (dir, journal_path, store) = journal_store("doc-tree-clear-after-import");
        let mut tree = LegacyDocTree::empty(8);
        tree.put_body(
            PathKey::from_slash_path("docs/readme").unwrap(),
            b"legacy body",
            ColdTierKind::Cold,
            1,
            None,
            &store,
        )
        .unwrap();
        tree.publish(&store, test_meta(1)).unwrap();
        assert!(LegacyDocTree::open_legacy_committed(&store)
            .unwrap()
            .is_some());

        let chunks = store.chunk_store();
        let catalog_address = chunks.read_root().unwrap().unwrap();
        let catalog = validate_catalog_root(chunks.as_chunk_store(), catalog_address).unwrap();
        let graph_root = catalog.graph_root().unwrap();
        let error = LegacyDocTree::clear_legacy_committed_root_after_graph_import(
            &store,
            Hash::of(b"stale graph import"),
            test_meta(2),
        )
        .unwrap_err();
        assert_eq!(error.code, "legacy_doc_tree_graph_import_not_durable");
        assert!(LegacyDocTree::open_legacy_committed(&store)
            .unwrap()
            .is_some());

        let cleared = LegacyDocTree::clear_legacy_committed_root_after_graph_import(
            &store,
            graph_root,
            test_meta(3),
        )
        .unwrap()
        .expect("legacy root should be removed once");
        assert_eq!(chunks.read_root().unwrap(), Some(cleared.catalog_root));
        assert!(LegacyDocTree::open_legacy_committed(&store)
            .unwrap()
            .is_none());
        assert!(
            LegacyDocTree::clear_legacy_committed_root_after_graph_import(
                &store,
                graph_root,
                test_meta(4),
            )
            .unwrap()
            .is_none()
        );

        drop(chunks);
        drop(store);
        let reopened = Arc::new(JournalChunkStore::open(&journal_path).unwrap());
        let reopened_unified: Arc<dyn UnifiedChunkStore> = reopened;
        let reopened_store = DiskObjectStore::from_chunk_store(reopened_unified).unwrap();
        assert!(LegacyDocTree::open_legacy_committed(&reopened_store)
            .unwrap()
            .is_none());
        drop(reopened_store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn committed_document_root_fails_closed_on_missing_or_corrupt_leaf() {
        let cases = [
            ("missing", None),
            (
                "corrupt",
                Some(vec![DOMAIN_DOC_ENTRY, DOC_TREE_FORMAT_VERSION]),
            ),
        ];
        for (index, (name, leaf_bytes)) in cases.into_iter().enumerate() {
            let (dir, _journal_path, store) = journal_store(name);
            let chunks = store.chunk_store();
            let path = PathKey::from_slash_path("tenant/project/doc/broken").unwrap();
            let leaf_address = leaf_bytes
                .as_deref()
                .map(Hash::of)
                .unwrap_or_else(|| Hash::of(b"intentionally missing document leaf"));
            if let Some(bytes) = leaf_bytes.as_deref() {
                chunks.write(&[(leaf_address, bytes)]).unwrap();
            }
            let root_bytes = encode_doc_root(8, std::iter::once((&path, leaf_address))).unwrap();
            let root_address = Hash::of(&root_bytes);
            chunks
                .write(&[(root_address, root_bytes.as_slice())])
                .unwrap();
            chunks
                .commit_catalog_updates(
                    &BTreeMap::from([(ROOT_NAME_DOCUMENTS.to_string(), Some(root_address))]),
                    test_meta(u64::try_from(index + 1).unwrap()),
                )
                .unwrap();

            let error = LegacyDocTree::open_legacy_committed(&store).unwrap_err();
            if name == "missing" {
                assert_eq!(error.code, "doc_tree_entry_missing");
            } else {
                assert_eq!(error.code, "doc_tree_chunk_truncated");
            }

            drop(chunks);
            drop(store);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
