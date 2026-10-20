//! Replay one RedCore store's retained graph history into another.
//!
//! The source is read through its pinned history; nothing is written to it.
//! Every retained source commit becomes one target commit carrying the
//! source's author, message and timestamp, so the target's history (and every
//! reader built on it, such as Wiki history) shows the same edits in the same
//! order. Commit hashes and transaction numbers cannot be preserved: the
//! target appends after its own head.
//!
//! Three properties make the import safe to run against a store that already
//! holds valuable data:
//!
//! - **Preflight before any write.** [`plan_graph_import`] compares every key
//!   the replay will touch with the target. A key the target changed
//!   independently is a conflict and refuses the whole run, unless the
//!   operator accepted that exact target revision by content address. The
//!   target's prior value is still recoverable from its own history.
//! - **Resumable by construction.** A cursor node in the target names the
//!   last source transaction applied, and it is written in the same atomic
//!   batch as that transaction's records. A re-run, or a later catch-up of
//!   source commits made since, starts exactly after it.
//! - **Payloads before references.** File manifests, file body chunks,
//!   vectors and raw payloads a record references are copied (verified by
//!   address) and pinned before the commit that makes them reachable.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Serialize;
use serde_json::{json, Value};

use crate::chunk_store::{validate_catalog_root, ChunkPublicationWindow, ChunkStore};
use crate::events::CONTROL_EVENT_LABEL;
use crate::file_payload::decode_file_manifest;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::graph_history::{
    GraphCommitMetadata, GraphHistoryCoverage, PinnedGraphHistory, PinnedGraphRevision,
    MAX_HISTORY_PAGE,
};
use crate::graph_store::{unix_ms, RedCoreGraphStore};
use crate::hash::Hash;
use crate::object_store::{DiskObjectStore, UnifiedChunkStore};
use crate::ordered::OrderedDesignation;
use crate::reachability::packed_graph_node_payload_references;
use crate::versioned_graph::GraphPayloadRecord;
use crate::VectorDesignation;

/// Label of the node that records how far a source has been replayed.
pub const GRAPH_IMPORT_CURSOR_LABEL: &str = "GraphImportCursor";
/// Author recorded on the one baseline commit a partial-history source needs.
pub const GRAPH_IMPORT_AUTHOR: &str = "graph-import";

/// Properties whose string values are counted in the tenant census.
const TENANT_PROPERTIES: [&str; 3] = ["tenant", "tenant_id", "tenant_slug"];
/// Catalog roots the graph replay itself carries.
const REPLAYED_CATALOG_ROOTS: [&str; 2] = ["graph", "redcore"];
/// Bound on the example keys a report lists for each kind of finding.
const REPORT_EXAMPLES: usize = 50;

pub fn graph_import_cursor_id(source_id: &str) -> String {
    format!("graph-import:cursor:{source_id}")
}

#[derive(Clone, Debug, Default)]
pub struct GraphImportOptions {
    /// Stable name of the source store; it keys the cursor.
    pub source_id: String,
    /// When set, a counted tenant value outside this set refuses the import.
    pub expected_tenants: Option<BTreeSet<String>>,
    /// Keys the operator allows the source to overwrite, each bound to the
    /// target content address the operator saw. A target that changed again
    /// since no longer matches and stays a conflict.
    pub accepted_overwrites: BTreeMap<String, String>,
    /// Replay at most this many source commits in one run.
    pub max_commits: Option<usize>,
}

/// Where the replay starts in the source's history.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphImportStart {
    /// A previous run recorded this source transaction in the target.
    Resume {
        after_transaction: u64,
        after_commit: Option<String>,
    },
    /// The source retains every commit; replay starts at its first.
    Complete,
    /// The source does not retain its earliest commits: its state at
    /// `transaction` is written as one baseline commit, then replay follows.
    Baseline {
        transaction: u64,
        commit: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GraphImportConflict {
    pub key: String,
    pub target_content_address: Option<String>,
    pub base_content_address: Option<String>,
    pub source_content_address: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GraphImportPlan {
    pub source_id: String,
    pub coverage: GraphHistoryCoverage,
    pub start: GraphImportStart,
    pub source_head_transaction: u64,
    /// Last source transaction this plan reaches; equals the head unless
    /// `max_commits` bounds the run.
    pub through_transaction: u64,
    pub commits_to_replay: usize,
    pub baseline_records: usize,
    pub touched_keys: usize,
    /// Touched keys holding records each store derives for itself on commit
    /// (control events). The target derives its own for the replayed writes;
    /// the source's stay in the source.
    pub derived_keys_skipped: usize,
    /// Keys where the target already holds the source's final value.
    pub converged_keys: usize,
    pub accepted_overwrites: Vec<String>,
    pub conflicts: Vec<GraphImportConflict>,
    pub conflict_count: usize,
    /// `property=value` counts over the source's final records for touched keys.
    pub tenant_census: BTreeMap<String, usize>,
    pub foreign_tenants: Vec<String>,
    pub source_vector_designations: Vec<VectorDesignation>,
    pub source_ordered_designations: Vec<OrderedDesignation>,
    pub designation_conflicts: Vec<String>,
    /// Source catalog roots outside the graph. Payloads referenced from graph
    /// records are copied; anything reachable only from these roots is not.
    pub uncarried_catalog_roots: Vec<String>,
}

impl GraphImportPlan {
    pub fn refusals(&self) -> Vec<String> {
        let mut refusals = Vec::new();
        if self.conflict_count > 0 {
            refusals.push(format!(
                "{} key(s) changed in the target independently of the source",
                self.conflict_count
            ));
        }
        if !self.foreign_tenants.is_empty() {
            refusals.push(format!(
                "records name tenants outside the expected set: {}",
                self.foreign_tenants.join(", ")
            ));
        }
        if !self.designation_conflicts.is_empty() {
            refusals.push(format!(
                "vector designations disagree: {}",
                self.designation_conflicts.join(", ")
            ));
        }
        refusals
    }

    pub fn is_admissible(&self) -> bool {
        self.refusals().is_empty()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GraphImportReport {
    pub commits_written: usize,
    pub baseline_written: bool,
    pub records_upserted: usize,
    pub records_deleted: usize,
    pub chunks_copied: usize,
    pub bytes_copied: u64,
    pub designations_applied: Vec<String>,
    /// Designations the source cleared during the replayed range. The target
    /// keeps its own; clearing one could remove an index the target relies on.
    pub designation_clears_not_applied: Vec<String>,
    pub last_source_transaction: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GraphImportLabelCount {
    pub source: usize,
    pub matched: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GraphImportVerification {
    pub source_transaction: u64,
    pub source_records: usize,
    /// Derived source records (control events), not carried by design.
    pub source_derived_records: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub missing_in_target: usize,
    pub mismatched_keys: Vec<String>,
    pub missing_keys: Vec<String>,
    /// Target records the source never had: the target's own data, kept.
    /// Records the target derived for itself are not counted.
    pub target_only_records: usize,
    pub by_label: BTreeMap<String, GraphImportLabelCount>,
    pub by_edge_type: BTreeMap<String, GraphImportLabelCount>,
    pub payload_refs_checked: usize,
    pub payload_refs_missing: usize,
    pub missing_payloads: Vec<String>,
}

impl GraphImportVerification {
    pub fn is_conserved(&self) -> bool {
        self.mismatched == 0 && self.missing_in_target == 0 && self.payload_refs_missing == 0
    }
}

#[derive(Clone, Debug)]
struct ImportCursor {
    last_source_transaction: u64,
    last_source_commit: Option<String>,
    imported_commits: u64,
}

/// One store's revision as the import reads it. `Empty` is the state before a
/// complete history's first commit.
enum Revision {
    Empty,
    Pinned(Box<PinnedGraphRevision>),
}

impl Revision {
    fn read(&self, key: &str) -> GraphStoreResult<Option<GraphPayloadRecord>> {
        let Self::Pinned(revision) = self else {
            return Ok(None);
        };
        let (kind, id) = split_key(key)?;
        let record = match kind {
            KeyKind::Node => revision.get_node(id)?.map(GraphPayloadRecord::Node),
            KeyKind::Edge => revision.get_edge(id)?.map(GraphPayloadRecord::Edge),
        };
        Ok(record.filter(|record| !is_tombstone(record)))
    }

    fn vector_designations(&self) -> &[VectorDesignation] {
        match self {
            Self::Empty => &[],
            Self::Pinned(revision) => &revision.roots.vector_designations,
        }
    }

    fn ordered_designations(&self) -> &[OrderedDesignation] {
        match self {
            Self::Empty => &[],
            Self::Pinned(revision) => &revision.roots.ordered_designations,
        }
    }
}

#[derive(Clone, Copy)]
enum KeyKind {
    Node,
    Edge,
}

fn split_key(key: &str) -> GraphStoreResult<(KeyKind, &str)> {
    if let Some(id) = key.strip_prefix("node/") {
        return Ok((KeyKind::Node, id));
    }
    if let Some(id) = key.strip_prefix("edge/") {
        return Ok((KeyKind::Edge, id));
    }
    Err(GraphStoreError::new(
        "graph_import_key_unknown",
        format!("changed object key {key:?} is outside the node/ and edge/ namespaces"),
    ))
}

fn record_key(record: &GraphPayloadRecord) -> String {
    match record {
        GraphPayloadRecord::Node(node) => format!("node/{}", node.id),
        GraphPayloadRecord::Edge(edge) => format!("edge/{}", edge.id),
    }
}

fn content_address(record: &GraphPayloadRecord) -> GraphStoreResult<String> {
    match record {
        GraphPayloadRecord::Node(node) => node.content_address(),
        GraphPayloadRecord::Edge(edge) => edge.content_address(),
    }
}

fn optional_address(record: &Option<GraphPayloadRecord>) -> GraphStoreResult<Option<String>> {
    record.as_ref().map(content_address).transpose()
}

/// A delete leaves a tombstoned copy of the prior record in the payload tree.
/// The import compares live state, so a tombstone reads as absence.
fn is_tombstone(record: &GraphPayloadRecord) -> bool {
    match record {
        GraphPayloadRecord::Node(node) => node.tombstone,
        GraphPayloadRecord::Edge(edge) => edge.tombstone,
    }
}

/// A record a store writes for itself on every commit, keyed by its own
/// transaction numbers: a control event. Carrying the source's would collide
/// with, and overwrite, the target's own events.
fn is_derived(record: &GraphPayloadRecord) -> bool {
    matches!(record, GraphPayloadRecord::Node(node)
        if node.labels.iter().any(|label| label == CONTROL_EVENT_LABEL))
}

fn either_derived(before: &Option<GraphPayloadRecord>, after: &Option<GraphPayloadRecord>) -> bool {
    before.as_ref().is_some_and(is_derived) || after.as_ref().is_some_and(is_derived)
}

/// The target's current head, or `Empty` when it has never published a root.
fn target_head(target: &RedCoreGraphStore) -> GraphStoreResult<Revision> {
    match target.pin_graph_history() {
        Ok(history) => Ok(Revision::Pinned(Box::new(history.head_revision()?))),
        Err(error) if error.code == "graph_history_uninitialized" => Ok(Revision::Empty),
        Err(error) => Err(error),
    }
}

fn source_history(source: &RedCoreGraphStore) -> GraphStoreResult<PinnedGraphHistory> {
    source.pin_graph_history().map_err(|error| {
        if error.code == "graph_history_uninitialized" {
            GraphStoreError::new(
                "graph_import_source_empty",
                "the source store has never published a graph root",
            )
        } else {
            error
        }
    })
}

fn read_cursor(target: &Revision, source_id: &str) -> GraphStoreResult<Option<ImportCursor>> {
    let Some(GraphPayloadRecord::Node(node)) =
        target.read(&format!("node/{}", graph_import_cursor_id(source_id)))?
    else {
        return Ok(None);
    };
    let transaction = node
        .properties
        .get("last_source_transaction")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            GraphStoreError::new(
                "graph_import_cursor_invalid",
                format!("cursor {} has no last_source_transaction", node.id),
            )
        })?;
    Ok(Some(ImportCursor {
        last_source_transaction: transaction,
        last_source_commit: node
            .properties
            .get("last_source_commit")
            .and_then(Value::as_str)
            .map(str::to_owned),
        imported_commits: node
            .properties
            .get("imported_commits")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }))
}

fn cursor_record(source_id: &str, cursor: &ImportCursor) -> NodeRecord {
    NodeRecord::new(
        graph_import_cursor_id(source_id),
        [GRAPH_IMPORT_CURSOR_LABEL],
        json!({
            "source_id": source_id,
            "last_source_transaction": cursor.last_source_transaction,
            "last_source_commit": cursor.last_source_commit,
            "imported_commits": cursor.imported_commits,
        }),
    )
}

/// The source revision at a recorded cursor.
fn source_revision_at(
    history: &PinnedGraphHistory,
    transaction: u64,
    commit: Option<&str>,
) -> GraphStoreResult<Revision> {
    let revision = match commit {
        Some(commit) => history.revision(Hash::parse_prefixed(commit)?)?,
        // No commit: the legacy boundary, or the head of a source that has
        // retained no history at all.
        None => match history.legacy_revision()? {
            Some(legacy) => legacy,
            None if history.coverage() == GraphHistoryCoverage::Unavailable => {
                history.head_revision()?
            }
            None => {
                return Err(GraphStoreError::new(
                    "graph_import_cursor_invalid",
                    "cursor names no commit and the source has no legacy boundary",
                ))
            }
        },
    };
    if revision.roots.transaction != transaction {
        return Err(GraphStoreError::new(
            "graph_import_cursor_mismatch",
            format!(
                "cursor names source transaction {transaction}, its revision is at {}",
                revision.roots.transaction
            ),
        ));
    }
    Ok(Revision::Pinned(Box::new(revision)))
}

/// Where replay starts, the revision the target already reflects (`base`) and
/// the baseline revision to write first, if any.
struct StartPoint {
    start: GraphImportStart,
    after_transaction: u64,
    base: Revision,
    baseline: Option<Revision>,
}

fn start_point(
    history: &PinnedGraphHistory,
    cursor: Option<&ImportCursor>,
) -> GraphStoreResult<StartPoint> {
    if let Some(cursor) = cursor {
        return Ok(StartPoint {
            start: GraphImportStart::Resume {
                after_transaction: cursor.last_source_transaction,
                after_commit: cursor.last_source_commit.clone(),
            },
            after_transaction: cursor.last_source_transaction,
            base: source_revision_at(
                history,
                cursor.last_source_transaction,
                cursor.last_source_commit.as_deref(),
            )?,
            baseline: None,
        });
    }
    let baseline = match history.coverage() {
        GraphHistoryCoverage::Complete => {
            return Ok(StartPoint {
                start: GraphImportStart::Complete,
                after_transaction: 0,
                base: Revision::Empty,
                baseline: None,
            });
        }
        GraphHistoryCoverage::Since {
            first_transaction, ..
        } => match history.legacy_revision()? {
            Some(legacy) => legacy,
            None => {
                let first = history.entries_after(first_transaction.saturating_sub(1), 1)?;
                let entry = first.first().ok_or_else(|| {
                    GraphStoreError::new(
                        "graph_history_unavailable",
                        "source history names a first transaction it does not hold",
                    )
                })?;
                history.revision(entry.commit.commit_hash)?
            }
        },
        GraphHistoryCoverage::Unavailable => history.head_revision()?,
    };
    let transaction = baseline.roots.transaction;
    Ok(StartPoint {
        start: GraphImportStart::Baseline {
            transaction,
            commit: baseline
                .commit()
                .map(|commit| commit.commit_hash.to_prefixed()),
        },
        after_transaction: transaction,
        base: Revision::Empty,
        baseline: Some(Revision::Pinned(Box::new(baseline))),
    })
}

/// Every live record of a revision, nodes first, paged from storage.
fn for_each_record(
    revision: &PinnedGraphRevision,
    mut visit: impl FnMut(GraphPayloadRecord) -> GraphStoreResult<()>,
) -> GraphStoreResult<()> {
    for prefix in ["node/", "edge/"] {
        let mut after: Option<String> = None;
        loop {
            let page = revision.records_page(prefix, after.as_deref(), MAX_HISTORY_PAGE)?;
            let Some(last) = page.last() else {
                break;
            };
            after = Some(record_key(last));
            let full = page.len() == MAX_HISTORY_PAGE;
            for record in page.into_iter().filter(|record| !is_tombstone(record)) {
                visit(record)?;
            }
            if !full {
                break;
            }
        }
    }
    Ok(())
}

/// Visit each source history entry after `after_transaction`, in order, up to
/// `limit` entries.
fn for_each_entry(
    history: &PinnedGraphHistory,
    after_transaction: u64,
    limit: usize,
    mut visit: impl FnMut(&crate::graph_history::GraphHistoryEntry) -> GraphStoreResult<()>,
) -> GraphStoreResult<()> {
    // A source without retained history has no entries to page.
    if history.coverage() == GraphHistoryCoverage::Unavailable {
        return Ok(());
    }
    let mut after = after_transaction;
    let mut remaining = limit;
    while remaining > 0 {
        let page = history.entries_after(after, remaining.min(MAX_HISTORY_PAGE))?;
        let Some(last) = page.last() else {
            break;
        };
        after = last.roots.transaction;
        remaining -= page.len();
        for entry in &page {
            visit(entry)?;
        }
    }
    Ok(())
}

fn source_catalog_roots(source: &RedCoreGraphStore) -> GraphStoreResult<Vec<String>> {
    let chunks = source.unified_chunk_store();
    let Some(address) = chunks.read_root()? else {
        return Ok(Vec::new());
    };
    let catalog = validate_catalog_root(chunks.as_chunk_store(), address)?;
    Ok(catalog
        .roots()
        .keys()
        .filter(|name| !REPLAYED_CATALOG_ROOTS.contains(&name.as_str()))
        .cloned()
        .collect())
}

fn census(record: &GraphPayloadRecord, counts: &mut BTreeMap<String, usize>) {
    let properties = match record {
        GraphPayloadRecord::Node(node) => &node.properties,
        GraphPayloadRecord::Edge(edge) => &edge.properties,
    };
    for property in TENANT_PROPERTIES {
        if let Some(value) = properties.get(property).and_then(Value::as_str) {
            *counts.entry(format!("{property}={value}")).or_default() += 1;
        }
    }
}

/// Compare everything the replay would touch with the target, without writing.
pub fn plan_graph_import(
    source: &RedCoreGraphStore,
    target: &RedCoreGraphStore,
    options: &GraphImportOptions,
) -> GraphStoreResult<GraphImportPlan> {
    validate_source_id(&options.source_id)?;
    let history = source_history(source)?;
    let target_now = target_head(target)?;
    let cursor = read_cursor(&target_now, &options.source_id)?;
    let point = start_point(&history, cursor.as_ref())?;
    let head_transaction = history.head_roots().transaction;
    let limit = options.max_commits.unwrap_or(usize::MAX);

    let mut touched = BTreeSet::<String>::new();
    let mut baseline_records = 0usize;
    if let Some(Revision::Pinned(baseline)) = &point.baseline {
        for_each_record(baseline, |record| {
            if !is_derived(&record) {
                baseline_records += 1;
                touched.insert(record_key(&record));
            }
            Ok(())
        })?;
    }
    let mut commits = 0usize;
    let mut through = point.after_transaction;
    let mut last_commit: Option<Hash> = None;
    for_each_entry(&history, point.after_transaction, limit, |entry| {
        commits += 1;
        through = entry.roots.transaction;
        last_commit = Some(entry.commit.commit_hash);
        touched.extend(entry.changed_object_keys.iter().cloned());
        Ok(())
    })?;
    let end = match last_commit {
        Some(commit) => Revision::Pinned(Box::new(history.revision(commit)?)),
        None => match point.baseline {
            Some(Revision::Pinned(ref baseline)) => Revision::Pinned(baseline.clone()),
            _ => match &point.base {
                Revision::Pinned(base) => Revision::Pinned(base.clone()),
                Revision::Empty => Revision::Empty,
            },
        },
    };

    let mut derived_skipped = 0usize;
    let mut converged = 0usize;
    let mut accepted = Vec::new();
    let mut conflicts = Vec::new();
    let mut conflict_count = 0usize;
    let mut tenant_census = BTreeMap::new();
    for key in &touched {
        let base_record = point.base.read(key)?;
        let final_record = end.read(key)?;
        if either_derived(&base_record, &final_record) {
            derived_skipped += 1;
            continue;
        }
        let target_record = target_now.read(key)?;
        if let Some(record) = &final_record {
            census(record, &mut tenant_census);
        }
        let target_address = optional_address(&target_record)?;
        let base_address = optional_address(&base_record)?;
        let final_address = optional_address(&final_record)?;
        if target_address == base_address {
            continue;
        }
        if target_address == final_address {
            converged += 1;
            continue;
        }
        if target_address.is_some()
            && options.accepted_overwrites.get(key) == target_address.as_ref()
        {
            accepted.push(key.clone());
            continue;
        }
        conflict_count += 1;
        if conflicts.len() < REPORT_EXAMPLES {
            conflicts.push(GraphImportConflict {
                key: key.clone(),
                target_content_address: target_address,
                base_content_address: base_address,
                source_content_address: final_address,
            });
        }
    }

    let foreign_tenants = match &options.expected_tenants {
        Some(expected) => tenant_census
            .keys()
            .filter(|entry| {
                let value = entry.split_once('=').map_or("", |(_, value)| value);
                !expected.contains(value)
            })
            .cloned()
            .collect(),
        None => Vec::new(),
    };

    let source_vectors = end.vector_designations().to_vec();
    let target_vectors = target.vector_designations();
    let designation_conflicts = source_vectors
        .iter()
        .filter_map(|source| {
            target_vectors
                .iter()
                .find(|target| {
                    target.label == source.label
                        && target.property == source.property
                        && target.dimension != source.dimension
                })
                .map(|target| {
                    format!(
                        "{}.{} is {} in the source and {} in the target",
                        source.label, source.property, source.dimension, target.dimension
                    )
                })
        })
        .collect();

    Ok(GraphImportPlan {
        source_id: options.source_id.clone(),
        coverage: history.coverage(),
        start: point.start,
        source_head_transaction: head_transaction,
        through_transaction: through,
        commits_to_replay: commits,
        baseline_records,
        touched_keys: touched.len() - derived_skipped,
        derived_keys_skipped: derived_skipped,
        converged_keys: converged,
        accepted_overwrites: accepted,
        conflicts,
        conflict_count,
        tenant_census,
        foreign_tenants,
        source_vector_designations: source_vectors,
        source_ordered_designations: end.ordered_designations().to_vec(),
        designation_conflicts,
        uncarried_catalog_roots: source_catalog_roots(source)?,
    })
}

fn validate_source_id(source_id: &str) -> GraphStoreResult<()> {
    if source_id.is_empty()
        || !source_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(GraphStoreError::new(
            "graph_import_source_id_invalid",
            "source id must be non-empty ASCII letters, digits, '-', '_' or '.'",
        ));
    }
    Ok(())
}

/// Reads source payload bytes, including bodies still in a legacy cold root.
struct PayloadCopier {
    source_objects: DiskObjectStore,
    target_chunks: Arc<dyn UnifiedChunkStore>,
}

impl PayloadCopier {
    fn new(source: &RedCoreGraphStore, target: &RedCoreGraphStore) -> GraphStoreResult<Self> {
        let source_chunks = source.unified_chunk_store();
        let source_objects = match source.data_dir() {
            Some(dir) => DiskObjectStore::from_chunk_store_with_legacy_root(
                Arc::clone(&source_chunks),
                dir.join("cold_objects"),
            )?,
            None => DiskObjectStore::from_chunk_store(Arc::clone(&source_chunks))?,
        };
        Ok(Self {
            source_objects,
            target_chunks: target.unified_chunk_store(),
        })
    }

    fn source_bytes(&self, address: Hash) -> GraphStoreResult<Vec<u8>> {
        let bytes = self
            .source_objects
            .get_document_bytes_read_only(&address.to_prefixed())?
            .ok_or_else(|| {
                GraphStoreError::new(
                    "graph_import_payload_missing",
                    format!("the source references payload {address} but does not hold it"),
                )
            })?;
        let actual = Hash::of(&bytes);
        if actual != address {
            return Err(GraphStoreError::new(
                "graph_import_payload_corrupt",
                format!("source payload {address} hashes to {actual}"),
            ));
        }
        Ok(bytes)
    }

    /// Every chunk address a node references, file bodies expanded.
    fn references(&self, node: &NodeRecord) -> GraphStoreResult<Vec<Hash>> {
        let (manifests, terminals) = packed_graph_node_payload_references(node)?;
        let mut addresses = BTreeSet::new();
        for manifest in manifests {
            let bytes = self.source_bytes(manifest)?;
            let decoded = decode_file_manifest(manifest, &bytes)?;
            addresses.insert(manifest);
            addresses.extend(decoded.chunks.iter().map(|chunk| chunk.address));
        }
        addresses.extend(terminals);
        Ok(addresses.into_iter().collect())
    }

    /// Copy what the target lacks and pin everything the batch references
    /// until the batch is durable.
    fn stage(
        &self,
        nodes: &[&NodeRecord],
        window: &ChunkPublicationWindow<dyn UnifiedChunkStore>,
        report: &mut GraphImportReport,
    ) -> GraphStoreResult<()> {
        let mut addresses = BTreeSet::new();
        for node in nodes {
            if !node.tombstone {
                addresses.extend(self.references(node)?);
            }
        }
        if addresses.is_empty() {
            return Ok(());
        }
        let addresses = addresses.into_iter().collect::<Vec<_>>();
        let missing = self
            .target_chunks
            .missing(&addresses)?
            .into_iter()
            .collect::<BTreeSet<_>>();
        for address in &addresses {
            if missing.contains(address) {
                let bytes = self.source_bytes(*address)?;
                window.write(&[(*address, bytes.as_slice())])?;
                report.chunks_copied += 1;
                report.bytes_copied += bytes.len() as u64;
            } else {
                window.protect(*address)?;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct Changes {
    node_upserts: Vec<NodeRecord>,
    edge_upserts: Vec<EdgeRecord>,
    edge_deletes: Vec<String>,
    node_deletes: Vec<String>,
}

impl Changes {
    fn push_upsert(&mut self, record: GraphPayloadRecord) {
        match record {
            GraphPayloadRecord::Node(node) => self.node_upserts.push(node),
            GraphPayloadRecord::Edge(edge) => self.edge_upserts.push(edge),
        }
    }

    fn push_delete(&mut self, key: &str) -> GraphStoreResult<()> {
        match split_key(key)? {
            (KeyKind::Node, id) => self.node_deletes.push(id.to_owned()),
            (KeyKind::Edge, id) => self.edge_deletes.push(id.to_owned()),
        }
        Ok(())
    }

    /// Endpoints exist before an edge names them; edges go before their nodes.
    fn into_batch(self, cursor: NodeRecord, report: &mut GraphImportReport) -> GraphMutationBatch {
        report.records_upserted += self.node_upserts.len() + self.edge_upserts.len();
        report.records_deleted += self.edge_deletes.len() + self.node_deletes.len();
        GraphMutationBatch::new(
            self.node_upserts
                .into_iter()
                .map(GraphMutation::NodeUpsert)
                .chain(self.edge_upserts.into_iter().map(GraphMutation::EdgeUpsert))
                .chain(self.edge_deletes.into_iter().map(GraphMutation::EdgeDelete))
                .chain(self.node_deletes.into_iter().map(GraphMutation::NodeDelete))
                .chain([GraphMutation::NodeUpsert(cursor)]),
        )
    }
}

/// Bring the target's designations up to a source revision's, before the
/// records that rely on them land. Clears are reported, never applied.
fn apply_designations(
    target: &mut RedCoreGraphStore,
    previous: &Revision,
    next: &Revision,
    report: &mut GraphImportReport,
) -> GraphStoreResult<()> {
    for designation in next.vector_designations() {
        let present = target.vector_designations().iter().any(|existing| {
            existing.label == designation.label
                && existing.property == designation.property
                && existing.dimension == designation.dimension
        });
        if !present {
            target.designate_vector_property(
                &designation.label,
                &designation.property,
                designation.dimension,
            )?;
            report.designations_applied.push(format!(
                "vector {}.{} ({})",
                designation.label, designation.property, designation.dimension
            ));
        }
    }
    for designation in next.ordered_designations() {
        if !target.ordered_designations().contains(designation) {
            target.designate_ordered_property(&designation.label, &designation.property)?;
            report.designations_applied.push(format!(
                "ordered {}.{}",
                designation.label, designation.property
            ));
        }
    }
    for designation in previous.vector_designations() {
        if !next
            .vector_designations()
            .iter()
            .any(|kept| kept.label == designation.label && kept.property == designation.property)
        {
            report.designation_clears_not_applied.push(format!(
                "vector {}.{}",
                designation.label, designation.property
            ));
        }
    }
    for designation in previous.ordered_designations() {
        if !next.ordered_designations().contains(designation) {
            report.designation_clears_not_applied.push(format!(
                "ordered {}.{}",
                designation.label, designation.property
            ));
        }
    }
    Ok(())
}

fn commit_with_payloads(
    target: &mut RedCoreGraphStore,
    copier: &PayloadCopier,
    changes: Changes,
    cursor: NodeRecord,
    metadata: GraphCommitMetadata,
    report: &mut GraphImportReport,
) -> GraphStoreResult<()> {
    let window = ChunkPublicationWindow::new(
        target.unified_chunk_store(),
        &format!("graph-import:{}", cursor.id),
    );
    let nodes = changes.node_upserts.iter().collect::<Vec<_>>();
    copier.stage(&nodes, &window, report)?;
    let batch = changes.into_batch(cursor, report);
    target.commit_batch_with_metadata(batch, metadata)?;
    window.finish_after_publication();
    report.commits_written += 1;
    Ok(())
}

fn target_holds(target: &RedCoreGraphStore, key: &str) -> GraphStoreResult<bool> {
    Ok(match split_key(key)? {
        (KeyKind::Node, id) => target.get_node(id)?.is_some_and(|node| !node.tombstone),
        (KeyKind::Edge, id) => target.get_edge(id)?.is_some_and(|edge| !edge.tombstone),
    })
}

/// Plan, refuse unless admissible, then replay. The source is never written.
pub fn apply_graph_import(
    source: &RedCoreGraphStore,
    target: &mut RedCoreGraphStore,
    options: &GraphImportOptions,
) -> GraphStoreResult<(GraphImportPlan, GraphImportReport)> {
    let plan = plan_graph_import(source, target, options)?;
    if !plan.is_admissible() {
        return Err(GraphStoreError::new(
            "graph_import_refused",
            plan.refusals().join("; "),
        ));
    }
    let history = source_history(source)?;
    let target_now = target_head(target)?;
    let cursor = read_cursor(&target_now, &options.source_id)?;
    let point = start_point(&history, cursor.as_ref())?;
    let copier = PayloadCopier::new(source, target)?;
    let mut report = GraphImportReport {
        last_source_transaction: point.after_transaction,
        ..GraphImportReport::default()
    };
    let mut state = cursor.unwrap_or(ImportCursor {
        last_source_transaction: 0,
        last_source_commit: None,
        imported_commits: 0,
    });

    let mut previous = point.base;
    if let Some(Revision::Pinned(baseline)) = point.baseline {
        let baseline = Revision::Pinned(baseline);
        apply_designations(target, &Revision::Empty, &baseline, &mut report)?;
        let Revision::Pinned(revision) = &baseline else {
            unreachable!("baseline was matched as pinned");
        };
        let mut changes = Changes::default();
        for_each_record(revision, |record| {
            if !is_derived(&record) {
                changes.push_upsert(record);
            }
            Ok(())
        })?;
        state = ImportCursor {
            last_source_transaction: revision.roots.transaction,
            last_source_commit: revision
                .commit()
                .map(|commit| commit.commit_hash.to_prefixed()),
            imported_commits: 0,
        };
        let metadata = GraphCommitMetadata {
            author: GRAPH_IMPORT_AUTHOR.to_owned(),
            message: format!(
                "import {}: state at source transaction {} (earlier history not retained)",
                options.source_id, revision.roots.transaction
            ),
            timestamp_unix_ms: revision.timestamp_unix_ms().unwrap_or_else(unix_ms),
        };
        commit_with_payloads(
            target,
            &copier,
            changes,
            cursor_record(&options.source_id, &state),
            metadata,
            &mut report,
        )?;
        report.baseline_written = true;
        report.last_source_transaction = revision.roots.transaction;
        previous = baseline;
    }

    let limit = options.max_commits.unwrap_or(usize::MAX);
    for_each_entry(&history, point.after_transaction, limit, |entry| {
        let next = Revision::Pinned(Box::new(history.revision(entry.commit.commit_hash)?));
        apply_designations(target, &previous, &next, &mut report)?;
        let mut changes = Changes::default();
        for key in &entry.changed_object_keys {
            let before = previous.read(key)?;
            let after = next.read(key)?;
            if either_derived(&before, &after) {
                continue;
            }
            match after {
                Some(record) => {
                    if optional_address(&before)? != Some(content_address(&record)?) {
                        changes.push_upsert(record);
                    }
                }
                // A target that already lacks the record has converged.
                None if before.is_some() && target_holds(target, key)? => {
                    changes.push_delete(key)?
                }
                None => {}
            }
        }
        state = ImportCursor {
            last_source_transaction: entry.roots.transaction,
            last_source_commit: Some(entry.commit.commit_hash.to_prefixed()),
            imported_commits: state.imported_commits + 1,
        };
        let metadata = GraphCommitMetadata {
            author: entry.commit.author.clone(),
            message: entry.commit.message.clone(),
            timestamp_unix_ms: entry.commit.timestamp_unix_ms,
        };
        commit_with_payloads(
            target,
            &copier,
            changes,
            cursor_record(&options.source_id, &state),
            metadata,
            &mut report,
        )?;
        report.last_source_transaction = entry.roots.transaction;
        previous = next;
        Ok(())
    })?;
    Ok((plan, report))
}

/// Compare the source, at the transaction the target's cursor names, with the
/// target: every record and every payload it references.
pub fn verify_graph_import(
    source: &RedCoreGraphStore,
    target: &RedCoreGraphStore,
    source_id: &str,
) -> GraphStoreResult<GraphImportVerification> {
    validate_source_id(source_id)?;
    let history = source_history(source)?;
    let target_now = target_head(target)?;
    let cursor = read_cursor(&target_now, source_id)?.ok_or_else(|| {
        GraphStoreError::new(
            "graph_import_not_started",
            format!("the target has no import cursor for {source_id}"),
        )
    })?;
    let Revision::Pinned(source_revision) = source_revision_at(
        &history,
        cursor.last_source_transaction,
        cursor.last_source_commit.as_deref(),
    )?
    else {
        unreachable!("source_revision_at always pins a revision");
    };
    let copier = PayloadCopier::new(source, target)?;
    let mut verification = GraphImportVerification {
        source_transaction: cursor.last_source_transaction,
        ..GraphImportVerification::default()
    };
    let mut source_keys = BTreeSet::new();
    for_each_record(&source_revision, |record| {
        if is_derived(&record) {
            verification.source_derived_records += 1;
            return Ok(());
        }
        let key = record_key(&record);
        let target_record = target_now.read(&key)?;
        let matched = match &target_record {
            Some(target_record) => content_address(target_record)? == content_address(&record)?,
            None => false,
        };
        verification.source_records += 1;
        let tally = |count: &mut GraphImportLabelCount| {
            count.source += 1;
            if matched {
                count.matched += 1;
            }
        };
        match &record {
            GraphPayloadRecord::Node(node) => {
                for label in &node.labels {
                    tally(verification.by_label.entry(label.clone()).or_default());
                }
            }
            GraphPayloadRecord::Edge(edge) => {
                tally(
                    verification
                        .by_edge_type
                        .entry(edge.edge_type.clone())
                        .or_default(),
                );
            }
        }
        if matched {
            verification.matched += 1;
        } else if target_record.is_some() {
            verification.mismatched += 1;
            if verification.mismatched_keys.len() < REPORT_EXAMPLES {
                verification.mismatched_keys.push(key.clone());
            }
        } else {
            verification.missing_in_target += 1;
            if verification.missing_keys.len() < REPORT_EXAMPLES {
                verification.missing_keys.push(key.clone());
            }
        }
        if let GraphPayloadRecord::Node(node) = &record {
            if !node.tombstone {
                for address in copier.references(node)? {
                    verification.payload_refs_checked += 1;
                    if !copier.target_chunks.has(address)? {
                        verification.payload_refs_missing += 1;
                        if verification.missing_payloads.len() < REPORT_EXAMPLES {
                            verification
                                .missing_payloads
                                .push(format!("{key} -> {address}"));
                        }
                    }
                }
            }
        }
        source_keys.insert(key);
        Ok(())
    })?;
    if let Revision::Pinned(target_revision) = &target_now {
        let cursor_key = format!("node/{}", graph_import_cursor_id(source_id));
        for_each_record(target_revision, |record| {
            let key = record_key(&record);
            if key != cursor_key && !is_derived(&record) && !source_keys.contains(&key) {
                verification.target_only_records += 1;
            }
            Ok(())
        })?;
    }
    Ok(verification)
}

#[cfg(test)]
#[path = "graph_import_tests.rs"]
mod tests;
