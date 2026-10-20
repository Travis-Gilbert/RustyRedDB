//! Positioned, bounded materialization for indexed fold surfaces.
//!
//! The durable [`PositionedFoldIndex`] is authoritative for a key at a pinned
//! log prefix. [`PartialFoldRuntime`] retains only a budgeted working set and
//! serializes reconstruction, later updates, and descendant eviction.

use crate::cold_index::{ColdIndex, ColdIndexEntry, FoldHotness};
use crate::fold::{FoldId, LogInterval, RetractOutcome};
use crate::fold_storage::{FoldStorage, FoldStorageMutation};
use crate::ordered::EvictionFrontier;
use crate::query_receipt::{ConsumerOverlapTrace, PinnedReadSet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use theorem_oplog::{LogPosition, WriteSet};

pub const PARTIAL_FOLD_INDEX_VERSION: u32 = 1;

const ENTRY_BASE_CHARGE: usize = 256;
const BUFFERED_MUTATION_CHARGE: usize = 96;
const DEPENDENCY_EDGE_CHARGE: usize = 128;
const HOTNESS_CHARGE: usize = 80;
const RETRACTION_LEDGER_ENTRY_CHARGE: usize = 192;
const RETRACTION_REQUEST_CHARGE: usize = 96;
const SUPPORT_VALUE_CHARGE: usize = 96;
const DEPENDENCY_KEY_COPIES: usize = 2;
const PARTIAL_FOLD_SCOPE: &str = "partial-fold";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialOperator {
    FactsEntityAttribute,
    FactsCurrentEntity,
    FactsProvenanceSource,
    GraphNode,
    GraphEdge,
    GraphNeighbors,
    GraphIndexedNodes,
    AlgebraRegion,
    AlgebraPointDerivation,
    AlgebraVector,
    HistoryInterval,
    HistoryContentChunk,
    HistoryLogicalRecord,
    DocumentsDocument,
    DocumentsKind,
    DocumentsCollection,
    DocumentsActor,
}

impl PartialOperator {
    pub fn fold_id(self) -> FoldId {
        match self {
            Self::FactsEntityAttribute | Self::FactsCurrentEntity | Self::FactsProvenanceSource => {
                FoldId::Facts
            }
            Self::GraphNode | Self::GraphEdge | Self::GraphNeighbors | Self::GraphIndexedNodes => {
                FoldId::Graph
            }
            Self::AlgebraRegion | Self::AlgebraPointDerivation | Self::AlgebraVector => {
                FoldId::Algebra
            }
            Self::HistoryInterval | Self::HistoryContentChunk | Self::HistoryLogicalRecord => {
                FoldId::History
            }
            Self::DocumentsDocument
            | Self::DocumentsKind
            | Self::DocumentsCollection
            | Self::DocumentsActor => FoldId::Documents,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FactsEntityAttribute => "facts_entity_attribute",
            Self::FactsCurrentEntity => "facts_current_entity",
            Self::FactsProvenanceSource => "facts_provenance_source",
            Self::GraphNode => "graph_node",
            Self::GraphEdge => "graph_edge",
            Self::GraphNeighbors => "graph_neighbors",
            Self::GraphIndexedNodes => "graph_indexed_nodes",
            Self::AlgebraRegion => "algebra_region",
            Self::AlgebraPointDerivation => "algebra_point_derivation",
            Self::AlgebraVector => "algebra_vector",
            Self::HistoryInterval => "history_interval",
            Self::HistoryContentChunk => "history_content_chunk",
            Self::HistoryLogicalRecord => "history_logical_record",
            Self::DocumentsDocument => "documents_document",
            Self::DocumentsKind => "documents_kind",
            Self::DocumentsCollection => "documents_collection",
            Self::DocumentsActor => "documents_actor",
        }
    }
}

impl fmt::Display for PartialOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoldSurface {
    FactsEntityAttribute,
    FactsCurrentEntity,
    FactsProvenanceSource,
    FactsAttributeValueScan,
    GraphNode,
    GraphEdge,
    GraphNeighbors,
    GraphIndexedNodes,
    GraphAllNodes,
    GraphGlobalAlgorithm,
    AlgebraRegion,
    AlgebraPointDerivation,
    AlgebraGlobalAlgorithm,
    AlgebraVector,
    AlgebraProjection,
    AlgebraTokenIndex,
    HistoryInterval,
    HistoryContentChunk,
    HistoryLogicalRecord,
    HistoryEnumeration,
    HistoryCheckout,
    DocumentsDocument,
    DocumentsKind,
    DocumentsCollection,
    DocumentsActor,
    DocumentsRecency,
    DocumentsEnumeration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterializationEligibility {
    IndexedPartial(PartialOperator),
    FullState,
}

impl FoldSurface {
    pub fn fold_id(self) -> FoldId {
        match self {
            Self::FactsEntityAttribute
            | Self::FactsCurrentEntity
            | Self::FactsProvenanceSource
            | Self::FactsAttributeValueScan => FoldId::Facts,
            Self::GraphNode
            | Self::GraphEdge
            | Self::GraphNeighbors
            | Self::GraphIndexedNodes
            | Self::GraphAllNodes
            | Self::GraphGlobalAlgorithm => FoldId::Graph,
            Self::AlgebraRegion
            | Self::AlgebraPointDerivation
            | Self::AlgebraGlobalAlgorithm
            | Self::AlgebraVector
            | Self::AlgebraProjection
            | Self::AlgebraTokenIndex => FoldId::Algebra,
            Self::HistoryInterval
            | Self::HistoryContentChunk
            | Self::HistoryLogicalRecord
            | Self::HistoryEnumeration
            | Self::HistoryCheckout => FoldId::History,
            Self::DocumentsDocument
            | Self::DocumentsKind
            | Self::DocumentsCollection
            | Self::DocumentsActor
            | Self::DocumentsRecency
            | Self::DocumentsEnumeration => FoldId::Documents,
        }
    }

    pub fn eligibility(self) -> MaterializationEligibility {
        let operator = match self {
            Self::FactsEntityAttribute => PartialOperator::FactsEntityAttribute,
            Self::FactsCurrentEntity => PartialOperator::FactsCurrentEntity,
            Self::FactsProvenanceSource => PartialOperator::FactsProvenanceSource,
            Self::GraphNode => PartialOperator::GraphNode,
            Self::GraphEdge => PartialOperator::GraphEdge,
            Self::GraphNeighbors => PartialOperator::GraphNeighbors,
            Self::GraphIndexedNodes => PartialOperator::GraphIndexedNodes,
            Self::AlgebraRegion => PartialOperator::AlgebraRegion,
            Self::AlgebraPointDerivation => PartialOperator::AlgebraPointDerivation,
            Self::AlgebraVector => PartialOperator::AlgebraVector,
            Self::HistoryInterval => PartialOperator::HistoryInterval,
            Self::HistoryContentChunk => PartialOperator::HistoryContentChunk,
            Self::HistoryLogicalRecord => PartialOperator::HistoryLogicalRecord,
            Self::DocumentsDocument => PartialOperator::DocumentsDocument,
            Self::DocumentsKind => PartialOperator::DocumentsKind,
            Self::DocumentsCollection => PartialOperator::DocumentsCollection,
            Self::DocumentsActor => PartialOperator::DocumentsActor,
            Self::FactsAttributeValueScan
            | Self::GraphAllNodes
            | Self::GraphGlobalAlgorithm
            | Self::AlgebraGlobalAlgorithm
            | Self::AlgebraProjection
            | Self::AlgebraTokenIndex
            | Self::HistoryEnumeration
            | Self::HistoryCheckout
            | Self::DocumentsRecency
            | Self::DocumentsEnumeration => return MaterializationEligibility::FullState,
        };
        MaterializationEligibility::IndexedPartial(operator)
    }
}

pub const FACTS_SURFACES: &[FoldSurface] = &[
    FoldSurface::FactsEntityAttribute,
    FoldSurface::FactsCurrentEntity,
    FoldSurface::FactsProvenanceSource,
    FoldSurface::FactsAttributeValueScan,
];
pub const GRAPH_SURFACES: &[FoldSurface] = &[
    FoldSurface::GraphNode,
    FoldSurface::GraphEdge,
    FoldSurface::GraphNeighbors,
    FoldSurface::GraphIndexedNodes,
    FoldSurface::GraphAllNodes,
    FoldSurface::GraphGlobalAlgorithm,
];
pub const ALGEBRA_SURFACES: &[FoldSurface] = &[
    FoldSurface::AlgebraRegion,
    FoldSurface::AlgebraPointDerivation,
    FoldSurface::AlgebraGlobalAlgorithm,
    FoldSurface::AlgebraVector,
    FoldSurface::AlgebraProjection,
    FoldSurface::AlgebraTokenIndex,
];
pub const HISTORY_SURFACES: &[FoldSurface] = &[
    FoldSurface::HistoryInterval,
    FoldSurface::HistoryContentChunk,
    FoldSurface::HistoryLogicalRecord,
    FoldSurface::HistoryEnumeration,
    FoldSurface::HistoryCheckout,
];
pub const DOCUMENTS_SURFACES: &[FoldSurface] = &[
    FoldSurface::DocumentsDocument,
    FoldSurface::DocumentsKind,
    FoldSurface::DocumentsCollection,
    FoldSurface::DocumentsActor,
    FoldSurface::DocumentsRecency,
    FoldSurface::DocumentsEnumeration,
];

pub const RETRACTION_WINDOW_APPLIED_ENTRIES: u32 = 100;
pub const RETRACTION_PROMOTION_MINIMUM: usize = 2;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct MutationOrder {
    pub position: LogPosition,
    pub ordinal: u32,
}

impl MutationOrder {
    pub fn new(position: LogPosition, ordinal: u32) -> Result<Self, PartialFoldError> {
        if position == LogPosition::ZERO {
            return Err(PartialFoldError::new(
                "partial_fold_position",
                "a persisted mutation must have a non-zero log position",
            ));
        }
        Ok(Self { position, ordinal })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PartialKey {
    pub operator: PartialOperator,
    pub key: Vec<u8>,
}

impl PartialKey {
    pub fn new(
        operator: PartialOperator,
        key: impl Into<Vec<u8>>,
    ) -> Result<Self, PartialFoldError> {
        let key = key.into();
        if key.is_empty() {
            return Err(PartialFoldError::new(
                "partial_fold_key",
                "partial materialization keys must be non-empty",
            ));
        }
        Ok(Self { operator, key })
    }

    fn frontier_id(&self) -> String {
        format!(
            "{}:{}:{}",
            self.operator.fold_id(),
            self.operator,
            blake3::hash(&self.key).to_hex()
        )
    }
}

#[derive(Clone)]
pub struct PositionedFoldIndex {
    storage: Arc<dyn FoldStorage>,
    cold_index: Arc<dyn ColdIndex>,
    publish_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for PositionedFoldIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PositionedFoldIndex")
            .field("floor", &self.storage.floor())
            .finish_non_exhaustive()
    }
}

impl PositionedFoldIndex {
    pub fn new(storage: Arc<dyn FoldStorage>, cold_index: Arc<dyn ColdIndex>) -> Self {
        Self {
            storage,
            cold_index,
            publish_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn publish(
        &self,
        operator: PartialOperator,
        key: &[u8],
        order: MutationOrder,
        payload: Option<&[u8]>,
    ) -> Result<(), PartialFoldError> {
        let _guard = self.publish_lock.lock().map_err(|_| {
            PartialFoldError::new(
                "partial_fold_publish_lock",
                "positioned index lock poisoned",
            )
        })?;
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let namespace = namespace(operator);
        let catalog_key = catalog_key(&partial_key.key);
        let snapshot_key = snapshot_key(&partial_key.key, order)?;
        let encoded_snapshot = encode_snapshot(payload);

        if let Some(existing) = self.storage.get(&namespace, &snapshot_key)? {
            if existing != encoded_snapshot {
                return Err(PartialFoldError::new(
                    "partial_fold_snapshot_conflict",
                    format!("{operator} key was republished with different bytes at {order:?}"),
                ));
            }
            self.record_locator(&partial_key, order, &encoded_snapshot)?;
            return Ok(());
        }

        let mut catalog = self
            .storage
            .get(&namespace, &catalog_key)?
            .map(|bytes| decode_catalog(&bytes))
            .transpose()?
            .unwrap_or_default();
        match catalog.binary_search(&order) {
            Ok(_) => {
                return Err(PartialFoldError::new(
                    "partial_fold_catalog_gap",
                    format!("catalog contains {order:?} without its snapshot"),
                ));
            }
            Err(index) => catalog.insert(index, order),
        }
        self.storage.write_batch(&[
            FoldStorageMutation::put(namespace.clone(), snapshot_key, encoded_snapshot.clone()),
            FoldStorageMutation::put(namespace, catalog_key, encode_catalog(&catalog)?),
        ])?;
        self.record_locator(&partial_key, order, &encoded_snapshot)
    }

    pub fn load_at(
        &self,
        operator: PartialOperator,
        key: &[u8],
        through: LogPosition,
    ) -> Result<PositionedSnapshot, PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let namespace = namespace(operator);
        let Some(catalog) = self.storage.get(&namespace, &catalog_key(key))? else {
            return Ok(PositionedSnapshot {
                operator,
                key: key.to_vec(),
                through,
                source_order: None,
                payload: None,
            });
        };
        let catalog = decode_catalog(&catalog)?;
        let source_order = catalog
            .iter()
            .rev()
            .find(|order| order.position <= through)
            .copied();
        let Some(source_order) = source_order else {
            return Ok(PositionedSnapshot {
                operator,
                key: key.to_vec(),
                through,
                source_order: None,
                payload: None,
            });
        };
        let encoded = self
            .storage
            .get(&namespace, &snapshot_key(key, source_order)?)?
            .ok_or_else(|| {
                PartialFoldError::new(
                    "partial_fold_snapshot_missing",
                    format!("catalog points to absent {operator} snapshot {source_order:?}"),
                )
            })?;
        let locator_id = locator_id(&partial_key, source_order);
        let locator = self.cold_index.lookup(&locator_id)?.ok_or_else(|| {
            PartialFoldError::new(
                "partial_fold_locator_missing",
                format!("ColdIndex has no locator for {locator_id}"),
            )
        })?;
        let expected_hash = blake3::hash(&encoded).to_hex().to_string();
        if locator.object_hash != expected_hash {
            return Err(PartialFoldError::new(
                "partial_fold_locator_hash",
                format!("ColdIndex locator {locator_id} does not match stored bytes"),
            ));
        }
        Ok(PositionedSnapshot {
            operator,
            key: key.to_vec(),
            through,
            source_order: Some(source_order),
            payload: decode_snapshot(&encoded)?,
        })
    }

    pub fn locator_id(
        &self,
        operator: PartialOperator,
        key: &[u8],
        order: MutationOrder,
    ) -> Result<String, PartialFoldError> {
        Ok(locator_id(&PartialKey::new(operator, key.to_vec())?, order))
    }

    fn record_locator(
        &self,
        key: &PartialKey,
        order: MutationOrder,
        encoded_snapshot: &[u8],
    ) -> Result<(), PartialFoldError> {
        self.cold_index
            .record(ColdIndexEntry::cold(
                locator_id(key, order),
                format!("partial:{}:{}", key.operator.fold_id(), key.operator),
                blake3::hash(encoded_snapshot).to_hex().to_string(),
            ))
            .map_err(PartialFoldError::from)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionedSnapshot {
    pub operator: PartialOperator,
    pub key: Vec<u8>,
    pub through: LogPosition,
    pub source_order: Option<MutationOrder>,
    pub payload: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReconstructionTicket {
    pub operator: PartialOperator,
    pub key: Vec<u8>,
    pub generation: u64,
    pub pin: LogPosition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedRead {
    pub payload: Option<Vec<u8>>,
    pub through: LogPosition,
    pub generation: u64,
    pub retained: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReconstructionStart {
    Resident(MaterializedRead),
    Transient(MaterializedRead),
    Started {
        ticket: ReconstructionTicket,
        snapshot: PositionedSnapshot,
    },
    Coalesced(ReconstructionTicket),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartialResidency {
    Evicted,
    Reconstructing {
        generation: u64,
        pin: LogPosition,
        buffered: usize,
    },
    Resident {
        generation: u64,
        through: LogPosition,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvictionNotice {
    pub operator: PartialOperator,
    pub key: Vec<u8>,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UpdateDisposition {
    AppliedResident,
    Buffered,
    AlreadyRepresented,
    Idempotent,
    DroppedEvicted { notices: Vec<EvictionNotice> },
    DroppedBudget { notices: Vec<EvictionNotice> },
}

#[derive(Clone, Debug)]
struct BufferedMutation {
    payload: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
enum EntryState {
    Reconstructing {
        generation: u64,
        pin: LogPosition,
        buffered: BTreeMap<MutationOrder, BufferedMutation>,
    },
    Resident {
        generation: u64,
        through: LogPosition,
        last_order: Option<MutationOrder>,
        payload: Option<Vec<u8>>,
    },
}

impl EntryState {
    fn generation(&self) -> u64 {
        match self {
            Self::Reconstructing { generation, .. } | Self::Resident { generation, .. } => {
                *generation
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PartialEntry {
    key: PartialKey,
    state: EntryState,
}

impl PartialEntry {
    fn accounting(&self) -> PartialFoldAccounting {
        let buffered_mutations = match &self.state {
            EntryState::Reconstructing { buffered, .. } => buffered
                .values()
                .map(|mutation| {
                    BUFFERED_MUTATION_CHARGE
                        + mutation.payload.as_ref().map(Vec::capacity).unwrap_or(0)
                })
                .sum(),
            EntryState::Resident { .. } => 0,
        };
        let payload_capacity = match &self.state {
            EntryState::Resident { payload, .. } => {
                payload.as_ref().map(Vec::capacity).unwrap_or(0)
            }
            EntryState::Reconstructing { .. } => 0,
        };
        PartialFoldAccounting {
            entry_slots: ENTRY_BASE_CHARGE,
            key_capacity: self.key.key.capacity(),
            payload_capacity,
            buffered_mutations,
            dependency_edges: 0,
            frontier_identifiers: 0,
            hotness: 0,
            retraction_ledger: 0,
            support_state: 0,
        }
    }
}

/// Conservative retained-capacity accounting for the bounded partial runtime.
///
/// Variable-size allocations are charged once for every owned copy retained by
/// the runtime. Fixed charges cover bounded map/set nodes and state-machine
/// slots. Durable storage and caller-owned returned values are outside this
/// counter by ADR05.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PartialFoldAccounting {
    pub entry_slots: usize,
    pub key_capacity: usize,
    pub payload_capacity: usize,
    pub buffered_mutations: usize,
    pub dependency_edges: usize,
    pub frontier_identifiers: usize,
    pub hotness: usize,
    pub retraction_ledger: usize,
    pub support_state: usize,
}

impl PartialFoldAccounting {
    pub fn total(self) -> usize {
        self.entry_slots
            .saturating_add(self.key_capacity)
            .saturating_add(self.payload_capacity)
            .saturating_add(self.buffered_mutations)
            .saturating_add(self.dependency_edges)
            .saturating_add(self.frontier_identifiers)
            .saturating_add(self.hotness)
            .saturating_add(self.retraction_ledger)
            .saturating_add(self.support_state)
    }

    fn add_assign(&mut self, other: Self) {
        self.entry_slots = self.entry_slots.saturating_add(other.entry_slots);
        self.key_capacity = self.key_capacity.saturating_add(other.key_capacity);
        self.payload_capacity = self.payload_capacity.saturating_add(other.payload_capacity);
        self.buffered_mutations = self
            .buffered_mutations
            .saturating_add(other.buffered_mutations);
        self.dependency_edges = self.dependency_edges.saturating_add(other.dependency_edges);
        self.frontier_identifiers = self
            .frontier_identifiers
            .saturating_add(other.frontier_identifiers);
        self.hotness = self.hotness.saturating_add(other.hotness);
        self.retraction_ledger = self
            .retraction_ledger
            .saturating_add(other.retraction_ledger);
        self.support_state = self.support_state.saturating_add(other.support_state);
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtremumOrder {
    Minimum,
    Maximum,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportPromotionState {
    ReplayOnly,
    BackfillPending,
    SupportCounted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetractionWindowReceipt {
    pub operator: PartialOperator,
    pub key_hash: String,
    pub first_order: MutationOrder,
    pub through_order: MutationOrder,
    pub applied_entries: u32,
    pub distinct_retractions: usize,
    pub promotion_required: bool,
}

#[derive(Clone, Debug, Default)]
struct RetractionWindow {
    first_order: Option<MutationOrder>,
    through_order: Option<MutationOrder>,
    applied_entries: u32,
    request_ids: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct RetractionLedgerEntry {
    order: ExtremumOrder,
    state: SupportPromotionState,
    window: RetractionWindow,
}

#[derive(Clone, Debug, Default)]
pub struct RetractionLedger {
    entries: BTreeMap<PartialKey, RetractionLedgerEntry>,
}

impl RetractionLedger {
    pub fn register_extremum(
        &mut self,
        key: PartialKey,
        order: ExtremumOrder,
    ) -> Result<(), PartialFoldError> {
        match self.entries.get(&key) {
            Some(existing) if existing.order == order => Ok(()),
            Some(_) => Err(PartialFoldError::new(
                "retraction_extremum_conflict",
                "a partial key cannot change extremum order without rollback",
            )),
            None => {
                self.entries.insert(
                    key,
                    RetractionLedgerEntry {
                        order,
                        state: SupportPromotionState::ReplayOnly,
                        window: RetractionWindow::default(),
                    },
                );
                Ok(())
            }
        }
    }

    pub fn unregister(&mut self, key: &PartialKey) {
        self.entries.remove(key);
    }

    pub fn state(&self, key: &PartialKey) -> Option<&SupportPromotionState> {
        self.entries.get(key).map(|entry| &entry.state)
    }

    pub fn order(&self, key: &PartialKey) -> Option<ExtremumOrder> {
        self.entries.get(key).map(|entry| entry.order)
    }

    pub fn observe_retraction(
        &mut self,
        key: &PartialKey,
        request_id: &str,
    ) -> Result<bool, PartialFoldError> {
        if request_id.trim().is_empty() {
            return Err(PartialFoldError::new(
                "retraction_request_id",
                "retraction request ids must be non-blank",
            ));
        }
        let entry = self.entries.get_mut(key).ok_or_else(|| {
            PartialFoldError::new(
                "retraction_extremum_unregistered",
                "retraction telemetry requires an explicitly registered extremum key",
            )
        })?;
        if entry.state != SupportPromotionState::ReplayOnly {
            return Ok(false);
        }
        Ok(entry.window.request_ids.insert(request_id.to_string()))
    }

    fn remove_request(&mut self, key: &PartialKey, request_id: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.window.request_ids.remove(request_id);
        }
    }

    pub fn observe_absorb(
        &mut self,
        key: &PartialKey,
        order: MutationOrder,
    ) -> Result<Option<RetractionWindowReceipt>, PartialFoldError> {
        let entry = self.entries.get_mut(key).ok_or_else(|| {
            PartialFoldError::new(
                "retraction_extremum_unregistered",
                "absorb telemetry requires an explicitly registered extremum key",
            )
        })?;
        if entry.state != SupportPromotionState::ReplayOnly {
            return Ok(None);
        }
        if entry
            .window
            .through_order
            .is_some_and(|previous| order <= previous)
        {
            return Err(PartialFoldError::new(
                "retraction_ledger_order",
                format!("absorb {order:?} is not after the current window tail"),
            ));
        }
        entry.window.first_order.get_or_insert(order);
        entry.window.through_order = Some(order);
        entry.window.applied_entries = entry.window.applied_entries.saturating_add(1);
        if entry.window.applied_entries < RETRACTION_WINDOW_APPLIED_ENTRIES {
            return Ok(None);
        }

        let distinct_retractions = entry.window.request_ids.len();
        let promotion_required = distinct_retractions >= RETRACTION_PROMOTION_MINIMUM;
        let receipt = RetractionWindowReceipt {
            operator: key.operator,
            key_hash: blake3::hash(&key.key).to_hex().to_string(),
            first_order: entry
                .window
                .first_order
                .expect("closed ledger window has a first order"),
            through_order: entry
                .window
                .through_order
                .expect("closed ledger window has a through order"),
            applied_entries: entry.window.applied_entries,
            distinct_retractions,
            promotion_required,
        };
        entry.window = RetractionWindow::default();
        if promotion_required {
            entry.state = SupportPromotionState::BackfillPending;
        }
        Ok(Some(receipt))
    }

    fn complete_backfill(&mut self, key: &PartialKey) -> Result<(), PartialFoldError> {
        let entry = self.entries.get_mut(key).ok_or_else(|| {
            PartialFoldError::new(
                "retraction_extremum_unregistered",
                "support backfill requires an explicitly registered extremum key",
            )
        })?;
        if entry.state != SupportPromotionState::BackfillPending {
            return Err(PartialFoldError::new(
                "retraction_backfill_state",
                "support backfill can complete only from BackfillPending",
            ));
        }
        entry.state = SupportPromotionState::SupportCounted;
        Ok(())
    }

    fn restore_pending(&mut self, key: &PartialKey) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.state = SupportPromotionState::BackfillPending;
        }
    }

    fn rollback(&mut self, key: &PartialKey) -> bool {
        let Some(entry) = self.entries.get_mut(key) else {
            return false;
        };
        entry.state = SupportPromotionState::ReplayOnly;
        entry.window = RetractionWindow::default();
        true
    }

    fn retained_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(key, entry)| {
                RETRACTION_LEDGER_ENTRY_CHARGE
                    .saturating_add(key.key.capacity())
                    .saturating_add(
                        entry
                            .window
                            .request_ids
                            .iter()
                            .map(|request| {
                                RETRACTION_REQUEST_CHARGE.saturating_add(request.capacity())
                            })
                            .sum::<usize>(),
                    )
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct SupportCountedExtrema {
    order: ExtremumOrder,
    counts: BTreeMap<Vec<u8>, u64>,
}

impl SupportCountedExtrema {
    pub fn from_values(
        order: ExtremumOrder,
        values: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<Self, PartialFoldError> {
        let mut support = Self {
            order,
            counts: BTreeMap::new(),
        };
        for value in values {
            support.insert(value)?;
        }
        if support.counts.is_empty() {
            return Err(PartialFoldError::new(
                "retraction_support_empty",
                "support backfill requires at least one extremal value",
            ));
        }
        Ok(support)
    }

    pub fn insert(&mut self, value: Vec<u8>) -> Result<(), PartialFoldError> {
        if value.is_empty() {
            return Err(PartialFoldError::new(
                "retraction_support_value",
                "support values must be non-empty",
            ));
        }
        let count = self.counts.entry(value).or_default();
        *count = count.checked_add(1).ok_or_else(|| {
            PartialFoldError::new(
                "retraction_support_overflow",
                "exact support count exceeds the u64 representation",
            )
        })?;
        Ok(())
    }

    pub fn retract(&mut self, value: &[u8]) -> Result<RetractOutcome, PartialFoldError> {
        let Some(count) = self.counts.get_mut(value) else {
            return Err(PartialFoldError::new(
                "retraction_support_missing",
                "the retracted value is absent from exact support state",
            ));
        };
        *count -= 1;
        if *count == 0 {
            self.counts.remove(value);
        }
        Ok(RetractOutcome::Retracted)
    }

    pub fn extremum(&self) -> Option<&[u8]> {
        match self.order {
            ExtremumOrder::Minimum => self.counts.first_key_value(),
            ExtremumOrder::Maximum => self.counts.last_key_value(),
        }
        .map(|(value, _)| value.as_slice())
    }

    fn retained_bytes(&self) -> usize {
        self.counts
            .keys()
            .map(|value| SUPPORT_VALUE_CHARGE.saturating_add(value.capacity()))
            .sum()
    }
}

#[derive(Debug)]
pub struct PartialFoldRuntime {
    index: PositionedFoldIndex,
    budget_bytes: usize,
    resident_bytes: usize,
    entries: BTreeMap<PartialKey, PartialEntry>,
    descendants: BTreeMap<PartialKey, BTreeSet<PartialKey>>,
    parents: BTreeMap<PartialKey, BTreeSet<PartialKey>>,
    frontier: EvictionFrontier,
    frontier_keys: BTreeMap<String, PartialKey>,
    hotness: FoldHotness,
    retraction_ledger: RetractionLedger,
    support_extrema: BTreeMap<PartialKey, SupportCountedExtrema>,
    access_clock: i64,
    next_generation: u64,
}

impl PartialFoldRuntime {
    pub fn new(index: PositionedFoldIndex, budget_bytes: usize) -> Result<Self, PartialFoldError> {
        if budget_bytes == 0 {
            return Err(PartialFoldError::new(
                "partial_fold_budget",
                "partial-fold budget B must be non-zero",
            ));
        }
        Ok(Self {
            index,
            budget_bytes,
            resident_bytes: 0,
            entries: BTreeMap::new(),
            descendants: BTreeMap::new(),
            parents: BTreeMap::new(),
            frontier: EvictionFrontier::new(),
            frontier_keys: BTreeMap::new(),
            hotness: FoldHotness::default(),
            retraction_ledger: RetractionLedger::default(),
            support_extrema: BTreeMap::new(),
            access_clock: 0,
            next_generation: 0,
        })
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn accounting(&self) -> PartialFoldAccounting {
        self.calculated_accounting()
    }

    pub fn resident_len(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| matches!(entry.state, EntryState::Resident { .. }))
            .count()
    }

    pub fn frontier_ops(&self) -> usize {
        self.frontier.ops()
    }

    pub fn index(&self) -> &PositionedFoldIndex {
        &self.index
    }

    pub fn residency(
        &self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<PartialResidency, PartialFoldError> {
        let key = PartialKey::new(operator, key.to_vec())?;
        Ok(match self.entries.get(&key).map(|entry| &entry.state) {
            None => PartialResidency::Evicted,
            Some(EntryState::Reconstructing {
                generation,
                pin,
                buffered,
            }) => PartialResidency::Reconstructing {
                generation: *generation,
                pin: *pin,
                buffered: buffered.len(),
            },
            Some(EntryState::Resident {
                generation,
                through,
                ..
            }) => PartialResidency::Resident {
                generation: *generation,
                through: *through,
            },
        })
    }

    pub fn resident(
        &self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<Option<MaterializedRead>, PartialFoldError> {
        let key = PartialKey::new(operator, key.to_vec())?;
        Ok(self.entries.get(&key).and_then(|entry| match &entry.state {
            EntryState::Resident {
                generation,
                through,
                payload,
                ..
            } => Some(MaterializedRead {
                payload: payload.clone(),
                through: *through,
                generation: *generation,
                retained: true,
            }),
            EntryState::Reconstructing { .. } => None,
        }))
    }

    pub fn begin_reconstruction(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        requested: LogPosition,
    ) -> Result<ReconstructionStart, PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        if let Some(entry) = self.entries.get(&partial_key) {
            match &entry.state {
                EntryState::Resident {
                    generation,
                    through,
                    payload,
                    ..
                } if *through == requested => {
                    let read = MaterializedRead {
                        payload: payload.clone(),
                        through: *through,
                        generation: *generation,
                        retained: true,
                    };
                    self.touch(&partial_key)?;
                    return Ok(ReconstructionStart::Resident(read));
                }
                EntryState::Resident { through, .. } if *through > requested => {
                    let snapshot = self.index.load_at(operator, key, requested)?;
                    return Ok(ReconstructionStart::Transient(MaterializedRead {
                        payload: snapshot.payload,
                        through: requested,
                        generation: entry.state.generation(),
                        retained: false,
                    }));
                }
                EntryState::Reconstructing {
                    generation, pin, ..
                } => {
                    return Ok(ReconstructionStart::Coalesced(ReconstructionTicket {
                        operator,
                        key: key.to_vec(),
                        generation: *generation,
                        pin: *pin,
                    }));
                }
                EntryState::Resident { .. } => {}
            }
            self.remove_entry(&partial_key);
        }

        let snapshot = self.index.load_at(operator, key, requested)?;
        let generation = self.allocate_generation();
        let ticket = ReconstructionTicket {
            operator,
            key: key.to_vec(),
            generation,
            pin: requested,
        };
        self.insert_entry(PartialEntry {
            key: partial_key.clone(),
            state: EntryState::Reconstructing {
                generation,
                pin: requested,
                buffered: BTreeMap::new(),
            },
        })?;
        if !self.enforce_budget(Some(&partial_key))? {
            self.remove_entry(&partial_key);
            return Ok(ReconstructionStart::Transient(MaterializedRead {
                payload: snapshot.payload,
                through: requested,
                generation,
                retained: false,
            }));
        }
        Ok(ReconstructionStart::Started { ticket, snapshot })
    }

    pub fn complete_reconstruction(
        &mut self,
        ticket: &ReconstructionTicket,
        snapshot: PositionedSnapshot,
    ) -> Result<MaterializedRead, PartialFoldError> {
        let key = PartialKey::new(ticket.operator, ticket.key.clone())?;
        if snapshot.operator != ticket.operator
            || snapshot.key != ticket.key
            || snapshot.through != ticket.pin
        {
            return Err(PartialFoldError::new(
                "partial_fold_response",
                "reconstruction response does not match its ticket",
            ));
        }
        let current = self.entries.get(&key).map(|entry| match &entry.state {
            EntryState::Reconstructing {
                generation, pin, ..
            } => Some((*generation, *pin)),
            EntryState::Resident { .. } => None,
        });
        if current != Some(Some((ticket.generation, ticket.pin))) {
            return Err(PartialFoldError::new(
                "partial_fold_stale_response",
                "reconstruction response is not for the current key generation",
            ));
        }
        let entry = self.entries.remove(&key).expect("validated current entry");
        self.remove_frontier(&key);
        let EntryState::Reconstructing {
            generation,
            pin: _,
            buffered,
        } = entry.state
        else {
            unreachable!("validated reconstructing entry")
        };

        let mut payload = snapshot.payload;
        let mut through = ticket.pin;
        let mut last_order = snapshot.source_order;
        for (order, mutation) in buffered {
            if order.position <= ticket.pin {
                continue;
            }
            payload = mutation.payload;
            through = through.max(order.position);
            last_order = Some(order);
        }
        let read = MaterializedRead {
            payload: payload.clone(),
            through,
            generation,
            retained: false,
        };
        self.insert_entry(PartialEntry {
            key: key.clone(),
            state: EntryState::Resident {
                generation,
                through,
                last_order,
                payload,
            },
        })?;
        if !self.enforce_budget(Some(&key))? {
            self.remove_entry(&key);
            return Ok(read);
        }
        Ok(MaterializedRead {
            retained: true,
            ..read
        })
    }

    pub fn read_at(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        requested: LogPosition,
    ) -> Result<MaterializedRead, PartialFoldError> {
        match self.begin_reconstruction(operator, key, requested)? {
            ReconstructionStart::Resident(read) | ReconstructionStart::Transient(read) => Ok(read),
            ReconstructionStart::Started { ticket, snapshot } => {
                self.complete_reconstruction(&ticket, snapshot)
            }
            ReconstructionStart::Coalesced(_) => Err(PartialFoldError::new(
                "partial_fold_read_pending",
                "a synchronous read cannot complete while the key is already reconstructing",
            )),
        }
    }

    pub fn apply_update(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        order: MutationOrder,
        payload: Option<Vec<u8>>,
    ) -> Result<UpdateDisposition, PartialFoldError> {
        self.index
            .publish(operator, key, order, payload.as_deref())?;
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let Some(mut entry) = self.entries.remove(&partial_key) else {
            let notices = self.evict_key(&partial_key)?;
            return Ok(UpdateDisposition::DroppedEvicted { notices });
        };
        self.remove_frontier(&partial_key);

        let disposition = match &mut entry.state {
            EntryState::Reconstructing { pin, buffered, .. } => {
                if order.position <= *pin {
                    self.insert_entry(entry)?;
                    self.enforce_budget(Some(&partial_key))?;
                    return Ok(UpdateDisposition::AlreadyRepresented);
                }
                match buffered.get(&order) {
                    Some(existing) if existing.payload == payload => {
                        self.insert_entry(entry)?;
                        self.enforce_budget(Some(&partial_key))?;
                        return Ok(UpdateDisposition::Idempotent);
                    }
                    Some(_) => {
                        self.insert_entry(entry)?;
                        self.enforce_budget(Some(&partial_key))?;
                        return Err(PartialFoldError::new(
                            "partial_fold_update_conflict",
                            format!("{operator} received different bytes for {order:?}"),
                        ));
                    }
                    None => {
                        buffered.insert(order, BufferedMutation { payload });
                        UpdateDisposition::Buffered
                    }
                }
            }
            EntryState::Resident {
                through,
                last_order,
                payload: resident_payload,
                ..
            } => {
                let represented = order.position < *through
                    || (order.position == *through && last_order.is_none_or(|last| order <= last));
                if represented {
                    self.insert_entry(entry)?;
                    self.enforce_budget(Some(&partial_key))?;
                    return Ok(UpdateDisposition::AlreadyRepresented);
                }
                if let Some(last) = *last_order {
                    if order <= last {
                        self.insert_entry(entry)?;
                        self.enforce_budget(Some(&partial_key))?;
                        return Err(PartialFoldError::new(
                            "partial_fold_update_order",
                            format!("{operator} update {order:?} is not after {last:?}"),
                        ));
                    }
                }
                *through = order.position;
                *last_order = Some(order);
                *resident_payload = payload;
                UpdateDisposition::AppliedResident
            }
        };
        self.insert_entry(entry)?;
        if !self.enforce_budget(Some(&partial_key))? {
            let notices = self.evict_key(&partial_key)?;
            return Ok(UpdateDisposition::DroppedBudget { notices });
        }
        Ok(disposition)
    }

    pub fn register_descendant(
        &mut self,
        parent_operator: PartialOperator,
        parent_key: &[u8],
        child_operator: PartialOperator,
        child_key: &[u8],
    ) -> Result<(), PartialFoldError> {
        let parent = PartialKey::new(parent_operator, parent_key.to_vec())?;
        let child = PartialKey::new(child_operator, child_key.to_vec())?;
        let inserted = self
            .descendants
            .entry(parent.clone())
            .or_default()
            .insert(child.clone());
        if !inserted {
            return Ok(());
        }
        self.parents
            .entry(child.clone())
            .or_default()
            .insert(parent.clone());
        if !self.enforce_budget(None)? {
            if let Some(children) = self.descendants.get_mut(&parent) {
                children.remove(&child);
                if children.is_empty() {
                    self.descendants.remove(&parent);
                }
            }
            if let Some(parents) = self.parents.get_mut(&child) {
                parents.remove(&parent);
                if parents.is_empty() {
                    self.parents.remove(&child);
                }
            }
            self.refresh_accounting();
            return Err(PartialFoldError::new(
                "partial_fold_budget",
                "dependency metadata cannot fit within B",
            ));
        }
        Ok(())
    }

    pub fn evict(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<Vec<EvictionNotice>, PartialFoldError> {
        self.evict_key(&PartialKey::new(operator, key.to_vec())?)
    }

    pub fn observe_hotness(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        reads: &PinnedReadSet,
        footprint: &WriteSet,
    ) -> Result<ConsumerOverlapTrace, PartialFoldError> {
        let key = PartialKey::new(operator, key.to_vec())?;
        let id = key.frontier_id();
        let trace = if self.entries.contains_key(&key) {
            self.hotness.observe(id, reads, footprint)
        } else {
            reads.hotness_trace(footprint)
        };
        if trace.overlaps && self.entries.contains_key(&key) {
            self.touch(&key)?;
            if !self.enforce_budget(Some(&key))? {
                self.evict_key(&key)?;
            }
        }
        Ok(trace)
    }

    pub fn hotness_count(
        &self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<u64, PartialFoldError> {
        Ok(self
            .hotness
            .count(&PartialKey::new(operator, key.to_vec())?.frontier_id()))
    }

    pub fn register_extremum(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        order: ExtremumOrder,
    ) -> Result<(), PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        self.retraction_ledger
            .register_extremum(partial_key.clone(), order)?;
        if !self.enforce_auxiliary_budget(&partial_key)? {
            self.retraction_ledger.unregister(&partial_key);
            self.refresh_accounting();
            return Err(PartialFoldError::new(
                "partial_fold_budget",
                "retraction ledger registration cannot fit within B",
            ));
        }
        Ok(())
    }

    pub fn promotion_state(
        &self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<Option<&SupportPromotionState>, PartialFoldError> {
        Ok(self
            .retraction_ledger
            .state(&PartialKey::new(operator, key.to_vec())?))
    }

    pub fn observe_retraction(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        request_id: &str,
    ) -> Result<bool, PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let inserted = self
            .retraction_ledger
            .observe_retraction(&partial_key, request_id)?;
        if !self.enforce_auxiliary_budget(&partial_key)? {
            if inserted {
                self.retraction_ledger
                    .remove_request(&partial_key, request_id);
            }
            self.refresh_accounting();
            return Err(PartialFoldError::new(
                "partial_fold_budget",
                "retraction telemetry cannot fit within B",
            ));
        }
        Ok(inserted)
    }

    pub fn observe_absorb(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        order: MutationOrder,
    ) -> Result<Option<RetractionWindowReceipt>, PartialFoldError> {
        let receipt = self
            .retraction_ledger
            .observe_absorb(&PartialKey::new(operator, key.to_vec())?, order)?;
        self.refresh_accounting();
        Ok(receipt)
    }

    pub fn complete_support_backfill(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        values: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<(), PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let order = self.retraction_ledger.order(&partial_key).ok_or_else(|| {
            PartialFoldError::new(
                "retraction_extremum_unregistered",
                "support backfill requires an explicitly registered extremum key",
            )
        })?;
        let support = SupportCountedExtrema::from_values(order, values)?;
        self.retraction_ledger.complete_backfill(&partial_key)?;
        self.support_extrema.insert(partial_key.clone(), support);
        if !self.enforce_auxiliary_budget(&partial_key)? {
            self.support_extrema.remove(&partial_key);
            self.retraction_ledger.restore_pending(&partial_key);
            self.refresh_accounting();
            return Err(PartialFoldError::new(
                "partial_fold_budget",
                "support-counted backfill cannot fit within B",
            ));
        }
        Ok(())
    }

    pub fn supported_extremum(
        &self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<Option<&[u8]>, PartialFoldError> {
        Ok(self
            .support_extrema
            .get(&PartialKey::new(operator, key.to_vec())?)
            .and_then(SupportCountedExtrema::extremum))
    }

    pub fn retract_supported_value(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
        value: &[u8],
        replay_interval: LogInterval,
    ) -> Result<RetractOutcome, PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        if self.retraction_ledger.state(&partial_key)
            != Some(&SupportPromotionState::SupportCounted)
        {
            return Ok(RetractOutcome::RequiresRecompute {
                from_interval: replay_interval,
            });
        }
        let outcome = self
            .support_extrema
            .get_mut(&partial_key)
            .ok_or_else(|| {
                PartialFoldError::new(
                    "retraction_support_missing",
                    "SupportCounted ledger state has no exact support map",
                )
            })?
            .retract(value)?;
        self.refresh_accounting();
        Ok(outcome)
    }

    pub fn rollback_support(
        &mut self,
        operator: PartialOperator,
        key: &[u8],
    ) -> Result<bool, PartialFoldError> {
        let partial_key = PartialKey::new(operator, key.to_vec())?;
        let removed = self.support_extrema.remove(&partial_key).is_some();
        let registered = self.retraction_ledger.rollback(&partial_key);
        self.refresh_accounting();
        Ok(removed || registered)
    }

    fn insert_entry(&mut self, entry: PartialEntry) -> Result<(), PartialFoldError> {
        let key = entry.key.clone();
        let id = key.frontier_id();
        self.entries.insert(key.clone(), entry);
        self.frontier_keys.insert(id.clone(), key);
        self.access_clock = self.access_clock.saturating_add(1);
        self.frontier
            .touch(PARTIAL_FOLD_SCOPE, &id, self.access_clock)?;
        Ok(())
    }

    fn touch(&mut self, key: &PartialKey) -> Result<(), PartialFoldError> {
        self.access_clock = self.access_clock.saturating_add(1);
        self.frontier
            .touch(PARTIAL_FOLD_SCOPE, &key.frontier_id(), self.access_clock)?;
        Ok(())
    }

    fn remove_frontier(&mut self, key: &PartialKey) {
        let id = key.frontier_id();
        self.frontier.forget(PARTIAL_FOLD_SCOPE, &id);
        self.frontier_keys.remove(&id);
    }

    fn remove_entry(&mut self, key: &PartialKey) -> Option<PartialEntry> {
        self.remove_frontier(key);
        self.hotness.forget(&key.frontier_id());
        let removed = self.entries.remove(key);
        self.refresh_accounting();
        removed
    }

    fn evict_key(&mut self, root: &PartialKey) -> Result<Vec<EvictionNotice>, PartialFoldError> {
        let mut stack = vec![root.clone()];
        let mut visited = BTreeSet::new();
        while let Some(key) = stack.pop() {
            if !visited.insert(key.clone()) {
                continue;
            }
            if let Some(children) = self.descendants.get(&key) {
                stack.extend(children.iter().cloned());
            }
        }
        let mut notices = Vec::new();
        for key in &visited {
            if let Some(entry) = self.entries.get(key) {
                notices.push(EvictionNotice {
                    operator: key.operator,
                    key: key.key.clone(),
                    generation: entry.state.generation().saturating_add(1),
                });
            }
        }
        for key in &visited {
            if let Some(children) = self.descendants.remove(key) {
                for child in children {
                    if let Some(parents) = self.parents.get_mut(&child) {
                        parents.remove(key);
                        if parents.is_empty() {
                            self.parents.remove(&child);
                        }
                    }
                }
            }
            if let Some(parents) = self.parents.remove(key) {
                for parent in parents {
                    if let Some(children) = self.descendants.get_mut(&parent) {
                        children.remove(key);
                        if children.is_empty() {
                            self.descendants.remove(&parent);
                        }
                    }
                }
            }
        }
        for key in visited {
            self.support_extrema.remove(&key);
            self.retraction_ledger.rollback(&key);
            self.remove_entry(&key);
        }
        self.refresh_accounting();
        Ok(notices)
    }

    fn enforce_budget(&mut self, protected: Option<&PartialKey>) -> Result<bool, PartialFoldError> {
        self.enforce_budget_inner(protected, true)
    }

    fn enforce_auxiliary_budget(
        &mut self,
        protected: &PartialKey,
    ) -> Result<bool, PartialFoldError> {
        self.enforce_budget_inner(Some(protected), false)
    }

    fn enforce_budget_inner(
        &mut self,
        protected: Option<&PartialKey>,
        require_protected_entry: bool,
    ) -> Result<bool, PartialFoldError> {
        while self.calculated_bytes() > self.budget_bytes {
            let candidates = self.frontier.coldest_below(PARTIAL_FOLD_SCOPE, i64::MAX, 2);
            let victim = candidates.into_iter().find_map(|(id, _)| {
                let key = self.frontier_keys.get(&id)?;
                (protected != Some(key)).then_some(key.clone())
            });
            let Some(victim) = victim else {
                self.refresh_accounting();
                return Ok(false);
            };
            self.evict_key(&victim)?;
            if require_protected_entry
                && protected.is_some_and(|key| !self.entries.contains_key(key))
            {
                self.refresh_accounting();
                return Ok(false);
            }
        }
        self.refresh_accounting();
        Ok(true)
    }

    fn calculated_accounting(&self) -> PartialFoldAccounting {
        let mut accounting = PartialFoldAccounting::default();
        for (directory_key, entry) in &self.entries {
            accounting.add_assign(entry.accounting());
            accounting.key_capacity = accounting
                .key_capacity
                .saturating_add(directory_key.key.capacity());
        }
        for (frontier_id, frontier_key) in &self.frontier_keys {
            accounting.frontier_identifiers = accounting
                .frontier_identifiers
                .saturating_add(frontier_id.capacity());
            accounting.key_capacity = accounting
                .key_capacity
                .saturating_add(frontier_key.key.capacity());
        }
        accounting.frontier_identifiers = accounting.frontier_identifiers.saturating_add(
            self.frontier
                .retained_identifier_capacity_bytes(PARTIAL_FOLD_SCOPE),
        );
        let hotness_identifiers = self.hotness.retained_identifier_capacity_bytes();
        accounting.hotness = hotness_identifiers
            .saturating_add(self.hotness.counts().len().saturating_mul(HOTNESS_CHARGE));
        accounting.retraction_ledger = self.retraction_ledger.retained_bytes();
        accounting.support_state = self
            .support_extrema
            .iter()
            .map(|(key, support)| key.key.capacity().saturating_add(support.retained_bytes()))
            .sum();
        for (parent, children) in &self.descendants {
            for child in children {
                accounting.dependency_edges = accounting.dependency_edges.saturating_add(
                    DEPENDENCY_EDGE_CHARGE.saturating_mul(2).saturating_add(
                        parent
                            .key
                            .capacity()
                            .saturating_add(child.key.capacity())
                            .saturating_mul(DEPENDENCY_KEY_COPIES),
                    ),
                );
            }
        }
        accounting
    }

    fn calculated_bytes(&self) -> usize {
        self.calculated_accounting().total()
    }

    fn refresh_accounting(&mut self) {
        self.resident_bytes = self.calculated_bytes();
    }

    fn allocate_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.saturating_add(1);
        self.next_generation
    }
}

#[derive(Debug)]
pub struct PartialFoldError {
    pub code: &'static str,
    pub detail: String,
}

impl PartialFoldError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for PartialFoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl Error for PartialFoldError {}

impl From<crate::fold_storage::FoldStorageError> for PartialFoldError {
    fn from(error: crate::fold_storage::FoldStorageError) -> Self {
        Self::new(error.code, error.detail)
    }
}

impl From<crate::graph::error::GraphStoreError> for PartialFoldError {
    fn from(error: crate::graph::error::GraphStoreError) -> Self {
        Self::new("partial_fold_graph", error.to_string())
    }
}

fn namespace(operator: PartialOperator) -> String {
    format!(
        "partial-v{PARTIAL_FOLD_INDEX_VERSION}:{}:{operator}",
        operator.fold_id()
    )
}

fn catalog_key(key: &[u8]) -> Vec<u8> {
    let mut encoded = b"catalog\0".to_vec();
    encoded.extend_from_slice(key);
    encoded
}

fn snapshot_key(key: &[u8], order: MutationOrder) -> Result<Vec<u8>, PartialFoldError> {
    let key_len = u32::try_from(key.len()).map_err(|_| {
        PartialFoldError::new(
            "partial_fold_key",
            "partial materialization key exceeds u32 length",
        )
    })?;
    let mut encoded = Vec::with_capacity(13 + key.len() + 12);
    encoded.extend_from_slice(b"snapshot\0");
    encoded.extend_from_slice(&key_len.to_be_bytes());
    encoded.extend_from_slice(key);
    encoded.extend_from_slice(&order.position.0.to_be_bytes());
    encoded.extend_from_slice(&order.ordinal.to_be_bytes());
    Ok(encoded)
}

fn encode_catalog(catalog: &[MutationOrder]) -> Result<Vec<u8>, PartialFoldError> {
    let count = u32::try_from(catalog.len()).map_err(|_| {
        PartialFoldError::new(
            "partial_fold_catalog",
            "position catalog exceeds u32 entries",
        )
    })?;
    let mut encoded = Vec::with_capacity(4 + catalog.len() * 12);
    encoded.extend_from_slice(&count.to_be_bytes());
    for order in catalog {
        encoded.extend_from_slice(&order.position.0.to_be_bytes());
        encoded.extend_from_slice(&order.ordinal.to_be_bytes());
    }
    Ok(encoded)
}

fn decode_catalog(bytes: &[u8]) -> Result<Vec<MutationOrder>, PartialFoldError> {
    let count_bytes: [u8; 4] = bytes
        .get(..4)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| PartialFoldError::new("partial_fold_catalog", "catalog header missing"))?;
    let count = u32::from_be_bytes(count_bytes) as usize;
    if bytes.len() != 4 + count * 12 {
        return Err(PartialFoldError::new(
            "partial_fold_catalog",
            "catalog length does not match its entry count",
        ));
    }
    let mut catalog = Vec::with_capacity(count);
    for chunk in bytes[4..].chunks_exact(12) {
        let position = u64::from_be_bytes(chunk[..8].try_into().expect("exact catalog chunk"));
        let ordinal = u32::from_be_bytes(chunk[8..].try_into().expect("exact catalog chunk"));
        catalog.push(MutationOrder::new(LogPosition(position), ordinal)?);
    }
    if catalog.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(PartialFoldError::new(
            "partial_fold_catalog",
            "catalog orders must be strictly increasing",
        ));
    }
    Ok(catalog)
}

fn encode_snapshot(payload: Option<&[u8]>) -> Vec<u8> {
    match payload {
        Some(payload) => {
            let mut encoded = Vec::with_capacity(1 + payload.len());
            encoded.push(1);
            encoded.extend_from_slice(payload);
            encoded
        }
        None => vec![0],
    }
}

fn decode_snapshot(encoded: &[u8]) -> Result<Option<Vec<u8>>, PartialFoldError> {
    match encoded.split_first() {
        Some((0, [])) => Ok(None),
        Some((1, payload)) => Ok(Some(payload.to_vec())),
        _ => Err(PartialFoldError::new(
            "partial_fold_snapshot",
            "snapshot tombstone/value tag is invalid",
        )),
    }
}

fn locator_id(key: &PartialKey, order: MutationOrder) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        key.operator.fold_id(),
        key.operator,
        blake3::hash(&key.key).to_hex(),
        order.position.0,
        order.ordinal
    )
}
