//! Atomic publication of compiled TensorBlocks into the derivation graph.
//!
//! Tensor payloads live in a content-addressed durable store.  The graph only
//! advances its generation pointer after both the manifest and payload can be
//! read back at their declared addresses.  The published TensorBlock is then a
//! normal `Value` produced by one functional `Derivation`, rather than a
//! parallel provenance mechanism.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    content_hash_bytes, stable_hash, DerivationAnnotation, DerivationGraphReader, DerivationKind,
    DerivationSpec, EdgeRecord, GraphMutation, GraphMutationBatch, GraphStore, GraphStoreError,
    GraphStoreResult, GraphWrite, InMemoryGraphStore, NodeQuery, NodeRecord, RedCoreGraphStore,
    ASSUMPTION_LABEL, CONSUMES_EDGE, DERIVATION_LABEL, OPERATOR_LABEL, PRODUCES_EDGE, UNDER_EDGE,
    VALUE_LABEL, VIA_EDGE,
};

pub const TENSOR_BLOCK_VALUE_LABEL: &str = "TensorBlock";
pub const TENSOR_BLOCK_POINTER_LABEL: &str = "TensorBlockGeneration";
pub const TENSOR_BLOCK_POINTER_EDGE: &str = "POINTS_TO";

/// One TensorBlock manifest lineage address paired with its canonical graph
/// value.  Publication rejects a graph input whose content address does not
/// match the manifest lineage it purports to represent.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorBlockLineageInput {
    pub content_id: String,
    pub value_ref: String,
}

/// A complete addressed edge from the TensorBlock manifest.  The `kind` and
/// `relation` fields make the graph derivation retain every lineage relation,
/// even when two manifest edges resolve to the same source content address.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct TensorBlockManifestLineage {
    pub content_id: String,
    pub kind: String,
    pub relation: String,
}

/// The four retractable premises required for compiled MemTensor publication.
/// They remain distinct fields so an omitted version cannot disappear inside
/// an untyped list.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorBlockAssumptionRefs {
    pub graph_commit: String,
    pub tensor_schema: String,
    pub embedding_model: String,
    pub calibration_epoch: String,
}

impl TensorBlockAssumptionRefs {
    pub fn all(&self) -> Vec<String> {
        vec![
            self.graph_commit.clone(),
            self.tensor_schema.clone(),
            self.embedding_model.clone(),
            self.calibration_epoch.clone(),
        ]
    }
}

/// Core-owned description of a TensorBlock publication.  The tensor crate
/// extracts this from a `TensorBlock`; core validates graph invariants and
/// owns the atomic pointer advance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TensorBlockPublicationInput {
    pub pointer_id: String,
    pub manifest_content_id: String,
    pub manifest_bytes: Vec<u8>,
    pub payload_content_id: String,
    pub payload_bytes: Vec<u8>,
    pub manifest_lineage: Vec<TensorBlockManifestLineage>,
    pub lineage_inputs: Vec<TensorBlockLineageInput>,
    pub assumptions: TensorBlockAssumptionRefs,
    pub operator_id: String,
    pub environment_version: String,
    #[serde(default)]
    pub annotation: DerivationAnnotation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorBlockPublicationReceipt {
    pub pointer_id: String,
    pub value_id: String,
    pub derivation_id: String,
    pub manifest_content_id: String,
    pub payload_content_id: String,
    pub idempotent_replay: bool,
}

/// A validated, graph-specific TensorBlock publication whose durable payload
/// writes and graph mutations have not run yet.
///
/// Callers may prepare several blocks, validate and merge them into one batch,
/// then publish the batch with one graph commit. The fields stay private so
/// payload pinning and mutation construction cannot be bypassed or rebuilt by
/// adapter crates.
#[derive(Clone, Debug)]
pub struct PreparedTensorBlockPublication {
    input: TensorBlockPublicationInput,
    receipt: TensorBlockPublicationReceipt,
    mutations: Vec<GraphMutation>,
}

impl PreparedTensorBlockPublication {
    pub fn receipt(&self) -> &TensorBlockPublicationReceipt {
        &self.receipt
    }
}

/// Several prepared TensorBlock publications validated as one atomic graph
/// mutation. Construct this only through
/// [`validate_and_merge_tensor_block_publications`].
#[derive(Clone, Debug)]
pub struct PreparedTensorBlockPublicationBatch {
    inputs: Vec<TensorBlockPublicationInput>,
    receipts: Vec<TensorBlockPublicationReceipt>,
    mutations: GraphMutationBatch,
}

impl PreparedTensorBlockPublicationBatch {
    pub fn receipts(&self) -> &[TensorBlockPublicationReceipt] {
        &self.receipts
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }
}

/// Content-addressed storage for TensorBlock manifests and payloads.  A store
/// must not report success until bytes survive under their content address.
pub trait TensorBlockPayloadStore: Send + Sync {
    fn put_tensor_block_bytes(
        &self,
        expected_content_id: &str,
        bytes: &[u8],
    ) -> GraphStoreResult<()>;
    fn get_tensor_block_bytes(&self, content_id: &str) -> GraphStoreResult<Option<Vec<u8>>>;

    /// Bind one anonymous writer receipt to an exact logical publication.
    fn bind_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<()> {
        let _ = (content_id, publication_key);
        Ok(())
    }

    /// Release one keyed writer receipt. A missing key returns false and never
    /// consumes another publication's receipt for the same content address.
    fn release_bound_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        let _ = (content_id, publication_key);
        Ok(false)
    }

    /// Create a fresh transaction-owned publication claim. Claims never
    /// consume pending writer receipts, including receipts for the same hash.
    fn claim_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let _ = content_id;
        Ok(())
    }

    /// Release one claimed receipt after the referencing graph root is
    /// durable. This never consumes a different writer's pending receipt.
    fn release_claimed_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let _ = content_id;
        Ok(())
    }

    /// Abort one pending writer receipt. Existing callers use this for failed
    /// property installation; fixtures and third-party stores remain source
    /// compatible through the no-op default.
    fn release_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let _ = content_id;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorPendingPinState {
    Anonymous,
    Bound,
    Released,
}

#[derive(Clone, Debug)]
struct TensorPayloadPinReceipt {
    content_id: String,
    publication_key: String,
    pending: TensorPendingPinState,
    claimed: bool,
}

struct TensorPayloadPinGuard<'a, Store: TensorBlockPayloadStore + ?Sized> {
    store: &'a Store,
    receipts: Vec<TensorPayloadPinReceipt>,
    armed: bool,
}

impl<'a, Store: TensorBlockPayloadStore + ?Sized> TensorPayloadPinGuard<'a, Store> {
    fn new(store: &'a Store) -> Self {
        Self {
            store,
            receipts: Vec::new(),
            armed: true,
        }
    }

    fn record_anonymous(&mut self, content_id: &str, publication_key: String) {
        self.receipts.push(TensorPayloadPinReceipt {
            content_id: content_id.to_string(),
            publication_key,
            pending: TensorPendingPinState::Anonymous,
            claimed: false,
        });
    }

    fn bind_all(&mut self) -> GraphStoreResult<()> {
        for receipt in &mut self.receipts {
            self.store
                .bind_tensor_block_pending(&receipt.content_id, &receipt.publication_key)?;
            receipt.pending = TensorPendingPinState::Bound;
        }
        Ok(())
    }

    fn claim_all(&mut self) -> GraphStoreResult<()> {
        for receipt in &mut self.receipts {
            self.store.claim_tensor_block_pin(&receipt.content_id)?;
            receipt.claimed = true;
        }
        Ok(())
    }

    fn release_receipt(&mut self, index: usize) -> GraphStoreResult<()> {
        let receipt = &mut self.receipts[index];
        match receipt.pending {
            TensorPendingPinState::Anonymous => {
                self.store.release_tensor_block_pin(&receipt.content_id)?;
                receipt.pending = TensorPendingPinState::Released;
            }
            TensorPendingPinState::Bound => {
                self.store.release_bound_tensor_block_pending(
                    &receipt.content_id,
                    &receipt.publication_key,
                )?;
                receipt.pending = TensorPendingPinState::Released;
            }
            TensorPendingPinState::Released => {}
        }
        if receipt.claimed {
            self.store
                .release_claimed_tensor_block_pin(&receipt.content_id)?;
            receipt.claimed = false;
        }
        Ok(())
    }

    fn release(mut self) -> GraphStoreResult<()> {
        while !self.receipts.is_empty() {
            let index = self.receipts.len() - 1;
            self.release_receipt(index)?;
            self.receipts.pop();
        }
        self.armed = false;
        Ok(())
    }
}

impl<Store: TensorBlockPayloadStore + ?Sized> Drop for TensorPayloadPinGuard<'_, Store> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        while !self.receipts.is_empty() {
            let index = self.receipts.len() - 1;
            // Publication errors are already being returned to the caller.
            // Cleanup is best effort during unwinding/early return; any state
            // already released is recorded so a partial retry cannot steal a
            // different writer's receipt for the same content address.
            let _ = self.release_receipt(index);
            self.receipts.pop();
        }
    }
}

/// In-memory payload store for unit and adapter tests.  Production publication
/// uses `DiskObjectStore`, which is the durable implementation below.
#[derive(Clone, Debug, Default)]
pub struct InMemoryTensorBlockPayloadStore {
    objects: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl InMemoryTensorBlockPayloadStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct content-addressed payloads retained by this fixture
    /// store. This is a physical-dedup oracle, not a logical write counter.
    pub fn object_count(&self) -> GraphStoreResult<usize> {
        self.objects
            .lock()
            .map(|objects| objects.len())
            .map_err(|_| {
                GraphStoreError::new("tensor_payload_store_poisoned", "payload store poisoned")
            })
    }

    /// Total physical payload bytes retained by this fixture store.
    pub fn stored_bytes(&self) -> GraphStoreResult<usize> {
        self.objects
            .lock()
            .map_err(|_| {
                GraphStoreError::new("tensor_payload_store_poisoned", "payload store poisoned")
            })?
            .values()
            .try_fold(0_usize, |total, bytes| total.checked_add(bytes.len()))
            .ok_or_else(|| {
                GraphStoreError::new(
                    "tensor_payload_store_size_overflow",
                    "payload byte accounting overflowed usize",
                )
            })
    }
}

impl TensorBlockPayloadStore for InMemoryTensorBlockPayloadStore {
    fn put_tensor_block_bytes(
        &self,
        expected_content_id: &str,
        bytes: &[u8],
    ) -> GraphStoreResult<()> {
        ensure_content_address(expected_content_id, bytes)?;
        self.objects
            .lock()
            .map_err(|_| {
                GraphStoreError::new("tensor_payload_store_poisoned", "payload store poisoned")
            })?
            .entry(expected_content_id.to_string())
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn get_tensor_block_bytes(&self, content_id: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        Ok(self
            .objects
            .lock()
            .map_err(|_| {
                GraphStoreError::new("tensor_payload_store_poisoned", "payload store poisoned")
            })?
            .get(content_id)
            .cloned())
    }
}

impl TensorBlockPayloadStore for crate::DiskObjectStore {
    fn put_tensor_block_bytes(
        &self,
        expected_content_id: &str,
        bytes: &[u8],
    ) -> GraphStoreResult<()> {
        ensure_content_address(expected_content_id, bytes)?;
        let address = crate::Hash::parse_prefixed(expected_content_id)?;
        let chunk_store = self.chunk_store();
        chunk_store.pin_in_flight(address)?;
        let write_result = (|| {
            let stored = self.put_unpublished_bytes(bytes)?;
            if stored != expected_content_id {
                return Err(GraphStoreError::new(
                    "tensor_payload_address_mismatch",
                    format!("stored TensorBlock bytes as {stored}, expected {expected_content_id}"),
                ));
            }
            Ok(())
        })();
        if let Err(write_error) = write_result {
            return match chunk_store.unpin_in_flight(address) {
                Ok(()) => Err(write_error),
                Err(release_error) => Err(GraphStoreError::new(
                    "tensor_payload_write_and_unpin_failed",
                    format!(
                        "payload write failed ({}: {}); releasing its in-flight pin also failed ({}: {})",
                        write_error.code,
                        write_error.message,
                        release_error.code,
                        release_error.message
                    ),
                )),
            };
        }
        Ok(())
    }

    fn get_tensor_block_bytes(&self, content_id: &str) -> GraphStoreResult<Option<Vec<u8>>> {
        self.get_unpublished_bytes(content_id)
    }

    fn bind_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<()> {
        let address = crate::Hash::parse_prefixed(content_id)?;
        self.chunk_store()
            .bind_pending_in_flight(address, publication_key)
    }

    fn release_bound_tensor_block_pending(
        &self,
        content_id: &str,
        publication_key: &str,
    ) -> GraphStoreResult<bool> {
        let address = crate::Hash::parse_prefixed(content_id)?;
        self.chunk_store()
            .release_bound_pending_in_flight(address, publication_key)
    }

    fn claim_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let address = crate::Hash::parse_prefixed(content_id)?;
        self.chunk_store().claim_in_flight(address)
    }

    fn release_claimed_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let address = crate::Hash::parse_prefixed(content_id)?;
        self.chunk_store().release_claimed_in_flight(address)
    }

    fn release_tensor_block_pin(&self, content_id: &str) -> GraphStoreResult<()> {
        let address = crate::Hash::parse_prefixed(content_id)?;
        self.chunk_store().unpin_in_flight(address)
    }
}

/// Owned-read plus atomic-graph-batch contract.  The durable payload write is
/// deliberately outside this transaction and must complete before it starts.
pub trait TensorBlockDerivationStore: DerivationGraphReader {
    fn commit_tensor_block_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()>;
}

impl TensorBlockDerivationStore for InMemoryGraphStore {
    fn commit_tensor_block_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        validate_tensor_block_batch(&batch)?;
        let mut staged = self.clone();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => {
                    crate::GraphWrite::upsert_node(&mut staged, node)?;
                }
                GraphMutation::EdgeUpsert(edge) => {
                    crate::GraphWrite::upsert_edge(&mut staged, edge)?;
                }
                GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                    return Err(GraphStoreError::new(
                        "tensor_block_delete_unsupported",
                        "TensorBlock publication batches cannot contain graph deletes",
                    ));
                }
            }
        }
        *self = staged;
        Ok(())
    }
}

impl TensorBlockDerivationStore for RedCoreGraphStore {
    fn commit_tensor_block_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        validate_tensor_block_batch(&batch)?;
        self.commit_batch(batch).map(|_| ())
    }
}

fn validate_tensor_block_batch(batch: &GraphMutationBatch) -> GraphStoreResult<()> {
    if batch.mutations.iter().any(|mutation| {
        matches!(
            mutation,
            GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_)
        )
    }) {
        return Err(GraphStoreError::new(
            "tensor_block_delete_unsupported",
            "TensorBlock publication batches cannot contain graph deletes",
        ));
    }
    Ok(())
}

/// Stable graph value identity for a content-addressed TensorBlock manifest.
pub fn tensor_block_value_id(manifest_content_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "value:tensor-block:{}",
        stable_hash(manifest_content_id.trim())?
    ))
}

fn tensor_block_pin_publication_key(
    pointer_id: &str,
    role: &str,
    content_id: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "tensor-block-publication:{}",
        stable_hash((pointer_id, role, content_id))?
    ))
}

/// Validate one TensorBlock publication and construct its graph mutations
/// without writing payload bytes or changing the graph.
pub fn prepare_tensor_block_publication(
    graph: &(impl TensorBlockDerivationStore + ?Sized),
    input: TensorBlockPublicationInput,
) -> GraphStoreResult<PreparedTensorBlockPublication> {
    validate_publication_input(&input)?;
    validate_graph_inputs(graph, &input)?;

    let value_id = tensor_block_value_id(&input.manifest_content_id)?;
    let mut lineage_inputs = input.lineage_inputs.clone();
    lineage_inputs.sort_by(|left, right| left.value_ref.cmp(&right.value_ref));
    let input_refs = lineage_inputs
        .iter()
        .map(|lineage| lineage.value_ref.clone())
        .collect::<Vec<_>>();
    let assumption_refs = input.assumptions.all();
    let annotation = DerivationAnnotation {
        confidence: input.annotation.confidence.clamp(0.0, 1.0),
        cost: input.annotation.cost.max(0.0),
    };
    let spec = DerivationSpec {
        kind: DerivationKind::Functional,
        operator_id: input.operator_id.clone(),
        input_refs,
        output_refs: vec![value_id.clone()],
        assumption_refs: assumption_refs.clone(),
        params: json!({
            "manifest_content_id": input.manifest_content_id.clone(),
            "payload_content_id": input.payload_content_id.clone(),
            "manifest_lineage": input.manifest_lineage.clone(),
            "lineage_inputs": lineage_inputs.clone(),
        }),
        environment_version: input.environment_version.clone(),
        annotation,
    };
    let derivation_id = spec.content_id()?;
    let value_node = NodeRecord::new(
        &value_id,
        [VALUE_LABEL, TENSOR_BLOCK_VALUE_LABEL],
        json!({
            "manifest_content_id": input.manifest_content_id.clone(),
            "payload_content_id": input.payload_content_id.clone(),
            "content_id": input.manifest_content_id.clone(),
        }),
    );
    let pointer_node = NodeRecord::new(
        &input.pointer_id,
        [TENSOR_BLOCK_POINTER_LABEL],
        json!({
            "value_ref": value_id.clone(),
            "derivation_ref": derivation_id.clone(),
            "manifest_content_id": input.manifest_content_id.clone(),
            "payload_content_id": input.payload_content_id.clone(),
        }),
    );

    let existing_pointer = graph.derivation_node(&input.pointer_id)?;
    if existing_pointer.as_ref().is_some_and(|node| {
        node.labels
            .iter()
            .any(|label| label == TENSOR_BLOCK_POINTER_LABEL)
            && node.properties == pointer_node.properties
    }) {
        return Ok(PreparedTensorBlockPublication {
            receipt: TensorBlockPublicationReceipt {
                pointer_id: input.pointer_id.clone(),
                value_id,
                derivation_id,
                manifest_content_id: input.manifest_content_id.clone(),
                payload_content_id: input.payload_content_id.clone(),
                idempotent_replay: true,
            },
            input,
            mutations: Vec::new(),
        });
    }
    if let Some(existing) = existing_pointer {
        if !existing
            .labels
            .iter()
            .any(|label| label == TENSOR_BLOCK_POINTER_LABEL)
        {
            return Err(GraphStoreError::new(
                "tensor_pointer_identity_collision",
                format!(
                    "{} is not a TensorBlock generation pointer",
                    input.pointer_id
                ),
            ));
        }
    }

    ensure_compatible_node(
        graph,
        &value_node,
        TENSOR_BLOCK_VALUE_LABEL,
        "tensor_value_identity_collision",
    )?;
    let operator_node = NodeRecord::new(
        &input.operator_id,
        [OPERATOR_LABEL],
        json!({"name": "TensorBlock compiler"}),
    );
    ensure_compatible_node(
        graph,
        &operator_node,
        OPERATOR_LABEL,
        "tensor_operator_identity_collision",
    )?;
    for producer in graph
        .derivation_neighbors(crate::NeighborQuery::in_(&value_id).with_edge_type(PRODUCES_EDGE))?
    {
        if producer.node_id != derivation_id {
            return Err(GraphStoreError::new(
                "tensor_value_multiple_producers",
                format!(
                    "TensorBlock Value {value_id} is already produced by {}",
                    producer.node_id
                ),
            ));
        }
    }
    let derivation_node = NodeRecord::new(
        &derivation_id,
        [DERIVATION_LABEL],
        serde_json::to_value(&spec).map_err(|error| {
            GraphStoreError::new("invalid_tensor_derivation", error.to_string())
        })?,
    );
    ensure_compatible_node(
        graph,
        &derivation_node,
        DERIVATION_LABEL,
        "tensor_derivation_identity_collision",
    )?;

    let mut mutations = Vec::new();
    if graph.derivation_node(&value_id)?.is_none() {
        mutations.push(GraphMutation::NodeUpsert(value_node.clone()));
    }
    if graph.derivation_node(&input.operator_id)?.is_none() {
        mutations.push(GraphMutation::NodeUpsert(operator_node));
    }
    if graph.derivation_node(&derivation_id)?.is_none() {
        mutations.push(GraphMutation::NodeUpsert(derivation_node));
    }
    mutations.push(GraphMutation::NodeUpsert(pointer_node));
    for input_ref in &spec.input_refs {
        mutations.push(GraphMutation::EdgeUpsert(tensor_edge(
            &derivation_id,
            CONSUMES_EDGE,
            input_ref,
            json!({}),
        )?));
    }
    mutations.push(GraphMutation::EdgeUpsert(tensor_edge(
        &derivation_id,
        PRODUCES_EDGE,
        &value_id,
        json!({"expected_content_hash": value_node.content_address()?}),
    )?));
    mutations.push(GraphMutation::EdgeUpsert(tensor_edge(
        &derivation_id,
        VIA_EDGE,
        &input.operator_id,
        json!({}),
    )?));
    for assumption_ref in &assumption_refs {
        mutations.push(GraphMutation::EdgeUpsert(tensor_edge(
            &derivation_id,
            UNDER_EDGE,
            assumption_ref,
            json!({}),
        )?));
    }
    mutations.push(GraphMutation::EdgeUpsert(EdgeRecord::new(
        format!(
            "tensor-generation-pointer-edge:{}",
            stable_hash((&input.pointer_id, TENSOR_BLOCK_POINTER_EDGE))?
        ),
        &input.pointer_id,
        TENSOR_BLOCK_POINTER_EDGE,
        &value_id,
        json!({"derivation_ref": derivation_id.clone()}),
    )));

    Ok(PreparedTensorBlockPublication {
        receipt: TensorBlockPublicationReceipt {
            pointer_id: input.pointer_id.clone(),
            value_id,
            derivation_id,
            manifest_content_id: input.manifest_content_id.clone(),
            payload_content_id: input.payload_content_id.clone(),
            idempotent_replay: false,
        },
        input,
        mutations,
    })
}

/// Validate cross-publication identities and merge several prepared blocks
/// into one graph mutation. Repeated compatible value/operator/derivation
/// records are deduplicated; competing pointer identities or producers are
/// refused before any durable write.
pub fn validate_and_merge_tensor_block_publications(
    publications: impl IntoIterator<Item = PreparedTensorBlockPublication>,
) -> GraphStoreResult<PreparedTensorBlockPublicationBatch> {
    let publications = publications.into_iter().collect::<Vec<_>>();
    if publications.is_empty() {
        return Err(GraphStoreError::new(
            "empty_tensor_publication_batch",
            "a TensorBlock publication batch must contain at least one prepared block",
        ));
    }

    let mut pointer_ids = BTreeSet::new();
    let mut value_producers = HashMap::<String, String>::new();
    let mut node_mutations = HashMap::<String, NodeRecord>::new();
    let mut edge_mutations = HashMap::<String, EdgeRecord>::new();
    let mut merged_mutations = Vec::new();
    let mut inputs = Vec::with_capacity(publications.len());
    let mut receipts = Vec::with_capacity(publications.len());

    for publication in publications {
        if !pointer_ids.insert(publication.receipt.pointer_id.clone()) {
            return Err(GraphStoreError::new(
                "duplicate_tensor_publication_pointer",
                format!(
                    "TensorBlock publication batch repeats pointer {}",
                    publication.receipt.pointer_id
                ),
            ));
        }
        if let Some(existing) = value_producers.insert(
            publication.receipt.value_id.clone(),
            publication.receipt.derivation_id.clone(),
        ) {
            if existing != publication.receipt.derivation_id {
                return Err(GraphStoreError::new(
                    "tensor_value_multiple_producers",
                    format!(
                        "TensorBlock Value {} has competing prepared producers {existing} and {}",
                        publication.receipt.value_id, publication.receipt.derivation_id
                    ),
                ));
            }
        }

        for mutation in publication.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => {
                    if let Some(existing) = node_mutations.get(&node.id) {
                        if existing != &node {
                            return Err(GraphStoreError::new(
                                "tensor_publication_batch_node_collision",
                                format!(
                                    "prepared TensorBlock publications disagree on node {}",
                                    node.id
                                ),
                            ));
                        }
                        continue;
                    }
                    node_mutations.insert(node.id.clone(), node.clone());
                    merged_mutations.push(GraphMutation::NodeUpsert(node));
                }
                GraphMutation::EdgeUpsert(edge) => {
                    if let Some(existing) = edge_mutations.get(&edge.id) {
                        if existing != &edge {
                            return Err(GraphStoreError::new(
                                "tensor_publication_batch_edge_collision",
                                format!(
                                    "prepared TensorBlock publications disagree on edge {}",
                                    edge.id
                                ),
                            ));
                        }
                        continue;
                    }
                    edge_mutations.insert(edge.id.clone(), edge.clone());
                    merged_mutations.push(GraphMutation::EdgeUpsert(edge));
                }
                GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                    return Err(GraphStoreError::new(
                        "tensor_block_delete_unsupported",
                        "TensorBlock publication batches cannot contain graph deletes",
                    ));
                }
            }
        }
        inputs.push(publication.input);
        receipts.push(publication.receipt);
    }

    let mutations = GraphMutationBatch::new(merged_mutations);
    validate_tensor_block_batch(&mutations)?;
    Ok(PreparedTensorBlockPublicationBatch {
        inputs,
        receipts,
        mutations,
    })
}

/// Persist every prepared block, retain its payload pins, and publish all
/// values, derivations, and generation pointers with one graph commit.
pub fn publish_prepared_tensor_block_batch(
    graph: &mut (impl TensorBlockDerivationStore + ?Sized),
    payload_store: &(impl TensorBlockPayloadStore + ?Sized),
    batch: PreparedTensorBlockPublicationBatch,
) -> GraphStoreResult<Vec<TensorBlockPublicationReceipt>> {
    let PreparedTensorBlockPublicationBatch {
        inputs,
        receipts,
        mutations,
    } = batch;

    // A graph pointer may never advertise bytes which have not survived a
    // round-trip through the durable object store. One guard owns every pin in
    // the merged batch so any preparation or commit failure releases all of
    // them together.
    let mut payload_pins = TensorPayloadPinGuard::new(payload_store);
    for input in &inputs {
        let manifest_publication_key = tensor_block_pin_publication_key(
            &input.pointer_id,
            "manifest_content_id",
            &input.manifest_content_id,
        )?;
        let payload_publication_key = tensor_block_pin_publication_key(
            &input.pointer_id,
            "payload_content_id",
            &input.payload_content_id,
        )?;
        payload_store.put_tensor_block_bytes(&input.manifest_content_id, &input.manifest_bytes)?;
        payload_pins.record_anonymous(&input.manifest_content_id, manifest_publication_key);
        payload_store.put_tensor_block_bytes(&input.payload_content_id, &input.payload_bytes)?;
        payload_pins.record_anonymous(&input.payload_content_id, payload_publication_key);
        verify_durable_bytes(
            payload_store,
            &input.manifest_content_id,
            &input.manifest_bytes,
        )?;
        verify_durable_bytes(
            payload_store,
            &input.payload_content_id,
            &input.payload_bytes,
        )?;
    }
    payload_pins.bind_all()?;
    payload_pins.claim_all()?;
    if !mutations.mutations.is_empty() {
        graph.commit_tensor_block_batch(mutations)?;
    }
    payload_pins.release()?;
    Ok(receipts)
}

/// Publish one TensorBlock after content-address verification. This is the
/// compatibility wrapper around the prepared batch API.
pub fn publish_tensor_block(
    graph: &mut (impl TensorBlockDerivationStore + ?Sized),
    payload_store: &(impl TensorBlockPayloadStore + ?Sized),
    input: TensorBlockPublicationInput,
) -> GraphStoreResult<TensorBlockPublicationReceipt> {
    let prepared = prepare_tensor_block_publication(graph, input)?;
    let batch = validate_and_merge_tensor_block_publications([prepared])?;
    let mut receipts = publish_prepared_tensor_block_batch(graph, payload_store, batch)?;
    receipts.pop().ok_or_else(|| {
        GraphStoreError::new(
            "empty_tensor_publication_receipt",
            "single TensorBlock publication returned no receipt",
        )
    })
}

fn validate_publication_input(input: &TensorBlockPublicationInput) -> GraphStoreResult<()> {
    for (field, value) in [
        ("pointer_id", &input.pointer_id),
        ("manifest_content_id", &input.manifest_content_id),
        ("payload_content_id", &input.payload_content_id),
        ("operator_id", &input.operator_id),
        ("environment_version", &input.environment_version),
    ] {
        if value.trim().is_empty() {
            return Err(GraphStoreError::new(
                "empty_graph_field",
                format!("{field} cannot be empty"),
            ));
        }
    }
    ensure_content_address(&input.manifest_content_id, &input.manifest_bytes)?;
    ensure_content_address(&input.payload_content_id, &input.payload_bytes)?;
    if input.lineage_inputs.is_empty() {
        return Err(GraphStoreError::new(
            "empty_tensor_lineage",
            "TensorBlock publication requires at least one manifest lineage input",
        ));
    }
    if input.manifest_lineage.is_empty() {
        return Err(GraphStoreError::new(
            "empty_tensor_manifest_lineage",
            "TensorBlock publication requires complete manifest lineage",
        ));
    }
    let mut content_ids = BTreeSet::new();
    let mut value_refs = BTreeSet::new();
    for lineage in &input.lineage_inputs {
        if lineage.content_id.trim().is_empty() || lineage.value_ref.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_tensor_lineage",
                "TensorBlock lineage content IDs and value refs are required",
            ));
        }
        if !content_ids.insert(lineage.content_id.clone())
            || !value_refs.insert(lineage.value_ref.clone())
        {
            return Err(GraphStoreError::new(
                "duplicate_tensor_lineage",
                "TensorBlock lineage inputs must be unique by content ID and value ref",
            ));
        }
    }
    let manifest_lineage = input
        .manifest_lineage
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if manifest_lineage.len() != input.manifest_lineage.len()
        || manifest_lineage.iter().any(|lineage| {
            lineage.content_id.trim().is_empty()
                || lineage.kind.trim().is_empty()
                || lineage.relation.trim().is_empty()
        })
    {
        return Err(GraphStoreError::new(
            "invalid_tensor_manifest_lineage",
            "TensorBlock manifest lineage edges must be unique and complete",
        ));
    }
    let manifest_content_ids = manifest_lineage
        .iter()
        .map(|lineage| lineage.content_id.as_str())
        .collect::<BTreeSet<_>>();
    if !content_ids
        .iter()
        .all(|content_id| manifest_content_ids.contains(content_id.as_str()))
        || !manifest_content_ids
            .iter()
            .all(|content_id| content_ids.contains(*content_id))
    {
        return Err(GraphStoreError::new(
            "tensor_manifest_input_mismatch",
            "TensorBlock manifest lineage and derivation inputs must name the same content addresses",
        ));
    }
    let assumptions = input.assumptions.all();
    if assumptions
        .iter()
        .any(|reference| reference.trim().is_empty())
        || assumptions.iter().collect::<BTreeSet<_>>().len() != assumptions.len()
    {
        return Err(GraphStoreError::new(
            "invalid_tensor_assumptions",
            "graph commit, tensor schema, embedding model, and calibration assumptions must be distinct",
        ));
    }
    Ok(())
}

fn validate_graph_inputs(
    graph: &(impl DerivationGraphReader + ?Sized),
    input: &TensorBlockPublicationInput,
) -> GraphStoreResult<()> {
    for lineage in &input.lineage_inputs {
        let node = graph.derivation_node(&lineage.value_ref)?.ok_or_else(|| {
            GraphStoreError::new(
                "missing_tensor_lineage_value",
                format!(
                    "TensorBlock lineage value {} does not exist",
                    lineage.value_ref
                ),
            )
        })?;
        if !node.labels.iter().any(|label| label == VALUE_LABEL) {
            return Err(GraphStoreError::new(
                "invalid_tensor_lineage_value",
                format!(
                    "TensorBlock lineage value {} is not a Value",
                    lineage.value_ref
                ),
            ));
        }
        if node.properties.get("content_id").and_then(Value::as_str) != Some(&lineage.content_id) {
            return Err(GraphStoreError::new(
                "tensor_lineage_content_mismatch",
                format!(
                    "TensorBlock lineage value {} does not represent {}",
                    lineage.value_ref, lineage.content_id
                ),
            ));
        }
    }
    for assumption_ref in input.assumptions.all() {
        let assumption = graph.derivation_node(&assumption_ref)?.ok_or_else(|| {
            GraphStoreError::new(
                "missing_tensor_assumption",
                format!("TensorBlock assumption {assumption_ref} does not exist"),
            )
        })?;
        if !assumption
            .labels
            .iter()
            .any(|label| label == ASSUMPTION_LABEL)
        {
            return Err(GraphStoreError::new(
                "invalid_tensor_assumption",
                format!("TensorBlock assumption {assumption_ref} is not an Assumption"),
            ));
        }
        if !assumption
            .properties
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(GraphStoreError::new(
                "inactive_tensor_assumption",
                format!("TensorBlock assumption {assumption_ref} is inactive"),
            ));
        }
    }
    Ok(())
}

fn ensure_content_address(expected_content_id: &str, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = content_hash_bytes(bytes);
    if actual != expected_content_id {
        return Err(GraphStoreError::new(
            "tensor_payload_address_mismatch",
            format!("TensorBlock bytes address {actual}, expected {expected_content_id}"),
        ));
    }
    Ok(())
}

fn verify_durable_bytes(
    store: &(impl TensorBlockPayloadStore + ?Sized),
    content_id: &str,
    expected: &[u8],
) -> GraphStoreResult<()> {
    let actual = store.get_tensor_block_bytes(content_id)?.ok_or_else(|| {
        GraphStoreError::new(
            "tensor_payload_not_durable",
            format!("TensorBlock bytes at {content_id} were not readable after persistence"),
        )
    })?;
    if actual != expected {
        return Err(GraphStoreError::new(
            "tensor_payload_readback_mismatch",
            format!("TensorBlock bytes at {content_id} changed before graph publication"),
        ));
    }
    Ok(())
}

fn ensure_compatible_node(
    graph: &(impl DerivationGraphReader + ?Sized),
    expected: &NodeRecord,
    required_label: &str,
    error_code: &str,
) -> GraphStoreResult<()> {
    let Some(existing) = graph.derivation_node(&expected.id)? else {
        return Ok(());
    };
    if !existing.labels.iter().any(|label| label == required_label)
        || existing.properties != expected.properties
    {
        return Err(GraphStoreError::new(
            error_code,
            format!(
                "existing graph node {} conflicts with TensorBlock publication",
                expected.id
            ),
        ));
    }
    Ok(())
}

fn tensor_edge(
    from_id: &str,
    edge_type: &str,
    to_id: &str,
    properties: Value,
) -> GraphStoreResult<EdgeRecord> {
    Ok(EdgeRecord::new(
        format!(
            "tensor-derivation-edge:{}",
            stable_hash((from_id, edge_type, to_id))?
        ),
        from_id,
        edge_type,
        to_id,
        properties,
    ))
}

const GRAPH_COMMIT_FAMILY: &str = "graph-commit";
const CALIBRATION_EPOCH_FAMILY: &str = "calibration-epoch";

/// Create or reactivate `assumption:graph-commit:<hash>` and deactivate any
/// prior graph-commit assumption for the same log.
pub fn ensure_assumption_for_pin(
    graph: &mut impl GraphStore,
    log_id: &str,
    log_content_hash: &str,
) -> GraphStoreResult<String> {
    let assumption_id = format!("assumption:graph-commit:{log_content_hash}");
    deactivate_family(graph, GRAPH_COMMIT_FAMILY, "log_id", log_id, &assumption_id)?;
    upsert_family_assumption(
        graph,
        &assumption_id,
        GRAPH_COMMIT_FAMILY,
        "log_id",
        log_id,
        log_content_hash,
    )?;
    Ok(assumption_id)
}

/// Create or reactivate `assumption:calibration-epoch:<id>` and deactivate any
/// prior calibration-epoch assumption for the same token region.
pub fn ensure_assumption_for_epoch(
    graph: &mut impl GraphStore,
    epoch_id: &str,
) -> GraphStoreResult<String> {
    let assumption_id = format!("assumption:calibration-epoch:{epoch_id}");
    deactivate_family(
        graph,
        CALIBRATION_EPOCH_FAMILY,
        "token_region",
        "algebra/tokens",
        &assumption_id,
    )?;
    upsert_family_assumption(
        graph,
        &assumption_id,
        CALIBRATION_EPOCH_FAMILY,
        "token_region",
        "algebra/tokens",
        epoch_id,
    )?;
    Ok(assumption_id)
}

fn deactivate_family(
    graph: &mut impl GraphStore,
    family: &str,
    scope_key: &str,
    scope_value: &str,
    keep_id: &str,
) -> GraphStoreResult<()> {
    let existing = graph.query_nodes(NodeQuery::label(ASSUMPTION_LABEL).with_limit(usize::MAX));
    for mut node in existing {
        if node.id == keep_id {
            continue;
        }
        let matches_family = node.properties.get("family").and_then(Value::as_str) == Some(family);
        let matches_scope =
            node.properties.get(scope_key).and_then(Value::as_str) == Some(scope_value);
        if matches_family && matches_scope {
            if let Some(properties) = node.properties.as_object_mut() {
                properties.insert("active".into(), Value::Bool(false));
            }
            GraphWrite::upsert_node(graph, node)?;
        }
    }
    Ok(())
}

fn upsert_family_assumption(
    graph: &mut impl GraphStore,
    assumption_id: &str,
    family: &str,
    scope_key: &str,
    scope_value: &str,
    name: &str,
) -> GraphStoreResult<()> {
    GraphWrite::upsert_node(
        graph,
        NodeRecord::new(
            assumption_id,
            [ASSUMPTION_LABEL],
            json!({
                "name": name,
                "active": true,
                "family": family,
                scope_key: scope_value,
            }),
        ),
    )?;
    Ok(())
}
