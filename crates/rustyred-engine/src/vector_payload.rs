//! Typed vector properties backed by the shared content-addressed payload store.
//!
//! Node JSON carries only [`VectorPropertyRef`]. The dense payload has one
//! canonical representation: consecutive little-endian IEEE-754 `f32` bits.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;
use crate::graph::vector_index::VectorDesignation;
use crate::hash::Hash;
use crate::tensor_derivation::TensorBlockPayloadStore;

/// Vector-reference positions that are part of the graph schema even when no
/// search index has been designated for them. These fields still require the
/// same publication pin and reachability treatment as indexed vectors.
pub(crate) const BUILTIN_VECTOR_PAYLOAD_PROPERTIES: [(&str, &str); 4] = [
    ("Item", "embedding"),
    ("Collection", "label_embedding"),
    ("Entity", "entity_embedding"),
    ("EpistemicShadow", "structural_role_vector"),
];

/// Exact graph-property reference to one packed vector payload.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VectorPropertyRef {
    pub content_id: String,
    pub dimension: usize,
}

impl VectorPropertyRef {
    /// Strictly decode the only accepted vector-reference JSON shape.
    pub fn from_value(value: &Value) -> GraphStoreResult<Self> {
        let object = value.as_object().ok_or_else(|| {
            GraphStoreError::new(
                "vector_property_reference_invalid",
                "vector property reference must be a JSON object",
            )
        })?;
        if object.len() != 2
            || !object.contains_key("content_id")
            || !object.contains_key("dimension")
        {
            return Err(GraphStoreError::new(
                "vector_property_reference_invalid",
                "vector property reference must contain only content_id and dimension",
            ));
        }
        let reference: Self = serde_json::from_value(value.clone()).map_err(|error| {
            GraphStoreError::new("vector_property_reference_invalid", error.to_string())
        })?;
        Hash::parse_prefixed(&reference.content_id).map_err(|error| {
            GraphStoreError::new(
                "vector_property_reference_invalid",
                format!("content_id is not canonical: {}", error.message),
            )
        })?;
        reference.dimension.checked_mul(4).ok_or_else(|| {
            GraphStoreError::new(
                "vector_property_dimension_overflow",
                "vector dimension exceeds the addressable packed-byte length",
            )
        })?;
        Ok(reference)
    }

    /// Encode the exact two-field graph-property value.
    pub fn to_value(&self) -> GraphStoreResult<Value> {
        Hash::parse_prefixed(&self.content_id)?;
        self.dimension.checked_mul(4).ok_or_else(|| {
            GraphStoreError::new(
                "vector_property_dimension_overflow",
                "vector dimension exceeds the addressable packed-byte length",
            )
        })?;
        let mut object = Map::new();
        object.insert(
            "content_id".to_string(),
            Value::String(self.content_id.clone()),
        );
        object.insert(
            "dimension".to_string(),
            Value::Number(serde_json::Number::from(
                u64::try_from(self.dimension).map_err(|_| {
                    GraphStoreError::new(
                        "vector_property_dimension_overflow",
                        "vector dimension exceeds the canonical JSON integer range",
                    )
                })?,
            )),
        );
        Ok(Value::Object(object))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VectorPropertyPendingWrite {
    content_id: String,
    publication_key: String,
}

/// Owns the exact pending-pin receipts created while detached nodes are being
/// prepared for one graph publication.
///
/// Successful graph-store preflight creates transaction-owned claims and
/// releases these keyed receipts. Until then, every staged receipt remains
/// owned here so an abandoned multi-node preparation can cancel only its own
/// `(node, property, content)` writes. Call
/// [`Self::finish_after_publication`] only after the graph-store operation
/// succeeds.
#[must_use = "staged vector writes must be published or explicitly cancelled"]
pub struct VectorPropertyStagingGuard {
    store: Arc<dyn TensorBlockPayloadStore>,
    pending: Vec<VectorPropertyPendingWrite>,
}

impl fmt::Debug for VectorPropertyStagingGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VectorPropertyStagingGuard")
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl VectorPropertyStagingGuard {
    pub fn new(store: Arc<dyn TensorBlockPayloadStore>) -> Self {
        Self {
            store,
            pending: Vec::new(),
        }
    }

    /// Pack, durably write, verify, and install one vector-property reference
    /// while retaining ownership of any newly-created pending receipt.
    pub fn write_vector_property(
        &mut self,
        node: &mut NodeRecord,
        property: &str,
        vector: &[f32],
    ) -> GraphStoreResult<()> {
        if let Some(receipt) =
            write_vector_property_staged(node, property, vector, self.store.as_ref())?
        {
            self.pending.push(receipt);
        }
        Ok(())
    }

    /// Cancel every receipt still owned by this preparation. Each release is
    /// exact-keyed and ref-counted, including duplicate content payloads.
    pub fn cancel(&mut self) -> GraphStoreResult<()> {
        let mut failed = Vec::new();
        let mut errors = Vec::new();
        for receipt in std::mem::take(&mut self.pending).into_iter().rev() {
            if let Err(error) = self
                .store
                .release_bound_tensor_block_pending(&receipt.content_id, &receipt.publication_key)
            {
                failed.push(receipt);
                errors.push(error);
            }
        }
        failed.reverse();
        self.pending = failed;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphStoreError::new(
                "vector_staging_pin_cleanup_failed",
                errors
                    .into_iter()
                    .map(|error| format!("{}: {}", error.code, error.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            ))
        }
    }

    /// Finish after graph publication by exact-releasing every tracked key.
    /// Receipts already consumed by graph-store preflight return `false`; a
    /// staged receipt omitted from the published batch is released here.
    ///
    /// Once publication succeeds cleanup cannot change its transaction
    /// outcome. Failed releases remain in `pending` and `Drop` retries once.
    pub fn finish_after_publication(&mut self) {
        let _ = self.cancel();
    }
}

impl Drop for VectorPropertyStagingGuard {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}

/// Pack, durably write, verify, and install one vector-property reference.
pub fn write_vector_property(
    node: &mut NodeRecord,
    property: &str,
    vector: &[f32],
    store: &(impl TensorBlockPayloadStore + ?Sized),
) -> GraphStoreResult<()> {
    write_vector_property_staged(node, property, vector, store).map(|_| ())
}

fn write_vector_property_staged(
    node: &mut NodeRecord,
    property: &str,
    vector: &[f32],
    store: &(impl TensorBlockPayloadStore + ?Sized),
) -> GraphStoreResult<Option<VectorPropertyPendingWrite>> {
    let property = property.trim();
    if property.is_empty() {
        return Err(GraphStoreError::new(
            "vector_property_name_invalid",
            "vector property name must be non-empty",
        ));
    }
    let properties = node.properties.as_object().ok_or_else(|| {
        GraphStoreError::new(
            "vector_property_bag_invalid",
            "node properties must be a JSON object",
        )
    })?;
    let prior_reference = properties
        .get(property)
        .and_then(|value| VectorPropertyRef::from_value(value).ok());
    let bytes = pack_vector(vector);
    let content_id = Hash::of(&bytes).to_prefixed();
    let publication_key = vector_pin_publication_key(&node.id, property, &content_id);
    let reference_value = VectorPropertyRef {
        content_id: content_id.clone(),
        dimension: vector.len(),
    }
    .to_value()?;
    if prior_reference
        .as_ref()
        .is_some_and(|reference| reference.content_id == content_id)
    {
        let readback = store.get_tensor_block_bytes(&content_id)?.ok_or_else(|| {
            GraphStoreError::new(
                "vector_payload_not_durable",
                format!("payload store cannot read back vector {content_id}"),
            )
        })?;
        if readback != bytes {
            return Err(GraphStoreError::new(
                "vector_payload_readback_mismatch",
                format!("payload store returned different bytes for vector {content_id}"),
            ));
        }
        node.properties
            .as_object_mut()
            .expect("property bag was validated before vector readback")
            .insert(property.to_string(), reference_value);
        return Ok(None);
    }
    let prior_binding = prior_reference.map(|reference| {
        let publication_key = vector_pin_publication_key(&node.id, property, &reference.content_id);
        (reference.content_id, publication_key)
    });
    store.put_tensor_block_bytes(&content_id, &bytes)?;
    let mut pending_is_bound = false;
    let install_result = (|| {
        let readback = store.get_tensor_block_bytes(&content_id)?.ok_or_else(|| {
            GraphStoreError::new(
                "vector_payload_not_durable",
                format!("payload store cannot read back vector {content_id}"),
            )
        })?;
        if readback != bytes {
            return Err(GraphStoreError::new(
                "vector_payload_readback_mismatch",
                format!("payload store returned different bytes for vector {content_id}"),
            ));
        }

        store.bind_tensor_block_pending(&content_id, &publication_key)?;
        pending_is_bound = true;
        if let Some((prior_content_id, prior_publication_key)) = &prior_binding {
            store.release_bound_tensor_block_pending(prior_content_id, prior_publication_key)?;
        }
        node.properties
            .as_object_mut()
            .expect("property bag was validated before payload publication")
            .insert(property.to_string(), reference_value);
        Ok(())
    })();
    if let Err(write_error) = install_result {
        let release_result = if pending_is_bound {
            store
                .release_bound_tensor_block_pending(&content_id, &publication_key)
                .map(|_| ())
        } else {
            store.release_tensor_block_pin(&content_id)
        };
        return match release_result {
            Ok(()) => Err(write_error),
            Err(release_error) => Err(GraphStoreError::new(
                "vector_property_write_and_unpin_failed",
                format!(
                    "vector property write failed ({}: {}); releasing its in-flight pin also failed ({}: {})",
                    write_error.code,
                    write_error.message,
                    release_error.code,
                    release_error.message
                ),
            )),
        };
    }
    Ok(Some(VectorPropertyPendingWrite {
        content_id,
        publication_key,
    }))
}

/// Opaque ownership key for the pending write attached to one detached node
/// property. Content ID alone is insufficient because two nodes can publish
/// the same deduplicated payload concurrently.
pub(crate) fn vector_pin_publication_key(
    node_id: &str,
    property: &str,
    content_id: &str,
) -> String {
    let mut preimage = Vec::with_capacity(
        node_id
            .len()
            .saturating_add(property.len())
            .saturating_add(content_id.len())
            .saturating_add(24),
    );
    for field in [
        node_id.as_bytes(),
        property.as_bytes(),
        content_id.as_bytes(),
    ] {
        preimage.extend_from_slice(&(field.len() as u64).to_le_bytes());
        preimage.extend_from_slice(field);
    }
    format!("vector-property:{}", Hash::of(&preimage).to_prefixed())
}

/// Resolve and bit-exactly decode one vector property.
pub fn read_vector_property(
    node: &NodeRecord,
    property: &str,
    store: &(impl TensorBlockPayloadStore + ?Sized),
) -> GraphStoreResult<Option<Vec<f32>>> {
    let Some((reference, bytes)) = read_vector_property_packed(node, property, store)? else {
        return Ok(None);
    };
    unpack_vector(&bytes, reference.dimension).map(Some)
}

/// Resolve and verify the canonical packed bytes without decoding them into a
/// second vector allocation. The designated index moves this buffer directly
/// into its packed exact-search representation.
pub(crate) fn read_vector_property_packed(
    node: &NodeRecord,
    property: &str,
    store: &(impl TensorBlockPayloadStore + ?Sized),
) -> GraphStoreResult<Option<(VectorPropertyRef, Vec<u8>)>> {
    let property = property.trim();
    if property.is_empty() {
        return Err(GraphStoreError::new(
            "vector_property_name_invalid",
            "vector property name must be non-empty",
        ));
    }
    let Some(value) = node.properties.get(property) else {
        return Ok(None);
    };
    let reference = VectorPropertyRef::from_value(value)?;
    let bytes = store
        .get_tensor_block_bytes(&reference.content_id)?
        .ok_or_else(|| {
            GraphStoreError::new(
                "vector_payload_missing",
                format!("vector payload {} is missing", reference.content_id),
            )
        })?;
    let actual = Hash::of(&bytes).to_prefixed();
    if actual != reference.content_id {
        return Err(GraphStoreError::new(
            "vector_payload_address_mismatch",
            format!(
                "vector payload bytes hash to {actual}, expected {}",
                reference.content_id
            ),
        ));
    }
    validate_packed_vector_len(&bytes, reference.dimension)?;
    Ok(Some((reference, bytes)))
}

pub fn decode_legacy_vector_array(value: &Value) -> GraphStoreResult<Option<Vec<f32>>> {
    let Some(values) = value.as_array() else {
        return Ok(None);
    };
    let mut vector = Vec::with_capacity(values.len());
    for value in values {
        let number = value.as_f64().ok_or_else(|| {
            GraphStoreError::new(
                "legacy_vector_property_invalid",
                "designated legacy vector arrays must contain only JSON numbers",
            )
        })?;
        let packed = number as f32;
        if !packed.is_finite() {
            return Err(GraphStoreError::new(
                "legacy_vector_property_invalid",
                "designated legacy vector value is outside the finite f32 range",
            ));
        }
        vector.push(packed);
    }
    Ok(Some(vector))
}

/// Collect vector payload addresses only at positions named by the durable
/// vector-designation registry. An object in such a position is necessarily a
/// vector reference and therefore decodes strictly; unrelated metadata is not
/// sniffed recursively.
pub(crate) fn vector_payload_references(
    node: &NodeRecord,
    designations: &[VectorDesignation],
) -> GraphStoreResult<Vec<Hash>> {
    let mut references = BTreeSet::new();
    for designation in designations {
        if !node.labels.iter().any(|label| label == &designation.label) {
            continue;
        }
        let Some(value) = node.properties.get(&designation.property) else {
            continue;
        };
        if value.is_array() {
            // Legacy inline vectors have no external payload to retain. The
            // recovery migration replaces these before the store is returned.
            continue;
        }
        let reference = VectorPropertyRef::from_value(value)?;
        if reference.dimension != designation.dimension {
            return Err(GraphStoreError::new(
                "dimension_mismatch",
                format!(
                    "designated property {} expects {} dimensions, reference declares {}",
                    designation.property, designation.dimension, reference.dimension
                ),
            ));
        }
        references.insert(Hash::parse_prefixed(&reference.content_id)?);
    }
    for (label, property) in BUILTIN_VECTOR_PAYLOAD_PROPERTIES {
        if !node.labels.iter().any(|candidate| candidate == label) {
            continue;
        }
        let Some(value) = node.properties.get(property) else {
            continue;
        };
        if value.is_array() {
            // The recovery/preflight migration owns legacy arrays. There is no
            // external payload to retain until that migration has run.
            continue;
        }
        let reference = VectorPropertyRef::from_value(value)?;
        references.insert(Hash::parse_prefixed(&reference.content_id)?);
    }
    Ok(references.into_iter().collect())
}

pub(crate) fn pack_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len().saturating_mul(4));
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn unpack_vector(bytes: &[u8], dimension: usize) -> GraphStoreResult<Vec<f32>> {
    validate_packed_vector_len(bytes, dimension)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte vector lane")))
        .collect())
}

fn validate_packed_vector_len(bytes: &[u8], dimension: usize) -> GraphStoreResult<()> {
    let expected_len = dimension.checked_mul(4).ok_or_else(|| {
        GraphStoreError::new(
            "vector_property_dimension_overflow",
            "vector dimension exceeds the addressable packed-byte length",
        )
    })?;
    if bytes.len() != expected_len {
        return Err(GraphStoreError::new(
            "vector_payload_length_mismatch",
            format!(
                "vector reference declares {dimension} dimensions ({expected_len} bytes), payload has {} bytes",
                bytes.len()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use super::*;
    use crate::tensor_derivation::InMemoryTensorBlockPayloadStore;

    #[derive(Default)]
    struct FaultingStagingStore {
        objects: Mutex<BTreeMap<String, Vec<u8>>>,
        bound: Mutex<BTreeMap<String, usize>>,
        release_attempts: AtomicUsize,
        fail_next_release: AtomicBool,
    }

    impl FaultingStagingStore {
        fn bound_count(&self) -> usize {
            self.bound
                .lock()
                .expect("fixture bound-pin lock")
                .values()
                .sum()
        }

        fn consume_one_as_graph_preflight(&self) {
            let mut bound = self.bound.lock().expect("fixture bound-pin lock");
            let key = bound.keys().next().expect("one staged receipt").clone();
            let count = bound.get_mut(&key).expect("selected receipt exists");
            *count -= 1;
            if *count == 0 {
                bound.remove(&key);
            }
        }
    }

    impl TensorBlockPayloadStore for FaultingStagingStore {
        fn put_tensor_block_bytes(
            &self,
            expected_content_id: &str,
            bytes: &[u8],
        ) -> GraphStoreResult<()> {
            self.objects
                .lock()
                .expect("fixture object lock")
                .insert(expected_content_id.to_string(), bytes.to_vec());
            Ok(())
        }

        fn get_tensor_block_bytes(&self, content_id: &str) -> GraphStoreResult<Option<Vec<u8>>> {
            Ok(self
                .objects
                .lock()
                .expect("fixture object lock")
                .get(content_id)
                .cloned())
        }

        fn bind_tensor_block_pending(
            &self,
            _content_id: &str,
            publication_key: &str,
        ) -> GraphStoreResult<()> {
            *self
                .bound
                .lock()
                .expect("fixture bound-pin lock")
                .entry(publication_key.to_string())
                .or_insert(0) += 1;
            Ok(())
        }

        fn release_bound_tensor_block_pending(
            &self,
            _content_id: &str,
            publication_key: &str,
        ) -> GraphStoreResult<bool> {
            self.release_attempts.fetch_add(1, Ordering::SeqCst);
            if self.fail_next_release.swap(false, Ordering::SeqCst) {
                return Err(GraphStoreError::new(
                    "fixture_pin_release_failure",
                    "fixture pending-pin release failure",
                ));
            }
            let mut bound = self.bound.lock().expect("fixture bound-pin lock");
            let Some(count) = bound.get_mut(publication_key) else {
                return Ok(false);
            };
            *count -= 1;
            if *count == 0 {
                bound.remove(publication_key);
            }
            Ok(true)
        }
    }

    #[test]
    fn vector_property_golden_bytes_and_bit_identity() -> GraphStoreResult<()> {
        let store = InMemoryTensorBlockPayloadStore::new();
        let vector = [1.0_f32, -0.0, 0.5];
        let mut node = NodeRecord::new("vector:golden", ["Doc"], json!({}));
        write_vector_property(&mut node, "embedding", &vector, &store)?;

        let reference = VectorPropertyRef::from_value(&node.properties["embedding"])?;
        assert_eq!(reference.dimension, 3);
        assert_eq!(
            reference.content_id,
            "sha256:6460c6cd7f2302b98c98ccf5628b1022812d337e5ca84b2839fe84694cf6656a"
        );
        assert_eq!(
            store
                .get_tensor_block_bytes(&reference.content_id)?
                .expect("golden payload"),
            vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x3f]
        );
        let decoded = read_vector_property(&node, "embedding", &store)?.expect("vector");
        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vector
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn vector_property_reference_is_exact_and_payloads_deduplicate() -> GraphStoreResult<()> {
        let store = InMemoryTensorBlockPayloadStore::new();
        let vector = [0.25_f32, 0.75];
        let mut first = NodeRecord::new("vector:first", ["Doc"], json!({}));
        let mut second = NodeRecord::new("vector:second", ["Doc"], json!({}));
        write_vector_property(&mut first, "embedding", &vector, &store)?;
        write_vector_property(&mut second, "embedding", &vector, &store)?;
        assert_eq!(
            first.properties["embedding"],
            second.properties["embedding"]
        );
        assert_eq!(store.object_count()?, 1);
        assert!(VectorPropertyRef::from_value(&json!({
            "content_id": first.properties["embedding"]["content_id"].clone(),
            "dimension": 2,
            "extra": true
        }))
        .is_err());
        Ok(())
    }

    #[test]
    fn staging_guard_attempts_all_receipts_after_partial_cleanup_failure() -> GraphStoreResult<()> {
        let store = Arc::new(FaultingStagingStore::default());
        let payloads: Arc<dyn TensorBlockPayloadStore> = store.clone();
        let mut guard = VectorPropertyStagingGuard::new(payloads);
        let vector = [0.25_f32, 0.75];
        let mut first = NodeRecord::new("vector:first", ["Doc"], json!({}));
        let mut second = NodeRecord::new("vector:second", ["Doc"], json!({}));
        guard.write_vector_property(&mut first, "embedding", &vector)?;
        guard.write_vector_property(&mut second, "embedding", &vector)?;
        assert_eq!(store.bound_count(), 2);

        store.fail_next_release.store(true, Ordering::SeqCst);
        let error = guard.cancel().expect_err("one release is faulted");
        assert_eq!(error.code, "vector_staging_pin_cleanup_failed");
        assert_eq!(store.release_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            store.bound_count(),
            1,
            "the later receipt must still be released after the first fault"
        );

        guard.cancel()?;
        assert_eq!(store.release_attempts.load(Ordering::SeqCst), 3);
        assert_eq!(store.bound_count(), 0);
        Ok(())
    }

    #[test]
    fn postpublication_cleanup_failure_is_retried_without_returning_failure() -> GraphStoreResult<()>
    {
        let store = Arc::new(FaultingStagingStore::default());
        let payloads: Arc<dyn TensorBlockPayloadStore> = store.clone();
        let mut guard = VectorPropertyStagingGuard::new(payloads);
        let mut node = NodeRecord::new("vector:postcommit-cleanup", ["Doc"], json!({}));
        guard.write_vector_property(&mut node, "embedding", &[0.25, 0.75])?;
        assert_eq!(store.bound_count(), 1);

        store.fail_next_release.store(true, Ordering::SeqCst);
        guard.finish_after_publication();
        assert_eq!(store.bound_count(), 1, "failed receipt is retained");
        drop(guard);
        assert_eq!(store.bound_count(), 0, "Drop retries the retained receipt");
        Ok(())
    }

    #[test]
    fn staging_guard_finish_releases_receipt_omitted_from_publication() -> GraphStoreResult<()> {
        let store = Arc::new(FaultingStagingStore::default());
        let payloads: Arc<dyn TensorBlockPayloadStore> = store.clone();
        let mut guard = VectorPropertyStagingGuard::new(payloads);
        let vector = [0.25_f32, 0.75];
        let mut published = NodeRecord::new("vector:published", ["Doc"], json!({}));
        let mut omitted = NodeRecord::new("vector:omitted", ["Doc"], json!({}));
        guard.write_vector_property(&mut published, "embedding", &vector)?;
        guard.write_vector_property(&mut omitted, "embedding", &vector)?;
        assert_eq!(store.bound_count(), 2);

        store.consume_one_as_graph_preflight();
        assert_eq!(store.bound_count(), 1);
        guard.finish_after_publication();
        assert_eq!(store.bound_count(), 0);
        Ok(())
    }
}
