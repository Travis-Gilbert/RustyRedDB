//! Read-only physical inventory. This is a diagnostic, never a migration owner.
//!
//! Unknown bytes and missing producer provenance remain undecidable. Numeric
//! arrays are candidates, not evidence that a property belongs to an embedding
//! model. No dimension, property name, or node label authorizes replacement.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::chunk_store::{ChunkStore, RootCatalog, ROOT_NAME_REDCORE};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;
use crate::graph::vector_index::VectorDesignation;
use crate::graph_store::decode_redcore_reachability_state;
use crate::hash::Hash;
use crate::journal::JournalChunkStore;
use crate::reachability::{walk_committed_reachable, CoreReachabilityCodec};
use crate::rr_file::RustyRed;
use crate::vector_payload::VectorPropertyRef;
use crate::versioned_graph::{
    collect_tree_entries, decode_canonical_graph_payload, GraphPayloadRecord, GraphProllyTree,
    ProllyReader, DOMAIN_GRAPH_PAYLOAD,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InventoryFileKind {
    BareJournal,
    RustyRed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PhysicalChunkObservation {
    pub address: Hash,
    pub payload_bytes: u64,
    /// None means typed reachability failed; false never implies safe deletion.
    pub committed_reachable: Option<bool>,
    pub interpretation: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorPropertyObservation {
    pub record_address: Hash,
    pub node_id: String,
    pub node_version: u64,
    pub tombstone: bool,
    pub labels: Vec<String>,
    pub current_record: bool,
    pub property_path: String,
    pub representation: String,
    pub dimension: usize,
    pub payload_address: Option<Hash>,
    pub payload_present: bool,
    pub finite: Option<bool>,
    pub l2_norm: Option<f64>,
    pub designated: bool,
    pub protection: String,
    /// Selected identity strings only. No input text or raw vector is exported.
    pub provenance: BTreeMap<String, String>,
    pub producer_identity_status: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NodeSchemaObservation {
    pub record_address: Hash,
    pub node_id: String,
    pub labels: Vec<String>,
    pub current_record: bool,
    pub tombstone: bool,
    pub property_types: BTreeMap<String, String>,
    pub properties_digest: Hash,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorRegistryObservation {
    pub state_address: Hash,
    pub current_registry: bool,
    pub designations: Vec<VectorDesignation>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SemanticInventory {
    pub schema: String,
    pub file_kind: InventoryFileKind,
    pub source_file_digest: Hash,
    pub source_file_bytes: u64,
    pub catalog_root: Option<Hash>,
    pub physical_catalog_complete: bool,
    pub typed_reachability_complete: bool,
    /// False whenever unknown bytes/identities remain; inventory is not approval.
    pub semantic_classification_complete: bool,
    pub named_roots: BTreeMap<String, Hash>,
    pub vector_designations: Vec<VectorDesignation>,
    pub physical_vector_registries: Vec<VectorRegistryObservation>,
    pub physical_chunks: Vec<PhysicalChunkObservation>,
    pub vector_properties: Vec<VectorPropertyObservation>,
    pub node_schemas: Vec<NodeSchemaObservation>,
    pub diagnostics: Vec<String>,
}

impl SemanticInventory {
    /// Canonical, relocation-independent migration input identity. All arrays
    /// and maps are sorted by durable identifiers, and no timestamp is injected.
    pub fn fingerprint(&self) -> GraphStoreResult<Hash> {
        encode(self).map(|bytes| Hash::of(&bytes))
    }
}

/// Does not acquire a writer claim or call `RedCoreGraphStore::open`.
/// A file changing during inspection is refused rather than reported complete.
pub fn inventory_semantic_file(
    path: &Path,
    kind: InventoryFileKind,
) -> GraphStoreResult<SemanticInventory> {
    let before = file_digest(path)?;
    let store = match kind {
        InventoryFileKind::BareJournal => Arc::new(JournalChunkStore::open_read_only(path)?),
        InventoryFileKind::RustyRed => Arc::clone(RustyRed::open_read_only(path)?.store()),
    };
    let committed = store.committed_root()?;
    let mut report = SemanticInventory {
        schema: "theorem.semantic.physical-inventory/1".to_owned(),
        file_kind: kind,
        source_file_digest: before.0,
        source_file_bytes: before.1,
        catalog_root: committed.as_ref().map(|root| root.root),
        physical_catalog_complete: true,
        typed_reachability_complete: false,
        semantic_classification_complete: false,
        named_roots: BTreeMap::new(),
        vector_designations: Vec::new(),
        physical_vector_registries: Vec::new(),
        physical_chunks: Vec::new(),
        vector_properties: Vec::new(),
        node_schemas: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut current_records = BTreeSet::new();
    if let Some(root) = report.catalog_root {
        let catalog = RootCatalog::from_bytes(&required(store.as_ref(), root)?)?;
        report.named_roots = catalog.roots().clone();
        if let Some(state_address) = catalog.get(ROOT_NAME_REDCORE) {
            let (_, payload_root, _, designations) = decode_redcore_reachability_state(
                state_address,
                &required(store.as_ref(), state_address)?,
            )?;
            report.vector_designations = designations;
            let mut reader = ProllyReader::new(store.as_ref());
            let tree = GraphProllyTree {
                root: payload_root,
                entries_total: reader.read(payload_root)?.entries_total(),
            };
            current_records.extend(
                collect_tree_entries(store.as_ref(), &tree)?
                    .into_iter()
                    .map(|entry| entry.object_hash),
            );
        }
    }
    report.vector_designations.sort_by(|a, b| {
        (&a.label, &a.property, a.dimension).cmp(&(&b.label, &b.property, b.dimension))
    });
    let reachable = if report.catalog_root.is_some() {
        match walk_committed_reachable(store.as_ref(), &CoreReachabilityCodec) {
            Ok(walk) => {
                report.typed_reachability_complete = true;
                Some(walk.reachable)
            }
            Err(error) => {
                report
                    .diagnostics
                    .push(format!("typed_reachability:{}", error.code));
                None
            }
        }
    } else {
        report.typed_reachability_complete = true;
        Some(BTreeSet::new())
    };
    let physical_chunks = match store.physical_chunks() {
        Ok(chunks) => chunks,
        Err(error) if error.code.starts_with("chunk_store_inventory_") => {
            report.physical_catalog_complete = false;
            report.diagnostics.push(format!("physical_catalog:{}", error.code));
            store.observed_physical_chunks()?
        }
        Err(error) => return Err(error),
    };
    for (address, payload_bytes) in physical_chunks {
        let bytes = required(store.as_ref(), address)?;
        let mut interpretation = "opaque_protected_undecidable".to_owned();
        // Diagnostic candidate sniffing is explicitly untrusted. Only a full
        // canonical decoder can label the chunk as a graph payload. An opaque
        // payload sharing this first byte stays opaque, with a diagnostic.
        if let Ok((_, _, _, mut designations)) = decode_redcore_reachability_state(address, &bytes)
        {
            designations.sort_by(|a, b| {
                (&a.label, &a.property, a.dimension).cmp(&(&b.label, &b.property, b.dimension))
            });
            report
                .physical_vector_registries
                .push(VectorRegistryObservation {
                    state_address: address,
                    current_registry: report.named_roots.get(ROOT_NAME_REDCORE) == Some(&address),
                    designations,
                });
            interpretation = "canonical_redcore_registry".to_owned();
        }
        if bytes.first() == Some(&DOMAIN_GRAPH_PAYLOAD) {
            match decode_canonical_graph_payload(address, &bytes) {
                Ok(GraphPayloadRecord::Node(node)) => {
                    interpretation = "canonical_node_payload".to_owned();
                    inventory_node(
                        &mut report,
                        store.as_ref(),
                        address,
                        &node,
                        current_records.contains(&address),
                    )?;
                }
                Ok(GraphPayloadRecord::Edge(_)) => {
                    interpretation = "canonical_edge_payload".to_owned()
                }
                Err(error) => report
                    .diagnostics
                    .push(format!("opaque_candidate:{address}:{}", error.code)),
            }
        }
        report.physical_chunks.push(PhysicalChunkObservation {
            address,
            payload_bytes,
            committed_reachable: reachable.as_ref().map(|set| set.contains(&address)),
            interpretation,
        });
    }
    report.vector_properties.sort_by(|a, b| {
        (a.record_address, &a.property_path).cmp(&(b.record_address, &b.property_path))
    });
    report.node_schemas.sort_by_key(|item| item.record_address);
    report.diagnostics.sort();
    report.diagnostics.dedup();
    if file_digest(path)? != before {
        return Err(GraphStoreError::new(
            "semantic_inventory_source_changed",
            "source journal changed while the read-only inventory ran; retry a quiescent snapshot",
        ));
    }
    Ok(report)
}

fn inventory_node(
    report: &mut SemanticInventory,
    store: &dyn ChunkStore,
    address: Hash,
    node: &NodeRecord,
    current: bool,
) -> GraphStoreResult<()> {
    let Some(properties) = node.properties.as_object() else {
        report
            .diagnostics
            .push(format!("non_object_node_properties:{address}"));
        return Ok(());
    };
    report.node_schemas.push(NodeSchemaObservation {
        record_address: address,
        node_id: node.id.clone(),
        labels: node.labels.clone(),
        current_record: current,
        tombstone: node.tombstone,
        property_types: properties
            .iter()
            .map(|(key, value)| (key.clone(), value_kind(value).to_owned()))
            .collect(),
        properties_digest: Hash::of(&encode(&node.properties)?),
    });
    let provenance = identity_fields(node);
    for (key, value) in properties {
        inspect_property(
            report,
            store,
            address,
            node,
            current,
            &format!("/{}", pointer_token(key)),
            value,
            &provenance,
        )?;
    }
    Ok(())
}

fn inspect_property(
    report: &mut SemanticInventory,
    store: &dyn ChunkStore,
    address: Hash,
    node: &NodeRecord,
    current: bool,
    path: &str,
    value: &Value,
    provenance: &BTreeMap<String, String>,
) -> GraphStoreResult<()> {
    if value
        .as_object()
        .is_some_and(|fields| fields.contains_key("content_id") && fields.contains_key("dimension"))
        && VectorPropertyRef::from_value(value).is_err()
    {
        report
            .diagnostics
            .push(format!("malformed_vector_candidate:{address}:{path}"));
    }
    let reference = VectorPropertyRef::from_value(value).ok();
    let array = value
        .as_array()
        .filter(|values| !values.is_empty() && values.iter().all(Value::is_number));
    if reference.is_some() || array.is_some() {
        let (representation, dimension, payload_address, payload_present, finite, norm) =
            if let Some(reference) = reference {
                let payload_address = Hash::parse_prefixed(&reference.content_id)?;
                match store.get(payload_address)? {
                    Some(bytes) if bytes.len() == reference.dimension.saturating_mul(4) => {
                        let values = bytes.chunks_exact(4).map(|chunk| {
                            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64
                        });
                        let (finite, norm) = numeric_summary(values);
                        (
                            "packed_le_f32",
                            reference.dimension,
                            Some(payload_address),
                            true,
                            Some(finite),
                            norm,
                        )
                    }
                    Some(_) => {
                        report
                            .diagnostics
                            .push(format!("vector_length_mismatch:{address}:{path}"));
                        (
                            "packed_le_f32",
                            reference.dimension,
                            Some(payload_address),
                            true,
                            None,
                            None,
                        )
                    }
                    None => {
                        report
                            .diagnostics
                            .push(format!("vector_payload_missing:{address}:{path}"));
                        (
                            "packed_le_f32",
                            reference.dimension,
                            Some(payload_address),
                            false,
                            None,
                            None,
                        )
                    }
                }
            } else {
                let values = array.expect("candidate array checked");
                let (finite, norm) =
                    numeric_summary(values.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)));
                (
                    "inline_numeric_candidate",
                    values.len(),
                    None,
                    true,
                    Some(finite),
                    norm,
                )
            };
        let designated = report.vector_designations.iter().any(|designation| {
            node.labels.contains(&designation.label)
                && format!("/{}", pointer_token(&designation.property)) == path
        });
        let protection = if node.labels.iter().any(|label| {
            matches!(
                label.as_str(),
                "EpistemicShadow" | "CalibrationEpoch" | "ModelWeights"
            )
        }) || path.contains("structural_role")
            || path.contains("weight")
            || path.contains("centroid")
        {
            "protected_structural_or_model_state"
        } else {
            "undecidable_requires_schema_and_producer_review"
        };
        // Provenance is descriptive and property-specific. A receipt elsewhere
        // on the node cannot establish ownership of this property.
        let property_provenance = property_identity_fields(node, path, provenance);
        report.vector_properties.push(VectorPropertyObservation {
            record_address: address,
            node_id: node.id.clone(),
            node_version: node.version,
            tombstone: node.tombstone,
            labels: node.labels.clone(),
            current_record: current,
            property_path: path.to_owned(),
            representation: representation.to_owned(),
            dimension,
            payload_address,
            payload_present,
            finite,
            l2_norm: norm,
            designated,
            protection: protection.to_owned(),
            producer_identity_status: if [
                "producer_semantic_space_id",
                "producer_checkpoint_digest",
                "producer_revision",
                "task_profile_id",
                "input_processing_profile_id",
            ]
            .iter()
            .all(|field| {
                property_provenance
                    .get(*field)
                    .is_some_and(|value| !value.is_empty())
            }) || [
                "semantic_space_id",
                "checkpoint_digest",
                "query_task_profile",
                "corpus_task_profile",
                "processing_profile_id",
                "physical_vector_property",
            ]
            .iter()
            .all(|field| {
                property_provenance
                    .get(*field)
                    .is_some_and(|value| !value.is_empty())
            }) {
                "declared_unverified".to_owned()
            } else {
                "missing_undecidable".to_owned()
            },
            provenance: property_provenance,
        });
        return Ok(());
    }
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                inspect_property(
                    report,
                    store,
                    address,
                    node,
                    current,
                    &format!("{path}/{}", pointer_token(key)),
                    child,
                    provenance,
                )?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                inspect_property(
                    report,
                    store,
                    address,
                    node,
                    current,
                    &format!("{path}/{index}"),
                    child,
                    provenance,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn pointer_token(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn identity_fields(node: &NodeRecord) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for field in [
        "embedding_space_id",
        "embedding_role",
        "semantic_generation_id",
    ] {
        if let Some(value) = node.properties.get(field).and_then(Value::as_str) {
            result.insert(field.to_owned(), value.to_owned());
        }
    }
    let Some(provenance) = node
        .properties
        .pointer("/extra/semantic_embedding_provenance")
    else {
        return result;
    };
    for field in [
        "producer_semantic_space_id",
        "producer_checkpoint_revision",
        "producer_checkpoint_digest",
        "producer_revision",
        "task_profile_id",
        "input_processing_profile_id",
        "loaded_modalities",
        "source_input_digest",
        "vector_digest",
        "source_ref",
        "source_revision",
        "tenant_id",
        "schema",
    ] {
        if let Some(value) = provenance.get(field).and_then(Value::as_str) {
            result.insert(field.to_owned(), value.to_owned());
        }
    }
    result
}
/// Producer metadata is observational and must name this exact physical
/// property. Never transfer a node's Gemma identity to a legacy vector field.
fn property_identity_fields(
    node: &NodeRecord,
    path: &str,
    som_provenance: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut result = if path == "/semantic_vec_eg2_256" {
        som_provenance.clone()
    } else {
        BTreeMap::new()
    };
    let Some(encoded_property) = path.strip_prefix('/') else {
        return result;
    };
    if encoded_property.contains('/') {
        return result;
    }
    let property = encoded_property.replace("~1", "/").replace("~0", "~");
    let identity_key = format!("{property}_identity");
    let extraction = node
        .properties
        .get("extraction")
        .and_then(|value| value.get("vector_identities"))
        .and_then(|value| value.get(&property));
    let identity = node
        .properties
        .get(&identity_key)
        .or_else(|| extraction.map(|value| value.get("semantic_identity").unwrap_or(value)))
        .or_else(|| {
            node.labels
                .iter()
                .any(|label| label == "SemanticMigrationProgress")
                .then(|| node.properties.get("semantic_identity"))
                .flatten()
        });
    if let Some(identity) = identity.filter(|value| {
        value
            .get("physical_vector_property")
            .and_then(Value::as_str)
            == Some(property.as_str())
    }) {
        for field in [
            "semantic_space_id",
            "checkpoint_digest",
            "query_task_profile",
            "corpus_task_profile",
            "processing_profile_id",
            "physical_vector_property",
        ] {
            if let Some(value) = identity.get(field).and_then(Value::as_str) {
                result.insert(field.to_owned(), value.to_owned());
            }
        }
        for field in ["model", "dimension", "metric", "normalized"] {
            if let Some(value) = node
                .properties
                .get(format!("{property}_{field}"))
                .filter(|value| !value.is_object() && !value.is_array())
            {
                result.insert(format!("declared_{field}"), value.to_string());
            }
        }
        if node
            .labels
            .iter()
            .any(|label| label == "SemanticMigrationProgress")
        {
            for field in ["model_id", "vector_digest"] {
                if let Some(value) = node.properties.get(field).and_then(Value::as_str) {
                    result.insert(format!("declared_{field}"), value.to_owned());
                }
            }
        }
        for metadata in [
            node.properties.get(format!("{property}_source")),
            extraction,
            node.labels
                .iter()
                .any(|label| label == "SemanticMigrationProgress")
                .then(|| node.properties.get("source"))
                .flatten(),
        ] {
            if let Some(metadata) = metadata {
                for field in [
                    "source_ref",
                    "source_revision",
                    "input_digest",
                    "tenant_id",
                    "node_id",
                    "node_version",
                    "property",
                    "model_id",
                    "producer_revision",
                    "source_id",
                    "properties_digest",
                ] {
                    if let Some(value) = metadata
                        .get(field)
                        .filter(|value| value.is_string() || value.is_number())
                    {
                        result.insert(format!("declared_source_{field}"), value.to_string());
                    }
                }
            }
        }
    }
    result
}

fn numeric_summary(values: impl Iterator<Item = f64>) -> (bool, Option<f64>) {
    let mut finite = true;
    let mut squares = 0.0;
    for value in values {
        finite &= value.is_finite();
        squares += value * value;
    }
    let norm = squares.sqrt();
    (finite, (finite && norm.is_finite()).then_some(norm))
}
fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
fn required(store: &dyn ChunkStore, address: Hash) -> GraphStoreResult<Arc<[u8]>> {
    store.get(address)?.ok_or_else(|| {
        GraphStoreError::new("semantic_inventory_chunk_missing", address.to_string())
    })
}
fn encode(value: &impl Serialize) -> GraphStoreResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|error| GraphStoreError::new("semantic_inventory_encode", error.to_string()))
}
fn file_digest(path: &Path) -> GraphStoreResult<(Hash, u64)> {
    let mut file = File::open(path).map_err(io_error)?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        total += count as u64;
    }
    Ok((Hash::from_bytes(digest.finalize().into()), total))
}
fn io_error(error: std::io::Error) -> GraphStoreError {
    GraphStoreError::new("semantic_inventory_io", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioned_graph::canonical_graph_payload_preimage;
    use crate::{RedCoreGraphStore, RedCoreOptions};
    use serde_json::json;

    #[test]
    fn consumer_identities_are_property_scoped_and_remain_unverified() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity-fixture.journal");
        let writer = JournalChunkStore::open(&path).unwrap();
        let identity = |property: &str| {
            json!({
                "semantic_space_id":"declared-fixture-space", "checkpoint_digest":"a".repeat(64),
                "query_task_profile":"search-query/1", "corpus_task_profile":"document/1",
                "processing_profile_id":"declared-fixture-profile", "physical_vector_property":property,
            })
        };
        let node = NodeRecord::new(
            "consumer-identity-fixture",
            ["Page"],
            json!({
                "semantic_vec_eg2_256":[1.0,0.0],
                "semantic_vec_eg2_256_identity": identity("semantic_vec_eg2_256"),
                "semantic_vec_eg2_256_dimension":256,
                "semantic_vec_eg2_256_source":{"source_ref":"fixture-page", "source_revision":"v1", "text":"do not disclose private text"},
                "life_vector":[0.0,1.0],
                "extraction":{"vector_identities":{"life_vector":{
                    "semantic_space_id":"declared-fixture-space", "checkpoint_digest":"a".repeat(64),
                "query_task_profile":"search-query/1", "corpus_task_profile":"document/1",
                "processing_profile_id":"declared-fixture-profile", "physical_vector_property":"life_vector",
                "source_ref":"fixture-life", "source_revision":"v2"
                }}},
                "legacy_embedding":[1.0,0.0],
                "legacy_embedding_identity":identity("different_property")
            }),
        );
        let payload = canonical_graph_payload_preimage(&GraphPayloadRecord::Node(node)).unwrap();
        writer.write(&[(Hash::of(&payload), &payload)]).unwrap();
        let before = std::fs::read(&path).unwrap();
        let report = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        assert_eq!(before, std::fs::read(&path).unwrap());
        let field = |path: &str| {
            report
                .vector_properties
                .iter()
                .find(|field| field.property_path == path)
                .unwrap()
        };
        assert_eq!(field("/semantic_vec_eg2_256").dimension, 2); // declared dimension never overrides observed bytes
        assert_eq!(
            field("/semantic_vec_eg2_256").producer_identity_status,
            "declared_unverified"
        );
        assert_eq!(
            field("/life_vector").producer_identity_status,
            "declared_unverified"
        );
        assert_eq!(
            field("/legacy_embedding").producer_identity_status,
            "missing_undecidable"
        );
        assert!(field("/legacy_embedding").provenance.is_empty());
        assert!(!String::from_utf8(encode(&report).unwrap())
            .unwrap()
            .contains("do not disclose private text"));
        assert!(!report.semantic_classification_complete);
    }

    #[test]
    fn physical_inventory_covers_orphans_and_nondesignated_vectors_without_writes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source.journal");
        let writer = JournalChunkStore::open(&path).unwrap();
        let packed = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat();
        let packed_address = Hash::of(&packed);
        let node = NodeRecord::new(
            "fixture-only/orphan",
            vec!["EpistemicShadow".to_owned()],
            json!({
                "structural_role_vector": {"content_id":packed_address.to_prefixed(),"dimension":2},
                "unknown_numeric_array":[0.0,1.0], "text":"private text must not appear",
                "slash/key":[0.0,1.0], "slash":{"key":[0.0,1.0]}
            }),
        );
        let payload = canonical_graph_payload_preimage(&GraphPayloadRecord::Node(node)).unwrap();
        let unknown = b"unreferenced unknown physical bytes";
        writer
            .write(&[
                (packed_address, &packed),
                (Hash::of(&payload), &payload),
                (Hash::of(unknown), unknown),
            ])
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let lock_path = path.with_extension("lock");
        let lock_before = lock_path
            .exists()
            .then(|| std::fs::read(&lock_path).unwrap());
        // The read-only diagnostic can operate while the writer claim is held.
        let first = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        let second = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());
        assert_eq!(before, std::fs::read(&path).unwrap());
        assert_eq!(
            lock_before,
            lock_path
                .exists()
                .then(|| std::fs::read(&lock_path).unwrap())
        );
        assert_eq!(first.physical_chunks.len(), 3);
        assert!(first
            .physical_chunks
            .iter()
            .all(|chunk| chunk.committed_reachable == Some(false)));
        assert_eq!(first.vector_properties.len(), 4);
        assert!(first
            .vector_properties
            .iter()
            .any(|v| v.property_path == "/slash~1key"));
        assert!(first
            .vector_properties
            .iter()
            .any(|v| v.property_path == "/slash/key"));
        assert!(first
            .vector_properties
            .iter()
            .all(|vector| !vector.designated && !vector.current_record));
        assert!(first
            .vector_properties
            .iter()
            .all(|vector| vector.producer_identity_status == "missing_undecidable"));
        assert!(first
            .vector_properties
            .iter()
            .all(|vector| vector.protection == "protected_structural_or_model_state"));
        assert_eq!(
            first
                .vector_properties
                .iter()
                .find(|v| v.property_path == "/structural_role_vector")
                .unwrap()
                .l2_norm,
            Some(1.0)
        );
        assert!(!first.semantic_classification_complete);
        assert!(!String::from_utf8(encode(&first).unwrap())
            .unwrap()
            .contains("private text must not appear"));
        drop(writer);
        // Deleting an unused fixture writer lock makes lock recreation observable.
        if lock_path.exists() {
            std::fs::remove_file(&lock_path).unwrap();
        }
        inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        assert!(!lock_path.exists());
    }

    #[test]
    fn physical_inventory_reads_current_registry_and_preserves_legacy_array_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let mut writer =
            RedCoreGraphStore::open(directory.path(), RedCoreOptions::default()).unwrap();
        writer
            .designate_vector_property("Item", "legacy_vec", 2)
            .unwrap();
        writer
            .upsert_node(NodeRecord::new(
                "fixture/current",
                vec!["Item".to_owned()],
                json!({
                    "legacy_vec":[1.0,0.0],"other_vec":[0.0,1.0],"title":"fixture"
                }),
            ))
            .unwrap();
        let path = directory.path().join("redcore.journal");
        let before = std::fs::read(&path).unwrap();
        let inventory = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        assert_eq!(before, std::fs::read(&path).unwrap());
        assert_eq!(inventory.vector_designations.len(), 1);
        assert!(inventory
            .physical_vector_registries
            .iter()
            .any(|registry| registry.current_registry));
        assert!(inventory.typed_reachability_complete);
        let current: Vec<_> = inventory
            .vector_properties
            .iter()
            .filter(|v| v.current_record)
            .collect();
        assert_eq!(current.len(), 2);
        assert!(
            current
                .iter()
                .find(|v| v.property_path == "/legacy_vec")
                .unwrap()
                .designated
        );
        assert!(
            !current
                .iter()
                .find(|v| v.property_path == "/other_vec")
                .unwrap()
                .designated
        );
        assert_eq!(
            inventory
                .node_schemas
                .iter()
                .filter(|s| s.current_record && s.node_id == "fixture/current")
                .count(),
            1
        );
    }

    #[test]
    fn declared_gemma_provenance_cannot_attribute_a_legacy_vector_property() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mixed.journal");
        let writer = JournalChunkStore::open(&path).unwrap();
        let node = NodeRecord::new(
            "fixture/mixed",
            ["Item"],
            json!({
                "embedding": [1.0, 0.0], "semantic_vec_eg2_256": [0.0, 1.0],
                "embedding_space_id": "declared-fixture-som-space", "embedding_role": "Document",
                "extra": {"semantic_embedding_provenance": {
                    "producer_semantic_space_id": "declared-fixture-space",
                    "producer_checkpoint_digest": "declared-fixture-digest",
                    "producer_revision": "declared-fixture-revision",
                    "task_profile_id": "document/1",
                    "input_processing_profile_id": "declared-fixture-profile"
                }}
            }),
        );
        let bytes = canonical_graph_payload_preimage(&GraphPayloadRecord::Node(node)).unwrap();
        writer.write(&[(Hash::of(&bytes), &bytes)]).unwrap();
        let report = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        let legacy = report
            .vector_properties
            .iter()
            .find(|v| v.property_path == "/embedding")
            .unwrap();
        let declared = report
            .vector_properties
            .iter()
            .find(|v| v.property_path == "/semantic_vec_eg2_256")
            .unwrap();
        assert!(legacy.provenance.is_empty());
        assert_eq!(legacy.producer_identity_status, "missing_undecidable");
        assert_eq!(declared.producer_identity_status, "declared_unverified");
        assert_eq!(
            declared
                .provenance
                .get("embedding_role")
                .map(String::as_str),
            Some("Document")
        );
        assert!(!report.semantic_classification_complete);
    }

    #[test]
    fn native_rr_inventory_does_not_change_header_or_writer_claim() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.RR");
        let writer = RustyRed::create(
            &path,
            &theorem_surface_contracts::desktop::DeviceId::new("inventory fixture writer"),
        )
        .unwrap();
        writer
            .store()
            .write(&[(Hash::of(b"uncommitted fixture"), b"uncommitted fixture")])
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let entries_before: BTreeSet<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let report = inventory_semantic_file(&path, InventoryFileKind::RustyRed).unwrap();
        assert_eq!(report.file_kind, InventoryFileKind::RustyRed);
        assert_eq!(report.physical_chunks.len(), 1);
        assert_eq!(before, std::fs::read(&path).unwrap());
        assert_eq!(
            entries_before,
            std::fs::read_dir(directory.path())
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect()
        );
        assert!(writer.is_writer());
    }

    #[test]
    fn read_only_inventory_marks_torn_tail_incomplete_without_repair_or_owner_claim() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("torn.journal");
        let writer = JournalChunkStore::open(&path).unwrap();
        writer.write(&[(Hash::of(b"fixture"), b"fixture")]).unwrap();
        drop(writer);
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0xff, 0x01])
            .unwrap();
        let before = std::fs::read(&path).unwrap();
        let entries_before: BTreeSet<_> = std::fs::read_dir(directory.path())
            .unwrap().map(|entry| entry.unwrap().file_name()).collect();
        let report = inventory_semantic_file(&path, InventoryFileKind::BareJournal).unwrap();
        assert!(!report.physical_catalog_complete);
        assert!(!report.semantic_classification_complete);
        assert_eq!(report.physical_chunks.len(), 1);
        assert!(report.diagnostics.iter().any(|reason|
            reason == "physical_catalog:chunk_store_inventory_torn_tail"));
        assert_eq!(before, std::fs::read(&path).unwrap());
        assert_eq!(entries_before, std::fs::read_dir(directory.path())
            .unwrap().map(|entry| entry.unwrap().file_name()).collect());
        // Secondary readers can still consume the safe prefix. Inventory must
        // neither repair the tail nor turn that prefix into complete coverage.
        let reader = JournalChunkStore::open_read_only(&path).unwrap();
        assert!(reader.get(Hash::of(b"fixture")).unwrap().is_some());
        assert_eq!(reader.physical_chunks().unwrap_err().code,
            "chunk_store_inventory_torn_tail");
        let relocated_directory = tempfile::tempdir().unwrap();
        let relocated = relocated_directory.path().join("relocated.journal");
        std::fs::write(&relocated, &before).unwrap();
        let moved_report = inventory_semantic_file(&relocated, InventoryFileKind::BareJournal).unwrap();
        assert_eq!(report.fingerprint().unwrap(), moved_report.fingerprint().unwrap());
        assert_eq!(before, std::fs::read(&relocated).unwrap());
    }
}
