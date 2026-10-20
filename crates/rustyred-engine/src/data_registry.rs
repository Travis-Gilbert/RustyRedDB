//! Canonical declarative DataRegistry records.
//!
//! The registry is intentionally a graph contract, not an in-memory catalog.
//! Source manifests and publication receipts are content addressed and
//! immutable; a stable tenant/source entry points at exactly one manifest.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::{stable_hash, EdgeRecord, NodeRecord};

pub const DATA_REGISTRY_ENTRY_LABEL: &str = "DataRegistryEntry";
pub const DATA_SOURCE_MANIFEST_LABEL: &str = "DataSourceManifest";
pub const DATA_REGISTRY_RECEIPT_LABEL: &str = "DataRegistryPublicationReceipt";
pub const DATA_REGISTRY_CONFLICT_LABEL: &str = "DataRegistryConflict";
pub const DATA_REGISTRY_HAS_MANIFEST: &str = "HAS_DATA_SOURCE_MANIFEST";
pub const DATA_REGISTRY_PUBLISHED_MANIFEST: &str = "PUBLISHED_DATA_SOURCE_MANIFEST";
pub const DATA_REGISTRY_CONFLICTS_WITH: &str = "CONFLICTS_WITH_DATA_REGISTRY";

pub const MAX_DATA_REGISTRY_SOURCE_ID_BYTES: usize = 256;
pub const MAX_DATA_REGISTRY_LOCATOR_BYTES: usize = 2_048;
pub const MAX_DATA_REGISTRY_COLLECTIONS: usize = 128;
pub const MAX_DATA_REGISTRY_OBLIGATIONS: usize = 128;
pub const MAX_DATA_REGISTRY_METADATA_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataRegistryContractError {
    pub code: &'static str,
    pub message: String,
}

impl DataRegistryContractError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

fn hash_contract_value(value: impl Serialize) -> Result<String, DataRegistryContractError> {
    stable_hash(value).map_err(|error| {
        DataRegistryContractError::new(
            "data_registry_hash_failed",
            format!("{}: {}", error.code, error.message),
        )
    })
}

impl fmt::Display for DataRegistryContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for DataRegistryContractError {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRegistryObligationStatus {
    Unknown,
    Open,
    Satisfied,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DataRegistryObligation {
    pub obligation_id: String,
    pub status: DataRegistryObligationStatus,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DataSourceManifestInput {
    pub source_id: String,
    pub source_kind: String,
    pub locator: String,
    pub content_anchor: String,
    pub schema_version: String,
    #[serde(default)]
    pub collections: Vec<String>,
    #[serde(default)]
    pub obligations: Vec<DataRegistryObligation>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl DataSourceManifestInput {
    pub fn normalize(mut self) -> Result<Self, DataRegistryContractError> {
        self.source_id = required_bounded(
            "source_id",
            self.source_id,
            MAX_DATA_REGISTRY_SOURCE_ID_BYTES,
        )?;
        self.source_kind = required_bounded("source_kind", self.source_kind, 128)?;
        self.locator = required_bounded("locator", self.locator, MAX_DATA_REGISTRY_LOCATOR_BYTES)?;
        self.schema_version = required_bounded("schema_version", self.schema_version, 128)?;
        self.content_anchor = self.content_anchor.trim().to_ascii_lowercase();
        if !is_sha256_anchor(&self.content_anchor) {
            return Err(DataRegistryContractError::new(
                "data_registry_content_anchor_invalid",
                "content_anchor must be sha256:<64 lowercase hexadecimal characters>",
            ));
        }
        self.collections = normalize_unique(self.collections, MAX_DATA_REGISTRY_COLLECTIONS, 256)?;
        if self.obligations.len() > MAX_DATA_REGISTRY_OBLIGATIONS {
            return Err(DataRegistryContractError::new(
                "data_registry_obligations_limit",
                format!("obligations exceeds the {MAX_DATA_REGISTRY_OBLIGATIONS}-item limit"),
            ));
        }
        for obligation in &mut self.obligations {
            obligation.obligation_id =
                required_bounded("obligation_id", obligation.obligation_id.clone(), 256)?;
            obligation.detail =
                required_bounded("obligation detail", obligation.detail.clone(), 2_048)?;
        }
        self.obligations.sort();
        if self
            .obligations
            .windows(2)
            .any(|pair| pair[0].obligation_id == pair[1].obligation_id)
        {
            return Err(DataRegistryContractError::new(
                "data_registry_obligation_duplicate",
                "obligation_id values must be unique within a source manifest",
            ));
        }
        let metadata_bytes = serde_json::to_vec(&self.metadata)
            .map_err(|error| {
                DataRegistryContractError::new(
                    "data_registry_metadata_invalid",
                    format!("metadata could not be encoded: {error}"),
                )
            })?
            .len();
        if metadata_bytes > MAX_DATA_REGISTRY_METADATA_BYTES {
            return Err(DataRegistryContractError::new(
                "data_registry_metadata_limit",
                format!("metadata exceeds the {MAX_DATA_REGISTRY_METADATA_BYTES}-byte limit"),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataRegistryIdentityBinding {
    pub principal_id: String,
    pub actor: String,
    pub binding_id: String,
    pub project_id: Option<String>,
    pub project_slug: Option<String>,
    pub resolution_path: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DataSourceManifest {
    pub manifest_id: String,
    pub manifest_anchor: String,
    pub tenant_id: String,
    #[serde(flatten)]
    pub input: DataSourceManifestInput,
}

impl DataSourceManifest {
    pub fn compile(
        tenant: &str,
        input: DataSourceManifestInput,
    ) -> Result<Self, DataRegistryContractError> {
        let tenant = required_bounded("tenant", tenant.to_string(), 256)?;
        let input = input.normalize()?;
        let manifest_anchor = hash_contract_value(json!({
            "contract": "theorem.data-registry.manifest.v1",
            "tenant_id": tenant,
            "input": input,
        }))?;
        Ok(Self {
            manifest_id: format!("data:source-manifest:{manifest_anchor}"),
            manifest_anchor,
            tenant_id: tenant,
            input,
        })
    }

    pub fn unknown_obligations(&self) -> Vec<DataRegistryObligation> {
        self.input
            .obligations
            .iter()
            .filter(|obligation| {
                matches!(
                    obligation.status,
                    DataRegistryObligationStatus::Unknown | DataRegistryObligationStatus::Open
                )
            })
            .cloned()
            .collect()
    }

    pub fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            &self.manifest_id,
            [DATA_SOURCE_MANIFEST_LABEL],
            serde_json::to_value(self).expect("data source manifest is serializable"),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, DataRegistryContractError> {
        decode_labeled(node, DATA_SOURCE_MANIFEST_LABEL, "data source manifest")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataRegistryPublicationReceipt {
    pub receipt_id: String,
    pub receipt_anchor: String,
    pub tenant_id: String,
    pub source_id: String,
    pub registry_id: String,
    pub manifest_id: String,
    pub manifest_anchor: String,
    pub graph_version_before: u64,
    pub identity: DataRegistryIdentityBinding,
    pub status: String,
}

impl DataRegistryPublicationReceipt {
    pub fn compile(
        manifest: &DataSourceManifest,
        registry_id: &str,
        identity: DataRegistryIdentityBinding,
        graph_version_before: u64,
    ) -> Result<Self, DataRegistryContractError> {
        let receipt_anchor = hash_contract_value(json!({
            "contract": "theorem.data-registry.publication-receipt.v1",
            "tenant_id": manifest.tenant_id,
            "source_id": manifest.input.source_id,
            "registry_id": registry_id,
            "manifest_id": manifest.manifest_id,
            "manifest_anchor": manifest.manifest_anchor,
            "graph_version_before": graph_version_before,
            "identity": identity,
        }))?;
        Ok(Self {
            receipt_id: format!("data:registry-receipt:{receipt_anchor}"),
            receipt_anchor,
            tenant_id: manifest.tenant_id.clone(),
            source_id: manifest.input.source_id.clone(),
            registry_id: registry_id.to_string(),
            manifest_id: manifest.manifest_id.clone(),
            manifest_anchor: manifest.manifest_anchor.clone(),
            graph_version_before,
            identity,
            status: "published".to_string(),
        })
    }

    pub fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            &self.receipt_id,
            ["Receipt", DATA_REGISTRY_RECEIPT_LABEL],
            serde_json::to_value(self).expect("data registry receipt is serializable"),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, DataRegistryContractError> {
        decode_labeled(node, DATA_REGISTRY_RECEIPT_LABEL, "data registry receipt")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataRegistryEntry {
    pub registry_id: String,
    pub tenant_id: String,
    pub source_id: String,
    pub manifest_id: String,
    pub manifest_anchor: String,
    pub publication_receipt_id: String,
}

impl DataRegistryEntry {
    pub fn compile(
        manifest: &DataSourceManifest,
        receipt: &DataRegistryPublicationReceipt,
    ) -> Result<Self, DataRegistryContractError> {
        Ok(Self {
            registry_id: data_registry_entry_id(&manifest.tenant_id, &manifest.input.source_id)?,
            tenant_id: manifest.tenant_id.clone(),
            source_id: manifest.input.source_id.clone(),
            manifest_id: manifest.manifest_id.clone(),
            manifest_anchor: manifest.manifest_anchor.clone(),
            publication_receipt_id: receipt.receipt_id.clone(),
        })
    }

    pub fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            &self.registry_id,
            [DATA_REGISTRY_ENTRY_LABEL],
            serde_json::to_value(self).expect("data registry entry is serializable"),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, DataRegistryContractError> {
        decode_labeled(node, DATA_REGISTRY_ENTRY_LABEL, "data registry entry")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataRegistryConflict {
    pub conflict_id: String,
    pub conflict_anchor: String,
    pub tenant_id: String,
    pub source_id: String,
    pub registry_id: String,
    pub existing_manifest_id: String,
    pub existing_manifest_anchor: String,
    pub requested_manifest_id: String,
    pub requested_manifest_anchor: String,
    pub identity: DataRegistryIdentityBinding,
    pub status: String,
    pub detail: String,
}

impl DataRegistryConflict {
    pub fn compile(
        entry: &DataRegistryEntry,
        requested: &DataSourceManifest,
        identity: DataRegistryIdentityBinding,
    ) -> Result<Self, DataRegistryContractError> {
        let conflict_anchor = hash_contract_value(json!({
            "contract": "theorem.data-registry.conflict.v1",
            "tenant_id": entry.tenant_id,
            "source_id": entry.source_id,
            "existing_manifest_anchor": entry.manifest_anchor,
            "requested_manifest_anchor": requested.manifest_anchor,
            "identity": identity,
        }))?;
        Ok(Self {
            conflict_id: format!("data:registry-conflict:{conflict_anchor}"),
            conflict_anchor,
            tenant_id: entry.tenant_id.clone(),
            source_id: entry.source_id.clone(),
            registry_id: entry.registry_id.clone(),
            existing_manifest_id: entry.manifest_id.clone(),
            existing_manifest_anchor: entry.manifest_anchor.clone(),
            requested_manifest_id: requested.manifest_id.clone(),
            requested_manifest_anchor: requested.manifest_anchor.clone(),
            identity,
            status: "conflict".to_string(),
            detail: "source_id is already bound to a different immutable manifest".to_string(),
        })
    }

    pub fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            &self.conflict_id,
            [DATA_REGISTRY_CONFLICT_LABEL],
            serde_json::to_value(self).expect("data registry conflict is serializable"),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, DataRegistryContractError> {
        decode_labeled(node, DATA_REGISTRY_CONFLICT_LABEL, "data registry conflict")
    }
}

pub fn data_registry_entry_id(
    tenant: &str,
    source_id: &str,
) -> Result<String, DataRegistryContractError> {
    Ok(format!(
        "data:registry:{}",
        hash_contract_value(json!(["theorem.data-registry.entry.v1", tenant, source_id]))?
    ))
}

pub fn publication_edges(
    entry: &DataRegistryEntry,
    receipt: &DataRegistryPublicationReceipt,
) -> [EdgeRecord; 2] {
    [
        EdgeRecord::new(
            format!("{}:manifest", entry.registry_id),
            &entry.registry_id,
            DATA_REGISTRY_HAS_MANIFEST,
            &entry.manifest_id,
            json!({ "tenant_id": entry.tenant_id, "source_id": entry.source_id }),
        ),
        EdgeRecord::new(
            format!("{}:manifest", receipt.receipt_id),
            &receipt.receipt_id,
            DATA_REGISTRY_PUBLISHED_MANIFEST,
            &entry.manifest_id,
            json!({ "tenant_id": entry.tenant_id, "source_id": entry.source_id }),
        ),
    ]
}

pub fn conflict_edge(conflict: &DataRegistryConflict) -> EdgeRecord {
    EdgeRecord::new(
        format!("{}:registry", conflict.conflict_id),
        &conflict.conflict_id,
        DATA_REGISTRY_CONFLICTS_WITH,
        &conflict.registry_id,
        json!({
            "tenant_id": conflict.tenant_id,
            "source_id": conflict.source_id,
            "existing_manifest_anchor": conflict.existing_manifest_anchor,
            "requested_manifest_anchor": conflict.requested_manifest_anchor,
        }),
    )
}

fn required_bounded(
    field: &'static str,
    value: String,
    max_bytes: usize,
) -> Result<String, DataRegistryContractError> {
    let value = value.trim().to_string();
    if value.is_empty() {
        return Err(DataRegistryContractError::new(
            "data_registry_required_field",
            format!("{field} must not be blank"),
        ));
    }
    if value.len() > max_bytes {
        return Err(DataRegistryContractError::new(
            "data_registry_field_limit",
            format!("{field} exceeds the {max_bytes}-byte limit"),
        ));
    }
    Ok(value)
}

fn normalize_unique(
    values: Vec<String>,
    max_items: usize,
    max_bytes: usize,
) -> Result<Vec<String>, DataRegistryContractError> {
    if values.len() > max_items {
        return Err(DataRegistryContractError::new(
            "data_registry_collections_limit",
            format!("collections exceeds the {max_items}-item limit"),
        ));
    }
    let mut normalized = BTreeSet::new();
    for value in values {
        normalized.insert(required_bounded("collection", value, max_bytes)?);
    }
    Ok(normalized.into_iter().collect())
}

fn is_sha256_anchor(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_labeled<T: for<'de> Deserialize<'de>>(
    node: &NodeRecord,
    label: &str,
    kind: &str,
) -> Result<T, DataRegistryContractError> {
    if !node.labels.iter().any(|candidate| candidate == label) {
        return Err(DataRegistryContractError::new(
            "data_registry_record_type_mismatch",
            format!("{} is not a {kind}", node.id),
        ));
    }
    serde_json::from_value(node.properties.clone()).map_err(|error| {
        DataRegistryContractError::new(
            "data_registry_record_invalid",
            format!("{kind} {} could not be decoded: {error}", node.id),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> DataSourceManifestInput {
        DataSourceManifestInput {
            source_id: "source:parcel-export".to_string(),
            source_kind: "csv".to_string(),
            locator: "s3://fixtures/parcel.csv".to_string(),
            content_anchor:
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            schema_version: "parcel-v1".to_string(),
            collections: vec!["parcels".to_string(), "parcels".to_string()],
            obligations: vec![DataRegistryObligation {
                obligation_id: "confirm-null-owner-semantics".to_string(),
                status: DataRegistryObligationStatus::Unknown,
                detail: "The source does not distinguish missing from withheld owners".to_string(),
            }],
            metadata: Map::new(),
        }
    }

    #[test]
    fn manifest_and_receipt_addresses_cover_normalized_content(
    ) -> Result<(), DataRegistryContractError> {
        let first = DataSourceManifest::compile("tenant-a", input())?;
        let second = DataSourceManifest::compile("tenant-a", input())?;
        assert_eq!(first, second);
        assert_eq!(first.input.collections, vec!["parcels"]);
        assert_eq!(first.unknown_obligations().len(), 1);

        let identity = DataRegistryIdentityBinding {
            principal_id: "principal:codex".to_string(),
            actor: "codex".to_string(),
            binding_id: "agent:codex".to_string(),
            project_id: Some("project:theorem".to_string()),
            project_slug: Some("Theorem".to_string()),
            resolution_path: "token->principal".to_string(),
        };
        let registry_id = data_registry_entry_id("tenant-a", "source:parcel-export")?;
        let receipt =
            DataRegistryPublicationReceipt::compile(&first, &registry_id, identity.clone(), 7)?;
        let identical =
            DataRegistryPublicationReceipt::compile(&first, &registry_id, identity.clone(), 7)?;
        let replay = DataRegistryPublicationReceipt::compile(&second, &registry_id, identity, 99)?;
        assert_eq!(receipt, identical);
        assert_ne!(receipt.receipt_id, replay.receipt_id);
        assert_ne!(receipt.receipt_anchor, replay.receipt_anchor);
        Ok(())
    }

    #[test]
    fn invalid_anchor_and_duplicate_obligations_refuse() {
        let mut invalid = input();
        invalid.content_anchor = "latest".to_string();
        assert_eq!(
            DataSourceManifest::compile("tenant-a", invalid)
                .unwrap_err()
                .code,
            "data_registry_content_anchor_invalid"
        );

        let mut duplicate = input();
        duplicate.obligations.push(duplicate.obligations[0].clone());
        assert_eq!(
            DataSourceManifest::compile("tenant-a", duplicate)
                .unwrap_err()
                .code,
            "data_registry_obligation_duplicate"
        );
    }
}
