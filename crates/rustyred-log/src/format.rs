use crate::{EntryProvenance, WriteSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fmt;
use ulid::Ulid;

pub const PHYSICAL_LOG_PROTOCOL_V2: u32 = 2;
pub const PHYSICAL_LOG_PROTOCOL_V3: u32 = 3;
pub const PHYSICAL_LOG_PROTOCOL_VERSION: u32 = PHYSICAL_LOG_PROTOCOL_V3;

/// Trusted identity of the deployment that owns one physical log root.
///
/// This value is intentionally not a request identity. It is supplied when
/// the durable root is opened and is repeated in every canonical record.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct DeploymentId(String);

impl DeploymentId {
    pub fn new(value: impl Into<String>) -> Result<Self, FormatError> {
        Ok(Self(normalize_identifier(value.into(), "deployment_id")?))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeploymentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for DeploymentId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct LogId(pub Ulid);

impl LogId {
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    /// Deterministic empty identity for folds that have not yet been bound to a log.
    pub fn nil() -> Self {
        Self(Ulid::nil())
    }
}

impl Default for LogId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LogId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(transparent)]
pub struct LogPosition(pub u64);

impl LogPosition {
    pub const ZERO: Self = Self(0);

    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

impl fmt::Display for LogPosition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RecordId(pub Ulid);

impl RecordId {
    pub fn new() -> Self {
        Self(Ulid::new())
    }
}

impl Default for RecordId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

macro_rules! string_digest {
    ($name:ident) => {
        #[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
                Self(blake3::hash(bytes.as_ref()).to_hex().to_string())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_digest!(ContentHash);
string_digest!(OpDigest);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IntentKind {
    namespace: String,
    name: String,
    version: u32,
}

impl IntentKind {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        version: u32,
    ) -> Result<Self, FormatError> {
        let kind = Self {
            namespace: normalize_identifier(namespace.into(), "intent.namespace")?,
            name: normalize_identifier(name.into(), "intent.name")?,
            version,
        };
        kind.validate()?;
        Ok(kind)
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub(crate) fn validate(&self) -> Result<(), FormatError> {
        require_canonical_identifier(&self.namespace, "intent.namespace")?;
        require_canonical_identifier(&self.name, "intent.name")?;
        if self.version == 0 {
            return Err(FormatError::ZeroVersion);
        }
        let normalized_name = self.name.to_ascii_lowercase().replace(['-', '.'], "_");
        if matches!(
            normalized_name.as_str(),
            "node_upsert"
                | "edge_upsert"
                | "node_delete"
                | "edge_delete"
                | "graph_mutation"
                | "graph_mutation_batch"
                | "crud_diff"
        ) {
            return Err(FormatError::GenericCrudKind(self.name.clone()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LogIntent {
    pub kind: IntentKind,
    pub payload: Value,
}

impl LogIntent {
    pub fn new(kind: IntentKind, payload: Value) -> Result<Self, FormatError> {
        let intent = Self { kind, payload };
        intent.validate()?;
        Ok(intent)
    }

    pub(crate) fn validate(&self) -> Result<(), FormatError> {
        self.kind.validate()?;
        reject_generic_crud_payload(&self.payload)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LogRecord {
    pub protocol_version: u32,
    pub deployment_id: DeploymentId,
    pub log_id: LogId,
    pub position: LogPosition,
    pub record_id: RecordId,
    pub previous_record_hash: Option<ContentHash>,
    pub payload_hash: ContentHash,
    pub record_hash: ContentHash,
    pub op_digest: OpDigest,
    #[serde(default)]
    pub provenance: EntryProvenance,
    pub intent: LogIntent,
    pub write_set: WriteSet,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FormatError {
    EmptyIdentifier(&'static str),
    NonCanonicalIdentifier(&'static str),
    ZeroVersion,
    GenericCrudKind(String),
    GenericCrudPayload(String),
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::NonCanonicalIdentifier(field) => {
                write!(formatter, "{field} must not contain surrounding whitespace")
            }
            Self::ZeroVersion => formatter.write_str("intent version must be greater than zero"),
            Self::GenericCrudKind(kind) => {
                write!(
                    formatter,
                    "generic CRUD-diff intent kind is forbidden: {kind}"
                )
            }
            Self::GenericCrudPayload(operation) => write!(
                formatter,
                "generic CRUD-diff payload operation is forbidden: {operation}"
            ),
        }
    }
}

impl Error for FormatError {}

fn normalize_identifier(value: String, field: &'static str) -> Result<String, FormatError> {
    let value = value.trim();
    if value.is_empty() {
        Err(FormatError::EmptyIdentifier(field))
    } else {
        Ok(value.to_string())
    }
}

fn require_canonical_identifier(value: &str, field: &'static str) -> Result<(), FormatError> {
    if value.is_empty() {
        Err(FormatError::EmptyIdentifier(field))
    } else if value != value.trim() {
        Err(FormatError::NonCanonicalIdentifier(field))
    } else {
        Ok(())
    }
}

fn reject_generic_crud_payload(payload: &Value) -> Result<(), FormatError> {
    let Some(object) = payload.as_object() else {
        return Ok(());
    };
    if let Some(operation) = object.get("op").and_then(Value::as_str) {
        reject_generic_operation(operation)?;
    }
    if let Some(mutations) = object.get("mutations").and_then(Value::as_array) {
        for mutation in mutations {
            if let Some(operation) = mutation
                .as_object()
                .and_then(|mutation| mutation.get("op"))
                .and_then(Value::as_str)
            {
                reject_generic_operation(operation)?;
            }
        }
    }
    Ok(())
}

fn reject_generic_operation(operation: &str) -> Result<(), FormatError> {
    let normalized = operation
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '.'], "_");
    if matches!(
        normalized.as_str(),
        "node_upsert" | "edge_upsert" | "node_delete" | "edge_delete"
    ) {
        Err(FormatError::GenericCrudPayload(operation.to_string()))
    } else {
        Ok(())
    }
}
