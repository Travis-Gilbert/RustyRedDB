//! Durable byte floors below deterministic fold semantics.
//!
//! The floor owns atomic, immediately durable byte batches. It does not own
//! log positions, fold schemas, checkpoints, or partial-residency policy;
//! those remain above this boundary. Both engines use the same injective
//! composite-key representation so their observable behavior is identical.

#[cfg(feature = "fjall-floor")]
mod fjall;
#[cfg(feature = "redb-floor")]
mod redb;

#[cfg(feature = "fjall-floor")]
pub use fjall::FjallFoldStorage;
#[cfg(feature = "redb-floor")]
pub use redb::RedbFoldStorage;

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

pub const FOLD_STORAGE_FORMAT_VERSION: u32 = 1;
pub(crate) const FORMAT_KEY: &[u8] = b"\0\0\0\0format";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoldStorageFloor {
    Redb,
    Fjall,
}

impl fmt::Display for FoldStorageFloor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Redb => "redb",
            Self::Fjall => "fjall",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FoldStorageMutation {
    Put {
        namespace: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        namespace: String,
        key: Vec<u8>,
    },
}

impl FoldStorageMutation {
    pub fn put(
        namespace: impl Into<String>,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
    ) -> Self {
        Self::Put {
            namespace: namespace.into(),
            key: key.into(),
            value: value.into(),
        }
    }

    pub fn delete(namespace: impl Into<String>, key: impl Into<Vec<u8>>) -> Self {
        Self::Delete {
            namespace: namespace.into(),
            key: key.into(),
        }
    }

    fn namespace(&self) -> &str {
        match self {
            Self::Put { namespace, .. } | Self::Delete { namespace, .. } => namespace,
        }
    }

    fn key(&self) -> &[u8] {
        match self {
            Self::Put { key, .. } | Self::Delete { key, .. } => key,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldStorageEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldStorageCommit {
    pub floor: FoldStorageFloor,
    pub mutations: usize,
    pub value_bytes: usize,
    pub immediately_durable: bool,
}

pub trait FoldStorage: Send + Sync {
    fn floor(&self) -> FoldStorageFloor;

    fn get(&self, namespace: &str, key: &[u8]) -> Result<Option<Vec<u8>>, FoldStorageError>;

    /// Apply one all-or-nothing batch and return only after its engine-specific
    /// immediate durability boundary succeeds.
    fn write_batch(
        &self,
        mutations: &[FoldStorageMutation],
    ) -> Result<FoldStorageCommit, FoldStorageError>;

    /// Return at most `limit` keys in byte order from one namespace/prefix.
    fn scan_prefix(
        &self,
        namespace: &str,
        prefix: &[u8],
        limit: usize,
    ) -> Result<Vec<FoldStorageEntry>, FoldStorageError>;

    fn disk_bytes(&self) -> Result<u64, FoldStorageError>;
}

#[derive(Debug)]
pub struct FoldStorageError {
    pub code: &'static str,
    pub detail: String,
}

impl FoldStorageError {
    pub(crate) fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for FoldStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl Error for FoldStorageError {}

pub(crate) fn validate_batch(
    mutations: &[FoldStorageMutation],
) -> Result<Vec<Vec<u8>>, FoldStorageError> {
    if mutations.is_empty() {
        return Err(FoldStorageError::new(
            "fold_storage_empty_batch",
            "a durable batch must contain at least one mutation",
        ));
    }
    mutations
        .iter()
        .map(|mutation| composite_key(mutation.namespace(), mutation.key()))
        .collect()
}

pub(crate) fn validate_scan(
    namespace: &str,
    prefix: &[u8],
    limit: usize,
) -> Result<Vec<u8>, FoldStorageError> {
    if limit == 0 {
        return Err(FoldStorageError::new(
            "fold_storage_unbounded_scan",
            "prefix scans require a non-zero result limit",
        ));
    }
    composite_key(namespace, prefix)
}

pub(crate) fn composite_key(namespace: &str, key: &[u8]) -> Result<Vec<u8>, FoldStorageError> {
    let canonical = namespace.trim();
    if canonical.is_empty() || canonical != namespace {
        return Err(FoldStorageError::new(
            "fold_storage_namespace",
            "namespace must be a canonical non-empty string",
        ));
    }
    let namespace_len = u32::try_from(canonical.len()).map_err(|_| {
        FoldStorageError::new(
            "fold_storage_namespace",
            "namespace length exceeds the u32 composite-key limit",
        )
    })?;
    let mut encoded = Vec::with_capacity(4 + canonical.len() + key.len());
    encoded.extend_from_slice(&namespace_len.to_be_bytes());
    encoded.extend_from_slice(canonical.as_bytes());
    encoded.extend_from_slice(key);
    Ok(encoded)
}

pub(crate) fn namespace_prefix_len(namespace: &str) -> Result<usize, FoldStorageError> {
    composite_key(namespace, &[]).map(|prefix| prefix.len())
}

pub(crate) fn format_value(floor: FoldStorageFloor) -> Vec<u8> {
    let mut value = b"rustyred-fold-storage\0".to_vec();
    value.extend_from_slice(&FOLD_STORAGE_FORMAT_VERSION.to_be_bytes());
    value.extend_from_slice(floor.to_string().as_bytes());
    value
}

pub(crate) fn require_format(
    floor: FoldStorageFloor,
    actual: &[u8],
) -> Result<(), FoldStorageError> {
    let expected = format_value(floor);
    if actual == expected {
        Ok(())
    } else {
        Err(FoldStorageError::new(
            "fold_storage_format",
            format!(
                "{floor} floor format marker is incompatible with version {FOLD_STORAGE_FORMAT_VERSION}"
            ),
        ))
    }
}

pub(crate) fn mutation_value_bytes(mutations: &[FoldStorageMutation]) -> usize {
    mutations
        .iter()
        .map(|mutation| match mutation {
            FoldStorageMutation::Put { value, .. } => value.len(),
            FoldStorageMutation::Delete { .. } => 0,
        })
        .sum()
}

pub(crate) fn directory_size(path: &Path) -> Result<u64, FoldStorageError> {
    let metadata = fs::metadata(path).map_err(storage_io("stat fold storage"))?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut bytes = 0_u64;
    for entry in fs::read_dir(path).map_err(storage_io("read fold storage directory"))? {
        let path = entry
            .map_err(storage_io("read fold storage directory entry"))?
            .path();
        bytes = bytes.saturating_add(directory_size(&path)?);
    }
    Ok(bytes)
}

pub(crate) fn storage_io(context: &'static str) -> impl Fn(std::io::Error) -> FoldStorageError {
    move |error| FoldStorageError::new("fold_storage_io", format!("{context}: {error}"))
}
