//! Origin- and partition-scoped IndexedDB storage inside one RustyRed file.
//!
//! Servo supplies already order-preserving encoded keys.  The broker resolves
//! an opaque grant below the browser API, stages a complete database mutation,
//! and publishes one new `browser` entry in the RR
//! [`RootCatalog`](crate::chunk_store::RootCatalog). Persistent
//! state is a content-addressed browser/partition/origin/database hierarchy;
//! ephemeral state never reaches the chunk store.

mod model;
mod persistence;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use rand_chacha::rand_core::{OsRng, RngCore};
use theorem_capability_scope::browser::{OpaqueGrantTable, OpaqueGrantTableError};
use theorem_surface_contracts::desktop::{
    BrowserPartitionId, Origin, PartitionResidency, StorageGrant, StorageGrantId, StorageScope,
};

use crate::chunk_store::ChunkStoreCatalog;
use crate::graph::error::GraphStoreError;
use crate::hash::Hash;

use model::DatabaseState;
pub use model::{
    BrowserMutation, BrowserOperation, BrowserOperationResult, BrowserRecord, BrowserTransaction,
    BrowserTransactionProgram, DatabaseSnapshot, EncodedKey, EncodedKeyRange, IndexEntry,
    IndexKeys, IndexSchema, KeyPath, ObjectStoreSchema, QueryTarget,
};
use persistence::BrowserStorageLens;
pub(crate) use persistence::{
    decode_browser_database, decode_browser_origin, decode_browser_partition, decode_browser_root,
};

/// Stable RootCatalog entry for durable third-party browser storage.
pub const ROOT_NAME_BROWSER: &str = "browser";

/// Object-safe marker for stores which support atomic named-root publication.
pub trait BrowserStorageChunkStore: ChunkStoreCatalog {}

impl<T: ChunkStoreCatalog> BrowserStorageChunkStore for T {}

pub type BrowserStorageResult<T> = Result<T, BrowserStorageError>;

/// Structured failure returned across the host storage boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserStorageError {
    pub code: String,
    pub message: String,
}

impl BrowserStorageError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    fn grant_refused() -> Self {
        Self::new(
            "browser_storage_grant_refused",
            "the browser storage grant is unknown, revoked, or outside the expected scope",
        )
    }

    fn invalid_scope(message: impl Into<String>) -> Self {
        Self::new("browser_storage_scope_invalid", message)
    }

    fn invalid_transaction(message: impl Into<String>) -> Self {
        Self::new("browser_storage_transaction_invalid", message)
    }

    fn store_not_found() -> Self {
        Self::new(
            "browser_storage_store_not_found",
            "the IndexedDB object store does not exist",
        )
    }

    fn index_not_found() -> Self {
        Self::new(
            "browser_storage_index_not_found",
            "the IndexedDB index does not exist in this object store",
        )
    }

    fn already_exists(kind: &str) -> Self {
        Self::new(
            "browser_storage_already_exists",
            format!("the IndexedDB {kind} already exists"),
        )
    }

    fn constraint(message: impl Into<String>) -> Self {
        Self::new("browser_storage_constraint", message)
    }

    fn corrupt(message: impl Into<String>) -> Self {
        Self::new("browser_storage_corrupt", message)
    }

    fn backend(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(code, message)
    }
}

impl From<GraphStoreError> for BrowserStorageError {
    fn from(error: GraphStoreError) -> Self {
        Self::backend(error.code, error.message)
    }
}

impl From<OpaqueGrantTableError> for BrowserStorageError {
    fn from(error: OpaqueGrantTableError) -> Self {
        Self::backend("browser_storage_grant_table", error.to_string())
    }
}

impl fmt::Display for BrowserStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for BrowserStorageError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentPublication {
    pub browser_root: Hash,
    pub catalog_root: Hash,
    pub commit: Hash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransactionReceipt {
    pub changed: bool,
    /// `None` means the transaction used the process-local ephemeral overlay.
    pub publication: Option<PersistentPublication>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransactionExecution {
    pub results: Vec<BrowserOperationResult>,
    pub receipt: TransactionReceipt,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BrowserRootDifferential {
    pub browser: Option<Hash>,
    pub partition: Option<Hash>,
    pub origin: Option<Hash>,
    pub database: Option<Hash>,
}

struct BrokerState {
    grants: OpaqueGrantTable<StorageGrant>,
    ephemeral: BTreeMap<StorageScope, DatabaseState>,
}

impl Default for BrokerState {
    fn default() -> Self {
        Self {
            grants: OpaqueGrantTable::default(),
            ephemeral: BTreeMap::new(),
        }
    }
}

/// The only browser-facing path to the RR IndexedDB lens.
///
/// All methods accepting browser authority take only [`StorageGrantId`].  The
/// concrete scope remains in `state.grants` on the host side, and one mutex
/// serializes persistent root publication with ephemeral lifecycle changes.
pub struct BrowserStorageBroker {
    lens: BrowserStorageLens,
    state: Mutex<BrokerState>,
}

impl BrowserStorageBroker {
    #[must_use]
    pub fn new(store: Arc<dyn BrowserStorageChunkStore>) -> Self {
        Self {
            lens: BrowserStorageLens::new(store),
            state: Mutex::new(BrokerState::default()),
        }
    }

    /// Mint an unpredictable capability for one exact storage scope.
    pub fn issue_grant(
        &self,
        scope: StorageScope,
        residency: PartitionResidency,
    ) -> BrowserStorageResult<StorageGrantId> {
        validate_scope_for_grant(&scope)?;
        let mut state = self.lock_state()?;
        for _ in 0..8 {
            let id = StorageGrantId::new(random_token()?);
            let grant = StorageGrant {
                id: id.clone(),
                scope: scope.clone(),
                residency,
            };
            match state.grants.register(id.as_str(), grant) {
                Ok(()) => return Ok(id),
                Err(OpaqueGrantTableError::DuplicateToken) => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(BrowserStorageError::backend(
            "browser_storage_grant_collision",
            "could not mint a distinct browser storage grant",
        ))
    }

    /// Host-side resolution for wiring a grant to a concrete database engine.
    /// Browser code should use the transaction/query methods below instead.
    pub fn resolve_grant(&self, id: &StorageGrantId) -> BrowserStorageResult<StorageGrant> {
        self.lock_state()?
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)
    }

    /// Resolve while checking an expected scope and residency without exposing
    /// which private dimension mismatched.
    pub fn resolve_grant_for(
        &self,
        id: &StorageGrantId,
        expected_scope: &StorageScope,
        expected_residency: PartitionResidency,
    ) -> BrowserStorageResult<StorageGrant> {
        self.lock_state()?
            .grants
            .resolve_matching(id.as_str(), |grant| {
                grant.scope == *expected_scope && grant.residency == expected_residency
            })
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)
    }

    pub fn revoke_grant(&self, id: &StorageGrantId) -> BrowserStorageResult<bool> {
        Ok(self.lock_state()?.grants.revoke(id.as_str()).is_some())
    }

    pub fn transact(
        &self,
        id: &StorageGrantId,
        transaction: &BrowserTransaction,
    ) -> BrowserStorageResult<TransactionReceipt> {
        let mut state = self.lock_state()?;
        let grant = state
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)?;

        match grant.residency {
            PartitionResidency::Persistent => {
                let (_, publication) = self.lens.transact(&grant.scope, transaction)?;
                Ok(TransactionReceipt {
                    changed: publication.is_some(),
                    publication,
                })
            }
            PartitionResidency::Ephemeral => {
                let current = state
                    .ephemeral
                    .get(&grant.scope)
                    .cloned()
                    .unwrap_or_default();
                let next = current.apply(transaction)?;
                let changed = next != current;
                if changed {
                    state.ephemeral.insert(grant.scope, next);
                }
                Ok(TransactionReceipt {
                    changed,
                    publication: None,
                })
            }
        }
    }

    /// Execute Servo's ordered request queue against one staged snapshot.
    /// Query results are returned only after any changed durable root has been
    /// published successfully.
    pub fn execute_transaction(
        &self,
        id: &StorageGrantId,
        program: &BrowserTransactionProgram,
    ) -> BrowserStorageResult<TransactionExecution> {
        let mut state = self.lock_state()?;
        let grant = state
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)?;

        let (results, publication, ephemeral_changed) = match grant.residency {
            PartitionResidency::Persistent => {
                let (_, results, publication) = self.lens.execute(&grant.scope, program)?;
                (results, publication, false)
            }
            PartitionResidency::Ephemeral => {
                let current = state
                    .ephemeral
                    .get(&grant.scope)
                    .cloned()
                    .unwrap_or_default();
                let (next, results) = current.execute(program)?;
                let changed = next != current;
                if changed {
                    state.ephemeral.insert(grant.scope, next);
                }
                (results, None, changed)
            }
        };
        Ok(TransactionExecution {
            results,
            receipt: TransactionReceipt {
                changed: publication.is_some() || ephemeral_changed,
                publication,
            },
        })
    }

    pub fn database(&self, id: &StorageGrantId) -> BrowserStorageResult<DatabaseSnapshot> {
        let state = self.lock_state()?;
        let grant = state
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)?;
        match grant.residency {
            PartitionResidency::Persistent => {
                self.lens.database(&grant.scope).map(|db| db.snapshot())
            }
            PartitionResidency::Ephemeral => Ok(state
                .ephemeral
                .get(&grant.scope)
                .cloned()
                .unwrap_or_default()
                .snapshot()),
        }
    }

    pub fn query(
        &self,
        id: &StorageGrantId,
        store: &str,
        target: &QueryTarget,
        range: &EncodedKeyRange,
        limit: Option<u32>,
    ) -> BrowserStorageResult<Vec<BrowserRecord>> {
        self.database(id)?.query(store, target, range, limit)
    }

    pub fn delete_database(
        &self,
        id: &StorageGrantId,
    ) -> BrowserStorageResult<Option<PersistentPublication>> {
        let mut state = self.lock_state()?;
        let grant = state
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)?;
        match grant.residency {
            PartitionResidency::Persistent => self.lens.delete_database(&grant.scope),
            PartitionResidency::Ephemeral => {
                state.ephemeral.remove(&grant.scope);
                Ok(None)
            }
        }
    }

    /// Clear every database for one host-selected origin without rebuilding
    /// unrelated partition/origin/database chunks.
    pub fn clear_origin(
        &self,
        partition: &BrowserPartitionId,
        origin: &Origin,
    ) -> BrowserStorageResult<Option<PersistentPublication>> {
        let mut state = self.lock_state()?;
        // Publish the fallible durable half first. Once it succeeds, removing
        // matching process-local state cannot fail, so callers never observe
        // a half-cleared origin when RR publication is refused.
        let publication = self
            .lens
            .clear_origin(partition.as_str(), origin.as_str())?;
        state
            .ephemeral
            .retain(|scope, _| scope.partition != *partition || scope.origin != *origin);
        Ok(publication)
    }

    /// Destroy a process-local partition and revoke all of its ephemeral
    /// capabilities.  No durable browser root is read or written.
    pub fn destroy_ephemeral_partition(
        &self,
        partition: &BrowserPartitionId,
    ) -> BrowserStorageResult<usize> {
        let mut state = self.lock_state()?;
        state
            .ephemeral
            .retain(|scope, _| scope.partition != *partition);
        Ok(state.grants.revoke_where(|grant| {
            grant.residency == PartitionResidency::Ephemeral && grant.scope.partition == *partition
        }))
    }

    /// Content-address differential for verification and host diagnostics.
    /// Ephemeral grants always report an empty differential.
    pub fn root_differential(
        &self,
        id: &StorageGrantId,
    ) -> BrowserStorageResult<BrowserRootDifferential> {
        let state = self.lock_state()?;
        let grant = state
            .grants
            .resolve(id.as_str())
            .cloned()
            .ok_or_else(BrowserStorageError::grant_refused)?;
        match grant.residency {
            PartitionResidency::Persistent => self.lens.differential(&grant.scope),
            PartitionResidency::Ephemeral => Ok(BrowserRootDifferential::default()),
        }
    }

    fn lock_state(&self) -> BrowserStorageResult<MutexGuard<'_, BrokerState>> {
        self.state.lock().map_err(|_| {
            BrowserStorageError::backend(
                "browser_storage_poisoned",
                "browser storage broker mutex was poisoned",
            )
        })
    }
}

fn validate_scope_for_grant(scope: &StorageScope) -> BrowserStorageResult<()> {
    for (label, value) in [
        ("partition", scope.partition.as_str()),
        ("origin", scope.origin.as_str()),
    ] {
        if value.is_empty() {
            return Err(BrowserStorageError::invalid_scope(format!(
                "browser storage {label} cannot be empty"
            )));
        }
    }
    Ok(())
}

fn random_token() -> BrowserStorageResult<String> {
    let mut bytes = [0_u8; 24];
    OsRng.try_fill_bytes(&mut bytes).map_err(|error| {
        BrowserStorageError::backend(
            "browser_storage_random",
            format!("mint browser storage grant: {error}"),
        )
    })?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        token.push(HEX[usize::from(byte >> 4)] as char);
        token.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    Ok(token)
}
