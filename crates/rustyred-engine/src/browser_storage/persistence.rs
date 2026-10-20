use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use theorem_surface_contracts::desktop::StorageScope;

use crate::chunk_store::{CatalogCommitResult, CommitMeta, validate_catalog_root};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;

use super::model::{
    BrowserOperationResult, BrowserTransaction, BrowserTransactionProgram, DatabaseState,
};
use super::{
    BrowserRootDifferential, BrowserStorageError, BrowserStorageResult, PersistentPublication,
    ROOT_NAME_BROWSER,
};

const BROWSER_ROOT_FORMAT: &str = "theorem-browser-root/1";
const PARTITION_ROOT_FORMAT: &str = "theorem-browser-partition/1";
const ORIGIN_ROOT_FORMAT: &str = "theorem-browser-origin/1";

// `commit_catalog_updates` merges named roots, but two independently built
// `browser` hierarchies cannot be merged at that boundary. Serialize the full
// read-build-publish window across every broker in this process.
static BROWSER_PUBLICATION_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct RootEntry {
    name: String,
    root: Hash,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRoot {
    format: String,
    partitions: Vec<RootEntry>,
}

impl Default for BrowserRoot {
    fn default() -> Self {
        Self {
            format: BROWSER_ROOT_FORMAT.to_owned(),
            partitions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct PartitionRoot {
    format: String,
    origins: Vec<RootEntry>,
}

impl Default for PartitionRoot {
    fn default() -> Self {
        Self {
            format: PARTITION_ROOT_FORMAT.to_owned(),
            origins: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct OriginRoot {
    format: String,
    databases: Vec<RootEntry>,
}

impl Default for OriginRoot {
    fn default() -> Self {
        Self {
            format: ORIGIN_ROOT_FORMAT.to_owned(),
            databases: Vec::new(),
        }
    }
}

pub(super) struct BrowserStorageLens {
    store: Arc<dyn super::BrowserStorageChunkStore>,
}

impl BrowserStorageLens {
    pub(super) fn new(store: Arc<dyn super::BrowserStorageChunkStore>) -> Self {
        Self { store }
    }

    pub(super) fn database(&self, scope: &StorageScope) -> BrowserStorageResult<DatabaseState> {
        self.load_path(scope).map(|path| path.database)
    }

    pub(super) fn differential(
        &self,
        scope: &StorageScope,
    ) -> BrowserStorageResult<BrowserRootDifferential> {
        let path = self.load_path(scope)?;
        Ok(BrowserRootDifferential {
            browser: path.browser_address,
            partition: path.partition_address,
            origin: path.origin_address,
            database: path.database_address,
        })
    }

    pub(super) fn transact(
        &self,
        scope: &StorageScope,
        transaction: &BrowserTransaction,
    ) -> BrowserStorageResult<(DatabaseState, Option<PersistentPublication>)> {
        let _publication_guard = self.lock_publication()?;
        let path = self.load_path(scope)?;
        let next_database = path.database.apply(transaction)?;
        if next_database == path.database {
            return Ok((next_database, None));
        }
        let publication = self.publish_database(scope, path, &next_database)?;
        Ok((next_database, Some(publication)))
    }

    pub(super) fn execute(
        &self,
        scope: &StorageScope,
        program: &BrowserTransactionProgram,
    ) -> BrowserStorageResult<(
        DatabaseState,
        Vec<BrowserOperationResult>,
        Option<PersistentPublication>,
    )> {
        let _publication_guard = self.lock_publication()?;
        let path = self.load_path(scope)?;
        let (next_database, results) = path.database.execute(program)?;
        if next_database == path.database {
            return Ok((next_database, results, None));
        }
        let publication = self.publish_database(scope, path, &next_database)?;
        Ok((next_database, results, Some(publication)))
    }

    pub(super) fn delete_database(
        &self,
        scope: &StorageScope,
    ) -> BrowserStorageResult<Option<PersistentPublication>> {
        let _publication_guard = self.lock_publication()?;
        let mut path = self.load_path(scope)?;
        if remove_entry(&mut path.origin.databases, &scope.database).is_none() {
            return Ok(None);
        }

        let mut chunks = Vec::<(Hash, Vec<u8>)>::new();
        if path.origin.databases.is_empty() {
            remove_entry(&mut path.partition.origins, scope.origin.as_str());
        } else {
            let bytes = encode_canonical(&path.origin, "browser origin root")?;
            let address = Hash::of(&bytes);
            chunks.push((address, bytes));
            upsert_entry(&mut path.partition.origins, scope.origin.as_str(), address);
        }

        if path.partition.origins.is_empty() {
            remove_entry(&mut path.browser.partitions, scope.partition.as_str());
        } else {
            let bytes = encode_canonical(&path.partition, "browser partition root")?;
            let address = Hash::of(&bytes);
            chunks.push((address, bytes));
            upsert_entry(
                &mut path.browser.partitions,
                scope.partition.as_str(),
                address,
            );
        }

        self.publish_changed_browser_root(path.browser, chunks)
            .map(Some)
    }

    pub(super) fn clear_origin(
        &self,
        partition: &str,
        origin: &str,
    ) -> BrowserStorageResult<Option<PersistentPublication>> {
        let _publication_guard = self.lock_publication()?;
        let mut browser = self.load_browser_root()?.0;
        let Some(partition_address) = entry_root(&browser.partitions, partition) else {
            return Ok(None);
        };
        let mut partition_root: PartitionRoot =
            self.read_node(partition_address, "browser partition root")?;
        validate_partition_root(&partition_root)?;
        if remove_entry(&mut partition_root.origins, origin).is_none() {
            return Ok(None);
        }

        let mut chunks = Vec::<(Hash, Vec<u8>)>::new();
        if partition_root.origins.is_empty() {
            remove_entry(&mut browser.partitions, partition);
        } else {
            let bytes = encode_canonical(&partition_root, "browser partition root")?;
            let address = Hash::of(&bytes);
            chunks.push((address, bytes));
            upsert_entry(&mut browser.partitions, partition, address);
        }
        self.publish_changed_browser_root(browser, chunks).map(Some)
    }

    fn publish_database(
        &self,
        scope: &StorageScope,
        mut path: LoadedPath,
        database: &DatabaseState,
    ) -> BrowserStorageResult<PersistentPublication> {
        let database_bytes = encode_canonical(database, "browser database")?;
        let database_address = Hash::of(&database_bytes);
        upsert_entry(
            &mut path.origin.databases,
            &scope.database,
            database_address,
        );

        let origin_bytes = encode_canonical(&path.origin, "browser origin root")?;
        let origin_address = Hash::of(&origin_bytes);
        upsert_entry(
            &mut path.partition.origins,
            scope.origin.as_str(),
            origin_address,
        );

        let partition_bytes = encode_canonical(&path.partition, "browser partition root")?;
        let partition_address = Hash::of(&partition_bytes);
        upsert_entry(
            &mut path.browser.partitions,
            scope.partition.as_str(),
            partition_address,
        );

        let browser_bytes = encode_canonical(&path.browser, "browser root")?;
        let browser_address = Hash::of(&browser_bytes);
        let chunks = [
            (database_address, database_bytes),
            (origin_address, origin_bytes),
            (partition_address, partition_bytes),
            (browser_address, browser_bytes),
        ];
        write_chunks(self.store.as_ref(), &chunks)?;
        let published = publish_browser_root(self.store.as_ref(), browser_address)?;

        Ok(PersistentPublication {
            browser_root: browser_address,
            catalog_root: published.catalog_root,
            commit: published.commit,
        })
    }

    fn publish_changed_browser_root(
        &self,
        browser: BrowserRoot,
        mut chunks: Vec<(Hash, Vec<u8>)>,
    ) -> BrowserStorageResult<PersistentPublication> {
        let browser_bytes = encode_canonical(&browser, "browser root")?;
        let browser_address = Hash::of(&browser_bytes);
        chunks.push((browser_address, browser_bytes));
        write_chunks(self.store.as_ref(), &chunks)?;
        let published = publish_browser_root(self.store.as_ref(), browser_address)?;
        Ok(PersistentPublication {
            browser_root: browser_address,
            catalog_root: published.catalog_root,
            commit: published.commit,
        })
    }

    fn load_path(&self, scope: &StorageScope) -> BrowserStorageResult<LoadedPath> {
        validate_scope(scope)?;
        let (browser, browser_address) = self.load_browser_root()?;
        let (partition, partition_address) =
            if let Some(address) = entry_root(&browser.partitions, scope.partition.as_str()) {
                let partition: PartitionRoot = self.read_node(address, "browser partition root")?;
                validate_partition_root(&partition)?;
                (partition, Some(address))
            } else {
                (PartitionRoot::default(), None)
            };
        let (origin, origin_address) =
            if let Some(address) = entry_root(&partition.origins, scope.origin.as_str()) {
                let origin: OriginRoot = self.read_node(address, "browser origin root")?;
                validate_origin_root(&origin)?;
                (origin, Some(address))
            } else {
                (OriginRoot::default(), None)
            };
        let (database, database_address) =
            if let Some(address) = entry_root(&origin.databases, &scope.database) {
                let database: DatabaseState = self.read_node(address, "browser database")?;
                database.validate()?;
                (database, Some(address))
            } else {
                (DatabaseState::default(), None)
            };

        Ok(LoadedPath {
            browser,
            browser_address,
            partition,
            partition_address,
            origin,
            origin_address,
            database,
            database_address,
        })
    }

    fn load_browser_root(&self) -> BrowserStorageResult<(BrowserRoot, Option<Hash>)> {
        let Some(catalog_address) = self.store.read_root()? else {
            return Ok((BrowserRoot::default(), None));
        };
        let catalog = validate_catalog_root(self.store.as_ref(), catalog_address)?;
        let Some(browser_address) = catalog.get(ROOT_NAME_BROWSER) else {
            return Ok((BrowserRoot::default(), None));
        };
        let browser: BrowserRoot = self.read_node(browser_address, "browser root")?;
        validate_browser_root(&browser)?;
        Ok((browser, Some(browser_address)))
    }

    fn read_node<T: DeserializeOwned + Serialize>(
        &self,
        address: Hash,
        label: &str,
    ) -> BrowserStorageResult<T> {
        let bytes = self
            .store
            .get(address)?
            .ok_or_else(|| BrowserStorageError::corrupt(format!("{label} {address} is missing")))?;
        let value = serde_json::from_slice::<T>(&bytes).map_err(|error| {
            BrowserStorageError::corrupt(format!("decode {label} {address}: {error}"))
        })?;
        let canonical = encode_canonical(&value, label)?;
        if canonical.as_slice() != bytes.as_ref() {
            return Err(BrowserStorageError::corrupt(format!(
                "{label} {address} is not canonically encoded"
            )));
        }
        Ok(value)
    }

    fn lock_publication(&self) -> BrowserStorageResult<MutexGuard<'static, ()>> {
        BROWSER_PUBLICATION_LOCK.lock().map_err(|_| {
            BrowserStorageError::backend(
                "browser_storage_publication_poisoned",
                "the browser storage publication lock was poisoned",
            )
        })
    }
}

/// Typed reachability decoders for the durable browser hierarchy.
///
/// One function per level rather than a single decoder that sniffs the node it
/// is handed: the expected level is positional, so a partition root can never
/// be traversed as an origin root, and a database can never be read as an
/// origin. Each one verifies the format tag, the canonical encoding and the
/// ordering the writers guarantee, then reports exactly the addresses a
/// committed walk must descend into.
pub(crate) fn decode_browser_root(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<Vec<Hash>> {
    let root: BrowserRoot = decode_canonical_node(expected_address, bytes, "browser root")?;
    validate_browser_root(&root).map_err(into_graph_error)?;
    Ok(entry_roots(root.partitions))
}

pub(crate) fn decode_browser_partition(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<Vec<Hash>> {
    let root: PartitionRoot =
        decode_canonical_node(expected_address, bytes, "browser partition root")?;
    validate_partition_root(&root).map_err(into_graph_error)?;
    Ok(entry_roots(root.origins))
}

pub(crate) fn decode_browser_origin(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<Vec<Hash>> {
    let root: OriginRoot = decode_canonical_node(expected_address, bytes, "browser origin root")?;
    validate_origin_root(&root).map_err(into_graph_error)?;
    Ok(entry_roots(root.databases))
}

/// A database node is terminal: its object stores, records and indexes are all
/// inlined in this one chunk. It is still decoded, because compaction must not
/// carry a malformed database forward merely because nothing descends from it.
pub(crate) fn decode_browser_database(
    expected_address: Hash,
    bytes: &[u8],
) -> GraphStoreResult<()> {
    let database: DatabaseState =
        decode_canonical_node(expected_address, bytes, "browser database")?;
    database.validate().map_err(into_graph_error)
}

fn entry_roots(entries: Vec<RootEntry>) -> Vec<Hash> {
    entries.into_iter().map(|entry| entry.root).collect()
}

/// Decode one node of the browser hierarchy and refuse any encoding the
/// writer could not have produced.
fn decode_canonical_node<T>(
    expected_address: Hash,
    bytes: &[u8],
    label: &str,
) -> GraphStoreResult<T>
where
    T: DeserializeOwned + Serialize,
{
    let value = serde_json::from_slice::<T>(bytes).map_err(|error| {
        GraphStoreError::new(
            "reachability_browser_node_corrupt",
            format!("decode {label} {expected_address}: {error}"),
        )
    })?;
    let canonical = serde_json::to_vec(&value).map_err(|error| {
        GraphStoreError::new(
            "reachability_browser_node_corrupt",
            format!("re-encode {label} {expected_address}: {error}"),
        )
    })?;
    if canonical.as_slice() != bytes {
        return Err(GraphStoreError::new(
            "reachability_browser_node_corrupt",
            format!("{label} {expected_address} is not canonically encoded"),
        ));
    }
    Ok(value)
}

fn into_graph_error(error: BrowserStorageError) -> GraphStoreError {
    GraphStoreError::new(
        "reachability_browser_node_corrupt",
        format!("{}: {}", error.code, error.message),
    )
}

struct LoadedPath {
    browser: BrowserRoot,
    browser_address: Option<Hash>,
    partition: PartitionRoot,
    partition_address: Option<Hash>,
    origin: OriginRoot,
    origin_address: Option<Hash>,
    database: DatabaseState,
    database_address: Option<Hash>,
}

fn publish_browser_root(
    store: &dyn super::BrowserStorageChunkStore,
    root: Hash,
) -> BrowserStorageResult<CatalogCommitResult> {
    store
        .commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_BROWSER.to_owned(), Some(root))]),
            CommitMeta {
                parents: Vec::new(),
                author: "rustyred-browser-storage".to_owned(),
                message: "publish IndexedDB transaction".to_owned(),
                timestamp_unix_ms: 0,
                graph_version: 0,
            },
        )
        .map_err(Into::into)
}

fn write_chunks(
    store: &dyn super::BrowserStorageChunkStore,
    chunks: &[(Hash, Vec<u8>)],
) -> BrowserStorageResult<()> {
    let borrowed = chunks
        .iter()
        .map(|(address, bytes)| (*address, bytes.as_slice()))
        .collect::<Vec<_>>();
    store.write(&borrowed).map_err(Into::into)
}

fn encode_canonical<T: Serialize>(value: &T, label: &str) -> BrowserStorageResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|error| {
        BrowserStorageError::backend("browser_storage_encode", format!("encode {label}: {error}"))
    })
}

fn validate_scope(scope: &StorageScope) -> BrowserStorageResult<()> {
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

fn validate_browser_root(root: &BrowserRoot) -> BrowserStorageResult<()> {
    validate_root_entries(
        &root.format,
        BROWSER_ROOT_FORMAT,
        &root.partitions,
        "browser partitions",
        false,
    )
}

fn validate_partition_root(root: &PartitionRoot) -> BrowserStorageResult<()> {
    validate_root_entries(
        &root.format,
        PARTITION_ROOT_FORMAT,
        &root.origins,
        "browser origins",
        false,
    )
}

fn validate_origin_root(root: &OriginRoot) -> BrowserStorageResult<()> {
    validate_root_entries(
        &root.format,
        ORIGIN_ROOT_FORMAT,
        &root.databases,
        "IndexedDB databases",
        true,
    )
}

fn validate_root_entries(
    actual_format: &str,
    expected_format: &str,
    entries: &[RootEntry],
    label: &str,
    empty_names_allowed: bool,
) -> BrowserStorageResult<()> {
    if actual_format != expected_format {
        return Err(BrowserStorageError::corrupt(format!(
            "unknown {label} root format"
        )));
    }
    if !empty_names_allowed && entries.iter().any(|entry| entry.name.is_empty()) {
        return Err(BrowserStorageError::corrupt(format!(
            "{label} contain an empty identity"
        )));
    }
    if entries.windows(2).any(|pair| pair[0].name >= pair[1].name) {
        return Err(BrowserStorageError::corrupt(format!(
            "{label} are not strictly ordered"
        )));
    }
    Ok(())
}

fn entry_root(entries: &[RootEntry], name: &str) -> Option<Hash> {
    entries
        .binary_search_by(|entry| entry.name.as_str().cmp(name))
        .ok()
        .map(|at| entries[at].root)
}

fn upsert_entry(entries: &mut Vec<RootEntry>, name: &str, root: Hash) {
    match entries.binary_search_by(|entry| entry.name.as_str().cmp(name)) {
        Ok(at) => entries[at].root = root,
        Err(at) => entries.insert(
            at,
            RootEntry {
                name: name.to_owned(),
                root,
            },
        ),
    }
}

fn remove_entry(entries: &mut Vec<RootEntry>, name: &str) -> Option<Hash> {
    entries
        .binary_search_by(|entry| entry.name.as_str().cmp(name))
        .ok()
        .map(|at| entries.remove(at).root)
}
