use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{BrowserStorageError, BrowserStorageResult};

const DATABASE_FORMAT: &str = "theorem-indexeddb-database/1";

/// A key serialized by Servo's existing order-preserving IndexedDB codec.
///
/// The host lens deliberately does not decode or independently encode keys.
/// Lexicographic byte order is the contract shared with Servo's
/// `components/storage/indexeddb/engines/sqlite/encoding.rs` implementation.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EncodedKey(Vec<u8>);

impl EncodedKey {
    #[must_use]
    pub fn from_servo_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodedKeyRange {
    pub lower: Option<EncodedKey>,
    pub upper: Option<EncodedKey>,
    pub lower_open: bool,
    pub upper_open: bool,
}

impl EncodedKeyRange {
    #[must_use]
    pub fn only(key: EncodedKey) -> Self {
        Self {
            lower: Some(key.clone()),
            upper: Some(key),
            lower_open: false,
            upper_open: false,
        }
    }

    #[must_use]
    pub fn contains(&self, key: &EncodedKey) -> bool {
        let above_lower = self
            .lower
            .as_ref()
            .is_none_or(|lower| key > lower || (!self.lower_open && key == lower));
        let below_upper = self
            .upper
            .as_ref()
            .is_none_or(|upper| key < upper || (!self.upper_open && key == upper));
        above_lower && below_upper
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyPath {
    String(String),
    Sequence(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectStoreSchema {
    pub name: String,
    pub key_path: Option<KeyPath>,
    pub auto_increment: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexSchema {
    pub name: String,
    pub key_path: KeyPath,
    pub unique: bool,
    pub multi_entry: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexEntry {
    pub index_key: EncodedKey,
    pub primary_key: EncodedKey,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexKeys {
    pub index: String,
    pub keys: Vec<EncodedKey>,
}

/// Every state-changing operation the RR lens can stage in one transaction.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserMutation {
    CreateStore {
        schema: ObjectStoreSchema,
    },
    DeleteStore {
        store: String,
    },
    CreateIndex {
        store: String,
        schema: IndexSchema,
        #[serde(default)]
        backfill: Vec<IndexEntry>,
    },
    DeleteIndex {
        store: String,
        index: String,
    },
    Put {
        store: String,
        key: EncodedKey,
        value: Vec<u8>,
        overwrite: bool,
        /// The already-computed next generator value from the Servo adapter.
        /// The lens stores it atomically but never implements a second key
        /// encoder.
        next_key_generator: Option<i64>,
        #[serde(default)]
        index_updates: Vec<IndexKeys>,
    },
    Delete {
        store: String,
        range: EncodedKeyRange,
    },
    Clear {
        store: String,
    },
    SetKeyGenerator {
        store: String,
        current: i64,
    },
    SetVersion {
        version: u64,
    },
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BrowserTransaction {
    pub mutations: Vec<BrowserMutation>,
}

/// An ordered transaction program matching Servo's queued `KvsOperation`s.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BrowserTransactionProgram {
    pub operations: Vec<BrowserOperation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserOperation {
    Mutate {
        mutation: BrowserMutation,
    },
    Query {
        store: String,
        target: QueryTarget,
        range: EncodedKeyRange,
        limit: Option<u32>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserOperationResult {
    Mutated,
    Records { records: Vec<BrowserRecord> },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueryTarget {
    ObjectStore,
    Index { name: String },
}

/// One record as observed through either an object store or an index.
///
/// For object-store reads `key == primary_key`.  For index reads `key` is the
/// index key and `primary_key` remains the owning object-store key across
/// cursor movement.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserRecord {
    pub key: EncodedKey,
    pub primary_key: EncodedKey,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatabaseSnapshot {
    state: DatabaseState,
}

impl DatabaseSnapshot {
    #[must_use]
    pub fn version(&self) -> u64 {
        self.state.version
    }

    #[must_use]
    pub fn object_store_names(&self) -> Vec<String> {
        self.state
            .stores
            .iter()
            .map(|store| store.schema.name.clone())
            .collect()
    }

    #[must_use]
    pub fn key_path(&self, store: &str) -> Option<KeyPath> {
        self.state
            .store(store)
            .and_then(|store| store.schema.key_path.clone())
    }

    #[must_use]
    pub fn key_generator_current_number(&self, store: &str) -> Option<i64> {
        self.state
            .store(store)
            .and_then(|store| store.key_generator)
    }

    #[must_use]
    pub fn indexes(&self, store: &str) -> Option<Vec<IndexSchema>> {
        self.state.store(store).map(|store| {
            store
                .indexes
                .iter()
                .map(|index| index.schema.clone())
                .collect()
        })
    }

    pub fn query(
        &self,
        store: &str,
        target: &QueryTarget,
        range: &EncodedKeyRange,
        limit: Option<u32>,
    ) -> BrowserStorageResult<Vec<BrowserRecord>> {
        self.state.query(store, target, range, limit)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DatabaseState {
    format: String,
    version: u64,
    stores: Vec<ObjectStoreState>,
}

impl Default for DatabaseState {
    fn default() -> Self {
        Self {
            format: DATABASE_FORMAT.to_owned(),
            version: 0,
            stores: Vec::new(),
        }
    }
}

impl DatabaseState {
    pub(super) fn snapshot(&self) -> DatabaseSnapshot {
        DatabaseSnapshot {
            state: self.clone(),
        }
    }

    pub(super) fn apply(&self, transaction: &BrowserTransaction) -> BrowserStorageResult<Self> {
        if transaction.mutations.is_empty() {
            return Err(BrowserStorageError::invalid_transaction(
                "a browser storage transaction needs at least one mutation",
            ));
        }

        let mut staged = self.clone();
        for mutation in &transaction.mutations {
            staged.apply_mutation(mutation)?;
        }
        staged.validate()?;
        Ok(staged)
    }

    pub(super) fn execute(
        &self,
        program: &BrowserTransactionProgram,
    ) -> BrowserStorageResult<(Self, Vec<BrowserOperationResult>)> {
        if program.operations.is_empty() {
            return Err(BrowserStorageError::invalid_transaction(
                "a browser storage transaction program needs at least one operation",
            ));
        }

        let mut staged = self.clone();
        let mut results = Vec::with_capacity(program.operations.len());
        for operation in &program.operations {
            match operation {
                BrowserOperation::Mutate { mutation } => {
                    staged.apply_mutation(mutation)?;
                    results.push(BrowserOperationResult::Mutated);
                }
                BrowserOperation::Query {
                    store,
                    target,
                    range,
                    limit,
                } => {
                    let records = staged.query(store, target, range, *limit)?;
                    results.push(BrowserOperationResult::Records { records });
                }
            }
        }
        staged.validate()?;
        Ok((staged, results))
    }

    fn apply_mutation(&mut self, mutation: &BrowserMutation) -> BrowserStorageResult<()> {
        match mutation {
            BrowserMutation::CreateStore { schema } => {
                match self
                    .stores
                    .binary_search_by(|store| store.schema.name.cmp(&schema.name))
                {
                    Ok(_) => Err(BrowserStorageError::already_exists("object store")),
                    Err(at) => {
                        self.stores.insert(
                            at,
                            ObjectStoreState {
                                schema: schema.clone(),
                                key_generator: schema.auto_increment.then_some(1),
                                records: Vec::new(),
                                indexes: Vec::new(),
                            },
                        );
                        Ok(())
                    }
                }
            }
            BrowserMutation::DeleteStore { store } => {
                let at = self.store_index(store)?;
                self.stores.remove(at);
                Ok(())
            }
            BrowserMutation::CreateIndex {
                store,
                schema,
                backfill,
            } => self.store_mut(store)?.create_index(schema, backfill),
            BrowserMutation::DeleteIndex { store, index } => {
                self.store_mut(store)?.delete_index(index)
            }
            BrowserMutation::Put {
                store,
                key,
                value,
                overwrite,
                next_key_generator,
                index_updates,
            } => self.store_mut(store)?.put(
                key,
                value,
                *overwrite,
                *next_key_generator,
                index_updates,
            ),
            BrowserMutation::Delete { store, range } => self.store_mut(store)?.delete(range),
            BrowserMutation::Clear { store } => {
                self.store_mut(store)?.clear();
                Ok(())
            }
            BrowserMutation::SetKeyGenerator { store, current } => {
                if *current < 1 {
                    return Err(BrowserStorageError::invalid_transaction(
                        "an IndexedDB key generator must be positive",
                    ));
                }
                let store = self.store_mut(store)?;
                if !store.schema.auto_increment {
                    return Err(BrowserStorageError::invalid_transaction(
                        "a key generator cannot be set on a non-auto-increment store",
                    ));
                }
                store.key_generator = Some(*current);
                Ok(())
            }
            BrowserMutation::SetVersion { version } => {
                if *version < self.version {
                    return Err(BrowserStorageError::invalid_transaction(
                        "an IndexedDB version cannot move backwards",
                    ));
                }
                self.version = *version;
                Ok(())
            }
        }
    }

    fn store(&self, name: &str) -> Option<&ObjectStoreState> {
        self.stores
            .binary_search_by(|store| store.schema.name.as_str().cmp(name))
            .ok()
            .map(|at| &self.stores[at])
    }

    fn store_index(&self, name: &str) -> BrowserStorageResult<usize> {
        self.stores
            .binary_search_by(|store| store.schema.name.as_str().cmp(name))
            .map_err(|_| BrowserStorageError::store_not_found())
    }

    fn store_mut(&mut self, name: &str) -> BrowserStorageResult<&mut ObjectStoreState> {
        let at = self.store_index(name)?;
        Ok(&mut self.stores[at])
    }

    pub(super) fn query(
        &self,
        store: &str,
        target: &QueryTarget,
        range: &EncodedKeyRange,
        limit: Option<u32>,
    ) -> BrowserStorageResult<Vec<BrowserRecord>> {
        self.store(store)
            .ok_or_else(BrowserStorageError::store_not_found)?
            .query(target, range, limit)
    }

    pub(super) fn validate(&self) -> BrowserStorageResult<()> {
        if self.format != DATABASE_FORMAT {
            return Err(BrowserStorageError::corrupt(
                "unknown browser database format",
            ));
        }
        strictly_sorted_by(
            &self.stores,
            |store| store.schema.name.as_str(),
            "object stores",
        )?;
        for store in &self.stores {
            store.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObjectStoreState {
    schema: ObjectStoreSchema,
    key_generator: Option<i64>,
    records: Vec<StoredRecord>,
    indexes: Vec<IndexState>,
}

impl ObjectStoreState {
    fn create_index(
        &mut self,
        schema: &IndexSchema,
        backfill: &[IndexEntry],
    ) -> BrowserStorageResult<()> {
        let at = match self
            .indexes
            .binary_search_by(|index| index.schema.name.cmp(&schema.name))
        {
            Ok(_) => return Err(BrowserStorageError::already_exists("index")),
            Err(at) => at,
        };

        let mut entries = backfill.to_vec();
        entries.sort_by(compare_index_entry);
        entries.dedup();
        let candidate = IndexState {
            schema: schema.clone(),
            entries,
        };
        candidate.validate_against(&self.records)?;
        self.indexes.insert(at, candidate);
        Ok(())
    }

    fn delete_index(&mut self, name: &str) -> BrowserStorageResult<()> {
        let at = self
            .indexes
            .binary_search_by(|index| index.schema.name.as_str().cmp(name))
            .map_err(|_| BrowserStorageError::index_not_found())?;
        self.indexes.remove(at);
        Ok(())
    }

    fn put(
        &mut self,
        key: &EncodedKey,
        value: &[u8],
        overwrite: bool,
        next_key_generator: Option<i64>,
        index_updates: &[IndexKeys],
    ) -> BrowserStorageResult<()> {
        let record_at = self
            .records
            .binary_search_by(|record| record.primary_key.cmp(key));
        if record_at.is_ok() && !overwrite {
            return Err(BrowserStorageError::constraint(
                "the primary key already exists",
            ));
        }

        let mut updates = BTreeMap::<&str, BTreeSet<EncodedKey>>::new();
        for update in index_updates {
            if updates.contains_key(update.index.as_str()) {
                return Err(BrowserStorageError::invalid_transaction(
                    "an index appears more than once in one put operation",
                ));
            }
            let index = self
                .indexes
                .binary_search_by(|index| index.schema.name.cmp(&update.index))
                .map_err(|_| BrowserStorageError::index_not_found())?;
            if !self.indexes[index].schema.multi_entry && update.keys.len() > 1 {
                return Err(BrowserStorageError::constraint(
                    "a non-multi-entry index cannot receive several keys",
                ));
            }
            updates.insert(update.index.as_str(), update.keys.iter().cloned().collect());
        }

        for index in &mut self.indexes {
            index.entries.retain(|entry| entry.primary_key != *key);
        }

        match record_at {
            Ok(at) => self.records[at].value = value.to_vec(),
            Err(at) => self.records.insert(
                at,
                StoredRecord {
                    primary_key: key.clone(),
                    value: value.to_vec(),
                },
            ),
        }

        for index in &mut self.indexes {
            let keys = updates
                .remove(index.schema.name.as_str())
                .unwrap_or_default();
            for index_key in keys {
                if index.schema.unique
                    && index
                        .entries
                        .iter()
                        .any(|entry| entry.index_key == index_key && entry.primary_key != *key)
                {
                    return Err(BrowserStorageError::constraint(
                        "a unique index key already belongs to another record",
                    ));
                }
                let entry = IndexEntry {
                    index_key,
                    primary_key: key.clone(),
                };
                let at = index
                    .entries
                    .binary_search_by(|candidate| compare_index_entry(candidate, &entry))
                    .unwrap_or_else(|at| at);
                if index.entries.get(at) != Some(&entry) {
                    index.entries.insert(at, entry);
                }
            }
        }

        if let Some(current) = next_key_generator {
            if !self.schema.auto_increment || current < 1 {
                return Err(BrowserStorageError::invalid_transaction(
                    "invalid next key-generator value for this object store",
                ));
            }
            self.key_generator = Some(current);
        }
        Ok(())
    }

    fn delete(&mut self, range: &EncodedKeyRange) -> BrowserStorageResult<()> {
        validate_range(range)?;
        let removed = self
            .records
            .iter()
            .filter(|record| range.contains(&record.primary_key))
            .map(|record| record.primary_key.clone())
            .collect::<BTreeSet<_>>();
        self.records
            .retain(|record| !removed.contains(&record.primary_key));
        for index in &mut self.indexes {
            index
                .entries
                .retain(|entry| !removed.contains(&entry.primary_key));
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.records.clear();
        for index in &mut self.indexes {
            index.entries.clear();
        }
    }

    fn query(
        &self,
        target: &QueryTarget,
        range: &EncodedKeyRange,
        limit: Option<u32>,
    ) -> BrowserStorageResult<Vec<BrowserRecord>> {
        validate_range(range)?;
        let limit = limit.map_or(usize::MAX, |limit| limit as usize);
        match target {
            QueryTarget::ObjectStore => Ok(self
                .records
                .iter()
                .filter(|record| range.contains(&record.primary_key))
                .take(limit)
                .map(|record| BrowserRecord {
                    key: record.primary_key.clone(),
                    primary_key: record.primary_key.clone(),
                    value: record.value.clone(),
                })
                .collect()),
            QueryTarget::Index { name } => {
                let index = self
                    .indexes
                    .binary_search_by(|index| index.schema.name.cmp(name))
                    .ok()
                    .map(|at| &self.indexes[at])
                    .ok_or_else(BrowserStorageError::index_not_found)?;
                let mut records = Vec::new();
                for entry in index
                    .entries
                    .iter()
                    .filter(|entry| range.contains(&entry.index_key))
                    .take(limit)
                {
                    let record = self
                        .records
                        .binary_search_by(|record| record.primary_key.cmp(&entry.primary_key))
                        .ok()
                        .map(|at| &self.records[at])
                        .ok_or_else(|| {
                            BrowserStorageError::corrupt(
                                "an index entry points at a missing object-store record",
                            )
                        })?;
                    records.push(BrowserRecord {
                        key: entry.index_key.clone(),
                        primary_key: entry.primary_key.clone(),
                        value: record.value.clone(),
                    });
                }
                Ok(records)
            }
        }
    }

    fn validate(&self) -> BrowserStorageResult<()> {
        if self.schema.auto_increment != self.key_generator.is_some() {
            return Err(BrowserStorageError::corrupt(
                "object-store key-generator state disagrees with its schema",
            ));
        }
        if self.key_generator.is_some_and(|number| number < 1) {
            return Err(BrowserStorageError::corrupt(
                "object-store key generator is not positive",
            ));
        }
        strictly_sorted_by(
            &self.records,
            |record| &record.primary_key,
            "object-store records",
        )?;
        strictly_sorted_by(&self.indexes, |index| index.schema.name.as_str(), "indexes")?;
        for index in &self.indexes {
            index.validate_against(&self.records)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct StoredRecord {
    primary_key: EncodedKey,
    value: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexState {
    schema: IndexSchema,
    entries: Vec<IndexEntry>,
}

impl IndexState {
    fn validate_against(&self, records: &[StoredRecord]) -> BrowserStorageResult<()> {
        for pair in self.entries.windows(2) {
            if compare_index_entry(&pair[0], &pair[1]) != Ordering::Less {
                return Err(BrowserStorageError::corrupt(
                    "index entries are not strictly ordered",
                ));
            }
        }

        let mut prior_index_key: Option<&EncodedKey> = None;
        let mut primary_keys = BTreeSet::new();
        for entry in &self.entries {
            if records
                .binary_search_by(|record| record.primary_key.cmp(&entry.primary_key))
                .is_err()
            {
                return Err(BrowserStorageError::corrupt(
                    "an index entry points at a missing object-store record",
                ));
            }
            if self.schema.unique && prior_index_key == Some(&entry.index_key) {
                return Err(BrowserStorageError::constraint(
                    "a unique index contains the same key more than once",
                ));
            }
            if !self.schema.multi_entry && !primary_keys.insert(&entry.primary_key) {
                return Err(BrowserStorageError::constraint(
                    "a non-multi-entry index contains several keys for one record",
                ));
            }
            prior_index_key = Some(&entry.index_key);
        }
        Ok(())
    }
}

fn compare_index_entry(left: &IndexEntry, right: &IndexEntry) -> Ordering {
    left.index_key
        .cmp(&right.index_key)
        .then_with(|| left.primary_key.cmp(&right.primary_key))
}

fn validate_range(range: &EncodedKeyRange) -> BrowserStorageResult<()> {
    if let (Some(lower), Some(upper)) = (&range.lower, &range.upper) {
        if lower > upper || (lower == upper && (range.lower_open || range.upper_open)) {
            return Err(BrowserStorageError::invalid_transaction(
                "IndexedDB key range is empty or reversed",
            ));
        }
    }
    Ok(())
}

fn strictly_sorted_by<'a, T, K: Ord + ?Sized + 'a>(
    values: &'a [T],
    key: impl Fn(&'a T) -> &'a K,
    label: &str,
) -> BrowserStorageResult<()> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        Err(BrowserStorageError::corrupt(format!(
            "{label} are not strictly ordered"
        )))
    } else {
        Ok(())
    }
}
