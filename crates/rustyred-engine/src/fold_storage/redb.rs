use super::{
    composite_key, directory_size, format_value, mutation_value_bytes, namespace_prefix_len,
    require_format, validate_batch, validate_scan, FoldStorage, FoldStorageCommit,
    FoldStorageEntry, FoldStorageError, FoldStorageFloor, FoldStorageMutation, FORMAT_KEY,
};
use redb::{Database, Durability, ReadableTable, TableDefinition};
use std::fs;
use std::path::{Path, PathBuf};

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("fold_state");

#[derive(Debug)]
pub struct RedbFoldStorage {
    database: Database,
    path: PathBuf,
}

impl RedbFoldStorage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, FoldStorageError> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(super::storage_io("create redb fold directory"))?;
        let path = root.join("fold.redb");
        let database = Database::create(&path).map_err(redb_error("open redb fold floor"))?;
        let storage = Self { database, path };
        storage.initialize_format()?;
        Ok(storage)
    }

    fn initialize_format(&self) -> Result<(), FoldStorageError> {
        let mut transaction = self
            .database
            .begin_write()
            .map_err(redb_error("begin redb format transaction"))?;
        transaction.set_durability(Durability::Immediate);
        {
            let mut table = transaction
                .open_table(TABLE)
                .map_err(redb_error("open redb fold table"))?;
            let marker = table
                .get(FORMAT_KEY)
                .map_err(redb_error("read redb format marker"))?
                .map(|marker| marker.value().to_vec());
            match marker {
                Some(marker) => require_format(FoldStorageFloor::Redb, &marker)?,
                None => {
                    let marker = format_value(FoldStorageFloor::Redb);
                    table
                        .insert(FORMAT_KEY, marker.as_slice())
                        .map_err(redb_error("write redb format marker"))?;
                }
            }
        }
        transaction
            .commit()
            .map_err(redb_error("commit redb format transaction"))
    }
}

impl FoldStorage for RedbFoldStorage {
    fn floor(&self) -> FoldStorageFloor {
        FoldStorageFloor::Redb
    }

    fn get(&self, namespace: &str, key: &[u8]) -> Result<Option<Vec<u8>>, FoldStorageError> {
        let key = composite_key(namespace, key)?;
        let transaction = self
            .database
            .begin_read()
            .map_err(redb_error("begin redb read"))?;
        let table = transaction
            .open_table(TABLE)
            .map_err(redb_error("open redb fold table"))?;
        table
            .get(key.as_slice())
            .map(|value| value.map(|value| value.value().to_vec()))
            .map_err(redb_error("get redb fold value"))
    }

    fn write_batch(
        &self,
        mutations: &[FoldStorageMutation],
    ) -> Result<FoldStorageCommit, FoldStorageError> {
        let keys = validate_batch(mutations)?;
        let mut transaction = self
            .database
            .begin_write()
            .map_err(redb_error("begin redb write"))?;
        transaction.set_durability(Durability::Immediate);
        {
            let mut table = transaction
                .open_table(TABLE)
                .map_err(redb_error("open redb fold table"))?;
            for (mutation, key) in mutations.iter().zip(keys) {
                match mutation {
                    FoldStorageMutation::Put { value, .. } => {
                        table
                            .insert(key.as_slice(), value.as_slice())
                            .map_err(redb_error("insert redb fold value"))?;
                    }
                    FoldStorageMutation::Delete { .. } => {
                        table
                            .remove(key.as_slice())
                            .map_err(redb_error("delete redb fold value"))?;
                    }
                }
            }
        }
        transaction
            .commit()
            .map_err(redb_error("commit redb immediate transaction"))?;
        Ok(FoldStorageCommit {
            floor: self.floor(),
            mutations: mutations.len(),
            value_bytes: mutation_value_bytes(mutations),
            immediately_durable: true,
        })
    }

    fn scan_prefix(
        &self,
        namespace: &str,
        prefix: &[u8],
        limit: usize,
    ) -> Result<Vec<FoldStorageEntry>, FoldStorageError> {
        let encoded_prefix = validate_scan(namespace, prefix, limit)?;
        let namespace_prefix_len = namespace_prefix_len(namespace)?;
        let transaction = self
            .database
            .begin_read()
            .map_err(redb_error("begin redb prefix read"))?;
        let table = transaction
            .open_table(TABLE)
            .map_err(redb_error("open redb fold table"))?;
        let mut entries = Vec::new();
        let range = table
            .range(encoded_prefix.as_slice()..)
            .map_err(redb_error("range redb fold values"))?;
        for item in range.take(limit) {
            let (key, value) = item.map_err(redb_error("read redb fold range item"))?;
            let key = key.value();
            if !key.starts_with(&encoded_prefix) {
                break;
            }
            entries.push(FoldStorageEntry {
                key: key[namespace_prefix_len..].to_vec(),
                value: value.value().to_vec(),
            });
        }
        Ok(entries)
    }

    fn disk_bytes(&self) -> Result<u64, FoldStorageError> {
        directory_size(&self.path)
    }
}

fn redb_error<E: std::fmt::Display>(context: &'static str) -> impl Fn(E) -> FoldStorageError {
    move |error| FoldStorageError::new("fold_storage_redb", format!("{context}: {error}"))
}
