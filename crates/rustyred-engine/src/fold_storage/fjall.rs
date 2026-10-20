use super::{
    composite_key, directory_size, format_value, mutation_value_bytes, namespace_prefix_len,
    require_format, validate_batch, validate_scan, FoldStorage, FoldStorageCommit,
    FoldStorageEntry, FoldStorageError, FoldStorageFloor, FoldStorageMutation, FORMAT_KEY,
};
use fjall::{Config, PartitionCreateOptions, PersistMode, TxKeyspace, TxPartitionHandle};
use std::fs;
use std::path::{Path, PathBuf};

const PARTITION: &str = "fold_state";

#[derive(Clone)]
pub struct FjallFoldStorage {
    keyspace: TxKeyspace,
    partition: TxPartitionHandle,
    path: PathBuf,
}

impl std::fmt::Debug for FjallFoldStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FjallFoldStorage")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl FjallFoldStorage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, FoldStorageError> {
        let path = root.as_ref().join("fjall");
        fs::create_dir_all(&path).map_err(super::storage_io("create fjall fold directory"))?;
        let keyspace = Config::new(&path)
            .open_transactional()
            .map_err(fjall_error("open fjall transactional floor"))?;
        let partition = keyspace
            .open_partition(PARTITION, PartitionCreateOptions::default())
            .map_err(fjall_error("open fjall fold partition"))?;
        let storage = Self {
            keyspace,
            partition,
            path,
        };
        storage.initialize_format()?;
        Ok(storage)
    }

    fn initialize_format(&self) -> Result<(), FoldStorageError> {
        match self
            .partition
            .get(FORMAT_KEY)
            .map_err(fjall_error("read fjall format marker"))?
        {
            Some(marker) => require_format(FoldStorageFloor::Fjall, marker.as_ref()),
            None => {
                let mut transaction = self
                    .keyspace
                    .write_tx()
                    .durability(Some(PersistMode::SyncAll));
                transaction.insert(
                    &self.partition,
                    FORMAT_KEY,
                    format_value(FoldStorageFloor::Fjall),
                );
                transaction
                    .commit()
                    .map_err(fjall_error("commit fjall format transaction"))
            }
        }
    }
}

impl FoldStorage for FjallFoldStorage {
    fn floor(&self) -> FoldStorageFloor {
        FoldStorageFloor::Fjall
    }

    fn get(&self, namespace: &str, key: &[u8]) -> Result<Option<Vec<u8>>, FoldStorageError> {
        let key = composite_key(namespace, key)?;
        self.partition
            .get(key)
            .map(|value| value.map(|value| value.to_vec()))
            .map_err(fjall_error("get fjall fold value"))
    }

    fn write_batch(
        &self,
        mutations: &[FoldStorageMutation],
    ) -> Result<FoldStorageCommit, FoldStorageError> {
        let keys = validate_batch(mutations)?;
        let mut transaction = self
            .keyspace
            .write_tx()
            .durability(Some(PersistMode::SyncAll));
        for (mutation, key) in mutations.iter().zip(keys) {
            match mutation {
                FoldStorageMutation::Put { value, .. } => {
                    transaction.insert(&self.partition, key, value.clone());
                }
                FoldStorageMutation::Delete { .. } => {
                    transaction.remove(&self.partition, key);
                }
            }
        }
        transaction
            .commit()
            .map_err(fjall_error("commit fjall SyncAll transaction"))?;
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
        let read = self.keyspace.read_tx();
        read.prefix(&self.partition, encoded_prefix)
            .take(limit)
            .map(|item| {
                let (key, value) = item.map_err(fjall_error("read fjall fold prefix item"))?;
                Ok(FoldStorageEntry {
                    key: key[namespace_prefix_len..].to_vec(),
                    value: value.to_vec(),
                })
            })
            .collect()
    }

    fn disk_bytes(&self) -> Result<u64, FoldStorageError> {
        directory_size(&self.path)
    }
}

fn fjall_error(context: &'static str) -> impl Fn(fjall::Error) -> FoldStorageError {
    move |error| FoldStorageError::new("fold_storage_fjall", format!("{context}: {error}"))
}
