//! Content-addressed History publishing over an S3-compatible object store.

use crate::fold::history::{HistoryFold, HistoryInterval};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use async_trait::async_trait;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, ObjectStore, PutMode, PutOptions};
use std::collections::BTreeSet;
use std::env;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkPublishReceipt {
    pub address: Hash,
    pub key: String,
    pub disposition: ChunkPublishDisposition,
    pub e_tag: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkPublishDisposition {
    Published,
    AlreadyPresent,
}

#[async_trait]
pub trait PublishByChunkHash: Send + Sync {
    async fn put_if_absent(
        &self,
        address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<ChunkPublishReceipt>;

    async fn get_verified(&self, address: Hash) -> GraphStoreResult<Option<Vec<u8>>>;

    async fn delete_owned(&self, address: Hash) -> GraphStoreResult<bool>;
}

#[derive(Clone)]
pub struct S3CompatibleChunkPublisher {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

impl fmt::Debug for S3CompatibleChunkPublisher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S3CompatibleChunkPublisher")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl S3CompatibleChunkPublisher {
    pub fn new(
        endpoint: impl Into<String>,
        region: impl Into<String>,
        bucket: impl Into<String>,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        prefix: impl Into<String>,
    ) -> GraphStoreResult<Self> {
        let endpoint = required(endpoint.into(), "S3 endpoint")?;
        let region = required(region.into(), "S3 region")?;
        let bucket = required(bucket.into(), "S3 bucket")?;
        let access_key_id = required(access_key_id.into(), "S3 access key id")?;
        let secret_access_key = required(secret_access_key.into(), "S3 secret access key")?;
        let prefix = canonical_prefix(prefix.into())?;
        let store = AmazonS3Builder::new()
            .with_endpoint(endpoint.clone())
            .with_region(region)
            .with_bucket_name(bucket)
            .with_access_key_id(access_key_id)
            .with_secret_access_key(secret_access_key)
            .with_virtual_hosted_style_request(false)
            .with_allow_http(endpoint.starts_with("http://"))
            .build()
            .map_err(object_store_error)?;
        Ok(Self {
            store: Arc::new(store),
            prefix,
        })
    }

    pub fn from_env(prefix: impl Into<String>) -> GraphStoreResult<Self> {
        Self::new(
            required_env("AWS_ENDPOINT_URL_S3")?,
            env::var("AWS_REGION")
                .or_else(|_| env::var("AWS_DEFAULT_REGION"))
                .map_err(|_| missing_env("AWS_REGION or AWS_DEFAULT_REGION"))?,
            env::var("AWS_BUCKET_NAME").map_err(|_| missing_env("AWS_BUCKET_NAME"))?,
            env::var("AWS_ACCESS_KEY_ID").map_err(|_| missing_env("AWS_ACCESS_KEY_ID"))?,
            env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| missing_env("AWS_SECRET_ACCESS_KEY"))?,
            prefix,
        )
    }

    pub fn key_for(&self, address: Hash) -> String {
        let prefixed = address.to_prefixed();
        let digest = prefixed
            .strip_prefix("sha256:")
            .expect("Hash always renders with sha256 prefix");
        format!("{}/sha256/{digest}", self.prefix)
    }

    fn path_for(&self, address: Hash) -> GraphStoreResult<Path> {
        Path::parse(self.key_for(address)).map_err(object_store_error)
    }
}

#[async_trait]
impl PublishByChunkHash for S3CompatibleChunkPublisher {
    async fn put_if_absent(
        &self,
        address: Hash,
        bytes: &[u8],
    ) -> GraphStoreResult<ChunkPublishReceipt> {
        require_address(address, bytes)?;
        let path = self.path_for(address)?;
        let options = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        let (disposition, e_tag) = match self
            .store
            .put_opts(&path, bytes.to_vec().into(), options)
            .await
        {
            Ok(result) => (ChunkPublishDisposition::Published, result.e_tag),
            Err(ObjectStoreError::AlreadyExists { .. }) => {
                let stored = self
                    .store
                    .get(&path)
                    .await
                    .map_err(object_store_error)?
                    .bytes()
                    .await
                    .map_err(object_store_error)?;
                require_address(address, &stored)?;
                let metadata = self.store.head(&path).await.map_err(object_store_error)?;
                (ChunkPublishDisposition::AlreadyPresent, metadata.e_tag)
            }
            Err(error) => return Err(object_store_error(error)),
        };
        Ok(ChunkPublishReceipt {
            address,
            key: path.to_string(),
            disposition,
            e_tag,
        })
    }

    async fn get_verified(&self, address: Hash) -> GraphStoreResult<Option<Vec<u8>>> {
        let path = self.path_for(address)?;
        let bytes = match self.store.get(&path).await {
            Ok(result) => result.bytes().await.map_err(object_store_error)?.to_vec(),
            Err(ObjectStoreError::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(object_store_error(error)),
        };
        require_address(address, &bytes)?;
        Ok(Some(bytes))
    }

    async fn delete_owned(&self, address: Hash) -> GraphStoreResult<bool> {
        let path = self.path_for(address)?;
        match self.store.delete(&path).await {
            Ok(()) => Ok(true),
            Err(ObjectStoreError::NotFound { .. }) => Ok(false),
            Err(error) => Err(object_store_error(error)),
        }
    }
}

pub async fn publish_history_interval(
    publisher: &impl PublishByChunkHash,
    history: &HistoryFold,
    interval: &HistoryInterval,
) -> GraphStoreResult<Vec<ChunkPublishReceipt>> {
    let mut receipts = Vec::new();
    let mut seen = BTreeSet::new();
    for address in &interval.chunk_hashes {
        if !seen.insert(*address) {
            continue;
        }
        let bytes = history.chunk_bytes(*address).map_err(|error| {
            GraphStoreError::new("history_chunk_read_failed", error.to_string())
        })?;
        let bytes = bytes.ok_or_else(|| {
            GraphStoreError::new(
                "history_chunk_missing",
                format!("History interval references missing chunk {address}"),
            )
        })?;
        receipts.push(publisher.put_if_absent(*address, &bytes).await?);
    }
    Ok(receipts)
}

fn canonical_prefix(prefix: String) -> GraphStoreResult<String> {
    let prefix = prefix.trim().trim_matches('/').to_string();
    if prefix.is_empty() {
        return Err(GraphStoreError::new(
            "history_publish_prefix_empty",
            "History publish prefix must be non-empty",
        ));
    }
    Path::parse(&prefix).map_err(object_store_error)?;
    Ok(prefix)
}

fn required(value: String, field: &str) -> GraphStoreResult<String> {
    if value.trim().is_empty() {
        Err(GraphStoreError::new(
            "history_s3_config_missing",
            format!("{field} must be non-empty"),
        ))
    } else {
        Ok(value)
    }
}

fn required_env(name: &str) -> GraphStoreResult<String> {
    env::var(name).map_err(|_| missing_env(name))
}

fn missing_env(name: &str) -> GraphStoreError {
    GraphStoreError::new(
        "history_s3_env_missing",
        format!("required environment variable {name} is absent"),
    )
}

fn require_address(address: Hash, bytes: &[u8]) -> GraphStoreResult<()> {
    let actual = Hash::of(bytes);
    if actual == address {
        Ok(())
    } else {
        Err(GraphStoreError::new(
            "history_chunk_hash_mismatch",
            format!("History chunk hashes to {actual}, expected {address}"),
        ))
    }
}

fn object_store_error(error: impl ToString) -> GraphStoreError {
    GraphStoreError::new("history_object_store_error", error.to_string())
}
