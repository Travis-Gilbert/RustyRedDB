//! The browser storage backend: OPFS, through synchronous access handles.
//!
//! The origin private file system gives a browser a real file with real
//! offsets, which is what the ledger's six operations need. What it does not
//! give is a synchronous *open*: `createSyncAccessHandle` returns a promise,
//! and it exists only inside a dedicated worker. So the shape here is an async
//! constructor that acquires the handle once, and a synchronous [`LogFile`]
//! implementation over it -- which is exactly the split SPEC-THEOREM-APP-HOST-2.0's
//! `opfs-backend` task describes, and why `LogFile` itself does not carry the
//! opener.
//!
//! The worker that holds the handle is RustyRed's own (PWA specification
//! section 35), not TheoremWeb's, and the store is the browser projection of
//! RustyRed's log rather than a second application database (section 36).

use crate::physical::{PhysicalLogError, LOG_FILENAME};
use crate::storage::LogFile;
use js_sys::Reflect;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetDirectoryOptions,
    FileSystemGetFileOptions, FileSystemReadWriteOptions, FileSystemSyncAccessHandle,
    WorkerGlobalScope,
};

/// A log file held open as an OPFS synchronous access handle.
///
/// Closing matters: an open sync access handle is an exclusive lock on the
/// file for this origin, so a dropped handle that was never closed would keep
/// the next worker out. [`Drop`] closes it.
///
/// Public, and not only through the ledger's opener, for the same reason
/// [`NativeLogFile`](crate::NativeLogFile) is: the layer above the log owns its
/// own file through this seam, and the browser projection's chunk journal is
/// one of those owners.
#[derive(Debug)]
pub struct OpfsLogFile {
    handle: FileSystemSyncAccessHandle,
}

// `web_sys` handles are `!Send` because they wrap a JavaScript value, and
// `LogFile` requires `Send + Sync` to keep `PhysicalLog`'s auto-traits what
// they are on every platform. On `wasm32-unknown-unknown` without atomics
// there is one thread and no way for another to observe this handle, so the
// bound is satisfied vacuously: nothing here crosses a thread, because there
// is no second thread to cross to. The assertion is target-gated, and the
// native backends keep the real guarantee.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for OpfsLogFile {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for OpfsLogFile {}

impl OpfsLogFile {
    /// Opens (creating when absent) `<directory>/theorem.oplog` under the
    /// origin's OPFS root and returns a synchronous handle to it.
    ///
    /// Must be called from a dedicated worker: the main thread has no
    /// synchronous access handles, so this returns the browser's own refusal
    /// there rather than a handle.
    pub async fn open_in_worker(directory: &str) -> Result<Self, PhysicalLogError> {
        Self::open_named_in_worker(directory, LOG_FILENAME).await
    }

    /// The same open over an explicit file name.
    ///
    /// See [`NativeLogFile::create_or_open_named`](crate::NativeLogFile) for
    /// why the seam admits more than one name per directory: the ledger and the
    /// browser projection's chunk journal are two logs, and the second must not
    /// open the first's file. The name is resolved inside the origin's private
    /// file system, so it is a fragment below the OPFS root, never a path.
    pub async fn open_named_in_worker(
        directory: &str,
        file_name: &str,
    ) -> Result<Self, PhysicalLogError> {
        Self::open_named_with_create_in_worker(directory, file_name, true).await
    }

    /// Opens an existing file without creating a directory or file when absent.
    ///
    /// This is the OPFS adapter for read-only recovery inspection: the scan may
    /// acquire an access handle, but an absent store remains absent and no bytes
    /// are written unless a later caller explicitly opens for recovery.
    pub async fn open_existing_named_in_worker(
        directory: &str,
        file_name: &str,
    ) -> Result<Self, PhysicalLogError> {
        Self::open_named_with_create_in_worker(directory, file_name, false).await
    }

    async fn open_named_with_create_in_worker(
        directory: &str,
        file_name: &str,
        create: bool,
    ) -> Result<Self, PhysicalLogError> {
        let root = opfs_root().await?;
        let mut parent: FileSystemDirectoryHandle = root;
        for part in directory
            .split('/')
            .filter(|part| !part.is_empty() && *part != "." && *part != "..")
        {
            let options = FileSystemGetDirectoryOptions::new();
            options.set_create(create);
            let promise = parent.get_directory_handle_with_options(part, &options);
            parent = JsFuture::from(promise)
                .await
                .map_err(storage_error)?
                .unchecked_into();
        }

        let options = FileSystemGetFileOptions::new();
        options.set_create(create);
        let promise = parent.get_file_handle_with_options(file_name, &options);
        let file: FileSystemFileHandle = JsFuture::from(promise)
            .await
            .map_err(storage_error)?
            .unchecked_into();

        let promise = file.create_sync_access_handle();
        let handle = JsFuture::from(promise)
            .await
            .map_err(storage_error)?
            .unchecked_into();
        Ok(Self { handle })
    }
}

impl Drop for OpfsLogFile {
    fn drop(&mut self) {
        self.handle.close();
    }
}

impl LogFile for OpfsLogFile {
    fn len(&self) -> Result<u64, PhysicalLogError> {
        Ok(self.handle.get_size().map_err(storage_error)? as u64)
    }

    fn read_exact_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), PhysicalLogError> {
        let options = FileSystemReadWriteOptions::new();
        options.set_at(offset as f64);
        let read = self
            .handle
            .read_with_u8_array_and_options(buffer, &options)
            .map_err(storage_error)? as usize;
        if read != buffer.len() {
            return Err(PhysicalLogError::Storage(format!(
                "OPFS read at byte {offset} returned {read} of {} bytes",
                buffer.len()
            )));
        }
        Ok(())
    }

    fn append(&mut self, bytes: &[u8]) -> Result<(), PhysicalLogError> {
        let at = self.len()?;
        let options = FileSystemReadWriteOptions::new();
        options.set_at(at as f64);
        let written = self
            .handle
            .write_with_u8_array_and_options(bytes, &options)
            .map_err(storage_error)? as usize;
        if written != bytes.len() {
            return Err(PhysicalLogError::Storage(format!(
                "OPFS write at byte {at} accepted {written} of {} bytes",
                bytes.len()
            )));
        }
        Ok(())
    }

    fn truncate(&mut self, length: u64) -> Result<(), PhysicalLogError> {
        self.handle
            .truncate_with_f64(length as f64)
            .map_err(storage_error)
    }

    fn sync(&mut self) -> Result<(), PhysicalLogError> {
        self.handle.flush().map_err(storage_error)
    }

    fn sync_directory(&mut self) -> Result<(), PhysicalLogError> {
        // OPFS has no directory entries to make durable: the file handle was
        // already resolved, and `flush` is the durability point.
        Ok(())
    }
}

async fn opfs_root() -> Result<FileSystemDirectoryHandle, PhysicalLogError> {
    let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
    let navigator = scope.navigator();
    let storage = navigator.storage();
    let promise = storage.get_directory();
    let value = JsFuture::from(promise).await.map_err(storage_error)?;
    Ok(value.unchecked_into())
}

/// Turns a JavaScript rejection into the seam's storage error, keeping the
/// browser's own message -- a refusal names itself rather than arriving as a
/// generic I/O failure.
fn storage_error(error: JsValue) -> PhysicalLogError {
    // The exclusive `readwrite` sync-handle mode reports contention as a
    // DOMException, not as a generic storage fault. Preserve that typed fact
    // across the existing LogFile seam so the local runtime can return its
    // bounded `writer_already_owned` refusal instead of parsing browser text.
    if error_name(&error).as_deref() == Some("NoModificationAllowedError") {
        return PhysicalLogError::Locked(format!(
            "OPFS exclusive sync access handle is already held: {}",
            describe(&error)
        ));
    }
    PhysicalLogError::Storage(describe(&error))
}

fn error_name(error: &JsValue) -> Option<String> {
    Reflect::get(error, &JsValue::from_str("name"))
        .ok()
        .and_then(|value| value.as_string())
}

fn describe(error: &JsValue) -> String {
    if let Some(text) = error.as_string() {
        return text;
    }
    if let Some(message) = Reflect::get(error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
    {
        return message;
    }
    format!("{error:?}")
}

/// The oracle's entry point, and not a product API.
///
/// `apps/theoremweb/oracles/pwa-opfs-ledger.mjs` drives this from a worker to
/// prove the claim section 32 makes: a log opened here survives a reload with
/// no server, appends replay idempotently by digest, and a foreign deployment
/// is refused. The product-facing browser API above the log is
/// `fold-reader-wasm`'s task; this exists so the storage backend has a real
/// browser oracle before that API is designed.
///
/// One call = open the log (creating it when absent), append one deterministic
/// record for `payload`, re-read the whole file from OPFS, verify the chain,
/// and report what was found.
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn opfs_ledger_probe(
    directory: String,
    deployment_id: String,
    payload: String,
) -> Result<String, JsValue> {
    use crate::format::{IntentKind, LogIntent, OpDigest};
    use crate::overlap::{AccessPath, WriteSet};

    let file = OpfsLogFile::open_in_worker(&directory)
        .await
        .map_err(probe_error)?;
    let deployment_id =
        crate::format::DeploymentId::new(deployment_id).map_err(probe_javascript_error)?;
    let mut log = crate::physical::PhysicalLog::open_with_storage(
        std::path::PathBuf::from(&directory),
        deployment_id,
        crate::physical::PhysicalLogOptions::default(),
        Box::new(file),
    )
    .map_err(probe_error)?;

    let receipt = log
        .append(
            LogIntent::new(
                IntentKind::new("theorem.pwa.probe", "probe_append", 1)
                    .map_err(probe_javascript_error)?,
                serde_json::json!({ "payload": payload }),
            )
            .map_err(probe_javascript_error)?,
            WriteSet::new([AccessPath::Entity {
                entity: "note:pwa-probe".to_string(),
            }])
            .map_err(probe_javascript_error)?,
            OpDigest::from_bytes(&payload),
        )
        .map_err(probe_error)?;
    let report = log.verify().map_err(probe_error)?;

    let report = serde_json::json!({
        "directory": directory,
        "log_id": receipt.log_id.to_string(),
        "deployment_id": log.deployment_id().as_str(),
        "position": receipt.position.0,
        "record_hash": receipt.record_hash.as_str(),
        "replayed": receipt.replayed,
        "records": report.records,
        "through": report.through.0,
        "content_hash": report.content_hash.as_str(),
    });
    Ok(report.to_string())
}

fn probe_error(error: PhysicalLogError) -> JsValue {
    JsValue::from_str(&format!("{error}"))
}

fn probe_javascript_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
