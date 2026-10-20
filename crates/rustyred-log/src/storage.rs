//! The file layer beneath the physical log.
//!
//! [`PhysicalLog`](crate::physical::PhysicalLog) reaches storage only through
//! [`LogFile`]; `std::fs` appears in exactly one place, the native
//! implementation below. That is what lets the ledger be linked against a
//! platform store instead of a filesystem: the browser build of
//! SPEC-THEOREM-APP-HOST-2.0's `fold-reader-wasm` chain implements the same
//! operations over OPFS (`opfs-backend`), and a wasm32 build without that
//! backend refuses to open a log by name rather than failing at run time
//! inside a filesystem call that has no filesystem behind it.
//!
//! The surface is deliberately narrow. Nothing here lists directories,
//! renames, removes, or opens arbitrary names, because the ledger does none
//! of those.

use crate::physical::PhysicalLogError;
use std::fmt;

/// The file operations the physical log performs.
///
/// Five operations are the whole storage need of the ledger once a handle
/// exists -- length, read at an offset, append, truncate, sync -- plus the
/// directory-entry sync that makes a freshly created log durable. A store
/// without directory entries, such as a browser's OPFS, treats that last
/// operation as a no-op.
///
/// Opening is deliberately *not* in this trait. It is the one operation whose
/// shape differs by platform: the filesystem opens synchronously, and OPFS
/// cannot -- `createSyncAccessHandle` returns a promise, so a browser gets its
/// handle in a Worker and then uses it synchronously. Each backend therefore
/// offers its own opener (`native::NativeLogFile::create_or_open`, the browser
/// backend's async open) and hands the result to
/// `PhysicalLog::open_with_storage`.
///
/// Two contracts matter to the record format above this layer:
///
/// - `append` must land at the current end of the file, and a backend's opener
///   must never truncate an existing log. Recovery depends on both: a reader
///   that opened a present log and truncated it would destroy the ledger it
///   was opening.
/// - Every method is fallible. A store that cannot answer `len` or `sync`
///   must say so rather than report success, because the log's crash story is
///   built on those answers.
///
/// Exclusivity between writers is the backend's business, not this trait's:
/// the native backend holds an advisory lock for the life of the handle, and
/// a browser origin's OPFS has a single writing runtime by construction.
///
/// The trait is public because the layer *above* the log is built on it rather
/// than on either backend. RustyRed's browser projection owns durable local
/// state as a `ChunkStore` over this seam (`browser_projection` in
/// `rustyred-thg-core`, SPEC-THEOREMWEB-PWA-HTML-CANVAS-1.1 section 33), and
/// that store is written against these six operations so the same commit
/// protocol runs over OPFS in a browser and over a file on a host -- which is
/// also what makes the protocol testable where a browser is not available.
/// Neither backend's opener is reached through the trait; each is public in
/// its own right for the same reason.
pub trait LogFile: fmt::Debug + Send + Sync {
    /// The log file's length in bytes.
    fn len(&self) -> Result<u64, PhysicalLogError>;

    /// Reads exactly `buffer.len()` bytes starting at `offset`.
    fn read_exact_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), PhysicalLogError>;

    /// Writes `bytes` at the current end of the log file.
    fn append(&mut self, bytes: &[u8]) -> Result<(), PhysicalLogError>;

    /// Truncates the log file to `length` bytes.
    fn truncate(&mut self, length: u64) -> Result<(), PhysicalLogError>;

    /// Makes every byte appended so far durable.
    fn sync(&mut self) -> Result<(), PhysicalLogError>;

    /// Makes the log's directory entry durable, so a log created by
    /// `create_or_open` survives a crash.
    fn sync_directory(&mut self) -> Result<(), PhysicalLogError>;
}

/// The filesystem backend, and the only `std::fs` in this crate.
///
/// It is absent from wasm32 builds, where there is no filesystem to name.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native {
    use super::LogFile;
    use crate::physical::{PhysicalLogError, LOG_FILENAME};
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    pub struct NativeLogFile {
        directory: PathBuf,
        file: File,
    }

    impl NativeLogFile {
        /// Creates or opens the log's containing directory, then opens the log
        /// file inside it, creating the file when it is absent.
        ///
        /// Opening an existing ledger must not truncate it.
        pub(crate) fn create_or_open(directory: &Path) -> Result<Self, PhysicalLogError> {
            Self::create_or_open_named(directory, LOG_FILENAME)
        }

        /// The same open over an explicit file name.
        ///
        /// The ledger owns `<directory>/theorem.oplog`; anything else reaching
        /// this seam -- RustyRed's browser-projection chunk store among them --
        /// owns its own name, so that one directory can hold a ledger and a
        /// chunk journal without either opening the other's file. The advisory
        /// lock is taken on the named file, so two handles on two names in one
        /// directory do not exclude each other, which is what lets a host test
        /// the projection's store beside a real ledger.
        pub fn create_or_open_named(
            directory: &Path,
            file_name: &str,
        ) -> Result<Self, PhysicalLogError> {
            fs::create_dir_all(directory)?;
            let path = directory.join(file_name);
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)?;
            // The exclusive lock is what keeps two processes off one log
            // directory. It is held for as long as this handle lives and is
            // released when the log drops.
            fs2::FileExt::try_lock_exclusive(&file).map_err(|error| {
                PhysicalLogError::Locked(format!("{}: {error}", path.display()))
            })?;
            Ok(Self {
                directory: directory.to_path_buf(),
                file,
            })
        }
    }

    impl LogFile for NativeLogFile {
        fn len(&self) -> Result<u64, PhysicalLogError> {
            Ok(self.file.metadata()?.len())
        }

        fn read_exact_at(
            &mut self,
            offset: u64,
            buffer: &mut [u8],
        ) -> Result<(), PhysicalLogError> {
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.read_exact(buffer)?;
            Ok(())
        }

        fn append(&mut self, bytes: &[u8]) -> Result<(), PhysicalLogError> {
            self.file.seek(SeekFrom::End(0))?;
            self.file.write_all(bytes)?;
            Ok(())
        }

        fn truncate(&mut self, length: u64) -> Result<(), PhysicalLogError> {
            self.file.set_len(length)?;
            Ok(())
        }

        fn sync(&mut self) -> Result<(), PhysicalLogError> {
            self.file.sync_all()?;
            Ok(())
        }

        fn sync_directory(&mut self) -> Result<(), PhysicalLogError> {
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DeploymentId, IntentKind, LogIntent, LogPosition, OpDigest};
    use crate::overlap::{AccessPath, WriteSet};
    use crate::physical::{PhysicalLog, PhysicalLogOptions};
    use serde_json::json;
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    /// A test store: the six operations over a byte vector, so the ledger can
    /// be driven with no filesystem at all.
    ///
    /// It is a fixture, not a production backend. It never reaches a browser;
    /// the production browser store is OPFS (`opfs-backend` in
    /// SPEC-THEOREM-APP-HOST-2.0), and its `sync` is a real durability
    /// barrier where this one's is a no-op.
    #[derive(Clone, Debug, Default)]
    struct MemoryLogFile {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl MemoryLogFile {
        fn over(bytes: &Arc<Mutex<Vec<u8>>>) -> Self {
            Self {
                bytes: Arc::clone(bytes),
            }
        }
    }

    impl LogFile for MemoryLogFile {
        fn len(&self) -> Result<u64, PhysicalLogError> {
            Ok(self.bytes.lock().expect("memory store lock").len() as u64)
        }

        fn read_exact_at(
            &mut self,
            offset: u64,
            buffer: &mut [u8],
        ) -> Result<(), PhysicalLogError> {
            let bytes = self.bytes.lock().expect("memory store lock");
            let first = offset as usize;
            let last = first + buffer.len();
            if last > bytes.len() {
                return Err(PhysicalLogError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "memory store is shorter than the read",
                )));
            }
            buffer.copy_from_slice(&bytes[first..last]);
            Ok(())
        }

        fn append(&mut self, bytes: &[u8]) -> Result<(), PhysicalLogError> {
            self.bytes
                .lock()
                .expect("memory store lock")
                .extend_from_slice(bytes);
            Ok(())
        }

        fn truncate(&mut self, length: u64) -> Result<(), PhysicalLogError> {
            self.bytes
                .lock()
                .expect("memory store lock")
                .truncate(length as usize);
            Ok(())
        }

        fn sync(&mut self) -> Result<(), PhysicalLogError> {
            Ok(())
        }

        fn sync_directory(&mut self) -> Result<(), PhysicalLogError> {
            Ok(())
        }
    }

    fn deployment_id() -> DeploymentId {
        DeploymentId::new("test:storage-seam").unwrap()
    }

    fn open(store: &Arc<Mutex<Vec<u8>>>) -> Result<PhysicalLog, PhysicalLogError> {
        PhysicalLog::open_with_storage(
            PathBuf::from("memory://ledger"),
            deployment_id(),
            PhysicalLogOptions::default(),
            Box::new(MemoryLogFile::over(store)),
        )
    }

    fn append(log: &mut PhysicalLog, value: &str) -> Result<(), Box<dyn Error>> {
        log.append(
            LogIntent::new(
                IntentKind::new("theorem", "block_update", 1)?,
                json!({ "value": value }),
            )?,
            WriteSet::new([AccessPath::Entity {
                entity: "note:1".to_string(),
            }])?,
            OpDigest::from_bytes(value),
        )?;
        Ok(())
    }

    fn stored(store: &Arc<Mutex<Vec<u8>>>) -> Vec<u8> {
        store.lock().expect("memory store lock").clone()
    }

    #[test]
    fn the_ledger_runs_over_a_store_that_is_not_a_file() -> Result<(), Box<dyn Error>> {
        let store = Arc::new(Mutex::new(Vec::new()));
        let mut log = open(&store)?;
        assert_eq!(log.directory(), Path::new("memory://ledger"));
        append(&mut log, "one")?;
        let one_record_bytes = stored(&store).len();
        append(&mut log, "two")?;
        assert_eq!(log.verify()?.records, 2);
        let two_record_bytes = stored(&store).len();
        drop(log);

        // A second open over the same bytes recovers the whole ledger, which
        // is the property the format was designed for.
        let reopened = open(&store)?;
        assert_eq!(reopened.len(), 2);
        assert_eq!(
            reopened
                .records()
                .iter()
                .map(|record| record.position)
                .collect::<Vec<_>>(),
            [LogPosition(1), LogPosition(2)]
        );
        assert_eq!(reopened.verify()?.records, 2);

        // A tail cut inside the second frame is dropped and erased, not
        // merely ignored, and the log resumes at the recovered end.
        store
            .lock()
            .expect("memory store lock")
            .truncate(two_record_bytes - 8);
        let mut recovered = open(&store)?;
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered.latest_position(), LogPosition(1));
        assert_eq!(stored(&store).len(), one_record_bytes);
        append(&mut recovered, "three")?;
        assert_eq!(recovered.verify()?.records, 2);
        drop(recovered);

        let after_recovery = open(&store)?;
        assert_eq!(after_recovery.len(), 2);
        assert_eq!(
            after_recovery.records()[1].intent.payload["value"],
            json!("three")
        );
        Ok(())
    }
}
