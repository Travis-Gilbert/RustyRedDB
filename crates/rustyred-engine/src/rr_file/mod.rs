//! The native single-file `.RR` database.
//!
//! D4 makes `.RR` the canonical local database and D5 gives it exactly one
//! write owner. S29 asks for three entry points, [`RustyRed::create`],
//! [`RustyRed::open`] and [`RustyRed::open_read_only`], over a file that
//! identifies itself: magic, a format version, a capability version, a database
//! identity and integrity metadata, with "Extension alone is never trusted".
//!
//! # What is actually new here
//!
//! Very little of the durability is. `redcore.journal` already frames every
//! record with `RRCJ` magic and a CRC32, addresses every chunk by SHA-256,
//! publishes a root with a single fsync, and recovers a torn tail by
//! distinguishing an uncommitted suffix from real mid-file corruption. That
//! machinery already satisfies S31: before root publication the old state stays
//! authoritative, after it the new state does.
//!
//! What it could not do was say what the file is. So an `.RR` file is a
//! [`RR_HEADER_LEN`] identity block followed by that same frame stream, and
//! this module adds the three things the block makes possible: identification,
//! a single-writer boundary that can name its holder, and a read-only open that
//! does not repair what it reads.
//!
//! Reusing the frame stream rather than writing a second durable format is the
//! point. A parallel implementation would need its own crash tests and would
//! drift from the one that already passes them.
//!
//! # Out of scope here
//!
//! Compaction (S32) and journal-to-RR migration (S33) belong to W02 and live
//! under `rr_migrate`. This module only has to leave room for them, which it
//! does by keeping the header immutable after creation and by teaching the
//! existing compactor to carry the header across a rewrite.

mod header;
mod lock;

pub use header::{
    mint_database_id, RrHeader, RR_BODY_FORMAT_JOURNAL, RR_CAPABILITY_VERSION, RR_FORMAT_VERSION,
    RR_HEADER_LEN, RR_MAGIC, RR_MAX_DATABASE_ID_LEN,
};
pub use lock::{lock_path_for, read_owner, RrOwner, RrWriteLock};

impl crate::journal::CallerWriteClaim for RrWriteLock {
    fn adopt_replacement(&self, replacement: &std::fs::File) -> GraphStoreResult<()> {
        RrWriteLock::adopt_replacement(self, replacement)
    }

    fn publish_committed(&self) {
        RrWriteLock::publish_committed(self);
    }

    fn publish_failed(&self) {
        RrWriteLock::publish_failed(self);
    }
}

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use theorem_surface_contracts::desktop::{DatabaseId, DeviceId, RrAccess, RrOwnership};

use crate::chunk_store::{validate_catalog_root, ChunkStore, CommittedRoot, RootCatalog};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::journal::{JournalChunkStore, JournalCommitPolicy, SyncStrategy, SystemSync};

/// One open `.RR` database.
///
/// Holds the write claim for as long as it lives when opened writable, and for
/// as long after that as any clone of its store handle is alive.
///
/// The claim is not a field here, which looks like an omission and is the
/// correction V01 forced. The first version held it beside `store` and relied
/// on struct field order to drop the journal handle first. Field order is a
/// real rule and proves nothing here: `store()` hands out an `Arc`,
/// `JournalChunkStore` is itself `Clone`, and either copy outlives this struct.
/// The claim therefore lives inside `JournalInner`, behind the refcount both
/// clones share, exactly where `JournalFileLock` already lives for a bare
/// journal. ADR07 records it.
pub struct RustyRed {
    path: PathBuf,
    header: RrHeader,
    ownership: RrOwnership,
    store: Arc<JournalChunkStore>,
}

impl RustyRed {
    /// Create a new database at `path` and take its write claim.
    ///
    /// Refuses rather than truncating when something is already there. Creating
    /// over a live database would destroy it, and the caller who meant to open
    /// it should hear about the mistake.
    pub fn create(path: impl AsRef<Path>, device: &DeviceId) -> GraphStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        create_parent_of(&path)?;

        // Creating the file exclusively *is* the existence check, and is the
        // claim in the same step: of two concurrent creates exactly one
        // observes the file as absent, which a separate `exists()` test cannot
        // promise. The claim then holds an exclusive `flock` on that very
        // descriptor, so no second opener can reach the file by another name.
        let mut claim = lock::claim_new(&path, device)?;

        let header = match RrHeader::new(mint_database_id()).and_then(|header| {
            write_new_header(claim.database_mut(), &path, &header).map(|()| header)
        }) {
            Ok(header) => header,
            Err(error) => {
                // Everything on disk at this point was made by this call. A
                // half-written database left behind occupies its own path
                // forever, because `create` then refuses it as existing and
                // `open` refuses it as unreadable.
                lock::discard_new(claim, &path);
                return Err(error);
            }
        };

        let store = match JournalChunkStore::open_rr_body(
            &path,
            JournalCommitPolicy::Immediate,
            RR_HEADER_LEN,
            Box::new(claim),
        ) {
            Ok(store) => store,
            Err(error) => {
                // The claim went into the journal and was dropped with it,
                // which released the lock and removed the label. The file it
                // created is still this call's to clear.
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };

        Ok(Self {
            path,
            ownership: RrOwnership {
                database: header.database().clone(),
                access: RrAccess::Writer {
                    device: device.clone(),
                },
            },
            header,
            store: Arc::new(store),
        })
    }

    /// Open an existing database writable, or refuse by name.
    ///
    /// A second writable open returns `rr_writer_held` naming the device that
    /// holds the claim, which is O02's requirement rather than a blocking wait.
    pub fn open(path: impl AsRef<Path>, device: &DeviceId) -> GraphStoreResult<Self> {
        Self::open_writable_with(path.as_ref(), device, Arc::new(SystemSync))
    }

    /// [`RustyRed::open`] with the body's fsyncs routed through `syncer`.
    ///
    /// Test-only. The O03 replacement window, "atomically replace only after
    /// fsync, crash in the window leaves a recoverable state", can only be
    /// observed by making one specific fsync of a native `.RR` compaction fail,
    /// and the product open path hard-codes the platform syncer on purpose.
    /// Everything else about the open is [`RustyRed::open`].
    #[cfg(test)]
    pub(crate) fn open_with_syncer(
        path: impl AsRef<Path>,
        device: &DeviceId,
        syncer: Arc<dyn SyncStrategy>,
    ) -> GraphStoreResult<Self> {
        Self::open_writable_with(path.as_ref(), device, syncer)
    }

    fn open_writable_with(
        path: &Path,
        device: &DeviceId,
        syncer: Arc<dyn SyncStrategy>,
    ) -> GraphStoreResult<Self> {
        let path = path.to_path_buf();
        let claim = lock::claim_existing(&path, device)?;
        let header = read_header(&path)?;

        if !header.writable_by_this_build() {
            return Err(GraphStoreError::new(
                "rr_capability_unsupported",
                format!(
                    "{} declares capability {}, this build implements {RR_CAPABILITY_VERSION}; \
                     it can still be opened read-only",
                    path.display(),
                    header.capability_version()
                ),
            ));
        }

        let store = JournalChunkStore::open_rr_body_with_syncer(
            &path,
            JournalCommitPolicy::Immediate,
            RR_HEADER_LEN,
            Box::new(claim),
            syncer,
        )?;

        Ok(Self {
            path,
            ownership: RrOwnership {
                database: header.database().clone(),
                access: RrAccess::Writer {
                    device: device.clone(),
                },
            },
            header,
            store: Arc::new(store),
        })
    }

    /// Open an existing database read-only.
    ///
    /// Takes no claim, so any number of readers may sit beside the one writer.
    /// Accepts a capability version this build does not implement, because
    /// reading state it does not understand is safe and appending beside it is
    /// not.
    pub fn open_read_only(path: impl AsRef<Path>) -> GraphStoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        let header = read_header(&path)?;
        let store = JournalChunkStore::open_read_only_at(&path, RR_HEADER_LEN)?;
        Ok(Self {
            path,
            ownership: RrOwnership {
                database: header.database().clone(),
                access: RrAccess::ReadOnly,
            },
            header,
            store: Arc::new(store),
        })
    }

    /// This database's identity. Lives in the file, so it survives copy, move
    /// and rename.
    #[must_use]
    pub const fn database(&self) -> &DatabaseId {
        self.ownership_database()
    }

    const fn ownership_database(&self) -> &DatabaseId {
        &self.ownership.database
    }

    #[must_use]
    pub const fn header(&self) -> &RrHeader {
        &self.header
    }

    /// How this process holds the file.
    #[must_use]
    pub const fn ownership(&self) -> &RrOwnership {
        &self.ownership
    }

    #[must_use]
    pub fn is_writer(&self) -> bool {
        self.ownership.is_writer()
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The content store behind the header.
    ///
    /// Cloning it, by the `Arc` or by the handle's own `Clone`, clones the
    /// write claim with it: a handle that outlives this `RustyRed` still holds
    /// the file, and the file is released when the last one goes.
    #[must_use]
    pub fn store(&self) -> &Arc<JournalChunkStore> {
        &self.store
    }

    /// The last durably published root, or `None` for a database that has never
    /// committed one.
    pub fn committed_root(&self) -> GraphStoreResult<Option<CommittedRoot>> {
        self.store.committed_root()
    }

    /// The committed [`RootCatalog`], validated against the store.
    ///
    /// Validation rather than a bare decode, because S31 makes the committed
    /// root the authority: if it names a root the store cannot resolve, the
    /// file is not in the state it claims and the caller should hear that here
    /// rather than at the first missing read.
    pub fn catalog(&self) -> GraphStoreResult<Option<RootCatalog>> {
        let Some(address) = self.store.read_root()? else {
            return Ok(None);
        };
        validate_catalog_root(self.store.as_ref(), address).map(Some)
    }
}

/// Make sure the directory a new database is about to land in exists.
///
/// Separate from [`write_new_header`] because the claim is taken by creating
/// the database file itself, which has to happen before anything is written and
/// cannot happen before its directory exists.
fn create_parent_of(path: &Path) -> GraphStoreResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                GraphStoreError::new(
                    "rr_create_io",
                    format!("create parent of {}: {error}", path.display()),
                )
            })?;
        }
    }
    Ok(())
}

/// Write the header block of a brand new database and make it durable.
///
/// Takes the already-claimed descriptor rather than opening the path again: the
/// exclusive `flock` lives on that descriptor, and a second open would be a
/// second way to reach the file that the claim does not cover.
///
/// The parent directory is fsynced as well as the file. Without that, a crash
/// can leave a file whose contents reached the platter but whose directory
/// entry did not, which is a database that exists only until the next mount.
fn write_new_header(file: &mut File, path: &Path, header: &RrHeader) -> GraphStoreResult<()> {
    let io = |action: &str, error: std::io::Error| {
        GraphStoreError::new(
            "rr_create_io",
            format!("{action} {}: {error}", path.display()),
        )
    };
    file.write_all(&header.encode())
        .map_err(|e| io("write header of", e))?;
    file.sync_all().map_err(|e| io("sync", e))?;

    if let Some(parent) = path.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        // A directory fsync needs the directory opened read-only; on the
        // platforms that refuse it the file fsync above is still durable, so
        // this is best effort rather than fatal.
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Read and validate the header block at the front of `path`.
fn read_header(path: &Path) -> GraphStoreResult<RrHeader> {
    let mut file = File::open(path).map_err(|error| {
        GraphStoreError::new("rr_open_io", format!("open {}: {error}", path.display()))
    })?;
    let mut block = vec![0_u8; RR_HEADER_LEN as usize];
    let mut filled = 0_usize;
    while filled < block.len() {
        match file.read(&mut block[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(GraphStoreError::new(
                    "rr_open_io",
                    format!("read header of {}: {error}", path.display()),
                ))
            }
        }
    }
    block.truncate(filled);
    RrHeader::decode(&block)
}
