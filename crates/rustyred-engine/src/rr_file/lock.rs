//! The single-writer boundary for one `.RR` file.
//!
//! D5 and S30 want one writable owner per file, advisory locking that stops a
//! second process writing, and a *named* refusal on a conflicting writable
//! open. `RrOwnership`'s own doc comment is the exact requirement: "A second
//! writable open is refused by name rather than blocked, so the caller learns
//! which device holds the lock (O02)."
//!
//! That last clause rules out using an advisory lock on its own. `flock` gives
//! exclusion and nothing else: a failed `try_lock_exclusive` proves somebody
//! holds the file and never says who. So the boundary is two mechanisms, and
//! which one carries the guarantee is the part that has to be right.
//!
//! 1. An exclusive `flock` **on the database file itself** is the guarantee.
//! 2. A sibling `.lock` file carrying an owner record supplies the name.
//!
//! The first version of this module put the `flock` on the sibling and derived
//! that sibling's path by appending `.lock` to whatever the caller typed. A
//! file does not have a name, it has an inode: `Travis.RR` and a symlink or a
//! hard link beside it are one database reachable by two spellings, and two
//! spellings produced two sibling files, two inodes and two granted locks. V01
//! drove two writers into one journal that way and read back
//! `chunk_store_journal_corrupt`. `flock` is keyed on the open file
//! description, so locking the database file is inode-exact by construction and
//! every spelling that reaches the file conflicts. ADR07 records the choice.
//!
//! Demoting the sibling to a pure label is what makes the rest tractable. It is
//! created only once the claim is already held, so a refused open leaves no
//! debris, and it is removed on a clean close, so a closed database leaves no
//! stale sibling. Neither would be safe if it were still the guarantee.
//!
//! Keying the claim on the file has one consequence the caller has to help
//! with. Compaction publishes a compacted body by renaming a fresh file over
//! the live one, which changes the inode the claim is on. Neither end of that
//! rename is a safe moment to move: adopting the new inode after the rename
//! leaves it reachable and unclaimed, and letting go of the old one before the
//! rename leaves *it* unclaimed, because the database path still resolves to it
//! until the rename lands. So a publish is three calls, not one. See
//! [`RrWriteLock::adopt_replacement`].
//!
//! The order the two are consulted still matters. A process that dies takes its
//! `flock` with it, because the kernel releases locks when the descriptor
//! closes, but it can leave whatever bytes it last wrote. So the record can
//! outlive its writer while the lock cannot, and the record is read only after
//! acquisition has already failed. Reading it first would resurrect a dead
//! owner and refuse an open that should have succeeded.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Mutex, OnceLock};

use theorem_surface_contracts::desktop::DeviceId;

use crate::file_lock::FileExt;
use crate::graph::error::{GraphStoreError, GraphStoreResult};

/// Owner records start with this so a torn or foreign file is ignored rather
/// than parsed into a confident wrong answer.
const OWNER_RECORD_TAG: &str = "rr-owner/1";

/// Longest owner record worth reading. The record is one short line; anything
/// larger is not one.
const OWNER_RECORD_READ_LIMIT: u64 = 4096;

static RR_PROCESS_LOCKS: OnceLock<Mutex<BTreeSet<FileKey>>> = OnceLock::new();

/// Identifies the file a claim is on, so two spellings cannot take two claims.
///
/// On unix this is the device and inode read from the open descriptor, which is
/// the only identity a symlink and a hard link share. Elsewhere it falls back to
/// the canonical path, which is weaker against hard links. The exclusive
/// `flock` on the database file is the guarantee on every platform; this key
/// only decides whether a refusal says "this process" or "another process".
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FileKey {
    #[cfg(unix)]
    DeviceInode(u64, u64),
    #[cfg(not(unix))]
    CanonicalPath(PathBuf),
}

impl FileKey {
    /// The key the path names right now, or `None` when it names nothing that
    /// can be stat'd.
    ///
    /// Not the same question as [`FileKey::of`]. A descriptor goes on naming
    /// the file it was opened on for as long as it is held; a path can be made
    /// to name a different file by a rename at any moment. Follows symlinks,
    /// because `open` does, so the two answers agree on a symlinked database.
    fn at(path: &Path) -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let metadata = std::fs::metadata(path).ok()?;
            Some(Self::DeviceInode(metadata.dev(), metadata.ino()))
        }
        #[cfg(not(unix))]
        {
            // Off unix the key is the path itself, so it cannot disagree with
            // a descriptor opened from that same path.
            Some(Self::CanonicalPath(
                path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
            ))
        }
    }

    fn of(file: &File, path: &Path) -> GraphStoreResult<Self> {
        let _ = path;
        let metadata = file.metadata().map_err(|error| {
            GraphStoreError::new(
                "rr_writer_lock_io",
                format!("stat RR database {}: {error}", path.display()),
            )
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self::DeviceInode(metadata.dev(), metadata.ino()))
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            Ok(Self::CanonicalPath(
                path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
            ))
        }
    }
}

/// Who holds the write lock on a database.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RrOwner {
    /// The device that opened the file writable.
    pub device: DeviceId,
    /// The owning process id, for an operator who has to go find it.
    pub pid: u32,
}

/// A held single-writer claim. Releases on drop.
///
/// Owns a descriptor on the database file, because that descriptor is what
/// carries the `flock`. A writer therefore reaches the file through two
/// descriptors, this one and the journal's; only this one is locked, and
/// `flock` conflicts between descriptors even inside one process, which is why
/// a second open in this process is refused by the kernel as well as by the
/// registry.
///
/// Both the descriptor and the registry key are behind a `Mutex` because the
/// file this claim is on can be replaced underneath it. Compaction publishes a
/// compacted body by renaming a fresh file over the live one, and a claim keyed
/// on the file has to move to the new inode or it ends up locking one that no
/// path reaches. See [`RrWriteLock::adopt_replacement`].
#[derive(Debug)]
pub struct RrWriteLock {
    database: Mutex<File>,
    database_path: PathBuf,
    lock_path: PathBuf,
    process_key: Mutex<FileKey>,
    /// The descriptor this claim held before an adoption that has not yet been
    /// reported as published, still locked and still registered.
    ///
    /// Publishing a compacted body is a rename, and a rename is not the
    /// instant the claim moves: it is the instant the old inode stops being
    /// reachable. Between adopting the replacement and renaming it into place
    /// the database path still resolves to the old inode, so letting go of it
    /// at adoption time leaves the live file unclaimed for the length of a
    /// rename. Holding both is what closes that window, and it is safe to hold
    /// both because one process never conflicts with itself here: these are two
    /// inodes, not two claims on one.
    superseded: Mutex<Option<(File, FileKey)>>,
}

impl RrWriteLock {
    /// The sibling file naming this claim's holder.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// The locked database descriptor, for a caller that has to write the file
    /// it just claimed.
    pub(super) fn database_mut(&mut self) -> &mut File {
        // `&mut self` means nothing else can be inside the mutex, and a
        // poisoned mutex here still yields the one descriptor this claim holds.
        self.database
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Take a second claim, on the file that is about to be renamed over this
    /// one, without letting go of the first.
    ///
    /// Called by compaction, the one operation that replaces the file a claim
    /// is on. It has to happen before the rename rather than after: between a
    /// rename and a re-acquisition the new inode is reachable by the database
    /// path and unclaimed, and a second writer asking for it in that window
    /// would be granted it.
    ///
    /// Adopting does not finish the move. The old inode is still the one the
    /// database path resolves to until the rename lands, so this claim keeps
    /// holding it and the caller says which way the publish went with
    /// [`RrWriteLock::publish_committed`] or
    /// [`RrWriteLock::publish_failed`]. Saying neither leaves both inodes
    /// claimed, which refuses writers that should have been admitted but never
    /// admits one that should have been refused.
    pub(super) fn adopt_replacement(&self, replacement: &File) -> GraphStoreResult<()> {
        let io = |action: &str, error: std::io::Error| {
            GraphStoreError::new(
                "rr_writer_lock_io",
                format!("{action} {}: {error}", self.database_path.display()),
            )
        };

        // A dup shares the open file description, so this has to be a fresh
        // descriptor on the replacement, not a second handle onto the caller's.
        let adopted = replacement
            .try_clone()
            .map_err(|error| io("duplicate the compacted body of", error))?;
        FileExt::try_lock_exclusive(&adopted)
            .map_err(|error| io("claim the compacted body of", error))?;
        let adopted_key = FileKey::of(&adopted, &self.database_path)?;

        let mut held = self.database.lock().map_err(|_| poisoned())?;
        let mut key = self.process_key.lock().map_err(|_| poisoned())?;
        let mut superseded = self.superseded.lock().map_err(|_| poisoned())?;
        claim_process_key(&adopted_key, &self.database_path)?;

        let previous = std::mem::replace(&mut *held, adopted);
        let previous_key = std::mem::replace(&mut *key, adopted_key);
        // An adoption that was never reported can only mean a publish that
        // neither committed nor failed, which is not reachable from the one
        // caller. Release it rather than keeping a descriptor nobody can name.
        if let Some((stale, stale_key)) = superseded.replace((previous, previous_key)) {
            drop(stale);
            release_process_key(&stale_key);
        }
        Ok(())
    }

    /// The rename landed: the superseded inode is now unreachable, so let it go.
    ///
    /// Until this runs, the descriptor on the old inode is what keeps its
    /// number from being handed to a different file, which is why the registry
    /// key is removed here and not earlier.
    pub(super) fn publish_committed(&self) {
        let taken = self
            .superseded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some((superseded, superseded_key)) = taken {
            drop(superseded);
            release_process_key(&superseded_key);
        }
    }

    /// The rename did not land: the old inode is still the database, so put the
    /// claim back on it and drop the one taken on the replacement.
    pub(super) fn publish_failed(&self) {
        let mut held = self
            .database
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut key = self
            .process_key
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut superseded = self
            .superseded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((previous, previous_key)) = superseded.take() {
            let abandoned = std::mem::replace(&mut *held, previous);
            let abandoned_key = std::mem::replace(&mut *key, previous_key);
            drop(abandoned);
            release_process_key(&abandoned_key);
        }
    }
}

impl Drop for RrWriteLock {
    fn drop(&mut self) {
        // Best effort, and deliberately so. A failure here cannot be reported
        // and must not panic, and the kernel releases the lock when the
        // descriptor closes regardless of what happens to the label.
        //
        // The label goes first. Releasing the lock first would let another
        // process take the claim and publish its own record, which this drop
        // would then delete.
        let _ = std::fs::remove_file(&self.lock_path);
        let _ = FileExt::unlock(self.database_mut());
        let key = self
            .process_key
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        release_process_key(&key);
        // An adoption whose publish was never reported still holds the inode
        // it superseded. Closing the descriptor releases that lock; the key
        // has to come out by hand.
        if let Some((superseded, superseded_key)) = self
            .superseded
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            drop(superseded);
            release_process_key(&superseded_key);
        }
    }
}

/// The sibling file that names the holder of `database_path`.
///
/// Appends `.lock` rather than replacing the extension. `Path::with_extension`
/// would turn `Travis.RR` into `Travis.lock`, which collides with any sibling
/// `Travis.*`; the journal gets away with that only because `redcore.journal`
/// has no siblings.
///
/// Canonicalizes first where it can, so two spellings of one file agree on one
/// label. This is a label and not the guarantee: the guarantee is the `flock`
/// on the database file, which two hard links share even though `canonicalize`
/// reports them as two distinct real paths.
#[must_use]
pub fn lock_path_for(database_path: &Path) -> PathBuf {
    let resolved = database_path
        .canonicalize()
        .unwrap_or_else(|_| database_path.to_path_buf());
    let mut name = OsString::from(resolved.into_os_string());
    name.push(".lock");
    PathBuf::from(name)
}

/// Take the claim on a database that must not already exist.
///
/// `create_new` is the existence check and the claim in one step: of two
/// concurrent creates exactly one observes the file as absent, which no
/// separate `exists()` test can promise. On any later failure the caller is
/// expected to call [`discard_new`], because a half-created database that still
/// occupies its path can never be created again.
pub(super) fn claim_new(database_path: &Path, device: &DeviceId) -> GraphStoreResult<RrWriteLock> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(database_path)
        .map_err(|error| {
            if error.kind() == ErrorKind::AlreadyExists {
                GraphStoreError::new(
                    "rr_already_exists",
                    format!(
                        "{} already exists; open it instead of creating over it",
                        database_path.display()
                    ),
                )
            } else {
                GraphStoreError::new(
                    "rr_create_io",
                    format!("create {}: {error}", database_path.display()),
                )
            }
        })?;

    claim(file, database_path, device).map_err(|error| {
        // The file is ours: we created it in this call and nothing else has
        // seen it. Leaving it would brick the path for every later create.
        // The exception is the one error that says the path stopped naming it.
        // Then the file at the path is somebody else's published body, and
        // removing it would destroy a database this call never owned.
        if error.code != "rr_writer_superseded" {
            let _ = std::fs::remove_file(database_path);
        }
        error
    })
}

/// Take the claim on a database that must already exist.
///
/// Opening and locking are two steps and a publish can land between them, so
/// this is a loop rather than one call. Each turn opens whatever the path names
/// now and [`claim`] refuses when that stops being what it locked. A retry
/// reaches the truth: the replacement is held by whoever published it, so the
/// next turn is refused by the `flock` with a named holder. The bound is here
/// because a path republished without pause is a condition to report, not one
/// to spin on.
pub(super) fn claim_existing(
    database_path: &Path,
    device: &DeviceId,
) -> GraphStoreResult<RrWriteLock> {
    const ATTEMPTS: usize = 8;

    for _ in 0..ATTEMPTS {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(database_path)
            .map_err(|error| {
                GraphStoreError::new(
                    "rr_open_io",
                    format!("open {}: {error}", database_path.display()),
                )
            })?;
        match claim(file, database_path, device) {
            Err(error) if error.code == "rr_writer_superseded" => continue,
            outcome => return outcome,
        }
    }

    Err(GraphStoreError::new(
        "rr_writer_superseded",
        format!(
            "RR database {} was replaced under all {ATTEMPTS} attempts to open it",
            database_path.display()
        ),
    ))
}

/// Give up a claim taken by [`claim_new`] and remove the file it created.
pub(super) fn discard_new(claim: RrWriteLock, database_path: &Path) {
    drop(claim);
    let _ = std::fs::remove_file(database_path);
}

fn claim(file: File, database_path: &Path, device: &DeviceId) -> GraphStoreResult<RrWriteLock> {
    // O02 asks the refusal to name a device. A writer with no name cannot be
    // named, and answering "a device called nothing holds it" is worse than
    // refusing here, where the caller can still fix it.
    if device.as_str().is_empty() {
        return Err(GraphStoreError::new(
            "rr_writer_device_unnamed",
            format!(
                "cannot claim RR database {} for an empty device id; \
                 a second writer has to be told which device holds the file",
                database_path.display()
            ),
        ));
    }

    let process_key = FileKey::of(&file, database_path)?;
    let lock_path = lock_path_for(database_path);

    claim_process_key(&process_key, database_path)?;

    let result: GraphStoreResult<()> = (|| {
        FileExt::try_lock_exclusive(&file).map_err(|error| {
            if error.kind() == ErrorKind::WouldBlock {
                let holder = read_owner(&lock_path);
                let who = match &holder {
                    Some(owner) => format!("device {} (pid {})", owner.device, owner.pid),
                    None => "another process that has not published its identity".to_owned(),
                };
                GraphStoreError::new(
                    "rr_writer_held",
                    format!(
                        "RR database {} is open writable by {who}",
                        database_path.display()
                    ),
                )
            } else {
                GraphStoreError::new(
                    "rr_writer_lock_io",
                    format!("lock RR database {}: {error}", database_path.display()),
                )
            }
        })?;

        // The descriptor was opened before this lock was taken, and a publish
        // renames a fresh body over the path. If one landed in between, the
        // lock now sits on an inode no path reaches, while the caller would
        // believe it holds the database and write into a file already
        // unlinked. The answer is only stable once the lock is held, so the
        // comparison belongs here and not next to the open.
        if FileKey::at(database_path).as_ref() != Some(&process_key) {
            return Err(GraphStoreError::new(
                "rr_writer_superseded",
                format!(
                    "RR database {} was replaced while it was being opened",
                    database_path.display()
                ),
            ));
        }

        publish_owner(&lock_path, device).map_err(|error| {
            // The claim is real even if the label failed, so surrender it
            // rather than holding a lock nobody can attribute.
            let _ = FileExt::unlock(&file);
            error
        })?;

        Ok(())
    })();

    match result {
        Ok(()) => Ok(RrWriteLock {
            database: Mutex::new(file),
            database_path: database_path.to_path_buf(),
            lock_path,
            process_key: Mutex::new(process_key),
            superseded: Mutex::new(None),
        }),
        Err(error) => {
            release_process_key(&process_key);
            Err(error)
        }
    }
}

/// Read the owner record, if the sibling file carries a well-formed one.
///
/// Returns `None` rather than an error for every malformed case. A torn record,
/// an empty file, a missing file and a foreign file are all "cannot say who",
/// which is a weaker answer than naming a device but never a wrong one.
#[must_use]
pub fn read_owner(lock_path: &Path) -> Option<RrOwner> {
    let file = File::open(lock_path).ok()?;
    let mut text = String::new();
    file.take(OWNER_RECORD_READ_LIMIT)
        .read_to_string(&mut text)
        .ok()?;
    parse_owner(&text)
}

/// The record is `rr-owner/1 <pid> <device>`, with the device last and escaped.
///
/// The device is a name a human chose, so it can carry a space ("mac studio")
/// and in principle a newline. The first version split the whole line on spaces
/// and required exactly three fields, which turned any such name into `None`
/// and degraded O02's refusal to "a process that has not published its
/// identity" without saying so. Putting the device last means only the escape
/// has to be reversible, and the fixed-arity prefix still rejects a torn line.
fn parse_owner(text: &str) -> Option<RrOwner> {
    let line = text.lines().next()?;
    let rest = line.strip_prefix(OWNER_RECORD_TAG)?.strip_prefix(' ')?;
    let (pid, device) = rest.split_once(' ')?;
    let pid = pid.parse::<u32>().ok()?;
    let device = unescape_device(device)?;
    if device.is_empty() {
        return None;
    }
    Some(RrOwner {
        device: DeviceId::new(device),
        pid,
    })
}

fn escape_device(device: &str) -> String {
    let mut escaped = String::with_capacity(device.len());
    for character in device.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            other => escaped.push(other),
        }
    }
    escaped
}

fn unescape_device(escaped: &str) -> Option<String> {
    let mut device = String::with_capacity(escaped.len());
    let mut characters = escaped.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            device.push(character);
            continue;
        }
        match characters.next()? {
            '\\' => device.push('\\'),
            'n' => device.push('\n'),
            'r' => device.push('\r'),
            // An unknown escape is a record this build did not write, so it is
            // "cannot say" rather than a guess at what was meant.
            _ => return None,
        }
    }
    Some(device)
}

/// Write the owner record, creating the sibling file if it is not there.
///
/// Refuses to follow a symlink. The path is derived from the database path, so
/// an attacker or a stale tree can put a symlink where the label goes; opening
/// it read-write and truncating it would then destroy whatever it points at.
fn publish_owner(lock_path: &Path, device: &DeviceId) -> GraphStoreResult<()> {
    let io = |action: &str, error: std::io::Error| {
        GraphStoreError::new(
            "rr_writer_lock_io",
            format!("{action} {}: {error}", lock_path.display()),
        )
    };

    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(lock_path).map_err(|e| io("open RR lock", e))?;

    let record = format!(
        "{OWNER_RECORD_TAG} {} {}\n",
        process::id(),
        escape_device(device.as_str())
    );
    file.set_len(0)
        .map_err(|e| io("clear RR owner record in", e))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| io("seek RR owner record in", e))?;
    file.write_all(record.as_bytes())
        .map_err(|e| io("write RR owner record to", e))?;
    file.flush()
        .map_err(|e| io("flush RR owner record to", e))?;
    Ok(())
}

fn poisoned() -> GraphStoreError {
    GraphStoreError::new("rr_writer_lock_poisoned", "RR writer claim is poisoned")
}

/// Move this process's registry entry from one file identity to another.
///
/// One critical section, so no other thread can observe the claim as belonging
/// to neither file. The insert cannot collide: the caller already holds an
/// exclusive `flock` on the identity being inserted.
/// Record that this process holds the claim on `process_key`, or refuse.
///
/// The kernel would refuse a second `flock` too, since `flock` conflicts
/// between descriptors even inside one process. The registry exists to say
/// "this process" instead of "another process", which is the difference
/// between a bug in this program and a second instance of it.
fn claim_process_key(process_key: &FileKey, database_path: &Path) -> GraphStoreResult<()> {
    let registry = RR_PROCESS_LOCKS.get_or_init(|| Mutex::new(BTreeSet::new()));
    let mut held = registry.lock().map_err(|_| {
        GraphStoreError::new(
            "rr_writer_lock_poisoned",
            "RR writer lock registry is poisoned",
        )
    })?;
    if held.insert(process_key.clone()) {
        Ok(())
    } else {
        Err(GraphStoreError::new(
            "rr_writer_held",
            format!(
                "RR database {} is already open writable in this process",
                database_path.display()
            ),
        ))
    }
}

fn release_process_key(process_key: &FileKey) {
    if let Some(registry) = RR_PROCESS_LOCKS.get() {
        if let Ok(mut held) = registry.lock() {
            held.remove(process_key);
        }
    }
}

/// Tests for the three-state publish protocol.
///
/// They live here rather than beside the other `.RR` tests because the states
/// they discriminate between are only reachable through `pub(super)` entry
/// points: a compacted body is published by `journal.rs`, which drives this
/// protocol but cannot say what it did to the claim, and `graph::tests` can
/// reach `RustyRed` but never the claim inside it.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A directory of this test's own, without `SystemTime`, which miri refuses
    /// even with isolation disabled.
    fn scratch(name: &str) -> PathBuf {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rr-lock-{}-{}-{name}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        dir
    }

    /// Stand in for the compacted body that compaction is about to rename into
    /// place. Only its inode matters here, so it stays empty.
    fn replacement_in(dir: &Path) -> File {
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(dir.join("Travis.RR.compacting"))
            .expect("create the replacement body")
    }

    fn refusal(database: &Path) -> GraphStoreError {
        claim_existing(database, &DeviceId::new("second writer"))
            .err()
            .expect("a second writable open must be refused")
    }

    /// Adoption is not the moment the claim moves. Until the rename lands, the
    /// database path still resolves to the old inode, so that inode has to stay
    /// claimed: this is the window a second writer walked into before the
    /// protocol had three states.
    #[test]
    fn an_adopted_replacement_does_not_release_the_file_the_path_still_names() {
        let dir = scratch("adopted");
        let database = dir.join("Travis.RR");
        let claim = claim_new(&database, &DeviceId::new("mac studio")).expect("claim the database");

        claim
            .adopt_replacement(&replacement_in(&dir))
            .expect("adopt the compacted body");

        let error = refusal(&database);
        assert_eq!(error.code, "rr_writer_held");
        assert!(
            error.message.contains("in this process"),
            "the refusal should name this process: {}",
            error.message
        );

        drop(claim);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A contender can open the old inode before publication and reach the
    /// `flock` only after the publisher releases it. That stale descriptor must
    /// not become a claim for a path that now resolves to the replacement.
    #[test]
    fn a_preopened_superseded_inode_cannot_claim_the_replacement_path() {
        let dir = scratch("preopened-superseded");
        let database = dir.join("Travis.RR");
        let publisher =
            claim_new(&database, &DeviceId::new("publisher")).expect("claim the database");
        let stale = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&database)
            .expect("open the old inode before publication");

        let replacement_path = dir.join("Travis.RR.compacting");
        let replacement = replacement_in(&dir);
        publisher
            .adopt_replacement(&replacement)
            .expect("adopt the compacted body");
        std::fs::rename(&replacement_path, &database).expect("publish the replacement");
        publisher.publish_committed();

        let stale_claim = claim(stale, &database, &DeviceId::new("contender"));
        assert!(
            stale_claim.is_err(),
            "a descriptor for the superseded inode must not authorize the replacement path"
        );

        drop(publisher);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A publish that did not land leaves the old inode as the database, so the
    /// claim belongs on it and the one taken on the replacement has to go.
    #[test]
    fn a_publish_that_did_not_land_gives_the_claim_back_its_original_file() {
        let dir = scratch("failed");
        let database = dir.join("Travis.RR");
        let claim = claim_new(&database, &DeviceId::new("mac studio")).expect("claim the database");

        let replacement = replacement_in(&dir);
        claim
            .adopt_replacement(&replacement)
            .expect("adopt the compacted body");
        claim.publish_failed();

        assert_eq!(refusal(&database).code, "rr_writer_held");

        // The replacement is nobody's database now. Compaction removes it, and
        // a claim left on it would refuse an inode that later reuses its
        // number, so the release has to be observable.
        drop(replacement);
        let abandoned = dir.join("Travis.RR.compacting");
        let second = claim_existing(&abandoned, &DeviceId::new("second writer"))
            .expect("the abandoned replacement must no longer be claimed");
        drop(second);

        drop(claim);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A publish that landed made the old inode unreachable by any path, so
    /// holding it would only refuse writers that should be admitted.
    #[test]
    fn a_publish_that_landed_releases_the_file_the_rename_replaced() {
        let dir = scratch("committed");
        let database = dir.join("Travis.RR");
        let claim = claim_new(&database, &DeviceId::new("mac studio")).expect("claim the database");

        claim
            .adopt_replacement(&replacement_in(&dir))
            .expect("adopt the compacted body");
        claim.publish_committed();

        // No rename happened in this test, so the old inode is still reachable
        // and claiming it must now succeed. In the real sequence the rename has
        // already put the replacement here and there is nothing left to claim.
        let second = claim_existing(&database, &DeviceId::new("second writer"))
            .expect("the superseded file must no longer be claimed");
        drop(second);

        drop(claim);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
