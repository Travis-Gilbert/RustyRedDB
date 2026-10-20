//! Verified migration from a legacy journal store to a native `.RR` file (O03).
//!
//! The legacy store is a committed journal: a frame stream that publishes one
//! catalog, with no identity block at offset zero. A native `.RR` file is the
//! same committed journal behind a header. Migration therefore copies **only
//! reachable canonical objects** — a reachability walk from the legacy
//! committed catalog, not a file copy — and republishes the same named roots
//! under a new file that identifies itself.
//!
//! # The three rules this module exists to keep
//!
//! 1. **The source is never mutated.** It is opened through read-only recovery
//!    and read; nothing here writes to it, not even the recovery repair a
//!    writable open would perform on a torn tail, and the receipt says so.
//! 2. **Nothing is replaced until every named root matches.** The target is
//!    built beside the source, reopened independently, and compared root by
//!    root. Only then does the caller hear success. Callers that want the
//!    original path to become the native file do that as a separate, explicit
//!    step.
//! 3. **Every failure leaves the old file usable and removes the partial
//!    target.** An abandoned target is deleted along with its lock sibling, so
//!    a failed migration does not leave a path that `create` will refuse
//!    forever.
//!
//! # What is deliberately not here
//!
//! Compaction of the source, any in-place rewrite, and any partial-copy
//! resumption. Migration is copy-verify-publish; a migration that cannot finish
//! is a migration that did not happen.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use theorem_surface_contracts::desktop::{DatabaseId, DeviceId};

use crate::chunk_store::{ChunkStore, ChunkStoreCatalog, CommitMeta, RootCatalog};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::journal::JournalChunkStore;
use crate::reachability::{walk_committed_reachable, CoreReachabilityCodec};
use crate::rr_file::{lock_path_for, RustyRed};

/// What a migration did, in numbers a receipt can carry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationReceipt {
    /// The legacy journal that was read.
    pub source: PathBuf,
    /// The native file that was written.
    pub target: PathBuf,
    /// The identity the target now carries.
    pub database: DatabaseId,
    /// How many named roots were compared and matched.
    pub named_roots: usize,
    /// Reachable objects copied, not counting the header or the catalog.
    pub chunks_copied: usize,
    /// Bytes of reachable object content copied.
    pub bytes_copied: u64,
    /// The target's committed catalog, which must equal the source's address.
    pub catalog_root: Hash,
    /// Always true on success; stated in the receipt rather than assumed by the
    /// caller, because "the source is untouched" is the whole promise of a
    /// migration that has a retraction path.
    pub source_preserved: bool,
}

/// Why a migration refused, when it refused rather than failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationRefusal {
    /// The target path already exists. Migration never clobbers a file.
    TargetExists,
    /// The legacy store has never committed a catalog.
    SourceUncommitted,
    /// The source catalog names no graph root.
    SourceNotAGraphStore,
}

impl MigrationRefusal {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::TargetExists => "rr_migration_target_exists",
            Self::SourceUncommitted => "rr_migration_source_uncommitted",
            Self::SourceNotAGraphStore => "rr_migration_source_not_a_graph_store",
        }
    }
}

/// Migrate one legacy journal into a new native `.RR` file.
///
/// The target must not exist. On success the target is a complete native file
/// whose committed catalog names the same roots as the source's, and the source
/// is byte-identical to what it was before the call.
///
/// The target's database identity is minted when the native file is created —
/// a bare journal carries none to carry over — and the receipt reports the
/// identity the target actually holds, read back from an independent reopen.
pub fn migrate_journal_to_rr(
    source: impl AsRef<Path>,
    target: impl AsRef<Path>,
    device: &DeviceId,
) -> GraphStoreResult<MigrationReceipt> {
    let source = source.as_ref().to_path_buf();
    let target = target.as_ref().to_path_buf();

    if target.exists() {
        return Err(refusal(
            MigrationRefusal::TargetExists,
            format!("{} already exists", target.display()),
        ));
    }

    // 1. The legacy store, through read-only recovery. Read-only on purpose:
    //    repairing a torn tail is a write, and this call's receipt promises the
    //    source is byte-identical afterwards. A migration that repaired the
    //    torn stores it exists to migrate would make `source_preserved` a lie
    //    exactly when it matters.
    let legacy = JournalChunkStore::open_read_only_at(&source, 0)?;
    let source_catalog_root = legacy.read_root()?.ok_or_else(|| {
        refusal(
            MigrationRefusal::SourceUncommitted,
            format!("{} has never committed a catalog", source.display()),
        )
    })?;
    let source_catalog = read_catalog(&legacy, source_catalog_root)?;
    if source_catalog
        .get(crate::chunk_store::ROOT_NAME_GRAPH)
        .is_none()
    {
        return Err(refusal(
            MigrationRefusal::SourceNotAGraphStore,
            format!(
                "{} names {} roots and none of them is a graph",
                source.display(),
                source_catalog.roots().len()
            ),
        ));
    }

    // 2. Walk reachability once, and copy exactly that set.
    let reachable = walk_committed_reachable(&legacy, &CoreReachabilityCodec)?;
    let addresses = reachable.reachable.iter().copied().collect::<Vec<_>>();

    let outcome = copy_and_verify(
        &legacy,
        &addresses,
        &source_catalog,
        source_catalog_root,
        &target,
        device,
    );
    match outcome {
        Ok((database, named_roots, chunks_copied, bytes_copied)) => Ok(MigrationReceipt {
            source,
            target,
            database,
            named_roots,
            chunks_copied,
            bytes_copied,
            catalog_root: source_catalog_root,
            source_preserved: true,
        }),
        Err(error) => {
            // 3. A failed migration is a migration that did not happen.
            clean_up_failed_target(&target);
            Err(error)
        }
    }
}

fn copy_and_verify(
    legacy: &JournalChunkStore,
    addresses: &[Hash],
    source_catalog: &RootCatalog,
    source_catalog_root: Hash,
    target: &Path,
    device: &DeviceId,
) -> GraphStoreResult<(DatabaseId, usize, usize, u64)> {
    let created = RustyRed::create(target, device)?;
    // The identity this migration's target will carry: minted by the create,
    // and the value every later check has to agree with. A bare journal has no
    // identity, so there is nothing to carry over; what matters is that the
    // identity minted here is the one the file still reports when it is opened
    // by somebody who never saw this handle.
    let identity = created.database().clone();
    let store = created.store().clone();

    let mut bytes_copied = 0_u64;
    for address in addresses {
        let Some(bytes) = legacy.get(*address)? else {
            return Err(GraphStoreError::new(
                "rr_migration_source_missing_chunk",
                format!("the reachability walk named {address:?} and the source cannot resolve it"),
            ));
        };
        store.write(&[(*address, bytes.as_ref())])?;
        bytes_copied += u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    }

    // Republish every named root, so the target's catalog equals the source's.
    let mut updates = BTreeMap::new();
    for (name, address) in source_catalog.roots() {
        updates.insert(name.clone(), Some(*address));
    }
    let meta = CommitMeta {
        parents: Vec::new(),
        author: format!("rr_migrate from {}", legacy.path().display()),
        message: "native .RR migration".to_owned(),
        timestamp_unix_ms: 0,
        graph_version: 0,
    };
    let published = created.store().commit_catalog_updates(&updates, meta)?;
    if published.catalog_root != source_catalog_root {
        return Err(GraphStoreError::new(
            "rr_migration_catalog_mismatch",
            format!(
                "the target published catalog {:?} and the source committed {:?}",
                published.catalog_root, source_catalog_root
            ),
        ));
    }
    drop(created);

    // Reopen independently rather than trusting the handle we just wrote with.
    let reopened = RustyRed::open_read_only(target)?;
    let target_catalog = reopened.catalog()?.ok_or_else(|| {
        GraphStoreError::new(
            "rr_migration_target_uncommitted",
            "the target reopened without a committed catalog",
        )
    })?;
    if target_catalog.address()? != source_catalog_root {
        return Err(GraphStoreError::new(
            "rr_migration_catalog_mismatch",
            "the reopened target does not commit the source's catalog address",
        ));
    }
    for (name, address) in source_catalog.roots() {
        match target_catalog.get(name) {
            Some(target_address) if target_address == *address => {}
            Some(target_address) => {
                return Err(GraphStoreError::new(
                    "rr_migration_root_mismatch",
                    format!(
                        "root {name} is {target_address:?} in the target and {address:?} in the source"
                    ),
                ));
            }
            None => {
                return Err(GraphStoreError::new(
                    "rr_migration_root_missing",
                    format!("the target does not name root {name}"),
                ));
            }
        }
    }
    // The last check is the one a header cannot fake: the identity block names
    // the identity the created file minted, read back from an independent
    // reopen rather than trusted from the handle that wrote it.
    if *reopened.database() != identity {
        return Err(GraphStoreError::new(
            "rr_migration_identity_mismatch",
            format!(
                "the created target identified itself as {} and the reopened file reports {}",
                identity.as_str(),
                reopened.database().as_str()
            ),
        ));
    }

    Ok((
        identity,
        source_catalog.roots().len(),
        addresses.len(),
        bytes_copied,
    ))
}

fn read_catalog(store: &JournalChunkStore, address: Hash) -> GraphStoreResult<RootCatalog> {
    let bytes = store.get(address)?.ok_or_else(|| {
        GraphStoreError::new(
            "rr_migration_source_missing_catalog",
            "the source's committed root names a catalog it cannot resolve",
        )
    })?;
    RootCatalog::from_bytes(bytes.as_ref())
}

fn refusal(refusal: MigrationRefusal, message: String) -> GraphStoreError {
    GraphStoreError::new(refusal.code(), message)
}

/// Remove a partial target and its lock sibling.
///
/// Both are this function's to remove: the target because this call created it,
/// and the sibling because `RustyRed::create` takes the claim by creating it. A
/// leftover sibling would otherwise be invisible debris on a path the caller
/// believes is free.
fn clean_up_failed_target(target: &Path) {
    let sibling = lock_path_for(target);
    fs::remove_file(&sibling).ok();
    fs::remove_file(target).ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_store::{CommitMeta, RootCatalog, ROOT_NAME_GRAPH};

    fn migration_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("rr-migrate-{label}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// A legacy journal with one graph payload, one unreachable payload, and a
    /// committed catalog naming the graph.
    fn legacy_store(dir: &Path) -> (PathBuf, Hash) {
        let path = dir.join("legacy.journal");
        let store = JournalChunkStore::open(&path).expect("open legacy");
        let graph = Hash::of(b"legacy graph payload");
        store
            .write(&[(graph, b"legacy graph payload".as_slice())])
            .expect("write graph");
        let orphan = Hash::of(b"legacy unreachable payload");
        store
            .write(&[(orphan, b"legacy unreachable payload".as_slice())])
            .expect("write orphan");

        let mut roots = std::collections::BTreeMap::new();
        roots.insert(ROOT_NAME_GRAPH.to_owned(), graph);
        let catalog = RootCatalog::new(roots).expect("legacy catalog");
        let catalog_root = catalog.write_to(&store).expect("write catalog");
        store
            .commit(
                catalog_root,
                CommitMeta {
                    parents: Vec::new(),
                    author: "rr-migrate-test".to_owned(),
                    message: "legacy commit".to_owned(),
                    timestamp_unix_ms: 0,
                    graph_version: 1,
                },
            )
            .expect("commit legacy catalog");
        (path, graph)
    }

    #[test]
    fn rr_migration_refuses_an_untyped_graph_root_and_leaves_nothing_behind() {
        // Measured precondition: the reachability walk decodes the catalog's
        // graph root as a versioned tree, so a store whose graph root is raw
        // bytes is refused by name rather than copied as if it were a graph.
        // This is the refusal half of the migration contract; the accepting
        // half needs a legacy store built by `RedCoreGraphStore`, which is the
        // next step recorded in this module's docs.
        let dir = migration_dir("untyped");
        let (source, _) = legacy_store(&dir);
        let target = dir.join("Out.RR");
        let before = fs::read(&source).expect("read source before");

        let error = migrate_journal_to_rr(&source, &target, &DeviceId::new("migrate test device"))
            .expect_err("an untyped graph root must not be migrated as a graph");

        assert_eq!(error.code, "graph_tree_chunk_version_unsupported");
        assert!(
            !target.exists(),
            "a refused migration must leave no partial target"
        );
        assert!(
            !lock_path_for(&target).exists(),
            "a refused migration must leave no lock sibling"
        );
        assert_eq!(
            fs::read(&source).expect("read source after"),
            before,
            "a refused migration must leave the source byte-identical"
        );
    }

    #[test]
    fn rr_migration_refuses_an_existing_target_and_leaves_it_alone() {
        let dir = migration_dir("exists");
        let (source, _) = legacy_store(&dir);
        let target = dir.join("Taken.RR");
        fs::write(&target, b"do not clobber me").expect("seed target");

        let error = migrate_journal_to_rr(&source, &target, &DeviceId::new("migrate test device"))
            .expect_err("an existing target must be refused");
        assert_eq!(error.code, MigrationRefusal::TargetExists.code());
        assert_eq!(
            fs::read(&target).expect("read target"),
            b"do not clobber me",
            "the refusal must not touch the file it refused"
        );
    }

    #[test]
    fn rr_migration_removes_a_partial_target_when_the_source_is_uncommitted() {
        let dir = migration_dir("uncommitted");
        let source = dir.join("empty.journal");
        drop(JournalChunkStore::open(&source).expect("open empty legacy"));
        let target = dir.join("Out.RR");

        let error = migrate_journal_to_rr(&source, &target, &DeviceId::new("migrate test device"))
            .expect_err("an uncommitted source must be refused");
        assert_eq!(error.code, MigrationRefusal::SourceUncommitted.code());
        assert!(!target.exists(), "no partial target may survive a refusal");
        assert!(
            !lock_path_for(&target).exists(),
            "no lock sibling may survive a refusal"
        );
    }
}
