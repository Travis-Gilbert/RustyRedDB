// W02 oracles for spec S32 (RR compaction) and S33 (existing RedCore
// migration), against O03.
//
// This file is `include!`d by `graph/tests.rs`, so it lands in
// `graph_store::tests` and shares one module scope with `foundation.rs`,
// `durability.rs`, `indexes_and_ttl.rs`, `live_substrate.rs`, `rr_file.rs` and
// `browser_storage.rs`. Their `use` statements are already in scope, so only
// names none of them bring are imported below.
//
// What these tests can and cannot reach: a native `.RR` is a chunk store with
// an identity block, and the only graph-level readers today take a data
// directory whose journal is a bare frame stream. Parity here is therefore
// content-addressed — the committed catalog, every named root, the reachable
// set, and the resolved file payload must match the source exactly, and nodes
// and edges live under the graph root whose address is compared. The
// node-level reopen of a native file is the product open path's job, not this
// node's, and is deliberately not faked here.

use crate::file_payload::{
    install_file_manifest_reference, read_file_payload, stage_file_payload, FileManifestRef,
    FILE_CONTENT_MANIFEST_PROPERTY,
};
use crate::journal::test_support::CountingSync;
use crate::rr_migrate::migrate_journal_to_rr;

/// A legacy data directory built by the real store: two nodes, one edge, one
/// graph-native File, and one unreachable object of the kind §31 says an
/// interrupted writer leaves behind.
struct RrMigrationFixture {
    dir: std::path::PathBuf,
    journal: std::path::PathBuf,
    device: DeviceId,
    body: Vec<u8>,
    manifest: FileManifestRef,
    orphan: Hash,
}

impl RrMigrationFixture {
    fn remove(self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

const ORPHAN_BYTES: &[u8] = b"unreachable object left behind by an interrupted writer";

fn rr_migration_fixture(label: &str) -> RrMigrationFixture {
    let dir = unique_test_dir(&format!("rr-migration-{label}"));
    let device = DeviceId::new("w02-fixture-device");
    // Larger than one 16 KiB target chunk, so the manifest has to carry
    // several terminals across the migration rather than a single body.
    let body = (0..40 * 1024_u32)
        .map(|index| u8::try_from(index % 251).expect("byte pattern"))
        .collect::<Vec<u8>>();
    let orphan = Hash::of(ORPHAN_BYTES);

    let (journal, manifest) = {
        let mut store = RedCoreGraphStore::open(
            &dir,
            RedCoreOptions {
                durability: RedCoreDurability::AofAlways,
                snapshot_interval_writes: 100,
                strict_acid: false,
            },
        )
        .expect("open the legacy store");

        store
            .upsert_node(NodeRecord::new(
                "node:alpha",
                ["Note"],
                json!({ "title": "alpha" }),
            ))
            .expect("node alpha");
        store
            .upsert_node(NodeRecord::new(
                "node:beta",
                ["Note"],
                json!({ "title": "beta" }),
            ))
            .expect("node beta");
        store
            .upsert_edge(EdgeRecord::new(
                "edge:alpha-references-beta",
                "node:alpha",
                "REFERENCES",
                "node:beta",
                json!({ "rank": 1 }),
            ))
            .expect("edge alpha -> beta");

        // One graph-native File: the payload is staged in the unified store,
        // the manifest reference goes onto the File node, and the graph commit
        // publishes the node that makes the payload reachable.
        let publication_key =
            format!("graph-node:file:notes/keep.txt:property:{FILE_CONTENT_MANIFEST_PROPERTY}");
        let chunk_store = store.unified_chunk_store();
        let staged = stage_file_payload(chunk_store.as_ref(), &publication_key, &body)
            .expect("stage the file payload");
        let reference = staged.reference().clone();
        let mut properties = json!({ "path": "notes/keep.txt", "title": "keep" });
        install_file_manifest_reference(
            properties.as_object_mut().expect("object properties"),
            &reference,
            0,
        )
        .expect("install the manifest reference");
        store
            .upsert_node(NodeRecord::new("file:notes/keep.txt", ["File"], properties))
            .expect("file node");
        staged.finish_after_publication();

        assert!(
            store.verify().expect("verify the legacy store").ok,
            "the fixture must verify before it is migrated"
        );

        // An interrupted writer's object: physically present, named by no
        // committed root, and therefore reclaimable only by compaction.
        let chunks = store.chunk_store_handle().expect("durable journal");
        chunks
            .write(&[(orphan, ORPHAN_BYTES)])
            .expect("write the unreachable object");
        (chunks.path().to_path_buf(), reference)
    };

    RrMigrationFixture {
        dir,
        journal,
        device,
        body,
        manifest,
        orphan,
    }
}

/// The source's committed state, read the way the migration reads it: through
/// read-only recovery, which reports the last committed root without repairing
/// a torn tail.
fn rr_migration_source_state(
    fixture: &RrMigrationFixture,
) -> (Hash, RootCatalog, std::collections::BTreeSet<Hash>, usize) {
    let source =
        JournalChunkStore::open_read_only_at(&fixture.journal, 0).expect("read-only source open");
    let catalog_root = source
        .read_root()
        .expect("read the committed root")
        .expect("the fixture commits a catalog");
    let catalog_bytes = source
        .get(catalog_root)
        .expect("resolve the catalog")
        .expect("the committed root names its catalog");
    let catalog = RootCatalog::from_bytes(catalog_bytes.as_ref()).expect("decode the catalog");
    let reachable = walk_committed_reachable(&source, &CoreReachabilityCodec)
        .expect("walk the committed reachable set");
    let chunk_count = source.chunk_count().expect("count chunks");
    (catalog_root, catalog, reachable.reachable, chunk_count)
}

#[test]
fn rr_migration_accepts_a_redcore_store_and_preserves_graph_files_roots_and_identity() {
    let fixture = rr_migration_fixture("accept");
    let (source_catalog_root, source_catalog, source_reachable, source_chunks) =
        rr_migration_source_state(&fixture);
    let source_before = std::fs::read(&fixture.journal).expect("read the source");
    let target = fixture.dir.join("Travis.RR");

    let receipt = migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect("the accepting path must migrate a store built by RedCoreGraphStore");

    assert!(receipt.source_preserved);
    assert_eq!(receipt.catalog_root, source_catalog_root);
    assert_eq!(
        receipt.chunks_copied,
        source_reachable.len(),
        "exactly the reachable set is copied"
    );
    assert!(
        source_chunks > receipt.chunks_copied,
        "the unreachable object must stay behind: {source_chunks} physical chunks, \
         {} reachable",
        receipt.chunks_copied
    );
    assert_eq!(receipt.named_roots, source_catalog.roots().len());
    assert!(
        receipt.named_roots >= 2,
        "a graph store commits more than one named root, got {}",
        receipt.named_roots
    );
    assert_eq!(
        std::fs::read(&fixture.journal).expect("read the source again"),
        source_before,
        "migration must leave the source byte-identical"
    );

    // Reopen independently: identity, catalog, and every named root. The
    // identity is the one the created file minted, and the receipt has to
    // report exactly what this independent open sees.
    let reopened = RustyRed::open_read_only(&target).expect("reopen the target");
    assert_eq!(
        reopened.database(),
        &receipt.database,
        "the receipt must report the identity the target carries"
    );
    assert!(
        !receipt.database.as_str().is_empty(),
        "a migrated database must carry a minted identity, not an empty one"
    );
    let target_catalog = reopened
        .catalog()
        .expect("read the target catalog")
        .expect("the target commits a catalog");
    assert_eq!(
        target_catalog.address().expect("target catalog address"),
        source_catalog_root
    );
    assert_eq!(
        target_catalog.roots(),
        source_catalog.roots(),
        "every named root must match the source"
    );

    // Graph parity at the level a native file can answer it: the reachable set
    // under the same graph root is identical, node for node and edge for edge,
    // because every object is content-addressed.
    let target_reachable =
        walk_committed_reachable(reopened.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk the target");
    assert_eq!(
        target_reachable.reachable, source_reachable,
        "the target must hold exactly the source's reachable set"
    );
    assert!(!target_reachable.reachable.contains(&fixture.orphan));
    assert!(
        reopened
            .store()
            .get(fixture.orphan)
            .expect("probe the target")
            .is_none(),
        "an unreachable source object must not appear in the target"
    );

    // Graph-native file parity: the migrated file's own store resolves the
    // manifest to the original bytes.
    assert_eq!(
        read_file_payload(reopened.store().as_ref(), &fixture.manifest).expect("read the file"),
        fixture.body,
        "a graph-native File must survive the migration intact"
    );

    fixture.remove();
}

#[test]
fn rr_migration_migrates_a_torn_source_and_leaves_it_byte_identical() {
    let fixture = rr_migration_fixture("torn");
    let (source_catalog_root, _, source_reachable, _) = rr_migration_source_state(&fixture);

    // A writer that died mid-frame: a record kind and a length with no payload.
    let mut torn = std::fs::read(&fixture.journal).expect("read the source");
    torn.extend_from_slice(&[
        0x01, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'p', b'a',
    ]);
    std::fs::write(&fixture.journal, &torn).expect("tear the source");
    let target = fixture.dir.join("Travis.RR");

    let receipt = migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect("a torn tail above the last commit must not prevent migration");

    assert!(receipt.source_preserved);
    assert_eq!(
        receipt.catalog_root, source_catalog_root,
        "the last committed state is what migrates"
    );
    assert_eq!(
        std::fs::read(&fixture.journal).expect("read the source again"),
        torn,
        "read-only recovery must not repair the torn tail it read"
    );

    let reopened = RustyRed::open_read_only(&target).expect("reopen the target");
    let target_reachable =
        walk_committed_reachable(reopened.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk the target");
    assert_eq!(target_reachable.reachable, source_reachable);
    assert_eq!(
        read_file_payload(reopened.store().as_ref(), &fixture.manifest).expect("read the file"),
        fixture.body
    );

    fixture.remove();
}

#[test]
fn rr_compaction_reclaims_unreachable_objects_and_keeps_identity_roots_and_files() {
    let fixture = rr_migration_fixture("compact");
    let target = fixture.dir.join("Travis.RR");
    migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect("migrate before compacting");

    // Writable open: compaction must run under the real single-writer claim,
    // which is what turns its rename into adopt -> publish_committed.
    let native = RustyRed::open(&target, &fixture.device).expect("open the native file writable");
    let identity = native.database().clone();
    let before_catalog = native
        .catalog()
        .expect("read the catalog")
        .expect("committed catalog");
    let before_reachable =
        walk_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk before compaction");
    let orphan = Hash::of(b"unreachable object written into the native file");
    native
        .store()
        .write(&[(orphan, b"unreachable object written into the native file")])
        .expect("write the unreachable object");
    assert!(
        native.store().get(orphan).expect("probe").is_some(),
        "precondition: the unreachable object is physically present"
    );

    let report = compact_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
        .expect("compaction must verify its own replacement");

    assert_eq!(
        report.compaction.chunks_removed, 1,
        "only the unreachable object is reclaimable"
    );
    assert_eq!(
        report.reachability.catalog_root,
        before_reachable.catalog_root
    );
    assert_eq!(
        report.reachability.reachable, before_reachable.reachable,
        "compaction must not change the reachable set"
    );
    assert_eq!(
        native.database(),
        &identity,
        "logical identity must not change because physical storage was compacted"
    );
    let after_catalog = native
        .catalog()
        .expect("read the catalog again")
        .expect("committed catalog");
    assert_eq!(after_catalog.roots(), before_catalog.roots());
    assert!(
        native.store().get(orphan).expect("probe after").is_none(),
        "compaction must remove the unreachable object"
    );
    assert_eq!(
        read_file_payload(native.store().as_ref(), &fixture.manifest).expect("read the file"),
        fixture.body,
        "the replacement must carry graph-native files through the rewrite"
    );
    assert!(
        native.is_writer(),
        "the claim must survive its own publish and still admit this writer"
    );

    // The compacted bytes at the path are the database: a fresh independent
    // read agrees with the writer's view.
    drop(native);
    let reread = RustyRed::open_read_only(&target).expect("reopen after compaction");
    assert_eq!(reread.database(), &identity);
    assert_eq!(
        read_file_payload(reread.store().as_ref(), &fixture.manifest).expect("read the file"),
        fixture.body
    );
    let reread_reachable =
        walk_committed_reachable(reread.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk after compaction");
    assert_eq!(reread_reachable.reachable, before_reachable.reachable);

    fixture.remove();
}

#[test]
fn rr_migration_refuses_a_corrupt_reachable_object_and_leaves_no_target() {
    let fixture = rr_migration_fixture("corrupt");
    let source_before = std::fs::read(&fixture.journal).expect("read the source");

    // Flip one byte inside the committed File body. The first terminal chunk's
    // bytes are stored verbatim in the journal, so the corruption lands in a
    // reachable object rather than in framing or in the unreachable orphan:
    // the walk can still enumerate it, and the copy is the step that has to
    // refuse it instead of publishing a database with different bytes under the
    // same addresses.
    let needle = &fixture.body[..64];
    let at = source_before
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("the committed file body is stored verbatim in the journal");
    let mut corrupt = source_before.clone();
    corrupt[at + 7] ^= 0xff;
    std::fs::write(&fixture.journal, &corrupt).expect("corrupt the source");
    let target = fixture.dir.join("Travis.RR");

    let error = migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect_err("a corrupt reachable object must not migrate");

    assert!(
        !target.exists(),
        "a refused migration must leave no target: {}",
        error.code
    );
    assert!(
        !lock_path_for(&target).exists(),
        "a refused migration must leave no lock sibling: {}",
        error.code
    );
    assert_eq!(
        std::fs::read(&fixture.journal).expect("read the source again"),
        corrupt,
        "a refused migration must leave the source exactly as it found it"
    );

    fixture.remove();
}

/// Every name beside the `.RR`, so a test can prove that a refused or
/// interrupted compaction left nothing new behind it.
fn rr_directory_entries(dir: &std::path::Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("list the fixture directory")
        .map(|entry| {
            entry
                .expect("directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

/// O03's replacement window on the header-carrying path: "atomically replace
/// only after fsync, crash in the window leaves a recoverable state".
///
/// The ordering under test is `compact_reachable` over
/// `rewrite_compacted_journal`: write the replacement, fsync it, adopt it into
/// the write claim, rename it over the live file, fsync the parent directory.
/// The two fsyncs bound the window and each is made to fail in turn through
/// the injected syncer, which is the only way to reach that instant on a
/// native `.RR`: the product open path hard-codes the platform syncer.
///
/// Before the rename, a failed fsync must leave the old inode as the database:
/// byte-identical, still holding the orphan, still the writer's, readable by an
/// independent reader, and with no replacement file beside it. After the
/// rename, a failed directory fsync is still reported, but the rename has
/// landed and the compacted file at the path is the database: an independent
/// reopen resolves the same identity, committed catalog, named roots,
/// reachable set and graph-native File, and a writable reopen takes the claim.
/// Each half is what a crash at that instant leaves on disk.
#[test]
fn rr_compaction_sync_failure_in_the_replacement_window_leaves_the_rr_recoverable() {
    let fixture = rr_migration_fixture("crash-window");
    let target = fixture.dir.join("Travis.RR");
    migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect("migrate before compacting");

    let syncer = Arc::new(CountingSync::default());
    let native = RustyRed::open_with_syncer(&target, &fixture.device, syncer.clone())
        .expect("open the native file writable through the injected syncer");
    let identity = native.database().clone();
    let (before_root, before_catalog) = rr_committed_catalog(native.store().as_ref());
    let before_reachable =
        walk_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk before compaction")
            .reachable;
    let orphan_bytes: &[u8] = b"unreachable object written before the interrupted compaction";
    let orphan = Hash::of(orphan_bytes);
    native
        .store()
        .write(&[(orphan, orphan_bytes)])
        .expect("write the unreachable object");
    let bytes_before = std::fs::read(&target).expect("read the native file");
    let entries_before = rr_directory_entries(&fixture.dir);
    assert!(
        lock_path_for(&target).exists(),
        "precondition: the writer's claim is labelled beside the file"
    );

    // Boundary one: the replacement's own fsync, before adoption and rename.
    syncer.fail_on_call(syncer.calls() + 1);
    let error = compact_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
        .err()
        .expect("a replacement that did not reach the platter must not be published");
    assert_eq!(error.code, "chunk_store_journal_io", "{}", error.message);
    assert!(
        error.message.contains("sync compacted journal"),
        "the refused step must be the replacement's fsync: {}",
        error.message
    );
    assert_eq!(
        std::fs::read(&target).expect("read the native file again"),
        bytes_before,
        "the old inode is still the database, byte for byte"
    );
    assert_eq!(
        rr_directory_entries(&fixture.dir),
        entries_before,
        "no replacement file may survive a refused publish"
    );
    assert!(
        native
            .store()
            .get(orphan)
            .expect("probe the orphan")
            .is_some(),
        "nothing was reclaimed: the orphan is still physically present"
    );
    let (root_after_refusal, catalog_after_refusal) = rr_committed_catalog(native.store().as_ref());
    assert_eq!(root_after_refusal, before_root);
    assert_eq!(catalog_after_refusal.roots(), before_catalog.roots());
    assert_eq!(native.database(), &identity);
    assert!(
        native.is_writer(),
        "the claim must fall back onto the old inode and still admit this writer"
    );
    {
        // An independent reader beside the live writer sees the old state and
        // nothing of the abandoned replacement.
        let reader = RustyRed::open_read_only(&target).expect("read beside the writer");
        assert_eq!(reader.database(), &identity);
        let (reader_root, reader_catalog) = rr_committed_catalog(reader.store().as_ref());
        assert_eq!(reader_root, before_root);
        assert_eq!(reader_catalog.roots(), before_catalog.roots());
        assert!(
            reader.store().get(orphan).expect("probe").is_some(),
            "the reader sees the orphan the refused compaction did not reclaim"
        );
    }

    // Boundary two: the parent-directory fsync after the rename has landed.
    // The first fsync of this attempt is the replacement's and succeeds; the
    // second is the directory's and fails. Nothing else syncs in between:
    // adoption, rename and publish are not fsyncs.
    syncer.fail_on_call(syncer.calls() + 2);
    let error = compact_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
        .err()
        .expect("a directory fsync that failed must be reported");
    assert_eq!(error.code, "chunk_store_journal_io", "{}", error.message);
    assert!(
        error.message.contains("journal compaction"),
        "the refused step must be the directory fsync: {}",
        error.message
    );
    let bytes_after = std::fs::read(&target).expect("read the native file after the rename");
    assert_ne!(
        bytes_after, bytes_before,
        "the rename landed: the path names the compacted body"
    );
    assert!(
        bytes_after.len() < bytes_before.len(),
        "the compacted body no longer carries the orphan's frame"
    );
    assert_eq!(
        rr_directory_entries(&fixture.dir),
        entries_before,
        "no replacement file may survive: it is the database now"
    );
    assert!(
        native.store().get(orphan).expect("probe after").is_none(),
        "the writer's handle is on the compacted body"
    );
    assert_eq!(native.database(), &identity);
    assert!(
        native.is_writer(),
        "the claim moved onto the compacted body and still admits this writer"
    );

    // Drop the writer and reopen from nothing but the bytes at the path.
    drop(native);
    assert!(
        !lock_path_for(&target).exists(),
        "a released claim leaves no lock sibling"
    );
    let reopened = RustyRed::open_read_only(&target)
        .expect("reopen the file the interrupted compaction left behind");
    assert_eq!(reopened.database(), &identity);
    let (after_root, after_catalog) = rr_committed_catalog(reopened.store().as_ref());
    assert_eq!(
        after_root, before_root,
        "the committed catalog address must survive the interrupted publish"
    );
    assert_eq!(
        after_catalog.roots(),
        before_catalog.roots(),
        "every named root must survive the interrupted publish"
    );
    let after_reachable =
        walk_committed_reachable(reopened.store().as_ref(), &CoreReachabilityCodec)
            .expect("walk the reopened file")
            .reachable;
    assert_eq!(
        after_reachable, before_reachable,
        "the reachable set must survive the interrupted publish"
    );
    assert!(
        reopened
            .store()
            .get(orphan)
            .expect("probe reopened")
            .is_none(),
        "the reopened file does not carry the orphan"
    );
    assert_eq!(
        read_file_payload(reopened.store().as_ref(), &fixture.manifest).expect("read the file"),
        fixture.body,
        "the graph-native File must survive the interrupted publish"
    );
    drop(reopened);
    let writer = RustyRed::open(&target, &fixture.device)
        .expect("a writable open after the interrupted compaction takes the claim");
    assert!(writer.is_writer());
    assert_eq!(writer.database(), &identity);
    drop(writer);

    fixture.remove();
}

/// The compaction-path twin of
/// `rr_migration_refuses_a_corrupt_reachable_object_and_leaves_no_target`.
///
/// One byte flipped inside a committed terminal chunk breaks that frame's CRC.
/// The open path already refuses such a file by name, because a bad CRC below
/// the committed root is corruption and not a torn tail, so to reach
/// compaction at all the corruption lands underneath a live writer, which is
/// the shape bit rot takes on a file that was clean when it was opened.
/// Compaction verifies what it copies and refuses rather than publishing a
/// database whose bytes do not hash to their addresses, and the refusal leaves
/// the file exactly as it found it with nothing beside it.
///
/// Recorded outcome for the reopen: `open_read_only` and `open` both refuse
/// with `chunk_store_journal_corrupt`, neither repairs the file, and the
/// refused writable open leaves no lock sibling.
#[test]
fn rr_compaction_refuses_a_corrupt_reachable_object_and_keeps_the_source() {
    let fixture = rr_migration_fixture("compact-corrupt");
    let target = fixture.dir.join("Travis.RR");
    migrate_journal_to_rr(&fixture.journal, &target, &fixture.device)
        .expect("migrate before compacting");

    let native = RustyRed::open(&target, &fixture.device).expect("open the native file writable");
    let identity = native.database().clone();
    let entries_before = rr_directory_entries(&fixture.dir);

    // The first terminal chunk of the committed File body is stored verbatim,
    // so the flip lands in a reachable object rather than in framing or in an
    // orphan. Written through a second descriptor: the writer's index still
    // names the chunk, and the read is what has to catch it.
    let pristine = std::fs::read(&target).expect("read the native file");
    let needle = &fixture.body[..64];
    let at = pristine
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("the committed file body is stored verbatim in the native file");
    let flipped = at + 7;
    {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&target)
            .expect("open the native file for corruption");
        std::os::unix::fs::FileExt::write_at(&file, &[pristine[flipped] ^ 0xff], flipped as u64)
            .expect("flip one byte");
        file.sync_all().expect("make the corruption durable");
    }
    let mut corrupt = pristine.clone();
    corrupt[flipped] ^= 0xff;
    assert_eq!(
        std::fs::read(&target).expect("read the corrupted file"),
        corrupt,
        "precondition: exactly one byte differs"
    );

    let error = compact_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
        .err()
        .expect("a corrupt reachable object must not be compacted");
    assert_eq!(
        error.code, "chunk_store_journal_corrupt",
        "{}",
        error.message
    );
    assert_eq!(
        std::fs::read(&target).expect("read the file after the refusal"),
        corrupt,
        "a refused compaction must leave the file exactly as it found it"
    );
    assert_eq!(
        rr_directory_entries(&fixture.dir),
        entries_before,
        "a refused compaction must leave no replacement file"
    );
    assert_eq!(native.database(), &identity);
    assert!(native.is_writer(), "the claim is untouched by a refusal");

    drop(native);
    assert!(
        !lock_path_for(&target).exists(),
        "a released claim leaves no lock sibling"
    );
    assert_eq!(
        std::fs::read(&target).expect("read the file after release"),
        corrupt,
        "releasing the writer must not repair or rewrite the file"
    );

    let error = RustyRed::open_read_only(&target)
        .err()
        .expect("a corrupt reachable object refuses a read-only open");
    assert_eq!(
        error.code, "chunk_store_journal_corrupt",
        "{}",
        error.message
    );
    let error = RustyRed::open(&target, &fixture.device)
        .err()
        .expect("a corrupt reachable object refuses a writable open");
    assert_eq!(
        error.code, "chunk_store_journal_corrupt",
        "{}",
        error.message
    );
    assert!(
        !lock_path_for(&target).exists(),
        "a refused writable open must not leave its claim label behind"
    );
    assert_eq!(
        std::fs::read(&target).expect("read the file after the refused opens"),
        corrupt,
        "a refused open must not repair mid-file corruption"
    );

    fixture.remove();
}

/// The committed `RootCatalog` of a store, read the way the migration reads it.
fn rr_committed_catalog(store: &dyn ChunkStore) -> (Hash, RootCatalog) {
    let root = store
        .read_root()
        .expect("read the committed root")
        .expect("the fixture commits a catalog");
    let bytes = store
        .get(root)
        .expect("resolve the catalog")
        .expect("the committed root names its catalog");
    (
        root,
        RootCatalog::from_bytes(bytes.as_ref()).expect("decode the catalog"),
    )
}

/// O03 parity for the fifth committed root, which is what generation 2 exists
/// to close.
///
/// Before this, `decode_catalog_children` admitted `graph`, `redcore`,
/// `documents` and `objects` and refused every other name by name. A real
/// desktop database carries third-party IndexedDB in the same `.RR` under
/// `RootCatalog.browser`, so the refusal did not merely skip web state: it
/// failed the whole walk, which made migration and compaction impossible for
/// the database shape the product actually writes.
///
/// What this does not prove: that the product opens a native file, that a
/// window or plugin reaches its roots, or that the database survives a crash.
/// Those are W24, W10 and the V02 crash-window property respectively.
#[test]
fn rr_migration_and_compaction_traverse_the_root_catalog_browser_root() {
    let directory = unique_test_dir("rr-migration-browser-root");
    let device = DeviceId::new("w02-browser-fixture-device");
    let scope = theorem_surface_contracts::desktop::StorageScope {
        partition: theorem_surface_contracts::desktop::BrowserPartitionId::new("w02-partition"),
        origin: theorem_surface_contracts::desktop::Origin::new("https://example.test"),
        database: "notes".to_owned(),
    };
    let residency = theorem_surface_contracts::desktop::PartitionResidency::Persistent;
    // Golden bytes from the pinned Servo codec, the same encoding the W03
    // shard uses: type tag, each ASCII code unit plus one.
    let key = browser_storage_api::EncodedKey::from_servo_bytes(vec![0x30, b'n' + 1, b'o' + 1]);
    let value = b"a record only the browser root can reach".to_vec();

    let (journal, source_browser_root) = {
        let mut store = RedCoreGraphStore::open(
            &directory,
            RedCoreOptions {
                durability: RedCoreDurability::AofAlways,
                snapshot_interval_writes: 100,
                strict_acid: false,
            },
        )
        .expect("open the legacy store");
        store
            .upsert_node(NodeRecord::new(
                "node:browser-parity",
                ["Note"],
                json!({ "title": "browser parity" }),
            ))
            .expect("graph node");

        // Publish a real browser hierarchy through the real broker, so the
        // browser root under test is the one the product's writers produce
        // rather than a hand-built envelope.
        let chunks = store.chunk_store_handle().expect("durable journal");
        let broker = browser_storage_api::BrowserStorageBroker::new(chunks.clone());
        let grant = broker
            .issue_grant(scope.clone(), residency)
            .expect("issue the storage grant");
        let receipt = broker
            .transact(
                &grant,
                &browser_storage_api::BrowserTransaction {
                    mutations: vec![
                        browser_storage_api::BrowserMutation::CreateStore {
                            schema: browser_storage_api::ObjectStoreSchema {
                                name: "notes".to_owned(),
                                key_path: None,
                                auto_increment: false,
                            },
                        },
                        browser_storage_api::BrowserMutation::SetVersion { version: 1 },
                        browser_storage_api::BrowserMutation::Put {
                            store: "notes".to_owned(),
                            key: key.clone(),
                            value: value.clone(),
                            overwrite: false,
                            next_key_generator: None,
                            index_updates: Vec::new(),
                        },
                    ],
                    ..Default::default()
                },
            )
            .expect("publish the browser hierarchy");
        assert!(receipt.changed, "the browser transaction must publish");

        let (_, catalog) = rr_committed_catalog(chunks.as_ref());
        let browser_root = catalog
            .get(browser_storage_api::ROOT_NAME_BROWSER)
            .expect("the source catalog carries a browser root");
        (chunks.path().to_path_buf(), browser_root)
    };
    let target = directory.join("Travis.RR");

    // The source walk descends the browser hierarchy rather than stopping at
    // the catalog. Four nodes: root, partition, origin, database.
    let source = JournalChunkStore::open_read_only_at(&journal, 0).expect("read-only source open");
    let (_, source_catalog) = rr_committed_catalog(&source);
    let source_reachable = walk_committed_reachable(&source, &CoreReachabilityCodec)
        .expect("walk a source that carries browser state")
        .reachable;
    assert!(
        source_reachable.contains(&source_browser_root),
        "the walk must reach the committed browser root"
    );

    migrate_journal_to_rr(&journal, &target, &device).expect("migrate with browser state present");

    // Independent reopen first: the migrated file must stand alone and answer
    // the same committed roots the source did.
    let migrated_catalog = {
        let reader = RustyRed::open_read_only(&target).expect("reopen the migrated file read-only");
        let (_, catalog) = rr_committed_catalog(reader.store().as_ref());
        assert_eq!(
            catalog.get(browser_storage_api::ROOT_NAME_BROWSER),
            Some(source_browser_root),
            "the migrated catalog must carry the source's browser root unchanged"
        );
        let migrated_reachable =
            walk_committed_reachable(reader.store().as_ref(), &CoreReachabilityCodec)
                .expect("walk the migrated file")
                .reachable;
        for address in &source_reachable {
            assert!(
                migrated_reachable.contains(address),
                "migration must carry reachable object {address}"
            );
        }
        catalog
    };

    // The property that was impossible before generation 2: compaction over a
    // database that holds web storage. Compaction writes, so it runs under the
    // real single-writer claim rather than a read-only handle.
    let native = RustyRed::open(&target, &device).expect("open the native file writable");
    let report = compact_committed_reachable(native.store().as_ref(), &CoreReachabilityCodec)
        .expect("compact a database that carries browser state");
    let compacted_catalog = native
        .catalog()
        .expect("read the catalog after compaction")
        .expect("committed catalog");
    assert_eq!(
        compacted_catalog.get(browser_storage_api::ROOT_NAME_BROWSER),
        Some(source_browser_root),
        "compaction must not move the browser root: {}",
        report.compaction.chunks_removed
    );
    assert_eq!(
        compacted_catalog.get(ROOT_NAME_GRAPH),
        migrated_catalog.get(ROOT_NAME_GRAPH),
        "compaction must not move the graph root"
    );

    // Addresses are not the record. Read it back through the same broker the
    // product uses, against the compacted file.
    let broker = browser_storage_api::BrowserStorageBroker::new(native.store().clone());
    let grant = broker
        .issue_grant(scope, residency)
        .expect("re-issue the storage grant against the compacted file");
    let records = broker
        .query(
            &grant,
            "notes",
            &browser_storage_api::QueryTarget::ObjectStore,
            &browser_storage_api::EncodedKeyRange::only(key),
            None,
        )
        .expect("query the migrated database");
    assert_eq!(records.len(), 1, "exactly the one committed record");
    assert_eq!(
        records[0].value, value,
        "the record outlives migration and compaction byte for byte"
    );
    drop(native);

    std::fs::remove_dir_all(&directory).ok();
}
