// W01 oracles for the native `.RR` file: spec S29 (native-file requirements),
// S30 (single-writer locking) and S31 (crash semantics), against O02.
//
// This file is `include!`d by `graph/tests.rs`, so it lands in
// `graph_store::tests` and shares one module scope with `foundation.rs`,
// `durability.rs`, `indexes_and_ttl.rs` and `live_substrate.rs`. Their `use`
// statements are already in scope here and a repeat would be a duplicate
// import, so only names none of them bring are imported below.

use crate::chunk_store::{CommitMeta, RootCatalog, ROOT_NAME_GRAPH};
use crate::hash::Hash;
use crate::journal::JournalChunkStore;
use crate::rr_file::{
    lock_path_for, mint_database_id, read_owner, RrHeader, RustyRed, RR_BODY_FORMAT_JOURNAL,
    RR_CAPABILITY_VERSION, RR_FORMAT_VERSION, RR_HEADER_LEN, RR_MAGIC, RR_MAX_DATABASE_ID_LEN,
};
use theorem_surface_contracts::desktop::{DatabaseId, DeviceId, RrAccess};

/// Passed to the child process of the cross-process lock test.
const RR_LOCK_CHILD_DATABASE_ENV: &str = "THEOREM_RR_LOCK_CHILD_DATABASE";
const RR_LOCK_CHILD_DEVICE_ENV: &str = "THEOREM_RR_LOCK_CHILD_DEVICE";
/// Printed by the child once every assertion has run. See its doc comment.
const RR_CHILD_VERIFIED: &str = "rr-file-child-verified";

// Header field offsets, written as literals rather than borrowed from
// `rr_file::header`, whose layout constants are private. A test that reused the
// implementation's own constants would follow the layout wherever it moved and
// could never catch a field shifting. These literals are the on-disk contract.
const FORMAT_VERSION_AT: usize = 8;
const CAPABILITY_VERSION_AT: usize = 10;
const BODY_FORMAT_AT: usize = 12;
const DATABASE_ID_LEN_AT: usize = 16;
const DATABASE_ID_AT: usize = 20;
const DIGEST_AT: usize = 4064;
/// The domain byte the header digest is computed over. Domain 7 was the free
/// slot in the crate's domain-separation table when W01 claimed it.
const DOMAIN_RR_HEADER: u8 = 7;

fn rr_test_path(label: &str) -> std::path::PathBuf {
    let dir = unique_test_dir(label);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("Travis.RR")
}

fn rr_clean_up(path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::remove_dir_all(parent).ok();
    }
}

/// Sign a header block in place, so a field edited by a test still decodes.
fn rr_resign(block: &mut [u8]) {
    let mut signed = Vec::with_capacity(1 + DIGEST_AT);
    signed.push(DOMAIN_RR_HEADER);
    signed.extend_from_slice(&block[..DIGEST_AT]);
    block[DIGEST_AT..].copy_from_slice(Hash::of(&signed).as_bytes());
}

/// Rewrite one `u16` header field and re-sign, producing a block that is
/// well-formed but declares something this build may not accept.
fn rr_resign_with_u16(block: &mut [u8], at: usize, value: u16) {
    block[at..at + 2].copy_from_slice(&value.to_le_bytes());
    rr_resign(block);
}

fn rr_header_block(identity: &str) -> Vec<u8> {
    RrHeader::new(DatabaseId::new(identity)).unwrap().encode()
}

fn rr_write_raw(path: &Path, bytes: &[u8]) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn rr_len(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}

fn rr_truncate_to(path: &Path, len: u64) {
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_len(len).unwrap();
    file.sync_all().unwrap();
}

/// The code and message of a call that must not have succeeded.
///
/// Takes no `Debug` bound because `RustyRed` holds a file handle and does not
/// carry one; `what` supplies the description the panic would otherwise print.
fn rr_refused<T>(what: &str, result: GraphStoreResult<T>) -> (String, String) {
    match result {
        Ok(_) => panic!("{what} was expected to refuse and did not"),
        Err(error) => (error.code, error.message),
    }
}

fn rr_put_chunk(db: &RustyRed, bytes: &[u8]) -> Hash {
    let address = Hash::of(bytes);
    db.store().write(&[(address, bytes)]).unwrap();
    address
}

/// Publish a one-entry catalog naming `graph` and return the catalog address.
fn rr_commit_graph_root(db: &RustyRed, graph: Hash, version: u64) -> Hash {
    let catalog = RootCatalog::graph(graph);
    let bytes = catalog.to_bytes().unwrap();
    let address = Hash::of(&bytes);
    db.store().write(&[(address, bytes.as_slice())]).unwrap();
    db.store()
        .commit(
            address,
            CommitMeta {
                parents: Vec::new(),
                author: "rr-file-test".to_owned(),
                message: format!("root {version}"),
                timestamp_unix_ms: 0,
                graph_version: version,
            },
        )
        .unwrap();
    address
}

// ---------------------------------------------------------------------------
// S29: the file is a database because of what it contains
// ---------------------------------------------------------------------------

#[test]
fn rr_file_header_encodes_to_one_fixed_block_and_decodes_back() {
    let header = RrHeader::new(DatabaseId::new("travis")).unwrap();
    let block = header.encode();

    assert_eq!(block.len() as u64, RR_HEADER_LEN);
    assert_eq!(block[..RR_MAGIC.len()], RR_MAGIC);
    assert_eq!(header.format_version(), RR_FORMAT_VERSION);
    assert_eq!(header.capability_version(), RR_CAPABILITY_VERSION);
    assert_eq!(header.body_format(), RR_BODY_FORMAT_JOURNAL);
    assert_eq!(RrHeader::decode(&block).unwrap(), header);

    // A minted identity is unguessable and fits the block by construction.
    let first = mint_database_id();
    let second = mint_database_id();
    assert_ne!(first, second);
    assert_eq!(first.as_str().len(), 32);
    assert!(first.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    assert!(RrHeader::new(first).is_ok());
}

#[test]
fn rr_file_create_then_reopen_carries_one_identity_through_a_rename() {
    let path = rr_test_path("rr-identity");
    let device = DeviceId::new("mac-studio");
    let minted;
    {
        let db = RustyRed::create(&path, &device).unwrap();
        assert!(db.is_writer());
        assert_eq!(db.ownership().database, *db.database());
        assert!(
            db.committed_root().unwrap().is_none(),
            "a fresh database has published no root"
        );
        assert!(db.catalog().unwrap().is_none());
        assert_eq!(db.path(), path.as_path());
        minted = db.database().clone();
    }

    // O02 puts the DatabaseId in the file rather than deriving it from the
    // path, so moving the file must not mint a second database.
    let moved = path.with_file_name("Renamed.RR");
    std::fs::rename(&path, &moved).unwrap();
    let reopened = RustyRed::open(&moved, &DeviceId::new("macbook")).unwrap();
    assert_eq!(reopened.database(), &minted);
    assert_eq!(reopened.header().format_version(), RR_FORMAT_VERSION);
    match &reopened.ownership().access {
        RrAccess::Writer { device } => assert_eq!(device, &DeviceId::new("macbook")),
        RrAccess::ReadOnly => panic!("a writable open must report Writer access"),
    }

    drop(reopened);
    rr_clean_up(&path);
}

#[test]
fn rr_file_create_refuses_to_overwrite_an_existing_database() {
    let path = rr_test_path("rr-already-exists");
    let device = DeviceId::new("mac-studio");
    let first = RustyRed::create(&path, &device).unwrap();
    let minted = first.database().clone();
    drop(first);

    let (code, message) = rr_refused("a second create", RustyRed::create(&path, &device));
    assert_eq!(code, "rr_already_exists");
    assert!(message.contains("open it instead"), "{message}");

    // The refusal must leave the database it declined to overwrite openable,
    // which is the whole reason it refuses rather than truncating.
    assert_eq!(RustyRed::open(&path, &device).unwrap().database(), &minted);
    rr_clean_up(&path);
}

#[test]
fn rr_file_truncated_header_is_refused_at_every_boundary() {
    let path = rr_test_path("rr-truncated-header");
    let block = rr_header_block("travis");

    // Every prefix short of the full block, including the ones that stop
    // inside the magic, inside the length field and one byte short of the
    // digest. Truncation is reported ahead of "not a database" on purpose: a
    // reader that has not got the bytes cannot know what they would have said.
    for short in [0_usize, 1, 7, 8, 15, 19, 20, 4063, 4095] {
        rr_write_raw(&path, &block[..short]);
        let (code, _) = rr_refused("a short header", RustyRed::open_read_only(&path));
        assert_eq!(code, "rr_header_truncated", "at {short} bytes");
    }

    rr_write_raw(&path, &block);
    assert_eq!(
        RustyRed::open_read_only(&path).unwrap().database(),
        &DatabaseId::new("travis")
    );
    rr_clean_up(&path);
}

#[test]
fn rr_file_corrupt_digest_is_reported_before_any_version_field() {
    let path = rr_test_path("rr-corrupt-digest");

    // One flipped bit anywhere under the digest is damage.
    let mut block = rr_header_block("travis");
    block[100] ^= 0x01;
    rr_write_raw(&path, &block);
    assert_eq!(
        rr_refused("a flipped byte", RustyRed::open_read_only(&path)).0,
        "rr_header_corrupt"
    );

    // The order matters. A torn block whose format field happens to read as a
    // future version must be reported as damage, not as a newer build: the
    // version field of a block that failed its digest is not evidence about
    // anything. The corollary is recorded in `header.rs`: a future layout that
    // moves the digest must take a new magic, not only bump the format.
    let mut block = rr_header_block("travis");
    block[FORMAT_VERSION_AT..FORMAT_VERSION_AT + 2]
        .copy_from_slice(&(RR_FORMAT_VERSION + 1).to_le_bytes());
    rr_write_raw(&path, &block);
    assert_eq!(
        rr_refused("an unsigned format bump", RustyRed::open_read_only(&path)).0,
        "rr_header_corrupt"
    );

    // Zeroing the digest is the same answer, not a bypass.
    let mut block = rr_header_block("travis");
    block[DIGEST_AT..].fill(0);
    rr_write_raw(&path, &block);
    assert_eq!(
        rr_refused("a zeroed digest", RustyRed::open_read_only(&path)).0,
        "rr_header_corrupt"
    );

    rr_clean_up(&path);
}

#[test]
fn rr_file_extension_spoofing_cannot_bypass_header_validation() {
    let path = rr_test_path("rr-extension-spoof");
    let device = DeviceId::new("mac-studio");

    // A bare journal carries "RRCJ" frames and no identity block. Naming it
    // `.RR` does not give it one.
    let mut journal_shaped = b"RRCJ".to_vec();
    journal_shaped.resize(8192, 0);
    rr_write_raw(&path, &journal_shaped);
    let (code, message) = rr_refused("a journal named .RR", RustyRed::open_read_only(&path));
    assert_eq!(code, "rr_not_a_database");
    assert!(message.contains("whatever its extension says"), "{message}");

    // Neither does a large file of unrelated bytes, on either open path.
    rr_write_raw(&path, &vec![b'x'; 8192]);
    assert_eq!(
        rr_refused("a foreign file, read-only", RustyRed::open_read_only(&path)).0,
        "rr_not_a_database"
    );
    assert_eq!(
        rr_refused("a foreign file, writable", RustyRed::open(&path, &device)).0,
        "rr_not_a_database"
    );

    // The writable refusal above ran after the claim was taken, so it also
    // proves the claim is surrendered on the refusal path. A leaked claim
    // would make every later open in this process fail.
    std::fs::remove_file(&path).unwrap();
    assert!(RustyRed::create(&path, &device).is_ok());
    rr_clean_up(&path);
}

#[test]
fn rr_file_unsupported_format_version_is_refused_outright() {
    let path = rr_test_path("rr-format-unsupported");
    let mut block = rr_header_block("travis");
    rr_resign_with_u16(&mut block, FORMAT_VERSION_AT, RR_FORMAT_VERSION + 1);
    rr_write_raw(&path, &block);

    // Outright, both ways. A layout this build cannot parse is not safely
    // readable either, which is exactly the line between format_version and
    // capability_version.
    assert_eq!(
        rr_refused("read-only", RustyRed::open_read_only(&path)).0,
        "rr_format_unsupported"
    );
    assert_eq!(
        rr_refused("writable", RustyRed::open(&path, &DeviceId::new("mac-studio"))).0,
        "rr_format_unsupported"
    );
    rr_clean_up(&path);
}

#[test]
fn rr_file_unsupported_capability_refuses_the_writer_and_admits_the_reader() {
    let path = rr_test_path("rr-capability-unsupported");
    let mut block = rr_header_block("travis");
    rr_resign_with_u16(&mut block, CAPABILITY_VERSION_AT, RR_CAPABILITY_VERSION + 1);
    rr_write_raw(&path, &block);

    // Reading state this build does not understand is safe; appending beside
    // it is not. So the capability refusal is scoped to writes.
    let reader = RustyRed::open_read_only(&path).unwrap();
    assert_eq!(
        reader.header().capability_version(),
        RR_CAPABILITY_VERSION + 1
    );
    assert!(!reader.is_writer());
    assert_eq!(reader.ownership().access, RrAccess::ReadOnly);
    drop(reader);

    let (code, message) = rr_refused(
        "a writer on a newer capability",
        RustyRed::open(&path, &DeviceId::new("mac-studio")),
    );
    assert_eq!(code, "rr_capability_unsupported");
    assert!(
        message.contains("read-only"),
        "the refusal must point at the open that would have worked: {message}"
    );
    rr_clean_up(&path);
}

#[test]
fn rr_file_unsupported_body_format_is_refused() {
    let path = rr_test_path("rr-body-format-unsupported");
    let mut block = rr_header_block("travis");
    rr_resign_with_u16(&mut block, BODY_FORMAT_AT, RR_BODY_FORMAT_JOURNAL + 1);
    rr_write_raw(&path, &block);

    assert_eq!(
        rr_refused("read-only", RustyRed::open_read_only(&path)).0,
        "rr_body_format_unsupported"
    );
    assert_eq!(
        rr_refused("writable", RustyRed::open(&path, &DeviceId::new("mac-studio"))).0,
        "rr_body_format_unsupported"
    );
    rr_clean_up(&path);
}

#[test]
fn rr_file_header_refuses_an_identity_it_cannot_carry() {
    let path = rr_test_path("rr-identity-invalid");

    // A declared length of zero, one past the block limit, and bytes that are
    // not UTF-8. Each is well-signed, so the digest cannot be what rejects it.
    let mut zero_length = rr_header_block("travis");
    zero_length[DATABASE_ID_LEN_AT..DATABASE_ID_LEN_AT + 4].copy_from_slice(&0_u32.to_le_bytes());
    rr_resign(&mut zero_length);

    let mut over_long = rr_header_block("travis");
    let past_limit = u32::try_from(RR_MAX_DATABASE_ID_LEN + 1).unwrap();
    over_long[DATABASE_ID_LEN_AT..DATABASE_ID_LEN_AT + 4].copy_from_slice(&past_limit.to_le_bytes());
    rr_resign(&mut over_long);

    let mut not_utf8 = rr_header_block("travis");
    not_utf8[DATABASE_ID_LEN_AT..DATABASE_ID_LEN_AT + 4].copy_from_slice(&4_u32.to_le_bytes());
    not_utf8[DATABASE_ID_AT..DATABASE_ID_AT + 4].copy_from_slice(&[0xff, 0xfe, 0xff, 0xfe]);
    rr_resign(&mut not_utf8);

    for (label, block) in [
        ("a zero-length identity", zero_length),
        ("an over-long identity", over_long),
        ("a non-UTF-8 identity", not_utf8),
    ] {
        rr_write_raw(&path, &block);
        assert_eq!(
            rr_refused(label, RustyRed::open_read_only(&path)).0,
            "rr_database_id_invalid",
            "{label}"
        );
    }

    // Refused at construction too, so a bad identity never reaches the disk.
    assert_eq!(
        rr_refused("an empty identity", RrHeader::new(DatabaseId::new(""))).0,
        "rr_database_id_invalid"
    );
    assert_eq!(
        rr_refused(
            "an over-long identity",
            RrHeader::new(DatabaseId::new("z".repeat(RR_MAX_DATABASE_ID_LEN + 1))),
        )
        .0,
        "rr_database_id_invalid"
    );
    rr_clean_up(&path);
}

// ---------------------------------------------------------------------------
// S30: one writer, and a refusal that names who holds it
// ---------------------------------------------------------------------------

#[test]
fn rr_file_lock_path_appends_rather_than_replacing_the_extension() {
    // `Path::with_extension` would turn `Travis.RR` into `Travis.lock`, which
    // collides with any sibling named `Travis.*`. The claim must be named
    // after the exact file it protects.
    let database = Path::new("/data/Travis.RR");
    assert_eq!(lock_path_for(database), Path::new("/data/Travis.RR.lock"));
    assert_ne!(lock_path_for(database), database.with_extension("lock"));

    // An extensionless database is still locked beside itself, and two
    // databases whose names differ only by extension get distinct claims.
    assert_eq!(
        lock_path_for(Path::new("/data/Travis")),
        Path::new("/data/Travis.lock")
    );
    assert_ne!(
        lock_path_for(Path::new("/data/Travis.RR")),
        lock_path_for(Path::new("/data/Travis.db"))
    );
}

#[test]
fn rr_file_second_writable_open_in_this_process_is_refused_by_name() {
    let path = rr_test_path("rr-same-process-lock");
    let device = DeviceId::new("mac-studio");
    let held = RustyRed::create(&path, &device).unwrap();

    // Named as this process rather than as another instance: the difference
    // between a bug in this program and a second copy of it running.
    let (code, message) = rr_refused(
        "a second writer",
        RustyRed::open(&path, &DeviceId::new("macbook")),
    );
    assert_eq!(code, "rr_writer_held");
    assert!(message.contains("in this process"), "{message}");

    // Readers take no claim, so any number sit beside the one writer.
    let first_reader = RustyRed::open_read_only(&path).unwrap();
    let second_reader = RustyRed::open_read_only(&path).unwrap();
    assert!(!first_reader.is_writer());
    assert_eq!(first_reader.database(), held.database());
    assert_eq!(second_reader.database(), held.database());
    drop((first_reader, second_reader));

    // Dropping the writer is what releases the file.
    drop(held);
    let next = RustyRed::open(&path, &DeviceId::new("macbook")).unwrap();
    assert!(next.is_writer());
    drop(next);
    rr_clean_up(&path);
}

#[test]
fn rr_file_owner_record_names_the_live_writer_and_never_guesses() {
    let path = rr_test_path("rr-owner-record");
    let device = DeviceId::new("mac-studio");
    let held = RustyRed::create(&path, &device).unwrap();

    let lock_path = lock_path_for(&path);
    assert!(lock_path.exists());
    let owner = read_owner(&lock_path).expect("a live writer publishes its identity");
    assert_eq!(owner.device, device);
    assert_eq!(owner.pid, std::process::id());
    drop(held);

    // Every malformed record answers "cannot say" rather than naming the wrong
    // device. A torn write loses a name; it must never invent one. The record
    // is a bare tagged line rather than JSON precisely so a torn write fails
    // the prefix check instead of parsing into something plausible.
    for garbage in [
        "",
        "\n",
        "rr-owner/1",
        "rr-owner/1 mac-studio",
        "rr-owner/2 mac-studio 7",
        "rr-owner/1 mac-studio not-a-pid",
        "{\"device\":\"mac-studio\",\"pid\":7}",
    ] {
        rr_write_raw(&lock_path, garbage.as_bytes());
        assert!(
            read_owner(&lock_path).is_none(),
            "{garbage:?} must not name an owner"
        );
    }

    // A record left by a dead process is still only consulted after the kernel
    // has already refused the claim, so a stale record cannot resurrect an
    // owner: the claim below succeeds despite the record naming someone.
    rr_write_raw(&lock_path, b"rr-owner/1 ghost-desk 999999\n");
    let reclaimed = RustyRed::open(&path, &device).unwrap();
    assert!(reclaimed.is_writer());
    assert_eq!(read_owner(&lock_path).unwrap().device, device);

    drop(reclaimed);
    rr_clean_up(&path);
}

/// The child half of [`rr_file_second_process_writer_is_refused_by_device_name`].
///
/// Returns immediately unless the parent named a database, so an ordinary test
/// run executes it as a no-op. On success it prints [`RR_CHILD_VERIFIED`],
/// which the parent requires: a child that returned early, or an `--exact`
/// filter that matched nothing after a rename, both exit zero and would
/// otherwise read as a pass without a single assertion having run.
#[test]
fn rr_file_second_process_writer_child() {
    let Some(raw_path) = std::env::var_os(RR_LOCK_CHILD_DATABASE_ENV) else {
        return;
    };
    let path = std::path::PathBuf::from(raw_path);
    let holder = std::env::var(RR_LOCK_CHILD_DEVICE_ENV).unwrap();

    // O02 requires the second writable open to be refused *by name* rather
    // than blocked, so the caller learns which device holds the file.
    let (code, message) = rr_refused(
        "a writer in a second process",
        RustyRed::open(&path, &DeviceId::new("child-desk")),
    );
    assert_eq!(code, "rr_writer_held");
    assert!(
        message.contains(&holder),
        "the refusal must name the holding device: {message}"
    );
    assert!(
        message.contains("pid"),
        "the refusal must name the holding process: {message}"
    );

    // The refusal is not a lockout. A reader still gets the committed state
    // while the other process writes.
    let reader = RustyRed::open_read_only(&path).unwrap();
    assert!(!reader.is_writer());
    assert!(
        reader.committed_root().unwrap().is_some(),
        "the reader must see the writer's published root"
    );
    assert!(reader.catalog().unwrap().is_some());

    // And it refuses writes by name rather than corrupting the writer's file.
    let (code, _) = rr_refused(
        "a write through a read-only open",
        reader
            .store()
            .write(&[(Hash::of(b"child"), b"child".as_slice())]),
    );
    assert_eq!(code, "rr_read_only_database");

    println!("{RR_CHILD_VERIFIED}");
}

#[test]
fn rr_file_second_process_writer_is_refused_by_device_name() {
    let path = rr_test_path("rr-cross-process-lock");
    // A name with a space in it, because that is what people call their
    // machines and because the first record format turned one into "a process
    // that has not published its identity" without saying so.
    let device = DeviceId::new("mac studio");
    let held = RustyRed::create(&path, &device).unwrap();
    let graph = rr_put_chunk(&held, b"cross-process graph bytes");
    rr_commit_graph_root(&held, graph, 1);

    // `Command::output` blocks until the child exits, so the claim is held for
    // the child's entire life by construction. That is the synchronisation:
    // there is no readiness handshake to race and no window in which the
    // parent has already let go.
    let output = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("graph_store::tests::rr_file_second_process_writer_child")
        .arg("--nocapture")
        .env(RR_LOCK_CHILD_DATABASE_ENV, &path)
        .env(RR_LOCK_CHILD_DEVICE_ENV, device.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child stdout:\n{}\nchild stderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(RR_CHILD_VERIFIED),
        "the child exited zero without reaching its assertions; stdout:\n{stdout}"
    );

    // The parent's claim outlived the child, and the child's read-only open
    // left the file unchanged.
    assert!(held.is_writer());
    assert!(held.store().has(graph).unwrap());
    drop(held);
    rr_clean_up(&path);
}

// ---------------------------------------------------------------------------
// S31: crash semantics, replayed one boundary at a time
// ---------------------------------------------------------------------------

#[test]
fn rr_file_crash_before_the_first_root_leaves_the_identity_block_intact() {
    let path = rr_test_path("rr-crash-before-root");
    let device = DeviceId::new("mac-studio");
    let minted;
    {
        let db = RustyRed::create(&path, &device).unwrap();
        minted = db.database().clone();
        rr_put_chunk(&db, b"an object written before any root was published");
    }

    // Tear the only frame in the body. Recovery must stop at the end of the
    // header, not at zero: a scan that began at offset 0 would take the
    // identity block with it and leave a file that is no longer a database.
    let body_len = rr_len(&path) - RR_HEADER_LEN;
    let torn = RR_HEADER_LEN + body_len / 2;
    assert!(torn > RR_HEADER_LEN && torn < rr_len(&path));
    rr_truncate_to(&path, torn);

    let recovered = RustyRed::open(&path, &device).unwrap();
    assert_eq!(recovered.database(), &minted);
    assert!(recovered.committed_root().unwrap().is_none());
    assert_eq!(
        rr_len(&path),
        RR_HEADER_LEN,
        "recovery must truncate to the body offset, never past it"
    );

    // The bytes still on disk are a header, not just an in-memory copy of one.
    let on_disk = RrHeader::decode(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(&on_disk, recovered.header());

    drop(recovered);
    rr_clean_up(&path);
}

#[test]
fn rr_file_crash_after_a_root_keeps_the_root_and_drops_the_torn_tail() {
    let path = rr_test_path("rr-crash-after-root");
    let device = DeviceId::new("mac-studio");
    let interrupted = b"an object the crash caught mid-frame";
    let graph;
    let catalog;
    let committed_len;
    {
        let db = RustyRed::create(&path, &device).unwrap();
        graph = rr_put_chunk(&db, b"committed graph bytes");
        catalog = rr_commit_graph_root(&db, graph, 1);
        committed_len = rr_len(&path);
        rr_put_chunk(&db, interrupted);
    }

    let torn = committed_len + (rr_len(&path) - committed_len) / 2;
    assert!(torn > committed_len && torn < rr_len(&path));
    rr_truncate_to(&path, torn);

    let recovered = RustyRed::open(&path, &device).unwrap();
    assert_eq!(recovered.store().read_root().unwrap(), Some(catalog));
    assert!(recovered.store().has(graph).unwrap());
    assert!(
        !recovered.store().has(Hash::of(interrupted)).unwrap(),
        "the interrupted append must not survive as a readable object"
    );
    assert_eq!(rr_len(&path), committed_len);
    assert_eq!(
        recovered.catalog().unwrap().unwrap().get(ROOT_NAME_GRAPH),
        Some(graph)
    );

    // The identity is unchanged by recovery, and the writer can go on writing.
    let next = rr_put_chunk(&recovered, b"an object written after recovery");
    assert!(recovered.store().has(next).unwrap());

    drop(recovered);
    rr_clean_up(&path);
}

#[test]
fn rr_file_read_only_open_reads_a_torn_tail_without_repairing_it() {
    let path = rr_test_path("rr-read-only-no-repair");
    let device = DeviceId::new("mac-studio");
    let graph;
    let catalog;
    let committed_len;
    {
        let db = RustyRed::create(&path, &device).unwrap();
        graph = rr_put_chunk(&db, b"committed graph bytes");
        catalog = rr_commit_graph_root(&db, graph, 1);
        committed_len = rr_len(&path);
        rr_put_chunk(&db, b"an object the crash caught mid-frame");
    }
    rr_truncate_to(&path, committed_len + (rr_len(&path) - committed_len) / 2);
    let torn_len = rr_len(&path);

    // A reader may sit beside a writer, so it must never rewrite the file it
    // is reading. It reads up to the tear and stops.
    let reader = RustyRed::open_read_only(&path).unwrap();
    assert_eq!(
        rr_len(&path),
        torn_len,
        "a read-only open must not repair, because the writer owns the file"
    );
    assert_eq!(reader.store().read_root().unwrap(), Some(catalog));
    assert!(reader.store().has(graph).unwrap());

    let (code, message) = rr_refused(
        "a write through a read-only open",
        reader
            .store()
            .write(&[(Hash::of(b"reader"), b"reader".as_slice())]),
    );
    assert_eq!(code, "rr_read_only_database");
    assert!(message.contains("refuses writes"), "{message}");
    assert_eq!(rr_len(&path), torn_len);
    drop(reader);

    // Opening writable is what repairs it.
    let writer = RustyRed::open(&path, &device).unwrap();
    assert_eq!(rr_len(&path), committed_len);
    assert_eq!(writer.store().read_root().unwrap(), Some(catalog));

    drop(writer);
    rr_clean_up(&path);
}

#[test]
fn rr_file_compaction_rewrites_the_body_and_keeps_the_identity_block() {
    use crate::chunk_store::ChunkStoreCompaction;

    let path = rr_test_path("rr-compaction-header");
    let device = DeviceId::new("mac-studio");
    let db = RustyRed::create(&path, &device).unwrap();
    let minted = db.database().clone();
    let graph = rr_put_chunk(&db, b"reachable graph bytes");
    let unreachable = rr_put_chunk(&db, b"bytes no root names, which compaction reclaims");
    let catalog = rr_commit_graph_root(&db, graph, 1);
    let before = rr_len(&path);

    let report = db
        .store()
        .compact_reachable(Some(catalog), &[graph, catalog])
        .unwrap();
    assert!(report.chunks_removed >= 1);
    assert!(!db.store().has(unreachable).unwrap());
    assert!(db.store().has(graph).unwrap());
    assert_eq!(db.store().read_root().unwrap(), Some(catalog));
    assert!(rr_len(&path) < before);

    // Compaction writes a fresh file and renames it over the live one. Without
    // copying the identity block across first, it would leave a valid frame
    // stream that is no longer an identifiable database.
    let head = std::fs::read(&path).unwrap();
    let on_disk = RrHeader::decode(&head[..RR_HEADER_LEN as usize]).unwrap();
    assert_eq!(on_disk.database(), &minted);

    drop(db);
    let reopened = RustyRed::open(&path, &device).unwrap();
    assert_eq!(reopened.database(), &minted);
    assert_eq!(reopened.store().read_root().unwrap(), Some(catalog));

    drop(reopened);
    rr_clean_up(&path);
}

// ---------------------------------------------------------------------------
// The single-writer boundary, attacked rather than described.
//
// The first version of this file asserted the *string* shape of the lock path
// and never that two spellings of one file collapse to one claim, so it passed
// against an implementation that let two writers in. These oracles name the
// property instead: whatever the claim is keyed on, one file is one writer.
// ---------------------------------------------------------------------------

/// A second writable open through a symlink to a held database.
#[cfg(unix)]
#[test]
fn rr_file_a_symlink_is_the_same_database_and_the_second_writer_is_refused() {
    let path = rr_test_path("rr-symlink-is-one-database");
    let device = DeviceId::new("mac-studio");
    drop(RustyRed::create(&path, &device).unwrap());

    let alias = path.parent().unwrap().join("Alias.RR");
    std::os::unix::fs::symlink(&path, &alias).unwrap();

    let held = RustyRed::open(&path, &device).unwrap();
    let (code, message) = rr_refused(
        "a writable open through a symlink to a held database",
        RustyRed::open(&alias, &DeviceId::new("second-device")),
    );
    assert_eq!(code, "rr_writer_held", "{message}");

    drop(held);
    rr_clean_up(&path);
}

/// A second writable open through a hard link to a held database.
#[test]
fn rr_file_a_hard_link_is_the_same_database_and_the_second_writer_is_refused() {
    let path = rr_test_path("rr-hard-link-is-one-database");
    let device = DeviceId::new("mac-studio");
    drop(RustyRed::create(&path, &device).unwrap());

    let alias = path.parent().unwrap().join("Alias.RR");
    std::fs::hard_link(&path, &alias).unwrap();

    let held = RustyRed::open(&path, &device).unwrap();
    let (code, message) = rr_refused(
        "a writable open through a hard link to a held database",
        RustyRed::open(&alias, &DeviceId::new("second-device")),
    );
    assert_eq!(code, "rr_writer_held", "{message}");

    drop(held);
    rr_clean_up(&path);
}

/// The store handle is reference counted, so the claim has to be too.
///
/// There are two ways out of the `RustyRed` that produced it, and a claim that
/// survives only one of them is not a claim. `Arc::clone` escapes the struct;
/// `JournalChunkStore` is itself `Clone`, so `(**db.store()).clone()` escapes
/// any wrapper put around it as well. Both are exercised here because the first
/// correction to this defect stopped the first route and left the second open.
#[test]
fn rr_file_a_cloned_store_keeps_the_write_claim_alive() {
    for route in ["arc", "handle"] {
        let path = rr_test_path(&format!("rr-cloned-store-keeps-claim-{route}"));
        let device = DeviceId::new("mac-studio");

        let db = RustyRed::create(&path, &device).unwrap();
        let store: std::sync::Arc<JournalChunkStore> = match route {
            "arc" => std::sync::Arc::clone(db.store()),
            _ => std::sync::Arc::new((**db.store()).clone()),
        };
        drop(db);

        // The handle still writes, so whatever authorised it must still be held.
        let bytes = format!("written through the {route} route after the owner went away");
        let bytes = bytes.as_bytes();
        store.write(&[(Hash::of(bytes), bytes)]).unwrap();

        let (code, message) = rr_refused(
            "a writable open beside a live cloned store",
            RustyRed::open(&path, &DeviceId::new("second-device")),
        );
        assert_eq!(code, "rr_writer_held", "via {route}: {message}");

        // And the claim is released once the last handle goes, not leaked.
        drop(store);
        drop(RustyRed::open(&path, &device).unwrap());
        rr_clean_up(&path);
    }
}

/// O02 asks the refusal to name the device, so the name has to survive the
/// record. The first format split the whole line on spaces and required exactly
/// three fields, which silently turned "mac studio" into no name at all.
#[test]
fn rr_file_an_owner_record_carries_a_name_a_person_would_choose() {
    let path = rr_test_path("rr-owner-record-names");

    for name in ["mac studio", "mac\nstudio", "back\\slash", "  leading space"] {
        let device = DeviceId::new(name);
        let held = RustyRed::create(&path, &device).unwrap();
        let owner = read_owner(&lock_path_for(&path))
            .unwrap_or_else(|| panic!("{name:?} must still name its holder"));
        assert_eq!(owner.device, device, "{name:?} did not survive the record");
        assert_eq!(owner.pid, std::process::id());
        drop(held);
        std::fs::remove_file(&path).unwrap();
    }

    // A device with no name cannot be named. Refusing here, where the caller
    // can still fix it, beats publishing a record that reads as "device  (pid
    // 7)" and beats degrading the refusal to "somebody, unspecified".
    let (code, message) = rr_refused(
        "a claim for an unnamed device",
        RustyRed::create(&path, &DeviceId::new("")),
    );
    assert_eq!(code, "rr_writer_device_unnamed", "{message}");
    assert!(
        !path.exists(),
        "the refused create left {} behind",
        path.display()
    );

    rr_clean_up(&path);
}

/// The label is removed on a clean close, which is only safe because it stopped
/// being the guarantee.
#[test]
fn rr_file_a_clean_close_leaves_no_lock_file_behind() {
    let path = rr_test_path("rr-clean-close");
    let device = DeviceId::new("mac-studio");

    let held = RustyRed::create(&path, &device).unwrap();
    let lock_path = lock_path_for(&path);
    assert!(lock_path.exists(), "a live writer publishes its identity");

    drop(held);
    assert!(
        !lock_path.exists(),
        "a cleanly closed database left {} behind",
        lock_path.display()
    );

    // And the next writer publishes a fresh one rather than inheriting a stale
    // record, which is what made the old 0-byte leftovers ambiguous.
    let reopened = RustyRed::open(&path, &device).unwrap();
    assert_eq!(read_owner(&lock_path).unwrap().device, device);
    drop(reopened);
    assert!(!lock_path.exists());

    rr_clean_up(&path);
}

/// An open that is going to be refused must not leave a claim's debris behind.
#[test]
fn rr_file_a_refused_open_leaves_no_lock_file_behind() {
    let path = rr_test_path("rr-refused-open-is-clean");
    let device = DeviceId::new("mac-studio");
    let directory = path.parent().unwrap().to_path_buf();

    // A file that carries the magic and fails validation: the claim is taken
    // before the header is read, so this is where debris used to appear.
    let intruder = directory.join("NotReally.RR");
    let mut block = rr_header_block("intruder");
    block[100] ^= 0xff;
    rr_write_raw(&intruder, &block);
    let (code, message) = rr_refused(
        "an open of a corrupt database",
        RustyRed::open(&intruder, &device),
    );
    assert_eq!(code, "rr_header_corrupt", "{message}");
    assert!(
        !lock_path_for(&intruder).exists(),
        "a refused open left a lock file beside {}",
        intruder.display()
    );

    // And a path with nothing at it at all.
    let absent = directory.join("Absent.RR");
    let (code, message) = rr_refused(
        "an open of an absent database",
        RustyRed::open(&absent, &device),
    );
    assert_eq!(code, "rr_open_io", "{message}");
    assert!(!lock_path_for(&absent).exists());

    rr_clean_up(&path);
}

/// The label path is derived, not chosen, so something can be sitting at it.
#[cfg(unix)]
#[test]
fn rr_file_the_owner_record_is_never_written_through_a_symlink() {
    let path = rr_test_path("rr-lock-symlink");
    let device = DeviceId::new("mac-studio");
    drop(RustyRed::create(&path, &device).unwrap());

    let bystander = path.parent().unwrap().join("bystander.txt");
    let bytes = b"bytes that belong to somebody else".as_slice();
    rr_write_raw(&bystander, bytes);
    let lock_path = lock_path_for(&path);
    std::os::unix::fs::symlink(&bystander, &lock_path).unwrap();

    let (code, message) = rr_refused(
        "an open whose lock path is a symlink",
        RustyRed::open(&path, &device),
    );
    assert_eq!(code, "rr_writer_lock_io", "{message}");
    assert_eq!(
        std::fs::read(&bystander).unwrap(),
        bytes,
        "the lock path was followed and the file it pointed at was destroyed"
    );

    rr_clean_up(&path);
}

/// A create that fails after taking its claim must not occupy the path it
/// failed on, because `create` then refuses it as existing and `open` refuses
/// it as unreadable, and nothing can reach it again.
#[test]
fn rr_file_a_create_that_fails_after_the_claim_leaves_the_path_creatable() {
    let path = rr_test_path("rr-failed-create-is-clean");
    let device = DeviceId::new("mac-studio");

    // Occupying the label path with a directory fails the create after the
    // database file has already been made.
    let occupied = path
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
        .join("Travis.RR.lock");
    std::fs::create_dir(&occupied).unwrap();

    let (code, message) = rr_refused(
        "a create whose label path is occupied",
        RustyRed::create(&path, &device),
    );
    assert_eq!(code, "rr_writer_lock_io", "{message}");
    assert!(
        !path.exists(),
        "the failed create left a half-made database at {}",
        path.display()
    );

    std::fs::remove_dir(&occupied).unwrap();
    let db = RustyRed::create(&path, &device).unwrap();
    assert_eq!(
        lock_path_for(&path),
        occupied,
        "the test occupied a path the implementation does not use"
    );
    drop(db);
    rr_clean_up(&path);
}

/// Compaction publishes by renaming a fresh file over the live one, which
/// changes the inode. A claim keyed on the file has to move with it, or the
/// holder is left locking an inode nobody can reach and the next opener is
/// granted a claim on a database that already has a writer.
#[test]
fn rr_file_compaction_does_not_hand_the_claim_to_the_next_writer() {
    use crate::chunk_store::ChunkStoreCompaction;

    let path = rr_test_path("rr-compaction-keeps-claim");
    let device = DeviceId::new("mac-studio");
    let held = RustyRed::create(&path, &device).unwrap();

    let graph = rr_put_chunk(&held, b"reachable graph bytes");
    rr_put_chunk(&held, b"bytes no root names, which compaction reclaims");
    let catalog = rr_commit_graph_root(&held, graph, 1);
    held.store()
        .compact_reachable(Some(catalog), &[graph, catalog])
        .unwrap();

    let (code, message) = rr_refused(
        "a writable open after the holder compacted",
        RustyRed::open(&path, &DeviceId::new("second-device")),
    );
    assert_eq!(code, "rr_writer_held", "{message}");

    // And the holder is still the writer, not a handle onto a stranded inode.
    let more = rr_put_chunk(&held, b"written after compaction by the same holder");
    assert!(held.store().has(more).unwrap());

    drop(held);
    drop(RustyRed::open(&path, &device).unwrap());
    rr_clean_up(&path);
}

/// Set or clear the macOS immutable flag, which is what makes a rename onto
/// this path fail without touching anything else about it.
#[cfg(target_os = "macos")]
fn rr_set_immutable(path: &Path, immutable: bool) {
    use std::os::unix::ffi::OsStrExt;

    let encoded = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    let flags = if immutable { libc::UF_IMMUTABLE } else { 0 };
    let outcome = unsafe { libc::chflags(encoded.as_ptr(), flags) };
    assert_eq!(
        outcome,
        0,
        "chflags on {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
}

/// Clears the flag however the test ends, so a failure does not leave an
/// immutable file in the scratch directory for the next run to trip over.
#[cfg(target_os = "macos")]
struct RrImmutable<'a>(&'a Path);

#[cfg(target_os = "macos")]
impl Drop for RrImmutable<'_> {
    fn drop(&mut self) {
        rr_set_immutable(self.0, false);
    }
}

/// A publish that cannot land leaves the claim on the file the path still names.
///
/// Compaction takes a second claim on the compacted body before renaming it
/// into place, because letting go of the first one early leaves the live file
/// unclaimed for the length of the rename. That makes the rename a decision
/// point: if it does not land, the old inode is still the database and the
/// claim belongs back on it. Nothing else in the suite reaches that branch,
/// because a rename between siblings in one directory does not ordinarily fail.
///
/// Made deterministic by the destination rather than by timing. The immutable
/// flag on the database file leaves the temp file creation and the rewrite
/// alone, both of which happen elsewhere in the directory, and fails exactly
/// `rename` onto it with EPERM. V01 reached the same window with a watcher
/// thread racing a 3.3 MB compaction and `chflags uchg` on the parent
/// directory; flagging the destination needs no thread and no race.
#[cfg(target_os = "macos")]
#[cfg_attr(miri, ignore = "chflags is not one of miri's supported shims")]
#[test]
fn rr_file_a_publish_that_cannot_land_leaves_the_claim_on_the_live_database() {
    use crate::chunk_store::ChunkStoreCompaction;

    let path = rr_test_path("rr-publish-cannot-land");
    let device = DeviceId::new("mac studio");
    let held = RustyRed::create(&path, &device).unwrap();

    let graph = rr_put_chunk(&held, b"reachable graph bytes");
    rr_put_chunk(&held, b"bytes no root names, which compaction would reclaim");
    let catalog = rr_commit_graph_root(&held, graph, 1);

    let before = std::fs::read(&path).unwrap();

    let error = {
        let _immutable = RrImmutable(&path);
        rr_set_immutable(&path, true);
        held.store()
            .compact_reachable(Some(catalog), &[graph, catalog])
            .unwrap_err()
    };
    assert_eq!(error.code, "chunk_store_journal_io", "{}", error.message);
    assert!(
        error.message.contains("replace journal during compaction"),
        "the failure should name the publish, not something earlier: {}",
        error.message
    );

    // The publish is the only thing that failed. The database is the file it
    // always was.
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a publish that did not land must leave the database byte-identical"
    );

    // The claim is the thing under test. It was adopted onto a replacement that
    // no longer exists, and it has to be back on this file.
    let (code, message) = rr_refused(
        "a writable open after a publish that could not land",
        RustyRed::open(&path, &DeviceId::new("second-device")),
    );
    assert_eq!(code, "rr_writer_held", "{message}");

    // And the holder still holds a database it can write, not a stranded inode.
    let more = rr_put_chunk(&held, b"written after the publish could not land");
    assert!(held.store().has(more).unwrap());

    drop(held);
    drop(RustyRed::open(&path, &device).unwrap());
    rr_clean_up(&path);
}

/// Renaming a database while it is open moves the file, and the claim follows.
///
/// The claim is on the inode and the registry key is the inode's device and
/// number, so both survive the rename and a second opener at the new name is
/// refused exactly. The label is at a path and does not follow, so it is left
/// at the old name. Inside this process that costs nothing, as asserted below.
/// From another process it costs the *name*: that opener finds no record at the
/// new path and is told "another process that has not published its identity".
/// The refusal is still exact; only O02's naming degrades, which is a weaker
/// answer and never a wrong one. Written down here as a known limit of a
/// path-derived label rather than left for the next head to hit.
#[test]
fn rr_file_a_rename_under_a_live_writer_moves_the_file_not_the_claim() {
    let path = rr_test_path("rr-rename-under-writer");
    let device = DeviceId::new("mac-studio");
    let held = RustyRed::create(&path, &device).unwrap();

    let renamed = path.parent().unwrap().join("Renamed.RR");
    std::fs::rename(&path, &renamed).unwrap();

    let (code, message) = rr_refused(
        "a writable open of a renamed but still-held database",
        RustyRed::open(&renamed, &DeviceId::new("second-device")),
    );
    assert_eq!(code, "rr_writer_held", "{message}");
    assert!(
        message.contains("in this process"),
        "the registry key is the inode, so a rename must not make the holder \
         look like a different process: {message}"
    );

    // The holder still writes to the file it claimed, under its new name.
    let bytes = b"written after the file was renamed underneath the writer".as_slice();
    let address = Hash::of(bytes);
    held.store().write(&[(address, bytes)]).unwrap();
    assert!(held.store().has(address).unwrap());

    drop(held);
    let reopened = RustyRed::open(&renamed, &device).unwrap();
    assert!(reopened.store().has(address).unwrap());
    drop(reopened);
    rr_clean_up(&renamed);
}
