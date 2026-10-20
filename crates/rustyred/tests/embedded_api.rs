use std::collections::BTreeMap;

use rustyred::{
    Database, EdgeRecord, ErrorKind, GraphMutation, GraphMutationBatch, NodeRecord, OpenOptions,
};
use serde_json::json;

fn opts() -> OpenOptions {
    OpenOptions::new().device_name("writer-a")
}

fn must_fail<T, E>(result: Result<T, E>) -> E {
    match result {
        Ok(_) => panic!("expected this call to fail"),
        Err(error) => error,
    }
}

#[test]
fn new_database_has_no_application_bootstrap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh.rr");
    let db = Database::create_with(&path, &opts()).unwrap();
    let inspection = db.inspect().unwrap();
    assert_eq!(inspection.nodes, 0);
    assert_eq!(inspection.edges, 0);
    assert_eq!(inspection.version, 0);
    assert!(db.vector_designation_count() == 0);
    assert!(inspection.database_id.is_some());
    let rows = db.query("MATCH (n) RETURN n", &BTreeMap::new()).unwrap();
    assert!(rows.rows.is_empty());
}

#[test]
fn create_refuses_an_existing_file_and_open_refuses_a_missing_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.rr");
    let missing = must_fail(Database::open(&path));
    assert_eq!(missing.kind, ErrorKind::NotFound);
    assert!(!path.exists(), "a missing open must not create a file");

    let db = Database::create_with(&path, &opts()).unwrap();
    drop(db);
    let again = must_fail(Database::create_with(&path, &opts()));
    assert_eq!(again.kind, ErrorKind::AlreadyExists);
}

#[test]
fn second_writer_is_refused_until_the_first_handle_drops() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lock.rr");
    let first = Database::create_with(&path, &opts()).unwrap();
    let second = must_fail(Database::open_with(
        &path,
        &OpenOptions::new().device_name("writer-b"),
    ));
    assert_eq!(second.code, "rr_writer_held");
    drop(first);
    let _reopened = Database::open_with(&path, &opts()).unwrap();
}

#[test]
fn read_only_mutation_is_refused_and_verify_does_not_change_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro.rr");
    let mut db = Database::create_with(&path, &opts()).unwrap();
    db.upsert_node(NodeRecord::new("n1", ["Part"], json!({"name": "bolt"})))
        .unwrap();
    drop(db);
    let before = std::fs::read(&path).unwrap();
    let mut reader = Database::open_read_only(&path).unwrap();
    let err = reader
        .upsert_node(NodeRecord::new("n2", ["Part"], json!({"name": "nut"})))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ReadOnly);
    assert_eq!(must_fail(reader.engine_mut()).kind, ErrorKind::ReadOnly);
    let reachable_store = reader.engine().unified_chunk_store();
    let bytes = b"a read-only engine must not offer an escape hatch";
    let address = rustyred_thg_core::hash::Hash::of(bytes);
    assert_eq!(
        reachable_store.write(&[(address, bytes)]).unwrap_err().code,
        "rr_read_only_database"
    );
    assert!(reachable_store.get(address).unwrap().is_none());
    drop(reachable_store);
    let _ = reader.verify().unwrap();
    let _ = reader
        .query("MATCH (p:Part) RETURN p", &BTreeMap::new())
        .unwrap();
    drop(reader);
    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after);
}

#[test]
fn queries_reject_writes_bad_text_and_the_empty_graph_is_empty() {
    let mut db = Database::in_memory();
    let write = db
        .query("CREATE (p:Part {name: 'bolt'})", &BTreeMap::new())
        .unwrap_err();
    assert_eq!(write.code, "cypher_write_unsupported");
    let bad = db.query("NOT CYPHER AT ALL", &BTreeMap::new()).unwrap_err();
    assert_eq!(bad.kind, ErrorKind::Query);
    db.upsert_node(NodeRecord::new("part:1", ["Part"], json!({"name": "bolt"})))
        .unwrap();
    db.upsert_node(NodeRecord::new(
        "supplier:1",
        ["Supplier"],
        json!({"name": "acme"}),
    ))
    .unwrap();
    db.mutate(GraphMutationBatch::new([GraphMutation::EdgeUpsert(
        EdgeRecord::new("s:1", "supplier:1", "SUPPLIES", "part:1", json!({})),
    )]))
    .unwrap();
    let rows = db
        .query(
            "MATCH (s:Supplier)-[:SUPPLIES]->(p:Part) RETURN s, p",
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
}

#[test]
fn wrong_vector_dimensions_are_refused() {
    let mut db = Database::in_memory();
    db.designate_vector("Part", "embedding", 3).unwrap();
    let err = db
        .upsert_node(NodeRecord::new(
            "part:vec",
            ["Part"],
            json!({"embedding": [1.0, 0.0]}),
        ))
        .unwrap_err();
    assert_eq!(err.code, "dimension_mismatch");
    db.upsert_node(NodeRecord::new(
        "part:vec",
        ["Part"],
        json!({"embedding": [1.0, 0.0, 0.0]}),
    ))
    .unwrap();
}

#[test]
fn directory_reopen_keeps_a_committed_node() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let mut db = Database::create_directory_with(&path, &opts()).unwrap();
    db.upsert_node(NodeRecord::new("part:9", ["Part"], json!({"sku": "b-9"})))
        .unwrap();
    drop(db);
    let missing = must_fail(Database::open_directory(dir.path().join("absent")));
    assert_eq!(missing.kind, ErrorKind::NotFound);
    let db = Database::open_directory(&path).unwrap();
    let rows = db
        .query("MATCH (p:Part) RETURN p", &BTreeMap::new())
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
}

#[test]
fn journal_migration_preserves_source_roots_payloads_history_and_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("legacy");
    let mut db = Database::create_directory_with(&store, &opts()).unwrap();
    db.designate_vector("Part", "embedding", 3).unwrap();
    db.upsert_node(NodeRecord::new(
        "part:m",
        ["Part"],
        json!({"sku": "original", "embedding": [1.0, 0.0, 0.0]}),
    ))
    .unwrap();
    let original = db
        .engine()
        .pin_graph_history()
        .unwrap()
        .head_roots()
        .logical_commit;
    db.put_file_node("file:m", "migration.bin", b"migration\x00\xffbytes")
        .unwrap();
    db.upsert_node(NodeRecord::new("supplier:m", ["Supplier"], json!({})))
        .unwrap();
    db.mutate(GraphMutationBatch::new([GraphMutation::EdgeUpsert(
        EdgeRecord::new(
            "edge:m",
            "supplier:m",
            "SUPPLIES",
            "part:m",
            json!({"quantity": 7}),
        ),
    )]))
    .unwrap();
    db.upsert_node(NodeRecord::new(
        "part:m",
        ["Part"],
        json!({"sku": "migrated", "embedding": [1.0, 0.0, 0.0]}),
    ))
    .unwrap();
    let roots = db.engine().unified_chunk_store().read_root().unwrap();
    drop(db);
    let journal = store.join("redcore.journal");
    let before = std::fs::read(&journal).unwrap();
    let target = dir.path().join("migrated.rr");
    let report = Database::migrate_journal(&journal, &target, &opts()).unwrap();
    assert!(report.source_preserved);
    assert_eq!(std::fs::read(&journal).unwrap(), before);
    let db = Database::open_read_only(&target).unwrap();
    assert_eq!(
        db.inspect().unwrap().database_id.as_deref(),
        Some(report.database_id.as_str())
    );
    assert_eq!(
        db.engine().unified_chunk_store().read_root().unwrap(),
        roots
    );
    assert_eq!(
        db.read_file_node("file:m").unwrap(),
        b"migration\x00\xffbytes"
    );
    assert_eq!(
        db.engine().get_edge("edge:m").unwrap().unwrap().properties["quantity"],
        7
    );
    assert_eq!(
        db.engine()
            .vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 1)
            .unwrap()[0]
            .0,
        "part:m"
    );
    let historical = db
        .engine()
        .pin_graph_history()
        .unwrap()
        .revision(original)
        .unwrap();
    assert_eq!(
        historical.get_node("part:m").unwrap().unwrap().properties["sku"],
        "original"
    );
    assert_eq!(
        db.engine().get_node("part:m").unwrap().unwrap().properties["sku"],
        "migrated"
    );
    assert!(db.verify().unwrap().ok);
}

#[cfg(feature = "test-fault-injection")]
#[test]
fn rr_compaction_preserves_pinned_history_vectors_and_file_payloads() {
    use rustyred_thg_core::chunk_store::ChunkStore;
    use rustyred_thg_core::file_payload::{
        read_file_payload, FileManifestRef, FILE_CONTENT_MANIFEST_PROPERTY,
    };
    use rustyred_thg_core::hash::Hash;
    use rustyred_thg_core::reachability::{compact_committed_reachable, CoreReachabilityCodec};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compact.rr");
    let mut db = Database::create(&path).unwrap();
    db.designate_vector("Part", "embedding", 3).unwrap();
    db.upsert_node(NodeRecord::new(
        "part",
        ["Part"],
        json!({"embedding": [1.0, 0.0, 0.0]}),
    ))
    .unwrap();
    db.put_file_node("file", "old.bin", b"old\0\xffpayload")
        .unwrap();
    let pinned = db.engine().pin_graph_history().unwrap();
    let original = pinned.head_roots().logical_commit;
    db.upsert_node(NodeRecord::new(
        "part",
        ["Part"],
        json!({"embedding": [0.0, 1.0, 0.0]}),
    ))
    .unwrap();
    db.put_file_node("file", "new.bin", b"new\0\xfepayload")
        .unwrap();
    let catalog = db.journal_for_test().read_root().unwrap();
    let orphan = Hash::of(b"unreachable bytes");
    db.journal_for_test()
        .write(&[(orphan, b"unreachable bytes")])
        .unwrap();
    let report =
        compact_committed_reachable(db.journal_for_test(), &CoreReachabilityCodec).unwrap();
    assert!(report.compaction.chunks_removed >= 1);
    assert_eq!(db.journal_for_test().read_root().unwrap(), catalog);
    assert!(db.journal_for_test().get(orphan).unwrap().is_none());
    let old = pinned.revision(original).unwrap();
    let old_file = old.get_node("file").unwrap().unwrap();
    let old_ref =
        FileManifestRef::from_value(&old_file.properties[FILE_CONTENT_MANIFEST_PROPERTY]).unwrap();
    assert_eq!(
        read_file_payload(db.engine().unified_chunk_store().as_chunk_store(), &old_ref).unwrap(),
        b"old\0\xffpayload"
    );
    let old_part = old.get_node("part").unwrap().unwrap();
    let payloads = rustyred_thg_core::object_store::DiskObjectStore::from_chunk_store(
        db.engine().unified_chunk_store(),
    )
    .unwrap();
    assert_eq!(
        rustyred_thg_core::vector_payload::read_vector_property(&old_part, "embedding", &payloads)
            .unwrap()
            .unwrap(),
        [1.0, 0.0, 0.0]
    );
    drop(payloads);
    assert_eq!(db.read_file_node("file").unwrap(), b"new\0\xfepayload");
    assert_eq!(
        db.engine()
            .vector_search(Some("Part"), "embedding", &[0.0, 1.0, 0.0], 1)
            .unwrap()[0]
            .0,
        "part"
    );
    drop(old);
    drop(pinned);
    drop(db);
    let reopened = Database::open_read_only(&path).unwrap();
    assert_eq!(
        reopened.read_file_node("file").unwrap(),
        b"new\0\xfepayload"
    );
    let historical = reopened
        .engine()
        .pin_graph_history()
        .unwrap()
        .revision(original)
        .unwrap();
    assert_eq!(historical.get_node("file").unwrap().unwrap(), old_file);
    let payloads = rustyred_thg_core::object_store::DiskObjectStore::from_chunk_store(
        reopened.engine().unified_chunk_store(),
    )
    .unwrap();
    assert_eq!(
        rustyred_thg_core::vector_payload::read_vector_property(
            &historical.get_node("part").unwrap().unwrap(),
            "embedding",
            &payloads
        )
        .unwrap()
        .unwrap(),
        [1.0, 0.0, 0.0]
    );
    assert!(reopened.verify().unwrap().ok);
}

#[test]
fn writable_chunk_clone_keeps_ownership_after_database_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cloned.rr");
    let db = Database::create(&path).unwrap();
    let writable_clone = db.engine().unified_chunk_store();
    drop(db);
    assert_eq!(must_fail(Database::open(&path)).code, "rr_writer_held");
    drop(writable_clone);
    drop(Database::open(&path).unwrap());
}

#[cfg(unix)]
#[test]
fn path_aliases_do_not_acquire_a_second_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("original.rr");
    let db = Database::create(&path).unwrap();
    let link = dir.path().join("symbolic.rr");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    let hard = dir.path().join("hard.rr");
    std::fs::hard_link(&path, &hard).unwrap();
    assert_eq!(must_fail(Database::open(&link)).code, "rr_writer_held");
    assert_eq!(must_fail(Database::open(&hard)).code, "rr_writer_held");
    drop(db);
    assert!(
        Database::open_read_only(&link)
            .unwrap()
            .verify()
            .unwrap()
            .ok
    );
}

#[cfg(feature = "test-fault-injection")]
#[test]
fn rr_compaction_refuses_unverified_named_root_without_changing_file() {
    use rustyred_thg_core::chunk_store::{
        validate_catalog_root, ChunkStore, ChunkStoreCatalog, CommitMeta,
    };
    use rustyred_thg_core::hash::Hash;
    use rustyred_thg_core::reachability::{compact_committed_reachable, CoreReachabilityCodec};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unknown-root.rr");
    let mut db = Database::create(&path).unwrap();
    db.upsert_node(NodeRecord::new(
        "retained",
        ["Record"],
        json!({"value": 42}),
    ))
    .unwrap();
    let opaque = b"future subsystem root with unverified reachability";
    let opaque_address = Hash::of(opaque);
    let orphan = b"must survive when native compaction is refused";
    let orphan_address = Hash::of(orphan);
    let journal = db.journal_for_test();
    journal
        .write(&[(opaque_address, opaque), (orphan_address, orphan)])
        .unwrap();
    let publication = journal
        .commit_catalog_updates(
            &BTreeMap::from([("future-subsystem".to_string(), Some(opaque_address))]),
            CommitMeta {
                parents: vec![],
                author: "storage-oracle".to_string(),
                message: "publish a synthetic future subsystem root".to_string(),
                timestamp_unix_ms: 1,
                graph_version: db.stats().unwrap().version,
            },
        )
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    let entries_before = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();

    let error = compact_committed_reachable(journal, &CoreReachabilityCodec).unwrap_err();
    assert_eq!(error.code, "reachability_catalog_name_unsupported");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(journal.read_root().unwrap(), Some(publication.catalog_root));
    assert_eq!(
        journal.get(opaque_address).unwrap().unwrap().as_ref(),
        opaque
    );
    assert_eq!(
        journal.get(orphan_address).unwrap().unwrap().as_ref(),
        orphan
    );
    assert_eq!(
        std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<std::collections::BTreeSet<_>>(),
        entries_before,
        "a refused compaction must not leave replacement files"
    );
    drop(db);

    let reader = Database::open_read_only(&path).unwrap();
    let store = reader.engine().unified_chunk_store();
    assert_eq!(store.read_root().unwrap(), Some(publication.catalog_root));
    let catalog = validate_catalog_root(store.as_chunk_store(), publication.catalog_root).unwrap();
    assert_eq!(catalog.get("future-subsystem"), Some(opaque_address));
    assert_eq!(store.get(opaque_address).unwrap().unwrap().as_ref(), opaque);
    assert_eq!(store.get(orphan_address).unwrap().unwrap().as_ref(), orphan);
    assert_eq!(
        reader
            .engine()
            .get_node("retained")
            .unwrap()
            .unwrap()
            .properties["value"],
        42
    );
    assert!(reader.verify().unwrap().ok);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}
