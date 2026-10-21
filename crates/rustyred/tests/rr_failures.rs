use rustyred::{Database, ErrorKind, OpenOptions};

#[test]
fn a_corrupt_header_is_a_bounded_open_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.rr");
    let db = Database::create_with(&path, &OpenOptions::new().device_name("writer")).unwrap();
    drop(db);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..8].copy_from_slice(b"NOTREDDB");
    std::fs::write(&path, &bytes).unwrap();
    let err = match Database::open(&path) {
        Ok(_) => panic!("expected a corrupt header to fail"),
        Err(error) => error,
    };
    assert_eq!(err.kind, ErrorKind::Storage);
    assert!(!err.code.is_empty());
    assert!(!err.message.is_empty());
}

#[test]
fn corrupt_committed_content_is_refused_without_repair() {
    use rustyred::NodeRecord;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt-content.rr");
    let mut db = Database::create(&path).unwrap();
    db.upsert_node(NodeRecord::new(
        "n",
        ["Record"],
        serde_json::json!({"value": "committed"}),
    ))
    .unwrap();
    drop(db);
    let mut bytes = std::fs::read(&path).unwrap();
    // The immutable format-1 header precedes RRCJ records. Alter the first
    // chunk's encoded payload while retaining its committed root and CRC.
    assert_eq!(&bytes[4096..4100], b"RRCJ");
    bytes[4096 + 47] ^= 0x40;
    std::fs::write(&path, &bytes).unwrap();
    let err = match Database::open_read_only(&path) {
        Ok(_) => panic!("corrupt committed content must fail"),
        Err(error) => error,
    };
    assert_eq!(err.kind, ErrorKind::Storage);
    assert!(!err.code.is_empty());
    assert!(err.message.len() < 4096);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn an_interrupted_tail_is_not_published_or_repaired_by_a_reader() {
    use rustyred::NodeRecord;
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("torn-tail.rr");
    let mut db = Database::create(&path).unwrap();
    db.upsert_node(NodeRecord::new(
        "durable",
        ["Record"],
        serde_json::json!({}),
    ))
    .unwrap();
    drop(db);
    let clean = std::fs::read(&path).unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"RRCJ\x01\x01torn")
        .unwrap();
    let torn = std::fs::read(&path).unwrap();
    let reader = Database::open_read_only(&path).unwrap();
    assert!(reader.engine().get_node("durable").unwrap().is_some());
    assert_eq!(std::fs::read(&path).unwrap(), torn);
    drop(reader);
    let writer = Database::open(&path).unwrap();
    assert!(writer.engine().get_node("durable").unwrap().is_some());
    drop(writer);
    assert_eq!(std::fs::read(&path).unwrap(), clean);
}

#[cfg(feature = "test-fault-injection")]
mod injected {
    use super::*;
    use rustyred::NodeRecord;
    use serde_json::json;

    fn record(id: &str) -> NodeRecord {
        NodeRecord::new(id, ["Record"], json!({"value": id}))
    }

    fn seeded(path: &std::path::Path) -> Database {
        let mut db = Database::create(path).unwrap();
        db.upsert_node(record("durable")).unwrap();
        db
    }

    fn old_state_after_reopen(path: &std::path::Path) {
        let db = Database::open_read_only(path).unwrap();
        assert!(db.engine().get_node("durable").unwrap().is_some());
        assert!(db.engine().get_node("refused").unwrap().is_none());
        assert!(db.verify().unwrap().ok);
    }

    #[test]
    fn torn_chunk_append_rolls_back_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunk.rr");
        let mut db = seeded(&path);
        db.journal_for_test()
            .inject_append_failure_after(0, 7, false);
        let error = db.upsert_node(record("refused")).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Storage);
        assert_eq!(error.code, "chunk_store_injected_append_failure");
        assert!(db.engine().get_node("refused").unwrap().is_none());
        drop(db);
        old_state_after_reopen(&path);
    }

    #[test]
    fn failure_before_root_and_torn_root_leave_the_prior_commit() {
        for after_bytes in [0, 7] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("root.rr");
            let mut db = seeded(&path);
            db.journal_for_test()
                .inject_root_append_failure_for_test(after_bytes, false);
            let error = db.upsert_node(record("refused")).unwrap_err();
            assert_eq!(error.kind, ErrorKind::Storage);
            assert_eq!(error.code, "chunk_store_injected_append_failure");
            drop(db);
            old_state_after_reopen(&path);
        }
    }

    #[test]
    fn failed_file_and_vector_publication_do_not_publish_half_payload_transactions() {
        for file in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("payload-failure.rr");
            let mut db = seeded(&path);
            db.designate_vector("Part", "embedding", 3).unwrap();
            db.journal_for_test()
                .inject_root_append_failure_for_test(7, false);
            let result = if file {
                db.put_file_node("refused", "partial.bin", b"not-published\x00\xff")
            } else {
                db.upsert_node(NodeRecord::new(
                    "refused",
                    ["Part"],
                    json!({"embedding": [1.0, 0.0, 0.0]}),
                ))
            };
            assert_eq!(result.unwrap_err().kind, ErrorKind::Storage);
            drop(db);
            old_state_after_reopen(&path);
            let db = Database::open_read_only(&path).unwrap();
            assert!(db
                .engine()
                .vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 1)
                .unwrap()
                .is_empty());
            if file {
                assert_eq!(
                    db.read_file_node("refused").unwrap_err().kind,
                    ErrorKind::NotFound
                );
            }
        }
    }

    #[test]
    fn root_sync_failure_with_confirmed_rollback_keeps_the_prior_commit() {
        for after_sync in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("sync.rr");
            let mut db = seeded(&path);
            db.journal_for_test()
                .inject_root_sync_failure_for_test(after_sync, false);
            let error = db.upsert_node(record("refused")).unwrap_err();
            assert_eq!(error.kind, ErrorKind::Storage);
            assert_eq!(error.code, "chunk_store_journal_io");
            drop(db);
            old_state_after_reopen(&path);
        }
    }

    #[test]
    fn an_unconfirmed_durable_publication_is_typed_unknown_and_never_retried() {
        for vector in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("unknown.rr");
            let mut db = seeded(&path);
            db.designate_vector("Part", "embedding", 3).unwrap();
            let uncertain = if vector {
                NodeRecord::new("uncertain", ["Part"], json!({"embedding": [1.0, 0.0, 0.0]}))
            } else {
                record("uncertain")
            };
            // Sync the actual root, then lose its acknowledgement and refuse rollback.
            db.journal_for_test()
                .inject_root_sync_failure_for_test(true, true);
            let error = db.upsert_node(uncertain).unwrap_err();
            assert_eq!(
                error.kind,
                ErrorKind::CommitOutcomeUnknown,
                "vector={vector}: {error}"
            );
            assert_eq!(error.code, "chunk_store_journal_poisoned");
            assert_eq!(
                db.upsert_node(record("later")).unwrap_err().kind,
                ErrorKind::CommitOutcomeUnknown
            );
            drop(db);
            let db = Database::open_read_only(&path).unwrap();
            assert!(
                db.engine().get_node("uncertain").unwrap().is_some(),
                "the uncertain write actually landed"
            );
            assert!(db.engine().get_node("later").unwrap().is_none());
            if vector {
                assert_eq!(
                    db.engine()
                        .vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 1)
                        .unwrap()[0]
                        .0,
                    "uncertain"
                );
            }
            assert!(db.verify().unwrap().ok);
        }
    }

    #[test]
    fn committed_vector_write_stays_successful_when_pin_maintenance_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("maintenance.rr");
        let mut db = seeded(&path);
        db.designate_vector("Part", "embedding", 3).unwrap();
        db.journal_for_test()
            .inject_claimed_release_failure_for_test();
        db.upsert_node(NodeRecord::new(
            "part",
            ["Part"],
            json!({"embedding": [1.0, 0.0, 0.0]}),
        ))
        .unwrap();
        assert!(
            !db.journal_for_test()
                .claimed_release_failure_pending_for_test(),
            "the real cleanup seam was exercised"
        );
        drop(db);
        let db = Database::open_read_only(&path).unwrap();
        assert_eq!(
            db.engine()
                .vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 1)
                .unwrap()[0]
                .0,
            "part"
        );
        assert!(db.verify().unwrap().ok);
    }
}
