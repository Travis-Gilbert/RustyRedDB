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
fn journal_migration_preserves_source_bytes_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("legacy");
    let mut db = Database::create_directory_with(&store, &opts()).unwrap();
    db.upsert_node(NodeRecord::new(
        "part:m",
        ["Part"],
        json!({"sku": "migrated"}),
    ))
    .unwrap();
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
    let rows = db
        .query("MATCH (p:Part) RETURN p", &BTreeMap::new())
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
}
