use rustyred::engine::NodeQuery;
use rustyred::{Database, ErrorKind, NodeRecord, OpenOptions};
use serde_json::json;

fn control_event_count(db: &Database) -> usize {
    db.engine()
        .query_nodes(NodeQuery::label("ControlEvent"))
        .unwrap()
        .len()
}

#[test]
fn control_events_follow_the_open_option() {
    let dir = tempfile::tempdir().unwrap();
    let mut logged = Database::create(dir.path().join("logged.rr")).unwrap();
    logged
        .upsert_node(NodeRecord::new("a", ["Note"], json!({"title": "A"})))
        .unwrap();
    assert!(control_event_count(&logged) > 0);

    let path = dir.path().join("index.rr");
    let options = OpenOptions::new().control_events(false);
    let mut index = Database::create_with(&path, &options).unwrap();
    index
        .upsert_node(NodeRecord::new("a", ["Note"], json!({"title": "A"})))
        .unwrap();
    assert_eq!(control_event_count(&index), 0);
    assert_eq!(index.stats().unwrap().nodes_total, 1);
    drop(index);

    let mut reopened = Database::open_with(&path, &options).unwrap();
    reopened
        .upsert_node(NodeRecord::new("b", ["Note"], json!({"title": "B"})))
        .unwrap();
    assert_eq!(control_event_count(&reopened), 0);
    assert_eq!(reopened.stats().unwrap().nodes_total, 2);
}

fn derived() -> OpenOptions {
    OpenOptions::new().control_events(false).keep_history(false)
}

fn note(round: usize) -> NodeRecord {
    NodeRecord::new(
        "note",
        ["Note"],
        json!({"title": format!("round {round}"), "body": "x".repeat(1000)}),
    )
}

#[test]
fn without_history_compaction_keeps_only_the_live_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("churn.rr");
    let mut db = Database::create_with(&path, &derived()).unwrap();
    db.designate_vector("Note", "embedding", 2).unwrap();
    db.put_file_node("file", "a.bin", b"payload\0bytes")
        .unwrap();
    for round in 0..200 {
        db.upsert_node(note(round)).unwrap();
    }
    let grown = std::fs::metadata(&path).unwrap().len();
    let report = db.compact().unwrap();
    let compacted = std::fs::metadata(&path).unwrap().len();
    assert!(report.chunks_removed > 0);
    assert!(
        compacted * 20 < grown,
        "compaction left {compacted} of {grown} bytes"
    );

    // What the file holds now is exactly the head.
    assert_eq!(
        db.engine().get_node("note").unwrap().unwrap().properties["title"],
        "round 199"
    );
    assert_eq!(db.read_file_node("file").unwrap(), b"payload\0bytes");
    assert_eq!(db.vector_designation_count(), 1);
    let history = db.engine().pin_graph_history().unwrap();
    assert!(matches!(
        history.coverage(),
        rustyred::engine::graph_history::GraphHistoryCoverage::Since {
            prior_history_unavailable: true,
            ..
        }
    ));
    drop(history);

    // Writes continue after compaction, and the file reopens whole.
    db.upsert_node(note(200)).unwrap();
    drop(db);
    let reopened = Database::open_with(&path, &derived()).unwrap();
    assert!(reopened.verify().unwrap().ok);
    assert_eq!(
        reopened
            .engine()
            .get_node("note")
            .unwrap()
            .unwrap()
            .properties["title"],
        "round 200"
    );
    assert_eq!(reopened.read_file_node("file").unwrap(), b"payload\0bytes");
    assert_eq!(reopened.vector_designation_count(), 1);
}

#[test]
fn with_history_compaction_keeps_every_revision() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kept.rr");
    let mut db = Database::create(&path).unwrap();
    for round in 0..3 {
        db.upsert_node(note(round)).unwrap();
    }
    let history = db.engine().pin_graph_history().unwrap();
    let first = history.entries_after(0, 10).unwrap()[0].commit.commit_hash;
    drop(history);
    db.compact().unwrap();
    let history = db.engine().pin_graph_history().unwrap();
    assert_eq!(history.entries_after(0, 10).unwrap().len(), 3);
    assert_eq!(
        history
            .revision(first)
            .unwrap()
            .get_node("note")
            .unwrap()
            .unwrap()
            .properties["title"],
        "round 0"
    );
}

#[test]
fn compaction_is_refused_without_a_writable_native_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro.rr");
    drop(Database::create(&path).unwrap());
    let mut reader = Database::open_read_only(&path).unwrap();
    assert_eq!(reader.compact().unwrap_err().kind, ErrorKind::ReadOnly);
    let mut memory = Database::in_memory();
    assert_eq!(memory.compact().unwrap_err().code, "compaction_unsupported");
}
