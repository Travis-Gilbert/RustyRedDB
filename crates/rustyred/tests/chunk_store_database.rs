//! A database over a caller's chunk store: the constructor a browser uses,
//! exercised here over the same projection store on native files.

use std::sync::Arc;

use rustyred::engine::browser_projection::ProjectionChunkStore;
use rustyred::engine::UnifiedChunkStore;
use rustyred::{Database, ErrorKind, NodeRecord, OpenOptions};
use serde_json::json;

fn open(directory: &std::path::Path) -> Database {
    let store: Arc<dyn UnifiedChunkStore> =
        Arc::new(ProjectionChunkStore::open_at(directory).expect("projection store"));
    Database::with_chunk_store(store, &OpenOptions::default()).expect("database")
}

#[test]
fn what_it_commits_is_there_when_the_store_is_opened_again() {
    let directory = tempfile::tempdir().expect("tempdir");
    let version = {
        let mut database = open(directory.path());
        database
            .upsert_node(NodeRecord::new(
                "org:a",
                ["Org"],
                json!({ "name": "Alder" }),
            ))
            .expect("commit");
        database.stats().expect("stats").version
    };

    let database = open(directory.path());
    let node = database
        .engine()
        .get_node("org:a")
        .expect("read")
        .expect("the committed node");
    assert_eq!(node.properties["name"], "Alder");
    assert_eq!(database.stats().expect("stats").version, version);
    assert!(database.verify().expect("verify").ok);
}

#[test]
fn it_has_no_path_and_refuses_compaction() {
    let directory = tempfile::tempdir().expect("tempdir");
    let mut database = open(directory.path());
    assert!(database.path().is_none());
    assert!(database.is_writable());
    let refused = database.compact().expect_err("not a native file");
    assert_eq!(refused.kind, ErrorKind::Storage);
    assert_eq!(refused.code, "compaction_unsupported");
}
