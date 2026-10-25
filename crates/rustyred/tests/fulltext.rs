#![cfg(feature = "fulltext")]

use std::path::Path;

use rustyred::{Database, ErrorKind, GraphMutation, GraphMutationBatch, NodeRecord, OpenOptions};
use serde_json::json;

fn derived() -> OpenOptions {
    OpenOptions::new().control_events(false).keep_history(false)
}

fn note(id: &str, title: &str, body: &str) -> NodeRecord {
    NodeRecord::new(id, ["Note"], json!({ "title": title, "body": body }))
}

fn indexed(path: &Path) -> Database {
    let mut db = Database::create_with(path, &derived()).unwrap();
    db.designate_fulltext("Note", &[("title", 5.0), ("body", 1.0)])
        .unwrap();
    db
}

/// Hit ids in rank order.
fn ranked(db: &Database, query: &str) -> Vec<String> {
    db.fulltext_search("Note", query, 20, None)
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

/// Hit ids, sorted, for queries whose order does not matter.
fn found(db: &Database, query: &str) -> Vec<String> {
    let mut ids = ranked(db, query);
    ids.sort();
    ids
}

fn snippet(db: &Database, query: &str) -> String {
    db.fulltext_search("Note", query, 1, None).unwrap()[0]
        .snippet
        .clone()
}

#[test]
fn every_word_must_match_and_the_last_may_be_a_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = indexed(&dir.path().join("index.rr"));
    db.upsert_node(note(
        "plan",
        "Project plan",
        "The launch checklist and budget",
    ))
    .unwrap();
    db.upsert_node(note("lunch", "Journal", "Lunch with the budget committee"))
        .unwrap();
    db.upsert_node(note("cafe", "Café", "state of the art espresso"))
        .unwrap();

    assert_eq!(found(&db, "budget"), ["lunch", "plan"]);
    assert_eq!(found(&db, "budget launch"), ["plan"]);
    assert_eq!(found(&db, "budget lau"), ["plan"]);
    assert_eq!(found(&db, "budg"), ["lunch", "plan"]);
    // Only the last word is a prefix.
    assert!(found(&db, "lau budget").is_empty());
    assert!(found(&db, "budget nothing").is_empty());
    // Case and accents fold.
    assert_eq!(found(&db, "CAFE"), ["cafe"]);
    assert_eq!(found(&db, "café"), ["cafe"]);
    // A word that splits into several terms is a phrase.
    assert_eq!(found(&db, "state-of-the-art"), ["cafe"]);
    assert!(found(&db, "art-of-the-state").is_empty());
    assert!(found(&db, "   ").is_empty());
    assert!(found(&db, "!!!").is_empty());
}

#[test]
fn boosted_fields_rank_first_and_arrays_are_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::create_with(dir.path().join("index.rr"), &derived()).unwrap();
    db.designate_fulltext("Note", &[("title", 5.0), ("aliases", 2.0), ("body", 1.0)])
        .unwrap();
    db.upsert_node(note("body", "Notes", "the garden was green in the garden"))
        .unwrap();
    db.upsert_node(note("title", "Garden", "notes")).unwrap();
    db.upsert_node(NodeRecord::new(
        "alias",
        ["Note"],
        json!({ "title": "Roadmap", "aliases": ["Big plan", "Quarterly goals"] }),
    ))
    .unwrap();

    assert_eq!(ranked(&db, "garden"), ["title", "body"]);
    assert_eq!(found(&db, "quarterly"), ["alias"]);
    assert_eq!(found(&db, "big plan"), ["alias"]);
}

#[test]
fn snippets_mark_matches_and_escape_the_text() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = indexed(&dir.path().join("index.rr"));
    db.upsert_node(note("x", "Snippet", "a <b>bold</b> claim about gardens"))
        .unwrap();
    db.upsert_node(note(
        "long",
        "Long",
        &format!("{}needle", "word ".repeat(100)),
    ))
    .unwrap();

    let marked = snippet(&db, "gardens");
    assert!(marked.contains("<mark>gardens</mark>"), "{marked}");
    assert!(marked.contains("&lt;b&gt;bold&lt;/b&gt;"), "{marked}");
    assert!(!marked.contains("<b>"), "{marked}");
    // A prefix marks the word it matched.
    assert!(snippet(&db, "gard").contains("<mark>gardens</mark>"));
    // A passage cut from the middle of the text says so.
    let cut = snippet(&db, "needle");
    assert!(
        cut.starts_with('…') && cut.ends_with("<mark>needle</mark>"),
        "{cut}"
    );
}

#[test]
fn within_keeps_only_the_given_ids() {
    let mut db = Database::in_memory();
    db.designate_fulltext("Note", &[("title", 1.0)]).unwrap();
    for id in ["a", "b", "c"] {
        db.upsert_node(note(id, "alpha", "")).unwrap();
    }
    let within = ["b".to_string(), "c".to_string(), "z".to_string()];
    let mut ids = db
        .fulltext_search("Note", "alpha", 10, Some(&within))
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, ["b", "c"]);
    assert!(db
        .fulltext_search("Note", "alpha", 10, Some(&[]))
        .unwrap()
        .is_empty());
    assert!(db
        .fulltext_search("Note", "alpha", 0, None)
        .unwrap()
        .is_empty());
}

#[test]
fn the_index_follows_deletes_and_label_changes() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = indexed(&dir.path().join("index.rr"));
    for id in ["a", "b", "c"] {
        db.upsert_node(note(id, "shared", "")).unwrap();
    }
    db.mutate(GraphMutationBatch::new([
        GraphMutation::NodeDelete("a".into()),
        GraphMutation::NodeUpsert(NodeRecord::new(
            "b",
            ["Other"],
            json!({ "title": "shared" }),
        )),
        GraphMutation::NodeUpsert(note("c", "shared retitled", "")),
        GraphMutation::NodeUpsert(note("d", "shared", "")),
        GraphMutation::NodeDelete("c".into()),
        GraphMutation::NodeDelete("d".into()),
    ]))
    .unwrap();
    // Whatever the batch left in the graph is exactly what the index holds.
    let live = ["a", "b", "c", "d"]
        .into_iter()
        .filter(|id| {
            db.engine().get_node(id).unwrap().is_some_and(|node| {
                !node.tombstone && node.labels.iter().any(|label| label == "Note")
            })
        })
        .collect::<Vec<_>>();
    assert!(!live.contains(&"a") && !live.contains(&"b"));
    assert_eq!(found(&db, "shared"), live);

    db.upsert_node(note("b", "shared again", "")).unwrap();
    assert!(found(&db, "shared").contains(&"b".to_string()));
}

#[test]
fn the_index_is_kept_through_reopen_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.rr");
    let mut db = indexed(&path);
    db.upsert_node(note("keep", "Keeper", "stays")).unwrap();
    for round in 0..50 {
        db.upsert_node(note("churn", &format!("round{round}"), "moves"))
            .unwrap();
    }
    drop(db);

    let mut db = Database::open_with(&path, &derived()).unwrap();
    assert_eq!(found(&db, "keeper"), ["keep"]);
    assert_eq!(found(&db, "round49"), ["churn"]);
    assert!(found(&db, "round48").is_empty());

    let report = db.compact().unwrap();
    assert!(report.chunks_removed > 0);
    assert_eq!(found(&db, "stays"), ["keep"]);
    db.upsert_node(note("after", "After compaction", ""))
        .unwrap();
    drop(db);

    let db = Database::open_with(&path, &derived()).unwrap();
    assert!(db.verify().unwrap().ok);
    assert_eq!(found(&db, "moves stays"), Vec::<String>::new());
    assert_eq!(found(&db, "compaction"), ["after"]);
    assert_eq!(found(&db, "round49"), ["churn"]);
}

#[test]
fn a_designation_made_before_any_write_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.rr");
    drop(indexed(&path));
    let mut db = Database::open_with(&path, &derived()).unwrap();
    db.upsert_node(note("a", "First", "")).unwrap();
    assert_eq!(found(&db, "first"), ["a"]);
}

#[test]
fn designations_are_checked_and_searches_need_one() {
    let mut db = Database::in_memory();
    let code = |result: Result<(), rustyred::Error>| result.unwrap_err().code;
    assert_eq!(
        code(db.designate_fulltext("Note", &[])),
        "fulltext_designation_invalid"
    );
    assert_eq!(
        code(db.designate_fulltext("Note", &[("title", 0.0)])),
        "fulltext_designation_invalid"
    );
    assert_eq!(
        code(db.designate_fulltext("Note", &[("title", 1.0), ("title", 2.0)])),
        "fulltext_designation_invalid"
    );
    assert_eq!(
        db.fulltext_search("Note", "x", 5, None).unwrap_err().code,
        "fulltext_not_designated"
    );

    db.designate_fulltext("Note", &[("title", 1.0)]).unwrap();
    db.upsert_node(note("a", "Alpha", "beta")).unwrap();
    db.designate_fulltext("Note", &[("title", 1.0)]).unwrap();
    assert_eq!(found(&db, "alpha"), ["a"]);
    // New fields reindex what is already there.
    db.designate_fulltext("Note", &[("body", 1.0)]).unwrap();
    assert_eq!(found(&db, "beta"), ["a"]);
    assert!(found(&db, "alpha").is_empty());
}

#[test]
fn a_read_only_handle_searches_but_cannot_designate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.rr");
    let mut db = indexed(&path);
    db.upsert_node(note("a", "Alpha", "")).unwrap();
    drop(db);

    let mut reader = Database::open_read_only(&path).unwrap();
    assert_eq!(found(&reader, "alpha"), ["a"]);
    let refused = reader
        .designate_fulltext("Note", &[("body", 1.0)])
        .unwrap_err();
    assert_eq!(refused.kind, ErrorKind::ReadOnly);
}
