//! Cross-process proof: the closed `.RR` copy alone retains the database.
use std::collections::BTreeMap;
use std::process::Command;
use std::{env, fs};

use rustyred::{Database, EdgeRecord, GraphMutation, GraphMutationBatch, NodeRecord, OpenOptions};
use rustyred_thg_core::hash::Hash;
use serde_json::{json, Value};

const BODY: &[u8] = b"bolt-drawing-v1-\x00\xff binary";

fn main() {
    match env::var("RR_ROUNDTRIP_ROLE").ok().as_deref() {
        Some("reopen") => reopen(),
        Some("second-writer") => second_writer(),
        _ => rr_graph_and_payload_roundtrip_across_processes(),
    }
}

fn rr_graph_and_payload_roundtrip_across_processes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.rr");
    let first_commit;
    {
        let mut db =
            Database::create_with(&path, &OpenOptions::new().device_name("parent")).unwrap();
        db.designate_vector("Part", "embedding", 3).unwrap();
        db.upsert_node(NodeRecord::new(
            "part:bolt",
            ["Part"],
            json!({"name": "bolt-v1", "embedding": [1.0, 0.0, 0.0]}),
        ))
        .unwrap();
        first_commit = db
            .engine()
            .pin_graph_history()
            .unwrap()
            .head_roots()
            .logical_commit
            .to_prefixed();
        db.upsert_node(NodeRecord::new(
            "part:nut",
            ["Part"],
            json!({"name": "nut", "embedding": [0.0, 1.0, 0.0]}),
        ))
        .unwrap();
        db.upsert_node(NodeRecord::new(
            "supplier:acme",
            ["Supplier"],
            json!({"name": "acme"}),
        ))
        .unwrap();
        db.mutate(GraphMutationBatch::new([GraphMutation::EdgeUpsert(
            EdgeRecord::new(
                "supplies:bolt",
                "supplier:acme",
                "SUPPLIES",
                "part:bolt",
                json!({"quantity": 42}),
            ),
        )]))
        .unwrap();
        db.put_file_node("file:drawing", "drawings/bolt.bin", BODY)
            .unwrap();
        db.upsert_node(NodeRecord::new(
            "part:bolt",
            ["Part"],
            json!({"name": "bolt-v2", "embedding": [1.0, 0.0, 0.0]}),
        ))
        .unwrap();
        assert!(db.verify().unwrap().ok);
    }
    let held = Database::open(&path).unwrap();
    let status = Command::new(env::current_exe().unwrap())
        .env("RR_ROUNDTRIP_ROLE", "second-writer")
        .env("RR_ROUNDTRIP_PATH", &path)
        .status()
        .unwrap();
    assert_eq!(
        status.code(),
        Some(3),
        "another process must not acquire the writer"
    );
    drop(held);

    // Copy into a fresh directory: no lock, cache, journal or payload sidecar can help.
    let copied_dir = tempfile::tempdir().unwrap();
    let copied = copied_dir.path().join("copy.rr");
    fs::copy(&path, &copied).unwrap();
    assert_eq!(fs::read_dir(copied_dir.path()).unwrap().count(), 1);
    let output = Command::new(env::current_exe().unwrap())
        .env("RR_ROUNDTRIP_ROLE", "reopen")
        .env("RR_ROUNDTRIP_PATH", &copied)
        .env("RR_ROUNDTRIP_FIRST_COMMIT", &first_commit)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["file_bytes"], json!(BODY));
    assert_eq!(report["current_name"], "bolt-v2");
    assert_eq!(report["historical_name"], "bolt-v1");
    assert_eq!(report["edge"]["from_id"], "supplier:acme");
    assert_eq!(report["edge"]["to_id"], "part:bolt");
    assert_eq!(report["edge"]["properties"]["quantity"], 42);
    assert_eq!(report["nearest"][0][0], "part:bolt");
    assert_eq!(report["nearest"][1][0], "part:nut");
    assert_eq!(report["parts"], 2);
    assert!(report["verified"].as_bool().unwrap());
    assert_eq!(
        fs::read_dir(copied_dir.path()).unwrap().count(),
        1,
        "read-only child writes no sidecars"
    );
}

fn reopen() {
    let path = env::var("RR_ROUNDTRIP_PATH").unwrap();
    let before = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let rows = db
        .query("MATCH (p:Part) RETURN p", &BTreeMap::new())
        .unwrap();
    let history = db.engine().pin_graph_history().unwrap();
    let original = Hash::parse_prefixed(&env::var("RR_ROUNDTRIP_FIRST_COMMIT").unwrap()).unwrap();
    let old = history
        .revision(original)
        .unwrap()
        .get_node("part:bolt")
        .unwrap()
        .unwrap();
    let part = db.engine().get_node("part:bolt").unwrap().unwrap();
    let report = json!({
        "parts": rows.rows.len(),
        "file_bytes": db.read_file_node("file:drawing").unwrap(),
        "current_name": part.properties["name"],
        "historical_name": old.properties["name"],
        "edge": db.engine().get_edge("supplies:bolt").unwrap().unwrap(),
        "nearest": db.engine().vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 2).unwrap(),
        "verified": db.verify().unwrap().ok,
    });
    drop(history);
    drop(db);
    assert_eq!(fs::read(path).unwrap(), before);
    println!("{report}");
}

fn second_writer() {
    let path = env::var("RR_ROUNDTRIP_PATH").unwrap();
    match Database::open_with(&path, &OpenOptions::new().device_name("intruder")) {
        Err(err) if err.code == "rr_writer_held" => std::process::exit(3),
        Ok(_) => std::process::exit(0),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
