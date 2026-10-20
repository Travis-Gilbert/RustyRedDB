//! Cross-process proof that a closed `.RR` file is the database.
//!
//! `cargo test --test rr_roundtrip` runs `main`. The child is this same binary
//! with `RR_ROUNDTRIP_ROLE=reopen`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::Command;

use rustyred::{Database, NodeRecord, OpenOptions};
use serde_json::{json, Value};

fn main() {
    match env::var("RR_ROUNDTRIP_ROLE").ok().as_deref() {
        Some("reopen") => reopen(),
        Some("second-writer") => second_writer(),
        _ => parent(),
    }
}

fn parent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("catalog.rr");
    let body = b"bolt-drawing-v1-\x00\xff binary";
    {
        let mut db =
            Database::create_with(&path, &OpenOptions::new().device_name("parent")).unwrap();
        db.designate_vector("Part", "embedding", 3).unwrap();
        db.upsert_node(NodeRecord::new(
            "part:bolt",
            ["Part"],
            json!({"name": "bolt", "embedding": [1.0, 0.0, 0.0]}),
        ))
        .unwrap();
        db.upsert_node(NodeRecord::new(
            "supplier:acme",
            ["Supplier"],
            json!({"name": "acme"}),
        ))
        .unwrap();
        db.put_file_node("file:drawing", "drawings/bolt.bin", body)
            .unwrap();
        assert!(db.verify().unwrap().ok);
    }

    let held = Database::open_with(&path, &OpenOptions::new().device_name("holder")).unwrap();
    let status = Command::new(env::current_exe().unwrap())
        .env("RR_ROUNDTRIP_ROLE", "second-writer")
        .env("RR_ROUNDTRIP_PATH", &path)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(3), "a second process must be refused");
    drop(held);

    let copied = dir.path().join("catalog-copy.rr");
    fs::copy(&path, &copied).unwrap();
    let output = Command::new(env::current_exe().unwrap())
        .env("RR_ROUNDTRIP_ROLE", "reopen")
        .env("RR_ROUNDTRIP_PATH", &copied)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("child json");
    assert_eq!(report["file_len"], body.len());
    assert_eq!(report["part_id"], "part:bolt");
    assert_eq!(report["vector_dimension"], 3);
    assert_eq!(report["nodes"], 6);
    let listed = report["listed"].as_array().expect("listed nodes");
    let ids = listed
        .iter()
        .map(|node| node["id"].as_str().unwrap_or(""))
        .collect::<Vec<_>>();
    assert!(ids.contains(&"part:bolt"), "{report}");
    assert!(ids.contains(&"supplier:acme"), "{report}");
    assert!(ids.contains(&"file:drawing"), "{report}");
    assert_eq!(
        listed
            .iter()
            .filter(|node| node["labels"]
                .as_array()
                .is_some_and(|labels| labels.iter().any(|label| label == "ControlEvent")))
            .count(),
        3,
        "each created record also keeps its control event"
    );
}

fn reopen() {
    let path = env::var("RR_ROUNDTRIP_PATH").unwrap();
    let db = Database::open_read_only(&path).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    let rows = db
        .query("MATCH (n) RETURN n", &BTreeMap::new())
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });
    let file = db.read_file_node("file:drawing").unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    let inspection = db.inspect().unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    let part = rows
        .rows
        .iter()
        .find(|row| row[0]["id"] == "part:bolt")
        .expect("part row");
    let report = json!({
        "nodes": inspection.nodes,
        "file_len": file.len(),
        "part_id": part[0]["id"],
        "vector_dimension": part[0]["properties"]["embedding"]["dimension"],
        "listed": rows.rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
    });
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
