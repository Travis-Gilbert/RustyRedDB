//! Parts and suppliers, stored in a native `.RR` file.
//!
//! Vectors in this example are caller-supplied coordinates, not model embeddings.
//! Run it twice with the same path: the second run reads what the first committed.

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use rustyred::{Database, NodeRecord};
use serde_json::json;

fn main() -> ExitCode {
    let path = PathBuf::from(env::args().nth(1).unwrap_or_else(|| "parts.rr".to_string()));
    let mut db = if path.exists() {
        match Database::open(&path) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(3);
            }
        }
    } else {
        match Database::create(&path) {
            Ok(db) => db,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(3);
            }
        }
    };

    if db.inspect().map(|item| item.nodes).unwrap_or(0) == 0 {
        if let Err(err) = seed(&mut db) {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    }

    match db.query(
        "MATCH (s:Supplier)-[:SUPPLIES]->(p:Part) RETURN s, p",
        &BTreeMap::new(),
    ) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "path": path,
                    "rows": result.rows,
                })
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(4)
        }
    }
}

fn seed(db: &mut Database) -> Result<(), rustyred::Error> {
    db.designate_vector("Part", "embedding", 3)?;
    db.upsert_node(NodeRecord::new(
        "part:bolt",
        ["Part"],
        json!({"name": "bolt", "sku": "B-1", "embedding": [1.0, 0.0, 0.0]}),
    ))?;
    db.upsert_node(NodeRecord::new(
        "part:nut",
        ["Part"],
        json!({"name": "nut", "sku": "N-1", "embedding": [0.0, 1.0, 0.0]}),
    ))?;
    db.upsert_node(NodeRecord::new(
        "supplier:acme",
        ["Supplier"],
        json!({"name": "Acme Fasteners"}),
    ))?;
    db.mutate(rustyred::GraphMutationBatch::new([
        rustyred::GraphMutation::EdgeUpsert(rustyred::EdgeRecord::new(
            "supply:bolt",
            "supplier:acme",
            "SUPPLIES",
            "part:bolt",
            json!({"lead_days": 4}),
        )),
        rustyred::GraphMutation::EdgeUpsert(rustyred::EdgeRecord::new(
            "supply:nut",
            "supplier:acme",
            "SUPPLIES",
            "part:nut",
            json!({"lead_days": 4}),
        )),
    ]))?;
    db.put_file_node("file:bolt-drawing", "drawings/bolt.bin", b"BOLT\0drawing")?;
    Ok(())
}
