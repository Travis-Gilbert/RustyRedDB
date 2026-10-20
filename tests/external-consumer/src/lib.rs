use std::collections::BTreeMap;

use rustyred::{Database, NodeRecord};
use serde_json::json;

#[test]
fn an_outside_crate_can_create_and_reopen() {
    let dir = std::env::temp_dir().join(format!("rustyred-external-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("outside.rr");
    let mut db = Database::create(&path).unwrap();
    db.upsert_node(NodeRecord::new("part:1", ["Part"], json!({"sku": "X"}))).unwrap();
    drop(db);
    let db = Database::open_read_only(&path).unwrap();
    let rows = db.query("MATCH (p:Part) RETURN p", &BTreeMap::new()).unwrap();
    assert_eq!(rows.rows.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}
