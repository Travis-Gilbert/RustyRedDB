use rustyred_thg_core::{
    ChunkStore, JournalChunkStore, RedCoreDurability, RedCoreGraphStore, RedCoreOptions,
};
use serde_json::json;
use sha2::{Digest, Sha256};

#[test]
fn migrates_legacy_checksum_over_original_mutation_bytes_once() {
    let data_dir = unique_test_dir("redcore-legacy-aof-raw-checksum");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mutation = r#"{"op":"node_upsert","record":{"id":"node:legacy","labels":["MemoryAtom"],"properties":{"zeta":{"second":2,"first":1},"alpha":"preserve-order"},"version":1,"tombstone":false}}"#;
    let later_mutation = r#"{"op":"node_upsert","record":{"id":"node:later","labels":["MemoryAtom"],"properties":{"alpha":"canonical","zeta":{"first":1,"second":2}},"version":2,"tombstone":false}}"#;
    let aof = format!(
        "{}\n{}\n",
        aof_frame(1, mutation),
        aof_frame(2, later_mutation)
    );
    std::fs::write(data_dir.join("graph.aof.current"), &aof).unwrap();

    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let store = RedCoreGraphStore::open(&data_dir, options.clone())
        .expect("legacy AOF checksum must validate against persisted mutation bytes");

    let node = store
        .get_node("node:legacy")
        .unwrap()
        .expect("legacy AOF node should replay");
    assert_eq!(node.properties["zeta"]["second"], json!(2));
    assert!(store.get_node("node:later").unwrap().is_some());
    assert_eq!(store.status().recovered_frames, 2);
    drop(store);

    let backup_aof = data_dir
        .join("legacy-redcore-v1.backup")
        .join("graph.aof.current");
    assert!(!data_dir.join("graph.aof.current").exists());
    assert_eq!(std::fs::read_to_string(&backup_aof).unwrap(), aof);
    assert!(data_dir.join("redcore.journal").is_file());

    let reopened = RedCoreGraphStore::open(&data_dir, options)
        .expect("the migrated graph must recover from its committed journal root");
    assert!(reopened.get_node("node:legacy").unwrap().is_some());
    assert!(reopened.get_node("node:later").unwrap().is_some());
    assert_eq!(reopened.status().recovered_frames, 1);
    assert_eq!(std::fs::read_to_string(&backup_aof).unwrap(), aof);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn rejects_changed_legacy_mutation_bytes_without_publishing_a_journal_root() {
    let data_dir = unique_test_dir("redcore-aof-raw-checksum-corruption");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mutation = r#"{"op":"node_upsert","record":{"id":"node:original","labels":["MemoryAtom"],"properties":{"value":"before"},"version":1,"tombstone":false}}"#;
    let later_mutation = r#"{"op":"node_upsert","record":{"id":"node:later","labels":["MemoryAtom"],"properties":{"value":"valid"},"version":2,"tombstone":false}}"#;
    let corrupted_frame = aof_frame(1, mutation).replace("before", "tampered");
    let aof = format!("{corrupted_frame}\n{}\n", aof_frame(2, later_mutation));
    let aof_path = data_dir.join("graph.aof.current");
    std::fs::write(&aof_path, &aof).unwrap();

    let error = RedCoreGraphStore::open(
        &data_dir,
        RedCoreOptions {
            durability: RedCoreDurability::AofAlways,
            snapshot_interval_writes: 100,
            strict_acid: true,
        },
    )
    .expect_err("changed persisted mutation bytes must fail checksum validation");

    assert_eq!(error.code, "redcore_aof_corrupt_frame");
    assert_eq!(std::fs::read_to_string(&aof_path).unwrap(), aof);
    let journal = JournalChunkStore::open(data_dir.join("redcore.journal")).unwrap();
    assert_eq!(journal.read_root().unwrap(), None);
    drop(journal);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[ignore = "requires REDCORE_AOF_FIXTURE_DIR"]
fn migrates_external_aof_fixture_without_checksum_error() {
    let data_dir = std::env::var_os("REDCORE_AOF_FIXTURE_DIR")
        .map(std::path::PathBuf::from)
        .expect("REDCORE_AOF_FIXTURE_DIR must point to a copied RedCore tenant directory");

    let options = RedCoreOptions::default();
    let store = RedCoreGraphStore::open(&data_dir, options.clone())
        .expect("external AOF fixture should migrate without a checksum error");
    let status = store.status();
    assert!(status.last_recovery_ok);
    assert!(
        status.recovered_frames > 0,
        "fixture must exercise AOF replay beyond its snapshot"
    );
    assert!(store.verify().unwrap().ok);
    eprintln!(
        "recovered_frames={} last_txn_id={} snapshot_txn_id={}",
        status.recovered_frames, status.last_txn_id, status.snapshot_txn_id
    );
    drop(store);

    assert!(data_dir.join("redcore.journal").is_file());
    assert!(data_dir.join("legacy-redcore-v1.backup").is_dir());
    let reopened = RedCoreGraphStore::open(&data_dir, options)
        .expect("migrated external fixture should recover from the journal root");
    assert_eq!(reopened.status().recovered_frames, 1);
    assert!(reopened.verify().unwrap().ok);
}

fn aof_frame(txn_id: u64, mutation: &str) -> String {
    let checksum = format!("sha256:{:x}", Sha256::digest(mutation.as_bytes()));
    format!(
        r#"{{"magic":"RRGDB_AOF","version":1,"txn_id":{txn_id},"graph_version":{txn_id},"timestamp_unix_ms":1,"payload_checksum":"{checksum}","mutation":{mutation}}}"#
    )
}

fn unique_test_dir(label: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{label}-{unique}"))
}
