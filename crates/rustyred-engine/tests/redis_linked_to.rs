//! Live Redis-wire writer acceptance, using only a private temporary Valkey.
//! This never reads an external endpoint or accesses an existing server.
#![cfg(feature = "redis-store")]

use std::collections::BTreeMap;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use rustyred_thg_core::{EdgeRecord, Hash, NodeRecord, RedisGraphStore};
use serde_json::{json, Value};
use theorem_surface_contracts::link::{
    linked_to_identity_bytes, IdentityEvidence, Justification, JustificationKind, LinkKind,
    LinkedToRecord, LINKED_TO_SCHEMA, LINKED_TO_TYPE,
};

// Same isolated-process pattern as theorem-valkey-client's live acceptance
// tests; storage and transactions remain the actual RedisGraphStore/Valkey path.
struct IsolatedValkey {
    child: Child,
    _directory: tempfile::TempDir,
    url: String,
}

impl Drop for IsolatedValkey {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl IsolatedValkey {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let child = Command::new("valkey-server")
            .args([
                "--bind",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--save",
                "",
                "--appendonly",
                "no",
                "--protected-mode",
                "yes",
            ])
            .arg("--dir")
            .arg(directory.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("install valkey-server to run the isolated live oracle");
        let server = Self {
            child,
            _directory: directory,
            url: format!("redis://127.0.0.1:{port}/"),
        };
        for _ in 0..100 {
            if server.store().ping().is_ok() {
                let mut connection = redis::Client::open(server.url.as_str())
                    .unwrap()
                    .get_connection()
                    .unwrap();
                let info: redis::InfoDict = redis::cmd("INFO")
                    .arg("server")
                    .query(&mut connection)
                    .unwrap();
                assert_eq!(
                    info.get::<u32>("process_id"),
                    Some(server.child.id()),
                    "refusing a Redis endpoint not owned by this test child"
                );
                return server;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("isolated Valkey did not start");
    }

    fn store(&self) -> RedisGraphStore {
        RedisGraphStore::new(&self.url, "linked-to-acceptance").unwrap()
    }

    fn seeded_store(&self) -> RedisGraphStore {
        let mut store = self.store();
        for id in ["source", "target"] {
            store
                .upsert_node(NodeRecord::new(id, ["Document"], json!({})))
                .unwrap();
        }
        store
    }

    // Compare every persisted key, including indexes, allocated version and
    // event history. DUMP is the server's serialization, not a storage stand-in.
    fn persisted_state(&self) -> BTreeMap<String, Vec<u8>> {
        let mut connection = redis::Client::open(self.url.as_str())
            .unwrap()
            .get_connection()
            .unwrap();
        let keys: Vec<String> = redis::cmd("KEYS").arg("*").query(&mut connection).unwrap();
        keys.into_iter()
            .map(|key| {
                let value = redis::cmd("DUMP").arg(&key).query(&mut connection).unwrap();
                (key, value)
            })
            .collect()
    }
}

fn canonical_edge() -> EdgeRecord {
    let record = LinkedToRecord {
        schema: LINKED_TO_SCHEMA.into(),
        kind: LinkKind::Inline,
        justification: JustificationKind::Identity,
        generator: "identity-generator".into(),
        score: 0.9,
        model_version: "identity-v1".into(),
        receipt_id: "receipt-1".into(),
        evidence: Justification::Identity(IdentityEvidence::Identifier {
            field: "slug".into(),
            value: "canonical-target".into(),
            edge_def_id: "identity-rule".into(),
            definition_version: 1,
        }),
        visibility: json!({"scope": "private-test"}),
        written_at_graph_version: None,
    };
    let id = format!(
        "linked-to:{}",
        Hash::of(&linked_to_identity_bytes("source", "target", record.kind)).to_prefixed()
    );
    EdgeRecord::new(
        id,
        "source",
        LINKED_TO_TYPE,
        "target",
        serde_json::to_value(record).unwrap(),
    )
}

fn first_version(edge: &EdgeRecord) -> u64 {
    serde_json::from_value::<LinkedToRecord<Value>>(edge.properties.clone())
        .unwrap()
        .written_at_graph_version
        .expect("a persisted LINKED_TO must have an allocated first version")
}

#[test]
#[ignore = "isolated live oracle requires valkey-server on PATH"]
fn redis_linked_to_create_stamps_actual_version_before_hashing() {
    let server = IsolatedValkey::start();
    let mut store = server.seeded_store();
    let edge = canonical_edge();
    let write = store.upsert_edge(edge.clone()).unwrap();
    assert_eq!(write.version, 3);
    let reopened = server.store();
    let persisted = reopened.get_edge(&edge.id).unwrap().unwrap();
    assert_eq!(first_version(&persisted), write.version);
    assert_eq!(persisted.version, write.version);
    assert_eq!(persisted.checksum().unwrap(), write.checksum);
    let ordinary = EdgeRecord::new(
        "ordinary-edge",
        "source",
        "REFERENCES",
        "target",
        json!({"note": "unreserved"}),
    );
    store.upsert_edge(ordinary.clone()).unwrap();
    let ordinary_persisted = reopened.get_edge(&ordinary.id).unwrap().unwrap();
    assert_eq!(ordinary_persisted.properties, ordinary.properties);
    assert!(reopened.verify().unwrap().ok);
}

#[test]
#[ignore = "isolated live oracle requires valkey-server on PATH"]
fn redis_linked_to_updates_preserve_first_version() {
    let server = IsolatedValkey::start();
    let mut store = server.seeded_store();
    let mut edge = canonical_edge();
    let first = store.upsert_edge(edge.clone()).unwrap().version;
    for supplied in [Value::Null, json!(first)] {
        edge.properties["written_at_graph_version"] = supplied;
        edge.properties["score"] = json!(0.75);
        let write = store.upsert_edge(edge.clone()).unwrap();
        assert!(write.version > first);
        let persisted = server.store().get_edge(&edge.id).unwrap().unwrap();
        assert_eq!(first_version(&persisted), first);
        assert_eq!(persisted.version, write.version);
        assert_eq!(persisted.checksum().unwrap(), write.checksum);
    }
    assert_eq!(store.stats().unwrap().edges_total, 1);
    assert!(store.verify().unwrap().ok);
}

#[test]
#[ignore = "isolated live oracle requires valkey-server on PATH"]
fn redis_linked_to_invalid_bindings_do_not_publish_any_state() {
    let server = IsolatedValkey::start();
    let mut store = server.seeded_store();
    let canonical = canonical_edge();
    let mut invalid = Vec::new();
    let mut edge = canonical.clone();
    edge.id = "forged-id".into();
    invalid.push(edge);
    let mut edge = canonical.clone();
    std::mem::swap(&mut edge.from_id, &mut edge.to_id);
    invalid.push(edge);
    for (field, value) in [
        ("kind", json!("see_also")),
        ("visibility", json!("not-an-object")),
        ("schema", json!("unknown-schema")),
        ("unrecognized_field", json!(true)),
        ("justification", json!("similarity")),
        ("written_at_graph_version", json!(100)),
    ] {
        let mut edge = canonical.clone();
        edge.properties[field] = value;
        invalid.push(edge);
    }
    let before = server.persisted_state();
    for edge in invalid {
        let error = store.upsert_edge(edge).unwrap_err();
        assert_eq!(error.code, "invalid_linked_to_record");
        assert_eq!(server.persisted_state(), before);
    }
    assert!(server.store().get_edge(&canonical.id).unwrap().is_none());
    assert!(store.verify().unwrap().ok);
}

#[test]
#[ignore = "isolated live oracle requires valkey-server on PATH"]
fn redis_linked_to_first_version_is_immutable_without_partial_updates() {
    let server = IsolatedValkey::start();
    let mut store = server.seeded_store();
    let edge = canonical_edge();
    let first = store.upsert_edge(edge.clone()).unwrap().version;
    let before = server.persisted_state();
    for forged in [0, first - 1, first + 1, u64::MAX] {
        let mut update = edge.clone();
        update.properties["score"] = json!(0.25);
        update.properties["written_at_graph_version"] = json!(forged);
        let error = store.upsert_edge(update).unwrap_err();
        assert_eq!(error.code, "invalid_linked_to_record");
        assert_eq!(server.persisted_state(), before);
    }
    let reopened = server.store();
    assert_eq!(
        first_version(&reopened.get_edge(&edge.id).unwrap().unwrap()),
        first
    );
    assert!(reopened.verify().unwrap().ok);
}
