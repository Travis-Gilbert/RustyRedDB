//! Retirement acceptance tests for the historical document tier.
//!
//! The live filesystem contract is graph `File`/`Directory` nodes plus
//! manifest-backed payloads. `LegacyDocTree` is intentionally exercised only
//! as a bounded, read-only migration decoder. Historical root construction is
//! available solely through the explicitly named fixture type.

use std::path::PathBuf;

use rustyred_thg_core::{
    content_hash_bytes, read_file_payload, stage_file_payload, ColdTierKind, DiskObjectStore,
    DocEntry, FileManifestRef, LegacyDocTree, LegacyDocTreeFixture, NodeRecord, PathKey,
    PathPrefix, FILE_CONTENT_MANIFEST_PROPERTY,
};
use serde_json::json;

fn temp_store(name: &str) -> (PathBuf, DiskObjectStore) {
    let dir = std::env::temp_dir().join(format!(
        "legacy-doctree-accept-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let store = DiskObjectStore::open(&dir).expect("open unified object facade");
    (dir, store)
}

fn legacy_entry(body: &[u8], created_ms: i64) -> DocEntry {
    DocEntry {
        content_hash: Some(content_hash_bytes(body)),
        inline: Some(body.to_vec()),
        real_path: None,
        inline_compressed: false,
        tier: ColdTierKind::Cold,
        size: u64::try_from(body.len()).expect("fixture body fits u64"),
        created_ms,
        gist: None,
        previous_hashes: Vec::new(),
        history_elided_count: 0,
    }
}

#[test]
fn legacy_json_is_a_bounded_read_only_import_view() {
    let (_dir, store) = temp_store("json-import");
    let alpha = b"historical alpha body";
    let beta = b"historical beta body";
    let encoded = serde_json::to_vec(&vec![
        ("tenant/project/beta.txt".to_string(), legacy_entry(beta, 2)),
        (
            "tenant/project/alpha.txt".to_string(),
            legacy_entry(alpha, 1),
        ),
    ])
    .expect("encode historical JSON fixture");

    let decoded = LegacyDocTree::decode_legacy_json(&encoded).expect("decode legacy JSON");
    assert_eq!(decoded.len(), 2);
    assert_eq!(
        decoded
            .legacy_entries()
            .map(|(path, _)| path.to_slash_path())
            .collect::<Vec<_>>(),
        ["tenant/project/alpha.txt", "tenant/project/beta.txt"]
    );
    assert_eq!(
        decoded
            .resolve_body(
                &PathKey::from_slash_path("tenant/project/alpha.txt").unwrap(),
                &store,
            )
            .unwrap()
            .as_deref(),
        Some(alpha.as_slice())
    );

    let duplicate = serde_json::to_vec(&vec![
        (
            "tenant/project/repeated".to_string(),
            legacy_entry(alpha, 1),
        ),
        ("tenant/project/repeated".to_string(), legacy_entry(beta, 2)),
    ])
    .unwrap();
    assert_eq!(
        LegacyDocTree::decode_legacy_json(&duplicate)
            .unwrap_err()
            .code,
        "legacy_doc_tree_duplicate_path"
    );
}

#[test]
fn historical_namespace_writes_are_fixture_named() {
    let (_dir, store) = temp_store("fixture-only");
    let path = PathKey::from_slash_path("legacy/readme.txt").unwrap();
    let body = b"historical committed-root fixture";
    let mut fixture = LegacyDocTreeFixture::new(8);
    let entry = fixture
        .put_body(
            path.clone(),
            body,
            ColdTierKind::Cold,
            7,
            Some("migration fixture".to_string()),
            &store,
        )
        .expect("seed historical entry");

    assert_eq!(
        entry.content_hash.as_deref(),
        Some(content_hash_bytes(body).as_str())
    );
    assert_eq!(
        fixture.resolve_body(&path, &store).unwrap().as_deref(),
        Some(body.as_slice())
    );
    assert_eq!(store.chunk_count().unwrap(), 1);
}

#[test]
fn live_file_payload_is_manifest_backed_and_bodyless_in_the_graph() {
    let (_dir, store) = temp_store("file-manifest");
    let chunks = store.chunk_store();
    let body = (0..200_000_u32)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut staged = stage_file_payload(chunks.as_ref(), "file:accept/content_manifest", &body)
        .expect("stage manifest-backed body");
    let reference = staged.reference().clone();
    let report = staged.report();

    assert_eq!(
        read_file_payload(chunks.as_chunk_store(), &reference).unwrap(),
        body
    );
    assert!(report.total_chunks > 0);
    assert!(report.bytes_written > 0);

    let path = PathKey::from_slash_path("tenant/project/src/main.rs").unwrap();
    let mut properties = serde_json::Map::new();
    properties.insert("path".to_string(), json!(path.as_str()));
    properties.insert(
        FILE_CONTENT_MANIFEST_PROPERTY.to_string(),
        reference.to_value().unwrap(),
    );
    let node = NodeRecord::new(
        "file:accept",
        ["File"],
        serde_json::Value::Object(properties),
    );
    assert!(node.properties.get("body").is_none());
    assert_eq!(
        FileManifestRef::from_value(&node.properties[FILE_CONTENT_MANIFEST_PROPERTY]).unwrap(),
        reference
    );

    staged.cancel().expect("release fixture staging pins");
}

#[test]
fn graph_path_keys_keep_exact_and_prefix_domains_distinct() {
    let exact = PathKey::from_segments(["tenant", "project", "src", "control\u{1f}.rs"])
        .expect("raw graph path");
    let prefix = PathPrefix::from_segments(["tenant", "project", "src"]).expect("raw graph prefix");
    let collision = PathKey::from_slash_path("tenant/project/src2/control.rs").unwrap();

    assert!(prefix.matches(&exact));
    assert!(!prefix.matches(&collision));
    assert!(exact.as_str().contains('\0'));
    assert_eq!(exact.to_slash_path(), "tenant/project/src/control\u{1f}.rs");
}
