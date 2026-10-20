use rustyred_thg_core::{
    IdentityIndex, IdentityIndexDefinition, IdentityIndexKey, IdentityInsertOutcome,
    IdentityTarget, ReceiptScope,
};
use serde_json::json;

fn tenant_scope() -> ReceiptScope {
    ReceiptScope::from([
        ("tenant".to_string(), "Travis-Gilbert".to_string()),
        ("project".to_string(), "Theorem".to_string()),
    ])
}

#[test]
fn reingesting_same_url_resolves_existing_webdoc_without_duplicate() {
    let mut index = IdentityIndex::new();
    index
        .register_definition(IdentityIndexDefinition::new(
            "identity:webdoc:url",
            "Canonical URL",
            "WebDoc",
            ["canonical_url"],
        ))
        .unwrap();

    let key = index
        .key_from_properties(
            "identity:webdoc:url",
            tenant_scope(),
            &json!({ "canonical_url": "https://theoremsweb.com/docs" }),
        )
        .unwrap();
    let target = IdentityTarget::node("webdoc:theoremsweb-docs", "WebDoc", 1);

    let first = index
        .resolve_or_insert(key.clone(), target.clone())
        .unwrap();
    assert_eq!(first, IdentityInsertOutcome::Inserted(target.clone()));

    let second = index.resolve_or_insert(key, target.clone()).unwrap();
    assert_eq!(second, IdentityInsertOutcome::Existing(target));
    assert_eq!(index.len(), 1);
    assert!(index.problems().is_empty());
}

#[test]
fn replaying_same_run_id_does_not_create_second_run_identity() {
    let mut index = IdentityIndex::new();
    index
        .register_definition(IdentityIndexDefinition::new(
            "identity:harness-run:run-id",
            "Harness run id",
            "HarnessRun",
            ["run_id"],
        ))
        .unwrap();

    let key = index
        .key_from_properties(
            "identity:harness-run:run-id",
            tenant_scope(),
            &json!({ "run_id": "run:phase1" }),
        )
        .unwrap();
    let target = IdentityTarget::node("run:phase1", "HarnessRun", 3);

    assert!(matches!(
        index
            .resolve_or_insert(key.clone(), target.clone())
            .unwrap(),
        IdentityInsertOutcome::Inserted(_)
    ));
    assert!(matches!(
        index.resolve_or_insert(key, target).unwrap(),
        IdentityInsertOutcome::Existing(_)
    ));
}

#[test]
fn repo_commit_file_and_symbol_path_resolve_stable_code_symbol_identity() {
    let mut index = IdentityIndex::new();
    index
        .register_definition(IdentityIndexDefinition::new(
            "identity:code-symbol",
            "Code symbol",
            "CodeSymbol",
            ["repo", "commit", "file_path", "symbol_path"],
        ))
        .unwrap();

    let properties = json!({
        "repo": "Travis-Gilbert/Theorem",
        "commit": "abc123",
        "file_path": "rustyredcore_THG/crates/rustyred-thg-core/src/identity_index.rs",
        "symbol_path": "IdentityIndex::resolve_or_insert"
    });
    let key = index
        .key_from_properties("identity:code-symbol", tenant_scope(), &properties)
        .unwrap();
    let target = IdentityTarget::node(
        "symbol:abc123:identity-index-resolve-or-insert",
        "CodeSymbol",
        5,
    );

    index
        .resolve_or_insert(key.clone(), target.clone())
        .unwrap();
    assert_eq!(index.resolve(&key), Some(&target));
    assert_eq!(index.len(), 1);
}

#[test]
fn content_hash_collision_creates_problem_record_without_overwriting_identity() {
    let mut index = IdentityIndex::new();
    index
        .register_definition(IdentityIndexDefinition::new(
            "identity:artifact:content-hash",
            "Artifact content hash",
            "Artifact",
            ["content_hash"],
        ))
        .unwrap();
    let key = IdentityIndexKey::new(
        "identity:artifact:content-hash",
        tenant_scope(),
        [("content_hash", "sha256:deadbeef")],
    );

    let existing = IdentityTarget::node("artifact:a", "Artifact", 10);
    let attempted = IdentityTarget::node("artifact:b", "Artifact", 11);

    index
        .resolve_or_insert(key.clone(), existing.clone())
        .unwrap();
    let collision = index
        .resolve_or_insert(key.clone(), attempted.clone())
        .unwrap();

    let IdentityInsertOutcome::Collision(problem) = collision else {
        panic!("expected explicit identity collision problem");
    };
    assert_eq!(problem.kind, "identity_collision");
    assert_eq!(problem.existing, existing);
    assert_eq!(problem.attempted, attempted);
    assert_eq!(index.resolve(&key).unwrap().node_id, "artifact:a");
    assert_eq!(index.problems(), &[problem]);
}
