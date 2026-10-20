use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rustyred"))
}

#[test]
fn create_inspect_query_and_verify_agree() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cli.rr");
    let path_arg = path.display().to_string();

    let created = bin().args(["create", &path_arg]).output().unwrap();
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stderr)
    );
    let created_json: serde_json::Value = serde_json::from_slice(&created.stdout).unwrap();
    assert_eq!(created_json["ok"], true);

    let again = bin().args(["create", &path_arg]).output().unwrap();
    assert_eq!(again.status.code(), Some(3));
    let again_json: serde_json::Value = serde_json::from_slice(&again.stdout).unwrap();
    assert_eq!(again_json["ok"], false);
    assert_eq!(again_json["code"], "rr_already_exists");

    let inspected = bin()
        .args(["inspect", &path_arg, "--json"])
        .output()
        .unwrap();
    assert!(inspected.status.success());
    let inspected_json: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(inspected_json["nodes"], 0);
    assert_eq!(inspected_json["writable"], false);

    let queried = bin()
        .args([
            "query",
            &path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "--params-json",
            "{}",
        ])
        .output()
        .unwrap();
    assert!(
        queried.status.success(),
        "{}",
        String::from_utf8_lossy(&queried.stderr)
    );
    let queried_json: serde_json::Value = serde_json::from_slice(&queried.stdout).unwrap();
    assert_eq!(queried_json["rows"], serde_json::json!([]));

    let verified = bin()
        .args(["verify", &path_arg, "--json"])
        .output()
        .unwrap();
    assert!(verified.status.success());
    let verified_json: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(verified_json["ok"], true);

    let missing = dir.path().join("missing.rr");
    let opened = bin()
        .args(["inspect", &missing.display().to_string(), "--json"])
        .output()
        .unwrap();
    assert_eq!(opened.status.code(), Some(3));
    let opened_json: serde_json::Value = serde_json::from_slice(&opened.stdout).unwrap();
    assert_eq!(opened_json["code"], "not_found");
}
