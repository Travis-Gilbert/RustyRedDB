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

    let default_params = bin()
        .args(["query", &path_arg, "--cypher", "MATCH (n) RETURN n"])
        .output()
        .unwrap();
    assert!(default_params.status.success(), "{default_params:?}");
    let default_json: serde_json::Value = serde_json::from_slice(&default_params.stdout).unwrap();
    assert_eq!(default_json["rows"], serde_json::json!([]));

    let invalid_query = bin()
        .args(["query", &path_arg, "--cypher", "CREATE (n)"])
        .output()
        .unwrap();
    assert_eq!(invalid_query.status.code(), Some(4));
    assert!(!invalid_query.stderr.is_empty());
    let error_json: serde_json::Value = serde_json::from_slice(&invalid_query.stdout).unwrap();
    assert_eq!(error_json["ok"], false);

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

#[test]
fn help_and_version_succeed_without_opening_a_database() {
    let dir = tempfile::tempdir().unwrap();
    for args in [
        vec!["--help"],
        vec!["create", "--help"],
        vec!["query", "--help"],
        vec!["inspect", "--help"],
        vec!["verify", "--help"],
        vec!["--version"],
    ] {
        let output = bin().current_dir(dir.path()).args(&args).output().unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{args:?}: {output:?}");
        assert!(!output.stdout.is_empty(), "{args:?}");
        if args.contains(&"--help") {
            assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
        }
    }
    assert_eq!(dir.path().read_dir().unwrap().count(), 0);
}

#[test]
fn invalid_arguments_are_rejected_before_storage_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("must-not-exist.rr");
    let path_arg = path.to_str().unwrap();
    let cases = [
        vec![],
        vec!["serve", path_arg],
        vec!["help"],
        vec!["create"],
        vec!["query"],
        vec!["inspect"],
        vec!["verify"],
        vec!["create", path_arg, "--unknown"],
        vec!["create", path_arg, "unexpected"],
        vec!["create", path_arg, "--json"],
        vec!["query", path_arg],
        vec!["query", path_arg, "--cypher"],
        vec!["query", path_arg, "--cypher", ""],
        vec![
            "query",
            path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "--params-json",
        ],
        vec![
            "query",
            path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "--unknown",
        ],
        vec![
            "query",
            path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "unexpected",
        ],
        vec![
            "query",
            path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "--cypher",
            "MATCH (m) RETURN m",
        ],
        vec![
            "query",
            path_arg,
            "--cypher",
            "MATCH (n) RETURN n",
            "--params-json",
            "{}",
            "--params-json",
            "{}",
        ],
        vec!["inspect", path_arg, "--json", "--unknown"],
        vec!["inspect", path_arg, "--json", "unexpected"],
        vec!["inspect", path_arg, "--json", "--json"],
        vec!["verify", path_arg, "--json", "--unknown"],
        vec!["verify", path_arg, "--json", "unexpected"],
        vec!["verify", path_arg, "--json", "--json"],
    ];
    for args in cases {
        let output = bin().args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{args:?}: {output:?}");
        assert!(!output.stderr.is_empty(), "{args:?}: {output:?}");
        assert!(!path.exists(), "{args:?} created a database");
        assert_eq!(dir.path().read_dir().unwrap().count(), 0, "{args:?}");
    }
}

#[test]
fn invalid_json_parameters_are_rejected_before_storage_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("must-not-exist.rr");
    for params in ["{", "[]", "null", "true", "42", r#""text""#] {
        let output = bin()
            .args([
                "query",
                path.to_str().unwrap(),
                "--cypher",
                "MATCH (n) RETURN n",
                "--params-json",
                params,
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{params}: {output:?}");
        assert!(output.stdout.is_empty(), "{params}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("--params-json"));
        assert!(!path.exists());
        assert_eq!(dir.path().read_dir().unwrap().count(), 0);
    }
}
