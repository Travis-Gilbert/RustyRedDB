use serde_json::Value;

#[test]
fn compiled_receipts_match_unchanged_python_reference_bytes() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../fixtures/python_reference.json")).unwrap();
    assert_eq!(cases.len(), 45);
    for case in cases {
        let actual = datalog::derive_legacy_receipt(&case["input"]).unwrap();
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&case["expected"]).unwrap(),
            "Python receipt differs for {}\nactual={actual}\nexpected={}",
            case["name"],
            case["expected"],
        );
    }
}
