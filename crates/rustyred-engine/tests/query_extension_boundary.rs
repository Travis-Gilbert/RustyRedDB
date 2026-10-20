//! The engine must stay below its optional query-rewrite extension.

#[test]
fn core_does_not_depend_on_query_saturate() {
    let manifest = include_str!("../Cargo.toml");
    assert!(!manifest.contains("rustyred-thg-query-saturate"));
    assert!(!manifest.contains("rustyred-query-saturate"));
    let _ = rustyred_thg_core::RankingRegistry::with_native_defaults();
}
