use rustyred_thg_core::{
    FieldedFullTextDefinition, FieldedFullTextDocument, FieldedFullTextIndex, IndexBackend,
    IndexCreatedBy, IndexKind, IndexScope,
};

#[test]
fn fielded_fulltext_definition_projects_to_bm25_manifest() {
    let definition = FieldedFullTextDefinition::bm25(
        "fulltext:codesymbol",
        "CodeSymbol",
        ["name", "docstring", "body"],
    );

    let manifest = definition
        .to_manifest(IndexScope::Repo, IndexCreatedBy::System)
        .expect("full-text manifest should be content-addressable");

    assert_eq!(manifest.kind, IndexKind::FullText);
    assert_eq!(manifest.backend, IndexBackend::Bm25);
    assert_eq!(manifest.target_label, "CodeSymbol");
    assert_eq!(
        manifest.target_properties,
        vec!["name", "docstring", "body"]
    );
}

#[test]
fn fielded_fulltext_preserves_code_identifiers_and_field_boosts() {
    let mut index = FieldedFullTextIndex::new("CodeSymbol");
    index.set_field_boost("name", 4.0);
    index.set_field_boost("body", 1.0);

    index.upsert(
        FieldedFullTextDocument::new("symbol:identity")
            .with_field("name", "IdentityIndex::resolve_or_insert")
            .with_field(
                "body",
                "Exact identity registry collision handling for graph objects.",
            ),
    );
    index.upsert(
        FieldedFullTextDocument::new("symbol:generic")
            .with_field("name", "generic_insert")
            .with_field("body", "Resolve existing object in general terms."),
    );

    let hits = index.search("resolve_or_insert", 5);

    assert_eq!(hits[0].doc_id, "symbol:identity");
    assert!(
        hits[0].field_scores["name"] > hits[0].field_scores.get("body").copied().unwrap_or(0.0)
    );
}

#[test]
fn phrase_prefix_and_fuzzy_search_return_source_offsets() {
    let mut index = FieldedFullTextIndex::new("Postmortem");
    index.set_field_boost("failure_mode", 3.0);

    index.upsert(
        FieldedFullTextDocument::new("postmortem:recall")
            .with_field("failure_mode", "recall read path timeout")
            .with_field(
                "root_cause",
                "The identity registry lookup blocked the context compiler.",
            ),
    );

    let phrase = index.phrase_search("identity registry", 5);
    assert_eq!(phrase[0].doc_id, "postmortem:recall");
    assert_eq!(phrase[0].snippets[0].field, "root_cause");
    assert!(phrase[0].snippets[0].start < phrase[0].snippets[0].end);

    let prefix = index.prefix_search("rec", 5);
    assert_eq!(prefix[0].doc_id, "postmortem:recall");

    let fuzzy = index.fuzzy_search("regstry", 2, 5);
    assert_eq!(fuzzy[0].doc_id, "postmortem:recall");
}

#[test]
fn url_and_domain_terms_find_webdoc_records() {
    let mut index = FieldedFullTextIndex::new("WebDoc");
    index.set_field_boost("url", 3.0);
    index.set_field_boost("domain", 2.0);

    index.upsert(
        FieldedFullTextDocument::new("webdoc:theorems")
            .with_field("title", "Theorems Harness Docs")
            .with_field("domain", "theoremsweb.com")
            .with_field("url", "https://theoremsweb.com/docs/rustyred-indexes"),
    );

    let hits = index.search("theoremsweb.com rustyred indexes", 5);

    assert_eq!(hits[0].doc_id, "webdoc:theorems");
    assert!(hits[0].field_scores.contains_key("url"));
}
