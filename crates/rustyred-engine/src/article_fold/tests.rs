use super::*;
mod links;
use crate::graph::capability::{GraphRead, GraphWrite};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::snapshot::GraphSnapshot;
use crate::statement::{collapse_if_corroborated, propose_same_as, Confidence};
use serde_json::json;

/// Explicit local fixture authority. Production routes must use a real pinned
/// tenant reader, not this deterministic in-memory test adapter.
struct FixtureArticleRead {
    nodes: BTreeMap<String, NodeRecord>,
    edges: Vec<EdgeRecord>,
    excluded: BTreeSet<String>,
    sources: BTreeMap<String, SourceAdmission>,
    receipts: BTreeMap<String, Vec<CitedEvidence>>,
    pins: Vec<PinnedSection>,
    revision: Revision,
    fail_read: bool,
}

impl Default for FixtureArticleRead {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: vec![],
            excluded: BTreeSet::new(),
            sources: BTreeMap::new(),
            receipts: BTreeMap::new(),
            pins: vec![],
            fail_read: false,
            revision: Revision {
                commit: None,
                head: HeadReference {
                    commit_id: Some("fixture-known-legacy-head".into()),
                    root: "fixture-root".into(),
                    transaction: Some(42),
                    log_position: None,
                },
                coverage: HistoryCoverage::Unavailable,
                seq: 11,
                stream_epoch: "fixture-stream".into(),
            },
        }
    }
}

impl ArticleRead for FixtureArticleRead {
    fn node(&self, id: &str) -> Result<ArticleNode, ArticleFoldError> {
        if self.fail_read {
            return Err(ArticleFoldError::Read("fixture corruption".into()));
        }
        if self.excluded.contains(id) {
            return Ok(ArticleNode::Excluded);
        }
        Ok(self
            .nodes
            .get(id)
            .cloned()
            .map(ArticleNode::Present)
            .unwrap_or(ArticleNode::Missing))
    }
    fn edges(&self, id: &str, direction: Direction) -> Result<Vec<EdgeRecord>, ArticleFoldError> {
        Ok(self
            .edges
            .iter()
            .filter(|e| match direction {
                Direction::Out => e.from_id == id,
                Direction::In => e.to_id == id,
            })
            .cloned()
            .collect())
    }
    fn identity_candidates(&self, exact: &str) -> Result<IdentityCandidates, ArticleFoldError> {
        let mut out = IdentityCandidates::default();
        for node in self.nodes.values() {
            if node.id == exact
                || string(node, "slug") == Some(exact)
                || (label(node, "MemoryDocument") && string(node, "doc_id") == Some(exact))
                || (label(node, "MemoryNode") && string(node, "node_id") == Some(exact))
                || (label(node, "MemoryProject") && string(node, "project_slug") == Some(exact))
                || (label(node, "AliasWitness") && string(node, "alias") == Some(exact))
            {
                if self.excluded.contains(&node.id) {
                    out.excluded = true;
                } else {
                    out.nodes.push(node.clone());
                }
            }
        }
        Ok(out)
    }
    fn source_admission(&self, reference: &str) -> Result<SourceAdmission, ArticleFoldError> {
        Ok(self
            .sources
            .get(reference)
            .cloned()
            .unwrap_or(SourceAdmission::Unresolved))
    }
    fn evidence(&self, node: &NodeRecord) -> Result<Vec<CitedEvidence>, ArticleFoldError> {
        Ok(self.receipts.get(&node.id).cloned().unwrap_or_default())
    }
    fn pinned_sections(&self, _: &str) -> Result<Vec<PinnedSection>, ArticleFoldError> {
        Ok(self.pins.clone())
    }
    fn revision(&self, _: &BTreeSet<String>) -> Result<Revision, ArticleFoldError> {
        Ok(self.revision.clone())
    }
}

fn fixture() -> (FixtureArticleRead, Value) {
    let json: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/article_fold/cluster.json"
    ))
    .unwrap();
    let mut read = FixtureArticleRead::default();
    for node in json["nodes"].as_array().unwrap() {
        let node = NodeRecord::new(
            node["id"].as_str().unwrap(),
            node["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap()),
            node["properties"].clone(),
        );
        read.nodes.insert(node.id.clone(), node);
    }
    for edge in json["edges"].as_array().unwrap() {
        read.edges.push(EdgeRecord::new(
            edge["id"].as_str().unwrap(),
            edge["from"].as_str().unwrap(),
            edge["type"].as_str().unwrap(),
            edge["to"].as_str().unwrap(),
            json!({}),
        ));
    }
    (read, json["expected"].clone())
}

fn put(read: &mut FixtureArticleRead, id: &str, kind: &str, content: &str) {
    read.nodes.insert(
        id.into(),
        NodeRecord::new(
            id,
            ["Entity"],
            json!({"kind":kind,"title":id,"gist":content}),
        ),
    );
}

#[test]
fn article_fold_fixture_matches_all_fixed_fields_and_missing_references() {
    let (read, expected) = fixture();
    let article = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert_eq!(article.title, expected["title"]);
    assert_eq!(
        json!(article.infobox.iter().map(|r| &r.id).collect::<Vec<_>>()),
        expected["infobox"]
    );
    assert_eq!(
        json!(article.sections.iter().map(|r| &r.id).collect::<Vec<_>>()),
        expected["sections"]
    );
    assert_eq!(
        json!(article
            .red_links
            .iter()
            .map(|r| &r.target)
            .collect::<Vec<_>>()),
        expected["redLinks"]
    );
    assert_eq!(
        json!(article
            .backlinks
            .iter()
            .map(|r| &r.entity_id)
            .collect::<Vec<_>>()),
        expected["backlinks"]
    );
    assert_eq!(
        article
            .lead
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>(),
        [
            "Theorem is a graph workspace.",
            "Its readers share provenance."
        ]
    );
    assert!(article.hatnote.is_none());
    assert!(!article.stub);
    assert!(!article.cluster_ids.contains(&"outsider".into()));
    assert!(article
        .dependency_keys
        .contains(&"node/future:target".into()));
    assert!(article.dependency_keys.contains(&"edge/e:dangling".into()));
    assert_eq!(article.revision, read.revision);
    assert_eq!(
        serde_json::from_value::<ArticleProjection>(serde_json::to_value(&article).unwrap())
            .unwrap(),
        article
    );
}

#[test]
fn article_fold_memory_labels_resolve_doc_node_and_project_identity() {
    let (mut read, _) = fixture();
    let article = fold_article(&read, "SPEC-WIKI-1", ArticleTier::Full).unwrap();
    assert_eq!(article.kind, "spec");
    assert!(!article.stub);
    assert_eq!(article.lead[0].text, "Articles are graph projections.");
    assert_eq!(article.lead.len(), 1);
    let body = article
        .sections
        .iter()
        .find(|s| s.id == "specifications")
        .unwrap();
    let SectionBody::Entries { entries } = &body.body else {
        panic!("folded specification body");
    };
    assert_eq!(
        entries[0].source_text,
        "Articles are graph projections.\n\nThe rest belongs in the article."
    );
    assert_eq!(
        fold_article(&read, "decision/wiki", ArticleTier::Full)
            .unwrap()
            .kind,
        "decision"
    );
    read.nodes.insert(
        "memory:project".into(),
        NodeRecord::new(
            "memory:project",
            ["MemoryProject"],
            json!({"project_slug":"Theorem"}),
        ),
    );
    let project = fold_article(&read, "Theorem", ArticleTier::Full).unwrap();
    assert_eq!(project.kind, "project");
    assert_eq!(project.title, "Theorem");
}

#[test]
fn article_fold_eligibility_uses_exact_kind_degree_and_provenance_threshold() {
    for kind in ARTICLE_KINDS {
        let mut read = FixtureArticleRead::default();
        put(&mut read, "subject", kind, "Text.");
        assert!(
            !fold_article(&read, "subject", ArticleTier::Full)
                .unwrap()
                .stub,
            "{kind}"
        );
    }
    let mut read = FixtureArticleRead::default();
    put(&mut read, "subject", "concept", "Text.");
    for index in 0..5 {
        let id = format!("source:{index}");
        put(&mut read, &id, "document", "Source.");
        read.edges.push(EdgeRecord::new(
            format!("edge:{index}"),
            id,
            "REFERENCES",
            "subject",
            json!({"source":"one"}),
        ));
    }
    let stub = fold_article(&read, "subject", ArticleTier::Full).unwrap();
    assert!(stub.stub);
    assert!(stub.categories.is_empty());
    assert!(stub.sections.is_empty());
    read.edges[0].properties["source"] = json!("two");
    assert!(
        !fold_article(&read, "subject", ArticleTier::Full)
            .unwrap()
            .stub
    );
    read.edges.pop();
    assert!(
        fold_article(&read, "subject", ArticleTier::Full)
            .unwrap()
            .stub
    );
}

#[test]
fn article_fold_excluded_name_collision_does_not_change_visible_resolution() {
    let (mut read, _) = fixture();
    let expected = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    put(&mut read, "private:project", "project", "Private source.");
    read.nodes.get_mut("private:project").unwrap().properties["slug"] = json!("theorem");
    read.excluded.insert("private:project".into());
    assert_eq!(
        fold_article(&read, "theorem", ArticleTier::Full).unwrap(),
        expected
    );
    read.nodes.get_mut("repo:theorem").unwrap().properties["slug"] = json!("private:project");
    assert_eq!(
        fold_article(&read, "private:project", ArticleTier::Full),
        Err(ArticleFoldError::Excluded)
    );
    assert_eq!(
        fold_article(&read, "theorem", ArticleTier::Full),
        Err(ArticleFoldError::Excluded)
    );
}

#[test]
fn article_fold_missing_excluded_and_corruption_never_collapse() {
    let (mut read, _) = fixture();
    read.excluded.insert("future:target".into());
    let article = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert!(!article
        .red_links
        .iter()
        .any(|r| r.target == "future:target"));
    assert_eq!(
        fold_article(&read, "future:target", ArticleTier::Full),
        Err(ArticleFoldError::Excluded)
    );
    assert_eq!(
        fold_article(&read, "absent", ArticleTier::Full),
        Err(ArticleFoldError::Missing("absent".into()))
    );
    read.fail_read = true;
    assert!(matches!(
        fold_article(&read, "theorem", ArticleTier::Full),
        Err(ArticleFoldError::Read(_))
    ));
}

#[test]
fn article_fold_foreign_dependencies_and_receipts_exclude_content_not_just_footnotes() {
    let (mut read, _) = fixture();
    put(
        &mut read,
        "foreign:source",
        "document",
        "Do not expose this.",
    );
    read.nodes.get_mut("foreign:source").unwrap().properties["source_ref"] =
        json!("theorem://foreign/block/1");
    read.sources.insert(
        "theorem://foreign/block/1".into(),
        SourceAdmission::Foreign {
            dependency_node_ids: vec!["federation:cache:verified".into()],
        },
    );
    read.nodes.get_mut("memory:decision").unwrap().properties["dependency_fact_ids"] =
        json!(["foreign:source"]);
    read.receipts.insert(
        "memory:spec".into(),
        vec![CitedEvidence {
            reference: Reference {
                id: ReferenceId("foreign-receipt".into()),
                receipt_id: "foreign-receipt".into(),
                source_kind: "fixture".into(),
                replay_handle: "foreign-replay".into(),
            },
            dependency_node_ids: vec!["foreign:source".into()],
        }],
    );
    let article = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert!(!article
        .sections
        .iter()
        .any(|s| matches!(s.id.as_str(), "decisions" | "specifications")));
    assert!(!article
        .red_links
        .iter()
        .any(|r| matches!(r.target.as_str(), "memory:decision" | "memory:spec")));
    assert!(article.references.is_empty());
    assert!(article
        .dependency_keys
        .contains(&"node/federation:cache:verified".into()));
    // Short ids are not globally blacklisted because some foreign origin also
    // happened to call its block "1".
    put(&mut read, "1", "document", "Local one.");
    assert!(fold_article(&read, "1", ArticleTier::Full).is_ok());
}

#[test]
fn article_fold_receipts_deduplicate_and_pins_remove_folded_body() {
    let (mut read, _) = fixture();
    put(&mut read, "receipt:actual", "receipt", "Fixture receipt.");
    let receipt = CitedEvidence {
        reference: Reference {
            id: ReferenceId("receipt:actual".into()),
            receipt_id: "receipt:actual".into(),
            source_kind: "fixture-replay".into(),
            replay_handle: "fixture:actual".into(),
        },
        dependency_node_ids: vec!["receipt:actual".into()],
    };
    read.receipts
        .insert("repo:theorem".into(), vec![receipt.clone()]);
    let mut alias_receipt = receipt;
    alias_receipt.reference.id = ReferenceId("receipt-alias".into());
    read.receipts
        .insert("memory:decision".into(), vec![alias_receipt]);
    let article = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert_eq!(article.references.len(), 1);
    assert!(article
        .lead
        .iter()
        .all(|s| s.citations == [ReferenceId("receipt:actual".into())]));
    assert_eq!(
        article.sections[0].citations,
        [ReferenceId("receipt:actual".into())]
    );
    put(&mut read, "doc:pin", "document", "Authored block.");
    put(&mut read, "artifact:pin", "artifact", "Snapshot bytes.");
    read.pins.push(PinnedSection {
        section_id: "decisions".into(),
        block: BlockRef {
            document_id: "doc:pin".into(),
            block_id: "b1".into(),
            position: 8,
            content_digest: "fixture-document-digest".into(),
        },
        dependency_node_ids: vec!["doc:pin".into(), "artifact:pin".into()],
    });
    let pinned = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert!(matches!(pinned.sections[0].body, SectionBody::Empty));
    assert!(pinned.sections[0].pinned.is_some());
    assert!(pinned.sections[0].citations.is_empty());
    assert!(pinned.dependency_keys.contains(&"node/artifact:pin".into()));
}

#[test]
fn article_fold_aliases_follow_real_corroborated_collapse_not_egraph_shadows() {
    let mut graph = InMemoryGraphStore::new();
    graph
        .upsert_node(NodeRecord::new(
            "repo:a",
            ["Repo"],
            json!({"title":"Canonical repository","gist":"Actual members."}),
        ))
        .unwrap();
    graph
        .upsert_node(NodeRecord::new(
            "repo:b",
            ["Repo"],
            json!({"title":"Old repository"}),
        ))
        .unwrap();
    for id in ["fact:one", "fact:two"] {
        graph
            .upsert_node(NodeRecord::new(id, ["Value"], json!({})))
            .unwrap();
    }
    propose_same_as(
        &mut graph,
        "repo:a",
        "repo:b",
        Confidence::new(0.98),
        vec![String::from("fact:one"), String::from("fact:two")],
    )
    .unwrap();
    let canonical = collapse_if_corroborated(
        &mut graph,
        &["repo:a".into(), "repo:b".into()],
        Confidence::new(0.95),
    )
    .unwrap()
    .unwrap();
    let mut read = FixtureArticleRead {
        nodes: graph
            .live_node_records()
            .into_iter()
            .map(|n| (n.id.clone(), n))
            .collect(),
        edges: graph.live_edge_records(),
        ..Default::default()
    };
    let article = fold_article(&read, "repo:b", ArticleTier::Full).unwrap();
    assert_eq!(article.entity_id, canonical.id);
    assert_eq!(article.title, "Canonical repository");
    assert_eq!(article.hatnote.unwrap().alias, "repo:b");
    assert!(article.cluster_ids.contains(&"repo:a".into()));
    for edge in &mut read.edges {
        if edge.edge_type == "SameEClass" {
            edge.properties["evidence"] = json!("egraph_congruence");
            edge.properties["quarantine"] = json!(true);
        }
    }
    assert_eq!(
        fold_article(&read, "repo:b", ArticleTier::Full)
            .unwrap()
            .entity_id,
        "repo:b"
    );
}

#[test]
fn article_fold_exact_fact_alias_collision_is_refused() {
    let mut read = FixtureArticleRead::default();
    for index in 0..2 {
        let fact = format!("fact:{index}");
        let alias = format!("alias:{index}");
        read.nodes.insert(
            fact.clone(),
            NodeRecord::new(
                &fact,
                ["CanonicalFact"],
                json!({"typed_fact_hash":format!("typed:{index}")}),
            ),
        );
        read.nodes.insert(alias.clone(), NodeRecord::new(&alias, ["AliasWitness"], json!({"alias":"old name","canonical_node_id":fact,"typed_fact_hash":format!("typed:{index}"),"authority":"witness_not_truth"})));
        read.edges.push(EdgeRecord::new(
            format!("alias-edge:{index}"),
            alias,
            "ALIAS_WITNESS_FOR",
            fact,
            json!({}),
        ));
    }
    assert!(matches!(
        fold_article(&read, "old name", ArticleTier::Full),
        Err(ArticleFoldError::AmbiguousIdentity(_))
    ));
    read.nodes.remove("alias:1");
    read.edges.pop();
    let article = fold_article(&read, "old name", ArticleTier::Full).unwrap();
    assert_eq!(article.entity_id, "fact:0");
    assert_eq!(article.hatnote.unwrap().alias, "old name");
}

#[test]
fn article_fold_link_caps_and_order_ignore_storage_enumeration() {
    let (mut read, _) = fixture();
    for index in 0..31 {
        let id = format!("many:{index:02}");
        put(&mut read, &id, "document", "A source.");
        read.edges.push(EdgeRecord::new(
            format!("many-edge:{index}"),
            id,
            "REFERENCES",
            "repo:theorem",
            json!({"source":"fixture"}),
        ));
    }
    let expected = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    assert_eq!(expected.backlinks.len(), 20);
    assert_eq!(expected.see_also.len(), 20);
    read.edges.reverse();
    assert_eq!(
        fold_article(&read, "theorem", ArticleTier::Full).unwrap(),
        expected
    );
}

#[test]
fn article_fold_gist_tier_does_not_publish_hidden_section_receipts() {
    let (read, _) = fixture();
    let article = fold_article(&read, "theorem", ArticleTier::Gist).unwrap();
    assert!(!article.lead.is_empty());
    assert!(article.sections.is_empty());
    assert!(article.infobox.is_empty());
    assert!(article.references.is_empty());
}

#[test]
fn article_fold_person_rows_respect_typed_edge_direction() {
    let mut read = FixtureArticleRead::default();
    put(&mut read, "person", "person", "A person.");
    put(&mut read, "org", "organization", "The organization.");
    put(&mut read, "project", "project", "The project.");
    put(
        &mut read,
        "incoming-project",
        "project",
        "Different direction.",
    );
    read.nodes.get_mut("person").unwrap().properties["role"] = json!("Engineer");
    for (id, from, relation, to) in [
        ("member", "person", "MEMBER_OF", "org"),
        ("works", "person", "WORKS_ON", "project"),
        ("incoming", "incoming-project", "WORKS_ON", "person"),
    ] {
        read.edges
            .push(EdgeRecord::new(id, from, relation, to, json!({})));
    }
    let article = fold_article(&read, "person", ArticleTier::Full).unwrap();
    assert_eq!(
        article
            .infobox
            .iter()
            .map(|row| row.id.as_str())
            .collect::<Vec<_>>(),
        ["role", "organization", "projects"]
    );
    assert_eq!(
        article.infobox[2].values,
        [InfoboxValue::Link {
            link: link(read.nodes.get("project").unwrap())
        }]
    );
}

#[test]
fn article_fold_real_history_abi_pins_revision_and_current_admission() {
    use super::snapshot::HistoryArticleRead;
    use crate::fold::{history::HistoryFold, FoldRuntime};
    use crate::graph::fold_store::FoldGraphStore;
    let directory = tempfile::tempdir().unwrap();
    let deployment = theorem_oplog::DeploymentId::new("test:article-fold").unwrap();
    let mut store = FoldGraphStore::open(directory.path(), deployment).unwrap();
    let (fixture, _) = fixture();
    let snapshot = GraphSnapshot {
        nodes: fixture.nodes.into_values().collect(),
        edges: fixture
            .edges
            .into_iter()
            .filter(|e| e.to_id != "future:target")
            .collect(),
        ..Default::default()
    };
    let operations = snapshot
        .nodes
        .iter()
        .cloned()
        .map(GraphMutation::NodeUpsert)
        .chain(
            snapshot
                .edges
                .iter()
                .cloned()
                .map(GraphMutation::EdgeUpsert),
        );
    store
        .commit_batch(GraphMutationBatch::new(operations))
        .unwrap();
    let old_position = store.latest_position();
    let mut next = snapshot
        .nodes
        .iter()
        .find(|n| n.id == "memory:decision")
        .unwrap()
        .clone();
    next.properties["gist"] = json!("An actual later write.");
    next.properties["content"] = json!("An actual later write.");
    store.upsert_node(next).unwrap();
    let mut history = FoldRuntime::for_log(HistoryFold::default(), store.log()).unwrap();
    history.apply_all(store.log()).unwrap();
    let pins = |_: &str| Ok(vec![]);
    let allowed = |_: &NodeRecord| true;
    let before = HistoryArticleRead::new(
        history.inner(),
        old_position,
        "fixture",
        &allowed,
        &pins,
        12,
        "stream:one",
        100,
    )
    .unwrap();
    let after = HistoryArticleRead::new(
        history.inner(),
        store.latest_position(),
        "fixture",
        &allowed,
        &pins,
        13,
        "stream:one",
        100,
    )
    .unwrap();
    let old = fold_article(&before, "theorem", ArticleTier::Full).unwrap();
    let new = fold_article(&after, "theorem", ArticleTier::Full).unwrap();
    assert_ne!(old.sections[0].body, new.sections[0].body);
    assert_ne!(
        old.revision.commit.as_ref().unwrap().id,
        new.revision.commit.as_ref().unwrap().id
    );
    assert_eq!(
        old.revision.commit.as_ref().unwrap().log_position,
        Some(old_position.0)
    );
    assert_eq!(old.revision.commit.as_ref().unwrap().transaction, None);
    assert_eq!(
        old,
        fold_article(&before, "theorem", ArticleTier::Full).unwrap()
    );
    let revoked = |node: &NodeRecord| node.id != "repo:theorem";
    let gated = HistoryArticleRead::new(
        history.inner(),
        old_position,
        "fixture",
        &revoked,
        &pins,
        12,
        "stream:one",
        100,
    )
    .unwrap();
    assert_eq!(
        fold_article(&gated, "theorem", ArticleTier::Full),
        Err(ArticleFoldError::Excluded)
    );
}

#[test]
fn article_fold_real_foreign_cache_signature_and_legacy_origin_exclusion() {
    use super::snapshot::HistoryArticleRead;
    use crate::federated_blocks::{
        FederatedGraphStore, FederatedVisibility, ForeignBlockRecord, ForeignOriginBinding,
        ForeignVerificationReceipt, VerifiedForeignBlock,
    };
    use crate::fold::{history::HistoryFold, FoldRuntime};
    use crate::graph::fold_store::FoldGraphStore;
    use ed25519_dalek::{Signer, SigningKey};

    let key = SigningKey::from_bytes(&[7; 32]);
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let binding = ForeignOriginBinding {
        origin_identity: "foreign-tenant".into(),
        resolved_endpoint_id: "fixture-authenticated-endpoint".into(),
        origin_verifying_key_hex: hex(&key.verifying_key().to_bytes()),
    };
    let mut block = ForeignBlockRecord {
        block_url: "theorem://foreign/block/1".into(),
        block_id: "1".into(),
        origin_identity: binding.origin_identity.clone(),
        shape_ref: "fixture@1".into(),
        state: json!({"text":"Foreign material"}),
        custody_chain: vec![binding.origin_identity.clone()],
        visibility: FederatedVisibility::Public,
        revision: 1,
        content_hash: String::new(),
        public_embedding: None,
    };
    block.content_hash = block.computed_content_hash().unwrap();
    let receipt = ForeignVerificationReceipt {
        origin_identity: binding.origin_identity.clone(),
        resolved_endpoint_id: binding.resolved_endpoint_id.clone(),
        origin_verifying_key_hex: binding.origin_verifying_key_hex.clone(),
        signature_hex: hex(&key.sign(block.content_hash.as_bytes()).to_bytes()),
        content_hash: block.content_hash.clone(),
    };
    let mut graph = InMemoryGraphStore::new();
    FederatedGraphStore::new(&mut graph, "fixture")
        .cache_verified(
            VerifiedForeignBlock {
                block,
                verification: receipt,
            },
            &binding,
        )
        .unwrap();
    let mut cache = graph.live_node_records().into_iter().next().unwrap();
    // Actual legacy serialization omitted this redundant boolean. Identity
    // binding still proves foreign origin and must continue to exclude it.
    cache
        .properties
        .as_object_mut()
        .unwrap()
        .remove("foreign_origin");
    let mut corrupted = cache.clone();
    corrupted.properties["verified"]["block"]["state"] = json!({"text":"Unverified replacement"});
    assert!(matches!(
        source::verified_source_admission(&corrupted, "theorem://foreign/block/1", "fixture"),
        Err(ArticleFoldError::InvalidAuthority(_))
    ));
    let foreign = NodeRecord::new(
        "foreign-article",
        ["MemoryDocument"],
        json!({"kind":"spec","title":"Foreign text must not appear","source_ref":"theorem://foreign/block/1","gist":"Not visible."}),
    );
    let local = NodeRecord::new(
        "1",
        ["Document"],
        json!({"title":"Local same short id","gist":"Still visible."}),
    );
    let directory = tempfile::tempdir().unwrap();
    let mut store = FoldGraphStore::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:article-foreign").unwrap(),
    )
    .unwrap();
    store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(cache),
            GraphMutation::NodeUpsert(foreign),
            GraphMutation::NodeUpsert(local),
        ]))
        .unwrap();
    let mut history = FoldRuntime::for_log(HistoryFold::default(), store.log()).unwrap();
    history.apply_all(store.log()).unwrap();
    let admission = |_: &NodeRecord| true;
    let pins = |_: &str| Ok(vec![]);
    let read = HistoryArticleRead::new(
        history.inner(),
        store.latest_position(),
        "fixture",
        &admission,
        &pins,
        0,
        "fixture-stream",
        100,
    )
    .unwrap();
    assert_eq!(
        fold_article(&read, "foreign-article", ArticleTier::Full),
        Err(ArticleFoldError::Excluded)
    );
    assert_eq!(
        fold_article(&read, "1", ArticleTier::Full).unwrap().title,
        "Local same short id"
    );
}

#[test]
fn article_fold_history_visibility_uses_explicit_pin_not_log_position() {
    use super::snapshot::HistoryArticleRead;
    use crate::fold::{history::HistoryFold, FoldRuntime};
    use crate::graph::fold_store::FoldGraphStore;
    let directory = tempfile::tempdir().unwrap();
    let mut store = FoldGraphStore::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:article-visibility").unwrap(),
    )
    .unwrap();
    store.upsert_node(NodeRecord::new("timed", ["Document"], json!({"title":"Visible at the pinned time","gist":"Preserved source.","_ttl_expires_at_ms":4102444800000_i64}))).unwrap();
    let mut history = FoldRuntime::for_log(HistoryFold::default(), store.log()).unwrap();
    history.apply_all(store.log()).unwrap();
    let allowed = |_: &NodeRecord| true;
    let pins = |_: &str| Ok(vec![]);
    let early = HistoryArticleRead::new(
        history.inner(),
        store.latest_position(),
        "fixture",
        &allowed,
        &pins,
        1,
        "fixture-stream",
        4102444799999,
    )
    .unwrap();
    let late = HistoryArticleRead::new(
        history.inner(),
        store.latest_position(),
        "fixture",
        &allowed,
        &pins,
        1,
        "fixture-stream",
        4102444800001,
    )
    .unwrap();
    let article = fold_article(&early, "timed", ArticleTier::Full).unwrap();
    assert_eq!(
        fold_article(&late, "timed", ArticleTier::Full),
        Err(ArticleFoldError::Missing("timed".into()))
    );
    assert_eq!(
        fold_article(&early, "timed", ArticleTier::Full).unwrap(),
        article
    );
}

#[test]
fn article_fold_deleted_edge_remains_an_article_revision() {
    use super::snapshot::HistoryArticleRead;
    use crate::fold::{history::HistoryFold, FoldRuntime};
    use crate::graph::fold_store::FoldGraphStore;
    let directory = tempfile::tempdir().unwrap();
    let mut store = FoldGraphStore::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:article-edge-delete").unwrap(),
    )
    .unwrap();
    store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "article",
                ["Project"],
                json!({"title":"Article","gist":"The project."}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "decision",
                ["Decision"],
                json!({"gist":"A former decision."}),
            )),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "removed-edge",
                "article",
                "HAS_DECISION",
                "decision",
                json!({}),
            )),
        ]))
        .unwrap();
    store.delete_edge("removed-edge").unwrap();
    let head = store.latest_position();
    let mut history = FoldRuntime::for_log(HistoryFold::default(), store.log()).unwrap();
    history.apply_all(store.log()).unwrap();
    let allowed = |_: &NodeRecord| true;
    let pins = |_: &str| Ok(vec![]);
    let read = HistoryArticleRead::new(
        history.inner(),
        head,
        "fixture",
        &allowed,
        &pins,
        2,
        "fixture-stream",
        100,
    )
    .unwrap();
    let article = fold_article(&read, "article", ArticleTier::Full).unwrap();
    assert!(article.sections.is_empty());
    assert_eq!(article.revision.commit.unwrap().log_position, Some(head.0));
}
