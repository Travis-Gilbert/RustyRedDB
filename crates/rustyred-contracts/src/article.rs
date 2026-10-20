//! The renderer-neutral Wiki projection. These records describe a read result,
//! never a second article store or a client-selected storage authority.

use serde::{Deserialize, Serialize};

pub const ARTICLE_SCHEMA: &str = "theorem-wiki-article/1";
pub const ARTICLE_KINDS: [&str; 12] = [
    "person",
    "organization",
    "project",
    "repo",
    "spec",
    "plan",
    "goal",
    "decision",
    "place",
    "document",
    "tool",
    "model",
];
pub const ARTICLE_LINK_LIMIT: usize = 20;
pub const ARTICLE_SECTION_ORDER: [&str; 6] = [
    "decisions",
    "specifications",
    "plans",
    "discussions",
    "evidence",
    "related",
];

/// Native read context. This describes the admitted projection; requests never
/// accept it as caller-selected authorization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ArticleReadScope {
    #[default]
    Scoped,
    PublicMetadata,
}

pub fn is_scoped_read(scope: &ArticleReadScope) -> bool {
    *scope == ArticleReadScope::Scoped
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleProjection {
    #[serde(default, skip_serializing_if = "is_scoped_read")]
    pub read_scope: ArticleReadScope,
    pub schema: String,
    pub entity_id: String,
    pub title: String,
    pub slug: String,
    pub kind: String,
    /// Every sentence preserves its own citations; receipt data stays in the
    /// footnote list rather than being interpolated into prose.
    pub lead: Vec<Sentence>,
    pub hatnote: Option<Hatnote>,
    pub infobox: Vec<InfoboxRow>,
    pub sections: Vec<Section>,
    pub references: Vec<Reference>,
    pub categories: Vec<Category>,
    pub red_links: Vec<RedLink>,
    pub backlinks: Vec<Backlink>,
    pub see_also: Vec<Backlink>,
    /// Optional native Link projection at this exact graph pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub links: Option<crate::article_links::ArticleLinkProjection>,
    pub revision: Revision,
    pub stub: bool,
    /// Sorted canonical and one-hop graph node identities.
    pub cluster_ids: Vec<String>,
    /// Sorted raw node and edge identities read to produce this projection,
    /// including aliases, receipts, pins, and absent reference targets.
    pub dependency_ids: Vec<String>,
    /// Graph object keys distinguish a node from an edge with the same id.
    pub dependency_keys: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Hatnote {
    pub alias: String,
    pub canonical_entity_id: String,
    pub canonical_title: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sentence {
    pub text: String,
    pub citations: Vec<ReferenceId>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReferenceId(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reference {
    pub id: ReferenceId,
    pub receipt_id: String,
    pub source_kind: String,
    /// Opaque, authority-issued replay target. The renderer opens it through
    /// OpenSource; it does not derive URLs or execute strings from the article.
    pub replay_handle: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InfoboxRow {
    pub id: String,
    pub label: String,
    pub values: Vec<InfoboxValue>,
    pub citations: Vec<ReferenceId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InfoboxValue {
    Text { text: String },
    Link { link: Backlink },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Section {
    pub id: String,
    pub level: u8,
    pub title: String,
    pub body: SectionBody,
    pub pinned: Option<BlockRef>,
    pub citations: Vec<ReferenceId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SectionBody {
    /// Required for authored sections: no stale folded text beneath the pin.
    Empty,
    Entries {
        entries: Vec<SectionEntry>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionEntry {
    /// Exact source text, including paragraph breaks and authored Markdown.
    pub source_text: String,
    pub entity_id: String,
    pub title: String,
    pub slug: String,
    pub sentences: Vec<Sentence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockRef {
    pub document_id: String,
    pub block_id: String,
    pub position: u64,
    pub content_digest: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Category {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RedLink {
    pub target: String,
    pub title: String,
    pub referring_entity_id: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Backlink {
    pub entity_id: String,
    pub slug: String,
    pub title: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Revision {
    /// Newest *known real* commit touching the dependency closure. None is
    /// explicit unavailability, never a synthetic baseline or a zero hash.
    pub commit: Option<ArticleCommit>,
    pub head: HeadReference,
    pub coverage: HistoryCoverage,
    pub seq: u64,
    /// Agent-space seq resets across process lifetimes. Comparison is valid
    /// only within this authority-issued stream epoch.
    pub stream_epoch: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleCommit {
    pub id: String,
    pub tree: String,
    pub transaction: Option<u64>,
    pub log_position: Option<u64>,
    pub author: String,
    pub summary: String,
    /// The immutable commit still has its original header. A scoped native
    /// presentation blanks author/summary when that header is not fully
    /// admitted, and marks the omission instead of inventing a new header.
    #[serde(default)]
    pub metadata_redacted: bool,
    /// Decimal text preserves the full substrate u128 timestamp on the web.
    pub timestamp_unix_ms: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadReference {
    /// Existing journals may know this hash without retaining its header.
    pub commit_id: Option<String>,
    pub root: String,
    pub transaction: Option<u64>,
    pub log_position: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum HistoryCoverage {
    Complete,
    Since {
        first_transaction: u64,
        prior_history_unavailable: bool,
    },
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn article_projection_preserves_unavailable_history_and_cited_sentences() {
        let article = ArticleProjection {
            schema: ARTICLE_SCHEMA.into(),
            read_scope: Default::default(),
            links: None,
            entity_id: "memory:doc:spec".into(),
            title: "Theorem".into(),
            slug: "Theorem".into(),
            kind: "spec".into(),
            lead: vec![Sentence {
                text: "A specification.".into(),
                citations: vec![ReferenceId("r1".into())],
            }],
            hatnote: None,
            infobox: vec![],
            sections: vec![Section {
                id: "decisions".into(),
                title: "Decisions".into(),
                level: 2,
                body: SectionBody::Empty,
                pinned: Some(BlockRef {
                    document_id: "doc:1".into(),
                    block_id: "b1".into(),
                    position: 12,
                    content_digest: "actual-document-digest".into(),
                }),
                citations: vec![],
            }],
            references: vec![Reference {
                id: ReferenceId("r1".into()),
                receipt_id: "receipt:1".into(),
                source_kind: "derivation".into(),
                replay_handle: "derivation:1".into(),
            }],
            categories: vec![],
            red_links: vec![],
            backlinks: vec![],
            see_also: vec![],
            revision: Revision {
                commit: None,
                head: HeadReference {
                    commit_id: Some("legacy-known-hash".into()),
                    root: "actual-root".into(),
                    transaction: Some(15),
                    log_position: None,
                },
                coverage: HistoryCoverage::Since {
                    first_transaction: 16,
                    prior_history_unavailable: true,
                },
                seq: 3,
                stream_epoch: "stream:a".into(),
            },
            stub: false,
            cluster_ids: vec!["memory:doc:spec".into()],
            dependency_ids: vec!["memory:doc:spec".into()],
            dependency_keys: vec!["node/memory:doc:spec".into()],
        };
        let json = serde_json::to_value(&article).unwrap();
        assert!(json["revision"]["commit"].is_null());
        assert_eq!(json["revision"]["head"]["transaction"], 15);
        assert_eq!(json["sections"][0]["body"]["kind"], "empty");
        assert_eq!(json["lead"][0]["citations"][0], "r1");
        assert_eq!(
            serde_json::from_value::<ArticleProjection>(json).unwrap(),
            article
        );
    }
}
