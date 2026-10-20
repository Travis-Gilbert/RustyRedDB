//! Link annotations over exact folded article bytes. The native Link service
//! supplies evidence at the article's pin; renderers never generate candidates.
use crate::{
    article::Backlink,
    link::{JustificationKind, LinkSpan},
    link_api::LinkRefusal,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleTextBlock {
    /// Exact renderer key. Offsets refer to UTF-8 bytes in `text`.
    pub block_id: String,
    /// Authority identity of this prose, which may differ from the article.
    pub source_id: String,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleLinkReference {
    pub candidate_id: String,
    pub source_id: String,
    pub target: Backlink,
    /// Native scoped receipt endpoint; never an executable or external URL.
    pub receipt_id: String,
    pub justification: JustificationKind,
    pub above_threshold: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleInlineLink {
    pub reference: ArticleLinkReference,
    pub span: LinkSpan,
    pub anchor_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArticleLinkProjection {
    pub graph_root: String,
    pub inline: Vec<ArticleInlineLink>,
    /// Ranked by the native scorer, then deduplicated by admitted target.
    pub see_also: Vec<ArticleLinkReference>,
    /// Computed only by the existing native MaskRule projection. No private
    /// endpoint, candidate identity, receipt or feature accompanies this count.
    pub masked_identity_count: u64,
    pub refusal: Option<LinkRefusal>,
}

/// One shared key/byte mapping for the native generator and every renderer.
/// Pinned authored sections deliberately have no folded prose to annotate.
pub fn article_text_blocks(article: &crate::article::ArticleProjection) -> Vec<ArticleTextBlock> {
    let mut blocks = article
        .lead
        .iter()
        .enumerate()
        .map(|(index, sentence)| ArticleTextBlock {
            block_id: format!("lead:{index}"),
            source_id: article.entity_id.clone(),
            text: sentence.text.clone(),
        })
        .collect::<Vec<_>>();
    for section in &article.sections {
        if section.pinned.is_some() {
            continue;
        }
        if let crate::article::SectionBody::Entries { entries } = &section.body {
            blocks.extend(entries.iter().map(|entry| ArticleTextBlock {
                block_id: format!("{}:{}", section.id, entry.entity_id),
                source_id: entry.entity_id.clone(),
                text: entry.source_text.clone(),
            }));
        }
    }
    blocks
}
