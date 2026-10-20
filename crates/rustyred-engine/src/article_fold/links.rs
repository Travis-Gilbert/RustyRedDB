//! Optional Link consumer over one immutable article read. The callback belongs
//! to the native host: core has no dependency on the Link executor or scorer.
use super::*;
use theorem_surface_contracts::{article_links::*, link::JustificationKind, link_api::LinkRefusal};

/// Fold ordinary authoritative prose, then call the candidate path with the
/// exact emitted bytes and root. Failure of the optional Link projection stays
/// explicit alongside the readable article rather than inventing empty evidence.
pub fn fold_article_with_links<R: ArticleRead + ?Sized>(
    read: &R,
    entity_id: &str,
    tier: ArticleTier,
    candidates: impl FnOnce(
        &ArticleProjection,
        &[ArticleTextBlock],
    ) -> Result<ArticleLinkProjection, LinkRefusal>,
) -> Result<ArticleProjection, ArticleFoldError> {
    let mut article = fold_article(read, entity_id, tier)?;
    let blocks = article_text_blocks(&article);
    let projected = candidates(&article, &blocks)
        .and_then(|projected| validate(read, &mut article, &blocks, projected));
    article.links = Some(match projected {
        Ok(projected) => projected,
        Err(refusal) => ArticleLinkProjection {
            graph_root: article.revision.head.root.clone(),
            inline: vec![],
            see_also: vec![],
            masked_identity_count: 0,
            refusal: Some(refusal),
        },
    });
    Ok(article)
}

fn validate<R: ArticleRead + ?Sized>(
    read: &R,
    article: &mut ArticleProjection,
    blocks: &[ArticleTextBlock],
    mut projected: ArticleLinkProjection,
) -> Result<ArticleLinkProjection, LinkRefusal> {
    if projected.graph_root != article.revision.head.root {
        return Err(LinkRefusal::StaleCandidate);
    }
    if let Some(refusal) = projected.refusal {
        return Err(refusal);
    }
    let by_id = blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block))
        .collect::<BTreeMap<_, _>>();
    let sources = blocks
        .iter()
        .map(|block| block.source_id.as_str())
        .chain(std::iter::once(article.entity_id.as_str()))
        .collect::<BTreeSet<_>>();
    let mut reads = Reads::default();
    let mut seen = BTreeSet::new();
    let mut accepted: Vec<ArticleInlineLink> = Vec::new();
    for mut inline in projected.inline {
        let Some(block) = by_id.get(inline.span.block_id.as_str()) else {
            return Err(LinkRefusal::ProjectionUnavailable);
        };
        let start =
            usize::try_from(inline.span.offset).map_err(|_| LinkRefusal::ProjectionUnavailable)?;
        let length =
            usize::try_from(inline.span.length).map_err(|_| LinkRefusal::ProjectionUnavailable)?;
        let end = start
            .checked_add(length)
            .ok_or(LinkRefusal::ProjectionUnavailable)?;
        if inline.reference.source_id != block.source_id
            || length == 0
            || block.text.get(start..end) != Some(inline.anchor_text.as_str())
        {
            return Err(LinkRefusal::ProjectionUnavailable);
        }
        if inline.reference.justification != JustificationKind::Mention
            || !inline.reference.above_threshold
        {
            continue;
        }
        if !admit_reference(read, &sources, &mut inline.reference, &mut reads)? {
            continue;
        }
        if !seen.insert(inline.reference.candidate_id.clone()) {
            continue;
        }
        // Candidates arrive in scorer rank order. The highest ranked evidence
        // owns an overlapping span; then the renderer gets source order.
        if accepted.iter().any(|other| {
            other.span.block_id == inline.span.block_id
                && other.span.offset < inline.span.offset + inline.span.length
                && inline.span.offset < other.span.offset + other.span.length
        }) {
            continue;
        }
        accepted.push(inline);
    }
    accepted.sort_by(|a, b| {
        (&a.span.block_id, a.span.offset, &a.reference.candidate_id).cmp(&(
            &b.span.block_id,
            b.span.offset,
            &b.reference.candidate_id,
        ))
    });
    let mut targets = BTreeSet::new();
    let mut related = Vec::new();
    for mut reference in projected.see_also {
        if !reference.above_threshold
            || !matches!(
                reference.justification,
                JustificationKind::Similarity
                    | JustificationKind::Structure
                    | JustificationKind::Inference
            )
        {
            continue;
        }
        if !admit_reference(read, &sources, &mut reference, &mut reads)? {
            continue;
        }
        if reference.target.entity_id != article.entity_id
            && targets.insert(reference.target.entity_id.clone())
        {
            related.push(reference);
            if related.len() == ARTICLE_LINK_LIMIT {
                break;
            }
        }
    }
    let keys = article
        .dependency_keys
        .iter()
        .cloned()
        .chain(reads.keys)
        .collect::<BTreeSet<_>>();
    // The dependency closure now includes admitted link endpoints. Its header
    // must still describe exactly the same graph pin before any article update.
    let revision = read
        .revision(&keys)
        .map_err(|_| LinkRefusal::ProjectionUnavailable)?;
    if revision.head.root != projected.graph_root {
        return Err(LinkRefusal::StaleCandidate);
    }
    article.dependency_ids.extend(reads.ids);
    article.dependency_ids.sort();
    article.dependency_ids.dedup();
    article.dependency_keys = keys.into_iter().collect();
    article.revision = revision;
    projected.inline = accepted;
    projected.see_also = related;
    projected.refusal = None;
    Ok(projected)
}

fn admit_reference<R: ArticleRead + ?Sized>(
    read: &R,
    sources: &BTreeSet<&str>,
    reference: &mut ArticleLinkReference,
    reads: &mut Reads,
) -> Result<bool, LinkRefusal> {
    if !sources.contains(reference.source_id.as_str())
        || reference.candidate_id.is_empty()
        || reference.receipt_id.is_empty()
    {
        return Err(LinkRefusal::ProjectionUnavailable);
    }
    let mut admitted_reads = Reads::default();
    let target = resolve(read, &reference.target.entity_id, &mut admitted_reads)
        .map_err(|_| LinkRefusal::ProjectionUnavailable)?;
    match target {
        ArticleNode::Excluded => Ok(false),
        ArticleNode::Missing => Err(LinkRefusal::ProjectionUnavailable),
        ArticleNode::Present(target) => {
            // Labels/slugs are read from the admitted canonical node, never
            // copied out of a potentially stale candidate presentation.
            reference.target = link(&target);
            // An excluded endpoint must not leak through dependency IDs even
            // when its candidate presentation is correctly hidden.
            reads.ids.extend(admitted_reads.ids);
            reads.keys.extend(admitted_reads.keys);
            Ok(true)
        }
    }
}
