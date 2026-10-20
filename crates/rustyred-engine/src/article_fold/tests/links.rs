use super::*;
use crate::article_fold::links::fold_article_with_links;
use theorem_surface_contracts::{
    article_links::*,
    link::{JustificationKind, LinkSpan},
    link_api::LinkRefusal,
};

fn reference(target: &str, kind: JustificationKind, above: bool) -> ArticleLinkReference {
    ArticleLinkReference {
        candidate_id: format!("fixture:{target}:{kind:?}:{above}"),
        source_id: "repo:theorem".into(),
        target: Backlink {
            entity_id: target.into(),
            title: "untrusted stale title".into(),
            slug: "stale".into(),
        },
        receipt_id: "fixture-scoped-receipt".into(),
        justification: kind,
        above_threshold: above,
    }
}
fn projection(article: &ArticleProjection) -> ArticleLinkProjection {
    ArticleLinkProjection {
        graph_root: article.revision.head.root.clone(),
        inline: vec![],
        see_also: vec![],
        masked_identity_count: 0,
        refusal: None,
    }
}
fn mention(target: &str, above: bool) -> ArticleInlineLink {
    ArticleInlineLink {
        reference: reference(target, JustificationKind::Mention, above),
        span: LinkSpan {
            block_id: "lead:0".into(),
            offset: 0,
            length: 7,
        },
        anchor_text: "Theorem".into(),
    }
}

#[test]
fn article_fold_links_calls_exact_folded_text_and_admits_only_above_threshold_mentions() {
    let (read, _) = fixture();
    let article =
        fold_article_with_links(&read, "theorem", ArticleTier::Full, |article, blocks| {
            assert_eq!(
                blocks[0],
                ArticleTextBlock {
                    block_id: "lead:0".into(),
                    source_id: "repo:theorem".into(),
                    text: "Theorem is a graph workspace.".into()
                }
            );
            assert!(blocks
                .iter()
                .any(|block| block.block_id == "decisions:memory:decision"
                    && block.text
                        == "Use a recessed sidebar and lifted article panel. Preserve the shell."));
            let mut out = projection(article);
            out.inline = vec![mention("tool:rust", false), mention("tool:rust", true)];
            Ok(out)
        })
        .unwrap();
    let links = article.links.unwrap();
    assert_eq!(links.inline.len(), 1);
    assert_eq!(links.inline[0].reference.target.title, "Rust");
    assert_eq!(links.inline[0].reference.target.slug, "tool:rust");
    assert_eq!(links.inline[0].anchor_text, "Theorem");
}

#[test]
fn article_fold_links_preserves_rank_and_deduplicates_twenty_typed_see_also_targets() {
    let (mut read, _) = fixture();
    for index in 0..25 {
        put(
            &mut read,
            &format!("candidate:{index}"),
            "spec",
            "Candidate prose.",
        );
    }
    let article = fold_article_with_links(&read, "theorem", ArticleTier::Full, |article, _| {
        let mut out = projection(article);
        out.see_also
            .push(reference("tool:rust", JustificationKind::Mention, true));
        for index in 0..25 {
            let kind = [
                JustificationKind::Similarity,
                JustificationKind::Structure,
                JustificationKind::Inference,
            ][index % 3];
            out.see_also
                .push(reference(&format!("candidate:{index}"), kind, true));
            out.see_also
                .push(reference(&format!("candidate:{index}"), kind, true));
        }
        Ok(out)
    })
    .unwrap();
    let related = article.links.unwrap().see_also;
    assert_eq!(related.len(), 20);
    assert_eq!(
        related
            .iter()
            .map(|link| link.target.entity_id.clone())
            .collect::<Vec<_>>(),
        (0..20)
            .map(|index| format!("candidate:{index}"))
            .collect::<Vec<_>>()
    );
}

#[test]
fn article_fold_links_refuses_stale_roots_and_non_utf8_or_forged_anchor_spans() {
    let (mut read, _) = fixture();
    read.nodes.get_mut("repo:theorem").unwrap().properties["gist"] = json!("Café Theorem.");
    for mode in 0..3 {
        let article = fold_article_with_links(&read, "theorem", ArticleTier::Full, |article, _| {
            let mut out = projection(article);
            let mut link = mention("tool:rust", true);
            match mode {
                0 => out.graph_root = "another-commit".into(),
                1 => {
                    link.span.offset = 4;
                    link.span.length = 1;
                    link.anchor_text = "é".into();
                }
                _ => {
                    link.span.offset = 0;
                    link.span.length = 5;
                    link.anchor_text = "forged".into();
                }
            }
            out.inline.push(link);
            Ok(out)
        })
        .unwrap();
        assert_eq!(article.lead[0].text, "Café Theorem.");
        let links = article.links.unwrap();
        assert!(links.inline.is_empty());
        assert_eq!(
            links.refusal,
            Some(if mode == 0 {
                LinkRefusal::StaleCandidate
            } else {
                LinkRefusal::ProjectionUnavailable
            })
        );
    }
}

#[test]
fn article_fold_links_excluded_endpoint_does_not_escape_through_dependency_ids() {
    let (mut read, _) = fixture();
    put(&mut read, "private:unrelated", "spec", "Private prose.");
    read.excluded.insert("private:unrelated".into());
    let article = fold_article_with_links(&read, "theorem", ArticleTier::Full, |article, _| {
        let mut out = projection(article);
        out.inline.push(mention("private:unrelated", true));
        // This scalar is an explicit native MaskRule fixture result. The
        // fold does not infer it by counting rejected endpoint identities.
        out.masked_identity_count = 3;
        Ok(out)
    })
    .unwrap();
    let wire = serde_json::to_string(&article).unwrap();
    assert!(!wire.contains("private:unrelated"));
    let links = article.links.unwrap();
    assert!(links.inline.is_empty());
    assert_eq!(links.masked_identity_count, 3);
}

#[test]
fn article_fold_links_policy_refusal_does_not_erase_authoritative_prose() {
    let (read, _) = fixture();
    let plain = fold_article(&read, "theorem", ArticleTier::Full).unwrap();
    let mut linked = fold_article_with_links(&read, "theorem", ArticleTier::Full, |_, _| {
        Err(LinkRefusal::ThresholdsNotConfigured)
    })
    .unwrap();
    assert_eq!(
        linked.links.take().unwrap().refusal,
        Some(LinkRefusal::ThresholdsNotConfigured)
    );
    assert_eq!(linked, plain);
}
