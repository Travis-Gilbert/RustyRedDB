//! Shared source interpretation for pinned ArticleRead implementations.
use super::*;
use crate::derivation::DerivationSpec;
use crate::federated_blocks::ForeignCacheEntry;

/// Exact persisted identity fields. The caller applies present admission and
/// visibility before returning candidate records; this is not a fuzzy resolver.
pub fn exact_identity_matches(node: &NodeRecord, exact: &str) -> bool {
    node.id == exact
        || string(node, "slug") == Some(exact)
        || (label(node, "MemoryDocument") && string(node, "doc_id") == Some(exact))
        || (label(node, "MemoryNode") && string(node, "node_id") == Some(exact))
        || (label(node, "MemoryProject") && string(node, "project_slug") == Some(exact))
        || (label(node, "AliasWitness") && string(node, "alias") == Some(exact))
}

/// Reuse the existing content-addressed derivation receipt schema. Reads must
/// stay on the same pin and distinguish exclusions from genuinely absent data.
pub fn derivation_evidence<R: ArticleRead + ?Sized>(
    read: &R,
    node: &NodeRecord,
) -> Result<Vec<CitedEvidence>, ArticleFoldError> {
    let mut ids = BTreeSet::new();
    if label(node, "Derivation") {
        ids.insert(node.id.clone());
    }
    if let Some(id) = string(node, "derivation_ref") {
        if id.starts_with(crate::memory_admission::LEGACY_MEMORY_DERIVATION_PREFIX) {
            // The canonical memory producer does not persist a receipt for this
            // observational admission. Accept only its exact source binding;
            // an arbitrary prefix must not hide a missing or forged receipt.
            let valid = label(node, "MemoryDocument")
                && !label(node, "Derivation")
                && node.properties.get("assumptions")
                    == Some(&serde_json::json!([
                        crate::memory_admission::LEGACY_MEMORY_ASSUMPTION
                    ]))
                && string(node, "tenant_slug")
                    .zip(string(node, "doc_id"))
                    .zip(string(node, "content"))
                    .is_some_and(|((tenant, document), content)| {
                        !tenant.trim().is_empty()
                            && !document.trim().is_empty()
                            && !content.trim().is_empty()
                            && id
                                == crate::memory_admission::legacy_memory_derivation_ref(
                                    tenant, document, content,
                                )
                    });
            if !valid {
                return Err(ArticleFoldError::InvalidAuthority(
                    "invalid legacy memory admission binding".into(),
                ));
            }
        } else {
            ids.insert(id.to_owned());
        }
    }
    let mut result = Vec::new();
    for id in ids {
        let receipt = match read.node(&id)? {
            ArticleNode::Present(node) => node,
            ArticleNode::Excluded => return Err(ArticleFoldError::Excluded),
            ArticleNode::Missing => {
                return Err(ArticleFoldError::IncompleteRead(
                    "derivation receipt missing".into(),
                ));
            }
        };
        if !label(&receipt, "Derivation") {
            return Err(ArticleFoldError::InvalidAuthority(
                "derivation_ref does not name a Derivation".into(),
            ));
        }
        let spec: DerivationSpec =
            serde_json::from_value(receipt.properties.clone()).map_err(|e| {
                ArticleFoldError::InvalidAuthority(format!("invalid derivation receipt: {e}"))
            })?;
        if spec
            .content_id()
            .map_err(|e| ArticleFoldError::Read(e.to_string()))?
            != receipt.id
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "derivation content identity mismatch".into(),
            ));
        }
        let mut dependencies = BTreeSet::from([id.clone(), spec.operator_id]);
        dependencies.extend(spec.input_refs);
        dependencies.extend(spec.assumption_refs);
        result.push(CitedEvidence {
            reference: Reference {
                id: ReferenceId(id.clone()),
                receipt_id: id.clone(),
                source_kind: "derivation".into(),
                replay_handle: id,
            },
            dependency_node_ids: dependencies.into_iter().collect(),
        });
    }
    Ok(result)
}

/// Interpret one exact persisted verified admission, including old rows whose
/// redundant foreign_origin boolean is absent. None means this row does not
/// bind the requested source/receiving tenant. Never match an unscoped block id.
pub fn verified_source_admission(
    node: &NodeRecord,
    source_ref: &str,
    tenant: &str,
) -> Result<Option<SourceAdmission>, ArticleFoldError> {
    if node.tombstone
        || !label(node, "FederatedForeignCache")
        || string(node, "receiving_tenant") != Some(tenant)
    {
        return Ok(None);
    }
    if node
        .properties
        .pointer("/verified/block/block_url")
        .and_then(Value::as_str)
        != Some(source_ref)
    {
        return Ok(None);
    }
    let entry: ForeignCacheEntry =
        serde_json::from_value(node.properties.clone()).map_err(|e| {
            ArticleFoldError::InvalidAuthority(format!("invalid federation admission: {e}"))
        })?;
    entry
        .verified
        .validate_against(&entry.origin_binding)
        .map_err(|e| ArticleFoldError::InvalidAuthority(format!("invalid admitted source: {e}")))?;
    let dependency_node_ids = vec![node.id.clone()];
    if entry.foreign_origin || entry.verified.block.origin_identity != tenant {
        Ok(Some(SourceAdmission::Foreign {
            dependency_node_ids,
        }))
    } else {
        Ok(Some(SourceAdmission::Local {
            dependency_node_ids,
        }))
    }
}
