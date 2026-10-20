//! Legacy memory admission identifiers shared by the native producer and its
//! readers. These mark an observational admission; they are not persisted
//! Derivation receipts and must never be rendered as proof citations.

pub const LEGACY_MEMORY_ASSUMPTION: &str = "assumption:legacy";
pub const LEGACY_MEMORY_DERIVATION_PREFIX: &str = "derivation:legacy:";

/// Preserve the exact existing producer preimage, including source Markdown.
pub fn legacy_memory_derivation_ref(tenant_slug: &str, doc_id: &str, content: &str) -> String {
    format!(
        "{LEGACY_MEMORY_DERIVATION_PREFIX}{}",
        crate::symbolic::stable_hash_value(&serde_json::json!({
            "tenant_slug": tenant_slug,
            "doc_id": doc_id,
            "content": content,
        }))
        .expect("a JSON object containing only strings always serializes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_memory_admission_preserves_original_producer_hash() {
        // SHA-256 of the canonical JSON object, captured from the same
        // preimage the product hasher used: sorted keys, compact JSON.
        let cases = [
            (
                "plain",
                "13541fda57c3cb2060e888400cfdd4fdd605f44e5ee85b9a582e49cfeeee1de6",
            ),
            (
                "Unicode λ 世界 and \"quoted\" text",
                "3d417083cf71fe2f54d59db6ea82e3e8cb48396c04a5edc8bcc2fc0aaa551df3",
            ),
            (
                "line\nline\t\\control\0",
                "683a0f6e5b750753679ca9931fe19c60531fd7cac36160028f496a83f59ab41c",
            ),
        ];
        for (source, expected) in cases {
            assert_eq!(
                legacy_memory_derivation_ref("tenant", "document", source),
                format!("{LEGACY_MEMORY_DERIVATION_PREFIX}{expected}")
            );
        }
    }
}
