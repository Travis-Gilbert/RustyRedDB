//! Verification refinery primitives.
//!
//! The refinery is a shadow over source content. It writes document records,
//! extracted claim statements, epistemic shadows, and certification artifacts
//! without mutating source corpus bytes.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::epistemic::{
    epistemic_egraph_dedup, epistemic_shadow_node_id, structural_epistemic_pass,
    EpistemicCandidatePair, EpistemicDedupConfig, EpistemicRelationInput, EpistemicRelationKind,
    EpistemicSourceKind, StructuralEpistemicConfig, StructuralEpistemicInput, EPISTEMIC_SUPPORTS,
    SAME_ECLASS, UNDERCUTS,
};
use crate::graph::capability::{GraphStore, GraphVectorPayloadAccess};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord};
use crate::state::stable_hash;
use crate::statement::{
    write_statement, StatementFieldProvenance, StatementProvenance, StatementRecord,
    StatementSemiring, STATEMENT_LABEL,
};

pub const REFINERY_DOCUMENT_LABEL: &str = "RefineryDocument";
pub const REFINERY_CLAIM_LABEL: &str = "RefineryClaim";
pub const REFINERY_SCOPE_LABEL: &str = "RefineryScope";
pub const REFINERY_HAS_DOCUMENT: &str = "RefineryHasDocument";
pub const REFINERY_SOURCE_KIND_WEB_PAGE: &str = "web_page";

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CorpusDeclaration {
    pub corpus_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal_rights: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_use: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RefineryDocumentInput {
    pub tenant_slug: String,
    pub corpus_id: String,
    pub path: String,
    pub text: String,
    pub content_hash: String,
    pub acquired_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WebRefineryDocumentInput {
    pub tenant_slug: String,
    pub corpus_id: String,
    pub path: String,
    pub text: String,
    pub content_hash: String,
    pub fetched_at_ms: i64,
    pub source_url: String,
    pub final_url: String,
    pub fetch_tier: String,
    pub source_class: String,
    pub citation_strategy: String,
    pub crawl_depth: usize,
    pub domain: String,
    pub robots_allowed: bool,
    pub robots_policy_state: String,
    pub robots_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_reliability_prior: Option<f64>,
    #[serde(default)]
    pub stale: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_content_hash: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct WebRefineryCrawlSummaryInput {
    pub tenant_slug: String,
    pub corpus_id: String,
    pub run_id: String,
    #[serde(default)]
    pub seeds: Vec<String>,
    #[serde(default)]
    pub domain_allowlist: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    pub depth: usize,
    pub page_budget: usize,
    pub pages_fetched: usize,
    pub pages_skipped_by_robots: usize,
    #[serde(default = "default_true_bool")]
    pub respect_robots: bool,
    #[serde(default)]
    pub skipped_urls: Vec<String>,
    pub fetch_window_start_ms: i64,
    pub fetch_window_end_ms: i64,
    #[serde(default)]
    pub changed_pages: usize,
    #[serde(default)]
    pub unchanged_pages: usize,
    #[serde(default)]
    pub stale_pages: usize,
    #[serde(default)]
    pub recrawl: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recrawl_of_manifest_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_terms: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal_basis: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RefineryClaimDerivationInput {
    pub tenant_slug: String,
    pub corpus_id: String,
    #[serde(default = "default_extractor_id")]
    pub extractor_id: String,
    #[serde(default = "default_extractor_model")]
    pub model_id: String,
    #[serde(default = "default_refinery_version")]
    pub extractor_version: String,
    pub computed_at_ms: i64,
    #[serde(default)]
    pub max_claims_per_document: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RefineryClaimDerivationReport {
    pub corpus_id: String,
    pub documents_seen: usize,
    pub claims_written: usize,
    pub unanchored_claims_written: usize,
    pub statement_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RefineryStandingInput {
    pub tenant_slug: String,
    pub corpus_id: String,
    #[serde(default = "default_standing_engine")]
    pub engine: String,
    #[serde(default = "default_refinery_version")]
    pub engine_version: String,
    pub computed_at_ms: i64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SourceReliabilityStanding {
    pub source_document_id: String,
    pub alpha: f64,
    pub beta: f64,
    pub mean: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VerificationQueueItem {
    pub claim_id: String,
    pub source_document_id: String,
    pub priority: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RefineryStandingReport {
    pub corpus_id: String,
    pub claims_seen: usize,
    pub supports_written: usize,
    pub undercuts_written: usize,
    pub same_eclass_written: usize,
    pub shadows_written: usize,
    pub source_reliability: Vec<SourceReliabilityStanding>,
    pub verification_queue: Vec<VerificationQueueItem>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RefineryManifestInput {
    pub tenant_slug: String,
    pub declaration: CorpusDeclaration,
    #[serde(default = "default_standing_engine")]
    pub engine: String,
    #[serde(default = "default_refinery_version")]
    pub engine_version: String,
    pub generated_at_ms: i64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CorpusFileEntry {
    pub path: String,
    pub content_hash: String,
    pub document_id: String,
    pub acquired_at_ms: i64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CertifiedClaim {
    pub claim_id: String,
    pub text_hash: String,
    pub source_document_id: String,
    pub source_path: String,
    pub source_content_hash: String,
    pub span_start: usize,
    pub span_end: usize,
    pub provenance_chain: Vec<String>,
    pub relation_refs: Vec<String>,
    pub source_reliability: f64,
    pub support_count: u64,
    pub undercut_count: u64,
    pub same_eclass_count: u64,
    pub grounded_status: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RefinerySignature {
    pub algorithm: String,
    pub public_key_hex: String,
    pub signature_hex: String,
    pub signed_manifest_hash: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RefineryManifest {
    pub schema: String,
    pub tenant_slug: String,
    pub corpus_id: String,
    pub generated_at_ms: i64,
    pub declaration: CorpusDeclaration,
    pub corpus_files: Vec<CorpusFileEntry>,
    pub certified_claims: Vec<CertifiedClaim>,
    pub source_reliability: Vec<SourceReliabilityStanding>,
    pub contradiction_density: f64,
    pub coverage: BTreeMap<String, Value>,
    pub tool_versions: BTreeMap<String, String>,
    pub manifest_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<RefinerySignature>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ProvEntity {
    pub id: String,
    pub kind: String,
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ProvActivity {
    pub id: String,
    pub kind: String,
    pub used: Vec<String>,
    pub generated: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ProvAgent {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ProvExport {
    pub prefix: BTreeMap<String, String>,
    pub entity: Vec<ProvEntity>,
    pub activity: Vec<ProvActivity>,
    pub agent: Vec<ProvAgent>,
    pub was_derived_from: Vec<BTreeMap<String, String>>,
    pub was_generated_by: Vec<BTreeMap<String, String>>,
    pub was_attributed_to: Vec<BTreeMap<String, String>>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsExport {
    pub standard: String,
    pub standard_version: String,
    pub dataset_id: String,
    #[serde(rename = "source")]
    pub source_group: DpsSourceGroup,
    pub provenance: DpsProvenanceGroup,
    #[serde(rename = "use")]
    pub use_group: DpsUseGroup,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub theorem_extensions: BTreeMap<String, Value>,
    #[serde(
        rename = "declared_source",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal_rights: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,
    pub generation_method: String,
    pub generated_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_use: Option<String>,
    pub corpus_file_count: usize,
    pub certified_claim_count: usize,
    pub manifest_hash: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsSourceGroup {
    pub version: String,
    pub title: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    pub issuer: Vec<DpsOrganization>,
    pub description: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsProvenanceGroup {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<DpsOrganization>,
    #[serde(rename = "origin-geography")]
    pub origin_geography: Vec<DpsLocation>,
    pub date: String,
    #[serde(
        rename = "previous-date",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub previous_date: Option<String>,
    #[serde(
        rename = "generation-period",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub generation_period: Option<DpsPeriod>,
    #[serde(rename = "generation-method")]
    pub generation_method: Vec<DpsConcept>,
    pub format: Vec<String>,
    #[serde(
        rename = "sub-provenance",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub sub_provenance: Vec<DpsProvenanceGroup>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsUseGroup {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classification: Vec<DpsClassification>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub consents: Vec<String>,
    #[serde(
        rename = "privacy-enhancing",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub privacy_enhancing: Vec<DpsPrivacyEnhancing>,
    #[serde(
        rename = "processing-included",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub processing_included: Vec<DpsLocation>,
    #[serde(
        rename = "processing-excluded",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub processing_excluded: Vec<DpsLocation>,
    #[serde(
        rename = "storage-allowed",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub storage_allowed: Vec<DpsLocation>,
    #[serde(
        rename = "storage-forbidden",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub storage_forbidden: Vec<DpsLocation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub license: Vec<String>,
    #[serde(rename = "intended-purpose")]
    pub intended_purpose: Vec<DpsConcept>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub copyright: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patent: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trademark: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsOrganization {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsLocation {
    pub description: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsPeriod {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsConcept {
    pub code: String,
    pub description: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsClassification {
    pub domain: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DpsPrivacyEnhancing {
    pub code: String,
    pub description: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RefineryCertification {
    pub manifest: RefineryManifest,
    pub prov: ProvExport,
    pub dps: DpsExport,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RefinerySpanVerificationReport {
    pub ok: bool,
    pub checked_files: usize,
    pub checked_claims: usize,
    pub failures: Vec<String>,
}

pub fn write_refinery_document<S: GraphStore>(
    store: &mut S,
    input: RefineryDocumentInput,
) -> GraphStoreResult<NodeRecord> {
    validate_scope(&input.tenant_slug, &input.corpus_id)?;
    let document_id = refinery_document_id(
        &input.tenant_slug,
        &input.corpus_id,
        &input.path,
        &input.content_hash,
    )?;
    let node = NodeRecord::new(
        document_id.clone(),
        ["Content", REFINERY_DOCUMENT_LABEL],
        json!({
            "tenant_slug": input.tenant_slug,
            "corpus_id": input.corpus_id,
            "path": input.path,
            "content": input.text,
            "content_hash": input.content_hash,
            "acquired_at_ms": input.acquired_at_ms,
            "media_type": input.media_type,
            "source_ref": input.source_ref,
            "source_kind": "corpus_file",
            "certification_source": true
        }),
    );
    store.upsert_node(node.clone())?;
    let tenant = prop_str(&node.properties, "tenant_slug").unwrap_or_default();
    let corpus = prop_str(&node.properties, "corpus_id").unwrap_or_default();
    let scope = NodeRecord::new(
        refinery_scope_id(&tenant, &corpus)?,
        [REFINERY_SCOPE_LABEL],
        json!({ "tenant_slug": tenant, "corpus_id": corpus }),
    );
    store.upsert_node(scope.clone())?;
    store.upsert_edge(
        EdgeRecord::new(
            refinery_scope_document_edge_id(&scope.id, &document_id)?,
            scope.id,
            REFINERY_HAS_DOCUMENT,
            document_id,
            json!({ "corpus_id": corpus }),
        )
        .with_epistemic_type(EpistemicType::Derives),
    )?;
    Ok(node)
}

pub fn write_web_refinery_document<S: GraphStore>(
    store: &mut S,
    input: WebRefineryDocumentInput,
) -> GraphStoreResult<NodeRecord> {
    validate_scope(&input.tenant_slug, &input.corpus_id)?;
    let document_id = refinery_document_id(
        &input.tenant_slug,
        &input.corpus_id,
        &input.path,
        &input.content_hash,
    )?;
    let reliability_prior = input
        .source_reliability_prior
        .unwrap_or_else(|| source_class_reliability_prior(&input.source_class))
        .clamp(0.0, 1.0);
    let node = NodeRecord::new(
        document_id.clone(),
        ["Content", REFINERY_DOCUMENT_LABEL],
        json!({
            "tenant_slug": input.tenant_slug,
            "corpus_id": input.corpus_id,
            "path": input.path,
            "content": input.text,
            "content_hash": input.content_hash,
            "acquired_at_ms": input.fetched_at_ms,
            "media_type": input.media_type,
            "source_ref": input.source_ref,
            "source_kind": REFINERY_SOURCE_KIND_WEB_PAGE,
            "certification_source": true,
            "source_url": input.source_url,
            "final_url": input.final_url,
            "fetch_timestamp_ms": input.fetched_at_ms,
            "fetch_tier": input.fetch_tier,
            "source_class": input.source_class,
            "citation_strategy": input.citation_strategy,
            "crawl_depth": input.crawl_depth,
            "domain": input.domain,
            "robots_allowed": input.robots_allowed,
            "robots_policy_state": input.robots_policy_state,
            "robots_reason": input.robots_reason,
            "run_id": input.run_id,
            "source_reliability_prior": reliability_prior,
            "stale": input.stale,
            "previous_content_hash": input.previous_content_hash
        }),
    );
    store.upsert_node(node.clone())?;
    let tenant = prop_str(&node.properties, "tenant_slug").unwrap_or_default();
    let corpus = prop_str(&node.properties, "corpus_id").unwrap_or_default();
    let scope = NodeRecord::new(
        refinery_scope_id(&tenant, &corpus)?,
        [REFINERY_SCOPE_LABEL],
        json!({ "tenant_slug": tenant, "corpus_id": corpus, "source_kind": "web_corpus" }),
    );
    store.upsert_node(scope.clone())?;
    store.upsert_edge(
        EdgeRecord::new(
            refinery_scope_document_edge_id(&scope.id, &document_id)?,
            scope.id,
            REFINERY_HAS_DOCUMENT,
            document_id,
            json!({ "corpus_id": corpus, "source_kind": REFINERY_SOURCE_KIND_WEB_PAGE }),
        )
        .with_epistemic_type(EpistemicType::Derives),
    )?;
    Ok(node)
}

pub fn write_web_refinery_crawl_summary<S: GraphStore>(
    store: &mut S,
    input: WebRefineryCrawlSummaryInput,
) -> GraphStoreResult<NodeRecord> {
    validate_scope(&input.tenant_slug, &input.corpus_id)?;
    let scope_id = refinery_scope_id(&input.tenant_slug, &input.corpus_id)?;
    let mut properties = store
        .get_node(&scope_id)
        .map(|node| node.properties.clone())
        .unwrap_or_else(|| json!({}));
    if !properties.is_object() {
        properties = json!({});
    }
    let map = properties
        .as_object_mut()
        .expect("object created when properties was not an object");
    map.insert("tenant_slug".to_string(), json!(input.tenant_slug));
    map.insert("corpus_id".to_string(), json!(input.corpus_id));
    map.insert("source_kind".to_string(), json!("web_corpus"));
    map.insert("web_run_id".to_string(), json!(input.run_id));
    map.insert("web_seeds".to_string(), json!(input.seeds));
    map.insert(
        "web_domain_allowlist".to_string(),
        json!(input.domain_allowlist),
    );
    map.insert("web_topic".to_string(), json!(input.topic));
    map.insert("web_depth".to_string(), json!(input.depth));
    map.insert("web_page_budget".to_string(), json!(input.page_budget));
    map.insert("web_pages_fetched".to_string(), json!(input.pages_fetched));
    map.insert(
        "web_pages_skipped_by_robots".to_string(),
        json!(input.pages_skipped_by_robots),
    );
    map.insert(
        "web_respected_robots".to_string(),
        json!(input.respect_robots),
    );
    map.insert("web_skipped_urls".to_string(), json!(input.skipped_urls));
    map.insert(
        "web_fetch_window_start_ms".to_string(),
        json!(input.fetch_window_start_ms),
    );
    map.insert(
        "web_fetch_window_end_ms".to_string(),
        json!(input.fetch_window_end_ms),
    );
    map.insert("web_changed_pages".to_string(), json!(input.changed_pages));
    map.insert(
        "web_unchanged_pages".to_string(),
        json!(input.unchanged_pages),
    );
    map.insert("web_stale_pages".to_string(), json!(input.stale_pages));
    map.insert("web_recrawl".to_string(), json!(input.recrawl));
    map.insert(
        "web_recrawl_of_manifest_hash".to_string(),
        json!(input.recrawl_of_manifest_hash),
    );
    map.insert("web_source_terms".to_string(), json!(input.source_terms));
    map.insert("web_legal_basis".to_string(), json!(input.legal_basis));

    let node = NodeRecord::new(scope_id, [REFINERY_SCOPE_LABEL], properties);
    store.upsert_node(node.clone())?;
    Ok(node)
}

pub fn derive_refinery_claims<S: GraphStore>(
    store: &mut S,
    input: RefineryClaimDerivationInput,
) -> GraphStoreResult<RefineryClaimDerivationReport> {
    validate_scope(&input.tenant_slug, &input.corpus_id)?;
    let mut documents = refinery_documents(store, &input.tenant_slug, &input.corpus_id);
    documents.sort_by(|left, right| left.id.cmp(&right.id));

    let mut report = RefineryClaimDerivationReport {
        corpus_id: input.corpus_id.clone(),
        documents_seen: documents.len(),
        ..RefineryClaimDerivationReport::default()
    };
    for document in documents {
        let text = prop_str(&document.properties, "content").unwrap_or_default();
        let claims = extract_span_claims(&text, input.max_claims_per_document);
        for claim in claims {
            let anchored = claim.end > claim.start && text.get(claim.start..claim.end).is_some();
            let provenance = StatementProvenance::new(
                StatementFieldProvenance::new(
                    EpistemicSourceKind::Learned,
                    input.extractor_id.clone(),
                    input.extractor_version.clone(),
                    input.computed_at_ms,
                ),
                [
                    document.id.clone(),
                    prop_str(&document.properties, "content_hash").unwrap_or_default(),
                ],
                "refinery_claim_extraction",
                StatementSemiring::Why,
            );
            let mut statement = StatementRecord::derive_literal(
                &document.id,
                "refinery_claim",
                json!(claim.text),
                provenance,
                json!({
                    "tenant_slug": input.tenant_slug,
                    "corpus_id": input.corpus_id,
                    "claim_text": claim.text,
                    "text_hash": sha256_string(&claim.text),
                    "source_document_id": document.id,
                    "source_path": prop_str(&document.properties, "path"),
                    "source_content_hash": prop_str(&document.properties, "content_hash"),
                    "span_start": claim.start,
                    "span_end": claim.end,
                    "anchored": anchored,
                    "certification_eligible": anchored,
                    "extractor_id": input.extractor_id,
                    "model_id": input.model_id,
                    "computed_at_ms": input.computed_at_ms,
                    "confidence": 1.0
                }),
            )?;
            if !statement
                .labels
                .iter()
                .any(|label| label == REFINERY_CLAIM_LABEL)
            {
                statement.labels.push(REFINERY_CLAIM_LABEL.to_string());
                statement.labels.sort();
            }
            let receipt = write_statement(store, statement, true)?;
            if anchored {
                report.claims_written += 1;
            } else {
                report.unanchored_claims_written += 1;
            }
            report.statement_ids.push(receipt.statement_id);
        }
    }
    report.statement_ids.sort();
    report.statement_ids.dedup();
    Ok(report)
}

pub fn run_refinery_standing<S: GraphStore + GraphVectorPayloadAccess>(
    store: &mut S,
    input: RefineryStandingInput,
) -> GraphStoreResult<RefineryStandingReport> {
    validate_scope(&input.tenant_slug, &input.corpus_id)?;
    let claims = certified_claim_nodes(store, &input.tenant_slug, &input.corpus_id);
    let claim_ids = claims
        .iter()
        .map(|node| node.id.clone())
        .collect::<Vec<_>>();
    let relations = deterministic_relations(&claims);
    let undercuts_written = relations
        .iter()
        .filter(|relation| relation.kind == EpistemicRelationKind::Undercuts)
        .count();
    let supports_written = relations
        .iter()
        .filter(|relation| relation.kind == EpistemicRelationKind::Supports)
        .count();
    let readout = structural_epistemic_pass(
        store,
        StructuralEpistemicInput {
            batch_node_ids: claim_ids.clone(),
            candidate_pairs: Vec::<EpistemicCandidatePair>::new(),
            explicit_relations: relations,
            config: StructuralEpistemicConfig {
                engine: input.engine.clone(),
                engine_version: input.engine_version.clone(),
                computed_at: input.computed_at_ms,
                ..StructuralEpistemicConfig::default()
            },
        },
    )?;
    let dedup = epistemic_egraph_dedup(
        store,
        &claim_ids,
        EpistemicDedupConfig {
            engine: input.engine.clone(),
            engine_version: input.engine_version.clone(),
            computed_at: input.computed_at_ms,
            ..EpistemicDedupConfig::default()
        },
    )?;
    let source_reliability = compute_source_reliability(store, &claims, &input.engine_version)?;
    let verification_queue = compute_verification_queue(store, &claims, &input.engine_version)?;
    Ok(RefineryStandingReport {
        corpus_id: input.corpus_id,
        claims_seen: claims.len(),
        supports_written,
        undercuts_written,
        same_eclass_written: dedup.same_eclass_edges_written,
        shadows_written: readout.shadows.len(),
        source_reliability,
        verification_queue,
    })
}

pub fn build_refinery_manifest<S: GraphStore>(
    store: &S,
    input: RefineryManifestInput,
) -> GraphStoreResult<RefineryCertification> {
    validate_scope(&input.tenant_slug, &input.declaration.corpus_id)?;
    let corpus_id = input.declaration.corpus_id.clone();
    let mut documents = refinery_documents(store, &input.tenant_slug, &corpus_id);
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    let corpus_files = documents
        .iter()
        .map(|document| CorpusFileEntry {
            path: prop_str(&document.properties, "path").unwrap_or_default(),
            content_hash: prop_str(&document.properties, "content_hash").unwrap_or_default(),
            document_id: document.id.clone(),
            acquired_at_ms: prop_i64(&document.properties, "acquired_at_ms").unwrap_or(0),
        })
        .collect::<Vec<_>>();

    let claims = certified_claim_nodes(store, &input.tenant_slug, &corpus_id);
    let reliability = compute_source_reliability(store, &claims, &input.engine_version)?;
    let reliability_by_doc = reliability
        .iter()
        .map(|entry| (entry.source_document_id.clone(), entry.mean))
        .collect::<BTreeMap<_, _>>();
    let certified_claims = claims
        .iter()
        .map(|claim| {
            certified_claim_from_node(store, claim, &input.engine_version, &reliability_by_doc)
        })
        .collect::<GraphStoreResult<Vec<_>>>()?;
    let undercuts = certified_claims
        .iter()
        .map(|claim| claim.undercut_count as usize)
        .sum::<usize>();
    let contradiction_density = if certified_claims.is_empty() {
        0.0
    } else {
        undercuts as f64 / certified_claims.len() as f64
    };
    let mut coverage = BTreeMap::new();
    coverage.insert("documents".to_string(), json!(corpus_files.len()));
    coverage.insert(
        "certified_claims".to_string(),
        json!(certified_claims.len()),
    );
    coverage.insert(
        "excluded_unanchored_claims".to_string(),
        json!(excluded_unanchored_claims(
            store,
            &input.tenant_slug,
            &corpus_id
        )),
    );
    let mut tool_versions = BTreeMap::new();
    tool_versions.insert("refinery".to_string(), input.engine_version.clone());
    tool_versions.insert("standing_engine".to_string(), input.engine);
    let mut manifest = RefineryManifest {
        schema: "theorem.refinery.manifest.v1".to_string(),
        tenant_slug: input.tenant_slug,
        corpus_id,
        generated_at_ms: input.generated_at_ms,
        declaration: input.declaration,
        corpus_files,
        certified_claims,
        source_reliability: reliability,
        contradiction_density,
        coverage,
        tool_versions,
        manifest_hash: String::new(),
        signature: None,
    };
    manifest.manifest_hash = manifest_hash(&manifest)
        .map_err(|error| GraphStoreError::new("refinery_manifest_hash", error))?;
    let prov = prov_export(&manifest);
    let dps = dps_export(&manifest);
    Ok(RefineryCertification {
        manifest,
        prov,
        dps,
    })
}

pub fn sign_refinery_manifest(
    manifest: &mut RefineryManifest,
    signing_key_hex: &str,
) -> Result<(), String> {
    let normalized_key = signing_key_hex
        .trim()
        .strip_prefix("ed25519:")
        .unwrap_or(signing_key_hex.trim());
    let bytes = decode_hex(normalized_key)?;
    let key_bytes: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!("ed25519 signing key must be 32 bytes, got {}", bytes.len())
    })?;
    manifest.signature = None;
    manifest.manifest_hash = manifest_hash(manifest)?;
    let signing_key = SigningKey::from_bytes(&key_bytes);
    let signature = signing_key.sign(manifest.manifest_hash.as_bytes());
    manifest.signature = Some(RefinerySignature {
        algorithm: "ed25519".to_string(),
        public_key_hex: encode_hex(&signing_key.verifying_key().to_bytes()),
        signature_hex: encode_hex(&signature.to_bytes()),
        signed_manifest_hash: manifest.manifest_hash.clone(),
    });
    Ok(())
}

pub fn verify_refinery_manifest_signature(manifest: &RefineryManifest) -> Result<(), String> {
    let Some(signature) = manifest.signature.as_ref() else {
        return Err("manifest is unsigned".to_string());
    };
    if signature.algorithm != "ed25519" {
        return Err(format!(
            "unsupported signature algorithm {}",
            signature.algorithm
        ));
    }
    let mut unsigned = manifest.clone();
    unsigned.signature = None;
    let hash = manifest_hash(&unsigned)?;
    if hash != signature.signed_manifest_hash || hash != manifest.manifest_hash {
        return Err("manifest hash does not match signed hash".to_string());
    }
    let public_bytes = decode_hex(&signature.public_key_hex)?;
    let public_bytes: [u8; 32] = public_bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!("ed25519 public key must be 32 bytes, got {}", bytes.len())
    })?;
    let signature_bytes = decode_hex(&signature.signature_hex)?;
    let signature_bytes: [u8; 64] = signature_bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!("ed25519 signature must be 64 bytes, got {}", bytes.len())
    })?;
    let verifying_key =
        VerifyingKey::from_bytes(&public_bytes).map_err(|error| error.to_string())?;
    verifying_key
        .verify(hash.as_bytes(), &Signature::from_bytes(&signature_bytes))
        .map_err(|error| format!("signature verification failed: {error}"))
}

pub fn verify_refinery_manifest_spans(
    manifest: &RefineryManifest,
    corpus_root: &Path,
) -> RefinerySpanVerificationReport {
    let mut report = RefinerySpanVerificationReport {
        ok: true,
        ..RefinerySpanVerificationReport::default()
    };
    if let Err(error) = verify_refinery_manifest_signature(manifest) {
        report.ok = false;
        report.failures.push(error);
    }
    for file in &manifest.corpus_files {
        let path = corpus_root.join(&file.path);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                report.ok = false;
                report.failures.push(format!("{}: {error}", file.path));
                continue;
            }
        };
        report.checked_files += 1;
        let hash = sha256_bytes(&bytes);
        if hash != file.content_hash {
            report.ok = false;
            report.failures.push(format!(
                "{}: content hash mismatch expected {} got {}",
                file.path, file.content_hash, hash
            ));
        }
        let text = String::from_utf8_lossy(&bytes);
        for claim in manifest
            .certified_claims
            .iter()
            .filter(|claim| claim.source_path == file.path)
        {
            report.checked_claims += 1;
            let Some(span) = text.get(claim.span_start..claim.span_end) else {
                report.ok = false;
                report.failures.push(format!(
                    "{}: invalid span for {}",
                    file.path, claim.claim_id
                ));
                continue;
            };
            if sha256_string(span) != claim.text_hash {
                report.ok = false;
                report.failures.push(format!(
                    "{}: span hash mismatch for {}",
                    file.path, claim.claim_id
                ));
            }
        }
    }
    report
}

fn refinery_documents<S: GraphStore>(
    store: &S,
    tenant_slug: &str,
    corpus_id: &str,
) -> Vec<NodeRecord> {
    store.query_nodes(
        NodeQuery::label(REFINERY_DOCUMENT_LABEL)
            .with_property("tenant_slug", json!(tenant_slug))
            .with_property("corpus_id", json!(corpus_id))
            .with_limit(100_000),
    )
}

fn certified_claim_nodes<S: GraphStore>(
    store: &S,
    tenant_slug: &str,
    corpus_id: &str,
) -> Vec<NodeRecord> {
    let mut nodes = store.query_nodes(
        NodeQuery::label(STATEMENT_LABEL)
            .with_property("tenant_slug", json!(tenant_slug))
            .with_property("corpus_id", json!(corpus_id))
            .with_property("certification_eligible", json!(true))
            .with_limit(100_000),
    );
    nodes.retain(|node| {
        node.labels
            .iter()
            .any(|label| label == REFINERY_CLAIM_LABEL)
    });
    nodes.sort_by(|left, right| left.id.cmp(&right.id));
    nodes
}

fn extract_span_claims(text: &str, max_claims: usize) -> Vec<SpanClaim> {
    let mut claims = Vec::new();
    let mut start = 0;
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '\n') {
            push_span_claim(text, start, idx, &mut claims);
            start = idx + ch.len_utf8();
            if max_claims > 0 && claims.len() >= max_claims {
                return claims;
            }
        }
    }
    push_span_claim(text, start, text.len(), &mut claims);
    if max_claims > 0 && claims.len() > max_claims {
        claims.truncate(max_claims);
    }
    claims
}

#[derive(Clone, Debug)]
struct SpanClaim {
    text: String,
    start: usize,
    end: usize,
}

fn push_span_claim(text: &str, start: usize, end: usize, claims: &mut Vec<SpanClaim>) {
    let Some(raw) = text.get(start..end) else {
        return;
    };
    let trimmed = raw.trim();
    if trimmed.split_whitespace().count() < 3 {
        return;
    }
    let leading = raw.len() - raw.trim_start().len();
    let trailing = raw.len() - raw.trim_end().len();
    claims.push(SpanClaim {
        text: trimmed.to_string(),
        start: start + leading,
        end: end.saturating_sub(trailing),
    });
}

fn deterministic_relations(claims: &[NodeRecord]) -> Vec<EpistemicRelationInput> {
    let mut relations = Vec::new();
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            let left_text = prop_str(&claims[i].properties, "claim_text").unwrap_or_default();
            let right_text = prop_str(&claims[j].properties, "claim_text").unwrap_or_default();
            let left = normalized_claim_basis(&left_text);
            let right = normalized_claim_basis(&right_text);
            if left.prop.is_empty() || right.prop.is_empty() || left.prop != right.prop {
                continue;
            }
            // Same proposition, opposite negation attacks the conclusion, not
            // the inference. EpistemicRelationKind carries only Supports and
            // Undercuts, so the finer row travels in canonical_edge_type.
            let (kind, canonical_edge_type) = if left.negated != right.negated {
                (
                    EpistemicRelationKind::Undercuts,
                    Some("Contradicts".to_string()),
                )
            } else {
                (EpistemicRelationKind::Supports, None)
            };
            relations.push(EpistemicRelationInput {
                from_content_id: claims[i].id.clone(),
                to_content_id: claims[j].id.clone(),
                kind,
                confidence: 0.92,
                evidence: "refinery_deterministic_relation".to_string(),
                source_kind: EpistemicSourceKind::Structural,
                score: Some(0.92),
                model_id: Some("refinery-deterministic-v1".to_string()),
                calibration_version: Some("fixture-v1".to_string()),
                feature_version: Some("claim-normalization-v1".to_string()),
                connection_features: None,
                canonical_edge_type,
            });
        }
    }
    relations
}

#[derive(Clone, Debug)]
struct ClaimBasis {
    prop: String,
    negated: bool,
}

fn normalized_claim_basis(text: &str) -> ClaimBasis {
    let normalized = text
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>();
    let mut negated = false;
    let tokens = normalized
        .split_whitespace()
        .filter_map(|token| {
            if matches!(token, "not" | "no" | "never" | "without") {
                negated = !negated;
                None
            } else {
                Some(token)
            }
        })
        .collect::<Vec<_>>();
    ClaimBasis {
        prop: tokens.join(" "),
        negated,
    }
}

fn compute_source_reliability<S: GraphStore>(
    store: &S,
    claims: &[NodeRecord],
    engine_version: &str,
) -> GraphStoreResult<Vec<SourceReliabilityStanding>> {
    let mut scores = BTreeMap::<String, (f64, f64)>::new();
    for claim in claims {
        let source = prop_str(&claim.properties, "source_document_id").unwrap_or_default();
        let shadow_id = epistemic_shadow_node_id(&claim.id, engine_version)?;
        let shadow = store.get_node(&shadow_id);
        let support = shadow
            .and_then(|node| prop_u64(&node.properties, "support_in_degree"))
            .unwrap_or(0);
        let attack = shadow
            .and_then(|node| prop_u64(&node.properties, "attack_in_degree"))
            .unwrap_or(0);
        let status = shadow
            .and_then(|node| prop_str(&node.properties, "grounded_extension_status"))
            .unwrap_or_else(|| "undecided".to_string());
        let entry = scores.entry(source).or_insert((1.0, 1.0));
        if status == "in" {
            entry.0 += 1.0 + support as f64;
        } else if status == "out" {
            entry.1 += 1.0 + attack as f64;
        } else {
            entry.0 += 0.5;
            entry.1 += 0.5;
        }
    }
    Ok(scores
        .into_iter()
        .map(
            |(source_document_id, (alpha, beta))| SourceReliabilityStanding {
                source_document_id,
                alpha,
                beta,
                mean: alpha / (alpha + beta),
            },
        )
        .collect())
}

fn compute_verification_queue<S: GraphStore>(
    store: &S,
    claims: &[NodeRecord],
    engine_version: &str,
) -> GraphStoreResult<Vec<VerificationQueueItem>> {
    let mut queue = Vec::with_capacity(claims.len());
    for claim in claims {
        let shadow_id = epistemic_shadow_node_id(&claim.id, engine_version)?;
        let shadow = store.get_node(&shadow_id);
        let support = shadow
            .and_then(|node| prop_u64(&node.properties, "support_in_degree"))
            .unwrap_or(0);
        let attack = shadow
            .and_then(|node| prop_u64(&node.properties, "attack_in_degree"))
            .unwrap_or(0);
        let status = shadow
            .and_then(|node| prop_str(&node.properties, "grounded_extension_status"))
            .unwrap_or_else(|| "undecided".to_string());
        let ambiguity = if status == "undecided" { 1.0 } else { 0.0 };
        let conflict = attack as f64 / (support + attack + 1) as f64;
        queue.push(VerificationQueueItem {
            claim_id: claim.id.clone(),
            source_document_id: prop_str(&claim.properties, "source_document_id")
                .unwrap_or_default(),
            priority: ambiguity + conflict,
            reason: if ambiguity > 0.0 {
                "grounded_status_undecided".to_string()
            } else if attack > 0 {
                "claim_has_undercut".to_string()
            } else {
                "low_marginal_value".to_string()
            },
        });
    }
    queue.sort_by(|left, right| {
        right
            .priority
            .partial_cmp(&left.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.claim_id.cmp(&right.claim_id))
    });
    Ok(queue)
}

fn certified_claim_from_node<S: GraphStore>(
    store: &S,
    claim: &NodeRecord,
    engine_version: &str,
    reliability_by_doc: &BTreeMap<String, f64>,
) -> GraphStoreResult<CertifiedClaim> {
    let shadow_id = epistemic_shadow_node_id(&claim.id, engine_version)?;
    let shadow = store.get_node(&shadow_id);
    let support_count = shadow
        .and_then(|node| prop_u64(&node.properties, "support_in_degree"))
        .unwrap_or(0);
    let undercut_count = shadow
        .and_then(|node| prop_u64(&node.properties, "attack_in_degree"))
        .unwrap_or(0);
    let grounded_status = shadow
        .and_then(|node| prop_str(&node.properties, "grounded_extension_status"))
        .unwrap_or_else(|| "undecided".to_string());
    let relation_refs = relation_refs_for_claim(store, &shadow_id);
    let source_document_id = prop_str(&claim.properties, "source_document_id").unwrap_or_default();
    Ok(CertifiedClaim {
        claim_id: claim.id.clone(),
        text_hash: prop_str(&claim.properties, "text_hash").unwrap_or_default(),
        source_document_id: source_document_id.clone(),
        source_path: prop_str(&claim.properties, "source_path").unwrap_or_default(),
        source_content_hash: prop_str(&claim.properties, "source_content_hash").unwrap_or_default(),
        span_start: prop_u64(&claim.properties, "span_start").unwrap_or(0) as usize,
        span_end: prop_u64(&claim.properties, "span_end").unwrap_or(0) as usize,
        provenance_chain: StatementRecord::why(claim),
        relation_refs,
        source_reliability: reliability_by_doc
            .get(&source_document_id)
            .copied()
            .unwrap_or(0.5),
        support_count,
        undercut_count,
        same_eclass_count: same_eclass_count(store, &shadow_id),
        grounded_status,
    })
}

fn relation_refs_for_claim<S: GraphStore>(store: &S, shadow_id: &str) -> Vec<String> {
    let mut refs = store
        .neighbors(NeighborQuery::out(shadow_id))
        .into_iter()
        .chain(store.neighbors(NeighborQuery::in_(shadow_id)))
        .filter_map(|hit| store.get_edge(&hit.edge_id).map(|edge| edge.id.clone()))
        .filter(|id| {
            id.contains(EPISTEMIC_SUPPORTS) || id.contains(UNDERCUTS) || id.contains(SAME_ECLASS)
        })
        .collect::<Vec<_>>();
    refs.sort();
    refs.dedup();
    refs
}

fn same_eclass_count<S: GraphStore>(store: &S, shadow_id: &str) -> u64 {
    store
        .neighbors(NeighborQuery::out(shadow_id).with_edge_type(SAME_ECLASS))
        .len() as u64
        + store
            .neighbors(NeighborQuery::in_(shadow_id).with_edge_type(SAME_ECLASS))
            .len() as u64
}

fn prov_export(manifest: &RefineryManifest) -> ProvExport {
    let mut export = ProvExport::default();
    export
        .prefix
        .insert("prov".to_string(), "http://www.w3.org/ns/prov#".to_string());
    export.agent.push(ProvAgent {
        id: "agent:theorem-refinery".to_string(),
        kind: "softwareAgent".to_string(),
    });
    let activity_id = format!("activity:refinery:{}", manifest.manifest_hash);
    export.activity.push(ProvActivity {
        id: activity_id.clone(),
        kind: "certification".to_string(),
        used: manifest
            .corpus_files
            .iter()
            .map(|file| file.document_id.clone())
            .collect(),
        generated: manifest
            .certified_claims
            .iter()
            .map(|claim| claim.claim_id.clone())
            .collect(),
    });
    for file in &manifest.corpus_files {
        export.entity.push(ProvEntity {
            id: file.document_id.clone(),
            kind: "document".to_string(),
            attributes: BTreeMap::from([
                ("path".to_string(), json!(file.path)),
                ("content_hash".to_string(), json!(file.content_hash)),
            ]),
        });
    }
    for claim in &manifest.certified_claims {
        export.entity.push(ProvEntity {
            id: claim.claim_id.clone(),
            kind: "claim".to_string(),
            attributes: BTreeMap::from([
                ("text_hash".to_string(), json!(claim.text_hash)),
                ("grounded_status".to_string(), json!(claim.grounded_status)),
                (
                    "source_span".to_string(),
                    json!([claim.span_start, claim.span_end]),
                ),
            ]),
        });
        export.was_derived_from.push(BTreeMap::from([
            ("generatedEntity".to_string(), claim.claim_id.clone()),
            ("usedEntity".to_string(), claim.source_document_id.clone()),
        ]));
        export.was_generated_by.push(BTreeMap::from([
            ("entity".to_string(), claim.claim_id.clone()),
            ("activity".to_string(), activity_id.clone()),
        ]));
        export.was_attributed_to.push(BTreeMap::from([
            ("entity".to_string(), claim.claim_id.clone()),
            ("agent".to_string(), "agent:theorem-refinery".to_string()),
        ]));
    }
    export
}

fn dps_export(manifest: &RefineryManifest) -> DpsExport {
    let generated_at = unix_ms_to_iso8601(manifest.generated_at_ms);
    let issuer_name = manifest
        .declaration
        .source
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Theorem Refinery".to_string());
    let origin = manifest
        .declaration
        .source
        .as_ref()
        .map(|source| DpsOrganization {
            name: source.clone(),
            url: None,
        });
    let formats = manifest_media_types(manifest);
    let intended_description = manifest
        .declaration
        .intended_use
        .clone()
        .unwrap_or_else(|| "Theorem refinery epistemic certification".to_string());
    let mut theorem_extensions = BTreeMap::new();
    theorem_extensions.insert("manifest_hash".to_string(), json!(manifest.manifest_hash));
    theorem_extensions.insert(
        "corpus_file_count".to_string(),
        json!(manifest.corpus_files.len()),
    );
    theorem_extensions.insert(
        "certified_claim_count".to_string(),
        json!(manifest.certified_claims.len()),
    );
    theorem_extensions.insert(
        "contradiction_density".to_string(),
        json!(manifest.contradiction_density),
    );
    theorem_extensions.insert("coverage".to_string(), json!(manifest.coverage));
    DpsExport {
        standard: "Data & Trust Alliance Data Provenance Standards".to_string(),
        standard_version: "1.0.0".to_string(),
        dataset_id: manifest.corpus_id.clone(),
        source_group: DpsSourceGroup {
            version: "1.0.0".to_string(),
            title: manifest.corpus_id.clone(),
            id: format!(
                "urn:theorem:refinery:{}:{}",
                manifest.corpus_id, manifest.manifest_hash
            ),
            location: None,
            issuer: vec![DpsOrganization {
                name: issuer_name,
                url: None,
            }],
            description: format!(
                "Theorem verification refinery certification metadata for corpus {}.",
                manifest.corpus_id
            ),
        },
        provenance: DpsProvenanceGroup {
            source: manifest.declaration.lineage.clone(),
            origin,
            origin_geography: vec![DpsLocation {
                description: "not declared by operator".to_string(),
            }],
            date: generated_at.clone(),
            previous_date: None,
            generation_period: None,
            generation_method: vec![DpsConcept {
                code: "machine-generated".to_string(),
                description:
                    "Theorem verification refinery claim extraction, standing derivation, and manifest certification"
                        .to_string(),
            }],
            format: formats,
            sub_provenance: Vec::new(),
        },
        use_group: DpsUseGroup {
            classification: manifest
                .declaration
                .privacy
                .as_ref()
                .map(|privacy| DpsClassification {
                    domain: "operator-declared-privacy".to_string(),
                    value: privacy.clone(),
                })
                .into_iter()
                .collect(),
            consents: Vec::new(),
            privacy_enhancing: Vec::new(),
            processing_included: Vec::new(),
            processing_excluded: Vec::new(),
            storage_allowed: Vec::new(),
            storage_forbidden: Vec::new(),
            license: manifest
                .declaration
                .legal_rights
                .clone()
                .into_iter()
                .collect(),
            intended_purpose: vec![DpsConcept {
                code: "ai-other".to_string(),
                description: intended_description,
            }],
            copyright: Vec::new(),
            patent: Vec::new(),
            trademark: Vec::new(),
        },
        theorem_extensions,
        source: manifest.declaration.source.clone(),
        legal_rights: manifest.declaration.legal_rights.clone(),
        privacy: manifest.declaration.privacy.clone(),
        lineage: manifest.declaration.lineage.clone(),
        generation_method: "Theorem verification refinery claim-level epistemic certification"
            .to_string(),
        generated_at_ms: manifest.generated_at_ms,
        intended_use: manifest.declaration.intended_use.clone(),
        corpus_file_count: manifest.corpus_files.len(),
        certified_claim_count: manifest.certified_claims.len(),
        manifest_hash: manifest.manifest_hash.clone(),
    }
}

fn manifest_media_types(manifest: &RefineryManifest) -> Vec<String> {
    let mut formats = manifest
        .corpus_files
        .iter()
        .filter_map(|file| {
            if file.path.ends_with(".csv") {
                Some("text/csv")
            } else if file.path.ends_with(".json") || file.path.ends_with(".jsonl") {
                Some("application/json")
            } else if file.path.ends_with(".html") || file.path.ends_with(".htm") {
                Some("text/html")
            } else if file.path.ends_with(".md")
                || file.path.ends_with(".markdown")
                || file.path.ends_with(".txt")
            {
                Some("text/plain")
            } else {
                None
            }
        })
        .map(str::to_string)
        .collect::<Vec<_>>();
    formats.sort();
    formats.dedup();
    if formats.is_empty() {
        formats.push("text/plain".to_string());
    }
    formats
}

fn unix_ms_to_iso8601(ms: i64) -> String {
    let seconds = ms.div_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let shifted = days_since_epoch + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    }
    .div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524
        - day_of_era / 146_096)
        .div_euclid(365);
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_param = (5 * day_of_year + 2).div_euclid(153);
    let day = day_of_year - (153 * month_param + 2).div_euclid(5) + 1;
    let month = month_param + if month_param < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year, month as u32, day as u32)
}

fn excluded_unanchored_claims<S: GraphStore>(
    store: &S,
    tenant_slug: &str,
    corpus_id: &str,
) -> usize {
    store
        .query_nodes(
            NodeQuery::label(STATEMENT_LABEL)
                .with_property("tenant_slug", json!(tenant_slug))
                .with_property("corpus_id", json!(corpus_id))
                .with_property("certification_eligible", json!(false))
                .with_limit(100_000),
        )
        .len()
}

fn manifest_hash(manifest: &RefineryManifest) -> Result<String, String> {
    let mut unsigned = manifest.clone();
    unsigned.signature = None;
    unsigned.manifest_hash.clear();
    serde_json::to_vec(&unsigned)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| format!("manifest serialization failed: {error}"))
}

fn validate_scope(tenant_slug: &str, corpus_id: &str) -> GraphStoreResult<()> {
    if tenant_slug.trim().is_empty() {
        return Err(GraphStoreError::new(
            "missing_tenant_slug",
            "tenant_slug is required",
        ));
    }
    if corpus_id.trim().is_empty() {
        return Err(GraphStoreError::new(
            "missing_corpus_id",
            "corpus_id is required",
        ));
    }
    Ok(())
}

fn refinery_scope_id(tenant_slug: &str, corpus_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "refinery:scope:{}",
        stable_hash(json!([tenant_slug, corpus_id]))?
    ))
}

fn refinery_document_id(
    tenant_slug: &str,
    corpus_id: &str,
    path: &str,
    content_hash: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "refinery:document:{}",
        stable_hash(json!([tenant_slug, corpus_id, path, content_hash]))?
    ))
}

fn refinery_scope_document_edge_id(scope_id: &str, document_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "refinery:has-document:{}",
        stable_hash(json!([scope_id, document_id]))?
    ))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn sha256_string(text: &str) -> String {
    sha256_bytes(text.as_bytes())
}

fn prop_str(properties: &Value, key: &str) -> Option<String> {
    properties
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn prop_i64(properties: &Value, key: &str) -> Option<i64> {
    properties.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
    })
}

fn prop_u64(properties: &Value, key: &str) -> Option<u64> {
    properties.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
    })
}

pub fn source_class_reliability_prior(source_class: &str) -> f64 {
    match source_class.trim().to_ascii_lowercase().as_str() {
        "paper" | "government" | "docs" => 0.82,
        "github" | "stackoverflow" | "news" | "pdf" => 0.68,
        "unknown" => 0.50,
        "forum" | "blog" | "product" => 0.38,
        _ => 0.50,
    }
}

fn default_true_bool() -> bool {
    true
}

fn default_extractor_id() -> String {
    "refinery.span_extractor".to_string()
}

fn default_extractor_model() -> String {
    "configured-model-lane-or-deterministic-fixture".to_string()
}

fn default_standing_engine() -> String {
    "refinery.standing".to_string()
}

fn default_refinery_version() -> String {
    "refinery-v1".to_string()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    let value = value.trim();
    if value.len() % 2 != 0 {
        return Err("hex value has odd length".to_string());
    }
    (0..value.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&value[idx..idx + 2], 16).map_err(|error| error.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_negation_pair_is_tagged_as_a_contradiction_not_an_undercut() {
        // Same proposition, opposite negation attacks the conclusion. The kind
        // enum cannot say so, so the finer row must travel in
        // canonical_edge_type or the persisted edge claims an undercut.
        let claims = vec![
            claim_node("c1", "the cache is warm"),
            claim_node("c2", "the cache is not warm"),
            claim_node("c3", "the cache is warm"),
        ];
        let relations = deterministic_relations(&claims);

        let contradiction = relations
            .iter()
            .find(|r| r.kind == EpistemicRelationKind::Undercuts)
            .expect("the negation pair produces a defeating relation");
        assert_eq!(
            contradiction.canonical_edge_type.as_deref(),
            Some("Contradicts"),
            "a negation pair must be tagged Contradicts"
        );

        let support = relations
            .iter()
            .find(|r| r.kind == EpistemicRelationKind::Supports)
            .expect("the identical pair produces a supporting relation");
        assert_eq!(
            support.canonical_edge_type, None,
            "a support falls back to the kind's own row"
        );
    }

    fn claim_node(id: &str, text: &str) -> NodeRecord {
        NodeRecord::new(id, ["Claim"], serde_json::json!({ "claim_text": text }))
    }
    use super::*;
    use crate::InMemoryGraphStore;

    fn signing_key_hex() -> String {
        encode_hex(&[7_u8; 32])
    }

    #[test]
    fn refinery_derives_span_claims_standing_and_signed_manifest() {
        let mut store = InMemoryGraphStore::new();
        let text = "Alpha is stable. Alpha is not stable. Beta has evidence.";
        write_refinery_document(
            &mut store,
            RefineryDocumentInput {
                tenant_slug: "tenant".to_string(),
                corpus_id: "corpus".to_string(),
                path: "a.txt".to_string(),
                text: text.to_string(),
                content_hash: sha256_string(text),
                acquired_at_ms: 1,
                media_type: Some("text/plain".to_string()),
                source_ref: None,
            },
        )
        .unwrap();
        let derive = derive_refinery_claims(
            &mut store,
            RefineryClaimDerivationInput {
                tenant_slug: "tenant".to_string(),
                corpus_id: "corpus".to_string(),
                extractor_id: default_extractor_id(),
                model_id: default_extractor_model(),
                extractor_version: default_refinery_version(),
                computed_at_ms: 2,
                max_claims_per_document: 0,
            },
        )
        .unwrap();
        assert_eq!(derive.claims_written, 3);
        let standing = run_refinery_standing(
            &mut store,
            RefineryStandingInput {
                tenant_slug: "tenant".to_string(),
                corpus_id: "corpus".to_string(),
                computed_at_ms: 3,
                engine: default_standing_engine(),
                engine_version: default_refinery_version(),
            },
        )
        .unwrap();
        assert!(standing.undercuts_written >= 1);
        assert!(!standing.verification_queue.is_empty());

        let mut cert = build_refinery_manifest(
            &store,
            RefineryManifestInput {
                tenant_slug: "tenant".to_string(),
                declaration: CorpusDeclaration {
                    corpus_id: "corpus".to_string(),
                    source: Some("fixture".to_string()),
                    legal_rights: Some("declared".to_string()),
                    privacy: Some("none".to_string()),
                    lineage: Some("fixture".to_string()),
                    intended_use: Some("test".to_string()),
                },
                engine: default_standing_engine(),
                engine_version: default_refinery_version(),
                generated_at_ms: 4,
            },
        )
        .unwrap();
        assert_eq!(cert.manifest.certified_claims.len(), 3);
        sign_refinery_manifest(&mut cert.manifest, &signing_key_hex()).unwrap();
        verify_refinery_manifest_signature(&cert.manifest).unwrap();
        cert.manifest.certified_claims[0].span_end += 1;
        assert!(verify_refinery_manifest_signature(&cert.manifest).is_err());
    }
}
