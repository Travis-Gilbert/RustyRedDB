//! Connection-native foreign blocks backed by the canonical [`GraphStore`].
//!
//! This module deliberately contains no side registry. Cache entries, visibility
//! invalidations, custody, introductions, and presence are graph records, so the
//! same durable RedCore/Redis implementations and graph-version counters apply.

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph::capability::{GraphSnapshotSource, GraphStore, GraphTtl};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::relational::ResolvedRow;
use crate::stream::{AckPolicy, StreamEvent, StreamStore};
use crate::{stable_hash, ScalarValue, StreamUrgency};

pub const FOREIGN_CACHE_LABEL: &str = "FederatedForeignCache";
pub const CUSTODY_EVENT_LABEL: &str = "FederatedCustodyEvent";
pub const INTRODUCTION_LABEL: &str = "FederatedIntroduction";
pub const DISCOVERY_OPT_IN_LABEL: &str = "FederatedDiscoveryOptIn";
pub const BLOCK_PRESENCE_LABEL: &str = "FederatedBlockPresence";
pub const VISIBILITY_INVALIDATION_KIND: &str = "federation.visibility.invalidated.v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FederatedVisibility {
    Public,
    Private,
    Contract {
        contract_id: String,
        allowed_identities: Vec<String>,
        revision: u64,
        revoked: bool,
    },
}

impl FederatedVisibility {
    pub fn revision(&self) -> u64 {
        match self {
            Self::Contract { revision, .. } => *revision,
            Self::Public | Self::Private => 0,
        }
    }

    pub fn permits(&self, identity: Option<&str>) -> bool {
        match self {
            Self::Public => true,
            Self::Private => false,
            Self::Contract {
                allowed_identities,
                revoked,
                ..
            } => {
                !revoked
                    && identity
                        .map(|identity| allowed_identities.iter().any(|item| item == identity))
                        .unwrap_or(false)
            }
        }
    }

    pub fn is_public(&self) -> bool {
        matches!(self, Self::Public)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForeignBlockRecord {
    pub block_url: String,
    pub block_id: String,
    pub origin_identity: String,
    pub shape_ref: String,
    pub state: Value,
    pub custody_chain: Vec<String>,
    pub visibility: FederatedVisibility,
    pub revision: u64,
    pub content_hash: String,
    #[serde(default)]
    pub public_embedding: Option<Vec<f32>>,
}

impl ForeignBlockRecord {
    pub fn computed_content_hash(&self) -> GraphStoreResult<String> {
        stable_hash(json!({
            "block_url": self.block_url,
            "block_id": self.block_id,
            "origin_identity": self.origin_identity,
            "shape_ref": self.shape_ref,
            "state": self.state,
            "custody_chain": self.custody_chain,
            "visibility": self.visibility,
            "revision": self.revision,
            "public_embedding": self.public_embedding,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ForeignVerificationReceipt {
    pub origin_identity: String,
    /// Authenticated transport endpoint selected by the resolution record.
    pub resolved_endpoint_id: String,
    pub origin_verifying_key_hex: String,
    pub signature_hex: String,
    pub content_hash: String,
}

/// Locally trusted result of resolving an origin identity. This is deliberately
/// separate from the remote envelope: a sender cannot attest its own ownership.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ForeignOriginBinding {
    pub origin_identity: String,
    pub resolved_endpoint_id: String,
    pub origin_verifying_key_hex: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifiedForeignBlock {
    pub block: ForeignBlockRecord,
    pub verification: ForeignVerificationReceipt,
}

impl VerifiedForeignBlock {
    /// Verify content identity, the resolution-record binding, and the Ed25519
    /// signature before the block is eligible for durable cache admission.
    pub fn validate_against(&self, binding: &ForeignOriginBinding) -> GraphStoreResult<()> {
        if self.verification.resolved_endpoint_id.trim().is_empty()
            || self.verification.origin_identity != self.block.origin_identity
            || binding.origin_identity != self.block.origin_identity
            || binding.resolved_endpoint_id != self.verification.resolved_endpoint_id
            || binding.origin_verifying_key_hex != self.verification.origin_verifying_key_hex
            || self.verification.content_hash != self.block.content_hash
            || self.block.computed_content_hash()? != self.block.content_hash
        {
            return Err(verification_error());
        }
        let key_bytes = decode_array::<32>(
            &self.verification.origin_verifying_key_hex,
            "origin verifying key",
        )?;
        let signature_bytes = decode_array::<64>(&self.verification.signature_hex, "signature")?;
        let key = VerifyingKey::from_bytes(&key_bytes).map_err(|_| verification_error())?;
        let signature = Signature::from_bytes(&signature_bytes);
        key.verify(self.block.content_hash.as_bytes(), &signature)
            .map_err(|_| verification_error())
    }
}

/// Read depth admitted for a cached foreign block.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignReadDepth {
    #[default]
    Gist,
    Full,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForeignCacheEntry {
    pub receiving_tenant: String,
    pub origin_binding: ForeignOriginBinding,
    pub verified: VerifiedForeignBlock,
    pub validated_visibility_revision: u64,
    pub stale: bool,
    /// True when the block's origin identity is not the receiving tenant.
    /// Same shape as FORME `has_stochastic_origin`: a provenance dimension
    /// discharged by verification, not a caller-checked boolean.
    #[serde(default)]
    pub foreign_origin: bool,
    /// Capability grants attached at admission. Foreign origins admit with none.
    #[serde(default)]
    pub grants: Vec<String>,
    /// Maximum read depth admitted with this cache entry.
    #[serde(default)]
    pub max_read_depth: ForeignReadDepth,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CustodyMergeReceipt {
    pub block_url: String,
    pub block_id: String,
    pub origin_identity: String,
    pub recipient_identity: String,
    pub custody_chain: Vec<String>,
    pub merged_content_hash: String,
    pub graph_version: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryCandidate {
    pub block_url: String,
    pub block_id: String,
    pub origin_identity: String,
    pub visibility: FederatedVisibility,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntroductionRecord {
    pub id: String,
    pub requesting_tenant: String,
    pub candidate_block_url: String,
    pub candidate_block_id: String,
    pub origin_identity: String,
    pub similarity: f32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FederatedPresence {
    pub block_url: String,
    pub tenant_identity: String,
    pub actor_identity: String,
    pub expires_at_ms: u64,
}

impl FederatedPresence {
    pub fn active_at(&self, now_ms: u64) -> bool {
        now_ms < self.expires_at_ms
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VisibilityInvalidation {
    pub block_url: String,
    pub visibility_revision: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InvalidationReceipt {
    pub delivered: usize,
    pub invalidated: usize,
    pub acknowledged: usize,
    pub redelivered: usize,
}

/// Network/HTTP adapters implement this trait. The store remains responsible
/// for verifying and admitting the returned envelope.
pub trait ForeignBlockResolver {
    fn fetch_verified(&mut self, block_url: &str) -> GraphStoreResult<VerifiedForeignBlock>;
    fn resolve_origin_binding(&mut self, block_url: &str)
        -> GraphStoreResult<ForeignOriginBinding>;
}

/// A thin service over a caller-owned GraphStore. It owns no durable state.
pub struct FederatedGraphStore<'a, S: GraphStore + GraphSnapshotSource + GraphTtl> {
    store: &'a mut S,
    receiving_tenant: String,
}

impl<'a, S: GraphStore + GraphSnapshotSource + GraphTtl> FederatedGraphStore<'a, S> {
    pub fn new(store: &'a mut S, receiving_tenant: impl Into<String>) -> Self {
        Self {
            store,
            receiving_tenant: receiving_tenant.into(),
        }
    }

    pub fn cache_verified(
        &mut self,
        verified: VerifiedForeignBlock,
        binding: &ForeignOriginBinding,
    ) -> GraphStoreResult<()> {
        verified.validate_against(binding)?;
        if let Some(existing) = self.cache_entry(&verified.block.block_url)? {
            let incoming_revision = verified.block.visibility.revision();
            if incoming_revision < existing.validated_visibility_revision {
                return Err(GraphStoreError::new(
                    "foreign_visibility_revision_regressed",
                    format!(
                        "visibility revision {incoming_revision} is older than cached revision {}",
                        existing.validated_visibility_revision
                    ),
                ));
            }
            if incoming_revision == existing.validated_visibility_revision
                && verified.block.content_hash != existing.verified.block.content_hash
            {
                return Err(GraphStoreError::new(
                    "foreign_visibility_revision_collision",
                    "changed foreign content must advance the visibility revision",
                ));
            }
        }
        let foreign_origin = verified.block.origin_identity != self.receiving_tenant;
        let entry = ForeignCacheEntry {
            receiving_tenant: self.receiving_tenant.clone(),
            origin_binding: binding.clone(),
            validated_visibility_revision: verified.block.visibility.revision(),
            verified,
            stale: false,
            foreign_origin,
            // Foreign blocks admit with zero grants and gist depth (NW5).
            grants: Vec::new(),
            max_read_depth: if foreign_origin {
                ForeignReadDepth::Gist
            } else {
                ForeignReadDepth::Full
            },
        };
        let node = NodeRecord::new(
            cache_node_id(&self.receiving_tenant, &entry.verified.block.block_url)?,
            [FOREIGN_CACHE_LABEL],
            serde_json::to_value(entry).map_err(json_error("encode foreign cache"))?,
        );
        self.store.upsert_node(node)?;
        Ok(())
    }

    pub fn resolve_or_fetch<R: ForeignBlockResolver>(
        &mut self,
        block_url: &str,
        requester_identity: &str,
        resolver: &mut R,
    ) -> GraphStoreResult<ForeignBlockRecord> {
        if let Some(entry) = self.cache_entry(block_url)? {
            if !entry.stale {
                if let Ok(block) = self.resolve_cached(block_url, requester_identity) {
                    return Ok(block);
                }
            }
        }
        let verified = resolver.fetch_verified(block_url)?;
        let binding = resolver.resolve_origin_binding(block_url)?;
        if verified.block.block_url != block_url {
            return Err(GraphStoreError::new(
                "foreign_resolution_mismatch",
                "resolver returned a block for a different URL",
            ));
        }
        self.cache_verified(verified, &binding)?;
        self.resolve_cached(block_url, requester_identity)
    }

    pub fn resolve_cached(
        &mut self,
        block_url: &str,
        requester_identity: &str,
    ) -> GraphStoreResult<ForeignBlockRecord> {
        self.resolve_cached_at_depth(block_url, requester_identity, ForeignReadDepth::Gist)
    }

    /// Resolve a cached block at an explicit read depth.
    ///
    /// Foreign-origin entries refuse [`ForeignReadDepth::Full`].
    pub fn resolve_cached_at_depth(
        &mut self,
        block_url: &str,
        requester_identity: &str,
        depth: ForeignReadDepth,
    ) -> GraphStoreResult<ForeignBlockRecord> {
        let entry = self.cache_entry(block_url)?.ok_or_else(|| {
            GraphStoreError::new("foreign_block_not_found", "foreign block is not cached")
        })?;
        entry.verified.validate_against(&entry.origin_binding)?;
        if entry.stale
            || !entry
                .verified
                .block
                .visibility
                .permits(Some(requester_identity))
        {
            return Err(GraphStoreError::new(
                "foreign_block_not_visible",
                "cached block is stale or current visibility denies access",
            ));
        }
        if matches!(depth, ForeignReadDepth::Full)
            && matches!(entry.max_read_depth, ForeignReadDepth::Gist)
        {
            return Err(GraphStoreError::new(
                "foreign_read_depth_denied",
                "foreign-origin cache entry admits gist depth only",
            ));
        }
        if entry.foreign_origin && !entry.grants.is_empty() {
            return Err(GraphStoreError::new(
                "foreign_grants_invariant",
                "foreign-origin cache entry must carry zero grants",
            ));
        }
        Ok(entry.verified.block)
    }

    /// Inspect the cache admission record (grants, foreign_origin, depth).
    pub fn cache_admission(&self, block_url: &str) -> GraphStoreResult<Option<ForeignCacheEntry>> {
        self.cache_entry(block_url)
    }

    pub fn invalidate_cache(
        &mut self,
        block_url: &str,
        visibility_revision: u64,
    ) -> GraphStoreResult<bool> {
        let Some(mut entry) = self.cache_entry(block_url)? else {
            return Ok(false);
        };
        entry.stale = true;
        entry.validated_visibility_revision =
            entry.validated_visibility_revision.max(visibility_revision);
        self.write_cache_entry(&entry)?;
        Ok(true)
    }

    /// Authorize an origin-scoped control frame against the locally trusted
    /// binding persisted when this tenant admitted the cached block.
    pub fn require_cached_origin(
        &self,
        block_url: &str,
        binding: &ForeignOriginBinding,
    ) -> GraphStoreResult<()> {
        let entry = self.cache_entry(block_url)?.ok_or_else(|| {
            GraphStoreError::new(
                "foreign_block_not_found",
                "origin control frame targets an uncached block",
            )
        })?;
        if &entry.origin_binding != binding {
            return Err(GraphStoreError::new(
                "foreign_origin_binding_mismatch",
                "origin control frame does not match the cached resolution binding",
            ));
        }
        Ok(())
    }

    pub fn consume_visibility_invalidations(
        &mut self,
        streams: &mut StreamStore,
        tenant: &str,
        consumer: &str,
        stream_key: &str,
        now_ms: u64,
    ) -> GraphStoreResult<InvalidationReceipt> {
        let delivery = streams
            .read_delivery(
                tenant,
                consumer,
                &[stream_key.to_string()],
                AckPolicy::Explicit,
                now_ms,
                0,
                5,
                256,
            )
            .map_err(stream_error)?;
        let redelivered = delivery
            .pending
            .iter()
            .filter(|pending| pending.delivered_count > 1)
            .count();
        let mut invalidated = 0;
        let mut ack_tokens = Vec::new();
        for event in &delivery.events {
            if event.kind != VISIBILITY_INVALIDATION_KIND {
                continue;
            }
            let invalidation: VisibilityInvalidation =
                serde_json::from_value(event.payload.clone())
                    .map_err(json_error("decode visibility invalidation"))?;
            invalidated += usize::from(
                self.invalidate_cache(&invalidation.block_url, invalidation.visibility_revision)?,
            );
            ack_tokens.push(event.ordering_token);
        }
        let acknowledged = streams
            .ack(tenant, consumer, stream_key, &ack_tokens)
            .map_err(stream_error)?;
        Ok(InvalidationReceipt {
            delivered: delivery.events.len(),
            invalidated,
            acknowledged,
            redelivered,
        })
    }

    pub fn merge_with_custody(
        &mut self,
        block_url: &str,
        recipient_identity: &str,
        confidence: f64,
    ) -> GraphStoreResult<CustodyMergeReceipt> {
        let entry = self.cache_entry(block_url)?.ok_or_else(|| {
            GraphStoreError::new("foreign_block_not_found", "foreign block is not cached")
        })?;
        entry.verified.validate_against(&entry.origin_binding)?;
        let mut block = entry.verified.block;
        if block.custody_chain.last().map(String::as_str) != Some(recipient_identity) {
            block.custody_chain.push(recipient_identity.to_string());
        }
        let merged_content_hash = stable_hash(json!({
            "source_content_hash": block.content_hash,
            "state": block.state,
            "custody_chain": block.custody_chain,
            "merge_confidence": confidence.clamp(0.0, 1.0),
        }))?;
        let merged_node_id = format!("federation:merged:{}", stable_hash(block_url)?);
        self.store.upsert_node(NodeRecord::new(
            &merged_node_id,
            ["FederatedMergedBlock"],
            json!({
                "block_url": block_url,
                "block_id": block.block_id,
                "origin_identity": block.origin_identity,
                "state": block.state,
                "source_content_hash": block.content_hash,
                "merged_content_hash": merged_content_hash,
                "merge_confidence": confidence.clamp(0.0, 1.0),
            }),
        ))?;

        for identity in &block.custody_chain {
            self.store.upsert_node(NodeRecord::new(
                identity_node_id(identity)?,
                ["FederatedIdentity"],
                json!({"identity": identity}),
            ))?;
        }
        for (sequence, pair) in block.custody_chain.windows(2).enumerate() {
            self.store.upsert_edge(
                EdgeRecord::new(
                    format!(
                        "federation:custody-edge:{}",
                        stable_hash(json!([block_url, pair[0], pair[1], sequence]))?
                    ),
                    identity_node_id(&pair[0])?,
                    "FEDERATED_CUSTODY_TRANSFER",
                    identity_node_id(&pair[1])?,
                    json!({
                        "block_url": block_url,
                        "sequence": sequence,
                        "source_content_hash": block.content_hash,
                    }),
                )
                .with_confidence(confidence),
            )?;
        }
        let graph_version = self.store.stats().version;
        let receipt = CustodyMergeReceipt {
            block_url: block_url.to_string(),
            block_id: block.block_id,
            origin_identity: block.origin_identity,
            recipient_identity: recipient_identity.to_string(),
            custody_chain: block.custody_chain,
            merged_content_hash,
            graph_version,
        };
        let receipt_id = format!(
            "federation:custody:{}",
            stable_hash(json!([block_url, recipient_identity, graph_version]))?
        );
        self.store.upsert_node(NodeRecord::new(
            &receipt_id,
            [CUSTODY_EVENT_LABEL],
            serde_json::to_value(&receipt).map_err(json_error("encode custody receipt"))?,
        ))?;
        self.store.upsert_edge(EdgeRecord::new(
            format!("{receipt_id}:block"),
            receipt_id,
            "CUSTODY_OF",
            merged_node_id,
            json!({"block_url": block_url}),
        ))?;
        Ok(receipt)
    }

    pub fn custody_chain(&self, block_url: &str) -> GraphStoreResult<Vec<String>> {
        let snapshot = self.store.graph_snapshot()?;
        let mut receipts = snapshot
            .nodes
            .into_iter()
            .filter(|node| node.labels.iter().any(|label| label == CUSTODY_EVENT_LABEL))
            .filter_map(|node| serde_json::from_value::<CustodyMergeReceipt>(node.properties).ok())
            .filter(|receipt| receipt.block_url == block_url)
            .collect::<Vec<_>>();
        receipts.sort_by_key(|receipt| receipt.graph_version);
        Ok(receipts
            .pop()
            .map(|receipt| receipt.custody_chain)
            .unwrap_or_default())
    }

    pub fn set_discovery_opt_in(
        &mut self,
        tenant_identity: &str,
        enabled: bool,
    ) -> GraphStoreResult<()> {
        self.store.upsert_node(NodeRecord::new(
            discovery_opt_in_node_id(tenant_identity)?,
            [DISCOVERY_OPT_IN_LABEL],
            json!({"tenant_identity": tenant_identity, "enabled": enabled}),
        ))?;
        Ok(())
    }

    pub fn discover(
        &mut self,
        requesting_tenant: &str,
        query_embedding: &[f32],
        candidates: &[DiscoveryCandidate],
        limit: usize,
    ) -> GraphStoreResult<Vec<IntroductionRecord>> {
        let enabled = self
            .store
            .get_node_record(&discovery_opt_in_node_id(requesting_tenant)?)
            .and_then(|node| node.properties.get("enabled").and_then(Value::as_bool))
            .unwrap_or(false);
        if !enabled {
            return Ok(Vec::new());
        }
        let mut introductions = Vec::new();
        for candidate in candidates
            .iter()
            .filter(|candidate| candidate.visibility.is_public())
        {
            let Some(similarity) = cosine_similarity(query_embedding, &candidate.embedding) else {
                continue;
            };
            let id = format!(
                "federation:introduction:{}",
                stable_hash(json!([
                    requesting_tenant,
                    candidate.block_url,
                    candidate.block_id
                ]))?
            );
            introductions.push(IntroductionRecord {
                id,
                requesting_tenant: requesting_tenant.to_string(),
                candidate_block_url: candidate.block_url.clone(),
                candidate_block_id: candidate.block_id.clone(),
                origin_identity: candidate.origin_identity.clone(),
                similarity,
            });
        }
        introductions.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.candidate_block_id.cmp(&right.candidate_block_id))
        });
        introductions.truncate(limit);
        for introduction in &introductions {
            self.store.upsert_node(NodeRecord::new(
                &introduction.id,
                [INTRODUCTION_LABEL],
                serde_json::to_value(introduction)
                    .map_err(json_error("encode introduction record"))?,
            ))?;
        }
        Ok(introductions)
    }

    pub fn observe_presence(&mut self, presence: FederatedPresence) -> GraphStoreResult<()> {
        let node_id = presence_node_id(
            &presence.block_url,
            &presence.tenant_identity,
            &presence.actor_identity,
        )?;
        self.store.upsert_node(NodeRecord::new(
            &node_id,
            [BLOCK_PRESENCE_LABEL],
            serde_json::to_value(&presence).map_err(json_error("encode federated presence"))?,
        ))?;
        self.store
            .set_node_ttl(&node_id, Some(presence.expires_at_ms as i64))?;
        Ok(())
    }

    pub fn active_presence(
        &self,
        block_url: &str,
        now_ms: u64,
    ) -> GraphStoreResult<Vec<FederatedPresence>> {
        let mut presence = self
            .store
            .graph_snapshot()?
            .nodes
            .into_iter()
            .filter(|node| {
                node.labels
                    .iter()
                    .any(|label| label == BLOCK_PRESENCE_LABEL)
            })
            .filter_map(|node| serde_json::from_value::<FederatedPresence>(node.properties).ok())
            .filter(|presence| presence.block_url == block_url && presence.active_at(now_ms))
            .collect::<Vec<_>>();
        presence.sort_by(|left, right| {
            left.tenant_identity
                .cmp(&right.tenant_identity)
                .then_with(|| left.actor_identity.cmp(&right.actor_identity))
        });
        Ok(presence)
    }

    pub fn resolve_as_relational_row(
        &mut self,
        block_url: &str,
        requester_identity: &str,
        relation: &str,
    ) -> GraphStoreResult<ResolvedRow> {
        let block = self.resolve_cached(block_url, requester_identity)?;
        let values = block
            .state
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(key, value)| {
                ScalarValue::from_json(value).map(|value| (key.clone(), value))
            })
            .chain([
                (
                    "block_url".to_string(),
                    ScalarValue::String(block.block_url.clone()),
                ),
                (
                    "origin_identity".to_string(),
                    ScalarValue::String(block.origin_identity.clone()),
                ),
            ])
            .collect::<BTreeMap<_, _>>();
        let mut row = ResolvedRow::new(relation, block.block_id, values);
        row.properties = block.state;
        Ok(row)
    }

    fn cache_entry(&self, block_url: &str) -> GraphStoreResult<Option<ForeignCacheEntry>> {
        self.store
            .get_node_record(&cache_node_id(&self.receiving_tenant, block_url)?)
            .map(|node| {
                serde_json::from_value(node.properties)
                    .map_err(json_error("decode foreign cache entry"))
            })
            .transpose()
    }

    fn write_cache_entry(&mut self, entry: &ForeignCacheEntry) -> GraphStoreResult<()> {
        self.store.upsert_node(NodeRecord::new(
            cache_node_id(&self.receiving_tenant, &entry.verified.block.block_url)?,
            [FOREIGN_CACHE_LABEL],
            serde_json::to_value(entry).map_err(json_error("encode foreign cache entry"))?,
        ))?;
        Ok(())
    }
}

pub fn publish_visibility_invalidation(
    streams: &mut StreamStore,
    tenant: &str,
    topic: &str,
    actor: &str,
    invalidation: &VisibilityInvalidation,
) -> GraphStoreResult<StreamEvent> {
    streams
        .publish(
            tenant,
            topic,
            actor,
            VISIBILITY_INVALIDATION_KIND,
            serde_json::to_value(invalidation)
                .map_err(json_error("encode visibility invalidation"))?,
            StreamUrgency::Info,
            None,
        )
        .map_err(stream_error)
}

fn cache_node_id(receiving_tenant: &str, block_url: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "federation:cache:{}",
        stable_hash(json!([receiving_tenant, block_url]))?
    ))
}

fn identity_node_id(identity: &str) -> GraphStoreResult<String> {
    Ok(format!("federation:identity:{}", stable_hash(identity)?))
}

fn discovery_opt_in_node_id(identity: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "federation:discovery-opt-in:{}",
        stable_hash(identity)?
    ))
}

fn presence_node_id(block_url: &str, tenant: &str, actor: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "federation:presence:{}",
        stable_hash(json!([block_url, tenant, actor]))?
    ))
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.is_empty() || left.len() != right.len() {
        return None;
    }
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    (left_norm > 0.0 && right_norm > 0.0).then(|| dot / (left_norm * right_norm))
}

fn decode_array<const N: usize>(encoded: &str, field: &str) -> GraphStoreResult<[u8; N]> {
    let bytes = hex_decode(encoded).ok_or_else(verification_error)?;
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        GraphStoreError::new(
            "foreign_verification_failed",
            format!("{field} must be {N} bytes, got {}", bytes.len()),
        )
    })
}

fn hex_decode(encoded: &str) -> Option<Vec<u8>> {
    if !encoded.len().is_multiple_of(2) {
        return None;
    }
    (0..encoded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&encoded[index..index + 2], 16).ok())
        .collect()
}

fn verification_error() -> GraphStoreError {
    GraphStoreError::new(
        "foreign_verification_failed",
        "foreign block failed content, signature, or resolution verification",
    )
}

fn json_error(action: &'static str) -> impl FnOnce(serde_json::Error) -> GraphStoreError {
    move |error| GraphStoreError::new("federation_json_error", format!("{action}: {error}"))
}

fn stream_error(error: crate::StreamError) -> GraphStoreError {
    GraphStoreError::new(error.code, error.message)
}
