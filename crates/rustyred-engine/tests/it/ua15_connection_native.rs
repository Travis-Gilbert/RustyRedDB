use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use rustyred_thg_core::{
    execute_bounded_federation_join, now_ms, publish_visibility_invalidation, AckPolicy,
    DiscoveryCandidate, FederatedGraphStore, FederatedPresence, FederatedRowProvider,
    FederatedVisibility, FederationJoinQuery, FederationRelation, ForeignBlockRecord,
    ForeignBlockResolver, ForeignOriginBinding, ForeignVerificationReceipt, GraphSnapshotSource,
    GraphStore, GraphStoreError, GraphStoreResult, GraphTtl, Projection, QueryRelation,
    RedCoreDurability, RedCoreGraphStore, RedCoreOptions, RelationalLens, ResolvedRow, ScalarValue,
    StreamStore, VerifiedForeignBlock, VisibilityInvalidation,
};
use serde_json::json;

struct FixtureResolver {
    block: VerifiedForeignBlock,
    calls: usize,
}

impl ForeignBlockResolver for FixtureResolver {
    fn fetch_verified(&mut self, block_url: &str) -> GraphStoreResult<VerifiedForeignBlock> {
        self.calls += 1;
        if self.block.block.block_url != block_url {
            return Err(GraphStoreError::new("fixture_miss", "unknown fixture URL"));
        }
        Ok(self.block.clone())
    }

    fn resolve_origin_binding(
        &mut self,
        block_url: &str,
    ) -> GraphStoreResult<ForeignOriginBinding> {
        if self.block.block.block_url != block_url {
            return Err(GraphStoreError::new("fixture_miss", "unknown fixture URL"));
        }
        Ok(binding_for(&self.block))
    }
}

struct CachedProvider<'service, 'store> {
    service: &'service mut FederatedGraphStore<'store, RedCoreGraphStore>,
    requester: &'service str,
    calls: usize,
}

impl FederatedRowProvider for CachedProvider<'_, '_> {
    fn resolve_foreign_row(&mut self, block_url: &str) -> GraphStoreResult<ResolvedRow> {
        self.calls += 1;
        self.service
            .resolve_as_relational_row(block_url, self.requester, "foreign_contacts")
    }
}

#[test]
fn ua15_connection_native_persists_resolution_custody_discovery_and_presence() {
    let dir = unique_temp_dir("persistent");
    let options = durable_options();
    let key = SigningKey::from_bytes(&[7_u8; 32]);
    let visibility = contract(1, false);
    let block_url = "https://tenant-a.example/.well-known/theorem/blocks/contact-1";
    let verified = signed_block(
        &key,
        block_url,
        visibility.clone(),
        json!({"email": "ada@example.com", "name": "Ada"}),
    );

    {
        let mut store = RedCoreGraphStore::open(&dir, options.clone()).unwrap();
        let mut federation = FederatedGraphStore::new(&mut store, "tenant-b");
        let mut resolver = FixtureResolver {
            block: verified.clone(),
            calls: 0,
        };
        let resolved = federation
            .resolve_or_fetch(block_url, "tenant-b", &mut resolver)
            .unwrap();
        assert_eq!(resolved.origin_identity, "tenant-a");
        assert_eq!(resolver.calls, 1);

        let custody = federation
            .merge_with_custody(block_url, "tenant-b", 0.92)
            .unwrap();
        assert_eq!(custody.custody_chain, vec!["tenant-a", "tenant-b"]);
        assert!(custody.graph_version > 0);
        assert_eq!(
            federation.custody_chain(block_url).unwrap(),
            vec!["tenant-a", "tenant-b"]
        );

        let public = DiscoveryCandidate {
            block_url: "https://tenant-a.example/blocks/public".into(),
            block_id: "public:block".into(),
            origin_identity: "tenant-a".into(),
            visibility: FederatedVisibility::Public,
            embedding: vec![1.0, 0.0],
        };
        let private = DiscoveryCandidate {
            block_url: "https://tenant-a.example/blocks/private".into(),
            block_id: "private:block".into(),
            origin_identity: "tenant-a".into(),
            visibility: FederatedVisibility::Private,
            embedding: vec![1.0, 0.0],
        };
        assert!(federation
            .discover(
                "tenant-b",
                &[1.0, 0.0],
                &[public.clone(), private.clone()],
                10,
            )
            .unwrap()
            .is_empty());
        federation.set_discovery_opt_in("tenant-b", true).unwrap();
        let introductions = federation
            .discover("tenant-b", &[1.0, 0.0], &[public, private], 10)
            .unwrap();
        assert_eq!(introductions.len(), 1);
        assert_eq!(introductions[0].candidate_block_id, "public:block");

        let base = now_ms() as u64 + 10_000;
        federation
            .observe_presence(FederatedPresence {
                block_url: block_url.into(),
                tenant_identity: "tenant-a".into(),
                actor_identity: "claude".into(),
                expires_at_ms: base + 100,
            })
            .unwrap();
        federation
            .observe_presence(FederatedPresence {
                block_url: block_url.into(),
                tenant_identity: "tenant-b".into(),
                actor_identity: "codex".into(),
                expires_at_ms: base + 200,
            })
            .unwrap();
        assert_eq!(
            federation
                .active_presence(block_url, base + 99)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            federation
                .active_presence(block_url, base + 100)
                .unwrap()
                .len(),
            1
        );
        assert!(federation
            .active_presence(block_url, base + 200)
            .unwrap()
            .is_empty());
    }

    // The second process/reopen reads the same graph records, not a test helper map.
    {
        let mut reopened = RedCoreGraphStore::open(&dir, options).unwrap();
        let mut federation = FederatedGraphStore::new(&mut reopened, "tenant-b");
        assert_eq!(
            federation
                .resolve_cached(block_url, "tenant-b")
                .unwrap()
                .block_id,
            "contact-1"
        );
        assert_eq!(
            federation.custody_chain(block_url).unwrap(),
            vec!["tenant-a", "tenant-b"]
        );
        assert_eq!(
            federation
                .discover("tenant-b", &[1.0, 0.0], &[], 10)
                .unwrap(),
            Vec::new()
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ua15_revocation_invalidation_redelivers_then_acks_idempotently() {
    let dir = unique_temp_dir("invalidation");
    let key = SigningKey::from_bytes(&[8_u8; 32]);
    let block_url = "https://tenant-a.example/blocks/contact-1";
    let verified = signed_block(
        &key,
        block_url,
        contract(1, false),
        json!({"email": "ada@example.com"}),
    );
    let mut store = RedCoreGraphStore::open(&dir, durable_options()).unwrap();
    let mut federation = FederatedGraphStore::new(&mut store, "tenant-b");
    cache(&mut federation, verified).unwrap();
    let private_url = "https://tenant-a.example/blocks/private";
    federation
        .cache_verified_with_fixture(signed_block(
            &key,
            private_url,
            FederatedVisibility::Private,
            json!({"email": "private@example.com"}),
        ))
        .unwrap();
    assert_eq!(
        federation
            .resolve_cached(private_url, "tenant-b")
            .unwrap_err()
            .code,
        "foreign_block_not_visible"
    );

    let read_validation_url = "https://tenant-a.example/blocks/read-validation";
    federation
        .cache_verified_with_fixture(signed_block(
            &key,
            read_validation_url,
            contract(1, false),
            json!({"email": "grace@example.com"}),
        ))
        .unwrap();
    let revoked = contract(2, true);
    federation
        .cache_verified_with_fixture(signed_block(
            &key,
            read_validation_url,
            revoked,
            json!({"email": "grace@example.com"}),
        ))
        .unwrap();
    assert_eq!(
        federation
            .resolve_cached(read_validation_url, "tenant-b")
            .unwrap_err()
            .code,
        "foreign_block_not_visible"
    );

    let mut streams = StreamStore::default();
    let event = publish_visibility_invalidation(
        &mut streams,
        "tenant-b",
        "federation-invalidations",
        "tenant-a",
        &VisibilityInvalidation {
            block_url: block_url.into(),
            visibility_revision: 2,
        },
    )
    .unwrap();
    let stream_key = event.stream_key;

    // First consumer process crashes after delivery and before applying/acking.
    let first = streams
        .read_delivery(
            "tenant-b",
            "cache-invalidator",
            &[stream_key.clone()],
            AckPolicy::Explicit,
            1,
            0,
            5,
            10,
        )
        .unwrap();
    assert_eq!(first.pending[0].delivered_count, 1);

    let receipt = federation
        .consume_visibility_invalidations(
            &mut streams,
            "tenant-b",
            "cache-invalidator",
            &stream_key,
            2,
        )
        .unwrap();
    assert_eq!(receipt.redelivered, 1);
    assert_eq!(receipt.invalidated, 1);
    assert_eq!(receipt.acknowledged, 1);
    assert!(federation.invalidate_cache(block_url, 1).unwrap());
    assert_eq!(
        federation
            .resolve_cached(block_url, "tenant-b")
            .unwrap_err()
            .code,
        "foreign_block_not_visible"
    );
    let empty = federation
        .consume_visibility_invalidations(
            &mut streams,
            "tenant-b",
            "cache-invalidator",
            &stream_key,
            3,
        )
        .unwrap();
    assert_eq!(empty.delivered, 0);
    assert_eq!(
        federation
            .cache_verified_with_fixture(signed_block(
                &key,
                block_url,
                contract(1, false),
                json!({"email": "ada@example.com"}),
            ))
            .unwrap_err()
            .code,
        "foreign_visibility_revision_regressed"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ua15_planner_joins_with_receipt_and_refuses_before_foreign_reads() {
    let dir = unique_temp_dir("planner");
    let key = SigningKey::from_bytes(&[9_u8; 32]);
    let block_url = "https://tenant-a.example/blocks/contact-1";
    let visibility = contract(1, false);
    let mut graph = RedCoreGraphStore::open(&dir, durable_options()).unwrap();
    let mut federation = FederatedGraphStore::new(&mut graph, "tenant-b");
    federation
        .cache_verified_with_fixture(signed_block(
            &key,
            block_url,
            visibility.clone(),
            json!({"email": "ada@example.com", "name": "Ada"}),
        ))
        .unwrap();

    let mut local = RelationalLens::new();
    local
        .upsert_row(ResolvedRow::new(
            "tasks",
            "task-1",
            BTreeMap::from([
                (
                    "owner_email".to_string(),
                    ScalarValue::String("ada@example.com".into()),
                ),
                ("title".to_string(), ScalarValue::String("Review".into())),
            ]),
        ))
        .unwrap();

    let mut provider = CachedProvider {
        service: &mut federation,
        requester: "tenant-b",
        calls: 0,
    };
    let result = execute_bounded_federation_join(
        &local,
        federation_query(vec![block_url.into()], 1),
        &mut provider,
    )
    .unwrap();
    assert_eq!(provider.calls, 1);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].get("contact.name"),
        Some(&ScalarValue::String("Ada".into()))
    );
    assert_eq!(result.trace.federation_foreign_reads, 1);
    assert_eq!(result.trace.federation_foreign_read_budget, Some(1));
    assert!(result.trace.federation_receipt_hash.is_some());
    assert_eq!(
        result.trace.join_algorithm.as_deref(),
        Some("bounded_federation_hash_join")
    );

    provider.calls = 0;
    let error = execute_bounded_federation_join(
        &local,
        federation_query(vec![block_url.into(), "https://tenant-c/block".into()], 1),
        &mut provider,
    )
    .unwrap_err();
    assert_eq!(error.code, "federation_join_budget_exceeded");
    assert_eq!(
        provider.calls, 0,
        "bound refusal must precede network reads"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ua15_rejects_tampered_signature_before_cache_admission() {
    let dir = unique_temp_dir("tamper");
    let key = SigningKey::from_bytes(&[10_u8; 32]);
    let mut verified = signed_block(
        &key,
        "https://tenant-a.example/blocks/tampered",
        FederatedVisibility::Public,
        json!({"name": "before"}),
    );
    verified.block.state = json!({"name": "after"});
    let mut graph = RedCoreGraphStore::open(&dir, durable_options()).unwrap();
    let mut federation = FederatedGraphStore::new(&mut graph, "tenant-b");
    assert_eq!(
        cache(&mut federation, verified).unwrap_err().code,
        "foreign_verification_failed"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ua15_cache_is_tenant_scoped_and_rejects_spoofed_origin_binding() {
    let dir = unique_temp_dir("tenant-scope");
    let key = SigningKey::from_bytes(&[11_u8; 32]);
    let block_url = "https://tenant-a.example/blocks/scoped";
    let verified = signed_block(
        &key,
        block_url,
        FederatedVisibility::Public,
        json!({"name": "scoped"}),
    );
    let binding = binding_for(&verified);
    let mut store = RedCoreGraphStore::open(&dir, durable_options()).unwrap();
    {
        let mut tenant_b = FederatedGraphStore::new(&mut store, "tenant-b");
        tenant_b.cache_verified(verified.clone(), &binding).unwrap();
    }
    {
        let mut tenant_c = FederatedGraphStore::new(&mut store, "tenant-c");
        assert_eq!(
            tenant_c
                .resolve_cached(block_url, "tenant-c")
                .unwrap_err()
                .code,
            "foreign_block_not_found"
        );
        let mut spoofed = binding.clone();
        spoofed.origin_identity = "tenant-z".into();
        assert_eq!(
            tenant_c
                .cache_verified(verified, &spoofed)
                .unwrap_err()
                .code,
            "foreign_verification_failed"
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

fn federation_query(block_urls: Vec<String>, max_foreign_reads: usize) -> FederationJoinQuery {
    FederationJoinQuery {
        local: QueryRelation {
            alias: "task".into(),
            relation: "tasks".into(),
            predicates: Vec::new(),
        },
        local_join_column: "owner_email".into(),
        foreign: FederationRelation {
            alias: "contact".into(),
            block_urls,
            join_column: "email".into(),
            max_foreign_reads,
        },
        projection: vec![
            Projection {
                alias: "task".into(),
                column: "title".into(),
            },
            Projection {
                alias: "contact".into(),
                column: "name".into(),
            },
        ],
        limit: None,
    }
}

fn contract(revision: u64, revoked: bool) -> FederatedVisibility {
    FederatedVisibility::Contract {
        contract_id: "visibility:tenant-a:contact-1".into(),
        allowed_identities: vec!["tenant-b".into()],
        revision,
        revoked,
    }
}

fn signed_block(
    key: &SigningKey,
    block_url: &str,
    visibility: FederatedVisibility,
    state: serde_json::Value,
) -> VerifiedForeignBlock {
    let mut block = ForeignBlockRecord {
        block_url: block_url.into(),
        block_id: block_url.rsplit('/').next().unwrap().into(),
        origin_identity: "tenant-a".into(),
        shape_ref: "com.example.contact.record@1".into(),
        state,
        custody_chain: vec!["tenant-a".into()],
        visibility,
        revision: 1,
        content_hash: String::new(),
        public_embedding: None,
    };
    block.content_hash = block
        .computed_content_hash()
        .expect("fixture block should be content-addressable");
    let signature = key.sign(block.content_hash.as_bytes());
    VerifiedForeignBlock {
        verification: ForeignVerificationReceipt {
            origin_identity: block.origin_identity.clone(),
            resolved_endpoint_id: "fixture-endpoint".into(),
            origin_verifying_key_hex: hex(key.verifying_key().as_bytes()),
            signature_hex: hex(&signature.to_bytes()),
            content_hash: block.content_hash.clone(),
        },
        block,
    }
}

fn binding_for(verified: &VerifiedForeignBlock) -> ForeignOriginBinding {
    ForeignOriginBinding {
        origin_identity: verified.verification.origin_identity.clone(),
        resolved_endpoint_id: verified.verification.resolved_endpoint_id.clone(),
        origin_verifying_key_hex: verified.verification.origin_verifying_key_hex.clone(),
    }
}

fn cache(
    federation: &mut FederatedGraphStore<'_, RedCoreGraphStore>,
    verified: VerifiedForeignBlock,
) -> GraphStoreResult<()> {
    let binding = binding_for(&verified);
    federation.cache_verified(verified, &binding)
}

trait FixtureCache {
    fn cache_verified_with_fixture(
        &mut self,
        verified: VerifiedForeignBlock,
    ) -> GraphStoreResult<()>;
}

impl<S: GraphStore + GraphSnapshotSource + GraphTtl> FixtureCache for FederatedGraphStore<'_, S> {
    fn cache_verified_with_fixture(
        &mut self,
        verified: VerifiedForeignBlock,
    ) -> GraphStoreResult<()> {
        let binding = binding_for(&verified);
        self.cache_verified(verified, &binding)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn durable_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1,
        strict_acid: true,
    }
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "theorem-ua15-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
