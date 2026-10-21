//! Component and real native-file oracles for the logical transport boundary.
//! Product Document/Record/schema and real OPFS acceptance are separate cases.
use super::*;
use crate::chunk_store::MemChunkStore;
use crate::graph_store::{GraphMutation, GraphMutationBatch, NodeRecord};
use crate::rr_file::RustyRed;
use serde_json::json;
use theorem_surface_contracts::desktop::DeviceId;

#[derive(Clone)]
struct MemoryArchive(Vec<u8>);
impl ExchangeReader for MemoryArchive {
    fn len(&self) -> GraphStoreResult<u64> {
        Ok(self.0.len() as u64)
    }
    fn read_exact(&self, offset: u64, length: usize) -> GraphStoreResult<Vec<u8>> {
        let start =
            usize::try_from(offset).map_err(|_| refusal("test_offset", "offset overflow"))?;
        self.0
            .get(
                start
                    ..start
                        .checked_add(length)
                        .ok_or_else(|| refusal("test_offset", "length overflow"))?,
            )
            .map(<[u8]>::to_vec)
            .ok_or_else(|| refusal("test_short_read", "short read"))
    }
}
impl ExchangeWriter for MemoryArchive {
    fn append(&mut self, bytes: &[u8]) -> GraphStoreResult<()> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
}
fn limits() -> ExchangeLimits {
    ExchangeLimits {
        max_archive_bytes: 8 * 1024 * 1024,
        max_header_bytes: 65536,
        max_chunk_bytes: 128 * 1024,
        max_chunks: 4096,
        max_depth: 128,
    }
}
fn identity() -> ExchangeIdentity {
    ExchangeIdentity {
        tenant: "tenant:exchange".into(),
        principal: "account:exchange".into(),
    }
}
fn admit(record: &GraphPayloadRecord) -> GraphStoreResult<()> {
    let properties = match record {
        GraphPayloadRecord::Node(node) => &node.properties,
        GraphPayloadRecord::Edge(edge) => &edge.properties,
    };
    if properties.get("tenant").and_then(serde_json::Value::as_str) == Some("tenant:exchange")
        || matches!(record,GraphPayloadRecord::Node(node) if node.labels==["ControlEvent"])
    {
        Ok(())
    } else {
        Err(refusal(
            "test_foreign_tenant",
            "component fixture is outside admitted tenant",
        ))
    }
}
fn populate(engine: &mut RedCoreGraphStore) {
    for body in ["saved one", "saved two", "saved three"] {
        engine
            .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                NodeRecord::new(
                    "exchange:record",
                    ["ExchangeTestRecord"],
                    json!({"tenant":"tenant:exchange","body":body}),
                ),
            )]))
            .unwrap();
    }
}
fn memory_source() -> (RedCoreGraphStore, MemoryArchive) {
    let mut engine = RedCoreGraphStore::open_with_chunk_store_and_paged_mutation_log(
        Arc::new(MemChunkStore::new()),
        crate::graph_store::RedCoreOptions::default(),
    )
    .unwrap();
    populate(&mut engine);
    let mut archive = MemoryArchive(vec![]);
    export_logical_exchange(&engine, &identity(), limits(), &mut archive, &mut admit).unwrap();
    (engine, archive)
}
fn validated(archive: MemoryArchive) -> ValidatedExchange {
    validate_logical_exchange(Arc::new(archive), &identity(), limits(), &mut admit).unwrap()
}
#[test]
fn logical_exchange_full_history_native_rr_reopens_without_source_write() {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.RR");
    let target_path = directory.path().join("target.RR");
    let source = RustyRed::create(&source_path, &DeviceId::new("exchange-source-device")).unwrap();
    let mut engine = RedCoreGraphStore::open_with_chunk_store_and_paged_mutation_log(
        source.store().clone(),
        crate::graph_store::RedCoreOptions::default(),
    )
    .unwrap();
    populate(&mut engine);
    let old_bytes = std::fs::read(&source_path).unwrap();
    let expected = engine.get_node("exchange:record").unwrap().unwrap();
    let expected_history = engine
        .pin_graph_history()
        .unwrap()
        .page(None, 32)
        .unwrap()
        .entries;
    assert_eq!(expected_history.len(), 3);
    let mut archive = MemoryArchive(vec![]);
    let exported =
        export_logical_exchange(&engine, &identity(), limits(), &mut archive, &mut admit).unwrap();
    assert_eq!(std::fs::read(&source_path).unwrap(), old_bytes);
    let accepted = validated(archive);
    assert_eq!(accepted.receipt().catalog, exported.catalog);
    let target = RustyRed::create(&target_path, &DeviceId::new("exchange-target-device")).unwrap();
    assert_ne!(
        source.database(),
        target.database(),
        "physical file identity is explicitly new"
    );
    accepted
        .apply_to_empty(target.store().as_ref(), None)
        .unwrap();
    drop(target);
    let reopened = RustyRed::open_read_only(&target_path).unwrap();
    accepted
        .verify_destination(reopened.store().clone())
        .unwrap();
    let restored = RedCoreGraphStore::open_with_chunk_store_and_paged_mutation_log(
        reopened.store().clone(),
        crate::graph_store::RedCoreOptions::default(),
    )
    .unwrap();
    assert_eq!(
        restored.get_node("exchange:record").unwrap().unwrap(),
        expected
    );
    assert_eq!(
        restored
            .pin_graph_history()
            .unwrap()
            .page(None, 32)
            .unwrap()
            .entries,
        expected_history
    );
    assert_eq!(std::fs::read(&source_path).unwrap(), old_bytes);
}
#[test]
fn logical_exchange_unknown_truncated_identity_and_capacity_refuse_before_destination_writes() {
    let (_, archive) = memory_source();
    let destination = MemChunkStore::new();
    let mut unknown = archive.clone();
    unknown.0[7] = b'9';
    assert_eq!(
        validate_logical_exchange(Arc::new(unknown), &identity(), limits(), &mut admit)
            .err()
            .unwrap()
            .code,
        "exchange_version_unsupported"
    );
    for end in [0, 1, 7, 8, 9, 11, 12, archive.0.len() - 1] {
        assert!(validate_logical_exchange(
            Arc::new(MemoryArchive(archive.0[..end].to_vec())),
            &identity(),
            limits(),
            &mut admit
        )
        .is_err());
    }
    let mut other = identity();
    other.principal = "account:other".into();
    assert_eq!(
        validate_logical_exchange(Arc::new(archive.clone()), &other, limits(), &mut admit)
            .err()
            .unwrap()
            .code,
        "exchange_identity_refused"
    );
    let mut bounded = limits();
    bounded.max_archive_bytes = 16;
    bounded.max_header_bytes = 8;
    bounded.max_chunk_bytes = 8;
    assert_eq!(
        validate_logical_exchange(Arc::new(archive), &identity(), bounded, &mut admit)
            .err()
            .unwrap()
            .code,
        "exchange_archive_limit"
    );
    assert_eq!(destination.read_root().unwrap(), None);
    assert_eq!(destination.chunk_count().unwrap(), 0);
}
#[test]
fn logical_exchange_exact_empty_facade_allowed_but_existing_domain_refused() {
    let (_, archive) = memory_source();
    let accepted = validated(archive);
    let empty = Arc::new(MemChunkStore::new());
    let _facade = crate::object_store::DiskObjectStore::from_chunk_store(empty.clone()).unwrap();
    let initial = empty.read_root().unwrap();
    assert!(initial.is_some());
    accepted.apply_to_empty(empty.as_ref(), initial).unwrap();
    let before = empty.read_root().unwrap();
    let count = empty.chunk_count().unwrap();
    assert_eq!(
        accepted
            .apply_to_empty(empty.as_ref(), before)
            .err()
            .unwrap()
            .code,
        "exchange_destination_not_empty"
    );
    assert_eq!(empty.read_root().unwrap(), before);
    assert_eq!(empty.chunk_count().unwrap(), count);
}
#[test]
fn logical_exchange_foreign_historical_payload_refuses_even_when_current_head_is_admitted() {
    let mut source = RedCoreGraphStore::open_with_chunk_store_and_paged_mutation_log(
        Arc::new(MemChunkStore::new()),
        crate::graph_store::RedCoreOptions::default(),
    )
    .unwrap();
    source
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new(
                "exchange:record",
                ["ExchangeTestRecord"],
                json!({"tenant":"tenant:foreign"}),
            ),
        )]))
        .unwrap();
    populate(&mut source);
    let mut archive = MemoryArchive(vec![]);
    assert_eq!(
        export_logical_exchange(&source, &identity(), limits(), &mut archive, &mut admit)
            .err()
            .unwrap()
            .code,
        "test_foreign_tenant"
    );
    assert!(
        archive.0.is_empty(),
        "preflight finishes before temporary export frames are exposed"
    );
}
#[test]
fn logical_exchange_forged_head_descriptor_refuses() {
    let (_, archive) = memory_source();
    let header_len = u32::from_le_bytes(archive.0[8..12].try_into().unwrap()) as usize;
    let mut header: Header = serde_json::from_slice(&archive.0[12..12 + header_len]).unwrap();
    header.pack.commit.message = "forged metadata under a different valid commit".into();
    header.pack.commit.commit_hash = recompute_commit_hash(&header.pack.commit).unwrap();
    header.pack.manifest.commit_hash = header.pack.commit.commit_hash;
    header.pack = crate::versioned_graph::retarget_graph_pack_version(
        header.pack.clone(),
        header.pack.commit.graph_version,
    )
    .unwrap();
    let rewritten = serde_json::to_vec(&header).unwrap();
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(rewritten.len() as u32).to_le_bytes());
    bytes.extend(rewritten);
    bytes.extend_from_slice(&archive.0[12 + header_len..]);
    assert_eq!(
        validate_logical_exchange(
            Arc::new(MemoryArchive(bytes)),
            &identity(),
            limits(),
            &mut admit
        )
        .err()
        .unwrap()
        .code,
        "exchange_head_binding_mismatch"
    );
}

struct ConcurrentPublicationBarrier {
    inner: Arc<MemChunkStore>,
    arrived: Arc<std::sync::Barrier>,
    published: Arc<std::sync::Barrier>,
}
impl ChunkStore for ConcurrentPublicationBarrier {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.inner.get(address)
    }
    fn get_bounded(&self, address: Hash, maximum: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.inner.get_bounded(address, maximum)
    }
    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.inner.has(address)
    }
    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.inner.missing(addresses)
    }
    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.inner.write(chunks)
    }
    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.inner.commit(root, meta)
    }
    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.inner.read_root()
    }
}
impl ChunkStoreCatalog for ConcurrentPublicationBarrier {
    fn commit_catalog_updates(
        &self,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        self.inner.commit_catalog_updates(updates, meta)
    }
    fn supports_catalog_compare_and_swap(&self) -> bool {
        true
    }
    fn commit_catalog_updates_if_root(
        &self,
        expected: Option<Hash>,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        self.arrived.wait();
        self.published.wait();
        self.inner
            .commit_catalog_updates_if_root(expected, updates, meta)
    }
}
#[test]
fn logical_exchange_concurrent_accepted_publication_between_check_and_commit_is_preserved() {
    let (_, archive) = memory_source();
    let accepted = validated(archive);
    let target = Arc::new(MemChunkStore::new());
    let arrived = Arc::new(std::sync::Barrier::new(2));
    let published = Arc::new(std::sync::Barrier::new(2));
    let fenced = ConcurrentPublicationBarrier {
        inner: target.clone(),
        arrived: arrived.clone(),
        published: published.clone(),
    };
    let importer = std::thread::spawn(move || accepted.apply_to_empty(&fenced, None));
    arrived.wait();
    let mut other = RedCoreGraphStore::open_with_chunk_store_and_paged_mutation_log(
        target.clone(),
        crate::graph_store::RedCoreOptions::default(),
    )
    .unwrap();
    other
        .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
            NodeRecord::new(
                "concurrent:accepted",
                ["ExchangeTestRecord"],
                json!({"tenant":"tenant:exchange","body":"do not overwrite this accepted write"}),
            ),
        )]))
        .unwrap();
    let concurrent_root = target.read_root().unwrap();
    published.wait();
    assert_eq!(
        importer.join().unwrap().err().unwrap().code,
        "chunk_store_catalog_precondition_failed"
    );
    assert_eq!(target.read_root().unwrap(), concurrent_root);
    assert!(other.get_node("concurrent:accepted").unwrap().is_some());
}

#[derive(Clone)]
struct MutableStagedArchive(Arc<Mutex<Vec<u8>>>);
impl ExchangeReader for MutableStagedArchive {
    fn len(&self) -> GraphStoreResult<u64> {
        Ok(self.0.lock().unwrap().len() as u64)
    }
    fn read_exact(&self, offset: u64, length: usize) -> GraphStoreResult<Vec<u8>> {
        let bytes = self.0.lock().unwrap();
        let offset = usize::try_from(offset).unwrap();
        bytes
            .get(offset..offset + length)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| refusal("test_short_read", "short mutable fixture read"))
    }
}
#[test]
fn logical_exchange_changed_staged_chunk_is_rehashed_before_publication() {
    let (_, archive) = memory_source();
    let mutable = MutableStagedArchive(Arc::new(Mutex::new(archive.0)));
    let accepted =
        validate_logical_exchange(Arc::new(mutable.clone()), &identity(), limits(), &mut admit)
            .unwrap();
    let mut bytes = mutable.0.lock().unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    drop(bytes);
    let target = MemChunkStore::new();
    assert_eq!(
        accepted.apply_to_empty(&target, None).err().unwrap().code,
        "exchange_chunk_hash"
    );
    assert_eq!(
        target.read_root().unwrap(),
        None,
        "no imported head was acknowledged"
    );
}
#[test]
fn logical_exchange_native_grouped_catalog_compare_and_swap_has_one_winner() {
    use crate::journal::{JournalChunkStore, JournalCommitPolicy};
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(
        JournalChunkStore::open_with_commit_policy(
            directory.path().join("grouped.journal"),
            JournalCommitPolicy::Grouped {
                max_wait: std::time::Duration::from_millis(10),
            },
        )
        .unwrap(),
    );
    let leaf = crate::versioned_graph::canonical_leaf_preimage(&[]).unwrap();
    let graph = Hash::of(&leaf);
    store.write(&[(graph, &leaf)]).unwrap();
    let start = Arc::new(std::sync::Barrier::new(3));
    let mut threads = vec![];
    for _ in 0..2 {
        let store = store.clone();
        let start = start.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            store.commit_catalog_updates_if_root(
                None,
                &BTreeMap::from([("graph".into(), Some(graph))]),
                meta(),
            )
        }));
    }
    start.wait();
    let results = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|result|matches!(result,Err(error) if error.code=="chunk_store_catalog_precondition_failed")).count(),1);
}
#[test]
fn logical_exchange_projection_capacity_matches_actual_v1_v2_dedup_and_reserved_bounds() {
    use crate::browser_projection::{ProjectionChunkStore, ProjectionStorageBounds};
    use theorem_oplog::NativeLogFile;
    let (_, archive) = memory_source();
    let accepted = validated(archive);
    let directory = tempfile::tempdir().unwrap();
    let mut physical = vec![];
    for compressed in [false, true] {
        let file = NativeLogFile::create_or_open_named(
            directory.path(),
            if compressed { "v2" } else { "v1" },
        )
        .unwrap();
        let bounds = ProjectionStorageBounds {
            max_log_bytes: 8 * 1024 * 1024,
            commit_headroom_bytes: 65536,
            recovery_headroom_bytes: 65536,
        };
        let store = if compressed {
            ProjectionChunkStore::open_with_compressed_chunks(Box::new(file), Some(bounds))
        } else {
            ProjectionChunkStore::open_with_storage_bounds(Box::new(file), Some(bounds))
        }
        .unwrap();
        // A real pre-existing orphan chunk is deduplicated by both the exact
        // preflight and the production writer; it remains uncommitted here.
        let first = *accepted.addresses.first().unwrap();
        let bytes = accepted.store.get(first).unwrap().unwrap();
        store.write(&[(first, &bytes)]).unwrap();
        let before = store.storage_snapshot().unwrap();
        let admission = store.preflight_logical_exchange(&accepted, None).unwrap();
        assert_eq!(
            store.storage_snapshot().unwrap(),
            before,
            "admission does not mutate pressure or file bytes"
        );
        accepted.apply_to_empty(&store, None).unwrap();
        let after = store.storage_snapshot().unwrap();
        assert_eq!(after.log_bytes, admission.projected_log_bytes);
        assert_eq!(
            after.log_bytes - before.log_bytes,
            admission.staged_chunk_bytes + admission.publication_bytes
        );
        physical.push(after.log_bytes);
        drop(store);
        let file = NativeLogFile::create_or_open_named(
            directory.path(),
            if compressed { "v2" } else { "v1" },
        )
        .unwrap();
        let reopened = Arc::new(
            ProjectionChunkStore::open_with_storage_bounds(Box::new(file), Some(bounds)).unwrap(),
        );
        accepted.verify_destination(reopened).unwrap();
    }
    assert!(
        physical[1] < physical[0],
        "actual archive is compressible; estimates follow each physical source version"
    );
    let file = NativeLogFile::create_or_open_named(directory.path(), "capacity-refusal").unwrap();
    let bounds = ProjectionStorageBounds {
        max_log_bytes: 4096,
        commit_headroom_bytes: 1024,
        recovery_headroom_bytes: 512,
    };
    let target =
        ProjectionChunkStore::open_with_storage_bounds(Box::new(file), Some(bounds)).unwrap();
    let before = target.storage_snapshot().unwrap();
    assert_eq!(
        target
            .preflight_logical_exchange(&accepted, None)
            .err()
            .unwrap()
            .code,
        "local_storage_pressure_refused"
    );
    assert_eq!(
        target.storage_snapshot().unwrap(),
        before,
        "complete capacity refusal occurs before canonical staging"
    );
}

#[test]
fn logical_exchange_admission_revoked_after_staging_refuses_before_catalog_publication() {
    let (_, archive) = memory_source();
    let accepted = validated(archive);
    let target = MemChunkStore::new();
    let mut checks = 0;
    let result = accepted.apply_to_empty_with_admission(&target, None, &mut || {
        checks += 1;
        if checks == 2 {
            Err(refusal(
                "test_admission_generation_changed",
                "component host admission changed",
            ))
        } else {
            Ok(())
        }
    });
    assert_eq!(
        result.err().unwrap().code,
        "test_admission_generation_changed"
    );
    assert_eq!(checks, 2);
    assert_eq!(target.read_root().unwrap(), None);
}

#[test]
fn logical_exchange_metadata_capabilities_refuse_unknown_kind_and_forged_hash() {
    let (_, archive) = memory_source();
    let header_len = u32::from_le_bytes(archive.0[8..12].try_into().unwrap()) as usize;
    for unknown in [false, true] {
        let mut header: Header = serde_json::from_slice(&archive.0[12..12 + header_len]).unwrap();
        assert_eq!(header.pack.capabilities.len(), 2);
        if unknown {
            header.pack.capabilities[0].kind = "unknown-required-schema/9".into();
        } else {
            header.pack.capabilities[0].content_hash = "forged-content-hash".into();
        }
        let rewritten = serde_json::to_vec(&header).unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(rewritten.len() as u32).to_le_bytes());
        bytes.extend(rewritten);
        bytes.extend_from_slice(&archive.0[12 + header_len..]);
        let error = validate_logical_exchange(
            Arc::new(MemoryArchive(bytes)),
            &identity(),
            limits(),
            &mut admit,
        )
        .err()
        .unwrap();
        assert_eq!(
            error.code,
            if unknown {
                "exchange_capability_kind_unsupported"
            } else {
                "exchange_capability_invalid"
            }
        );
    }
}

/// The actual native caller already owns a committed chunk store. Any
/// accidental unbounded read in compact-head recovery must fail this oracle.
struct BoundedOnlySource(Arc<dyn UnifiedChunkStore>);
impl ChunkStore for BoundedOnlySource {
    fn get(&self, _: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        Err(refusal(
            "test_unbounded_source_read",
            "native export must preflight physical chunk size",
        ))
    }
    fn get_bounded(&self, address: Hash, maximum: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.0.get_bounded(address, maximum)
    }
    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.0.has(address)
    }
    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.0.missing(addresses)
    }
    fn write(&self, _: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        Err(refusal("test_source_write", "read-only source"))
    }
    fn commit(&self, _: Hash, _: CommitMeta) -> GraphStoreResult<Hash> {
        Err(refusal("test_source_write", "read-only source"))
    }
    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.0.read_root()
    }
}
impl ChunkStoreCatalog for BoundedOnlySource {
    fn commit_catalog_updates(
        &self,
        _: &BTreeMap<String, Option<Hash>>,
        _: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        Err(refusal("test_source_write", "read-only source"))
    }
}
#[test]
fn logical_exchange_native_chunk_source_never_requires_unbounded_facade_recovery() {
    let (engine, _) = memory_source();
    let root = engine.unified_chunk_store().read_root().unwrap();
    let source: Arc<dyn UnifiedChunkStore> =
        Arc::new(BoundedOnlySource(engine.unified_chunk_store()));
    drop(engine);
    let mut archive = MemoryArchive(vec![]);
    let receipt = export_logical_exchange_from_store(
        source.clone(),
        &identity(),
        limits(),
        &mut archive,
        &mut admit,
    )
    .unwrap();
    assert_eq!(Some(receipt.catalog), root);
    let accepted = validated(archive);
    assert_eq!(accepted.receipt().catalog, receipt.catalog);
    assert_eq!(source.read_root().unwrap(), root);
    let mut small = limits();
    small.max_chunk_bytes = 1;
    let mut refused_output = MemoryArchive(vec![]);
    assert!(export_logical_exchange_from_store(
        source,
        &identity(),
        small,
        &mut refused_output,
        &mut admit
    )
    .is_err());
    assert!(
        refused_output.0.is_empty(),
        "physical read refusal precedes archive publication"
    );
}
