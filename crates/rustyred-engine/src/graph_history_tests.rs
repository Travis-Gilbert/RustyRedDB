use super::*;
use crate::graph_history::{GraphCommitMetadata, GraphHistoryCoverage};
use crate::reachability::{
    compact_committed_reachable, compact_committed_reachable_with_limits, CoreReachabilityCodec,
    ReachabilityLimits,
};
use serde_json::json;

fn history_metadata(n: u64) -> GraphCommitMetadata {
    GraphCommitMetadata {
        author: format!("verified-user:{n}"),
        message: format!("authored edit {n}"),
        timestamp_unix_ms: 1_700_000_000_000 + u128::from(n),
    }
}
fn history_node(text: &str) -> NodeRecord {
    NodeRecord::new("article:a", ["Memory"], json!({"body":text}))
}
fn history_write(store: &mut RedCoreGraphStore, n: u64) -> GraphStoreResult<GraphCommitReceipt> {
    store.commit_batch_with_metadata(
        GraphMutationBatch::new([GraphMutation::NodeUpsert(history_node(&format!(
            "body {n}"
        )))]),
        history_metadata(n),
    )
}
fn history_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    }
}

#[test]
fn graph_history_full_header_survives_reopen() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let (first, second) = {
        let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
        (history_write(&mut store, 1)?, history_write(&mut store, 2)?)
    };
    let store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let pinned = store.pin_graph_history()?;
    assert_eq!(pinned.coverage(), GraphHistoryCoverage::Complete);
    let one = pinned.entry(first.commit.commit_hash)?;
    let two = pinned.entry(second.commit.commit_hash)?;
    assert_eq!(one.commit, first.commit);
    assert_eq!(two.commit, second.commit);
    assert_eq!(two.commit.parent_commits, vec![first.commit.commit_hash]);
    assert_eq!(two.roots.transaction, second.transaction.txn_id);
    assert_eq!(
        pinned
            .revision(first.commit.commit_hash)?
            .get_node("article:a")?
            .unwrap()
            .properties["body"],
        "body 1"
    );
    assert_eq!(
        pinned
            .head_revision()?
            .get_node("article:a")?
            .unwrap()
            .properties["body"],
        "body 2"
    );
    assert_eq!(store.current_pack.as_ref().unwrap().commit, second.commit);
    Ok(())
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(16))]

    #[test]
    fn graph_history_multiple_designations_survive_reopen(
        indexes in proptest::collection::btree_map(
            ("[a-z]{1,4}", "[a-z]{1,4}"), 1usize..5, 2..10,
        ),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let expected_vectors: Vec<_> = indexes.iter().map(|((label, property), &dimension)| {
            VectorDesignation { label: label.clone(), property: property.clone(), dimension }
        }).collect();
        let expected_ordered: Vec<_> = indexes.keys().map(|(label, property)| {
            OrderedDesignation { label: label.clone(), property: property.clone() }
        }).collect();
        let commit = {
            let mut store = RedCoreGraphStore::open(dir.path(), history_options()).unwrap();
            for ((label, property), &dimension) in indexes.iter().rev() {
                store.designate_vector_property(label, property, dimension).unwrap();
                store.designate_ordered_property(label, property).unwrap();
            }
            history_write(&mut store, 1).unwrap().commit.commit_hash
        };
        let mut reopened = RedCoreGraphStore::open(dir.path(), history_options()).unwrap();
        let pinned = reopened.pin_graph_history().unwrap();
        let entry = pinned.entry(commit).unwrap();
        proptest::prop_assert_eq!(&entry.roots.vector_designations, &expected_vectors);
        proptest::prop_assert_eq!(&entry.roots.ordered_designations, &expected_ordered);
        // A second write reconstructs roots from the reloaded index maps.
        history_write(&mut reopened, 2).unwrap();
        proptest::prop_assert_eq!(pinned.entry(commit).unwrap(), entry);
    }
}

#[test]
fn graph_history_failed_publication_has_no_phantom_header() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let first;
    {
        let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
        first = history_write(&mut store, 1)?;
        let before = store.chunk_store_handle().unwrap().read_root()?;
        store.inject_journal_append_failure_for_test(0, 0, false)?;
        assert!(history_write(&mut store, 2).is_err());
        assert_eq!(store.chunk_store_handle().unwrap().read_root()?, before);
        assert_eq!(store.pin_graph_history()?.page(None, 10)?.entries.len(), 1);
        assert_eq!(
            store.get_node("article:a")?.unwrap().properties["body"],
            "body 1"
        );
    }
    let mut reopened = RedCoreGraphStore::open(dir.path(), history_options())?;
    assert_eq!(
        reopened.pin_graph_history()?.head_revision()?.commit(),
        Some(&first.commit)
    );
    let third = history_write(&mut reopened, 3)?;
    assert_eq!(third.commit.author, "verified-user:3");
    assert_eq!(third.commit.message, "authored edit 3");
    assert_eq!(
        reopened.pin_graph_history()?.page(None, 10)?.entries.len(),
        2
    );
    Ok(())
}

#[test]
fn graph_history_pinned_indexes_and_payloads_survive_270_commits_and_gc() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
        let first = history_write(&mut store, 1)?;
        history_write(&mut store, 2)?;
        let pin = store.pin_graph_history()?;
        let initial_page = pin.page(None, 1)?;
        assert!(initial_page.next.is_some());
        for n in 3..=270 {
            history_write(&mut store, n)?;
        }
        let journal = store.chunk_store_handle().unwrap();
        let report = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(report.reachability.max_depth < 256);
        assert_eq!(pin.page(None, 1)?, initial_page);
        assert_eq!(
            pin.page(initial_page.next, 1)?.entries[0].commit,
            first.commit
        );
        assert_eq!(
            pin.revision(first.commit.commit_hash)?
                .get_node("article:a")?
                .unwrap()
                .properties["body"],
            "body 1"
        );
        assert_eq!(
            store.pin_graph_history()?.page(None, 512)?.entries.len(),
            270
        );
    }
    let store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let pin = store.pin_graph_history()?;
    assert_eq!(pin.page(None, 512)?.entries.len(), 270);
    assert_eq!(
        pin.head_revision()?
            .get_node("article:a")?
            .unwrap()
            .properties["body"],
        "body 270"
    );
    Ok(())
}

#[test]
fn graph_history_gc_budget_refuses_before_deleting() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let first = history_write(&mut store, 1)?;
    history_write(&mut store, 2)?;
    let journal = store.chunk_store_handle().unwrap();
    let bytes_before = std::fs::metadata(journal.path()).unwrap().len();
    let root_before = journal.read_root()?;
    assert!(compact_committed_reachable_with_limits(
        journal.as_ref(),
        &CoreReachabilityCodec,
        ReachabilityLimits {
            max_visits: 1,
            max_depth: 256
        }
    )
    .is_err());
    assert_eq!(
        std::fs::metadata(journal.path()).unwrap().len(),
        bytes_before
    );
    assert_eq!(journal.read_root()?, root_before);
    assert_eq!(
        store
            .pin_graph_history()?
            .entry(first.commit.commit_hash)?
            .commit,
        first.commit
    );
    Ok(())
}

#[test]
fn graph_history_v4_boundary_never_invents_a_header() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let first;
    {
        let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
        first = history_write(&mut store, 1)?;
        let journal = store.chunk_store_handle().unwrap();
        let catalog = crate::chunk_store::RootCatalog::from_bytes(
            &journal.get(journal.read_root()?.unwrap())?.unwrap(),
        )?;
        let bytes = journal
            .get(catalog.get(ROOT_NAME_REDCORE).unwrap())?
            .unwrap();
        let mut state = RedCoreJournalState::from_bytes(&bytes)?;
        state.history_root = None;
        let mut legacy = state.to_bytes()?;
        // Exact preserved v4 layout: v5 inserts its optional root immediately
        // before the unchanged mutation-log tail. This fixture restores v4.
        let offset = legacy.len() - state.mutation_log.encode_tail()?.len() - 1;
        assert_eq!(legacy.remove(offset), 0);
        legacy[1] = MUTATION_LOG_REDCORE_JOURNAL_STATE_VERSION;
        assert_eq!(RedCoreJournalState::from_bytes(&legacy)?.history_root, None);
        let address = Hash::of(&legacy);
        journal.write(&[(address, &legacy)])?;
        journal.commit_catalog_updates(
            &BTreeMap::from([(ROOT_NAME_REDCORE.into(), Some(address))]),
            CommitMeta {
                parents: vec![],
                author: "v4 fixture writer".into(),
                message: "publish preserved v4 state layout".into(),
                timestamp_unix_ms: 1,
                graph_version: state.graph_version,
            },
        )?;
    }
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let legacy = store.pin_graph_history()?;
    assert_eq!(legacy.coverage(), GraphHistoryCoverage::Unavailable);
    assert!(legacy.head_revision()?.commit().is_none());
    assert_eq!(legacy.head_roots().logical_commit, first.commit.commit_hash);
    assert!(legacy.page(None, 10)?.entries.is_empty());
    assert!(legacy.entries_after(0, 10).is_err());
    history_write(&mut store, 2)?;
    let pin = store.pin_graph_history()?;
    assert_eq!(
        pin.coverage(),
        GraphHistoryCoverage::Since {
            first_transaction: 2,
            prior_history_unavailable: true
        }
    );
    assert_eq!(pin.page(None, 10)?.entries.len(), 1);
    assert!(pin.entries_after(0, 10).is_err());
    assert_eq!(pin.entries_after(first.transaction.txn_id, 10)?.len(), 1);
    let prior = pin.legacy_revision()?.unwrap();
    assert!(prior.commit().is_none());
    assert_eq!(
        prior.get_node("article:a")?.unwrap().properties["body"],
        "body 1"
    );
    let journal = store.chunk_store_handle().unwrap();
    compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
    assert_eq!(
        pin.legacy_revision()?
            .unwrap()
            .get_node("article:a")?
            .unwrap()
            .properties["body"],
        "body 1"
    );
    Ok(())
}

#[test]
fn graph_history_pinned_point_reads_match_authoritative_hydration() -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    let first = history_write(&mut store, 1)?;
    store.upsert_node(NodeRecord::new(
        "article:b",
        ["Memory"],
        json!({"body":"other"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "article:a",
        "relates_to",
        "article:b",
        json!({}),
    ))?;
    let pinned = store.pin_graph_history()?;
    let revision = pinned.head_revision()?;
    let snapshot = hydrate_graph_snapshot_from_payload_index(
        store.unified_chunk_store().as_chunk_store(),
        &revision.roots.logical_tree,
        &revision.roots.payload_tree,
        revision.roots.graph_version,
    )?;
    for node in snapshot.nodes {
        assert_eq!(revision.get_node(&node.id)?, Some(node));
    }
    for edge in snapshot.edges {
        assert_eq!(revision.get_edge(&edge.id)?, Some(edge));
    }
    assert!(revision.get_node("missing")?.is_none());
    assert_eq!(
        revision.incident_edges("article:a", Direction::Out)?.len(),
        1
    );
    assert_eq!(
        pinned
            .revision(first.commit.commit_hash)?
            .get_node("article:a")?
            .unwrap()
            .properties["body"],
        "body 1"
    );
    assert_eq!(
        revision.timestamp_unix_ms(),
        Some(revision.commit().unwrap().timestamp_unix_ms)
    );
    Ok(())
}

#[test]
fn graph_history_changed_keys_track_final_payload_and_rejected_roots() -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    history_write(&mut store, 1)?;
    let before = store.pin_graph_history()?;
    let node = store.get_node("article:a")?.unwrap().clone();
    let result = store.commit_batch_with_metadata(
        GraphMutationBatch::new([
            GraphMutation::NodeUpsert(node.clone()),
            GraphMutation::NodeUpsert(node),
        ]),
        history_metadata(2),
    )?;
    // Each accepted upsert also writes its genuine control event. The header
    // must cover those records while deduplicating the repeated article key.
    assert_eq!(
        store
            .pin_graph_history()?
            .entry(result.commit.commit_hash)?
            .changed_object_keys,
        vec![
            "node/article:a",
            "node/controlevent:2:0:Memory.updated",
            "node/controlevent:2:1:Memory.updated",
        ]
    );
    let stale = GraphMutationBatch::new([GraphMutation::NodeUpsert(history_node("must not land"))])
        .with_expected_root(before.head_roots().logical_tree.root);
    // Metadata-only writes can preserve the logical root; use an unrelated
    // invalid fixture commitment to exercise the existing expected-root guard.
    let stale = stale.with_expected_root(Hash::of(b"invalid stale root fixture"));
    let head = store.pin_graph_history()?.head_roots().clone();
    assert!(store
        .commit_batch_with_metadata(stale, history_metadata(3))
        .is_err());
    assert_eq!(store.pin_graph_history()?.head_roots(), &head);
    assert_eq!(store.pin_graph_history()?.page(None, 10)?.entries.len(), 2);
    Ok(())
}

#[test]
fn graph_history_removed_edge_is_latest_endpoint_revision() -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    history_write(&mut store, 1)?;
    store.upsert_node(NodeRecord::new(
        "article:b",
        ["Memory"],
        json!({"body":"other"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "article:a",
        "relates_to",
        "article:b",
        json!({}),
    ))?;
    let edge_head = store.pin_graph_history()?.head_roots().logical_commit;
    let deleted = store.commit_batch_with_metadata(
        GraphMutationBatch::new([GraphMutation::EdgeDelete("edge:ab".into())]),
        history_metadata(9),
    )?;
    let pin = store.pin_graph_history()?;
    let keys = BTreeSet::from(["node/article:a".into()]);
    assert_eq!(pin.newest_touching(&keys)?.unwrap().commit, deleted.commit);
    let earlier = pin.view_at(edge_head)?;
    assert_eq!(
        earlier.newest_touching(&keys)?.unwrap().commit.commit_hash,
        edge_head
    );
    assert!(earlier.entry(deleted.commit.commit_hash).is_err());
    assert!(earlier
        .page(None, 512)?
        .entries
        .iter()
        .all(|entry| entry.roots.transaction <= earlier.head_roots().transaction));
    Ok(())
}

#[test]
fn graph_history_article_clock_and_metadata_are_explicit() -> GraphStoreResult<()> {
    use crate::article_fold::{ArticleNode, ArticleRead};
    use crate::graph_history::article::*;
    use crate::graph_history::{GraphHistoryEntry, PinnedGraphHistory, PinnedGraphRevision};
    let mut store = RedCoreGraphStore::memory();
    let node = NodeRecord::new(
        "article:a",
        ["Memory"],
        json!({"body":"historical", "_ttl_expires_at_ms":1_700_000_000_500i64}),
    );
    let receipt = store.commit_batch_with_metadata(
        GraphMutationBatch::new([GraphMutation::NodeUpsert(node)]),
        history_metadata(1),
    )?;
    let history = store.pin_graph_history()?;
    let admitted = |_: &NodeRecord| true;
    let denied = |_: &NodeRecord| false;
    let pins = |_: &PinnedGraphRevision, _: &str| Ok(Vec::new());
    let restricted =
        |_: &PinnedGraphHistory, _: &GraphHistoryEntry| Ok(CommitMetadataVisibility::Redacted);
    for observed_at_ms in [1_700_000_000_600i64, 1_900_000_000_000i64] {
        let context = RedCoreArticleReadContext {
            tenant: "tenant-a",
            admission: &admitted,
            pins: &pins,
            metadata_visibility: &restricted,
            seq: 7,
            stream_epoch: "actual-test-epoch",
            observed_at_ms,
        };
        let read = RedCoreArticleRead::new(
            &history,
            ArticleRevisionTarget::Commit(receipt.commit.commit_hash),
            context,
        )
        .unwrap();
        assert!(matches!(
            read.node("article:a").unwrap(),
            ArticleNode::Present(_)
        ));
        let revision = read
            .revision(&BTreeSet::from(["node/article:a".into()]))
            .unwrap();
        let projected = revision.commit.unwrap();
        assert!(projected.metadata_redacted);
        assert!(projected.author.is_empty() && projected.summary.is_empty());
        assert_eq!(projected.id, receipt.commit.commit_hash.to_string());
        assert_eq!(projected.transaction, Some(receipt.transaction.txn_id));
        assert_eq!(projected.log_position, None);
        assert_eq!(
            projected.timestamp_unix_ms,
            receipt.commit.timestamp_unix_ms.to_string()
        );
    }
    let context = RedCoreArticleReadContext {
        tenant: "tenant-a",
        admission: &denied,
        pins: &pins,
        metadata_visibility: &restricted,
        seq: 7,
        stream_epoch: "actual-test-epoch",
        observed_at_ms: 1_900_000_000_000i64,
    };
    let revoked = RedCoreArticleRead::new(
        &history,
        ArticleRevisionTarget::Commit(receipt.commit.commit_hash),
        context,
    )
    .unwrap();
    assert!(matches!(
        revoked.node("article:a").unwrap(),
        ArticleNode::Excluded
    ));
    let context = RedCoreArticleReadContext {
        tenant: "tenant-a",
        admission: &admitted,
        pins: &pins,
        metadata_visibility: &restricted,
        seq: 7,
        stream_epoch: "actual-test-epoch",
        observed_at_ms: 1_900_000_000_000i64,
    };
    let live = RedCoreArticleRead::new(&history, ArticleRevisionTarget::Head, context).unwrap();
    assert!(matches!(
        live.node("article:a").unwrap(),
        ArticleNode::Missing
    ));
    assert_eq!(
        history.entry(receipt.commit.commit_hash)?.commit,
        receipt.commit
    );
    Ok(())
}

#[test]
fn graph_history_gc_uses_the_revisions_removed_vector_designation() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let first_hash;
    let payload_address;
    {
        let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
        store.designate_vector_property("HistoryVector", "historical_embedding", 2)?;
        let first = store.commit_batch_with_metadata(
            GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "article:vector",
                ["HistoryVector"],
                json!({"historical_embedding":[1.0,2.0]}),
            ))]),
            history_metadata(1),
        )?;
        first_hash = first.commit.commit_hash;
        let prior = store.pin_graph_history()?.revision(first_hash)?;
        let node = prior.get_node("article:vector")?.unwrap();
        let reference = crate::vector_payload::VectorPropertyRef::from_value(
            &node.properties["historical_embedding"],
        )?;
        payload_address = Hash::parse_prefixed(&reference.content_id)?;
        store.clear_vector_property("HistoryVector", "historical_embedding")?;
        assert_eq!(
            store.pin_graph_history()?.page(None, 10)?.entries.len(),
            1,
            "designation-only mutation must not invent a graph history row"
        );
        store.commit_batch_with_metadata(
            GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "article:vector",
                ["HistoryVector"],
                json!({"body":"without vector"}),
            ))]),
            history_metadata(2),
        )?;
        assert!(store.vector_designations().is_empty());
        let journal = store.chunk_store_handle().unwrap();
        let original = journal.get(payload_address)?.unwrap();
        let report = compact_committed_reachable(journal.as_ref(), &CoreReachabilityCodec)?;
        assert!(report.reachability.reachable.contains(&payload_address));
        assert_eq!(journal.get(payload_address)?.unwrap(), original);
        assert!(prior
            .get_node("article:vector")?
            .unwrap()
            .properties
            .get("historical_embedding")
            .is_some());
    }
    let store = RedCoreGraphStore::open(dir.path(), history_options())?;
    assert!(store.vector_designations().is_empty());
    assert!(store
        .chunk_store_handle()
        .unwrap()
        .get(payload_address)?
        .is_some());
    assert!(store
        .pin_graph_history()?
        .revision(first_hash)?
        .get_node("article:vector")?
        .unwrap()
        .properties
        .get("historical_embedding")
        .is_some());
    Ok(())
}

#[test]
fn graph_history_checked_attribution_checks_synthesized_replacements_before_publication(
) -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    history_write(&mut store, 1)?;
    let condition = ConditionalNodePropertyUpdate {
        node_id: "article:a".into(),
        property: "body".into(),
        expected: json!("body 1"),
        new_value: json!("guarded body"),
    };
    let head = store.pin_graph_history()?.head_roots().clone();
    let refused = store.commit_conditional_batch_with_metadata_checked(
        vec![condition.clone()],
        GraphMutationBatch::default(),
        history_metadata(2),
        |_, prepared| {
            assert_eq!(prepared.mutations.len(), 1);
            let GraphMutation::NodeUpsert(node) = &prepared.mutations[0] else {
                panic!("condition must synthesize the replacement");
            };
            assert_eq!(node.properties["body"], "guarded body");
            Err(GraphStoreError::new(
                "quota_fixture_refusal",
                "reject complete prepared batch",
            ))
        },
    );
    assert_eq!(refused.unwrap_err().code, "quota_fixture_refusal");
    assert_eq!(store.pin_graph_history()?.head_roots(), &head);
    let (receipt, prepared) = store
        .commit_conditional_batch_with_metadata_checked(
            vec![condition.clone()],
            GraphMutationBatch::default(),
            history_metadata(3),
            |_, _| Ok(()),
        )?
        .unwrap();
    assert_eq!(prepared.mutations.len(), 1);
    assert_eq!(receipt.commit.author, "verified-user:3");
    assert_eq!(
        store.get_node("article:a")?.unwrap().properties["body"],
        "guarded body"
    );
    assert!(store
        .commit_conditional_batch_with_metadata_checked(
            vec![condition],
            GraphMutationBatch::default(),
            history_metadata(4),
            |_, _| panic!("rejected CAS must not reach quota or publication")
        )?
        .is_none());
    assert_eq!(store.pin_graph_history()?.page(None, 10)?.entries.len(), 2);
    Ok(())
}

#[test]
fn graph_history_forged_header_is_an_integrity_error() -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    let receipt = history_write(&mut store, 1)?;
    let mut entry = store
        .pin_graph_history()?
        .entry(receipt.commit.commit_hash)?;
    entry.commit.author = "forged fixture actor".into();
    assert!(entry.to_bytes().is_err());
    let mut bytes = vec![17u8, 1u8];
    bytes.extend(serde_json::to_vec(&entry).unwrap());
    assert!(crate::graph_history::GraphHistoryEntry::from_bytes(Hash::of(&bytes), &bytes).is_err());
    Ok(())
}

#[test]
fn graph_history_deleted_node_covers_cascaded_edges_and_surviving_endpoint() -> GraphStoreResult<()>
{
    let mut store = RedCoreGraphStore::memory();
    history_write(&mut store, 1)?;
    store.upsert_node(NodeRecord::new(
        "article:b",
        ["Memory"],
        json!({"body":"other"}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "article:a",
        "relates_to",
        "article:b",
        json!({}),
    ))?;
    let old = store.pin_graph_history()?.head_revision()?;
    let removed = store.commit_batch_with_metadata(
        GraphMutationBatch::new([GraphMutation::NodeDelete("article:a".into())]),
        history_metadata(8),
    )?;
    let history = store.pin_graph_history()?;
    let entry = history.entry(removed.commit.commit_hash)?;
    assert!(entry.changed_object_keys.contains(&"node/article:a".into()));
    assert!(entry.changed_object_keys.contains(&"edge/edge:ab".into()));
    assert!(old.get_edge("edge:ab")?.is_some_and(|edge| !edge.tombstone));
    assert!(history
        .head_revision()?
        .get_edge("edge:ab")?
        .is_none_or(|edge| edge.tombstone));
    let survivor = BTreeSet::from(["node/article:b".into()]);
    assert_eq!(
        history.newest_touching(&survivor)?.unwrap().commit,
        removed.commit
    );
    Ok(())
}

#[test]
fn graph_commit_observer_covers_generic_attributed_rewired_and_cascade_writes_without_dispatcher(
) -> GraphStoreResult<()> {
    use crate::hooks::GraphCommitObserver;
    let mut store = RedCoreGraphStore::memory();
    let notices = Arc::new(std::sync::Mutex::new(Vec::new()));
    let received = Arc::clone(&notices);
    store.attach_commit_observer(GraphCommitObserver::new(move |event| {
        received.lock().unwrap().push(event.clone());
    }));
    assert!(!store.has_hook_emitter());
    store.set_hook_emit_depth(100);
    store.upsert_node(history_node("generic writer"))?;
    assert_eq!(notices.lock().unwrap().len(), 1);
    assert_eq!(notices.lock().unwrap()[0].commit.author, "rustyred");
    let explicit = history_write(&mut store, 2)?;
    assert_eq!(
        notices.lock().unwrap().last().unwrap().commit,
        explicit.commit
    );
    for id in ["article:b", "article:c"] {
        store.upsert_node(NodeRecord::new(id, ["Memory"], json!({})))?;
    }
    store.upsert_edge(EdgeRecord::new(
        "edge:move",
        "article:a",
        "relates_to",
        "article:b",
        json!({}),
    ))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:move",
        "article:b",
        "relates_to",
        "article:c",
        json!({}),
    ))?;
    let rewired = notices.lock().unwrap().last().unwrap().clone();
    assert_eq!(
        rewired.changed_edge_endpoints,
        vec!["article:a", "article:b", "article:c"]
    );
    assert!(rewired
        .changed_object_keys
        .contains(&"edge/edge:move".into()));
    let before_refusal = notices.lock().unwrap().len();
    assert!(store
        .commit_batch_with_metadata(
            GraphMutationBatch::new([GraphMutation::NodeDelete("article:b".into())])
                .with_expected_root(Hash::of(b"stale observer fixture")),
            history_metadata(7),
        )
        .is_err());
    assert_eq!(notices.lock().unwrap().len(), before_refusal);
    store.delete_node("article:b")?;
    let deleted = notices.lock().unwrap().last().unwrap().clone();
    assert_eq!(
        deleted.changed_edge_endpoints,
        vec!["article:b", "article:c"]
    );
    let history = store.pin_graph_history()?;
    for event in notices.lock().unwrap().iter() {
        let entry = history.entry(event.commit.commit_hash)?;
        assert_eq!(entry.commit, event.commit);
        assert_eq!(entry.roots.transaction, event.transaction);
        assert_eq!(entry.changed_object_keys, event.changed_object_keys);
    }
    Ok(())
}

#[test]
fn graph_commit_observer_emits_only_after_real_journal_publication_and_contains_panics(
) -> GraphStoreResult<()> {
    use crate::hooks::GraphCommitObserver;
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let notices = Arc::new(std::sync::Mutex::new(Vec::new()));
    let received = Arc::clone(&notices);
    store.attach_commit_observer(GraphCommitObserver::new(move |event| {
        received.lock().unwrap().push(event.clone());
    }));
    let first = history_write(&mut store, 1)?;
    store.inject_journal_append_failure_for_test(0, 0, false)?;
    assert!(history_write(&mut store, 2).is_err());
    assert_eq!(notices.lock().unwrap().len(), 1);
    assert_eq!(notices.lock().unwrap()[0].commit, first.commit);
    drop(store);
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    store.attach_commit_observer(GraphCommitObserver::new(|_| {
        panic!("observer fixture panic")
    }));
    let final_commit = history_write(&mut store, 3)?;
    assert_eq!(
        store
            .pin_graph_history()?
            .entry(final_commit.commit.commit_hash)?
            .commit,
        final_commit.commit
    );
    assert_eq!(
        store.get_node("article:a")?.unwrap().properties["body"],
        "body 3"
    );
    Ok(())
}

#[test]
fn graph_history_attributed_guarded_mutations_preserve_cas_quota_and_version_boundaries(
) -> GraphStoreResult<()> {
    let mut store = RedCoreGraphStore::memory();
    assert!(store
        .insert_node_if_absent_with_metadata_checked(
            history_node("body"),
            history_metadata(1),
            |_, _| Ok(())
        )?
        .is_some());
    assert!(store
        .insert_node_if_absent_with_metadata_checked(
            history_node("duplicate"),
            history_metadata(2),
            |_, _| panic!("dedup must not check a losing batch")
        )?
        .is_none());
    assert!(store
        .insert_node_if_never_existed_with_metadata_checked(
            history_node("reuse"),
            history_metadata(2),
            |_, _| panic!("historical guard must reject")
        )?
        .is_none());
    let head = store.pin_graph_history()?.head_roots().clone();
    let rejected = store.compare_and_set_node_property_with_metadata_checked(
        "article:a",
        "body",
        &json!("body"),
        json!("changed"),
        history_metadata(3),
        |_, batch| {
            let GraphMutation::NodeUpsert(node) = &batch.mutations[0] else {
                panic!("prepared replacement");
            };
            assert_eq!(node.properties["body"], "changed");
            Err(GraphStoreError::new("quota-fixture", "replacement refused"))
        },
    );
    assert_eq!(rejected.unwrap_err().code, "quota-fixture");
    assert_eq!(store.pin_graph_history()?.head_roots(), &head);
    assert!(store
        .compare_and_set_node_property_with_metadata_checked(
            "article:a",
            "body",
            &json!("wrong"),
            json!("changed"),
            history_metadata(3),
            |_, _| panic!("CAS must reject before quota")
        )?
        .is_none());
    assert!(store
        .compare_and_set_node_property_with_metadata_checked(
            "article:a",
            "body",
            &json!("body"),
            json!("changed"),
            history_metadata(4),
            |_, _| Ok(())
        )?
        .is_some());
    assert_eq!(
        store
            .pin_graph_history()?
            .head_revision()?
            .commit()
            .unwrap()
            .author,
        "verified-user:4"
    );
    let version = store.stats()?.version;
    assert!(store
        .commit_batch_if_version_with_metadata(
            version + 1,
            GraphMutationBatch::new([GraphMutation::NodeUpsert(history_node("bad version"))]),
            history_metadata(5)
        )
        .is_err());
    let transaction = store.commit_batch_if_version_with_metadata(
        version,
        GraphMutationBatch::new([GraphMutation::NodeUpsert(history_node("versioned"))]),
        history_metadata(6),
    )?;
    let committed = store.pin_graph_history()?.head_revision()?;
    assert_eq!(committed.roots.transaction, transaction.txn_id);
    assert_eq!(committed.commit().unwrap().author, "verified-user:6");
    assert_eq!(store.pin_graph_history()?.page(None, 10)?.entries.len(), 3);
    Ok(())
}

#[test]
fn graph_history_forward_catchup_is_ordered_bounded_and_pinned() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    let mut commits = Vec::new();
    for i in 1..=7 {
        commits.push(history_write(&mut store, i)?);
    }
    let captured = store.pin_graph_history()?;
    let old = captured.view_at(commits[3].commit.commit_hash)?;
    history_write(&mut store, 8)?;
    let mut after = 0;
    let mut actual = Vec::new();
    loop {
        let page = captured.entries_after(after, 2)?;
        assert!(page.len() <= 2);
        if page.is_empty() {
            break;
        }
        for entry in page {
            assert!(entry.roots.transaction > after);
            after = entry.roots.transaction;
            actual.push(entry.commit.commit_hash);
        }
    }
    assert_eq!(
        actual,
        commits
            .iter()
            .map(|receipt| receipt.commit.commit_hash)
            .collect::<Vec<_>>()
    );
    assert_eq!(old.entries_after(0, 20)?.len(), 4);
    assert!(old.entries_after(commits[4].transaction.txn_id, 2).is_err());
    assert!(captured.entries_after(0, 0).is_err());
    assert!(captured
        .entries_after(0, crate::graph_history::MAX_HISTORY_PAGE + 1)
        .is_err());
    assert_eq!(captured.entries_after(after, 2)?.len(), 0);
    drop(old);
    drop(captured);
    drop(store);
    let reopened = RedCoreGraphStore::open(dir.path(), history_options())?;
    assert_eq!(
        reopened.pin_graph_history()?.entries_after(after, 2)?.len(),
        1
    );
    Ok(())
}

#[test]
fn graph_history_prefix_pages_are_bounded_isolated_and_pinned() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), history_options())?;
    store.commit_batch_with_metadata(
        GraphMutationBatch::new(
            [
                "wiki-watch:alpha:a",
                "wiki-watch:alpha:é",
                "wiki-watch:beta:a",
                "wiki-watch:alphabeta:a",
            ]
            .map(|id| GraphMutation::NodeUpsert(NodeRecord::new(id, ["Fixture"], json!({})))),
        ),
        history_metadata(1),
    )?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let prefix = "node/wiki-watch:alpha:";
    let first = pin.records_prefix_page(prefix, None, 1)?;
    assert_eq!(
        first.iter().map(|record| record.key()).collect::<Vec<_>>(),
        ["node/wiki-watch:alpha:a"]
    );
    let second = pin.records_prefix_page(prefix, Some(&first[0].key()), 1)?;
    assert_eq!(
        second.iter().map(|record| record.key()).collect::<Vec<_>>(),
        ["node/wiki-watch:alpha:é"]
    );
    assert!(pin
        .records_prefix_page(prefix, Some(&second[0].key()), 16)?
        .is_empty());
    assert!(pin
        .records_prefix_page(prefix, Some("node/wiki-watch:beta:a"), 1)
        .is_err());
    assert!(pin.records_prefix_page("untyped/", None, 1).is_err());
    assert!(
        pin.records_page(prefix, None, 1).is_err(),
        "old namespace-only API stays strict"
    );
    store.commit_batch_with_metadata(
        GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "wiki-watch:alpha:future",
            ["Fixture"],
            json!({}),
        ))]),
        history_metadata(2),
    )?;
    assert_eq!(pin.records_prefix_page(prefix, None, 16)?.len(), 2);
    assert_eq!(
        store
            .pin_graph_history()?
            .head_revision()?
            .records_prefix_page(prefix, None, 16)?
            .len(),
        3
    );
    drop(pin);
    drop(store);
    let reopened = RedCoreGraphStore::open(dir.path(), history_options())?;
    assert_eq!(
        reopened
            .pin_graph_history()?
            .head_revision()?
            .records_prefix_page(prefix, None, 16)?
            .len(),
        3
    );
    Ok(())
}
