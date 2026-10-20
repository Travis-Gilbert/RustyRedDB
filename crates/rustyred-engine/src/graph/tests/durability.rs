#[test]
fn graph_store_trait_covers_memory_oracle_contract() {
    fn write_fixture(store: &mut dyn GraphStore) {
        store
            .upsert_node(NodeRecord::new("node:a", ["Fixture"], json!({})))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("node:b", ["Fixture"], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "LINKS",
                "node:b",
                json!({}),
            ))
            .unwrap();
    }

    let mut store = InMemoryGraphStore::new();
    write_fixture(&mut store);

    assert_eq!(store.stats().nodes_total, 2);
    assert_eq!(store.verify().ok, true);
    assert_eq!(
        store.neighbors(NeighborQuery::out("node:a"))[0].node_id,
        "node:b"
    );
}

#[test]
fn redcore_embedded_store_recovers_nodes_edges_and_indexes_from_journal() {
    let data_dir = unique_test_dir("redcore-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:a",
                ["File"],
                json!({ "path": "src/lib.rs", "repo": "rusty-red" }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:b",
                ["File"],
                json!({ "path": "src/main.rs", "repo": "rusty-red" }),
            ))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "IMPORTS",
                "node:b",
                json!({ "rank": 1 }),
            ))
            .unwrap();
        assert_eq!(store.verify().unwrap().ok, true);
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let hits = store
        .query_nodes(
            NodeQuery::label("File")
                .with_property("repo", json!("rusty-red"))
                .with_property("path", json!("src/lib.rs")),
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "node:a");
    assert_eq!(
        store.neighbors(NeighborQuery::out("node:a")).unwrap()[0].node_id,
        "node:b"
    );
    let status = store.status();
    assert_eq!(status.recovered_frames, 1);
    assert_eq!(status.skipped_aof_frames, 0);
    assert_eq!(status.aof_replay_bytes, 0);
    assert!(status.aof_size_bytes > 0);
    assert_eq!(status.txn_lag_since_snapshot, 0);
    assert_eq!(status.truncated_aof_tail, false);
    assert!(store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .is_some());
    for legacy in [
        super::REDCORE_AOF_FILE,
        super::REDCORE_SNAPSHOT_FILE,
        super::REDCORE_MANIFEST_FILE,
    ] {
        assert!(!data_dir.join(legacy).exists(), "legacy artifact {legacy}");
    }
    assert_eq!(store.verify().unwrap().ok, true);

    std::fs::remove_dir_all(data_dir).ok();
}

// ---- RedCore TTL decorator tests (TTL-03 design amendment) -------
//
// These cover the durability behavior of the RedCore wrapper around
// InMemory's TTL state. The TTL semantics themselves are covered by
// the 18 inline InMemory tests further down in this module; the two
// tests below verify only that the journal recovery path reconstructs both
// the TTL property AND the in-memory ttl_index correctly across a
// process restart cycle.

#[test]
fn redcore_set_node_ttl_persists_through_journal_recovery() {
    let data_dir = unique_test_dir("redcore-ttl-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    // Far-future expiration so the node remains live across recovery.
    let expires_at_ms = now_ms() + 60_000;

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:atom",
                ["MemoryAtom"],
                json!({ "title": "mention-1" }),
            ))
            .unwrap();
        store
            .set_node_ttl("node:atom", Some(expires_at_ms))
            .unwrap();
        assert_eq!(store.ttl_active_count(), 1);
    }

    // Reopen hydrates the rooted record and rebuilds the derived TTL index.
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let node = store
        .get_node("node:atom")
        .unwrap()
        .expect("node should be live after recovery");
    assert_eq!(
        node_ttl_expires_at_ms(&node),
        Some(expires_at_ms),
        "TTL property should survive journal recovery"
    );
    assert_eq!(
        store.ttl_active_count(),
        1,
        "ttl_index should be rebuilt during recovery"
    );
    let expiring = store.nodes_expiring_before(expires_at_ms + 1, 10);
    assert_eq!(expiring.len(), 1);
    assert_eq!(expiring[0].id, "node:atom");

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_guarded_insert_persists_only_winner() {
    let data_dir = unique_test_dir("redcore-guarded-insert");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let first = store
            .insert_node_if_absent(NodeRecord::new(
                "node:url",
                ["url"],
                json!({ "state": "frontier" }),
            ))
            .unwrap();
        let second = store
            .insert_node_if_absent(NodeRecord::new(
                "node:url",
                ["url"],
                json!({ "state": "frontier" }),
            ))
            .unwrap();
        assert!(first.is_some());
        assert!(second.is_none());
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:url").unwrap().is_some());
    assert_eq!(store.status().recovered_frames, 1);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_guarded_insert_durably_replaces_expired_node_before_purge() {
    let data_dir = unique_test_dir("redcore-guarded-insert-expired");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:url",
                ["url"],
                json!({ "state": "expired", TTL_PROPERTY: now_ms() - 60_000 }),
            ))
            .unwrap();
        let replacement = store
            .insert_node_if_absent(NodeRecord::new(
                "node:url",
                ["url"],
                json!({ "state": "refreshed" }),
            ))
            .unwrap();
        assert!(replacement.is_some());
        assert_eq!(
            store.get_node("node:url").unwrap().unwrap().properties["state"],
            "refreshed"
        );
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let node = reopened.get_node("node:url").unwrap().unwrap();
    assert_eq!(node.properties["state"], "refreshed");
    assert_eq!(node_ttl_expires_at_ms(&node), None);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_historical_registry_incompleteness_persists_and_fails_closed_after_reopen() {
    let data_dir = unique_test_dir("redcore-historical-registry-incomplete");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store.store.historical_node_id_limit = 2;
        store.upsert_node(plain_node("history:a")).unwrap();
        store.upsert_node(plain_node("history:b")).unwrap();
        store.upsert_node(plain_node("history:c")).unwrap();
        assert_eq!(store.store.historical_node_ids.len(), 2);
        assert!(!store.store.historical_node_history_complete);
        let error = store
            .insert_node_if_never_existed(plain_node("history:unknown-before-reopen"))
            .unwrap_err();
        assert_eq!(error.code, "historical_registry_incomplete");
    }

    {
        let mut reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
        assert_eq!(reopened.store.historical_node_ids.len(), 2);
        assert!(!reopened.store.historical_node_history_complete);
        assert!(reopened.get_node("history:c").unwrap().is_some());
        let error = reopened
            .insert_node_if_never_existed(plain_node("history:unknown-after-reopen"))
            .unwrap_err();
        assert_eq!(error.code, "historical_registry_incomplete");
    }

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_compare_and_set_node_property_claims_once() {
    let data_dir = unique_test_dir("redcore-guarded-cas");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:url",
                ["url"],
                json!({ "state": "frontier" }),
            ))
            .unwrap();
        let first = store
            .compare_and_set_node_property(
                "node:url",
                "state",
                &json!("frontier"),
                json!("in_flight"),
            )
            .unwrap();
        let second = store
            .compare_and_set_node_property(
                "node:url",
                "state",
                &json!("frontier"),
                json!("in_flight"),
            )
            .unwrap();
        assert!(first.is_some());
        assert!(second.is_none());
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let state = store
        .get_node("node:url")
        .unwrap()
        .and_then(|node| node.properties.get("state").cloned());
    assert_eq!(state, Some(json!("in_flight")));

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_get_node_filters_expired_after_journal_recovery() {
    let data_dir = unique_test_dir("redcore-ttl-expired-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    // Past expiration: the node is already expired at write time.
    // After recovery, get_node must filter it out (TTL semantics are
    // lazy until TTL-04 adds the sweep), but the data is still in
    // storage and visible via get_node_including_expired.
    let expires_at_ms = now_ms() - 1_000;

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:stale",
                ["MemoryAtom"],
                json!({ "title": "expired-mention", TTL_PROPERTY: expires_at_ms }),
            ))
            .unwrap();
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(
        store.get_node("node:stale").unwrap().is_none(),
        "expired node should be filtered after journal recovery"
    );
    assert!(
        store
            .get_node_including_expired("node:stale")
            .unwrap()
            .is_some(),
        "expired node should still be visible to forensic reads"
    );
    let live = GraphRead::live_node_records(&store);
    assert!(
        live.iter().all(|node| node.id != "node:stale"),
        "expired nodes must not remain visible through live enumeration"
    );
    assert_eq!(
        live.iter()
            .filter(|node| node.labels.iter().any(|label| label == CONTROL_EVENT_LABEL))
            .count(),
        1,
        "the durable blind upsert event remains a live control-plane node"
    );
    assert_eq!(
        store.ttl_active_count(),
        1,
        "ttl_index should still hold the expired node (sweep is TTL-04)"
    );

    std::fs::remove_dir_all(data_dir).ok();
}

// ---- RedCore durable-purge tests (TTL-04) ------------------------
//
// These cover the logical tombstone + RedCore::purge_expired_nodes
// wiring. The InMemory side is covered by the existing
// `purge_expired_nodes_returns_count_and_clears_storage` test; what
// these add is the journal-root durability contract that the
// RedCore decorator brings.

#[test]
fn redcore_purge_expired_survives_journal_recovery() {
    let data_dir = unique_test_dir("redcore-ttl-durable-purge");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    // One node already past expiration, one node still live. After
    // purge the expired one must be gone from BOTH memory AND the
    // journal-recovered reopen (the durable-purge property).
    let past = now_ms() - 10_000;
    let future = now_ms() + 60_000;

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:expired",
                ["MemoryAtom"],
                json!({ "title": "stale", TTL_PROPERTY: past }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:live",
                ["MemoryAtom"],
                json!({ "title": "fresh", TTL_PROPERTY: future }),
            ))
            .unwrap();
        assert_eq!(store.ttl_active_count(), 2);

        // Purge publishes a logical tombstone and new catalog root.
        let purged = store.purge_expired_nodes().unwrap();
        assert_eq!(purged, 1, "exactly one expired node should be purged");
        assert!(
            store.get_node("node:expired").unwrap().is_none(),
            "purged node should be gone from memory"
        );
        assert!(
            store.get_node("node:live").unwrap().is_some(),
            "live node should remain"
        );
        assert_eq!(store.ttl_active_count(), 1);
    }

    // Reopen from the current journal root. End state matches the
    // pre-close state without replaying a second mutation stream.
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(
        store.get_node("node:expired").unwrap().is_none(),
        "purged node must stay gone after journal recovery"
    );
    assert!(
        store.get_node("node:live").unwrap().is_some(),
        "live node should survive journal recovery"
    );
    assert_eq!(
        store.ttl_active_count(),
        1,
        "ttl_index should reflect the rooted post-purge state"
    );
    assert_eq!(store.status().recovered_frames, 1);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_delete_node_survives_journal_recovery() {
    let data_dir = unique_test_dir("redcore-delete-node-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "file:src/lib.rs",
                ["File"],
                json!({ "doc_tree_path": "src/lib.rs" }),
            ))
            .unwrap();

        assert!(store.delete_node("file:src/lib.rs").unwrap());
        assert_eq!(store.status().graph_version, 4);
        assert!(store
            .query_nodes(NodeQuery::label("File"))
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
                .unwrap()
                .len(),
            2
        );
        assert!(
            !store.delete_node("file:src/lib.rs").unwrap(),
            "missing delete is a no-op"
        );
        assert!(store.get_node("file:src/lib.rs").unwrap().is_none());
        assert_eq!(
            store.status().last_txn_id,
            2,
            "upsert + one real delete transaction; no-op delete writes nothing"
        );
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(
        store.get_node("file:src/lib.rs").unwrap().is_none(),
        "deleted node must stay gone after journal recovery"
    );
    assert_eq!(
        store.status().recovered_frames,
        1,
        "recovery hydrates one current catalog root"
    );
    assert_eq!(store.status().graph_version, 4);
    assert!(store
        .query_nodes(NodeQuery::label("File"))
        .unwrap()
        .is_empty());
    assert_eq!(
        store
            .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
            .unwrap()
            .len(),
        2
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn entity_resolution_redcore_delete_edge_is_versioned_durable_and_blocks_stale_readmission() {
    let data_dir = unique_test_dir("redcore-delete-edge-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let stale;
    let deleted_version;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for id in ["a", "b"] {
            store
                .upsert_node(NodeRecord::new(id, ["Record"], json!({})))
                .unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new("ab", "a", "match_edge", "b", json!({})))
            .unwrap();
        stale = store.store.edges["ab"].clone();
        let before = store.store.version;
        store.start_recording_redcore_mutations();
        assert!(store.delete_edge("ab").unwrap());
        deleted_version = store.store.version;
        assert_eq!(
            store.take_recorded_redcore_mutations().unwrap().mutations,
            vec![RecordedRedCoreMutation::EdgeDelete {
                id: "ab".into(),
                version: deleted_version
            }]
        );
        assert_eq!(deleted_version, before + 1);
        assert!(store.get_edge("ab").unwrap().is_none());
        assert!(store
            .store
            .neighbors(NeighborQuery::out("a").with_edge_type("match_edge"))
            .is_empty());
        store.store.readmit_edge(stale.clone()).unwrap();
        assert!(
            store.get_edge("ab").unwrap().is_none(),
            "stale warm edge cannot re-admit"
        );
        store.snapshot_now().unwrap();
    }
    let mut reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_edge("ab").unwrap().is_none());
    assert!(reopened.store.version >= deleted_version);
    reopened.store.readmit_edge(stale).unwrap();
    assert!(reopened.get_edge("ab").unwrap().is_none());
    let version = reopened.store.version;
    reopened
        .apply_recovered_mutation(RedCoreMutation::EdgeDelete("ab".into()))
        .unwrap();
    reopened
        .apply_recovered_mutation(RedCoreMutation::EdgeDelete("ab".into()))
        .unwrap();
    assert_eq!(
        reopened.store.version, version,
        "duplicate replay is idempotent"
    );
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn entity_resolution_edge_delete_reopens_from_journal_without_legacy_snapshot() {
    let data_dir = unique_test_dir("entity-edge-delete-journal-only");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 10_000,
        strict_acid: false,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for id in ["a", "b"] {
            store
                .upsert_node(NodeRecord::new(id, ["Record"], json!({})))
                .unwrap();
        }
        store
            .upsert_edge(EdgeRecord::new("ab", "a", "match_edge", "b", json!({})))
            .unwrap();
        assert!(store.delete_edge("ab").unwrap());
    }
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_edge("ab").unwrap().is_none());
    assert_eq!(reopened.status().recovered_frames, 1);
    assert!(!data_dir.join(super::REDCORE_SNAPSHOT_FILE).exists());
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_purge_no_op_does_not_advance_catalog_root() {
    let data_dir = unique_test_dir("redcore-ttl-purge-no-op");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let future = now_ms() + 60_000;

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:fresh",
                ["MemoryAtom"],
                json!({ "title": "fresh", TTL_PROPERTY: future }),
            ))
            .unwrap();
        let txn_id_before = store.status().last_txn_id;
        let root_before = store.journal.as_ref().unwrap().read_root().unwrap();

        let purged = store.purge_expired_nodes().unwrap();
        assert_eq!(purged, 0, "nothing expired should return 0");

        let txn_id_after = store.status().last_txn_id;
        assert_eq!(
            txn_id_before, txn_id_after,
            "no-op purge must not advance last_txn_id"
        );
        assert_eq!(
            store.journal.as_ref().unwrap().read_root().unwrap(),
            root_before,
            "no-op purge must not publish a catalog root"
        );
    }

    // Reopen confirms the original rooted node remains the current state.
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(store.status().recovered_frames, 1);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_purge_batched_when_multiple_expire_at_once() {
    let data_dir = unique_test_dir("redcore-ttl-purge-batch");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let past = now_ms() - 5_000;

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for n in 0..5 {
            store
                .upsert_node(NodeRecord::new(
                    format!("node:exp{n}"),
                    ["MemoryAtom"],
                    json!({ "title": format!("atom-{n}"), TTL_PROPERTY: past }),
                ))
                .unwrap();
        }
        let txn_id_before = store.status().last_txn_id;

        let purged = store.purge_expired_nodes().unwrap();
        assert_eq!(purged, 5);

        // One logical batch publishes one new catalog root.
        assert_eq!(store.status().last_txn_id, txn_id_before + 1);
    }

    // Reopen from the single current root; all five tombstones remain final.
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    for n in 0..5 {
        assert!(
            store.get_node(&format!("node:exp{n}")).unwrap().is_none(),
            "node:exp{n} should be gone after batch-purge recovery"
        );
    }
    assert_eq!(store.ttl_active_count(), 0);
    assert_eq!(store.status().recovered_frames, 1);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_ttl_purge_durably_removes_incident_edges() {
    let data_dir = unique_test_dir("redcore-ttl-purge-incident-edges");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };

    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:expired",
                ["MemoryAtom"],
                json!({ TTL_PROPERTY: now_ms() - 60_000 }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("node:live", ["MemoryAtom"], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "edge:expired-live",
                "node:expired",
                "LINKS",
                "node:live",
                json!({}),
            ))
            .unwrap();

        assert_eq!(store.purge_expired_nodes().unwrap(), 1);
        assert!(store.get_edge("edge:expired-live").unwrap().is_none());
        assert!(store.verify().unwrap().ok);
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:expired").unwrap().is_none());
    assert!(reopened.get_edge("edge:expired-live").unwrap().is_none());
    assert!(reopened.verify().unwrap().ok);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn non_finite_edge_confidence_is_refused_by_direct_and_batch_writes() {
    let mut direct = InMemoryGraphStore::new();
    direct
        .upsert_node(NodeRecord::new("node:a", ["Claim"], json!({})))
        .unwrap();
    direct
        .upsert_node(NodeRecord::new("node:b", ["Claim"], json!({})))
        .unwrap();
    let mut edge = EdgeRecord::new("edge:bad", "node:a", "LINKS", "node:b", json!({}));
    edge.confidence = Some(f64::NAN);
    assert_eq!(
        direct.upsert_edge(edge.clone()).unwrap_err().code,
        "edge_confidence_non_finite"
    );
    edge.confidence = Some(f64::INFINITY);
    assert_eq!(
        direct.upsert_edge(edge.clone()).unwrap_err().code,
        "edge_confidence_non_finite"
    );

    let mut batch = RedCoreGraphStore::memory();
    let error = batch
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["Claim"], json!({}))),
            GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["Claim"], json!({}))),
            GraphMutation::EdgeUpsert(edge),
        ]))
        .unwrap_err();
    assert_eq!(error.code, "edge_confidence_non_finite");
    assert!(batch.get_node("node:a").unwrap().is_none());
    assert_eq!(batch.status().graph_version, 0);
}

#[test]
fn redcore_batch_commit_is_all_or_nothing() {
    let mut store = RedCoreGraphStore::memory();
    let error = store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:missing",
                "node:a",
                "IMPORTS",
                "node:missing",
                json!({}),
            )),
        ]))
        .unwrap_err();

    assert_eq!(error.code, "missing_graph_endpoint");
    assert!(store.get_node("node:a").unwrap().is_none());
    assert_eq!(store.status().graph_version, 0);
    assert_eq!(store.status().last_txn_id, 0);

    let transaction = store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
            GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["File"], json!({}))),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "IMPORTS",
                "node:b",
                json!({}),
            )),
        ]))
        .unwrap();

    assert_eq!(transaction.txn_id, 1);
    assert_eq!(transaction.graph_version, 5);
    assert_eq!(transaction.writes.len(), 3);
    assert_eq!(
        store
            .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
            .unwrap()
            .len(),
        2
    );
    assert_eq!(store.status().last_txn_id, 1);
    assert_eq!(store.verify().unwrap().ok, true);
}

#[test]
fn redcore_records_committed_mutations_for_external_mirrors() {
    let mut store = RedCoreGraphStore::memory();
    store.start_recording_committed_mutations();

    store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
            GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["File"], json!({}))),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "IMPORTS",
                "node:b",
                json!({}),
            )),
        ]))
        .unwrap();

    let recorded = store
        .take_recorded_committed_mutations()
        .expect("recording session returns a batch");
    assert_eq!(recorded.mutations.len(), 5);
    let wire = serde_json::to_value(&recorded).unwrap();
    assert_eq!(wire["mutations"][0]["op"], json!("node_upsert"));
    assert_eq!(wire["mutations"][0]["record"]["id"], json!("node:a"));
    let _: GraphMutationBatch = serde_json::from_value(wire).unwrap();
    match &recorded.mutations[0] {
        GraphMutation::NodeUpsert(node) => {
            assert_eq!(node.id, "node:a");
            assert_eq!(node.version, 1);
            assert!(node.content_hash.is_some());
        }
        _ => panic!("first recorded mutation should be node:a"),
    }
    match &recorded.mutations[2] {
        GraphMutation::EdgeUpsert(edge) => {
            assert_eq!(edge.id, "edge:ab");
            assert_eq!(edge.version, 3);
            assert!(edge.content_hash.is_some());
        }
        _ => panic!("third recorded mutation should be edge:ab"),
    }
    assert_eq!(
        recorded
            .mutations
            .iter()
            .filter(|mutation| matches!(
                mutation,
                GraphMutation::NodeUpsert(node)
                    if node.labels.iter().any(|label| label == CONTROL_EVENT_LABEL)
            ))
            .count(),
        2
    );
    assert!(store.take_recorded_committed_mutations().is_none());
}

#[test]
fn redcore_recovers_batch_commit_from_journal_as_one_transaction() {
    let data_dir = unique_test_dir("redcore-batch-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let transaction = store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:a",
                    ["File"],
                    json!({ "path": "src/lib.rs" }),
                )),
                GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:b",
                    ["File"],
                    json!({ "path": "src/main.rs" }),
                )),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ab",
                    "node:a",
                    "IMPORTS",
                    "node:b",
                    json!({}),
                )),
            ]))
            .unwrap();

        assert_eq!(transaction.txn_id, 1);
        assert_eq!(transaction.graph_version, 5);
        assert_eq!(transaction.writes.len(), 3);
        assert_eq!(store.status().last_txn_id, 1);
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();

    assert_eq!(store.status().recovered_frames, 1);
    assert_eq!(store.status().last_txn_id, 1);
    assert_eq!(store.status().graph_version, 5);
    assert_eq!(
        store
            .query_nodes(NodeQuery::label(CONTROL_EVENT_LABEL))
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        store.neighbors(NeighborQuery::out("node:a")).unwrap()[0].node_id,
        "node:b"
    );
    assert_eq!(store.verify().unwrap().ok, true);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn ordinary_upsert_only_commit_advances_incremental_journal_root() {
    let data_dir = unique_test_dir("redcore-incremental-journal-root");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let first_root;
    let second_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:incremental",
                ["File"],
                json!({ "path": "src/lib.rs" }),
            ))
            .unwrap();
        first_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:incremental",
                ["File"],
                json!({ "path": "src/updated.rs" }),
            ))
            .unwrap();
        second_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        assert_ne!(first_root, second_root);
        assert_eq!(store.status().last_txn_id, 2);
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        reopened
            .get_node("node:incremental")
            .unwrap()
            .unwrap()
            .properties["path"],
        "src/updated.rs"
    );
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(second_root)
    );
    assert!(!data_dir.join(super::REDCORE_AOF_FILE).exists());
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn version_checked_upsert_publishes_only_after_version_gate() {
    let data_dir = unique_test_dir("redcore-version-checked-journal-route");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let committed_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        let error = store
            .commit_batch_if_version(
                1,
                GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:rejected",
                    ["File"],
                    json!({}),
                ))]),
            )
            .unwrap_err();
        assert_eq!(error.code, "graph_version_conflict");
        assert_eq!(store.journal.as_ref().unwrap().read_root().unwrap(), None);
        store
            .commit_batch_if_version(
                0,
                GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                    "node:checked",
                    ["File"],
                    json!({ "path": "src/checked.rs" }),
                ))]),
            )
            .unwrap();
        committed_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:checked").unwrap().is_some());
    assert!(reopened.get_node("node:rejected").unwrap().is_none());
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(committed_root)
    );
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_mixed_transaction_publishes_one_atomic_catalog_root() {
    let data_dir = unique_test_dir("redcore-mixed-transaction");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let first_root;
    let mixed_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
                GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["File"], json!({}))),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ab",
                    "node:a",
                    "LINKS",
                    "node:b",
                    json!({}),
                )),
            ]))
            .unwrap();
        first_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();

        let transaction = store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ac",
                    "node:a",
                    "LINKS",
                    "node:c",
                    json!({}),
                )),
                GraphMutation::NodeUpsert(NodeRecord::new("node:c", ["File"], json!({}))),
                GraphMutation::NodeDelete("node:b".to_string()),
                GraphMutation::EdgeDelete("edge:ab".to_string()),
            ]))
            .unwrap();

        assert_eq!(transaction.txn_id, 2);
        mixed_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        assert_ne!(mixed_root, first_root);
        assert!(store.get_node("node:b").unwrap().is_none());
        assert!(store.get_node("node:c").unwrap().is_some());
        assert!(store.get_edge("edge:ab").unwrap().is_none());
        assert!(store.get_edge("edge:ac").unwrap().is_some());
        assert!(store.verify().unwrap().ok);
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:b").unwrap().is_none());
    assert!(store.get_node("node:c").unwrap().is_some());
    assert!(store.get_edge("edge:ab").unwrap().is_none());
    assert!(store.get_edge("edge:ac").unwrap().is_some());
    assert_eq!(store.status().last_txn_id, 2);
    assert_eq!(
        store.journal.as_ref().unwrap().read_root().unwrap(),
        Some(mixed_root)
    );
    let catalog = crate::chunk_store::validate_catalog_root(
        store.journal.as_ref().unwrap().as_ref(),
        mixed_root,
    )
    .unwrap();
    assert!(catalog.get(crate::chunk_store::ROOT_NAME_GRAPH).is_some());
    assert!(catalog.get(crate::chunk_store::ROOT_NAME_REDCORE).is_some());
    assert!(store.verify().unwrap().ok);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_publication_window_survives_old_root_compaction_interleaving() {
    let data_dir = unique_test_dir("redcore-publication-window-compaction");
    let options = redcore_journal_test_options();
    let final_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:prior",
                ["File"],
                json!({"path": "prior.rs"}),
            ))
            .unwrap();
        let journal = Arc::clone(store.journal.as_ref().unwrap());
        let prior_reachability =
            walk_committed_reachable(journal.as_ref(), &CoreReachabilityCodec).unwrap();
        let prior_reachable_count = prior_reachability.reachable.len();
        store.journal_precommit_compaction_reachable =
            Some(prior_reachability.reachable.iter().copied().collect());

        store
            .upsert_node(NodeRecord::new(
                "node:published-after-compaction",
                ["File"],
                json!({"path": "published.rs", "body": "novel payload"}),
            ))
            .unwrap();

        let interleaving = store
            .journal_precommit_compaction_report
            .take()
            .expect("precommit compaction hook ran");
        assert!(
            interleaving.chunks_after > prior_reachable_count,
            "unpublished chunks survived solely through publication-window pins"
        );
        assert!(store
            .get_node("node:published-after-compaction")
            .unwrap()
            .is_some());
        final_root = journal.read_root().unwrap().unwrap();
        let catalog =
            crate::chunk_store::validate_catalog_root(journal.as_ref(), final_root).unwrap();
        for address in [
            catalog.graph_root().unwrap(),
            catalog.get(crate::chunk_store::ROOT_NAME_REDCORE).unwrap(),
        ] {
            assert_eq!(journal.in_flight_pin_count(address).unwrap(), 0);
        }
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(final_root)
    );
    assert!(reopened
        .get_node("node:published-after-compaction")
        .unwrap()
        .is_some());
    assert!(reopened.verify().unwrap().ok);
    drop(reopened);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_invalid_mixed_transaction_leaves_store_and_root_unchanged() {
    let data_dir = unique_test_dir("redcore-mixed-journal-publish-gate");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
            GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["File"], json!({}))),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:ab",
                "node:a",
                "LINKS",
                "node:b",
                json!({}),
            )),
        ]))
        .unwrap();
    let before = store.status();
    let before_root = store.journal.as_ref().unwrap().read_root().unwrap();

    let error = store
        .commit_batch(GraphMutationBatch::new([
            GraphMutation::EdgeDelete("edge:ab".to_string()),
            GraphMutation::NodeDelete("node:b".to_string()),
            GraphMutation::NodeUpsert(NodeRecord::new("node:c", ["File"], json!({}))),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:ac",
                "node:a",
                "LINKS",
                "node:missing",
                json!({}),
            )),
        ]))
        .unwrap_err();

    assert_eq!(error.code, "missing_graph_endpoint");
    assert_eq!(store.status().graph_version, before.graph_version);
    assert_eq!(store.status().last_txn_id, before.last_txn_id);
    assert_eq!(
        store.journal.as_ref().unwrap().read_root().unwrap(),
        before_root
    );
    assert!(store.get_node("node:b").unwrap().is_some());
    assert!(store.get_node("node:c").unwrap().is_none());
    assert!(store.get_edge("edge:ab").unwrap().is_some());
    assert!(store.get_edge("edge:ac").unwrap().is_none());
    assert!(store.verify().unwrap().ok);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_version_checked_batch_conflict_does_not_publish() {
    let mut store = RedCoreGraphStore::memory();
    store
        .upsert_node(NodeRecord::new("node:a", ["File"], json!({})))
        .unwrap();
    let before = store.status();

    let error = store
        .commit_batch_if_version(
            0,
            GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
                "node:b",
                ["File"],
                json!({}),
            ))]),
        )
        .unwrap_err();

    assert_eq!(error.code, "graph_version_conflict");
    assert_eq!(store.status().graph_version, before.graph_version);
    assert_eq!(store.status().last_txn_id, before.last_txn_id);
    assert!(store.get_node("node:b").unwrap().is_none());
}

#[test]
fn torn_mixed_journal_recovers_prior_or_complete_catalog_root_only() {
    let data_dir = unique_test_dir("redcore-mixed-torn-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let journal_path = data_dir.join(super::REDCORE_JOURNAL_FILE);
    let prior_bytes;
    let prior_root;
    let complete_bytes;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::NodeUpsert(NodeRecord::new("node:a", ["File"], json!({}))),
                GraphMutation::NodeUpsert(NodeRecord::new("node:b", ["File"], json!({}))),
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    "edge:ab",
                    "node:a",
                    "LINKS",
                    "node:b",
                    json!({}),
                )),
            ]))
            .unwrap();
        prior_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        prior_bytes = std::fs::read(&journal_path).unwrap();
        store
            .commit_batch(GraphMutationBatch::new([
                GraphMutation::EdgeDelete("edge:ab".to_string()),
                GraphMutation::NodeDelete("node:b".to_string()),
                GraphMutation::NodeUpsert(NodeRecord::new("node:c", ["File"], json!({}))),
            ]))
            .unwrap();
        complete_bytes = std::fs::read(&journal_path).unwrap();
    }
    assert!(complete_bytes.len() > prior_bytes.len());
    let delta_len = complete_bytes.len() - prior_bytes.len();
    let torn_len = prior_bytes.len() + delta_len.saturating_div(2).max(1);
    std::fs::write(&journal_path, &complete_bytes[..torn_len]).unwrap();
    {
        let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        assert_eq!(
            store.journal.as_ref().unwrap().read_root().unwrap(),
            Some(prior_root)
        );
        assert!(store.get_node("node:b").unwrap().is_some());
        assert!(store.get_node("node:c").unwrap().is_none());
        assert!(store.get_edge("edge:ab").unwrap().is_some());
        assert!(store.verify().unwrap().ok);
    }

    std::fs::write(&journal_path, &complete_bytes).unwrap();
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:b").unwrap().is_none());
    assert!(store.get_node("node:c").unwrap().is_some());
    assert!(store.get_edge("edge:ab").unwrap().is_none());
    assert!(store.verify().unwrap().ok);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn graph_snapshot_recovery_skips_orphan_edges_instead_of_poisoning_store() {
    let snapshot = GraphSnapshot {
        historical_node_ids: Default::default(),
        historical_node_history_complete: false,
        version: 2,
        nodes: vec![NodeRecord::new(
            "actor:sandbox",
            ["Actor"],
            json!({ "actor_id": "sandbox" }),
        )],
        edges: vec![EdgeRecord::new(
            "edge:orphan-created-by",
            "mem:missing-memory-atom",
            "CREATED_BY",
            "actor:sandbox",
            json!({ "actor_kind": "sandbox" }),
        )],
    };

    let store = InMemoryGraphStore::from_snapshot(snapshot).unwrap();

    assert!(store.get_node("actor:sandbox").is_some());
    assert!(store.get_edge("edge:orphan-created-by").is_none());
    assert_eq!(store.verify().ok, true);
}

#[test]
fn redcore_legacy_aof_path_collision_does_not_gate_journal_publication() {
    let data_dir = unique_test_dir("redcore-legacy-aof-path-collision");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    std::fs::create_dir(data_dir.join(super::REDCORE_AOF_FILE)).unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:published",
            ["File"],
            json!({ "path": "published.rs" }),
        ))
        .unwrap();

    assert!(store.get_node("node:published").unwrap().is_some());
    assert_eq!(store.status().graph_version, 2);
    assert_eq!(store.status().last_txn_id, 1);
    assert!(store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .is_some());

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_snapshot_only_mode_uses_journal_not_legacy_snapshot_writer() {
    let data_dir = unique_test_dir("redcore-snapshot-mode-journal-route");
    let options = RedCoreOptions {
        durability: RedCoreDurability::SnapshotOnly,
        snapshot_interval_writes: 1,
        strict_acid: false,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
    std::fs::create_dir(data_dir.join("graph.snapshot.current.tmp")).unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:published",
            ["File"],
            json!({ "path": "published.rs" }),
        ))
        .unwrap();

    assert!(store.get_node("node:published").unwrap().is_some());
    assert_eq!(store.status().snapshot_txn_id, 1);
    assert!(store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .is_some());
    drop(store);
    std::fs::remove_dir(data_dir.join("graph.snapshot.current.tmp")).unwrap();
    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:published").unwrap().is_some());
    assert!(!data_dir.join(super::REDCORE_SNAPSHOT_FILE).exists());

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_legacy_manifest_path_collision_does_not_gate_journal_publication() {
    let data_dir = unique_test_dir("redcore-legacy-manifest-path-collision");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    std::fs::create_dir(data_dir.join("manifest.json.tmp")).unwrap();

    store
        .upsert_node(NodeRecord::new(
            "node:published",
            ["File"],
            json!({ "path": "published.rs" }),
        ))
        .unwrap();

    assert!(store.get_node("node:published").unwrap().is_some());
    assert_eq!(store.status().last_txn_id, 1);
    assert!(store
        .journal
        .as_ref()
        .unwrap()
        .read_root()
        .unwrap()
        .is_some());
    assert!(!data_dir.join(super::REDCORE_MANIFEST_FILE).exists());

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_snapshot_only_compatibility_mode_recovers_from_journal_without_redis() {
    let data_dir = unique_test_dir("redcore-snapshot-mode-journal-recovery");
    let options = RedCoreOptions {
        durability: RedCoreDurability::SnapshotOnly,
        snapshot_interval_writes: 1,
        strict_acid: false,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:snapshot",
                ["Snapshot"],
                json!({ "mode": "snapshot_only" }),
            ))
            .unwrap();
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        store.get_node("node:snapshot").unwrap().unwrap().labels,
        vec!["Snapshot".to_string()]
    );
    assert_eq!(store.status().snapshot_txn_id, 1);
    assert_eq!(store.verify().unwrap().ok, true);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_checked_in_legacy_fixture_migrates_to_current_journal() {
    let data_dir = unique_test_dir("redcore-format-fixture");
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("redcore-v1");
    std::fs::create_dir_all(&data_dir).unwrap();
    for file in ["manifest.json", "graph.snapshot.current"] {
        std::fs::copy(fixture_dir.join(file), data_dir.join(file)).unwrap();
    }

    let options = RedCoreOptions {
        durability: RedCoreDurability::SnapshotOnly,
        snapshot_interval_writes: 10,
        strict_acid: false,
    };
    let mut store = RedCoreGraphStore::open(&data_dir, options).unwrap();

    assert!(data_dir.join(super::REDCORE_JOURNAL_FILE).is_file());
    assert!(!data_dir.join(super::REDCORE_MANIFEST_FILE).exists());
    assert!(!data_dir.join(super::REDCORE_SNAPSHOT_FILE).exists());
    let backup = data_dir.join(super::REDCORE_LEGACY_BACKUP_DIR);
    assert!(backup.join(super::REDCORE_MANIFEST_FILE).is_file());
    assert!(backup.join(super::REDCORE_SNAPSHOT_FILE).is_file());
    assert_eq!(
        store.get_node("doc:a").unwrap().unwrap().labels,
        vec!["Doc"]
    );
    assert_eq!(
        store
            .query_nodes(NodeQuery {
                label: Some("Doc".to_string()),
                properties: std::collections::BTreeMap::new(),
                limit: Some(10),
                include_expired: false,
            })
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        store.neighbors(NeighborQuery::out("doc:a")).unwrap()[0].node_id,
        "doc:b"
    );
    assert!(store.verify().unwrap().ok);
    assert!(store.rebuild_indexes().unwrap().after.ok);
    store
        .designate_vector_property("Doc", "embedding", 2)
        .unwrap();
    let vector_hits = store
        .vector_search(Some("Doc"), "embedding", &[1.0, 0.0], 2)
        .unwrap();
    assert_eq!(vector_hits[0].0, "doc:a");

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_strict_acid_accepts_all_durable_journal_modes_and_rejects_none() {
    for (label, durability) in [
        ("always", RedCoreDurability::AofAlways),
        ("everysec", RedCoreDurability::AofEverysec),
        ("snapshot", RedCoreDurability::SnapshotOnly),
    ] {
        let data_dir = unique_test_dir(&format!("redcore-strict-{label}"));
        let options = RedCoreOptions {
            durability,
            snapshot_interval_writes: 100,
            strict_acid: true,
        };
        {
            let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
            store
                .upsert_node(NodeRecord::new("node:strict", ["Doc"], json!({})))
                .unwrap();
            assert!(store
                .journal
                .as_ref()
                .unwrap()
                .read_root()
                .unwrap()
                .is_some());
        }
        let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
        assert!(reopened.get_node("node:strict").unwrap().is_some());
        drop(reopened);
        std::fs::remove_dir_all(data_dir).ok();
    }

    let data_dir = unique_test_dir("redcore-strict-none-refused");
    let error = RedCoreGraphStore::open(
        &data_dir,
        RedCoreOptions {
            durability: RedCoreDurability::None,
            snapshot_interval_writes: 100,
            strict_acid: true,
        },
    )
    .unwrap_err();

    assert_eq!(error.code, "redcore_strict_mode_invalid");
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_durability_parser_rejects_unknown_policy() {
    assert_eq!(
        RedCoreDurability::parse("aof_everysec").unwrap(),
        RedCoreDurability::AofEverysec
    );
    let error = RedCoreDurability::parse("aof-always").unwrap_err();
    assert_eq!(error.code, "redcore_invalid_durability");
}

#[test]
fn redcore_aof_everysec_compatibility_mode_uses_journal_root_barriers() {
    let data_dir = unique_test_dir("redcore-aof-everysec-journal-barrier");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofEverysec,
        snapshot_interval_writes: 100,
        strict_acid: false,
    };
    let second_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new("node:first", ["Doc"], json!({})))
            .unwrap();
        let first_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        store
            .upsert_node(NodeRecord::new("node:second", ["Doc"], json!({})))
            .unwrap();
        second_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        assert_ne!(first_root, second_root);
        assert!(store.last_fsync.is_some());
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(reopened.get_node("node:first").unwrap().is_some());
    assert!(reopened.get_node("node:second").unwrap().is_some());
    assert_eq!(
        reopened.journal.as_ref().unwrap().read_root().unwrap(),
        Some(second_root)
    );
    assert!(reopened.verify().unwrap().ok);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_open_holds_exclusive_data_directory_lock() {
    let data_dir = unique_test_dir("redcore-file-lock");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let _store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
    let error = RedCoreGraphStore::open(&data_dir, options).unwrap_err();

    assert_eq!(error.code, "redcore_lock_unavailable");
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_resident_readiness_reuses_its_owned_journal() {
    let data_dir = unique_test_dir("redcore-resident-readiness");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();

    let competing = RedCoreGraphStore::readiness_check(
        &data_dir,
        options.durability,
        options.strict_acid,
    )
    .unwrap_err();
    assert_eq!(competing.code, "chunk_store_journal_lock_unavailable");
    assert!(store.resident_readiness_check().is_ok());
    assert!(store.resident_readiness_check().is_ok());

    drop(store);
    assert!(RedCoreGraphStore::readiness_check(
        &data_dir,
        options.durability,
        options.strict_acid,
    )
    .is_ok());
    std::fs::remove_dir_all(data_dir).ok();
}

#[cfg(unix)]
#[test]
fn redcore_resident_readiness_rejects_a_replaced_journal_path() {
    let data_dir = unique_test_dir("redcore-replaced-resident-journal");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let journal_path = data_dir.join(super::REDCORE_JOURNAL_FILE);
    let detached_path = data_dir.join("redcore.journal.detached");

    std::fs::rename(&journal_path, &detached_path).unwrap();
    std::fs::write(&journal_path, []).unwrap();

    let error = store.resident_readiness_check().unwrap_err();
    assert_eq!(error.code, "chunk_store_journal_replaced");

    drop(store);
    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_recovery_truncates_torn_journal_tail() {
    let data_dir = unique_test_dir("redcore-torn-journal-tail");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let committed_root;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:stable",
                ["File"],
                json!({ "path": "stable.rs" }),
            ))
            .unwrap();
        committed_root = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
    }
    let journal_path = data_dir.join(super::REDCORE_JOURNAL_FILE);
    let committed_len = std::fs::metadata(&journal_path).unwrap().len();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&journal_path)
        .unwrap()
        .write_all(b"RRCJ\x01\x01torn")
        .unwrap();
    assert!(std::fs::metadata(&journal_path).unwrap().len() > committed_len);

    let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
    assert!(store.get_node("node:stable").unwrap().is_some());
    assert_eq!(store.status().recovered_frames, 1);
    assert_eq!(
        store.journal.as_ref().unwrap().read_root().unwrap(),
        Some(committed_root)
    );
    assert_eq!(
        std::fs::metadata(&journal_path).unwrap().len(),
        committed_len
    );

    store
        .upsert_node(NodeRecord::new(
            "node:after-recovery",
            ["File"],
            json!({ "path": "after.rs" }),
        ))
        .unwrap();
    drop(store);
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:after-recovery").unwrap().is_some());
    assert_eq!(store.verify().unwrap().ok, true);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_reads_covered_aof_transaction_from_trusted_prefix() {
    let raw = r#"{"magic":"RRGDB_AOF","version":1,"txn_id":4000,"graph_version":9,"timestamp_unix_ms":1,"payload_checksum":"sha256:test","mutation":{"op":"batch","record":[{"large":"payload"}]}}"#;

    assert_eq!(super::covered_aof_txn_id(raw, 4_000), Some(4_000));
    assert_eq!(super::covered_aof_txn_id(raw, 3_999), None);
    assert_eq!(
        super::covered_aof_txn_id(
            r#"{"magic":"WRONG","version":1,"txn_id":1,"mutation":{}}"#,
            1,
        ),
        None
    );
}

#[test]
fn redcore_recovery_refuses_mid_journal_corruption_without_truncating() {
    let data_dir = unique_test_dir("redcore-mid-journal-corruption");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for index in 1..=3 {
            store
                .upsert_node(NodeRecord::new(
                    format!("node:{index}"),
                    ["File"],
                    json!({ "index": index }),
                ))
                .unwrap();
        }
    }

    let journal_path = data_dir.join(super::REDCORE_JOURNAL_FILE);
    let mut corrupted_journal = std::fs::read(&journal_path).unwrap();
    assert!(corrupted_journal.len() > 4);
    corrupted_journal[0] ^= 0x80;
    std::fs::write(&journal_path, &corrupted_journal).unwrap();

    let error = RedCoreGraphStore::open(&data_dir, options).unwrap_err();
    assert_eq!(error.code, "chunk_store_journal_corrupt");
    assert_eq!(std::fs::read(&journal_path).unwrap(), corrupted_journal);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_snapshot_now_republishes_catalog_without_legacy_stream() {
    let data_dir = unique_test_dir("redcore-journal-republish");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 2,
        strict_acid: true,
    };
    let published_catalog;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        for index in 1..=4 {
            store
                .upsert_node(NodeRecord::new(
                    format!("node:{index}"),
                    ["Snapshot"],
                    json!({ "index": index }),
                ))
                .unwrap();
        }

        store
            .upsert_node(NodeRecord::new(
                "node:5",
                ["Snapshot"],
                json!({ "index": 5 }),
            ))
            .unwrap();
        let journal = store.journal.as_ref().unwrap().clone();
        published_catalog = journal.read_root().unwrap().unwrap();
        let first_commit = journal.committed_root().unwrap().unwrap().commit;
        let before = store.status();
        store.snapshot_now().unwrap();
        let second_commit = journal.committed_root().unwrap().unwrap().commit;
        assert_ne!(first_commit, second_commit);
        assert_eq!(journal.read_root().unwrap(), Some(published_catalog));
        store.snapshot_now().unwrap();
        let third_commit = journal.committed_root().unwrap().unwrap().commit;
        assert_ne!(second_commit, third_commit);
        assert_eq!(store.status().last_txn_id, before.last_txn_id);
        assert_eq!(store.status().graph_version, before.graph_version);
        assert_eq!(store.status().snapshot_txn_id, before.last_txn_id);
        for legacy in [
            super::REDCORE_AOF_FILE,
            super::REDCORE_SNAPSHOT_FILE,
            super::REDCORE_MANIFEST_FILE,
        ] {
            assert!(!data_dir.join(legacy).exists(), "legacy artifact {legacy}");
        }
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    for index in 1..=5 {
        assert!(store.get_node(&format!("node:{index}")).unwrap().is_some());
    }
    assert_eq!(
        store.journal.as_ref().unwrap().read_root().unwrap(),
        Some(published_catalog)
    );
    assert!(store.verify().unwrap().ok);

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn redcore_current_journal_ignores_late_corrupt_legacy_snapshot() {
    let data_dir = unique_test_dir("redcore-ignore-late-legacy-snapshot");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 1,
        strict_acid: true,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:first",
                ["Snapshot"],
                json!({ "step": 1 }),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new(
                "node:second",
                ["Snapshot"],
                json!({ "step": 2 }),
            ))
            .unwrap();
    }
    std::fs::write(
        data_dir.join(super::REDCORE_SNAPSHOT_FILE),
        b"{corrupt snapshot",
    )
    .unwrap();

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert!(store.get_node("node:first").unwrap().is_some());
    assert!(store.get_node("node:second").unwrap().is_some());
    assert_eq!(store.status().last_txn_id, 2);
    assert_eq!(store.verify().unwrap().ok, true);

    std::fs::remove_dir_all(data_dir).ok();
}

fn unique_test_dir(label: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{label}-{unique}"))
}

#[test]
fn cold_eviction_is_cache_only_and_rooted_record_rehydrates_on_reopen() {
    let data_dir = unique_test_dir("redcore-cold-cache-rehydrate");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 5,
        strict_acid: true,
    };
    let root_after_delete;
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "memory:cold",
                ["MemoryAtom"],
                json!({"search_text": "rollback survives"}),
            ))
            .unwrap();
        store
            .upsert_node(NodeRecord::new("memory:hot", ["MemoryAtom"], json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new(
                "edge:cold-hot",
                "memory:cold",
                "RELATED_TO",
                "memory:hot",
                json!({}),
            ))
            .unwrap();
        assert!(crate::GraphResidency::evict_node(&mut store, "memory:cold")
            .unwrap()
            .is_some());
        store
            .upsert_node(NodeRecord::new("commit:trigger", ["Probe"], json!({})))
            .unwrap();
        assert!(store.delete_edge("edge:cold-hot").unwrap());
        root_after_delete = store
            .journal
            .as_ref()
            .unwrap()
            .read_root()
            .unwrap()
            .unwrap();
        assert!(store.get_node("memory:cold").unwrap().is_none());
        assert!(store.get_edge("edge:cold-hot").unwrap().is_none());
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        store.journal.as_ref().unwrap().read_root().unwrap(),
        Some(root_after_delete)
    );
    let restored = store.get_node("memory:cold").unwrap().unwrap();
    assert!(restored.labels.iter().any(|label| label == "MemoryAtom"));
    assert_eq!(restored.properties["search_text"], "rollback survives");
    assert!(store.get_edge("edge:cold-hot").unwrap().is_none());
    assert!(store.verify().unwrap().ok);
    for legacy in [
        super::REDCORE_AOF_FILE,
        super::REDCORE_SNAPSHOT_FILE,
        super::REDCORE_MANIFEST_FILE,
    ] {
        assert!(!data_dir.join(legacy).exists(), "legacy artifact {legacy}");
    }

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
fn cold_eviction_then_upsert_publishes_latest_payload() -> GraphStoreResult<()> {
    let data_dir = unique_test_dir("redcore-cold-evict-upsert");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "memory:cold",
                ["MemoryAtom"],
                json!({"search_text": "before rollback"}),
            ))
            .unwrap();
        assert!(crate::GraphResidency::evict_node(&mut store, "memory:cold")
            .unwrap()
            .is_some());
        store
            .upsert_node(NodeRecord::new(
                "memory:cold",
                ["MemoryAtom"],
                json!({"search_text": "updated after eviction"}),
            ))
            .unwrap();
    }

    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    let recovered = store.get_node("memory:cold").unwrap().unwrap();
    assert_eq!(
        recovered.properties["search_text"],
        "updated after eviction"
    );
    assert!(store.verify().unwrap().ok);

    std::fs::remove_dir_all(data_dir).ok();
    Ok(())
}

#[test]
fn cold_eviction_preserves_former_reserved_property_as_user_data() -> GraphStoreResult<()> {
    let data_dir = unique_test_dir("redcore-cold-user-property");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone()).unwrap();
        store
            .upsert_node(NodeRecord::new(
                "memory:reserved",
                ["MemoryAtom"],
                json!({ "_rustyred_cold_rollback_record": "user-data" }),
            ))
            .unwrap();
        assert!(
            crate::GraphResidency::evict_node(&mut store, "memory:reserved")
                .unwrap()
                .is_some()
        );
        assert!(store.get_node("memory:reserved").unwrap().is_none());
    }
    let store = RedCoreGraphStore::open(&data_dir, options).unwrap();
    assert_eq!(
        store
            .get_node("memory:reserved")
            .unwrap()
            .unwrap()
            .properties["_rustyred_cold_rollback_record"],
        "user-data"
    );

    std::fs::remove_dir_all(data_dir).ok();
    Ok(())
}

#[test]
fn resident_edge_budget_keeps_nodes_live_and_backpressures_complete_edge_reads(
) -> GraphStoreResult<()> {
    let data_dir = unique_test_dir("redcore-resident-edge-budget");
    let options = RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: true,
    };
    let (durable_nodes_before, durable_edges_before) = {
        let mut store = RedCoreGraphStore::open(&data_dir, options.clone())?;
        store.upsert_node(NodeRecord::new("node:a", ["Fixture"], json!({})))?;
        store.upsert_node(NodeRecord::new("node:b", ["Fixture"], json!({})))?;
        store.upsert_edge(EdgeRecord::new(
            "edge:ab",
            "node:a",
            "LINKS",
            "node:b",
            json!({"weight": 1}),
        ))?;
        let status = store.status();
        (status.durable_nodes_total, status.durable_edges_total)
    };

    {
        let mut bounded =
            RedCoreGraphStore::open_with_resident_edge_budget(&data_dir, options.clone(), 0)?;
        let status = bounded.status();
        assert!(!status.edges_fully_resident);
        assert_eq!(status.durable_nodes_total, durable_nodes_before);
        assert_eq!(status.durable_edges_total, durable_edges_before);
        assert_eq!(status.resident_edges_total, 0);
        assert_eq!(bounded.stats()?.edges_total, 1);
        assert!(bounded.get_node("node:a")?.is_some());

        for error in [
            bounded.get_edge("edge:ab").unwrap_err(),
            bounded.neighbors(NeighborQuery::out("node:a")).unwrap_err(),
            bounded.graph_snapshot_checked().unwrap_err(),
        ] {
            assert_eq!(error.code, "resident_edges_nonresident");
        }

        bounded.upsert_node(NodeRecord::new(
            "node:c",
            ["Fixture"],
            json!({"bounded": true}),
        ))?;
        bounded.upsert_edge(EdgeRecord::new(
            "edge:bc",
            "node:b",
            "LINKS",
            "node:c",
            json!({"weight": 2}),
        ))?;
        let status = bounded.status();
        assert_eq!(status.durable_edges_total, 2);
        assert_eq!(status.resident_edges_total, 0);
        assert_eq!(
            bounded.get_edge("edge:bc").unwrap_err().code,
            "resident_edges_nonresident"
        );
        let delete_error = bounded.delete_node("node:a").unwrap_err();
        assert_eq!(delete_error.code, "resident_edges_nonresident");
    }

    let reopened = RedCoreGraphStore::open(&data_dir, options)?;
    assert!(reopened.get_node("node:c")?.is_some());
    assert!(reopened.get_edge("edge:ab")?.is_some());
    assert!(reopened.get_edge("edge:bc")?.is_some());
    assert!(reopened.verify()?.ok);

    std::fs::remove_dir_all(data_dir).ok();
    Ok(())
}
