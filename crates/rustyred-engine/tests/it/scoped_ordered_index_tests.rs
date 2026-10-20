use rustyred_thg_core::{OrderedMode, ScopedOrderedIndex, ScopedOrderedIndexManifest};

fn frontier() -> ScopedOrderedIndex {
    ScopedOrderedIndex::new(ScopedOrderedIndexManifest::new(
        "ordered:crawl-frontier",
        "Crawl frontier",
        "tenant/project",
        "priority + freshness + trust + politeness",
        "WebDoc",
        OrderedMode::Transient,
    ))
    .unwrap()
}

#[test]
fn scoped_ordered_frontier_pops_highest_priority_without_scan() {
    let mut index = frontier();
    let scope = "Travis-Gilbert/Theorem";

    for i in 0..10_000 {
        index
            .add_or_update(scope, &format!("url:{i:05}"), (i % 997) as f64, i as u64)
            .unwrap();
    }
    index
        .add_or_update(scope, "url:highest", 10_000.0, 10_000)
        .unwrap();
    assert_eq!(index.cardinality(scope), 10_001);

    index.reset_ops();
    let popped = index.pop_max(scope).unwrap();

    assert_eq!(popped.id, "url:highest");
    assert_eq!(popped.score.get(), 10_000.0);
    assert_eq!(popped.hydration_handle.label, "WebDoc");
    assert_eq!(index.ops(), 1, "frontier pop is one ordered-index op");
    assert_eq!(index.cardinality(scope), 10_000);
}

#[test]
fn scoped_ordered_ttl_queue_pops_next_expiring_without_scan() {
    let mut index = ScopedOrderedIndex::new(ScopedOrderedIndexManifest::new(
        "ordered:ttl-sweep",
        "TTL sweep queue",
        "tenant/project",
        "expires_at_ms",
        "ContextAtom",
        OrderedMode::Transient,
    ))
    .unwrap();
    let scope = "Travis-Gilbert/Theorem";

    for i in 0..10_000 {
        index
            .add_or_update(scope, &format!("atom:{i:05}"), 2_000.0 + i as f64, i)
            .unwrap();
    }
    index.add_or_update(scope, "atom:next", 10.0, 10).unwrap();

    index.reset_ops();
    let next = index.pop_min(scope).unwrap();

    assert_eq!(next.id, "atom:next");
    assert_eq!(next.score.get(), 10.0);
    assert_eq!(next.hydration_handle.label, "ContextAtom");
    assert_eq!(index.ops(), 1, "ttl pop is one ordered-index op");
    assert_eq!(index.cardinality(scope), 10_000);
}

#[test]
fn scoped_ordered_range_rank_remove_and_tie_breaks_are_stable() {
    let mut index = ScopedOrderedIndex::new(ScopedOrderedIndexManifest::new(
        "ordered:training-examples",
        "Training examples",
        "tenant/project",
        "quality_score",
        "LabeledTrainingRun",
        OrderedMode::Persistent,
    ))
    .unwrap();
    let scope = "Travis-Gilbert/Theorem";

    index.add_or_update(scope, "training:b", 9.0, 1).unwrap();
    index.add_or_update(scope, "training:a", 9.0, 1).unwrap();
    index.add_or_update(scope, "training:c", 12.0, 1).unwrap();

    let entries = index.range_by_score(scope, 0.0, 10.0, None).unwrap();
    let ids = entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["training:a", "training:b"]);
    assert_eq!(index.rank(scope, "training:a"), Some(0));
    assert_eq!(index.rank(scope, "training:b"), Some(1));

    assert!(index.remove(scope, "training:a"));
    assert_eq!(index.rank(scope, "training:b"), Some(0));
    assert_eq!(index.cardinality(scope), 2);

    let best = index.pop_max(scope).unwrap();
    assert_eq!(best.id, "training:c");
    assert_eq!(best.hydration_handle.object_id, "training:c");
}
