use std::error::Error;

use rustyred_thg_core::cold_index::FoldHotness;
use rustyred_thg_core::query_receipt::{OverlapConsumer, PinnedReadSet};
use rustyred_thg_core::{NeighborQuery, NodeQuery};
use serde_json::json;
use theorem_oplog::{
    AccessPath, CommitOutcome, IntentKind, LogIntent, LogPosition, OpDigest, PhysicalLog, Proposal,
    ProposalCommitter, ReadSet, WriteSet,
};

#[test]
fn every_consumer_uses_the_same_normalized_overlap_trace() -> Result<(), Box<dyn Error>> {
    assert!(PinnedReadSet::point_node(LogPosition::ZERO, "   ").is_err());
    assert_eq!(
        PinnedReadSet::point_node(LogPosition::ZERO, " alpha ")?
            .read_set
            .entries(),
        &[entity("node:alpha")]
    );

    let cases = vec![
        Case::new(
            "point_hit",
            PinnedReadSet::point_node(LogPosition::ZERO, "alpha")?,
            WriteSet::new([entity("node:alpha")])?,
            true,
            false,
        ),
        Case::new(
            "point_miss",
            PinnedReadSet::point_node(LogPosition::ZERO, "alpha")?,
            WriteSet::new([entity("node:beta")])?,
            false,
            false,
        ),
        Case::new(
            "node_query_false_positive",
            NodeQuery::label("MemoryAtom").capture_reads(LogPosition::ZERO),
            WriteSet::new([entity("edge:unrelated")])?,
            true,
            true,
        ),
        Case::new(
            "neighbor_query_false_positive",
            NeighborQuery::out("node:alpha").capture_reads(LogPosition::ZERO),
            WriteSet::new([entity("node:unrelated")])?,
            true,
            true,
        ),
        Case::new(
            "merged_interval_hit",
            PinnedReadSet::new(
                LogPosition::ZERO,
                ReadSet::new([
                    interval("note:1", "body", 0, 10),
                    interval("note:1", "body", 10, 20),
                ])?,
            ),
            WriteSet::new([interval("note:1", "body", 19, 30)])?,
            true,
            false,
        ),
        Case::new(
            "adjacent_interval_miss",
            PinnedReadSet::new(
                LogPosition::ZERO,
                ReadSet::new([interval("note:1", "body", 0, 10)])?,
            ),
            WriteSet::new([interval("note:1", "body", 10, 20)])?,
            false,
            false,
        ),
    ];

    let mut hotness = FoldHotness::default();
    for case in cases {
        let occ = ProposalCommitter::evaluate_occ(&case.reads.read_set, &case.writes);
        let cache = case.reads.cache_invalidation_trace(&case.writes);
        let subscription = case.reads.subscription_trace(&case.writes);
        let hot = hotness.observe(case.name, &case.reads, &case.writes);

        assert_eq!(occ.overlaps, case.expected, "{} OCC", case.name);
        assert_eq!(cache.overlaps, case.expected, "{} cache", case.name);
        assert_eq!(
            subscription.overlaps, case.expected,
            "{} subscription",
            case.name
        );
        assert_eq!(hot.overlaps, case.expected, "{} hotness", case.name);
        assert_eq!(cache.consumer, OverlapConsumer::CacheInvalidation);
        assert_eq!(subscription.consumer, OverlapConsumer::Subscription);
        assert_eq!(hot.consumer, OverlapConsumer::FoldHotness);

        for trace in [&cache, &subscription, &hot] {
            assert_eq!(
                trace.read_position, case.reads.position,
                "{} position",
                case.name
            );
            assert_eq!(&trace.read_set, occ.read_set, "{} reads", case.name);
            assert_eq!(&trace.write_set, occ.write_set, "{} writes", case.name);
        }
        assert_eq!(
            hotness.count(case.name),
            u64::from(case.expected),
            "{} hotness count",
            case.name
        );
        assert_eq!(
            occ_conflicted(case.name, case.reads.read_set.clone(), case.writes.clone())?,
            case.expected,
            "{} physical OCC outcome",
            case.name
        );

        if case.subscription_false_positive {
            assert_eq!(case.reads.read_set.entries(), &[AccessPath::Wildcard]);
            assert!(subscription.overlaps);
        }
    }
    Ok(())
}

struct Case {
    name: &'static str,
    reads: PinnedReadSet,
    writes: WriteSet,
    expected: bool,
    subscription_false_positive: bool,
}

impl Case {
    fn new(
        name: &'static str,
        reads: PinnedReadSet,
        writes: WriteSet,
        expected: bool,
        subscription_false_positive: bool,
    ) -> Self {
        Self {
            name,
            reads,
            writes,
            expected,
            subscription_false_positive,
        }
    }
}

fn occ_conflicted(
    name: &str,
    reads: ReadSet,
    prior_writes: WriteSet,
) -> Result<bool, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer = ProposalCommitter::new(PhysicalLog::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:overlap-consumers")?,
    )?);
    let baseline = proposal(
        LogPosition::ZERO,
        format!("{name}:baseline"),
        ReadSet::default(),
        prior_writes,
    )?;
    assert!(matches!(
        committer.commit(baseline)?,
        CommitOutcome::Committed(_)
    ));

    let candidate = proposal(
        LogPosition::ZERO,
        format!("{name}:candidate"),
        reads,
        WriteSet::new([entity(&format!("audit:{name}"))])?,
    )?;
    Ok(matches!(
        committer.commit(candidate)?,
        CommitOutcome::Refused(_)
    ))
}

fn proposal(
    begin_position: LogPosition,
    digest: String,
    reads: ReadSet,
    writes: WriteSet,
) -> Result<Proposal, Box<dyn Error>> {
    Ok(Proposal::new(
        begin_position,
        reads,
        writes,
        OpDigest::from_bytes(&digest),
        LogIntent::new(
            IntentKind::new("theorem", "overlap_probe", 1)?,
            json!({"digest": digest}),
        )?,
    ))
}

fn entity(entity: &str) -> AccessPath {
    AccessPath::Entity {
        entity: entity.to_string(),
    }
}

fn interval(entity: &str, attribute: &str, start: i64, end: i64) -> AccessPath {
    AccessPath::Interval {
        entity: entity.to_string(),
        attribute: attribute.to_string(),
        start,
        end,
    }
}
