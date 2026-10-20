use rustyred_thg_core::{
    EdgeRecord, FactsFold, FactsQuery, Fold, FoldCapabilities, FoldError, FoldGraphStore, FoldId,
    FoldRuntime, GraphFold, GraphMutation, GraphMutationBatch, GraphWrite, HistoryFold, NodeRecord,
    RestoreDisposition, SegmentTransition, TransitionComposition,
};
use serde_json::json;
use std::error::Error;
use theorem_oplog::{LogPosition, PhysicalLog};

#[test]
fn concrete_fold_descriptors_are_deterministic_and_truthful() -> Result<(), Box<dyn Error>> {
    let descriptors = [
        FactsFold::default().descriptor(),
        GraphFold::default().descriptor(),
        rustyred_thg_core::fold::algebra::descriptor(),
        HistoryFold::default().descriptor(),
        rustyred_thg_core::DocumentsFold::default().descriptor(),
    ];
    let repeated = [
        FactsFold::default().descriptor(),
        GraphFold::default().descriptor(),
        rustyred_thg_core::fold::algebra::descriptor(),
        HistoryFold::default().descriptor(),
        rustyred_thg_core::DocumentsFold::default().descriptor(),
    ];
    assert_eq!(
        serde_json::to_vec(&descriptors)?,
        serde_json::to_vec(&repeated)?,
        "fresh descriptor construction must have stable wire bytes"
    );
    assert_eq!(
        descriptors
            .iter()
            .map(|item| item.fold_id)
            .collect::<Vec<_>>(),
        FoldId::ALL
    );
    for descriptor in descriptors {
        assert_eq!(
            descriptor.capabilities.transition_composition,
            TransitionComposition::OrderedStateTransitions
        );
        assert_eq!(descriptor.capabilities.segment_summary, None);
    }
    Ok(())
}

#[test]
fn real_folds_obey_ordered_transition_identity_and_associativity() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    build_context_sensitive_log(directory.path())?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;

    assert_transition_laws::<FactsFold>(&log)?;
    assert_transition_laws::<GraphFold>(&log)?;
    assert_transition_laws::<HistoryFold>(&log)?;
    Ok(())
}

#[test]
fn context_sensitive_tail_is_not_accepted_as_snapshot_summary() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    build_context_sensitive_log(directory.path())?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let latest = log.latest_position();
    let tail = SegmentTransition::from_log(&log, LogPosition(latest.0 - 1), latest)?;

    let mut without_prefix = FoldRuntime::for_log(GraphFold::default(), &log)?;
    let error = without_prefix
        .apply_transition(&log, &tail)
        .expect_err("a delete/update tail cannot act as a context-free snapshot summary");
    assert!(error.to_string().contains("runtime is at 0"));
    assert_eq!(
        without_prefix.descriptor().capabilities,
        FoldCapabilities::ordered_transitions_only()
    );

    let mut with_prefix = FoldRuntime::for_log(GraphFold::default(), &log)?;
    with_prefix.apply_range(&log, tail.from_exclusive)?;
    with_prefix.apply_transition(&log, &tail)?;
    let mut full = FoldRuntime::for_log(GraphFold::default(), &log)?;
    full.apply_all(&log)?;
    assert_eq!(
        with_prefix.inner().checkpoint()?,
        full.inner().checkpoint()?
    );
    Ok(())
}

fn assert_transition_laws<F>(log: &PhysicalLog) -> Result<(), Box<dyn Error>>
where
    F: LawFold,
{
    let latest = log.latest_position();
    let first = SegmentTransition::from_log(log, LogPosition::ZERO, LogPosition(1))?;
    let middle = SegmentTransition::from_log(log, LogPosition(1), LogPosition(3))?;
    let last = SegmentTransition::from_log(log, LogPosition(3), latest)?;
    let left = first.compose(&middle)?.compose(&last)?;
    let right = first.compose(&middle.compose(&last)?)?;
    assert_eq!(left, right);
    assert_eq!(
        SegmentTransition::identity(log, LogPosition::ZERO)?.compose(&left)?,
        left
    );
    assert_eq!(
        left.compose(&SegmentTransition::identity(log, latest)?)?,
        left
    );
    let mut backwards = first.clone();
    backwards.from_exclusive = LogPosition(2);
    backwards
        .compose(&middle)
        .expect_err("composition must refuse a forged backward interval");

    let mut composed = FoldRuntime::for_log(F::default(), log)?;
    composed.apply_transition(log, &left)?;
    let mut segmented = FoldRuntime::for_log(F::default(), log)?;
    segmented.apply_transition(log, &first)?;
    segmented.apply_transition(log, &middle)?;
    segmented.apply_transition(log, &last)?;
    let mut replayed = FoldRuntime::for_log(F::default(), log)?;
    replayed.apply_all(log)?;

    let mut checkpoint_source = FoldRuntime::for_log(F::default(), log)?;
    checkpoint_source.apply_range(log, LogPosition(2))?;
    let checkpoint = checkpoint_source.checkpoint(log)?;
    let mut restored = FoldRuntime::for_log(F::default(), log)?;
    assert!(matches!(
        restored.restore_or_rebuild(&checkpoint, log)?,
        RestoreDisposition::Restored { .. }
    ));

    let expected = replayed.inner().law_observation(latest)?;
    assert_eq!(composed.inner().law_observation(latest)?, expected);
    assert_eq!(segmented.inner().law_observation(latest)?, expected);
    assert_eq!(restored.inner().law_observation(latest)?, expected);
    assert_eq!(
        composed.quarantine_receipts(),
        replayed.quarantine_receipts()
    );
    Ok(())
}

trait LawFold: Fold + Default {
    fn law_observation(&self, position: LogPosition) -> Result<Vec<u8>, FoldError>;
}

impl LawFold for FactsFold {
    fn law_observation(&self, position: LogPosition) -> Result<Vec<u8>, FoldError> {
        let observations = ["node:a", "node:b"].map(|entity| {
            self.facts_at(
                &FactsQuery {
                    entity: entity.to_string(),
                    attribute: None,
                },
                position,
            )
        });
        serde_json::to_vec(&observations).map_err(|error| FoldError::Codec(error.to_string()))
    }
}

impl LawFold for GraphFold {
    fn law_observation(&self, _position: LogPosition) -> Result<Vec<u8>, FoldError> {
        self.checkpoint()
    }
}

impl LawFold for HistoryFold {
    fn law_observation(&self, _position: LogPosition) -> Result<Vec<u8>, FoldError> {
        self.checkpoint()
    }
}

fn build_context_sensitive_log(directory: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let mut store = FoldGraphStore::open(directory, deployment_id())?;
    store.commit_batch(GraphMutationBatch::new([
        GraphMutation::NodeUpsert(node("node:a", 1)),
        GraphMutation::NodeUpsert(node("node:b", 2)),
    ]))?;
    store.upsert_edge(EdgeRecord::new(
        "edge:ab",
        "node:a",
        "LINKS",
        "node:b",
        json!({"rank": 1}),
    ))?;
    store.upsert_node(node("node:a", 3))?;
    store.delete_node("node:b")?;
    assert_eq!(store.latest_position(), LogPosition(4));
    Ok(())
}

fn node(id: &str, value: i64) -> NodeRecord {
    NodeRecord::new(id, ["FoldLawFixture"], json!({"value": value}))
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:fold-algebra-laws").unwrap()
}
