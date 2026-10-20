#![cfg(feature = "graphblas")]

use rustyred_thg_core::fold::algebra::{AlgebraFold, AlgebraReachability};
use rustyred_thg_core::fold::history::{HistoryCheckout, HistoryFold};
use rustyred_thg_core::fold::{
    Fold, FoldRuntime, LogInterval, RetractOutcome, RetractionReceipt, RetractionStrategy,
};
use rustyred_thg_core::partial_fold::{
    ExtremumOrder, SupportPromotionState, RETRACTION_WINDOW_APPLIED_ENTRIES,
};
use rustyred_thg_core::{
    ColdIndex, DiskColdIndex, EdgeRecord, FactTuple, FactsFold, FoldGraphStore, FoldStorage,
    GraphFold, GraphMutation, GraphMutationBatch, MutationOrder, NodeRecord, PartialFoldRuntime,
    PartialOperator, PositionedFoldIndex, RedbFoldStorage,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::sync::Arc;
use tempfile::TempDir;
use theorem_oplog::{LogPosition, PhysicalLog};

const TARGET: LogPosition = LogPosition(6);

struct LogFixture {
    _directory: TempDir,
    log: PhysicalLog,
}

#[test]
fn every_concrete_fold_declares_bounded_replay() {
    let strategies = [
        FactsFold::default().capabilities().retraction,
        GraphFold::default().capabilities().retraction,
        rustyred_thg_core::fold::algebra::descriptor()
            .capabilities
            .retraction,
        HistoryFold::default().capabilities().retraction,
    ];
    assert!(strategies
        .iter()
        .all(|strategy| *strategy == RetractionStrategy::BoundedReplay));
}

#[test]
fn all_folds_retract_through_the_nearest_strictly_earlier_checkpoint() -> Result<(), Box<dyn Error>>
{
    let original = build_log(false)?;
    let retained = build_log(true)?;

    let (facts, facts_receipt) = retracted_runtime::<FactsFold>(&original.log)?;
    assert_bounded_receipt(&facts_receipt);
    let facts_value = current_fact_value(&facts, &original.log, "edge:e", "type")?;
    let retained_facts = full_runtime::<FactsFold>(&retained.log)?;
    assert_eq!(
        facts_value,
        current_fact_value(&retained_facts, &retained.log, "edge:e", "type")?
    );

    let (graph, graph_receipt) = retracted_runtime::<GraphFold>(&original.log)?;
    assert_bounded_receipt(&graph_receipt);
    let graph_edge = graph.query(&json!({"kind": "edge", "id": "edge:e"}), LogPosition(7))?;
    let retained_graph = full_runtime::<GraphFold>(&retained.log)?;
    let retained_edge = retained_graph.query(
        &json!({"kind": "edge", "id": "edge:e"}),
        retained.log.latest_position(),
    )?;
    assert_eq!(graph_edge, retained_edge);
    assert!(graph_edge.is_object());

    let (mut algebra, algebra_receipt) = retracted_runtime::<AlgebraFold>(&original.log)?;
    assert_bounded_receipt(&algebra_receipt);
    let query = json!({
        "kind": "reachable",
        "edge_type": "LINKS",
        "sources": ["node:a"]
    });
    let algebra_actual =
        decode_reachability(algebra.upquery(&original.log, &query, LogPosition(7))?)?;
    let mut retained_algebra = full_runtime::<AlgebraFold>(&retained.log)?;
    let algebra_expected = decode_reachability(retained_algebra.upquery(
        &retained.log,
        &query,
        retained.log.latest_position(),
    )?)?;
    assert_eq!(algebra_actual.reachable, algebra_expected.reachable);
    assert_eq!(algebra_actual.reachable, vec!["node:a", "node:b", "node:c"]);

    let (history, history_receipt) = retracted_runtime::<HistoryFold>(&original.log)?;
    assert_bounded_receipt(&history_receipt);
    let history_actual =
        decode_checkout(history.query(&json!({"kind": "checkout"}), LogPosition(7))?)?;
    let retained_history = full_runtime::<HistoryFold>(&retained.log)?;
    let history_expected = decode_checkout(
        retained_history.query(&json!({"kind": "checkout"}), retained.log.latest_position())?,
    )?;
    assert_eq!(history_actual.snapshot, history_expected.snapshot);

    let checkpoint = algebra.checkpoint(&original.log)?;
    assert_eq!(checkpoint.retracted_positions.len(), 1);
    assert!(checkpoint.retracted_positions.contains(&TARGET));
    assert_eq!(checkpoint.retraction_receipts, vec![algebra_receipt]);
    let checkpoint = serde_json::from_slice(&serde_json::to_vec(&checkpoint)?)?;
    let mut restored = FoldRuntime::for_log(AlgebraFold::default(), &original.log)?;
    restored.restore_or_rebuild(&checkpoint, &original.log)?;
    assert_eq!(
        restored.retracted_positions(),
        algebra.retracted_positions()
    );
    assert_eq!(
        restored.retraction_receipts(),
        algebra.retraction_receipts()
    );
    let restored_answer = decode_reachability(restored.upquery(
        &original.log,
        &query,
        original.log.latest_position(),
    )?)?;
    assert_eq!(restored_answer.reachable, algebra_actual.reachable);
    let retry = restored.retract_position(&original.log, TARGET, &[])?;
    assert!(!retry.newly_retracted);
    assert_eq!(retry.strategy, RetractionStrategy::BoundedReplay);
    assert_eq!(retry.outcome, checkpoint.retraction_receipts[0].outcome);
    assert_eq!(retry.replay.applied, 0);
    assert_eq!(retry.replay.retracted, 0);
    assert_eq!(retry.replay.through, LogPosition(7));
    Ok(())
}

#[test]
fn seeded_multi_retractions_match_independent_retained_logs() -> Result<(), Box<dyn Error>> {
    for seed in 0_u64..8 {
        let targets = generated_targets(seed);
        let target_set = targets.iter().copied().collect::<BTreeSet<_>>();
        let original = build_generated_log(seed, &BTreeSet::new())?;
        let retained = build_generated_log(seed, &target_set)?;
        let entities = generated_entity_ids();

        let facts = round_trip_runtime(
            retract_generated::<FactsFold>(&original.log, &targets)?,
            &original.log,
        )?;
        let retained_facts = full_runtime::<FactsFold>(&retained.log)?;
        assert_eq!(
            normalized_facts(&facts, &entities)?,
            normalized_facts(&retained_facts, &entities)?,
            "Facts mismatch for seed {seed}"
        );

        let graph = round_trip_runtime(
            retract_generated::<GraphFold>(&original.log, &targets)?,
            &original.log,
        )?;
        let retained_graph = full_runtime::<GraphFold>(&retained.log)?;
        assert_eq!(
            graph_observation(&graph, &entities)?,
            graph_observation(&retained_graph, &entities)?,
            "Graph mismatch for seed {seed}"
        );

        let mut algebra = round_trip_runtime(
            retract_generated::<AlgebraFold>(&original.log, &targets)?,
            &original.log,
        )?;
        let mut retained_algebra = full_runtime::<AlgebraFold>(&retained.log)?;
        assert_eq!(
            algebra_observation(&mut algebra, &original.log)?,
            algebra_observation(&mut retained_algebra, &retained.log)?,
            "Algebra mismatch for seed {seed}"
        );

        let history = round_trip_runtime(
            retract_generated::<HistoryFold>(&original.log, &targets)?,
            &original.log,
        )?;
        let retained_history = full_runtime::<HistoryFold>(&retained.log)?;
        let history_checkout = decode_checkout(
            history.query(&json!({"kind": "checkout"}), history.current_position())?,
        )?;
        let retained_checkout = decode_checkout(retained_history.query(
            &json!({"kind": "checkout"}),
            retained_history.current_position(),
        )?)?;
        assert_eq!(
            history_checkout.snapshot, retained_checkout.snapshot,
            "History mismatch for seed {seed}"
        );
    }
    Ok(())
}

#[test]
fn promotion_is_measured_at_zero_one_and_two_per_hundred() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(DiskColdIndex::open(directory.path().join("cold"))?);
    let mut runtime = PartialFoldRuntime::new(PositionedFoldIndex::new(storage, cold), 128 * 1024)?;
    let operator = PartialOperator::FactsEntityAttribute;

    for (index, retractions) in [0_usize, 1, 2].into_iter().enumerate() {
        let key = format!("grade:{index}");
        runtime.register_extremum(operator, key.as_bytes(), ExtremumOrder::Maximum)?;
        let mut closed = None;
        for position in 1..=RETRACTION_WINDOW_APPLIED_ENTRIES {
            if position == 50 {
                for request in 0..retractions {
                    assert!(runtime.observe_retraction(
                        operator,
                        key.as_bytes(),
                        &format!("request:{index}:{request}"),
                    )?);
                }
                if retractions != 0 {
                    assert!(!runtime.observe_retraction(
                        operator,
                        key.as_bytes(),
                        &format!("request:{index}:0"),
                    )?);
                }
            }
            closed = runtime.observe_absorb(
                operator,
                key.as_bytes(),
                MutationOrder::new(LogPosition(u64::from(position)), 0)?,
            )?;
            if position < RETRACTION_WINDOW_APPLIED_ENTRIES {
                assert!(closed.is_none());
                assert_eq!(
                    runtime.promotion_state(operator, key.as_bytes())?,
                    Some(&SupportPromotionState::ReplayOnly)
                );
            }
        }
        let receipt = closed.expect("the hundredth applied entry closes the window");
        assert_eq!(receipt.applied_entries, 100);
        assert_eq!(receipt.distinct_retractions, retractions);
        assert_eq!(receipt.promotion_required, retractions >= 2);
        let expected = if retractions >= 2 {
            SupportPromotionState::BackfillPending
        } else {
            SupportPromotionState::ReplayOnly
        };
        assert_eq!(
            runtime.promotion_state(operator, key.as_bytes())?,
            Some(&expected)
        );
    }
    assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    Ok(())
}

#[test]
fn support_backfill_is_exact_budgeted_and_reversible() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(DiskColdIndex::open(directory.path().join("cold"))?);
    let mut runtime = PartialFoldRuntime::new(PositionedFoldIndex::new(storage, cold), 64 * 1024)?;
    let operator = PartialOperator::FactsEntityAttribute;
    let key = b"epistemic:grade";
    runtime.register_extremum(operator, key, ExtremumOrder::Maximum)?;
    runtime.observe_retraction(operator, key, "retract:first")?;
    runtime.observe_retraction(operator, key, "retract:second")?;
    for position in 1..=100 {
        runtime.observe_absorb(operator, key, MutationOrder::new(LogPosition(position), 0)?)?;
    }
    let fallback = LogInterval::new(LogPosition(4), LogPosition(12))?;
    assert_eq!(
        runtime.retract_supported_value(operator, key, &[30], fallback)?,
        RetractOutcome::RequiresRecompute {
            from_interval: fallback
        }
    );

    runtime.complete_support_backfill(operator, key, [vec![10], vec![20], vec![20], vec![30]])?;
    assert_eq!(
        runtime.promotion_state(operator, key)?,
        Some(&SupportPromotionState::SupportCounted)
    );
    assert_eq!(runtime.supported_extremum(operator, key)?, Some(&[30][..]));
    assert_eq!(
        runtime.retract_supported_value(operator, key, &[30], fallback)?,
        RetractOutcome::Retracted
    );
    assert_eq!(runtime.supported_extremum(operator, key)?, Some(&[20][..]));
    runtime.retract_supported_value(operator, key, &[20], fallback)?;
    assert_eq!(runtime.supported_extremum(operator, key)?, Some(&[20][..]));
    runtime.retract_supported_value(operator, key, &[20], fallback)?;
    assert_eq!(runtime.supported_extremum(operator, key)?, Some(&[10][..]));
    assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    assert!(runtime.accounting().retraction_ledger > 0);
    assert!(runtime.accounting().support_state > 0);

    assert!(runtime.rollback_support(operator, key)?);
    assert_eq!(
        runtime.promotion_state(operator, key)?,
        Some(&SupportPromotionState::ReplayOnly)
    );
    assert_eq!(runtime.supported_extremum(operator, key)?, None);
    assert_eq!(
        runtime.retract_supported_value(operator, key, &[10], fallback)?,
        RetractOutcome::RequiresRecompute {
            from_interval: fallback
        }
    );

    let denied_directory = tempfile::tempdir()?;
    let denied_storage: Arc<dyn FoldStorage> = Arc::new(RedbFoldStorage::open(
        denied_directory.path().join("floor"),
    )?);
    let denied_cold: Arc<dyn ColdIndex> =
        Arc::new(DiskColdIndex::open(denied_directory.path().join("cold"))?);
    let mut denied = PartialFoldRuntime::new(
        PositionedFoldIndex::new(denied_storage, denied_cold),
        8 * 1024,
    )?;
    denied.apply_update(
        operator,
        key,
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"resident".to_vec()),
    )?;
    assert!(denied.read_at(operator, key, LogPosition(1))?.retained);
    denied.register_extremum(operator, key, ExtremumOrder::Maximum)?;
    denied.observe_retraction(operator, key, "denied:first")?;
    denied.observe_retraction(operator, key, "denied:second")?;
    for position in 2..=101 {
        denied.observe_absorb(operator, key, MutationOrder::new(LogPosition(position), 0)?)?;
    }
    let error = denied
        .complete_support_backfill(operator, key, [vec![42; 64 * 1024]])
        .unwrap_err();
    assert_eq!(error.code, "partial_fold_budget");
    assert_eq!(
        denied.promotion_state(operator, key)?,
        Some(&SupportPromotionState::BackfillPending)
    );
    assert_eq!(denied.supported_extremum(operator, key)?, None);
    assert!(denied.resident(operator, key)?.is_some());
    assert!(denied.resident_bytes() <= denied.budget_bytes());
    Ok(())
}

fn retracted_runtime<F: Fold + Default>(
    log: &PhysicalLog,
) -> Result<(FoldRuntime<F>, RetractionReceipt), Box<dyn Error>> {
    let mut runtime = FoldRuntime::for_log(F::default(), log)?;
    runtime.apply_range(log, LogPosition(5))?;
    let before = runtime.checkpoint(log)?;
    runtime.apply_range(log, TARGET)?;
    let at = runtime.checkpoint(log)?;
    runtime.apply_all(log)?;
    let after = runtime.checkpoint(log)?;
    let receipt = runtime.retract_position(log, TARGET, &[after, at, before])?;
    Ok((runtime, receipt))
}

fn full_runtime<F: Fold + Default>(log: &PhysicalLog) -> Result<FoldRuntime<F>, Box<dyn Error>> {
    let mut runtime = FoldRuntime::for_log(F::default(), log)?;
    runtime.apply_all(log)?;
    Ok(runtime)
}

fn retract_generated<F: Fold + Default>(
    log: &PhysicalLog,
    targets: &[LogPosition],
) -> Result<FoldRuntime<F>, Box<dyn Error>> {
    let mut runtime = FoldRuntime::for_log(F::default(), log)?;
    let mut retracted = BTreeSet::new();
    for target in targets {
        let before_position = LogPosition(target.0 - 1);
        runtime.apply_range(log, before_position)?;
        let before = runtime.checkpoint(log)?;

        let foreign = if retracted.is_empty() {
            None
        } else {
            let mut without_overlay = FoldRuntime::for_log(F::default(), log)?;
            without_overlay.apply_range(log, before_position)?;
            Some(without_overlay.checkpoint(log)?)
        };

        runtime.apply_range(log, *target)?;
        let at = runtime.checkpoint(log)?;
        let through = LogPosition(target.0 + 1);
        runtime.apply_range(log, through)?;
        let after = runtime.checkpoint(log)?;
        let mut corrupt = before.clone();
        corrupt.checkpoint_hash.push_str("-corrupt");

        let mut candidates = vec![after, at, corrupt];
        if let Some(foreign) = foreign {
            candidates.push(foreign);
        }
        candidates.push(before);
        let receipt = runtime.retract_position(log, *target, &candidates)?;
        assert_eq!(receipt.strategy, RetractionStrategy::BoundedReplay);
        assert_eq!(
            receipt.outcome,
            RetractOutcome::RequiresRecompute {
                from_interval: LogInterval::new(before_position, through)?
            }
        );
        assert_eq!(receipt.replay.applied, 1);
        assert_eq!(receipt.replay.retracted, 1);
        assert_eq!(receipt.replay.quarantined, 0);
        retracted.insert(*target);
    }
    runtime.apply_all(log)?;
    assert_eq!(runtime.retracted_positions(), &retracted);
    assert_eq!(runtime.retraction_receipts().len(), retracted.len());
    Ok(runtime)
}

fn round_trip_runtime<F: Fold + Default>(
    runtime: FoldRuntime<F>,
    log: &PhysicalLog,
) -> Result<FoldRuntime<F>, Box<dyn Error>> {
    let expected_positions = runtime.retracted_positions().clone();
    let expected_receipts = runtime.retraction_receipts().clone();
    let encoded = serde_json::to_vec(&runtime.checkpoint(log)?)?;
    let checkpoint = serde_json::from_slice(&encoded)?;
    let mut restored = FoldRuntime::for_log(F::default(), log)?;
    restored.restore_or_rebuild(&checkpoint, log)?;
    assert_eq!(restored.retracted_positions(), &expected_positions);
    assert_eq!(restored.retraction_receipts(), &expected_receipts);
    Ok(restored)
}

fn normalized_facts(
    runtime: &FoldRuntime<FactsFold>,
    entities: &[String],
) -> Result<Vec<(String, String, String)>, Box<dyn Error>> {
    let mut facts = Vec::new();
    for entity in entities {
        let value = runtime.query(&json!({"entity": entity}), runtime.current_position())?;
        for tuple in serde_json::from_value::<Vec<FactTuple>>(value)? {
            facts.push((
                tuple.entity,
                tuple.attribute,
                serde_json::to_string(&tuple.value)?,
            ));
        }
    }
    facts.sort();
    Ok(facts)
}

fn graph_observation(
    runtime: &FoldRuntime<GraphFold>,
    entities: &[String],
) -> Result<BTreeMap<String, Value>, Box<dyn Error>> {
    let mut values = BTreeMap::new();
    for entity in entities {
        let kind = if entity.starts_with("edge:") {
            "edge"
        } else {
            "node"
        };
        values.insert(
            entity.clone(),
            runtime.query(
                &json!({"kind": kind, "id": entity}),
                runtime.current_position(),
            )?,
        );
    }
    Ok(values)
}

fn algebra_observation(
    runtime: &mut FoldRuntime<AlgebraFold>,
    log: &PhysicalLog,
) -> Result<BTreeMap<String, Vec<String>>, Box<dyn Error>> {
    let mut values = BTreeMap::new();
    for source in ["node:a", "node:b", "node:c", "node:d", "node:e"] {
        let answer = decode_reachability(runtime.upquery(
            log,
            &json!({
                "kind": "reachable",
                "edge_type": "LINKS",
                "sources": [source]
            }),
            runtime.current_position(),
        )?)?;
        values.insert(source.to_string(), answer.reachable);
    }
    Ok(values)
}

fn assert_bounded_receipt(receipt: &RetractionReceipt) {
    assert!(receipt.newly_retracted);
    assert_eq!(receipt.position, TARGET);
    assert_eq!(receipt.strategy, RetractionStrategy::BoundedReplay);
    assert_eq!(
        receipt.outcome,
        RetractOutcome::RequiresRecompute {
            from_interval: LogInterval {
                from_exclusive: LogPosition(5),
                through: LogPosition(7),
            }
        }
    );
    assert_eq!(receipt.replay.applied, 1);
    assert_eq!(receipt.replay.retracted, 1);
    assert_eq!(receipt.replay.through, LogPosition(7));
}

fn current_fact_value(
    runtime: &FoldRuntime<FactsFold>,
    _log: &PhysicalLog,
    entity: &str,
    attribute: &str,
) -> Result<Vec<Value>, Box<dyn Error>> {
    let position = runtime.current_position();
    let value = runtime.query(&json!({"entity": entity, "attribute": attribute}), position)?;
    let mut values = serde_json::from_value::<Vec<FactTuple>>(value)?
        .into_iter()
        .map(|tuple| tuple.value)
        .collect::<Vec<_>>();
    values.sort_by_key(Value::to_string);
    Ok(values)
}

fn decode_reachability(value: Value) -> Result<AlgebraReachability, serde_json::Error> {
    serde_json::from_value(value)
}

fn decode_checkout(value: Value) -> Result<HistoryCheckout, Box<dyn Error>> {
    serde_json::from_value::<Option<HistoryCheckout>>(value)?
        .ok_or_else(|| "History fold produced no checkout".into())
}

fn build_log(without_target: bool) -> Result<LogFixture, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    let batches = fixture_batches();
    for (index, batch) in batches.into_iter().enumerate() {
        let position = index as u64 + 1;
        if without_target && position == TARGET.0 {
            continue;
        }
        store.commit_batch(batch)?;
    }
    drop(store);
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    Ok(LogFixture {
        _directory: directory,
        log,
    })
}

fn build_generated_log(
    seed: u64,
    skipped: &BTreeSet<LogPosition>,
) -> Result<LogFixture, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    for (index, batch) in generated_batches(seed).into_iter().enumerate() {
        let position = LogPosition(index as u64 + 1);
        if !skipped.contains(&position) {
            store.commit_batch(batch)?;
        }
    }
    drop(store);
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    Ok(LogFixture {
        _directory: directory,
        log,
    })
}

fn generated_targets(seed: u64) -> Vec<LogPosition> {
    vec![
        LogPosition(4 + seed % 2),
        LogPosition(9 + (seed / 2) % 2),
        LogPosition(14 + (seed / 4) % 2),
    ]
}

fn generated_entity_ids() -> Vec<String> {
    let mut ids = ["node:a", "node:b", "node:c", "node:d", "node:e"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    ids.extend((2..=20).map(|position| format!("edge:generated:{position}")));
    ids
}

fn generated_batches(seed: u64) -> Vec<GraphMutationBatch> {
    let nodes = ["node:a", "node:b", "node:c", "node:d", "node:e"];
    let mut batches = vec![GraphMutationBatch::new(nodes.into_iter().map(|id| {
        GraphMutation::NodeUpsert(NodeRecord::new(
            id,
            ["GeneratedRetractionFixture"],
            json!({"seed": seed}),
        ))
    }))];
    let mut state = seed.wrapping_add(1);
    for position in 2_u64..=20 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let from_index = (state as usize) % nodes.len();
        state = state.rotate_left(17) ^ position;
        let mut to_index = (state as usize) % nodes.len();
        if to_index == from_index {
            to_index = (to_index + 1) % nodes.len();
        }
        batches.push(GraphMutationBatch::new([GraphMutation::EdgeUpsert(
            EdgeRecord::new(
                format!("edge:generated:{position}"),
                nodes[from_index],
                "LINKS",
                nodes[to_index],
                json!({"seed": seed, "position": position}),
            ),
        )]));
    }
    batches
}

fn fixture_batches() -> Vec<GraphMutationBatch> {
    vec![
        GraphMutationBatch::new([
            GraphMutation::NodeUpsert(node("node:a", 1)),
            GraphMutation::NodeUpsert(node("node:b", 1)),
            GraphMutation::NodeUpsert(node("node:c", 1)),
        ]),
        GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge(
            "edge:e", "node:a", "node:b", 1,
        ))]),
        GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge(
            "edge:f", "node:b", "node:c", 1,
        ))]),
        GraphMutationBatch::new([GraphMutation::NodeUpsert(node("node:c", 2))]),
        GraphMutationBatch::new([GraphMutation::EdgeUpsert(edge(
            "edge:e", "node:a", "node:b", 2,
        ))]),
        GraphMutationBatch::new([GraphMutation::EdgeDelete("edge:e".to_string())]),
        GraphMutationBatch::new([GraphMutation::NodeUpsert(node("node:a", 2))]),
    ]
}

fn node(id: &str, revision: u64) -> NodeRecord {
    NodeRecord::new(id, ["RetractionFixture"], json!({"revision": revision}))
}

fn edge(id: &str, from: &str, to: &str, revision: u64) -> EdgeRecord {
    EdgeRecord::new(id, from, "LINKS", to, json!({"revision": revision}))
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:fold-retraction-differential").unwrap()
}
