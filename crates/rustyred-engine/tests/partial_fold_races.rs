use std::error::Error;
use std::sync::Arc;

use rustyred_thg_core::query_receipt::PinnedReadSet;
use rustyred_thg_core::{
    ColdIndex, FoldStorage, InMemoryColdIndex, MutationOrder, PartialFoldRuntime, PartialOperator,
    PartialResidency, PositionedFoldIndex, ReconstructionStart, RedbFoldStorage, UpdateDisposition,
};
use theorem_oplog::{AccessPath, LogPosition, WriteSet};

#[test]
fn reconstruction_coalesces_and_drains_later_updates_in_total_order() -> Result<(), Box<dyn Error>>
{
    let (_directory, mut runtime) = runtime(64 * 1024)?;
    let operator = PartialOperator::GraphNode;
    runtime.apply_update(
        operator,
        b"node:1",
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"v1".to_vec()),
    )?;
    let (ticket, snapshot) =
        started(runtime.begin_reconstruction(operator, b"node:1", LogPosition(1))?);
    assert_eq!(
        runtime.begin_reconstruction(operator, b"node:1", LogPosition(1))?,
        ReconstructionStart::Coalesced(ticket.clone())
    );
    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:1",
            MutationOrder::new(LogPosition(3), 0)?,
            Some(b"v3.0".to_vec()),
        )?,
        UpdateDisposition::Buffered
    );
    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:1",
            MutationOrder::new(LogPosition(2), 0)?,
            Some(b"v2".to_vec()),
        )?,
        UpdateDisposition::Buffered
    );
    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:1",
            MutationOrder::new(LogPosition(3), 1)?,
            Some(b"v3.1".to_vec()),
        )?,
        UpdateDisposition::Buffered
    );
    let result = runtime.complete_reconstruction(&ticket, snapshot)?;
    assert_eq!(result.payload, Some(b"v3.1".to_vec()));
    assert_eq!(result.through, LogPosition(3));
    assert!(result.retained);

    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:1",
            MutationOrder::new(LogPosition(4), 0)?,
            Some(b"v4.0".to_vec()),
        )?,
        UpdateDisposition::AppliedResident
    );
    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:1",
            MutationOrder::new(LogPosition(4), 1)?,
            Some(b"v4.1".to_vec()),
        )?,
        UpdateDisposition::AppliedResident
    );
    assert_eq!(
        runtime
            .resident(operator, b"node:1")?
            .expect("same-position ordinal remains resident")
            .payload,
        Some(b"v4.1".to_vec())
    );
    Ok(())
}

#[test]
fn stale_response_cannot_replace_or_remove_a_new_generation() -> Result<(), Box<dyn Error>> {
    let (_directory, mut runtime) = runtime(64 * 1024)?;
    let operator = PartialOperator::FactsCurrentEntity;
    runtime.apply_update(
        operator,
        b"entity:1",
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"old".to_vec()),
    )?;
    let (old_ticket, old_snapshot) =
        started(runtime.begin_reconstruction(operator, b"entity:1", LogPosition(1))?);
    runtime.evict(operator, b"entity:1")?;
    runtime.apply_update(
        operator,
        b"entity:1",
        MutationOrder::new(LogPosition(2), 0)?,
        Some(b"new".to_vec()),
    )?;
    let (new_ticket, new_snapshot) =
        started(runtime.begin_reconstruction(operator, b"entity:1", LogPosition(2))?);
    let error = runtime
        .complete_reconstruction(&old_ticket, old_snapshot)
        .unwrap_err();
    assert_eq!(error.code, "partial_fold_stale_response");
    assert!(matches!(
        runtime.residency(operator, b"entity:1")?,
        PartialResidency::Reconstructing { generation, .. } if generation == new_ticket.generation
    ));
    let current = runtime.complete_reconstruction(&new_ticket, new_snapshot)?;
    assert_eq!(current.payload, Some(b"new".to_vec()));
    Ok(())
}

#[test]
fn evicted_dependency_drops_update_and_evicts_descendants_once() -> Result<(), Box<dyn Error>> {
    let (_directory, mut runtime) = runtime(64 * 1024)?;
    let parent = PartialOperator::GraphNode;
    let child = PartialOperator::AlgebraRegion;
    runtime.apply_update(
        child,
        b"region:1",
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"matrix".to_vec()),
    )?;
    runtime.read_at(child, b"region:1", LogPosition(1))?;
    runtime.register_descendant(parent, b"node:1", child, b"region:1")?;
    runtime.register_descendant(child, b"region:1", parent, b"node:1")?;

    let disposition = runtime.apply_update(
        parent,
        b"node:1",
        MutationOrder::new(LogPosition(2), 0)?,
        Some(b"node".to_vec()),
    )?;
    let UpdateDisposition::DroppedEvicted { notices } = disposition else {
        panic!("update on an evicted dependency was not dropped");
    };
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].operator, child);
    assert_eq!(
        runtime.residency(child, b"region:1")?,
        PartialResidency::Evicted
    );
    assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    Ok(())
}

#[test]
fn oversized_result_is_returned_without_residency_and_budget_evicts_coldest(
) -> Result<(), Box<dyn Error>> {
    let (_directory, mut runtime) = runtime(1_200)?;
    for index in 0..4_u64 {
        let key = format!("node:{index}");
        runtime.apply_update(
            PartialOperator::GraphNode,
            key.as_bytes(),
            MutationOrder::new(LogPosition(index + 1), 0)?,
            Some(vec![index as u8; 300]),
        )?;
        runtime.read_at(
            PartialOperator::GraphNode,
            key.as_bytes(),
            LogPosition(index + 1),
        )?;
        assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    }
    assert!(runtime.resident_len() < 4);
    assert!(runtime.frontier_ops() > 0);

    runtime.apply_update(
        PartialOperator::GraphNode,
        b"huge",
        MutationOrder::new(LogPosition(5), 0)?,
        Some(vec![9; 4_096]),
    )?;
    let huge = runtime.read_at(PartialOperator::GraphNode, b"huge", LogPosition(5))?;
    assert!(!huge.retained);
    assert_eq!(huge.payload.as_ref().map(Vec::len), Some(4_096));
    assert_eq!(
        runtime.residency(PartialOperator::GraphNode, b"huge")?,
        PartialResidency::Evicted
    );
    assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    Ok(())
}

#[test]
fn represented_older_update_preserves_residency_and_overlap_increments_hotness(
) -> Result<(), Box<dyn Error>> {
    let (_directory, mut runtime) = runtime(64 * 1024)?;
    let operator = PartialOperator::GraphNode;
    runtime.apply_update(
        operator,
        b"node:hot",
        MutationOrder::new(LogPosition(3), 0)?,
        Some(b"current".to_vec()),
    )?;
    runtime.read_at(operator, b"node:hot", LogPosition(3))?;

    assert_eq!(
        runtime.apply_update(
            operator,
            b"node:hot",
            MutationOrder::new(LogPosition(2), 0)?,
            Some(b"older".to_vec()),
        )?,
        UpdateDisposition::AlreadyRepresented
    );
    assert_eq!(
        runtime
            .resident(operator, b"node:hot")?
            .expect("resident survives refusal")
            .payload,
        Some(b"current".to_vec())
    );

    let reads = PinnedReadSet::point_node(LogPosition(3), "hot")?;
    let footprint = WriteSet::new([AccessPath::Entity {
        entity: "node:hot".to_owned(),
    }])?;
    let trace = runtime.observe_hotness(operator, b"node:hot", &reads, &footprint)?;
    assert!(trace.overlaps);
    assert_eq!(runtime.hotness_count(operator, b"node:hot")?, 1);
    assert!(runtime.accounting().hotness > 0);
    assert!(runtime.resident_bytes() <= runtime.budget_bytes());
    Ok(())
}

#[test]
fn long_keys_charge_every_retained_copy_before_admission() -> Result<(), Box<dyn Error>> {
    let key = vec![b'k'; 4_096];
    let operator = PartialOperator::HistoryLogicalRecord;

    let (_small_directory, mut small) = runtime(6_000)?;
    small.apply_update(
        operator,
        &key,
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"value".to_vec()),
    )?;
    let transient = small.read_at(operator, &key, LogPosition(1))?;
    assert!(!transient.retained);
    assert_eq!(transient.payload, Some(b"value".to_vec()));
    assert_eq!(small.residency(operator, &key)?, PartialResidency::Evicted);
    assert_eq!(small.resident_bytes(), 0);
    assert_eq!(small.accounting().total(), 0);

    let (_large_directory, mut large) = runtime(32 * 1024)?;
    large.apply_update(
        operator,
        &key,
        MutationOrder::new(LogPosition(1), 0)?,
        Some(b"value".to_vec()),
    )?;
    assert!(large.read_at(operator, &key, LogPosition(1))?.retained);
    let accounting = large.accounting();
    assert!(accounting.key_capacity >= key.len() * 3);
    assert!(accounting.frontier_identifiers > 0);
    assert_eq!(accounting.total(), large.resident_bytes());
    assert!(accounting.total() <= large.budget_bytes());

    let child_key = vec![b'c'; 2_048];
    large.register_descendant(operator, &key, PartialOperator::GraphNode, &child_key)?;
    let accounting = large.accounting();
    assert!(accounting.dependency_edges >= (key.len() + child_key.len()) * 2);
    assert_eq!(accounting.total(), large.resident_bytes());
    assert!(accounting.total() <= large.budget_bytes());
    Ok(())
}

fn runtime(budget: usize) -> Result<(tempfile::TempDir, PartialFoldRuntime), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(InMemoryColdIndex::new());
    let runtime = PartialFoldRuntime::new(PositionedFoldIndex::new(storage, cold), budget)?;
    Ok((directory, runtime))
}

fn started(
    start: ReconstructionStart,
) -> (
    rustyred_thg_core::ReconstructionTicket,
    rustyred_thg_core::PositionedSnapshot,
) {
    match start {
        ReconstructionStart::Started { ticket, snapshot } => (ticket, snapshot),
        other => panic!("expected reconstruction start, found {other:?}"),
    }
}
