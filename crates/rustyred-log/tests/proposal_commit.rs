use serde_json::json;
use std::error::Error;
use theorem_oplog::{
    AccessPath, CommitOutcome, IntentKind, LogIntent, LogPosition, OpDigest, PhysicalLog, Proposal,
    ProposalCommitter, ReadSet, RefusalCode, WriteSet,
};

#[test]
fn conflicting_proposals_yield_one_commit_and_typed_refusal() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    let first = proposal(0, "first", entity("note:1"), attribute("note:1", "body"))?;
    let second = proposal(
        0,
        "second",
        attribute("note:1", "body"),
        attribute("note:1", "title"),
    )?;

    let CommitOutcome::Committed(first) = committer.commit(first)? else {
        panic!("first proposal must commit");
    };
    assert_eq!(first.deployment_id, deployment_id());
    assert_eq!(first.append.position, LogPosition(1));
    assert!(!first.append.replayed);

    let CommitOutcome::Refused(refusal) = committer.commit(second)? else {
        panic!("second proposal must be refused");
    };
    assert_eq!(refusal.deployment_id, deployment_id());
    assert_eq!(refusal.code, RefusalCode::OccConflict);
    assert_eq!(serde_json::to_value(refusal.code)?, json!("occ_conflict"));
    assert_eq!(refusal.conflicting_positions, vec![LogPosition(1)]);
    assert_eq!(committer.log().len(), 1);
    Ok(())
}

#[test]
fn disjoint_history_does_not_conflict() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    let first = proposal(0, "first", entity("note:1"), entity("note:1"))?;
    let second = proposal(0, "second", entity("note:2"), entity("note:2"))?;

    assert!(matches!(
        committer.commit(first)?,
        CommitOutcome::Committed(_)
    ));
    let CommitOutcome::Committed(second) = committer.commit(second)? else {
        panic!("disjoint proposal must commit");
    };
    assert_eq!(second.append.position, LogPosition(2));
    assert_eq!(committer.log().verify()?.records, 2);
    Ok(())
}

#[test]
fn duplicate_digest_replays_original_receipt_without_append() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    let original = proposal(0, "same", entity("note:1"), entity("note:1"))?;
    let duplicate = original.clone();

    let CommitOutcome::Committed(original) = committer.commit(original)? else {
        panic!("original must commit");
    };
    let CommitOutcome::Committed(replayed) = committer.commit(duplicate)? else {
        panic!("duplicate must replay");
    };

    assert_eq!(original.append.position, replayed.append.position);
    assert_eq!(original.append.record_id, replayed.append.record_id);
    assert_eq!(original.append.record_hash, replayed.append.record_hash);
    assert!(replayed.append.replayed);
    assert_eq!(committer.log().len(), 1);
    Ok(())
}

#[test]
fn duplicate_digest_with_different_content_is_rejected() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    let original = proposal(0, "same", entity("note:1"), entity("note:1"))?;
    let mut changed = proposal(0, "different", entity("note:2"), entity("note:2"))?;
    changed.op_digest = original.op_digest.clone();
    assert!(matches!(
        committer.commit(original)?,
        CommitOutcome::Committed(_)
    ));

    let error = committer.commit(changed).unwrap_err();
    assert!(error.to_string().contains("reused with different intent"));
    assert_eq!(committer.log().len(), 1);
    Ok(())
}

#[test]
fn deterministic_retry_rebuilds_from_latest_position() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    assert!(matches!(
        committer.commit(proposal(0, "winner", entity("note:1"), entity("note:1"))?)?,
        CommitOutcome::Committed(_)
    ));

    let stale = proposal(
        0,
        "retry",
        attribute("note:1", "body"),
        attribute("note:1", "body"),
    )?;
    let mut reruns = 0;
    let CommitOutcome::Committed(retried) =
        committer.commit_with_retry(stale, 1, |latest, refusal| {
            reruns += 1;
            assert_eq!(refusal.code, RefusalCode::OccConflict);
            proposal(
                latest.0,
                "retry",
                attribute("note:1", "body"),
                attribute("note:1", "body"),
            )
            .map_err(|error| panic!("fixture proposal failed: {error}"))
        })?
    else {
        panic!("deterministic retry must commit");
    };
    assert_eq!(reruns, 1);
    assert_eq!(retried.attempts, 2);
    assert_eq!(retried.append.position, LogPosition(2));
    Ok(())
}

#[test]
fn typed_boundary_rejects_generic_crud_diffs() {
    let kind = IntentKind::new("graph", "node_upsert", 1).unwrap_err();
    assert!(kind.to_string().contains("CRUD-diff intent kind"));

    let kind = IntentKind::new("theorem", "mutation", 1).unwrap();
    let payload = LogIntent::new(
        kind,
        json!({"mutations": [{"op": "edge_delete", "id": "edge:1"}]}),
    )
    .unwrap_err();
    assert!(payload.to_string().contains("CRUD-diff payload"));

    let payload = LogIntent::new(
        IntentKind::new("theorem", "mutation", 1).unwrap(),
        json!({"op": "node_upsert", "id": "note:1"}),
    )
    .unwrap_err();
    assert!(payload.to_string().contains("CRUD-diff payload"));

    let payload = LogIntent::new(
        IntentKind::new("theorem", "mutation", 1).unwrap(),
        json!({"mutations": [{"op": " node-upsert ", "id": "note:1"}]}),
    )
    .unwrap_err();
    assert!(payload.to_string().contains("CRUD-diff payload"));

    LogIntent::new(
        IntentKind::new("theorem", "typed_semantic_change", 1).unwrap(),
        json!({
            "record": {
                "properties": {"op": "node_upsert"}
            }
        }),
    )
    .expect("arbitrary nested schemaless data must not be parsed as a CRUD envelope");

    let noncanonical: IntentKind = serde_json::from_value(json!({
        "namespace": " theorem",
        "name": "block_update",
        "version": 1
    }))
    .unwrap();
    let payload = LogIntent::new(noncanonical, json!({"value": "unsafe"})).unwrap_err();
    assert!(payload.to_string().contains("surrounding whitespace"));
}

#[test]
fn a_physical_log_has_exactly_one_process_writer() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let first = PhysicalLog::open(directory.path(), deployment_id())?;
    let error = PhysicalLog::open(directory.path(), deployment_id()).unwrap_err();
    assert!(error.to_string().contains("already has a writer"));
    drop(first);
    PhysicalLog::open(directory.path(), deployment_id())?;
    Ok(())
}

#[test]
fn position_and_record_identity_are_distinct_and_prefixes_are_pinned() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let mut committer =
        ProposalCommitter::new(PhysicalLog::open(directory.path(), deployment_id())?);
    let CommitOutcome::Committed(first) =
        committer.commit(proposal(0, "first", entity("note:1"), entity("note:1"))?)?
    else {
        panic!("first must commit");
    };
    let CommitOutcome::Committed(second) =
        committer.commit(proposal(1, "second", entity("note:2"), entity("note:2"))?)?
    else {
        panic!("second must commit");
    };
    assert_eq!(first.append.position, LogPosition(1));
    assert_eq!(second.append.position, LogPosition(2));
    assert_ne!(first.append.record_id, second.append.record_id);
    assert_ne!(
        committer.log().pinned_prefix(LogPosition(1))?.content_hash,
        committer.log().pinned_prefix(LogPosition(2))?.content_hash
    );
    Ok(())
}

fn proposal(
    begin_position: u64,
    digest: &str,
    read: AccessPath,
    write: AccessPath,
) -> Result<Proposal, Box<dyn Error>> {
    Ok(Proposal::new(
        LogPosition(begin_position),
        ReadSet::new([read])?,
        WriteSet::new([write])?,
        OpDigest::from_bytes(digest),
        LogIntent::new(
            IntentKind::new("theorem", "block_update", 1)?,
            json!({"value": digest}),
        )?,
    ))
}

fn entity(entity: &str) -> AccessPath {
    AccessPath::Entity {
        entity: entity.to_string(),
    }
}

fn attribute(entity: &str, attribute: &str) -> AccessPath {
    AccessPath::Attribute {
        entity: entity.to_string(),
        attribute: attribute.to_string(),
    }
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:proposal-commit").unwrap()
}
