use serde::Deserialize;
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use theorem_oplog::{
    canonical_record_hash, AccessPath, BridgeTransport, CommitAuthority, CommitOutcome,
    ContentHash, DeploymentId, EntryOrigin, EntryProvenance, IntentKind, LogIntent, LogPosition,
    LogRecord, OpDigest, PhysicalLog, Proposal, ProposalCommitter, ReadSet, VerificationState,
    WriteSet, WriterId, PHYSICAL_LOG_PROTOCOL_V2, PHYSICAL_LOG_PROTOCOL_V3,
};
use ulid::Ulid;

const HEADER_BYTES: usize = 56;
const FRAME_CHECKSUM_BYTES: usize = 32;
const V2_FIXTURE: &str = include_str!("fixtures/physical-log-v2/record.json");

#[derive(Deserialize)]
struct V2Fixture {
    deployment_id: String,
    log_id: String,
    expected_payload_hash: String,
    expected_record_hash: String,
    record: Value,
}

#[test]
fn v2_fixture_reopens_and_accepts_a_hash_linked_v3_tail() -> Result<(), Box<dyn Error>> {
    let fixture = v2_fixture()?;
    let record: LogRecord = serde_json::from_value(fixture.record.clone())?;
    let payload_hash = ContentHash::from_bytes(serde_json::to_vec(&record.intent.payload)?);
    assert_eq!(
        payload_hash.as_str(),
        fixture.expected_payload_hash,
        "fixture payload hash changed"
    );
    assert_eq!(record.payload_hash, payload_hash);
    let record_hash = canonical_record_hash(&record)?;
    assert_eq!(
        record_hash.as_str(),
        fixture.expected_record_hash,
        "v2 canonical preimage changed"
    );
    assert_eq!(record.record_hash, record_hash);
    assert!(record.provenance.is_explicit_legacy());

    let directory = tempfile::tempdir()?;
    materialize_fixture(directory.path(), &fixture, &fixture.record)?;
    let log = PhysicalLog::open(directory.path(), fixture_deployment(&fixture)?)?;
    assert_eq!(log.records()[0].protocol_version, PHYSICAL_LOG_PROTOCOL_V2);
    let v2_hash = log.records()[0].record_hash.clone();

    let mut committer = ProposalCommitter::new(log);
    let CommitOutcome::Committed(receipt) = committer.commit_with_provenance(
        proposal("v3-tail", 1)?,
        admitted_provenance(EntryOrigin::Mutation, "head:migration"),
    )?
    else {
        panic!("v3 tail must commit after a valid v2 prefix");
    };
    assert_eq!(receipt.append.position, LogPosition(2));
    let tail = &committer.log().records()[1];
    assert_eq!(tail.protocol_version, PHYSICAL_LOG_PROTOCOL_V3);
    assert_eq!(tail.previous_record_hash.as_ref(), Some(&v2_hash));
    assert_eq!(tail.provenance.origin, EntryOrigin::Mutation);
    drop(committer);

    let reopened = PhysicalLog::open(directory.path(), fixture_deployment(&fixture)?)?;
    let report = reopened.verify()?;
    assert_eq!(report.records, 2);
    assert_eq!(report.through, LogPosition(2));
    assert_eq!(
        reopened.records()[0].protocol_version,
        PHYSICAL_LOG_PROTOCOL_V2
    );
    assert_eq!(
        reopened.records()[1].protocol_version,
        PHYSICAL_LOG_PROTOCOL_V3
    );
    Ok(())
}

#[test]
fn v3_hash_commits_every_provenance_field() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut committer = ProposalCommitter::new(log);
    let provenance = admitted_provenance(EntryOrigin::Mutation, "head:codex");
    let CommitOutcome::Committed(_) =
        committer.commit_with_provenance(proposal("field-matrix", 0)?, provenance)?
    else {
        panic!("fixture proposal must commit");
    };

    let record = committer.log().records()[0].clone();
    let baseline = canonical_record_hash(&record)?;
    assert_eq!(baseline, record.record_hash);

    let mut mutations = Vec::new();
    let mut changed = record.clone();
    changed.provenance.origin = EntryOrigin::Action;
    mutations.push(("origin", changed));

    let mut changed = record.clone();
    changed.provenance.authority = CommitAuthority::Internal;
    mutations.push(("authority", changed));

    let mut changed = record.clone();
    changed.provenance.transport = Some(BridgeTransport::Service);
    mutations.push(("transport", changed));

    let mut changed = record.clone();
    changed.provenance.verification = VerificationState::GroundingVerified;
    mutations.push(("verification", changed));

    let mut changed = record.clone();
    changed.provenance.writer = Some(WriterId::new("head:fable")?);
    mutations.push(("writer", changed));

    for (field, changed) in mutations {
        assert_ne!(
            canonical_record_hash(&changed)?,
            baseline,
            "changing provenance.{field} must change the v3 record hash"
        );
    }
    Ok(())
}

#[test]
fn provenance_tamper_survives_frame_checksum_but_fails_chain_hash() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut committer = ProposalCommitter::new(log);
    let CommitOutcome::Committed(_) = committer.commit_with_provenance(
        proposal("tamper", 0)?,
        admitted_provenance(EntryOrigin::Mutation, "head:codex"),
    )?
    else {
        panic!("fixture proposal must commit");
    };
    drop(committer);

    let path = directory.path().join("theorem.oplog");
    let bytes = fs::read(&path)?;
    let tampered = rewrite_first_frame(&bytes, |record| {
        record["provenance"]["origin"] = json!("action");
    })?;
    fs::write(&path, tampered)?;

    let error = PhysicalLog::open(directory.path(), deployment_id()).unwrap_err();
    assert!(error.to_string().contains("chain hash mismatch"));
    Ok(())
}

#[test]
fn v2_cannot_smuggle_unhashed_provenance() -> Result<(), Box<dyn Error>> {
    let fixture = v2_fixture()?;
    let mut smuggled = fixture.record.clone();
    smuggled["provenance"] =
        serde_json::to_value(admitted_provenance(EntryOrigin::Mutation, "head:smuggled"))?;

    let directory = tempfile::tempdir()?;
    materialize_fixture(directory.path(), &fixture, &smuggled)?;
    let error = PhysicalLog::open(directory.path(), fixture_deployment(&fixture)?).unwrap_err();
    assert!(error.to_string().contains("outside its canonical hash"));
    Ok(())
}

#[test]
fn v3_duplicate_digest_requires_the_same_provenance() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let mut committer = ProposalCommitter::new(log);
    let proposal = proposal("duplicate", 0)?;
    let CommitOutcome::Committed(_) = committer.commit_with_provenance(
        proposal.clone(),
        admitted_provenance(EntryOrigin::Mutation, "head:first"),
    )?
    else {
        panic!("first proposal must commit");
    };

    let error = committer
        .commit_with_provenance(
            proposal,
            admitted_provenance(EntryOrigin::Mutation, "head:second"),
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("different intent, write set, or provenance"));
    assert_eq!(committer.log().len(), 1);
    Ok(())
}

fn v2_fixture() -> Result<V2Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(V2_FIXTURE)?)
}

fn fixture_deployment(fixture: &V2Fixture) -> Result<DeploymentId, Box<dyn Error>> {
    Ok(DeploymentId::new(fixture.deployment_id.clone())?)
}

fn materialize_fixture(
    directory: &Path,
    fixture: &V2Fixture,
    record: &Value,
) -> Result<(), Box<dyn Error>> {
    let payload = serde_json::to_vec(record)?;
    let log_id = Ulid::from_str(&fixture.log_id)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"THOPLG02");
    bytes.extend_from_slice(&log_id.to_bytes());
    bytes.extend_from_slice(blake3::hash(fixture.deployment_id.as_bytes()).as_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(blake3::hash(&payload).as_bytes());
    fs::write(directory.join("theorem.oplog"), bytes)?;
    Ok(())
}

fn rewrite_first_frame(
    bytes: &[u8],
    mutate: impl FnOnce(&mut Value),
) -> Result<Vec<u8>, Box<dyn Error>> {
    let old_length = u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into()?) as usize;
    let payload_start = HEADER_BYTES + 4;
    let payload_end = payload_start + old_length;
    let old_frame_end = payload_end + FRAME_CHECKSUM_BYTES;
    let mut record: Value = serde_json::from_slice(&bytes[payload_start..payload_end])?;
    mutate(&mut record);
    let payload = serde_json::to_vec(&record)?;

    let mut rewritten = Vec::new();
    rewritten.extend_from_slice(&bytes[..HEADER_BYTES]);
    rewritten.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    rewritten.extend_from_slice(&payload);
    rewritten.extend_from_slice(blake3::hash(&payload).as_bytes());
    rewritten.extend_from_slice(&bytes[old_frame_end..]);
    Ok(rewritten)
}

fn admitted_provenance(origin: EntryOrigin, writer: &str) -> EntryProvenance {
    EntryProvenance {
        origin,
        authority: CommitAuthority::Mutation,
        transport: Some(BridgeTransport::InProcess),
        verification: VerificationState::Unverified,
        writer: Some(WriterId::new(writer).expect("fixture writer identity is valid")),
    }
}

fn proposal(digest: &str, begin_position: u64) -> Result<Proposal, Box<dyn Error>> {
    Ok(Proposal::new(
        LogPosition(begin_position),
        ReadSet::default(),
        WriteSet::new([AccessPath::Entity {
            entity: format!("note:{digest}"),
        }])?,
        OpDigest::from_bytes(digest),
        LogIntent::new(
            IntentKind::new("theorem", "provenance_fixture", 1)?,
            json!({"digest": digest}),
        )?,
    ))
}

fn deployment_id() -> DeploymentId {
    DeploymentId::new("test:provenance-commit").unwrap()
}
