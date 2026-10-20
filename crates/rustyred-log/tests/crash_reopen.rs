use serde::Deserialize;
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use theorem_oplog::{
    AccessPath, ContentHash, CrashInjection, DeploymentId, IntentKind, LogIntent, LogPosition,
    OpDigest, PhysicalLog, WriteSet, PHYSICAL_LOG_PROTOCOL_V2, PHYSICAL_LOG_PROTOCOL_V3,
};
use ulid::Ulid;

const V2_FIXTURE: &str = include_str!("fixtures/physical-log-v2/record.json");

#[derive(Deserialize)]
struct V2Fixture {
    deployment_id: String,
    log_id: String,
    expected_record_hash: String,
    record: Value,
}

#[test]
#[ignore = "subprocess crash matrix"]
fn crash_matrix_recovers_only_verified_prefixes() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("THEOREM_RUN_CRASH_MATRIX").is_none() {
        return Ok(());
    }
    for stage in [
        "before_write",
        "after_length",
        "after_payload",
        "after_checksum_before_sync",
        "after_sync",
    ] {
        let directory = tempfile::tempdir()?;
        let status = Command::new(std::env::current_exe()?)
            .arg("--ignored")
            .arg("--exact")
            .arg("crash_child")
            .arg("--nocapture")
            .env("THEOREM_OPLOG_CRASH_CHILD", "1")
            .env("THEOREM_OPLOG_CRASH_DIRECTORY", directory.path())
            .env("THEOREM_OPLOG_CRASH_STAGE", stage)
            .status()?;
        assert!(!status.success(), "stage {stage} did not crash");

        let log = PhysicalLog::open(directory.path(), deployment_id())?;
        let report = log.verify()?;
        assert!(
            matches!(report.records, 1 | 2),
            "stage {stage} recovered {} records",
            report.records
        );
        if stage == "after_sync" {
            assert_eq!(report.records, 2, "fsynced append must survive");
        } else if stage == "before_write" {
            assert_eq!(report.records, 1, "pre-write crash must not append");
        }
        assert_eq!(report.through, LogPosition(report.records as u64));
    }
    Ok(())
}

#[test]
#[ignore = "crash-matrix child"]
fn crash_child() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("THEOREM_OPLOG_CRASH_CHILD").is_none() {
        return Ok(());
    }
    let directory = std::env::var("THEOREM_OPLOG_CRASH_DIRECTORY")?;
    let stage = match std::env::var("THEOREM_OPLOG_CRASH_STAGE")?.as_str() {
        "before_write" => CrashInjection::BeforeWrite,
        "after_length" => CrashInjection::AfterLength,
        "after_payload" => CrashInjection::AfterPayload,
        "after_checksum_before_sync" => CrashInjection::AfterChecksumBeforeSync,
        "after_sync" => CrashInjection::AfterSync,
        other => panic!("unknown crash stage {other}"),
    };
    let mut log = PhysicalLog::open(&directory, deployment_id())?;
    append(&mut log, "durable")?;
    log.set_crash_injection(Some(stage));
    append(&mut log, "crash")?;
    panic!("crash injection did not fire");
}

#[test]
#[ignore = "mixed-version subprocess crash matrix"]
fn mixed_v2_prefix_v3_crash_matrix_preserves_legacy_prefix() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("THEOREM_RUN_CRASH_MATRIX").is_none() {
        return Ok(());
    }
    let fixture: V2Fixture = serde_json::from_str(V2_FIXTURE)?;
    for stage in [
        "before_write",
        "after_length",
        "after_payload",
        "after_checksum_before_sync",
        "after_sync",
    ] {
        let directory = tempfile::tempdir()?;
        materialize_v2_fixture(directory.path(), &fixture)?;
        let status = Command::new(std::env::current_exe()?)
            .arg("--ignored")
            .arg("--exact")
            .arg("mixed_version_crash_child")
            .arg("--nocapture")
            .env("THEOREM_OPLOG_MIXED_CRASH_CHILD", "1")
            .env("THEOREM_OPLOG_CRASH_DIRECTORY", directory.path())
            .env("THEOREM_OPLOG_CRASH_STAGE", stage)
            .status()?;
        assert!(!status.success(), "mixed stage {stage} did not crash");

        let log = PhysicalLog::open(
            directory.path(),
            DeploymentId::new(fixture.deployment_id.clone())?,
        )?;
        let report = log.verify()?;
        assert!(
            matches!(report.records, 1 | 2),
            "mixed stage {stage} recovered {} records",
            report.records
        );
        assert_eq!(log.records()[0].protocol_version, PHYSICAL_LOG_PROTOCOL_V2);
        assert_eq!(
            log.records()[0].record_hash.as_str(),
            fixture.expected_record_hash
        );
        if stage == "after_sync" {
            assert_eq!(report.records, 2, "fsynced v3 tail must survive");
        } else if stage == "before_write" {
            assert_eq!(report.records, 1, "pre-write crash preserves only v2");
        }
        if let Some(tail) = log.records().get(1) {
            assert_eq!(tail.protocol_version, PHYSICAL_LOG_PROTOCOL_V3);
            assert_eq!(
                tail.previous_record_hash.as_ref(),
                Some(&log.records()[0].record_hash)
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "mixed-version crash-matrix child"]
fn mixed_version_crash_child() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("THEOREM_OPLOG_MIXED_CRASH_CHILD").is_none() {
        return Ok(());
    }
    let fixture: V2Fixture = serde_json::from_str(V2_FIXTURE)?;
    let directory = std::env::var("THEOREM_OPLOG_CRASH_DIRECTORY")?;
    let stage = crash_stage(&std::env::var("THEOREM_OPLOG_CRASH_STAGE")?);
    let mut log = PhysicalLog::open(&directory, DeploymentId::new(fixture.deployment_id)?)?;
    assert_eq!(log.len(), 1);
    assert_eq!(log.records()[0].protocol_version, PHYSICAL_LOG_PROTOCOL_V2);
    log.set_crash_injection(Some(stage));
    append(&mut log, "v3-tail")?;
    panic!("mixed-version crash injection did not fire");
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:crash-reopen").unwrap()
}

fn append(log: &mut PhysicalLog, value: &str) -> Result<(), Box<dyn Error>> {
    log.append(
        LogIntent::new(
            IntentKind::new("theorem", "block_update", 1)?,
            json!({"value": value}),
        )?,
        WriteSet::new([AccessPath::Entity {
            entity: "note:1".to_string(),
        }])?,
        OpDigest::from_bytes(value),
    )?;
    Ok(())
}

fn crash_stage(stage: &str) -> CrashInjection {
    match stage {
        "before_write" => CrashInjection::BeforeWrite,
        "after_length" => CrashInjection::AfterLength,
        "after_payload" => CrashInjection::AfterPayload,
        "after_checksum_before_sync" => CrashInjection::AfterChecksumBeforeSync,
        "after_sync" => CrashInjection::AfterSync,
        other => panic!("unknown crash stage {other}"),
    }
}

fn materialize_v2_fixture(directory: &Path, fixture: &V2Fixture) -> Result<(), Box<dyn Error>> {
    let payload = serde_json::to_vec(&fixture.record)?;
    let record_hash: ContentHash = serde_json::from_value(fixture.record["record_hash"].clone())?;
    assert_eq!(record_hash.as_str(), fixture.expected_record_hash);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"THOPLG02");
    bytes.extend_from_slice(&Ulid::from_str(&fixture.log_id)?.to_bytes());
    bytes.extend_from_slice(blake3::hash(fixture.deployment_id.as_bytes()).as_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(blake3::hash(&payload).as_bytes());
    fs::write(directory.join("theorem.oplog"), bytes)?;
    Ok(())
}
