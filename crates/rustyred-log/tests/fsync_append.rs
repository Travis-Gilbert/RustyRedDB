use serde_json::json;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;
use theorem_oplog::{
    AccessPath, IntentKind, LogIntent, LogPosition, OpDigest, PhysicalLog, WriteSet,
};

const CHILD_FLAG: &str = "THEOREM_OPLOG_FSYNC_CHILD";
const CHILD_PATH: &str = "THEOREM_OPLOG_FSYNC_PATH";

#[test]
fn acknowledged_append_survives_abrupt_process_exit() -> Result<(), Box<dyn Error>> {
    if std::env::var_os(CHILD_FLAG).is_some() {
        let path = PathBuf::from(std::env::var(CHILD_PATH)?);
        let mut log = PhysicalLog::open(path, deployment_id())?;
        let receipt = log.append(
            LogIntent::new(
                IntentKind::new("theorem", "fsync_probe", 1)?,
                json!({"value": "durable"}),
            )?,
            WriteSet::new([AccessPath::Entity {
                entity: "probe:durable".to_string(),
            }])?,
            OpDigest::from_bytes("fsync-probe"),
        )?;
        assert_eq!(receipt.position, LogPosition(1));
        std::process::abort();
    }

    let directory = tempfile::tempdir()?;
    let status = Command::new(std::env::current_exe()?)
        .arg("--exact")
        .arg("acknowledged_append_survives_abrupt_process_exit")
        .arg("--nocapture")
        .env(CHILD_FLAG, "1")
        .env(CHILD_PATH, directory.path())
        .status()?;
    assert!(!status.success(), "fsync child did not abort");

    let reopened = PhysicalLog::open(directory.path(), deployment_id())?;
    let report = reopened.verify()?;
    assert_eq!(report.records, 1);
    assert_eq!(report.through, LogPosition(1));
    Ok(())
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:fsync-append").unwrap()
}
