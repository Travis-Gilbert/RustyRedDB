use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use rustyred_thg_core::{FoldStorage, FoldStorageMutation};

const CHILD_FLAG: &str = "THEOREM_FOLD_FLOOR_CRASH_CHILD";
const CHILD_PATH: &str = "THEOREM_FOLD_FLOOR_CRASH_PATH";
const CHILD_KIND: &str = "THEOREM_FOLD_FLOOR_CRASH_KIND";
const MID_BATCH_CHILD_FLAG: &str = "THEOREM_FOLD_FLOOR_MID_BATCH_CHILD";
const INTERRUPTED_BATCH_SIZE: usize = 256;

#[test]
fn acknowledged_batch_survives_abrupt_process_exit() -> Result<(), Box<dyn Error>> {
    if std::env::var_os(CHILD_FLAG).is_some() {
        let path = PathBuf::from(std::env::var(CHILD_PATH)?);
        let kind = FloorKind::parse(&std::env::var(CHILD_KIND)?)?;
        let storage = open_floor(kind, &path)?;
        let receipt = storage.write_batch(&[
            FoldStorageMutation::put("facts", b"node:1".to_vec(), b"first".to_vec()),
            FoldStorageMutation::put("facts", b"node:2".to_vec(), b"second".to_vec()),
        ])?;
        assert!(receipt.immediately_durable);
        std::process::abort();
    }

    for kind in enabled_floors() {
        let directory = tempfile::tempdir()?;
        let status = Command::new(std::env::current_exe()?)
            .arg("--exact")
            .arg("acknowledged_batch_survives_abrupt_process_exit")
            .arg("--nocapture")
            .env(CHILD_FLAG, "1")
            .env(CHILD_PATH, directory.path())
            .env(CHILD_KIND, kind.name())
            .status()?;
        assert!(!status.success(), "{} child did not abort", kind.name());

        let reopened = open_floor(kind, directory.path())?;
        assert_eq!(reopened.get("facts", b"node:1")?, Some(b"first".to_vec()));
        assert_eq!(reopened.get("facts", b"node:2")?, Some(b"second".to_vec()));
        let entries = reopened.scan_prefix("facts", b"node:", 8)?;
        assert_eq!(entries.len(), 2);
    }
    Ok(())
}

#[test]
fn crash_during_batch_never_exposes_partial_state() -> Result<(), Box<dyn Error>> {
    if std::env::var_os(MID_BATCH_CHILD_FLAG).is_some() {
        let path = PathBuf::from(std::env::var(CHILD_PATH)?);
        let kind = FloorKind::parse(&std::env::var(CHILD_KIND)?)?;
        let storage = open_floor(kind, &path)?;
        storage.write_batch(&[FoldStorageMutation::put(
            "crash",
            b"acknowledged".to_vec(),
            b"seed".to_vec(),
        )])?;
        println!("ACK");
        std::io::stdout().flush()?;

        let interrupted = (0..INTERRUPTED_BATCH_SIZE)
            .map(|index| {
                FoldStorageMutation::put(
                    "crash",
                    format!("batch:{index:04}").into_bytes(),
                    vec![index as u8; 4096],
                )
            })
            .collect::<Vec<_>>();
        println!("START");
        std::io::stdout().flush()?;
        storage.write_batch(&interrupted)?;
        println!("COMMIT");
        std::io::stdout().flush()?;
        thread::sleep(Duration::from_secs(5));
        return Ok(());
    }

    for kind in enabled_floors() {
        for delay_ms in [0, 1, 5, 20] {
            let directory = tempfile::tempdir()?;
            let mut child = Command::new(std::env::current_exe()?)
                .arg("--exact")
                .arg("crash_during_batch_never_exposes_partial_state")
                .arg("--nocapture")
                .env(MID_BATCH_CHILD_FLAG, "1")
                .env(CHILD_PATH, directory.path())
                .env(CHILD_KIND, kind.name())
                .stdout(Stdio::piped())
                .spawn()?;
            let mut output = BufReader::new(child.stdout.take().expect("piped child stdout"));
            wait_for_marker(&mut output, "ACK")?;
            wait_for_marker(&mut output, "START")?;

            thread::sleep(Duration::from_millis(delay_ms));
            child.kill()?;
            let status = child.wait()?;
            drop(output);
            assert!(!status.success(), "{} child was not killed", kind.name());

            let reopened = open_floor(kind, directory.path())?;
            assert_eq!(
                reopened.get("crash", b"acknowledged")?,
                Some(b"seed".to_vec()),
                "{} lost the acknowledged seed at {delay_ms}ms",
                kind.name()
            );
            let visible = reopened.scan_prefix("crash", b"batch:", INTERRUPTED_BATCH_SIZE + 1)?;
            assert!(
                visible.is_empty() || visible.len() == INTERRUPTED_BATCH_SIZE,
                "{} exposed {} of {} interrupted mutations at {delay_ms}ms",
                kind.name(),
                visible.len(),
                INTERRUPTED_BATCH_SIZE
            );
            if visible.len() == INTERRUPTED_BATCH_SIZE {
                for (index, entry) in visible.into_iter().enumerate() {
                    assert_eq!(entry.key, format!("batch:{index:04}").into_bytes());
                    assert_eq!(entry.value, vec![index as u8; 4096]);
                }
            }
        }
    }
    Ok(())
}

fn wait_for_marker(
    output: &mut impl BufRead,
    expected: &'static str,
) -> Result<(), Box<dyn Error>> {
    let mut line = String::new();
    loop {
        line.clear();
        if output.read_line(&mut line)? == 0 {
            return Err(format!("child exited before {expected} marker").into());
        }
        if line.trim() == expected {
            return Ok(());
        }
    }
}

#[test]
fn floor_format_reopens_after_overwrite_and_delete() -> Result<(), Box<dyn Error>> {
    for kind in enabled_floors() {
        let directory = tempfile::tempdir()?;
        {
            let storage = open_floor(kind, directory.path())?;
            storage.write_batch(&[
                FoldStorageMutation::put("graph", b"node:1".to_vec(), b"old".to_vec()),
                FoldStorageMutation::put("graph", b"node:2".to_vec(), b"remove".to_vec()),
            ])?;
            storage.write_batch(&[
                FoldStorageMutation::put("graph", b"node:1".to_vec(), b"new".to_vec()),
                FoldStorageMutation::delete("graph", b"node:2".to_vec()),
            ])?;
        }
        let reopened = open_floor(kind, directory.path())?;
        assert_eq!(reopened.get("graph", b"node:1")?, Some(b"new".to_vec()));
        assert_eq!(reopened.get("graph", b"node:2")?, None);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum FloorKind {
    #[cfg(feature = "redb-floor")]
    Redb,
    #[cfg(feature = "fjall-floor")]
    Fjall,
}

impl FloorKind {
    fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "redb-floor")]
            Self::Redb => "redb",
            #[cfg(feature = "fjall-floor")]
            Self::Fjall => "fjall",
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            #[cfg(feature = "redb-floor")]
            "redb" => Ok(Self::Redb),
            #[cfg(feature = "fjall-floor")]
            "fjall" => Ok(Self::Fjall),
            _ => Err(format!("floor {value} is not enabled").into()),
        }
    }
}

fn enabled_floors() -> Vec<FloorKind> {
    vec![
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb,
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall,
    ]
}

fn open_floor(kind: FloorKind, path: &Path) -> Result<Box<dyn FoldStorage>, Box<dyn Error>> {
    match kind {
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb => Ok(Box::new(rustyred_thg_core::RedbFoldStorage::open(path)?)),
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall => Ok(Box::new(rustyred_thg_core::FjallFoldStorage::open(path)?)),
    }
}
