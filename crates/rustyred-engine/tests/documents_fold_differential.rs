use rustyred_thg_core::{
    DocumentsFold, FoldError, FoldRuntime, RestoreDisposition, DOCUMENTS_INTENT_NAMESPACE,
    DOCUMENTS_INTENT_VERSION, INTENT_RECORD_CREATED,
};
use serde_json::{json, Value};
use std::error::Error;
use tempfile::TempDir;
use theorem_oplog::{
    AccessPath, IntentKind, LogIntent, LogPosition, OpDigest, PhysicalLog, WriteSet,
};

#[test]
fn documents_fold_checkpoint_plus_tail_matches_full_replay() -> Result<(), Box<dyn Error>> {
    for seed in 0_u64..8 {
        let (_directory, log) = generated_document_log(seed, 16)?;
        let latest = log.latest_position();
        let mut full = FoldRuntime::for_log(DocumentsFold::default(), &log)?;
        full.apply_all(&log)?;
        let expected = observation(&full, latest)?;

        for cut in [0, 1, latest.0 / 2, latest.0] {
            let mut prefix = FoldRuntime::for_log(DocumentsFold::default(), &log)?;
            prefix.apply_range(&log, LogPosition(cut))?;
            let checkpoint = prefix.checkpoint(&log)?;
            let mut resumed = FoldRuntime::for_log(DocumentsFold::default(), &log)?;
            assert!(matches!(
                resumed.restore_or_rebuild(&checkpoint, &log)?,
                RestoreDisposition::Restored { .. }
            ));
            assert_eq!(observation(&resumed, latest)?, expected);
            let query = json!({"view": "document", "key": "doc:0"});
            let from_query = resumed.query(&query, latest)?;
            let from_upquery = resumed.upquery(&log, &query, latest)?;
            assert_eq!(from_query["record"], from_upquery["record"]);
            let kind_query = json!({"view": "by_kind", "key": "authored"});
            assert_eq!(
                resumed.query(&kind_query, latest)?["ids"],
                resumed.upquery(&log, &kind_query, latest)?["ids"]
            );
        }
    }
    Ok(())
}

fn generated_document_log(seed: u64, count: usize) -> Result<(TempDir, PhysicalLog), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut log = PhysicalLog::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:documents-fold-differential")?,
    )?;
    let mut random = seed ^ 0x51ed_0c01_d0c5;
    for index in 0..count {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let id = format!("doc:{}", (random as usize) % 5);
        let invalid = random % 9 == 0;
        let payload = if invalid {
            json!({"id": "", "kind": "authored"})
        } else {
            json!({
                "id": id,
                "kind": if random % 2 == 0 { "authored" } else { "skill" },
                "actors": [if random % 3 == 0 { "human" } else { "agent" }],
                "collection_refs": [format!("col:{}", random % 3)],
                "created_position": 1,
                "updated_position": index as u64 + 1,
                "content_digest": format!("sha256:{seed:x}:{index}")
            })
        };
        log.append(
            LogIntent::new(
                IntentKind::new(
                    DOCUMENTS_INTENT_NAMESPACE,
                    INTENT_RECORD_CREATED,
                    DOCUMENTS_INTENT_VERSION,
                )?,
                payload,
            )?,
            WriteSet::new([AccessPath::Entity { entity: id }])?,
            OpDigest::from_bytes(format!("{seed}:{index}")),
        )?;
    }
    Ok((directory, log))
}

fn observation(
    runtime: &FoldRuntime<DocumentsFold>,
    position: LogPosition,
) -> Result<Vec<u8>, FoldError> {
    let answers = [
        json!({"view": "by_kind", "key": "authored"}),
        json!({"view": "by_kind", "key": "skill"}),
        json!({"view": "by_actor", "key": "human"}),
        json!({"view": "by_recency", "key": ""}),
        json!({"view": "document", "key": "doc:0"}),
    ]
    .into_iter()
    .map(|query| runtime.query(&query, position))
    .collect::<Result<Vec<Value>, _>>()?;
    serde_json::to_vec(&json!({
        "position": runtime.current_position(),
        "answers": answers,
        "quarantine": runtime.quarantine_receipts(),
    }))
    .map_err(|error| FoldError::Codec(error.to_string()))
}
