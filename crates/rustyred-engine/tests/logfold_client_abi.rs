//! SPEC-THEOREM-APP-HOST-2.0 client fold ABI (`LogFold`) differential.
//!
//! The client ABI must not become a second implementation. These tests hold it
//! to the V02 byte-equal discipline: checkpoint plus tail equals full replay,
//! by-hash upqueries answer only from the prefix the client pinned, and the
//! only write path is a proposal through the one committer.

use rustyred_thg_core::{
    ClientPin, DocumentsFold, FoldError, FoldId, FoldRuntime, LogFold, RestoreDisposition,
    DOCUMENTS_INTENT_NAMESPACE, DOCUMENTS_INTENT_VERSION, INTENT_RECORD_CREATED,
};
use serde_json::{json, Value};
use std::error::Error;
use tempfile::TempDir;
use theorem_oplog::{
    AccessPath, CommitOutcome, ContentHash, DeploymentId, IntentKind, LogIntent, LogPosition,
    OpDigest, PhysicalLog, Proposal, ProposalCommitter, ReadSet, WriteSet,
};

#[test]
fn client_checkpoint_plus_tail_equals_full_replay() -> Result<(), Box<dyn Error>> {
    for seed in 0_u64..4 {
        let (_directory, log) = generated_document_log(seed, 12)?;
        let latest = log.latest_position();
        let queries = observation_queries();

        let mut full = LogFold::for_log(DocumentsFold::default(), &log)?;
        full.apply_tail(&log)?;
        let expected = observation(full.runtime(), latest)?;

        for cut in [0, 1, latest.0 / 2, latest.0] {
            let mut prefix = LogFold::for_log(DocumentsFold::default(), &log)?;
            prefix.runtime_mut().apply_range(&log, LogPosition(cut))?;
            let checkpoint = prefix.runtime().checkpoint(&log)?;

            let mut resumed = LogFold::for_log(DocumentsFold::default(), &log)?;
            match resumed.load_checkpoint(&checkpoint, &log)? {
                RestoreDisposition::Restored { tail } => {
                    assert_eq!(tail.through, latest);
                    // The tail interval is fully accounted for: applied plus
                    // quarantined is exactly the records after the cut.
                    assert_eq!(tail.applied + tail.quarantined, (latest.0 - cut) as usize);
                }
                other => panic!("expected a restored checkpoint, got {other:?}"),
            }
            assert_eq!(observation(resumed.runtime(), latest)?, expected);

            // By-hash upqueries answer at the pinned prefix and agree with a
            // direct query at the same position.
            let pin = ClientPin::of_checkpoint(&checkpoint);
            resumed.require_pin(&log, &pin)?;
            for query in &queries {
                assert_eq!(
                    resumed.upquery_by_hash(&log, query, &pin)?,
                    resumed.runtime().query(query, pin.through)?,
                );
            }
            // Answering at the pin does not move the runtime backward.
            assert_eq!(resumed.position(), latest);

            // A pin whose hash names another prefix refuses; it never answers
            // from a prefix the client did not name.
            let wrong = ClientPin::new(pin.through, ContentHash::from_bytes(b"not-this-log"));
            let mismatch = resumed
                .upquery_by_hash(&log, &queries[0], &wrong)
                .unwrap_err();
            assert!(
                matches!(mismatch, FoldError::PinHashMismatch { .. }),
                "expected PinHashMismatch, got {mismatch:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn client_fold_subset_is_declared_once_and_matches_the_constant() {
    // Every fold is a client fold today; splitting the constant is the
    // deliberate act a new fold must take, not a silent omission.
    assert_eq!(FoldId::CLIENT_SUBSET, FoldId::ALL);
    for id in FoldId::CLIENT_SUBSET {
        assert!(FoldId::ALL.contains(&id));
        assert!(
            !id.surfaces().is_empty(),
            "{id} declares no client surfaces"
        );
    }

    // The ABI names the constant, not a copy of it.
    let fold_source = include_str!("../src/fold.rs");
    assert!(fold_source.contains("FoldId::CLIENT_SUBSET"));
}

#[test]
fn client_proposals_are_the_only_write_path() -> Result<(), Box<dyn Error>> {
    let (_directory, log) = generated_document_log(11, 3)?;
    let mut committer = ProposalCommitter::new(log);
    let mut client = LogFold::for_log(DocumentsFold::default(), committer.log())?;
    client.apply_tail(committer.log())?;
    let head = committer.log().latest_position();
    let before = observation(client.runtime(), head)?;

    let created = json!({
        "id": "doc:client",
        "kind": "authored",
        "actors": ["agent"],
        "collection_refs": ["col:client"],
        "created_position": head.0 + 1,
        "updated_position": head.0 + 1,
        "content_digest": "sha256:client"
    });
    let outcome = client.submit_proposal(
        &mut committer,
        document_proposal(head, "client-committed", &created, ReadSet::default())?,
    )?;
    assert!(matches!(outcome, CommitOutcome::Committed(_)));

    // The fold has not moved until the client applies the tail: a proposal is
    // not a state write.
    assert_eq!(client.position(), head);
    assert_eq!(observation(client.runtime(), head)?, before);
    client.apply_tail(committer.log())?;
    assert_eq!(client.position(), committer.log().latest_position());
    let answer = client.runtime().query(
        &json!({"view": "document", "key": "doc:client"}),
        client.position(),
    )?;
    assert_eq!(answer["record"]["kind"], "authored");

    // A stale proposal that read doc:client before the commit is refused with
    // the conflicting positions, and nothing is appended.
    let stale_head = committer.log().latest_position();
    let stale = document_proposal(
        LogPosition::ZERO,
        "client-refused",
        &created,
        ReadSet::new([AccessPath::Entity {
            entity: "doc:client".to_string(),
        }])?,
    )?;
    match client.submit_proposal(&mut committer, stale)? {
        CommitOutcome::Refused(receipt) => {
            assert!(!receipt.conflicting_positions.is_empty());
        }
        other => panic!("expected an OCC refusal, got {other:?}"),
    }
    assert_eq!(committer.log().latest_position(), stale_head);
    Ok(())
}

fn document_proposal(
    begin: LogPosition,
    digest: &str,
    payload: &Value,
    read_set: ReadSet,
) -> Result<Proposal, Box<dyn Error>> {
    let entity = payload
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(Proposal::new(
        begin,
        read_set,
        WriteSet::new([AccessPath::Entity { entity }])?,
        OpDigest::from_bytes(digest.as_bytes()),
        LogIntent::new(
            IntentKind::new(
                DOCUMENTS_INTENT_NAMESPACE,
                INTENT_RECORD_CREATED,
                DOCUMENTS_INTENT_VERSION,
            )?,
            payload.clone(),
        )?,
    ))
}

fn generated_document_log(
    seed: u64,
    count: usize,
) -> Result<(TempDir, PhysicalLog), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut log = PhysicalLog::open(
        directory.path(),
        DeploymentId::new("test:logfold-client-abi")?,
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

fn observation_queries() -> Vec<Value> {
    vec![
        json!({"view": "by_kind", "key": "authored"}),
        json!({"view": "by_kind", "key": "skill"}),
        json!({"view": "by_actor", "key": "human"}),
        json!({"view": "by_recency", "key": ""}),
        json!({"view": "document", "key": "doc:0"}),
    ]
}

fn observation(
    runtime: &FoldRuntime<DocumentsFold>,
    position: LogPosition,
) -> Result<Vec<u8>, FoldError> {
    let answers = observation_queries()
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
