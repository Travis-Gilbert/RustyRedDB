use rustyred_thg_core::{
    CheckpointRejectCode, Fold, FoldCapabilities, FoldConsumeError, FoldError, FoldId,
    FoldRefusalCode, FoldRuntime, FoldVersion, RestoreDisposition, SegmentSummaryDescriptor,
    SegmentTransition,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::error::Error;
use tempfile::TempDir;
use theorem_oplog::{
    AccessPath, ContentHash, IntentKind, LogId, LogIntent, LogPosition, LogRecord, OpDigest,
    PhysicalLog, WriteSet,
};

const ENTITIES: [&str; 5] = ["note:0", "note:1", "note:2", "note:3", "note:4"];

#[derive(Debug, Deserialize, Serialize)]
struct FixtureFold {
    id: FoldId,
    version: FoldVersion,
    values: BTreeMap<String, Vec<(LogPosition, i64)>>,
}

impl FixtureFold {
    fn new(id: FoldId, version: u32) -> Result<Self, FoldError> {
        Ok(Self {
            id,
            version: FoldVersion::new(version)?,
            values: BTreeMap::new(),
        })
    }

    fn payload(record: &LogRecord) -> Result<(String, i64), FoldConsumeError> {
        if record.intent.kind.namespace() != "fixture" || record.intent.kind.name() != "fact_delta"
        {
            return Err(FoldConsumeError::refused(
                FoldRefusalCode::UnsupportedIntent,
                "fixture fold accepts only fixture.fact_delta@1",
            ));
        }
        let object = record.intent.payload.as_object().ok_or_else(|| {
            FoldConsumeError::refused(FoldRefusalCode::InvalidPayload, "payload must be an object")
        })?;
        let entity = object
            .get("entity")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FoldConsumeError::refused(
                    FoldRefusalCode::SchemaViolation,
                    "entity must be a string",
                )
            })?;
        if entity.is_empty() || entity != entity.trim() {
            return Err(FoldConsumeError::refused(
                FoldRefusalCode::SchemaViolation,
                "entity must be a canonical non-empty string",
            ));
        }
        let delta = object.get("delta").and_then(Value::as_i64).ok_or_else(|| {
            FoldConsumeError::refused(
                FoldRefusalCode::SchemaViolation,
                "delta must be a signed integer",
            )
        })?;
        Ok((entity.to_string(), delta))
    }

    fn query_entity(query: &Value) -> Result<&str, FoldError> {
        query
            .get("entity")
            .and_then(Value::as_str)
            .ok_or_else(|| FoldError::Codec("query.entity must be a string".to_string()))
    }
}

impl Fold for FixtureFold {
    fn id(&self) -> FoldId {
        self.id
    }

    fn version(&self) -> FoldVersion {
        self.version
    }

    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        let (entity, delta) = Self::payload(entry)?;
        self.values
            .entry(entity)
            .or_default()
            .push((entry.position, delta));
        Ok(())
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        let entity = Self::query_entity(query)?;
        let value = self
            .values
            .get(entity)
            .into_iter()
            .flatten()
            .filter(|(at, _)| *at <= position)
            .map(|(_, delta)| delta)
            .sum::<i64>();
        Ok(json!({"entity": entity, "value": value}))
    }

    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError> {
        let entity = Self::query_entity(key)?;
        let value = pinned_records
            .iter()
            .filter_map(|record| Self::payload(record).ok())
            .filter(|(candidate, _)| candidate == entity)
            .map(|(_, delta)| delta)
            .sum::<i64>();
        Ok(json!({"entity": entity, "value": value, "position": position}))
    }

    fn checkpoint(&self) -> Result<Vec<u8>, FoldError> {
        serde_json::to_vec(&self.values).map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        self.values = serde_json::from_slice(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        self.values.clear();
        Ok(())
    }
}

#[test]
fn randomized_full_segmented_and_checkpoint_replay_are_byte_equal() -> Result<(), Box<dyn Error>> {
    for seed in 0_u64..16 {
        let (_directory, log) = generated_log(seed, 24)?;
        let latest = log.latest_position();
        assert!(log
            .records()
            .iter()
            .all(|record| !record.provenance.is_explicit_legacy()));
        for fold_id in FoldId::ALL {
            let mut full = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
            full.apply_all(&log)?;
            let expected = observation(&full, latest)?;

            let transitions = random_transitions(&log, seed ^ fold_id as u64)?;
            let mut segmented = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
            for transition in &transitions {
                segmented.apply_transition(&log, transition)?;
            }
            assert_eq!(observation(&segmented, latest)?, expected);

            let composite = transitions
                .iter()
                .skip(1)
                .try_fold(transitions[0].clone(), |combined, transition| {
                    combined.compose(transition)
                })?;
            assert_eq!(
                SegmentTransition::identity(&log, LogPosition::ZERO)?.compose(&composite)?,
                composite
            );
            assert_eq!(
                composite.compose(&SegmentTransition::identity(&log, latest)?)?,
                composite
            );
            let mut composed = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
            composed.apply_transition(&log, &composite)?;
            assert_eq!(observation(&composed, latest)?, expected);

            for cut in [0, 1, latest.0 / 3, latest.0 * 2 / 3, latest.0] {
                let mut prefix = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
                prefix.apply_range(&log, LogPosition(cut))?;
                let checkpoint = prefix.checkpoint(&log)?;

                let mut resumed = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
                assert!(matches!(
                    resumed.restore_or_rebuild(&checkpoint, &log)?,
                    RestoreDisposition::Restored { .. }
                ));
                assert_eq!(observation(&resumed, latest)?, expected);

                for entity in ENTITIES {
                    let query = json!({"entity": entity});
                    let expected_value = resumed.query(&query, latest)?["value"].clone();
                    let upquery = resumed.upquery(&log, &query, latest)?;
                    assert_eq!(upquery["value"], expected_value);
                    assert_eq!(upquery["position"], json!(latest));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn every_fold_quarantines_invalid_entries_and_continues() -> Result<(), Box<dyn Error>> {
    let (_directory, log) = generated_log(7, 12)?;
    for fold_id in FoldId::ALL {
        let mut runtime = FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?;
        let report = runtime.apply_all(&log)?;
        assert!(report.applied > 0);
        assert!(report.quarantined > 0);
        assert_eq!(report.through, log.latest_position());
        assert!(runtime
            .quarantine_receipts()
            .iter()
            .all(|receipt| receipt.fold_id == fold_id
                && receipt.fold_version == FoldVersion::new(1).unwrap()
                && receipt.log_id == log.log_id()));
        assert_eq!(
            runtime.quarantine_receipts(),
            FoldRuntime::for_log(FixtureFold::new(fold_id, 1)?, &log)?
                .tap_apply(&log)?
                .quarantine_receipts()
        );
    }
    Ok(())
}

#[test]
fn incompatible_checkpoints_are_rebuilt_never_migrated() -> Result<(), Box<dyn Error>> {
    let (_directory, log) = generated_log(91, 18)?;
    let latest = log.latest_position();
    let mut source = FoldRuntime::for_log(FixtureFold::new(FoldId::Facts, 1)?, &log)?;
    source.apply_range(&log, LogPosition(9))?;
    let checkpoint = source.checkpoint(&log)?;

    let mut baseline = FoldRuntime::for_log(FixtureFold::new(FoldId::Facts, 1)?, &log)?;
    baseline.full_rebuild(&log)?;
    let expected = observation(&baseline, latest)?;

    let mut cases = Vec::new();
    let mut bad = checkpoint.clone();
    bad.protocol_version += 1;
    cases.push((bad, CheckpointRejectCode::ProtocolVersion));

    let mut bad = checkpoint.clone();
    bad.log_id = LogId::new();
    cases.push((bad, CheckpointRejectCode::LogIdentity));

    let mut bad = checkpoint.clone();
    bad.fold_id = FoldId::Graph;
    cases.push((bad, CheckpointRejectCode::FoldIdentity));

    let mut bad = checkpoint.clone();
    bad.fold_version = FoldVersion::new(2)?;
    cases.push((bad, CheckpointRejectCode::FoldVersion));

    let mut bad = checkpoint.clone();
    bad.through = latest.next().unwrap();
    cases.push((bad, CheckpointRejectCode::Position));

    let mut bad = checkpoint.clone();
    bad.log_content_hash = ContentHash::from_bytes("foreign-prefix");
    cases.push((bad, CheckpointRejectCode::LogContentHash));

    let mut bad = checkpoint;
    bad.state.push(0xff);
    cases.push((bad, CheckpointRejectCode::CheckpointHash));

    let mut bad = source.checkpoint(&log)?;
    bad.fold_descriptor.capabilities.segment_summary = Some(SegmentSummaryDescriptor::new(
        "fixture.unsound_snapshot",
        1,
    )?);
    cases.push((bad, CheckpointRejectCode::FoldCapabilities));

    for (bad, expected_code) in cases {
        let mut runtime = FoldRuntime::for_log(FixtureFold::new(FoldId::Facts, 1)?, &log)?;
        let RestoreDisposition::Rebuilt { rejection, replay } =
            runtime.restore_or_rebuild(&bad, &log)?
        else {
            panic!("incompatible checkpoint was restored");
        };
        assert_eq!(rejection.code, expected_code);
        assert_eq!(replay.through, latest);
        assert_eq!(observation(&runtime, latest)?, expected);
    }
    Ok(())
}

trait ApplyFixture {
    fn tap_apply(self, log: &PhysicalLog) -> Result<Self, FoldError>
    where
        Self: Sized;
}

impl ApplyFixture for FoldRuntime<FixtureFold> {
    fn tap_apply(mut self, log: &PhysicalLog) -> Result<Self, FoldError> {
        self.apply_all(log)?;
        Ok(self)
    }
}

fn generated_log(seed: u64, count: usize) -> Result<(TempDir, PhysicalLog), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut log = PhysicalLog::open(
        directory.path(),
        theorem_oplog::DeploymentId::new("test:fold-replay-differential")?,
    )?;
    let mut random = seed ^ 0x8f3c_29a1_54d7_b6e0;
    for index in 0..count {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let entity = ENTITIES[(random as usize) % ENTITIES.len()];
        let invalid = random % 7 == 0 || index == count / 2;
        let payload = if invalid {
            json!({"entity": entity, "delta": "not-an-integer"})
        } else {
            json!({"entity": entity, "delta": (random % 17) as i64 - 8})
        };
        let digest_input = format!("{seed}:{index}:{random}");
        log.append(
            LogIntent::new(IntentKind::new("fixture", "fact_delta", 1)?, payload)?,
            WriteSet::new([AccessPath::Entity {
                entity: entity.to_string(),
            }])?,
            OpDigest::from_bytes(digest_input),
        )?;
    }
    Ok((directory, log))
}

fn random_transitions(log: &PhysicalLog, seed: u64) -> Result<Vec<SegmentTransition>, FoldError> {
    let mut random = seed ^ 0xd1b5_4a32_d192_ed03;
    let mut from = LogPosition::ZERO;
    let mut transitions = Vec::new();
    while from < log.latest_position() {
        random = random
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        let width = random % 5 + 1;
        let through = LogPosition((from.0 + width).min(log.latest_position().0));
        transitions.push(SegmentTransition::from_log(log, from, through)?);
        from = through;
    }
    Ok(transitions)
}

fn observation(
    runtime: &FoldRuntime<FixtureFold>,
    position: LogPosition,
) -> Result<Vec<u8>, FoldError> {
    let mut answers = BTreeMap::new();
    for entity in ENTITIES {
        answers.insert(entity, runtime.query(&json!({"entity": entity}), position)?);
    }
    serde_json::to_vec(&json!({
        "position": runtime.current_position(),
        "answers": answers,
        "quarantine": runtime.quarantine_receipts(),
    }))
    .map_err(|error| FoldError::Codec(error.to_string()))
}
