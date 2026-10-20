use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::labeled_training_run::LabeledTrainingRun;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExportKind {
    ContextRank,
    ToolPolicy,
    AdapterPolicy,
    ValidatorPolicy,
    MemoryRecall,
    MapReuse,
    ArtifactOutcome,
    TrainingSft,
    TrainingPreference,
    GraphPack,
    MultiVectorRerank,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionStatus {
    Pending,
    Safe,
    Redacted,
    Blocked,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainingExportRecord {
    pub export_kind: TrainingExportKind,
    pub graph_version: u64,
    pub redaction_status: RedactionStatus,
    pub source_run_id: String,
    pub source_artifact_id: String,
    pub payload: Value,
}

impl TrainingExportRecord {
    pub fn from_labeled_run(
        export_kind: TrainingExportKind,
        run: &LabeledTrainingRun,
        redaction_status: RedactionStatus,
    ) -> GraphStoreResult<Self> {
        let payload = serde_json::to_value(run).map_err(|err| {
            GraphStoreError::new(
                "training_export_serialize_error",
                format!("could not serialize labeled training run {}: {err}", run.id),
            )
        })?;
        Ok(Self {
            export_kind,
            graph_version: run.graph_version,
            redaction_status,
            source_run_id: run.run_id.clone(),
            source_artifact_id: run.artifact_id.clone(),
            payload,
        })
    }
}

pub fn export_records_jsonl(records: &[TrainingExportRecord]) -> GraphStoreResult<String> {
    let mut out = String::new();
    for record in records {
        let line = serde_json::to_string(record).map_err(|err| {
            GraphStoreError::new(
                "training_export_serialize_error",
                format!("could not serialize training export record: {err}"),
            )
        })?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatSample {
    pub messages: Vec<ChatMessage>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PreferenceSample {
    pub input: ChatSample,
    pub preferred_output: ChatMessage,
    pub non_preferred_output: ChatMessage,
}

pub fn chat_samples_jsonl(samples: &[ChatSample]) -> GraphStoreResult<String> {
    let mut out = String::new();
    for sample in samples {
        let line = serde_json::to_string(sample).map_err(|err| {
            GraphStoreError::new(
                "training_export_serialize_error",
                format!("could not serialize chat sample: {err}"),
            )
        })?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

pub fn preference_samples_jsonl(samples: &[PreferenceSample]) -> GraphStoreResult<String> {
    let mut out = String::new();
    for sample in samples {
        let line = serde_json::to_string(sample).map_err(|err| {
            GraphStoreError::new(
                "training_export_serialize_error",
                format!("could not serialize preference sample: {err}"),
            )
        })?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

/// Emit only records whose redaction status is Safe.
pub fn sft_samples_from_records(
    records: &[TrainingExportRecord],
) -> GraphStoreResult<Vec<ChatSample>> {
    let mut samples = Vec::new();
    for record in records {
        if record.redaction_status != RedactionStatus::Safe {
            continue;
        }
        samples.push(chat_sample_from_record(record)?);
    }
    Ok(samples)
}

fn chat_sample_from_record(record: &TrainingExportRecord) -> GraphStoreResult<ChatSample> {
    if let Some(messages) = record
        .payload
        .get("messages")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let role = item.get("role")?.as_str()?.to_string();
                    let content = item.get("content")?.as_str()?.to_string();
                    Some(ChatMessage { role, content })
                })
                .collect::<Vec<_>>()
        })
        .filter(|messages| !messages.is_empty())
    {
        return Ok(ChatSample { messages });
    }

    let system = record
        .payload
        .get("system")
        .or_else(|| record.payload.get("instruction"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let user = record
        .payload
        .get("user")
        .or_else(|| record.payload.get("input"))
        .or_else(|| record.payload.get("task"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let assistant = record
        .payload
        .get("assistant")
        .or_else(|| record.payload.get("output"))
        .or_else(|| record.payload.get("completion"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let mut messages = Vec::new();
    if !system.trim().is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: if user.trim().is_empty() {
            record.source_run_id.clone()
        } else {
            user
        },
    });
    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: if assistant.trim().is_empty() {
            record.payload.to_string()
        } else {
            assistant
        },
    });
    Ok(ChatSample { messages })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(status: RedactionStatus, payload: Value) -> TrainingExportRecord {
        TrainingExportRecord {
            export_kind: TrainingExportKind::TrainingSft,
            graph_version: 1,
            redaction_status: status,
            source_run_id: "run:1".into(),
            source_artifact_id: "artifact:1".into(),
            payload,
        }
    }

    #[test]
    fn sft_samples_from_records_keeps_only_safe() {
        let records = vec![
            record(
                RedactionStatus::Safe,
                json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}),
            ),
            record(
                RedactionStatus::Pending,
                json!({"messages":[{"role":"user","content":"no"}]}),
            ),
            record(
                RedactionStatus::Blocked,
                json!({"messages":[{"role":"user","content":"no"}]}),
            ),
            record(
                RedactionStatus::Redacted,
                json!({"messages":[{"role":"user","content":"no"}]}),
            ),
        ];
        let samples = sft_samples_from_records(&records).unwrap();
        assert_eq!(samples.len(), 1);
        let jsonl = chat_samples_jsonl(&samples).unwrap();
        for line in jsonl.lines() {
            let value: Value = serde_json::from_str(line).unwrap();
            assert!(value.get("messages").and_then(Value::as_array).is_some());
        }
    }
}
