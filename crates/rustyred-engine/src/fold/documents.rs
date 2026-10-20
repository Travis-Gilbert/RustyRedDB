//! Document corpus fold: state from log entries, keyed views, upquery-by-id.
//!
//! Secondary reads (`by_kind`, `by_collection`, `by_recency`, `by_actor`) are
//! fold-internal keys over this state, not a second index. Entries are integer,
//! hash, and string shaped.

use super::{Fold, FoldCapabilities, FoldError, FoldId, FoldVersion};
use crate::fold_quarantine::{FoldConsumeError, FoldRefusalCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use theorem_oplog::{LogPosition, LogRecord};

pub const FOLD_ID: FoldId = FoldId::Documents;
pub const DOCUMENTS_FOLD_VERSION: u32 = 1;
pub const DOCUMENTS_INTENT_NAMESPACE: &str = "theorem.documents";
pub const DOCUMENTS_INTENT_VERSION: u32 = 1;

pub const INTENT_RECORD_CREATED: &str = "record_created";
pub const INTENT_SNAPSHOT_COMMITTED: &str = "snapshot_committed";
pub const INTENT_REVISION: &str = "revision";
pub const INTENT_SKILL_EDITION_SELECTED: &str = "skill_edition_selected";
pub const INTENT_PROVENANCE_SPAN_APPENDED: &str = "provenance_span_appended";

const VIEW_DOCUMENT: &str = "document";
const VIEW_BY_KIND: &str = "by_kind";
const VIEW_BY_COLLECTION: &str = "by_collection";
const VIEW_BY_RECENCY: &str = "by_recency";
const VIEW_BY_ACTOR: &str = "by_actor";

/// One document as the fold stores it: strings, integers, and hashes only.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DocumentsFoldRecord {
    pub id: String,
    pub kind: String,
    pub actors: Vec<String>,
    pub collection_refs: Vec<String>,
    pub created_position: u64,
    pub updated_position: u64,
    pub content_digest: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_skill_digest: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct DocumentsFoldCheckpoint {
    versions: BTreeMap<String, BTreeMap<u64, DocumentsFoldRecord>>,
}

/// Query over the documents fold. Secondary reads are keyed views.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "view", rename_all = "snake_case")]
pub enum DocumentsQuery {
    Document {
        key: String,
    },
    ByKind {
        key: String,
    },
    ByCollection {
        key: String,
    },
    ByRecency {
        #[serde(default)]
        key: String,
    },
    ByActor {
        key: String,
    },
}

#[derive(Clone, Debug, Default)]
pub struct DocumentsFold {
    /// doc_id -> log_position -> record. Latest log position at or before a pin wins.
    versions: BTreeMap<String, BTreeMap<u64, DocumentsFoldRecord>>,
}

impl DocumentsFold {
    pub fn descriptor() -> super::FoldDescriptor {
        super::FoldDescriptor::new(
            FOLD_ID,
            FoldVersion::new(DOCUMENTS_FOLD_VERSION).expect("Documents fold version is non-zero"),
            FoldCapabilities::ordered_transitions_only(),
        )
    }

    pub fn apply_intent(
        &mut self,
        name: &str,
        payload: &Value,
        log_position: LogPosition,
    ) -> Result<(), FoldConsumeError> {
        if !matches!(
            name,
            INTENT_RECORD_CREATED
                | INTENT_SNAPSHOT_COMMITTED
                | INTENT_REVISION
                | INTENT_SKILL_EDITION_SELECTED
                | INTENT_PROVENANCE_SPAN_APPENDED
        ) {
            return Err(FoldConsumeError::refused(
                FoldRefusalCode::UnsupportedIntent,
                format!("documents fold does not consume intent {name}"),
            ));
        }
        let record = decode_record(payload)?;
        if record.id.is_empty() || record.id != record.id.trim() {
            return Err(schema_refusal("id must be a canonical non-empty string"));
        }
        if record.kind.is_empty() || record.kind != record.kind.trim() {
            return Err(schema_refusal("kind must be a canonical non-empty string"));
        }
        if record.content_digest.is_empty() {
            return Err(schema_refusal("content_digest must be a hash string"));
        }
        let mut staged = self.clone();
        staged
            .versions
            .entry(record.id.clone())
            .or_default()
            .insert(log_position.0, record);
        *self = staged;
        Ok(())
    }

    pub fn records_at(&self, position: LogPosition) -> Vec<DocumentsFoldRecord> {
        let mut records = Vec::new();
        for versions in self.versions.values() {
            if let Some(record) = latest_at(versions, position.0) {
                records.push(record.clone());
            }
        }
        records.sort_by(|left, right| {
            right
                .updated_position
                .cmp(&left.updated_position)
                .then_with(|| left.id.cmp(&right.id))
        });
        records
    }

    pub fn records_at_updated_position(&self, through: u64) -> Vec<DocumentsFoldRecord> {
        let mut records = Vec::new();
        for versions in self.versions.values() {
            let Some(record) = versions
                .values()
                .rev()
                .find(|record| through == 0 || record.updated_position <= through)
            else {
                continue;
            };
            records.push(record.clone());
        }
        records.sort_by(|left, right| {
            right
                .updated_position
                .cmp(&left.updated_position)
                .then_with(|| left.id.cmp(&right.id))
        });
        records
    }

    pub fn ids_for_view(&self, view: &str, key: &str, through: u64) -> Vec<String> {
        ids_from_records(&self.records_at_updated_position(through), view, key)
    }

    pub fn record_by_id(&self, id: &str, through: u64) -> Option<DocumentsFoldRecord> {
        self.records_at_updated_position(through)
            .into_iter()
            .find(|record| record.id == id)
    }

    fn answer(&self, query: &DocumentsQuery, position: LogPosition) -> Value {
        let through = position.0;
        match query {
            DocumentsQuery::Document { key } => {
                let empty = BTreeMap::new();
                let record = latest_at(self.versions.get(key).unwrap_or(&empty), through);
                json!({
                    "view": VIEW_DOCUMENT,
                    "key": key,
                    "through": through,
                    "record": record,
                })
            }
            DocumentsQuery::ByKind { key } => self.view_answer(VIEW_BY_KIND, key, through),
            DocumentsQuery::ByCollection { key } => {
                self.view_answer(VIEW_BY_COLLECTION, key, through)
            }
            DocumentsQuery::ByActor { key } => self.view_answer(VIEW_BY_ACTOR, key, through),
            DocumentsQuery::ByRecency { key } => self.view_answer(VIEW_BY_RECENCY, key, through),
        }
    }

    fn view_answer(&self, view: &str, key: &str, through: u64) -> Value {
        json!({
            "view": view,
            "key": key,
            "through": through,
            "ids": ids_from_records(&self.records_at(LogPosition(through)), view, key),
        })
    }
}

impl Fold for DocumentsFold {
    fn id(&self) -> FoldId {
        FOLD_ID
    }

    fn version(&self) -> FoldVersion {
        FoldVersion::new(DOCUMENTS_FOLD_VERSION).expect("Documents fold version is non-zero")
    }

    fn capabilities(&self) -> FoldCapabilities {
        FoldCapabilities::ordered_transitions_only()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        if entry.intent.kind.namespace() != DOCUMENTS_INTENT_NAMESPACE {
            return Err(FoldConsumeError::refused(
                FoldRefusalCode::UnsupportedIntent,
                format!(
                    "expected {DOCUMENTS_INTENT_NAMESPACE} intents, found {}",
                    entry.intent.kind.namespace()
                ),
            ));
        }
        if entry.intent.kind.version() != DOCUMENTS_INTENT_VERSION {
            return Err(FoldConsumeError::refused(
                FoldRefusalCode::UnsupportedIntent,
                format!(
                    "expected {DOCUMENTS_INTENT_NAMESPACE}@{DOCUMENTS_INTENT_VERSION}, found {}",
                    entry.intent.kind.version()
                ),
            ));
        }
        self.apply_intent(
            entry.intent.kind.name(),
            &entry.intent.payload,
            entry.position,
        )
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        let query = parse_query(query)?;
        Ok(self.answer(&query, position))
    }

    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError> {
        let query = parse_query(key)?;
        let mut rebuilt = Self::default();
        for record in pinned_records {
            match rebuilt.apply(record) {
                Ok(()) | Err(FoldConsumeError::Refused(_)) => {}
                Err(FoldConsumeError::Fatal(reason)) => {
                    return Err(FoldError::FatalConsume(reason));
                }
            }
        }
        Ok(rebuilt.answer(&query, position))
    }

    fn checkpoint(&self) -> Result<Vec<u8>, FoldError> {
        serde_json::to_vec(&DocumentsFoldCheckpoint {
            versions: self.versions.clone(),
        })
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        let checkpoint = serde_json::from_slice::<DocumentsFoldCheckpoint>(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        self.versions = checkpoint.versions;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        *self = Self::default();
        Ok(())
    }
}

fn latest_at(
    versions: &BTreeMap<u64, DocumentsFoldRecord>,
    through: u64,
) -> Option<&DocumentsFoldRecord> {
    versions
        .iter()
        .rev()
        .find(|(position, _)| **position <= through)
        .map(|(_, record)| record)
}

fn ids_from_records(visible: &[DocumentsFoldRecord], view: &str, key: &str) -> Vec<String> {
    match view {
        VIEW_DOCUMENT => visible
            .iter()
            .filter(|record| record.id == key)
            .map(|record| record.id.clone())
            .collect(),
        VIEW_BY_KIND => visible
            .iter()
            .filter(|record| record.kind == key)
            .map(|record| record.id.clone())
            .collect(),
        VIEW_BY_COLLECTION => visible
            .iter()
            .filter(|record| {
                record
                    .collection_refs
                    .iter()
                    .any(|collection| collection == key)
            })
            .map(|record| record.id.clone())
            .collect(),
        VIEW_BY_ACTOR => visible
            .iter()
            .filter(|record| record.actors.iter().any(|actor| actor == key))
            .map(|record| record.id.clone())
            .collect(),
        VIEW_BY_RECENCY => {
            if key.is_empty() {
                visible.iter().map(|record| record.id.clone()).collect()
            } else {
                visible
                    .iter()
                    .filter(|record| record.id.contains(key))
                    .map(|record| record.id.clone())
                    .collect()
            }
        }
        _ => Vec::new(),
    }
}

fn decode_record(payload: &Value) -> Result<DocumentsFoldRecord, FoldConsumeError> {
    serde_json::from_value(payload.clone()).map_err(|error| {
        FoldConsumeError::refused(
            FoldRefusalCode::InvalidPayload,
            format!("documents fold payload is not a record: {error}"),
        )
    })
}

fn parse_query(query: &Value) -> Result<DocumentsQuery, FoldError> {
    if let Ok(parsed) = serde_json::from_value::<DocumentsQuery>(query.clone()) {
        return Ok(parsed);
    }
    let view = query
        .get("view")
        .or_else(|| query.get("fold"))
        .and_then(Value::as_str)
        .unwrap_or(VIEW_DOCUMENT);
    let key = query
        .get("key")
        .or_else(|| query.get("id"))
        .or_else(|| query.get("doc_id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    match view {
        VIEW_DOCUMENT | "id" => Ok(DocumentsQuery::Document { key }),
        VIEW_BY_KIND => Ok(DocumentsQuery::ByKind { key }),
        VIEW_BY_COLLECTION => Ok(DocumentsQuery::ByCollection { key }),
        VIEW_BY_RECENCY => Ok(DocumentsQuery::ByRecency { key }),
        VIEW_BY_ACTOR => Ok(DocumentsQuery::ByActor { key }),
        other => Err(FoldError::Codec(format!(
            "unknown documents fold view {other}"
        ))),
    }
}

fn schema_refusal(reason: impl Into<String>) -> FoldConsumeError {
    FoldConsumeError::refused(FoldRefusalCode::SchemaViolation, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn view_keys_answer_from_fold_state() {
        let mut fold = DocumentsFold::default();
        fold.apply_intent(
            INTENT_RECORD_CREATED,
            &json!({
                "id": "a",
                "kind": "postmortem",
                "actors": ["agent"],
                "collection_refs": ["col:run"],
                "created_position": 1,
                "updated_position": 1,
                "content_digest": "sha256:abc"
            }),
            LogPosition(1),
        )
        .unwrap();
        assert_eq!(
            fold.ids_for_view(VIEW_BY_KIND, "postmortem", 1),
            vec!["a".to_string()]
        );
        assert_eq!(
            fold.ids_for_view(VIEW_BY_COLLECTION, "col:run", 1),
            vec!["a".to_string()]
        );
        assert_eq!(
            fold.ids_for_view(VIEW_BY_ACTOR, "agent", 1),
            vec!["a".to_string()]
        );
        assert_eq!(
            fold.ids_for_view(VIEW_BY_RECENCY, "", 1),
            vec!["a".to_string()]
        );
        assert!(fold.records_at(LogPosition::ZERO).is_empty());
        assert!(fold.answer(
            &DocumentsQuery::Document { key: "a".into() },
            LogPosition::ZERO
        )["record"]
            .is_null());
        assert_eq!(
            fold.record_by_id("a", 0).map(|record| record.id),
            Some("a".into()),
            "store wrapper: through=0 still means the latest document pin"
        );
    }
}
