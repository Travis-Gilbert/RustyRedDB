//! Control event plane (SPEC-THEOREM-CONTROL-PRIMITIVES-1.0 CP1).
//!
//! Record and metadata operations compile to stable reversible strings.
//! Emitted control events name what changed; they never carry record payloads.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph::record::NodeRecord;
use crate::hooks::MutationKind;

pub const CONTROL_EVENT_LABEL: &str = "ControlEvent";

/// Schema registry object-type label. Metadata plane keys on this.
pub const METADATA_OBJECT_TYPE_ENTITY: &str = "object_type";
/// Schema registry field-spec label.
pub const METADATA_FIELD_SPEC_ENTITY: &str = "field_spec";
pub const OBJECT_TYPE_LABEL: &str = "ObjectType";
pub const FIELD_SPEC_LABEL: &str = "FieldSpec";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventOperation {
    Record {
        object: ObjectSelector,
        event: RecordEvent,
    },
    Metadata {
        entity: MetadataSelector,
        operation: MetadataOperation,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectSelector {
    Named(String),
    Any,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordEvent {
    Created,
    Updated,
    Deleted,
    Restored,
    Any,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataSelector {
    Named(String),
    Any,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataOperation {
    Created,
    Updated,
    Deleted,
    Any,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ControlEventRecord {
    pub event_id: String,
    pub tenant: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal: Option<String>,
    pub graph_version_before: u64,
    pub graph_version_after: u64,
    pub affected_node_ids: Vec<String>,
    pub operation: EventOperation,
    pub compiled: String,
    pub committed_at_ms: u64,
    pub txn_id: u64,
    pub sequence: u64,
}

impl EventOperation {
    /// Total compilation to a stable subscription string.
    pub fn compile(&self) -> String {
        match self {
            Self::Record { object, event } => {
                format!("{}.{}", object.compile(), event.compile())
            }
            Self::Metadata { entity, operation } => {
                format!("metadata.{}.{}", entity.compile(), operation.compile())
            }
        }
    }

    /// Inverse of [`Self::compile`]. Accepts only forms this plane emits.
    pub fn parse(compiled: &str) -> Result<Self, EventCompileError> {
        let compiled = compiled.trim();
        if compiled.is_empty() {
            return Err(EventCompileError::empty());
        }
        if let Some(rest) = compiled.strip_prefix("metadata.") {
            let (entity, operation) = split_two(rest).ok_or_else(|| {
                EventCompileError::invalid(format!(
                    "expected metadata.entity.operation, got {compiled}"
                ))
            })?;
            return Ok(Self::Metadata {
                entity: MetadataSelector::parse(entity)?,
                operation: MetadataOperation::parse(operation)?,
            });
        }
        let (object, event) = split_two(compiled).ok_or_else(|| {
            EventCompileError::invalid(format!("expected object.event, got {compiled}"))
        })?;
        Ok(Self::Record {
            object: ObjectSelector::parse(object)?,
            event: RecordEvent::parse(event)?,
        })
    }

    /// Wildcard match: named equals named, Any matches any concrete value.
    pub fn matches(&self, concrete: &EventOperation) -> bool {
        match (self, concrete) {
            (
                Self::Record {
                    object: filter_object,
                    event: filter_event,
                },
                Self::Record {
                    object: concrete_object,
                    event: concrete_event,
                },
            ) => filter_object.matches(concrete_object) && filter_event.matches(concrete_event),
            (
                Self::Metadata {
                    entity: filter_entity,
                    operation: filter_operation,
                },
                Self::Metadata {
                    entity: concrete_entity,
                    operation: concrete_operation,
                },
            ) => {
                filter_entity.matches(concrete_entity)
                    && filter_operation.matches(concrete_operation)
            }
            _ => false,
        }
    }
}

impl ObjectSelector {
    fn compile(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Any => "*".into(),
        }
    }

    fn parse(raw: &str) -> Result<Self, EventCompileError> {
        if raw == "*" {
            Ok(Self::Any)
        } else if raw.is_empty() || raw.contains('.') {
            Err(EventCompileError::invalid(format!(
                "invalid object selector: {raw}"
            )))
        } else {
            Ok(Self::Named(raw.to_string()))
        }
    }

    fn matches(&self, other: &Self) -> bool {
        match self {
            Self::Any => true,
            Self::Named(name) => match other {
                Self::Named(other_name) => name == other_name,
                Self::Any => false,
            },
        }
    }
}

impl RecordEvent {
    fn compile(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Updated => "updated",
            Self::Deleted => "deleted",
            Self::Restored => "restored",
            Self::Any => "*",
        }
    }

    fn parse(raw: &str) -> Result<Self, EventCompileError> {
        match raw {
            "created" => Ok(Self::Created),
            "updated" => Ok(Self::Updated),
            "deleted" => Ok(Self::Deleted),
            "restored" => Ok(Self::Restored),
            "*" => Ok(Self::Any),
            other => Err(EventCompileError::invalid(format!(
                "invalid record event: {other}"
            ))),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        matches!(self, Self::Any) || self == other
    }
}

impl MetadataSelector {
    fn compile(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Any => "*".into(),
        }
    }

    fn parse(raw: &str) -> Result<Self, EventCompileError> {
        if raw == "*" {
            Ok(Self::Any)
        } else if raw.is_empty() || raw.contains('.') {
            Err(EventCompileError::invalid(format!(
                "invalid metadata selector: {raw}"
            )))
        } else {
            Ok(Self::Named(raw.to_string()))
        }
    }

    fn matches(&self, other: &Self) -> bool {
        match self {
            Self::Any => true,
            Self::Named(name) => match other {
                Self::Named(other_name) => name == other_name,
                Self::Any => false,
            },
        }
    }
}

impl MetadataOperation {
    fn compile(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Updated => "updated",
            Self::Deleted => "deleted",
            Self::Any => "*",
        }
    }

    fn parse(raw: &str) -> Result<Self, EventCompileError> {
        match raw {
            "created" => Ok(Self::Created),
            "updated" => Ok(Self::Updated),
            "deleted" => Ok(Self::Deleted),
            "*" => Ok(Self::Any),
            other => Err(EventCompileError::invalid(format!(
                "invalid metadata operation: {other}"
            ))),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        matches!(self, Self::Any) || self == other
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventCompileError {
    pub code: &'static str,
    pub message: String,
}

impl EventCompileError {
    fn empty() -> Self {
        Self {
            code: "event_compile_empty",
            message: "compiled event string is empty".into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "event_compile_invalid",
            message: message.into(),
        }
    }
}

impl std::fmt::Display for EventCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for EventCompileError {}

fn split_two(raw: &str) -> Option<(&str, &str)> {
    let (left, right) = raw.rsplit_once('.')?;
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some((left, right))
}

/// Inputs required to project one committed node mutation into a control event.
#[derive(Clone, Debug)]
pub struct NodeMutationProjection {
    pub kind: MutationKind,
    pub node_id: String,
    pub labels: Vec<String>,
    pub prior_was_tombstone: bool,
    pub had_prior: bool,
}

/// Project a node mutation into at most one blind control operation.
pub fn project_node_mutation(mutation: &NodeMutationProjection) -> Option<EventOperation> {
    if mutation
        .labels
        .iter()
        .any(|label| label == CONTROL_EVENT_LABEL)
    {
        return None;
    }

    if let Some(entity) = metadata_entity_from_labels(&mutation.labels) {
        let operation = match mutation.kind {
            MutationKind::NodeUpserted if !mutation.had_prior || mutation.prior_was_tombstone => {
                MetadataOperation::Created
            }
            MutationKind::NodeUpserted => MetadataOperation::Updated,
            MutationKind::NodeDeleted => MetadataOperation::Deleted,
            MutationKind::EdgeUpserted | MutationKind::EdgeDeleted => return None,
        };
        return Some(EventOperation::Metadata {
            entity: MetadataSelector::Named(entity.to_string()),
            operation,
        });
    }

    let object = object_name_from_labels(&mutation.labels)?;
    let event = match mutation.kind {
        MutationKind::NodeUpserted if mutation.prior_was_tombstone => RecordEvent::Restored,
        MutationKind::NodeUpserted if !mutation.had_prior => RecordEvent::Created,
        MutationKind::NodeUpserted => RecordEvent::Updated,
        MutationKind::NodeDeleted => RecordEvent::Deleted,
        MutationKind::EdgeUpserted | MutationKind::EdgeDeleted => return None,
    };
    Some(EventOperation::Record {
        object: ObjectSelector::Named(object),
        event,
    })
}

fn metadata_entity_from_labels(labels: &[String]) -> Option<&'static str> {
    if labels.iter().any(|label| label == OBJECT_TYPE_LABEL) {
        Some(METADATA_OBJECT_TYPE_ENTITY)
    } else if labels.iter().any(|label| label == FIELD_SPEC_LABEL) {
        Some(METADATA_FIELD_SPEC_ENTITY)
    } else {
        None
    }
}

fn object_name_from_labels(labels: &[String]) -> Option<String> {
    labels
        .iter()
        .find(|label| {
            *label != CONTROL_EVENT_LABEL
                && *label != OBJECT_TYPE_LABEL
                && *label != FIELD_SPEC_LABEL
        })
        .cloned()
}

impl ControlEventRecord {
    pub fn new(
        tenant: impl Into<String>,
        actor: Option<String>,
        principal: Option<String>,
        graph_version_before: u64,
        graph_version_after: u64,
        affected_node_ids: Vec<String>,
        operation: EventOperation,
        committed_at_ms: u64,
        txn_id: u64,
        sequence: u64,
    ) -> Self {
        let compiled = operation.compile();
        let event_id = format!("controlevent:{txn_id}:{sequence}:{compiled}");
        Self {
            event_id,
            tenant: tenant.into(),
            actor,
            principal,
            graph_version_before,
            graph_version_after,
            affected_node_ids,
            operation,
            compiled,
            committed_at_ms,
            txn_id,
            sequence,
        }
    }

    pub fn to_node(&self) -> NodeRecord {
        // Blind plane: identity and selectors only. No record payload fields.
        NodeRecord::new(
            self.event_id.clone(),
            [CONTROL_EVENT_LABEL],
            json!({
                "event_id": self.event_id,
                "tenant": self.tenant,
                "actor": self.actor,
                "principal": self.principal,
                "graph_version_before": self.graph_version_before,
                "graph_version_after": self.graph_version_after,
                "affected_node_ids": self.affected_node_ids,
                "compiled": self.compiled,
                "operation": self.operation,
                "committed_at_ms": self.committed_at_ms,
                "txn_id": self.txn_id,
                "sequence": self.sequence,
            }),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, EventCompileError> {
        if !node.labels.iter().any(|label| label == CONTROL_EVENT_LABEL) {
            return Err(EventCompileError::invalid(
                "node is not labeled ControlEvent",
            ));
        }
        let mut props = node.properties.clone();
        if let Some(object) = props.as_object_mut() {
            object
                .entry("event_id".to_string())
                .or_insert_with(|| Value::String(node.id.clone()));
        }
        serde_json::from_value(props).map_err(|error| {
            EventCompileError::invalid(format!("control event decode failed: {error}"))
        })
    }

    /// True when properties contain no record body fields (blind plane).
    pub fn is_blind(&self) -> bool {
        let value = serde_json::to_value(self).unwrap_or(Value::Null);
        let Some(object) = value.as_object() else {
            return false;
        };
        !object.contains_key("properties")
            && !object.contains_key("payload")
            && !object.contains_key("record")
            && !object.contains_key("body")
    }
}

/// Build durable control-event nodes from projected node mutations.
pub fn control_events_from_projections(
    tenant: &str,
    actor: Option<&str>,
    principal: Option<&str>,
    graph_version_before: u64,
    graph_version_after: u64,
    committed_at_ms: u64,
    txn_id: u64,
    projections: &[NodeMutationProjection],
) -> Vec<ControlEventRecord> {
    let mut out = Vec::new();
    let mut sequence = 0u64;
    for projection in projections {
        let Some(operation) = project_node_mutation(projection) else {
            continue;
        };
        out.push(ControlEventRecord::new(
            tenant,
            actor.map(str::to_string),
            principal.map(str::to_string),
            graph_version_before,
            graph_version_after,
            vec![projection.node_id.clone()],
            operation,
            committed_at_ms,
            txn_id,
            sequence,
        ));
        sequence += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_round_trips_named_and_wildcards() {
        let cases = [
            EventOperation::Record {
                object: ObjectSelector::Named("gclba_property".into()),
                event: RecordEvent::Created,
            },
            EventOperation::Metadata {
                entity: MetadataSelector::Named("object_type".into()),
                operation: MetadataOperation::Updated,
            },
            EventOperation::Record {
                object: ObjectSelector::Any,
                event: RecordEvent::Any,
            },
            EventOperation::Metadata {
                entity: MetadataSelector::Any,
                operation: MetadataOperation::Any,
            },
        ];
        for op in cases {
            let compiled = op.compile();
            let parsed = EventOperation::parse(&compiled).unwrap();
            assert_eq!(parsed, op);
            assert_eq!(parsed.compile(), compiled);
        }
        assert_eq!(
            EventOperation::Record {
                object: ObjectSelector::Any,
                event: RecordEvent::Any,
            }
            .compile(),
            "*.*"
        );
        assert_eq!(
            EventOperation::Metadata {
                entity: MetadataSelector::Named("object_type".into()),
                operation: MetadataOperation::Created,
            }
            .compile(),
            "metadata.object_type.created"
        );
    }

    #[test]
    fn wildcards_match_concrete_ops() {
        let filter = EventOperation::Record {
            object: ObjectSelector::Any,
            event: RecordEvent::Created,
        };
        let concrete = EventOperation::Record {
            object: ObjectSelector::Named("note".into()),
            event: RecordEvent::Created,
        };
        assert!(filter.matches(&concrete));
        assert!(!filter.matches(&EventOperation::Record {
            object: ObjectSelector::Named("note".into()),
            event: RecordEvent::Updated,
        }));

        let meta_filter = EventOperation::Metadata {
            entity: MetadataSelector::Any,
            operation: MetadataOperation::Any,
        };
        assert!(meta_filter.matches(&EventOperation::Metadata {
            entity: MetadataSelector::Named("object_type".into()),
            operation: MetadataOperation::Created,
        }));
        assert!(!meta_filter.matches(&concrete));
    }

    #[test]
    fn project_object_type_as_metadata_created() {
        let op = project_node_mutation(&NodeMutationProjection {
            kind: MutationKind::NodeUpserted,
            node_id: "schema:t1:Note".into(),
            labels: vec![OBJECT_TYPE_LABEL.into()],
            prior_was_tombstone: false,
            had_prior: false,
        })
        .unwrap();
        assert_eq!(op.compile(), "metadata.object_type.created");
    }

    #[test]
    fn project_record_created_and_skips_control_event_nodes() {
        let op = project_node_mutation(&NodeMutationProjection {
            kind: MutationKind::NodeUpserted,
            node_id: "n1".into(),
            labels: vec!["gclba_property".into()],
            prior_was_tombstone: false,
            had_prior: false,
        })
        .unwrap();
        assert_eq!(op.compile(), "gclba_property.created");

        assert!(project_node_mutation(&NodeMutationProjection {
            kind: MutationKind::NodeUpserted,
            node_id: "controlevent:1:0:x".into(),
            labels: vec![CONTROL_EVENT_LABEL.into()],
            prior_was_tombstone: false,
            had_prior: false,
        })
        .is_none());
    }

    #[test]
    fn control_event_node_is_blind() {
        let event = ControlEventRecord::new(
            "t1",
            Some("actor".into()),
            Some("principal".into()),
            1,
            2,
            vec!["n1".into()],
            EventOperation::Record {
                object: ObjectSelector::Named("note".into()),
                event: RecordEvent::Created,
            },
            10,
            7,
            0,
        );
        assert!(event.is_blind());
        let node = event.to_node();
        let props = node.properties.as_object().unwrap();
        assert!(!props.contains_key("payload"));
        assert!(!props.contains_key("properties"));
        assert!(!props.contains_key("body"));
        let round = ControlEventRecord::from_node(&node).unwrap();
        assert_eq!(round.compiled, "note.created");
        assert_eq!(round.affected_node_ids, vec!["n1".to_string()]);
    }

    #[test]
    fn redcore_write_emits_one_blind_control_event_and_survives_reopen() {
        use crate::graph::capability::GraphRead;
        use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
        use crate::graph::record::NodeRecord;
        use crate::graph_store::{RedCoreDurability, RedCoreGraphStore, RedCoreOptions};
        use serde_json::json;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("cp1-events-{stamp}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        {
            let mut store = RedCoreGraphStore::open(
                &dir,
                RedCoreOptions {
                    durability: RedCoreDurability::AofAlways,
                    ..RedCoreOptions::default()
                },
            )
            .unwrap();
            store.set_hook_tenant("t1");
            store.set_hook_actor("actor-a");
            store
                .commit_batch(GraphMutationBatch::new([GraphMutation::NodeUpsert(
                    NodeRecord::new("n1", ["gclba_property"], json!({"title": "Parcel"})),
                )]))
                .unwrap();

            let events: Vec<_> = GraphRead::query_nodes(
                &store,
                crate::graph::query::NodeQuery::label(CONTROL_EVENT_LABEL),
            )
            .into_iter()
            .map(|node| ControlEventRecord::from_node(&node).unwrap())
            .collect();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].compiled, "gclba_property.created");
            assert_eq!(events[0].affected_node_ids, vec!["n1".to_string()]);
            assert!(events[0].is_blind());
            assert_eq!(events[0].actor.as_deref(), Some("actor-a"));
            let encoded = serde_json::to_value(&events[0]).unwrap();
            assert!(encoded.get("title").is_none());
        }

        let reopened = RedCoreGraphStore::open(
            &dir,
            RedCoreOptions {
                durability: RedCoreDurability::AofAlways,
                ..RedCoreOptions::default()
            },
        )
        .unwrap();
        let events: Vec<_> = GraphRead::query_nodes(
            &reopened,
            crate::graph::query::NodeQuery::label(CONTROL_EVENT_LABEL),
        )
        .into_iter()
        .map(|node| ControlEventRecord::from_node(&node).unwrap())
        .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].compiled, "gclba_property.created");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
