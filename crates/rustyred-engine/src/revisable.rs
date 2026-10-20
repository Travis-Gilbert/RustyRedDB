//! Shared revisable draft/publish primitive (SPEC-THEOREM-CONTROL-PRIMITIVES-1.0 CP4).
//!
//! Programs, object types, and views share one lifecycle. A fifth independent
//! draft/publish implementation is a defect.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::mutation::{GraphMutation, GraphMutationBatch};
use crate::graph::record::{EdgeRecord, NodeRecord};

pub const REVISION_LABEL: &str = "Revision";
pub const SUPERSEDES_REVISION: &str = "SUPERSEDES_REVISION";
pub const REVISION_OF_LINEAGE: &str = "REVISION_OF_LINEAGE";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionState {
    Draft,
    Published,
    Superseded,
    Retired,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RevisionId(pub String);

impl RevisionId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for RevisionId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for RevisionId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

pub trait Revisable: Clone {
    fn revision_of(&self) -> Option<RevisionId>;
    fn revision_state(&self) -> RevisionState;
    fn content_anchor(&self) -> &str;
    fn lineage_id(&self) -> &str;
    fn revision_id(&self) -> RevisionId;
    fn with_revision_state(&self, state: RevisionState) -> Self;
    fn with_revision_of(&self, parent: Option<RevisionId>) -> Self;
    fn with_revision_id(&self, id: RevisionId) -> Self;
    fn to_node(&self) -> NodeRecord;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RevisionError {
    pub code: &'static str,
    pub message: String,
}

impl RevisionError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new("revision_publish_conflict", message)
    }
}

impl std::fmt::Display for RevisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RevisionError {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PublishReceipt {
    pub published_revision_id: RevisionId,
    pub superseded_revision_id: Option<RevisionId>,
    pub lineage_id: String,
    pub content_anchor: String,
}

/// Draft an editable copy that shares lineage with a published revision.
pub fn draft_from<T: Revisable>(published: &T) -> Result<T, RevisionError> {
    if published.revision_state() != RevisionState::Published {
        return Err(RevisionError::new(
            "revision_draft_requires_published",
            format!(
                "draft_from requires Published, got {:?}",
                published.revision_state()
            ),
        ));
    }
    let draft_id = RevisionId(format!(
        "{}:draft:{}",
        published.lineage_id(),
        published.content_anchor()
    ));
    Ok(published
        .with_revision_id(draft_id)
        .with_revision_of(Some(published.revision_id()))
        .with_revision_state(RevisionState::Draft))
}

/// Publish a draft. When `prior` is supplied, mark it superseded and write the
/// lineage edge. Exactly one published revision per lineage is the caller's
/// invariant; use [`publish_if_head`] for concurrent safety.
pub fn publish<T: Revisable>(
    draft: T,
    prior: Option<&T>,
) -> Result<(T, PublishReceipt, GraphMutationBatch), RevisionError> {
    if draft.revision_state() != RevisionState::Draft {
        return Err(RevisionError::new(
            "revision_publish_requires_draft",
            format!("publish requires Draft, got {:?}", draft.revision_state()),
        ));
    }
    if let Some(prior) = prior {
        if prior.revision_state() != RevisionState::Published {
            return Err(RevisionError::new(
                "revision_publish_prior_not_published",
                format!(
                    "prior revision must be Published, got {:?}",
                    prior.revision_state()
                ),
            ));
        }
        if prior.lineage_id() != draft.lineage_id() {
            return Err(RevisionError::new(
                "revision_lineage_mismatch",
                format!(
                    "draft lineage {} does not match prior {}",
                    draft.lineage_id(),
                    prior.lineage_id()
                ),
            ));
        }
    }

    let published = match prior {
        Some(prior) => draft
            .with_revision_of(Some(prior.revision_id()))
            .with_revision_state(RevisionState::Published),
        None => draft.with_revision_state(RevisionState::Published),
    };
    let batch = match prior {
        Some(prior) => supersede(prior, &published),
        None => GraphMutationBatch::new([
            GraphMutation::NodeUpsert(lineage_node(published.lineage_id())),
            GraphMutation::NodeUpsert(published.to_node()),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                format!(
                    "revision:lineage:{}:{}",
                    published.lineage_id(),
                    published.revision_id().0
                ),
                published.revision_id().0.clone(),
                REVISION_OF_LINEAGE,
                published.lineage_id().to_string(),
                json!({}),
            )),
        ]),
    };
    let receipt = PublishReceipt {
        published_revision_id: published.revision_id(),
        superseded_revision_id: prior.map(|item| item.revision_id()),
        lineage_id: published.lineage_id().to_string(),
        content_anchor: published.content_anchor().to_string(),
    };
    Ok((published, receipt, batch))
}

/// Concurrent publish helper: if `expected_published` is still the live head,
/// supersede it; otherwise return a structured conflict without mutating.
pub fn publish_if_head<T: Revisable>(
    draft: T,
    expected_published: Option<&T>,
    live_published: Option<&T>,
) -> Result<(T, PublishReceipt, GraphMutationBatch), RevisionError> {
    match (expected_published, live_published) {
        (None, None) => publish(draft, None),
        (Some(expected), Some(live))
            if expected.revision_id() == live.revision_id()
                && live.revision_state() == RevisionState::Published =>
        {
            publish(draft, Some(live))
        }
        (Some(expected), Some(live)) => Err(RevisionError::conflict(format!(
            "lineage {} expected published {} but live head is {}",
            draft.lineage_id(),
            expected.revision_id().0,
            live.revision_id().0
        ))),
        (None, Some(live)) => Err(RevisionError::conflict(format!(
            "lineage {} already has published head {}",
            draft.lineage_id(),
            live.revision_id().0
        ))),
        (Some(expected), None) => Err(RevisionError::conflict(format!(
            "lineage {} expected published {} but no live head exists",
            draft.lineage_id(),
            expected.revision_id().0
        ))),
    }
}

pub fn supersede<T: Revisable>(prev: &T, next: &T) -> GraphMutationBatch {
    GraphMutationBatch::new([
        GraphMutation::NodeUpsert(lineage_node(next.lineage_id())),
        GraphMutation::NodeUpsert(
            prev.with_revision_state(RevisionState::Superseded)
                .to_node(),
        ),
        GraphMutation::NodeUpsert(next.with_revision_state(RevisionState::Published).to_node()),
        GraphMutation::EdgeUpsert(EdgeRecord::new(
            format!(
                "revision:supersedes:{}:{}",
                prev.revision_id().0,
                next.revision_id().0
            ),
            next.revision_id().0.clone(),
            SUPERSEDES_REVISION,
            prev.revision_id().0.clone(),
            json!({ "lineage_id": next.lineage_id() }),
        )),
        GraphMutation::EdgeUpsert(EdgeRecord::new(
            format!(
                "revision:lineage:{}:{}",
                next.lineage_id(),
                next.revision_id().0
            ),
            next.revision_id().0.clone(),
            REVISION_OF_LINEAGE,
            next.lineage_id().to_string(),
            json!({}),
        )),
    ])
}

fn lineage_node(lineage_id: &str) -> NodeRecord {
    NodeRecord::new(
        lineage_id.to_string(),
        ["RevisionLineage"],
        json!({ "lineage_id": lineage_id }),
    )
}

/// Shared revision envelope used by programs, object types, and views.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RevisionEnvelope {
    pub revision_id: RevisionId,
    pub lineage_id: String,
    pub revision_state: RevisionState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_of: Option<RevisionId>,
    pub content_anchor: String,
    pub kind: String,
    #[serde(default)]
    pub body: Value,
}

impl Revisable for RevisionEnvelope {
    fn revision_of(&self) -> Option<RevisionId> {
        self.revision_of.clone()
    }

    fn revision_state(&self) -> RevisionState {
        self.revision_state
    }

    fn content_anchor(&self) -> &str {
        &self.content_anchor
    }

    fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    fn revision_id(&self) -> RevisionId {
        self.revision_id.clone()
    }

    fn with_revision_state(&self, state: RevisionState) -> Self {
        let mut next = self.clone();
        next.revision_state = state;
        next
    }

    fn with_revision_of(&self, parent: Option<RevisionId>) -> Self {
        let mut next = self.clone();
        next.revision_of = parent;
        next
    }

    fn with_revision_id(&self, id: RevisionId) -> Self {
        let mut next = self.clone();
        next.revision_id = id;
        next
    }

    fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            self.revision_id.0.clone(),
            [REVISION_LABEL, self.kind.as_str()],
            serde_json::to_value(self).unwrap_or_else(|_| json!({})),
        )
    }
}

impl RevisionEnvelope {
    pub fn published(kind: &str, lineage_id: &str, content_anchor: &str, body: Value) -> Self {
        Self {
            revision_id: RevisionId(format!("{lineage_id}:rev:{content_anchor}")),
            lineage_id: lineage_id.to_string(),
            revision_state: RevisionState::Published,
            revision_of: None,
            content_anchor: content_anchor.to_string(),
            kind: kind.to_string(),
            body,
        }
    }

    pub fn from_node(node: &NodeRecord) -> GraphStoreResult<Self> {
        serde_json::from_value(node.properties.clone())
            .map_err(|error| GraphStoreError::new("revision_decode", error.to_string()))
    }
}

/// Retrofit adapters: programs, object types, and views all share [`RevisionEnvelope`].
pub mod retrofit {
    use super::*;

    pub fn program(lineage_id: &str, content_anchor: &str, source: &str) -> RevisionEnvelope {
        RevisionEnvelope::published(
            "program",
            lineage_id,
            content_anchor,
            json!({ "source": source }),
        )
    }

    pub fn object_type(lineage_id: &str, content_anchor: &str, name: &str) -> RevisionEnvelope {
        RevisionEnvelope::published(
            "object_type",
            lineage_id,
            content_anchor,
            json!({ "name": name }),
        )
    }

    pub fn view(lineage_id: &str, content_anchor: &str, name: &str) -> RevisionEnvelope {
        RevisionEnvelope::published("view", lineage_id, content_anchor, json!({ "name": name }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::capability::{GraphRead, GraphWrite};
    use crate::graph::in_memory::InMemoryGraphStore;

    fn shared_suite(kind: &str, make: fn(&str, &str, &str) -> RevisionEnvelope) {
        let published = make("lineage-a", "anchor-1", "alpha");
        assert_eq!(published.kind, kind);
        assert_eq!(published.revision_state(), RevisionState::Published);

        let draft = draft_from(&published).unwrap();
        assert_eq!(draft.revision_state(), RevisionState::Draft);
        assert_eq!(draft.lineage_id(), published.lineage_id());
        assert_eq!(draft.revision_of(), Some(published.revision_id()));

        let (next, receipt, batch) = publish(draft, Some(&published)).unwrap();
        assert_eq!(next.revision_state(), RevisionState::Published);
        assert_eq!(
            receipt.superseded_revision_id,
            Some(published.revision_id())
        );
        assert!(!batch.mutations.is_empty());

        let mut store = InMemoryGraphStore::new();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => {
                    GraphWrite::upsert_node(&mut store, node).unwrap();
                }
                GraphMutation::EdgeUpsert(edge) => {
                    GraphWrite::upsert_edge(&mut store, edge).unwrap();
                }
                other => panic!("unexpected mutation {other:?}"),
            }
        }

        // Superseded prior remains readable via explicit id.
        let loaded = GraphRead::get_node_record(&store, published.revision_id().as_str())
            .expect("superseded revision readable");
        let loaded = RevisionEnvelope::from_node(&loaded).unwrap();
        assert_eq!(loaded.revision_state(), RevisionState::Superseded);

        // Concurrent publish conflict.
        let draft_b = draft_from(&published).unwrap();
        let err = publish_if_head(draft_b, Some(&published), Some(&next)).unwrap_err();
        assert_eq!(err.code, "revision_publish_conflict");
    }

    #[test]
    fn programs_object_types_and_views_share_suite() {
        shared_suite("program", |lineage, anchor, body| {
            retrofit::program(lineage, anchor, body)
        });
        shared_suite("object_type", |lineage, anchor, body| {
            retrofit::object_type(lineage, anchor, body)
        });
        shared_suite("view", |lineage, anchor, body| {
            retrofit::view(lineage, anchor, body)
        });
    }

    #[test]
    fn draft_from_rejects_non_published() {
        let draft = retrofit::program("l", "a", "src").with_revision_state(RevisionState::Draft);
        assert_eq!(
            draft_from(&draft).unwrap_err().code,
            "revision_draft_requires_published"
        );
    }
}
