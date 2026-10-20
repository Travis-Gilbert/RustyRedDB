use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

use theorem_surface_contracts::sync::SyncCursor;

pub type GraphStoreResult<T> = Result<T, GraphStoreError>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphStoreError {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<SyncCursor>,
}

impl GraphStoreError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            cursor: None,
        }
    }

    pub fn stale_write_base(cursor: SyncCursor) -> Self {
        Self {
            code: "stale_write_base".into(),
            message: format!(
                "write expected a different graph root than txn {}",
                cursor.txn_id
            ),
            cursor: Some(cursor),
        }
    }

    pub(crate) fn empty_field(field: &str) -> Self {
        Self::new("empty_graph_field", format!("{field} is required"))
    }

    pub(crate) fn missing_endpoint(edge_id: &str, endpoint: &str, node_id: &str) -> Self {
        Self::new(
            "missing_graph_endpoint",
            format!("edge {edge_id} {endpoint} endpoint {node_id} does not exist"),
        )
    }

    pub(crate) fn empty_transaction() -> Self {
        Self::new(
            "empty_graph_transaction",
            "graph transaction requires at least one mutation",
        )
    }

    /// Public so downstream crates retiring legacy on-disk artifacts report the
    /// same `redcore_io_error` code as the core journal paths.
    pub fn io(action: impl AsRef<str>, err: impl std::fmt::Display) -> Self {
        Self::new("redcore_io_error", format!("{}: {err}", action.as_ref()))
    }

    pub(crate) fn tombstoned_endpoint(edge_id: &str, endpoint: &str, node_id: &str) -> Self {
        Self::new(
            "tombstoned_graph_endpoint",
            format!("edge {edge_id} {endpoint} endpoint {node_id} is tombstoned"),
        )
    }

    #[cfg(feature = "redis-store")]
    pub(crate) fn invalid_record(record_type: &str, id: &str, err: impl std::fmt::Display) -> Self {
        Self::new(
            "invalid_graph_record",
            format!("{record_type} {id} could not be decoded: {err}"),
        )
    }
}

impl fmt::Display for GraphStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl Error for GraphStoreError {}

pub(crate) fn vector_pin_cleanup_error(
    context: &str,
    publish_error: GraphStoreError,
    release_error: GraphStoreError,
) -> GraphStoreError {
    GraphStoreError::new(
        "vector_payload_publication_and_unpin_failed",
        format!(
            "{context} failed ({}: {}); releasing its vector payload pins also failed ({}: {})",
            publish_error.code, publish_error.message, release_error.code, release_error.message
        ),
    )
}

pub(crate) fn is_recoverable_orphan_edge(error: &GraphStoreError) -> bool {
    // Ported from RustyRed-Graph-Database 365d073. Recovery may legally skip an
    // edge whose endpoint node was never durably written (only shadowed) or was
    // tombstoned; both surface as these codes.
    matches!(
        error.code.as_str(),
        "missing_graph_endpoint" | "tombstoned_graph_endpoint"
    )
}

#[cfg(feature = "redis-store")]
impl From<redis::RedisError> for GraphStoreError {
    fn from(err: redis::RedisError) -> Self {
        Self::new("redis_graph_store_error", err.to_string())
    }
}
