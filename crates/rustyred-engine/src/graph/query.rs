use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use theorem_oplog::LogPosition;

use crate::graph::record::{Direction, EpistemicType};
use crate::query_receipt::PinnedReadSet;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NeighborQuery {
    pub node_id: String,
    pub direction: Direction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// When true, neighbor results include nodes past their TTL expiration.
    /// Default false (TTL-expired targets are filtered out). Used by audit
    /// and forensics callers; normal callers should leave this false.
    #[serde(default)]
    pub include_expired: bool,
}

impl NeighborQuery {
    pub fn out(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            direction: Direction::Out,
            edge_type: None,
            include_expired: false,
        }
    }

    pub fn in_(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            direction: Direction::In,
            edge_type: None,
            include_expired: false,
        }
    }

    pub fn with_edge_type(mut self, edge_type: impl Into<String>) -> Self {
        let edge_type = edge_type.into();
        if !edge_type.trim().is_empty() {
            self.edge_type = Some(edge_type);
        }
        self
    }

    pub fn with_include_expired(mut self, include_expired: bool) -> Self {
        self.include_expired = include_expired;
        self
    }

    /// Capture this set-valued query's reads at one immutable log prefix.
    ///
    /// Graph mutation write sets currently publish record identities rather
    /// than adjacency-index keys. A wildcard is therefore the narrowest sound
    /// footprint: it may produce false positives, but cannot miss a newly
    /// matching edge or a target-node visibility change.
    pub fn capture_reads(&self, position: LogPosition) -> PinnedReadSet {
        PinnedReadSet::wildcard(position)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NeighborHit {
    pub edge_id: String,
    pub node_id: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic_type: Option<EpistemicType>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NodeQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// When true, query results include nodes past their TTL expiration.
    /// Default false (TTL-expired nodes are filtered out). Used by audit
    /// and forensics callers; normal callers should leave this false.
    #[serde(default)]
    pub include_expired: bool,
}

impl NodeQuery {
    pub fn label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..Self::default()
        }
    }

    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        if limit > 0 {
            self.limit = Some(limit);
        }
        self
    }

    pub fn with_include_expired(mut self, include_expired: bool) -> Self {
        self.include_expired = include_expired;
        self
    }

    /// Capture this set-valued query's reads at one immutable log prefix.
    ///
    /// Label/property index membership is not yet represented in graph write
    /// sets. Until it is, using a wildcard is deliberately conservative and
    /// makes subscription/cache false positives visible in the shared trace
    /// instead of permitting false negatives.
    pub fn capture_reads(&self, position: LogPosition) -> PinnedReadSet {
        PinnedReadSet::wildcard(position)
    }

    pub(crate) fn normalized_label(&self) -> Option<String> {
        self.label
            .as_deref()
            .map(str::trim)
            .filter(|label| !label.is_empty())
            .map(str::to_string)
    }

    /// The number of nodes this query may return.
    ///
    /// `limit` is the caller's explicit ceiling. `None` means the caller asked
    /// for no ceiling, and the read is unbounded.
    ///
    /// This used to answer `100` for `None`, which silently truncated every
    /// unbounded read in the product. The damage was not cosmetic: a run's event
    /// ledger is read with an unbounded query, so the hundredth event was the
    /// last one any caller could see, and a run's terminal `RUN.CLOSED` sits
    /// past it. The stream tail therefore never observed a terminal status, the
    /// SSE connection never closed, and no chat turn could ever settle -- on the
    /// wire it looked like a stream that simply stopped. It truncated list reads
    /// the same way, so any table built from this query showed its first hundred
    /// rows and called the set complete.
    ///
    /// A caller that wants a hundred rows asks for a hundred rows. See
    /// `GraphSnapshot::query_nodes`, which used to spell the same silent default
    /// a second time.
    pub fn bounded_limit(&self) -> usize {
        self.limit.filter(|limit| *limit > 0).unwrap_or(usize::MAX)
    }
}
