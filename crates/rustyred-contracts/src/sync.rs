//! Store-neutral graph synchronization wire types.
//!
//! A renderer never owns graph truth. It holds a cursor, asks what changed,
//! and renders a runtime projection over those changes. The concrete graph
//! store stays below this crate so the same contracts compile for native and
//! wasm consumers.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// One durable graph version.
///
/// The content-addressed root is authoritative. `txn_id` is the inexpensive
/// linear position used to retrieve entries on the current branch.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(
    export,
    export_to = "theorem-surface-contracts/sync/",
    optional_fields = nullable
)]
pub struct SyncCursor {
    pub root: [u8; 32],
    pub txn_id: u64,
}

impl SyncCursor {
    /// The cursor before a durable graph has published its first root.
    pub const GENESIS: Self = Self {
        root: [0; 32],
        txn_id: 0,
    };
}

/// The graph record operation committed at one cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(
    export,
    export_to = "theorem-surface-contracts/sync/",
    optional_fields = nullable
)]
pub enum MutationKind {
    NodeUpserted,
    NodeDeleted,
    EdgeUpserted,
    EdgeDeleted,
}

/// Identity and shape of one committed graph mutation.
///
/// Payloads are intentionally absent. A mirror hydrates the named record from
/// the authoritative store instead of trusting a notification as a snapshot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(
    export,
    export_to = "theorem-surface-contracts/sync/",
    optional_fields = nullable
)]
pub struct MutationLogEntry {
    pub txn_id: u64,
    pub root: [u8; 32],
    pub kind: MutationKind,
    pub id: String,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub changed_props: Vec<String>,
    pub tenant: String,
    pub committed_at_ms: u64,
    pub sequence: u64,
}

/// The complete incremental answer, or an explicit demand for rehydration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
#[ts(
    export,
    export_to = "theorem-surface-contracts/sync/",
    optional_fields = nullable
)]
pub enum ChangesSince {
    Changes {
        cursor: SyncCursor,
        entries: Vec<MutationLogEntry>,
    },
    Truncated {
        cursor: SyncCursor,
    },
}

/// Lightweight notification that tells a client to pull from its own cursor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(
    export,
    export_to = "theorem-surface-contracts/sync/",
    optional_fields = nullable
)]
pub struct SyncPing {
    pub root: [u8; 32],
    pub txn_id: u64,
    pub tenant: String,
}

impl SyncPing {
    #[must_use]
    pub const fn cursor(&self) -> SyncCursor {
        SyncCursor {
            root: self.root,
            txn_id: self.txn_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_since_never_collapses_truncation_into_an_empty_delta() {
        let cursor = SyncCursor {
            root: [7; 32],
            txn_id: 42,
        };
        let value = serde_json::to_value(ChangesSince::Truncated { cursor }).unwrap();
        assert_eq!(value["status"], "truncated");
        assert_eq!(value["cursor"]["txnId"], 42);
        assert!(value.get("entries").is_none());
    }
}
