//! Durable, bounded mutation log minted at the RedCore commit point.
//!
//! Identity and shape only. A consumer that needs a record reads the node.
//! Wire types live in `theorem-surface-contracts`; this module stores and
//! encodes them inside `RedCoreJournalState`.

use std::collections::HashMap;

use theorem_surface_contracts::sync::{
    ChangesSince, MutationKind as WireKind, MutationLogEntry, SyncCursor,
};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::hash::Hash;
use crate::hooks::MutationKind;

/// Oldest retained window. Dropping the front advances `log_base_*`.
pub const MAX_MUTATION_LOG_ENTRIES: usize = 10_000;

const GENESIS_SYNC_ROOT_PREIMAGE: &[u8] = b"redcore.sync.root.v1:empty";

/// Bounded append-only log plus the ancestry map used by `changes_since`.
#[derive(Clone, Debug, PartialEq)]
pub struct MutationLog {
    pub log_base_txn_id: u64,
    pub log_base_root: [u8; 32],
    pub entries: Vec<MutationLogEntry>,
}

impl MutationLog {
    /// Empty log whose next retained txn will be `txn_id + 1`.
    #[must_use]
    pub fn empty_after(logical_root: Hash, txn_id: u64) -> Self {
        Self {
            log_base_txn_id: txn_id.saturating_add(1),
            log_base_root: *logical_root.as_bytes(),
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub fn genesis() -> Self {
        Self::empty_after(genesis_sync_root(), 0)
    }

    #[must_use]
    pub fn contains_txn(&self, txn_id: u64) -> bool {
        self.entries.iter().any(|entry| entry.txn_id == txn_id)
    }

    pub fn append_entries(&mut self, entries: Vec<MutationLogEntry>) {
        if entries.is_empty() {
            return;
        }
        if self.entries.is_empty() {
            if let Some(first) = entries.first() {
                self.log_base_txn_id = first.txn_id;
                self.log_base_root = first.root;
            }
        }
        self.entries.extend(entries);
        while self.entries.len() > MAX_MUTATION_LOG_ENTRIES {
            self.entries.remove(0);
            if let Some(first) = self.entries.first() {
                self.log_base_txn_id = first.txn_id;
                self.log_base_root = first.root;
            }
        }
    }

    #[must_use]
    pub fn parent_map(&self, head: Hash) -> HashMap<Hash, Hash> {
        let mut parents = HashMap::new();
        let mut previous: Option<Hash> = None;
        for entry in &self.entries {
            let root = Hash::from_bytes(entry.root);
            if let Some(parent) = previous {
                if parent != root {
                    parents.insert(root, parent);
                }
            } else {
                let base = Hash::from_bytes(self.log_base_root);
                if base != root {
                    parents.insert(root, base);
                }
            }
            previous = Some(root);
        }
        if let Some(last) = previous {
            if last != head {
                parents.insert(head, last);
            }
        }
        parents
    }

    pub fn encode_tail(&self) -> GraphStoreResult<Vec<u8>> {
        let mut output = Vec::new();
        output.extend_from_slice(&self.log_base_txn_id.to_le_bytes());
        output.extend_from_slice(&self.log_base_root);
        let count = u32::try_from(self.entries.len()).map_err(|_| {
            GraphStoreError::new(
                "redcore_mutation_log_length_overflow",
                "mutation log entry count exceeds u32",
            )
        })?;
        output.extend_from_slice(&count.to_le_bytes());
        for entry in &self.entries {
            output.extend_from_slice(&entry.txn_id.to_le_bytes());
            output.extend_from_slice(&entry.root);
            output.push(wire_kind_byte(entry.kind));
            push_counted_string(&mut output, &entry.id, "mutation log id")?;
            push_counted_strings(&mut output, &entry.labels, "mutation log labels")?;
            push_counted_strings(
                &mut output,
                &entry.changed_props,
                "mutation log changed props",
            )?;
            push_counted_string(&mut output, &entry.tenant, "mutation log tenant")?;
            output.extend_from_slice(&entry.committed_at_ms.to_le_bytes());
            output.extend_from_slice(&entry.sequence.to_le_bytes());
        }
        Ok(output)
    }

    pub fn decode_tail(bytes: &[u8]) -> GraphStoreResult<Self> {
        let mut cursor = TailCursor { bytes, offset: 0 };
        let log_base_txn_id = cursor.u64("log_base_txn_id")?;
        let log_base_root = cursor.hash_bytes("log_base_root")?;
        let count = cursor.u32("mutation log entry count")?;
        if count as usize > MAX_MUTATION_LOG_ENTRIES {
            return Err(GraphStoreError::new(
                "redcore_mutation_log_count_limit",
                format!("mutation log has {count} entries, exceeding {MAX_MUTATION_LOG_ENTRIES}"),
            ));
        }
        let mut entries = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let txn_id = cursor.u64("entry txn_id")?;
            let root = cursor.hash_bytes("entry root")?;
            let kind = wire_kind_from_byte(cursor.byte("entry kind")?)?;
            let id = cursor.string("entry id")?;
            let labels = cursor.strings("entry labels")?;
            let changed_props = cursor.strings("entry changed_props")?;
            let tenant = cursor.string("entry tenant")?;
            let committed_at_ms = cursor.u64("entry committed_at_ms")?;
            let sequence = cursor.u64("entry sequence")?;
            entries.push(MutationLogEntry {
                txn_id,
                root,
                kind,
                id,
                labels,
                changed_props,
                tenant,
                committed_at_ms,
                sequence,
            });
        }
        cursor.finish()?;
        Ok(Self {
            log_base_txn_id,
            log_base_root,
            entries,
        })
    }
}

#[must_use]
pub fn genesis_sync_root() -> Hash {
    Hash::of(GENESIS_SYNC_ROOT_PREIMAGE)
}

#[must_use]
pub fn memory_sync_root(parent: Hash, txn_id: u64) -> Hash {
    let mut bytes = Vec::with_capacity(GENESIS_SYNC_ROOT_PREIMAGE.len() + 8 + 32);
    bytes.extend_from_slice(b"redcore.sync.root.v1");
    bytes.extend_from_slice(&txn_id.to_le_bytes());
    bytes.extend_from_slice(parent.as_bytes());
    Hash::of(&bytes)
}

#[must_use]
pub fn wire_kind(kind: MutationKind) -> WireKind {
    match kind {
        MutationKind::NodeUpserted => WireKind::NodeUpserted,
        MutationKind::NodeDeleted => WireKind::NodeDeleted,
        MutationKind::EdgeUpserted => WireKind::EdgeUpserted,
        MutationKind::EdgeDeleted => WireKind::EdgeDeleted,
    }
}

#[must_use]
pub fn is_sync_ancestor(parents: &HashMap<Hash, Hash>, candidate: Hash, head: Hash) -> bool {
    if candidate == head {
        return true;
    }
    let mut current = head;
    for _ in 0..1_000_000 {
        let Some(parent) = parents.get(&current).copied() else {
            return false;
        };
        if parent == candidate {
            return true;
        }
        if parent == current {
            return false;
        }
        current = parent;
    }
    false
}

pub fn changes_since(
    log: &MutationLog,
    parents: &HashMap<Hash, Hash>,
    current: SyncCursor,
    since: &SyncCursor,
) -> ChangesSince {
    if since.root == current.root && since.txn_id == current.txn_id {
        return ChangesSince::Changes {
            cursor: current,
            entries: Vec::new(),
        };
    }
    let since_root = Hash::from_bytes(since.root);
    let head = Hash::from_bytes(current.root);
    if !is_sync_ancestor(parents, since_root, head) {
        return ChangesSince::Truncated { cursor: current };
    }
    if log.entries.is_empty() {
        if since.txn_id < current.txn_id {
            return ChangesSince::Truncated { cursor: current };
        }
        return ChangesSince::Changes {
            cursor: current,
            entries: Vec::new(),
        };
    }
    let first_retained = log.entries[0].txn_id;
    if first_retained > since.txn_id.saturating_add(1) {
        return ChangesSince::Truncated { cursor: current };
    }
    let mut entries: Vec<MutationLogEntry> = log
        .entries
        .iter()
        .filter(|entry| entry.txn_id > since.txn_id)
        .cloned()
        .collect();
    entries.sort_by_key(|entry| (entry.txn_id, entry.sequence));
    ChangesSince::Changes {
        cursor: current,
        entries,
    }
}

fn wire_kind_byte(kind: WireKind) -> u8 {
    match kind {
        WireKind::NodeUpserted => 0,
        WireKind::NodeDeleted => 1,
        WireKind::EdgeUpserted => 2,
        WireKind::EdgeDeleted => 3,
    }
}

fn wire_kind_from_byte(byte: u8) -> GraphStoreResult<WireKind> {
    match byte {
        0 => Ok(WireKind::NodeUpserted),
        1 => Ok(WireKind::NodeDeleted),
        2 => Ok(WireKind::EdgeUpserted),
        3 => Ok(WireKind::EdgeDeleted),
        other => Err(GraphStoreError::new(
            "redcore_mutation_log_kind_invalid",
            format!("mutation log kind byte {other} is not 0..3"),
        )),
    }
}

fn push_counted_string(output: &mut Vec<u8>, value: &str, field: &str) -> GraphStoreResult<()> {
    let length = u32::try_from(value.len()).map_err(|_| {
        GraphStoreError::new(
            "redcore_mutation_log_length_overflow",
            format!("{field} exceeds u32"),
        )
    })?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_counted_strings(
    output: &mut Vec<u8>,
    values: &[String],
    field: &str,
) -> GraphStoreResult<()> {
    let count = u32::try_from(values.len()).map_err(|_| {
        GraphStoreError::new(
            "redcore_mutation_log_length_overflow",
            format!("{field} count exceeds u32"),
        )
    })?;
    output.extend_from_slice(&count.to_le_bytes());
    for value in values {
        push_counted_string(output, value, field)?;
    }
    Ok(())
}

struct TailCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> TailCursor<'a> {
    fn take(&mut self, length: usize, field: &str) -> GraphStoreResult<&'a [u8]> {
        let end = self.offset.checked_add(length).ok_or_else(|| {
            GraphStoreError::new(
                "redcore_mutation_log_truncated",
                format!("{field} offset overflowed"),
            )
        })?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            GraphStoreError::new(
                "redcore_mutation_log_truncated",
                format!("mutation log ended while reading {field}"),
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self, field: &str) -> GraphStoreResult<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &str) -> GraphStoreResult<u32> {
        let raw: [u8; 4] = self.take(4, field)?.try_into().expect("fixed-width slice");
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self, field: &str) -> GraphStoreResult<u64> {
        let raw: [u8; 8] = self.take(8, field)?.try_into().expect("fixed-width slice");
        Ok(u64::from_le_bytes(raw))
    }

    fn hash_bytes(&mut self, field: &str) -> GraphStoreResult<[u8; 32]> {
        let raw: [u8; 32] = self.take(32, field)?.try_into().expect("fixed-width slice");
        Ok(raw)
    }

    fn string(&mut self, field: &str) -> GraphStoreResult<String> {
        let length = usize::try_from(self.u32(field)?).expect("u32 fits usize");
        String::from_utf8(self.take(length, field)?.to_vec()).map_err(|error| {
            GraphStoreError::new(
                "redcore_mutation_log_utf8",
                format!("{field} is not UTF-8: {error}"),
            )
        })
    }

    fn strings(&mut self, field: &str) -> GraphStoreResult<Vec<String>> {
        let count = usize::try_from(self.u32(field)?).expect("u32 fits usize");
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string(field)?);
        }
        Ok(values)
    }

    fn finish(&self) -> GraphStoreResult<()> {
        if self.offset != self.bytes.len() {
            return Err(GraphStoreError::new(
                "redcore_mutation_log_trailing_bytes",
                format!(
                    "mutation log has {} trailing bytes",
                    self.bytes.len() - self.offset
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_round_trips() {
        let log = MutationLog::genesis();
        let bytes = log.encode_tail().unwrap();
        assert_eq!(MutationLog::decode_tail(&bytes).unwrap(), log);
    }

    #[test]
    fn bounded_log_drops_oldest() {
        let mut log = MutationLog::genesis();
        let mut entries = Vec::new();
        for i in 0..(MAX_MUTATION_LOG_ENTRIES as u64 + 3) {
            entries.push(MutationLogEntry {
                txn_id: i + 1,
                root: [(i % 255) as u8; 32],
                kind: WireKind::NodeUpserted,
                id: format!("n{i}"),
                labels: vec!["Item".into()],
                changed_props: Vec::new(),
                tenant: "t".into(),
                committed_at_ms: i,
                sequence: 0,
            });
        }
        log.append_entries(entries);
        assert_eq!(log.entries.len(), MAX_MUTATION_LOG_ENTRIES);
        assert_eq!(log.log_base_txn_id, 4);
        assert_eq!(log.entries.first().unwrap().id, "n3");
    }

    #[test]
    fn changes_since_truncates_when_the_retained_window_has_a_gap() {
        let mut log = MutationLog::genesis();
        log.append_entries(vec![MutationLogEntry {
            txn_id: 4,
            root: [4; 32],
            kind: WireKind::NodeUpserted,
            id: "n4".into(),
            labels: vec!["Item".into()],
            changed_props: Vec::new(),
            tenant: "t".into(),
            committed_at_ms: 4,
            sequence: 0,
        }]);
        let head = Hash::from_bytes([4; 32]);
        let parents = log.parent_map(head);
        let current = SyncCursor {
            root: [4; 32],
            txn_id: 4,
        };
        let since = SyncCursor {
            root: [4; 32],
            txn_id: 1,
        };
        match changes_since(&log, &parents, current, &since) {
            ChangesSince::Truncated { cursor } => assert_eq!(cursor, current),
            ChangesSince::Changes { .. } => panic!("a gapped window must truncate"),
        }
    }
}
