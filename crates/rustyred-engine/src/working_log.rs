use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph::ttl::now_ms;

const MAX_WORKING_LOG_LEN: usize = 100_000;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkingLogEventKind {
    Episode,
    Access,
    Mutation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WorkingLogEvent {
    pub cursor: u64,
    pub ts_ms: i64,
    pub kind: WorkingLogEventKind,
    pub row_id: Option<String>,
    pub payload: Value,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct WorkingLog {
    events: Vec<WorkingLogEvent>,
    next_cursor: u64,
}

impl WorkingLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(
        &mut self,
        kind: WorkingLogEventKind,
        row_id: Option<String>,
        payload: Value,
    ) -> WorkingLogEvent {
        self.next_cursor += 1;
        let event = WorkingLogEvent {
            cursor: self.next_cursor,
            ts_ms: now_ms(),
            kind,
            row_id,
            payload,
        };
        self.events.push(event.clone());
        if self.events.len() > MAX_WORKING_LOG_LEN {
            let excess = self.events.len() - MAX_WORKING_LOG_LEN;
            self.events.drain(..excess);
        }
        event
    }

    pub fn append_episode(&mut self, row_id: impl Into<String>, payload: Value) -> WorkingLogEvent {
        self.append(WorkingLogEventKind::Episode, Some(row_id.into()), payload)
    }

    pub fn append_access(&mut self, row_id: impl Into<String>) -> WorkingLogEvent {
        self.append(
            WorkingLogEventKind::Access,
            Some(row_id.into()),
            json!({ "access": true }),
        )
    }

    pub fn append_mutation(
        &mut self,
        row_id: impl Into<String>,
        payload: Value,
    ) -> WorkingLogEvent {
        self.append(WorkingLogEventKind::Mutation, Some(row_id.into()), payload)
    }

    pub fn subscribe_after(&self, cursor: u64, limit: usize) -> Vec<WorkingLogEvent> {
        let start = self.events.partition_point(|event| event.cursor <= cursor);
        self.events[start..]
            .iter()
            .take(if limit == 0 { usize::MAX } else { limit })
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Persist the full working log to a JSON file for crash recovery.
    pub fn write_to_path(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = path.with_extension(
            path.extension()
                .map(|extension| format!("{}.tmp", extension.to_string_lossy()))
                .unwrap_or_else(|| "tmp".to_string()),
        );
        match fs::remove_file(&tmp_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(&json)?;
        file.sync_all()?;
        drop(file);
        #[cfg(windows)]
        let _ = fs::remove_file(path);
        fs::rename(&tmp_path, path)?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    /// Load a working log from a JSON file previously written by [`write_to_path`].
    /// Returns `None` when the file does not exist (first run).
    pub fn read_from_path(path: &std::path::Path) -> std::io::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json)
                .map(Some)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RecencyCounter {
    rows: BTreeMap<String, RecencyState>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecencyState {
    pub row_id: String,
    pub last_accessed_ms: i64,
    pub access_count: u64,
    pub recency_score: f64,
}

impl RecencyCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn touch(&mut self, row_id: impl Into<String>, at_ms: i64) -> RecencyState {
        let row_id = row_id.into();
        let state = self.rows.entry(row_id.clone()).or_insert(RecencyState {
            row_id,
            last_accessed_ms: at_ms,
            access_count: 0,
            recency_score: 0.0,
        });
        state.last_accessed_ms = state.last_accessed_ms.max(at_ms);
        state.access_count = state.access_count.saturating_add(1);
        state.recency_score = recency_score(state.access_count, state.last_accessed_ms);
        state.clone()
    }

    pub fn state(&self, row_id: &str) -> Option<&RecencyState> {
        self.rows.get(row_id)
    }

    pub fn rows_aged_before(&self, cutoff_ms: i64) -> Vec<String> {
        self.rows
            .values()
            .filter(|state| state.last_accessed_ms <= cutoff_ms)
            .map(|state| state.row_id.clone())
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TemporalFact {
    pub fact_id: String,
    pub t_valid: Option<i64>,
    pub t_invalid: Option<i64>,
    pub t_created: i64,
    pub t_expired: Option<i64>,
    pub payload: Value,
}

impl TemporalFact {
    pub fn new(fact_id: impl Into<String>, t_created: i64, payload: Value) -> Self {
        Self {
            fact_id: fact_id.into(),
            t_valid: Some(t_created),
            t_invalid: None,
            t_created,
            t_expired: None,
            payload,
        }
    }

    pub fn invalidate(&self, t_invalid: i64, t_expired: i64) -> Self {
        let mut invalidated = self.clone();
        invalidated.t_invalid = Some(t_invalid);
        invalidated.t_expired = Some(t_expired);
        invalidated
    }

    pub fn is_valid_at(&self, ts_ms: i64) -> bool {
        self.t_valid.map(|start| ts_ms >= start).unwrap_or(true)
            && self.t_invalid.map(|end| ts_ms < end).unwrap_or(true)
    }
}

fn recency_score(access_count: u64, last_accessed_ms: i64) -> f64 {
    (access_count as f64).ln_1p() + (last_accessed_ms.max(0) as f64 / 1_000.0)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn working_log_subscriber_resumes_from_cursor_in_order() {
        let mut log = WorkingLog::new();
        let first = log.append_episode("episode:1", json!({ "text": "hello" }));
        log.append_access("episode:1");
        log.append_mutation("episode:1", json!({ "field": "status" }));

        let events = log.subscribe_after(first.cursor, 0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].cursor, first.cursor + 1);
        assert_eq!(events[0].kind, WorkingLogEventKind::Access);
        assert_eq!(events[1].kind, WorkingLogEventKind::Mutation);

        let limited = log.subscribe_after(0, 2);
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].kind, WorkingLogEventKind::Episode);
    }

    #[test]
    fn recency_counter_tracks_hot_rows_and_aged_drop_candidates() {
        let mut counter = RecencyCounter::new();
        counter.touch("row:old", 10);
        counter.touch("row:hot", 100);
        counter.touch("row:hot", 110);

        assert_eq!(counter.state("row:hot").unwrap().access_count, 2);
        assert!(counter.state("row:hot").unwrap().recency_score > 0.0);
        assert_eq!(counter.rows_aged_before(50), vec!["row:old".to_string()]);
    }

    #[test]
    fn temporal_fact_invalidation_preserves_validity_history() {
        let fact = TemporalFact::new("fact:1", 100, json!({ "claim": "enabled" }));
        let invalidated = fact.invalidate(200, 250);

        assert!(invalidated.is_valid_at(150));
        assert!(!invalidated.is_valid_at(220));
        assert_eq!(invalidated.t_valid, Some(100));
        assert_eq!(invalidated.t_invalid, Some(200));
        assert_eq!(invalidated.t_created, 100);
        assert_eq!(invalidated.t_expired, Some(250));
    }

    #[test]
    fn working_log_round_trip_write_read_equality() {
        let mut log = WorkingLog::new();
        log.append_episode("ep:a", json!({ "text": "hello" }));
        log.append_access("ep:a");
        log.append_mutation("ep:a", json!({ "field": "x" }));

        let dir = std::env::temp_dir().join("thg-core-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("working_log_roundtrip.json");
        log.write_to_path(&path).unwrap();

        let restored = WorkingLog::read_from_path(&path)
            .unwrap()
            .expect("log file should exist after write");
        assert_eq!(restored, log);
    }

    #[test]
    fn working_log_read_nonexistent_returns_none() {
        let path = std::path::Path::new("/tmp/thg-core-nonexistent-file.json");
        let _ = std::fs::remove_file(path);
        let result = WorkingLog::read_from_path(path).unwrap();
        assert!(result.is_none(), "nonexistent file should return None");
    }

    #[test]
    fn working_log_retention_keeps_latest_cursor_window() {
        let mut log = WorkingLog::new();
        for index in 0..=MAX_WORKING_LOG_LEN {
            log.append_episode(format!("episode:{index}"), json!({}));
        }
        assert_eq!(log.len(), MAX_WORKING_LOG_LEN);
        let first = log.subscribe_after(0, 1);
        assert_eq!(first[0].cursor, 2);
    }
}
