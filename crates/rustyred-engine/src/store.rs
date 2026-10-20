use std::sync::Arc;

#[cfg(feature = "redis-store")]
use crate::errors::ThgError;
use crate::errors::ThgResult;
use crate::state::ThgState;

pub trait ThgStore {
    fn load(&self) -> ThgResult<ThgState>;
    fn load_snapshot(&self) -> ThgResult<Arc<ThgState>> {
        self.load().map(Arc::new)
    }
    fn save(&mut self, state: &ThgState) -> ThgResult<()>;
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryThgStore {
    state: Arc<ThgState>,
}

impl InMemoryThgStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Arc<ThgState> {
        Arc::clone(&self.state)
    }
}

impl ThgStore for InMemoryThgStore {
    fn load(&self) -> ThgResult<ThgState> {
        Ok(self.state.as_ref().clone())
    }

    fn load_snapshot(&self) -> ThgResult<Arc<ThgState>> {
        Ok(self.snapshot())
    }

    fn save(&mut self, state: &ThgState) -> ThgResult<()> {
        self.state = Arc::new(state.clone());
        Ok(())
    }
}

#[cfg(feature = "redis-store")]
#[derive(Clone, Debug)]
pub struct RedisThgStore {
    client: redis::Client,
    key: String,
}

#[cfg(feature = "redis-store")]
impl RedisThgStore {
    pub fn new(redis_url: &str, key: impl Into<String>) -> redis::RedisResult<Self> {
        Ok(Self {
            client: redis::Client::open(redis_url)?,
            key: key.into(),
        })
    }

    pub fn ping(&self) -> redis::RedisResult<()> {
        let mut connection = self.client.get_connection()?;
        redis::cmd("PING").query::<String>(&mut connection)?;
        Ok(())
    }
}

#[cfg(feature = "redis-store")]
impl ThgStore for RedisThgStore {
    fn load(&self) -> ThgResult<ThgState> {
        let mut connection = self.client.get_connection().map_err(|error| {
            ThgError::new(
                "store_unavailable",
                format!("Redis connection failed: {error}"),
            )
        })?;
        let raw: redis::RedisResult<Option<String>> =
            redis::cmd("GET").arg(&self.key).query(&mut connection);
        let raw = raw.map_err(|error| {
            ThgError::new("store_unavailable", format!("Redis GET failed: {error}"))
        })?;
        match raw {
            Some(raw) => serde_json::from_str::<ThgState>(&raw).map_err(|error| {
                ThgError::new(
                    "state_corrupt",
                    format!("stored THG state is invalid JSON: {error}"),
                )
            }),
            None => Ok(ThgState::default()),
        }
    }

    fn save(&mut self, state: &ThgState) -> ThgResult<()> {
        let mut connection = self.client.get_connection().map_err(|error| {
            ThgError::new(
                "persist_failed",
                format!("Redis connection failed: {error}"),
            )
        })?;
        let raw = serde_json::to_string(state).map_err(|error| {
            ThgError::new(
                "persist_failed",
                format!("THG state serialization failed: {error}"),
            )
        })?;
        redis::cmd("SET")
            .arg(&self.key)
            .arg(raw)
            .query::<()>(&mut connection)
            .map_err(|error| {
                ThgError::new("persist_failed", format!("Redis SET failed: {error}"))
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryThgStore, ThgStore};
    use crate::state::RunState;
    use std::sync::Arc;

    #[test]
    fn in_memory_store_round_trips_state() {
        let mut store = InMemoryThgStore::new();
        let mut state = store.load().unwrap();
        state.runs.insert(
            "run:redis-contract".to_string(),
            RunState {
                run_id: "run:redis-contract".to_string(),
                task: "persist THG".to_string(),
                actor: "agent".to_string(),
                scope: serde_json::json!({ "source": "test" }),
                status: "running".to_string(),
                steps: Vec::new(),
            },
        );

        store.save(&state).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.runs["run:redis-contract"].task, "persist THG");
    }

    #[test]
    fn in_memory_store_load_snapshot_is_arc_shared() {
        let mut store = InMemoryThgStore::new();
        let mut state = store.load().unwrap();
        state.runs.insert(
            "run:shared".to_string(),
            RunState {
                run_id: "run:shared".to_string(),
                task: "shared snapshot".to_string(),
                actor: "agent".to_string(),
                scope: serde_json::json!({}),
                status: "running".to_string(),
                steps: Vec::new(),
            },
        );
        store.save(&state).unwrap();

        let first = store.load_snapshot().unwrap();
        let second = store.load_snapshot().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.runs["run:shared"].task, "shared snapshot");
    }
}
