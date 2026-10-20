use imbl::OrdMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::stream::StreamStore;
use crate::symbolic::canonical_json;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThgState {
    pub seq: u64,
    pub runs: OrdMap<String, RunState>,
    pub contexts: OrdMap<String, ContextState>,
    pub patches: OrdMap<String, PatchState>,
    /// Append-only coordination streams: the live awareness channel that
    /// replaces turn-start room polling. Persisted and hashed with the rest of
    /// the harness state so cursors survive restarts and every publish advances
    /// the state hash.
    #[serde(default)]
    pub streams: StreamStore,
}

impl ThgState {
    pub fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    pub fn hash(&self) -> GraphStoreResult<String> {
        stable_hash(self)
    }

    pub fn graph(&self) -> (Vec<ThgNode>, Vec<ThgEdge>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for run in self.runs.values() {
            nodes.push(run.node());
            for step in &run.steps {
                nodes.push(step.node(&run.run_id));
                edges.push(ThgEdge {
                    from_id: run.run_id.clone(),
                    edge_type: "HAS_STEP".to_string(),
                    to_id: step.step_id.clone(),
                    properties: json!({ "index": step.index }),
                });
            }
        }
        for context in self.contexts.values() {
            nodes.push(context.node());
        }
        for patch in self.patches.values() {
            nodes.push(patch.node());
            if !patch.run_id.is_empty() {
                edges.push(ThgEdge {
                    from_id: patch.run_id.clone(),
                    edge_type: "PROPOSED_PATCH".to_string(),
                    to_id: patch.patch_id.clone(),
                    properties: Value::Object(Default::default()),
                });
            }
        }

        (nodes, edges)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunState {
    pub run_id: String,
    pub task: String,
    pub actor: String,
    pub scope: Value,
    pub status: String,
    pub steps: Vec<StepState>,
}

impl RunState {
    pub fn node(&self) -> ThgNode {
        ThgNode {
            id: self.run_id.clone(),
            labels: vec!["AgentRun".to_string()],
            properties: json!({
                "task": self.task,
                "actor": self.actor,
                "scope": self.scope,
                "status": self.status,
            }),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct StepState {
    pub step_id: String,
    pub kind: String,
    pub index: i64,
    pub payload: Value,
}

impl StepState {
    pub fn node(&self, run_id: &str) -> ThgNode {
        ThgNode {
            id: self.step_id.clone(),
            labels: vec!["Step".to_string()],
            properties: json!({
                "run_id": run_id,
                "kind": self.kind,
                "index": self.index,
                "payload": self.payload,
            }),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContextState {
    pub artifact_id: String,
    pub status: String,
    pub sections: Value,
    pub token_ledger: Value,
}

impl ContextState {
    pub fn node(&self) -> ThgNode {
        ThgNode {
            id: self.artifact_id.clone(),
            labels: vec!["ContextArtifact".to_string()],
            properties: json!({
                "status": self.status,
                "sections": self.sections,
                "token_ledger": self.token_ledger,
            }),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PatchState {
    pub patch_id: String,
    pub run_id: String,
    pub status: String,
    pub patch: Value,
    pub findings: Value,
}

impl PatchState {
    pub fn node(&self) -> ThgNode {
        ThgNode {
            id: self.patch_id.clone(),
            labels: vec!["MemoryPatch".to_string()],
            properties: json!({
                "status": self.status,
                "patch": self.patch,
                "findings": self.findings,
            }),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThgNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThgEdge {
    pub from_id: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub to_id: String,
    pub properties: Value,
}

pub fn stable_hash<T: Serialize>(value: T) -> GraphStoreResult<String> {
    let logical_value = serde_json::to_value(value).map_err(|error| {
        GraphStoreError::new(
            "stable_hash_serialize",
            format!("could not serialize stable-hash input: {error}"),
        )
    })?;
    let canonical = canonical_json(&logical_value).map_err(|message| {
        GraphStoreError::new(
            "stable_hash_serialize",
            format!("could not canonicalize stable-hash input: {message}"),
        )
    })?;
    Ok(sha256_address(canonical.as_bytes()))
}

/// Frozen raw-serde preimage for the versioned-graph v1 format.
///
/// Phase A makes hashing fallible and canonical without changing existing
/// graph roots. The v1 graph format historically hashed derived-struct field
/// order, so its object, tree, boundary, and commit identities must retain that
/// exact preimage until the intentional format bump in Phase C.
pub(crate) fn stable_hash_graph_v1<T: Serialize>(value: T) -> GraphStoreResult<String> {
    let encoded = serde_json::to_vec(&value).map_err(|error| {
        GraphStoreError::new(
            "stable_hash_serialize",
            format!("could not serialize versioned-graph v1 hash input: {error}"),
        )
    })?;
    Ok(sha256_address(&encoded))
}

fn sha256_address(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serializer;
    use serde_json::Map;

    struct FailingSerialize(&'static str);

    impl Serialize for FailingSerialize {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Err(serde::ser::Error::custom(self.0))
        }
    }

    #[test]
    fn stable_hash_propagates_serialization_failures_without_shared_addresses() {
        let first = stable_hash(FailingSerialize("first serialization failure")).unwrap_err();
        let second = stable_hash(FailingSerialize("second serialization failure")).unwrap_err();

        assert_eq!(first.code, "stable_hash_serialize");
        assert_eq!(second.code, "stable_hash_serialize");
        assert!(first.message.contains("first serialization failure"));
        assert!(second.message.contains("second serialization failure"));
        assert_ne!(first.message, second.message);
    }

    #[test]
    fn stable_hash_sorts_nested_object_keys() {
        let mut nested = Map::new();
        nested.insert("z".to_string(), json!(1));
        nested.insert("a".to_string(), json!(2));
        let mut outer = Map::new();
        outer.insert("nested".to_string(), Value::Object(nested));

        assert_eq!(
            stable_hash(Value::Object(outer)).unwrap(),
            stable_hash(json!({"nested": {"a": 2, "z": 1}})).unwrap()
        );
    }

    #[test]
    fn versioned_graph_v1_hash_preserves_derived_struct_field_order() {
        #[derive(Serialize)]
        struct Fixture<'a> {
            protocol_version: &'a str,
            level: u32,
            first_key: &'a str,
            last_key: &'a str,
            entries: [u8; 0],
            children: [u8; 0],
        }

        assert_eq!(
            stable_hash_graph_v1(Fixture {
                protocol_version: "rustyred-versioned-graph-v1",
                level: 0,
                first_key: "",
                last_key: "",
                entries: [],
                children: [],
            })
            .unwrap(),
            "sha256:e70bdbcacf6dd146f39a28edc8ef8537a5571d7d82f453527616318c205005cf"
        );
    }
}
