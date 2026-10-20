use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::capability::GraphStore;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::hash::Hash;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphWriteResult {
    pub id: String,
    pub version: u64,
    pub checksum: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", content = "record", rename_all = "snake_case")]
pub enum GraphMutation {
    NodeUpsert(NodeRecord),
    EdgeUpsert(EdgeRecord),
    NodeDelete(String),
    EdgeDelete(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ConditionalNodePropertyUpdate {
    pub node_id: String,
    pub property: String,
    pub expected: Value,
    pub new_value: Value,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GraphMutationBatch {
    pub mutations: Vec<GraphMutation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_root: Option<Hash>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum RecordedRedCoreMutation {
    NodeUpsert { record: NodeRecord },
    EdgeUpsert { record: EdgeRecord },
    NodeDelete { id: String, version: u64 },
    EdgeDelete { id: String, version: u64 },
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RecordedRedCoreMutationBatch {
    pub mutations: Vec<RecordedRedCoreMutation>,
}

impl GraphMutationBatch {
    pub fn new(mutations: impl IntoIterator<Item = GraphMutation>) -> Self {
        Self {
            mutations: mutations.into_iter().collect(),
            expected_root: None,
        }
    }

    #[must_use]
    pub fn with_expected_root(mut self, expected_root: Hash) -> Self {
        self.expected_root = Some(expected_root);
        self
    }
}

pub(crate) fn prepare_conditional_batch<S: GraphStore>(
    store: &S,
    conditions: Vec<ConditionalNodePropertyUpdate>,
    batch: GraphMutationBatch,
) -> GraphStoreResult<Option<GraphMutationBatch>> {
    let expected_root = batch.expected_root;
    let mut conditional_nodes = BTreeMap::<String, NodeRecord>::new();
    let mut condition_keys = BTreeSet::<(String, String)>::new();
    let mut condition_owned_node_ids = BTreeSet::<String>::new();
    let mut check_only_node_ids = BTreeSet::<String>::new();
    let mut check_only_properties = BTreeMap::<String, BTreeMap<String, Value>>::new();
    let mut guarded_replacements = BTreeMap::<String, BTreeMap<String, Value>>::new();
    for condition in conditions {
        let property = condition.property.trim();
        if property.is_empty() {
            return Err(GraphStoreError::empty_field("node.property"));
        }
        let condition_key = (condition.node_id.clone(), property.to_string());
        if !condition_keys.insert(condition_key) {
            return Err(GraphStoreError::new(
                "duplicate_conditional_node_property",
                format!(
                    "conditional batch repeats node {} property {property}",
                    condition.node_id
                ),
            ));
        }
        let had_prepared_node = conditional_nodes.contains_key(&condition.node_id);
        let mut node = if let Some(node) = conditional_nodes.remove(&condition.node_id) {
            node
        } else {
            let Some(node) = store
                .get_node_record(&condition.node_id)
                .filter(|node| !node.tombstone)
            else {
                return Ok(None);
            };
            node
        };
        let current = node
            .properties
            .as_object()
            .and_then(|properties| properties.get(property))
            .cloned()
            .unwrap_or(Value::Null);
        if current != condition.expected {
            return Ok(None);
        }
        if condition.new_value == condition.expected {
            check_only_node_ids.insert(condition.node_id.clone());
            check_only_properties
                .entry(condition.node_id.clone())
                .or_default()
                .insert(property.to_string(), condition.expected);
            if had_prepared_node {
                conditional_nodes.insert(condition.node_id, node);
            }
            continue;
        }
        condition_owned_node_ids.insert(condition.node_id.clone());
        let properties = node.properties.as_object_mut().ok_or_else(|| {
            GraphStoreError::new(
                "invalid_node_properties",
                "node properties must be an object",
            )
        })?;
        if condition.new_value.is_null() {
            properties.remove(property);
        } else {
            properties.insert(property.to_string(), condition.new_value.clone());
        }
        guarded_replacements
            .entry(condition.node_id.clone())
            .or_default()
            .insert(property.to_string(), condition.new_value);
        conditional_nodes.insert(condition.node_id, node);
    }

    let mut caller_upserted_condition_nodes = BTreeSet::<String>::new();
    let mut remaining_mutations = Vec::with_capacity(batch.mutations.len());
    for mutation in batch.mutations {
        match mutation {
            GraphMutation::NodeUpsert(caller) => {
                if let Some(checked) = check_only_properties.get(&caller.id) {
                    let caller_properties = caller.properties.as_object().ok_or_else(|| {
                        GraphStoreError::new(
                            "invalid_node_properties",
                            "caller node properties must be an object",
                        )
                    })?;
                    let preserves_checked_properties = !caller.tombstone
                        && checked.iter().all(|(property, expected)| {
                            caller_properties
                                .get(property)
                                .map_or_else(|| expected.is_null(), |actual| actual == expected)
                        });
                    if !preserves_checked_properties {
                        return Err(GraphStoreError::new(
                            "conditional_batch_node_conflict",
                            format!(
                                "caller upsert changes a check-only property on node {}",
                                caller.id
                            ),
                        ));
                    }
                }
                if !condition_owned_node_ids.contains(&caller.id) {
                    remaining_mutations.push(GraphMutation::NodeUpsert(caller));
                    continue;
                }
                if caller.tombstone || !caller_upserted_condition_nodes.insert(caller.id.clone()) {
                    return Err(GraphStoreError::new(
                        "conditional_batch_node_conflict",
                        format!(
                            "caller batch has an ambiguous upsert for condition-owned node {}",
                            caller.id
                        ),
                    ));
                }
                let guarded = conditional_nodes.get_mut(&caller.id).ok_or_else(|| {
                    GraphStoreError::new(
                        "conditional_batch_internal_error",
                        format!("condition-owned node {} was not prepared", caller.id),
                    )
                })?;
                let guarded_properties = guarded.properties.as_object_mut().ok_or_else(|| {
                    GraphStoreError::new(
                        "invalid_node_properties",
                        "node properties must be an object",
                    )
                })?;
                let caller_properties = caller.properties.as_object().ok_or_else(|| {
                    GraphStoreError::new(
                        "invalid_node_properties",
                        "caller node properties must be an object",
                    )
                })?;
                for (property, value) in caller_properties {
                    guarded_properties.insert(property.clone(), value.clone());
                }
                for (property, value) in &guarded_replacements[&caller.id] {
                    if value.is_null() {
                        guarded_properties.remove(property);
                    } else {
                        guarded_properties.insert(property.clone(), value.clone());
                    }
                }
                let mut labels = guarded.labels.iter().cloned().collect::<BTreeSet<_>>();
                labels.extend(caller.labels);
                guarded.labels = labels.into_iter().collect();
                guarded.version = 0;
                guarded.tombstone = false;
                guarded.content_hash = None;
                guarded.parent_hashes = caller.parent_hashes;
            }
            GraphMutation::NodeDelete(node_id)
                if check_only_node_ids.contains(&node_id)
                    || condition_owned_node_ids.contains(&node_id) =>
            {
                return Err(GraphStoreError::new(
                    "conditional_batch_node_conflict",
                    format!("caller batch also mutates condition-owned node {node_id}"),
                ));
            }
            mutation => remaining_mutations.push(mutation),
        }
    }

    let mut mutations = Vec::with_capacity(conditional_nodes.len() + remaining_mutations.len());
    mutations.extend(
        conditional_nodes
            .into_values()
            .map(GraphMutation::NodeUpsert),
    );
    mutations.extend(remaining_mutations);
    let mut prepared = GraphMutationBatch::new(mutations);
    prepared.expected_root = expected_root;
    Ok(Some(prepared))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphTransaction {
    pub txn_id: u64,
    pub graph_version: u64,
    pub writes: Vec<GraphWriteResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::in_memory::InMemoryGraphStore;
    use serde_json::json;

    #[test]
    fn conditional_preparation_preserves_the_frozen_merge_shape() -> GraphStoreResult<()> {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(NodeRecord::new(
            "node:guarded",
            ["Base"],
            json!({"keep": 1, "state": "before"}),
        ))?;

        let caller = NodeRecord::new(
            "node:guarded",
            ["Incoming"],
            json!({"caller": true, "state": "caller"}),
        );
        let prepared = prepare_conditional_batch(
            &store,
            vec![ConditionalNodePropertyUpdate {
                node_id: "node:guarded".to_string(),
                property: "state".to_string(),
                expected: json!("before"),
                new_value: json!("guarded"),
            }],
            GraphMutationBatch::new([GraphMutation::NodeUpsert(caller)]),
        )?
        .expect("matching guard prepares the batch");

        assert_eq!(
            serde_json::to_value(prepared).expect("prepared batch serializes"),
            json!({
                "mutations": [{
                    "op": "node_upsert",
                    "record": {
                        "id": "node:guarded",
                        "labels": ["Base", "Incoming"],
                        "properties": {
                            "caller": true,
                            "keep": 1,
                            "state": "guarded"
                        },
                        "version": 0,
                        "tombstone": false
                    }
                }]
            })
        );
        Ok(())
    }

    #[test]
    fn equal_conditional_replacement_is_a_read_only_guard() -> GraphStoreResult<()> {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(NodeRecord::new(
            "node:schema",
            ["Schema"],
            json!({"revision": "schema:1"}),
        ))?;
        let unrelated = NodeRecord::new("node:record", ["Record"], json!({"title": "next"}));
        let condition = ConditionalNodePropertyUpdate {
            node_id: "node:schema".to_string(),
            property: "revision".to_string(),
            expected: json!("schema:1"),
            new_value: json!("schema:1"),
        };

        let prepared = prepare_conditional_batch(
            &store,
            vec![condition.clone()],
            GraphMutationBatch::new([GraphMutation::NodeUpsert(unrelated)]),
        )?
        .expect("matching read-only guard prepares the caller batch");
        assert_eq!(prepared.mutations.len(), 1);
        assert!(matches!(
            prepared.mutations.as_slice(),
            [GraphMutation::NodeUpsert(node)] if node.id == "node:record"
        ));

        store.upsert_node(NodeRecord::new(
            "node:schema",
            ["Schema"],
            json!({"revision": "schema:2"}),
        ))?;
        assert!(
            prepare_conditional_batch(&store, vec![condition], GraphMutationBatch::default())?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn check_only_condition_after_mutation_keeps_the_prepared_node() -> GraphStoreResult<()> {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(NodeRecord::new(
            "node:schema",
            ["Schema"],
            json!({"revision": "schema:1", "schema_kind": "case"}),
        ))?;
        let prepared = prepare_conditional_batch(
            &store,
            vec![
                ConditionalNodePropertyUpdate {
                    node_id: "node:schema".into(),
                    property: "revision".into(),
                    expected: json!("schema:1"),
                    new_value: json!("schema:2"),
                },
                ConditionalNodePropertyUpdate {
                    node_id: "node:schema".into(),
                    property: "schema_kind".into(),
                    expected: json!("case"),
                    new_value: json!("case"),
                },
            ],
            GraphMutationBatch::default(),
        )?
        .expect("both guards match");

        assert!(matches!(
            prepared.mutations.as_slice(),
            [GraphMutation::NodeUpsert(node)]
                if node.properties["revision"] == json!("schema:2")
                    && node.properties["schema_kind"] == json!("case")
        ));
        Ok(())
    }

    #[test]
    fn caller_upsert_must_preserve_every_check_only_property() -> GraphStoreResult<()> {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(NodeRecord::new(
            "node:schema",
            ["Schema"],
            json!({"revision": "schema:1", "label": "Before"}),
        ))?;
        let condition = ConditionalNodePropertyUpdate {
            node_id: "node:schema".into(),
            property: "revision".into(),
            expected: json!("schema:1"),
            new_value: json!("schema:1"),
        };
        let preserving = NodeRecord::new(
            "node:schema",
            ["Schema"],
            json!({"revision": "schema:1", "label": "After"}),
        );
        let prepared = prepare_conditional_batch(
            &store,
            vec![condition.clone()],
            GraphMutationBatch::new([GraphMutation::NodeUpsert(preserving)]),
        )?
        .expect("the checked property is preserved");
        assert!(matches!(
            prepared.mutations.as_slice(),
            [GraphMutation::NodeUpsert(node)] if node.properties["label"] == json!("After")
        ));

        for violating in [
            NodeRecord::new(
                "node:schema",
                ["Schema"],
                json!({"revision": "schema:2", "label": "Changed"}),
            ),
            NodeRecord::new("node:schema", ["Schema"], json!({"label": "Omitted"})),
        ] {
            let error = prepare_conditional_batch(
                &store,
                vec![condition.clone()],
                GraphMutationBatch::new([GraphMutation::NodeUpsert(violating)]),
            )
            .unwrap_err();
            assert_eq!(error.code, "conditional_batch_node_conflict");
        }

        let delete_error = prepare_conditional_batch(
            &store,
            vec![condition],
            GraphMutationBatch::new([GraphMutation::NodeDelete("node:schema".into())]),
        )
        .unwrap_err();
        assert_eq!(delete_error.code, "conditional_batch_node_conflict");
        Ok(())
    }

    #[test]
    fn mutation_batch_without_expected_root_remains_wire_compatible() {
        let decoded: GraphMutationBatch =
            serde_json::from_value(json!({ "mutations": [] })).expect("legacy batch decodes");

        assert_eq!(decoded.expected_root, None);
        assert_eq!(
            serde_json::to_value(decoded).expect("batch serializes"),
            json!({ "mutations": [] })
        );
    }

    #[test]
    fn conditional_preparation_preserves_expected_root() -> GraphStoreResult<()> {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(NodeRecord::new(
            "node:guarded",
            ["Base"],
            json!({"state": "before"}),
        ))?;
        let expected_root = Hash::of(b"expected-root");

        let prepared = prepare_conditional_batch(
            &store,
            vec![ConditionalNodePropertyUpdate {
                node_id: "node:guarded".to_string(),
                property: "state".to_string(),
                expected: json!("before"),
                new_value: json!("after"),
            }],
            GraphMutationBatch::default().with_expected_root(expected_root),
        )?
        .expect("matching guard prepares the batch");

        assert_eq!(prepared.expected_root, Some(expected_root));
        Ok(())
    }
}
