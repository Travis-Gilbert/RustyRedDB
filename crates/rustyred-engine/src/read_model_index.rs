use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context_view::HydrationHandle;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;
use crate::query_receipt::ReceiptScope;
use crate::state::stable_hash;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompositeIndexDefinition {
    pub id: String,
    pub target_label: String,
    pub scope_fields: Vec<String>,
    pub key_fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_id: Option<String>,
}

impl CompositeIndexDefinition {
    pub fn new(
        id: impl Into<String>,
        target_label: impl Into<String>,
        scope_fields: impl IntoIterator<Item = impl Into<String>>,
        key_fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            target_label: target_label.into(),
            scope_fields: scope_fields.into_iter().map(Into::into).collect(),
            key_fields: key_fields.into_iter().map(Into::into).collect(),
            manifest_id: None,
        }
    }

    pub fn with_manifest_id(mut self, manifest_id: impl Into<String>) -> Self {
        self.manifest_id = Some(manifest_id.into());
        self
    }

    pub fn validate(&self) -> GraphStoreResult<()> {
        if self.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_composite_index",
                "composite index id is required",
            ));
        }
        if self.target_label.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_composite_index",
                "composite index target label is required",
            ));
        }
        if self.scope_fields.is_empty() || self.key_fields.is_empty() {
            return Err(GraphStoreError::new(
                "invalid_composite_index",
                "composite index requires scope fields and key fields",
            ));
        }
        if !self.key_fields.starts_with(&self.scope_fields) {
            return Err(GraphStoreError::new(
                "composite_index_not_scope_first",
                "composite index key fields must start with scope fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CompositeIndexKey {
    pub index_id: String,
    pub values: Vec<(String, String)>,
}

impl CompositeIndexKey {
    pub fn from_properties(
        index_id: impl Into<String>,
        fields: &[String],
        properties: &Value,
    ) -> GraphStoreResult<Self> {
        let mut values = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(token) = properties.get(field).and_then(index_value_token) else {
                return Err(GraphStoreError::new(
                    "missing_composite_key_field",
                    format!("composite key field {field} is missing or not indexable"),
                ));
            };
            values.push((field.clone(), token));
        }
        Ok(Self {
            index_id: index_id.into(),
            values,
        })
    }

    pub fn left_prefix(&self, len: usize) -> Self {
        Self {
            index_id: self.index_id.clone(),
            values: self.values.iter().take(len).cloned().collect(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompositeIndexEntry {
    pub node_id: String,
    pub source_version: u64,
    pub hydration_handle: HydrationHandle,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompositeIndex {
    definitions: BTreeMap<String, CompositeIndexDefinition>,
    exact: BTreeMap<CompositeIndexKey, BTreeMap<String, CompositeIndexEntry>>,
}

impl CompositeIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_definition(
        &mut self,
        definition: CompositeIndexDefinition,
    ) -> GraphStoreResult<()> {
        definition.validate()?;
        if self.definitions.contains_key(&definition.id) {
            return Err(GraphStoreError::new(
                "composite_index_exists",
                format!("composite index {} is already registered", definition.id),
            ));
        }
        self.definitions.insert(definition.id.clone(), definition);
        Ok(())
    }

    pub fn upsert_node(
        &mut self,
        index_id: &str,
        node: &NodeRecord,
        scope: &ReceiptScope,
    ) -> GraphStoreResult<()> {
        let definition = self.definitions.get(index_id).ok_or_else(|| {
            GraphStoreError::new(
                "composite_index_not_found",
                format!("composite index {index_id} is not registered"),
            )
        })?;
        let target_label = definition.target_label.clone();
        let key_fields = definition.key_fields.clone();
        if !node.labels.iter().any(|label| label == &target_label) || node.tombstone {
            self.remove_node(index_id, &node.id);
            return Ok(());
        }
        self.remove_node(index_id, &node.id);
        let properties = merge_scope_properties(scope, &node.properties);
        let key = CompositeIndexKey::from_properties(index_id, &key_fields, &properties)?;
        self.exact.entry(key).or_default().insert(
            node.id.clone(),
            CompositeIndexEntry {
                node_id: node.id.clone(),
                source_version: node.version,
                hydration_handle: HydrationHandle::new(
                    node.id.clone(),
                    target_label.clone(),
                    node.version,
                    format!("graph://{}/{}", target_label, node.id),
                ),
            },
        );
        Ok(())
    }

    pub fn remove_node(&mut self, index_id: &str, node_id: &str) -> bool {
        let mut removed = false;
        self.exact.retain(|key, entries| {
            if key.index_id == index_id && entries.remove(node_id).is_some() {
                removed = true;
            }
            !entries.is_empty()
        });
        removed
    }

    pub fn query_exact(&self, key: &CompositeIndexKey) -> Vec<CompositeIndexEntry> {
        self.exact
            .get(key)
            .map(|entries| entries.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn query_left_prefix(&self, prefix: &CompositeIndexKey) -> Vec<CompositeIndexEntry> {
        let mut out = BTreeMap::new();
        for (key, entries) in &self.exact {
            if key.index_id == prefix.index_id && key.values.starts_with(&prefix.values) {
                for (node_id, entry) in entries {
                    out.insert(node_id.clone(), entry.clone());
                }
            }
        }
        out.into_values().collect()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialPredicateOp {
    Equals,
    NotEquals,
    Exists,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PartialPredicateClause {
    pub field: String,
    pub op: PartialPredicateOp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

impl PartialPredicateClause {
    pub fn equals(field: impl Into<String>, value: Value) -> Self {
        Self {
            field: field.into(),
            op: PartialPredicateOp::Equals,
            value: Some(value),
        }
    }

    pub fn exists(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            op: PartialPredicateOp::Exists,
            value: None,
        }
    }

    pub fn evaluate(&self, properties: &Value) -> bool {
        let actual = properties.get(&self.field);
        match self.op {
            PartialPredicateOp::Equals => actual == self.value.as_ref(),
            PartialPredicateOp::NotEquals => actual != self.value.as_ref(),
            PartialPredicateOp::Exists => actual.is_some(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PartialIndexDefinition {
    pub id: String,
    pub target_label: String,
    pub clauses: Vec<PartialPredicateClause>,
    pub predicate_hash: String,
}

impl PartialIndexDefinition {
    pub fn new(
        id: impl Into<String>,
        target_label: impl Into<String>,
        clauses: Vec<PartialPredicateClause>,
    ) -> GraphStoreResult<Self> {
        let mut definition = Self {
            id: id.into(),
            target_label: target_label.into(),
            clauses,
            predicate_hash: String::new(),
        };
        definition.refresh_predicate_hash()?;
        Ok(definition)
    }

    pub fn refresh_predicate_hash(&mut self) -> GraphStoreResult<()> {
        self.predicate_hash = stable_hash(&self.clauses)?;
        Ok(())
    }

    pub fn evaluate_node(&self, node: &NodeRecord) -> bool {
        !node.tombstone
            && node.labels.iter().any(|label| label == &self.target_label)
            && self
                .clauses
                .iter()
                .all(|clause| clause.evaluate(&node.properties))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PartialIndexEntry {
    pub node_id: String,
    pub source_version: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PartialIndex {
    pub definition: PartialIndexDefinition,
    predicate_hash: String,
    entries: BTreeMap<String, PartialIndexEntry>,
}

impl PartialIndex {
    pub fn new(definition: PartialIndexDefinition) -> Self {
        Self {
            predicate_hash: definition.predicate_hash.clone(),
            definition,
            entries: BTreeMap::new(),
        }
    }

    pub fn upsert_node(&mut self, node: &NodeRecord) -> bool {
        if self.definition.evaluate_node(node) {
            self.entries.insert(
                node.id.clone(),
                PartialIndexEntry {
                    node_id: node.id.clone(),
                    source_version: node.version,
                },
            );
            true
        } else {
            self.entries.remove(&node.id);
            false
        }
    }

    pub fn predicate_drifted(&self) -> bool {
        self.predicate_hash != self.definition.predicate_hash
    }

    pub fn ids(&self) -> BTreeSet<String> {
        self.entries.keys().cloned().collect()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CoveringIndexDefinition {
    pub id: String,
    pub target_label: String,
    pub covering_fields: Vec<String>,
}

impl CoveringIndexDefinition {
    pub fn new(
        id: impl Into<String>,
        target_label: impl Into<String>,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            target_label: target_label.into(),
            covering_fields: fields.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CoveringRow {
    pub object_id: String,
    pub source_version: u64,
    pub fields: BTreeMap<String, Value>,
    pub stale: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CoveringIndex {
    pub definition: CoveringIndexDefinition,
    rows: BTreeMap<String, CoveringRow>,
}

impl CoveringIndex {
    pub fn new(definition: CoveringIndexDefinition) -> Self {
        Self {
            definition,
            rows: BTreeMap::new(),
        }
    }

    pub fn upsert_node(&mut self, node: &NodeRecord) -> Option<CoveringRow> {
        if node.tombstone
            || !node
                .labels
                .iter()
                .any(|label| label == &self.definition.target_label)
        {
            self.rows.remove(&node.id);
            return None;
        }
        let fields = self
            .definition
            .covering_fields
            .iter()
            .filter_map(|field| {
                node.properties
                    .get(field)
                    .cloned()
                    .map(|value| (field.clone(), value))
            })
            .collect::<BTreeMap<_, _>>();
        let row = CoveringRow {
            object_id: node.id.clone(),
            source_version: node.version,
            fields,
            stale: false,
        };
        self.rows.insert(node.id.clone(), row.clone());
        Some(row)
    }

    pub fn mark_stale_if_older(&mut self, object_id: &str, source_version: u64) -> bool {
        let Some(row) = self.rows.get_mut(object_id) else {
            return false;
        };
        if row.source_version < source_version {
            row.stale = true;
        }
        row.stale
    }

    pub fn get(&self, object_id: &str) -> Option<&CoveringRow> {
        self.rows.get(object_id)
    }

    pub fn list(&self) -> Vec<&CoveringRow> {
        self.rows.values().collect()
    }
}

fn merge_scope_properties(scope: &ReceiptScope, properties: &Value) -> Value {
    let mut merged = properties
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    for (key, value) in scope {
        merged
            .entry(key.clone())
            .or_insert_with(|| Value::String(value.clone()));
    }
    Value::Object(merged)
}

fn index_value_token(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Value::Bool(value) => Some(value.to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).ok(),
    }
}
