use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context_view::HydrationHandle;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::query_receipt::ReceiptScope;
use crate::state::stable_hash;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdentityIndexDefinition {
    pub id: String,
    pub name: String,
    pub target_label: String,
    pub key_fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_id: Option<String>,
}

impl IdentityIndexDefinition {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        target_label: impl Into<String>,
        key_fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            target_label: target_label.into(),
            key_fields: key_fields.into_iter().map(Into::into).collect(),
            resolver: None,
            manifest_id: None,
        }
    }

    pub fn with_resolver(mut self, resolver: impl Into<String>) -> Self {
        self.resolver = Some(resolver.into());
        self
    }

    pub fn with_manifest_id(mut self, manifest_id: impl Into<String>) -> Self {
        self.manifest_id = Some(manifest_id.into());
        self
    }

    fn validate(&self) -> GraphStoreResult<()> {
        if self.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_identity_index",
                "identity index id is required",
            ));
        }
        if self.target_label.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_identity_index",
                "identity index target label is required",
            ));
        }
        if self.key_fields.is_empty() || self.key_fields.iter().any(|field| field.trim().is_empty())
        {
            return Err(GraphStoreError::new(
                "invalid_identity_index",
                "identity index requires non-empty key fields",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct IdentityIndexKey {
    pub index_id: String,
    pub scope: ReceiptScope,
    pub values: BTreeMap<String, String>,
}

impl IdentityIndexKey {
    pub fn new(
        index_id: impl Into<String>,
        scope: ReceiptScope,
        values: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            index_id: index_id.into(),
            scope,
            values: values
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        }
    }

    pub fn from_properties(
        index_id: impl Into<String>,
        scope: ReceiptScope,
        properties: &Value,
        fields: &[String],
    ) -> GraphStoreResult<Self> {
        let index_id = index_id.into();
        let mut values = BTreeMap::new();
        for field in fields {
            let field = field.trim();
            if field.is_empty() {
                return Err(GraphStoreError::new(
                    "invalid_identity_key",
                    "identity key field must not be empty",
                ));
            }
            let Some(value) = properties.get(field).and_then(identity_value_token) else {
                return Err(GraphStoreError::new(
                    "missing_identity_key_field",
                    format!("identity key field {field} is missing or not indexable"),
                ));
            };
            values.insert(field.to_string(), value);
        }
        Ok(Self {
            index_id,
            scope,
            values,
        })
    }

    fn validate(&self) -> GraphStoreResult<()> {
        if self.index_id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_identity_key",
                "identity key index id is required",
            ));
        }
        if self.values.is_empty()
            || self
                .values
                .iter()
                .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
        {
            return Err(GraphStoreError::new(
                "invalid_identity_key",
                "identity key requires non-empty field values",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdentityTarget {
    pub node_id: String,
    pub target_label: String,
    pub hydration_handle: HydrationHandle,
}

impl IdentityTarget {
    pub fn node(
        node_id: impl Into<String>,
        target_label: impl Into<String>,
        graph_version: u64,
    ) -> Self {
        let node_id = node_id.into();
        let target_label = target_label.into();
        Self {
            hydration_handle: HydrationHandle::new(
                node_id.clone(),
                target_label.clone(),
                graph_version,
                format!("graph://{target_label}/{node_id}"),
            ),
            node_id,
            target_label,
        }
    }

    fn validate(&self) -> GraphStoreResult<()> {
        if self.node_id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_identity_target",
                "identity target node id is required",
            ));
        }
        if self.target_label.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_identity_target",
                "identity target label is required",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdentityProblemRecord {
    pub id: String,
    pub kind: String,
    pub key: IdentityIndexKey,
    pub existing: IdentityTarget,
    pub attempted: IdentityTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver: Option<String>,
}

impl IdentityProblemRecord {
    fn collision(
        key: IdentityIndexKey,
        existing: IdentityTarget,
        attempted: IdentityTarget,
        resolver: Option<String>,
    ) -> GraphStoreResult<Self> {
        #[derive(Serialize)]
        struct ProblemHashInput<'a> {
            kind: &'static str,
            key: &'a IdentityIndexKey,
            existing_node_id: &'a str,
            attempted_node_id: &'a str,
            resolver: &'a Option<String>,
        }

        let id = stable_hash(ProblemHashInput {
            kind: "identity_collision",
            key: &key,
            existing_node_id: &existing.node_id,
            attempted_node_id: &attempted.node_id,
            resolver: &resolver,
        })?;
        Ok(Self {
            id,
            kind: "identity_collision".to_string(),
            key,
            existing,
            attempted,
            resolver,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityInsertOutcome {
    Inserted(IdentityTarget),
    Existing(IdentityTarget),
    Collision(IdentityProblemRecord),
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdentityIndex {
    definitions: BTreeMap<String, IdentityIndexDefinition>,
    entries: BTreeMap<IdentityIndexKey, IdentityTarget>,
    problems: Vec<IdentityProblemRecord>,
}

impl IdentityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_definition(
        &mut self,
        definition: IdentityIndexDefinition,
    ) -> GraphStoreResult<()> {
        definition.validate()?;
        if self.definitions.contains_key(&definition.id) {
            return Err(GraphStoreError::new(
                "identity_index_exists",
                format!("identity index {} is already registered", definition.id),
            ));
        }
        self.definitions.insert(definition.id.clone(), definition);
        Ok(())
    }

    pub fn definition(&self, id: &str) -> Option<&IdentityIndexDefinition> {
        self.definitions.get(id)
    }

    pub fn key_from_properties(
        &self,
        index_id: &str,
        scope: ReceiptScope,
        properties: &Value,
    ) -> GraphStoreResult<IdentityIndexKey> {
        let definition = self.definitions.get(index_id).ok_or_else(|| {
            GraphStoreError::new(
                "identity_index_not_found",
                format!("identity index {index_id} is not registered"),
            )
        })?;
        IdentityIndexKey::from_properties(index_id, scope, properties, &definition.key_fields)
    }

    pub fn resolve(&self, key: &IdentityIndexKey) -> Option<&IdentityTarget> {
        self.entries.get(key)
    }

    pub fn resolve_or_insert(
        &mut self,
        key: IdentityIndexKey,
        target: IdentityTarget,
    ) -> GraphStoreResult<IdentityInsertOutcome> {
        key.validate()?;
        target.validate()?;
        let definition = self.definitions.get(&key.index_id).ok_or_else(|| {
            GraphStoreError::new(
                "identity_index_not_found",
                format!("identity index {} is not registered", key.index_id),
            )
        })?;
        if definition.target_label != target.target_label {
            return Err(GraphStoreError::new(
                "identity_target_label_mismatch",
                format!(
                    "identity index {} targets {}, got {}",
                    definition.id, definition.target_label, target.target_label
                ),
            ));
        }

        if let Some(existing) = self.entries.get(&key).cloned() {
            if existing.node_id == target.node_id {
                return Ok(IdentityInsertOutcome::Existing(existing));
            }
            let problem = IdentityProblemRecord::collision(
                key,
                existing,
                target,
                definition.resolver.clone(),
            )?;
            self.problems.push(problem.clone());
            return Ok(IdentityInsertOutcome::Collision(problem));
        }

        self.entries.insert(key, target.clone());
        Ok(IdentityInsertOutcome::Inserted(target))
    }

    pub fn problems(&self) -> &[IdentityProblemRecord] {
        &self.problems
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn identity_value_token(value: &Value) -> Option<String> {
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
