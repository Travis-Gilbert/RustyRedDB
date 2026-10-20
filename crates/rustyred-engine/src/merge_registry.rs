use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Hlc, NodeRecord};

const CRDT_HLC_PROPERTY: &str = "_crdt_hlc";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeRegistryStrategy {
    LwwRegister,
    OrSet,
    DeltaCrdt,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MergeRegistryEntry {
    pub node_label: String,
    pub field_name: String,
    pub strategy: MergeRegistryStrategy,
}

impl MergeRegistryEntry {
    pub fn new(
        node_label: impl Into<String>,
        field_name: impl Into<String>,
        strategy: MergeRegistryStrategy,
    ) -> Self {
        Self {
            node_label: node_label.into(),
            field_name: field_name.into(),
            strategy,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MergeRegistry {
    entries: Vec<MergeRegistryEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeRegistryResolution {
    pub node: NodeRecord,
    pub reason: String,
}

impl MergeRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn substrate_sync_defaults() -> Self {
        let mut registry = Self::default();
        for label in ["MemoryItem", "MemoryDocument"] {
            registry.register(label, "status", MergeRegistryStrategy::LwwRegister);
            registry.register(label, "confidence", MergeRegistryStrategy::LwwRegister);
            registry.register(label, "tags", MergeRegistryStrategy::OrSet);
        }
        registry
    }

    pub fn register(
        &mut self,
        node_label: impl Into<String>,
        field_name: impl Into<String>,
        strategy: MergeRegistryStrategy,
    ) {
        self.entries
            .push(MergeRegistryEntry::new(node_label, field_name, strategy));
    }

    pub fn entries(&self) -> &[MergeRegistryEntry] {
        &self.entries
    }

    pub fn resolve_node(
        &self,
        base: Option<&NodeRecord>,
        ours: &NodeRecord,
        theirs: &NodeRecord,
    ) -> Option<MergeRegistryResolution> {
        if self.entries.is_empty() || ours.id != theirs.id || ours.tombstone != theirs.tombstone {
            return None;
        }

        let labels = merge_labels(base, ours, theirs);
        let base_props = object_props(base.map(|node| &node.properties));
        let ours_props = object_props(Some(&ours.properties));
        let theirs_props = object_props(Some(&theirs.properties));

        let mut keys = BTreeSet::new();
        keys.extend(base_props.keys().cloned());
        keys.extend(ours_props.keys().cloned());
        keys.extend(theirs_props.keys().cloned());

        let mut merged_props = Map::new();
        let mut used_registry = Vec::new();

        for key in keys {
            if key == CRDT_HLC_PROPERTY {
                continue;
            }
            let base_value = base_props.get(&key);
            let ours_value = ours_props.get(&key);
            let theirs_value = theirs_props.get(&key);

            if ours_value == theirs_value {
                insert_optional(&mut merged_props, &key, ours_value.cloned());
                continue;
            }
            if let Some(strategy) = self.strategy_for(&labels, &key) {
                let resolved = match strategy {
                    MergeRegistryStrategy::LwwRegister => lww_register(
                        base.map(|node| &node.properties),
                        &ours.properties,
                        &theirs.properties,
                        &key,
                    ),
                    MergeRegistryStrategy::OrSet => or_set(
                        base.map(|node| &node.properties),
                        &ours.properties,
                        &theirs.properties,
                        &key,
                    ),
                    MergeRegistryStrategy::DeltaCrdt => return None,
                }?;
                insert_optional(&mut merged_props, &key, resolved);
                used_registry.push(format!("{key}:{}", strategy_name(strategy)));
                continue;
            }
            if base_value == ours_value {
                insert_optional(&mut merged_props, &key, theirs_value.cloned());
                continue;
            }
            if base_value == theirs_value {
                insert_optional(&mut merged_props, &key, ours_value.cloned());
                continue;
            }

            return None;
        }

        if used_registry.is_empty() {
            return None;
        }

        merge_crdt_meta(&mut merged_props, base.map(|node| &node.properties));
        merge_crdt_meta(&mut merged_props, Some(&ours.properties));
        merge_crdt_meta(&mut merged_props, Some(&theirs.properties));

        let mut node = ours.clone();
        node.labels = labels;
        node.properties = Value::Object(merged_props);
        node.version = ours.version.max(theirs.version).saturating_add(1);
        node.content_hash = None;
        node.parent_hashes = parent_hashes(base, ours, theirs);

        Some(MergeRegistryResolution {
            node,
            reason: format!("merge_registry({})", used_registry.join(",")),
        })
    }

    fn strategy_for(&self, labels: &[String], field_name: &str) -> Option<MergeRegistryStrategy> {
        self.entries
            .iter()
            .find(|entry| {
                entry.field_name == field_name
                    && labels.iter().any(|label| label == &entry.node_label)
            })
            .map(|entry| entry.strategy)
    }
}

fn lww_register(
    base: Option<&Value>,
    ours: &Value,
    theirs: &Value,
    field: &str,
) -> Option<Option<Value>> {
    let ours_hlc = field_hlc(ours, field).or_else(|| record_hlc(ours));
    let theirs_hlc = field_hlc(theirs, field).or_else(|| record_hlc(theirs));
    match (ours_hlc, theirs_hlc) {
        (Some(ours_hlc), Some(theirs_hlc)) => {
            if ours_hlc >= theirs_hlc {
                Some(object_props(Some(ours)).get(field).cloned())
            } else {
                Some(object_props(Some(theirs)).get(field).cloned())
            }
        }
        (Some(_), None) => Some(object_props(Some(ours)).get(field).cloned()),
        (None, Some(_)) => Some(object_props(Some(theirs)).get(field).cloned()),
        (None, None) => {
            let base_value = object_props(base).get(field).cloned();
            let ours_value = object_props(Some(ours)).get(field).cloned();
            let theirs_value = object_props(Some(theirs)).get(field).cloned();
            if ours_value != base_value {
                Some(ours_value)
            } else {
                Some(theirs_value)
            }
        }
    }
}

fn or_set(
    base: Option<&Value>,
    ours: &Value,
    theirs: &Value,
    field: &str,
) -> Option<Option<Value>> {
    if !field_is_string_array_or_missing(base, field)
        || !field_is_string_array_or_missing(Some(ours), field)
        || !field_is_string_array_or_missing(Some(theirs), field)
    {
        return None;
    }
    let mut state = OrSetState::default();
    state.observe_base(base, field);
    state.observe_side(base, ours, field);
    state.observe_side(base, theirs, field);
    Some(Some(Value::Array(
        state
            .resolve()
            .into_iter()
            .map(Value::String)
            .collect::<Vec<_>>(),
    )))
}

fn field_is_string_array_or_missing(record: Option<&Value>, field: &str) -> bool {
    match record
        .and_then(Value::as_object)
        .and_then(|map| map.get(field))
    {
        None => true,
        Some(Value::Array(values)) => values.iter().all(Value::is_string),
        Some(_) => false,
    }
}

#[derive(Default)]
struct OrSetState {
    adds: BTreeMap<String, Hlc>,
    removes: BTreeMap<String, Hlc>,
}

impl OrSetState {
    fn observe_base(&mut self, base: Option<&Value>, field: &str) {
        let Some(base) = base else {
            return;
        };
        let stamp = field_hlc(base, field)
            .or_else(|| record_hlc(base))
            .unwrap_or_default();
        for item in string_set(object_props(Some(base)).get(field)) {
            self.adds.entry(item).or_insert(stamp);
        }
    }

    fn observe_side(&mut self, base: Option<&Value>, side: &Value, field: &str) {
        let side_items = string_set(object_props(Some(side)).get(field));
        let base_items = base
            .and_then(|base| object_props(Some(base)).get(field).cloned())
            .map(|value| string_set(Some(&value)))
            .unwrap_or_default();
        let stamp = field_hlc(side, field)
            .or_else(|| record_hlc(side))
            .unwrap_or_default();

        for item in &side_items {
            self.adds
                .entry(item.clone())
                .and_modify(|existing| {
                    if stamp > *existing {
                        *existing = stamp;
                    }
                })
                .or_insert(stamp);
        }
        for item in base_items.difference(&side_items) {
            self.removes
                .entry(item.clone())
                .and_modify(|existing| {
                    if stamp > *existing {
                        *existing = stamp;
                    }
                })
                .or_insert(stamp);
        }
    }

    fn resolve(self) -> Vec<String> {
        let mut out = Vec::new();
        for (item, add_hlc) in self.adds {
            let remove_hlc = self.removes.get(&item).copied().unwrap_or_default();
            if add_hlc >= remove_hlc {
                out.push(item);
            }
        }
        out
    }
}

fn object_props(value: Option<&Value>) -> Map<String, Value> {
    value
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn insert_optional(map: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if key == CRDT_HLC_PROPERTY {
        return;
    }
    if let Some(value) = value {
        map.insert(key.to_string(), value);
    }
}

fn merge_labels(base: Option<&NodeRecord>, ours: &NodeRecord, theirs: &NodeRecord) -> Vec<String> {
    let mut labels = base
        .into_iter()
        .flat_map(|node| node.labels.iter())
        .chain(ours.labels.iter())
        .chain(theirs.labels.iter())
        .filter(|label| !label.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn parent_hashes(base: Option<&NodeRecord>, ours: &NodeRecord, theirs: &NodeRecord) -> Vec<String> {
    let mut hashes = Vec::new();
    hashes.extend(base.and_then(|node| node.content_hash.clone()));
    hashes.extend(ours.content_hash.clone());
    hashes.extend(theirs.content_hash.clone());
    hashes.extend(ours.parent_hashes.iter().cloned());
    hashes.extend(theirs.parent_hashes.iter().cloned());
    hashes.sort();
    hashes.dedup();
    hashes
}

fn string_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn merge_crdt_meta(out: &mut Map<String, Value>, source: Option<&Value>) {
    let Some(source_meta) = source
        .and_then(Value::as_object)
        .and_then(|map| map.get(CRDT_HLC_PROPERTY))
        .and_then(Value::as_object)
    else {
        return;
    };
    let out_meta = out
        .entry(CRDT_HLC_PROPERTY.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !out_meta.is_object() {
        *out_meta = Value::Object(Map::new());
    }
    let out_meta = out_meta.as_object_mut().expect("metadata is object");

    for scalar in ["record", "tombstone"] {
        let Some(candidate) = source_meta.get(scalar).and_then(value_to_hlc) else {
            continue;
        };
        let current = out_meta
            .get(scalar)
            .and_then(value_to_hlc)
            .unwrap_or_default();
        if candidate > current {
            out_meta.insert(scalar.to_string(), hlc_to_value(candidate));
        }
    }

    if let Some(source_props) = source_meta.get("properties").and_then(Value::as_object) {
        let props = out_meta
            .entry("properties".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !props.is_object() {
            *props = Value::Object(Map::new());
        }
        let props = props
            .as_object_mut()
            .expect("metadata properties is object");
        for (field, value) in source_props {
            let Some(candidate) = value_to_hlc(value) else {
                continue;
            };
            let current = props.get(field).and_then(value_to_hlc).unwrap_or_default();
            if candidate > current {
                props.insert(field.clone(), hlc_to_value(candidate));
            }
        }
    }
}

fn field_hlc(value: &Value, field: &str) -> Option<Hlc> {
    value
        .as_object()
        .and_then(|map| map.get(CRDT_HLC_PROPERTY))
        .and_then(Value::as_object)
        .and_then(|meta| meta.get("properties"))
        .and_then(Value::as_object)
        .and_then(|props| props.get(field))
        .and_then(value_to_hlc)
}

fn record_hlc(value: &Value) -> Option<Hlc> {
    value
        .as_object()
        .and_then(|map| map.get(CRDT_HLC_PROPERTY))
        .and_then(Value::as_object)
        .and_then(|meta| meta.get("record"))
        .and_then(value_to_hlc)
}

fn value_to_hlc(value: &Value) -> Option<Hlc> {
    serde_json::from_value(value.clone()).ok()
}

fn hlc_to_value(hlc: Hlc) -> Value {
    serde_json::to_value(hlc).unwrap_or(Value::Null)
}

fn strategy_name(strategy: MergeRegistryStrategy) -> &'static str {
    match strategy {
        MergeRegistryStrategy::LwwRegister => "lww",
        MergeRegistryStrategy::OrSet => "or_set",
        MergeRegistryStrategy::DeltaCrdt => "delta_crdt",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn or_set_refuses_non_string_members_without_destroying_them() {
        let mut registry = MergeRegistry::empty();
        registry.register("MemoryItem", "tags", MergeRegistryStrategy::OrSet);
        let base = NodeRecord::new("memory:1", ["MemoryItem"], json!({"tags": [1]}));
        let ours = NodeRecord::new("memory:1", ["MemoryItem"], json!({"tags": [1, 2]}));
        let theirs = NodeRecord::new("memory:1", ["MemoryItem"], json!({"tags": [1, 3]}));
        assert!(registry.resolve_node(Some(&base), &ours, &theirs).is_none());
    }
}
