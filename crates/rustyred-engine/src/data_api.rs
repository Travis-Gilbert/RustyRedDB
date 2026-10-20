use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::graph::record::{EdgeRecord, NodeRecord, Provenance};

pub const DEFAULT_DATA_QUERY_LIMIT: usize = 20;
pub const MAX_DATA_QUERY_LIMIT: usize = 500;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataScope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collections: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataQuery {
    #[serde(default)]
    pub scope: DataScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<DataFilter>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sort: Vec<DataSort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default)]
    pub hydrate: DataHydrate,
    #[serde(default)]
    pub broad_scan: bool,
    #[serde(default = "default_trace")]
    pub trace: bool,
}

impl DataQuery {
    pub fn bounded_limit(&self) -> usize {
        self.limit
            .filter(|limit| *limit > 0)
            .unwrap_or(DEFAULT_DATA_QUERY_LIMIT)
            .min(MAX_DATA_QUERY_LIMIT)
    }
}

fn default_trace() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFilterOperator {
    Equals,
    Contains,
    Prefix,
}

impl Default for DataFilterOperator {
    fn default() -> Self {
        Self::Equals
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataFilter {
    pub field: String,
    #[serde(default)]
    pub value: Value,
    #[serde(default)]
    pub op: DataFilterOperator,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataSortDirection {
    Asc,
    Desc,
}

impl Default for DataSortDirection {
    fn default() -> Self {
        Self::Asc
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataSort {
    pub field: String,
    #[serde(default)]
    pub direction: DataSortDirection,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataHydrate {
    #[serde(default)]
    pub links: bool,
    #[serde(default)]
    pub content_preview_chars: Option<usize>,
}

impl Default for DataHydrate {
    fn default() -> Self {
        Self {
            links: false,
            content_preview_chars: Some(1200),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataResult {
    #[serde(default)]
    pub records: Vec<DataRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default)]
    pub trace: DataTrace,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<DataDegradedState>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataCursor {
    pub value: String,
    #[serde(default)]
    pub offset: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataTrace {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub filters_applied: Vec<String>,
    #[serde(default)]
    pub candidate_count: usize,
    #[serde(default)]
    pub returned_count: usize,
    #[serde(default)]
    pub broad_scan: bool,
    #[serde(default)]
    pub source: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stats: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataDegradedState {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataRecord {
    pub id: String,
    #[serde(rename = "type")]
    pub record_type: String,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub fields: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<DataLink>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provenance: Vec<DataProvenance>,
    #[serde(default)]
    pub rank_signals: DataRankSignals,
}

impl DataRecord {
    pub fn from_node(node: &NodeRecord) -> Self {
        let fields = object_fields(&node.properties);
        let provenance = DataProvenance::from_fields(&fields);
        Self {
            id: node.id.clone(),
            record_type: record_type_from_labels(&node.labels),
            labels: node.labels.clone(),
            fields,
            score: None,
            edges: Vec::new(),
            provenance,
            rank_signals: DataRankSignals::default(),
        }
    }

    pub fn with_score(mut self, score: Option<f64>) -> Self {
        self.score = score;
        self.rank_signals.final_score = score;
        self
    }

    pub fn with_edges(mut self, edges: Vec<DataLink>) -> Self {
        self.edges = edges;
        self
    }

    pub fn with_criticism_signals(
        mut self,
        criticism_survival: Option<f64>,
        test_burden: Option<f64>,
    ) -> Self {
        self.rank_signals.criticism_survival = criticism_survival;
        self.rank_signals.test_burden = test_burden;
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataLink {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub direction: String,
    #[serde(default)]
    pub fields: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provenance: Vec<DataProvenance>,
}

impl DataLink {
    pub fn from_edge(edge: &EdgeRecord, direction: impl Into<String>) -> Self {
        let fields = object_fields(&edge.properties);
        let mut provenance = DataProvenance::from_fields(&fields);
        if let Some(edge_provenance) = edge.provenance.as_ref().and_then(DataProvenance::from_edge)
        {
            provenance.push(edge_provenance);
        }
        Self {
            id: edge.id.clone(),
            from_id: edge.from_id.clone(),
            to_id: edge.to_id.clone(),
            edge_type: edge.edge_type.clone(),
            direction: direction.into(),
            fields,
            confidence: edge.confidence,
            provenance,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
}

impl DataProvenance {
    pub fn from_fields(fields: &Map<String, Value>) -> Vec<Self> {
        let provenance = Self {
            source: field_string(fields, &["source", "surface"]),
            source_id: field_string(fields, &["source_id", "sourceId", "path", "repo"]),
            timestamp: field_string(
                fields,
                &["updated_at", "updatedAt", "created_at", "createdAt"],
            ),
            method: field_string(fields, &["method", "kind"]),
        };
        if provenance.source.is_some()
            || provenance.source_id.is_some()
            || provenance.timestamp.is_some()
            || provenance.method.is_some()
        {
            vec![provenance]
        } else {
            Vec::new()
        }
    }

    fn from_edge(provenance: &Provenance) -> Option<Self> {
        let value = Self {
            source: None,
            source_id: provenance.source_id.clone(),
            timestamp: provenance.timestamp.clone(),
            method: provenance.method.clone(),
        };
        if value.source.is_some()
            || value.source_id.is_some()
            || value.timestamp.is_some()
            || value.method.is_some()
        {
            Some(value)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DataRankSignals {
    #[serde(default)]
    pub exact_match: bool,
    #[serde(default)]
    pub filter_match: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recency_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_score: Option<f64>,
    /// Acceptability × saturated survived-severity (SPEC-CRITICISM-SURVIVAL).
    /// Returned separately from source priors; not blended into `final_score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criticism_survival: Option<f64>,
    /// Saturating count of survived severe criticism (independent of Acc).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_burden: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

fn object_fields(value: &Value) -> Map<String, Value> {
    match value {
        Value::Object(object) => object.clone(),
        other => {
            let mut fields = Map::new();
            fields.insert("value".to_string(), other.clone());
            fields
        }
    }
}

fn field_string(fields: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| fields.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn record_type_from_labels(labels: &[String]) -> String {
    if labels.iter().any(|label| label == "MemoryDocument") {
        return "memory_doc".to_string();
    }
    if labels.iter().any(|label| label == "CodeFile") {
        return "code_file".to_string();
    }
    if labels.iter().any(|label| label == "CodeSymbol") {
        return "code_symbol".to_string();
    }
    if labels.iter().any(|label| label == "CoordinationRecord") {
        return "coordination_record".to_string();
    }
    if labels.iter().any(|label| label == "DataView") {
        return "data_view".to_string();
    }
    labels
        .first()
        .map(|label| camel_to_snake(label))
        .unwrap_or_else(|| "record".to_string())
}

fn camel_to_snake(value: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if ch.is_ascii_uppercase() && idx > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn maps_graph_node_to_stable_data_record() {
        let node = NodeRecord::new(
            "mem:1",
            ["MemoryDocument"],
            json!({
                "title": "Routing decision",
                "source": "codex",
                "source_id": "thread:1",
                "updated_at": "2026-07-01T00:00:00Z"
            }),
        );

        let record = DataRecord::from_node(&node);

        assert_eq!(record.id, "mem:1");
        assert_eq!(record.record_type, "memory_doc");
        assert_eq!(record.fields["title"], "Routing decision");
        assert_eq!(record.provenance.len(), 1);
        assert_eq!(record.provenance[0].source.as_deref(), Some("codex"));

        let serialized = serde_json::to_value(&record).unwrap();
        assert_eq!(serialized["type"], "memory_doc");
        assert!(serialized.get("fields").is_some());
        assert!(serialized.get("rank_signals").is_some());
    }

    #[test]
    fn maps_edge_to_data_link_with_provenance() {
        let edge = EdgeRecord::new(
            "edge:1",
            "mem:1",
            "CITES",
            "file:1",
            json!({"source": "test", "method": "fixture"}),
        )
        .with_confidence(0.8)
        .with_provenance(Provenance {
            source_id: Some("call:1".to_string()),
            timestamp: Some("unix_ms:1".to_string()),
            method: Some("manual".to_string()),
        });

        let link = DataLink::from_edge(&edge, "out");

        assert_eq!(link.id, "edge:1");
        assert_eq!(link.edge_type, "CITES");
        assert_eq!(link.confidence, Some(0.8));
        assert_eq!(link.provenance.len(), 2);
    }

    #[test]
    fn data_query_limit_is_bounded() {
        let query = DataQuery {
            limit: Some(MAX_DATA_QUERY_LIMIT + 100),
            ..Default::default()
        };
        assert_eq!(query.bounded_limit(), MAX_DATA_QUERY_LIMIT);
    }

    #[test]
    fn data_record_exposes_criticism_survival_separately_from_score() {
        let node = NodeRecord::new(
            "claim:1",
            ["Claim"],
            json!({"title": "survived conjecture"}),
        );
        let record = DataRecord::from_node(&node)
            .with_score(Some(0.9))
            .with_criticism_signals(Some(0.31), Some(1.0));
        assert_eq!(record.score, Some(0.9));
        assert_eq!(record.rank_signals.final_score, Some(0.9));
        assert_eq!(record.rank_signals.criticism_survival, Some(0.31));
        assert_eq!(record.rank_signals.test_burden, Some(1.0));
        let serialized = serde_json::to_value(&record).unwrap();
        assert_eq!(serialized["rank_signals"]["criticism_survival"], 0.31);
        assert_eq!(serialized["rank_signals"]["test_burden"], 1.0);
    }
}
