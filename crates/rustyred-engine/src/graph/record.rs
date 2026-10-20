use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::state::stable_hash_graph_v1;

pub const EMBEDDING_PROPERTY: &str = "_embedding";
pub const EMBEDDING_EXCLUDED_PROPERTY: &str = "_embedding_excluded";
pub const VALUATION_PROPERTY: &str = "_valuation";
pub const CALIBRATION_EPOCH_LABEL: &str = "CalibrationEpoch";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Out,
    In,
}

/// The epistemic relation an edge asserts.
///
/// The Pollock distinction is carried explicitly: `Undercuts` attacks the
/// inference, `Contradicts` attacks the conclusion, and `Tension` marks
/// unresolved mutual pressure with no directed attack. All three classify as
/// `AlgebraicClass::Defeating` in the edge ontology, so every consumer that
/// previously treated the old `Contradicts`-for-everything encoding uniformly
/// continues to.
///
/// `Undercuts` appends at the end of the variant list. The enum serializes by
/// name rather than discriminant, so edges serialized before it existed
/// deserialize byte-identically and content addressing of existing records is
/// unaffected.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicType {
    Supports,
    Contradicts,
    Tension,
    Derives,
    Cites,
    Undercuts,
}

impl std::fmt::Display for EpistemicType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Supports => write!(f, "supports"),
            Self::Contradicts => write!(f, "contradicts"),
            Self::Tension => write!(f, "tension"),
            Self::Derives => write!(f, "derives"),
            Self::Cites => write!(f, "cites"),
            Self::Undercuts => write!(f, "undercuts"),
        }
    }
}

impl std::str::FromStr for EpistemicType {
    type Err = GraphStoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "supports" => Ok(Self::Supports),
            "contradicts" => Ok(Self::Contradicts),
            "tension" => Ok(Self::Tension),
            "derives" => Ok(Self::Derives),
            "cites" => Ok(Self::Cites),
            "undercuts" => Ok(Self::Undercuts),
            _ => Err(GraphStoreError::new(
                "invalid_epistemic_type",
                format!("unknown epistemic type: {s}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NodeRecord {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: Value,
    pub version: u64,
    pub tombstone: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_hashes: Vec<String>,
}

impl NodeRecord {
    pub fn new(
        id: impl Into<String>,
        labels: impl IntoIterator<Item = impl Into<String>>,
        properties: Value,
    ) -> Self {
        Self {
            id: id.into(),
            labels: normalize_labels(labels),
            properties,
            version: 0,
            tombstone: false,
            content_hash: None,
            parent_hashes: Vec::new(),
        }
    }

    pub fn content_address(&self) -> GraphStoreResult<String> {
        #[derive(Serialize)]
        struct NodeContent<'a> {
            kind: &'static str,
            id: &'a str,
            labels: &'a [String],
            properties: &'a Value,
            tombstone: bool,
        }

        stable_hash_graph_v1(NodeContent {
            kind: "node",
            id: &self.id,
            labels: &self.labels,
            properties: &self.properties,
            tombstone: self.tombstone,
        })
    }

    pub fn checksum(&self) -> GraphStoreResult<String> {
        self.content_address()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EdgeRecord {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub properties: Value,
    pub version: u64,
    pub tombstone: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic_type: Option<EpistemicType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_hashes: Vec<String>,
}

impl EdgeRecord {
    pub fn new(
        id: impl Into<String>,
        from_id: impl Into<String>,
        edge_type: impl Into<String>,
        to_id: impl Into<String>,
        properties: Value,
    ) -> Self {
        Self {
            id: id.into(),
            from_id: from_id.into(),
            to_id: to_id.into(),
            edge_type: edge_type.into(),
            properties,
            version: 0,
            tombstone: false,
            confidence: None,
            epistemic_type: None,
            provenance: None,
            content_hash: None,
            parent_hashes: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    pub fn with_epistemic_type(mut self, epistemic_type: EpistemicType) -> Self {
        self.epistemic_type = Some(epistemic_type);
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    pub fn effective_confidence(&self) -> f64 {
        self.confidence.unwrap_or(1.0)
    }

    pub fn content_address(&self) -> GraphStoreResult<String> {
        #[derive(Serialize)]
        struct EdgeContent<'a> {
            kind: &'static str,
            id: &'a str,
            from_id: &'a str,
            to_id: &'a str,
            edge_type: &'a str,
            properties: &'a Value,
            tombstone: bool,
            confidence: Option<f64>,
            epistemic_type: &'a Option<EpistemicType>,
            provenance: &'a Option<Provenance>,
        }

        stable_hash_graph_v1(EdgeContent {
            kind: "edge",
            id: &self.id,
            from_id: &self.from_id,
            to_id: &self.to_id,
            edge_type: &self.edge_type,
            properties: &self.properties,
            tombstone: self.tombstone,
            confidence: self.confidence,
            epistemic_type: &self.epistemic_type,
            provenance: &self.provenance,
        })
    }

    pub fn checksum(&self) -> GraphStoreResult<String> {
        self.content_address()
    }
}

pub(crate) fn normalize_labels(labels: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
    let mut labels = labels
        .into_iter()
        .map(Into::into)
        .map(|label| label.trim().to_string())
        .filter(|label| !label.is_empty())
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}
