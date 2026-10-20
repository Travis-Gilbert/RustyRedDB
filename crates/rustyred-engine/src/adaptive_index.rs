use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::access_method::{ScalarBound, ScalarValue};
use crate::cold_fragments::{ColdFragment, CompressionFilter, ZoneMap};
use crate::index_manifest::{
    IndexBackend, IndexBuildStatus, IndexCreatedBy, IndexKind, IndexManifest, IndexScope,
};
use crate::spatial::SpatialDesignation;
use crate::working_log::TemporalFact;
use crate::GraphStoreResult;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStructuralIndexKind {
    Adjacency,
    PersonalizedPagerank,
    Motif,
    SupportAttack,
    Bridge,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphStructuralIndexDefinition {
    pub manifest_id: String,
    pub structural_kind: GraphStructuralIndexKind,
    pub target_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_edge_type: Option<String>,
    #[serde(default)]
    pub seed_properties: Vec<String>,
    pub graph_version: u64,
}

impl GraphStructuralIndexDefinition {
    pub fn new(
        manifest_id: impl Into<String>,
        structural_kind: GraphStructuralIndexKind,
        target_label: impl Into<String>,
        graph_version: u64,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            structural_kind,
            target_label: target_label.into(),
            target_edge_type: None,
            seed_properties: Vec::new(),
            graph_version,
        }
    }

    pub fn with_edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.target_edge_type = Some(edge_type.into());
        self
    }

    pub fn with_seed_properties(
        mut self,
        properties: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.seed_properties = properties.into_iter().map(Into::into).collect();
        self
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{:?} structural index", self.structural_kind),
            IndexKind::GraphStructural,
            IndexBackend::RustyredCore,
            scope,
            self.target_label.clone(),
            created_by,
        )?
        .with_target_properties(self.seed_properties.clone())?;
        if let Some(edge_type) = &self.target_edge_type {
            manifest = manifest.with_target_edge_type(edge_type.clone())?;
        }
        manifest.graph_version = self.graph_version;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphStructuralIndexReceipt {
    pub manifest_id: String,
    pub graph_version: u64,
    pub indexed_node_count: usize,
    pub indexed_edge_count: usize,
    pub problem_count: usize,
}

impl GraphStructuralIndexReceipt {
    pub fn stale_for_graph_version(&self, current_graph_version: u64) -> bool {
        self.graph_version != current_graph_version
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TemporalIndexDefinition {
    pub manifest_id: String,
    pub target_label: String,
    pub valid_from_property: String,
    pub invalid_at_property: String,
    pub transaction_time_property: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_property: Option<String>,
}

impl TemporalIndexDefinition {
    pub fn new(
        manifest_id: impl Into<String>,
        target_label: impl Into<String>,
        valid_from_property: impl Into<String>,
        invalid_at_property: impl Into<String>,
        transaction_time_property: impl Into<String>,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            target_label: target_label.into(),
            valid_from_property: valid_from_property.into(),
            invalid_at_property: invalid_at_property.into(),
            transaction_time_property: transaction_time_property.into(),
            ttl_property: None,
        }
    }

    pub fn with_ttl_property(mut self, ttl_property: impl Into<String>) -> Self {
        self.ttl_property = Some(ttl_property.into());
        self
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut properties = vec![
            self.valid_from_property.clone(),
            self.invalid_at_property.clone(),
            self.transaction_time_property.clone(),
        ];
        if let Some(ttl_property) = &self.ttl_property {
            properties.push(ttl_property.clone());
        }
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} temporal index", self.target_label),
            IndexKind::Temporal,
            IndexBackend::RustyredCore,
            scope,
            self.target_label.clone(),
            created_by,
        )?
        .with_target_properties(properties)?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TemporalContextSlice {
    pub as_of_ms: i64,
    pub fact_ids: Vec<String>,
    pub facts: Vec<TemporalFact>,
}

pub fn reconstruct_temporal_context(facts: &[TemporalFact], as_of_ms: i64) -> TemporalContextSlice {
    let mut facts = facts
        .iter()
        .filter(|fact| fact.is_valid_at(as_of_ms))
        .cloned()
        .collect::<Vec<_>>();
    facts.sort_by(|left, right| left.fact_id.cmp(&right.fact_id));
    let fact_ids = facts.iter().map(|fact| fact.fact_id.clone()).collect();
    TemporalContextSlice {
        as_of_ms,
        fact_ids,
        facts,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialIndexBackend {
    H3,
    S2,
}

impl SpatialIndexBackend {
    fn index_backend(self) -> IndexBackend {
        match self {
            Self::H3 => IndexBackend::H3,
            Self::S2 => IndexBackend::S2,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SpatialIndexDefinition {
    pub manifest_id: String,
    pub designation: SpatialDesignation,
    pub backend: SpatialIndexBackend,
}

impl SpatialIndexDefinition {
    pub fn h3(manifest_id: impl Into<String>, designation: SpatialDesignation) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            designation,
            backend: SpatialIndexBackend::H3,
        }
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} spatial index", self.designation.label),
            IndexKind::Spatial,
            self.backend.index_backend(),
            scope,
            self.designation.label.clone(),
            created_by,
        )?
        .with_target_properties([
            self.designation.lat_property.clone(),
            self.designation.lon_property.clone(),
        ])?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ColdSkipIndexDefinition {
    pub manifest_id: String,
    pub relation: String,
    pub indexed_columns: Vec<String>,
}

impl ColdSkipIndexDefinition {
    pub fn new(
        manifest_id: impl Into<String>,
        relation: impl Into<String>,
        indexed_columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            relation: relation.into(),
            indexed_columns: indexed_columns.into_iter().map(Into::into).collect(),
        }
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} cold skip index", self.relation),
            IndexKind::ColdSkip,
            IndexBackend::ColdFragment,
            scope,
            self.relation.clone(),
            created_by,
        )?
        .with_target_properties(self.indexed_columns.clone())?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ColdFragmentSkipMetadata {
    pub fragment_id: String,
    pub relation: String,
    pub row_count: usize,
    pub zone_maps: BTreeMap<String, ZoneMap>,
    pub compression_filters: BTreeMap<String, Vec<CompressionFilter>>,
}

impl ColdFragmentSkipMetadata {
    pub fn from_fragment(fragment: &ColdFragment) -> Self {
        let mut zone_maps = BTreeMap::new();
        let mut compression_filters = BTreeMap::new();
        for (column, fragment_column) in &fragment.columns {
            if let Some(zone_map) = &fragment_column.zone_map {
                zone_maps.insert(column.clone(), zone_map.clone());
            }
            compression_filters.insert(column.clone(), fragment_column.filters.clone());
        }
        Self {
            fragment_id: fragment.fragment_id.clone(),
            relation: fragment.relation.clone(),
            row_count: fragment.row_ids.len(),
            zone_maps,
            compression_filters,
        }
    }

    pub fn excludes_range(&self, column: &str, lo: &ScalarBound, hi: &ScalarBound) -> bool {
        self.zone_maps
            .get(column)
            .map(|zone_map| zone_map.excludes_range(lo, hi))
            .unwrap_or(false)
    }

    pub fn no_false_negative_for_range(
        &self,
        fragment: &ColdFragment,
        column: &str,
        lo: ScalarBound,
        hi: ScalarBound,
    ) -> bool {
        if !self.excludes_range(column, &lo, &hi) {
            return true;
        }
        fragment
            .range_query(column, lo, hi)
            .map(|result| result.rows.is_empty())
            .unwrap_or(false)
    }
}

pub fn scalar_i64(value: i64) -> ScalarValue {
    ScalarValue::I64(value)
}
