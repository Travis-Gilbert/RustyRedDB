use serde::{Deserialize, Serialize};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::state::stable_hash;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexKind {
    Identity,
    Ordered,
    Composite,
    Partial,
    Covering,
    FullText,
    /// Trigram candidate index: candidate-then-verify substring search.
    Trigram,
    Vector,
    SparseVector,
    MultiVector,
    GraphStructural,
    Temporal,
    Spatial,
    ColdSkip,
    LearnedRouting,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexBackend {
    RustyredCore,
    Turbovec,
    Bm25,
    Tantivy,
    H3,
    S2,
    ColdFragment,
    AdvisorShadow,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexScope {
    Global,
    Tenant,
    Project,
    Repo,
    User,
    Session,
    Run,
    Artifact,
    Domain,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexBuildStatus {
    Proposed,
    Building,
    Shadow,
    Active,
    Degraded,
    Retired,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexCreatedBy {
    Manual,
    Migration,
    Advisor,
    System,
}

/// Durable eval gate for Shadow → Active promotion (O11).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexEvalReceipt {
    pub receipt_id: String,
    pub generation_id: String,
    pub manifest_id: String,
    pub graph_watermark: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_at_k: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ndcg_at_10: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexSegmentRef {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_path: Option<String>,
    pub doc_count: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexCoverage {
    pub document_count: u64,
    pub segment_count: u64,
    #[serde(default)]
    pub fields: Vec<String>,
}

/// Query-service view of the generation currently serving for a logical index.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ServingIndexView {
    pub logical_id: String,
    pub manifest_id: String,
    pub generation_id: String,
    pub graph_watermark: String,
    pub build_status: IndexBuildStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_generation: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexManifest {
    pub id: String,
    pub name: String,
    pub kind: IndexKind,
    pub backend: IndexBackend,
    pub scope: IndexScope,
    pub target_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_edge_type: Option<String>,
    #[serde(default)]
    pub target_properties: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(default)]
    pub sort_keys: Vec<String>,
    #[serde(default)]
    pub covering_fields: Vec<String>,
    pub version: u64,
    pub schema_hash: String,
    pub graph_version: u64,
    pub state_hash: String,
    pub build_status: IndexBuildStatus,
    pub created_by: IndexCreatedBy,
    pub hit_count: u64,
    pub miss_count: u64,
    pub avg_latency_ms: f64,
    pub avg_latency_saved_ms: f64,
    pub memory_bytes: u64,
    pub disk_bytes: u64,
    pub write_amplification: f64,
    pub quality_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retirement_reason: Option<String>,
    /// Groups generations of one logical index (defaults to [`Self::id`] when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_index_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_watermark: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_contract_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_dimension: Option<usize>,
    #[serde(default)]
    pub segments: Vec<IndexSegmentRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<IndexCoverage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_receipt: Option<IndexEvalReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_generation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_compatibility: Option<String>,
}

impl IndexManifest {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        kind: IndexKind,
        backend: IndexBackend,
        scope: IndexScope,
        target_label: impl Into<String>,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<Self> {
        let mut manifest = Self {
            id: id.into(),
            name: name.into(),
            kind,
            backend,
            scope,
            target_label: target_label.into(),
            target_edge_type: None,
            target_properties: Vec::new(),
            predicate: None,
            sort_keys: Vec::new(),
            covering_fields: Vec::new(),
            version: 1,
            schema_hash: String::new(),
            graph_version: 0,
            state_hash: String::new(),
            build_status: IndexBuildStatus::Proposed,
            created_by,
            hit_count: 0,
            miss_count: 0,
            avg_latency_ms: 0.0,
            avg_latency_saved_ms: 0.0,
            memory_bytes: 0,
            disk_bytes: 0,
            write_amplification: 0.0,
            quality_score: 0.0,
            retirement_reason: None,
            logical_id: None,
            physical_index_ref: None,
            generation_id: None,
            graph_watermark: None,
            embedding_contract_hash: None,
            vector_dimension: None,
            segments: Vec::new(),
            coverage: None,
            eval_receipt: None,
            previous_generation: None,
            serving_compatibility: None,
        };
        manifest.refresh_hashes()?;
        Ok(manifest)
    }

    pub fn logical_index_id(&self) -> &str {
        self.logical_id.as_deref().unwrap_or(self.id.as_str())
    }

    pub fn with_logical_id(mut self, logical_id: impl Into<String>) -> GraphStoreResult<Self> {
        self.logical_id = Some(logical_id.into());
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_target_edge_type(mut self, edge_type: impl Into<String>) -> GraphStoreResult<Self> {
        self.target_edge_type = Some(edge_type.into());
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_target_properties(
        mut self,
        properties: impl IntoIterator<Item = impl Into<String>>,
    ) -> GraphStoreResult<Self> {
        self.target_properties = properties.into_iter().map(Into::into).collect();
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_predicate(mut self, predicate: impl Into<String>) -> GraphStoreResult<Self> {
        self.predicate = Some(predicate.into());
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_sort_keys(
        mut self,
        sort_keys: impl IntoIterator<Item = impl Into<String>>,
    ) -> GraphStoreResult<Self> {
        self.sort_keys = sort_keys.into_iter().map(Into::into).collect();
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_covering_fields(
        mut self,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> GraphStoreResult<Self> {
        self.covering_fields = fields.into_iter().map(Into::into).collect();
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_embedding_contract(
        mut self,
        contract_hash: impl Into<String>,
        vector_dimension: usize,
    ) -> GraphStoreResult<Self> {
        self.embedding_contract_hash = Some(contract_hash.into());
        self.vector_dimension = Some(vector_dimension);
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_physical_index_ref(
        mut self,
        physical_index_ref: impl Into<String>,
    ) -> GraphStoreResult<Self> {
        self.physical_index_ref = Some(physical_index_ref.into());
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn with_serving_compatibility(
        mut self,
        serving_compatibility: impl Into<String>,
    ) -> GraphStoreResult<Self> {
        self.serving_compatibility = Some(serving_compatibility.into());
        self.refresh_hashes()?;
        Ok(self)
    }

    pub fn validate(&self) -> GraphStoreResult<()> {
        if self.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_manifest",
                "index manifest id is required",
            ));
        }
        if self.name.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_manifest",
                "index manifest name is required",
            ));
        }
        if self.target_label.trim().is_empty()
            && self
                .target_edge_type
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
        {
            return Err(GraphStoreError::new(
                "invalid_index_manifest",
                "index manifest requires a target label or target edge type",
            ));
        }
        Ok(())
    }

    pub fn mark_building(&mut self) -> GraphStoreResult<()> {
        self.build_status = IndexBuildStatus::Building;
        self.refresh_hashes()
    }

    pub fn mark_shadow(&mut self, graph_version: u64) -> GraphStoreResult<()> {
        let generation_id = self
            .generation_id
            .clone()
            .unwrap_or_else(|| self.id.clone());
        let graph_watermark = self
            .graph_watermark
            .clone()
            .unwrap_or_else(|| format!("graph:{graph_version}"));
        self.mark_shadow_generation(graph_version, generation_id, graph_watermark)
    }

    /// Enter Shadow with generation identity + watermark (search-spine lifecycle).
    pub fn mark_shadow_generation(
        &mut self,
        graph_version: u64,
        generation_id: impl Into<String>,
        graph_watermark: impl Into<String>,
    ) -> GraphStoreResult<()> {
        let generation_id = generation_id.into();
        if generation_id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_generation",
                "generation_id is required for shadow generations",
            ));
        }
        let graph_watermark = graph_watermark.into();
        if graph_watermark.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_watermark",
                "graph_watermark is required for shadow generations",
            ));
        }
        self.build_status = IndexBuildStatus::Shadow;
        self.graph_version = graph_version;
        self.generation_id = Some(generation_id);
        self.graph_watermark = Some(graph_watermark);
        self.eval_receipt = None;
        self.refresh_hashes()
    }

    /// Attach a durable eval receipt while in Shadow. Does not activate.
    pub fn attach_eval_receipt(&mut self, receipt: IndexEvalReceipt) -> GraphStoreResult<()> {
        if self.build_status != IndexBuildStatus::Shadow {
            return Err(GraphStoreError::new(
                "index_eval_wrong_status",
                format!(
                    "eval receipt requires Shadow status, found {:?}",
                    self.build_status
                ),
            ));
        }
        self.validate_eval_receipt(&receipt)?;
        self.eval_receipt = Some(receipt);
        self.refresh_hashes()
    }

    /// Activate a non-Shadow manifest (Proposed/Building/Degraded bootstrap path).
    ///
    /// Shadow → Active is refused here; use [`Self::activate_with_eval`] or the
    /// registry's atomic `activate_shadow_generation`.
    pub fn activate(&mut self, graph_version: u64) -> GraphStoreResult<()> {
        if self.build_status == IndexBuildStatus::Shadow {
            return Err(GraphStoreError::new(
                "index_shadow_activation_requires_eval",
                format!(
                    "shadow generation {} cannot become Active without an eval receipt",
                    self.generation_id.as_deref().unwrap_or(&self.id)
                ),
            ));
        }
        self.build_status = IndexBuildStatus::Active;
        self.graph_version = graph_version;
        self.retirement_reason = None;
        self.refresh_hashes()
    }

    /// Promote Shadow → Active only when a passed eval receipt is bound.
    pub fn activate_with_eval(
        &mut self,
        graph_version: u64,
        receipt: IndexEvalReceipt,
    ) -> GraphStoreResult<()> {
        if self.build_status != IndexBuildStatus::Shadow {
            return Err(GraphStoreError::new(
                "index_eval_activation_wrong_status",
                format!(
                    "activate_with_eval requires Shadow status, found {:?}",
                    self.build_status
                ),
            ));
        }
        self.validate_eval_receipt(&receipt)?;
        if !receipt.passed {
            return Err(GraphStoreError::new(
                "index_eval_receipt_failed",
                format!(
                    "eval receipt {} did not pass; Shadow generation stays Shadow",
                    receipt.receipt_id
                ),
            ));
        }
        self.eval_receipt = Some(receipt);
        self.build_status = IndexBuildStatus::Active;
        self.graph_version = graph_version;
        self.retirement_reason = None;
        self.refresh_hashes()
    }

    pub fn serving_view(&self) -> GraphStoreResult<ServingIndexView> {
        if self.build_status != IndexBuildStatus::Active {
            return Err(GraphStoreError::new(
                "index_not_serving",
                format!(
                    "manifest {} is {:?}; only Active generations serve",
                    self.id, self.build_status
                ),
            ));
        }
        let generation_id = self.generation_id.clone().ok_or_else(|| {
            GraphStoreError::new(
                "index_generation_missing",
                format!("active manifest {} has no generation_id", self.id),
            )
        })?;
        let graph_watermark = self.graph_watermark.clone().ok_or_else(|| {
            GraphStoreError::new(
                "index_watermark_missing",
                format!("active manifest {} has no graph_watermark", self.id),
            )
        })?;
        Ok(ServingIndexView {
            logical_id: self.logical_index_id().to_string(),
            manifest_id: self.id.clone(),
            generation_id,
            graph_watermark,
            build_status: self.build_status.clone(),
            previous_generation: self.previous_generation.clone(),
        })
    }

    pub fn degrade(&mut self) -> GraphStoreResult<()> {
        self.build_status = IndexBuildStatus::Degraded;
        self.refresh_hashes()
    }

    pub fn fail(&mut self, reason: impl Into<String>) -> GraphStoreResult<()> {
        self.build_status = IndexBuildStatus::Failed;
        self.retirement_reason = Some(reason.into());
        self.refresh_hashes()
    }

    pub fn retire(&mut self, reason: impl Into<String>) -> GraphStoreResult<()> {
        self.build_status = IndexBuildStatus::Retired;
        self.retirement_reason = Some(reason.into());
        self.refresh_hashes()
    }

    pub fn record_hit(&mut self, latency_ms: f64, latency_saved_ms: f64) -> GraphStoreResult<()> {
        let observations = self.hit_count.saturating_add(self.miss_count);
        self.avg_latency_ms = rolling_average(self.avg_latency_ms, observations, latency_ms);
        self.avg_latency_saved_ms =
            rolling_average(self.avg_latency_saved_ms, self.hit_count, latency_saved_ms);
        self.hit_count = self.hit_count.saturating_add(1);
        self.refresh_hashes()
    }

    pub fn record_miss(&mut self, latency_ms: f64) -> GraphStoreResult<()> {
        let observations = self.hit_count.saturating_add(self.miss_count);
        self.avg_latency_ms = rolling_average(self.avg_latency_ms, observations, latency_ms);
        self.miss_count = self.miss_count.saturating_add(1);
        self.refresh_hashes()
    }

    pub fn refresh_hashes(&mut self) -> GraphStoreResult<()> {
        self.schema_hash = self.compute_schema_hash()?;
        self.state_hash = self.compute_state_hash()?;
        Ok(())
    }

    fn validate_eval_receipt(&self, receipt: &IndexEvalReceipt) -> GraphStoreResult<()> {
        if receipt.receipt_id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_eval_receipt",
                "eval receipt_id is required",
            ));
        }
        if receipt.manifest_id != self.id {
            return Err(GraphStoreError::new(
                "index_eval_manifest_mismatch",
                format!(
                    "eval receipt manifest_id {} does not match {}",
                    receipt.manifest_id, self.id
                ),
            ));
        }
        let Some(generation_id) = self.generation_id.as_deref() else {
            return Err(GraphStoreError::new(
                "index_generation_missing",
                format!("shadow manifest {} has no generation_id", self.id),
            ));
        };
        if receipt.generation_id != generation_id {
            return Err(GraphStoreError::new(
                "index_eval_generation_mismatch",
                format!(
                    "eval receipt generation_id {} does not match {}",
                    receipt.generation_id, generation_id
                ),
            ));
        }
        let watermark = self.graph_watermark.as_deref().ok_or_else(|| {
            GraphStoreError::new(
                "index_watermark_missing",
                format!("shadow manifest {} has no graph_watermark", self.id),
            )
        })?;
        if receipt.graph_watermark != watermark {
            return Err(GraphStoreError::new(
                "index_eval_watermark_mismatch",
                format!(
                    "eval receipt watermark {} does not match {}",
                    receipt.graph_watermark, watermark
                ),
            ));
        }
        Ok(())
    }

    fn compute_schema_hash(&self) -> GraphStoreResult<String> {
        #[derive(Serialize)]
        struct SchemaHashInput<'a> {
            kind: &'a IndexKind,
            backend: &'a IndexBackend,
            scope: &'a IndexScope,
            target_label: &'a str,
            target_edge_type: &'a Option<String>,
            target_properties: &'a [String],
            predicate: &'a Option<String>,
            sort_keys: &'a [String],
            covering_fields: &'a [String],
            embedding_contract_hash: &'a Option<String>,
            vector_dimension: &'a Option<usize>,
            physical_index_ref: &'a Option<String>,
            serving_compatibility: &'a Option<String>,
        }

        stable_hash(SchemaHashInput {
            kind: &self.kind,
            backend: &self.backend,
            scope: &self.scope,
            target_label: &self.target_label,
            target_edge_type: &self.target_edge_type,
            target_properties: &self.target_properties,
            predicate: &self.predicate,
            sort_keys: &self.sort_keys,
            covering_fields: &self.covering_fields,
            embedding_contract_hash: &self.embedding_contract_hash,
            vector_dimension: &self.vector_dimension,
            physical_index_ref: &self.physical_index_ref,
            serving_compatibility: &self.serving_compatibility,
        })
    }

    fn compute_state_hash(&self) -> GraphStoreResult<String> {
        #[derive(Serialize)]
        struct StateHashInput<'a> {
            id: &'a str,
            name: &'a str,
            schema_hash: &'a str,
            version: u64,
            graph_version: u64,
            build_status: &'a IndexBuildStatus,
            created_by: &'a IndexCreatedBy,
            hit_count: u64,
            miss_count: u64,
            avg_latency_ms: f64,
            avg_latency_saved_ms: f64,
            memory_bytes: u64,
            disk_bytes: u64,
            write_amplification: f64,
            quality_score: f64,
            retirement_reason: &'a Option<String>,
            logical_id: &'a Option<String>,
            generation_id: &'a Option<String>,
            graph_watermark: &'a Option<String>,
            previous_generation: &'a Option<String>,
            eval_receipt_id: Option<&'a str>,
            segment_ids: Vec<&'a str>,
            coverage: &'a Option<IndexCoverage>,
        }

        stable_hash(StateHashInput {
            id: &self.id,
            name: &self.name,
            schema_hash: &self.schema_hash,
            version: self.version,
            graph_version: self.graph_version,
            build_status: &self.build_status,
            created_by: &self.created_by,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            avg_latency_ms: self.avg_latency_ms,
            avg_latency_saved_ms: self.avg_latency_saved_ms,
            memory_bytes: self.memory_bytes,
            disk_bytes: self.disk_bytes,
            write_amplification: self.write_amplification,
            quality_score: self.quality_score,
            retirement_reason: &self.retirement_reason,
            logical_id: &self.logical_id,
            generation_id: &self.generation_id,
            graph_watermark: &self.graph_watermark,
            previous_generation: &self.previous_generation,
            eval_receipt_id: self.eval_receipt.as_ref().map(|r| r.receipt_id.as_str()),
            segment_ids: self.segments.iter().map(|s| s.id.as_str()).collect(),
            coverage: &self.coverage,
        })
    }
}

fn rolling_average(current_avg: f64, current_count: u64, next: f64) -> f64 {
    if current_count == 0 {
        next
    } else {
        ((current_avg * current_count as f64) + next) / (current_count.saturating_add(1) as f64)
    }
}
