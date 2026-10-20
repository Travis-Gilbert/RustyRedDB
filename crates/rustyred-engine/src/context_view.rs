use serde::{Deserialize, Serialize};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::query_receipt::ReceiptScope;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextViewType {
    RepoOnboarding,
    PrReview,
    BugFix,
    Research,
    Postmortem,
    ToolSelection,
    UserMemory,
    RuleMap,
    ProjectMap,
    TrainingSlice,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessStatus {
    Fresh,
    Stale,
    Drifted,
    NeedsRebuild,
}

impl Default for FreshnessStatus {
    fn default() -> Self {
        Self::Fresh
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HydrationHandle {
    pub object_id: String,
    pub label: String,
    pub graph_version: u64,
    pub handle: String,
}

impl HydrationHandle {
    pub fn new(
        object_id: impl Into<String>,
        label: impl Into<String>,
        graph_version: u64,
        handle: impl Into<String>,
    ) -> Self {
        Self {
            object_id: object_id.into(),
            label: label.into(),
            graph_version,
            handle: handle.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContextView {
    pub id: String,
    pub view_type: ContextViewType,
    pub scope: ReceiptScope,
    #[serde(default)]
    pub source_artifact_ids: Vec<String>,
    #[serde(default)]
    pub source_run_ids: Vec<String>,
    #[serde(default)]
    pub source_map_ids: Vec<String>,
    pub source_graph_version: u64,
    #[serde(default)]
    pub included_atom_ids: Vec<String>,
    #[serde(default)]
    pub excluded_atom_ids: Vec<String>,
    #[serde(default)]
    pub positive_label_ids: Vec<String>,
    #[serde(default)]
    pub negative_label_ids: Vec<String>,
    pub materialized_summary: String,
    #[serde(default)]
    pub hydration_handles: Vec<HydrationHandle>,
    pub freshness_status: FreshnessStatus,
    pub version: u64,
    pub export_eligible: bool,
}

impl ContextView {
    pub fn new(
        id: impl Into<String>,
        view_type: ContextViewType,
        scope: ReceiptScope,
        source_graph_version: u64,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            view_type,
            scope,
            source_artifact_ids: Vec::new(),
            source_run_ids: Vec::new(),
            source_map_ids: Vec::new(),
            source_graph_version,
            included_atom_ids: Vec::new(),
            excluded_atom_ids: Vec::new(),
            positive_label_ids: Vec::new(),
            negative_label_ids: Vec::new(),
            materialized_summary: summary.into(),
            hydration_handles: Vec::new(),
            freshness_status: FreshnessStatus::Fresh,
            version: 1,
            export_eligible: false,
        }
    }

    pub fn has_atom_identity(&self) -> bool {
        !self.included_atom_ids.is_empty()
            || !self.excluded_atom_ids.is_empty()
            || !self.positive_label_ids.is_empty()
            || !self.negative_label_ids.is_empty()
    }

    pub fn validate_not_summary_only(&self) -> GraphStoreResult<()> {
        if self.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_context_view",
                "context view id is required",
            ));
        }
        if self.materialized_summary.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_context_view",
                "context view summary is required",
            ));
        }
        if !self.has_atom_identity() {
            return Err(GraphStoreError::new(
                "summary_only_context_view",
                "context view must preserve included, excluded, positive, or negative atom identity",
            ));
        }
        if self.hydration_handles.is_empty() {
            return Err(GraphStoreError::new(
                "context_view_missing_handles",
                "context view must include hydration handles",
            ));
        }
        Ok(())
    }

    pub fn mark_stale(&mut self) {
        self.freshness_status = FreshnessStatus::Stale;
        self.version = self.version.saturating_add(1);
    }

    pub fn add_hydration_handle(&mut self, handle: HydrationHandle) {
        self.hydration_handles.push(handle);
    }
}
