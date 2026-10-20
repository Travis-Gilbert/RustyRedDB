use serde::{Deserialize, Serialize};

use crate::query_receipt::ReceiptScope;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingTaskType {
    Plan,
    Review,
    Fix,
    Explain,
    Search,
    Crawl,
    Route,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingOutcome {
    Success,
    Failure,
    Partial,
    Inconclusive,
}

impl Default for TrainingOutcome {
    fn default() -> Self {
        Self::Inconclusive
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExportStatus {
    Pending,
    Exported,
    Redacted,
    Blocked,
}

impl Default for TrainingExportStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingLabelFamily {
    ContextRanking,
    ToolPolicy,
    AdapterPolicy,
    ValidatorPolicy,
    Memory,
    Map,
    Artifact,
    RunOutcome,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TrainingLabel {
    pub family: TrainingLabelFamily,
    pub label: String,
    pub target_id: String,
}

impl TrainingLabel {
    pub fn new(
        family: TrainingLabelFamily,
        label: impl Into<String>,
        target_id: impl Into<String>,
    ) -> Self {
        Self {
            family,
            label: label.into(),
            target_id: target_id.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ValidatorResult {
    pub validator: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ValidatorResult {
    pub fn new(validator: impl Into<String>, status: impl Into<String>) -> Self {
        Self {
            validator: validator.into(),
            status: status.into(),
            detail: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LabeledTrainingRun {
    pub id: String,
    pub run_id: String,
    pub artifact_id: String,
    pub task_type: TrainingTaskType,
    pub actor: String,
    pub scope: ReceiptScope,
    pub graph_version: u64,
    #[serde(default)]
    pub candidate_atom_ids: Vec<String>,
    #[serde(default)]
    pub included_atom_ids: Vec<String>,
    #[serde(default)]
    pub excluded_atom_ids: Vec<String>,
    #[serde(default)]
    pub cited_atom_ids: Vec<String>,
    #[serde(default)]
    pub dismissed_atom_ids: Vec<String>,
    #[serde(default)]
    pub map_section_ids: Vec<String>,
    #[serde(default)]
    pub tool_candidates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_selected: Option<String>,
    #[serde(default)]
    pub adapter_candidates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_selected: Option<String>,
    #[serde(default)]
    pub validator_results: Vec<ValidatorResult>,
    pub outcome: TrainingOutcome,
    pub accepted: bool,
    #[serde(default)]
    pub labels: Vec<TrainingLabel>,
    pub export_status: TrainingExportStatus,
}

impl LabeledTrainingRun {
    pub fn new(
        id: impl Into<String>,
        run_id: impl Into<String>,
        artifact_id: impl Into<String>,
        task_type: TrainingTaskType,
        actor: impl Into<String>,
        scope: ReceiptScope,
        graph_version: u64,
    ) -> Self {
        Self {
            id: id.into(),
            run_id: run_id.into(),
            artifact_id: artifact_id.into(),
            task_type,
            actor: actor.into(),
            scope,
            graph_version,
            candidate_atom_ids: Vec::new(),
            included_atom_ids: Vec::new(),
            excluded_atom_ids: Vec::new(),
            cited_atom_ids: Vec::new(),
            dismissed_atom_ids: Vec::new(),
            map_section_ids: Vec::new(),
            tool_candidates: Vec::new(),
            tool_selected: None,
            adapter_candidates: Vec::new(),
            adapter_selected: None,
            validator_results: Vec::new(),
            outcome: TrainingOutcome::Inconclusive,
            accepted: false,
            labels: Vec::new(),
            export_status: TrainingExportStatus::Pending,
        }
    }

    pub fn add_label(&mut self, label: TrainingLabel) {
        self.labels.push(label);
    }

    pub fn has_positive_context_labels(&self) -> bool {
        !self.included_atom_ids.is_empty()
            || !self.cited_atom_ids.is_empty()
            || self.labels.iter().any(|label| {
                label.family == TrainingLabelFamily::ContextRanking
                    && matches!(
                        label.label.as_str(),
                        "included" | "cited" | "useful" | "accepted"
                    )
            })
    }

    pub fn has_negative_context_labels(&self) -> bool {
        !self.excluded_atom_ids.is_empty()
            || !self.dismissed_atom_ids.is_empty()
            || self.labels.iter().any(|label| {
                label.family == TrainingLabelFamily::ContextRanking
                    && matches!(
                        label.label.as_str(),
                        "excluded" | "dismissed" | "wasted" | "redundant" | "stale" | "blocked"
                    )
            })
    }

    pub fn is_exportable(&self) -> bool {
        self.export_status == TrainingExportStatus::Pending
            && self.has_positive_context_labels()
            && self.has_negative_context_labels()
    }
}
