use serde::{Deserialize, Serialize};

use crate::index_manifest::IndexManifest;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexRiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexProposalStatus {
    Proposed,
    Rejected,
    ShadowBuilding,
    ShadowActive,
    Promoted,
    Retired,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PromotionThreshold {
    pub min_latency_saved_ms: f64,
    pub min_scan_reduction: f64,
    pub max_recall_drop: f64,
    pub max_write_amplification: f64,
    pub require_scope_policy_ttl_tombstone_filters: bool,
}

impl Default for PromotionThreshold {
    fn default() -> Self {
        Self {
            min_latency_saved_ms: 1.0,
            min_scan_reduction: 0.0,
            max_recall_drop: 0.0,
            max_write_amplification: 2.0,
            require_scope_policy_ttl_tombstone_filters: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexProposal {
    pub id: String,
    pub manifest_draft: IndexManifest,
    pub reason: String,
    #[serde(default)]
    pub supporting_receipts: Vec<String>,
    pub estimated_latency_saved_ms: f64,
    pub estimated_scan_reduction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_token_savings: Option<u64>,
    pub estimated_memory_cost: u64,
    pub estimated_disk_cost: u64,
    pub estimated_write_amplification: f64,
    pub risk_level: IndexRiskLevel,
    pub shadow_validation_plan: String,
    pub promotion_threshold: PromotionThreshold,
    pub status: IndexProposalStatus,
}

impl IndexProposal {
    pub fn new(
        id: impl Into<String>,
        manifest_draft: IndexManifest,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            manifest_draft,
            reason: reason.into(),
            supporting_receipts: Vec::new(),
            estimated_latency_saved_ms: 0.0,
            estimated_scan_reduction: 0.0,
            estimated_token_savings: None,
            estimated_memory_cost: 0,
            estimated_disk_cost: 0,
            estimated_write_amplification: 0.0,
            risk_level: IndexRiskLevel::Low,
            shadow_validation_plan: String::new(),
            promotion_threshold: PromotionThreshold::default(),
            status: IndexProposalStatus::Proposed,
        }
    }

    pub fn with_supporting_receipts(
        mut self,
        receipts: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.supporting_receipts = receipts.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_estimates(
        mut self,
        latency_saved_ms: f64,
        scan_reduction: f64,
        memory_cost: u64,
        disk_cost: u64,
        write_amplification: f64,
    ) -> Self {
        self.estimated_latency_saved_ms = latency_saved_ms;
        self.estimated_scan_reduction = scan_reduction;
        self.estimated_memory_cost = memory_cost;
        self.estimated_disk_cost = disk_cost;
        self.estimated_write_amplification = write_amplification;
        self
    }

    pub fn with_shadow_validation_plan(mut self, plan: impl Into<String>) -> Self {
        self.shadow_validation_plan = plan.into();
        self
    }

    pub fn reject(&mut self) {
        self.status = IndexProposalStatus::Rejected;
    }

    pub fn start_shadow_build(&mut self) {
        self.status = IndexProposalStatus::ShadowBuilding;
    }

    pub fn mark_shadow_active(&mut self) {
        self.status = IndexProposalStatus::ShadowActive;
    }

    pub fn promote(&mut self) {
        self.status = IndexProposalStatus::Promoted;
    }

    pub fn retire(&mut self) {
        self.status = IndexProposalStatus::Retired;
    }
}
