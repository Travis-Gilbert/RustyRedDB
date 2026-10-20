use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::index_manifest::{
    IndexBackend, IndexBuildStatus, IndexCreatedBy, IndexKind, IndexManifest, IndexScope,
};
use crate::query_receipt::ReceiptScope;
use crate::GraphStoreResult;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorSearchBackend {
    ExactCosine,
    Turbovec,
}

impl VectorSearchBackend {
    fn index_backend(self) -> IndexBackend {
        match self {
            Self::ExactCosine => IndexBackend::RustyredCore,
            Self::Turbovec => IndexBackend::Turbovec,
        }
    }
}

impl Default for VectorSearchBackend {
    fn default() -> Self {
        Self::ExactCosine
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VectorSearchWork {
    pub backend: VectorSearchBackend,
    pub indexed_points: usize,
    pub distance_evaluations: usize,
    pub retained_candidates: usize,
    pub accelerated_candidates: usize,
    pub index_rebuilds: usize,
    /// Stable fallback evidence. One unique reason remains byte-for-byte
    /// unchanged. Multiple reasons are sorted, deduplicated, escaped, and
    /// joined with `;`; `\\` escapes a literal `\\` or `;` inside a reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VectorSearchReceipt {
    pub hits: Vec<(String, f32)>,
    pub work: VectorSearchWork,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VectorIndexDefinition {
    pub manifest_id: String,
    pub target_label: String,
    pub property: String,
    pub dimension: usize,
    pub backend: VectorSearchBackend,
    #[serde(default)]
    pub scope_fields: Vec<String>,
}

impl VectorIndexDefinition {
    pub fn new(
        manifest_id: impl Into<String>,
        target_label: impl Into<String>,
        property: impl Into<String>,
        dimension: usize,
        backend: VectorSearchBackend,
    ) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            target_label: target_label.into(),
            property: property.into(),
            dimension,
            backend,
            scope_fields: Vec::new(),
        }
    }

    pub fn with_scope_fields(
        mut self,
        scope_fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.scope_fields = scope_fields.into_iter().map(Into::into).collect();
        self
    }

    pub fn to_manifest(
        &self,
        scope: IndexScope,
        created_by: IndexCreatedBy,
    ) -> GraphStoreResult<IndexManifest> {
        let mut manifest = IndexManifest::new(
            self.manifest_id.clone(),
            format!("{} {} vector", self.target_label, self.property),
            IndexKind::Vector,
            self.backend.index_backend(),
            scope,
            self.target_label.clone(),
            created_by,
        )?
        .with_target_properties([self.property.clone()])?;
        manifest.build_status = IndexBuildStatus::Active;
        manifest.memory_bytes = self.dimension.saturating_mul(4) as u64;
        manifest.refresh_hashes()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorSearchCandidate {
    pub object_id: String,
    pub distance: f32,
    pub scope: ReceiptScope,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_status: Option<String>,
    pub policy_allowed: bool,
    pub tombstone: bool,
    pub ttl_expired: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl VectorSearchCandidate {
    pub fn new(object_id: impl Into<String>, distance: f32, scope: ReceiptScope) -> Self {
        Self {
            object_id: object_id.into(),
            distance,
            scope,
            labels: Vec::new(),
            repo: None,
            user: None,
            trust_status: None,
            freshness_status: None,
            policy_allowed: true,
            tombstone: false,
            ttl_expired: false,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_repo(mut self, repo: impl Into<String>) -> Self {
        self.repo = Some(repo.into());
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_trust_status(mut self, trust_status: impl Into<String>) -> Self {
        self.trust_status = Some(trust_status.into());
        self
    }

    pub fn with_freshness_status(mut self, freshness_status: impl Into<String>) -> Self {
        self.freshness_status = Some(freshness_status.into());
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct VectorFilterPolicy {
    pub scope: ReceiptScope,
    #[serde(default)]
    pub required_labels: BTreeSet<String>,
    #[serde(default)]
    pub allowed_trust_statuses: BTreeSet<String>,
    #[serde(default)]
    pub allowed_freshness_statuses: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    pub include_tombstones: bool,
    pub include_ttl_expired: bool,
    pub require_policy_allowed: bool,
}

impl VectorFilterPolicy {
    pub fn scoped(scope: ReceiptScope) -> Self {
        Self {
            scope,
            require_policy_allowed: true,
            ..Self::default()
        }
    }

    pub fn require_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.required_labels = labels.into_iter().map(Into::into).collect();
        self
    }

    pub fn allow_trust_statuses(
        mut self,
        statuses: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_trust_statuses = statuses.into_iter().map(Into::into).collect();
        self
    }

    pub fn allow_freshness_statuses(
        mut self,
        statuses: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_freshness_statuses = statuses.into_iter().map(Into::into).collect();
        self
    }

    pub fn for_repo(mut self, repo: impl Into<String>) -> Self {
        self.repo = Some(repo.into());
        self
    }

    pub fn for_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn accepts(&self, candidate: &VectorSearchCandidate) -> bool {
        for (field, expected) in &self.scope {
            if candidate.scope.get(field) != Some(expected) {
                return false;
            }
        }
        if self.require_policy_allowed && !candidate.policy_allowed {
            return false;
        }
        if !self.include_tombstones && candidate.tombstone {
            return false;
        }
        if !self.include_ttl_expired && candidate.ttl_expired {
            return false;
        }
        if !self.required_labels.is_empty()
            && !self.required_labels.iter().all(|label| {
                candidate
                    .labels
                    .iter()
                    .any(|candidate_label| candidate_label == label)
            })
        {
            return false;
        }
        if !self.allowed_trust_statuses.is_empty()
            && candidate
                .trust_status
                .as_ref()
                .map(|status| !self.allowed_trust_statuses.contains(status))
                .unwrap_or(true)
        {
            return false;
        }
        if !self.allowed_freshness_statuses.is_empty()
            && candidate
                .freshness_status
                .as_ref()
                .map(|status| !self.allowed_freshness_statuses.contains(status))
                .unwrap_or(true)
        {
            return false;
        }
        if let Some(repo) = &self.repo {
            if candidate.repo.as_ref() != Some(repo) {
                return false;
            }
        }
        if let Some(user) = &self.user {
            if candidate.user.as_ref() != Some(user) {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorRecallReport {
    pub exact_top_k: usize,
    pub candidate_top_k: usize,
    pub exact_count: usize,
    pub candidate_count: usize,
    pub overlap_count: usize,
    pub recall: f32,
    pub missing_object_ids: Vec<String>,
}

pub fn filter_vector_candidates(
    candidates: &[VectorSearchCandidate],
    policy: &VectorFilterPolicy,
    k: usize,
) -> Vec<VectorSearchCandidate> {
    if k == 0 {
        return Vec::new();
    }
    let mut filtered = candidates
        .iter()
        .filter(|candidate| candidate.distance.is_finite() && policy.accepts(candidate))
        .cloned()
        .collect::<Vec<_>>();
    sort_vector_candidates(&mut filtered);
    filtered.truncate(k);
    filtered
}

pub fn vector_recall_against_exact(
    exact_ranked: &[VectorSearchCandidate],
    candidate_ranked: &[VectorSearchCandidate],
    exact_top_k: usize,
    candidate_top_k: usize,
) -> VectorRecallReport {
    let exact_count = exact_ranked.len().min(exact_top_k);
    let candidate_count = candidate_ranked.len().min(candidate_top_k);
    let candidate_ids = candidate_ranked
        .iter()
        .take(candidate_count)
        .map(|candidate| candidate.object_id.as_str())
        .collect::<BTreeSet<_>>();

    let mut overlap_count = 0;
    let mut missing_object_ids = Vec::new();
    for candidate in exact_ranked.iter().take(exact_count) {
        if candidate_ids.contains(candidate.object_id.as_str()) {
            overlap_count += 1;
        } else {
            missing_object_ids.push(candidate.object_id.clone());
        }
    }

    VectorRecallReport {
        exact_top_k,
        candidate_top_k,
        exact_count,
        candidate_count,
        overlap_count,
        recall: if exact_count == 0 {
            0.0
        } else {
            overlap_count as f32 / exact_count as f32
        },
        missing_object_ids,
    }
}

fn sort_vector_candidates(candidates: &mut [VectorSearchCandidate]) {
    candidates.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.object_id.cmp(&right.object_id))
    });
}
