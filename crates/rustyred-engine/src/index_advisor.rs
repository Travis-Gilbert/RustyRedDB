use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::index_manifest::IndexManifest;
use crate::index_proposal::IndexProposal;
use crate::query_receipt::{QueryKind, QueryReceipt, ReceiptScope};
use crate::state::stable_hash;
use crate::GraphStoreResult;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexAdvisorConfig {
    pub min_repeated_receipts: usize,
    pub min_total_full_scans: usize,
    pub min_total_latency_ms: f64,
    pub min_candidate_waste_ratio: f64,
    pub min_token_cost: u64,
    pub min_recall_missing: usize,
    pub min_cold_reads: usize,
}

impl Default for IndexAdvisorConfig {
    fn default() -> Self {
        Self {
            min_repeated_receipts: 2,
            min_total_full_scans: 2,
            min_total_latency_ms: 500.0,
            min_candidate_waste_ratio: 4.0,
            min_token_cost: 1_000,
            min_recall_missing: 1,
            min_cold_reads: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexPainKind {
    FullScan,
    Latency,
    CandidateWaste,
    TokenCost,
    PoorRecall,
    ColdRead,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ReceiptCluster {
    pub cluster_key: String,
    pub query_signature: String,
    pub query_kind: QueryKind,
    pub scope: ReceiptScope,
    pub access_path_keys: Vec<String>,
    pub receipt_ids: Vec<String>,
    pub total_full_scans: usize,
    pub total_results: usize,
    pub total_candidate_count: usize,
    pub total_latency_ms: f64,
    pub total_token_cost: u64,
    pub total_cold_reads: usize,
    pub recall_missing_count: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IndexPainSignal {
    pub pain_kind: IndexPainKind,
    pub query_signature: String,
    pub supporting_receipts: Vec<String>,
    pub total_full_scans: usize,
    pub total_results: usize,
    pub total_candidate_count: usize,
    pub total_latency_ms: f64,
    pub total_token_cost: u64,
    pub total_cold_reads: usize,
    pub recall_missing_count: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ShadowValidationReport {
    pub replayed_receipts: Vec<String>,
    pub latency_saved_ms: f64,
    pub scan_reduction: f64,
    pub recall_drop: f64,
    pub write_amplification: f64,
    pub scope_policy_ttl_tombstone_filters_enforced: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain_manifest_id: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct IndexAdvisor {
    config: IndexAdvisorConfig,
}

impl IndexAdvisor {
    pub fn new(config: IndexAdvisorConfig) -> Self {
        Self { config }
    }

    pub fn detect_full_scan_pain<'a>(
        &self,
        receipts: impl IntoIterator<Item = &'a QueryReceipt>,
    ) -> GraphStoreResult<Vec<IndexPainSignal>> {
        Ok(self
            .detect_pain(receipts)?
            .into_iter()
            .filter(|signal| signal.pain_kind == IndexPainKind::FullScan)
            .collect())
    }

    pub fn cluster_receipts<'a>(
        &self,
        receipts: impl IntoIterator<Item = &'a QueryReceipt>,
    ) -> GraphStoreResult<Vec<ReceiptCluster>> {
        #[derive(Default)]
        struct Accumulator {
            receipt_ids: Vec<String>,
            total_full_scans: usize,
            total_results: usize,
            total_candidate_count: usize,
            total_latency_ms: f64,
            total_token_cost: u64,
            total_cold_reads: usize,
            recall_missing_count: usize,
            access_path_keys: Vec<String>,
        }

        let mut metadata = BTreeMap::new();
        let mut accumulators: BTreeMap<String, Accumulator> = BTreeMap::new();
        for receipt in receipts {
            let mut access_path_keys = receipt
                .access_paths_used
                .iter()
                .map(|path| path.stage_key())
                .collect::<Vec<_>>();
            access_path_keys.sort();
            access_path_keys.dedup();
            let cluster_key = receipt_cluster_key(receipt, &access_path_keys)?;
            metadata.entry(cluster_key.clone()).or_insert_with(|| {
                (
                    receipt.query_signature.clone(),
                    receipt.query_kind.clone(),
                    receipt.scope.clone(),
                )
            });
            let entry = accumulators.entry(cluster_key).or_default();
            entry.receipt_ids.push(receipt.id.clone());
            entry.total_full_scans = entry
                .total_full_scans
                .saturating_add(receipt.full_scan_count);
            entry.total_results = entry.total_results.saturating_add(receipt.result_count);
            entry.total_candidate_count = entry
                .total_candidate_count
                .saturating_add(candidate_count(receipt));
            entry.total_latency_ms += receipt.latency_by_stage_ms.values().sum::<f64>();
            entry.total_token_cost = entry
                .total_token_cost
                .saturating_add(receipt.token_cost.unwrap_or(0));
            entry.total_cold_reads = entry.total_cold_reads.saturating_add(cold_reads(receipt));
            entry.recall_missing_count = entry
                .recall_missing_count
                .saturating_add(recall_missing(receipt));
            entry.access_path_keys.extend(access_path_keys);
            entry.access_path_keys.sort();
            entry.access_path_keys.dedup();
        }

        Ok(accumulators
            .into_iter()
            .map(|(cluster_key, accumulator)| {
                let (query_signature, query_kind, scope) =
                    metadata.remove(&cluster_key).expect("cluster metadata");
                ReceiptCluster {
                    cluster_key,
                    query_signature,
                    query_kind,
                    scope,
                    access_path_keys: accumulator.access_path_keys,
                    receipt_ids: accumulator.receipt_ids,
                    total_full_scans: accumulator.total_full_scans,
                    total_results: accumulator.total_results,
                    total_candidate_count: accumulator.total_candidate_count,
                    total_latency_ms: accumulator.total_latency_ms,
                    total_token_cost: accumulator.total_token_cost,
                    total_cold_reads: accumulator.total_cold_reads,
                    recall_missing_count: accumulator.recall_missing_count,
                }
            })
            .collect())
    }

    pub fn detect_pain<'a>(
        &self,
        receipts: impl IntoIterator<Item = &'a QueryReceipt>,
    ) -> GraphStoreResult<Vec<IndexPainSignal>> {
        let mut signals = Vec::new();
        for cluster in self.cluster_receipts(receipts)? {
            if cluster.receipt_ids.len() >= self.config.min_repeated_receipts
                && cluster.total_full_scans >= self.config.min_total_full_scans
            {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::FullScan,
                    format!(
                        "query family performed {} full scans across {} receipts",
                        cluster.total_full_scans,
                        cluster.receipt_ids.len()
                    ),
                ));
            }
            if cluster.total_latency_ms >= self.config.min_total_latency_ms {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::Latency,
                    format!(
                        "query family spent {:.1} ms across stages",
                        cluster.total_latency_ms
                    ),
                ));
            }
            if cluster.total_results > 0
                && (cluster.total_candidate_count as f64 / cluster.total_results as f64)
                    >= self.config.min_candidate_waste_ratio
            {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::CandidateWaste,
                    format!(
                        "query family considered {} candidates for {} returned results",
                        cluster.total_candidate_count, cluster.total_results
                    ),
                ));
            }
            if cluster.total_token_cost >= self.config.min_token_cost {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::TokenCost,
                    format!(
                        "query family consumed {} context tokens",
                        cluster.total_token_cost
                    ),
                ));
            }
            if cluster.recall_missing_count >= self.config.min_recall_missing {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::PoorRecall,
                    format!(
                        "query family missed {} exact-oracle results",
                        cluster.recall_missing_count
                    ),
                ));
            }
            if cluster.total_cold_reads >= self.config.min_cold_reads {
                signals.push(signal_for_cluster(
                    &cluster,
                    IndexPainKind::ColdRead,
                    format!(
                        "query family touched {} cold fragments",
                        cluster.total_cold_reads
                    ),
                ));
            }
        }
        Ok(signals)
    }

    pub fn proposal_for_pain(
        &self,
        id: impl Into<String>,
        manifest_draft: IndexManifest,
        pain: &IndexPainSignal,
    ) -> IndexProposal {
        IndexProposal::new(id, manifest_draft, pain.reason.clone())
            .with_supporting_receipts(pain.supporting_receipts.clone())
            .with_estimates(
                pain.total_full_scans as f64,
                pain.total_full_scans as f64,
                pain.total_candidate_count.saturating_mul(64) as u64,
                pain.total_cold_reads.saturating_mul(128) as u64,
                0.0,
            )
            .with_shadow_validation_plan("replay supporting receipts against shadow index")
    }

    pub fn apply_shadow_validation(
        &self,
        proposal: &mut IndexProposal,
        report: &ShadowValidationReport,
    ) -> bool {
        let threshold = &proposal.promotion_threshold;
        let explain_matches = report
            .explain_manifest_id
            .as_ref()
            .map(|id| id == &proposal.manifest_draft.id)
            .unwrap_or(false);
        let filters_ok = !threshold.require_scope_policy_ttl_tombstone_filters
            || report.scope_policy_ttl_tombstone_filters_enforced;
        let accepted = report.latency_saved_ms >= threshold.min_latency_saved_ms
            && report.scan_reduction >= threshold.min_scan_reduction
            && report.recall_drop <= threshold.max_recall_drop
            && report.write_amplification <= threshold.max_write_amplification
            && filters_ok
            && explain_matches;
        if accepted {
            proposal.promote();
        } else {
            proposal.reject();
        }
        accepted
    }

    pub fn retire_unused_or_harmful(
        &self,
        manifest: &mut IndexManifest,
        min_observations: u64,
    ) -> GraphStoreResult<bool> {
        let observations = manifest.hit_count.saturating_add(manifest.miss_count);
        let harmful = observations >= min_observations
            && (manifest.hit_count == 0
                || manifest.miss_count > manifest.hit_count.saturating_mul(4)
                || manifest.quality_score < 0.0);
        if harmful {
            manifest.retire("advisor_retired_unused_or_harmful_index")?;
        }
        Ok(harmful)
    }
}

fn receipt_cluster_key(
    receipt: &QueryReceipt,
    access_path_keys: &[String],
) -> GraphStoreResult<String> {
    #[derive(Serialize)]
    struct Shape<'a> {
        query_signature: &'a str,
        query_kind: &'a QueryKind,
        scope: &'a ReceiptScope,
        access_path_keys: &'a [String],
    }

    stable_hash(Shape {
        query_signature: &receipt.query_signature,
        query_kind: &receipt.query_kind,
        scope: &receipt.scope,
        access_path_keys,
    })
}

fn signal_for_cluster(
    cluster: &ReceiptCluster,
    pain_kind: IndexPainKind,
    reason: String,
) -> IndexPainSignal {
    IndexPainSignal {
        pain_kind,
        query_signature: cluster.query_signature.clone(),
        supporting_receipts: cluster.receipt_ids.clone(),
        total_full_scans: cluster.total_full_scans,
        total_results: cluster.total_results,
        total_candidate_count: cluster.total_candidate_count,
        total_latency_ms: cluster.total_latency_ms,
        total_token_cost: cluster.total_token_cost,
        total_cold_reads: cluster.total_cold_reads,
        recall_missing_count: cluster.recall_missing_count,
        reason,
    }
}

fn candidate_count(receipt: &QueryReceipt) -> usize {
    receipt
        .candidate_counts_by_stage
        .get("candidate_set")
        .copied()
        .unwrap_or_else(|| receipt.candidate_counts_by_stage.values().sum())
}

fn cold_reads(receipt: &QueryReceipt) -> usize {
    receipt
        .candidate_counts_by_stage
        .get("cold_reads")
        .or_else(|| {
            receipt
                .candidate_counts_by_stage
                .get("cold_fragments_visited")
        })
        .copied()
        .unwrap_or(0)
}

fn recall_missing(receipt: &QueryReceipt) -> usize {
    receipt
        .candidate_counts_by_stage
        .get("recall_missing")
        .or_else(|| receipt.candidate_counts_by_stage.get("missing_recall"))
        .copied()
        .unwrap_or(0)
}
