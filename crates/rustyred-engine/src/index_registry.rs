use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::index_manifest::{IndexBuildStatus, IndexEvalReceipt, IndexManifest, ServingIndexView};
use crate::index_proposal::IndexProposal;
use crate::query_receipt::QueryReceipt;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
struct GenerationSlot {
    /// Manifest id currently served for this logical index.
    active_id: Option<String>,
    /// Prior Active kept for rollback (not served).
    previous_id: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct IndexRegistry {
    manifests: BTreeMap<String, IndexManifest>,
    receipts: BTreeMap<String, QueryReceipt>,
    proposals: BTreeMap<String, IndexProposal>,
    /// logical_id → serving / rollback slot. Query service reads this only.
    serving: BTreeMap<String, GenerationSlot>,
    /// Durable eval receipts keyed by receipt_id.
    eval_receipts: BTreeMap<String, IndexEvalReceipt>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_manifest(&mut self, manifest: IndexManifest) -> GraphStoreResult<()> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.id) {
            return Err(GraphStoreError::new(
                "index_manifest_exists",
                format!("index manifest {} is already registered", manifest.id),
            ));
        }
        let should_serve = manifest.build_status == IndexBuildStatus::Active
            && manifest.generation_id.is_some()
            && manifest.graph_watermark.is_some();
        let logical_id = manifest.logical_index_id().to_string();
        let manifest_id = manifest.id.clone();
        self.manifests.insert(manifest.id.clone(), manifest);
        if should_serve {
            self.serving.insert(
                logical_id,
                GenerationSlot {
                    active_id: Some(manifest_id),
                    previous_id: None,
                },
            );
        }
        Ok(())
    }

    pub fn upsert_manifest(&mut self, mut manifest: IndexManifest) -> GraphStoreResult<()> {
        manifest.validate()?;
        manifest.refresh_hashes()?;
        if let Some(existing) = self.manifests.get(&manifest.id) {
            if existing == &manifest {
                return Ok(());
            }
            if existing.generation_id.is_some() || manifest.generation_id.is_some() {
                return Err(GraphStoreError::new(
                    "index_generation_immutable",
                    format!(
                        "generation manifest {} cannot be changed outside lifecycle transitions",
                        manifest.id
                    ),
                ));
            }
        }
        self.manifests.insert(manifest.id.clone(), manifest);
        Ok(())
    }

    pub fn get_manifest(&self, id: &str) -> Option<&IndexManifest> {
        self.manifests.get(id)
    }

    pub fn list_manifests(&self) -> Vec<&IndexManifest> {
        self.manifests.values().collect()
    }

    pub fn retire_manifest(&mut self, id: &str, reason: impl Into<String>) -> GraphStoreResult<()> {
        let Some(manifest) = self.manifests.get_mut(id) else {
            return Err(GraphStoreError::new(
                "index_manifest_not_found",
                format!("index manifest {id} is not registered"),
            ));
        };
        let logical_id = manifest.logical_index_id().to_string();
        manifest.retire(reason)?;
        if let Some(slot) = self.serving.get_mut(&logical_id) {
            if slot.active_id.as_deref() == Some(id) {
                slot.active_id = None;
            }
            if slot.previous_id.as_deref() == Some(id) {
                slot.previous_id = None;
            }
        }
        Ok(())
    }

    /// Register a Shadow generation for a logical index (build N+1).
    pub fn register_shadow_generation(&mut self, manifest: IndexManifest) -> GraphStoreResult<()> {
        manifest.validate()?;
        if manifest.build_status != IndexBuildStatus::Shadow {
            return Err(GraphStoreError::new(
                "index_shadow_required",
                format!(
                    "register_shadow_generation requires Shadow status, found {:?}",
                    manifest.build_status
                ),
            ));
        }
        if manifest
            .generation_id
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            return Err(GraphStoreError::new(
                "invalid_index_generation",
                "shadow generation requires generation_id",
            ));
        }
        if manifest
            .graph_watermark
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            return Err(GraphStoreError::new(
                "invalid_index_watermark",
                "shadow generation requires graph_watermark",
            ));
        }
        if self.manifests.contains_key(&manifest.id) {
            return Err(GraphStoreError::new(
                "index_manifest_exists",
                format!("index manifest {} is already registered", manifest.id),
            ));
        }
        // Shadow never enters the serving slot.
        self.manifests.insert(manifest.id.clone(), manifest);
        Ok(())
    }

    /// Persist an eval receipt and bind it to the Shadow manifest.
    pub fn attach_eval_receipt(
        &mut self,
        manifest_id: &str,
        receipt: IndexEvalReceipt,
    ) -> GraphStoreResult<()> {
        let Some(manifest) = self.manifests.get_mut(manifest_id) else {
            return Err(GraphStoreError::new(
                "index_manifest_not_found",
                format!("index manifest {manifest_id} is not registered"),
            ));
        };
        manifest.attach_eval_receipt(receipt.clone())?;
        self.eval_receipts
            .insert(receipt.receipt_id.clone(), receipt);
        Ok(())
    }

    pub fn get_eval_receipt(&self, receipt_id: &str) -> Option<&IndexEvalReceipt> {
        self.eval_receipts.get(receipt_id)
    }

    /// Atomically promote a Shadow generation to Active from the query-service POV.
    ///
    /// Either the serving slot still points at N, or it points at N+1 with generation
    /// id + watermark — never a half-applied flip.
    pub fn activate_shadow_generation(
        &mut self,
        shadow_manifest_id: &str,
    ) -> GraphStoreResult<ServingIndexView> {
        let mut staged = self.clone();
        let view = staged.activate_shadow_generation_in_place(shadow_manifest_id)?;
        *self = staged;
        Ok(view)
    }

    fn activate_shadow_generation_in_place(
        &mut self,
        shadow_manifest_id: &str,
    ) -> GraphStoreResult<ServingIndexView> {
        let shadow = self.manifests.get(shadow_manifest_id).ok_or_else(|| {
            GraphStoreError::new(
                "index_manifest_not_found",
                format!("index manifest {shadow_manifest_id} is not registered"),
            )
        })?;
        if shadow.build_status != IndexBuildStatus::Shadow {
            return Err(GraphStoreError::new(
                "index_shadow_required",
                format!(
                    "activate_shadow_generation requires Shadow, found {:?}",
                    shadow.build_status
                ),
            ));
        }
        let receipt = shadow.eval_receipt.clone().ok_or_else(|| {
            GraphStoreError::new(
                "index_shadow_activation_requires_eval",
                format!(
                    "shadow generation {} cannot become Active without an eval receipt",
                    shadow
                        .generation_id
                        .as_deref()
                        .unwrap_or(shadow_manifest_id)
                ),
            )
        })?;
        if !receipt.passed {
            return Err(GraphStoreError::new(
                "index_eval_receipt_failed",
                format!(
                    "eval receipt {} did not pass; refusing activation",
                    receipt.receipt_id
                ),
            ));
        }

        let logical_id = shadow.logical_index_id().to_string();
        let graph_version = shadow.graph_version;
        let previous_active_id = self
            .serving
            .get(&logical_id)
            .and_then(|slot| slot.active_id.clone());
        let retire_id = self
            .serving
            .get(&logical_id)
            .and_then(|slot| slot.previous_id.clone());
        let previous_generation_label = previous_active_id
            .as_ref()
            .and_then(|id| self.manifests.get(id))
            .and_then(|m| m.generation_id.clone())
            .or_else(|| previous_active_id.clone());

        // Apply demotions / promotions in one &mut self critical section.
        if let Some(old_previous) = retire_id {
            if let Some(manifest) = self.manifests.get_mut(&old_previous) {
                manifest.retire("superseded_by_newer_generation")?;
            }
        }

        // The prior serving generation remains available for rollback, but it
        // must stop advertising itself as Active once N+1 is promoted. The
        // serving slot is authoritative for reads; Shadow accurately records
        // that the retained generation is standby-only.
        if let Some(previous_active) = previous_active_id.as_ref() {
            if let Some(manifest) = self.manifests.get_mut(previous_active) {
                manifest.build_status = IndexBuildStatus::Shadow;
                manifest.refresh_hashes()?;
            }
        }

        {
            let shadow = self
                .manifests
                .get_mut(shadow_manifest_id)
                .expect("shadow present");
            shadow.previous_generation = previous_generation_label;
            // Re-bind receipt (already on shadow) and flip status.
            shadow.activate_with_eval(graph_version, receipt)?;
        }

        self.serving.insert(
            logical_id.clone(),
            GenerationSlot {
                active_id: Some(shadow_manifest_id.to_string()),
                previous_id: previous_active_id,
            },
        );

        self.serving_view(&logical_id)?.ok_or_else(|| {
            GraphStoreError::new(
                "index_serving_missing",
                format!("serving view missing after activating {shadow_manifest_id}"),
            )
        })
    }

    /// Roll serving back to the previous generation (N) for `logical_id`.
    pub fn rollback_generation(&mut self, logical_id: &str) -> GraphStoreResult<ServingIndexView> {
        let mut staged = self.clone();
        let view = staged.rollback_generation_in_place(logical_id)?;
        *self = staged;
        Ok(view)
    }

    fn rollback_generation_in_place(
        &mut self,
        logical_id: &str,
    ) -> GraphStoreResult<ServingIndexView> {
        let slot = self.serving.get(logical_id).cloned().ok_or_else(|| {
            GraphStoreError::new(
                "index_serving_missing",
                format!("no serving slot for logical index {logical_id}"),
            )
        })?;
        let previous_id = slot.previous_id.clone().ok_or_else(|| {
            GraphStoreError::new(
                "index_rollback_unavailable",
                format!("no previous generation to roll back for {logical_id}"),
            )
        })?;
        let current_id = slot.active_id.clone();

        let previous = self.manifests.get(&previous_id).ok_or_else(|| {
            GraphStoreError::new(
                "index_manifest_not_found",
                format!("previous generation manifest {previous_id} is not registered"),
            )
        })?;
        if previous.generation_id.is_none() || previous.graph_watermark.is_none() {
            return Err(GraphStoreError::new(
                "index_rollback_incomplete",
                format!("previous generation {previous_id} lacks generation_id or graph_watermark"),
            ));
        }

        // Demote current Active out of serving (keep for potential re-promote).
        if let Some(ref active_id) = current_id {
            if let Some(manifest) = self.manifests.get_mut(active_id) {
                manifest.build_status = IndexBuildStatus::Shadow;
                manifest.refresh_hashes()?;
            }
        }

        // Restore previous as Active serving.
        {
            let prev = self
                .manifests
                .get_mut(&previous_id)
                .expect("previous present");
            prev.build_status = IndexBuildStatus::Active;
            prev.retirement_reason = None;
            prev.refresh_hashes()?;
        }

        self.serving.insert(
            logical_id.to_string(),
            GenerationSlot {
                active_id: Some(previous_id),
                previous_id: current_id,
            },
        );

        self.serving_view(logical_id)?.ok_or_else(|| {
            GraphStoreError::new(
                "index_serving_missing",
                format!("serving view missing after rollback of {logical_id}"),
            )
        })
    }

    /// Query-service read of the Active generation (generation id + watermark).
    pub fn serving_view(&self, logical_id: &str) -> GraphStoreResult<Option<ServingIndexView>> {
        let Some(slot) = self.serving.get(logical_id) else {
            return Ok(None);
        };
        let Some(active_id) = slot.active_id.as_deref() else {
            return Ok(None);
        };
        let Some(manifest) = self.manifests.get(active_id) else {
            return Err(GraphStoreError::new(
                "index_manifest_not_found",
                format!("serving manifest {active_id} is not registered"),
            ));
        };
        Ok(Some(manifest.serving_view()?))
    }

    pub fn record_receipt(&mut self, receipt: QueryReceipt) -> GraphStoreResult<()> {
        if receipt.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_query_receipt",
                "query receipt id is required",
            ));
        }
        if self.receipts.contains_key(&receipt.id) {
            return Err(GraphStoreError::new(
                "query_receipt_exists",
                format!("query receipt {} is already recorded", receipt.id),
            ));
        }
        self.receipts.insert(receipt.id.clone(), receipt);
        Ok(())
    }

    pub fn get_receipt(&self, id: &str) -> Option<&QueryReceipt> {
        self.receipts.get(id)
    }

    pub fn receipts_for_signature(&self, query_signature: &str) -> Vec<&QueryReceipt> {
        self.receipts
            .values()
            .filter(|receipt| receipt.query_signature == query_signature)
            .collect()
    }

    pub fn register_proposal(&mut self, proposal: IndexProposal) -> GraphStoreResult<()> {
        if proposal.id.trim().is_empty() {
            return Err(GraphStoreError::new(
                "invalid_index_proposal",
                "index proposal id is required",
            ));
        }
        if self.proposals.contains_key(&proposal.id) {
            return Err(GraphStoreError::new(
                "index_proposal_exists",
                format!("index proposal {} is already registered", proposal.id),
            ));
        }
        self.proposals.insert(proposal.id.clone(), proposal);
        Ok(())
    }

    pub fn get_proposal(&self, id: &str) -> Option<&IndexProposal> {
        self.proposals.get(id)
    }

    pub fn list_proposals(&self) -> Vec<&IndexProposal> {
        self.proposals.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexBackend, IndexCreatedBy, IndexKind, IndexScope};

    fn generation_manifest(id: &str, generation: &str, status: IndexBuildStatus) -> IndexManifest {
        let mut manifest = IndexManifest::new(
            id,
            id,
            IndexKind::FullText,
            IndexBackend::Tantivy,
            IndexScope::Tenant,
            "Page",
            IndexCreatedBy::System,
        )
        .unwrap()
        .with_logical_id("web:page:fulltext")
        .unwrap();
        manifest.generation_id = Some(generation.to_string());
        manifest.graph_watermark = Some(format!("graph:{generation}"));
        manifest.graph_version = generation.parse().unwrap();
        manifest.build_status = status;
        manifest.refresh_hashes().unwrap();
        manifest
    }

    #[test]
    fn activation_leaves_exactly_one_active_manifest_per_logical_index() {
        let mut registry = IndexRegistry::new();
        registry
            .register_manifest(generation_manifest("idx@1", "1", IndexBuildStatus::Active))
            .unwrap();
        let shadow = generation_manifest("idx@2", "2", IndexBuildStatus::Shadow);
        registry.register_shadow_generation(shadow.clone()).unwrap();
        registry
            .attach_eval_receipt(
                &shadow.id,
                IndexEvalReceipt {
                    receipt_id: "eval:2".into(),
                    generation_id: "2".into(),
                    manifest_id: shadow.id.clone(),
                    graph_watermark: "graph:2".into(),
                    passed: true,
                    recall_at_k: Some(1.0),
                    ndcg_at_10: Some(1.0),
                    notes: None,
                },
            )
            .unwrap();
        registry.activate_shadow_generation(&shadow.id).unwrap();

        let active = registry
            .list_manifests()
            .into_iter()
            .filter(|manifest| manifest.build_status == IndexBuildStatus::Active)
            .collect::<Vec<_>>();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "idx@2");
        assert_eq!(
            registry.get_manifest("idx@1").unwrap().build_status,
            IndexBuildStatus::Shadow
        );
    }

    #[test]
    fn shadow_activation_requires_a_passed_eval_without_partial_mutation() {
        let shadow = generation_manifest("idx@2", "2", IndexBuildStatus::Shadow);
        let mut registry = IndexRegistry::new();
        registry.register_shadow_generation(shadow.clone()).unwrap();

        let before_missing = registry.clone();
        let missing = registry.activate_shadow_generation(&shadow.id).unwrap_err();
        assert_eq!(missing.code, "index_shadow_activation_requires_eval");
        assert_eq!(registry, before_missing);

        registry
            .attach_eval_receipt(
                &shadow.id,
                IndexEvalReceipt {
                    receipt_id: "eval:2:failed".into(),
                    generation_id: "2".into(),
                    manifest_id: shadow.id.clone(),
                    graph_watermark: "graph:2".into(),
                    passed: false,
                    recall_at_k: Some(0.0),
                    ndcg_at_10: Some(0.0),
                    notes: Some("fixture gate failure".into()),
                },
            )
            .unwrap();
        let before_failed = registry.clone();
        let failed = registry.activate_shadow_generation(&shadow.id).unwrap_err();
        assert_eq!(failed.code, "index_eval_receipt_failed");
        assert_eq!(registry, before_failed);
    }

    #[test]
    fn rollback_restores_previous_generation_and_serving_receipt() {
        let logical_id = "web:page:fulltext";
        let mut registry = IndexRegistry::new();
        registry
            .register_manifest(generation_manifest("idx@1", "1", IndexBuildStatus::Active))
            .unwrap();
        let shadow = generation_manifest("idx@2", "2", IndexBuildStatus::Shadow);
        registry.register_shadow_generation(shadow.clone()).unwrap();
        registry
            .attach_eval_receipt(
                &shadow.id,
                IndexEvalReceipt {
                    receipt_id: "eval:2".into(),
                    generation_id: "2".into(),
                    manifest_id: shadow.id.clone(),
                    graph_watermark: "graph:2".into(),
                    passed: true,
                    recall_at_k: Some(1.0),
                    ndcg_at_10: Some(1.0),
                    notes: None,
                },
            )
            .unwrap();
        registry.activate_shadow_generation(&shadow.id).unwrap();

        let rolled_back = registry.rollback_generation(logical_id).unwrap();
        assert_eq!(rolled_back.manifest_id, "idx@1");
        assert_eq!(rolled_back.generation_id, "1");
        assert_eq!(rolled_back.graph_watermark, "graph:1");
        assert_eq!(
            registry.get_manifest("idx@1").unwrap().build_status,
            IndexBuildStatus::Active
        );
        assert_eq!(
            registry.get_manifest("idx@2").unwrap().build_status,
            IndexBuildStatus::Shadow
        );
    }
}
