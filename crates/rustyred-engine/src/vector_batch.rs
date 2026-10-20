use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::graph::capability::GraphStore;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::query::NodeQuery;
use crate::graph::record::NodeRecord;
use crate::graph_store::RedCoreGraphStore;
use crate::query_receipt::ReceiptScope;
use crate::vector_eval::{
    filter_vector_candidates, VectorFilterPolicy, VectorIndexDefinition, VectorSearchCandidate,
    VectorSearchReceipt, VectorSearchWork,
};

/// A detached point remains useful as an ingestion DTO, but batch search no
/// longer accepts detached vectors or builds an index from them. Publish the
/// point through a designated graph property, then query that persistent index.
#[derive(Clone, Debug)]
pub struct ScopedVectorPoint {
    pub candidate: VectorSearchCandidate,
    pub vector: Vec<f32>,
}

impl ScopedVectorPoint {
    pub fn new(candidate: VectorSearchCandidate, vector: Vec<f32>) -> Self {
        Self { candidate, vector }
    }
}

#[derive(Clone, Debug)]
pub struct ScopedVectorQuery {
    pub object_id: String,
    pub vector: Vec<f32>,
}

impl ScopedVectorQuery {
    pub fn new(object_id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            object_id: object_id.into(),
            vector,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScopedVectorBatchReceipt {
    pub matches: BTreeMap<String, Vec<VectorSearchCandidate>>,
    pub work: VectorSearchWork,
    pub admitted_points: usize,
    pub rejected_points: usize,
}

/// Concrete graph stores expose their already-designated, persistent vector
/// index through this narrow seam. Implementations must not construct an index
/// from caller-provided vectors while answering a query.
pub trait PersistentVectorBatchStore: GraphStore {
    fn search_persistent_vector_index(
        &self,
        definition: &VectorIndexDefinition,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt>;
}

impl PersistentVectorBatchStore for InMemoryGraphStore {
    fn search_persistent_vector_index(
        &self,
        definition: &VectorIndexDefinition,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        self.vector_search_with_receipt(
            Some(&definition.target_label),
            &definition.property,
            query,
            k,
        )
    }
}

impl PersistentVectorBatchStore for RedCoreGraphStore {
    fn search_persistent_vector_index(
        &self,
        definition: &VectorIndexDefinition,
        query: &[f32],
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        self.vector_search_with_receipt(
            Some(&definition.target_label),
            &definition.property,
            query,
            k,
        )
    }
}

/// Search one persistent designated index for many scoped queries. Scope and
/// policy filtering is applied to graph metadata; dense point bytes stay in the
/// payload store and the persistent index for the entire batch.
pub fn search_scoped_vector_batch<S: PersistentVectorBatchStore>(
    store: &S,
    definition: &VectorIndexDefinition,
    policy: &VectorFilterPolicy,
    queries: &[ScopedVectorQuery],
    k: usize,
) -> GraphStoreResult<ScopedVectorBatchReceipt> {
    if let Some(field) = definition
        .scope_fields
        .iter()
        .find(|field| !policy.scope.contains_key(*field))
    {
        return Err(GraphStoreError::new(
            "incomplete_vector_scope_policy",
            format!("vector scope policy is missing required field `{field}`"),
        ));
    }

    let mut query_ids = BTreeSet::new();
    for query in queries {
        if !query_ids.insert(query.object_id.as_str()) {
            return Err(GraphStoreError::new(
                "duplicate_vector_query_id",
                format!(
                    "scoped vector batch contains duplicate query id `{}`",
                    query.object_id
                ),
            ));
        }
        if query.vector.len() != definition.dimension {
            return Err(GraphStoreError::new(
                "dimension_mismatch",
                format!(
                    "expected {} dimensions, got {}",
                    definition.dimension,
                    query.vector.len()
                ),
            ));
        }
    }

    let candidates = store
        .query_nodes(NodeQuery::label(&definition.target_label).with_limit(usize::MAX))
        .into_iter()
        .filter(|node| node.properties.get(&definition.property).is_some())
        .map(|node| candidate_from_node(&node, definition, 0.0))
        .collect::<GraphStoreResult<Vec<_>>>()?;
    let admitted_points = candidates
        .iter()
        .filter(|candidate| policy.accepts(candidate))
        .count();
    let rejected_points = candidates.len().saturating_sub(admitted_points);

    if queries.is_empty() {
        return Ok(ScopedVectorBatchReceipt {
            matches: BTreeMap::new(),
            work: VectorSearchWork::default(),
            admitted_points,
            rejected_points,
        });
    }

    // Fetch every live point from the designated index before applying scope.
    // This preserves the old pre-index scope semantics without rebuilding a
    // tenant-local index or copying stored vectors into a transient structure.
    let fetch_limit = candidates.len();
    let mut matches = BTreeMap::new();
    let mut work = VectorSearchWork::default();
    let mut backend_initialized = false;
    for query in queries {
        let receipt =
            store.search_persistent_vector_index(definition, &query.vector, fetch_limit)?;
        if !backend_initialized {
            work.backend = receipt.work.backend;
            backend_initialized = true;
        } else if work.backend != receipt.work.backend {
            work.backend = Default::default();
        }
        work.indexed_points = work.indexed_points.max(receipt.work.indexed_points);
        work.distance_evaluations = work
            .distance_evaluations
            .saturating_add(receipt.work.distance_evaluations);
        work.accelerated_candidates = work
            .accelerated_candidates
            .saturating_add(receipt.work.accelerated_candidates);
        work.index_rebuilds = work.index_rebuilds.max(receipt.work.index_rebuilds);
        merge_fallback_reason(&mut work.fallback_reason, receipt.work.fallback_reason);

        let candidates = receipt
            .hits
            .into_iter()
            .filter(|(object_id, _)| object_id != &query.object_id)
            .filter_map(|(object_id, distance)| {
                store
                    .get_node(&object_id)
                    .map(|node| candidate_from_node(node, definition, distance))
            })
            .collect::<GraphStoreResult<Vec<_>>>()?;
        let candidates = filter_vector_candidates(&candidates, policy, k);
        work.retained_candidates = work.retained_candidates.saturating_add(candidates.len());
        matches.insert(query.object_id.clone(), candidates);
    }

    Ok(ScopedVectorBatchReceipt {
        matches,
        work,
        admitted_points,
        rejected_points,
    })
}

fn candidate_from_node(
    node: &NodeRecord,
    definition: &VectorIndexDefinition,
    distance: f32,
) -> GraphStoreResult<VectorSearchCandidate> {
    let mut scope = ReceiptScope::new();
    for field in &definition.scope_fields {
        let Some(value) = node.properties.get(field) else {
            continue;
        };
        let value = scalar_string(value).ok_or_else(|| {
            GraphStoreError::new(
                "vector_scope_field_invalid",
                format!("node {} scope field `{field}` is not scalar", node.id),
            )
        })?;
        scope.insert(field.clone(), value);
    }
    let mut candidate = VectorSearchCandidate::new(&node.id, distance, scope)
        .with_labels(node.labels.iter().cloned());
    candidate.repo = string_property(node, "repo_id").or_else(|| string_property(node, "repo"));
    candidate.user = string_property(node, "user_id").or_else(|| string_property(node, "user"));
    candidate.trust_status = string_property(node, "trust_status");
    candidate.freshness_status = string_property(node, "freshness_status");
    candidate.policy_allowed = node
        .properties
        .get("policy_allowed")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    for (key, value) in node.properties.as_object().into_iter().flatten() {
        if let Some(value) = scalar_string(value) {
            candidate.metadata.insert(key.clone(), value);
        }
    }
    Ok(candidate)
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn string_property(node: &NodeRecord, property: &str) -> Option<String> {
    node.properties
        .get(property)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn merge_fallback_reason(target: &mut Option<String>, incoming: Option<String>) {
    let Some(incoming) = incoming.filter(|reason| !reason.is_empty()) else {
        return;
    };
    let mut reasons = target
        .take()
        .into_iter()
        .chain([incoming])
        .collect::<Vec<_>>();
    reasons.sort();
    reasons.dedup();
    *target = Some(reasons.join(";"));
}
