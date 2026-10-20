use std::collections::{BTreeMap, HashMap};

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use turbovec::IdMapIndex;

use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::vector_eval::{VectorSearchBackend, VectorSearchReceipt, VectorSearchWork};

const VECTOR_INDEX_BIT_WIDTH: usize = 4;
pub const VECTOR_ACCELERATION_MIN_POINTS: usize = 64;

#[derive(Clone, Debug)]
pub struct VectorPoint {
    pub(crate) packed_le: Vec<u8>,
    pub(crate) norm: f32,
}

impl VectorPoint {
    pub fn new(raw: &[f32]) -> Self {
        let mut packed_le = Vec::with_capacity(raw.len().saturating_mul(4));
        for value in raw {
            packed_le.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            packed_le,
            norm: vector_norm(raw.iter().copied()),
        }
    }

    fn from_packed(packed_le: Vec<u8>, dimension: usize) -> GraphStoreResult<Self> {
        let expected_len = dimension.checked_mul(4).ok_or_else(|| {
            GraphStoreError::new(
                "vector_property_dimension_overflow",
                "vector dimension exceeds the addressable packed-byte length",
            )
        })?;
        if packed_le.len() != expected_len {
            return Err(GraphStoreError::new(
                "vector_payload_length_mismatch",
                format!(
                    "vector index expects {dimension} dimensions ({expected_len} bytes), payload has {} bytes",
                    packed_le.len()
                ),
            ));
        }
        let norm = vector_norm(packed_values(&packed_le));
        Ok(Self { packed_le, norm })
    }

    fn extend_normalized(&self, output: &mut Vec<f32>) {
        let scale = normalization_scale(self.norm);
        output.extend(packed_values(&self.packed_le).map(|value| value / scale));
    }

    fn cosine_distance_to(&self, query: &[f32], query_norm: f32) -> f32 {
        // Preserve the historical exact-search operation order. Dividing each
        // lane before multiplication is mathematically equivalent to dividing
        // the raw dot product once, but it is not bit-equivalent in f32 and can
        // reorder tied neighbors.
        let query_scale = normalization_scale(query_norm);
        let point_scale = normalization_scale(self.norm);
        let dot = query
            .iter()
            .copied()
            .zip(packed_values(&self.packed_le))
            .map(|(left, right)| (left / query_scale) * (right / point_scale))
            .sum::<f32>();
        1.0 - dot.clamp(-1.0, 1.0)
    }
}

fn packed_values(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(4).map(|chunk| {
        f32::from_le_bytes(
            chunk
                .try_into()
                .expect("validated packed vector lanes are four bytes"),
        )
    })
}

fn vector_norm(values: impl Iterator<Item = f32>) -> f32 {
    values.map(|value| value * value).sum::<f32>().sqrt()
}

fn normalization_scale(norm: f32) -> f32 {
    if norm < 1e-10 {
        1.0
    } else {
        norm
    }
}

#[derive(Clone, Debug)]
struct RankedVectorHit {
    node_id: String,
    distance: f32,
}

impl PartialEq for RankedVectorHit {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.node_id == other.node_id
    }
}

impl Eq for RankedVectorHit {}

impl PartialOrd for RankedVectorHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedVectorHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

pub struct VectorIndex {
    pub(crate) points: Vec<VectorPoint>,
    pub(crate) node_ids: Vec<String>,
    pub(crate) slot_by_node_id: HashMap<String, usize>,
    dense_id_by_slot: Vec<u64>,
    slot_by_dense_id: HashMap<u64, usize>,
    turbovec: Option<IdMapIndex>,
    next_dense_id: u64,
    calibration_seed: u64,
    full_builds: usize,
    pub dimension: usize,
}

/// Read-only shape of one designated live vector index.
///
/// `point_count` is the number of stored, non-tombstoned points that the
/// index's exact path evaluates for a finite query of `dimension`. It includes
/// points inserted directly through
/// [`crate::graph::in_memory::InMemoryGraphStore::index_vector`] even
/// when the source node has no matching vector property.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VectorIndexShape {
    pub dimension: usize,
    pub point_count: usize,
}

impl Clone for VectorIndex {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            node_ids: self.node_ids.clone(),
            slot_by_node_id: self.slot_by_node_id.clone(),
            dense_id_by_slot: self.dense_id_by_slot.clone(),
            slot_by_dense_id: self.slot_by_dense_id.clone(),
            turbovec: self.turbovec.as_ref().map(|index| {
                IdMapIndex::from_bytes(&index.to_bytes())
                    .expect("resident TurboVec snapshot is well-formed")
            }),
            next_dense_id: self.next_dense_id,
            calibration_seed: self.calibration_seed,
            full_builds: self.full_builds,
            dimension: self.dimension,
        }
    }
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("dimension", &self.dimension)
            .field("count", &self.points.len())
            .field("full_builds", &self.full_builds)
            .finish()
    }
}

impl VectorIndex {
    pub(crate) fn new(dimension: usize, calibration_seed: u64) -> Self {
        Self {
            points: Vec::new(),
            node_ids: Vec::new(),
            slot_by_node_id: HashMap::new(),
            dense_id_by_slot: Vec::new(),
            slot_by_dense_id: HashMap::new(),
            turbovec: None,
            next_dense_id: 1,
            calibration_seed,
            full_builds: 0,
            dimension,
        }
    }

    pub(crate) fn insert(&mut self, node_id: &str, vector: &[f32]) {
        self.upsert_point(node_id, VectorPoint::new(vector));
    }

    pub(crate) fn insert_packed(
        &mut self,
        node_id: &str,
        packed_le: Vec<u8>,
    ) -> GraphStoreResult<()> {
        let point = VectorPoint::from_packed(packed_le, self.dimension)?;
        self.upsert_point(node_id, point);
        Ok(())
    }

    fn upsert_point(&mut self, node_id: &str, point: VectorPoint) {
        if let Some(&slot) = self.slot_by_node_id.get(node_id) {
            self.points[slot] = point;
            self.sync_resident_slot(slot);
            return;
        }
        let slot = self.node_ids.len();
        self.points.push(point);
        self.node_ids.push(node_id.to_string());
        self.slot_by_node_id.insert(node_id.to_string(), slot);
        let dense_id = self.next_dense_id;
        self.next_dense_id = self.next_dense_id.saturating_add(1);
        self.dense_id_by_slot.push(dense_id);
        self.slot_by_dense_id.insert(dense_id, slot);
        if self.turbovec.is_some() {
            let padded = self.padded_normalized(slot);
            if let Some(index) = self.turbovec.as_mut() {
                if index.add_with_ids(&padded, &[dense_id]).is_err() {
                    self.turbovec = None;
                }
            }
        } else if self.points.len() >= VECTOR_ACCELERATION_MIN_POINTS {
            self.build_resident_index();
        }
    }

    pub(crate) fn remove(&mut self, node_id: &str) {
        let Some(slot) = self.slot_by_node_id.remove(node_id) else {
            return;
        };
        let dense_id = self.dense_id_by_slot[slot];
        if let Some(index) = self.turbovec.as_mut() {
            index.remove(dense_id);
        }
        self.slot_by_dense_id.remove(&dense_id);
        self.points.swap_remove(slot);
        self.node_ids.swap_remove(slot);
        self.dense_id_by_slot.swap_remove(slot);
        if slot < self.node_ids.len() {
            self.slot_by_node_id
                .insert(self.node_ids[slot].clone(), slot);
            let moved_id = self.dense_id_by_slot[slot];
            self.slot_by_dense_id.insert(moved_id, slot);
        }
    }

    fn sync_resident_slot(&mut self, slot: usize) {
        if self.turbovec.is_none() {
            return;
        }
        let padded = self.padded_normalized(slot);
        let Some(index) = self.turbovec.as_mut() else {
            return;
        };
        let old_id = self.dense_id_by_slot[slot];
        index.remove(old_id);
        self.slot_by_dense_id.remove(&old_id);
        let new_id = self.next_dense_id;
        self.next_dense_id = self.next_dense_id.saturating_add(1);
        if index.add_with_ids(&padded, &[new_id]).is_err() {
            self.turbovec = None;
            return;
        }
        self.dense_id_by_slot[slot] = new_id;
        self.slot_by_dense_id.insert(new_id, slot);
    }

    fn padded_normalized(&self, slot: usize) -> Vec<f32> {
        let turbovec_dimension = self.dimension.next_multiple_of(8);
        let mut vector = Vec::with_capacity(turbovec_dimension);
        self.points[slot].extend_normalized(&mut vector);
        vector.resize(turbovec_dimension, 0.0);
        vector
    }

    fn build_resident_index(&mut self) {
        let turbovec_dimension = self.dimension.next_multiple_of(8);
        let mut index = match IdMapIndex::new(turbovec_dimension, VECTOR_INDEX_BIT_WIDTH) {
            Ok(index) => index,
            Err(_) => return,
        };
        // TQ+ calibration needs a representative population. Repeating a
        // small or degenerate corpus up to MIN_CALIBRATION_ROWS invents sample
        // diversity and can make calibration reject an otherwise searchable
        // index. In that case plain TurboQuant is the honest acceleration
        // mode; a later full rebuild can calibrate once the corpus is large.
        if self.points.len() >= turbovec::MIN_CALIBRATION_ROWS {
            let sample = self.calibration_sample(turbovec::RECOMMENDED_CALIBRATION_ROWS);
            if index.calibrate_2d(&sample, turbovec_dimension).is_err() {
                index = match IdMapIndex::new(turbovec_dimension, VECTOR_INDEX_BIT_WIDTH) {
                    Ok(index) => index,
                    Err(_) => return,
                };
            }
        }
        let mut vectors = Vec::with_capacity(self.points.len() * turbovec_dimension);
        for slot in 0..self.points.len() {
            vectors.extend(self.padded_normalized(slot));
        }
        let ids = self.dense_id_by_slot.clone();
        if index.add_with_ids(&vectors, &ids).is_err() {
            return;
        }
        self.turbovec = Some(index);
        self.full_builds = self.full_builds.saturating_add(1);
    }

    fn calibration_sample(&self, take: usize) -> Vec<f32> {
        let turbovec_dimension = self.dimension.next_multiple_of(8);
        let count = self.points.len().min(take);
        let mut rng =
            ChaCha8Rng::from_seed(*blake3::hash(&self.calibration_seed.to_le_bytes()).as_bytes());
        let mut order = (0..self.points.len()).collect::<Vec<_>>();
        let n = order.len();
        for i in 0..n.min(count) {
            let span = n - i;
            let j = i + (rng.next_u64() as usize % span.max(1));
            order.swap(i, j.min(n.saturating_sub(1)));
        }
        let mut sample = Vec::with_capacity(count * turbovec_dimension);
        for slot in order.iter().take(count) {
            sample.extend(self.padded_normalized(*slot));
        }
        sample
    }

    pub(crate) fn search_with_receipt(&self, query: &[f32], k: usize) -> VectorSearchReceipt {
        if k == 0 || query.iter().any(|value| !value.is_finite()) {
            return VectorSearchReceipt::default();
        }
        let Some(index) = &self.turbovec else {
            let mut receipt = self.exact_search_with_receipt(query, k);
            receipt.work.index_rebuilds = self.full_builds;
            receipt.work.fallback_reason = Some("below_acceleration_minimum".to_string());
            return receipt;
        };
        let mut turbovec_query = Vec::with_capacity(self.dimension.next_multiple_of(8));
        let query_norm = vector_norm(query.iter().copied());
        let query_scale = normalization_scale(query_norm);
        turbovec_query.extend(query.iter().map(|value| value / query_scale));
        turbovec_query.resize(self.dimension.next_multiple_of(8), 0.0);
        let recall_k = k.saturating_mul(4).max(k).min(self.node_ids.len());
        let (_scores, ids) = index.search(&turbovec_query, recall_k);
        let accelerated_candidates = ids.len();
        let mut hits = ids
            .into_iter()
            .filter_map(|id| {
                let slot = *self.slot_by_dense_id.get(&id)?;
                let node_id = self.node_ids.get(slot)?;
                let point = self.points.get(slot)?;
                Some(RankedVectorHit {
                    node_id: node_id.clone(),
                    distance: point.cosine_distance_to(query, query_norm),
                })
            })
            .collect::<Vec<_>>();
        hits.sort();
        hits.truncate(k);
        VectorSearchReceipt {
            work: VectorSearchWork {
                backend: VectorSearchBackend::Turbovec,
                indexed_points: self.points.len(),
                retained_candidates: hits.len(),
                accelerated_candidates,
                index_rebuilds: self.full_builds,
                ..VectorSearchWork::default()
            },
            hits: hits
                .into_iter()
                .map(|hit| (hit.node_id, hit.distance))
                .collect(),
        }
    }

    pub(crate) fn exact_search_with_receipt(&self, query: &[f32], k: usize) -> VectorSearchReceipt {
        use std::collections::BinaryHeap;

        if k == 0 || query.iter().any(|value| !value.is_finite()) {
            return VectorSearchReceipt::default();
        }
        let query_norm = vector_norm(query.iter().copied());
        let mut heap = BinaryHeap::with_capacity(k.min(self.points.len()));
        for (node_id, point) in self.node_ids.iter().zip(&self.points) {
            let hit = RankedVectorHit {
                node_id: node_id.clone(),
                distance: point.cosine_distance_to(query, query_norm),
            };
            if heap.len() < k {
                heap.push(hit);
            } else if heap.peek().is_some_and(|worst| hit < *worst) {
                heap.pop();
                heap.push(hit);
            }
        }
        let mut hits = heap.into_vec();
        hits.sort();
        VectorSearchReceipt {
            work: VectorSearchWork {
                backend: VectorSearchBackend::ExactCosine,
                indexed_points: self.points.len(),
                distance_evaluations: self.points.len(),
                retained_candidates: hits.len(),
                ..VectorSearchWork::default()
            },
            hits: hits
                .into_iter()
                .map(|hit| (hit.node_id, hit.distance))
                .collect(),
        }
    }

    /// Exact kNN within an already-admitted blocking posting. Dense bytes and
    /// cosine semantics remain in this maintained index; no detached ANN or
    /// whole-index candidate enumeration is performed.
    pub(crate) fn search_candidates_with_receipt(
        &self,
        query: &[f32],
        candidates: &std::collections::BTreeSet<String>,
        k: usize,
    ) -> GraphStoreResult<VectorSearchReceipt> {
        if query.len() != self.dimension || query.iter().any(|v| !v.is_finite()) || k == 0 {
            return Err(GraphStoreError::new(
                "invalid_vector_candidate_query",
                "finite matching dimensions and positive k required",
            ));
        }
        let query_norm = vector_norm(query.iter().copied());
        let mut hits = Vec::new();
        for id in candidates {
            let Some(slot) = self.slot_by_node_id.get(id) else {
                continue;
            };
            let point = &self.points[*slot];
            hits.push(RankedVectorHit {
                node_id: id.clone(),
                distance: point.cosine_distance_to(query, query_norm),
            });
        }
        let evaluated = hits.len();
        hits.sort();
        hits.truncate(k);
        Ok(VectorSearchReceipt {
            work: VectorSearchWork {
                backend: VectorSearchBackend::ExactCosine,
                indexed_points: self.points.len(),
                distance_evaluations: evaluated,
                retained_candidates: hits.len(),
                ..Default::default()
            },
            hits: hits.into_iter().map(|h| (h.node_id, h.distance)).collect(),
        })
    }

    pub(crate) fn resident_len(&self) -> Option<usize> {
        self.turbovec.as_ref().map(IdMapIndex::len)
    }

    pub(crate) fn full_builds(&self) -> usize {
        self.full_builds
    }

    pub(crate) fn index_bytes(&self) -> Option<Vec<u8>> {
        self.turbovec.as_ref().map(IdMapIndex::to_bytes)
    }
}

pub(crate) fn serialize_vector_fallback_reasons(mut reasons: Vec<String>) -> Option<String> {
    reasons.sort();
    reasons.dedup();
    if reasons.is_empty() {
        return None;
    }
    if reasons.len() == 1 {
        return reasons.pop();
    }
    Some(
        reasons
            .into_iter()
            .map(|reason| reason.replace('\\', "\\\\").replace(';', "\\;"))
            .collect::<Vec<_>>()
            .join(";"),
    )
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorDesignation {
    pub label: String,
    pub property: String,
    pub dimension: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HybridScoringConfig {
    pub alpha: f32,
    #[serde(default = "default_confidence_weighted_graph_distance")]
    pub confidence_weighted_graph_distance: bool,
    #[serde(default = "default_hybrid_edge_type_weights")]
    pub edge_type_weights: BTreeMap<String, f32>,
}

impl Default for HybridScoringConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            confidence_weighted_graph_distance: true,
            edge_type_weights: default_hybrid_edge_type_weights(),
        }
    }
}

impl HybridScoringConfig {
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    pub(crate) fn edge_type_weight(&self, edge_type: &str) -> f32 {
        self.edge_type_weights
            .get(edge_type)
            .or_else(|| self.edge_type_weights.get(&edge_type.to_ascii_lowercase()))
            .copied()
            .unwrap_or(1.0)
    }
}

fn default_confidence_weighted_graph_distance() -> bool {
    true
}

pub fn default_hybrid_edge_type_weights() -> BTreeMap<String, f32> {
    BTreeMap::from([
        ("contradicts".to_string(), -1.0),
        ("CONTRADICTS".to_string(), -1.0),
        ("tension".to_string(), -0.5),
        ("TENSION".to_string(), -0.5),
    ])
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use crate::vector_eval::VectorSearchBackend;

    fn varied(dim: usize, seed: usize) -> Vec<f32> {
        let mut vector = Vec::with_capacity(dim);
        for col in 0..dim {
            let raw = ((seed * 31 + col * 17) % 97) as f32 / 97.0 - 0.5;
            vector.push(raw);
        }
        let norm = vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for value in &mut vector {
            *value /= norm;
        }
        vector
    }

    #[test]
    fn insert_then_remove_keeps_one_full_build() {
        let mut index = VectorIndex::new(128, 9);
        for i in 0..200 {
            index.insert(&format!("n{i}"), &varied(128, i));
        }
        for i in 150..200 {
            index.remove(&format!("n{i}"));
        }
        assert_eq!(index.resident_len(), Some(150));
        assert_eq!(index.full_builds(), 1);
    }

    #[test]
    fn updates_do_not_rebuild() {
        let mut index = VectorIndex::new(128, 11);
        let beta = varied(128, 10_000);
        for i in 0..64 {
            index.insert(&format!("n{i}"), &varied(128, i));
        }
        for _ in 0..100 {
            index.insert("n0", &beta);
        }
        assert_eq!(index.full_builds(), 1);
        let receipt = index.search_with_receipt(&beta, 1);
        assert_eq!(receipt.hits[0].0, "n0");
        assert_eq!(receipt.work.backend, VectorSearchBackend::Turbovec);
    }

    #[test]
    fn identical_seeds_write_identical_bytes() {
        let mut left = VectorIndex::new(128, 13);
        let mut right = VectorIndex::new(128, 13);
        for i in 0..64 {
            let vector = varied(128, i);
            left.insert(&format!("n{i}"), &vector);
            right.insert(&format!("n{i}"), &vector);
        }
        assert_eq!(left.index_bytes(), right.index_bytes());
    }

    #[test]
    fn threshold_is_inclusive_at_sixty_four() {
        let mut under = VectorIndex::new(128, 15);
        for i in 0..63 {
            under.insert(&format!("n{i}"), &varied(128, i));
        }
        let probe = varied(128, 0);
        assert_eq!(
            under.search_with_receipt(&probe, 1).work.backend,
            VectorSearchBackend::ExactCosine
        );
        under.insert("n63", &varied(128, 63));
        assert_eq!(
            under.search_with_receipt(&probe, 1).work.backend,
            VectorSearchBackend::Turbovec
        );
    }

    #[test]
    fn degenerate_small_corpus_uses_honest_uncalibrated_turbovec() {
        let mut index = VectorIndex::new(128, 17);
        let repeated = varied(128, 1);
        for i in 0..64 {
            index.insert(&format!("n{i}"), &repeated);
        }

        assert_eq!(index.resident_len(), Some(64));
        let receipt = index.search_with_receipt(&repeated, 1);
        assert_eq!(receipt.work.backend, VectorSearchBackend::Turbovec);
        assert_eq!(receipt.hits.len(), 1);
    }
}
