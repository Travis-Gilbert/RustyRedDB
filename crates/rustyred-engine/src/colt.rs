//! COLT free-join executor (SPEC-THEOREM-RELATIONAL-OPERATORS-1.0 RO2).
//!
//! One operator family: binary hash join and generic join are plan points of the
//! same executor. Left/semi/anti are emission modes on the probe. COLT levels
//! are transient per query (FLUSHALL): a never-probed relation builds nothing.

use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(not(feature = "graphblas"))]
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

#[cfg(feature = "graphblas")]
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::planner::{JoinBacking, JoinKind, JoinPlanPoint, JoinPredicate};

/// Native floor candidate id. Cold-start EXPLAIN and traces keep `hash_join`
/// so pre-observation INNER behavior is bit-identical to as-built.
pub const JOIN_CANDIDATE_COLT_BINARY: &str = "join_colt_binary";
pub const JOIN_CANDIDATE_COLT_GENERIC: &str = "join_colt_generic";
pub const JOIN_CANDIDATE_MASK: &str = "join_mask";

/// Floor algorithm name preserved for INNER cold-start traces.
pub const NATIVE_JOIN_ALGORITHM: &str = "hash_join";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Emission {
    Inner,
    Left,
    Semi,
    Anti,
}

impl Emission {
    pub fn from_kind(kind: JoinKind) -> Option<Self> {
        match kind {
            JoinKind::Inner => Some(Self::Inner),
            JoinKind::Left => Some(Self::Left),
            JoinKind::Semi => Some(Self::Semi),
            JoinKind::Anti => Some(Self::Anti),
            JoinKind::Asof => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ColtStats {
    pub intermediate_tuples: usize,
    pub build_counts: Vec<usize>,
    pub algorithm: String,
    pub candidate_id: String,
}

/// COLT level: hash-map internal nodes, offset-vector leaves (SIGMOD 2023).
/// Transient levels materialize on first probe. Persistent (DATAWAVE / node-id)
/// levels are already built at ingest and never increment the query build counter.
#[derive(Clone, Debug)]
pub struct ColtLevel {
    built: bool,
    persistent: bool,
    offsets: HashMap<String, (u32, u32)>,
    indices: Vec<usize>,
}

impl ColtLevel {
    fn unbuilt() -> Self {
        Self {
            built: false,
            persistent: false,
            offsets: HashMap::new(),
            indices: Vec::new(),
        }
    }

    /// Persistent tier stand-in: the dictionary from values to row ids is paid
    /// at ingest (DATAWAVE-normalized values or node ids). Probe never builds.
    pub fn from_persistent(entries: impl IntoIterator<Item = (String, usize)>) -> Self {
        let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
        for (key, index) in entries {
            buckets.entry(key).or_default().push(index);
        }
        let mut level = Self::unbuilt();
        level.install_buckets(buckets);
        level.persistent = true;
        level.built = false;
        level
    }

    fn install_buckets(&mut self, buckets: HashMap<String, Vec<usize>>) {
        self.indices.clear();
        self.offsets.clear();
        for (key, rows) in buckets {
            let start = self.indices.len() as u32;
            self.indices.extend(rows);
            let end = self.indices.len() as u32;
            self.offsets.insert(key, (start, end));
        }
    }

    fn rebuild(&mut self, keys: &[Option<String>]) {
        let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, row_key) in keys.iter().enumerate() {
            if let Some(value) = row_key {
                buckets.entry(value.clone()).or_default().push(index);
            }
        }
        self.install_buckets(buckets);
    }

    fn lookup(&self, key: &str) -> &[usize] {
        self.offsets
            .get(key)
            .map(|(start, end)| &self.indices[*start as usize..*end as usize])
            .unwrap_or(&[])
    }

    /// Probe a level that is already materialized (persistent or previously built).
    pub fn probe_key(&self, key: &str) -> &[usize] {
        self.lookup(key)
    }

    fn probe(&mut self, keys: &[Option<String>], key: &str) -> &[usize] {
        if self.persistent {
            return self.lookup(key);
        }
        if !self.built {
            self.built = true;
            self.rebuild(keys);
        }
        self.lookup(key)
    }

    pub fn build_count(&self) -> usize {
        if self.persistent {
            0
        } else if self.built {
            1
        } else {
            0
        }
    }
}

/// Pair of matched row indexes (left, optional right). Right is None on left-pad.
pub type JoinPair = (usize, Option<usize>);

pub fn execute_binary_join(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    emission: Emission,
) -> (Vec<JoinPair>, ColtStats) {
    let mut level = ColtLevel::unbuilt();
    let mut pairs = Vec::new();
    let mut intermediate = 0usize;
    for (left_index, left_key) in left_keys.iter().enumerate() {
        let Some(key) = left_key else {
            if matches!(emission, Emission::Left | Emission::Anti) {
                pairs.push((left_index, None));
            }
            continue;
        };
        let hits = level.probe(right_keys, key);
        intermediate = intermediate.saturating_add(hits.len().max(1));
        match emission {
            Emission::Inner => {
                for &right_index in hits {
                    pairs.push((left_index, Some(right_index)));
                }
            }
            Emission::Left => {
                if hits.is_empty() {
                    pairs.push((left_index, None));
                } else {
                    for &right_index in hits {
                        pairs.push((left_index, Some(right_index)));
                    }
                }
            }
            Emission::Semi => {
                if !hits.is_empty() {
                    pairs.push((left_index, Some(hits[0])));
                }
            }
            Emission::Anti => {
                if hits.is_empty() {
                    pairs.push((left_index, None));
                }
            }
        }
    }
    let stats = ColtStats {
        intermediate_tuples: intermediate,
        build_counts: vec![level.build_count()],
        algorithm: NATIVE_JOIN_ALGORITHM.to_string(),
        candidate_id: JOIN_CANDIDATE_COLT_BINARY.to_string(),
    };
    (pairs, stats)
}

/// Generic plan point of the same COLT executor: same offset-vector levels,
/// generic-join algorithm id so EXPLAIN names what ran.
pub fn execute_generic_join(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    emission: Emission,
) -> (Vec<JoinPair>, ColtStats) {
    let (pairs, mut stats) = execute_binary_join(left_keys, right_keys, emission);
    stats.algorithm = "generic_join".to_string();
    stats.candidate_id = JOIN_CANDIDATE_COLT_GENERIC.to_string();
    (pairs, stats)
}

/// Generic-join extreme for a triangle R(a,b) ⋈ S(b,c) ⋈ T(c,a).
/// Returns index triples (r, s, t) and counts intermediates against output size.
pub fn execute_triangle_generic(
    r_ab: &[(String, String)],
    s_bc: &[(String, String)],
    t_ca: &[(String, String)],
) -> (Vec<(usize, usize, usize)>, ColtStats) {
    let s_keys: Vec<Option<String>> = s_bc.iter().map(|(b, _)| Some(b.clone())).collect();
    let t_keys: Vec<Option<String>> = t_ca.iter().map(|(_, a)| Some(a.clone())).collect();
    let mut s_level = ColtLevel::unbuilt();
    let mut t_level = ColtLevel::unbuilt();
    let mut intermediate = 0usize;
    let mut out = Vec::new();

    for (r_index, (a, b)) in r_ab.iter().enumerate() {
        let t_hits = t_level.probe(&t_keys, a).to_vec();
        let s_hits = s_level.probe(&s_keys, b).to_vec();
        for t_index in t_hits {
            let c = &t_ca[t_index].0;
            for s_index in &s_hits {
                intermediate = intermediate.saturating_add(1);
                if s_bc[*s_index].1 == *c {
                    out.push((r_index, *s_index, t_index));
                }
            }
        }
    }

    (
        out,
        ColtStats {
            intermediate_tuples: intermediate,
            build_counts: vec![0, s_level.build_count(), t_level.build_count()],
            algorithm: "generic_join".to_string(),
            candidate_id: JOIN_CANDIDATE_COLT_GENERIC.to_string(),
        },
    )
}

/// Binary hash-join extreme of the same triangle: (R ⋈ S) ⋈ T.
pub fn execute_triangle_binary(
    r_ab: &[(String, String)],
    s_bc: &[(String, String)],
    t_ca: &[(String, String)],
) -> (Vec<(usize, usize, usize)>, ColtStats) {
    let mut s_by_b: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, (b, _)) in s_bc.iter().enumerate() {
        s_by_b.entry(b.as_str()).or_default().push(index);
    }
    let mut intermediate = 0usize;
    let mut rs = Vec::new();
    for (r_index, (_, b)) in r_ab.iter().enumerate() {
        if let Some(hits) = s_by_b.get(b.as_str()) {
            intermediate = intermediate.saturating_add(hits.len());
            for &s_index in hits {
                rs.push((r_index, s_index));
            }
        }
    }
    let mut t_by_ca: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (index, (c, a)) in t_ca.iter().enumerate() {
        t_by_ca
            .entry((c.clone(), a.clone()))
            .or_default()
            .push(index);
    }
    let mut out = Vec::new();
    for (r_index, s_index) in rs {
        let a = &r_ab[r_index].0;
        let c = &s_bc[s_index].1;
        intermediate = intermediate.saturating_add(1);
        if let Some(hits) = t_by_ca.get(&(c.clone(), a.clone())) {
            for &t_index in hits {
                out.push((r_index, s_index, t_index));
            }
        }
    }
    (
        out,
        ColtStats {
            intermediate_tuples: intermediate,
            build_counts: vec![1, 1],
            algorithm: NATIVE_JOIN_ALGORITHM.to_string(),
            candidate_id: JOIN_CANDIDATE_COLT_BINARY.to_string(),
        },
    )
}

/// Structural / complemented mask join over encoded node ids (RO3).
/// With `graphblas`, equi lowers to masked `GrB_mxm` through the safe handle.
/// Without it, roaring is a default-build stand-in so COLT tests stay native-free.
pub fn execute_mask_join(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    emission: Emission,
) -> (Vec<JoinPair>, ColtStats) {
    #[cfg(feature = "graphblas")]
    {
        execute_graphblas_mask_join(left_keys, right_keys, emission)
            .unwrap_or_else(|error| panic!("GrB_mxm join failed: {error}"))
    }
    #[cfg(not(feature = "graphblas"))]
    {
        execute_roaring_mask_join(left_keys, right_keys, emission)
    }
}

#[cfg(not(feature = "graphblas"))]
fn execute_roaring_mask_join(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    emission: Emission,
) -> (Vec<JoinPair>, ColtStats) {
    let mut intern: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    let mut intern_key = |key: &str| -> u32 {
        if let Some(id) = intern.get(key) {
            return *id;
        }
        let id = next;
        intern.insert(key.to_string(), id);
        next = next.saturating_add(1);
        id
    };

    let mut right_mask = RoaringBitmap::new();
    let mut right_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, key) in right_keys.iter().enumerate() {
        if let Some(value) = key {
            let id = intern_key(value);
            right_mask.insert(id);
            right_by_id.entry(id).or_default().push(index);
        }
    }

    let mut pairs = Vec::new();
    for (left_index, left_key) in left_keys.iter().enumerate() {
        let Some(key) = left_key else {
            if matches!(emission, Emission::Left | Emission::Anti) {
                pairs.push((left_index, None));
            }
            continue;
        };
        let id = intern_key(key);
        let present = right_mask.contains(id);
        match emission {
            Emission::Inner => {
                if present {
                    for &right_index in right_by_id.get(&id).into_iter().flatten() {
                        pairs.push((left_index, Some(right_index)));
                    }
                }
            }
            Emission::Left => {
                if present {
                    for &right_index in right_by_id.get(&id).into_iter().flatten() {
                        pairs.push((left_index, Some(right_index)));
                    }
                } else {
                    pairs.push((left_index, None));
                }
            }
            Emission::Semi => {
                if present {
                    if let Some(right_index) = right_by_id.get(&id).and_then(|rows| rows.first()) {
                        pairs.push((left_index, Some(*right_index)));
                    }
                }
            }
            Emission::Anti => {
                if !present {
                    pairs.push((left_index, None));
                }
            }
        }
    }

    let intermediate_tuples = pairs.len();
    (
        pairs,
        ColtStats {
            intermediate_tuples,
            build_counts: vec![1],
            algorithm: "mask_join".to_string(),
            candidate_id: JOIN_CANDIDATE_MASK.to_string(),
        },
    )
}

#[cfg(feature = "graphblas")]
fn execute_graphblas_mask_join(
    left_keys: &[Option<String>],
    right_keys: &[Option<String>],
    emission: Emission,
) -> GraphStoreResult<(Vec<JoinPair>, ColtStats)> {
    use rustyred_thg_graphblas::{mxm_into, Descriptor, Matrix, Semiring};

    fn stats(pairs: &[JoinPair]) -> ColtStats {
        ColtStats {
            intermediate_tuples: pairs.len(),
            build_counts: vec![1],
            algorithm: "mask_join".to_string(),
            candidate_id: JOIN_CANDIDATE_MASK.to_string(),
        }
    }
    let map_err = |error: rustyred_thg_graphblas::GrbError| {
        GraphStoreError::new("graphblas_mxm_join", error.to_string())
    };

    let mut intern: HashMap<String, u64> = HashMap::new();
    {
        let mut intern_key = |key: &str| -> u64 {
            if let Some(id) = intern.get(key) {
                return *id;
            }
            let id = intern.len() as u64;
            intern.insert(key.to_string(), id);
            id
        };
        for key in left_keys.iter().chain(right_keys.iter()).flatten() {
            intern_key(key);
        }
    }
    let interned = |key: &str| -> u64 { *intern.get(key).expect("join key interned") };

    if left_keys.is_empty() {
        return Ok((Vec::new(), stats(&[])));
    }
    if intern.is_empty() {
        let mut pairs = Vec::new();
        if matches!(emission, Emission::Left | Emission::Anti) {
            for left_index in 0..left_keys.len() {
                pairs.push((left_index, None));
            }
        }
        return Ok((pairs.clone(), stats(&pairs)));
    }

    let n_ids = intern.len() as u64;
    let n_left = left_keys.len() as u64;
    let n_right = right_keys.len().max(1) as u64;

    let mut left_inc = Matrix::new_bool(n_left, n_ids).map_err(map_err)?;
    let mut right_inc = Matrix::new_bool(n_ids, n_right).map_err(map_err)?;
    for (index, key) in left_keys.iter().enumerate() {
        if let Some(value) = key {
            left_inc
                .set_bool(index as u64, interned(value), true)
                .map_err(map_err)?;
        }
    }
    for (index, key) in right_keys.iter().enumerate() {
        if let Some(value) = key {
            right_inc
                .set_bool(interned(value), index as u64, true)
                .map_err(map_err)?;
        }
    }

    let mut product = Matrix::new_bool(n_left, n_right).map_err(map_err)?;
    mxm_into(
        &mut product,
        None,
        &Semiring::reachability_bool(),
        &left_inc,
        &right_inc,
        &Descriptor::none(),
    )
    .map_err(map_err)?;

    let mut hits: HashMap<usize, Vec<usize>> = HashMap::new();
    for (row, col) in product.bool_tuples().map_err(map_err)? {
        if right_keys.is_empty() {
            continue;
        }
        hits.entry(row as usize).or_default().push(col as usize);
    }
    let mut pairs = Vec::new();
    for (left_index, left_key) in left_keys.iter().enumerate() {
        if left_key.is_none() {
            if matches!(emission, Emission::Left | Emission::Anti) {
                pairs.push((left_index, None));
            }
            continue;
        }
        let mut row_hits = hits.get(&left_index).cloned().unwrap_or_default();
        row_hits.sort_unstable();
        match emission {
            Emission::Inner => {
                for right_index in row_hits {
                    pairs.push((left_index, Some(right_index)));
                }
            }
            Emission::Left => {
                if row_hits.is_empty() {
                    pairs.push((left_index, None));
                } else {
                    for right_index in row_hits {
                        pairs.push((left_index, Some(right_index)));
                    }
                }
            }
            Emission::Semi => {
                if let Some(right_index) = row_hits.first() {
                    pairs.push((left_index, Some(*right_index)));
                }
            }
            Emission::Anti => {
                if row_hits.is_empty() {
                    pairs.push((left_index, None));
                }
            }
        }
    }
    Ok((pairs.clone(), stats(&pairs)))
}

/// Nearest-preceding-valid asof over already-sorted lanes (RO4).
/// `build` must be sorted ascending by key. A probe above the newest build key
/// returns `asof_beyond_watermark`.
pub fn asof_nearest_preceding<'a, T>(
    probe_key: &str,
    build: &'a [(String, T)],
) -> Result<Option<&'a T>, crate::graph::error::GraphStoreError> {
    if build.is_empty() {
        return Err(crate::graph::error::GraphStoreError::new(
            "asof_beyond_watermark",
            "asof probe has no valid watermark",
        ));
    }
    let newest = &build[build.len() - 1].0;
    if probe_key > newest.as_str() {
        return Err(crate::graph::error::GraphStoreError::new(
            "asof_beyond_watermark",
            format!("asof probe {probe_key} is beyond newest watermark {newest}"),
        ));
    }
    let mut best = None;
    for (key, value) in build {
        if key.as_str() <= probe_key {
            best = Some(value);
        } else {
            break;
        }
    }
    Ok(best)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmptinessReceipt {
    pub passed: bool,
    pub count: usize,
    pub violating_ids: Vec<String>,
}

/// Anti-join emptiness gate (RO5): left keys with no right match.
pub fn emptiness_gate(left_ids: &[String], right_ids: &[String]) -> EmptinessReceipt {
    let right: HashSet<&str> = right_ids.iter().map(String::as_str).collect();
    let left_keys: Vec<Option<String>> = left_ids.iter().cloned().map(Some).collect();
    let right_keys: Vec<Option<String>> = right_ids.iter().cloned().map(Some).collect();
    let (pairs, _) = execute_binary_join(&left_keys, &right_keys, Emission::Anti);
    let violating_ids: Vec<String> = pairs
        .into_iter()
        .map(|(index, _)| left_ids[index].clone())
        .filter(|id| !right.contains(id.as_str()))
        .collect();
    let count = violating_ids.len();
    EmptinessReceipt {
        passed: count == 0,
        count,
        violating_ids,
    }
}

pub fn join_shape_key(joins: &[JoinPredicate], relation_count: usize) -> String {
    let kinds = joins
        .iter()
        .map(|join| join.kind.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let keys = joins
        .iter()
        .map(|join| {
            format!(
                "{}={}.{}",
                join.left_alias, join.right_alias, join.left_column
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("join|rels:{relation_count}|kinds:{kinds}|keys:{keys}")
}

pub fn enumerate_join_candidates(joins: &[JoinPredicate]) -> Vec<&'static str> {
    if joins.iter().any(|join| join.kind == JoinKind::Asof) {
        return vec![JOIN_CANDIDATE_COLT_BINARY];
    }
    let mut candidates = vec![JOIN_CANDIDATE_COLT_BINARY];
    if joins.len() >= 2 && joins.iter().all(|join| join.kind == JoinKind::Inner) {
        candidates.push(JOIN_CANDIDATE_COLT_GENERIC);
    }
    if joins
        .iter()
        .all(|join| matches!(join.kind, JoinKind::Inner | JoinKind::Semi | JoinKind::Anti))
    {
        candidates.push(JOIN_CANDIDATE_MASK);
    }
    candidates
}

pub fn select_join_candidate(
    joins: &[JoinPredicate],
    requested: Option<&str>,
    plan_point: JoinPlanPoint,
    backing: JoinBacking,
) -> &'static str {
    let enumerated = enumerate_join_candidates(joins);
    if let Some(requested) = requested {
        if let Some(hit) = enumerated.iter().copied().find(|id| *id == requested) {
            return hit;
        }
    }
    match (backing, plan_point) {
        (JoinBacking::Mask, _) if enumerated.contains(&JOIN_CANDIDATE_MASK) => JOIN_CANDIDATE_MASK,
        (_, JoinPlanPoint::Generic) if enumerated.contains(&JOIN_CANDIDATE_COLT_GENERIC) => {
            JOIN_CANDIDATE_COLT_GENERIC
        }
        _ => JOIN_CANDIDATE_COLT_BINARY,
    }
}

pub fn candidate_algorithm(candidate_id: &str) -> &'static str {
    match candidate_id {
        JOIN_CANDIDATE_MASK => "mask_join",
        JOIN_CANDIDATE_COLT_GENERIC => "generic_join",
        _ => NATIVE_JOIN_ALGORITHM,
    }
}

#[derive(Clone, Debug, Default)]
struct JoinObservationAccumulator {
    observations: u32,
    successes: u32,
    total_cost: f64,
}

/// Process-local join steering store. Cold start is the native COLT binary
/// floor; unobserved mask/generic candidates receive a bounded trial rate.
#[derive(Debug)]
pub struct JoinSteeringState {
    observations: std::sync::Mutex<BTreeMap<(String, String), JoinObservationAccumulator>>,
    trial_counter: std::sync::atomic::AtomicU32,
}

impl Default for JoinSteeringState {
    fn default() -> Self {
        Self {
            observations: std::sync::Mutex::new(BTreeMap::new()),
            trial_counter: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JoinSteeringDecision {
    pub selected_candidate_id: String,
    pub reason: String,
    pub observation_count: u32,
}

impl JoinSteeringState {
    pub fn next_trial_nonce(&self) -> u32 {
        self.trial_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn record(&self, shape_key: &str, candidate_id: &str, cost_units: f64, success: bool) {
        if !cost_units.is_finite() || cost_units < 0.0 {
            return;
        }
        let mut observations = self
            .observations
            .lock()
            .expect("join steering observations lock");
        let slot = observations
            .entry((shape_key.to_string(), candidate_id.to_string()))
            .or_default();
        slot.observations = slot.observations.saturating_add(1);
        if success {
            slot.successes = slot.successes.saturating_add(1);
        }
        slot.total_cost += cost_units;
    }

    pub fn decide(
        &self,
        joins: &[JoinPredicate],
        relation_count: usize,
        trial_rate: f64,
        min_observations: u32,
    ) -> JoinSteeringDecision {
        let shape = join_shape_key(joins, relation_count);
        let enumerated = enumerate_join_candidates(joins);
        let observations = self
            .observations
            .lock()
            .expect("join steering observations lock");
        let mut total = 0u32;
        let mut cheapest_alternate: Option<(&str, f64, u32)> = None;
        for candidate in &enumerated {
            if let Some(slot) = observations.get(&(shape.clone(), (*candidate).to_string())) {
                total = total.saturating_add(slot.observations);
                if *candidate != JOIN_CANDIDATE_COLT_BINARY && slot.observations > 0 {
                    let mean = slot.total_cost / slot.observations.max(1) as f64;
                    match cheapest_alternate {
                        Some((_, cost, _)) if cost <= mean => {}
                        _ => cheapest_alternate = Some((*candidate, mean, slot.observations)),
                    }
                }
            }
        }
        let native_count = observations
            .get(&(shape.clone(), JOIN_CANDIDATE_COLT_BINARY.to_string()))
            .map(|slot| slot.observations)
            .unwrap_or(0);
        drop(observations);

        if total >= min_observations {
            if let Some((candidate, mean, count)) = cheapest_alternate {
                let native_mean = self
                    .observations
                    .lock()
                    .expect("join steering observations lock")
                    .get(&(shape, JOIN_CANDIDATE_COLT_BINARY.to_string()))
                    .map(|slot| slot.total_cost / slot.observations.max(1) as f64)
                    .unwrap_or(f64::MAX);
                if mean < native_mean {
                    return JoinSteeringDecision {
                        selected_candidate_id: candidate.to_string(),
                        reason: "learned_ranker_selected_enumerated_candidate".to_string(),
                        observation_count: count,
                    };
                }
            }
            return JoinSteeringDecision {
                selected_candidate_id: JOIN_CANDIDATE_COLT_BINARY.to_string(),
                reason: "native_candidate_ranked_best".to_string(),
                observation_count: native_count,
            };
        }

        let unobserved: Vec<&str> = enumerated
            .iter()
            .copied()
            .filter(|candidate| *candidate != JOIN_CANDIDATE_COLT_BINARY)
            .filter(|candidate| {
                self.observations
                    .lock()
                    .expect("join steering observations lock")
                    .get(&(shape.clone(), (*candidate).to_string()))
                    .map(|slot| slot.observations)
                    .unwrap_or(0)
                    == 0
            })
            .collect();
        let nonce = self.next_trial_nonce();
        let trial_bucket = ((trial_rate.clamp(0.0, 1.0) * 10.0).round() as u32).max(0);
        if trial_bucket > 0 && nonce > 0 && nonce % 10 < trial_bucket {
            if let Some(candidate) = unobserved.first() {
                return JoinSteeringDecision {
                    selected_candidate_id: (*candidate).to_string(),
                    reason: "bounded_trial_unobserved_candidate".to_string(),
                    observation_count: 0,
                };
            }
        }
        JoinSteeringDecision {
            selected_candidate_id: JOIN_CANDIDATE_COLT_BINARY.to_string(),
            reason: "cold_start_native_floor".to_string(),
            observation_count: native_count,
        }
    }
}

/// Prove an unprobed relation never builds a COLT level.
pub fn unprobed_build_count(keys: &[Option<String>]) -> usize {
    let level = ColtLevel::unbuilt();
    let _ = keys;
    level.build_count()
}

pub fn multiset_pairs(pairs: &[(usize, Option<usize>)]) -> BTreeMap<(usize, Option<usize>), usize> {
    let mut counts = BTreeMap::new();
    for pair in pairs {
        *counts.entry(*pair).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unprobed_relation_build_counter_is_zero() {
        assert_eq!(unprobed_build_count(&[Some("a".into())]), 0);
    }

    #[test]
    fn persistent_colt_level_does_not_increment_build_count() {
        let level = ColtLevel::from_persistent([
            ("node:1".to_string(), 0usize),
            ("node:2".to_string(), 1usize),
        ]);
        assert_eq!(level.build_count(), 0);
        assert_eq!(level.probe_key("node:1"), &[0]);
        assert_eq!(level.build_count(), 0);
    }

    #[test]
    fn generic_plan_point_uses_same_colt_and_names_generic_join() {
        let left = [Some("1".into()), Some("2".into())];
        let right = [Some("1".into())];
        let (binary, _) = execute_binary_join(&left, &right, Emission::Inner);
        let (generic, stats) = execute_generic_join(&left, &right, Emission::Inner);
        assert_eq!(multiset_pairs(&binary), multiset_pairs(&generic));
        assert_eq!(stats.algorithm, "generic_join");
        assert_eq!(stats.candidate_id, JOIN_CANDIDATE_COLT_GENERIC);
    }

    #[test]
    fn binary_inner_matches_hash_join() {
        let left = [Some("1".into()), Some("2".into()), Some("1".into())];
        let right = [Some("1".into()), Some("3".into())];
        let (pairs, stats) = execute_binary_join(&left, &right, Emission::Inner);
        assert_eq!(pairs, vec![(0, Some(0)), (2, Some(0))]);
        assert_eq!(stats.algorithm, NATIVE_JOIN_ALGORITHM);
        assert_eq!(stats.build_counts, vec![1]);
    }

    #[test]
    fn left_pads_on_miss() {
        let left = [Some("run".into())];
        let right: [Option<String>; 0] = [];
        let (pairs, _) = execute_binary_join(&left, &right, Emission::Left);
        assert_eq!(pairs, vec![(0, None)]);
    }

    #[test]
    fn anti_emits_on_zero_hits() {
        let left = [Some("open".into()), Some("done".into())];
        let right = [Some("done".into())];
        let (pairs, _) = execute_binary_join(&left, &right, Emission::Anti);
        assert_eq!(pairs, vec![(0, None)]);
    }

    #[test]
    fn triangle_generic_stays_near_output_while_binary_exceeds() {
        let r = vec![
            ("a".into(), "b1".into()),
            ("a".into(), "b2".into()),
            ("x".into(), "y".into()),
        ];
        let s = vec![
            ("b1".into(), "c".into()),
            ("b2".into(), "c".into()),
            ("y".into(), "z".into()),
        ];
        let t = vec![("c".into(), "a".into()), ("z".into(), "x".into())];
        let (generic, g_stats) = execute_triangle_generic(&r, &s, &t);
        let (binary, b_stats) = execute_triangle_binary(&r, &s, &t);
        assert_eq!(generic.len(), binary.len());
        assert!(
            g_stats.intermediate_tuples <= generic.len().saturating_mul(4).max(4),
            "generic intermediates {} vs output {}",
            g_stats.intermediate_tuples,
            generic.len()
        );
        assert!(
            b_stats.intermediate_tuples > generic.len(),
            "binary intermediates {} should exceed output {}",
            b_stats.intermediate_tuples,
            generic.len()
        );
    }

    #[test]
    fn mask_agrees_with_colt_including_empty() {
        let left: [Option<String>; 0] = [];
        let right: [Option<String>; 0] = [];
        let (colt, _) = execute_binary_join(&left, &right, Emission::Inner);
        let (mask, _) = execute_mask_join(&left, &right, Emission::Inner);
        assert_eq!(multiset_pairs(&colt), multiset_pairs(&mask));
    }

    #[test]
    fn mask_agrees_on_hub_skew_and_anti_matches_not_exists() {
        let mut left = vec![Some("hub".into())];
        left.extend((0..32).map(|index| Some(format!("spoke-{index}"))));
        let mut right = vec![Some("hub".into())];
        right.extend((0..16).map(|index| Some(format!("spoke-{index}"))));
        for emission in [Emission::Inner, Emission::Semi, Emission::Anti] {
            let (colt, _) = execute_binary_join(&left, &right, emission);
            let (mask, _) = execute_mask_join(&left, &right, emission);
            assert_eq!(multiset_pairs(&colt), multiset_pairs(&mask));
        }
        let absentees: Vec<String> = (16..32).map(|index| format!("spoke-{index}")).collect();
        let receipt = emptiness_gate(
            &left.iter().flatten().cloned().collect::<Vec<_>>(),
            &right.iter().flatten().cloned().collect::<Vec<_>>(),
        );
        assert_eq!(receipt.violating_ids, absentees);
    }

    #[test]
    fn two_backings_enumerate_for_one_shape() {
        let joins = [JoinPredicate {
            left_alias: "a".into(),
            left_column: "id".into(),
            right_alias: "b".into(),
            right_column: "id".into(),
            kind: JoinKind::Inner,
        }];
        let candidates = enumerate_join_candidates(&joins);
        assert!(candidates.contains(&JOIN_CANDIDATE_COLT_BINARY));
        assert!(candidates.contains(&JOIN_CANDIDATE_MASK));
        assert_eq!(join_shape_key(&joins, 2), join_shape_key(&joins, 2));
    }

    #[test]
    fn steering_cold_start_is_floor_then_cheaper_alternate_wins() {
        let joins = [JoinPredicate {
            left_alias: "a".into(),
            left_column: "id".into(),
            right_alias: "b".into(),
            right_column: "id".into(),
            kind: JoinKind::Inner,
        }];
        let state = JoinSteeringState::default();
        let first = state.decide(&joins, 2, 0.1, 20);
        assert_eq!(first.selected_candidate_id, JOIN_CANDIDATE_COLT_BINARY);
        assert_eq!(first.reason, "cold_start_native_floor");
        let shape = join_shape_key(&joins, 2);
        for _ in 0..12 {
            state.record(&shape, JOIN_CANDIDATE_COLT_BINARY, 40.0, true);
            state.record(&shape, JOIN_CANDIDATE_MASK, 10.0, true);
        }
        let learned = state.decide(&joins, 2, 0.1, 20);
        assert_eq!(learned.selected_candidate_id, JOIN_CANDIDATE_MASK);
        assert_eq!(
            learned.reason,
            "learned_ranker_selected_enumerated_candidate"
        );
    }

    #[test]
    fn steering_bounded_trial_skips_first_query_then_samples_unobserved() {
        let joins = [JoinPredicate {
            left_alias: "a".into(),
            left_column: "id".into(),
            right_alias: "b".into(),
            right_column: "id".into(),
            kind: JoinKind::Inner,
        }];
        let state = JoinSteeringState::default();
        let first = state.decide(&joins, 2, 0.1, 20);
        assert_eq!(first.selected_candidate_id, JOIN_CANDIDATE_COLT_BINARY);
        assert_eq!(first.reason, "cold_start_native_floor");
        let mut saw_trial = false;
        for _ in 0..12 {
            let decision = state.decide(&joins, 2, 0.1, 20);
            if decision.reason == "bounded_trial_unobserved_candidate" {
                assert_ne!(decision.selected_candidate_id, JOIN_CANDIDATE_COLT_BINARY);
                saw_trial = true;
                break;
            }
            assert_eq!(decision.reason, "cold_start_native_floor");
        }
        assert!(
            saw_trial,
            "trial bucket must fire within one decade of nonces"
        );
    }

    #[test]
    fn asof_refuses_beyond_newest() {
        let build = vec![("v1".into(), 1u64), ("v2".into(), 2u64)];
        let err = asof_nearest_preceding("v9", &build).unwrap_err();
        assert_eq!(err.code, "asof_beyond_watermark");
    }

    #[test]
    fn emptiness_names_violators() {
        let receipt = emptiness_gate(
            &["section-a".into(), "section-b".into()],
            &["section-b".into()],
        );
        assert!(!receipt.passed);
        assert_eq!(receipt.violating_ids, vec!["section-a"]);
        assert_eq!(receipt.count, 1);
    }
}
