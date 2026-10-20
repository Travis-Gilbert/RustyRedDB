//! Scoped cache-aside helpers for structural PPR vectors.
//!
//! The cache key carries the graph version, so graph writes naturally miss old
//! structural priors. Callers still apply volatile multipliers such as fitness
//! and activation after PPR.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::dynamic_ppr::{
    DynamicPprEdgeUpdate, DynamicPprMaintainer, DynamicPprParams, DynamicPprStats,
};
use crate::graph_csr::CsrGraph;
use crate::state::stable_hash;
use crate::GraphStoreResult;

const DEFAULT_SCOPED_PPR_CACHE_CAPACITY: usize = 256;
const DEFAULT_SCOPED_PPR_CACHE_TTL: Duration = Duration::from_secs(300);
const DEFAULT_MAINTAINED_PPR_CACHE_CAPACITY: usize = 32;
const DEFAULT_MAINTAINED_PPR_CACHE_TTL: Duration = Duration::from_secs(300);

#[derive(Clone, Debug)]
struct CachedPprEntry {
    scores: HashMap<String, f64>,
    inserted_at: Instant,
}

#[derive(Debug)]
struct ScopedPprCache {
    entries: HashMap<String, CachedPprEntry>,
    order: VecDeque<String>,
    capacity: usize,
    ttl: Duration,
}

impl Default for ScopedPprCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity: DEFAULT_SCOPED_PPR_CACHE_CAPACITY,
            ttl: DEFAULT_SCOPED_PPR_CACHE_TTL,
        }
    }
}

impl ScopedPprCache {
    fn get(&mut self, key: &str) -> Option<HashMap<String, f64>> {
        let Some(entry) = self.entries.get(key) else {
            return None;
        };
        if entry.inserted_at.elapsed() > self.ttl {
            self.entries.remove(key);
            self.order.retain(|queued| queued != key);
            return None;
        }
        Some(entry.scores.clone())
    }

    fn insert(&mut self, key: String, value: HashMap<String, f64>) {
        if self.entries.contains_key(&key) {
            self.entries.insert(
                key,
                CachedPprEntry {
                    scores: value,
                    inserted_at: Instant::now(),
                },
            );
            return;
        }
        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            CachedPprEntry {
                scores: value,
                inserted_at: Instant::now(),
            },
        );
        while self.entries.len() > self.capacity {
            let Some(evicted) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&evicted);
        }
    }
}

static SCOPED_PPR_CACHE: OnceLock<Mutex<ScopedPprCache>> = OnceLock::new();
#[derive(Debug)]
struct MaintainedPprEntry {
    maintainer: DynamicPprMaintainer,
    last_touched: Instant,
}

static MAINTAINED_PPR_CACHE: OnceLock<Mutex<HashMap<String, MaintainedPprEntry>>> = OnceLock::new();
static CSR_PPR_RUNS: AtomicUsize = AtomicUsize::new(0);
static CSR_BUILDS_FROM_ADJACENCY: AtomicUsize = AtomicUsize::new(0);

fn cache() -> &'static Mutex<ScopedPprCache> {
    SCOPED_PPR_CACHE.get_or_init(|| Mutex::new(ScopedPprCache::default()))
}

fn maintained_cache() -> &'static Mutex<HashMap<String, MaintainedPprEntry>> {
    MAINTAINED_PPR_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScopedPprExecutionCounters {
    pub csr_ppr_runs: usize,
    pub csr_builds_from_adjacency: usize,
}

pub fn scoped_ppr_execution_counters() -> ScopedPprExecutionCounters {
    ScopedPprExecutionCounters {
        csr_ppr_runs: CSR_PPR_RUNS.load(Ordering::SeqCst),
        csr_builds_from_adjacency: CSR_BUILDS_FROM_ADJACENCY.load(Ordering::SeqCst),
    }
}

pub fn reset_scoped_ppr_execution_counters() {
    CSR_PPR_RUNS.store(0, Ordering::SeqCst);
    CSR_BUILDS_FROM_ADJACENCY.store(0, Ordering::SeqCst);
}

/// Run PPR through the process-local scoped cache.
///
/// The key is `scope + graph_version + PPR params + seed set`. The adjacency is
/// intentionally not hashed; callers must pass the store snapshot version that
/// changes whenever the structural graph changes.
pub fn cached_personalized_pagerank(
    scope: &str,
    graph_version: u64,
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    seeds: &HashMap<String, f64>,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> GraphStoreResult<HashMap<String, f64>> {
    if seeds.is_empty() {
        return Ok(HashMap::new());
    }
    let key = scoped_ppr_cache_key(scope, graph_version, seeds, alpha, epsilon, max_pushes)?;
    if let Ok(mut guard) = cache().lock() {
        if let Some(cached) = guard.get(&key) {
            return Ok(cached);
        }
    }

    CSR_BUILDS_FROM_ADJACENCY.fetch_add(1, Ordering::SeqCst);
    let csr = CsrGraph::from_weighted_adjacency(adjacency, seeds.keys());
    let scores = run_csr_personalized_pagerank(&csr, seeds, alpha, epsilon, max_pushes);
    if let Ok(mut guard) = cache().lock() {
        guard.insert(key, scores.clone());
    }
    Ok(scores)
}

/// Run PPR through the scoped cache when the caller already has a CSR view.
pub fn cached_csr_personalized_pagerank(
    scope: &str,
    graph_version: u64,
    csr: &CsrGraph,
    seeds: &HashMap<String, f64>,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> GraphStoreResult<HashMap<String, f64>> {
    if seeds.is_empty() {
        return Ok(HashMap::new());
    }
    let key = scoped_ppr_cache_key(scope, graph_version, seeds, alpha, epsilon, max_pushes)?;
    if let Ok(mut guard) = cache().lock() {
        if let Some(cached) = guard.get(&key) {
            return Ok(cached);
        }
    }

    let scores = run_csr_personalized_pagerank(csr, seeds, alpha, epsilon, max_pushes);
    if let Ok(mut guard) = cache().lock() {
        guard.insert(key, scores.clone());
    }
    Ok(scores)
}

/// Cache a unit single-seed PPR vector, then scale it for the requested weight.
///
/// This is the stable-seed decomposition used by task-type and project-anchor
/// priors. Dynamic query seeds can be computed fresh and added to this vector.
pub fn cached_single_seed_personalized_pagerank(
    scope: &str,
    graph_version: u64,
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    seed: &str,
    weight: f64,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> GraphStoreResult<HashMap<String, f64>> {
    if seed.trim().is_empty() || weight <= 0.0 {
        return Ok(HashMap::new());
    }
    let mut seeds = HashMap::new();
    seeds.insert(seed.to_string(), 1.0);
    let mut scores = cached_personalized_pagerank(
        scope,
        graph_version,
        adjacency,
        &seeds,
        alpha,
        epsilon,
        max_pushes,
    )?;
    if (weight - 1.0).abs() > f64::EPSILON {
        for score in scores.values_mut() {
            *score *= weight;
        }
    }
    Ok(scores)
}

/// Register a stable-seed PPR vector for delta maintenance.
///
/// Maintained entries are keyed without the graph version because edge deltas
/// migrate the vector forward in place. Non-maintained callers keep using the
/// version-keyed cache above.
pub fn register_maintained_ppr_entry(
    scope: &str,
    adjacency: HashMap<String, Vec<(String, f64)>>,
    seeds: HashMap<String, f64>,
    params: DynamicPprParams,
) -> GraphStoreResult<String> {
    let key = maintained_ppr_cache_key(scope, &seeds, params)?;
    let mut maintainer = DynamicPprMaintainer::new(adjacency, seeds, params);
    maintainer.reset_full_recompute_counter();
    if let Ok(mut guard) = maintained_cache().lock() {
        guard.retain(|_, entry| entry.last_touched.elapsed() <= DEFAULT_MAINTAINED_PPR_CACHE_TTL);
        if guard.len() >= DEFAULT_MAINTAINED_PPR_CACHE_CAPACITY {
            if let Some(oldest) = guard
                .iter()
                .min_by_key(|(_, entry)| entry.last_touched)
                .map(|(key, _)| key.clone())
            {
                guard.remove(&oldest);
            }
        }
        guard.insert(
            key.clone(),
            MaintainedPprEntry {
                maintainer,
                last_touched: Instant::now(),
            },
        );
    }
    Ok(key)
}

pub fn maintained_ppr_scores(key: &str) -> Option<HashMap<String, f64>> {
    maintained_cache().lock().ok().and_then(|mut guard| {
        let entry = guard.get_mut(key)?;
        entry.last_touched = Instant::now();
        Some(entry.maintainer.scores().clone())
    })
}

pub fn maintained_ppr_stats(key: &str) -> Option<DynamicPprStats> {
    maintained_cache().lock().ok().and_then(|mut guard| {
        let entry = guard.get_mut(key)?;
        entry.last_touched = Instant::now();
        Some(entry.maintainer.stats())
    })
}

pub fn apply_maintained_ppr_update(
    key: &str,
    update: DynamicPprEdgeUpdate,
) -> Option<DynamicPprStats> {
    maintained_cache().lock().ok().and_then(|mut guard| {
        let entry = guard.get_mut(key)?;
        entry.last_touched = Instant::now();
        Some(entry.maintainer.apply_update(update))
    })
}

pub fn clear_maintained_ppr_cache() {
    if let Ok(mut guard) = maintained_cache().lock() {
        guard.clear();
    }
}

pub fn maintained_ppr_cache_len() -> usize {
    maintained_cache()
        .lock()
        .map(|guard| guard.len())
        .unwrap_or_default()
}

/// Add `source` scores into `target`.
pub fn merge_ppr_scores(target: &mut HashMap<String, f64>, source: HashMap<String, f64>) {
    for (node, score) in source {
        *target.entry(node).or_insert(0.0) += score;
    }
}

fn run_csr_personalized_pagerank(
    csr: &CsrGraph,
    seeds: &HashMap<String, f64>,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> HashMap<String, f64> {
    CSR_PPR_RUNS.fetch_add(1, Ordering::SeqCst);
    csr.personalized_pagerank_local_push_named(seeds, alpha, epsilon, max_pushes)
}

pub fn clear_scoped_ppr_cache() {
    if let Ok(mut guard) = cache().lock() {
        guard.entries.clear();
        guard.order.clear();
    }
    clear_maintained_ppr_cache();
}

pub fn scoped_ppr_cache_len() -> usize {
    cache()
        .lock()
        .map(|guard| guard.entries.len())
        .unwrap_or_default()
}

fn scoped_ppr_cache_key(
    scope: &str,
    graph_version: u64,
    seeds: &HashMap<String, f64>,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> GraphStoreResult<String> {
    let mut normalized = seeds
        .iter()
        .map(|(node, weight)| (node.as_str(), normalized_f64_bits(*weight)))
        .collect::<Vec<_>>();
    normalized.sort_by(|left, right| left.0.cmp(right.0));
    stable_hash((
        "scoped-ppr-cache-v1",
        scope,
        graph_version,
        normalized,
        normalized_f64_bits(alpha),
        normalized_f64_bits(epsilon),
        max_pushes,
    ))
}

fn maintained_ppr_cache_key(
    scope: &str,
    seeds: &HashMap<String, f64>,
    params: DynamicPprParams,
) -> GraphStoreResult<String> {
    let mut normalized = seeds
        .iter()
        .map(|(node, weight)| (node.as_str(), normalized_f64_bits(*weight)))
        .collect::<Vec<_>>();
    normalized.sort_by(|left, right| left.0.cmp(right.0));
    stable_hash((
        "maintained-ppr-cache-v1",
        scope,
        normalized,
        normalized_f64_bits(params.alpha),
        normalized_f64_bits(params.epsilon),
        params.max_pushes,
    ))
}

fn normalized_f64_bits(value: f64) -> u64 {
    if !value.is_finite() || value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_CACHE_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

    fn cache_test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_CACHE_GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap()
    }

    fn chain() -> HashMap<String, Vec<(String, f64)>> {
        HashMap::from([
            ("a".to_string(), vec![("b".to_string(), 1.0)]),
            ("b".to_string(), vec![("c".to_string(), 1.0)]),
        ])
    }

    #[test]
    fn scoped_cache_reuses_identical_structural_query() {
        let _guard = cache_test_guard();
        clear_scoped_ppr_cache();
        let adjacency = chain();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);

        let first =
            cached_personalized_pagerank("scope", 7, &adjacency, &seeds, 0.15, 1e-5, 100).unwrap();
        let second =
            cached_personalized_pagerank("scope", 7, &adjacency, &seeds, 0.15, 1e-5, 100).unwrap();

        assert_eq!(first, second);
        assert_eq!(scoped_ppr_cache_len(), 1);

        cached_personalized_pagerank("scope", 8, &adjacency, &seeds, 0.15, 1e-5, 100).unwrap();
        assert_eq!(scoped_ppr_cache_len(), 2);
    }

    #[test]
    fn single_seed_cache_scales_weight_after_lookup() {
        let _guard = cache_test_guard();
        clear_scoped_ppr_cache();
        let adjacency = chain();

        let unit = cached_single_seed_personalized_pagerank(
            "single", 1, &adjacency, "a", 1.0, 0.15, 1e-5, 100,
        )
        .unwrap();
        let doubled = cached_single_seed_personalized_pagerank(
            "single", 1, &adjacency, "a", 2.0, 0.15, 1e-5, 100,
        )
        .unwrap();

        assert_eq!(scoped_ppr_cache_len(), 1);
        assert!((doubled["a"] - unit["a"] * 2.0).abs() < 1e-9);
    }

    #[test]
    fn maintained_entry_retracts_edge_and_counts_refresh() {
        let _guard = cache_test_guard();
        clear_scoped_ppr_cache();
        clear_maintained_ppr_cache();
        reset_scoped_ppr_execution_counters();
        let adjacency = HashMap::from([(
            "claim:seed".to_string(),
            vec![
                ("claim:retracted".to_string(), 1.0),
                ("claim:kept".to_string(), 1.0),
            ],
        )]);
        let seeds = HashMap::from([("claim:seed".to_string(), 1.0)]);
        let params = DynamicPprParams {
            alpha: 0.15,
            epsilon: 1e-6,
            max_pushes: 10_000,
        };
        let key =
            register_maintained_ppr_entry("epistemic-shadow", adjacency, seeds, params).unwrap();

        assert!(maintained_ppr_scores(&key)
            .expect("scores")
            .contains_key("claim:retracted"));
        let stats = apply_maintained_ppr_update(
            &key,
            DynamicPprEdgeUpdate::delete("claim:seed", "claim:retracted"),
        )
        .expect("update stats");
        let after = maintained_ppr_scores(&key).expect("updated scores");

        assert!(
            !after.contains_key("claim:retracted"),
            "retracted belief should fall out of maintained ranked output"
        );
        assert!(after.contains_key("claim:kept"));
        assert_eq!(stats.full_recompute_count, 0);
        assert_eq!(stats.incremental_update_count, 1);
        assert_eq!(scoped_ppr_execution_counters().csr_ppr_runs, 0);
        assert_eq!(maintained_ppr_cache_len(), 1);
        clear_maintained_ppr_cache();
    }

    #[test]
    fn maintained_cache_evicts_oldest_entry_at_capacity() {
        let _guard = cache_test_guard();
        clear_maintained_ppr_cache();
        let params = DynamicPprParams {
            alpha: 0.15,
            epsilon: 1e-5,
            max_pushes: 100,
        };
        for index in 0..=DEFAULT_MAINTAINED_PPR_CACHE_CAPACITY {
            register_maintained_ppr_entry(
                &format!("scope:{index}"),
                HashMap::new(),
                HashMap::from([(format!("seed:{index}"), 1.0)]),
                params,
            )
            .unwrap();
        }
        assert_eq!(
            maintained_ppr_cache_len(),
            DEFAULT_MAINTAINED_PPR_CACHE_CAPACITY
        );
        clear_maintained_ppr_cache();
    }
}
