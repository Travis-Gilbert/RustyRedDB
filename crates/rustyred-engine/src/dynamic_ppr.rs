//! Dynamic maintenance for stable-seed Personalized PageRank vectors.
//!
//! This module keeps the maintenance contract deliberately narrow: a caller
//! registers stable seeds and applies edge deltas to the owned adjacency view.
//! Updates refresh the maintained vector through the bounded local-push routine
//! over the current adjacency and expose counters that keep refresh costs visible.
//!
//! ## Incremental maintenance (arXiv:1603.07796)
//!
//! Every forward push preserves the invariant `p + ppr(r) = ppr(s)` where `p`
//! is the reserve vector, `r` is the residual vector, and `s` is the seed
//! distribution. By persisting the residual across edge updates and applying a
//! local restore step at the changed endpoint, this maintainer avoids
//! recomputing PPR from scratch on every edge delta.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicPprParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub max_pushes: usize,
}

impl Default for DynamicPprParams {
    fn default() -> Self {
        Self {
            alpha: 0.15,
            epsilon: 1e-6,
            max_pushes: 10_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicPprUpdateKind {
    Insert,
    Delete,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicPprEdgeUpdate {
    pub from_id: String,
    pub to_id: String,
    pub weight: f64,
    pub kind: DynamicPprUpdateKind,
}

impl DynamicPprEdgeUpdate {
    pub fn insert(from_id: impl Into<String>, to_id: impl Into<String>, weight: f64) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            weight,
            kind: DynamicPprUpdateKind::Insert,
        }
    }

    pub fn delete(from_id: impl Into<String>, to_id: impl Into<String>) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            weight: 0.0,
            kind: DynamicPprUpdateKind::Delete,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DynamicPprStats {
    pub update_count: usize,
    pub full_recompute_count: usize,
    pub incremental_update_count: usize,
    pub last_touched_nodes: usize,
    pub last_pushes: usize,
    pub total_pushes: usize,
}

#[derive(Clone, Debug)]
pub struct DynamicPprMaintainer {
    adjacency: HashMap<String, Vec<(String, f64)>>,
    seeds: HashMap<String, f64>,
    params: DynamicPprParams,
    /// Reserve vector `p` — the maintained PPR scores.
    scores: HashMap<String, f64>,
    /// Residual vector `r` — persisted across incremental updates (D1).
    residual: HashMap<String, f64>,
    stats: DynamicPprStats,
}

impl DynamicPprMaintainer {
    pub fn new(
        adjacency: HashMap<String, Vec<(String, f64)>>,
        seeds: HashMap<String, f64>,
        params: DynamicPprParams,
    ) -> Self {
        let (scores_full, residual, counters) = full_push_scores(&adjacency, &seeds, params);
        let scores: HashMap<String, f64> =
            scores_full.into_iter().filter(|(_, v)| *v != 0.0).collect();
        Self {
            adjacency,
            seeds,
            params,
            scores,
            residual,
            stats: DynamicPprStats {
                full_recompute_count: 1,
                last_touched_nodes: counters.touched_nodes,
                last_pushes: counters.pushes,
                total_pushes: counters.pushes,
                update_count: 0,
                incremental_update_count: 0,
            },
        }
    }

    pub fn scores(&self) -> &HashMap<String, f64> {
        &self.scores
    }

    pub fn stats(&self) -> DynamicPprStats {
        self.stats
    }

    pub fn reset_full_recompute_counter(&mut self) {
        self.stats.full_recompute_count = 0;
    }

    /// Refresh the maintained PPR vector from seeds against the current
    /// adjacency via a full forward push. Replaces both reserve and residual.
    /// Increments `full_recompute_count` by 1 (D6).
    pub fn refresh_full(&mut self) -> DynamicPprStats {
        let (scores_full, residual, counters) =
            full_push_scores(&self.adjacency, &self.seeds, self.params);
        self.scores = scores_full.into_iter().filter(|(_, v)| *v != 0.0).collect();
        self.residual = residual;
        self.stats.full_recompute_count += 1;
        self.stats.last_touched_nodes = counters.touched_nodes;
        self.stats.last_pushes = counters.pushes;
        self.stats.total_pushes += counters.pushes;
        self.stats
    }

    /// Apply one edge delta incrementally (D4).
    ///
    /// 1. Capture old out-neighbours of `from_id`.
    /// 2. Mutate adjacency.
    /// 3. Restore the local-push invariant at `from_id` (D3).
    /// 4. Resume pushing from the affected frontier.
    ///
    /// `full_recompute_count` is *not* incremented; `incremental_update_count`
    /// is incremented instead (D5). Deletions that leave the restored residual
    /// outside tolerance fall back to a bounded local re-push or a full refresh
    /// (D7).
    pub fn apply_update(&mut self, update: DynamicPprEdgeUpdate) -> DynamicPprStats {
        let from_key = &update.from_id;
        let old_neighbors = self.adjacency.get(from_key).cloned().unwrap_or_default();

        // ---- mutate adjacency -------------------------------------------------
        apply_adjacency_update(&mut self.adjacency, update.clone());

        let new_neighbors = self.adjacency.get(from_key).cloned().unwrap_or_default();

        // ---- restore invariant at the affected node (D3) ----------------------
        let frontier = restore_step(
            &mut self.residual,
            &mut self.scores,
            from_key,
            &old_neighbors,
            &new_neighbors,
            self.params.alpha,
        );

        // D7: deletion fallback — if restore left any negative residual, attempt
        //      a bounded re-push before falling through to a full refresh.
        if update.kind == DynamicPprUpdateKind::Delete {
            let any_negative = frontier
                .iter()
                .any(|n| self.residual.get(n).copied().unwrap_or(0.0) < -self.params.epsilon);
            if any_negative {
                let ok = bounded_local_repush(
                    &self.adjacency,
                    &mut self.scores,
                    &mut self.residual,
                    &update.from_id,
                    &update.to_id,
                    self.params,
                );
                if !ok {
                    return self.refresh_full();
                }
            }
        }

        // ---- resume push from the affected frontier (D2 / D4) -----------------
        let counters = resume_push(
            &self.adjacency,
            &mut self.scores,
            &mut self.residual,
            frontier,
            self.params,
        );

        self.stats.update_count += 1;
        self.stats.incremental_update_count += 1;
        self.stats.last_touched_nodes = counters.touched_nodes;
        self.stats.last_pushes = counters.pushes;
        self.stats.total_pushes += counters.pushes;

        // Remove entries that converged to near-zero after incremental push.
        // The constructor and refresh_full filter with !=0.0, but incremental
        // pushes may leave entries at ~ε (asymptotic convergence) rather than
        // exactly zero. Use epsilon so we match the constructor contract.
        self.scores.retain(|_, v| v.abs() >= self.params.epsilon);

        self.stats
    }

    pub fn apply_updates<I>(&mut self, updates: I) -> DynamicPprStats
    where
        I: IntoIterator<Item = DynamicPprEdgeUpdate>,
    {
        let mut stats = self.stats;
        for update in updates {
            stats = self.apply_update(update);
        }
        stats
    }

    /// Replace the stable seed vector by injecting its signed delta into the
    /// maintained residual and resuming local push. PageRank is linear in the
    /// seed vector, so this preserves the same invariant as an edge update
    /// without discarding the current reserve.
    pub fn replace_seeds_incremental(&mut self, seeds: HashMap<String, f64>) -> DynamicPprStats {
        let keys = self
            .seeds
            .keys()
            .chain(seeds.keys())
            .cloned()
            .collect::<HashSet<_>>();
        let mut frontier = HashSet::new();
        for key in keys {
            let old_mass = self.seeds.get(&key).copied().unwrap_or(0.0);
            let new_mass = seeds.get(&key).copied().unwrap_or(0.0);
            let delta = new_mass - old_mass;
            if delta.abs() <= f64::EPSILON {
                continue;
            }
            *self.residual.entry(key.clone()).or_insert(0.0) += delta;
            frontier.insert(key);
        }
        if frontier.is_empty() {
            return self.stats;
        }

        self.seeds = seeds;
        let counters = resume_push(
            &self.adjacency,
            &mut self.scores,
            &mut self.residual,
            frontier,
            self.params,
        );
        self.stats.update_count += 1;
        self.stats.incremental_update_count += 1;
        self.stats.last_touched_nodes = counters.touched_nodes;
        self.stats.last_pushes = counters.pushes;
        self.stats.total_pushes += counters.pushes;
        self.scores
            .retain(|_, value| value.abs() >= self.params.epsilon);
        self.stats
    }
}

// ---- reference helpers (public, unchanged API) -------------------------------

pub fn dynamic_ppr_full_reference(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    seeds: &HashMap<String, f64>,
    params: DynamicPprParams,
) -> HashMap<String, f64> {
    full_push_scores(adjacency, seeds, params).0
}

pub fn ppr_vectors_within_epsilon(
    left: &HashMap<String, f64>,
    right: &HashMap<String, f64>,
    epsilon: f64,
) -> bool {
    let keys = left
        .keys()
        .chain(right.keys())
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    keys.into_iter().all(|key| {
        let l = left.get(key).copied().unwrap_or(0.0);
        let r = right.get(key).copied().unwrap_or(0.0);
        (l - r).abs() <= epsilon
    })
}

// ---- push infrastructure ----------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct LocalPushCounters {
    pushes: usize,
    touched_nodes: usize,
}

/// The shared inner push loop: drain `queue`, pushing residual mass into
/// reserve and forwarding to neighbours.  Returns push and touched-node counts.
fn inner_push_loop(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    reserve: &mut HashMap<String, f64>,
    residual: &mut HashMap<String, f64>,
    queue: &mut VecDeque<String>,
    in_queue: &mut HashSet<String>,
    params: DynamicPprParams,
) -> LocalPushCounters {
    let mut touched = HashSet::new();
    let mut pushes = 0usize;

    while let Some(node) = queue.pop_front() {
        in_queue.remove(&node);
        if pushes >= params.max_pushes {
            break;
        }
        let residual_mass = residual.remove(&node).unwrap_or(0.0);
        let degree = adjacency
            .get(&node)
            .map(Vec::len)
            .unwrap_or_default()
            .max(1) as f64;
        if residual_mass.abs() <= params.epsilon * degree {
            // Preserve below-epsilon residual — it may become significant after
            // edge changes during incremental maintenance.
            *residual.entry(node.clone()).or_insert(0.0) += residual_mass;
            continue;
        }
        pushes += 1;
        touched.insert(node.clone());
        *reserve.entry(node.clone()).or_insert(0.0) += params.alpha * residual_mass;

        let Some(neighbors) = adjacency.get(&node) else {
            *residual.entry(node.clone()).or_insert(0.0) += (1.0 - params.alpha) * residual_mass;
            let current = *residual.get(&node).unwrap_or(&0.0);
            if current.abs() > params.epsilon * degree && in_queue.insert(node.clone()) {
                queue.push_back(node);
            }
            continue;
        };
        let total_weight = neighbors.iter().map(|(_, w)| w.max(0.0)).sum::<f64>();
        if neighbors.is_empty() || total_weight <= 0.0 {
            *residual.entry(node.clone()).or_insert(0.0) += (1.0 - params.alpha) * residual_mass;
            let current = *residual.get(&node).unwrap_or(&0.0);
            if current.abs() > params.epsilon * degree && in_queue.insert(node.clone()) {
                queue.push_back(node);
            }
            continue;
        }
        for (target, weight) in neighbors {
            let share = (1.0 - params.alpha) * residual_mass * weight.max(0.0) / total_weight;
            *residual.entry(target.clone()).or_insert(0.0) += share;
            let target_degree = adjacency
                .get(target)
                .map(Vec::len)
                .unwrap_or_default()
                .max(1) as f64;
            if residual[target].abs() > params.epsilon * target_degree
                && in_queue.insert(target.clone())
            {
                queue.push_back(target.clone());
            }
        }
    }

    LocalPushCounters {
        pushes,
        touched_nodes: touched.len(),
    }
}

/// Full forward push from seeds.  Returns `(reserve, residual, counters)`.
/// The `reserve` map is *not* filtered; the caller strips zero entries when
/// building the public scores view.
fn full_push_scores(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    seeds: &HashMap<String, f64>,
    params: DynamicPprParams,
) -> (
    HashMap<String, f64>,
    HashMap<String, f64>,
    LocalPushCounters,
) {
    if seeds.is_empty() {
        return (HashMap::new(), HashMap::new(), LocalPushCounters::default());
    }
    let mut reserve = HashMap::new();
    let mut residual = HashMap::new();
    let mut queue = VecDeque::new();
    let mut in_queue = HashSet::new();
    for (seed, mass) in seeds {
        if *mass <= 0.0 || seed.trim().is_empty() {
            continue;
        }
        *residual.entry(seed.clone()).or_insert(0.0) += *mass;
        if in_queue.insert(seed.clone()) {
            queue.push_back(seed.clone());
        }
    }
    let counters = inner_push_loop(
        adjacency,
        &mut reserve,
        &mut residual,
        &mut queue,
        &mut in_queue,
        params,
    );
    (reserve, residual, counters)
}

/// Resume a forward push from an existing `(reserve, residual)` state, seeded
/// on the nodes in `frontier`.  Mutates both maps in place.
fn resume_push(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    reserve: &mut HashMap<String, f64>,
    residual: &mut HashMap<String, f64>,
    frontier: HashSet<String>,
    params: DynamicPprParams,
) -> LocalPushCounters {
    let mut queue = VecDeque::new();
    let mut in_queue = HashSet::new();
    for node in &frontier {
        in_queue.insert(node.clone());
        queue.push_back(node.clone());
    }
    inner_push_loop(
        adjacency,
        reserve,
        residual,
        &mut queue,
        &mut in_queue,
        params,
    )
}

/// Bounded local re-push seeded from `source` and `dest` (D7).  Returns
/// `true` if the residual is non-negative after the limited push.
fn bounded_local_repush(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    reserve: &mut HashMap<String, f64>,
    residual: &mut HashMap<String, f64>,
    source: &str,
    dest: &str,
    params: DynamicPprParams,
) -> bool {
    let mut queue = VecDeque::new();
    let mut in_queue = HashSet::new();
    for node in [source, dest] {
        if residual.contains_key(node) && in_queue.insert(node.to_string()) {
            queue.push_back(node.to_string());
        }
    }
    // Tight bound — this is a safety net, not the primary path.
    let bounded_params = DynamicPprParams {
        max_pushes: 200,
        ..params
    };
    inner_push_loop(
        adjacency,
        reserve,
        residual,
        &mut queue,
        &mut in_queue,
        bounded_params,
    );
    !residual.values().any(|&v| v < -params.epsilon)
}

// ---- restore step (D3, arXiv:1603.07796 forward-push section) ----------------

/// Restore the local-push invariant at `source` after its outgoing adjacency
/// changed from `old_neighbors` to `new_neighbors`.
///
/// The operation re-weights the *effective* residual at `source` from the old
/// out-distribution to the new one: subtract old contributions, add new
/// contributions.  Returns the set of affected nodes (the resume frontier).
///
/// After a forward push most pushed nodes have `r[source] ≈ 0` even though
/// large mass already flowed through them.  The effective pre-push residual is
/// recovered as `max(r[source], p[source] / α)`.
fn restore_step(
    residual: &mut HashMap<String, f64>,
    reserve: &mut HashMap<String, f64>,
    source: &str,
    old_neighbors: &[(String, f64)],
    new_neighbors: &[(String, f64)],
    alpha: f64,
) -> HashSet<String> {
    let mut frontier = HashSet::new();
    frontier.insert(source.to_string());
    let old_total: f64 = old_neighbors.iter().map(|(_, w)| w.max(0.0)).sum();
    let new_total: f64 = new_neighbors.iter().map(|(_, w)| w.max(0.0)).sum();

    // A dangling node repeatedly absorbs its forwarded mass back into its own
    // reserve. When its first outgoing edge appears, that absorbed score is
    // the effective visitation mass, not `reserve / alpha`. Keep the teleport
    // share at the source and forward the remainder exactly once.
    if old_total <= 0.0 && new_total > 0.0 {
        let absorbed =
            reserve.get(source).copied().unwrap_or(0.0) + residual.remove(source).unwrap_or(0.0);
        if absorbed <= 0.0 {
            return frontier;
        }
        reserve.insert(source.to_string(), alpha * absorbed);
        frontier.remove(source);
        for (neighbor, weight) in new_neighbors {
            let share = (1.0 - alpha) * absorbed * weight.max(0.0) / new_total;
            *residual.entry(neighbor.clone()).or_insert(0.0) += share;
            frontier.insert(neighbor.clone());
        }
        return frontier;
    }

    // Effective residual: the mass that was at `source` before any push.
    // For pushed nodes r[source]=0 but p[source]/α recovers the original mass.
    // For unpushed nodes p[source]=0 so r[source] is the effective value.
    let r_stored = residual.get(source).copied().unwrap_or(0.0);
    let r_effective = r_stored.max(reserve.get(source).copied().unwrap_or(0.0) / alpha);
    if r_effective <= 0.0 {
        return frontier;
    }

    let one_minus_alpha = 1.0 - alpha;

    // Build an index of old neighbour weights for O(1) lookup.
    let old_weight: HashMap<&str, f64> = old_neighbors
        .iter()
        .map(|(n, w)| (n.as_str(), w.max(0.0)))
        .collect();

    // First pass: neighbours present in the OLD out-distribution.
    for (neighbor, old_w) in old_neighbors {
        let old_contrib = if old_total > 0.0 {
            one_minus_alpha * r_effective * old_w.max(0.0) / old_total
        } else {
            0.0
        };
        let new_w = new_neighbors
            .iter()
            .find(|(n, _)| n == neighbor)
            .map(|(_, w)| w.max(0.0))
            .unwrap_or(0.0);
        let new_contrib = if new_total > 0.0 {
            one_minus_alpha * r_effective * new_w / new_total
        } else {
            0.0
        };
        *residual.entry(neighbor.clone()).or_insert(0.0) += new_contrib - old_contrib;
        frontier.insert(neighbor.clone());
    }

    // Second pass: neighbours present ONLY in the NEW out-distribution.
    for (neighbor, new_w) in new_neighbors {
        if old_weight.contains_key(neighbor.as_str()) {
            continue; // already handled above
        }
        let new_contrib = if new_total > 0.0 {
            one_minus_alpha * r_effective * new_w.max(0.0) / new_total
        } else {
            0.0
        };
        *residual.entry(neighbor.clone()).or_insert(0.0) += new_contrib;
        frontier.insert(neighbor.clone());
    }

    // When source becomes a sink (lost all outgoing edges), the
    // (1-α)*r_effective mass previously distributed through old edges
    // was subtracted from neighbors in Pass 1 but never reclaimed.
    // Reclaim it to source's residual so the sink-convergence loop
    // can push it all to score, preserving mass conservation.
    if new_total <= 0.0 && old_total > 0.0 {
        *residual.entry(source.to_string()).or_insert(0.0) += one_minus_alpha * r_effective;
    }

    frontier
}

// ---- adjacency mutation -----------------------------------------------------

fn apply_adjacency_update(
    adjacency: &mut HashMap<String, Vec<(String, f64)>>,
    update: DynamicPprEdgeUpdate,
) {
    match update.kind {
        DynamicPprUpdateKind::Insert => {
            let neighbors = adjacency.entry(update.from_id).or_default();
            if let Some((_, weight)) = neighbors
                .iter_mut()
                .find(|(target, _)| target == &update.to_id)
            {
                *weight = update.weight.max(0.0);
            } else {
                neighbors.push((update.to_id, update.weight.max(0.0)));
            }
        }
        DynamicPprUpdateKind::Delete => {
            if let Some(neighbors) = adjacency.get_mut(&update.from_id) {
                neighbors.retain(|(target, _)| target != &update.to_id);
            }
        }
    }
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_adjacency() -> HashMap<String, Vec<(String, f64)>> {
        let mut adjacency = HashMap::new();
        adjacency.insert("a".to_string(), vec![("b".to_string(), 1.0)]);
        adjacency.insert("b".to_string(), vec![("c".to_string(), 1.0)]);
        adjacency.insert("c".to_string(), vec![("d".to_string(), 1.0)]);
        for idx in 0..200 {
            adjacency.insert(format!("isolated:{idx}"), Vec::new());
        }
        adjacency
    }

    fn default_params() -> DynamicPprParams {
        DynamicPprParams {
            alpha: 0.15,
            epsilon: 1e-6,
            max_pushes: 10_000,
        }
    }

    // ---- A1 / A2 / A3: mixed-edge updates match reference, counters reflect
    //      incremental operation, and the frontier stays bounded ----------------

    #[test]
    fn mixed_edge_updates_match_full_reference_and_count_refreshes() {
        let params = default_params();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let mut maintainer = DynamicPprMaintainer::new(fixture_adjacency(), seeds.clone(), params);
        maintainer.reset_full_recompute_counter();

        maintainer.apply_updates([
            DynamicPprEdgeUpdate::insert("b", "x", 1.0),
            DynamicPprEdgeUpdate::delete("c", "d"),
        ]);

        let mut expected_adjacency = fixture_adjacency();
        apply_adjacency_update(
            &mut expected_adjacency,
            DynamicPprEdgeUpdate::insert("b", "x", 1.0),
        );
        apply_adjacency_update(
            &mut expected_adjacency,
            DynamicPprEdgeUpdate::delete("c", "d"),
        );
        let expected = dynamic_ppr_full_reference(&expected_adjacency, &seeds, params);
        let stats = maintainer.stats();

        // DEBUG: print score differences
        let scores = maintainer.scores().clone();
        let all_keys: std::collections::BTreeSet<_> =
            scores.keys().chain(expected.keys()).collect();
        let mut diffs: Vec<(&str, f64, f64, f64)> = vec![];
        for k in &all_keys {
            let s = scores.get(*k).copied().unwrap_or(0.0);
            let e = expected.get(*k).copied().unwrap_or(0.0);
            let d = s - e;
            if d.abs() > params.epsilon {
                diffs.push((k, s, e, d));
            }
        }
        diffs.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
        eprintln!("Score diffs > epsilon ({} total):", diffs.len());
        for (k, s, e, d) in &diffs {
            eprintln!("  {}: maint={:.12} ref={:.12} diff={:.12}", k, s, e, d);
        }
        if diffs.len() > 0 {
            // also print the maintainer scores & reference scores side by side
            eprintln!("\nAll scores:");
            let mut all_sorted: Vec<_> = all_keys.iter().collect();
            all_sorted.sort();
            for k in &all_sorted {
                let s = scores.get(k.as_str()).copied().unwrap_or(0.0);
                let e = expected.get(k.as_str()).copied().unwrap_or(0.0);
                if s.abs() > 1e-9 || e.abs() > 1e-9 {
                    eprintln!("  {}: maint={:.10e} ref={:.10e}", k, s, e);
                }
            }
        }

        // A1: scores match full reference within 10× epsilon.
        assert!(ppr_vectors_within_epsilon(
            &scores,
            &expected,
            10.0 * params.epsilon
        ));

        // A2: incremental semantics — full_recompute_count stays at 0.
        assert_eq!(stats.full_recompute_count, 0);
        assert_eq!(stats.incremental_update_count, 2);
        assert_eq!(stats.update_count, 2);

        // A3: bounded frontier — delta push should not touch isolated nodes.
        assert!(
            stats.last_touched_nodes < 10,
            "delta frontier should avoid touching isolated nodes"
        );
    }

    // ---- A4: insert-then-delete round-trip -----------------------------------

    #[test]
    fn insert_then_delete_round_trip_returns_to_epsilon() {
        let params = default_params();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let mut maintainer = DynamicPprMaintainer::new(fixture_adjacency(), seeds.clone(), params);
        let pre_insertion = maintainer.scores().clone();

        // Insert a novel edge, then delete it.
        maintainer.apply_update(DynamicPprEdgeUpdate::insert("c", "z", 0.5));
        maintainer.apply_update(DynamicPprEdgeUpdate::delete("c", "z"));

        assert!(ppr_vectors_within_epsilon(
            maintainer.scores(),
            &pre_insertion,
            10.0 * params.epsilon
        ));
    }

    #[test]
    fn incremental_seed_replacement_matches_full_reference() {
        let params = default_params();
        let adjacency = HashMap::from([
            ("a".to_string(), vec![("b".to_string(), 1.0)]),
            ("b".to_string(), vec![("c".to_string(), 1.0)]),
            ("c".to_string(), Vec::new()),
        ]);
        let initial = HashMap::from([("a".to_string(), 1.0)]);
        let replacement = HashMap::from([
            ("a".to_string(), 1.0 / 3.0),
            ("b".to_string(), 1.0 / 3.0),
            ("c".to_string(), 1.0 / 3.0),
        ]);
        let mut maintainer = DynamicPprMaintainer::new(adjacency.clone(), initial, params);

        maintainer.replace_seeds_incremental(replacement.clone());

        let expected = dynamic_ppr_full_reference(&adjacency, &replacement, params);
        assert!(ppr_vectors_within_epsilon(
            maintainer.scores(),
            &expected,
            10.0 * params.epsilon
        ));
        assert_eq!(maintainer.stats().full_recompute_count, 1);
        assert_eq!(maintainer.stats().incremental_update_count, 1);
    }

    #[test]
    fn first_outgoing_edges_from_dangling_nodes_match_full_reference() {
        let params = default_params();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let mut maintainer = DynamicPprMaintainer::new(HashMap::new(), seeds.clone(), params);
        maintainer.apply_update(DynamicPprEdgeUpdate::insert("a", "b", 1.0));
        maintainer.apply_update(DynamicPprEdgeUpdate::insert("b", "c", 1.0));

        let adjacency = HashMap::from([
            ("a".to_string(), vec![("b".to_string(), 1.0)]),
            ("b".to_string(), vec![("c".to_string(), 1.0)]),
        ]);
        let expected = dynamic_ppr_full_reference(&adjacency, &seeds, params);
        assert!(ppr_vectors_within_epsilon(
            maintainer.scores(),
            &expected,
            10.0 * params.epsilon
        ));
        assert_eq!(maintainer.stats().full_recompute_count, 1);
        assert_eq!(maintainer.stats().incremental_update_count, 2);
    }

    // ---- A5: refresh_full resets to reference and bumps the counter ----------

    #[test]
    fn refresh_full_resets_to_reference() {
        let params = default_params();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let mut maintainer = DynamicPprMaintainer::new(fixture_adjacency(), seeds.clone(), params);
        maintainer.reset_full_recompute_counter();

        // Apply several incremental deltas.
        maintainer.apply_updates([
            DynamicPprEdgeUpdate::insert("b", "x", 1.0),
            DynamicPprEdgeUpdate::insert("c", "y", 1.0),
            DynamicPprEdgeUpdate::delete("c", "d"),
        ]);
        assert_eq!(maintainer.stats().full_recompute_count, 0);
        assert_eq!(maintainer.stats().incremental_update_count, 3);

        // Now refresh full.
        let stats = maintainer.refresh_full();

        // After refresh the scores must match a reference from scratch.
        let expected = dynamic_ppr_full_reference(&maintainer.adjacency, &seeds, params);
        assert!(ppr_vectors_within_epsilon(
            maintainer.scores(),
            &expected,
            params.epsilon
        ));
        assert_eq!(stats.full_recompute_count, 1);
    }

    // ---- D7 smoke: deletion with dangling residual ---------------------------

    #[test]
    fn delete_all_out_edges_and_recover() {
        let mut adjacency = HashMap::new();
        adjacency.insert("a".to_string(), vec![("b".to_string(), 1.0)]);
        adjacency.insert("b".to_string(), Vec::new());

        let params = default_params();
        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let mut maintainer = DynamicPprMaintainer::new(adjacency.clone(), seeds.clone(), params);

        // Delete the only outgoing edge from "a".
        maintainer.apply_update(DynamicPprEdgeUpdate::delete("a", "b"));

        let mut expected_adjacency = adjacency;
        apply_adjacency_update(
            &mut expected_adjacency,
            DynamicPprEdgeUpdate::delete("a", "b"),
        );
        let expected = dynamic_ppr_full_reference(&expected_adjacency, &seeds, params);

        assert!(ppr_vectors_within_epsilon(
            maintainer.scores(),
            &expected,
            10.0 * params.epsilon
        ));
        // Full recompute counter should be 0 even on hard deletion
        // (the bounded fallback should not have tripped a full refresh).
        assert_eq!(maintainer.stats().full_recompute_count, 1); // 1 from ctor
    }

    // ---- A6: verify stats present the new incremental_update_count field -----

    #[test]
    fn stats_default_has_incremental_update_field() {
        let stats = DynamicPprStats::default();
        assert_eq!(stats.incremental_update_count, 0);
    }
}
