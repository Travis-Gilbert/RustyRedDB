//! Criticism-survival ranking signals (SPEC-CRITICISM-SURVIVAL).
//!
//! Two quantities, not one:
//! - Acceptability Acc(c): weighted h-categoriser over the attack graph.
//! - Test burden Burden(c): saturating count of survived severe criticism.
//!
//! Exported signal: CriticismSurvival(c) = Acc(c) * (1 - exp(-λ * Burden(c))).
//! An untested claim scores zero regardless of source pedigree.
//!
//! Attack edges already exist in RustyRed as `undercuts` / `contradicts` /
//! `attacks` / `ATTACKS` / `tension` (see `epistemic.rs` and `graph_store.rs`).
//! [`AttackGraph`] is the slot-indexed CSR view those edge types project into.

use crate::node_state::{NodeStateBlock, NodeStateChannel};

/// Edge types that count as attacks when projecting a store graph.
pub const ATTACK_EDGE_TYPES: &[&str] =
    &["undercuts", "contradicts", "attacks", "ATTACKS", "tension"];

/// Bidirectional attack adjacency over dense claim slots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttackGraph {
    /// attacker slot → targets
    pub attacks: Vec<Vec<u32>>,
    /// target slot → attackers
    pub attacked_by: Vec<Vec<u32>>,
}

impl AttackGraph {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            attacks: vec![Vec::new(); n],
            attacked_by: vec![Vec::new(); n],
        }
    }

    pub fn len(&self) -> usize {
        self.attacks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attacks.is_empty()
    }

    /// Insert a directed attack `attacker → target`. Idempotent per pair.
    pub fn add_attack(&mut self, attacker: u32, target: u32) {
        let a = attacker as usize;
        let t = target as usize;
        if a >= self.attacks.len() || t >= self.attacked_by.len() {
            return;
        }
        if !self.attacks[a].contains(&target) {
            self.attacks[a].push(target);
        }
        if !self.attacked_by[t].contains(&attacker) {
            self.attacked_by[t].push(attacker);
        }
    }

    /// Build from `(attacker, target)` pairs. `n` is the slot count.
    pub fn from_pairs(n: usize, pairs: &[(u32, u32)]) -> Self {
        let mut graph = Self::with_capacity(n);
        for &(attacker, target) in pairs {
            graph.add_attack(attacker, target);
        }
        graph
    }

    pub fn is_attack_edge_type(edge_type: &str) -> bool {
        ATTACK_EDGE_TYPES
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(edge_type))
            || edge_type.eq_ignore_ascii_case("ATTACKS")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorroborationParams {
    /// θ: Acc below this counts as eliminated for severity discrimination.
    pub severity_threshold: f64,
    /// λ: saturation rate for CriticismSurvival.
    pub burden_lambda: f64,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for CorroborationParams {
    fn default() -> Self {
        Self {
            severity_threshold: 0.5,
            burden_lambda: 0.35,
            max_iter: 100,
            tolerance: 1e-9,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CorroborationStats {
    pub iterations: usize,
    pub converged: bool,
    pub max_delta: f64,
}

/// Joint fixed point over acceptability and severity.
///
/// Writes `Acc` into channel `Corroboration` and `Pressure` into channel
/// `CriticismPressure`. Severity (and Acc in the pressure sum) are read from
/// the previous iterate (Jacobi), matching the pagerank-style loop in
/// `CsrGraph::pagerank`.
pub fn solve_acceptability(
    graph: &AttackGraph,
    state: &mut NodeStateBlock,
    live: &[bool],
    params: CorroborationParams,
) -> CorroborationStats {
    let n = graph.len();
    if state.len() < n {
        state.resize(n);
    }
    // Unattacked default: Acc = 1.0, Pressure = 0.0.
    state.fill(NodeStateChannel::Corroboration, 1.0);
    state.fill(NodeStateChannel::CriticismPressure, 0.0);

    let mut prev_acc = vec![1.0; n];
    let mut stats = CorroborationStats::default();

    for iter in 0..params.max_iter {
        let mut max_delta = 0.0f64;
        let mut next_acc = prev_acc.clone();
        let mut next_pressure = vec![0.0; n];

        for slot in 0..n {
            if live.get(slot).copied().unwrap_or(true) == false {
                continue;
            }
            let mut pressure = 0.0;
            for &attacker in &graph.attacked_by[slot] {
                let severity = attack_severity(graph, &prev_acc, attacker, slot as u32, params);
                let attacker_acc = prev_acc
                    .get(attacker as usize)
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                pressure += attacker_acc * severity;
            }
            let acc = 1.0 / (1.0 + pressure);
            max_delta = max_delta.max((acc - prev_acc[slot]).abs());
            next_acc[slot] = acc;
            next_pressure[slot] = pressure;
        }

        prev_acc = next_acc;
        for slot in 0..n {
            state.set(NodeStateChannel::Corroboration, slot as u32, prev_acc[slot]);
            state.set(
                NodeStateChannel::CriticismPressure,
                slot as u32,
                next_pressure[slot],
            );
        }

        stats.iterations = iter + 1;
        stats.max_delta = max_delta;
        if max_delta < params.tolerance {
            stats.converged = true;
            break;
        }
    }

    stats
}

/// Severity proxy: fraction of the attacker's *other* targets that are
/// eliminated (Acc < θ). When the attacker has only one target, discrimination
/// is undefined and the attack is treated as full-strength (1.0), which is
/// what makes an unrefuted singleton attacker drive Acc(target) → 0.5.
pub fn attack_severity(
    graph: &AttackGraph,
    acc: &[f64],
    attacker: u32,
    target: u32,
    params: CorroborationParams,
) -> f64 {
    let targets = graph
        .attacks
        .get(attacker as usize)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if targets.len() <= 1 {
        return 1.0;
    }
    let eliminated = targets
        .iter()
        .filter(|&&t| t != target)
        .filter(|&&t| acc.get(t as usize).copied().unwrap_or(1.0) < params.severity_threshold)
        .count();
    eliminated as f64 / (targets.len() - 1) as f64
}

/// Requires [`solve_acceptability`] to have run.
pub fn test_burden(
    graph: &AttackGraph,
    state: &NodeStateBlock,
    slot: u32,
    params: CorroborationParams,
) -> f64 {
    let idx = slot as usize;
    let Some(attackers) = graph.attacked_by.get(idx) else {
        return 0.0;
    };
    let acc = state.channel(NodeStateChannel::Corroboration);
    let mut burden = 0.0;
    for &attacker in attackers {
        let severity = attack_severity(graph, acc, attacker, slot, params);
        let attacker_acc = state
            .get(NodeStateChannel::Corroboration, attacker)
            .clamp(0.0, 1.0);
        burden += severity * (1.0 - attacker_acc);
    }
    burden
}

/// Requires [`solve_acceptability`] to have run.
pub fn criticism_survival(
    graph: &AttackGraph,
    state: &NodeStateBlock,
    slot: u32,
    params: CorroborationParams,
) -> f64 {
    let acc = state
        .get(NodeStateChannel::Corroboration, slot)
        .clamp(0.0, 1.0);
    let burden = test_burden(graph, state, slot, params);
    acc * (1.0 - (-params.burden_lambda * burden).exp())
}

/// Spearman rank correlation between two equal-length score vectors.
/// Returns NaN when either side is constant (undefined rank correlation).
pub fn spearman_rho(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    let n = left.len();
    if n < 2 {
        return f64::NAN;
    }
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    pearson(&left_ranks, &right_ranks)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // 1-based average rank over the tied window.
        let avg = ((i + 1) + j) as f64 / 2.0;
        for &idx in &order[i..j] {
            ranks[idx] = avg;
        }
        i = j;
    }
    ranks
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len() as f64;
    let mean_l = left.iter().sum::<f64>() / n;
    let mean_r = right.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_l = 0.0;
    let mut den_r = 0.0;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let dl = l - mean_l;
        let dr = r - mean_r;
        num += dl * dr;
        den_l += dl * dl;
        den_r += dr * dr;
    }
    let den = (den_l * den_r).sqrt();
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn solve(
        pairs: &[(u32, u32)],
        n: usize,
    ) -> (
        AttackGraph,
        NodeStateBlock,
        CorroborationStats,
        CorroborationParams,
    ) {
        let graph = AttackGraph::from_pairs(n, pairs);
        let mut state = NodeStateBlock::new(n);
        let live = vec![true; n];
        let params = CorroborationParams::default();
        let stats = solve_acceptability(&graph, &mut state, &live, params);
        (graph, state, stats, params)
    }

    #[test]
    fn unattacked_claim_is_standing_untested() {
        let (graph, state, stats, params) = solve(&[], 1);
        assert!(stats.converged);
        assert!((state.get(NodeStateChannel::Corroboration, 0) - 1.0).abs() < 1e-9);
        assert!((test_burden(&graph, &state, 0, params) - 0.0).abs() < 1e-9);
        assert!((criticism_survival(&graph, &state, 0, params) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn single_unrefuted_full_strength_attacker_halves_acceptability() {
        // attacker 0 → target 1
        let (graph, state, stats, _) = solve(&[(0, 1)], 2);
        assert!(stats.converged);
        assert!((state.get(NodeStateChannel::Corroboration, 0) - 1.0).abs() < 1e-9);
        assert!((state.get(NodeStateChannel::Corroboration, 1) - 0.5).abs() < 1e-9);
        let _ = graph;
    }

    #[test]
    fn refuted_attacker_recovers_downstream_claim() {
        // x=0 attacks y=1; y attacks z=2; nothing attacks x.
        let (_, state, stats, _) = solve(&[(0, 1), (1, 2)], 3);
        assert!(stats.converged);
        let acc_y = state.get(NodeStateChannel::Corroboration, 1);
        let acc_z = state.get(NodeStateChannel::Corroboration, 2);
        assert!(
            acc_z > acc_y,
            "Acc(z)={acc_z} should exceed Acc(y)={acc_y} when y is itself under attack"
        );
    }

    #[test]
    fn severity_discrimination_eliminated_vs_standing() {
        let params = CorroborationParams::default();
        // a=0 targets {c=1, t1=2, t2=3, t3=4}. Drive Acc(t*) well below θ=0.5
        // with multiple unrefuted critics each (strict < θ, not ≤).
        let mut pairs = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let mut next = 10u32;
        for &target in &[2u32, 3, 4] {
            for _ in 0..4 {
                pairs.push((next, target));
                next += 1;
            }
        }
        let (graph, state, _, _) = solve(&pairs, next as usize);
        let acc = state.channel(NodeStateChannel::Corroboration).to_vec();
        assert!(
            acc[2] < params.severity_threshold
                && acc[3] < params.severity_threshold
                && acc[4] < params.severity_threshold,
            "fixture must refute other targets below θ: {:?}",
            &acc[2..=4]
        );
        let sev = attack_severity(&graph, &acc, 0, 1, params);
        assert!(
            (sev - 1.0).abs() < 1e-9,
            "severity should be 1.0 when other targets are refuted, got {sev}"
        );

        // Second fixture: other targets stand (no attacks on them).
        let standing_pairs = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let (graph2, state2, _, _) = solve(&standing_pairs, 5);
        let acc2 = state2.channel(NodeStateChannel::Corroboration).to_vec();
        let sev2 = attack_severity(&graph2, &acc2, 0, 1, params);
        assert!(
            (sev2 - 0.0).abs() < 1e-9,
            "severity should be 0.0 when other targets stand, got {sev2}"
        );
    }

    #[test]
    fn burden_saturation_is_monotone_with_diminishing_increments() {
        let params = CorroborationParams::default();
        // Criticisms that have been *thoroughly* defeated: Acc(attacker) ≈ 0 so
        // they add burden without collapsing Acc(claim). Mild half-refutations
        // still press Acc(c) and can make Survival non-monotone.
        let mut survivals = Vec::new();
        for k in 0..=10 {
            let mut pairs = Vec::new();
            let mut next = 100u32;
            for i in 0..k {
                let attacker = (i + 1) as u32;
                pairs.push((attacker, 0));
                // Many standing critics → Acc(attacker) ≈ 1/(1+n) ≈ 0 so
                // residual pressure does not overturn Acc(claim) for k ≤ 10.
                for _ in 0..100 {
                    pairs.push((next, attacker));
                    next += 1;
                }
            }
            let n = next as usize;
            let (graph, state, _, _) = solve(&pairs, n);
            if k > 0 {
                let attacker_acc = state.get(NodeStateChannel::Corroboration, 1);
                assert!(
                    attacker_acc < 0.02,
                    "attacker must be thoroughly refuted, Acc={attacker_acc}"
                );
            }
            survivals.push(criticism_survival(&graph, &state, 0, params));
        }
        for window in survivals.windows(2) {
            assert!(
                window[1] + 1e-12 >= window[0],
                "CriticismSurvival must be monotone non-decreasing: {:?}",
                survivals
            );
        }
        let inc_1_2 = survivals[2] - survivals[1];
        let inc_9_10 = survivals[10] - survivals[9];
        assert!(
            inc_9_10 < inc_1_2,
            "increment 9→10 ({inc_9_10}) must be < increment 1→2 ({inc_1_2})"
        );
    }

    #[test]
    fn corroboration_length_neutrality_vs_viterbi_decay() {
        use crate::provenance_semiring::{
            bounded_recursive_semiring_derivation, ProvenanceSemiringKind, RecursiveSemiringEdge,
        };
        use std::collections::BTreeMap;

        let five_chain = vec![
            RecursiveSemiringEdge {
                from_id: "s".into(),
                to_id: "a".into(),
                value: 0.9,
            },
            RecursiveSemiringEdge {
                from_id: "a".into(),
                to_id: "b".into(),
                value: 0.9,
            },
            RecursiveSemiringEdge {
                from_id: "b".into(),
                to_id: "c".into(),
                value: 0.9,
            },
            RecursiveSemiringEdge {
                from_id: "c".into(),
                to_id: "d".into(),
                value: 0.9,
            },
            RecursiveSemiringEdge {
                from_id: "d".into(),
                to_id: "long".into(),
                value: 0.9,
            },
        ];
        let one_edge = vec![RecursiveSemiringEdge {
            from_id: "s".into(),
            to_id: "short".into(),
            value: 0.9,
        }];
        let mut edges = five_chain;
        edges.extend(one_edge);
        let seeds = BTreeMap::from([("s".to_string(), 1.0)]);

        let corr = bounded_recursive_semiring_derivation(
            &edges,
            &seeds,
            ProvenanceSemiringKind::Corroboration,
            8,
        )
        .expect("corroboration fold");
        assert!(
            (corr.scores["long"] - corr.scores["short"]).abs() < 1e-9,
            "max-min must be length-neutral: long={} short={}",
            corr.scores["long"],
            corr.scores["short"]
        );
        assert!((corr.scores["long"] - 0.9).abs() < 1e-9);

        let vit = bounded_recursive_semiring_derivation(
            &edges,
            &seeds,
            ProvenanceSemiringKind::Viterbi,
            8,
        )
        .expect("viterbi fold");
        assert!(
            vit.scores["long"] < vit.scores["short"] - 1e-9,
            "viterbi must penalize chain length: long={} short={}",
            vit.scores["long"],
            vit.scores["short"]
        );
    }

    #[test]
    fn mutual_attack_cycle_converges_to_equal_scores() {
        let (graph, state, stats, _) = solve(&[(0, 1), (1, 0)], 2);
        assert!(stats.converged, "cycle must converge within max_iter");
        let a = state.get(NodeStateChannel::Corroboration, 0);
        let b = state.get(NodeStateChannel::Corroboration, 1);
        assert!(
            (a - b).abs() < 1e-9,
            "mutual attackers must receive equal Acc: {a} vs {b}"
        );
        let _ = graph;
    }

    #[test]
    fn kill_criterion_criticism_survival_not_redundant_with_source_prior() {
        // Synthetic corpus: deliberately anti-align source prior with survival
        // so the signal is informative. High prior + never tested vs low prior
        // + many survived attacks.
        const N: usize = 220;
        let mut pairs = Vec::new();
        let mut prior = vec![0.5; N];

        // Claims 0..99: high pedigree, zero attacks (survival = 0).
        for i in 0..100 {
            prior[i] = 0.85 + (i as f64) * 0.001;
        }
        // Claims 100..199: weak pedigree, each survives a thoroughly refuted attacker.
        let mut next_slot = 220u32;
        for i in 100..200 {
            prior[i] = 0.15 + ((i - 100) as f64) * 0.001;
            let attacker = next_slot;
            next_slot += 1;
            pairs.push((attacker, i as u32));
            for _ in 0..40 {
                pairs.push((next_slot, attacker));
                next_slot += 1;
            }
        }
        // Claims 200..219: mid prior, mixed mild attacks (noise).
        for i in 200..220 {
            prior[i] = 0.5;
        }

        let slot_n = next_slot as usize;
        let graph = AttackGraph::from_pairs(slot_n, &pairs);
        let mut state = NodeStateBlock::new(slot_n);
        let live = vec![true; slot_n];
        let params = CorroborationParams::default();
        let stats = solve_acceptability(&graph, &mut state, &live, params);
        assert!(stats.converged);

        let mut survival = Vec::with_capacity(N);
        for slot in 0..N {
            survival.push(criticism_survival(&graph, &state, slot as u32, params));
        }
        let prior_slice = &prior[..N];
        let rho = spearman_rho(prior_slice, &survival);
        assert!(
            rho.is_finite() && rho.abs() <= 0.9,
            "kill criterion: |rho|={rho} > 0.9 means CriticismSurvival is redundant with the source prior"
        );

        // Rank disagreement: largest |rank_prior - rank_survival|.
        let prior_ranks = average_ranks(prior_slice);
        let surv_ranks = average_ranks(&survival);
        let mut disagreements: Vec<(usize, f64, f64, f64, f64)> = (0..N)
            .map(|i| (i, prior[i], survival[i], prior_ranks[i], surv_ranks[i]))
            .collect();
        disagreements.sort_by(|a, b| {
            let da = (a.3 - a.4).abs();
            let db = (b.3 - b.4).abs();
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top20 = &disagreements[..20];

        let mut fixture = String::from(
            "# CriticismSurvival vs justification_confidence_prior rank disagreements\n",
        );
        fixture.push_str(&format!("# spearman_rho={rho:.6}\n"));
        fixture.push_str(
            "claim_slot,justification_confidence_prior,criticism_survival,prior_rank,survival_rank,rank_delta\n",
        );
        for (slot, p, s, pr, sr) in top20 {
            fixture.push_str(&format!(
                "{slot},{p:.6},{s:.6},{pr:.1},{sr:.1},{:.1}\n",
                (pr - sr).abs()
            ));
        }

        let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("criticism_survival_rank_disagreements.csv");
        fs::create_dir_all(out.parent().expect("fixtures dir")).expect("mkdir fixtures");
        fs::write(&out, &fixture).expect("write disagreement fixture");
        let written = fs::read_to_string(&out).expect("read fixture");
        assert!(written.contains("spearman_rho="));
        assert_eq!(written.lines().count(), 23); // header comment*2 + csv header + 20 rows
    }
}
