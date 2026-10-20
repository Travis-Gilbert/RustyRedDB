use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSemiringKind {
    /// Max-times (GraphBLAS `max_times_fp64`).
    ///
    /// MCP `fold_semiring` historically advertised this algebra as
    /// `"probability"`, but the derivation-crate `ProbabilitySemiring` is a
    /// different noisy-OR union. Accept the alias on this enum so inbound JSON
    /// that still says `probability` maps here, and prefer the name `viterbi`
    /// in new call sites. See `parse_name`.
    #[default]
    #[serde(alias = "probability")]
    Viterbi,
    Tropical,
    Boolean,
    Counting,
    /// Max-min bottleneck fold: sum = max, product = min on `[0, 1]`.
    /// Length-neutral derivational support (unlike Viterbi max-times).
    Corroboration,
    /// Noisy-OR probability union (`a + b - a·b`). Distinct from Viterbi.
    /// Wire: `probability_union`. Legacy `probability` still aliases Viterbi.
    #[serde(rename = "probability_union")]
    Probability,
}

impl ProvenanceSemiringKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Viterbi => "viterbi",
            Self::Tropical => "tropical",
            Self::Boolean => "boolean",
            Self::Counting => "counting",
            Self::Corroboration => "corroboration",
            Self::Probability => "probability",
        }
    }

    /// Parse a wire/MCP name. `probability` aliases to [`Self::Viterbi`]
    /// (max-times). The derivation-fold noisy-OR algebra keeps the name
    /// `probability` on that separate surface; do not conflate the two.
    pub fn parse_name(name: &str) -> Option<Self> {
        match name {
            "viterbi" | "probability" => Some(Self::Viterbi),
            "tropical" => Some(Self::Tropical),
            "boolean" => Some(Self::Boolean),
            "counting" | "count" => Some(Self::Counting),
            "corroboration" => Some(Self::Corroboration),
            "probability_union" => Some(Self::Probability),
            _ => None,
        }
    }

    pub fn graphblas_semiring(self) -> Option<&'static str> {
        match self {
            Self::Viterbi => Some("max_times_fp64"),
            Self::Tropical => Some("min_plus_fp64"),
            Self::Boolean => Some("lor_land_bool"),
            Self::Counting => Some("plus_times_fp64"),
            Self::Corroboration => Some("max_min_fp64"),
            Self::Probability => None,
        }
    }

    /// True when this semiring is safe for recursive derivation given a
    /// `distinct` step at each iteration. Counting qualifies; the absorptive
    /// semirings qualify without needing one.
    pub fn recursion_safe_with_distinct(self) -> bool {
        matches!(
            self,
            Self::Viterbi
                | Self::Tropical
                | Self::Boolean
                | Self::Counting
                | Self::Corroboration
                | Self::Probability
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProvenanceSemiring {
    pub kind: ProvenanceSemiringKind,
}

impl ProvenanceSemiring {
    pub fn new(kind: ProvenanceSemiringKind) -> Self {
        Self { kind }
    }

    pub fn zero(self) -> f64 {
        match self.kind {
            ProvenanceSemiringKind::Viterbi => 0.0,
            ProvenanceSemiringKind::Tropical => f64::INFINITY,
            ProvenanceSemiringKind::Boolean => 0.0,
            ProvenanceSemiringKind::Counting => 0.0,
            ProvenanceSemiringKind::Corroboration => 0.0,
            ProvenanceSemiringKind::Probability => 0.0,
        }
    }

    pub fn one(self) -> f64 {
        match self.kind {
            ProvenanceSemiringKind::Viterbi => 1.0,
            ProvenanceSemiringKind::Tropical => 0.0,
            ProvenanceSemiringKind::Boolean => 1.0,
            ProvenanceSemiringKind::Counting => 1.0,
            ProvenanceSemiringKind::Corroboration => 1.0,
            ProvenanceSemiringKind::Probability => 1.0,
        }
    }

    pub fn sum(self, left: f64, right: f64) -> f64 {
        match self.kind {
            ProvenanceSemiringKind::Viterbi => left.max(right).clamp(0.0, 1.0),
            ProvenanceSemiringKind::Tropical => left.min(right),
            ProvenanceSemiringKind::Boolean => bool_value(left != 0.0 || right != 0.0),
            ProvenanceSemiringKind::Counting => left + right,
            ProvenanceSemiringKind::Corroboration => left.max(right).clamp(0.0, 1.0),
            ProvenanceSemiringKind::Probability => (left + right - left * right).clamp(0.0, 1.0),
        }
    }

    pub fn product(self, left: f64, right: f64) -> f64 {
        match self.kind {
            ProvenanceSemiringKind::Viterbi => {
                (left.clamp(0.0, 1.0) * right.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            }
            ProvenanceSemiringKind::Tropical => left + right,
            ProvenanceSemiringKind::Boolean => bool_value(left != 0.0 && right != 0.0),
            ProvenanceSemiringKind::Counting => left * right,
            ProvenanceSemiringKind::Corroboration => {
                left.clamp(0.0, 1.0).min(right.clamp(0.0, 1.0))
            }
            ProvenanceSemiringKind::Probability => {
                (left.clamp(0.0, 1.0) * right.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            }
        }
    }

    pub fn is_absorptive(self) -> bool {
        matches!(
            self.kind,
            ProvenanceSemiringKind::Viterbi
                | ProvenanceSemiringKind::Tropical
                | ProvenanceSemiringKind::Corroboration
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SemiringAnnotation {
    pub semiring: ProvenanceSemiringKind,
    pub value: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecursiveSemiringEdge {
    pub from_id: String,
    pub to_id: String,
    pub value: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecursiveSemiringResult {
    pub semiring: ProvenanceSemiringKind,
    pub scores: BTreeMap<String, f64>,
    pub max_depth: usize,
    pub expansions: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecursiveSemiringError {
    pub code: String,
    pub message: String,
}

impl RecursiveSemiringError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

pub fn bounded_recursive_semiring_derivation(
    edges: &[RecursiveSemiringEdge],
    seeds: &BTreeMap<String, f64>,
    kind: ProvenanceSemiringKind,
    max_depth: usize,
) -> Result<RecursiveSemiringResult, RecursiveSemiringError> {
    let semiring = ProvenanceSemiring::new(kind);
    if !semiring.is_absorptive() {
        return Err(RecursiveSemiringError::new(
            "recursive_semiring_not_absorptive",
            format!(
                "recursive derivation supports viterbi, tropical, and corroboration only; {} would require unbounded provenance polynomial expansion",
                kind.name()
            ),
        ));
    }
    let adjacency = edges.iter().fold(
        BTreeMap::<String, Vec<&RecursiveSemiringEdge>>::new(),
        |mut map, edge| {
            map.entry(edge.from_id.clone()).or_default().push(edge);
            map
        },
    );
    let mut scores = BTreeMap::new();
    let mut queue = VecDeque::new();
    for (seed, value) in seeds {
        let value = normalize_recursive_seed(kind, *value);
        scores.insert(seed.clone(), semiring.sum(semiring.zero(), value));
        queue.push_back((seed.clone(), value, 0usize));
    }
    let mut expansions = 0usize;
    while let Some((node, value, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let Some(out_edges) = adjacency.get(&node) else {
            continue;
        };
        for edge in out_edges {
            expansions += 1;
            let next = semiring.product(value, edge.value);
            let previous = scores
                .get(&edge.to_id)
                .copied()
                .unwrap_or_else(|| semiring.zero());
            let joined = semiring.sum(previous, next);
            if recursive_score_improved(kind, previous, joined) {
                scores.insert(edge.to_id.clone(), joined);
                queue.push_back((edge.to_id.clone(), joined, depth + 1));
            }
        }
    }
    Ok(RecursiveSemiringResult {
        semiring: kind,
        scores,
        max_depth,
        expansions,
    })
}

fn bool_value(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn normalize_recursive_seed(kind: ProvenanceSemiringKind, value: f64) -> f64 {
    match kind {
        ProvenanceSemiringKind::Viterbi => value.clamp(0.0, 1.0),
        ProvenanceSemiringKind::Tropical => value.max(0.0),
        ProvenanceSemiringKind::Corroboration => value.clamp(0.0, 1.0),
        _ => value,
    }
}

fn recursive_score_improved(kind: ProvenanceSemiringKind, previous: f64, next: f64) -> bool {
    match kind {
        ProvenanceSemiringKind::Viterbi => next > previous + 1e-12,
        ProvenanceSemiringKind::Tropical => next + 1e-12 < previous,
        ProvenanceSemiringKind::Corroboration => next > previous + 1e-12,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) -> bool {
        if left.is_infinite() || right.is_infinite() {
            return left == right;
        }
        (left - right).abs() < 1e-9
    }

    fn law_values(kind: ProvenanceSemiringKind) -> Vec<f64> {
        match kind {
            ProvenanceSemiringKind::Viterbi => vec![0.2, 0.5, 0.9],
            ProvenanceSemiringKind::Tropical => vec![1.0, 2.0, 5.0],
            ProvenanceSemiringKind::Boolean => vec![bool_value(false), bool_value(true)],
            ProvenanceSemiringKind::Counting => vec![1.0, 2.0, 3.0],
            ProvenanceSemiringKind::Corroboration => vec![0.2, 0.5, 0.9],
            ProvenanceSemiringKind::Probability => vec![0.2, 0.5, 0.9],
        }
    }

    fn assert_laws(kind: ProvenanceSemiringKind) {
        let semiring = ProvenanceSemiring::new(kind);
        let values = law_values(kind);
        for &a in &values {
            assert!(close(semiring.sum(a, semiring.zero()), a));
            assert!(close(semiring.sum(semiring.zero(), a), a));
            assert!(close(semiring.product(a, semiring.one()), a));
            assert!(close(semiring.product(semiring.one(), a), a));
            for &b in &values {
                for &c in &values {
                    assert!(close(
                        semiring.sum(semiring.sum(a, b), c),
                        semiring.sum(a, semiring.sum(b, c))
                    ));
                    assert!(close(
                        semiring.product(semiring.product(a, b), c),
                        semiring.product(a, semiring.product(b, c))
                    ));
                }
            }
            if kind == ProvenanceSemiringKind::Corroboration {
                // Absorption: max(a, min(a, b)) = a
                for &b in &values {
                    assert!(close(semiring.sum(a, semiring.product(a, b)), a));
                }
            }
        }
    }

    #[test]
    fn semirings_satisfy_identity_and_associativity_laws() {
        for kind in [
            ProvenanceSemiringKind::Viterbi,
            ProvenanceSemiringKind::Tropical,
            ProvenanceSemiringKind::Boolean,
            ProvenanceSemiringKind::Counting,
            ProvenanceSemiringKind::Corroboration,
        ] {
            assert_laws(kind);
            assert!(kind.graphblas_semiring().is_some());
        }
    }

    #[test]
    fn probability_aliases_to_viterbi_on_provenance_surface() {
        assert_eq!(
            ProvenanceSemiringKind::parse_name("probability"),
            Some(ProvenanceSemiringKind::Viterbi)
        );
        assert_eq!(
            ProvenanceSemiringKind::parse_name("viterbi"),
            Some(ProvenanceSemiringKind::Viterbi)
        );
        let aliased: ProvenanceSemiringKind =
            serde_json::from_str("\"probability\"").expect("alias");
        assert_eq!(aliased, ProvenanceSemiringKind::Viterbi);
        assert_eq!(
            ProvenanceSemiringKind::parse_name("corroboration"),
            Some(ProvenanceSemiringKind::Corroboration)
        );
    }

    #[test]
    fn required_recursive_semirings_are_absorptive() {
        assert!(ProvenanceSemiring::new(ProvenanceSemiringKind::Viterbi).is_absorptive());
        assert!(ProvenanceSemiring::new(ProvenanceSemiringKind::Tropical).is_absorptive());
        assert!(ProvenanceSemiring::new(ProvenanceSemiringKind::Corroboration).is_absorptive());
        assert!(!ProvenanceSemiring::new(ProvenanceSemiringKind::Counting).is_absorptive());
    }

    #[test]
    fn counting_is_recursion_safe_with_distinct_but_not_absorptive() {
        assert!(ProvenanceSemiringKind::Counting.recursion_safe_with_distinct());
        assert!(!ProvenanceSemiring::new(ProvenanceSemiringKind::Counting).is_absorptive());
        for kind in [
            ProvenanceSemiringKind::Viterbi,
            ProvenanceSemiringKind::Tropical,
            ProvenanceSemiringKind::Boolean,
            ProvenanceSemiringKind::Corroboration,
        ] {
            assert!(kind.recursion_safe_with_distinct());
        }
    }

    #[test]
    fn bounded_recursive_derivation_uses_viterbi_max_and_tropical_min() {
        let confidence_edges = vec![
            RecursiveSemiringEdge {
                from_id: "a".to_string(),
                to_id: "b".to_string(),
                value: 0.8,
            },
            RecursiveSemiringEdge {
                from_id: "b".to_string(),
                to_id: "c".to_string(),
                value: 0.7,
            },
            RecursiveSemiringEdge {
                from_id: "a".to_string(),
                to_id: "c".to_string(),
                value: 0.4,
            },
            RecursiveSemiringEdge {
                from_id: "b".to_string(),
                to_id: "a".to_string(),
                value: 0.1,
            },
        ];
        let seeds = BTreeMap::from([("a".to_string(), 1.0)]);
        let viterbi = bounded_recursive_semiring_derivation(
            &confidence_edges,
            &seeds,
            ProvenanceSemiringKind::Viterbi,
            4,
        )
        .expect("viterbi recursion");
        assert!((viterbi.scores["c"] - 0.56).abs() < 1e-9);

        let cost_edges = vec![
            RecursiveSemiringEdge {
                from_id: "a".to_string(),
                to_id: "b".to_string(),
                value: 1.0,
            },
            RecursiveSemiringEdge {
                from_id: "b".to_string(),
                to_id: "c".to_string(),
                value: 1.0,
            },
            RecursiveSemiringEdge {
                from_id: "a".to_string(),
                to_id: "c".to_string(),
                value: 5.0,
            },
        ];
        let tropical = bounded_recursive_semiring_derivation(
            &cost_edges,
            &BTreeMap::from([("a".to_string(), 0.0)]),
            ProvenanceSemiringKind::Tropical,
            4,
        )
        .expect("tropical recursion");
        assert!((tropical.scores["c"] - 2.0).abs() < 1e-9);

        let err = bounded_recursive_semiring_derivation(
            &confidence_edges,
            &seeds,
            ProvenanceSemiringKind::Counting,
            4,
        )
        .expect_err("counting recursion rejected");
        assert_eq!(err.code, "recursive_semiring_not_absorptive");
        assert!(err.message.contains("provenance polynomial"));
    }
}
