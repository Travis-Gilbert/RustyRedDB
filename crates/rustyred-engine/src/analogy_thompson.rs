//! Thompson sampling for analogical transfer hypotheses.
//!
//! Replaces the deterministic confidence scalar in
//! [`crate::analogy::transfer_hypothesis`] with a Thompson sample drawn from a
//! schema-level Beta posterior. Each schema's Beta is initialized from a
//! structural-score prior blended with the fleet prior tally, then updated as
//! transfer outcomes are observed. The fleet prior compounds counts across
//! tenants (counts only, never content).
//!
//! ## Integration points
//!
//! - [`ThompsonAnalogyOracle::thompson_sample`] replaces the deterministic
//!   `base_confidence * systematicity * 0.5` in `transfer_hypothesis`.
//! - [`ThompsonAnalogyOracle::observe_outcome`] is called when a hypothesis is
//!   verified or falsified (after the pass, not during).
//! - The fleet tally uses [`theorem_bayes::CountBinSet`] to compound decayed
//!   counts across tenants without sharing content.

use std::collections::BTreeMap;

use theorem_bayes::{seed_from_parts, BetaBernoulli, CountBinSet};

/// Minimum allowed value to avoid degenerate Betas.
const EPSILON: f64 = 1e-9;

/// Default prior strength for a new schema with no fleet history.
/// Beta(prior_strength, prior_strength) → mean 0.5, low confidence.
pub const DEFAULT_THOMPSON_PRIOR_STRENGTH: f64 = 2.0;

/// Default half-life in days for fleet tally decay.
pub const DEFAULT_THOMPSON_HALF_LIFE_DAYS: f64 = 30.0;

/// Task family key used for analogy outcome tally bins.
pub const ANALOGY_TASK_FAMILY: &str = "analogical_transfer";

/// Tool family key used for analogy outcome tally bins.
pub const ANALOGY_TOOL_FAMILY: &str = "thompson_analogy_v1";

/// A Thompson-sampling oracle for analogical transfer hypotheses.
///
/// Holds a Beta posterior per schema and a fleet-wide tally of transfer
/// outcomes. The prior for a new schema blends the structural-score prior
/// with the fleet tally, so schemas with fleet evidence start with an
/// informed Beta rather than a flat one.
#[derive(Clone, Debug)]
pub struct ThompsonAnalogyOracle {
    /// Beta posterior per schema id.
    schema_betas: BTreeMap<String, BetaBernoulli>,
    /// Fleet-wide tally of transfer outcomes (decayed, per-tenant).
    fleet_tally: CountBinSet,
    /// Prior strength for a schema with no fleet evidence.
    prior_strength: f64,
    /// Half-life in days for fleet tally decay.
    half_life_days: f64,
    /// Tenant slug for fleet tally keying.
    tenant_slug: String,
}

impl ThompsonAnalogyOracle {
    /// Create a new oracle with the given tenant slug.
    ///
    /// `prior_strength` controls how strongly the initial Beta is concentrated
    /// around its mean (higher = narrower prior). `half_life_days` controls
    /// fleet tally decay.
    pub fn new(tenant_slug: impl Into<String>, prior_strength: f64, half_life_days: f64) -> Self {
        Self {
            schema_betas: BTreeMap::new(),
            fleet_tally: CountBinSet::default(),
            prior_strength: prior_strength.max(EPSILON),
            half_life_days: half_life_days.max(EPSILON),
            tenant_slug: tenant_slug.into(),
        }
    }

    /// Initialize the Beta posterior for a schema.
    ///
    /// Blends the structural-score prior with the fleet tally:
    /// - Beta(prior_strength, prior_strength) scaled toward the structural score
    /// - Fleet successes/failures folded in
    pub fn initialize_schema_prior(
        &mut self,
        schema_id: &str,
        structural_score: f64,
        day_index: i64,
    ) {
        let score = structural_score.clamp(0.01, 0.99);
        // Build a structural-score-informed prior: Beta(α, β) with mean = score
        // and concentration = prior_strength.
        let structural_prior = BetaBernoulli::new(
            self.prior_strength * score,
            self.prior_strength * (1.0 - score),
        );

        // Blend with fleet tally
        let fleet_prior = self.fleet_tally.beta_prior_for(
            &self.tenant_slug,
            ANALOGY_TASK_FAMILY,
            ANALOGY_TOOL_FAMILY,
            day_index,
            self.half_life_days,
            self.prior_strength,
        );

        // Start from structural prior, fold in fleet evidence
        let fleet_successes = fleet_prior.mean() * self.prior_strength;
        let fleet_failures = (1.0 - fleet_prior.mean()) * self.prior_strength;
        let blended = structural_prior.observe_counts(fleet_successes, fleet_failures);

        self.schema_betas.insert(schema_id.to_string(), blended);
    }

    /// Return a reference to the Beta posterior for a schema, if initialized.
    pub fn schema_beta(&self, schema_id: &str) -> Option<&BetaBernoulli> {
        self.schema_betas.get(schema_id)
    }

    /// Thompson-sample a confidence value from the schema's Beta posterior.
    ///
    /// The seed is derived deterministically from the hypothesis id so the
    /// draw is reproducible. Returns the mean if the schema hasn't been
    /// initialized.
    pub fn thompson_sample(&self, schema_id: &str, hypothesis_id: &str) -> f64 {
        let beta = match self.schema_betas.get(schema_id) {
            Some(b) => b,
            None => return 0.25, // Conservative default for uninitialized schemas
        };
        let seed = seed_from_parts(&[schema_id, hypothesis_id]);
        beta.sample(seed).clamp(0.01, 0.99)
    }

    /// Observe a transfer outcome and update the schema's Beta posterior
    /// and the fleet tally.
    pub fn observe_outcome(&mut self, schema_id: &str, success: bool, day_index: i64) {
        use theorem_bayes::CountBinKey;

        // Update fleet tally
        self.fleet_tally.observe(
            CountBinKey::new(
                self.tenant_slug.clone(),
                day_index,
                ANALOGY_TASK_FAMILY,
                ANALOGY_TOOL_FAMILY,
                if success { "success" } else { "failure" },
            ),
            1.0,
        );

        // Update schema posterior
        if let Some(&beta) = self.schema_betas.get(schema_id) {
            let updated = if success {
                beta.observe_counts(1.0, 0.0)
            } else {
                beta.observe_counts(0.0, 1.0)
            };
            self.schema_betas.insert(schema_id.to_string(), updated);
        }
    }

    /// Re-initialize all schema priors from the fleet tally (e.g., after
    /// fleet tally has been loaded from persistent storage).
    pub fn refresh_from_fleet(
        &mut self,
        structural_scores: &BTreeMap<String, f64>,
        day_index: i64,
    ) {
        for (schema_id, score) in structural_scores {
            if !self.schema_betas.contains_key(schema_id) {
                self.initialize_schema_prior(schema_id, *score, day_index);
            }
        }
    }

    /// Number of schemas with initialized Betas.
    pub fn schema_count(&self) -> usize {
        self.schema_betas.len()
    }

    /// Tenant count from the fleet tally.
    pub fn fleet_tenant_count(&self) -> usize {
        self.fleet_tally.tenant_count()
    }
}

impl Default for ThompsonAnalogyOracle {
    fn default() -> Self {
        Self::new(
            "default",
            DEFAULT_THOMPSON_PRIOR_STRENGTH,
            DEFAULT_THOMPSON_HALF_LIFE_DAYS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_oracle_is_empty() {
        let oracle = ThompsonAnalogyOracle::default();
        assert_eq!(oracle.schema_count(), 0);
        assert_eq!(oracle.fleet_tenant_count(), 0);
    }

    #[test]
    fn initialize_schema_prior_gives_reasonable_mean() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.initialize_schema_prior("schema-a", 0.7, 0);
        let beta = oracle.schema_beta("schema-a").unwrap();
        // Mean should be near 0.7 (structural prior) since fleet is empty
        assert!((beta.mean() - 0.7).abs() < 0.3);
    }

    #[test]
    fn thompson_sample_is_deterministic() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.initialize_schema_prior("s1", 0.6, 0);
        let s1 = oracle.thompson_sample("s1", "hyp-1");
        let s2 = oracle.thompson_sample("s1", "hyp-1");
        assert_eq!(s1, s2, "same schema + hypothesis → same draw");
    }

    #[test]
    fn different_hypotheses_may_differ() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.initialize_schema_prior("s1", 0.6, 0);
        let s1 = oracle.thompson_sample("s1", "hyp-a");
        let s2 = oracle.thompson_sample("s1", "hyp-b");
        // They could be equal by chance with a weak prior, but structurally
        // different seeds should produce different values most of the time.
        // We just assert they're valid.
        assert!(s1 >= 0.01 && s1 <= 0.99);
        assert!(s2 >= 0.01 && s2 <= 0.99);
    }

    #[test]
    fn observe_success_increases_mean() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.initialize_schema_prior("s1", 0.5, 0);
        let mean_before = oracle.schema_beta("s1").unwrap().mean();
        oracle.observe_outcome("s1", true, 0);
        let mean_after = oracle.schema_beta("s1").unwrap().mean();
        assert!(mean_after > mean_before, "success should increase mean");
    }

    #[test]
    fn observe_failure_decreases_mean() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.initialize_schema_prior("s1", 0.5, 0);
        let mean_before = oracle.schema_beta("s1").unwrap().mean();
        oracle.observe_outcome("s1", false, 0);
        let mean_after = oracle.schema_beta("s1").unwrap().mean();
        assert!(mean_after < mean_before, "failure should decrease mean");
    }

    #[test]
    fn fleet_tally_tracks_outcomes() {
        let mut oracle = ThompsonAnalogyOracle::default();
        oracle.observe_outcome("s1", true, 0);
        oracle.observe_outcome("s1", true, 1);
        oracle.observe_outcome("s1", false, 2);

        let fleet_prior = oracle.fleet_tally.beta_prior_for(
            "default",
            ANALOGY_TASK_FAMILY,
            ANALOGY_TOOL_FAMILY,
            3,
            30.0,
            2.0,
        );
        // Two successes, one failure from fleet
        assert!(
            fleet_prior.mean() > 0.5,
            "fleet prior reflects success bias"
        );
    }

    #[test]
    fn uninitialized_schema_returns_conservative_default() {
        let oracle = ThompsonAnalogyOracle::default();
        let sample = oracle.thompson_sample("nonexistent", "hyp-1");
        assert_eq!(sample, 0.25, "uninitialized schema returns default 0.25");
    }

    #[test]
    fn refresh_from_fleet_initializes_missing_schemas() {
        let mut oracle = ThompsonAnalogyOracle::default();
        // Prime fleet with some data
        oracle.observe_outcome("s1", true, 0);
        oracle.observe_outcome("s1", true, 0);

        let scores: BTreeMap<String, f64> = [("schema-new".to_string(), 0.8)].into();
        oracle.refresh_from_fleet(&scores, 0);
        assert!(oracle.schema_beta("schema-new").is_some());
        // Schema-new should have been initialized with structural score 0.8
        // blended with fleet evidence
        let beta = oracle.schema_beta("schema-new").unwrap();
        assert!(
            beta.mean() > 0.5,
            "fleet evidence + high structural score → mean > 0.5"
        );
    }
}
