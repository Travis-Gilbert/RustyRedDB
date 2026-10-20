//! Bayesian posterior values for Theorem harness decision gates.
//!
//! This crate owns the small online path: conjugate updates, deterministic
//! seeded draws for replay, and content-addressed posterior receipts. Heavier
//! hierarchical modeling can stay in PyMC/offline jobs and compare against these
//! shapes through fixtures.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const EPSILON: f64 = 1e-9;

pub const BAYESIAN_TUNING_MODEL_VERSION: &str = "bayesian-harness-v1";
pub const DEFAULT_CONTEXT_LEAF: &str = "tenant/day/task_family/tool_family/outcome";
pub const DEFAULT_KAPPA: f64 = 1.0;
pub const DEFAULT_KAPPA_FLEET: f64 = 1.0;
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: f64 = 7.0;
pub const DEFAULT_DP_POLICY: &str = "counts-only-no-dp-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BayesianTuningDefaults {
    pub model_version: String,
    pub context_leaf: String,
    pub kappa: f64,
    pub kappa_fleet: f64,
    pub decay_half_life_days: f64,
    pub dp_policy: String,
}

impl BayesianTuningDefaults {
    pub fn current() -> Self {
        Self {
            model_version: BAYESIAN_TUNING_MODEL_VERSION.to_string(),
            context_leaf: DEFAULT_CONTEXT_LEAF.to_string(),
            kappa: DEFAULT_KAPPA,
            kappa_fleet: DEFAULT_KAPPA_FLEET,
            decay_half_life_days: DEFAULT_DECAY_HALF_LIFE_DAYS,
            dp_policy: DEFAULT_DP_POLICY.to_string(),
        }
    }

    pub fn as_value(&self) -> Value {
        serde_json::to_value(self).expect("bayesian tuning defaults serialize")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BetaBernoulli {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaBernoulli {
    /// Neutral Beta(1, 1) prior for binary success/failure evidence.
    pub fn default_prior() -> Self {
        Self::new(1.0, 1.0)
    }

    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(EPSILON),
            beta: beta.max(EPSILON),
        }
    }
}

impl Default for BetaBernoulli {
    fn default() -> Self {
        Self::default_prior()
    }
}

impl BetaBernoulli {
    pub fn from_mean_strength(mean: f64, strength: f64) -> Self {
        let mean = mean.clamp(EPSILON, 1.0 - EPSILON);
        let strength = strength.max(2.0 * EPSILON);
        Self::new(mean * strength, (1.0 - mean) * strength)
    }

    pub fn observe_counts(self, successes: f64, failures: f64) -> Self {
        Self::new(
            self.alpha + successes.max(0.0),
            self.beta + failures.max(0.0),
        )
    }

    pub fn observe_weighted(self, outcome_value: f64, outcome_weight: f64) -> Self {
        let value = outcome_value.clamp(0.0, 1.0);
        let weight = outcome_weight.max(0.0);
        self.observe_counts(value * weight, (1.0 - value) * weight)
    }

    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / ((total * total) * (total + 1.0))
    }

    pub fn sample(&self, seed: u64) -> f64 {
        let mut rng = SplitMix64::new(seed);
        let a = sample_gamma(self.alpha, &mut rng);
        let b = sample_gamma(self.beta, &mut rng);
        if a + b <= EPSILON {
            self.mean()
        } else {
            (a / (a + b)).clamp(0.0, 1.0)
        }
    }

    /// Equal-tailed credible interval at `mass` (e.g. 0.95 for a 95% credible interval).
    /// Returns `(lower, upper)` where P[lower ≤ θ ≤ upper] = mass.
    pub fn credible_interval(&self, mass: f64) -> (f64, f64) {
        let tail = (1.0 - mass) / 2.0;
        let lower = inverse_regularized_incomplete_beta(tail, self.alpha, self.beta);
        let upper = inverse_regularized_incomplete_beta(1.0 - tail, self.alpha, self.beta);
        (lower, upper)
    }

    pub fn as_value(&self) -> Value {
        let (lo, hi) = self.credible_interval(0.95);
        json!({
            "distribution": "beta",
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean(),
            "variance": self.variance(),
            "credible_interval_95": [lo, hi],
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryObservation {
    pub receipt_id: String,
    pub outcome_value: f64,
    pub outcome_weight: f64,
    #[serde(default)]
    pub children: Vec<BinaryObservation>,
}

impl BinaryObservation {
    pub fn new(receipt_id: impl Into<String>, outcome_value: f64, outcome_weight: f64) -> Self {
        Self {
            receipt_id: receipt_id.into(),
            outcome_value,
            outcome_weight,
            children: Vec::new(),
        }
    }

    pub fn with_children(mut self, children: Vec<BinaryObservation>) -> Self {
        self.children = children;
        self
    }

    pub fn success_failure_counts(&self) -> (f64, f64, f64) {
        let value = self.outcome_value.clamp(0.0, 1.0);
        let weight = self.outcome_weight.max(0.0);
        (value * weight, (1.0 - value) * weight, weight)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryReceiptFold {
    pub prior: BetaBernoulli,
    pub posterior: BetaBernoulli,
    pub successes: f64,
    pub failures: f64,
    pub total_weight: f64,
    pub observation_count: usize,
    pub max_depth_seen: usize,
}

impl BinaryReceiptFold {
    pub fn as_value(&self) -> Value {
        json!({
            "prior": self.prior.as_value(),
            "posterior": self.posterior.as_value(),
            "successes": self.successes,
            "failures": self.failures,
            "total_weight": self.total_weight,
            "observation_count": self.observation_count,
            "max_depth_seen": self.max_depth_seen,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReceiptFoldError {
    MaxNodesExceeded {
        max_nodes: usize,
    },
    MaxDepthExceeded {
        receipt_id: String,
        depth: usize,
        max_depth: usize,
    },
}

impl fmt::Display for ReceiptFoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxNodesExceeded { max_nodes } => {
                write!(formatter, "receipt fold exceeded max_nodes={max_nodes}")
            }
            Self::MaxDepthExceeded {
                receipt_id,
                depth,
                max_depth,
            } => write!(
                formatter,
                "receipt fold exceeded max_depth={max_depth} at {receipt_id} depth {depth}"
            ),
        }
    }
}

impl std::error::Error for ReceiptFoldError {}

pub fn fold_binary_receipts(
    prior: BetaBernoulli,
    roots: &[BinaryObservation],
    max_nodes: usize,
    max_depth: usize,
) -> Result<BinaryReceiptFold, ReceiptFoldError> {
    let mut stack = roots
        .iter()
        .rev()
        .map(|observation| (observation, 1_usize))
        .collect::<Vec<_>>();
    let mut successes = 0.0;
    let mut failures = 0.0;
    let mut total_weight = 0.0;
    let mut observation_count = 0_usize;
    let mut max_depth_seen = 0_usize;

    while let Some((observation, depth)) = stack.pop() {
        if depth > max_depth {
            return Err(ReceiptFoldError::MaxDepthExceeded {
                receipt_id: observation.receipt_id.clone(),
                depth,
                max_depth,
            });
        }
        if observation_count >= max_nodes {
            return Err(ReceiptFoldError::MaxNodesExceeded { max_nodes });
        }

        let (observation_successes, observation_failures, observation_weight) =
            observation.success_failure_counts();
        successes += observation_successes;
        failures += observation_failures;
        total_weight += observation_weight;
        observation_count += 1;
        max_depth_seen = max_depth_seen.max(depth);

        for child in observation.children.iter().rev() {
            stack.push((child, depth + 1));
        }
    }

    Ok(BinaryReceiptFold {
        prior,
        posterior: prior.observe_counts(successes, failures),
        successes,
        failures,
        total_weight,
        observation_count,
        max_depth_seen,
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CountBinKey {
    pub tenant_slug: String,
    pub day_index: i64,
    pub task_family: String,
    pub tool_family: String,
    pub outcome: String,
}

impl CountBinKey {
    pub fn new(
        tenant_slug: impl Into<String>,
        day_index: i64,
        task_family: impl Into<String>,
        tool_family: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            tenant_slug: tenant_slug.into(),
            day_index,
            task_family: task_family.into(),
            tool_family: tool_family.into(),
            outcome: outcome.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountBinRecord {
    pub key: CountBinKey,
    pub count: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountBinSummary {
    pub key: CountBinKey,
    pub count: f64,
    pub decayed_count: f64,
    pub read_day_index: i64,
    pub half_life_days: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CountBinSet {
    pub bins: Vec<CountBinRecord>,
}

impl CountBinSet {
    pub fn observe(&mut self, key: CountBinKey, weight: f64) {
        if let Some(record) = self.bins.iter_mut().find(|record| record.key == key) {
            record.count += weight.max(0.0);
        } else {
            self.bins.push(CountBinRecord {
                key,
                count: weight.max(0.0),
            });
            self.bins.sort_by(|left, right| left.key.cmp(&right.key));
        }
    }

    pub fn merged(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for record in &other.bins {
            out.observe(record.key.clone(), record.count);
        }
        out
    }

    pub fn decayed_counts(&self, read_day_index: i64, half_life_days: f64) -> Vec<CountBinSummary> {
        let half_life_days = half_life_days.max(EPSILON);
        self.bins
            .iter()
            .map(|record| {
                let key = &record.key;
                let age_days = read_day_index.saturating_sub(key.day_index).max(0) as f64;
                let decay = 0.5_f64.powf(age_days / half_life_days);
                CountBinSummary {
                    key: key.clone(),
                    count: record.count,
                    decayed_count: record.count * decay,
                    read_day_index,
                    half_life_days,
                }
            })
            .collect()
    }

    pub fn decayed_outcome_count(
        &self,
        tenant_slug: &str,
        task_family: &str,
        tool_family: &str,
        outcome: &str,
        read_day_index: i64,
        half_life_days: f64,
    ) -> f64 {
        self.decayed_counts(read_day_index, half_life_days)
            .into_iter()
            .filter(|summary| {
                summary.key.tenant_slug == tenant_slug
                    && summary.key.task_family == task_family
                    && summary.key.tool_family == tool_family
                    && summary.key.outcome == outcome
            })
            .map(|summary| summary.decayed_count)
            .sum()
    }

    pub fn beta_prior_for(
        &self,
        tenant_slug: &str,
        task_family: &str,
        tool_family: &str,
        read_day_index: i64,
        half_life_days: f64,
        prior_strength: f64,
    ) -> BetaBernoulli {
        let successes = self.decayed_outcome_count(
            tenant_slug,
            task_family,
            tool_family,
            "success",
            read_day_index,
            half_life_days,
        );
        let failures = self.decayed_outcome_count(
            tenant_slug,
            task_family,
            tool_family,
            "failure",
            read_day_index,
            half_life_days,
        );
        BetaBernoulli::new(prior_strength.max(EPSILON), prior_strength.max(EPSILON))
            .observe_counts(successes, failures)
    }

    pub fn tenant_count(&self) -> usize {
        self.bins
            .iter()
            .map(|record| record.key.tenant_slug.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn aggregated_for_tenant(&self, aggregate_tenant_slug: &str) -> Self {
        let mut aggregated = Self::default();
        for record in &self.bins {
            aggregated.observe(
                CountBinKey::new(
                    aggregate_tenant_slug,
                    record.key.day_index,
                    record.key.task_family.clone(),
                    record.key.tool_family.clone(),
                    record.key.outcome.clone(),
                ),
                record.count,
            );
        }
        aggregated
    }
}

pub const PRIOR_PACK_SCHEMA_VERSION: u32 = 1;
pub const PRIOR_PACK_AGGREGATE_TENANT: &str = "__fleet__";
pub const PRIOR_PACK_SIGNATURE_ALGORITHM: &str = "theorem-sha256-local-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorPackCompilerInput {
    pub model_id: String,
    pub graph_version: u64,
    pub generated_at_day_index: i64,
    pub aggregate_tenant_slug: String,
    pub source_bins: CountBinSet,
    pub tenant_count: Option<u64>,
    pub rollback_parent_pack_id: Option<String>,
}

impl PriorPackCompilerInput {
    pub fn new(
        model_id: impl Into<String>,
        graph_version: u64,
        generated_at_day_index: i64,
        source_bins: CountBinSet,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            graph_version,
            generated_at_day_index,
            aggregate_tenant_slug: PRIOR_PACK_AGGREGATE_TENANT.to_string(),
            source_bins,
            tenant_count: None,
            rollback_parent_pack_id: None,
        }
    }

    pub fn with_tenant_count(mut self, tenant_count: u64) -> Self {
        self.tenant_count = Some(tenant_count);
        self
    }

    pub fn with_rollback_parent(mut self, parent_pack_id: impl Into<String>) -> Self {
        self.rollback_parent_pack_id = Some(parent_pack_id.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorPackPayload {
    pub schema_version: u32,
    pub model_id: String,
    pub graph_version: u64,
    pub generated_at_day_index: i64,
    pub tenant_count: u64,
    pub source_bin_count: usize,
    pub aggregate_tenant_slug: String,
    pub rollback_parent_pack_id: Option<String>,
    pub provenance_digest: String,
    pub tuning: BayesianTuningDefaults,
    pub bins: CountBinSet,
}

impl PriorPackPayload {
    pub fn digest(&self) -> String {
        stable_hash_value(&serde_json::to_value(self).expect("prior pack payload serializes"))
    }

    pub fn pack_id(&self) -> String {
        format!("prior-pack:{}", self.digest())
    }

    pub fn verify_content_boundary(&self) -> Result<(), PriorPackError> {
        if self.schema_version != PRIOR_PACK_SCHEMA_VERSION {
            return Err(PriorPackError::UnsupportedSchema {
                schema_version: self.schema_version,
            });
        }
        if self.tenant_count == 0 {
            return Err(PriorPackError::InvalidPayload(
                "tenant_count must be positive".to_string(),
            ));
        }
        for record in &self.bins.bins {
            if record.key.tenant_slug != self.aggregate_tenant_slug {
                return Err(PriorPackError::ContentBoundaryViolation {
                    reason: "prior pack bin exposes a source tenant slug".to_string(),
                });
            }
        }
        ensure_no_content_keys(
            &serde_json::to_value(self).expect("prior pack payload serializes"),
            "$",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorPackSignature {
    pub signer_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub digest: String,
    pub signature: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignedPriorPack {
    pub payload: PriorPackPayload,
    pub signature: PriorPackSignature,
}

impl SignedPriorPack {
    pub fn pack_id(&self) -> String {
        self.payload.pack_id()
    }

    pub fn rollback_parent_pack_id(&self) -> Option<&str> {
        self.payload.rollback_parent_pack_id.as_deref()
    }

    pub fn verify(&self, secret: &[u8]) -> Result<(), PriorPackError> {
        self.payload.verify_content_boundary()?;
        let digest = self.payload.digest();
        if self.signature.algorithm != PRIOR_PACK_SIGNATURE_ALGORITHM {
            return Err(PriorPackError::UnsupportedSignatureAlgorithm {
                algorithm: self.signature.algorithm.clone(),
            });
        }
        if self.signature.digest != digest {
            return Err(PriorPackError::DigestMismatch {
                expected: digest,
                actual: self.signature.digest.clone(),
            });
        }
        let expected = prior_pack_signature(
            &self.signature.signer_id,
            &self.signature.key_id,
            &self.signature.digest,
            secret,
        );
        if self.signature.signature != expected {
            return Err(PriorPackError::BadSignature);
        }
        Ok(())
    }

    pub fn affordance_suggestions(
        &self,
        task_family: &str,
        read_day_index: i64,
        half_life_days: f64,
        prior_strength: f64,
    ) -> Vec<PriorPackAffordanceSuggestion> {
        let tool_families = self
            .payload
            .bins
            .bins
            .iter()
            .filter(|record| {
                record.key.tenant_slug == self.payload.aggregate_tenant_slug
                    && record.key.task_family == task_family
            })
            .map(|record| record.key.tool_family.as_str())
            .collect::<BTreeSet<_>>();
        let pack_id = self.pack_id();
        let mut suggestions = tool_families
            .into_iter()
            .map(|tool_family| {
                let posterior = self.payload.bins.beta_prior_for(
                    &self.payload.aggregate_tenant_slug,
                    task_family,
                    tool_family,
                    read_day_index,
                    half_life_days,
                    prior_strength,
                );
                PriorPackAffordanceSuggestion {
                    pack_id: pack_id.clone(),
                    task_family: task_family.to_string(),
                    tool_family: tool_family.to_string(),
                    posterior,
                    posterior_mean: posterior.mean(),
                    posterior_variance: posterior.variance(),
                }
            })
            .collect::<Vec<_>>();
        suggestions.sort_by(|left, right| {
            right
                .posterior_mean
                .partial_cmp(&left.posterior_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.posterior_variance
                        .partial_cmp(&right.posterior_variance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.tool_family.cmp(&right.tool_family))
        });
        suggestions
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriorPackAffordanceSuggestion {
    pub pack_id: String,
    pub task_family: String,
    pub tool_family: String,
    pub posterior: BetaBernoulli,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PriorPackError {
    UnsupportedSchema { schema_version: u32 },
    UnsupportedSignatureAlgorithm { algorithm: String },
    DigestMismatch { expected: String, actual: String },
    BadSignature,
    ContentBoundaryViolation { reason: String },
    InvalidPayload(String),
}

impl fmt::Display for PriorPackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { schema_version } => {
                write!(formatter, "unsupported prior pack schema {schema_version}")
            }
            Self::UnsupportedSignatureAlgorithm { algorithm } => {
                write!(
                    formatter,
                    "unsupported prior pack signature algorithm {algorithm}"
                )
            }
            Self::DigestMismatch { expected, actual } => {
                write!(
                    formatter,
                    "prior pack digest mismatch: expected {expected}, got {actual}"
                )
            }
            Self::BadSignature => write!(formatter, "prior pack signature verification failed"),
            Self::ContentBoundaryViolation { reason } => {
                write!(formatter, "prior pack content boundary violation: {reason}")
            }
            Self::InvalidPayload(reason) => {
                write!(formatter, "invalid prior pack payload: {reason}")
            }
        }
    }
}

impl std::error::Error for PriorPackError {}

pub fn compile_prior_pack(
    input: PriorPackCompilerInput,
) -> Result<PriorPackPayload, PriorPackError> {
    let aggregate_tenant_slug = if input.aggregate_tenant_slug.trim().is_empty() {
        PRIOR_PACK_AGGREGATE_TENANT.to_string()
    } else {
        input.aggregate_tenant_slug
    };
    let tenant_count = input
        .tenant_count
        .unwrap_or_else(|| input.source_bins.tenant_count() as u64);
    if tenant_count == 0 {
        return Err(PriorPackError::InvalidPayload(
            "cannot compile a prior pack from zero tenants".to_string(),
        ));
    }

    let source_bin_count = input.source_bins.bins.len();
    let provenance_digest = stable_hash_value(&json!({
        "model_id": input.model_id,
        "graph_version": input.graph_version,
        "generated_at_day_index": input.generated_at_day_index,
        "source_bin_count": source_bin_count,
        "tenant_count": tenant_count,
        "source_bins_digest": stable_hash_value(
            &serde_json::to_value(&input.source_bins).expect("source bins serialize")
        ),
    }));
    let payload = PriorPackPayload {
        schema_version: PRIOR_PACK_SCHEMA_VERSION,
        model_id: input.model_id,
        graph_version: input.graph_version,
        generated_at_day_index: input.generated_at_day_index,
        tenant_count,
        source_bin_count,
        aggregate_tenant_slug: aggregate_tenant_slug.clone(),
        rollback_parent_pack_id: input.rollback_parent_pack_id,
        provenance_digest,
        tuning: BayesianTuningDefaults::current(),
        bins: input
            .source_bins
            .aggregated_for_tenant(&aggregate_tenant_slug),
    };
    payload.verify_content_boundary()?;
    Ok(payload)
}

pub fn sign_prior_pack(
    payload: PriorPackPayload,
    signer_id: impl Into<String>,
    key_id: impl Into<String>,
    secret: &[u8],
) -> Result<SignedPriorPack, PriorPackError> {
    payload.verify_content_boundary()?;
    let signer_id = signer_id.into();
    let key_id = key_id.into();
    let digest = payload.digest();
    let signature = prior_pack_signature(&signer_id, &key_id, &digest, secret);
    Ok(SignedPriorPack {
        payload,
        signature: PriorPackSignature {
            signer_id,
            key_id,
            algorithm: PRIOR_PACK_SIGNATURE_ALGORITHM.to_string(),
            digest,
            signature,
        },
    })
}

fn prior_pack_signature(signer_id: &str, key_id: &str, digest: &str, secret: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PRIOR_PACK_SIGNATURE_ALGORITHM.as_bytes());
    hasher.update((signer_id.len() as u64).to_le_bytes());
    hasher.update(signer_id.as_bytes());
    hasher.update((key_id.len() as u64).to_le_bytes());
    hasher.update(key_id.as_bytes());
    hasher.update((digest.len() as u64).to_le_bytes());
    hasher.update(digest.as_bytes());
    hasher.update((secret.len() as u64).to_le_bytes());
    hasher.update(secret);
    to_hex(&hasher.finalize())
}

fn ensure_no_content_keys(value: &Value, path: &str) -> Result<(), PriorPackError> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let normalized = key.to_ascii_lowercase();
                if matches!(
                    normalized.as_str(),
                    "content" | "prompt" | "source_text" | "raw_text" | "body" | "document"
                ) {
                    return Err(PriorPackError::ContentBoundaryViolation {
                        reason: format!("field {path}.{key} is not allowed in aggregate packs"),
                    });
                }
                ensure_no_content_keys(child, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                ensure_no_content_keys(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirichletCategorical {
    pub categories: Vec<String>,
    pub alphas: Vec<f64>,
}

impl DirichletCategorical {
    pub fn new<I, S>(categories: I, prior_alpha: f64) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let categories = categories.into_iter().map(Into::into).collect::<Vec<_>>();
        let alphas = vec![prior_alpha.max(EPSILON); categories.len()];
        Self { categories, alphas }
    }

    pub fn observe(mut self, category: &str, weight: f64) -> Self {
        if let Some(index) = self.categories.iter().position(|known| known == category) {
            self.alphas[index] += weight.max(0.0);
        }
        self
    }

    pub fn mean_distribution(&self) -> Vec<f64> {
        let total = self.alphas.iter().sum::<f64>().max(EPSILON);
        self.alphas.iter().map(|alpha| alpha / total).collect()
    }

    pub fn sample_distribution(&self, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        let mut draws = self
            .alphas
            .iter()
            .map(|alpha| sample_gamma(*alpha, &mut rng))
            .collect::<Vec<_>>();
        let total = draws.iter().sum::<f64>();
        if total <= EPSILON {
            return self.mean_distribution();
        }
        for draw in &mut draws {
            *draw /= total;
        }
        draws
    }

    pub fn as_value(&self) -> Value {
        json!({
            "distribution": "dirichlet",
            "categories": self.categories,
            "alphas": self.alphas,
            "mean": self.mean_distribution(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalMoments {
    pub mean: f64,
    pub m2: f64,
    pub weight: f64,
}

impl NormalMoments {
    pub fn new(mean: f64, variance: f64, weight: f64) -> Self {
        let weight = weight.max(0.0);
        Self {
            mean,
            m2: variance.max(0.0) * weight.max(1.0),
            weight,
        }
    }

    pub fn empty() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            weight: 0.0,
        }
    }

    pub fn observe(self, value: f64) -> Self {
        let next_weight = self.weight + 1.0;
        let delta = value - self.mean;
        let mean = self.mean + delta / next_weight;
        let delta2 = value - mean;
        Self {
            mean,
            m2: self.m2 + delta * delta2,
            weight: next_weight,
        }
    }

    pub fn variance(&self) -> f64 {
        if self.weight <= 1.0 {
            0.0
        } else {
            self.m2 / (self.weight - 1.0)
        }
    }

    pub fn standard_error(&self) -> f64 {
        if self.weight <= 0.0 {
            1.0
        } else {
            (self.variance() / self.weight).sqrt()
        }
    }

    pub fn sample(&self, seed: u64) -> f64 {
        let mut rng = SplitMix64::new(seed);
        self.mean + sample_standard_normal(&mut rng) * self.standard_error()
    }

    pub fn as_value(&self) -> Value {
        json!({
            "distribution": "normal_moments",
            "mean": self.mean,
            "variance": self.variance(),
            "weight": self.weight,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PosteriorReceipt {
    pub kind: String,
    pub model_id: String,
    pub prior: Value,
    pub observations: Value,
    pub posterior: Value,
    pub metadata: Value,
    pub receipt_hash: String,
}

impl PosteriorReceipt {
    pub fn new(
        kind: impl Into<String>,
        model_id: impl Into<String>,
        prior: Value,
        observations: Value,
        posterior: Value,
        metadata: Value,
    ) -> Self {
        let kind = kind.into();
        let model_id = model_id.into();
        let hash_payload = json!({
            "kind": kind,
            "model_id": model_id,
            "prior": prior,
            "observations": observations,
            "posterior": posterior,
            "metadata": metadata,
        });
        let receipt_hash = stable_hash_value(&hash_payload);
        Self {
            kind,
            model_id,
            prior,
            observations,
            posterior,
            metadata,
            receipt_hash,
        }
    }

    pub fn as_value(&self) -> Value {
        serde_json::to_value(self).expect("posterior receipt serializes")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64_open01(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        ((value as f64) + 0.5) / ((1_u64 << 53) as f64)
    }
}

pub fn seed_from_parts(parts: &[&str]) -> u64 {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    let digest = hasher.finalize();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

pub fn stable_hash_value(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).expect("json value serializes");
    let digest = Sha256::digest(bytes);
    to_hex(&digest)
}

fn sample_gamma(shape: f64, rng: &mut SplitMix64) -> f64 {
    let shape = shape.max(EPSILON);
    if shape < 1.0 {
        let u = rng.next_f64_open01();
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }

    let d = shape - (1.0 / 3.0);
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x = sample_standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.next_f64_open01();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

fn sample_standard_normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64_open01();
    let u2 = rng.next_f64_open01();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// --- Regularized Incomplete Beta (for credible intervals) ---

/// Log-Gamma via Lanczos approximation (g=7, n=9).
fn ln_gamma(x: f64) -> f64 {
    let x = if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        const P: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let z = x + 7.0 + 0.5;
        let mut s = P[0];
        for (i, p) in P.iter().enumerate().skip(1) {
            s += p / (x + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt().ln() + s.ln() - z + (x + 0.5) * z.ln()
    };
    x
}

/// Log of the beta function: ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b).
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Regularized incomplete beta I_x(a, b) via continued fraction (modified Lentz).
/// Result is in [0, 1]; NaN for degenerate inputs.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }

    // Use left tail when x is left of the mode — avoids infinite recursion.
    let use_left = x * (a + b + 2.0) < (a + 1.0);
    if use_left || x <= a / (a + b) {
        let ln_front = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
        ln_front.exp() * incomplete_beta_cf(x, a, b)
    } else {
        1.0 - regularized_incomplete_beta(1.0 - x, b, a)
    }
}

/// Lentz continued fraction for the incomplete beta.
fn incomplete_beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-16;

    let mut c: f64 = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < EPS {
        d = EPS;
    }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..=MAX_ITER {
        let m2 = 2.0 * m as f64;

        // Even term: d_{2m}
        let mut t = m as f64 * (b - m as f64) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + t * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + t / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        f *= d * c;

        // Odd term: d_{2m+1}
        t = -(a + m as f64) * (a + b + m as f64) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + t * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + t / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    f
}

/// Inverse regularized incomplete beta: find x such that I_x(a,b) = p.
/// Uses Newton's method with a secure bracket.
fn inverse_regularized_incomplete_beta(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let mean = a / (a + b);

    // Initial guess via approximation
    let mut x = mean;
    if a > 1.0 && b > 1.0 {
        // Use normal approximation for initial guess
        let sigma = (a * b / ((a + b) * (a + b) * (a + b + 1.0))).sqrt();
        let z = normal_quantile(p);
        x = (mean + z * sigma).clamp(1e-8, 1.0 - 1e-8);
    }

    // Newton: x_{n+1} = x_n - (I_x - p) / beta_pdf(x; a, b)
    for _ in 0..50 {
        let ix = regularized_incomplete_beta(x, a, b);
        let err = ix - p;
        if err.abs() < 1e-12 {
            break;
        }
        // PDF of Beta at x: x^{a-1} (1-x)^{b-1} / B(a,b)
        let ln_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(a, b);
        let pdf = ln_pdf.exp();
        if pdf < 1e-300 {
            // Fall back to bisection
            let step = if err > 0.0 { -0.5 * x } else { 0.5 * (1.0 - x) };
            x = (x + step).clamp(1e-15, 1.0 - 1e-15);
        } else {
            let step = err / pdf;
            x = (x - step).clamp(1e-15, 1.0 - 1e-15);
        }
    }
    x
}

/// Approximate normal quantile (inverse CDF) using Peter Acklam's algorithm.
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    // Coefficients for the rational approximation (Acklam).
    const A1: f64 = -3.969683028665376e+01;
    const A2: f64 = 2.209460984245205e+02;
    const A3: f64 = -2.759285104469687e+02;
    const A4: f64 = 1.383_577_518_672_69e2;
    const A5: f64 = -3.066479806614716e+01;
    const A6: f64 = 2.506628277459239e+00;

    const B1: f64 = -5.447609879822406e+01;
    const B2: f64 = 1.615858368580409e+02;
    const B3: f64 = -1.556989798598866e+02;
    const B4: f64 = 6.680131188771972e+01;
    const B5: f64 = -1.328068155288572e+01;

    const C1: f64 = -7.784894002430293e-03;
    const C2: f64 = -3.223964580411365e-01;
    const C3: f64 = -2.400758277161838e+00;
    const C4: f64 = -2.549732539343734e+00;
    const C5: f64 = 4.374664141464968e+00;
    const C6: f64 = 2.938163982698783e+00;

    const D1: f64 = 7.784695709041462e-03;
    const D2: f64 = 3.224671290700398e-01;
    const D3: f64 = 2.445134137142996e+00;
    const D4: f64 = 3.754408661907416e+00;

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        q * (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6)
            / (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0)
    } else {
        let r = if q > 0.0 { 1.0 - p } else { p };
        let r = (-r.ln()).sqrt();
        let mut x = ((((C1 * r + C2) * r + C3) * r + C4) * r + C5) * r + C6;
        x /= (((D1 * r + D2) * r + D3) * r + D4) * r + 1.0;
        if q > 0.0 {
            -x
        } else {
            x
        }
    }
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_default_prior_and_serialization_contract_are_stable() {
        let prior = BetaBernoulli::default();

        assert_eq!(prior, BetaBernoulli::default_prior());
        assert_eq!(prior.alpha, 1.0);
        assert_eq!(prior.beta, 1.0);
        assert_eq!(
            serde_json::to_value(prior).unwrap(),
            json!({
                "alpha": 1.0,
                "beta": 1.0,
            })
        );
    }

    #[test]
    fn uniform_beta_credible_interval_matches_equal_tailed_oracle() {
        let (lower, upper) = BetaBernoulli::default().credible_interval(0.95);

        assert!((lower - 0.025).abs() < 1e-12);
        assert!((upper - 0.975).abs() < 1e-12);
    }

    #[test]
    fn asymmetric_beta_credible_interval_matches_closed_form_oracle() {
        fn beta_2_5_cdf(x: f64) -> f64 {
            15.0 * x.powi(2) - 40.0 * x.powi(3) + 45.0 * x.powi(4) - 24.0 * x.powi(5)
                + 5.0 * x.powi(6)
        }

        fn beta_2_5_quantile(probability: f64) -> f64 {
            let (mut lower, mut upper) = (0.0, 1.0);
            for _ in 0..100 {
                let midpoint = (lower + upper) / 2.0;
                if beta_2_5_cdf(midpoint) < probability {
                    lower = midpoint;
                } else {
                    upper = midpoint;
                }
            }
            (lower + upper) / 2.0
        }

        let posterior = BetaBernoulli::new(2.0, 5.0);
        let (lower, upper) = posterior.credible_interval(0.95);
        let expected_lower = beta_2_5_quantile(0.025);
        let expected_upper = beta_2_5_quantile(0.975);

        assert!((lower - expected_lower).abs() < 1e-10);
        assert!((upper - expected_upper).abs() < 1e-10);
        assert!(lower < posterior.mean());
        assert!(posterior.mean() < upper);
    }

    #[test]
    fn beta_updates_and_draws_are_deterministic() {
        let posterior = BetaBernoulli::new(1.0, 1.0).observe_weighted(0.75, 8.0);
        assert_eq!(posterior.alpha, 7.0);
        assert_eq!(posterior.beta, 3.0);
        assert!((posterior.mean() - 0.7).abs() < 1e-12);
        assert_eq!(posterior.sample(42), posterior.sample(42));
        assert_ne!(posterior.sample(42), posterior.sample(43));
    }

    #[test]
    fn beta_default_prior_and_credible_interval_are_stable() {
        assert_eq!(BetaBernoulli::default(), BetaBernoulli::new(1.0, 1.0));
        assert_eq!(BetaBernoulli::default_prior(), BetaBernoulli::new(1.0, 1.0));

        let posterior = BetaBernoulli::default_prior().observe_counts(8.0, 2.0);
        let interval = posterior.credible_interval(0.95);
        assert!(interval.0 < posterior.mean());
        assert!(interval.1 > posterior.mean());
        assert!((0.0..=1.0).contains(&interval.0));
        assert!((0.0..=1.0).contains(&interval.1));
    }

    #[test]
    fn recursive_receipt_fold_is_bounded_and_deterministic() {
        let observations = vec![
            BinaryObservation::new("root-success", 1.0, 2.0).with_children(vec![
                BinaryObservation::new("child-failure", 0.0, 1.0),
                BinaryObservation::new("child-success", 1.0, 4.0)
                    .with_children(vec![BinaryObservation::new("grandchild-failure", 0.0, 2.0)]),
            ]),
        ];
        let fold = fold_binary_receipts(BetaBernoulli::new(2.0, 3.0), &observations, 8, 4)
            .expect("fold stays within bounds");

        assert_eq!(fold.successes, 6.0);
        assert_eq!(fold.failures, 3.0);
        assert_eq!(fold.total_weight, 9.0);
        assert_eq!(fold.observation_count, 4);
        assert_eq!(fold.max_depth_seen, 3);
        assert_eq!(fold.posterior.alpha, 8.0);
        assert_eq!(fold.posterior.beta, 6.0);

        assert!(matches!(
            fold_binary_receipts(BetaBernoulli::new(2.0, 3.0), &observations, 3, 4),
            Err(ReceiptFoldError::MaxNodesExceeded { max_nodes: 3 })
        ));
        assert!(matches!(
            fold_binary_receipts(BetaBernoulli::new(2.0, 3.0), &observations, 8, 2),
            Err(ReceiptFoldError::MaxDepthExceeded {
                receipt_id,
                depth: 3,
                max_depth: 2,
            }) if receipt_id == "grandchild-failure"
        ));
    }

    #[test]
    fn recursive_receipt_fold_matches_oracle_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/affordance_posterior_oracle.json"
        ))
        .expect("fixture parses");
        let prior = BetaBernoulli::new(
            fixture["prior"]["alpha"].as_f64().expect("prior alpha"),
            fixture["prior"]["beta"].as_f64().expect("prior beta"),
        );
        let observations: Vec<BinaryObservation> =
            serde_json::from_value(fixture["observations"].clone())
                .expect("fixture observations parse");
        let bounds = &fixture["bounds"];
        let fold = fold_binary_receipts(
            prior,
            &observations,
            bounds["max_nodes"].as_u64().expect("max_nodes") as usize,
            bounds["max_depth"].as_u64().expect("max_depth") as usize,
        )
        .expect("fixture fold stays in bounds");
        let expected = &fixture["closed_form"];
        let posterior = &expected["posterior"];

        assert_close(fold.successes, expected["successes"].as_f64().unwrap());
        assert_close(fold.failures, expected["failures"].as_f64().unwrap());
        assert_close(
            fold.total_weight,
            expected["total_weight"].as_f64().unwrap(),
        );
        assert_eq!(
            fold.observation_count as u64,
            expected["observation_count"].as_u64().unwrap()
        );
        assert_eq!(
            fold.max_depth_seen as u64,
            expected["max_depth_seen"].as_u64().unwrap()
        );
        assert_close(fold.posterior.alpha, posterior["alpha"].as_f64().unwrap());
        assert_close(fold.posterior.beta, posterior["beta"].as_f64().unwrap());
        assert_close(fold.posterior.mean(), posterior["mean"].as_f64().unwrap());
        assert_close(
            fold.posterior.variance(),
            posterior["variance"].as_f64().unwrap(),
        );
    }

    #[test]
    fn count_bins_merge_commutatively_and_decay_at_read() {
        let mut first = CountBinSet::default();
        first.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            4.0,
        );
        let mut second = CountBinSet::default();
        second.observe(
            CountBinKey::new("tenant-a", 8, "coding", "search", "failure"),
            2.0,
        );

        let left = first.merged(&second);
        let right = second.merged(&first);
        assert_eq!(left, right);

        let summaries = left.decayed_counts(12, 2.0);
        let success = summaries
            .iter()
            .find(|summary| summary.key.outcome == "success")
            .unwrap();
        let failure = summaries
            .iter()
            .find(|summary| summary.key.outcome == "failure")
            .unwrap();
        assert_close(success.decayed_count, 2.0);
        assert_close(failure.decayed_count, 0.5);

        let prior = left.beta_prior_for("tenant-a", "coding", "search", 12, 2.0, 1.0);
        assert_close(prior.alpha, 3.0);
        assert_close(prior.beta, 1.5);
    }

    #[test]
    fn count_bins_keep_tenants_separate_and_serialize_without_content() {
        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            3.0,
        );
        bins.observe(
            CountBinKey::new("tenant-b", 10, "coding", "search", "success"),
            7.0,
        );

        assert_close(
            bins.decayed_outcome_count("tenant-a", "coding", "search", "success", 10, 7.0),
            3.0,
        );
        assert_close(
            bins.decayed_outcome_count("tenant-b", "coding", "search", "success", 10, 7.0),
            7.0,
        );

        let encoded = serde_json::to_string(&bins).expect("count bins serialize");
        assert!(!encoded.contains("content"));
        assert!(!encoded.contains("prompt"));
        assert!(!encoded.contains("source_text"));
    }

    #[test]
    fn prior_pack_compiles_aggregate_bins_without_source_tenants() {
        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            3.0,
        );
        bins.observe(
            CountBinKey::new("tenant-b", 10, "coding", "search", "success"),
            7.0,
        );
        bins.observe(
            CountBinKey::new("tenant-b", 10, "coding", "edit", "failure"),
            2.0,
        );

        let payload = compile_prior_pack(PriorPackCompilerInput::new(
            "affordance-prior-v1",
            42,
            10,
            bins,
        ))
        .expect("prior pack compiles");

        assert_eq!(payload.tenant_count, 2);
        assert_eq!(payload.source_bin_count, 3);
        assert_eq!(payload.bins.bins.len(), 2);
        assert_eq!(payload.aggregate_tenant_slug, PRIOR_PACK_AGGREGATE_TENANT);
        assert_eq!(
            payload.bins.decayed_outcome_count(
                PRIOR_PACK_AGGREGATE_TENANT,
                "coding",
                "search",
                "success",
                10,
                7.0,
            ),
            10.0
        );

        let encoded = serde_json::to_string(&payload).expect("payload serializes");
        assert!(!encoded.contains("tenant-a"));
        assert!(!encoded.contains("tenant-b"));
        assert!(!encoded.contains("content"));
        assert!(!encoded.contains("prompt"));
        assert!(!encoded.contains("source_text"));
    }

    #[test]
    fn prior_pack_signature_verifies_and_rejects_tampering() {
        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            3.0,
        );
        let payload = compile_prior_pack(PriorPackCompilerInput::new(
            "affordance-prior-v1",
            42,
            10,
            bins,
        ))
        .expect("prior pack compiles");
        let secret = b"local-test-secret";
        let signed = sign_prior_pack(payload, "fleet-signer", "key-1", secret).expect("pack signs");

        assert!(signed.verify(secret).is_ok());
        assert_eq!(signed.signature.digest, signed.payload.digest());
        assert_eq!(
            signed.verify(b"wrong-secret"),
            Err(PriorPackError::BadSignature)
        );

        let mut tampered_signature = signed.clone();
        tampered_signature.signature.signature = "bad".to_string();
        assert_eq!(
            tampered_signature.verify(secret),
            Err(PriorPackError::BadSignature)
        );

        let mut tampered_digest = signed.clone();
        tampered_digest.signature.digest = "different".to_string();
        assert!(matches!(
            tampered_digest.verify(secret),
            Err(PriorPackError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn prior_pack_preserves_rollback_identity() {
        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            1.0,
        );
        let payload = compile_prior_pack(
            PriorPackCompilerInput::new("affordance-prior-v1", 42, 10, bins)
                .with_rollback_parent("prior-pack:previous"),
        )
        .expect("prior pack compiles");
        let signed =
            sign_prior_pack(payload, "fleet-signer", "key-1", b"secret").expect("pack signs");

        assert_eq!(
            signed.rollback_parent_pack_id(),
            Some("prior-pack:previous")
        );
        assert!(signed.pack_id().starts_with("prior-pack:"));
    }

    #[test]
    fn prior_pack_produces_ranked_affordance_suggestions() {
        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            9.0,
        );
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "failure"),
            1.0,
        );
        bins.observe(
            CountBinKey::new("tenant-b", 10, "coding", "edit", "success"),
            2.0,
        );
        bins.observe(
            CountBinKey::new("tenant-b", 10, "coding", "edit", "failure"),
            8.0,
        );
        let payload = compile_prior_pack(PriorPackCompilerInput::new(
            "affordance-prior-v1",
            42,
            10,
            bins,
        ))
        .expect("prior pack compiles");
        let signed =
            sign_prior_pack(payload, "fleet-signer", "key-1", b"secret").expect("pack signs");

        let suggestions = signed.affordance_suggestions("coding", 10, 7.0, 1.0);

        assert_eq!(suggestions.len(), 2);
        assert_eq!(suggestions[0].tool_family, "search");
        assert_eq!(suggestions[1].tool_family, "edit");
        assert!(suggestions[0].posterior_mean > suggestions[1].posterior_mean);
        assert_eq!(suggestions[0].posterior.alpha, 10.0);
        assert_eq!(suggestions[0].posterior.beta, 2.0);
        assert_eq!(suggestions[0].pack_id, signed.pack_id());
    }

    #[test]
    fn tuning_defaults_are_named_and_receipt_visible() {
        let tuning = BayesianTuningDefaults::current();
        assert_eq!(tuning.model_version, BAYESIAN_TUNING_MODEL_VERSION);
        assert_eq!(tuning.context_leaf, DEFAULT_CONTEXT_LEAF);
        assert_eq!(tuning.kappa, DEFAULT_KAPPA);
        assert_eq!(tuning.kappa_fleet, DEFAULT_KAPPA_FLEET);
        assert_eq!(tuning.decay_half_life_days, DEFAULT_DECAY_HALF_LIFE_DAYS);
        assert_eq!(tuning.dp_policy, DEFAULT_DP_POLICY);

        let receipt = PosteriorReceipt::new(
            "tuning-defaults",
            BAYESIAN_TUNING_MODEL_VERSION,
            json!({}),
            json!({}),
            json!({}),
            json!({"tuning": tuning.as_value()}),
        );
        assert_eq!(
            receipt.metadata["tuning"]["context_leaf"],
            json!(DEFAULT_CONTEXT_LEAF)
        );
        assert_eq!(
            receipt.metadata["tuning"]["dp_policy"],
            json!(DEFAULT_DP_POLICY)
        );

        let mut bins = CountBinSet::default();
        bins.observe(
            CountBinKey::new("tenant-a", 10, "coding", "search", "success"),
            1.0,
        );
        let payload = compile_prior_pack(PriorPackCompilerInput::new(
            "affordance-prior-v1",
            42,
            10,
            bins,
        ))
        .expect("prior pack compiles");
        assert_eq!(payload.tuning.model_version, BAYESIAN_TUNING_MODEL_VERSION);
    }

    #[test]
    fn dirichlet_counts_and_means_are_stable() {
        let posterior = DirichletCategorical::new(["success", "failure"], 1.0)
            .observe("success", 3.0)
            .observe("failure", 1.0);
        assert_eq!(posterior.alphas, vec![4.0, 2.0]);
        assert_eq!(posterior.mean_distribution(), vec![2.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(
            posterior.sample_distribution(7),
            posterior.sample_distribution(7)
        );
    }

    #[test]
    fn normal_moments_track_reward_values() {
        let moments = NormalMoments::empty()
            .observe(2.0)
            .observe(4.0)
            .observe(6.0);
        assert_eq!(moments.weight, 3.0);
        assert!((moments.mean - 4.0).abs() < 1e-12);
        assert!((moments.variance() - 4.0).abs() < 1e-12);
        assert_eq!(moments.sample(99), moments.sample(99));
    }

    #[test]
    fn posterior_receipts_are_content_addressed() {
        let prior = BetaBernoulli::new(1.0, 1.0);
        let posterior = prior.observe_counts(3.0, 1.0);
        let receipt = PosteriorReceipt::new(
            "tool-success",
            "tool:a",
            prior.as_value(),
            json!({"successes": 3.0, "failures": 1.0}),
            posterior.as_value(),
            json!({"graph_version": 7}),
        );
        let same = PosteriorReceipt::new(
            "tool-success",
            "tool:a",
            prior.as_value(),
            json!({"successes": 3.0, "failures": 1.0}),
            posterior.as_value(),
            json!({"graph_version": 7}),
        );
        let changed = PosteriorReceipt::new(
            "tool-success",
            "tool:a",
            prior.as_value(),
            json!({"successes": 2.0, "failures": 2.0}),
            posterior.as_value(),
            json!({"graph_version": 7}),
        );

        assert_eq!(receipt.receipt_hash.len(), 64);
        assert_eq!(receipt.receipt_hash, same.receipt_hash);
        assert_ne!(receipt.receipt_hash, changed.receipt_hash);
    }

    #[test]
    fn seeds_are_content_stable() {
        assert_eq!(
            seed_from_parts(&["tenant", "task", "tool"]),
            seed_from_parts(&["tenant", "task", "tool"])
        );
        assert_ne!(
            seed_from_parts(&["tenant", "task", "tool"]),
            seed_from_parts(&["tenant", "task", "other-tool"])
        );
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-12,
            "left {left} differs from right {right}"
        );
    }
}
