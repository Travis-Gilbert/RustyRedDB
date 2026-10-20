use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use egg::{Id, RecExpr, Runner, SymbolLang};
use rustyred_thg_derivation::ontology::{default_registry, AlgebraicClass};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use theorem_bayes::BetaBernoulli;

use crate::graph::capability::{GraphStore, GraphVectorPayloadAccess};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord, Provenance};
use crate::graph::ttl::now_ms;
use crate::graph_csr::CsrGraph;
use crate::ppr_cache::cached_csr_personalized_pagerank;
use crate::state::stable_hash;
use crate::vector_payload::{
    decode_legacy_vector_array, read_vector_property, write_vector_property, VectorPropertyRef,
};

pub const EPISTEMIC_SHADOW_LABEL: &str = "EpistemicShadow";
/// Every edge type the shadow projection can emit for a valued relation.
///
/// [`write_shadow_relation`] emits the canonical name of the ontology row it
/// resolved, so a stored `Contradicts` or `Tension` relation writes that name
/// rather than `Undercuts`. Any consumer that enumerates shadow edge types has
/// to enumerate the same set: a hardcoded `[Undercuts, Supports]` silently
/// drops every preserved defeat row out of its view, which is how these edges
/// vanished from the shadow PPR topology and from the projection write guard.
pub fn valued_epistemic_edge_types() -> Vec<&'static str> {
    default_registry()
        .specs()
        .filter(|spec| {
            matches!(
                spec.class,
                AlgebraicClass::MonotonePositive | AlgebraicClass::Defeating
            )
        })
        .map(|spec| spec.canonical.as_str())
        .collect()
}

/// Whether `edge_type` names a valued epistemic row, resolving aliases and case
/// through the ontology.
///
/// [`valued_epistemic_edge_types`] lists canonical names, which is exactly what
/// the shadow projection writes, so a query over stored edges should ask for
/// those. A request arriving from outside carries whatever spelling its caller
/// chose, and the ontology carries aliases precisely because those spellings
/// vary. Testing exact membership let `contradicts`, `UNDERCUTS` and `attacks`
/// past the projection write guard and overwrite projection-maintained data.
pub fn is_valued_epistemic_edge_type(edge_type: &str) -> bool {
    default_registry().resolve(edge_type).is_some_and(|spec| {
        matches!(
            spec.class,
            AlgebraicClass::MonotonePositive | AlgebraicClass::Defeating
        )
    })
}

/// Whether `edge_type` names the equivalence or structural row, resolved the
/// same way.
///
/// The guard matched these two by exact string as well, so the same alias and
/// case hole applied to them.
pub fn is_structural_epistemic_edge_type(edge_type: &str) -> bool {
    default_registry().resolve(edge_type).is_some_and(|spec| {
        matches!(
            spec.class,
            AlgebraicClass::Equivalence | AlgebraicClass::Structural
        )
    })
}

pub const HAS_EPISTEMIC_SHADOW: &str = "HasEpistemicShadow";
pub const UNDERCUTS: &str = "Undercuts";
pub const EPISTEMIC_SUPPORTS: &str = "Supports";
pub const SAME_ECLASS: &str = "SameEClass";
pub const DEFAULT_EPISTEMIC_ENGINE_VERSION: &str = "epistemic-v1";
pub const STRUCTURAL_EPISTEMIC_ENGINE: &str = "rustyred-thg-core.structural_epistemic";
pub const LEARNED_EPISTEMIC_ENGINE: &str = "theseus.epistemic_enrichment";
pub const NLI_EPISTEMIC_ENGINE: &str = "theseus.nli_epistemic_enrichment";
pub const DEFAULT_NLI_MODEL_ID: &str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli";
pub const DEFAULT_CONNECTION_SCORER_MODEL_ID: &str = "theseus.learned_connection_scorer";
pub const DEFAULT_CONNECTION_FEATURE_VERSION: &str = "connection-features-v1";
pub const DEFAULT_CONNECTION_CALIBRATION_VERSION: &str = "connection-calibration-v1";
pub const EPISTEMIC_DETERMINISTIC_FALLBACK_ENV: &str =
    "THEOREM_EPISTEMIC_ALLOW_DETERMINISTIC_FALLBACK";
pub const EPISTEMIC_SCORER_ENDPOINT_ENV: &str = "THEOREM_EPISTEMIC_SCORER_ENDPOINT";
pub const EPISTEMIC_SCORER_MODEL_ENV: &str = "THEOREM_EPISTEMIC_SCORER_MODEL";
pub const EPISTEMIC_SCORER_CALIBRATION_ENV: &str = "THEOREM_EPISTEMIC_SCORER_CALIBRATION";
/// Engine tag for the e-graph dedup pass (Strand A phase 3 / cut 9). The
/// equivalence relation is a symbolic, model-free computation over the shadow
/// claim forms, so it carries the `structural` source kind on the edges it
/// writes.
pub const EGRAPH_EPISTEMIC_ENGINE: &str = "rustyred-thg-core.epistemic_egraph";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicSourceKind {
    Structural,
    Learned,
    Mixed,
}

impl EpistemicSourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Learned => "learned",
            Self::Mixed => "mixed",
        }
    }
}

impl Default for EpistemicSourceKind {
    fn default() -> Self {
        Self::Structural
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundedExtensionStatus {
    In,
    Out,
    Undecided,
}

impl GroundedExtensionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Undecided => "undecided",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PredictedEdgePointer {
    pub target_content_id: String,
    pub relation: String,
    pub confidence: f64,
    #[serde(default = "default_true")]
    pub quarantine: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SourceReliability {
    pub alpha: f64,
    pub beta: f64,
    pub mean: f64,
}

impl SourceReliability {
    pub fn from_posterior(posterior: BetaBernoulli) -> Self {
        Self {
            alpha: posterior.alpha,
            beta: posterior.beta,
            mean: posterior.mean(),
        }
    }

    pub fn posterior(&self) -> BetaBernoulli {
        BetaBernoulli::new(self.alpha, self.beta)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VerificationReliabilityObservation {
    #[serde(default)]
    pub source_id: String,
    pub corroborated: bool,
    #[serde(default = "default_observation_weight")]
    pub weight: f64,
}

pub fn source_reliability_from_verification_outcomes<I>(
    prior_alpha: f64,
    prior_beta: f64,
    observations: I,
) -> SourceReliability
where
    I: IntoIterator<Item = VerificationReliabilityObservation>,
{
    let mut posterior = BetaBernoulli::new(prior_alpha, prior_beta);
    for observation in observations {
        posterior = posterior.observe_weighted(
            if observation.corroborated { 1.0 } else { 0.0 },
            observation.weight,
        );
    }
    SourceReliability::from_posterior(posterior)
}

pub fn effective_confidence(
    base_confidence: f64,
    source_reliability: Option<&SourceReliability>,
) -> f64 {
    let base = normalized_confidence(base_confidence);
    source_reliability
        .map(|reliability| base * reliability.mean.clamp(0.0, 1.0))
        .unwrap_or(base)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicFieldProvenance {
    pub source_kind: EpistemicSourceKind,
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
}

impl EpistemicFieldProvenance {
    pub fn structural(config: &StructuralEpistemicConfig) -> Self {
        Self {
            source_kind: EpistemicSourceKind::Structural,
            engine: config.engine.clone(),
            engine_version: config.engine_version.clone(),
            computed_at: config.computed_at,
        }
    }

    pub fn learned(config: &EpistemicCronInput) -> Self {
        Self {
            source_kind: EpistemicSourceKind::Learned,
            engine: config.engine.clone(),
            engine_version: config.engine_version.clone(),
            computed_at: config.computed_at,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EpistemicReadout {
    pub shadows: Vec<EpistemicShadowReadout>,
    pub contradictions: Vec<EpistemicRelationReadout>,
    pub unsupported: Vec<String>,
    pub orphans: Vec<String>,
    pub chokepoints: Vec<EpistemicChokepoint>,
    pub checked_pair_count: usize,
    pub candidate_pair_bound: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicShadowReadout {
    pub content_node_id: String,
    pub shadow_node_id: String,
    pub grounded_extension_status: GroundedExtensionStatus,
    pub support_in_degree: u64,
    pub attack_in_degree: u64,
    pub unsupported_leaf: bool,
    pub orphan: bool,
    pub bridge_score: f64,
    pub contradiction_cycle_id: Option<String>,
    pub predicted_edges: Vec<PredictedEdgePointer>,
    pub completion_confidence: Option<f64>,
    pub structural_role_vector: Vec<f32>,
    pub source_reliability: Option<SourceReliability>,
    pub community_id: Option<String>,
    pub source_kind: EpistemicSourceKind,
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
    pub quarantine: bool,
    pub field_provenance: BTreeMap<String, EpistemicFieldProvenance>,
    /// Set when this shadow has been collapsed onto another shadow as a member
    /// of a `SameEClass` equivalence class (Strand A phase 3 / cut 9). `None`
    /// means the shadow is either a class representative or was never deduped.
    #[serde(default)]
    pub same_eclass: Option<SameEClassRef>,
}

/// A back-reference from a deduped shadow to the `SameEClass` representative it
/// was collapsed onto. Read off the member shadow's `SameEClass` out-edge, so it
/// is always consistent with the edge layer that `epistemic_shadow_ppr`
/// traverses.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SameEClassRef {
    pub representative_shadow_id: String,
    pub class_id: String,
    pub canonical_form: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicRelationKind {
    Supports,
    Undercuts,
}

impl EpistemicRelationKind {
    pub fn edge_type(&self) -> &'static str {
        match self {
            Self::Supports => EPISTEMIC_SUPPORTS,
            Self::Undercuts => UNDERCUTS,
        }
    }

    /// The relation kind an ontology class collapses to, or `None` for a class
    /// that carries no epistemic relation.
    ///
    /// The collapse is deliberate: the grounded semantics only asks whether an
    /// edge supports or defeats, so all three `Defeating` rows answer the same
    /// way. This is the single place that mapping is written down.
    pub fn for_class(class: AlgebraicClass) -> Option<Self> {
        match class {
            AlgebraicClass::MonotonePositive => Some(Self::Supports),
            AlgebraicClass::Defeating => Some(Self::Undercuts),
            AlgebraicClass::Equivalence | AlgebraicClass::Structural => None,
        }
    }

    /// The `record.rs` token this relation kind writes onto an edge.
    ///
    /// `Undercuts` maps to [`EpistemicType::Undercuts`], not to
    /// `Contradicts`. Conflating them erased the Pollock distinction: an
    /// undercut attacks the inference, a contradiction attacks the conclusion.
    /// Legacy edges already written as `Contradicts` stay valid, because the
    /// edge ontology classes both as `Defeating` and every consumer that
    /// treated them identically continues to.
    fn epistemic_type(&self) -> EpistemicType {
        match self {
            Self::Supports => EpistemicType::Supports,
            Self::Undercuts => EpistemicType::Undercuts,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ConnectionFeatures {
    pub from_content_id: String,
    pub to_content_id: String,
    #[serde(default)]
    pub premise: String,
    #[serde(default)]
    pub hypothesis: String,
    #[serde(default)]
    pub candidate_evidence: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<String>,
    #[serde(default)]
    pub nli_entailment_score: f64,
    #[serde(default)]
    pub nli_neutral_score: f64,
    #[serde(default)]
    pub nli_contradiction_score: f64,
    #[serde(default)]
    pub support_in_degree: u64,
    #[serde(default)]
    pub attack_in_degree: u64,
    #[serde(default)]
    pub bridge_score: f64,
    #[serde(default)]
    pub source_reliability_mean: Option<f64>,
    #[serde(default)]
    pub graph_edge_count: usize,
    #[serde(default = "default_connection_feature_version_string")]
    pub feature_version: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ConnectionScore {
    #[serde(default)]
    pub kind: Option<EpistemicRelationKind>,
    pub score: f64,
    pub confidence: f64,
    pub model_id: String,
    pub calibration_version: String,
    pub feature_version: String,
    #[serde(default)]
    pub evidence: String,
}

pub trait ConnectionScorer {
    fn score(
        &self,
        features: &ConnectionFeatures,
    ) -> Result<ConnectionScore, EpistemicEnrichmentError>;

    fn score_batch(
        &self,
        features: &[ConnectionFeatures],
    ) -> Result<Vec<ConnectionScore>, EpistemicEnrichmentError> {
        features.iter().map(|feature| self.score(feature)).collect()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LearnedConnectionScorerConfig {
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_connection_scorer_model_id")]
    pub model_id: String,
    #[serde(default = "default_connection_calibration_version_string")]
    pub calibration_version: String,
    #[serde(default = "default_connection_feature_version_string")]
    pub feature_version: String,
}

impl Default for LearnedConnectionScorerConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var(EPISTEMIC_SCORER_ENDPOINT_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty()),
            model_id: std::env::var(EPISTEMIC_SCORER_MODEL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(default_connection_scorer_model_id),
            calibration_version: std::env::var(EPISTEMIC_SCORER_CALIBRATION_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(default_connection_calibration_version_string),
            feature_version: default_connection_feature_version_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LearnedConnectionScorer {
    pub config: LearnedConnectionScorerConfig,
}

impl LearnedConnectionScorer {
    pub fn new(config: LearnedConnectionScorerConfig) -> Self {
        Self { config }
    }
}

impl ConnectionScorer for LearnedConnectionScorer {
    fn score(
        &self,
        _features: &ConnectionFeatures,
    ) -> Result<ConnectionScore, EpistemicEnrichmentError> {
        let Some(endpoint) = self.config.endpoint.as_deref() else {
            return Err(EpistemicEnrichmentError::new(
                "learned_scorer_unavailable",
                format!(
                    "learned connection scorer is the default; configure {EPISTEMIC_SCORER_ENDPOINT_ENV} or inject a scorer"
                ),
            ));
        };
        Err(EpistemicEnrichmentError::new(
            "learned_scorer_transport_unbound",
            format!(
                "learned scorer endpoint {endpoint} is configured, but this core crate only defines the RunPod scoring contract"
            ),
        ))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LearnedConnectionScorerPair {
    pub left_content_id: String,
    pub right_content_id: String,
    pub premise: String,
    pub hypothesis: String,
    pub features: ConnectionFeatures,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LearnedConnectionScorerRequest {
    pub model_id: String,
    pub calibration_version: String,
    pub feature_version: String,
    pub pairs: Vec<LearnedConnectionScorerPair>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LearnedConnectionScorerResponse {
    pub scores: Vec<ConnectionScore>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NliPairInput {
    pub left_content_id: String,
    pub right_content_id: String,
    pub premise: String,
    pub hypothesis: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NliVerdict {
    pub entailment: f64,
    pub neutral: f64,
    pub contradiction: f64,
    pub model_id: String,
}

pub trait NliClassifier {
    fn classify_batch(
        &self,
        pairs: &[NliPairInput],
    ) -> Result<Vec<NliVerdict>, EpistemicEnrichmentError>;
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NliPairSelectionConfig {
    pub max_pairs: usize,
}

impl Default for NliPairSelectionConfig {
    fn default() -> Self {
        Self { max_pairs: 64 }
    }
}

#[derive(Clone, Debug)]
pub struct NliEpistemicEnricher<C, S> {
    classifier: C,
    scorer: S,
    pair_selection: NliPairSelectionConfig,
}

impl<C, S> NliEpistemicEnricher<C, S> {
    pub fn new(classifier: C, scorer: S, pair_selection: NliPairSelectionConfig) -> Self {
        Self {
            classifier,
            scorer,
            pair_selection,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicRelationInput {
    pub from_content_id: String,
    pub to_content_id: String,
    pub kind: EpistemicRelationKind,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub evidence: String,
    #[serde(default)]
    pub source_kind: EpistemicSourceKind,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub calibration_version: Option<String>,
    #[serde(default)]
    pub feature_version: Option<String>,
    #[serde(default)]
    pub connection_features: Option<ConnectionFeatures>,
    /// Canonical ontology row of the stored edge this relation came from.
    ///
    /// [`EpistemicRelationKind`] is lossy on purpose: `Contradicts`, `Tension`,
    /// and `Undercuts` all classify as `Defeating` and collapse onto the single
    /// `Undercuts` kind. That is the right granularity for the grounded
    /// semantics, but it means the kind alone cannot say which edge type to
    /// write back, so the pass would rewrite a `Contradicts` edge as an
    /// undercut. Relations minted from something other than a stored edge
    /// carry `None` and fall back to the kind's own edge type.
    #[serde(default)]
    pub canonical_edge_type: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicCandidatePair {
    pub left_content_id: String,
    pub right_content_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicRelationReadout {
    pub from_content_id: String,
    pub to_content_id: String,
    pub from_shadow_id: String,
    pub to_shadow_id: String,
    pub kind: EpistemicRelationKind,
    pub confidence: f64,
    pub evidence: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicChokepoint {
    pub content_node_id: String,
    pub shadow_node_id: String,
    pub bridge_score: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StructuralEpistemicConfig {
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
    pub support_edge_types: Vec<String>,
    pub attack_edge_types: Vec<String>,
    pub candidate_top_k: usize,
}

impl Default for StructuralEpistemicConfig {
    /// Defaults come from [`default_registry`] filtered by algebraic class,
    /// not from hardcoded vectors.
    ///
    /// This is where the intended behavior change lands: `Tension` is a
    /// `Defeating` row, so the default attack list now includes it and an
    /// unresolved tension counts as epistemic pressure. A config that spells
    /// the old lists out explicitly keeps the old scope, so existing callers
    /// are unaffected.
    fn default() -> Self {
        let registry = default_registry();
        Self {
            engine: STRUCTURAL_EPISTEMIC_ENGINE.to_string(),
            engine_version: DEFAULT_EPISTEMIC_ENGINE_VERSION.to_string(),
            computed_at: now_ms(),
            support_edge_types: registry.tokens_in_class(AlgebraicClass::MonotonePositive),
            attack_edge_types: registry.tokens_in_class(AlgebraicClass::Defeating),
            candidate_top_k: 8,
        }
    }
}

/// The edge types one [`StructuralEpistemicConfig`] admits, resolved through
/// the edge ontology.
///
/// The configured lists select *which* edge types the pass considers; the
/// registry decides *how* each is classified. Splitting those two jobs is what
/// removes the third copy of the classification: a token can no longer be
/// support in one list and attack in another.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConfiguredEdgeClasses {
    admitted: BTreeSet<String>,
}

impl ConfiguredEdgeClasses {
    /// Resolve a config's edge-type lists through [`default_registry`].
    ///
    /// A string the registry does not know is a construction error naming the
    /// string, never a silently ignored entry.
    pub fn from_config(config: &StructuralEpistemicConfig) -> GraphStoreResult<Self> {
        let registry = default_registry();
        let mut admitted = BTreeSet::new();
        for token in config
            .support_edge_types
            .iter()
            .chain(config.attack_edge_types.iter())
        {
            let Some(spec) = registry.resolve(token) else {
                return Err(GraphStoreError::new(
                    "unknown_epistemic_edge_type",
                    format!(
                        "edge type {token:?} is not declared in the edge ontology; declare a                          registry row with an algebraic class before configuring it"
                    ),
                ));
            };
            admitted.insert(spec.canonical.clone());
        }
        Ok(Self { admitted })
    }

    /// The relation kind an edge type contributes, or `None` when the type is
    /// outside this config's scope or is never valued.
    pub fn kind_of(&self, edge_type: &str) -> Option<EpistemicRelationKind> {
        let spec = default_registry().resolve(edge_type)?;
        if !self.admitted.contains(&spec.canonical) {
            return None;
        }
        EpistemicRelationKind::for_class(spec.class)
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StructuralEpistemicInput {
    pub batch_node_ids: Vec<String>,
    #[serde(default)]
    pub candidate_pairs: Vec<EpistemicCandidatePair>,
    #[serde(default)]
    pub explicit_relations: Vec<EpistemicRelationInput>,
    #[serde(default)]
    pub config: StructuralEpistemicConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicEnrichmentMode {
    Delta,
    Full,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct UserSubgraph {
    pub nodes: Vec<NodeRecord>,
    pub edges: Vec<EdgeRecord>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EpistemicAnnotations {
    pub annotations: Vec<EpistemicAnnotation>,
    #[serde(default)]
    pub support_relations: Vec<EpistemicRelationInput>,
    #[serde(default)]
    pub attack_relations: Vec<EpistemicRelationInput>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EpistemicAnnotation {
    pub content_node_id: String,
    #[serde(default)]
    pub predicted_edges: Vec<PredictedEdgePointer>,
    #[serde(default)]
    pub completion_confidence: Option<f64>,
    #[serde(default)]
    pub structural_role_vector: Vec<f32>,
    #[serde(default)]
    pub source_reliability: Option<SourceReliability>,
    #[serde(default)]
    pub community_id: Option<String>,
    #[serde(default)]
    pub grounded_extension_status: Option<GroundedExtensionStatus>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicCronInput {
    pub content_node_ids: Vec<String>,
    pub mode: EpistemicEnrichmentMode,
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
    pub density_floor: f64,
}

impl Default for EpistemicCronInput {
    fn default() -> Self {
        Self {
            content_node_ids: Vec::new(),
            mode: EpistemicEnrichmentMode::Delta,
            engine: LEARNED_EPISTEMIC_ENGINE.to_string(),
            engine_version: DEFAULT_EPISTEMIC_ENGINE_VERSION.to_string(),
            computed_at: now_ms(),
            density_floor: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EpistemicCronReport {
    pub attempted: bool,
    pub no_op: bool,
    pub grpc_ok: bool,
    pub skipped_reason: String,
    pub annotations_received: usize,
    pub shadows_written: usize,
    pub shadow_edges_written: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EpistemicEnrichmentError {
    pub code: String,
    pub message: String,
}

impl EpistemicEnrichmentError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

pub trait EpistemicEnricher {
    fn enrich(
        &self,
        subgraph: UserSubgraph,
        mode: EpistemicEnrichmentMode,
    ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError>;
}

impl<C, S> EpistemicEnricher for NliEpistemicEnricher<C, S>
where
    C: NliClassifier,
    S: ConnectionScorer,
{
    fn enrich(
        &self,
        subgraph: UserSubgraph,
        _mode: EpistemicEnrichmentMode,
    ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError> {
        let pairs = select_nli_pairs(&subgraph, &self.pair_selection);
        if pairs.is_empty() {
            return Ok(EpistemicAnnotations::default());
        }
        let verdicts = self.classifier.classify_batch(&pairs)?;
        if verdicts.len() != pairs.len() {
            return Err(EpistemicEnrichmentError::new(
                "nli_result_count_mismatch",
                format!(
                    "classifier returned {} verdicts for {} pairs",
                    verdicts.len(),
                    pairs.len()
                ),
            ));
        }

        let graph_signals = graph_signal_index(&subgraph);
        let features = pairs
            .iter()
            .zip(verdicts.iter())
            .map(|(pair, verdict)| connection_features_for_pair(pair, verdict, &graph_signals))
            .collect::<Vec<_>>();
        let scores = self.scorer.score_batch(&features)?;
        if scores.len() != features.len() {
            return Err(EpistemicEnrichmentError::new(
                "connection_score_count_mismatch",
                format!(
                    "scorer returned {} scores for {} feature rows",
                    scores.len(),
                    features.len()
                ),
            ));
        }
        let mut support_relations = Vec::new();
        let mut attack_relations = Vec::new();
        for ((pair, features), score) in pairs.into_iter().zip(features.into_iter()).zip(scores) {
            let Some(kind) = score.kind.clone() else {
                continue;
            };
            let relation = EpistemicRelationInput {
                from_content_id: pair.left_content_id.clone(),
                to_content_id: pair.right_content_id.clone(),
                kind: kind.clone(),
                confidence: normalized_confidence(score.confidence),
                evidence: if score.evidence.trim().is_empty() {
                    format!("learned_connection_score: {}", score.model_id)
                } else {
                    score.evidence.clone()
                },
                source_kind: EpistemicSourceKind::Learned,
                score: Some(score.score.clamp(0.0, 1.0)),
                model_id: Some(score.model_id),
                calibration_version: Some(score.calibration_version),
                feature_version: Some(score.feature_version),
                connection_features: Some(features),
                canonical_edge_type: None,
            };
            match kind {
                EpistemicRelationKind::Supports => support_relations.push(relation),
                EpistemicRelationKind::Undercuts => attack_relations.push(relation),
            }
        }
        Ok(EpistemicAnnotations {
            support_relations,
            attack_relations,
            ..EpistemicAnnotations::default()
        })
    }
}

pub fn epistemic_shadow_node_id(
    content_node_id: &str,
    engine_version: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "epistemic:shadow:{}",
        stable_hash(json!([content_node_id, engine_version]))?
    ))
}

pub fn has_epistemic_shadow_edge_id(
    content_node_id: &str,
    shadow_node_id: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "epistemic:has_shadow:{}",
        stable_hash(json!([content_node_id, shadow_node_id]))?
    ))
}

pub fn epistemic_shadow_edge_id(
    kind: EpistemicRelationKind,
    from_shadow_id: &str,
    to_shadow_id: &str,
    engine_version: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "epistemic:edge:{}:{}",
        kind.edge_type(),
        stable_hash(json!([from_shadow_id, to_shadow_id, engine_version]))?
    ))
}

pub fn same_eclass_edge_id(
    from_shadow_id: &str,
    to_shadow_id: &str,
    engine_version: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "epistemic:edge:{}:{}",
        SAME_ECLASS,
        stable_hash(json!([from_shadow_id, to_shadow_id, engine_version]))?
    ))
}

pub fn structural_epistemic_pass<S: GraphStore + GraphVectorPayloadAccess>(
    store: &mut S,
    input: StructuralEpistemicInput,
) -> GraphStoreResult<EpistemicReadout> {
    let config = input.config;
    let edge_classes = ConfiguredEdgeClasses::from_config(&config)?;
    let mut node_ids = input
        .batch_node_ids
        .into_iter()
        .filter(|id| !id.trim().is_empty())
        .collect::<BTreeSet<_>>();
    for pair in &input.candidate_pairs {
        node_ids.insert(pair.left_content_id.clone());
        node_ids.insert(pair.right_content_id.clone());
    }
    for relation in &input.explicit_relations {
        node_ids.insert(relation.from_content_id.clone());
        node_ids.insert(relation.to_content_id.clone());
    }

    let nodes = node_ids
        .iter()
        .filter_map(|id| store.get_node(id).cloned())
        .collect::<Vec<_>>();
    let node_set = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    let existing_edges = induced_edges(store, &node_set);

    let mut relations = existing_epistemic_relations(&existing_edges, &edge_classes);
    let checked_pair_count = input.candidate_pairs.len();
    for pair in input.candidate_pairs {
        if deterministic_pair_fallback_enabled() {
            if let Some(relation) = infer_pair_relation(store, &pair) {
                relations.push(relation);
            }
        }
    }
    relations.extend(input.explicit_relations);
    dedupe_relations(&mut relations);

    let undirected = undirected_adjacency(&node_set, &existing_edges, &relations);
    let bridge_scores = bridge_scores(&node_set, &undirected);
    let cycle_ids = contradiction_cycles(&relations)?;
    let mut support_in = HashMap::<String, u64>::new();
    let mut attack_in = HashMap::<String, u64>::new();
    for relation in &relations {
        match relation.kind {
            EpistemicRelationKind::Supports => {
                *support_in
                    .entry(relation.to_content_id.clone())
                    .or_insert(0) += 1;
            }
            EpistemicRelationKind::Undercuts => {
                *attack_in.entry(relation.to_content_id.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut readout = EpistemicReadout {
        checked_pair_count,
        candidate_pair_bound: node_set.len().saturating_mul(config.candidate_top_k),
        ..EpistemicReadout::default()
    };
    for node in &nodes {
        let support = support_in.get(&node.id).copied().unwrap_or(0);
        let attack = attack_in.get(&node.id).copied().unwrap_or(0);
        let orphan = undirected
            .get(&node.id)
            .map(|neighbors| neighbors.is_empty())
            .unwrap_or(true);
        let unsupported_leaf = support == 0;
        let contradiction_cycle_id = cycle_ids.get(&node.id).cloned();
        let grounded_extension_status = if contradiction_cycle_id.is_some() {
            GroundedExtensionStatus::Undecided
        } else if attack > 0 {
            GroundedExtensionStatus::Out
        } else {
            GroundedExtensionStatus::In
        };
        let bridge_score = bridge_scores.get(&node.id).copied().unwrap_or(0.0);
        let shadow = write_structural_shadow(
            store,
            node,
            &config,
            grounded_extension_status,
            support,
            attack,
            unsupported_leaf,
            orphan,
            bridge_score,
            contradiction_cycle_id,
        )?;
        if unsupported_leaf {
            readout.unsupported.push(node.id.clone());
        }
        if orphan {
            readout.orphans.push(node.id.clone());
        }
        if bridge_score > 0.0 {
            readout.chokepoints.push(EpistemicChokepoint {
                content_node_id: node.id.clone(),
                shadow_node_id: shadow.shadow_node_id.clone(),
                bridge_score,
            });
        }
        readout.shadows.push(shadow);
    }

    for relation in &relations {
        let from_shadow =
            epistemic_shadow_node_id(&relation.from_content_id, &config.engine_version)?;
        let to_shadow = epistemic_shadow_node_id(&relation.to_content_id, &config.engine_version)?;
        if store.get_node(&from_shadow).is_none() || store.get_node(&to_shadow).is_none() {
            continue;
        }
        write_shadow_relation(
            store,
            relation,
            &from_shadow,
            &to_shadow,
            &config,
            relation.canonical_edge_type.as_deref(),
        )?;
        if relation.kind == EpistemicRelationKind::Undercuts {
            readout.contradictions.push(EpistemicRelationReadout {
                from_content_id: relation.from_content_id.clone(),
                to_content_id: relation.to_content_id.clone(),
                from_shadow_id: from_shadow,
                to_shadow_id: to_shadow,
                kind: relation.kind.clone(),
                confidence: relation.confidence,
                evidence: relation.evidence.clone(),
            });
        }
    }

    readout
        .shadows
        .sort_by(|left, right| left.content_node_id.cmp(&right.content_node_id));
    readout.unsupported.sort();
    readout.orphans.sort();
    readout.chokepoints.sort_by(|left, right| {
        right
            .bridge_score
            .partial_cmp(&left.bridge_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    readout.contradictions.sort_by(|left, right| {
        left.from_content_id
            .cmp(&right.from_content_id)
            .then_with(|| left.to_content_id.cmp(&right.to_content_id))
    });
    Ok(readout)
}

pub fn read_epistemic_shadow<S: GraphStore + GraphVectorPayloadAccess>(
    store: &S,
    content_node_id: &str,
) -> Option<EpistemicShadowReadout> {
    try_read_epistemic_shadow(store, content_node_id)
        .ok()
        .flatten()
}

pub fn try_read_epistemic_shadow<S: GraphStore + GraphVectorPayloadAccess>(
    store: &S,
    content_node_id: &str,
) -> GraphStoreResult<Option<EpistemicShadowReadout>> {
    for hit in store
        .neighbors(NeighborQuery::out(content_node_id).with_edge_type(HAS_EPISTEMIC_SHADOW))
        .into_iter()
    {
        let Some(shadow) = store.get_node(&hit.node_id) else {
            continue;
        };
        if shadow
            .labels
            .iter()
            .any(|label| label == EPISTEMIC_SHADOW_LABEL)
        {
            let mut readout = shadow_readout_from_node(store, content_node_id, shadow)?;
            readout.same_eclass = read_same_eclass(store, &readout.shadow_node_id);
            return Ok(Some(readout));
        }
    }
    Ok(None)
}

/// Read the `SameEClass` representative a shadow was collapsed onto, if any.
/// Returns `None` for a class representative (it has no outgoing `SameEClass`
/// edge) or a shadow that was never deduped.
pub fn read_same_eclass<S: GraphStore>(store: &S, shadow_node_id: &str) -> Option<SameEClassRef> {
    let hit = store
        .neighbors(NeighborQuery::out(shadow_node_id).with_edge_type(SAME_ECLASS))
        .into_iter()
        .next()?;
    let (class_id, canonical_form) = store
        .get_edge(&hit.edge_id)
        .map(|edge| {
            (
                prop_str(&edge.properties, "class_id").unwrap_or_default(),
                prop_str(&edge.properties, "canonical_form").unwrap_or_default(),
            )
        })
        .unwrap_or_default();
    Some(SameEClassRef {
        representative_shadow_id: hit.node_id,
        class_id,
        canonical_form,
    })
}

pub fn epistemic_shadow_ppr<S: GraphStore + GraphVectorPayloadAccess>(
    store: &S,
    seeds: &HashMap<String, f64>,
    top_k: usize,
    alpha: f64,
    epsilon: f64,
    max_pushes: usize,
) -> GraphStoreResult<Vec<(String, f64)>> {
    let mut shadow_seeds = HashMap::new();
    for (seed, weight) in seeds {
        if store
            .get_node(seed)
            .map(|node| {
                node.labels
                    .iter()
                    .any(|label| label == EPISTEMIC_SHADOW_LABEL)
            })
            .unwrap_or(false)
        {
            shadow_seeds.insert(seed.clone(), *weight);
        } else if let Some(shadow) = try_read_epistemic_shadow(store, seed)? {
            shadow_seeds.insert(shadow.shadow_node_id, *weight);
        }
    }
    if shadow_seeds.is_empty() {
        return Ok(Vec::new());
    }

    let shadow_nodes = store
        .query_nodes(NodeQuery::label(EPISTEMIC_SHADOW_LABEL).with_limit(100_000))
        .into_iter()
        .map(|node| node.id)
        .collect::<BTreeSet<_>>();
    let mut shadow_edges = Vec::new();
    for node_id in &shadow_nodes {
        // Every valued row plus the equivalence edge. Hardcoding
        // [UNDERCUTS, EPISTEMIC_SUPPORTS, SAME_ECLASS] dropped preserved
        // Contradicts and Tension edges out of the CSR topology and its cache
        // identity, changing rankings and disconnecting relationships this same
        // path traversed before they stopped being rewritten as Undercuts.
        for edge_type in valued_epistemic_edge_types()
            .into_iter()
            .chain(std::iter::once(SAME_ECLASS))
        {
            for hit in store
                .neighbors(NeighborQuery::out(node_id).with_edge_type(edge_type))
                .into_iter()
            {
                if shadow_nodes.contains(&hit.node_id) {
                    shadow_edges.push((
                        node_id.clone(),
                        hit.node_id,
                        hit.confidence.unwrap_or(1.0).max(0.0),
                    ));
                }
            }
        }
    }
    let topology_identity = stable_hash((
        "epistemic-shadow-ppr-topology-v1",
        &shadow_nodes,
        &shadow_edges,
    ))?;
    let cache_scope = format!("epistemic-shadow-ppr:{topology_identity}");
    let csr = CsrGraph::from_edges_and_nodes(shadow_edges, shadow_nodes.iter());
    let mut ranked = cached_csr_personalized_pagerank(
        &cache_scope,
        store.stats().version,
        &csr,
        &shadow_seeds,
        alpha,
        epsilon,
        max_pushes,
    )?
    .into_iter()
    .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k.max(1));
    Ok(ranked)
}

pub fn run_epistemic_cron_pass<S: GraphStore + GraphVectorPayloadAccess, E: EpistemicEnricher>(
    store: &mut S,
    input: EpistemicCronInput,
    enricher: &E,
) -> GraphStoreResult<EpistemicCronReport> {
    let subgraph = compile_user_subgraph(store, &input.content_node_ids);
    if subgraph.nodes.is_empty() {
        return Ok(EpistemicCronReport {
            no_op: true,
            skipped_reason: "empty_subgraph".to_string(),
            ..EpistemicCronReport::default()
        });
    }
    if input.density_floor > 0.0 {
        let density = edge_density(&subgraph);
        if density < input.density_floor {
            return Ok(EpistemicCronReport {
                no_op: true,
                skipped_reason: format!("edge_density_below_floor:{density:.6}"),
                ..EpistemicCronReport::default()
            });
        }
    }

    let annotations = match enricher.enrich(subgraph, input.mode.clone()) {
        Ok(annotations) => annotations,
        Err(err) => {
            return Ok(EpistemicCronReport {
                attempted: true,
                no_op: true,
                grpc_ok: false,
                skipped_reason: format!("{}: {}", err.code, err.message),
                ..EpistemicCronReport::default()
            });
        }
    };

    let received = annotations.annotations.len();
    let mut report = EpistemicCronReport {
        attempted: true,
        grpc_ok: true,
        annotations_received: received,
        ..EpistemicCronReport::default()
    };
    for annotation in &annotations.annotations {
        if write_learned_shadow(store, annotation, &input)?.is_some() {
            report.shadows_written += 1;
        }
    }
    let mut relations = annotations.support_relations;
    relations.extend(annotations.attack_relations);
    dedupe_relations(&mut relations);
    for relation in &relations {
        let from_shadow =
            epistemic_shadow_node_id(&relation.from_content_id, &input.engine_version)?;
        let to_shadow = epistemic_shadow_node_id(&relation.to_content_id, &input.engine_version)?;
        if store.get_node(&from_shadow).is_none() || store.get_node(&to_shadow).is_none() {
            continue;
        }
        let config = StructuralEpistemicConfig {
            engine: input.engine.clone(),
            engine_version: input.engine_version.clone(),
            computed_at: input.computed_at,
            ..StructuralEpistemicConfig::default()
        };
        write_shadow_relation(
            store,
            relation,
            &from_shadow,
            &to_shadow,
            &config,
            // An enricher can return a defeating relation whose canonical row
            // is Contradicts or Tension. Passing None here threw that away and
            // persisted Undercuts, losing the distinction the structural pass
            // preserves.
            relation.canonical_edge_type.as_deref(),
        )?;
        report.shadow_edges_written += 1;
    }
    Ok(report)
}

fn write_structural_shadow<S: GraphStore + GraphVectorPayloadAccess>(
    store: &mut S,
    content_node: &NodeRecord,
    config: &StructuralEpistemicConfig,
    grounded_extension_status: GroundedExtensionStatus,
    support_in_degree: u64,
    attack_in_degree: u64,
    unsupported_leaf: bool,
    orphan: bool,
    bridge_score: f64,
    contradiction_cycle_id: Option<String>,
) -> GraphStoreResult<EpistemicShadowReadout> {
    let shadow_id = epistemic_shadow_node_id(&content_node.id, &config.engine_version)?;
    let existing = store.get_node(&shadow_id).cloned();
    let mut properties = existing
        .as_ref()
        .map(|node| node.properties.clone())
        .unwrap_or_else(|| json!({}));
    let provenance = EpistemicFieldProvenance::structural(config);
    set_field_provenance(
        &mut properties,
        &[
            "grounded_extension_status",
            "support_in_degree",
            "attack_in_degree",
            "unsupported_leaf",
            "orphan",
            "bridge_score",
            "contradiction_cycle_id",
        ],
        &provenance,
    );
    let source_kind = if has_learned_fields(&properties) {
        EpistemicSourceKind::Mixed
    } else {
        EpistemicSourceKind::Structural
    };
    let object = ensure_object(&mut properties);
    object.insert("content_node_id".to_string(), json!(content_node.id));
    object.insert(
        "grounded_extension_status".to_string(),
        json!(grounded_extension_status.as_str()),
    );
    object.insert("support_in_degree".to_string(), json!(support_in_degree));
    object.insert("attack_in_degree".to_string(), json!(attack_in_degree));
    object.insert("unsupported_leaf".to_string(), json!(unsupported_leaf));
    object.insert("orphan".to_string(), json!(orphan));
    object.insert("bridge_score".to_string(), json!(round6(bridge_score)));
    object.insert(
        "contradiction_cycle_id".to_string(),
        contradiction_cycle_id
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    object.insert("source_kind".to_string(), json!(source_kind.as_str()));
    object.insert("engine".to_string(), json!(config.engine));
    object.insert("engine_version".to_string(), json!(config.engine_version));
    object.insert("computed_at".to_string(), json!(config.computed_at));
    object.insert("quarantine".to_string(), json!(true));
    object
        .entry("predicted_edges".to_string())
        .or_insert_with(|| json!([]));
    if let Some(tenant) = prop_str(&content_node.properties, "tenant_id") {
        object
            .entry("tenant_id".to_string())
            .or_insert(json!(tenant));
    }
    if let Some(repo) = prop_str(&content_node.properties, "repo_id") {
        object.entry("repo_id".to_string()).or_insert(json!(repo));
    }

    let mut shadow = NodeRecord::new(&shadow_id, [EPISTEMIC_SHADOW_LABEL], properties);
    migrate_legacy_structural_role_vector(store, &mut shadow)?;
    store.upsert_node(shadow)?;
    store.upsert_edge(EdgeRecord::new(
        has_epistemic_shadow_edge_id(&content_node.id, &shadow_id)?,
        &content_node.id,
        HAS_EPISTEMIC_SHADOW,
        &shadow_id,
        json!({
            "engine_version": config.engine_version,
            "computed_at": config.computed_at,
            "source": STRUCTURAL_EPISTEMIC_ENGINE,
        }),
    ))?;
    let shadow = store.get_node(&shadow_id).ok_or_else(|| {
        GraphStoreError::new(
            "epistemic_shadow_write_failed",
            format!("shadow {shadow_id} was not readable after write"),
        )
    })?;
    shadow_readout_from_node(store, &content_node.id, shadow)
}

fn write_learned_shadow<S: GraphStore + GraphVectorPayloadAccess>(
    store: &mut S,
    annotation: &EpistemicAnnotation,
    config: &EpistemicCronInput,
) -> GraphStoreResult<Option<EpistemicShadowReadout>> {
    if annotation.content_node_id.trim().is_empty() {
        return Ok(None);
    }
    let Some(content_node) = store.get_node(&annotation.content_node_id).cloned() else {
        return Ok(None);
    };
    let shadow_id = epistemic_shadow_node_id(&annotation.content_node_id, &config.engine_version)?;
    let existing = store.get_node(&shadow_id).cloned();
    let mut properties = existing
        .as_ref()
        .map(|node| node.properties.clone())
        .unwrap_or_else(|| json!({}));
    let provenance = EpistemicFieldProvenance::learned(config);
    set_field_provenance(
        &mut properties,
        &[
            "predicted_edges",
            "completion_confidence",
            "structural_role_vector",
            "source_reliability",
            "community_id",
        ],
        &provenance,
    );
    let object = ensure_object(&mut properties);
    object.insert(
        "content_node_id".to_string(),
        json!(annotation.content_node_id),
    );
    object.insert(
        "predicted_edges".to_string(),
        serde_json::to_value(&annotation.predicted_edges).unwrap_or_else(|_| json!([])),
    );
    object.insert(
        "completion_confidence".to_string(),
        annotation
            .completion_confidence
            .and_then(|value| serde_json::Number::from_f64(value.clamp(0.0, 1.0)))
            .map(Value::Number)
            .unwrap_or(Value::Null),
    );
    object.insert(
        "source_reliability".to_string(),
        annotation
            .source_reliability
            .as_ref()
            .map(|value| serde_json::to_value(value).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "community_id".to_string(),
        annotation
            .community_id
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or(Value::Null),
    );
    if let Some(status) = &annotation.grounded_extension_status {
        object.insert(
            "grounded_extension_status".to_string(),
            json!(status.as_str()),
        );
    }
    object.insert(
        "source_kind".to_string(),
        json!(if existing.is_some() {
            "mixed"
        } else {
            "learned"
        }),
    );
    object.insert("engine".to_string(), json!(config.engine));
    object.insert("engine_version".to_string(), json!(config.engine_version));
    object.insert("computed_at".to_string(), json!(config.computed_at));
    object.insert("quarantine".to_string(), json!(true));
    if let Some(tenant) = prop_str(&content_node.properties, "tenant_id") {
        object
            .entry("tenant_id".to_string())
            .or_insert(json!(tenant));
    }
    if let Some(repo) = prop_str(&content_node.properties, "repo_id") {
        object.entry("repo_id".to_string()).or_insert(json!(repo));
    }

    let mut shadow = NodeRecord::new(&shadow_id, [EPISTEMIC_SHADOW_LABEL], properties);
    if annotation.structural_role_vector.is_empty() {
        if let Some(properties) = shadow.properties.as_object_mut() {
            properties.remove("structural_role_vector");
        }
    } else {
        let payloads = store.vector_payload_store()?;
        write_vector_property(
            &mut shadow,
            "structural_role_vector",
            &annotation.structural_role_vector,
            payloads.as_ref(),
        )?;
    }
    store.upsert_node(shadow)?;
    store.upsert_edge(EdgeRecord::new(
        has_epistemic_shadow_edge_id(&annotation.content_node_id, &shadow_id)?,
        &annotation.content_node_id,
        HAS_EPISTEMIC_SHADOW,
        &shadow_id,
        json!({
            "engine_version": config.engine_version,
            "computed_at": config.computed_at,
            "source": LEARNED_EPISTEMIC_ENGINE,
        }),
    ))?;
    store
        .get_node(&shadow_id)
        .map(|shadow| shadow_readout_from_node(store, &annotation.content_node_id, shadow))
        .transpose()
}

/// `canonical_edge_type` is the ontology row the originating edge resolved to,
/// when the relation came from a stored edge.
///
/// `EpistemicRelationKind` has only `Supports` and `Undercuts`, so every
/// `Defeating` row collapses onto `Undercuts` on the way in. Emitting the shadow
/// edge from the kind alone therefore rewrote a `Contradicts` edge, and an
/// unresolved `Tension`, as a directed inference undercut, destroying exactly
/// the distinction this pass now classifies by. Carrying the resolved row keeps
/// the emitted edge type and `EpistemicType` faithful to the source while the
/// kind continues to drive attack counting and grounded status.
fn write_shadow_relation<S: GraphStore>(
    store: &mut S,
    relation: &EpistemicRelationInput,
    from_shadow_id: &str,
    to_shadow_id: &str,
    config: &StructuralEpistemicConfig,
    canonical_edge_type: Option<&str>,
) -> GraphStoreResult<()> {
    let mut properties = json!({
        "from_content_id": relation.from_content_id,
        "to_content_id": relation.to_content_id,
        "confidence": normalized_confidence(relation.confidence),
        "evidence": relation.evidence,
        "source_kind": relation.source_kind.as_str(),
        "engine": config.engine,
        "engine_version": config.engine_version,
        "computed_at": config.computed_at,
        "quarantine": true,
    });
    {
        let object = ensure_object(&mut properties);
        if let Some(score) = relation.score {
            object.insert("score".to_string(), json!(score.clamp(0.0, 1.0)));
        }
        if let Some(model_id) = &relation.model_id {
            object.insert("model_id".to_string(), json!(model_id));
        }
        if let Some(calibration_version) = &relation.calibration_version {
            object.insert(
                "calibration_version".to_string(),
                json!(calibration_version),
            );
        }
        if let Some(feature_version) = &relation.feature_version {
            object.insert("feature_version".to_string(), json!(feature_version));
        }
        if let Some(features) = &relation.connection_features {
            object.insert(
                "connection_features".to_string(),
                serde_json::to_value(features).unwrap_or_else(|_| json!({})),
            );
        }
    }
    // `canonical_edge_type` is a public deserializable field, so a caller can
    // send `kind: Supports` alongside `canonical_edge_type: "Contradicts"`.
    // Support and attack counts, and the grounded status derived from them, are
    // computed from `kind`; trusting a conflicting canonical here would emit a
    // shadow edge whose type contradicts the arithmetic that produced it. A row
    // whose class disagrees with the kind is dropped, and the kind's own edge
    // type stands, so a disagreeing caller loses only its unusable hint.
    let resolved = canonical_edge_type
        .and_then(|token| default_registry().resolve(token))
        .filter(|spec| {
            EpistemicRelationKind::for_class(spec.class).as_ref() == Some(&relation.kind)
        });
    let emitted_edge_type = resolved
        .map(|spec| spec.canonical.as_str())
        .unwrap_or_else(|| relation.kind.edge_type());
    let emitted_epistemic_type = resolved
        .and_then(|spec| spec.epistemic_type.as_deref())
        .and_then(|token| token.parse::<EpistemicType>().ok())
        .unwrap_or_else(|| relation.kind.epistemic_type());
    let edge = EdgeRecord::new(
        epistemic_shadow_edge_id(
            relation.kind.clone(),
            from_shadow_id,
            to_shadow_id,
            &config.engine_version,
        )?,
        from_shadow_id,
        emitted_edge_type,
        to_shadow_id,
        properties,
    )
    .with_confidence(normalized_confidence(relation.confidence))
    .with_epistemic_type(emitted_epistemic_type)
    .with_provenance(Provenance {
        source_id: Some(config.engine.clone()),
        timestamp: Some(config.computed_at.to_string()),
        method: Some("epistemic_shadow_relation".to_string()),
    });
    store.upsert_edge(edge)?;
    Ok(())
}

fn induced_edges<S: GraphStore>(store: &S, node_set: &BTreeSet<String>) -> Vec<EdgeRecord> {
    let mut seen = BTreeSet::new();
    let mut edges = Vec::new();
    for node_id in node_set {
        for hit in store
            .neighbors(NeighborQuery::out(node_id).with_include_expired(true))
            .into_iter()
        {
            if !node_set.contains(&hit.node_id) || !seen.insert(hit.edge_id.clone()) {
                continue;
            }
            if let Some(edge) = store.get_edge(&hit.edge_id) {
                edges.push(edge.clone());
            }
        }
    }
    edges
}

fn existing_epistemic_relations(
    edges: &[EdgeRecord],
    classes: &ConfiguredEdgeClasses,
) -> Vec<EpistemicRelationInput> {
    let mut relations = Vec::new();
    for edge in edges {
        let kind = classes.kind_of(&edge.edge_type);
        if let Some(kind) = kind {
            let canonical_edge_type = default_registry()
                .resolve(&edge.edge_type)
                .map(|spec| spec.canonical.clone());
            relations.push(EpistemicRelationInput {
                from_content_id: edge.from_id.clone(),
                to_content_id: edge.to_id.clone(),
                kind,
                confidence: edge.confidence.unwrap_or(1.0),
                evidence: prop_str(&edge.properties, "evidence").unwrap_or_default(),
                source_kind: EpistemicSourceKind::Structural,
                score: None,
                model_id: None,
                calibration_version: None,
                feature_version: None,
                connection_features: None,
                canonical_edge_type,
            });
        }
    }
    relations
}

fn infer_pair_relation<S: GraphStore>(
    store: &S,
    pair: &EpistemicCandidatePair,
) -> Option<EpistemicRelationInput> {
    let left = store.get_node(&pair.left_content_id)?;
    let right = store.get_node(&pair.right_content_id)?;
    let left_text = claim_text(left);
    let right_text = claim_text(right);
    if left_text.trim().is_empty() || right_text.trim().is_empty() {
        return None;
    }
    let left_norm = normalize_claim(&left_text);
    let right_norm = normalize_claim(&right_text);
    if left_norm.is_empty() || right_norm.is_empty() {
        return None;
    }
    let left_neg = contains_negation(&left_norm);
    let right_neg = contains_negation(&right_norm);
    if left_neg != right_neg && without_negation(&left_norm) == without_negation(&right_norm) {
        return Some(EpistemicRelationInput {
            from_content_id: pair.left_content_id.clone(),
            to_content_id: pair.right_content_id.clone(),
            kind: EpistemicRelationKind::Undercuts,
            confidence: 0.75,
            evidence: format!("bounded_pair_contradiction: {left_text} :: {right_text}"),
            source_kind: EpistemicSourceKind::Structural,
            score: None,
            model_id: None,
            calibration_version: None,
            feature_version: None,
            connection_features: None,
            // Same proposition, opposite negation: this attacks the conclusion,
            // not the inference that reached it. The kind enum carries only
            // Supports and Undercuts, so canonical_edge_type is the only way to
            // say so, and without it the persisted edge claims an undercut.
            canonical_edge_type: Some("Contradicts".to_string()),
        });
    }
    if left_norm == right_norm {
        return Some(EpistemicRelationInput {
            from_content_id: pair.left_content_id.clone(),
            to_content_id: pair.right_content_id.clone(),
            kind: EpistemicRelationKind::Supports,
            confidence: 0.6,
            evidence: "bounded_pair_equivalent_claim".to_string(),
            source_kind: EpistemicSourceKind::Structural,
            score: None,
            model_id: None,
            calibration_version: None,
            feature_version: None,
            connection_features: None,
            canonical_edge_type: None,
        });
    }
    None
}

fn undirected_adjacency(
    node_set: &BTreeSet<String>,
    edges: &[EdgeRecord],
    relations: &[EpistemicRelationInput],
) -> HashMap<String, BTreeSet<String>> {
    let mut adjacency = node_set
        .iter()
        .map(|id| (id.clone(), BTreeSet::new()))
        .collect::<HashMap<_, _>>();
    for edge in edges {
        if node_set.contains(&edge.from_id) && node_set.contains(&edge.to_id) {
            adjacency
                .entry(edge.from_id.clone())
                .or_default()
                .insert(edge.to_id.clone());
            adjacency
                .entry(edge.to_id.clone())
                .or_default()
                .insert(edge.from_id.clone());
        }
    }
    for relation in relations {
        if node_set.contains(&relation.from_content_id)
            && node_set.contains(&relation.to_content_id)
        {
            adjacency
                .entry(relation.from_content_id.clone())
                .or_default()
                .insert(relation.to_content_id.clone());
            adjacency
                .entry(relation.to_content_id.clone())
                .or_default()
                .insert(relation.from_content_id.clone());
        }
    }
    adjacency
}

fn bridge_scores(
    node_set: &BTreeSet<String>,
    adjacency: &HashMap<String, BTreeSet<String>>,
) -> HashMap<String, f64> {
    let edges = adjacency.iter().flat_map(|(from, neighbors)| {
        neighbors
            .iter()
            .map(move |to| (from.as_str(), to.as_str(), 1.0))
    });
    let graph = CsrGraph::from_edges_and_nodes(edges, node_set.iter().map(String::as_str));
    graph
        .component_increase_counts()
        .into_iter()
        .enumerate()
        .map(|(dense, increase)| {
            (
                graph.id_of(dense).expect("dense node id").to_string(),
                increase as f64,
            )
        })
        .collect()
}

#[cfg(test)]
fn naive_bridge_scores(
    node_set: &BTreeSet<String>,
    adjacency: &HashMap<String, BTreeSet<String>>,
) -> HashMap<String, f64> {
    let baseline = naive_component_count(node_set, adjacency, None);
    node_set
        .iter()
        .map(|node_id| {
            let without = naive_component_count(node_set, adjacency, Some(node_id));
            let score = without.saturating_sub(baseline) as f64;
            (node_id.clone(), score)
        })
        .collect()
}

#[cfg(test)]
fn naive_component_count(
    node_set: &BTreeSet<String>,
    adjacency: &HashMap<String, BTreeSet<String>>,
    removed: Option<&String>,
) -> usize {
    use std::collections::VecDeque;

    let mut visited = HashSet::new();
    let mut count = 0usize;
    for start in node_set {
        if removed == Some(start) || !visited.insert(start.clone()) {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::from([start.clone()]);
        while let Some(node) = queue.pop_front() {
            for neighbor in adjacency.get(&node).into_iter().flatten() {
                if removed == Some(neighbor) || !visited.insert(neighbor.clone()) {
                    continue;
                }
                queue.push_back(neighbor.clone());
            }
        }
    }
    count
}

fn contradiction_cycles(
    relations: &[EpistemicRelationInput],
) -> GraphStoreResult<HashMap<String, String>> {
    let mut attacks = HashSet::new();
    for relation in relations {
        if relation.kind == EpistemicRelationKind::Undercuts {
            attacks.insert((
                relation.from_content_id.clone(),
                relation.to_content_id.clone(),
            ));
        }
    }
    let mut cycles = HashMap::new();
    for (left, right) in &attacks {
        if attacks.contains(&(right.clone(), left.clone())) {
            let cycle_id = format!("contradiction:{}", stable_hash(json!([left, right]))?);
            cycles.insert(left.clone(), cycle_id.clone());
            cycles.insert(right.clone(), cycle_id);
        }
    }
    Ok(cycles)
}

fn dedupe_relations(relations: &mut Vec<EpistemicRelationInput>) {
    let mut seen = BTreeSet::new();
    relations.retain(|relation| {
        seen.insert((
            relation.from_content_id.clone(),
            relation.to_content_id.clone(),
            relation.kind.edge_type().to_string(),
        ))
    });
}

fn migrate_legacy_structural_role_vector<S: GraphStore + GraphVectorPayloadAccess>(
    store: &S,
    shadow: &mut NodeRecord,
) -> GraphStoreResult<()> {
    let Some(value) = shadow.properties.get("structural_role_vector").cloned() else {
        return Ok(());
    };
    let Some(vector) = decode_legacy_vector_array(&value)? else {
        VectorPropertyRef::from_value(&value)?;
        return Ok(());
    };
    if vector.is_empty() {
        shadow
            .properties
            .as_object_mut()
            .expect("epistemic shadow properties are constructed as an object")
            .remove("structural_role_vector");
        return Ok(());
    }
    let payloads = store.vector_payload_store()?;
    write_vector_property(shadow, "structural_role_vector", &vector, payloads.as_ref())
}

fn shadow_readout_from_node<S: GraphStore + GraphVectorPayloadAccess>(
    store: &S,
    content_node_id: &str,
    shadow: &NodeRecord,
) -> GraphStoreResult<EpistemicShadowReadout> {
    let props = &shadow.properties;
    let structural_role_vector = if props.get("structural_role_vector").is_some() {
        let payloads = store.vector_payload_store()?;
        read_vector_property(shadow, "structural_role_vector", payloads.as_ref())?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    Ok(EpistemicShadowReadout {
        content_node_id: prop_str(props, "content_node_id")
            .unwrap_or_else(|| content_node_id.to_string()),
        shadow_node_id: shadow.id.clone(),
        grounded_extension_status: parse_grounded_status(
            &prop_str(props, "grounded_extension_status")
                .unwrap_or_else(|| "undecided".to_string()),
        ),
        support_in_degree: prop_u64(props, "support_in_degree").unwrap_or(0),
        attack_in_degree: prop_u64(props, "attack_in_degree").unwrap_or(0),
        unsupported_leaf: prop_bool(props, "unsupported_leaf"),
        orphan: prop_bool(props, "orphan"),
        bridge_score: prop_f64(props, "bridge_score").unwrap_or(0.0),
        contradiction_cycle_id: prop_str(props, "contradiction_cycle_id")
            .filter(|value| !value.is_empty()),
        predicted_edges: serde_json::from_value(
            props
                .get("predicted_edges")
                .cloned()
                .unwrap_or_else(|| json!([])),
        )
        .unwrap_or_default(),
        completion_confidence: prop_f64(props, "completion_confidence"),
        structural_role_vector,
        source_reliability: props
            .get("source_reliability")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
        community_id: prop_str(props, "community_id").filter(|value| !value.is_empty()),
        source_kind: parse_source_kind(
            &prop_str(props, "source_kind").unwrap_or_else(|| "structural".to_string()),
        ),
        engine: prop_str(props, "engine").unwrap_or_default(),
        engine_version: prop_str(props, "engine_version").unwrap_or_default(),
        computed_at: prop_i64(props, "computed_at").unwrap_or(0),
        quarantine: prop_bool(props, "quarantine"),
        field_provenance: props
            .get("field_provenance")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        // Populated by callers that hold a store handle (e.g.
        // `read_epistemic_shadow`); the node alone cannot reach its out-edge.
        same_eclass: None,
    })
}

pub fn compile_user_subgraph<S: GraphStore>(store: &S, node_ids: &[String]) -> UserSubgraph {
    let nodes = if node_ids.is_empty() {
        store
            .query_nodes(NodeQuery::default().with_limit(100_000))
            .into_iter()
            .filter(|node| {
                !node
                    .labels
                    .iter()
                    .any(|label| label == EPISTEMIC_SHADOW_LABEL)
            })
            .collect::<Vec<_>>()
    } else {
        node_ids
            .iter()
            .filter_map(|id| store.get_node(id).cloned())
            .collect::<Vec<_>>()
    };
    let node_set = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    let edges = induced_edges(store, &node_set)
        .into_iter()
        .filter(|edge| edge.edge_type != HAS_EPISTEMIC_SHADOW)
        .collect();
    UserSubgraph { nodes, edges }
}

pub fn select_nli_pairs(
    subgraph: &UserSubgraph,
    config: &NliPairSelectionConfig,
) -> Vec<NliPairInput> {
    if config.max_pairs == 0 {
        return Vec::new();
    }
    let mut nodes = subgraph
        .nodes
        .iter()
        .filter_map(|node| {
            let text = claim_text(node);
            (!text.trim().is_empty()).then(|| (node.id.clone(), text))
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| left.0.cmp(&right.0));
    let mut pairs = Vec::new();
    for (left_id, premise) in &nodes {
        for (right_id, hypothesis) in &nodes {
            if left_id == right_id {
                continue;
            }
            pairs.push(NliPairInput {
                left_content_id: left_id.clone(),
                right_content_id: right_id.clone(),
                premise: premise.clone(),
                hypothesis: hypothesis.clone(),
            });
            if pairs.len() >= config.max_pairs {
                return pairs;
            }
        }
    }
    pairs
}

#[derive(Default)]
struct GraphSignalIndex {
    support_in: HashMap<String, u64>,
    attack_in: HashMap<String, u64>,
    edge_count: usize,
}

fn graph_signal_index(subgraph: &UserSubgraph) -> GraphSignalIndex {
    let mut index = GraphSignalIndex {
        edge_count: subgraph.edges.len(),
        ..GraphSignalIndex::default()
    };
    let registry = default_registry();
    for edge in &subgraph.edges {
        match registry.class_of(&edge.edge_type) {
            Some(AlgebraicClass::MonotonePositive) => {
                *index.support_in.entry(edge.to_id.clone()).or_insert(0) += 1;
            }
            // Tension is a `Defeating` row, so an unresolved tension now
            // reaches `attack_in_degree` instead of being dropped.
            Some(AlgebraicClass::Defeating) => {
                *index.attack_in.entry(edge.to_id.clone()).or_insert(0) += 1;
            }
            Some(AlgebraicClass::Equivalence) | Some(AlgebraicClass::Structural) | None => {}
        }
    }
    index
}

fn connection_features_for_pair(
    pair: &NliPairInput,
    verdict: &NliVerdict,
    graph_signals: &GraphSignalIndex,
) -> ConnectionFeatures {
    ConnectionFeatures {
        from_content_id: pair.left_content_id.clone(),
        to_content_id: pair.right_content_id.clone(),
        premise: pair.premise.clone(),
        hypothesis: pair.hypothesis.clone(),
        candidate_evidence: Vec::new(),
        provenance: vec![format!("nli_model:{}", verdict.model_id)],
        nli_entailment_score: verdict.entailment.clamp(0.0, 1.0),
        nli_neutral_score: verdict.neutral.clamp(0.0, 1.0),
        nli_contradiction_score: verdict.contradiction.clamp(0.0, 1.0),
        support_in_degree: graph_signals
            .support_in
            .get(&pair.right_content_id)
            .copied()
            .unwrap_or(0),
        attack_in_degree: graph_signals
            .attack_in
            .get(&pair.right_content_id)
            .copied()
            .unwrap_or(0),
        bridge_score: 0.0,
        source_reliability_mean: None,
        graph_edge_count: graph_signals.edge_count,
        feature_version: default_connection_feature_version_string(),
    }
}

fn edge_density(subgraph: &UserSubgraph) -> f64 {
    let n = subgraph.nodes.len();
    if n < 2 {
        return 0.0;
    }
    let possible = n.saturating_mul(n - 1) as f64;
    subgraph.edges.len() as f64 / possible
}

fn set_field_provenance(
    properties: &mut Value,
    fields: &[&str],
    provenance: &EpistemicFieldProvenance,
) {
    let serialized = serde_json::to_value(provenance).unwrap_or_else(|_| json!({}));
    let object = ensure_object(properties);
    let entry = object
        .entry("field_provenance".to_string())
        .or_insert_with(|| json!({}));
    let provenance_map = ensure_object(entry);
    for field in fields {
        provenance_map.insert((*field).to_string(), serialized.clone());
    }
}

fn has_learned_fields(properties: &Value) -> bool {
    properties
        .get("predicted_edges")
        .and_then(Value::as_array)
        .map(|items| !items.is_empty())
        .unwrap_or(false)
        || properties
            .get("completion_confidence")
            .is_some_and(|value| !value.is_null())
        || properties
            .get("source_reliability")
            .is_some_and(|value| !value.is_null())
}

fn claim_text(node: &NodeRecord) -> String {
    [
        "claim_text",
        "content",
        "summary",
        "doc",
        "signature",
        "snippet",
        "name",
    ]
    .into_iter()
    .filter_map(|key| prop_str(&node.properties, key))
    .filter(|part| !part.trim().is_empty())
    .collect::<Vec<_>>()
    .join(" ")
}

pub(crate) fn normalize_claim(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_negation(normalized: &str) -> bool {
    normalized
        .split_whitespace()
        .any(|token| matches!(token, "not" | "no" | "never" | "without"))
}

pub(crate) fn without_negation(normalized: &str) -> String {
    normalized
        .split_whitespace()
        .filter(|token| !matches!(*token, "not" | "no" | "never" | "without"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn deterministic_pair_fallback_enabled() -> bool {
    std::env::var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn default_connection_scorer_model_id() -> String {
    DEFAULT_CONNECTION_SCORER_MODEL_ID.to_string()
}

fn default_connection_feature_version_string() -> String {
    DEFAULT_CONNECTION_FEATURE_VERSION.to_string()
}

fn default_connection_calibration_version_string() -> String {
    DEFAULT_CONNECTION_CALIBRATION_VERSION.to_string()
}

fn parse_grounded_status(raw: &str) -> GroundedExtensionStatus {
    match raw {
        "in" => GroundedExtensionStatus::In,
        "out" => GroundedExtensionStatus::Out,
        _ => GroundedExtensionStatus::Undecided,
    }
}

fn parse_source_kind(raw: &str) -> EpistemicSourceKind {
    match raw {
        "learned" => EpistemicSourceKind::Learned,
        "mixed" => EpistemicSourceKind::Mixed,
        _ => EpistemicSourceKind::Structural,
    }
}

/// Shared with [`crate::graph::ontology_adapter`] so the dual polynomial and
/// the pass agree on what a relation's confidence means. A negative or
/// non-finite value reads as `1.0` here, so a filter that clamped instead would
/// drop relations the pass still counts.
pub(crate) fn normalized_confidence(confidence: f64) -> f64 {
    if !confidence.is_finite() || confidence < 0.0 {
        1.0
    } else {
        confidence.clamp(0.0, 1.0)
    }
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn prop_str(properties: &Value, key: &str) -> Option<String> {
    properties.get(key).and_then(|value| {
        value.as_str().map(str::to_string).or_else(|| {
            if value.is_null() {
                None
            } else {
                Some(value.to_string())
            }
        })
    })
}

fn prop_u64(properties: &Value, key: &str) -> Option<u64> {
    properties.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
    })
}

fn prop_i64(properties: &Value, key: &str) -> Option<i64> {
    properties.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
    })
}

fn prop_f64(properties: &Value, key: &str) -> Option<f64> {
    properties.get(key).and_then(Value::as_f64)
}

fn prop_bool(properties: &Value, key: &str) -> bool {
    properties
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    value.as_object_mut().expect("value is object")
}

fn default_true() -> bool {
    true
}

fn default_observation_weight() -> f64 {
    1.0
}

// =========================================================================== //
// Strand A phase 3 / cut 9: e-graph equivalence-class dedup.
//
// Collapses provably-equivalent epistemic shadow STATES many-to-one onto a
// single `SameEClass` representative. The equivalence relation is decided by an
// `egg` e-graph: each shadow becomes the term `(shadow <status> <claim-term>)`
// (or `<claim-term>` alone in claim-only mode), the e-graph runs the epistemic
// rewrite rules to congruence-closure saturation, and two shadows are
// equivalent iff their terms land in the same e-class.
//
// Why an e-graph and not a string compare: the claim-term encodes negation
// STRUCTURALLY -- `(not (not p))` -- and the rule `(not (not ?x)) => ?x` proves
// that "p" and "not not p" are the same proposition. A string-tuple dedup
// (`dedupe_relations`) can never see that. Grouping is driven by the e-graph's
// `find()`, so adding richer provable-equivalence rules later (commutativity,
// idempotence, domain rewrites -- the egglog/Datalog upgrade path) re-clusters
// automatically with no change to the grouping code.
//
// Non-breaking: this pass writes ONLY `SameEClass` edges. It never mutates a
// content node, and never mutates or deletes a shadow node. `epistemic_shadow_ppr`
// already traverses `SameEClass`, so the new edges light up clustering for free,
// and `read_epistemic_shadow` reads the membership back off the edge.
//
// Scope notes (surfaced, not buried):
//   * v1 congruence is the double-negation rule only. The negation parity is
//     best-effort: it reuses the existing whole-claim negation lexicon
//     (`count_negations` mirrors `without_negation`), so "without"/multi-negation
//     resolve to XOR parity rather than scoped logic. A mis-parity only ever
//     fails-to-merge (conservative) or mislabels the cosmetic `canonical_form`;
//     it cannot corrupt the content graph. Richer rules are an additive change.
//   * Egglog as the backend for the byte-parity Datalog layer (`symbolic.rs`
//     `derive_datalog_receipt`) is deliberately DEFERRED, not done here: that
//     engine has a Python-reference byte-parity gate an egglog rewrite would
//     break. This e-graph is structured as the reusable substrate for that
//     upgrade, but swapping the Datalog engine is its own gated change.
//   * Read path is wired: recall surfaces `same_eclass` via `read_epistemic_shadow`.
//     The WRITE trigger (a cron/MCP/server caller of `epistemic_egraph_dedup`)
//     is deferred and named, matching the rest of the epistemic strand's
//     core-first, wiring-later split.
// =========================================================================== //

/// Which fields define epistemic-shadow equivalence for the dedup pass.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicCongruence {
    /// Two shadows are equivalent iff their claim logical forms are congruent
    /// AND their grounded-extension status is identical. The default: a shadow
    /// STATE includes its standing, so the same claim marked `in` versus `out`
    /// is two distinct states and is not collapsed (collapsing it would hide a
    /// disagreement).
    #[default]
    ClaimAndStanding,
    /// Equivalence on the claim logical form alone; standing is ignored, so
    /// equivalent claims merge regardless of grounded status.
    ClaimOnly,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EpistemicDedupConfig {
    pub engine: String,
    pub engine_version: String,
    pub computed_at: i64,
    pub congruence: EpistemicCongruence,
}

impl Default for EpistemicDedupConfig {
    fn default() -> Self {
        Self {
            engine: EGRAPH_EPISTEMIC_ENGINE.to_string(),
            engine_version: DEFAULT_EPISTEMIC_ENGINE_VERSION.to_string(),
            computed_at: now_ms(),
            congruence: EpistemicCongruence::default(),
        }
    }
}

/// One equivalence class with at least two members (singletons are not
/// reported). `member_*` vectors are sorted and include the representative.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EpistemicEquivalenceClass {
    pub class_id: String,
    pub canonical_form: String,
    pub representative_content_id: String,
    pub representative_shadow_id: String,
    pub member_content_ids: Vec<String>,
    pub member_shadow_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EpistemicDedupReport {
    /// Shadows that produced an e-graph term (had a non-empty claim form).
    pub shadows_examined: usize,
    /// Shadows that formed their own singleton class (no equivalent peer).
    pub singleton_count: usize,
    /// Equivalence classes with >= 2 members.
    pub classes: Vec<EpistemicEquivalenceClass>,
    /// Sum over classes of (members - 1): the count of shadows folded onto a
    /// representative.
    pub members_collapsed: usize,
    /// `SameEClass` edges written (== `members_collapsed`).
    pub same_eclass_edges_written: usize,
}

/// The logical form of a claim: a proposition plus a negation parity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ClaimForm {
    /// Negation-stripped, normalized proposition tokens joined by spaces.
    pub(crate) prop: String,
    /// Number of negation tokens observed (the e-graph reduces parity).
    pub(crate) neg_count: usize,
}

impl ClaimForm {
    pub(crate) fn negated(&self) -> bool {
        self.neg_count % 2 == 1
    }

    /// A human-readable canonical label for the (proposition, parity) pair.
    pub(crate) fn canonical_label(&self) -> String {
        if self.negated() {
            format!("not {}", self.prop)
        } else {
            self.prop.clone()
        }
    }

    /// The `egg` term for the claim: a `p_`-prefixed proposition leaf wrapped in
    /// `neg_count` `(not ...)` layers, which the double-negation rule reduces to
    /// parity. The `p_` prefix keeps the leaf disjoint from the `not`/`shadow`
    /// operators and `s_*` status symbols.
    pub(crate) fn claim_term(&self) -> String {
        let mut term = format!("p_{}", self.prop.replace(' ', "_"));
        for _ in 0..self.neg_count {
            term = format!("(not {term})");
        }
        term
    }
}

pub(crate) fn claim_logical_form(text: &str) -> Option<ClaimForm> {
    let normalized = normalize_claim(text);
    let prop = without_negation(&normalized);
    if prop.is_empty() {
        return None;
    }
    Some(ClaimForm {
        prop,
        neg_count: count_negations(&normalized),
    })
}

// Best-effort negation parity: this MUST use the same lexicon as
// `without_negation` (above) or the stripped proposition and the parity would
// disagree. See the module-header scope notes for why this is intentionally
// best-effort (whole-claim, not scoped).
pub(crate) fn count_negations(normalized: &str) -> usize {
    normalized
        .split_whitespace()
        .filter(|token| matches!(*token, "not" | "no" | "never" | "without"))
        .count()
}

pub(crate) fn grounded_status_symbol(raw: &str) -> &'static str {
    match parse_grounded_status(raw) {
        GroundedExtensionStatus::In => "s_in",
        GroundedExtensionStatus::Out => "s_out",
        GroundedExtensionStatus::Undecided => "s_undecided",
    }
}

/// A shadow staged for the e-graph: the term plus the identity needed to write
/// edges and report membership.
struct DedupCandidate {
    shadow_id: String,
    content_id: String,
    form: ClaimForm,
    term: RecExpr<SymbolLang>,
}

/// Run the e-graph dedup pass over the given content nodes' shadows (or every
/// shadow when `content_node_ids` is empty). Writes `SameEClass` edges from each
/// non-representative member to its class representative and returns the report.
pub fn epistemic_egraph_dedup<S: GraphStore>(
    store: &mut S,
    content_node_ids: &[String],
    config: EpistemicDedupConfig,
) -> GraphStoreResult<EpistemicDedupReport> {
    let shadow_nodes = collect_dedup_shadows(store, content_node_ids);

    // Stage each shadow as an e-graph term.
    let mut candidates: Vec<DedupCandidate> = Vec::new();
    for shadow in &shadow_nodes {
        let content_id =
            prop_str(&shadow.properties, "content_node_id").unwrap_or_else(|| shadow.id.clone());
        let Some(content_node) = store.get_node(&content_id) else {
            continue;
        };
        let Some(form) = claim_logical_form(&claim_text(content_node)) else {
            continue;
        };
        let term_str = match config.congruence {
            EpistemicCongruence::ClaimOnly => form.claim_term(),
            EpistemicCongruence::ClaimAndStanding => {
                let status = grounded_status_symbol(
                    &prop_str(&shadow.properties, "grounded_extension_status")
                        .unwrap_or_else(|| "undecided".to_string()),
                );
                format!("(shadow {status} {})", form.claim_term())
            }
        };
        let Ok(term) = term_str.parse::<RecExpr<SymbolLang>>() else {
            continue;
        };
        candidates.push(DedupCandidate {
            shadow_id: shadow.id.clone(),
            content_id,
            form,
            term,
        });
    }

    let mut report = EpistemicDedupReport {
        shadows_examined: candidates.len(),
        ..EpistemicDedupReport::default()
    };
    if candidates.is_empty() {
        return Ok(report);
    }

    // Saturate one e-graph holding every term, then group by canonical e-class.
    let terms = candidates
        .iter()
        .map(|candidate| candidate.term.clone())
        .collect::<Vec<_>>();
    let class_roots = egraph_class_roots(&terms);
    let mut groups: BTreeMap<Id, Vec<usize>> = BTreeMap::new();
    for (idx, root) in class_roots.into_iter().enumerate() {
        groups.entry(root).or_default().push(idx);
    }

    let mut classes: Vec<EpistemicEquivalenceClass> = Vec::new();
    for members in groups.into_values() {
        if members.len() < 2 {
            report.singleton_count += 1;
            continue;
        }
        // Deterministic representative: smallest (content_id, shadow_id).
        let mut sorted = members.clone();
        sorted.sort_by(|&a, &b| {
            candidates[a]
                .content_id
                .cmp(&candidates[b].content_id)
                .then_with(|| candidates[a].shadow_id.cmp(&candidates[b].shadow_id))
        });
        let rep = &candidates[sorted[0]];
        let representative_content_id = rep.content_id.clone();
        let representative_shadow_id = rep.shadow_id.clone();
        let canonical_form = rep.form.canonical_label();

        let mut member_content_ids: Vec<String> = sorted
            .iter()
            .map(|&i| candidates[i].content_id.clone())
            .collect();
        let mut member_shadow_ids: Vec<String> = sorted
            .iter()
            .map(|&i| candidates[i].shadow_id.clone())
            .collect();
        member_content_ids.sort();
        member_shadow_ids.sort();
        member_content_ids.dedup();
        member_shadow_ids.dedup();
        let class_id = format!("eclass:{}", stable_hash(json!(member_shadow_ids))?);

        for &idx in sorted.iter().skip(1) {
            let member = &candidates[idx];
            write_same_eclass_edge(
                store,
                &member.shadow_id,
                &representative_shadow_id,
                &class_id,
                &canonical_form,
                &config,
            )?;
            report.same_eclass_edges_written += 1;
            report.members_collapsed += 1;
        }

        classes.push(EpistemicEquivalenceClass {
            class_id,
            canonical_form,
            representative_content_id,
            representative_shadow_id,
            member_content_ids,
            member_shadow_ids,
        });
    }
    classes.sort_by(|a, b| a.class_id.cmp(&b.class_id));
    report.classes = classes;
    Ok(report)
}

fn collect_dedup_shadows<S: GraphStore>(store: &S, content_node_ids: &[String]) -> Vec<NodeRecord> {
    if content_node_ids.is_empty() {
        // Full-store mode bounds at 100_000 shadows, matching the rest of the
        // epistemic module (`compile_user_subgraph`, `epistemic_shadow_ppr`). A
        // tenant whose shadow set exceeds that should drive the dedup with an
        // explicit `content_node_ids` batch rather than the whole-store scan.
        let mut nodes =
            store.query_nodes(NodeQuery::label(EPISTEMIC_SHADOW_LABEL).with_limit(100_000));
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        return nodes;
    }
    let mut seen = BTreeSet::new();
    let mut nodes = Vec::new();
    for content_id in content_node_ids {
        for hit in store
            .neighbors(NeighborQuery::out(content_id).with_edge_type(HAS_EPISTEMIC_SHADOW))
            .into_iter()
        {
            if !seen.insert(hit.node_id.clone()) {
                continue;
            }
            if let Some(shadow) = store.get_node(&hit.node_id) {
                if shadow
                    .labels
                    .iter()
                    .any(|label| label == EPISTEMIC_SHADOW_LABEL)
                {
                    nodes.push(shadow.clone());
                }
            }
        }
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    nodes
}

/// Saturate one e-graph over all candidate terms and return, per candidate (in
/// input order), the canonical e-class id it belongs to.
///
/// The whole seed graph is built with `with_expr` (unbounded `add_expr`) BEFORE
/// `run()`, and a large batch can push it past egg's default `node_limit`
/// (10_000). When the seed exceeds that limit, egg's `check_limits()` returns
/// `Err(NodeLimit)` and short-circuits rewrite application, so the congruence
/// rule would fire on ZERO terms and the dedup would silently degenerate to
/// exact-string hashconsing. We therefore lift the limits explicitly. This is
/// safe: the only rule `(not (not ?x)) => ?x` strictly shrinks term nesting and
/// never grows the graph beyond the seed, so it reaches `Saturated` in a handful
/// of iterations regardless of batch size.
pub(crate) fn egraph_class_roots(terms: &[RecExpr<SymbolLang>]) -> Vec<Id> {
    let rules: Vec<egg::Rewrite<SymbolLang, ()>> =
        vec![egg::rewrite!("epistemic-double-negation"; "(not (not ?x))" => "?x")];
    let seed_nodes = terms.iter().map(|term| term.as_ref().len()).sum::<usize>();
    let mut runner: Runner<SymbolLang, ()> = Runner::default()
        .with_node_limit(seed_nodes.saturating_mul(4).max(10_000))
        .with_iter_limit(64)
        .with_time_limit(std::time::Duration::from_secs(30));
    for term in terms {
        runner = runner.with_expr(term);
    }
    let runner = runner.run(&rules);
    runner
        .roots
        .iter()
        .map(|root| runner.egraph.find(*root))
        .collect()
}

fn write_same_eclass_edge<S: GraphStore>(
    store: &mut S,
    member_shadow_id: &str,
    representative_shadow_id: &str,
    class_id: &str,
    canonical_form: &str,
    config: &EpistemicDedupConfig,
) -> GraphStoreResult<()> {
    let edge = EdgeRecord::new(
        same_eclass_edge_id(
            member_shadow_id,
            representative_shadow_id,
            &config.engine_version,
        )?,
        member_shadow_id,
        SAME_ECLASS,
        representative_shadow_id,
        json!({
            "class_id": class_id,
            "canonical_form": canonical_form,
            "confidence": 1.0,
            "evidence": "egraph_congruence",
            "source_kind": "structural",
            "engine": config.engine,
            "engine_version": config.engine_version,
            "computed_at": config.computed_at,
            "quarantine": true,
        }),
    )
    .with_confidence(1.0)
    .with_provenance(Provenance {
        source_id: Some(config.engine.clone()),
        timestamp: Some(config.computed_at.to_string()),
        method: Some("epistemic_egraph_dedup".to_string()),
    });
    store.upsert_edge(edge)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::in_memory::InMemoryGraphStore;
    use crate::graph::ontology_adapter::{
        epistemic_node_polynomial, lift_edge, relation_token, RelationAnnotations, StoreAnnotations,
    };
    use rustyred_thg_derivation::dual::{EdgeToken, Polarity};
    use rustyred_thg_derivation::interpret::{
        interpret, net_probability, probability_readout, AnnotationSource, Standing,
    };
    use rustyred_thg_derivation::semiring::BooleanSemiring;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn claim(id: &str, text: &str) -> NodeRecord {
        NodeRecord::new(
            id,
            ["Claim"],
            json!({ "tenant_id": "t", "claim_text": text }),
        )
    }

    #[test]
    fn bridge_scores_match_naive_chokepoint_oracle_for_all_small_graphs() {
        for node_count in 0..=5 {
            let node_set = (0..node_count)
                .map(|node| format!("node:{node}"))
                .collect::<BTreeSet<_>>();
            let possible_edges = (0..node_count)
                .flat_map(|left| ((left + 1)..node_count).map(move |right| (left, right)))
                .collect::<Vec<_>>();

            for edge_mask in 0..(1usize << possible_edges.len()) {
                let mut adjacency = node_set
                    .iter()
                    .map(|node| (node.clone(), BTreeSet::new()))
                    .collect::<HashMap<_, _>>();
                for (edge_index, &(left, right)) in possible_edges.iter().enumerate() {
                    if edge_mask & (1usize << edge_index) == 0 {
                        continue;
                    }
                    let left = format!("node:{left}");
                    let right = format!("node:{right}");
                    adjacency
                        .get_mut(&left)
                        .expect("left endpoint")
                        .insert(right.clone());
                    adjacency
                        .get_mut(&right)
                        .expect("right endpoint")
                        .insert(left);
                }

                assert_eq!(
                    bridge_scores(&node_set, &adjacency),
                    naive_bridge_scores(&node_set, &adjacency),
                    "node_count={node_count}, edge_mask={edge_mask:#x}"
                );
            }
        }
    }

    #[test]
    fn structural_pass_does_not_use_deterministic_pair_fallback_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV);
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "cache is enabled")).unwrap();
        store
            .upsert_node(claim("b", "cache is not enabled"))
            .unwrap();

        let readout = structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                candidate_pairs: vec![EpistemicCandidatePair {
                    left_content_id: "a".to_string(),
                    right_content_id: "b".to_string(),
                }],
                config: StructuralEpistemicConfig {
                    candidate_top_k: 1,
                    ..StructuralEpistemicConfig::default()
                },
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();

        assert_eq!(readout.checked_pair_count, 1);
        assert_eq!(readout.candidate_pair_bound, 2);
        assert_eq!(readout.contradictions.len(), 0);
        let shadow = read_epistemic_shadow(&store, "b").expect("shadow");
        assert_eq!(shadow.attack_in_degree, 0);
        assert_eq!(
            shadow.grounded_extension_status,
            GroundedExtensionStatus::In
        );
        assert!(shadow.field_provenance.contains_key("support_in_degree"));
    }

    #[test]
    fn structural_pass_uses_deterministic_pair_fallback_only_when_enabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV, "1");
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "cache is enabled")).unwrap();
        store
            .upsert_node(claim("b", "cache is not enabled"))
            .unwrap();

        let readout = structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                candidate_pairs: vec![EpistemicCandidatePair {
                    left_content_id: "a".to_string(),
                    right_content_id: "b".to_string(),
                }],
                config: StructuralEpistemicConfig {
                    candidate_top_k: 1,
                    ..StructuralEpistemicConfig::default()
                },
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        std::env::remove_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV);

        assert_eq!(readout.contradictions.len(), 1);
        let shadow = read_epistemic_shadow(&store, "b").expect("shadow");
        assert_eq!(shadow.attack_in_degree, 1);
        assert_eq!(
            shadow.grounded_extension_status,
            GroundedExtensionStatus::Out
        );
    }

    #[test]
    fn shadow_ppr_ranks_over_shadow_edges() {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        store.upsert_node(claim("b", "B")).unwrap();
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                explicit_relations: vec![EpistemicRelationInput {
                    from_content_id: "a".to_string(),
                    to_content_id: "b".to_string(),
                    kind: EpistemicRelationKind::Supports,
                    confidence: 1.0,
                    evidence: "test".to_string(),
                    source_kind: EpistemicSourceKind::Structural,
                    score: None,
                    model_id: None,
                    calibration_version: None,
                    feature_version: None,
                    connection_features: None,
                    canonical_edge_type: None,
                }],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let mut seeds = HashMap::new();
        seeds.insert("a".to_string(), 1.0);
        let ranked = epistemic_shadow_ppr(&store, &seeds, 4, 0.15, 1e-5, 20_000).unwrap();
        let b_shadow = epistemic_shadow_node_id("b", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        assert!(ranked.iter().any(|(id, _)| id == &b_shadow));
    }

    #[test]
    fn shadow_ppr_cache_isolated_by_graph_topology() {
        crate::ppr_cache::clear_scoped_ppr_cache();
        let mut first = InMemoryGraphStore::new();
        let mut second = InMemoryGraphStore::new();
        for store in [&mut first, &mut second] {
            for (id, text) in [("a", "A"), ("b", "B"), ("c", "C")] {
                store.upsert_node(claim(id, text)).unwrap();
            }
        }
        for (store, target) in [(&mut first, "b"), (&mut second, "c")] {
            structural_epistemic_pass(
                store,
                StructuralEpistemicInput {
                    batch_node_ids: vec!["a".to_string(), target.to_string()],
                    explicit_relations: vec![EpistemicRelationInput {
                        from_content_id: "a".to_string(),
                        to_content_id: target.to_string(),
                        kind: EpistemicRelationKind::Supports,
                        confidence: 1.0,
                        evidence: "cache isolation".to_string(),
                        source_kind: EpistemicSourceKind::Structural,
                        score: None,
                        model_id: None,
                        calibration_version: None,
                        feature_version: None,
                        connection_features: None,
                        canonical_edge_type: None,
                    }],
                    ..StructuralEpistemicInput::default()
                },
            )
            .unwrap();
        }
        assert_eq!(first.stats().version, second.stats().version);

        let seeds = HashMap::from([("a".to_string(), 1.0)]);
        let first_ranked = epistemic_shadow_ppr(&first, &seeds, 8, 0.15, 1e-5, 20_000).unwrap();
        let second_ranked = epistemic_shadow_ppr(&second, &seeds, 8, 0.15, 1e-5, 20_000).unwrap();
        let b_shadow = epistemic_shadow_node_id("b", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        let c_shadow = epistemic_shadow_node_id("c", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        assert!(first_ranked.iter().any(|(id, _)| id == &b_shadow));
        assert!(!first_ranked.iter().any(|(id, _)| id == &c_shadow));
        assert!(second_ranked.iter().any(|(id, _)| id == &c_shadow));
        assert!(!second_ranked.iter().any(|(id, _)| id == &b_shadow));
        crate::ppr_cache::clear_scoped_ppr_cache();
    }

    struct DroppingEnricher;

    impl EpistemicEnricher for DroppingEnricher {
        fn enrich(
            &self,
            _subgraph: UserSubgraph,
            _mode: EpistemicEnrichmentMode,
        ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError> {
            Err(EpistemicEnrichmentError::new("unavailable", "grpc dropped"))
        }
    }

    #[test]
    fn cron_drop_noops_without_deleting_existing_shadow() {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string()],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let structural_shadow_id =
            epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        assert!(store
            .get_node(&structural_shadow_id)
            .unwrap()
            .properties
            .get("structural_role_vector")
            .is_none());
        let before = store.stats().nodes_total;
        let report = run_epistemic_cron_pass(
            &mut store,
            EpistemicCronInput {
                content_node_ids: vec!["a".to_string()],
                ..EpistemicCronInput::default()
            },
            &DroppingEnricher,
        )
        .unwrap();
        assert!(report.no_op);
        assert!(!report.grpc_ok);
        assert_eq!(store.stats().nodes_total, before);
        assert!(read_epistemic_shadow(&store, "a").is_some());
    }

    struct LearnedEnricher;

    impl EpistemicEnricher for LearnedEnricher {
        fn enrich(
            &self,
            _subgraph: UserSubgraph,
            _mode: EpistemicEnrichmentMode,
        ) -> Result<EpistemicAnnotations, EpistemicEnrichmentError> {
            Ok(EpistemicAnnotations {
                annotations: vec![EpistemicAnnotation {
                    content_node_id: "a".to_string(),
                    predicted_edges: vec![PredictedEdgePointer {
                        target_content_id: "b".to_string(),
                        relation: "depends_on".to_string(),
                        confidence: 0.8,
                        quarantine: true,
                    }],
                    completion_confidence: Some(0.8),
                    structural_role_vector: vec![0.1, 0.2],
                    source_reliability: Some(SourceReliability {
                        alpha: 3.0,
                        beta: 1.0,
                        mean: 0.75,
                    }),
                    community_id: Some("community:test".to_string()),
                    grounded_extension_status: None,
                }],
                ..EpistemicAnnotations::default()
            })
        }
    }

    #[test]
    fn learned_cron_adds_fields_to_same_shadow_without_overwriting_structural() {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        store.upsert_node(claim("b", "B")).unwrap();
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let before = read_epistemic_shadow(&store, "a").expect("structural shadow");
        assert!(before.structural_role_vector.is_empty());
        assert_eq!(
            before.shadow_node_id,
            epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap()
        );

        let report = run_epistemic_cron_pass(
            &mut store,
            EpistemicCronInput {
                content_node_ids: vec!["a".to_string(), "b".to_string()],
                ..EpistemicCronInput::default()
            },
            &LearnedEnricher,
        )
        .unwrap();
        assert_eq!(report.shadows_written, 1);

        let after = read_epistemic_shadow(&store, "a").expect("learned shadow");
        assert_eq!(after.shadow_node_id, before.shadow_node_id);
        assert_eq!(after.support_in_degree, before.support_in_degree);
        assert_eq!(after.attack_in_degree, before.attack_in_degree);
        assert_eq!(after.predicted_edges.len(), 1);
        assert!(after.predicted_edges[0].quarantine);
        assert_eq!(after.structural_role_vector, vec![0.1, 0.2]);
        assert_eq!(after.source_kind, EpistemicSourceKind::Mixed);
        let stored_vector = store
            .get_node(&after.shadow_node_id)
            .unwrap()
            .properties
            .get("structural_role_vector")
            .expect("learned vector reference");
        assert!(stored_vector.is_object());
        assert!(!stored_vector.is_array());
        let reference = VectorPropertyRef::from_value(stored_vector).unwrap();
        assert_eq!(reference.dimension, 2);
    }

    #[test]
    fn legacy_structural_role_vector_rejects_values_outside_finite_f32() {
        let store = InMemoryGraphStore::new();
        let mut shadow = NodeRecord::new(
            "legacy:shadow",
            [EPISTEMIC_SHADOW_LABEL],
            json!({"structural_role_vector":[1.0e300_f64]}),
        );

        let error = migrate_legacy_structural_role_vector(&store, &mut shadow).unwrap_err();
        assert_eq!(error.code, "legacy_vector_property_invalid");
        assert!(shadow.properties["structural_role_vector"].is_array());
    }

    struct StubNliClassifier;

    impl NliClassifier for StubNliClassifier {
        fn classify_batch(
            &self,
            pairs: &[NliPairInput],
        ) -> Result<Vec<NliVerdict>, EpistemicEnrichmentError> {
            Ok(pairs
                .iter()
                .map(|pair| {
                    if pair.left_content_id == "a" && pair.right_content_id == "b" {
                        NliVerdict {
                            entailment: 0.91,
                            neutral: 0.04,
                            contradiction: 0.05,
                            model_id: DEFAULT_NLI_MODEL_ID.to_string(),
                        }
                    } else {
                        NliVerdict {
                            entailment: 0.03,
                            neutral: 0.06,
                            contradiction: 0.91,
                            model_id: DEFAULT_NLI_MODEL_ID.to_string(),
                        }
                    }
                })
                .collect())
        }
    }

    struct FixtureLearnedScorer;

    impl ConnectionScorer for FixtureLearnedScorer {
        fn score(
            &self,
            features: &ConnectionFeatures,
        ) -> Result<ConnectionScore, EpistemicEnrichmentError> {
            let kind = if features.from_content_id == "a" && features.to_content_id == "b" {
                Some(EpistemicRelationKind::Supports)
            } else {
                Some(EpistemicRelationKind::Undercuts)
            };
            Ok(ConnectionScore {
                kind,
                score: 0.93,
                confidence: 0.91,
                model_id: "fixture-learned-connection-scorer".to_string(),
                calibration_version: DEFAULT_CONNECTION_CALIBRATION_VERSION.to_string(),
                feature_version: DEFAULT_CONNECTION_FEATURE_VERSION.to_string(),
                evidence: "fixture learned scorer response".to_string(),
            })
        }
    }

    #[test]
    fn nli_cron_uses_learned_scorer_by_default_and_writes_metadata() {
        let mut store = InMemoryGraphStore::new();
        store
            .upsert_node(claim("a", "cache layer is enabled"))
            .unwrap();
        store
            .upsert_node(claim("b", "cache layer accelerates reads"))
            .unwrap();
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let enricher = NliEpistemicEnricher::new(
            StubNliClassifier,
            FixtureLearnedScorer,
            NliPairSelectionConfig { max_pairs: 2 },
        );
        let report = run_epistemic_cron_pass(
            &mut store,
            EpistemicCronInput {
                content_node_ids: vec!["a".to_string(), "b".to_string()],
                engine: NLI_EPISTEMIC_ENGINE.to_string(),
                ..EpistemicCronInput::default()
            },
            &enricher,
        )
        .unwrap();

        assert_eq!(report.shadow_edges_written, 2);
        let from_shadow = epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        let to_shadow = epistemic_shadow_node_id("b", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        let edge_id = epistemic_shadow_edge_id(
            EpistemicRelationKind::Supports,
            &from_shadow,
            &to_shadow,
            DEFAULT_EPISTEMIC_ENGINE_VERSION,
        )
        .unwrap();
        let edge = store.get_edge(&edge_id).expect("learned support edge");
        assert_eq!(
            edge.properties.get("source_kind").and_then(Value::as_str),
            Some("learned")
        );
        assert_eq!(
            edge.properties.get("model_id").and_then(Value::as_str),
            Some("fixture-learned-connection-scorer")
        );
        assert_eq!(
            edge.properties.get("score").and_then(Value::as_f64),
            Some(0.93)
        );
        assert_eq!(
            edge.properties
                .get("connection_features")
                .and_then(|value| value.get("nli_entailment_score"))
                .and_then(Value::as_f64),
            Some(0.91)
        );
    }

    #[test]
    fn missing_learned_scorer_noops_without_deterministic_fallback() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(EPISTEMIC_SCORER_ENDPOINT_ENV);
        std::env::remove_var(EPISTEMIC_DETERMINISTIC_FALLBACK_ENV);
        let mut store = InMemoryGraphStore::new();
        store
            .upsert_node(claim("a", "cache layer is enabled"))
            .unwrap();
        store
            .upsert_node(claim("b", "cache layer is not enabled"))
            .unwrap();
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let before_edges = store.stats().edges_total;
        let enricher = NliEpistemicEnricher::new(
            StubNliClassifier,
            LearnedConnectionScorer::default(),
            NliPairSelectionConfig { max_pairs: 1 },
        );
        let report = run_epistemic_cron_pass(
            &mut store,
            EpistemicCronInput {
                content_node_ids: vec!["a".to_string(), "b".to_string()],
                engine: NLI_EPISTEMIC_ENGINE.to_string(),
                ..EpistemicCronInput::default()
            },
            &enricher,
        )
        .unwrap();

        assert!(report.no_op);
        assert!(!report.grpc_ok);
        assert!(report.skipped_reason.contains("learned_scorer_unavailable"));
        assert_eq!(store.stats().edges_total, before_edges);
    }

    fn shadows_for(store: &mut InMemoryGraphStore, ids: &[&str]) {
        structural_epistemic_pass(
            store,
            StructuralEpistemicInput {
                batch_node_ids: ids.iter().map(|s| s.to_string()).collect(),
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn egraph_dedup_collapses_double_negation_equivalent_claims() {
        // The e-graph win: "X" and "not not X" are proven equal by the
        // double-negation rule, so they collapse into one class. A plain
        // string-tuple dedup would keep them apart.
        let mut store = InMemoryGraphStore::new();
        store
            .upsert_node(claim("plain", "feature is enabled"))
            .unwrap();
        store
            .upsert_node(claim("double", "feature is not not enabled"))
            .unwrap();
        shadows_for(&mut store, &["plain", "double"]);

        let report =
            epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).unwrap();
        assert_eq!(report.shadows_examined, 2);
        assert_eq!(report.classes.len(), 1, "the two claims form one class");
        assert_eq!(report.members_collapsed, 1);
        assert_eq!(report.same_eclass_edges_written, 1);
        let class = &report.classes[0];
        assert_eq!(class.member_content_ids, vec!["double", "plain"]);
        // canonical_form is the parity-reduced proposition (no leading "not").
        assert_eq!(class.canonical_form, "feature is enabled");
    }

    #[test]
    fn egraph_dedup_single_negation_stays_distinct_from_positive() {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("pos", "cache is enabled")).unwrap();
        store
            .upsert_node(claim("neg", "cache is not enabled"))
            .unwrap();
        shadows_for(&mut store, &["pos", "neg"]);
        let report =
            epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).unwrap();
        assert_eq!(
            report.classes.len(),
            0,
            "p and (not p) are different states"
        );
        assert_eq!(report.same_eclass_edges_written, 0);
        assert_eq!(report.singleton_count, 2);
    }

    #[test]
    fn egraph_dedup_is_idempotent_and_many_to_one() {
        let mut store = InMemoryGraphStore::new();
        for id in ["a", "b", "c"] {
            store.upsert_node(claim(id, "the sky is blue")).unwrap();
        }
        shadows_for(&mut store, &["a", "b", "c"]);

        let first =
            epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).unwrap();
        assert_eq!(first.classes.len(), 1);
        // Three members, two collapse onto one representative (many-to-one).
        assert_eq!(first.members_collapsed, 2);
        let rep = &first.classes[0].representative_shadow_id;

        // Every non-representative member points at exactly the representative.
        let mut edge_targets = Vec::new();
        for content in ["a", "b", "c"] {
            let shadow_id =
                epistemic_shadow_node_id(content, DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
            if &shadow_id == rep {
                continue;
            }
            let same = read_same_eclass(&store, &shadow_id).expect("member has SameEClass edge");
            edge_targets.push(same.representative_shadow_id);
        }
        assert!(edge_targets.iter().all(|t| t == rep));

        // Re-running writes the same edge ids: edge count is unchanged.
        let edges_before = store.stats().edges_total;
        let second =
            epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).unwrap();
        assert_eq!(second.members_collapsed, first.members_collapsed);
        assert_eq!(second.classes[0].class_id, first.classes[0].class_id);
        assert_eq!(
            store.stats().edges_total,
            edges_before,
            "idempotent: no new edges"
        );
    }

    #[test]
    fn claim_and_standing_keeps_distinct_status_apart() {
        // Same claim, but planted so one shadow is grounded `out` (attacked) and
        // the other is `in`. ClaimAndStanding must NOT merge them.
        let mut store = InMemoryGraphStore::new();
        store
            .upsert_node(claim("winner", "the build passes"))
            .unwrap();
        store
            .upsert_node(claim("loser", "the build passes"))
            .unwrap();
        store
            .upsert_node(claim("attacker", "the build fails"))
            .unwrap();
        // attacker --CONTRADICTS--> loser  => loser is grounded `out`.
        store
            .upsert_edge(EdgeRecord::new(
                "atk",
                "attacker",
                "CONTRADICTS",
                "loser",
                json!({}),
            ))
            .unwrap();
        shadows_for(&mut store, &["winner", "loser", "attacker"]);

        let standing =
            epistemic_egraph_dedup(&mut store, &[], EpistemicDedupConfig::default()).unwrap();
        // winner(in) and loser(out) share a claim but differ in standing.
        assert!(
            !standing
                .classes
                .iter()
                .any(|c| c.member_content_ids.contains(&"winner".to_string())
                    && c.member_content_ids.contains(&"loser".to_string())),
            "ClaimAndStanding must keep in/out apart"
        );

        // ClaimOnly ignores standing and merges the two identical claims.
        let mut store2 = InMemoryGraphStore::new();
        store2
            .upsert_node(claim("winner", "the build passes"))
            .unwrap();
        store2
            .upsert_node(claim("loser", "the build passes"))
            .unwrap();
        store2
            .upsert_node(claim("attacker", "the build fails"))
            .unwrap();
        store2
            .upsert_edge(EdgeRecord::new(
                "atk",
                "attacker",
                "CONTRADICTS",
                "loser",
                json!({}),
            ))
            .unwrap();
        shadows_for(&mut store2, &["winner", "loser", "attacker"]);
        let claim_only = epistemic_egraph_dedup(
            &mut store2,
            &[],
            EpistemicDedupConfig {
                congruence: EpistemicCongruence::ClaimOnly,
                ..EpistemicDedupConfig::default()
            },
        )
        .unwrap();
        assert!(
            claim_only
                .classes
                .iter()
                .any(|c| c.member_content_ids.contains(&"winner".to_string())
                    && c.member_content_ids.contains(&"loser".to_string())),
            "ClaimOnly merges identical claims across standing"
        );
    }

    // -----------------------------------------------------------------
    // SPEC-RUSTYRED-EDGE-ONTOLOGY-DUAL-PROVENANCE-1.0, section 8
    // -----------------------------------------------------------------

    fn relation(from: &str, to: &str, kind: EpistemicRelationKind) -> EpistemicRelationInput {
        EpistemicRelationInput {
            from_content_id: from.to_string(),
            to_content_id: to.to_string(),
            kind,
            confidence: 1.0,
            evidence: "fixture".to_string(),
            source_kind: EpistemicSourceKind::Structural,
            score: None,
            model_id: None,
            calibration_version: None,
            feature_version: None,
            connection_features: None,
            canonical_edge_type: None,
        }
    }

    /// An `Undercuts` relation carrying `canonical` as its stored row.
    fn with_canonical(from: &str, to: &str, canonical: &str) -> EpistemicRelationInput {
        EpistemicRelationInput {
            canonical_edge_type: Some(canonical.to_string()),
            ..relation(from, to, EpistemicRelationKind::Undercuts)
        }
    }

    /// Run the pass over `relations` and pair each node's stored grounded
    /// status with the boolean dual readout of its polynomial.
    fn grounded_and_dual(
        node_ids: &[&str],
        relations: Vec<EpistemicRelationInput>,
    ) -> Vec<(String, GroundedExtensionStatus, Standing)> {
        let mut store = InMemoryGraphStore::new();
        for id in node_ids {
            store.upsert_node(claim(id, id)).unwrap();
        }
        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: node_ids.iter().map(|id| id.to_string()).collect(),
                explicit_relations: relations.clone(),
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();

        let cycles = contradiction_cycles(&relations)
            .unwrap()
            .into_keys()
            .collect::<BTreeSet<_>>();
        let annotations = RelationAnnotations(&relations, default_registry());
        node_ids
            .iter()
            .map(|id| {
                let grounded = read_epistemic_shadow(&store, id)
                    .expect("shadow")
                    .grounded_extension_status;
                let poly = epistemic_node_polynomial(id, &relations, &cycles, default_registry());
                // The pass applies a one-level defeat rule, so the dual side
                // is asked for that rule rather than the bare pair reading.
                let standing = interpret::<BooleanSemiring>(&poly, &annotations)
                    .into_grounded(&false)
                    .standing;
                (id.to_string(), grounded, standing)
            })
            .collect()
    }

    fn equivalent(grounded: &GroundedExtensionStatus, standing: &Standing) -> bool {
        matches!(
            (grounded, standing),
            (GroundedExtensionStatus::In, Standing::In)
                | (GroundedExtensionStatus::Out, Standing::Out)
                | (GroundedExtensionStatus::Undecided, Standing::Undecided)
        )
    }

    /// Section 8.5, acyclic half.
    #[test]
    fn grounded_parity_holds_on_acyclic_attack_graphs() {
        let observed = grounded_and_dual(
            &["a", "b", "c", "d", "e"],
            vec![
                relation("a", "b", EpistemicRelationKind::Supports),
                relation("c", "d", EpistemicRelationKind::Undercuts),
                relation("a", "e", EpistemicRelationKind::Supports),
                relation("c", "e", EpistemicRelationKind::Undercuts),
            ],
        );
        let expected = [
            ("a", GroundedExtensionStatus::In),
            ("b", GroundedExtensionStatus::In),
            ("c", GroundedExtensionStatus::In),
            ("d", GroundedExtensionStatus::Out),
            // Supported and attacked: the fixpoint's one-level rule says Out,
            // and defeat multiplying into every derivation says Out too.
            ("e", GroundedExtensionStatus::Out),
        ];
        for ((id, grounded, standing), (want_id, want)) in observed.iter().zip(expected) {
            assert_eq!(id, want_id);
            assert_eq!(grounded, &want, "grounded status changed at {id}");
            assert!(
                equivalent(grounded, standing),
                "{id}: grounded {grounded:?} disagrees with dual standing {standing:?}"
            );
        }
    }

    /// Section 8.5, cycle half.
    #[test]
    fn grounded_parity_reports_undecided_on_the_two_cycle_fixture() {
        let observed = grounded_and_dual(
            &["x", "y"],
            vec![
                relation("x", "y", EpistemicRelationKind::Undercuts),
                relation("y", "x", EpistemicRelationKind::Undercuts),
            ],
        );
        for (id, grounded, standing) in &observed {
            assert_eq!(grounded, &GroundedExtensionStatus::Undecided, "{id}");
            assert_eq!(standing, &Standing::Undecided, "{id}");
        }
    }

    /// Section 8.6: the pass's own edge-type constants, iterated rather than
    /// copied, all resolve in the default registry with the right class.
    #[test]
    fn every_edge_type_the_pass_emits_resolves_in_the_default_registry() {
        let registry = default_registry();
        let expected = [
            (EPISTEMIC_SUPPORTS, AlgebraicClass::MonotonePositive),
            (UNDERCUTS, AlgebraicClass::Defeating),
            (SAME_ECLASS, AlgebraicClass::Equivalence),
            (HAS_EPISTEMIC_SHADOW, AlgebraicClass::Structural),
        ];
        for (token, class) in expected {
            assert_eq!(
                registry.class_of(token),
                Some(class),
                "constant {token} does not resolve"
            );
        }
        for kind in [
            EpistemicRelationKind::Supports,
            EpistemicRelationKind::Undercuts,
        ] {
            assert!(
                registry.resolve(kind.edge_type()).is_some(),
                "relation kind {kind:?} writes an unregistered edge type"
            );
        }
    }

    /// Section 8.7: the old explicit string lists still deserialize and still
    /// classify identically.
    #[test]
    fn old_shape_config_json_deserializes_and_reads_out_identically() {
        let legacy = r#"{
            "engine": "rustyred-thg-core.structural_epistemic",
            "engine_version": "epistemic-v1",
            "computed_at": 0,
            "support_edge_types": ["Supports", "supports", "SUPPORTS", "CITES", "DERIVED_FROM"],
            "attack_edge_types": ["Undercuts", "CONTRADICTS", "contradicts", "ATTACKS", "attacks"],
            "candidate_top_k": 8
        }"#;
        let legacy: StructuralEpistemicConfig = serde_json::from_str(legacy).expect("deserialize");
        let classes = ConfiguredEdgeClasses::from_config(&legacy).expect("all tokens are declared");
        for token in ["Supports", "supports", "SUPPORTS", "CITES", "DERIVED_FROM"] {
            assert_eq!(
                classes.kind_of(token),
                Some(EpistemicRelationKind::Supports),
                "{token}"
            );
        }
        for token in [
            "Undercuts",
            "CONTRADICTS",
            "contradicts",
            "ATTACKS",
            "attacks",
        ] {
            assert_eq!(
                classes.kind_of(token),
                Some(EpistemicRelationKind::Undercuts),
                "{token}"
            );
        }
        // The old lists never named Tension, so a config that spells them out
        // keeps the old scope and existing callers see no change.
        assert_eq!(classes.kind_of("Tension"), None);

        let mut legacy_store = InMemoryGraphStore::new();
        let mut default_store = InMemoryGraphStore::new();
        for store in [&mut legacy_store, &mut default_store] {
            store.upsert_node(claim("a", "A")).unwrap();
            store.upsert_node(claim("b", "B")).unwrap();
            store
                .upsert_edge(EdgeRecord::new(
                    "e_ab",
                    "a",
                    EPISTEMIC_SUPPORTS,
                    "b",
                    json!({}),
                ))
                .unwrap();
        }
        let mut fixed = legacy.clone();
        fixed.computed_at = 0;
        let legacy_readout = structural_epistemic_pass(
            &mut legacy_store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                config: fixed,
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        let default_readout = structural_epistemic_pass(
            &mut default_store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                config: StructuralEpistemicConfig {
                    computed_at: 0,
                    ..StructuralEpistemicConfig::default()
                },
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();
        assert_eq!(legacy_readout, default_readout);
    }

    #[test]
    fn a_config_edge_type_the_registry_does_not_know_is_a_construction_error() {
        let config = StructuralEpistemicConfig {
            support_edge_types: vec!["Supports".to_string(), "ENDORSES".to_string()],
            ..StructuralEpistemicConfig::default()
        };
        let err = ConfiguredEdgeClasses::from_config(&config).unwrap_err();
        assert_eq!(err.code, "unknown_epistemic_edge_type");
        assert!(err.message.contains("ENDORSES"), "{}", err.message);

        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        let err = structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string()],
                config,
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.code, "unknown_epistemic_edge_type");
    }

    /// Section 7.3: an unresolved tension is epistemic pressure, and the pass
    /// had been silently dropping it.
    #[test]
    fn tension_edges_now_count_as_defeating_pressure() {
        assert_eq!(
            default_registry().class_of("Tension"),
            Some(AlgebraicClass::Defeating)
        );
        assert!(StructuralEpistemicConfig::default()
            .attack_edge_types
            .iter()
            .any(|token| token == "Tension"));

        let subgraph = UserSubgraph {
            nodes: vec![claim("a", "A"), claim("b", "B")],
            edges: vec![EdgeRecord::new("e", "a", "Tension", "b", json!({}))],
        };
        let index = graph_signal_index(&subgraph);
        assert_eq!(index.attack_in.get("b").copied(), Some(1));
        assert_eq!(index.support_in.get("b").copied(), None);
    }

    #[test]
    fn graph_signal_index_ignores_unclassified_and_never_valued_edge_types() {
        let subgraph = UserSubgraph {
            nodes: vec![claim("a", "A"), claim("b", "B")],
            edges: vec![
                EdgeRecord::new("e1", "a", "mentions", "b", json!({})),
                EdgeRecord::new("e2", "a", HAS_EPISTEMIC_SHADOW, "b", json!({})),
                EdgeRecord::new("e3", "a", SAME_ECLASS, "b", json!({})),
                EdgeRecord::new("e4", "a", "derived_from", "b", json!({})),
            ],
        };
        let index = graph_signal_index(&subgraph);
        assert_eq!(index.attack_in.get("b").copied(), None);
        assert_eq!(index.support_in.get("b").copied(), Some(1));
    }

    /// A Contradicts edge and a Tension edge must survive the pass as
    /// themselves. EpistemicRelationKind has only Supports and Undercuts, so
    /// every Defeating row collapses onto Undercuts on the way in; emitting the
    /// shadow edge from the kind alone rewrote both as directed inference
    /// undercuts and destroyed the distinction the pass classifies by.
    #[test]
    fn defeating_relations_keep_their_own_kind_through_the_pass() {
        for (edge_type, want_type, want_epistemic) in [
            ("CONTRADICTS", "Contradicts", EpistemicType::Contradicts),
            ("Tension", "Tension", EpistemicType::Tension),
            ("Undercuts", "Undercuts", EpistemicType::Undercuts),
        ] {
            let mut store = InMemoryGraphStore::new();
            store.upsert_node(claim("a", "A")).unwrap();
            store.upsert_node(claim("b", "B")).unwrap();
            store
                .upsert_edge(EdgeRecord::new("e", "a", edge_type, "b", json!({})))
                .unwrap();

            let readout = structural_epistemic_pass(
                &mut store,
                StructuralEpistemicInput {
                    batch_node_ids: vec!["a".to_string(), "b".to_string()],
                    ..StructuralEpistemicInput::default()
                },
            )
            .unwrap();

            // It still counts as defeat: b is attacked and grounded Out.
            assert_eq!(readout.contradictions.len(), 1, "{edge_type}");
            let shadow = read_epistemic_shadow(&store, "b").expect("shadow");
            assert_eq!(shadow.attack_in_degree, 1, "{edge_type}");
            assert_eq!(
                shadow.grounded_extension_status,
                GroundedExtensionStatus::Out,
                "{edge_type}"
            );

            // And the emitted shadow edge is still its own relation, not an
            // undercut standing in for all three.
            let from_shadow =
                epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
            let emitted = store
                .neighbors(NeighborQuery::out(&from_shadow))
                .into_iter()
                .filter_map(|hit| store.get_edge(&hit.edge_id).cloned())
                .find(|edge| edge.epistemic_type.is_some())
                .unwrap_or_else(|| panic!("{edge_type}: no shadow relation written"));
            assert_eq!(emitted.edge_type, want_type, "{edge_type} edge type");
            assert_eq!(
                emitted.epistemic_type,
                Some(want_epistemic),
                "{edge_type} epistemic type"
            );
        }
    }

    #[test]
    fn a_caller_supplied_canonical_type_that_contradicts_its_kind_is_ignored() {
        // canonical_edge_type is public and deserializable, so a caller can send
        // a row whose class disagrees with `kind`. Support and attack counts and
        // the grounded status come from `kind`, so honouring the canonical would
        // write a shadow edge whose type contradicts the arithmetic behind it.
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        store.upsert_node(claim("b", "B")).unwrap();

        let readout = structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                explicit_relations: vec![EpistemicRelationInput {
                    from_content_id: "a".to_string(),
                    to_content_id: "b".to_string(),
                    kind: EpistemicRelationKind::Supports,
                    confidence: 1.0,
                    evidence: "conflicting hint".to_string(),
                    source_kind: EpistemicSourceKind::Structural,
                    score: None,
                    model_id: None,
                    calibration_version: None,
                    feature_version: None,
                    connection_features: None,
                    canonical_edge_type: Some("Contradicts".to_string()),
                }],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();

        // The kind said Supports, so nothing was attacked.
        assert!(readout.contradictions.is_empty());
        let shadow = read_epistemic_shadow(&store, "b").expect("shadow");
        assert_eq!(shadow.attack_in_degree, 0);
        assert_eq!(shadow.support_in_degree, 1);

        // And the emitted edge agrees with the arithmetic, not with the hint.
        let from_shadow = epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        let emitted = store
            .neighbors(NeighborQuery::out(&from_shadow))
            .into_iter()
            .filter_map(|hit| store.get_edge(&hit.edge_id).cloned())
            .find(|edge| edge.epistemic_type.is_some())
            .expect("a shadow relation is written");
        assert_eq!(emitted.edge_type, "Supports");
        assert_eq!(emitted.epistemic_type, Some(EpistemicType::Supports));
    }

    #[test]
    fn a_pair_carrying_both_support_and_defeat_keeps_each_relation_distinct() {
        // The ontology row cannot be recovered from the node pair: the same
        // ordered pair can carry a MonotonePositive edge and a Defeating one at
        // once. Keying the row by (from, to) let whichever edge was seen first
        // decide the type of both, so the support was emitted as an attack or
        // the attack as a support. The row rides on the relation instead.
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        store.upsert_node(claim("b", "B")).unwrap();
        store
            .upsert_edge(EdgeRecord::new("e_con", "a", "CONTRADICTS", "b", json!({})))
            .unwrap();
        store
            .upsert_edge(EdgeRecord::new("e_sup", "a", "Supports", "b", json!({})))
            .unwrap();

        structural_epistemic_pass(
            &mut store,
            StructuralEpistemicInput {
                batch_node_ids: vec!["a".to_string(), "b".to_string()],
                ..StructuralEpistemicInput::default()
            },
        )
        .unwrap();

        let from_shadow = epistemic_shadow_node_id("a", DEFAULT_EPISTEMIC_ENGINE_VERSION).unwrap();
        let mut emitted = store
            .neighbors(NeighborQuery::out(&from_shadow))
            .into_iter()
            .filter_map(|hit| store.get_edge(&hit.edge_id).cloned())
            .filter(|edge| edge.epistemic_type.is_some())
            .map(|edge| (edge.edge_type.clone(), edge.epistemic_type))
            .collect::<Vec<_>>();
        emitted.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(
            emitted,
            vec![
                ("Contradicts".to_string(), Some(EpistemicType::Contradicts)),
                ("Supports".to_string(), Some(EpistemicType::Supports)),
            ],
            "each relation keeps its own ontology row"
        );
    }

    // -- section 4 / acceptance criterion 2 ----------------------------

    #[test]
    fn undercuts_round_trips_and_is_no_longer_conflated_with_contradicts() {
        assert_eq!(
            "undercuts".parse::<EpistemicType>().unwrap(),
            EpistemicType::Undercuts
        );
        assert_eq!(EpistemicType::Undercuts.to_string(), "undercuts");
        assert_eq!(
            EpistemicType::Undercuts
                .to_string()
                .parse::<EpistemicType>()
                .unwrap(),
            EpistemicType::Undercuts
        );
        assert_eq!(
            EpistemicRelationKind::Undercuts.epistemic_type(),
            EpistemicType::Undercuts
        );
        assert_ne!(EpistemicType::Undercuts, EpistemicType::Contradicts);
        // Both are Defeating, so consumers that treated them identically still do.
        for token in ["undercuts", "contradicts"] {
            assert_eq!(
                default_registry().class_of(token),
                Some(AlgebraicClass::Defeating)
            );
        }
    }

    #[test]
    fn edges_serialized_before_the_new_variant_deserialize_unchanged() {
        // A byte-for-byte edge payload from before `Undercuts` existed.
        let legacy = r#"{"id":"e1","from_id":"a","to_id":"b","type":"Undercuts",
            "properties":{},"version":3,"tombstone":false,"confidence":0.75,
            "epistemic_type":"contradicts"}"#;
        let edge: EdgeRecord = serde_json::from_str(legacy).expect("legacy edge deserializes");
        assert_eq!(edge.epistemic_type, Some(EpistemicType::Contradicts));
        assert_eq!(edge.content_address().unwrap(), {
            let mut rebuilt = EdgeRecord::new("e1", "a", "Undercuts", "b", json!({}))
                .with_epistemic_type(EpistemicType::Contradicts)
                .with_confidence(0.75);
            rebuilt.version = 3;
            rebuilt.content_address().unwrap()
        });
        // The new variant serializes under its own name, not the old one.
        let fresh = EdgeRecord::new("e2", "a", "Undercuts", "b", json!({}))
            .with_epistemic_type(EpistemicType::Undercuts);
        let json = serde_json::to_string(&fresh).unwrap();
        assert!(json.contains(r#""epistemic_type":"undercuts""#), "{json}");
    }

    // -- section 7 adapter ---------------------------------------------

    #[test]
    fn lift_edge_mints_polarity_from_the_registry_and_nothing_from_plumbing() {
        let registry = default_registry();
        let supports = EdgeRecord::new("e1", "a", "SUPPORTS", "b", json!({}));
        let generator = lift_edge(&supports, registry).expect("supports mints a generator");
        assert_eq!(generator.polarity, Polarity::Positive);
        assert_eq!(generator.canonical_type, "Supports");
        assert_eq!(generator.token.as_str(), "e1");

        let attacks = EdgeRecord::new("e2", "a", "attacks", "b", json!({}));
        let generator = lift_edge(&attacks, registry).expect("attacks mints a generator");
        assert_eq!(generator.polarity, Polarity::Negative);
        assert_eq!(generator.canonical_type, "Contradicts");

        for edge_type in [SAME_ECLASS, HAS_EPISTEMIC_SHADOW, "mentions"] {
            let edge = EdgeRecord::new("e3", "a", edge_type, "b", json!({}));
            assert!(
                lift_edge(&edge, registry).is_none(),
                "{edge_type} must mint nothing"
            );
        }
    }

    #[test]
    fn relation_tokens_survive_content_ids_containing_the_delimiter() {
        // `from="a->b", to="c"` and `from="a", to="b->c"` concatenate to the
        // same string under a naive delimiter join. Length prefixes keep them
        // distinct, so they cannot share a generator and annotation.
        let left = relation("a->b", "c", EpistemicRelationKind::Supports);
        let right = relation("a", "b->c", EpistemicRelationKind::Supports);
        assert_ne!(
            relation_token(&left, default_registry()),
            relation_token(&right, default_registry())
        );

        // Relation kind still separates the same pair.
        let supports = relation("a", "b", EpistemicRelationKind::Supports);
        let undercuts = relation("a", "b", EpistemicRelationKind::Undercuts);
        assert_ne!(
            relation_token(&supports, default_registry()),
            relation_token(&undercuts, default_registry())
        );

        // And the encoding stays deterministic.
        assert_eq!(
            relation_token(&supports, default_registry()),
            relation_token(&supports.clone(), default_registry())
        );
    }

    #[test]
    fn duplicate_relations_annotate_the_same_whichever_is_listed_first() {
        // Two assertions of one relation differing only in confidence mint the
        // same token by construction. A `.find()` read whichever came first, so
        // reversing the slice changed every weighted readout for the same
        // relation multiset.
        let strong = EpistemicRelationInput {
            confidence: 0.9,
            evidence: "strong source".to_string(),
            ..with_canonical("a", "b", "Contradicts")
        };
        let weak = EpistemicRelationInput {
            confidence: 0.3,
            evidence: "weak source".to_string(),
            ..with_canonical("a", "b", "Contradicts")
        };
        let token = EdgeToken::new(relation_token(&strong, default_registry()));
        assert_eq!(
            relation_token(&weak, default_registry()),
            token.as_str(),
            "the two assertions must share one token, or this test proves nothing"
        );

        let forward = RelationAnnotations(&[strong.clone(), weak.clone()], default_registry())
            .annotation(&token);
        let reversed = RelationAnnotations(&[weak, strong], default_registry()).annotation(&token);

        assert_eq!(
            forward, reversed,
            "input order must not change the annotation"
        );
        assert_eq!(forward, 0.9, "the strongest stated confidence stands");
    }

    #[test]
    fn an_alias_spelled_canonical_row_mints_one_token_not_two() {
        // `relation_token` embedded the caller's raw spelling while
        // `lift_relation` resolved it, so `Contradicts` and `CONTRADICTS` minted
        // one generator but two tokens. One relation became two provenance
        // variables, and probability_readout is inclusion-exclusion over
        // monomial supports, so it counted the duplicate as independent
        // evidence.
        let canonical = with_canonical("a", "b", "Contradicts");
        let shouted = with_canonical("a", "b", "CONTRADICTS");
        assert_eq!(
            relation_token(&canonical, default_registry()),
            relation_token(&shouted, default_registry()),
            "one relation spelled two ways is one provenance variable"
        );

        // The round 13 fix still holds: parallel defeating rows stay distinct.
        let tension = with_canonical("a", "b", "Tension");
        assert_ne!(
            relation_token(&canonical, default_registry()),
            relation_token(&tension, default_registry()),
            "Contradicts and Tension are different rows on the same pair"
        );

        // A value that resolves into another class is ignored by
        // `lift_relation`, so it must not change identity either.
        let cross_class = with_canonical("a", "b", "Supports");
        let unstated = relation("a", "b", EpistemicRelationKind::Undercuts);
        assert_eq!(
            relation_token(&cross_class, default_registry()),
            relation_token(&unstated, default_registry()),
            "a cross-class row falls back to the kind on both paths"
        );
    }

    #[test]
    fn the_bounded_pair_contradiction_is_tagged_as_a_contradiction() {
        // The evidence string this producer writes literally says
        // bounded_pair_contradiction, but the kind enum carries only Supports
        // and Undercuts. Without canonical_edge_type the persisted edge claims
        // an attack on the inference rather than on the conclusion.
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("l", "the cache is warm")).unwrap();
        store
            .upsert_node(claim("r", "the cache is not warm"))
            .unwrap();
        store.upsert_node(claim("s", "the cache is warm")).unwrap();

        let negation = infer_pair_relation(
            &store,
            &EpistemicCandidatePair {
                left_content_id: "l".to_string(),
                right_content_id: "r".to_string(),
            },
        )
        .expect("a negation pair infers a defeating relation");
        assert_eq!(negation.kind, EpistemicRelationKind::Undercuts);
        assert_eq!(
            negation.canonical_edge_type.as_deref(),
            Some("Contradicts"),
            "a negation pair must be tagged Contradicts"
        );

        // The equivalent-claim branch is a genuine support and keeps the
        // kind's own row.
        let same = infer_pair_relation(
            &store,
            &EpistemicCandidatePair {
                left_content_id: "l".to_string(),
                right_content_id: "s".to_string(),
            },
        )
        .expect("identical claims infer a supporting relation");
        assert_eq!(same.kind, EpistemicRelationKind::Supports);
        assert_eq!(same.canonical_edge_type, None);
    }

    #[test]
    fn parallel_defeating_rows_are_valued_by_their_own_confidence() {
        use crate::graph::ontology_adapter::lift_relation;
        // Contradicts and Tension on the same endpoints both collapse to
        // EpistemicRelationKind::Undercuts. Their generators stay distinct
        // because canonical_type differs, but the annotation lookup finds by
        // token, so a shared token valued both from whichever came first.
        let registry = default_registry();
        let mut strong = relation("a", "b", EpistemicRelationKind::Undercuts);
        strong.canonical_edge_type = Some("Contradicts".to_string());
        strong.confidence = 0.9;
        let mut weak = relation("a", "b", EpistemicRelationKind::Undercuts);
        weak.canonical_edge_type = Some("Tension".to_string());
        weak.confidence = 0.1;
        let relations = vec![strong.clone(), weak.clone()];

        let strong_generator = lift_relation(&strong, registry).expect("lifts");
        let weak_generator = lift_relation(&weak, registry).expect("lifts");
        assert_ne!(
            strong_generator.token, weak_generator.token,
            "parallel defeating rows must not share a provenance token"
        );

        let annotations = RelationAnnotations(&relations, default_registry());
        assert!(
            (annotations.annotation(&strong_generator.token) - 0.9).abs() < 1e-9,
            "Contradicts valued {}",
            annotations.annotation(&strong_generator.token)
        );
        assert!(
            (annotations.annotation(&weak_generator.token) - 0.1).abs() < 1e-9,
            "Tension valued {}",
            annotations.annotation(&weak_generator.token)
        );
    }

    #[test]
    fn edge_type_classification_resolves_aliases_and_case() {
        // The projection write guard classifies a caller-supplied spelling.
        // Exact membership against the canonical names let a bulk edge spelled
        // contradicts, UNDERCUTS or attacks past the guard and overwrite
        // projection-maintained data.
        for spelling in [
            "Contradicts",
            "contradicts",
            "CONTRADICTS",
            "attacks",
            "ATTACKS",
            "UNDERCUTS",
            "undercuts",
            "tension",
            "SUPPORTS",
            "derived_from",
        ] {
            assert!(
                is_valued_epistemic_edge_type(spelling),
                "{spelling} is a valued row and must classify as one"
            );
        }
        for spelling in ["SameEClass", "sameeclass", "HasEpistemicShadow"] {
            assert!(
                is_structural_epistemic_edge_type(spelling),
                "{spelling} must classify as structural"
            );
            assert!(!is_valued_epistemic_edge_type(spelling));
        }
        for spelling in ["NotAnEdgeType", "", "CHILD"] {
            assert!(!is_valued_epistemic_edge_type(spelling));
            assert!(!is_structural_epistemic_edge_type(spelling));
        }
    }

    #[test]
    fn the_valued_edge_type_enumeration_covers_every_row_the_pass_can_emit() {
        // Consumers enumerate shadow edge types from this. A row missing here
        // disappears from the shadow PPR topology and from the projection
        // write guard.
        let types = valued_epistemic_edge_types();
        for expected in [
            "Supports",
            "Cites",
            "Derives",
            "Contradicts",
            "Tension",
            "Undercuts",
        ] {
            assert!(
                types.contains(&expected),
                "{expected} is a valued row the pass can emit but is not enumerated: {types:?}"
            );
        }
        assert!(
            !types.contains(&SAME_ECLASS),
            "SameEClass is an equivalence row, not a valued one"
        );
        assert!(!types.contains(&HAS_EPISTEMIC_SHADOW));
    }

    #[test]
    fn a_lifted_relation_keeps_the_canonical_row_it_came_from() {
        use crate::graph::ontology_adapter::lift_relation;
        let registry = default_registry();

        // Every defeating relation collapses to EpistemicRelationKind::Undercuts,
        // so resolving the kind alone tags a stored Contradicts edge as
        // Undercuts and a caller cannot tell the three defeating relations
        // apart, which is what Generator::canonical_type exists to carry.
        for stored in ["Contradicts", "Tension", "Undercuts"] {
            let mut r = relation("a", "b", EpistemicRelationKind::Undercuts);
            r.canonical_edge_type = Some(stored.to_string());
            let generator = lift_relation(&r, registry).expect("defeating rows lift");
            assert_eq!(
                generator.canonical_type, stored,
                "lifting lost the stored row {stored}"
            );
            assert_eq!(generator.polarity, Polarity::Negative);
        }

        // A stored row from another class must not flip the polarity. It is
        // rejected and the kind's own row stands.
        let mut crossed = relation("a", "b", EpistemicRelationKind::Undercuts);
        crossed.canonical_edge_type = Some("Supports".to_string());
        let generator = lift_relation(&crossed, registry).expect("defeating rows lift");
        assert_eq!(
            generator.polarity,
            Polarity::Negative,
            "class must be validated"
        );
        assert_eq!(generator.canonical_type, "Undercuts");

        // No stored row: the kind still resolves.
        let bare = relation("a", "b", EpistemicRelationKind::Supports);
        let generator = lift_relation(&bare, registry).expect("supporting rows lift");
        assert_eq!(generator.polarity, Polarity::Positive);
        assert_eq!(generator.canonical_type, "Supports");
    }

    #[test]
    fn an_attacked_node_still_reports_a_usable_net_probability() {
        // net_probability was only ever exercised on a hand-built readout, so
        // nothing caught that no real node polynomial could produce one. Defeat
        // used to multiply into the base, which left no positive-only monomial,
        // so for_value read zero for every attacked node and the net collapsed
        // to zero however weak the attack.
        let registry = default_registry();
        let cycles = BTreeSet::new();

        let mut support = relation("a", "b", EpistemicRelationKind::Supports);
        support.confidence = 0.8;
        let mut attack = relation("c", "b", EpistemicRelationKind::Undercuts);
        attack.confidence = 0.25;
        let relations = vec![support, attack];

        let poly = epistemic_node_polynomial("b", &relations, &cycles, registry);
        let readout =
            probability_readout(&poly, &RelationAnnotations(&relations, default_registry()))
                .expect("two monomials is well inside the inclusion-exclusion bound");

        assert!(
            (readout.for_value - 0.8).abs() < 1e-9,
            "support must survive defeat, got {}",
            readout.for_value
        );
        assert!(
            (readout.against_value - 0.25).abs() < 1e-9,
            "attack strength must reach the against side, got {}",
            readout.against_value
        );
        // The documented noisy-and-not collapse, now reachable end to end.
        assert!(
            (net_probability(&readout) - 0.6).abs() < 1e-9,
            "expected 0.8 * (1 - 0.25) = 0.6, got {}",
            net_probability(&readout)
        );
        // Standing still goes Out, through the one-level defeat rule.
        assert_eq!(readout.into_grounded(&0.0).standing, Standing::Out);
    }

    #[test]
    fn a_zero_confidence_relation_asserts_nothing() {
        let registry = default_registry();
        let cycles = BTreeSet::new();

        let mut dead_attack = relation("a", "b", EpistemicRelationKind::Undercuts);
        dead_attack.confidence = 0.0;
        let relations = vec![dead_attack];

        // Admitting it would put a generator on the against side that carries
        // no weight, flipping the node Out on an attack that asserts nothing.
        // Skipping it leaves the node's own standing intact.
        let poly = epistemic_node_polynomial("b", &relations, &cycles, registry);
        let readout = interpret::<BooleanSemiring>(
            &poly,
            &RelationAnnotations(&relations, default_registry()),
        )
        .into_grounded(&false);
        assert!(readout.for_value, "the base must survive a null attack");
        assert!(!readout.against_value);
        assert_eq!(readout.standing, Standing::In);

        // A live attack at the same position still lands. The base survives it
        // on the for side, which is what keeps a net probability meaningful;
        // the standing goes Out through the one-level defeat rule, not by
        // erasing the support.
        let live = vec![relation("a", "b", EpistemicRelationKind::Undercuts)];
        let poly = epistemic_node_polynomial("b", &live, &cycles, registry);
        let readout =
            interpret::<BooleanSemiring>(&poly, &RelationAnnotations(&live, default_registry()))
                .into_grounded(&false);
        assert!(
            readout.for_value,
            "defeat must not erase the supporting side"
        );
        assert!(readout.against_value);
        assert_eq!(readout.standing, Standing::Out);
    }

    #[test]
    fn store_annotations_weight_confidence_by_recorded_source_reliability() {
        let mut store = InMemoryGraphStore::new();
        store.upsert_node(claim("a", "A")).unwrap();
        store.upsert_node(claim("b", "B")).unwrap();
        store
            .upsert_edge(
                EdgeRecord::new("e_plain", "a", EPISTEMIC_SUPPORTS, "b", json!({}))
                    .with_confidence(0.8),
            )
            .unwrap();
        store
            .upsert_edge(
                EdgeRecord::new(
                    "e_weighted",
                    "a",
                    EPISTEMIC_SUPPORTS,
                    "b",
                    json!({ "source_reliability": { "alpha": 1.0, "beta": 1.0, "mean": 0.5 } }),
                )
                .with_confidence(0.8),
            )
            .unwrap();

        let annotations = StoreAnnotations(&store);
        assert!((annotations.annotation(&EdgeToken::new("e_plain")) - 0.8).abs() < 1e-9);
        assert!((annotations.annotation(&EdgeToken::new("e_weighted")) - 0.4).abs() < 1e-9);
        // A token naming no edge annotates as the unweighted default.
        assert!((annotations.annotation(&EdgeToken::new("missing")) - 1.0).abs() < 1e-9);
    }
}
