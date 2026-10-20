//! Incremental Algebra fold: node-vector regions over the typed graph log.
//!
//! Native GraphBLAS matrices remain disposable derived state behind
//! `feature = "graphblas"`. Token codes, kind masks, and leaf vectors are
//! ungated fold state.

use super::{Fold, FoldCapabilities, FoldDescriptor, FoldError, FoldId, FoldVersion};
use crate::fold_quarantine::{FoldConsumeError, FoldRefusalCode};
use crate::graph::fold_store::{
    GraphIntentBatch, GraphSemanticChange, GRAPH_INTENT_NAME, GRAPH_INTENT_NAMESPACE,
    GRAPH_INTENT_VERSION,
};
use crate::graph::record::{
    EdgeRecord, NodeRecord, CALIBRATION_EPOCH_LABEL, EMBEDDING_EXCLUDED_PROPERTY,
    EMBEDDING_PROPERTY, VALUATION_PROPERTY,
};
use crate::intern::{intern, symbol_of, Symbol};
use crate::token_encoder::packed_bytes_per_vector;
use rustyred_thg_derivation::{resolve_algebra_token, AlgebraClass, LawProfile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::str::FromStr;
use theorem_oplog::{DeploymentId, LogPosition, LogRecord};
use turbovec::TurboQuantIndex;

pub const FOLD_ID: FoldId = FoldId::Algebra;
pub const FOLD_VERSION: u32 = 3;
pub const ALGEBRA_FOLD_VERSION: u32 = FOLD_VERSION;
pub const TURBOVEC_ENCODE_VERSION: &str = "1.0.0";
pub const TOKEN_BIT_WIDTH: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NodeIndex(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TokenSlot(pub u64);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlgebraLawIdentity {
    pub name: String,
    pub class: AlgebraClass,
    pub laws: LawProfile,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphBlasOperatorPairIdentity {
    pub name: String,
    pub domain: String,
    pub additive_monoid: String,
    pub multiply_operator: String,
    pub law_identity: AlgebraLawIdentity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlgebraOperationProfile {
    pub operation_id: String,
    pub graphblas_operator_pair: GraphBlasOperatorPairIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mathematical_semiring: Option<AlgebraLawIdentity>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlgebraValuationProfile {
    pub engine: String,
    pub operations: Vec<AlgebraOperationProfile>,
}

impl AlgebraValuationProfile {
    pub fn reachability() -> Self {
        Self {
            engine: "suitesparse_graphblas".to_string(),
            operations: vec![reachability_operation()],
        }
    }

    pub fn v3() -> Self {
        Self {
            engine: "suitesparse_graphblas".to_string(),
            operations: vec![
                reachability_operation(),
                fp32_operation(
                    "activation_spread",
                    "plus_times_fp32",
                    "plus_fp32",
                    "times_fp32",
                ),
                fp32_operation(
                    "competitive_activation",
                    "max_plus_fp32",
                    "max_fp32",
                    "plus_fp32",
                ),
                AlgebraOperationProfile {
                    operation_id: "memtensor_compose".to_string(),
                    graphblas_operator_pair: GraphBlasOperatorPairIdentity {
                        name: "plus_times_fp32".to_string(),
                        domain: "f32".to_string(),
                        additive_monoid: "plus_fp32".to_string(),
                        multiply_operator: "times_fp32".to_string(),
                        law_identity: operator_pair_identity("graphblas_plus_times_fp32"),
                    },
                    mathematical_semiring: None,
                },
            ],
        }
    }
}

fn reachability_operation() -> AlgebraOperationProfile {
    let mathematical = resolve_algebra_token("boolean")
        .expect("V01 registers the boolean algebra token")
        .require_mathematical_semiring()
        .expect("V01 certifies the boolean mathematical semiring");
    AlgebraOperationProfile {
        operation_id: "reachability".to_string(),
        graphblas_operator_pair: GraphBlasOperatorPairIdentity {
            name: "lor_land_bool".to_string(),
            domain: "bool".to_string(),
            additive_monoid: "lor_bool".to_string(),
            multiply_operator: "land_bool".to_string(),
            law_identity: operator_pair_identity("graphblas_lor_land_bool"),
        },
        mathematical_semiring: Some(AlgebraLawIdentity {
            name: mathematical.name.to_string(),
            class: mathematical.class,
            laws: mathematical.laws,
        }),
    }
}

fn fp32_operation(
    operation_id: &str,
    pair: &str,
    additive: &str,
    multiply: &str,
) -> AlgebraOperationProfile {
    AlgebraOperationProfile {
        operation_id: operation_id.to_string(),
        graphblas_operator_pair: GraphBlasOperatorPairIdentity {
            name: pair.to_string(),
            domain: "f32".to_string(),
            additive_monoid: additive.to_string(),
            multiply_operator: multiply.to_string(),
            law_identity: operator_pair_identity(&format!("graphblas_{pair}")),
        },
        mathematical_semiring: None,
    }
}

fn operator_pair_identity(name: &str) -> AlgebraLawIdentity {
    AlgebraLawIdentity {
        name: name.to_string(),
        class: AlgebraClass::GraphBlasOperatorPair,
        laws: LawProfile::GRAPHBLAS_OPERATOR_PAIR,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeafRole {
    Signal,
    Code,
    Index,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeafDtype {
    F32,
    U32,
    U64,
    Bool,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LeafKey(String);

impl LeafKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LeafKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for LeafKey {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty()
            || value.starts_with('/')
            || value.ends_with('/')
            || value.split('/').any(|segment| {
                segment.is_empty() || segment == "." || segment == ".." || segment == "__metadata__"
            })
        {
            return Err(format!("invalid leaf key {value:?}"));
        }
        Ok(Self(value.to_string()))
    }
}

impl<'de> Deserialize<'de> for LeafKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LeafSource {
    Valuation,
    KindLabel { label: String },
    Live,
    MultivectorCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LeafSpec {
    pub key: LeafKey,
    pub role: LeafRole,
    pub dtype: LeafDtype,
    pub source: LeafSource,
    pub governing_mask: Option<LeafKey>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LeafValues {
    F32(Vec<f32>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeafVector {
    pub spec: LeafSpec,
    pub values: LeafValues,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemTensorKind {
    Memory,
    Claim,
    Commitment,
    UserModel,
    AgentModelActivation,
}

impl MemTensorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Memory => "Memory",
            Self::Claim => "Claim",
            Self::Commitment => "Commitment",
            Self::UserModel => "UserModel",
            Self::AgentModelActivation => "AgentModelActivation",
        }
    }

    pub fn leaf_key(self) -> LeafKey {
        LeafKey::from_str(match self {
            Self::Memory => "kind/memory",
            Self::Claim => "kind/claim",
            Self::Commitment => "kind/commitment",
            Self::UserModel => "kind/user_model",
            Self::AgentModelActivation => "kind/agent_model_activation",
        })
        .expect("kind leaf keys are valid")
    }

    pub fn all() -> [Self; 5] {
        [
            Self::Memory,
            Self::Claim,
            Self::Commitment,
            Self::UserModel,
            Self::AgentModelActivation,
        ]
    }

    fn from_label(label: &str) -> Option<Self> {
        Self::all().into_iter().find(|kind| kind.label() == label)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingExclusion {
    GraphInternalKind,
    BodyEmpty,
    TenantPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowEmbedding {
    Pending,
    Excluded {
        reason: EmbeddingExclusion,
    },
    Set {
        id: String,
        epoch: String,
        exact_object_ref: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEpoch {
    pub id: String,
    pub node_id: String,
    pub model_id: String,
    pub model_version: String,
    pub dim: usize,
    pub bit_width: u8,
    pub turbovec_version: String,
    pub sample_block: String,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    pub active: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenCodes {
    pub vector_count: u32,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
}

pub struct TokenResidency {
    pub index: TurboQuantIndex,
    pub slot_node: Vec<u64>,
    pub offsets: Vec<u64>,
    pub kind_masks: BTreeMap<LeafKey, Vec<bool>>,
}

pub struct TokenRegion {
    pub epoch: Option<CalibrationEpoch>,
    pub codes: BTreeMap<u64, TokenCodes>,
    pub state: Vec<RowEmbedding>,
    residency: Option<TokenResidency>,
}

#[derive(Clone, Debug, Default)]
pub struct AlgebraIndex {
    /// Row identity in interning order. Symbols are process-local; the
    /// canonical string is `Symbol::as_str`, and every durability boundary
    /// (checkpoint, manifest, oplog) resolves back to it.
    pub nodes_by_index: Vec<Symbol>,
    pub node_indices: HashMap<Symbol, u64>,
    pub live_nodes: HashSet<Symbol>,
    pub labels_by_index: Vec<Vec<String>>,
    pub edges_by_id: BTreeMap<String, EdgeRecord>,
    pub coordinate_counts: BTreeMap<(String, u64, u64), usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AlgebraStats {
    pub interval_batches: u64,
    pub native_full_builds: u64,
    pub native_delta_updates: u64,
    pub token_index_builds: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AlgebraResidency {
    pub regions: usize,
    pub matrix_entries: u64,
    pub matrix_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AlgebraQuery {
    Reachable {
        edge_type: String,
        sources: Vec<String>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlgebraReachability {
    pub edge_type: String,
    pub sources: Vec<String>,
    pub reachable: Vec<String>,
    pub position: LogPosition,
}

#[derive(Clone, Debug)]
pub struct BranchArrays {
    pub counts: Vec<u32>,
    pub targets: Vec<u64>,
    pub weights: Vec<f32>,
    pub edge_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct CsrView<'a> {
    pub counts: &'a [u32],
    pub targets: &'a [u64],
    pub weights: &'a [f32],
    pub edge_ids: &'a [String],
}

pub struct AlgebraProjection<'a> {
    pub log_id: theorem_oplog::LogId,
    pub through: LogPosition,
    pub node_count: u64,
    /// Row identity by index. Resolve one with [`AlgebraProjection::node_id`];
    /// go the other way with [`AlgebraProjection::node_index`], which is the
    /// fold's own reverse index rather than a scan.
    pub nodes_by_index: &'a [Symbol],
    pub node_indices: &'a HashMap<Symbol, u64>,
    pub leaves: &'a BTreeMap<LeafKey, LeafVector>,
    pub epoch: &'a CalibrationEpoch,
    pub embedding_state: &'a [RowEmbedding],
    pub tokens: &'a TokenResidency,
    pub edge_types: Vec<String>,
    pub branches: BTreeMap<String, CsrView<'a>>,
}

impl AlgebraProjection<'_> {
    /// The canonical identity string of one row.
    #[must_use]
    pub fn node_id(&self, node: NodeIndex) -> Option<&str> {
        self.nodes_by_index
            .get(usize::try_from(node.0).ok()?)
            .map(|symbol| symbol.as_str())
    }

    /// The row index of one identity, through the fold's reverse index.
    ///
    /// This is a hash probe, not a scan over `nodes_by_index`, and an id the
    /// fold has never admitted resolves to `None` without interning it.
    #[must_use]
    pub fn node_index(&self, id: &str) -> Option<NodeIndex> {
        symbol_of(id)
            .and_then(|symbol| self.node_indices.get(&symbol).copied())
            .map(NodeIndex)
    }

    pub fn leaf(&self, key: &str) -> Option<&LeafValues> {
        key.parse()
            .ok()
            .and_then(|parsed: LeafKey| self.leaves.get(&parsed).map(|leaf| &leaf.values))
    }

    pub fn embedding(&self, index: usize) -> Option<&RowEmbedding> {
        self.embedding_state.get(index)
    }
}

pub enum AlgebraLeafCatalog {
    V3,
}

impl AlgebraLeafCatalog {
    pub fn v3_specs() -> Vec<LeafSpec> {
        vec![
            spec(
                "live",
                LeafRole::Index,
                LeafDtype::Bool,
                LeafSource::Live,
                None,
            ),
            kind_spec("kind/memory", "Memory"),
            kind_spec("kind/claim", "Claim"),
            kind_spec("kind/commitment", "Commitment"),
            kind_spec("kind/user_model", "UserModel"),
            kind_spec("kind/agent_model_activation", "AgentModelActivation"),
            valuation("activation/score", LeafDtype::F32, "kind/memory"),
            valuation("retrieval/recency", LeafDtype::F32, "kind/memory"),
            valuation("belief/confidence", LeafDtype::F32, "kind/claim"),
            valuation_code("belief/status_code", "kind/claim"),
            valuation("commitment/priority", LeafDtype::F32, "kind/commitment"),
            valuation_code("commitment/status_code", "kind/commitment"),
            valuation(
                "identity/preference_score",
                LeafDtype::F32,
                "kind/user_model",
            ),
            valuation("identity/recency", LeafDtype::F32, "kind/user_model"),
            valuation(
                "activation/source_reliability",
                LeafDtype::F32,
                "kind/agent_model_activation",
            ),
            valuation(
                "activation/traffic",
                LeafDtype::F32,
                "kind/agent_model_activation",
            ),
            spec(
                "multivector/count",
                LeafRole::Index,
                LeafDtype::U32,
                LeafSource::MultivectorCount,
                Some("live"),
            ),
        ]
    }
}

fn spec(
    key: &str,
    role: LeafRole,
    dtype: LeafDtype,
    source: LeafSource,
    governing: Option<&str>,
) -> LeafSpec {
    LeafSpec {
        key: LeafKey::from_str(key).expect("catalog keys are valid"),
        role,
        dtype,
        source,
        governing_mask: governing.map(|mask| LeafKey::from_str(mask).expect("mask keys are valid")),
    }
}

fn kind_spec(key: &str, label: &str) -> LeafSpec {
    spec(
        key,
        LeafRole::Index,
        LeafDtype::Bool,
        LeafSource::KindLabel {
            label: label.to_string(),
        },
        Some("live"),
    )
}

fn valuation(key: &str, dtype: LeafDtype, mask: &str) -> LeafSpec {
    spec(
        key,
        LeafRole::Signal,
        dtype,
        LeafSource::Valuation,
        Some(mask),
    )
}

fn valuation_code(key: &str, mask: &str) -> LeafSpec {
    spec(
        key,
        LeafRole::Code,
        LeafDtype::U32,
        LeafSource::Valuation,
        Some(mask),
    )
}

#[cfg(feature = "graphblas")]
struct ResidentRegion {
    matrix: rustyred_thg_graphblas::Matrix,
    through: LogPosition,
}

#[cfg(feature = "graphblas")]
#[derive(Default)]
struct NativeRegions {
    resident_regions: BTreeMap<String, ResidentRegion>,
}

pub struct AlgebraFold {
    deployment_id: Option<DeploymentId>,
    log_id: theorem_oplog::LogId,
    index: AlgebraIndex,
    leaves: BTreeMap<LeafKey, LeafVector>,
    tokens: TokenRegion,
    branches: BTreeMap<String, BranchArrays>,
    #[cfg(feature = "graphblas")]
    native: NativeRegions,
    applied_through: LogPosition,
    stats: AlgebraStats,
}

impl Default for AlgebraFold {
    fn default() -> Self {
        let mut leaves = BTreeMap::new();
        for spec in AlgebraLeafCatalog::v3_specs() {
            leaves.insert(
                spec.key.clone(),
                LeafVector {
                    values: empty_values(spec.dtype),
                    spec,
                },
            );
        }
        Self {
            deployment_id: None,
            log_id: theorem_oplog::LogId::nil(),
            index: AlgebraIndex::default(),
            leaves,
            tokens: TokenRegion {
                epoch: None,
                codes: BTreeMap::new(),
                state: Vec::new(),
                residency: None,
            },
            branches: BTreeMap::new(),
            #[cfg(feature = "graphblas")]
            native: NativeRegions::default(),
            applied_through: LogPosition::ZERO,
            stats: AlgebraStats::default(),
        }
    }
}

impl std::fmt::Debug for AlgebraFold {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AlgebraFold")
            .field("node_count", &self.index.nodes_by_index.len())
            .field("applied_through", &self.applied_through)
            .field("token_index_builds", &self.stats.token_index_builds)
            .finish()
    }
}

fn empty_values(dtype: LeafDtype) -> LeafValues {
    match dtype {
        LeafDtype::F32 => LeafValues::F32(Vec::new()),
        LeafDtype::U32 => LeafValues::U32(Vec::new()),
        LeafDtype::U64 => LeafValues::U64(Vec::new()),
        LeafDtype::Bool => LeafValues::Bool(Vec::new()),
    }
}

impl AlgebraFold {
    pub fn deployment_id(&self) -> Option<&DeploymentId> {
        self.deployment_id.as_ref()
    }

    pub fn set_log_id(&mut self, log_id: theorem_oplog::LogId) {
        self.log_id = log_id;
    }

    pub fn log_id(&self) -> theorem_oplog::LogId {
        self.log_id
    }

    pub fn stats(&self) -> AlgebraStats {
        self.stats
    }

    pub fn epoch(&self) -> Option<&CalibrationEpoch> {
        self.tokens.epoch.as_ref()
    }

    pub fn pending_embedding_count(&self) -> usize {
        self.tokens
            .state
            .iter()
            .filter(|state| matches!(state, RowEmbedding::Pending))
            .count()
    }

    pub fn kind_mask(&self, kind: MemTensorKind) -> &[bool] {
        match self.leaves.get(&kind.leaf_key()).map(|leaf| &leaf.values) {
            Some(LeafValues::Bool(values)) => values,
            _ => &[],
        }
    }

    pub fn evict_tokens(&mut self) -> bool {
        self.tokens.residency.take().is_some()
    }

    pub fn materialize_tokens(&mut self) -> Result<(), FoldError> {
        let epoch = self.tokens.epoch.as_ref().ok_or_else(|| {
            FoldError::Codec("Algebra token materialization requires an active epoch".to_string())
        })?;
        if epoch.turbovec_version != TURBOVEC_ENCODE_VERSION {
            return Err(FoldError::Codec(format!(
                "token epoch names turbovec {} rather than {TURBOVEC_ENCODE_VERSION}",
                epoch.turbovec_version
            )));
        }
        let dim = epoch.dim;
        let bit_width = epoch.bit_width;
        let bytes_per = packed_bytes_per_vector(dim, bit_width)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        let node_count = self.index.nodes_by_index.len();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut slot_node = Vec::new();
        let mut offsets = vec![0_u64; node_count.saturating_add(1)];
        let live = bool_leaf(&self.leaves, "live");
        for index in 0..node_count {
            offsets[index] = slot_node.len() as u64;
            if let Some(codes) = self.tokens.codes.get(&(index as u64)) {
                packed.extend_from_slice(&codes.packed_codes);
                scales.extend_from_slice(&codes.scales);
                slot_node.extend(std::iter::repeat(index as u64).take(codes.vector_count as usize));
            }
        }
        offsets[node_count] = slot_node.len() as u64;
        let n_vectors = slot_node.len();
        let index = TurboQuantIndex::from_parts(
            Some(dim),
            usize::from(bit_width),
            n_vectors,
            packed,
            scales,
            epoch.tqplus_shift.clone(),
            epoch.tqplus_scale.clone(),
        )
        .map_err(|error| FoldError::Codec(error.to_string()))?;
        let mut kind_masks = BTreeMap::new();
        for kind in MemTensorKind::all() {
            let node_mask = self.kind_mask(kind);
            let mut slot_mask = vec![false; n_vectors];
            for (slot, node) in slot_node.iter().copied().enumerate() {
                let node = node as usize;
                let live_bit = live.get(node).copied().unwrap_or(false);
                let kind_bit = node_mask.get(node).copied().unwrap_or(false);
                let set = matches!(self.tokens.state.get(node), Some(RowEmbedding::Set { .. }));
                slot_mask[slot] = live_bit && kind_bit && set;
            }
            kind_masks.insert(kind.leaf_key(), slot_mask);
        }
        self.tokens.residency = Some(TokenResidency {
            index,
            slot_node,
            offsets,
            kind_masks,
        });
        self.stats.token_index_builds = self.stats.token_index_builds.saturating_add(1);
        let _ = bytes_per;
        Ok(())
    }

    pub fn project(&self) -> Result<AlgebraProjection<'_>, FoldError> {
        let epoch = self.tokens.epoch.as_ref().ok_or_else(|| {
            FoldError::Codec("Algebra projection requires an active calibration epoch".to_string())
        })?;
        let tokens = self
            .tokens
            .residency
            .as_ref()
            .ok_or_else(|| FoldError::Codec("token_index_not_resident".to_string()))?;
        let mut branches = BTreeMap::new();
        let mut edge_types = self.branches.keys().cloned().collect::<Vec<_>>();
        edge_types.sort();
        for (edge_type, arrays) in &self.branches {
            branches.insert(
                edge_type.clone(),
                CsrView {
                    counts: &arrays.counts,
                    targets: &arrays.targets,
                    weights: &arrays.weights,
                    edge_ids: &arrays.edge_ids,
                },
            );
        }
        Ok(AlgebraProjection {
            log_id: self.log_id,
            through: self.applied_through,
            node_count: self.index.nodes_by_index.len() as u64,
            nodes_by_index: &self.index.nodes_by_index,
            node_indices: &self.index.node_indices,
            leaves: &self.leaves,
            epoch,
            embedding_state: &self.tokens.state,
            tokens,
            edge_types,
            branches,
        })
    }

    pub fn validate_batch(&self, batch: &GraphIntentBatch) -> Result<(), FoldConsumeError> {
        let mut staged = self.logical_duplicate();
        staged.apply_changes(batch.changes.clone())
    }

    pub fn evict_region(&mut self, edge_type: &str) -> bool {
        #[cfg(feature = "graphblas")]
        {
            return self.native.resident_regions.remove(edge_type).is_some();
        }
        #[cfg(not(feature = "graphblas"))]
        {
            let _ = edge_type;
            false
        }
    }

    pub fn evict_all(&mut self) {
        #[cfg(feature = "graphblas")]
        {
            self.native.resident_regions.clear();
        }
    }

    pub fn residency(&self) -> Result<AlgebraResidency, FoldError> {
        #[cfg(feature = "graphblas")]
        {
            return native_residency(&self.native);
        }
        #[cfg(not(feature = "graphblas"))]
        Ok(AlgebraResidency::default())
    }

    pub fn materialize_region(&mut self, edge_type: &str) -> Result<(), FoldError> {
        #[cfg(feature = "graphblas")]
        {
            return materialize_native_region(self, edge_type);
        }
        #[cfg(not(feature = "graphblas"))]
        {
            let _ = edge_type;
            Err(FoldError::Codec(
                "native GraphBLAS materialization is not linked".to_string(),
            ))
        }
    }

    pub fn activation_spread(
        &mut self,
        edge_type: &str,
        seed: &[f32],
        steps: usize,
    ) -> Result<Vec<f32>, FoldError> {
        #[cfg(feature = "graphblas")]
        {
            return activation_spread_native(self, edge_type, seed, steps);
        }
        #[cfg(not(feature = "graphblas"))]
        {
            let _ = (edge_type, seed, steps);
            Err(FoldError::Codec(
                "native GraphBLAS activation_spread is not linked".to_string(),
            ))
        }
    }

    fn logical_duplicate(&self) -> Self {
        Self {
            deployment_id: self.deployment_id.clone(),
            log_id: self.log_id,
            index: self.index.clone(),
            leaves: self.leaves.clone(),
            tokens: TokenRegion {
                epoch: self.tokens.epoch.clone(),
                codes: self.tokens.codes.clone(),
                state: self.tokens.state.clone(),
                residency: None,
            },
            branches: self.branches.clone(),
            #[cfg(feature = "graphblas")]
            native: NativeRegions::default(),
            applied_through: self.applied_through,
            stats: self.stats,
        }
    }

    fn logical_eq(&self, other: &Self) -> bool {
        self.deployment_id == other.deployment_id
            && self.index.nodes_by_index == other.index.nodes_by_index
            && self.index.live_nodes == other.index.live_nodes
            && self.index.edges_by_id == other.index.edges_by_id
            && self.applied_through == other.applied_through
    }

    fn apply_batch(
        &mut self,
        entry: &LogRecord,
        batch: GraphIntentBatch,
    ) -> Result<(), FoldConsumeError> {
        match &self.deployment_id {
            Some(expected) if expected != &entry.deployment_id => {
                return Err(FoldConsumeError::fatal(format!(
                    "Algebra fold is bound to deployment {expected}, not {}",
                    entry.deployment_id
                )));
            }
            None => self.deployment_id = Some(entry.deployment_id.clone()),
            Some(_) => {}
        }
        self.apply_changes(batch.changes)?;
        self.applied_through = entry.position;
        self.stats.interval_batches = self.stats.interval_batches.saturating_add(1);
        Ok(())
    }

    fn apply_changes(&mut self, changes: Vec<GraphSemanticChange>) -> Result<(), FoldConsumeError> {
        for change in normalize_changes(changes) {
            match change {
                GraphSemanticChange::NodeObserved { record } => self.observe_node(record)?,
                GraphSemanticChange::RelationshipObserved { record } => {
                    validate_edge(&record)?;
                    if record.tombstone {
                        self.remove_edge(&record.id)?;
                    } else {
                        self.replace_edge(record)?;
                    }
                }
                GraphSemanticChange::NodeRetracted { id } => {
                    require_consume_identifier(&id, "node.id")?;
                    self.remove_node(&id)?;
                }
                GraphSemanticChange::RelationshipRetracted { id } => {
                    require_consume_identifier(&id, "edge.id")?;
                    self.remove_edge(&id)?;
                }
            }
        }
        self.rebuild_branches()?;
        Ok(())
    }

    fn observe_node(&mut self, record: NodeRecord) -> Result<(), FoldConsumeError> {
        require_consume_identifier(&record.id, "node.id")?;
        if !record.properties.is_object() {
            return Err(schema_refusal("node properties must be an object"));
        }
        if record.tombstone {
            return self.remove_node(&record.id);
        }
        let is_epoch = record
            .labels
            .iter()
            .any(|label| label == CALIBRATION_EPOCH_LABEL);
        if is_epoch {
            self.observe_epoch(&record)?;
        } else if self
            .tokens
            .epoch
            .as_ref()
            .is_some_and(|epoch| epoch.node_id == record.id)
        {
            self.demote_sets_to_pending();
            self.tokens.epoch = None;
        }
        let index = self.ensure_node(&record.id)? as usize;
        self.index.labels_by_index[index] = record.labels.clone();
        self.set_bool_leaf("live", index, true);
        for kind in MemTensorKind::all() {
            let present = record
                .labels
                .iter()
                .any(|label| MemTensorKind::from_label(label) == Some(kind));
            self.set_bool_leaf(kind.leaf_key().as_str(), index, present);
        }
        self.apply_valuation(index, &record)?;
        self.apply_embedding(index, &record)?;
        Ok(())
    }

    fn observe_epoch(&mut self, record: &NodeRecord) -> Result<(), FoldConsumeError> {
        let epoch = CalibrationEpoch::from_node(record)?;
        validate_epoch(&epoch)?;
        if epoch.active {
            if let Some(previous) = self.tokens.epoch.as_ref() {
                if previous.id != epoch.id {
                    self.demote_sets_to_pending();
                }
            }
            self.tokens.epoch = Some(epoch);
        } else if self
            .tokens
            .epoch
            .as_ref()
            .is_some_and(|current| current.id == epoch.id)
        {
            self.demote_sets_to_pending();
            self.tokens.epoch = None;
        }
        Ok(())
    }

    fn demote_sets_to_pending(&mut self) {
        for state in &mut self.tokens.state {
            if matches!(state, RowEmbedding::Set { .. }) {
                *state = RowEmbedding::Pending;
            }
        }
        self.tokens.codes.clear();
        self.tokens.residency = None;
        let count_key = LeafKey::from_str("multivector/count").expect("count key");
        if let Some(LeafVector {
            values: LeafValues::U32(values),
            ..
        }) = self.leaves.get_mut(&count_key)
        {
            for value in values.iter_mut() {
                *value = 0;
            }
        }
    }

    fn apply_valuation(
        &mut self,
        index: usize,
        record: &NodeRecord,
    ) -> Result<(), FoldConsumeError> {
        let valuation = match record.properties.get(VALUATION_PROPERTY) {
            Some(value) => Some(
                value
                    .as_object()
                    .ok_or_else(|| schema_refusal("_valuation must be an object"))?,
            ),
            None => None,
        };
        let mut f32_updates = Vec::new();
        let mut u32_updates = Vec::new();
        if let Some(object) = valuation {
            for (key, value) in object {
                let leaf_key: LeafKey = key
                    .parse()
                    .map_err(|error| schema_refusal(format!("invalid valuation key: {error}")))?;
                let Some(leaf) = self.leaves.get(&leaf_key) else {
                    continue;
                };
                if !matches!(leaf.spec.source, LeafSource::Valuation) {
                    continue;
                }
                if !self.mask_allows(leaf.spec.governing_mask.as_ref(), index) {
                    return Err(schema_refusal(format!(
                        "ungoverned leaf {} on node {}",
                        leaf_key, record.id
                    )));
                }
                match (&leaf.spec.dtype, leaf.spec.role) {
                    (LeafDtype::F32, _) => {
                        let number = value
                            .as_f64()
                            .ok_or_else(|| schema_refusal(format!("{leaf_key} must be f32")))?;
                        f32_updates.push((leaf_key, checked_f32(number, key)?));
                    }
                    (LeafDtype::U32, role) => {
                        let raw = value
                            .as_u64()
                            .ok_or_else(|| schema_refusal(format!("{leaf_key} must be u32")))?;
                        let number = u32::try_from(raw)
                            .map_err(|_| schema_refusal(format!("{leaf_key} exceeds u32")))?;
                        if role == LeafRole::Code {
                            let max = if leaf_key.as_str() == "belief/status_code" {
                                4
                            } else {
                                2
                            };
                            if number > max {
                                return Err(schema_refusal(format!(
                                    "out-of-range Code value {number} for {leaf_key}"
                                )));
                            }
                        }
                        u32_updates.push((leaf_key, number));
                    }
                    _ => {
                        return Err(schema_refusal(format!(
                            "unsupported valuation dtype for {leaf_key}"
                        )));
                    }
                }
            }
        }
        self.clear_valuation(index);
        for (key, value) in f32_updates {
            self.set_f32_leaf(key.as_str(), index, value);
        }
        for (key, value) in u32_updates {
            self.set_u32_leaf(key.as_str(), index, value);
        }
        Ok(())
    }

    fn apply_embedding(
        &mut self,
        index: usize,
        record: &NodeRecord,
    ) -> Result<(), FoldConsumeError> {
        let has_embedding = record.properties.get(EMBEDDING_PROPERTY).is_some();
        let has_excluded = record.properties.get(EMBEDDING_EXCLUDED_PROPERTY).is_some();
        if has_embedding && has_excluded {
            return Err(schema_refusal(
                "node cannot carry both _embedding and _embedding_excluded",
            ));
        }
        if has_excluded {
            let reason = record
                .properties
                .get(EMBEDDING_EXCLUDED_PROPERTY)
                .and_then(|value| value.get("reason"))
                .cloned()
                .ok_or_else(|| schema_refusal("_embedding_excluded requires reason"))?;
            let reason: EmbeddingExclusion = serde_json::from_value(reason)
                .map_err(|error| schema_refusal(error.to_string()))?;
            self.tokens.state[index] = RowEmbedding::Excluded { reason };
            self.tokens.codes.remove(&(index as u64));
            self.set_u32_leaf("multivector/count", index, 0);
            self.tokens.residency = None;
            return Ok(());
        }
        if !has_embedding {
            self.clear_embedding(index);
            return Ok(());
        }
        let payload = record
            .properties
            .get(EMBEDDING_PROPERTY)
            .cloned()
            .ok_or_else(|| schema_refusal("_embedding missing"))?;
        let decoded = decode_embedding(&payload)?;
        let epoch = self.tokens.epoch.as_ref().ok_or_else(|| {
            schema_refusal("_embedding observed before any active calibration epoch")
        })?;
        if decoded.epoch != epoch.id {
            return Err(schema_refusal("_embedding names a stale calibration epoch"));
        }
        if decoded.dim != epoch.dim
            || decoded.bit_width != epoch.bit_width
            || decoded.model_id != epoch.model_id
            || decoded.model_version != epoch.model_version
        {
            return Err(schema_refusal(
                "_embedding does not match the active calibration epoch",
            ));
        }
        if decoded.content_id != record.id {
            return Err(schema_refusal(
                "_embedding content_id must equal the node id",
            ));
        }
        let vector_count = usize::try_from(decoded.vector_count)
            .map_err(|_| schema_refusal("_embedding vector_count exceeds usize"))?;
        let expected = packed_bytes_per_vector(decoded.dim, decoded.bit_width)
            .map_err(|error| schema_refusal(error.to_string()))?
            .checked_mul(vector_count)
            .ok_or_else(|| schema_refusal("_embedding packed-code size overflow"))?;
        if decoded.packed_codes.len() != expected {
            return Err(schema_refusal("packed_codes has the wrong length"));
        }
        if decoded.scales.len() != vector_count {
            return Err(schema_refusal("scales length must equal vector_count"));
        }
        if decoded
            .scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err(schema_refusal("scales must be finite and non-negative"));
        }
        self.tokens.codes.insert(
            index as u64,
            TokenCodes {
                vector_count: decoded.vector_count,
                packed_codes: decoded.packed_codes,
                scales: decoded.scales,
            },
        );
        self.tokens.state[index] = RowEmbedding::Set {
            id: decoded.embedding_set_id,
            epoch: decoded.epoch,
            exact_object_ref: decoded.exact_object_ref,
        };
        self.set_u32_leaf("multivector/count", index, decoded.vector_count);
        self.tokens.residency = None;
        Ok(())
    }

    fn clear_valuation(&mut self, index: usize) {
        for leaf in self.leaves.values_mut() {
            if !matches!(leaf.spec.source, LeafSource::Valuation) {
                continue;
            }
            match &mut leaf.values {
                LeafValues::F32(values) => {
                    if let Some(value) = values.get_mut(index) {
                        *value = 0.0;
                    }
                }
                LeafValues::U32(values) => {
                    if let Some(value) = values.get_mut(index) {
                        *value = 0;
                    }
                }
                LeafValues::U64(values) => {
                    if let Some(value) = values.get_mut(index) {
                        *value = 0;
                    }
                }
                LeafValues::Bool(values) => {
                    if let Some(value) = values.get_mut(index) {
                        *value = false;
                    }
                }
            }
        }
    }

    fn clear_embedding(&mut self, index: usize) {
        self.tokens.state[index] = RowEmbedding::Pending;
        self.tokens.codes.remove(&(index as u64));
        self.set_u32_leaf("multivector/count", index, 0);
        self.tokens.residency = None;
    }

    /// The canonical identity string of one row.
    ///
    /// Reads the fold's identity table directly, without the calibration epoch
    /// and token residency that [`Self::project`] requires.
    #[must_use]
    pub fn node_id(&self, node: NodeIndex) -> Option<&str> {
        self.index
            .nodes_by_index
            .get(usize::try_from(node.0).ok()?)
            .map(|symbol| symbol.as_str())
    }

    /// The row index of one identity, through the fold's reverse index.
    #[must_use]
    pub fn node_index(&self, id: &str) -> Option<NodeIndex> {
        self.node_index_of(id).map(NodeIndex)
    }

    /// Resolve an already-admitted identity to its row index. A miss costs a
    /// hash probe and mints nothing.
    fn node_index_of(&self, id: &str) -> Option<u64> {
        symbol_of(id).and_then(|symbol| self.row_index(symbol))
    }

    fn row_index(&self, symbol: Symbol) -> Option<u64> {
        self.index.node_indices.get(&symbol).copied()
    }

    fn is_live(&self, id: &str) -> bool {
        symbol_of(id).is_some_and(|symbol| self.index.live_nodes.contains(&symbol))
    }

    /// Both endpoints of an edge as row indices. An edge whose endpoints are
    /// not indexed is a fold-state contradiction, not a caller error, so it is
    /// fatal rather than a refusal.
    fn edge_endpoints(&self, edge: &EdgeRecord) -> Result<(u64, u64), FoldConsumeError> {
        let endpoint = |id: &str| {
            self.node_index_of(id).ok_or_else(|| {
                FoldConsumeError::fatal(format!(
                    "Algebra edge {} names endpoint {id}, which has no row index",
                    edge.id
                ))
            })
        };
        Ok((endpoint(&edge.from_id)?, endpoint(&edge.to_id)?))
    }

    /// Admit one identity and return its row index, interning on first sight.
    ///
    /// This is the only admission point into the process-global identity table
    /// (SPEC-THEOREM-MEMTENSOR-IDENTITY-INTERNER-1.0): every read path resolves
    /// through `symbol_of`, which never mints.
    fn ensure_node(&mut self, id: &str) -> Result<u64, FoldConsumeError> {
        let symbol = intern(id);
        self.index.live_nodes.insert(symbol);
        if let Some(&index) = self.index.node_indices.get(&symbol) {
            return Ok(index);
        }
        let index = self.index.nodes_by_index.len() as u64;
        self.index.nodes_by_index.push(symbol);
        self.index.labels_by_index.push(Vec::new());
        self.index.node_indices.insert(symbol, index);
        self.grow_leaves();
        self.tokens.state.push(RowEmbedding::Pending);
        Ok(index)
    }

    fn grow_leaves(&mut self) {
        for leaf in self.leaves.values_mut() {
            match &mut leaf.values {
                LeafValues::F32(values) => values.push(0.0),
                LeafValues::U32(values) => values.push(0),
                LeafValues::U64(values) => values.push(0),
                LeafValues::Bool(values) => values.push(false),
            }
        }
        for arrays in self.branches.values_mut() {
            arrays.counts.push(0);
        }
    }

    fn remove_node(&mut self, id: &str) -> Result<(), FoldConsumeError> {
        if self
            .tokens
            .epoch
            .as_ref()
            .is_some_and(|epoch| epoch.node_id == id)
        {
            self.demote_sets_to_pending();
            self.tokens.epoch = None;
        }
        let incident = self
            .index
            .edges_by_id
            .values()
            .filter(|edge| edge.from_id == id || edge.to_id == id)
            .map(|edge| edge.id.clone())
            .collect::<Vec<_>>();
        for edge_id in incident {
            self.remove_edge(&edge_id)?;
        }
        let symbol = symbol_of(id);
        if let Some(symbol) = symbol {
            self.index.live_nodes.remove(&symbol);
        }
        if let Some(index) = symbol.and_then(|symbol| self.row_index(symbol)) {
            let index = index as usize;
            self.set_bool_leaf("live", index, false);
            for kind in MemTensorKind::all() {
                self.set_bool_leaf(kind.leaf_key().as_str(), index, false);
            }
            self.zero_leaves(index);
            self.tokens.state[index] = RowEmbedding::Pending;
            self.tokens.codes.remove(&(index as u64));
            self.tokens.residency = None;
        }
        Ok(())
    }

    fn zero_leaves(&mut self, index: usize) {
        for leaf in self.leaves.values_mut() {
            match &mut leaf.values {
                LeafValues::F32(values) => {
                    if let Some(slot) = values.get_mut(index) {
                        *slot = 0.0;
                    }
                }
                LeafValues::U32(values) => {
                    if let Some(slot) = values.get_mut(index) {
                        *slot = 0;
                    }
                }
                LeafValues::U64(values) => {
                    if let Some(slot) = values.get_mut(index) {
                        *slot = 0;
                    }
                }
                LeafValues::Bool(values) => {
                    if let Some(slot) = values.get_mut(index) {
                        *slot = false;
                    }
                }
            }
        }
    }

    fn replace_edge(&mut self, edge: EdgeRecord) -> Result<(), FoldConsumeError> {
        if !self.is_live(&edge.from_id) || !self.is_live(&edge.to_id) {
            return Err(schema_refusal(format!(
                "edge {} endpoints must both be live nodes",
                edge.id
            )));
        }
        self.ensure_branch_leaf(&edge.edge_type)?;
        if let Some(previous) = self.index.edges_by_id.remove(&edge.id) {
            self.remove_coordinate(&previous)?;
        }
        self.add_coordinate(&edge)?;
        self.index.edges_by_id.insert(edge.id.clone(), edge);
        Ok(())
    }

    fn remove_edge(&mut self, id: &str) -> Result<(), FoldConsumeError> {
        if let Some(edge) = self.index.edges_by_id.remove(id) {
            self.remove_coordinate(&edge)?;
        }
        Ok(())
    }

    fn add_coordinate(&mut self, edge: &EdgeRecord) -> Result<(), FoldConsumeError> {
        let (from, to) = self.edge_endpoints(edge)?;
        let key = (edge.edge_type.clone(), from, to);
        let count = self.index.coordinate_counts.entry(key).or_default();
        *count = count.saturating_add(1);
        Ok(())
    }

    fn remove_coordinate(&mut self, edge: &EdgeRecord) -> Result<(), FoldConsumeError> {
        let (from, to) = self.edge_endpoints(edge)?;
        let key = (edge.edge_type.clone(), from, to);
        let Some(count) = self.index.coordinate_counts.get_mut(&key) else {
            return Err(FoldConsumeError::fatal(format!(
                "Algebra coordinate count missing for edge {}",
                edge.id
            )));
        };
        *count -= 1;
        if *count == 0 {
            self.index.coordinate_counts.remove(&key);
        }
        Ok(())
    }

    fn ensure_branch_leaf(&mut self, edge_type: &str) -> Result<LeafKey, FoldConsumeError> {
        let key = format!("branch/{edge_type}/count");
        let key = LeafKey::from_str(&key)
            .map_err(|error| schema_refusal(format!("invalid edge type {edge_type:?}: {error}")))?;
        if self.leaves.contains_key(&key) {
            return Ok(key);
        }
        let n = self.index.nodes_by_index.len();
        self.leaves.insert(
            key.clone(),
            LeafVector {
                spec: LeafSpec {
                    key: key.clone(),
                    role: LeafRole::Index,
                    dtype: LeafDtype::U32,
                    source: LeafSource::Live,
                    governing_mask: Some(LeafKey::from_str("live").expect("live")),
                },
                values: LeafValues::U32(vec![0; n]),
            },
        );
        Ok(key)
    }

    fn rebuild_branches(&mut self) -> Result<(), FoldConsumeError> {
        let node_count = self.index.nodes_by_index.len();
        let mut grouped: BTreeMap<String, Vec<EdgeRecord>> = BTreeMap::new();
        for edge in self.index.edges_by_id.values() {
            grouped
                .entry(edge.edge_type.clone())
                .or_default()
                .push(edge.clone());
        }
        self.branches.clear();
        for (key, leaf) in &mut self.leaves {
            if key.as_str().starts_with("branch/") && key.as_str().ends_with("/count") {
                if let LeafValues::U32(values) = &mut leaf.values {
                    values.fill(0);
                }
            }
        }
        for (edge_type, mut edges) in grouped {
            let count_key = self.ensure_branch_leaf(&edge_type)?;
            edges.sort_by(|left, right| {
                self.node_index_of(&left.from_id)
                    .cmp(&self.node_index_of(&right.from_id))
                    .then_with(|| left.id.cmp(&right.id))
            });
            let mut counts = vec![0_u32; node_count];
            let mut targets = Vec::new();
            let mut weights = Vec::new();
            let mut edge_ids = Vec::new();
            for edge in edges {
                let (from, to) = self.edge_endpoints(&edge)?;
                let from = from as usize;
                counts[from] = counts[from]
                    .checked_add(1)
                    .ok_or_else(|| schema_refusal("branch count exceeds u32"))?;
                targets.push(to);
                weights.push(checked_f32(
                    edge.confidence.unwrap_or(1.0),
                    "edge confidence",
                )?);
                edge_ids.push(edge.id.clone());
            }
            if let Some(LeafVector {
                values: LeafValues::U32(values),
                ..
            }) = self.leaves.get_mut(&count_key)
            {
                *values = counts.clone();
            }
            self.branches.insert(
                edge_type,
                BranchArrays {
                    counts,
                    targets,
                    weights,
                    edge_ids,
                },
            );
        }
        Ok(())
    }

    fn mask_allows(&self, mask: Option<&LeafKey>, index: usize) -> bool {
        let key = mask
            .cloned()
            .unwrap_or_else(|| LeafKey::from_str("live").expect("live"));
        match self.leaves.get(&key).map(|leaf| &leaf.values) {
            Some(LeafValues::Bool(values)) => values.get(index).copied().unwrap_or(false),
            _ => false,
        }
    }

    fn set_bool_leaf(&mut self, key: &str, index: usize, value: bool) {
        if let Some(LeafVector {
            values: LeafValues::Bool(values),
            ..
        }) = self.leaves.get_mut(&LeafKey::from_str(key).expect("key"))
        {
            if let Some(slot) = values.get_mut(index) {
                *slot = value;
            }
        }
    }

    fn set_f32_leaf(&mut self, key: &str, index: usize, value: f32) {
        if let Some(LeafVector {
            values: LeafValues::F32(values),
            ..
        }) = self.leaves.get_mut(&LeafKey::from_str(key).expect("key"))
        {
            if let Some(slot) = values.get_mut(index) {
                *slot = value;
            }
        }
    }

    fn set_u32_leaf(&mut self, key: &str, index: usize, value: u32) {
        if let Some(LeafVector {
            values: LeafValues::U32(values),
            ..
        }) = self.leaves.get_mut(&LeafKey::from_str(key).expect("key"))
        {
            if let Some(slot) = values.get_mut(index) {
                *slot = value;
            }
        }
    }

    fn rebuild_indexes(&mut self) -> Result<(), FoldError> {
        self.index.node_indices.clear();
        for (index, symbol) in self.index.nodes_by_index.iter().copied().enumerate() {
            if self
                .index
                .node_indices
                .insert(symbol, index as u64)
                .is_some()
            {
                return Err(FoldError::Codec(format!(
                    "Algebra checkpoint repeats node index {}",
                    symbol.as_str()
                )));
            }
        }
        if self
            .index
            .live_nodes
            .iter()
            .any(|symbol| !self.index.node_indices.contains_key(symbol))
        {
            return Err(FoldError::Codec(
                "Algebra checkpoint has a live node without an index".to_string(),
            ));
        }
        self.index.coordinate_counts.clear();
        for edge in self.index.edges_by_id.values() {
            let Some(from) = self.node_index_of(&edge.from_id) else {
                return Err(FoldError::Codec(format!(
                    "Algebra checkpoint edge {} has no from-node index",
                    edge.id
                )));
            };
            let Some(to) = self.node_index_of(&edge.to_id) else {
                return Err(FoldError::Codec(format!(
                    "Algebra checkpoint edge {} has no to-node index",
                    edge.id
                )));
            };
            *self
                .index
                .coordinate_counts
                .entry((edge.edge_type.clone(), from, to))
                .or_default() += 1;
        }
        self.rebuild_branches()
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        self.tokens.residency = None;
        #[cfg(feature = "graphblas")]
        {
            self.native.resident_regions.clear();
        }
        Ok(())
    }

    fn install_validated(&mut self, staged: Self) -> Result<(), FoldConsumeError> {
        #[cfg(feature = "graphblas")]
        {
            install_native_delta(self, &staged)?;
        }
        self.deployment_id = staged.deployment_id;
        self.index = staged.index;
        self.leaves = staged.leaves;
        self.tokens = staged.tokens;
        self.branches = staged.branches;
        self.applied_through = staged.applied_through;
        self.stats.interval_batches = staged.stats.interval_batches;
        Ok(())
    }
}

impl Fold for AlgebraFold {
    fn id(&self) -> FoldId {
        FOLD_ID
    }

    fn version(&self) -> FoldVersion {
        FoldVersion::new(ALGEBRA_FOLD_VERSION).expect("Algebra fold version is non-zero")
    }

    fn descriptor(&self) -> FoldDescriptor {
        descriptor()
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        let batch = decode_graph_batch(entry)?;
        let mut staged = self.logical_duplicate();
        staged.apply_batch(entry, batch)?;
        self.install_validated(staged)
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        if position != self.applied_through {
            return Err(FoldError::Codec(format!(
                "Algebra direct query is resident at {}; use upquery for pinned position {position}",
                self.applied_through
            )));
        }
        let query = serde_json::from_value::<AlgebraQuery>(query.clone())
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        serde_json::to_value(self.answer(&query, position)?)
            .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError> {
        let query = serde_json::from_value::<AlgebraQuery>(key.clone())
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        let AlgebraQuery::Reachable { edge_type, .. } = &query;
        let mut rebuilt = Self::default();
        for record in pinned_records {
            match rebuilt.apply(record) {
                Ok(()) | Err(FoldConsumeError::Refused(_)) => {}
                Err(FoldConsumeError::Fatal(reason)) => {
                    return Err(FoldError::FatalConsume(reason));
                }
            }
        }
        rebuilt.materialize_region(edge_type)?;
        let answer = rebuilt.answer(&query, position)?;
        if rebuilt.logical_eq(self) {
            #[cfg(feature = "graphblas")]
            {
                if let Some(region) = rebuilt.native.resident_regions.remove(edge_type) {
                    self.native
                        .resident_regions
                        .insert(edge_type.clone(), region);
                    self.stats.native_full_builds = self.stats.native_full_builds.saturating_add(1);
                }
            }
        }
        serde_json::to_value(answer).map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn checkpoint(&self) -> Result<Vec<u8>, FoldError> {
        serde_json::to_vec(&AlgebraCheckpoint {
            deployment_id: self.deployment_id.clone(),
            // The checkpoint stays string-keyed and byte-identical to the
            // pre-interner shape: symbols are in-memory only, so
            // ALGEBRA_FOLD_VERSION does not move and existing checkpoints
            // restore unchanged.
            nodes_by_index: self
                .index
                .nodes_by_index
                .iter()
                .map(|symbol| symbol.as_str().to_string())
                .collect(),
            live_nodes: self
                .index
                .live_nodes
                .iter()
                .map(|symbol| symbol.as_str().to_string())
                .collect(),
            labels_by_index: self.index.labels_by_index.clone(),
            edges_by_id: self.index.edges_by_id.clone(),
            leaves: self.leaves.clone(),
            epoch: self.tokens.epoch.clone(),
            codes: self.tokens.codes.clone(),
            state: self.tokens.state.clone(),
            applied_through: self.applied_through,
            interval_batches: self.stats.interval_batches,
            token_index_builds: self.stats.token_index_builds,
        })
        .map_err(|error| FoldError::Codec(error.to_string()))
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        let checkpoint = serde_json::from_slice::<AlgebraCheckpoint>(checkpoint)
            .map_err(|error| FoldError::Codec(error.to_string()))?;
        *self = Self {
            deployment_id: checkpoint.deployment_id,
            log_id: self.log_id,
            index: AlgebraIndex {
                nodes_by_index: checkpoint
                    .nodes_by_index
                    .iter()
                    .map(|id| intern(id))
                    .collect(),
                node_indices: HashMap::new(),
                live_nodes: checkpoint.live_nodes.iter().map(|id| intern(id)).collect(),
                labels_by_index: checkpoint.labels_by_index,
                edges_by_id: checkpoint.edges_by_id,
                coordinate_counts: BTreeMap::new(),
            },
            leaves: checkpoint.leaves,
            tokens: TokenRegion {
                epoch: checkpoint.epoch,
                codes: checkpoint.codes,
                state: checkpoint.state,
                residency: None,
            },
            branches: BTreeMap::new(),
            #[cfg(feature = "graphblas")]
            native: NativeRegions::default(),
            applied_through: checkpoint.applied_through,
            stats: AlgebraStats {
                interval_batches: checkpoint.interval_batches,
                token_index_builds: checkpoint.token_index_builds,
                ..AlgebraStats::default()
            },
        };
        self.rebuild_indexes()
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        *self = Self::default();
        Ok(())
    }

    fn advance_without_absorb(&mut self, entry: &LogRecord) -> Result<(), FoldError> {
        self.applied_through = entry.position;
        #[cfg(feature = "graphblas")]
        {
            for region in self.native.resident_regions.values_mut() {
                region.through = entry.position;
            }
        }
        Ok(())
    }
}

impl AlgebraFold {
    fn answer(
        &self,
        query: &AlgebraQuery,
        position: LogPosition,
    ) -> Result<AlgebraReachability, FoldError> {
        #[cfg(feature = "graphblas")]
        {
            return answer_native(self, query, position);
        }
        #[cfg(not(feature = "graphblas"))]
        {
            let _ = (query, position);
            Err(FoldError::Codec(
                "native GraphBLAS reachability is not linked".to_string(),
            ))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AlgebraCheckpoint {
    deployment_id: Option<DeploymentId>,
    nodes_by_index: Vec<String>,
    live_nodes: BTreeSet<String>,
    #[serde(default)]
    labels_by_index: Vec<Vec<String>>,
    edges_by_id: BTreeMap<String, EdgeRecord>,
    #[serde(default)]
    leaves: BTreeMap<LeafKey, LeafVector>,
    #[serde(default)]
    epoch: Option<CalibrationEpoch>,
    #[serde(default)]
    codes: BTreeMap<u64, TokenCodes>,
    #[serde(default)]
    state: Vec<RowEmbedding>,
    applied_through: LogPosition,
    interval_batches: u64,
    #[serde(default)]
    token_index_builds: u64,
}

struct DecodedEmbedding {
    embedding_set_id: String,
    content_id: String,
    model_id: String,
    model_version: String,
    epoch: String,
    dim: usize,
    bit_width: u8,
    vector_count: u32,
    packed_codes: Vec<u8>,
    scales: Vec<f32>,
    exact_object_ref: String,
}

fn decode_embedding(value: &Value) -> Result<DecodedEmbedding, FoldConsumeError> {
    let object = value
        .as_object()
        .ok_or_else(|| schema_refusal("_embedding must be an object"))?;
    let require = |key: &str| -> Result<String, FoldConsumeError> {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| schema_refusal(format!("_embedding missing {key}")))
    };
    let packed = object
        .get("packed_codes")
        .and_then(Value::as_str)
        .ok_or_else(|| schema_refusal("_embedding missing packed_codes"))?;
    let packed_codes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, packed)
        .map_err(|error| schema_refusal(format!("packed_codes base64: {error}")))?;
    let scales = object
        .get("scales")
        .and_then(Value::as_array)
        .ok_or_else(|| schema_refusal("_embedding missing scales"))?
        .iter()
        .map(|item| {
            let number = item
                .as_f64()
                .ok_or_else(|| schema_refusal("scale must be f32"))?;
            checked_f32(number, "scale")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dim = usize::try_from(
        object
            .get("dim")
            .and_then(Value::as_u64)
            .ok_or_else(|| schema_refusal("_embedding missing dim"))?,
    )
    .map_err(|_| schema_refusal("_embedding dim exceeds usize"))?;
    let bit_width = u8::try_from(
        object
            .get("bit_width")
            .and_then(Value::as_u64)
            .ok_or_else(|| schema_refusal("_embedding missing bit_width"))?,
    )
    .map_err(|_| schema_refusal("_embedding bit_width exceeds u8"))?;
    let vector_count = u32::try_from(
        object
            .get("vector_count")
            .and_then(Value::as_u64)
            .ok_or_else(|| schema_refusal("_embedding missing vector_count"))?,
    )
    .map_err(|_| schema_refusal("_embedding vector_count exceeds u32"))?;
    Ok(DecodedEmbedding {
        embedding_set_id: require("embedding_set_id")?,
        content_id: require("content_id")?,
        model_id: require("model_id")?,
        model_version: require("model_version")?,
        epoch: require("epoch")?,
        dim,
        bit_width,
        vector_count,
        packed_codes,
        scales,
        exact_object_ref: require("exact_object_ref")?,
    })
}

fn checked_f32(number: f64, field: &str) -> Result<f32, FoldConsumeError> {
    if !number.is_finite() || number < f32::MIN as f64 || number > f32::MAX as f64 {
        return Err(schema_refusal(format!(
            "{field} must be within the finite f32 range"
        )));
    }
    Ok(number as f32)
}

fn validate_epoch(epoch: &CalibrationEpoch) -> Result<(), FoldConsumeError> {
    if epoch.dim == 0 || epoch.dim % 8 != 0 {
        return Err(schema_refusal(
            "CalibrationEpoch dim must be a multiple of 8",
        ));
    }
    if !(2..=4).contains(&epoch.bit_width) {
        return Err(schema_refusal(
            "CalibrationEpoch bit_width must be in 2..=4",
        ));
    }
    if epoch.tqplus_shift.len() != epoch.dim || epoch.tqplus_scale.len() != epoch.dim {
        return Err(schema_refusal(
            "CalibrationEpoch TQ+ pair length must equal dim",
        ));
    }
    if epoch.tqplus_shift.iter().any(|shift| !shift.is_finite()) {
        return Err(schema_refusal(
            "CalibrationEpoch tqplus_shift must be finite",
        ));
    }
    if epoch
        .tqplus_scale
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(schema_refusal(
            "CalibrationEpoch tqplus_scale must be finite and positive",
        ));
    }
    if epoch.turbovec_version != TURBOVEC_ENCODE_VERSION {
        return Err(schema_refusal(format!(
            "CalibrationEpoch turbovec_version {} is not {TURBOVEC_ENCODE_VERSION}",
            epoch.turbovec_version
        )));
    }
    Ok(())
}

fn bool_leaf<'a>(leaves: &'a BTreeMap<LeafKey, LeafVector>, key: &str) -> &'a [bool] {
    match leaves.get(&LeafKey::from_str(key).expect("key")) {
        Some(LeafVector {
            values: LeafValues::Bool(values),
            ..
        }) => values,
        _ => &[],
    }
}

fn decode_graph_batch(record: &LogRecord) -> Result<GraphIntentBatch, FoldConsumeError> {
    if record.intent.kind.namespace() != GRAPH_INTENT_NAMESPACE
        || record.intent.kind.name() != GRAPH_INTENT_NAME
        || record.intent.kind.version() != GRAPH_INTENT_VERSION
    {
        return Err(FoldConsumeError::refused(
            FoldRefusalCode::UnsupportedIntent,
            format!("expected {GRAPH_INTENT_NAMESPACE}.{GRAPH_INTENT_NAME}@{GRAPH_INTENT_VERSION}"),
        ));
    }
    let batch = serde_json::from_value::<GraphIntentBatch>(record.intent.payload.clone()).map_err(
        |error| {
            FoldConsumeError::refused(
                FoldRefusalCode::InvalidPayload,
                format!("Algebra graph batch decode failed: {error}"),
            )
        },
    )?;
    if batch.changes.is_empty() {
        return Err(schema_refusal(
            "Algebra graph semantic batch must contain at least one change",
        ));
    }
    Ok(batch)
}

fn normalize_changes(changes: Vec<GraphSemanticChange>) -> Vec<GraphSemanticChange> {
    let mut edge_retractions = Vec::new();
    let mut node_retractions = Vec::new();
    let mut node_observations = Vec::new();
    let mut edge_observations = Vec::new();
    for change in changes {
        match change {
            GraphSemanticChange::RelationshipRetracted { .. } => edge_retractions.push(change),
            GraphSemanticChange::NodeRetracted { .. } => node_retractions.push(change),
            GraphSemanticChange::NodeObserved { .. } => node_observations.push(change),
            GraphSemanticChange::RelationshipObserved { .. } => edge_observations.push(change),
        }
    }
    edge_retractions
        .into_iter()
        .chain(node_retractions)
        .chain(node_observations)
        .chain(edge_observations)
        .collect()
}

fn validate_edge(edge: &EdgeRecord) -> Result<(), FoldConsumeError> {
    require_consume_identifier(&edge.id, "edge.id")?;
    require_consume_identifier(&edge.from_id, "edge.from_id")?;
    require_consume_identifier(&edge.to_id, "edge.to_id")?;
    require_consume_identifier(&edge.edge_type, "edge.type")?;
    LeafKey::from_str(&format!("branch/{}/count", edge.edge_type)).map_err(|error| {
        schema_refusal(format!("invalid edge type {:?}: {error}", edge.edge_type))
    })?;
    if let Some(confidence) = edge.confidence {
        checked_f32(confidence, "edge confidence")?;
    }
    if !edge.properties.is_object() {
        return Err(schema_refusal("edge properties must be an object"));
    }
    Ok(())
}

fn require_consume_identifier(value: &str, field: &'static str) -> Result<(), FoldConsumeError> {
    if value.is_empty() || value != value.trim() {
        Err(schema_refusal(format!(
            "{field} must be a canonical non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn schema_refusal(reason: impl Into<String>) -> FoldConsumeError {
    FoldConsumeError::refused(FoldRefusalCode::SchemaViolation, reason)
}

/// Stable Algebra fold identity, readable even when native materialization is disabled.
pub fn descriptor() -> FoldDescriptor {
    FoldDescriptor::new(
        FOLD_ID,
        FoldVersion::new(FOLD_VERSION).expect("Algebra fold version is non-zero"),
        FoldCapabilities::ordered_transitions_only(),
    )
    .with_valuation_profile(AlgebraValuationProfile::v3())
}

#[cfg(feature = "graphblas")]
fn native_error(error: impl ToString) -> FoldError {
    FoldError::Codec(format!("native GraphBLAS failure: {}", error.to_string()))
}

#[cfg(feature = "graphblas")]
fn native_consume_error(error: impl ToString) -> FoldConsumeError {
    FoldConsumeError::fatal(format!("native GraphBLAS failure: {}", error.to_string()))
}

#[cfg(feature = "graphblas")]
fn native_residency(native: &NativeRegions) -> Result<AlgebraResidency, FoldError> {
    let matrix_entries = native
        .resident_regions
        .values()
        .try_fold(0_u64, |total, region| {
            region
                .matrix
                .nvals()
                .map(|entries| total.saturating_add(entries))
                .map_err(native_error)
        })?;
    let matrix_bytes = native
        .resident_regions
        .values()
        .try_fold(0_usize, |total, region| {
            region
                .matrix
                .memory_usage()
                .map(|bytes| total.saturating_add(bytes))
                .map_err(native_error)
        })?;
    Ok(AlgebraResidency {
        regions: native.resident_regions.len(),
        matrix_entries,
        matrix_bytes,
    })
}

#[cfg(feature = "graphblas")]
fn materialize_native_region(fold: &mut AlgebraFold, edge_type: &str) -> Result<(), FoldError> {
    require_query_identifier(edge_type, "edge_type")?;
    if fold.native.resident_regions.contains_key(edge_type) {
        return Ok(());
    }
    let edges = fold
        .index
        .coordinate_counts
        .iter()
        .filter_map(|((candidate, from, to), count)| {
            (candidate == edge_type && *count != 0).then_some((*from, *to))
        })
        .collect::<Vec<_>>();
    let matrix = rustyred_thg_graphblas::Matrix::from_bool_edges(
        fold.index.nodes_by_index.len() as u64,
        &edges,
    )
    .map_err(native_error)?;
    fold.native.resident_regions.insert(
        edge_type.to_string(),
        ResidentRegion {
            matrix,
            through: fold.applied_through,
        },
    );
    fold.stats.native_full_builds = fold.stats.native_full_builds.saturating_add(1);
    Ok(())
}

#[cfg(feature = "graphblas")]
fn answer_native(
    fold: &AlgebraFold,
    query: &AlgebraQuery,
    position: LogPosition,
) -> Result<AlgebraReachability, FoldError> {
    let AlgebraQuery::Reachable { edge_type, sources } = query;
    require_query_identifier(edge_type, "edge_type")?;
    if sources.is_empty() {
        return Err(FoldError::Codec(
            "Algebra reachable query requires at least one source".to_string(),
        ));
    }
    let mut canonical_sources = sources.clone();
    canonical_sources.sort();
    canonical_sources.dedup();
    let source_indices = canonical_sources
        .iter()
        .map(|source| {
            require_query_identifier(source, "source")?;
            if !fold.is_live(source) {
                return Err(FoldError::Codec(format!(
                    "Algebra source {source} is not a live node"
                )));
            }
            fold.node_index_of(source).ok_or_else(|| {
                FoldError::Codec(format!("Algebra source {source} has no row index"))
            })
        })
        .collect::<Result<Vec<_>, FoldError>>()?;
    let region = fold.native.resident_regions.get(edge_type).ok_or_else(|| {
        FoldError::Codec(format!(
            "Algebra region {edge_type} is not resident; use FoldRuntime::upquery"
        ))
    })?;
    let reachable = rustyred_thg_graphblas::reachable_from(
        &region.matrix,
        fold.index.nodes_by_index.len() as u64,
        &source_indices,
    )
    .map_err(native_error)?
    .indices_bool()
    .map_err(native_error)?
    .into_iter()
    .filter_map(|index| fold.index.nodes_by_index.get(index as usize).copied())
    .filter(|symbol| fold.index.live_nodes.contains(symbol))
    .map(|symbol| symbol.as_str().to_string())
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect();
    Ok(AlgebraReachability {
        edge_type: edge_type.clone(),
        sources: canonical_sources,
        reachable,
        position,
    })
}

#[cfg(feature = "graphblas")]
fn activation_spread_native(
    fold: &mut AlgebraFold,
    edge_type: &str,
    seed: &[f32],
    steps: usize,
) -> Result<Vec<f32>, FoldError> {
    use rustyred_thg_graphblas::{mxv_into, Descriptor, Semiring, Vector};
    require_query_identifier(edge_type, "edge_type")?;
    fold.materialize_region(edge_type)?;
    let n = fold.index.nodes_by_index.len();
    if seed.len() != n {
        return Err(FoldError::Codec(
            "activation_spread seed length must equal node_count".to_string(),
        ));
    }
    let region =
        fold.native.resident_regions.get(edge_type).ok_or_else(|| {
            FoldError::Codec(format!("Algebra region {edge_type} is not resident"))
        })?;
    let live = Vector::from_dense_bool(bool_leaf(&fold.leaves, "live")).map_err(native_error)?;
    let mut state = Vector::from_dense_fp32(seed).map_err(native_error)?;
    let semi = Semiring::plus_times_fp32();
    // Matrices store A(from, to). Forward propagation is therefore A^T * state:
    // output[to] accumulates input[from] for each directed edge from -> to.
    let desc = Descriptor::transpose_a();
    for _ in 0..steps {
        let mut next = Vector::new_fp32(n as u64).map_err(native_error)?;
        mxv_into(&mut next, Some(&live), &semi, &region.matrix, &state, &desc)
            .map_err(native_error)?;
        state = next;
    }
    state.to_dense_fp32(n as u64, 0.0).map_err(native_error)
}

#[cfg(feature = "graphblas")]
fn install_native_delta(
    fold: &mut AlgebraFold,
    staged: &AlgebraFold,
) -> Result<(), FoldConsumeError> {
    let dimension = staged.index.nodes_by_index.len() as u64;
    let dimension_grew = staged.index.nodes_by_index.len() > fold.index.nodes_by_index.len();
    let mut native_delta_updates = 0_u64;
    for (edge_type, region) in &mut fold.native.resident_regions {
        if dimension_grew {
            region
                .matrix
                .resize(dimension, dimension)
                .map_err(native_consume_error)?;
            native_delta_updates = native_delta_updates.saturating_add(1);
        }
        let before = coordinates_for(&fold.index.coordinate_counts, edge_type);
        let after = coordinates_for(&staged.index.coordinate_counts, edge_type);
        for (from, to) in before.difference(&after) {
            region
                .matrix
                .remove(*from, *to)
                .map_err(native_consume_error)?;
            native_delta_updates = native_delta_updates.saturating_add(1);
        }
        for (from, to) in after.difference(&before) {
            region
                .matrix
                .set_bool(*from, *to, true)
                .map_err(native_consume_error)?;
            native_delta_updates = native_delta_updates.saturating_add(1);
        }
        region.through = staged.applied_through;
    }
    fold.stats.native_delta_updates = fold
        .stats
        .native_delta_updates
        .saturating_add(native_delta_updates);
    Ok(())
}

#[cfg(feature = "graphblas")]
fn coordinates_for(
    counts: &BTreeMap<(String, u64, u64), usize>,
    edge_type: &str,
) -> BTreeSet<(u64, u64)> {
    counts
        .iter()
        .filter_map(|((candidate, from, to), count)| {
            (candidate == edge_type && *count != 0).then_some((*from, *to))
        })
        .collect()
}

#[cfg(feature = "graphblas")]
fn require_query_identifier(value: &str, field: &'static str) -> Result<(), FoldError> {
    if value.is_empty() || value != value.trim() {
        Err(FoldError::Codec(format!(
            "Algebra {field} must be a canonical non-empty string"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fold_materializes_empty_tokens_after_epoch() {
        let mut fold = AlgebraFold::default();
        let mut sample = Vec::with_capacity(128 * 4);
        for row in 0..4 {
            for col in 0..128 {
                sample.push((((row * 31 + col * 17) % 97) as f32) / 97.0 - 0.5);
            }
        }
        let epoch = CalibrationEpoch::fit(
            &sample,
            128,
            TOKEN_BIT_WIDTH,
            "m",
            "v",
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        )
        .expect("fit");
        let node = epoch.to_node();
        fold.observe_node(node).expect("epoch");
        fold.materialize_tokens().expect("tokens");
        let projection = fold.project().expect("project");
        assert_eq!(projection.node_count, 1);
        assert_eq!(projection.tokens.slot_node.len(), 0);
    }
}
