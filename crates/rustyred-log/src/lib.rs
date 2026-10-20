//! Typed, append-only operation ledger for Theorem.
//!
//! Operations are the durable source. Receipts, timelines, actor diffs, and
//! HunkSets are deterministic projections over an [`OpRange`]. The mutable
//! lane-tip node is only a repairable acceleration structure, following the
//! existing Harness event-journal pattern.
//!
//! This journal extends the graph substrate; it does not replace the Harness
//! run-transition ledger. Harness transitions remain the lifecycle authority
//! for runs, while `TheoremOp` records typed user, model, and tool effects that
//! can be folded across run boundaries into review surfaces.

pub use theorem_oplog::{read_write_overlap, AccessPath, OverlapError, ReadSet, WriteSet};

use rustyred_thg_core::{GraphAtomic, GraphStore, GraphStoreError, NodeQuery, NodeRecord};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use theorem_blocks::prelude::{Shape, ShapeId, ShapeRegistry, ShapeVersion};
use ulid::Ulid;

pub const OP_PROTOCOL_VERSION: u32 = 1;
pub const OP_NODE_LABEL: &str = "TheoremOp";
pub const OP_LANE_LABEL: &str = "TheoremOpLane";
pub const OP_TENANT_TIP_LABEL: &str = "TheoremOpTenantTip";
pub const TEN_THOUSAND_OP_READ_BUDGET_MS: u128 = 2_000;
pub const MAX_OP_QUERY: usize = 1_000_000;

macro_rules! string_ref {
    ($name:ident) => {
        #[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_ref!(TenantSlug);
string_ref!(ActorId);
string_ref!(BlockId);
string_ref!(DerivationRef);
string_ref!(GrantRef);
string_ref!(SessionId);
string_ref!(VerificationRef);
string_ref!(ValueRef);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct OpId(pub Ulid);

impl OpId {
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    fn after(previous: Option<Self>, requested_ts_ms: u64) -> Self {
        let entropy = Ulid::new().random();
        let Some(previous) = previous else {
            return Self(Ulid::from_parts(requested_ts_ms, entropy));
        };
        let previous_ts_ms = previous.0.timestamp_ms();
        if requested_ts_ms > previous_ts_ms {
            return Self(Ulid::from_parts(requested_ts_ms, entropy));
        }
        const MAX_RANDOM: u128 = (1_u128 << 80) - 1;
        if previous.0.random() < MAX_RANDOM {
            return Self(Ulid::from_parts(previous_ts_ms, previous.0.random() + 1));
        }
        Self(Ulid::from_parts(previous_ts_ms.saturating_add(1), entropy))
    }
}

impl Default for OpId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Blake3Hash(pub String);

impl fmt::Display for Blake3Hash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Invertibility {
    Invertible,
    NonInvertible,
}

/// A typed operation kind backed by a registered UA2 Shape.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpKind {
    pub shape: ShapeId,
    pub invertibility: Invertibility,
}

impl OpKind {
    pub fn core(name: &str, invertibility: Invertibility) -> Self {
        Self {
            shape: ShapeId {
                namespace: "theorem.op".to_string(),
                name: name.to_string(),
                version: ShapeVersion::new(1, 0, 0),
            },
            invertibility,
        }
    }

    pub fn name(&self) -> &str {
        &self.shape.name
    }

    pub fn is_attention(&self) -> bool {
        self.name().starts_with("attention.")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StochasticDischarge {
    Undischarged,
    Discharged { verify_ref: VerificationRef },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StochasticOrigin {
    pub head_id: String,
    pub model_version: String,
    pub discharge: StochasticDischarge,
}

pub type OpPayload = Value;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NewOp {
    pub tenant: TenantSlug,
    pub actor_id: ActorId,
    pub kind: OpKind,
    #[serde(default)]
    pub target_blocks: Vec<BlockId>,
    pub payload: OpPayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation_ref: Option<DerivationRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_ref: Option<GrantRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stochastic: Option<StochasticOrigin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub ts_ms: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Op {
    pub protocol_version: u32,
    pub op_id: OpId,
    pub op_hash: Blake3Hash,
    pub tenant: TenantSlug,
    pub actor_id: ActorId,
    pub kind: OpKind,
    pub target_blocks: Vec<BlockId>,
    pub payload: OpPayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation_ref: Option<DerivationRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_ref: Option<GrantRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stochastic: Option<StochasticOrigin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_op_id: Option<OpId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_op_hash: Option<Blake3Hash>,
    pub ts_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpRange {
    pub first_op_id: OpId,
    pub last_op_id: OpId,
    pub range_hash: Blake3Hash,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AppendReceipt {
    pub op_id: OpId,
    pub op_hash: Blake3Hash,
    pub op_range: OpRange,
    pub replayed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpStream {
    pub range: OpRange,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RangeProof {
    pub range: OpRange,
    pub op_count: usize,
    pub verified: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpLogError {
    EmptyField(&'static str),
    TenantMismatch {
        expected: TenantSlug,
        actual: TenantSlug,
    },
    InvalidPayload(String),
    ShapeMissing(String),
    ShapeInvalid {
        shape: String,
        detail: String,
    },
    Store(String),
    AppendConflict(OpId),
    IdempotencyConflict(String),
    AppendContention,
    RangeEmpty,
    RangeEndpointMissing(OpId),
    RangeOrder,
    HashMismatch(OpId),
    ChainMismatch(OpId),
    RangeHashMismatch,
    Decode {
        node_id: String,
        detail: String,
    },
}

impl fmt::Display for OpLogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} must not be empty"),
            Self::TenantMismatch { expected, actual } => {
                write!(
                    formatter,
                    "operation tenant {actual} does not match ledger tenant {expected}"
                )
            }
            Self::InvalidPayload(detail) => write!(formatter, "op payload is invalid: {detail}"),
            Self::ShapeMissing(shape) => {
                write!(formatter, "operation shape is not registered: {shape}")
            }
            Self::ShapeInvalid { shape, detail } => {
                write!(
                    formatter,
                    "operation payload does not match {shape}: {detail}"
                )
            }
            Self::Store(detail) => write!(formatter, "operation store failed: {detail}"),
            Self::AppendConflict(op_id) => {
                write!(formatter, "operation id already exists: {op_id}")
            }
            Self::IdempotencyConflict(key) => {
                write!(
                    formatter,
                    "idempotency key was reused with different content: {key}"
                )
            }
            Self::AppendContention => {
                formatter.write_str("operation lane remained contended after 64 attempts")
            }
            Self::RangeEmpty => formatter.write_str("operation range is empty"),
            Self::RangeEndpointMissing(op_id) => {
                write!(formatter, "operation range endpoint is missing: {op_id}")
            }
            Self::RangeOrder => formatter.write_str("operation range ends before it starts"),
            Self::HashMismatch(op_id) => write!(formatter, "operation hash mismatch at {op_id}"),
            Self::ChainMismatch(op_id) => {
                write!(formatter, "operation hash chain mismatch at {op_id}")
            }
            Self::RangeHashMismatch => formatter.write_str("operation range hash mismatch"),
            Self::Decode { node_id, detail } => write!(
                formatter,
                "operation node {node_id} failed to decode: {detail}"
            ),
        }
    }
}

impl Error for OpLogError {}

impl From<GraphStoreError> for OpLogError {
    fn from(value: GraphStoreError) -> Self {
        Self::Store(format!("{}: {}", value.code, value.message))
    }
}

pub type OpLogResult<T> = Result<T, OpLogError>;

/// Register the initial operation families required by OP-LEDGER.
pub fn register_core_op_shapes(registry: &mut ShapeRegistry) {
    for (name, invertibility) in core_op_kinds() {
        registry.register(Shape {
            id: OpKind::core(name, invertibility).shape,
            description: format!("Typed Theorem operation payload for {name}"),
            mime_type: "application/json".to_string(),
            schema: op_shape_schema(name),
            active: true,
            column_contract: None,
        });
    }
}

pub fn core_op_registry() -> ShapeRegistry {
    let mut registry = ShapeRegistry::new();
    register_core_op_shapes(&mut registry);
    registry
}

pub fn core_op_kinds() -> Vec<(&'static str, Invertibility)> {
    use Invertibility::{Invertible, NonInvertible};
    vec![
        ("block.create", Invertible),
        ("block.update", Invertible),
        ("block.delete", NonInvertible),
        ("block.link", NonInvertible),
        ("block.unlink", NonInvertible),
        ("text.delta", Invertible),
        ("run.node_result", NonInvertible),
        ("tool.invoke", NonInvertible),
        ("forme.execute", NonInvertible),
        ("ingest.commit", NonInvertible),
        ("approval.grant", NonInvertible),
        ("approval.deny", NonInvertible),
        ("approval.human_discharge", NonInvertible),
        ("grant.issue", Invertible),
        ("grant.revoke", Invertible),
        ("attention.focus", NonInvertible),
        ("attention.blur", NonInvertible),
        ("attention.viewport", NonInvertible),
        ("undo.apply", NonInvertible),
        ("hunk.accept.requested", NonInvertible),
        ("hunk.accept.effect_failed", NonInvertible),
        ("hunk.accept", NonInvertible),
        ("hunk.reject.requested", NonInvertible),
        ("hunk.reject.effect_failed", NonInvertible),
        ("hunk.reject", NonInvertible),
        ("hunk.decline", NonInvertible),
        ("hunk.verify.requested", NonInvertible),
        ("hunk.verify", NonInvertible),
        ("object.action", NonInvertible),
    ]
}

fn op_shape_schema(name: &str) -> Value {
    let common = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": format!("Theorem operation {name}"),
        "type": "object",
        "additionalProperties": true
    });
    let mut schema = common.as_object().cloned().unwrap_or_default();
    let (properties, any_of) = if name.starts_with("block.") || name == "text.delta" {
        (
            json!({
                "capture": {"type": "string"},
                "invocation_id": {"type": "string", "minLength": 1},
                "action": {"type": "object"},
                "inverse_action": {"type": ["object", "null"]},
                "before_ref": {"type": ["string", "null"]},
                "after_ref": {"type": ["string", "null"]}
            }),
            Some(json!([
                {"required": ["action", "invocation_id"]},
                {"required": ["before_ref", "after_ref"]}
            ])),
        )
    } else if name == "run.node_result" {
        (
            json!({
                "run_id": {"type": "string", "minLength": 1},
                "invocation_id": {"type": "string", "minLength": 1},
                "head_id": {"type": "string", "minLength": 1},
                "binding_id": {"type": ["string", "null"]},
                "task": {"type": ["string", "null"]},
                "receipt_hash": {"type": "string", "minLength": 1},
                "result": {"type": "string", "enum": ["accepted", "rejected"]}
            }),
            Some(json!([{"required": ["run_id", "invocation_id", "head_id"]}])),
        )
    } else if name == "tool.invoke" {
        (
            json!({
                "invocation_id": {"type": "string", "minLength": 1},
                "action": {"type": "object"},
                "policy_receipt": {"type": "object"},
                "resolution_path": {"type": "string", "minLength": 1},
                "granted_scopes": {"type": "array", "items": {"type": "string"}}
            }),
            Some(json!([{
                "required": ["invocation_id", "action", "policy_receipt", "resolution_path"]
            }])),
        )
    } else if name == "forme.execute" {
        (
            json!({
                "forme_ref": {"type": "string", "minLength": 1},
                "execution_id": {"type": "string", "minLength": 1},
                "inputs": {"type": "object"}
            }),
            Some(json!([{"required": ["forme_ref", "execution_id"]}])),
        )
    } else if name == "ingest.commit" {
        (
            json!({
                "commit_id": {"type": "string", "minLength": 1},
                "source_ref": {"type": "string", "minLength": 1},
                "object_ids": {"type": "array", "items": {"type": "string"}}
            }),
            Some(json!([{"required": ["commit_id", "source_ref"]}])),
        )
    } else if name.starts_with("grant.") {
        (
            json!({
                "grant_id": {"type": "string", "minLength": 1},
                "principal_id": {"type": "string", "minLength": 1},
                "scopes": {"type": "array", "items": {"type": "string"}}
            }),
            Some(json!([{"required": ["grant_id", "principal_id"]}])),
        )
    } else if name == "undo.apply" {
        (
            json!({
                "source_range": {
                    "type": "object",
                    "required": ["first_op_id", "last_op_id"],
                    "properties": {
                        "first_op_id": {"type": "string", "minLength": 1},
                        "last_op_id": {"type": "string", "minLength": 1}
                    }
                },
                "applied_op_ids": {"type": "array", "items": {"type": "string"}}
            }),
            Some(json!([{"required": ["source_range", "applied_op_ids"]}])),
        )
    } else if name == "object.action" {
        (
            json!({
                "invocation_id": {"type": "string", "minLength": 1},
                "action": {"type": "object"},
                "inverse_action": {"type": ["object", "null"]},
                "policy_receipt": {"type": ["object", "null"]},
                "resolution_path": {"type": ["string", "null"]}
            }),
            Some(json!([{"required": ["invocation_id", "action"]}])),
        )
    } else if name.starts_with("attention.") {
        (
            json!({
                "target": {"type": ["string", "object", "array", "null"]},
                "viewport": {"type": ["object", "null"]}
            }),
            None,
        )
    } else if name == "hunk.verify" {
        (
            json!({
                "hunk_set_id": {"type": "string", "minLength": 1},
                "affected_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                "binding_id": {"type": "string", "minLength": 1},
                "verification_ref": {"type": "string", "minLength": 1},
                "run_id": {"type": "string", "minLength": 1},
                "receipt_hash": {"type": "string", "minLength": 1}
            }),
            Some(
                json!([{"required": ["hunk_set_id", "affected_ids", "binding_id", "verification_ref", "run_id", "receipt_hash"]}]),
            ),
        )
    } else if name == "approval.human_discharge" {
        (
            json!({
                "hunk_set_id": {"type": "string", "minLength": 1},
                "affected_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                "reviewer_id": {"type": "string", "minLength": 1},
                "verifier_id": {"type": "string", "minLength": 1}
            }),
            Some(
                json!([{"required": ["hunk_set_id", "affected_ids", "reviewer_id", "verifier_id"]}]),
            ),
        )
    } else if name.starts_with("hunk.") {
        (
            json!({
                "hunk_set_id": {"type": "string", "minLength": 1},
                "affected_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                "reason": {"type": "string"},
                "phase": {"type": "string"},
                "error": {"type": "string"},
                "binding_id": {"type": "string", "minLength": 1},
                "verification_refs": {"type": "array", "items": {"type": "string"}},
                "human_discharge": {"type": ["object", "null"]}
            }),
            Some(json!([{"required": ["hunk_set_id", "affected_ids"]}])),
        )
    } else {
        (
            json!({
                "capture": {"type": "string"},
                "invocation_id": {"type": "string"},
                "action": {"type": "object"},
                "receipt_ref": {"type": "string"}
            }),
            None,
        )
    };
    schema.insert("properties".to_string(), properties);
    if let Some(any_of) = any_of {
        schema.insert("anyOf".to_string(), any_of);
    }
    Value::Object(schema)
}

/// Graph-backed operation ledger. Operation nodes are authoritative; lane-tip
/// nodes are repairable projections used to find the prior session hash.
pub struct OpLedger<'a, S: GraphStore> {
    store: &'a mut S,
    registry: &'a ShapeRegistry,
    tenant: TenantSlug,
}

impl<'a, S: GraphStore> OpLedger<'a, S> {
    pub fn new(store: &'a mut S, registry: &'a ShapeRegistry, tenant: TenantSlug) -> Self {
        Self {
            store,
            registry,
            tenant,
        }
    }

    pub fn append(&mut self, new_op: NewOp) -> OpLogResult<AppendReceipt>
    where
        S: GraphAtomic,
    {
        if new_op.tenant != self.tenant {
            return Err(OpLogError::TenantMismatch {
                expected: self.tenant.clone(),
                actual: new_op.tenant,
            });
        }
        validate_new_op(self.registry, &new_op)?;
        let lane = lane_key(&new_op);
        self.ensure_lane_head(&new_op.tenant, &lane)?;
        self.ensure_tenant_head()?;
        if let Some(existing) = self.idempotent_op(&new_op)? {
            self.advance_tenant_head(&existing)?;
            return receipt_for_op(&existing, true);
        }

        for _ in 0..64 {
            let previous = self.lane_tip(&new_op.tenant, &lane)?;
            if let Some(previous) = &previous {
                if previous.idempotency_key == new_op.idempotency_key
                    && new_op.idempotency_key.is_some()
                {
                    ensure_same_invocation(previous, &new_op)?;
                    self.materialize_op(previous, &lane)?;
                    self.advance_tenant_head(previous)?;
                    return receipt_for_op(previous, true);
                }
            }
            let tenant_tip = self.tenant_tip()?;
            let mut op = Op {
                protocol_version: OP_PROTOCOL_VERSION,
                op_id: OpId::after(tenant_tip.map(|op| op.op_id), new_op.ts_ms),
                op_hash: Blake3Hash(String::new()),
                tenant: new_op.tenant.clone(),
                actor_id: new_op.actor_id.clone(),
                kind: new_op.kind.clone(),
                target_blocks: dedup_blocks(new_op.target_blocks.clone()),
                payload: new_op.payload.clone(),
                derivation_ref: new_op.derivation_ref.clone(),
                grant_ref: new_op.grant_ref.clone(),
                stochastic: new_op.stochastic.clone(),
                session_id: new_op.session_id.clone(),
                idempotency_key: new_op.idempotency_key.clone(),
                prev_op_id: previous.as_ref().map(|op| op.op_id),
                prev_op_hash: previous.as_ref().map(|op| op.op_hash.clone()),
                ts_ms: new_op.ts_ms,
            };
            op.op_hash = hash_op(&op)?;
            let expected = previous
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let next = serde_json::to_value(&op)
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?;
            if self
                .store
                .compare_and_set_node_property(
                    &lane_node_id(&new_op.tenant, &lane),
                    "head",
                    &expected,
                    next,
                )?
                .is_none()
            {
                continue;
            }
            self.materialize_op(&op, &lane)?;
            self.advance_tenant_head(&op)?;
            return receipt_for_op(&op, false);
        }
        Err(OpLogError::AppendContention)
    }

    pub fn read_range(&self, range: &OpRange) -> OpLogResult<OpStream> {
        let mut ops = self.tenant_ops_for_range(range)?;
        if ops.is_empty() {
            return Err(OpLogError::RangeEmpty);
        }
        if ops.first().map(|op| op.op_id) != Some(range.first_op_id) {
            return Err(OpLogError::RangeEndpointMissing(range.first_op_id));
        }
        if ops.last().map(|op| op.op_id) != Some(range.last_op_id) {
            return Err(OpLogError::RangeEndpointMissing(range.last_op_id));
        }
        ops.sort_by_key(|op| op.op_id);
        Ok(OpStream {
            range: range.clone(),
            ops,
        })
    }

    pub fn verify_range(&self, range: &OpRange) -> OpLogResult<RangeProof> {
        let stream = self.read_range(range)?;
        let mut last_by_lane: BTreeMap<String, &Op> = BTreeMap::new();
        for op in &stream.ops {
            if hash_op(op)? != op.op_hash {
                return Err(OpLogError::HashMismatch(op.op_id));
            }
            let lane = op_lane_key(op);
            if let Some(previous) = last_by_lane.get(&lane) {
                if op.prev_op_hash.as_ref() != Some(&previous.op_hash) {
                    return Err(OpLogError::ChainMismatch(op.op_id));
                }
            }
            last_by_lane.insert(lane, op);
        }
        if hash_range(&stream.ops) != range.range_hash {
            return Err(OpLogError::RangeHashMismatch);
        }
        Ok(RangeProof {
            range: range.clone(),
            op_count: stream.ops.len(),
            verified: true,
        })
    }

    pub fn range_from_ids(&self, first: OpId, last: OpId) -> OpLogResult<OpRange> {
        if last < first {
            return Err(OpLogError::RangeOrder);
        }
        let placeholder = OpRange {
            first_op_id: first,
            last_op_id: last,
            range_hash: Blake3Hash(String::new()),
        };
        let ops = self.tenant_ops_for_range(&placeholder)?;
        if ops.first().map(|op| op.op_id) != Some(first) {
            return Err(OpLogError::RangeEndpointMissing(first));
        }
        if ops.last().map(|op| op.op_id) != Some(last) {
            return Err(OpLogError::RangeEndpointMissing(last));
        }
        Ok(OpRange {
            first_op_id: first,
            last_op_id: last,
            range_hash: hash_range(&ops),
        })
    }

    /// Recover a durable operation by its tenant-scoped invocation key.
    ///
    /// Executors use this before repeating fallible external effects. The
    /// append path remains the authority for semantic retry equivalence.
    pub fn op_by_idempotency_key(&self, key: &str) -> OpLogResult<Option<Op>> {
        let mut matches = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&self.tenant.0))
                    .with_property("idempotency_key", json!(key))
                    .with_limit(2),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        if matches.len() > 1 {
            return Err(OpLogError::IdempotencyConflict(key.to_string()));
        }
        Ok(matches.pop())
    }

    /// Read tenant operations strictly after a durable cursor.
    pub fn ops_after(&self, cursor: OpId) -> OpLogResult<Vec<Op>> {
        let mut ops = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&self.tenant.0))
                    .with_limit(MAX_OP_QUERY),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        ops.retain(|op| op.op_id > cursor);
        ops.sort_by_key(|op| op.op_id);
        Ok(ops)
    }

    fn tenant_ops_for_range(&self, range: &OpRange) -> OpLogResult<Vec<Op>> {
        if range.last_op_id < range.first_op_id {
            return Err(OpLogError::RangeOrder);
        }
        let mut ops = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&self.tenant.0))
                    .with_limit(MAX_OP_QUERY),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        ops.retain(|op| op.op_id >= range.first_op_id && op.op_id <= range.last_op_id);
        ops.sort_by_key(|op| op.op_id);
        Ok(ops)
    }

    fn lane_tip(&self, tenant: &TenantSlug, lane: &str) -> OpLogResult<Option<Op>> {
        let node_id = lane_node_id(tenant, lane);
        if let Some(node) = self.store.get_node(&node_id) {
            if let Some(head) = node.properties.get("head") {
                if head.is_null() {
                    return Ok(None);
                }
                return serde_json::from_value(head.clone())
                    .map(Some)
                    .map_err(|error| OpLogError::Decode {
                        node_id: node.id.clone(),
                        detail: error.to_string(),
                    });
            }
            let op_id = node
                .properties
                .get("last_op_id")
                .and_then(Value::as_str)
                .ok_or_else(|| OpLogError::Decode {
                    node_id: node.id.clone(),
                    detail: "last_op_id is missing".to_string(),
                })?;
            let op_id = op_id
                .parse::<Ulid>()
                .map(OpId)
                .map_err(|error| OpLogError::Decode {
                    node_id: node.id.clone(),
                    detail: error.to_string(),
                })?;
            return self
                .store
                .get_node(&op_node_id(tenant, op_id))
                .map(|node| decode_op_node(node.clone()))
                .transpose();
        }

        let mut lane_ops = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&tenant.0))
                    .with_property("lane", json!(lane))
                    .with_limit(MAX_OP_QUERY),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        lane_ops.sort_by_key(|op| op.op_id);
        Ok(lane_ops.pop())
    }

    fn tenant_tip(&self) -> OpLogResult<Option<Op>> {
        if let Some(node) = self.store.get_node(&tenant_tip_node_id(&self.tenant)) {
            if let Some(head) = node.properties.get("head") {
                if head.is_null() {
                    return Ok(None);
                }
                return serde_json::from_value(head.clone())
                    .map(Some)
                    .map_err(|error| OpLogError::Decode {
                        node_id: node.id.clone(),
                        detail: error.to_string(),
                    });
            }
            let op_id = node
                .properties
                .get("last_op_id")
                .and_then(Value::as_str)
                .ok_or_else(|| OpLogError::Decode {
                    node_id: node.id.clone(),
                    detail: "last_op_id is missing".to_string(),
                })?
                .parse::<Ulid>()
                .map(OpId)
                .map_err(|error| OpLogError::Decode {
                    node_id: node.id.clone(),
                    detail: error.to_string(),
                })?;
            return self
                .store
                .get_node(&op_node_id(&self.tenant, op_id))
                .map(|node| decode_op_node(node.clone()))
                .transpose();
        }
        let mut ops = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&self.tenant.0))
                    .with_limit(MAX_OP_QUERY),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        ops.sort_by_key(|op| op.op_id);
        Ok(ops.pop())
    }

    fn ensure_lane_head(&mut self, tenant: &TenantSlug, lane: &str) -> OpLogResult<()>
    where
        S: GraphAtomic,
    {
        let node_id = lane_node_id(tenant, lane);
        if self.store.get_node(&node_id).is_none() {
            let recovered = self.lane_tip(tenant, lane)?;
            let head = recovered
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let _ = self.store.insert_node_if_absent(NodeRecord::new(
                node_id.clone(),
                [OP_LANE_LABEL],
                json!({"tenant": tenant, "lane": lane, "head": head}),
            ))?;
        }
        let needs_migration = self
            .store
            .get_node(&node_id)
            .is_some_and(|node| node.properties.get("head").is_none());
        if needs_migration {
            let legacy = self.lane_tip(tenant, lane)?;
            let head = legacy
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let _ =
                self.store
                    .compare_and_set_node_property(&node_id, "head", &Value::Null, head)?;
        }
        Ok(())
    }

    fn ensure_tenant_head(&mut self) -> OpLogResult<()>
    where
        S: GraphAtomic,
    {
        let node_id = tenant_tip_node_id(&self.tenant);
        if self.store.get_node(&node_id).is_none() {
            let recovered = self.tenant_tip()?;
            let head = recovered
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let _ = self.store.insert_node_if_absent(NodeRecord::new(
                node_id.clone(),
                [OP_TENANT_TIP_LABEL],
                json!({"tenant": self.tenant, "head": head}),
            ))?;
        }
        let needs_migration = self
            .store
            .get_node(&node_id)
            .is_some_and(|node| node.properties.get("head").is_none());
        if needs_migration {
            let legacy = self.tenant_tip()?;
            let head = legacy
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let _ =
                self.store
                    .compare_and_set_node_property(&node_id, "head", &Value::Null, head)?;
        }
        Ok(())
    }

    fn idempotent_op(&self, new_op: &NewOp) -> OpLogResult<Option<Op>> {
        let Some(key) = &new_op.idempotency_key else {
            return Ok(None);
        };
        let mut matches = self
            .store
            .query_nodes(
                NodeQuery::label(OP_NODE_LABEL)
                    .with_property("tenant", json!(&self.tenant.0))
                    .with_property("idempotency_key", json!(key))
                    .with_limit(2),
            )
            .into_iter()
            .map(decode_op_node)
            .collect::<OpLogResult<Vec<_>>>()?;
        if matches.len() > 1 {
            return Err(OpLogError::IdempotencyConflict(key.clone()));
        }
        if let Some(existing) = matches.pop() {
            ensure_same_invocation(&existing, new_op)?;
            return Ok(Some(existing));
        }
        Ok(None)
    }

    fn materialize_op(&mut self, op: &Op, lane: &str) -> OpLogResult<()>
    where
        S: GraphAtomic,
    {
        let node = op_node(op, lane)?;
        if self.store.insert_node_if_never_existed(node)?.is_some() {
            return Ok(());
        }
        let existing = self
            .store
            .get_node(&op_node_id(&op.tenant, op.op_id))
            .cloned()
            .ok_or(OpLogError::AppendConflict(op.op_id))?;
        if decode_op_node(existing)? == *op {
            Ok(())
        } else {
            Err(OpLogError::AppendConflict(op.op_id))
        }
    }

    fn advance_tenant_head(&mut self, op: &Op) -> OpLogResult<()>
    where
        S: GraphAtomic,
    {
        let node_id = tenant_tip_node_id(&self.tenant);
        for _ in 0..64 {
            let current = self.tenant_tip()?;
            if current.as_ref().is_some_and(|tip| tip.op_id >= op.op_id) {
                return Ok(());
            }
            let expected = current
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?
                .unwrap_or(Value::Null);
            let next = serde_json::to_value(op)
                .map_err(|error| OpLogError::InvalidPayload(error.to_string()))?;
            if self
                .store
                .compare_and_set_node_property(&node_id, "head", &expected, next)?
                .is_some()
            {
                return Ok(());
            }
        }
        Err(OpLogError::AppendContention)
    }
}

fn validate_new_op(registry: &ShapeRegistry, op: &NewOp) -> OpLogResult<()> {
    if op.tenant.0.trim().is_empty() {
        return Err(OpLogError::EmptyField("tenant"));
    }
    if op.actor_id.0.trim().is_empty() {
        return Err(OpLogError::EmptyField("actor_id"));
    }
    if op
        .idempotency_key
        .as_deref()
        .is_some_and(|key| key.trim().is_empty())
    {
        return Err(OpLogError::EmptyField("idempotency_key"));
    }
    if !op.payload.is_object() {
        return Err(OpLogError::InvalidPayload(
            "payload must be an object".to_string(),
        ));
    }
    let shape = registry
        .lookup_version(
            &op.kind.shape.namespace,
            &op.kind.shape.name,
            &op.kind.shape.version,
        )
        .ok_or_else(|| OpLogError::ShapeMissing(op.kind.shape.reference()))?;
    shape
        .validate(&op.payload)
        .map_err(|error| OpLogError::ShapeInvalid {
            shape: op.kind.shape.reference(),
            detail: error.to_string(),
        })
}

fn dedup_blocks(blocks: Vec<BlockId>) -> Vec<BlockId> {
    blocks
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn lane_key(op: &NewOp) -> String {
    op.session_id
        .as_ref()
        .map(|session| format!("session:{}", session.0))
        .unwrap_or_else(|| format!("actor:{}", op.actor_id.0))
}

fn op_lane_key(op: &Op) -> String {
    op.session_id
        .as_ref()
        .map(|session| format!("session:{}", session.0))
        .unwrap_or_else(|| format!("actor:{}", op.actor_id.0))
}

fn op_node(op: &Op, lane: &str) -> OpLogResult<NodeRecord> {
    let mut properties =
        serde_json::to_value(op).map_err(|error| OpLogError::InvalidPayload(error.to_string()))?;
    properties["lane"] = Value::String(lane.to_string());
    Ok(NodeRecord::new(
        op_node_id(&op.tenant, op.op_id),
        [OP_NODE_LABEL],
        properties,
    ))
}

fn op_node_id(tenant: &TenantSlug, op_id: OpId) -> String {
    format!("theorem-op:{}:{}", tenant.0, op_id)
}

fn lane_node_id(tenant: &TenantSlug, lane: &str) -> String {
    let digest = blake3::hash(lane.as_bytes()).to_hex();
    format!("theorem-op-lane:{}:{}", tenant.0, &digest[..24])
}

fn tenant_tip_node_id(tenant: &TenantSlug) -> String {
    format!("theorem-op-tenant-tip:{}", tenant.0)
}

fn decode_op_node(node: NodeRecord) -> OpLogResult<Op> {
    serde_json::from_value(node.properties).map_err(|error| OpLogError::Decode {
        node_id: node.id,
        detail: error.to_string(),
    })
}

fn hash_op(op: &Op) -> OpLogResult<Blake3Hash> {
    let material = json!({
        "protocol_version": op.protocol_version,
        "op_id": op.op_id,
        "tenant": op.tenant,
        "actor_id": op.actor_id,
        "kind": op.kind,
        "target_blocks": op.target_blocks,
        "payload": op.payload,
        "derivation_ref": op.derivation_ref,
        "grant_ref": op.grant_ref,
        "stochastic": op.stochastic,
        "session_id": op.session_id,
        "idempotency_key": op.idempotency_key,
        "prev_op_id": op.prev_op_id,
        "prev_op_hash": op.prev_op_hash,
        "ts_ms": op.ts_ms
    });
    Ok(hash_value(&material))
}

fn ensure_same_invocation(existing: &Op, requested: &NewOp) -> OpLogResult<()> {
    let same = existing.tenant == requested.tenant
        && existing.actor_id == requested.actor_id
        && existing.kind == requested.kind
        && existing.target_blocks == dedup_blocks(requested.target_blocks.clone())
        && semantic_idempotency_payload(&existing.payload)
            == semantic_idempotency_payload(&requested.payload)
        && existing.derivation_ref == requested.derivation_ref
        && existing.grant_ref == requested.grant_ref
        && existing.stochastic == requested.stochastic
        && existing.session_id == requested.session_id
        && existing.idempotency_key == requested.idempotency_key;
    if same {
        Ok(())
    } else {
        Err(OpLogError::IdempotencyConflict(
            requested.idempotency_key.clone().unwrap_or_default(),
        ))
    }
}

fn semantic_idempotency_payload(payload: &Value) -> Value {
    let mut payload = payload.clone();
    if payload.get("capture").and_then(Value::as_str) == Some("invocation") {
        if let Some(object) = payload.as_object_mut() {
            object.remove("inverse_action");
        }
    }
    payload
}

fn receipt_for_op(op: &Op, replayed: bool) -> OpLogResult<AppendReceipt> {
    Ok(AppendReceipt {
        op_id: op.op_id,
        op_hash: op.op_hash.clone(),
        op_range: OpRange {
            first_op_id: op.op_id,
            last_op_id: op.op_id,
            range_hash: hash_range(std::slice::from_ref(op)),
        },
        replayed,
    })
}

fn hash_range(ops: &[Op]) -> Blake3Hash {
    hash_value(&Value::Array(
        ops.iter()
            .map(|op| Value::String(op.op_hash.0.clone()))
            .collect(),
    ))
}

fn hash_value(value: &Value) -> Blake3Hash {
    Blake3Hash(
        blake3::hash(canonical_json(value).as_bytes())
            .to_hex()
            .to_string(),
    )
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("string encoding"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|key| format!(
                        "{}:{}",
                        canonical_json(&Value::String(key.clone())),
                        canonical_json(values.get(key).unwrap_or(&Value::Null))
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActorBlockDiff {
    pub block_id: BlockId,
    pub before_ref: Option<ValueRef>,
    pub after_ref: Option<ValueRef>,
    pub op_range: OpRange,
    pub derivation_refs: Vec<DerivationRef>,
    pub stochastic: Option<StochasticOrigin>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActorDiffFoldOutput {
    pub actor_id: ActorId,
    pub op_range: OpRange,
    pub blocks: Vec<ActorBlockDiff>,
    pub receipt_hash: Blake3Hash,
}

/// Deterministic per-actor block-state fold consumed directly by Hunk H1.
/// Attention operations are excluded by construction.
pub fn fold_actor_diff(stream: &OpStream, actor_id: &ActorId) -> OpLogResult<ActorDiffFoldOutput> {
    #[derive(Default)]
    struct Accumulator {
        first: Option<OpId>,
        last: Option<OpId>,
        before_ref: Option<ValueRef>,
        after_ref: Option<ValueRef>,
        derivations: BTreeSet<DerivationRef>,
        stochastic: Option<StochasticOrigin>,
    }

    let mut by_block: BTreeMap<BlockId, Accumulator> = BTreeMap::new();
    for op in stream
        .ops
        .iter()
        .filter(|op| &op.actor_id == actor_id && !op.kind.is_attention())
    {
        for block_id in &op.target_blocks {
            let entry = by_block.entry(block_id.clone()).or_default();
            entry.first.get_or_insert(op.op_id);
            entry.last = Some(op.op_id);
            if entry.before_ref.is_none() {
                entry.before_ref = payload_ref(&op.payload, "before_ref");
            }
            entry.after_ref = payload_ref(&op.payload, "after_ref").or(entry.after_ref.take());
            if let Some(derivation) = &op.derivation_ref {
                entry.derivations.insert(derivation.clone());
            }
            if op.stochastic.is_some() {
                entry.stochastic = op.stochastic.clone();
            }
        }
    }

    let mut blocks = Vec::with_capacity(by_block.len());
    for (block_id, entry) in by_block {
        let first = entry.first.ok_or(OpLogError::RangeEmpty)?;
        let last = entry.last.ok_or(OpLogError::RangeEmpty)?;
        let block_ops = stream
            .ops
            .iter()
            .filter(|op| {
                op.op_id >= first && op.op_id <= last && op.target_blocks.contains(&block_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        blocks.push(ActorBlockDiff {
            block_id,
            before_ref: entry.before_ref,
            after_ref: entry.after_ref,
            op_range: OpRange {
                first_op_id: first,
                last_op_id: last,
                range_hash: hash_range(&block_ops),
            },
            derivation_refs: entry.derivations.into_iter().collect(),
            stochastic: entry.stochastic,
        });
    }

    let material = json!({
        "actor_id": actor_id,
        "op_range": stream.range,
        "blocks": blocks
    });
    Ok(ActorDiffFoldOutput {
        actor_id: actor_id.clone(),
        op_range: stream.range.clone(),
        blocks,
        receipt_hash: hash_value(&material),
    })
}

fn payload_ref(payload: &Value, field: &str) -> Option<ValueRef> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ValueRef::from)
}

/// Align an append-ordered event to the config state that governed it (RO4).
pub fn config_state_for_event<'a, T>(
    event_key: &str,
    config_lane: &'a [(String, T)],
) -> rustyred_thg_core::GraphStoreResult<Option<&'a T>> {
    rustyred_thg_core::asof_nearest_preceding(event_key, config_lane)
}
mod tests {
    use super::*;
    use rustyred_thg_core::{
        ConditionalNodePropertyUpdate, EdgeRecord, GraphMutationBatch, GraphRead,
        GraphRebuildReport, GraphStats, GraphStoreResult, GraphVerify, GraphWrite,
        GraphWriteResult, InMemoryGraphStore, NeighborHit, NeighborQuery, VerifyReport,
    };

    struct ContendedStore {
        inner: InMemoryGraphStore,
        lose_next_lane_cas: bool,
    }

    impl ContendedStore {
        fn new() -> Self {
            Self {
                inner: InMemoryGraphStore::new(),
                lose_next_lane_cas: true,
            }
        }
    }

    impl GraphWrite for ContendedStore {
        fn upsert_node(&mut self, node: NodeRecord) -> GraphStoreResult<GraphWriteResult> {
            self.inner.upsert_node(node)
        }

        fn upsert_edge(&mut self, edge: EdgeRecord) -> GraphStoreResult<GraphWriteResult> {
            self.inner.upsert_edge(edge)
        }
    }

    impl GraphAtomic for ContendedStore {
        fn insert_node_if_absent(
            &mut self,
            node: NodeRecord,
        ) -> GraphStoreResult<Option<GraphWriteResult>> {
            self.inner.insert_node_if_absent(node)
        }

        fn insert_node_if_never_existed(
            &mut self,
            node: NodeRecord,
        ) -> GraphStoreResult<Option<GraphWriteResult>> {
            self.inner.insert_node_if_never_existed(node)
        }

        fn compare_and_set_node_property(
            &mut self,
            id: &str,
            property: &str,
            expected: &Value,
            new_value: Value,
        ) -> GraphStoreResult<Option<GraphWriteResult>> {
            if self.lose_next_lane_cas && id.starts_with("theorem-op-lane:") {
                self.lose_next_lane_cas = false;
                return Ok(None);
            }
            self.inner
                .compare_and_set_node_property(id, property, expected, new_value)
        }

        fn commit_conditional_batch_atomic(
            &mut self,
            conditions: Vec<ConditionalNodePropertyUpdate>,
            batch: GraphMutationBatch,
        ) -> GraphStoreResult<bool> {
            GraphAtomic::commit_conditional_batch_atomic(&mut self.inner, conditions, batch)
        }
    }

    impl GraphRead for ContendedStore {
        fn get_node(&self, id: &str) -> Option<&NodeRecord> {
            self.inner.get_node(id)
        }

        fn get_edge(&self, id: &str) -> Option<&EdgeRecord> {
            self.inner.get_edge(id)
        }

        fn live_edge_records(&self) -> Vec<EdgeRecord> {
            GraphRead::live_edge_records(&self.inner)
        }

        fn query_nodes(&self, query: NodeQuery) -> Vec<NodeRecord> {
            self.inner.query_nodes(query)
        }

        fn neighbors(&self, query: NeighborQuery) -> Vec<NeighborHit> {
            self.inner.neighbors(query)
        }

        fn stats(&self) -> GraphStats {
            self.inner.stats()
        }
    }

    impl GraphVerify for ContendedStore {
        fn verify(&self) -> VerifyReport {
            self.inner.verify()
        }

        fn rebuild_indexes(&mut self) -> GraphStoreResult<GraphRebuildReport> {
            self.inner.rebuild_indexes()
        }
    }

    fn registry() -> ShapeRegistry {
        let mut registry = ShapeRegistry::new();
        register_core_op_shapes(&mut registry);
        registry
    }

    fn ledger<'a>(
        store: &'a mut InMemoryGraphStore,
        registry: &'a ShapeRegistry,
    ) -> OpLedger<'a, InMemoryGraphStore> {
        OpLedger::new(store, registry, TenantSlug::from("Travis-Gilbert"))
    }

    fn op(actor: &str, block: &str, before: &str, after: &str, ts_ms: u64) -> NewOp {
        NewOp {
            tenant: TenantSlug::from("Travis-Gilbert"),
            actor_id: ActorId::from(actor),
            kind: OpKind::core("block.update", Invertibility::Invertible),
            target_blocks: vec![BlockId::from(block)],
            payload: json!({"before_ref": before, "after_ref": after}),
            derivation_ref: Some(DerivationRef::from("derivation:fixture")),
            grant_ref: None,
            stochastic: None,
            session_id: Some(SessionId::from("session:fixture")),
            idempotency_key: None,
            ts_ms,
        }
    }

    #[test]
    fn append_read_and_verify_round_trip() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let (first, second) = {
            let mut ledger = ledger(&mut store, &registry);
            let first = ledger
                .append(op("codex", "block:1", "v1", "v2", 1))
                .unwrap();
            let second = ledger
                .append(op("codex", "block:1", "v2", "v3", 2))
                .unwrap();
            (first, second)
        };
        let ledger = ledger(&mut store, &registry);
        let range = ledger.range_from_ids(first.op_id, second.op_id).unwrap();
        let stream = ledger.read_range(&range).unwrap();
        assert_eq!(stream.ops.len(), 2);
        assert_eq!(
            stream.ops[1].prev_op_hash,
            Some(stream.ops[0].op_hash.clone())
        );
        assert!(ledger.verify_range(&range).unwrap().verified);
    }

    #[test]
    fn unregistered_operation_shape_is_refused() {
        let registry = ShapeRegistry::new();
        let mut store = InMemoryGraphStore::new();
        let error = ledger(&mut store, &registry)
            .append(op("codex", "block:1", "v1", "v2", 1))
            .unwrap_err();
        assert!(matches!(error, OpLogError::ShapeMissing(_)));
        assert!(store
            .query_nodes(NodeQuery::label(OP_NODE_LABEL))
            .is_empty());
    }

    #[test]
    fn operation_families_enforce_distinct_minimal_payloads() {
        let registry = registry();
        let fixtures = [
            (
                "forme.execute",
                json!({"forme_ref": "forme:1", "execution_id": "execution:1"}),
            ),
            (
                "ingest.commit",
                json!({"commit_id": "commit:1", "source_ref": "source:1"}),
            ),
            (
                "grant.issue",
                json!({"grant_id": "grant:1", "principal_id": "agent:1"}),
            ),
            (
                "grant.revoke",
                json!({"grant_id": "grant:1", "principal_id": "agent:1"}),
            ),
            (
                "undo.apply",
                json!({
                    "source_range": {
                        "first_op_id": "1",
                        "last_op_id": "2"
                    },
                    "applied_op_ids": ["3"]
                }),
            ),
            (
                "object.action",
                json!({"invocation_id": "invocation:1", "action": {"kind": "select"}}),
            ),
            (
                "hunk.verify",
                json!({
                    "hunk_set_id": "hunk-set:1",
                    "affected_ids": ["block:1"],
                    "binding_id": "agent:hunk-verifier",
                    "verification_ref": "verify:1",
                    "run_id": "run:1",
                    "receipt_hash": "hash:1"
                }),
            ),
        ];

        for (name, valid_payload) in fixtures {
            let shape_ref = OpKind::core(name, Invertibility::NonInvertible).shape;
            let shape = registry
                .lookup_version(&shape_ref.namespace, &shape_ref.name, &shape_ref.version)
                .unwrap_or_else(|| panic!("missing shape for {name}"));
            assert!(shape.validate(&json!({})).is_err(), "{name} accepted {{}}");
            shape
                .validate(&valid_payload)
                .unwrap_or_else(|error| panic!("{name} rejected canonical payload: {error}"));
        }
    }

    #[test]
    fn mutated_payload_fails_with_the_named_op() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let receipt = ledger(&mut store, &registry)
            .append(op("codex", "block:1", "v1", "v2", 1))
            .unwrap();
        let node_id = op_node_id(&TenantSlug::from("Travis-Gilbert"), receipt.op_id);
        let mut node = store.get_node(&node_id).unwrap().clone();
        node.properties["payload"]["after_ref"] = json!("tampered");
        store.upsert_node(node).unwrap();
        let error = ledger(&mut store, &registry)
            .verify_range(&receipt.op_range)
            .unwrap_err();
        assert_eq!(error, OpLogError::HashMismatch(receipt.op_id));
    }

    #[test]
    fn actor_diff_is_deterministic_and_excludes_attention() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let (first, last) = {
            let mut ledger = ledger(&mut store, &registry);
            let first = ledger
                .append(op("codex", "block:1", "v1", "v2", 1))
                .unwrap();
            let mut attention = op("codex", "block:1", "v2", "v2", 2);
            attention.kind = OpKind::core("attention.focus", Invertibility::NonInvertible);
            ledger.append(attention).unwrap();
            let last = ledger
                .append(op("codex", "block:1", "v2", "v3", 3))
                .unwrap();
            (first, last)
        };
        let ledger = ledger(&mut store, &registry);
        let range = ledger.range_from_ids(first.op_id, last.op_id).unwrap();
        let stream = ledger.read_range(&range).unwrap();
        let left = fold_actor_diff(&stream, &ActorId::from("codex")).unwrap();
        let right = fold_actor_diff(&stream, &ActorId::from("codex")).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.blocks.len(), 1);
        assert_eq!(left.blocks[0].before_ref, Some(ValueRef::from("v1")));
        assert_eq!(left.blocks[0].after_ref, Some(ValueRef::from("v3")));
    }

    #[test]
    fn ledger_refuses_cross_tenant_append_and_range_reads_stay_scoped() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let mut wrong = op("codex", "block:1", "v1", "v2", 1);
        wrong.tenant = TenantSlug::from("another-tenant");
        let error = ledger(&mut store, &registry).append(wrong).unwrap_err();
        assert!(matches!(error, OpLogError::TenantMismatch { .. }));
        assert!(store
            .query_nodes(NodeQuery::label(OP_NODE_LABEL))
            .is_empty());
    }

    #[test]
    fn identical_idempotent_retry_returns_one_operation_and_changed_content_conflicts() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let mut request = op("codex", "block:1", "v1", "v2", 1);
        request.idempotency_key = Some("invocation:fixture".to_string());
        let first = ledger(&mut store, &registry)
            .append(request.clone())
            .unwrap();
        let retry = ledger(&mut store, &registry)
            .append(request.clone())
            .unwrap();
        assert_eq!(retry.op_id, first.op_id);
        assert_eq!(retry.op_hash, first.op_hash);
        assert!(retry.replayed);
        assert_eq!(store.query_nodes(NodeQuery::label(OP_NODE_LABEL)).len(), 1);

        request.payload["after_ref"] = json!("different");
        let error = ledger(&mut store, &registry).append(request).unwrap_err();
        assert_eq!(
            error,
            OpLogError::IdempotencyConflict("invocation:fixture".to_string())
        );
    }

    #[test]
    fn two_tenants_coexist_in_one_store_with_disjoint_ranges() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let first = ledger(&mut store, &registry)
            .append(op("codex", "block:1", "v1", "v2", 1))
            .unwrap();
        let mut other = op("codex", "block:2", "a", "b", 1);
        other.tenant = TenantSlug::from("tenant-b");
        let second = OpLedger::new(&mut store, &registry, TenantSlug::from("tenant-b"))
            .append(other)
            .unwrap();

        let tenant_a = OpLedger::new(&mut store, &registry, TenantSlug::from("Travis-Gilbert"));
        assert_eq!(tenant_a.read_range(&first.op_range).unwrap().ops.len(), 1);
        assert!(matches!(
            tenant_a.read_range(&second.op_range),
            Err(OpLogError::RangeEmpty | OpLogError::RangeEndpointMissing(_))
        ));
        let tenants = store
            .query_nodes(NodeQuery::label(OP_NODE_LABEL).with_limit(MAX_OP_QUERY))
            .into_iter()
            .filter_map(|node| node.properties["tenant"].as_str().map(str::to_string))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            tenants,
            BTreeSet::from(["Travis-Gilbert".to_string(), "tenant-b".to_string()])
        );
    }

    #[test]
    fn interleaved_actor_lanes_keep_independent_hash_chains() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let (first, last) = {
            let mut ledger = ledger(&mut store, &registry);
            let first = ledger
                .append(op("codex", "block:1", "v1", "v2", 1))
                .unwrap();
            let mut human = op("human", "block:2", "h1", "h2", 2);
            human.session_id = Some(SessionId::from("session:human"));
            ledger.append(human).unwrap();
            let last = ledger
                .append(op("codex", "block:1", "v2", "v3", 3))
                .unwrap();
            (first, last)
        };
        let ledger = ledger(&mut store, &registry);
        let range = ledger.range_from_ids(first.op_id, last.op_id).unwrap();
        assert_eq!(ledger.verify_range(&range).unwrap().op_count, 3);
    }

    #[test]
    fn lane_cas_contention_retries_without_forking_the_chain() {
        let registry = registry();
        let mut store = ContendedStore::new();
        let mut ledger = OpLedger::new(&mut store, &registry, TenantSlug::from("Travis-Gilbert"));
        let first = ledger
            .append(op("codex", "block:1", "v1", "v2", 1))
            .unwrap();
        let second = ledger
            .append(op("codex", "block:1", "v2", "v3", 2))
            .unwrap();
        let range = ledger.range_from_ids(first.op_id, second.op_id).unwrap();
        let stream = ledger.read_range(&range).unwrap();
        assert_eq!(stream.ops.len(), 2);
        assert_eq!(stream.ops[1].prev_op_id, Some(stream.ops[0].op_id));
        assert_eq!(
            stream.ops[1].prev_op_hash,
            Some(stream.ops[0].op_hash.clone())
        );
        assert!(ledger.verify_range(&range).unwrap().verified);
    }

    #[test]
    fn ten_thousand_op_range_read_meets_recorded_budget() {
        let registry = registry();
        let mut store = InMemoryGraphStore::new();
        let (first, last) = {
            let mut ledger = ledger(&mut store, &registry);
            let mut first = None;
            let mut last = None;
            for index in 0..10_000 {
                let receipt = ledger
                    .append(op(
                        "codex",
                        "block:1",
                        &format!("v{index}"),
                        &format!("v{}", index + 1),
                        index + 1,
                    ))
                    .unwrap();
                first.get_or_insert(receipt.op_id);
                last = Some(receipt.op_id);
            }
            (first.unwrap(), last.unwrap())
        };
        let ledger = ledger(&mut store, &registry);
        let range = ledger.range_from_ids(first, last).unwrap();
        let started = std::time::Instant::now();
        let stream = ledger.read_range(&range).unwrap();
        let elapsed_ms = started.elapsed().as_millis();
        eprintln!(
            "10k op range read: {elapsed_ms} ms (budget {TEN_THOUSAND_OP_READ_BUDGET_MS} ms)"
        );
        assert_eq!(stream.ops.len(), 10_000);
        assert!(elapsed_ms <= TEN_THOUSAND_OP_READ_BUDGET_MS,
            "10k op range read took {elapsed_ms} ms"
        );
    }

    #[test]
    fn asof_aligns_scripted_events_to_hand_checked_config() {
        let configs = vec![
            ("cfg-1".to_string(), "strict"),
            ("cfg-2".to_string(), "relaxed"),
        ];
        assert_eq!(
            config_state_for_event("cfg-1", &configs).unwrap(),
            Some(&"strict")
        );
        assert_eq!(
            config_state_for_event("cfg-1z", &configs).unwrap(),
            Some(&"strict")
        );
        assert_eq!(
            config_state_for_event("cfg-2", &configs).unwrap(),
            Some(&"relaxed")
        );
        let err = config_state_for_event("cfg-9", &configs).unwrap_err();
        assert_eq!(err.code, "asof_beyond_watermark");
    }
}
